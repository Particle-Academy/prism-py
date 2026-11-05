"""Request-scoped urllib handlers; no global socket timeout or opener changes.

The socket is obtained through HTTPConnection's lifecycle, rather than through
private buffered-reader chains. CPython's urllib and http.client extension seams:
https://docs.python.org/3/library/urllib.request.html#urllib.request.HTTPHandler
https://docs.python.org/3/library/http.client.html#http.client.HTTPConnection
"""

from __future__ import annotations

import http.client
import ipaddress
import queue
import socket
import threading
import time
import urllib.request
from collections.abc import Callable
from functools import partial
from typing import Any


def _lookup_worker(
    lookup: Callable[..., Any],
    args: tuple[Any, ...],
    result: queue.Queue[tuple[bool, Any]],
    ready: threading.Event,
) -> None:
    try:
        result.put((True, lookup(*args)))
    except BaseException as error:
        result.put((False, error))
    finally:
        ready.set()


def resolve_in_thread(
    lookup: Callable[..., Any], args: tuple[Any, ...], timeout: float | None
) -> Any:
    result: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)
    ready = threading.Event()
    # An abandoned daemon lookup holds no request data and ends at the OS resolver timeout.
    threading.Thread(target=_lookup_worker, args=(lookup, args, result, ready), daemon=True).start()
    if not ready.wait(timeout):
        raise TimeoutError("Prism request timed out.")
    succeeded, value = result.get_nowait()
    if not succeeded:
        raise value
    return value


class NetworkDeadline:
    def __init__(
        self,
        timeout: float,
        connect_timeout: float | None,
        pinned_host: str | None = None,
        pinned_addresses: tuple[str, ...] | None = None,
    ) -> None:
        self.deadline = time.monotonic() + timeout if timeout else None
        self.connect_timeout = connect_timeout
        self.pinned_host = pinned_host
        self.pinned_addresses = pinned_addresses
        self.expired = False
        self._sockets: list[socket.socket] = []
        self._lock = threading.Lock()
        self._active = bool(timeout)
        self._timer = threading.Timer(timeout, self._expire) if timeout else None
        if self._timer is not None:
            self._timer.daemon = True
            self._timer.start()

    def _expire(self) -> None:
        with self._lock:
            if not self._active:
                return
            self.expired = True
            sockets = self._sockets.copy()
        for connection in sockets:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass  # A completed redirect may already have closed this socket.

    def remaining(self) -> float | None:
        if self.expired:
            raise TimeoutError("Prism request timed out.")
        if self.deadline is None:
            return None
        seconds = self.deadline - time.monotonic()
        if seconds <= 0:
            self.expired = True
            raise TimeoutError("Prism request timed out.")
        return seconds

    def connection_timeout(self) -> float | None:
        remaining = self.remaining()
        connect = self.connect_timeout or None
        if remaining is None:
            return connect
        return min(remaining, connect) if connect is not None else remaining

    def connected(self, connection: socket.socket | None) -> None:
        if connection is None:
            raise ConnectionError("HTTP connection did not open a socket.")
        with self._lock:
            self._sockets.append(connection)
        connection.settimeout(self.remaining())

    def create_connection(
        self,
        address: tuple[str, int],
        _timeout: Any = None,
        source_address: tuple[str, int] | None = None,
    ) -> socket.socket:
        host, port = address
        connect_deadline = time.monotonic() + self.connect_timeout if self.connect_timeout else None

        def remaining_connect() -> float | None:
            remaining = self.remaining()
            if connect_deadline is not None:
                seconds = connect_deadline - time.monotonic()
                if seconds <= 0:
                    raise TimeoutError("Prism request timed out.")
                remaining = min(remaining, seconds) if remaining is not None else seconds
            return remaining

        if self.pinned_addresses is not None and host.lower() == (self.pinned_host or "").lower():
            records = []
            for value in self.pinned_addresses:
                parsed = ipaddress.ip_address(value)
                family = socket.AF_INET6 if parsed.version == 6 else socket.AF_INET
                destination = (value, port, 0, 0) if parsed.version == 6 else (value, port)
                records.append((family, socket.SOCK_STREAM, 0, "", destination))
        else:
            records = resolve_in_thread(
                socket.getaddrinfo, (host, port, 0, socket.SOCK_STREAM), remaining_connect()
            )
        # DNS and every address attempt share the original request budget.
        self.remaining()
        last_error: OSError | None = None
        for family, kind, protocol, _canonical, destination in records:
            connection = socket.socket(family, kind, protocol)
            try:
                connection.settimeout(remaining_connect())
                if source_address is not None:
                    connection.bind(source_address)
                with self._lock:
                    self._sockets.append(connection)
                connection.connect(destination)
                self.remaining()
                connection.settimeout(remaining_connect())
                return connection
            except OSError as error:
                connection.close()
                last_error = error
        if last_error is not None:
            raise last_error
        raise OSError("The hostname did not resolve to a usable address.")

    def stream_body(self, idle: float) -> None:
        self.remaining()
        with self._lock:
            self._active = False
        if self._timer is not None:
            self._timer.cancel()
        self.remaining()
        self.deadline = None
        for connection in self._sockets:
            try:
                connection.settimeout(idle)
            except OSError:
                pass  # Closed sockets from completed redirect hops.

    def close(self) -> None:
        with self._lock:
            self._active = False
        if self._timer is not None:
            self._timer.cancel()
        self._sockets.clear()


class _HTTPConnection(http.client.HTTPConnection):
    def __init__(self, host: str, *, budget: NetworkDeadline, **kwargs: Any) -> None:
        super().__init__(host, **kwargs)
        self._budget = budget
        # CPython's HTTPConnection hook replaces only TCP address selection;
        # HTTPSConnection still owns original-host SNI and certificate checks.
        self._create_connection = budget.create_connection

    def connect(self) -> None:
        self.timeout = self._budget.connection_timeout()
        super().connect()
        self._budget.connected(self.sock)


class _HTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host: str, *, budget: NetworkDeadline, **kwargs: Any) -> None:
        super().__init__(host, **kwargs)
        self._budget = budget
        self._create_connection = budget.create_connection

    def connect(self) -> None:
        self.timeout = self._budget.connection_timeout()
        super().connect()
        self._budget.connected(self.sock)


class _HTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, budget: NetworkDeadline) -> None:
        super().__init__()
        self._budget = budget

    def http_open(self, request: urllib.request.Request) -> Any:
        return self.do_open(partial(_HTTPConnection, budget=self._budget), request)


class _HTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, budget: NetworkDeadline) -> None:
        # Initialize the handler only; defer native trust-store creation to an
        # actual HTTPSConnection rather than loading it for every HTTP request.
        urllib.request.AbstractHTTPHandler.__init__(self)
        self._budget = budget

    def https_open(self, request: urllib.request.Request) -> Any:
        # HTTPSConnection's native default context verifies certificates and
        # hostnames. No custom CA, unverified context or TLS downgrade.
        return self.do_open(partial(_HTTPSConnection, budget=self._budget), request)


def opener(budget: NetworkDeadline, extra_handlers: list[Any]) -> urllib.request.OpenerDirector:
    # A proxy would resolve the destination independently of the validated IP.
    proxies = [urllib.request.ProxyHandler({})] if budget.pinned_addresses is not None else []
    return urllib.request.build_opener(
        _HTTPHandler(budget), _HTTPSHandler(budget), *proxies, *extra_handlers
    )
