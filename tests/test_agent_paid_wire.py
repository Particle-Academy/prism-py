"""Direct loopback callers cannot turn key presence or forged fields into paid authority."""

from __future__ import annotations

import json
import os
import queue
import shutil
import socket
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path
from typing import Any


def test_native_mcp_paid_refusal_keeps_free_discovery(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    fixture = tmp_path / "workspace" / "repo"
    fixture.mkdir(parents=True)
    for name in ("agent", "src"):
        shutil.copytree(
            root / name,
            fixture / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    shutil.copyfile(root / "pyproject.toml", fixture / "pyproject.toml")
    assert not (tmp_path / ".env").exists()
    assert not (fixture / ".env").exists()
    marker = tmp_path / "provider-attempts"
    launcher = tmp_path / "tripwire.py"
    launcher.write_text(
        "import pathlib,runpy,socket,sys,urllib.request\n"
        f"marker=pathlib.Path({str(marker)!r})\n"
        "def blocked(*args,**kwargs):\n"
        " marker.write_text('attempt')\n"
        " raise RuntimeError('PROVIDER_REQUEST_BLOCKED_BY_TEST')\n"
        "urllib.request.urlopen=blocked\n"
        "socket.socket.connect=blocked\n"
        "socket.socket.connect_ex=blocked\n"
        f"sys.path.insert(0,{str(fixture / 'agent')!r})\n"
        f"runpy.run_path({str(fixture / 'agent' / 'server.py')!r},run_name='__main__')\n",
        encoding="utf-8",
    )
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    allowed = ("PATH", "SystemRoot", "WINDIR", "TEMP", "TMP", "HOME", "USERPROFILE", "LOCALAPPDATA")
    env = {name: os.environ[name] for name in allowed if name in os.environ}
    env.update(
        PRISM_AGENT_PORT=str(port),
        ANTHROPIC_API_KEY="synthetic-test-key",
        PYTHONIOENCODING="utf-8",
    )
    child = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-B",
            "-X",
            f"pycache_prefix={tmp_path / 'bytecode'}",
            "-u",
            str(launcher),
        ],
        cwd=fixture,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )
    lines: queue.Queue[str] = queue.Queue()

    def read_output() -> None:
        assert child.stdout is not None
        for line in child.stdout:
            lines.put(line)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request_id = 0

    def rpc(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        nonlocal request_id
        request_id += 1
        payload = json.dumps(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}
        ).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/mcp",
            data=payload,
            headers={"content-type": "application/json", "MCP-Protocol-Version": "2026-07-28"},
        )
        with opener.open(request, timeout=3) as response:
            assert response.status == 200
            message = json.load(response)
        assert message["id"] == request_id
        assert "error" not in message
        result: dict[str, Any] = message["result"]
        return result

    try:
        ready = lines.get(timeout=7)
        assert f"listening on http://127.0.0.1:{port}/mcp" in ready
        assert rpc("server/discover")["protocolVersion"] == "2026-07-28"
        listed = rpc("tools/list")
        assert sorted(tool["name"] for tool in listed["tools"]) == [
            "consensus",
            "describe_port",
            "ecosystem_probe",
            "explain",
            "harness_probe",
            "run_conformance",
            "run_tests",
            "status",
        ]
        for name in ("explain", "consensus"):
            tool = next(tool for tool in listed["tools"] if tool["name"] == name)
            assert "paid_call_not_authorised" in tool["description"]
            assert "refused" in tool["description"]
            for arguments in (
                {},
                {
                    "subject": "Owned",
                    "question": "Owned",
                    "approved": True,
                    "owner": "SECRET_CALLER",
                    "run_id": "forged",
                    "authority": {"approved": True, "cap": 5},
                },
                None,
            ):
                result = rpc("tools/call", {"name": name, "arguments": arguments})
                assert result["isError"] is True
                assert result["structuredContent"]["ok"] is False
                assert result["structuredContent"]["code"] == "paid_call_not_authorised"
                assert json.loads(result["content"][0]["text"]) == result["structuredContent"]
                assert "SECRET_CALLER" not in json.dumps(result)
        status = rpc("tools/call", {"name": "status"})
        assert status["isError"] is False
        assert status["structuredContent"]["can_reason"] is False
        assert status["structuredContent"]["provider_key_present"] is True
        assert status["structuredContent"]["paid_call_refusal"] == "paid_call_not_authorised"
        described = rpc("tools/call", {"name": "describe_port"})
        assert described["isError"] is False
        assert "anthropic" in described["structuredContent"]["providers_implemented"]
        assert not marker.exists()
        assert (fixture / "agent" / "server.py").read_bytes() == (
            root / "agent" / "server.py"
        ).read_bytes()
    finally:
        child.terminate()
        child.wait(timeout=3)
        reader.join(timeout=3)
        assert not reader.is_alive()
        if child.stdout is not None:
            child.stdout.close()
