"""Request-local numeric reports and a narrow native read-failure boundary."""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterator
from typing import Any, Literal

from prism.errors import PrismError
from prism.streaming.events import ErrorEvent
from prism.streaming.sse import sse_data
from prism.value_objects.stream_failure_usage import StreamFailureUsage
from prism.value_objects.usage import Usage

_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "cache_write_input_tokens",
    "cache_read_input_tokens",
    "thought_tokens",
    "cost",
    "cache_write_5m_input_tokens",
    "cache_write_1h_input_tokens",
)


class StreamFailureUsageTracker:
    """One instance per provider request, independent of completed-step accounting."""

    def __init__(
        self, provider: Literal["anthropic", "openai", "mistral"], step_index: int = 0
    ) -> None:
        self.provider = provider
        self.step_index = step_index
        self._reported: dict[str, int] = {}

    def record(self, data: dict[str, Any]) -> None:
        raw = data.get("usage")
        nested = data.get("message" if self.provider == "anthropic" else "response")
        if self.provider != "mistral" and isinstance(nested, dict) and "usage" in nested:
            raw = nested["usage"]
        if not isinstance(raw, dict):
            return
        total = raw.get("cache_creation_input_tokens")
        if (
            self.provider == "anthropic"
            and type(total) is int
            and "cache_write_input_tokens" in self._reported
            and total != self._reported["cache_write_input_tokens"]
        ):
            self._reported.pop("cache_write_5m_input_tokens", None)
            self._reported.pop("cache_write_1h_input_tokens", None)
        if self.provider == "anthropic":
            paths = {
                "prompt_tokens": "input_tokens",
                "completion_tokens": "output_tokens",
                "cache_write_input_tokens": "cache_creation_input_tokens",
                "cache_read_input_tokens": "cache_read_input_tokens",
                "thought_tokens": "output_tokens_details.thinking_tokens",
                "cache_write_5m_input_tokens": "cache_creation.ephemeral_5m_input_tokens",
                "cache_write_1h_input_tokens": "cache_creation.ephemeral_1h_input_tokens",
            }
        else:
            paths = {
                "prompt_tokens": "input_tokens" if "input_tokens" in raw else "prompt_tokens",
                "completion_tokens": "output_tokens"
                if "output_tokens" in raw
                else "completion_tokens",
                "cache_read_input_tokens": "input_tokens_details.cached_tokens"
                if "input_tokens" in raw
                else "prompt_tokens_details.cached_tokens",
                "cache_write_input_tokens": "input_tokens_details.cache_write_tokens",
                "thought_tokens": "output_tokens_details.reasoning_tokens"
                if "output_tokens" in raw
                else "completion_tokens_details.reasoning_tokens",
            }
        for field, key in paths.items():
            count: Any = raw
            for part in key.split("."):
                count = count.get(part) if isinstance(count, dict) else None
            if type(count) is int and 0 <= count <= 9007199254740991:
                self._reported[field] = count

    def snapshot(self) -> StreamFailureUsage | None:
        r = self._reported
        fields = [field for field in _FIELDS if field in r]
        if not fields:
            return None
        prompt = r.get("prompt_tokens", 0)
        if self.provider != "anthropic":
            prompt = max(0, prompt - r.get("cache_read_input_tokens", 0))
        if self.provider == "openai":
            prompt = max(0, prompt - r.get("cache_write_input_tokens", 0))
        return StreamFailureUsage(
            Usage(
                prompt,
                r.get("completion_tokens", 0),
                r.get("cache_write_input_tokens"),
                r.get("cache_read_input_tokens"),
                r.get("thought_tokens"),
                None,
                r.get("cache_write_5m_input_tokens"),
                r.get("cache_write_1h_input_tokens"),
            ),
            self.step_index,
            fields,
        )


def failure_aware_sse(
    chunks: Iterator[str],
    snapshot: Callable[[], StreamFailureUsage | None],
) -> Generator[str | ErrorEvent, None, None]:
    """Catch only native next(); user exceptions at a yield are never reclassified."""
    source = sse_data(chunks)
    try:
        while True:
            failure: Exception | None = None
            value: str | None = None
            try:
                value = next(source)
            except StopIteration:
                return
            except Exception as error:
                failure = error
            if failure is not None:
                usage = snapshot()
                if isinstance(failure, PrismError):
                    failure.with_stream_failure_usage(usage)
                elif usage is not None:
                    yield ErrorEvent(
                        code="stream_transport_error",
                        message="The stream failed.",
                        recoverable=False,
                        failure_usage=usage,
                    )
                raise failure
            if value is not None:
                yield value
    finally:
        close_source = getattr(source, "close", None)
        if callable(close_source):
            close_source()
        close_chunks = getattr(chunks, "close", None)
        if callable(close_chunks):
            close_chunks()
