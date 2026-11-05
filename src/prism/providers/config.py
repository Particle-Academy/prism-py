"""Provider configuration presence, separate from empty and explicit null."""

from __future__ import annotations

import math
import os
import re
import threading
import warnings
from decimal import Decimal
from enum import Enum
from typing import Any


class Unset(Enum):
    VALUE = 0


UNSET = Unset.VALUE
_warned_legacy_beta = False
_legacy_beta_warning_lock = threading.Lock()


def _environment_value(name: str, fallback: Any) -> Any:
    value = os.environ.get(name)
    if value is None:
        return fallback
    if value.lower() in ("null", "(null)"):
        return None
    if value.lower() in ("true", "(true)"):
        return True
    if value.lower() in ("false", "(false)"):
        return False
    if value.lower() in ("empty", "(empty)"):
        return ""
    quoted = re.fullmatch(r"(['\"])(.*)\1", value)
    return quoted[2] if quoted else value


def configured_string(
    value: str | Unset | None, name: str, fallback: str, null_fallback: str | None = None
) -> str:
    resolved = _environment_value(name, fallback) if value is UNSET else value
    if resolved is None and null_fallback is not None:
        return null_fallback
    if not isinstance(resolved, str):
        raise TypeError(f"Invalid string configuration for {name}.")
    return resolved


def nullable_string(value: str | Unset | None, name: str) -> str | None:
    resolved = _environment_value(name, None) if value is UNSET else value
    if resolved is not None and not isinstance(resolved, str):
        raise TypeError(f"Invalid nullable string configuration for {name}.")
    return resolved


def anthropic_beta(value: str | Unset | None) -> str | None:
    global _warned_legacy_beta
    if value is not UNSET or "ANTHROPIC_BETA" in os.environ:
        return nullable_string(value, "ANTHROPIC_BETA")
    legacy = os.environ.get("ANTHROPIC_BETA_FEATURES")
    if legacy is not None:
        with _legacy_beta_warning_lock:
            if not _warned_legacy_beta:
                _warned_legacy_beta = True
                warnings.warn(
                    "ANTHROPIC_BETA_FEATURES is deprecated; use ANTHROPIC_BETA.",
                    FutureWarning,
                    stacklevel=3,
                )
    return legacy


def thinking_budget_default(value: Any = UNSET) -> int:
    """The reference casts the config default, not a noninteger request budget."""
    resolved = (
        _environment_value("ANTHROPIC_DEFAULT_THINKING_BUDGET", 1024) if value is UNSET else value
    )
    if resolved is None:
        return 0
    if isinstance(resolved, (bool, int)):
        return int(resolved)
    if isinstance(resolved, float):
        return int(resolved) if math.isfinite(resolved) else 0
    if isinstance(resolved, str):
        prefix = re.match(r"\s*[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", resolved)
        return int(Decimal(prefix[0])) if prefix else 0
    raise TypeError("Invalid ANTHROPIC_DEFAULT_THINKING_BUDGET configuration.")
