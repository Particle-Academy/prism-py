"""Canonical JSON encoding.

Every byte a port puts on the wire has to match the reference's bytes, so the
encoder is part of the contract rather than an implementation detail. The rules,
and why each one is here:

* **UTF-8, no insignificant whitespace.** ``separators=(",", ":")``.
* **Forward slashes are not escaped, non-ASCII is not escaped.** PHP escapes
  both by default and Python escapes non-ASCII by default, so both sides have to
  configure their encoder rather than accept its defaults (case ``trq-0023``).
* **Object keys keep insertion order.** Never sorted. The reference builds its
  payloads key by key and the order is observable in the goldens.
* **Integral floats render without a fraction.** PHP and JavaScript render the
  float ``1.0`` as ``1``; Python's :mod:`json` renders ``1.0``. Same JSON number,
  different bytes. This module normalises integral floats to ints so the wire
  bytes agree (case ``trq-0025``, which the corpus marks skipped for Python so
  the divergence stays visible there rather than hidden in here).
* **``None`` is a real JSON ``null``.** It is never dropped. Several cases
  require an explicit null in the output — ``max_output_tokens`` is always
  present and is usually null — so "set to None" and "never set" have to be
  different states in the caller, not in the encoder.
"""

from __future__ import annotations

import json
import math
from typing import Any

__all__ = ["encode", "normalize"]


def normalize(value: Any) -> Any:
    """Recursively rewrite ``value`` into a shape :func:`json.dumps` renders canonically.

    The only rewrite is the integral-float one; everything else is passed
    through so the caller's key order and explicit ``None``s survive.
    """
    if value is None or isinstance(value, (str, bool, int)):
        # bool before int on purpose: bool is a subclass of int in Python and
        # must keep rendering as true/false.
        return value

    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return int(value)
        return value

    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return normalize(to_dict())

    raise TypeError(f"{type(value).__name__} is not JSON encodable")


def encode(value: Any) -> str:
    """Encode ``value`` as a canonical JSON string."""
    return json.dumps(
        normalize(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=False,
        allow_nan=False,
    )
