"""Strict provider-vector decoding shared by the embedding handlers."""

from __future__ import annotations

import base64
import binascii
import math
import struct
from typing import Any

from prism.errors import ErrorCode, PrismError
from prism.value_objects.embedding import Embedding


def decode_embedding(value: Any) -> Embedding:
    if isinstance(value, list):
        return Embedding.from_list(value)

    if not isinstance(value, str):
        raise PrismError(
            ErrorCode.INVALID_EMBEDDING, "Invalid embedding: expected a vector or base64 string."
        )

    if not value.isascii():
        raise PrismError(ErrorCode.INVALID_EMBEDDING, "Invalid embedding: malformed base64.")

    try:
        raw = base64.b64decode(value, validate=True)
    except binascii.Error:
        raise PrismError(
            ErrorCode.INVALID_EMBEDDING, "Invalid embedding: malformed base64."
        ) from None

    # validate=True still admits noncanonical padding bits and some extra padding.
    if base64.b64encode(raw).decode("ascii") != value:
        raise PrismError(ErrorCode.INVALID_EMBEDDING, "Invalid embedding: malformed base64.")

    if not raw or len(raw) % 4:
        raise PrismError(
            ErrorCode.INVALID_EMBEDDING, "Invalid embedding: expected complete float32 values."
        )

    values = [number for (number,) in struct.iter_unpack("<f", raw)]
    if not all(math.isfinite(number) for number in values):
        raise PrismError(
            ErrorCode.INVALID_EMBEDDING, "Invalid embedding: non-finite float32 value."
        )

    return Embedding.from_list(values)
