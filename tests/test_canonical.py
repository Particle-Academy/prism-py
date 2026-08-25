"""The canonical encoder is part of the wire contract, not a formatting detail."""

from __future__ import annotations

import pytest

from prism import canonical


def test_no_insignificant_whitespace() -> None:
    assert canonical.encode({"a": 1, "b": [1, 2]}) == '{"a":1,"b":[1,2]}'


def test_keys_keep_insertion_order_and_are_never_sorted() -> None:
    assert canonical.encode({"z": 1, "a": 2, "m": 3}) == '{"z":1,"a":2,"m":3}'


def test_forward_slashes_are_not_escaped() -> None:
    assert canonical.encode("https://example.com/x") == '"https://example.com/x"'


def test_non_ascii_is_not_escaped() -> None:
    assert canonical.encode({"t": "über — 日本語"}) == '{"t":"über — 日本語"}'


def test_integral_floats_render_without_a_fraction() -> None:
    # PHP and JavaScript render the float 1.0 as 1; Python's json renders 1.0.
    # Same JSON number, different bytes — normalised so the bytes agree.
    assert canonical.encode(1.0) == "1"
    assert canonical.encode({"temperature": 1.0}) == '{"temperature":1}'
    assert canonical.encode([1.0, -2.0, 0.0]) == "[1,-2,0]"


def test_non_integral_floats_are_untouched() -> None:
    assert canonical.encode({"temperature": 0.7}) == '{"temperature":0.7}'
    assert canonical.encode({"cost": 0.00125}) == '{"cost":0.00125}'


def test_none_is_a_real_null_and_is_never_dropped() -> None:
    assert canonical.encode({"max_output_tokens": None}) == '{"max_output_tokens":null}'


def test_absent_and_none_are_different_states() -> None:
    # The distinction lives in the caller: the encoder emits exactly the keys it
    # is handed.
    assert canonical.encode({}) == "{}"
    assert canonical.encode({"k": None}) == '{"k":null}'


def test_false_and_zero_survive() -> None:
    assert canonical.encode({"store": False, "temperature": 0}) == '{"store":false,"temperature":0}'


def test_booleans_do_not_become_integers() -> None:
    # bool is a subclass of int in Python; the integral-float rewrite must not
    # catch it.
    assert canonical.encode([True, False, 1, 0]) == "[true,false,1,0]"


def test_nested_structures_are_normalised_throughout() -> None:
    assert canonical.encode({"a": [{"b": 2.0}]}) == '{"a":[{"b":2}]}'


def test_objects_that_can_serialise_themselves_are_asked_to() -> None:
    class Thing:
        def to_dict(self) -> dict[str, float]:
            return {"n": 3.0}

    assert canonical.encode({"thing": Thing()}) == '{"thing":{"n":3}}'


def test_non_finite_floats_are_refused() -> None:
    with pytest.raises(ValueError):
        canonical.encode(float("nan"))


def test_unencodable_values_are_refused() -> None:
    with pytest.raises(TypeError):
        canonical.encode(object())
