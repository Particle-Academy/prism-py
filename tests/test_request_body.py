"""Building the OpenAI request body, and the builder that feeds it."""

from __future__ import annotations

import pytest

from prism import (
    PendingRequest,
    Prism,
    PrismError,
    ProviderTool,
    Tool,
    ToolChoice,
    UserMessage,
    build_request_body,
    canonical,
)


def encoded(pending: PendingRequest) -> str:
    return canonical.encode(build_request_body(pending.to_request()))


def base() -> PendingRequest:
    return Prism.text().using("openai", "gpt-4o")


# -- the unconditional keys -------------------------------------------------


def test_max_output_tokens_is_present_even_when_never_set() -> None:
    # The single most important row in the corpus: a port that models "unset"
    # as "absent" drops the key and the bytes differ.
    assert encoded(base().with_prompt("Who are you?")) == (
        '{"model":"gpt-4o","input":[{"role":"user","content":'
        '[{"type":"input_text","text":"Who are you?"}]}],"max_output_tokens":null}'
    )


def test_max_output_tokens_carries_its_value_when_set() -> None:
    assert '"max_output_tokens":256' in encoded(base().with_prompt("hi").with_max_tokens(256))


# -- the not-null filter ----------------------------------------------------


def test_temperature_zero_survives() -> None:
    # Zero is falsy in PHP and JavaScript; the filter is on null, not falsiness.
    # Dropping it makes the model sample at its default instead.
    assert '"temperature":0' in encoded(base().with_prompt("hi").using_temperature(0))


def test_temperature_is_omitted_when_never_set() -> None:
    assert "temperature" not in encoded(base().with_prompt("hi"))


def test_top_p_uses_the_snake_case_wire_name() -> None:
    assert '"top_p":0.9' in encoded(base().with_prompt("hi").using_top_p(0.9))


def test_a_false_provider_option_reaches_the_wire() -> None:
    # Turning response storage off is the entire point of setting it.
    encoded_body = encoded(
        base().with_prompt("hi").with_provider_options({"store": False, "service_tier": "flex"})
    )

    assert encoded_body.endswith('"service_tier":"flex","store":false}')


def test_provider_option_keys_keep_the_reference_order() -> None:
    encoded_body = encoded(
        base()
        .with_prompt("hi")
        .with_provider_options({"store": True, "service_tier": "flex", "truncation": "auto"})
    )

    assert encoded_body.index('"service_tier"') < encoded_body.index('"store"')
    assert encoded_body.index('"store"') < encoded_body.index('"truncation"')


# -- tools ------------------------------------------------------------------


def test_an_empty_tool_list_omits_the_tools_key_entirely() -> None:
    # An empty list is falsy in PHP and truthy in Python. Sending `tools: []`
    # is rejected by some models and changes tool_choice defaults on others.
    assert "tools" not in encoded(base().with_prompt("hi").with_tools([]))


def test_a_tool_with_a_required_parameter_carries_its_schema() -> None:
    tool = Tool("weather", "Get the weather").with_string_parameter("city", "The city")

    assert encoded(base().with_prompt("hi").with_tools([tool])).endswith(
        '"tools":[{"type":"function","name":"weather","description":"Get the weather",'
        '"parameters":{"type":"object","properties":'
        '{"city":{"description":"The city","type":"string"}},"required":["city"]}}]}'
    )


def test_a_tool_with_no_parameters_omits_both_parameters_and_strict() -> None:
    assert encoded(base().with_prompt("hi").with_tools([Tool("ping", "Ping")])).endswith(
        '"tools":[{"type":"function","name":"ping","description":"Ping"}]}'
    )


def test_strict_appears_only_when_truthy() -> None:
    # Filtered on FALSINESS, unlike the provider options above. The
    # inconsistency is the reference's and is reproduced deliberately.
    tool = Tool("ping", "Ping").with_provider_options({"strict": True})
    assert '"strict":true' in encoded(base().with_prompt("hi").with_tools([tool]))

    off = Tool("ping", "Ping").with_provider_options({"strict": False})
    assert "strict" not in encoded(base().with_prompt("hi").with_tools([off]))


def test_the_tool_filter_uses_php_falsiness() -> None:
    # `array_filter` in the reference drops "0" along with "" and 0, because PHP
    # considers the string zero false and Python does not. No golden pins this,
    # but identity wins where the corpus is silent — a divergence here would be
    # a divergence in the bytes.
    encoded_body = encoded(base().with_prompt("hi").with_tools([Tool("0", "Zero")]))

    assert '{"type":"function","description":"Zero"}' in encoded_body


def test_tool_declaration_order_is_preserved() -> None:
    tools = [Tool("zebra", "Z"), Tool("alpha", "A")]
    encoded_body = encoded(base().with_prompt("hi").with_tools(tools))

    assert encoded_body.index('"zebra"') < encoded_body.index('"alpha"')


def test_provider_tools_are_merged_in_front_of_user_tools() -> None:
    encoded_body = encoded(
        base()
        .with_prompt("hi")
        .with_tools([Tool("ping", "Ping")])
        .with_provider_tools([ProviderTool("web_search_preview")])
    )

    assert encoded_body.endswith(
        '"tools":[{"type":"web_search_preview"},'
        '{"type":"function","name":"ping","description":"Ping"}]}'
    )


def test_provider_tool_options_are_spread_alongside_the_type() -> None:
    encoded_body = encoded(
        base()
        .with_prompt("hi")
        .with_provider_tools([ProviderTool("web_search_preview", options={"depth": 2})])
    )

    assert '{"type":"web_search_preview","depth":2}' in encoded_body


# -- tool choice ------------------------------------------------------------


def test_tool_choice_auto() -> None:
    pending = base().with_prompt("hi").with_tools([Tool("ping", "P")])
    assert '"tool_choice":"auto"' in encoded(pending.with_tool_choice(ToolChoice.AUTO))


def test_tool_choice_any_maps_to_required() -> None:
    # The one member whose wire name differs from its own. "any" is rejected.
    pending = base().with_prompt("hi").with_tools([Tool("ping", "P")])
    assert '"tool_choice":"required"' in encoded(pending.with_tool_choice(ToolChoice.ANY))


def test_tool_choice_none() -> None:
    pending = base().with_prompt("hi").with_tools([Tool("ping", "P")])
    assert '"tool_choice":"none"' in encoded(pending.with_tool_choice(ToolChoice.NONE))


def test_tool_choice_by_name_becomes_an_object() -> None:
    pending = base().with_prompt("hi").with_tools([Tool("ping", "P")])
    assert '"tool_choice":{"type":"function","name":"ping"}' in encoded(
        pending.with_tool_choice("ping")
    )


def test_tool_choice_accepts_a_tool() -> None:
    tool = Tool("ping", "P")
    pending = base().with_prompt("hi").with_tools([tool]).with_tool_choice(tool)

    assert '"tool_choice":{"type":"function","name":"ping"}' in encoded(pending)


# -- reasoning --------------------------------------------------------------


def test_reasoning_disabled_asks_for_minimal_effort() -> None:
    assert '"reasoning":{"effort":"minimal"}' in encoded(
        base().with_prompt("hi").with_reasoning(False)
    )


def test_reasoning_enabled_emits_nothing() -> None:
    # Asymmetric on purpose: true must not override a per-provider setting.
    assert "reasoning" not in encoded(base().with_prompt("hi").with_reasoning(True))


def test_an_explicit_reasoning_provider_option_wins() -> None:
    encoded_body = encoded(
        base()
        .with_prompt("hi")
        .with_reasoning(False)
        .with_provider_options({"reasoning": {"effort": "high"}})
    )

    assert '"reasoning":{"effort":"high"}' in encoded_body


def test_text_verbosity_becomes_a_text_block() -> None:
    encoded_body = encoded(
        base().with_prompt("hi").with_provider_options({"text_verbosity": "low"})
    )

    assert '"text":{"verbosity":"low"}' in encoded_body


# -- prompts and messages ---------------------------------------------------


def test_system_prompts_accumulate_rather_than_replace() -> None:
    pending = base().with_system_prompt("A").with_system_prompt("B").with_prompt("hi")

    assert encoded(pending) == (
        '{"model":"gpt-4o","input":[{"role":"system","content":"A"},'
        '{"role":"system","content":"B"},'
        '{"role":"user","content":[{"type":"input_text","text":"hi"}]}],'
        '"max_output_tokens":null}'
    )


def test_an_empty_prompt_produces_a_normal_user_message() -> None:
    # A deliberate divergence from the reference, which gates the prompt path on
    # PHP truthiness and drops the turn entirely.
    assert '"text":""' in encoded(base().with_prompt(""))


def test_a_prompt_that_is_the_string_zero_produces_a_normal_user_message() -> None:
    # "0" is falsy in PHP and is ordinary model input. The reference drops it.
    assert '"text":"0"' in encoded(base().with_prompt("0"))


def test_prompt_and_messages_together_are_refused() -> None:
    pending = base().with_prompt("Who are you?").with_messages([UserMessage("hi")])

    with pytest.raises(PrismError) as raised:
        pending.to_request()

    assert raised.value.code == "prompt_and_messages"


def test_a_falsy_prompt_alongside_messages_still_refuses() -> None:
    # Same refusal, with a prompt the reference's truthiness guard lets past.
    for prompt in ("0", ""):
        with pytest.raises(PrismError) as raised:
            base().with_prompt(prompt).with_messages([UserMessage("hi")]).to_request()

        assert raised.value.code == "prompt_and_messages"


def test_non_ascii_and_forward_slashes_are_not_escaped() -> None:
    prompt = "Explain https://example.com/über — 日本語 too"

    assert prompt in encoded(base().with_prompt(prompt))


# -- the builder itself -----------------------------------------------------


def test_the_builder_records_every_setting() -> None:
    request = (
        Prism.text()
        .using("openai", "gpt-4o")
        .with_prompt("hi")
        .with_max_tokens(10)
        .using_temperature(0.5)
        .using_top_p(0.9)
        .using_top_k(3)
        .with_max_steps(4)
        .with_client_options({"timeout": 1})
        .to_request()
    )

    assert request.model == "gpt-4o"
    assert request.provider_key == "openai"
    assert (request.max_tokens, request.temperature, request.top_p, request.top_k) == (
        10,
        0.5,
        0.9,
        3,
    )
    assert request.max_steps == 4
    assert request.client_options == {"timeout": 1}


def test_an_unknown_provider_is_refused_with_a_code() -> None:
    with pytest.raises(PrismError) as raised:
        Prism.text().using("nonesuch", "m")

    assert raised.value.code == "unsupported_provider_action"


def test_as_text_without_a_provider_is_refused_with_a_code() -> None:
    with pytest.raises(PrismError) as raised:
        Prism.text().with_prompt("hi").as_text()

    assert raised.value.code == "unsupported_provider_action"
