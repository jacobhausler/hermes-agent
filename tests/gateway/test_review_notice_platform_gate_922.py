"""Background self-improvement notice resolves per platform (est-2ek.1.922).

The intake: a profile with ``display.platforms.discord.memory_notifications: off`` still got
the "💾 Self-improvement review: …" notice on Discord. The gateway turn read only the GLOBAL
``display.memory_notifications`` while the TUI read side documented per-platform support, so
the override was config-truth with no consumer (and Discord's tier had no default for the key
at all, so a bare global ``off`` also leaked into every other surface the operator wanted).

Pinned here:
  * discord's built-in tier is ``off`` (a bot seat's guild/thread is a shared surface — the
    review itself always runs; only its chat notice is silenced);
  * the canonical chain platform → display → tier → global applies to this key like every
    other display setting;
  * YAML booleans (`off:` parses as False in YAML 1.1) normalise to the tristate;
  * the gateway turn wiring — not just the resolver — is what sets ``agent.memory_notifications``,
    which ``background_review`` consumes for ``summarize_background_review_actions``.
"""

from __future__ import annotations

import types

from gateway.config import Platform
from gateway.display_config import resolve_display_setting
from gateway.run_turn_runner import TurnRunner

KEY = "memory_notifications"


def _setting(cfg, platform_value):
    return resolve_display_setting(cfg, platform_value, KEY, "on")


# ---------------------------------------------------------------------------
# Resolver: built-in tier + chain
# ---------------------------------------------------------------------------


def test_discord_tier_defaults_off_others_on():
    assert _setting({}, "discord") == "off"
    for platform in ("telegram", "signal", "whatsapp", "cli"):
        assert _setting({}, platform) == "on", platform


def test_explicit_platform_override_beats_discord_tier():
    cfg = {"display": {"platforms": {"discord": {"memory_notifications": "on"}}}}
    assert _setting(cfg, "discord") == "on"
    # and it does not bleed into other platforms
    assert _setting(cfg, "telegram") == "on"


def test_explicit_global_beats_discord_tier():
    """A deliberate global value is respected on discord too (same precedence law as
    show_reasoning/#121230 — explicit global beats platform tier)."""
    cfg = {"display": {"memory_notifications": "verbose"}}
    assert _setting(cfg, "discord") == "verbose"


def test_explicit_platform_off_beats_explicit_global():
    cfg = {
        "display": {
            "memory_notifications": "verbose",
            "platforms": {"telegram": {"memory_notifications": "off"}},
        }
    }
    assert _setting(cfg, "telegram") == "off"
    assert _setting(cfg, "whatsapp") == "verbose"


def test_yaml_booleans_normalise_to_tristate():
    """`off:`/`on:` are unquoted YAML booleans (1.1), and 'true'/'false' strings reach us
    from desktop config writers — all four must land on a real mode, never literal 'False'."""
    assert _setting({"display": {KEY: False}}, "telegram") == "off"
    assert _setting({"display": {KEY: True}}, "telegram") == "on"
    assert _setting({"display": {"platforms": {"discord": {KEY: True}}}}, "discord") == "on"
    assert _setting({"display": {KEY: "false"}}, "telegram") == "off"
    assert _setting({"display": {KEY: " OFF "}}, "telegram") == "off"


def test_unrecognised_value_falls_back_to_default():
    assert _setting({"display": {KEY: "loud"}}, "telegram") == "on"


def test_summary_off_emits_nothing():
    """The mode the turn sets must actually silence the notice at the summariser."""
    import json as _json

    from agent.background_review import summarize_background_review_actions

    review_messages = [
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_mem1",
                "function": {"name": "memory", "arguments": _json.dumps(
                    {"action": "add", "target": "memory", "content": "terse"})},
            }],
        },
        {
            "role": "tool", "tool_call_id": "call_mem1",
            "content": _json.dumps({"success": True, "message": "Entry added.", "target": "memory"}),
        },
    ]
    assert summarize_background_review_actions(review_messages, [], notification_mode="off") == []
    assert summarize_background_review_actions(review_messages, [], notification_mode="on") != []


# ---------------------------------------------------------------------------
# Gateway turn wiring: the consumer that was broken
# ---------------------------------------------------------------------------


def _wire(user_config, platform):
    """Run `_wire_turn_agent_callbacks` over minimal fakes; return the agent.

    Carries the same ``resolve_display_setting`` binding a real turn context has
    (run_turn.py binds the real resolver into the ctx).
    """
    from gateway.display_config import resolve_display_setting as real_resolver

    agent = types.SimpleNamespace()
    ctx = types.SimpleNamespace(
        progress_callback=None,
        native_tool_start_callback=None,
        voice_ack_callback=None,
        _voice_ack_guild=[None],
        _native_slack_task_cards=False,
        native_tool_complete_callback=None,
        _step_callback_sync=None,
        _hooks_ref=types.SimpleNamespace(loaded_hooks=[]),
        _status_callback_sync=None,
        _event_callback_sync=None,
        _status_adapter=None,
        session_key="",
        user_config=user_config,
        source=types.SimpleNamespace(platform=platform),
        resolve_display_setting=real_resolver,
        mute_notification_reply=False,
        _thinking_enabled=False,
        agent_holder=[None],
        tools_holder=[None],
        process_task_id=None,
        process_baseline=None,
        run_generation=0,
    )
    holder = types.SimpleNamespace(
        _ctx=ctx,
        _runner=types.SimpleNamespace(
            _service_tier=None,
            _consume_pending_turn_sidecar_notes=lambda key: [],
        ),
        _make_bg_review_callbacks=lambda: (lambda message: None, lambda: None),
        _merge_turn_request_overrides=TurnRunner._merge_turn_request_overrides,
        _clarify_callback_sync=lambda *a, **k: None,
        _notice_callback_sync=lambda *a, **k: None,
        _attach_session_title_callback=lambda agent, ctx: None,
    )
    TurnRunner._wire_turn_agent_callbacks(holder, agent, {}, None, None, None, False)
    return agent


def test_turn_wiring_silences_discord_notice_by_default():
    agent = _wire({}, Platform.DISCORD)
    assert agent.memory_notifications == "off"


def test_turn_wiring_honours_discord_platform_opt_in():
    cfg = {"display": {"platforms": {"discord": {KEY: "on"}}}}
    agent = _wire(cfg, Platform.DISCORD)
    assert agent.memory_notifications == "on"


def test_turn_wiring_honours_global_off_across_platforms():
    cfg = {"display": {KEY: "off"}}
    assert _wire(cfg, Platform.TELEGRAM).memory_notifications == "off"
    assert _wire(cfg, Platform.DISCORD).memory_notifications == "off"


def test_turn_wiring_keeps_other_platforms_loud():
    assert _wire({}, Platform.TELEGRAM).memory_notifications == "on"
    assert _wire({}, Platform.SIGNAL).memory_notifications == "on"


def test_turn_wiring_yaml_bool_off():
    cfg = {"display": {"platforms": {"telegram": {KEY: False}}}}
    assert _wire(cfg, Platform.TELEGRAM).memory_notifications == "off"
