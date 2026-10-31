"""Redirect-mode busy ack obeys the busy_steer_ack_enabled display gate (est-2ek.1.931).

A mid-run plain-text correction on a session whose runtime supports turn redirect
lands via ``_handle_active_session_busy_message`` in ``interrupt`` mode and
``redirect()``s the live turn; the busy-ack path then posts a visible
"↪ Redirected current run. I'll adjust using your correction." bubble. The
suppression gate checked ONLY ``is_steer_mode``, so a platform configured with
``display.platforms.<p>.busy_steer_ack_enabled: false`` (a Discord bot seat that
wants mid-run steering silent) still got the bubble in the guild channel —
steer and redirect are the same behavior ("keep the run, drop the bubble") and
the setting's documented intent covers both.

Fix (salvaged from tyfpro's #108963): extend the gate to ``is_redirect_mode``.
These tests pin the whole contract:
  * redirect-mode ack suppressed when the gate is off (no bubble at all — the
    first-time onboarding hint rides inside the composed message, so it leaks
    with it and must be gone too);
  * redirect-mode ack still sent when the gate is on (the fix must not
    over-suppress — the bubble is default-on);
  * the redirect itself is still applied when the bubble is suppressed;
  * a plain (non-redirected) interrupt ack is NOT covered by the steer gate.
"""
import sys
import time
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Minimal telegram stubs so gateway imports cleanly (mirrors sibling tests).
_tg = types.ModuleType("telegram")
_tg.constants = types.ModuleType("telegram.constants")
_ct = MagicMock()
_ct.SUPERGROUP = "supergroup"
_ct.GROUP = "group"
_ct.PRIVATE = "private"
_tg.constants.ChatType = _ct
sys.modules.setdefault("telegram", _tg)
sys.modules.setdefault("telegram.constants", _tg.constants)
sys.modules.setdefault("telegram.ext", types.ModuleType("telegram.ext"))

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SessionSource, build_session_key
from gateway.platforms.event import MessageEvent, MessageType


def _make_event(text: str = "actually use postgres"):
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
        user_id="user1",
    )
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source, message_id="msg1")


def _make_runner(display_config: dict):
    """Bare GatewayRunner whose gateway config resolves to ``display_config``."""
    import gateway.run as gateway_run

    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._busy_ack_ts = {}
    runner._draining = False
    runner._busy_input_mode = "interrupt"
    runner._busy_text_mode = "interrupt"
    runner.adapters = {}
    runner.session_store = None
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = True
    runner._is_user_authorized = lambda _source: True
    return runner


def _make_adapter():
    adapter = MagicMock()
    adapter._pending_messages = {}
    adapter._send_with_retry = AsyncMock()
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter.platform = Platform.TELEGRAM
    return adapter


def _make_redirect_agent():
    agent = MagicMock()
    agent._supports_active_turn_redirect = True
    agent.redirect = MagicMock(return_value=True)
    agent.get_activity_summary.return_value = {
        "api_call_count": 7,
        "max_iterations": 60,
        "current_tool": "terminal",
        "last_activity_ts": time.time(),
        "last_activity_desc": "terminal",
        "seconds_since_activity": 1.0,
    }
    return agent


def _display(platform_off: bool):
    if not platform_off:
        return {}
    return {"display": {"platforms": {"telegram": {"busy_steer_ack_enabled": False}}}}


async def _drive_redirect(runner, agent):
    event = _make_event()
    sk = build_session_key(event.source)
    runner._running_agents[sk] = agent
    runner._running_agents_ts[sk] = time.time() - 60
    adapter = _make_adapter()
    runner.adapters[event.source.platform] = adapter
    handled = await runner._handle_active_session_busy_message(event, sk)
    return handled, adapter, event, sk


@pytest.mark.asyncio
async def test_redirect_ack_suppressed_when_steer_ack_disabled(monkeypatch, tmp_path):
    """Gate off ⇒ NO bubble at all (redirect still applied, nothing sent)."""
    import gateway.run as _gr

    monkeypatch.delenv("HERMES_GATEWAY_BUSY_STEER_ACK_ENABLED", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", raising=False)
    monkeypatch.setattr(_gr, "_load_gateway_config", lambda *a, **k: _display(platform_off=True))
    monkeypatch.setattr(_gr, "_hermes_home", tmp_path)

    runner = _make_runner(_display(platform_off=True))
    agent = _make_redirect_agent()
    handled, adapter, _event, _sk = await _drive_redirect(runner, agent)

    assert handled is True
    agent.redirect.assert_called_once()          # behavior kept
    agent.interrupt.assert_not_called()          # not degraded to interrupt
    adapter._send_with_retry.assert_not_called()  # bubble (and its hint) gone


@pytest.mark.asyncio
async def test_redirect_ack_sent_when_gate_default_on(monkeypatch, tmp_path):
    """Gate at its default ⇒ the redirect ack posts with the redirect wording."""
    import gateway.run as _gr

    monkeypatch.delenv("HERMES_GATEWAY_BUSY_STEER_ACK_ENABLED", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", raising=False)
    monkeypatch.setattr(_gr, "_load_gateway_config", lambda *a, **k: {})
    monkeypatch.setattr(_gr, "_hermes_home", tmp_path)

    runner = _make_runner({})
    agent = _make_redirect_agent()
    handled, adapter, _event, _sk = await _drive_redirect(runner, agent)

    assert handled is True
    agent.redirect.assert_called_once()
    adapter._send_with_retry.assert_called_once()
    call_kwargs = adapter._send_with_retry.call_args
    content = call_kwargs.kwargs.get("content") or (call_kwargs.args[-1] if call_kwargs.args else "")
    assert "Redirected" in content


@pytest.mark.asyncio
async def test_plain_interrupt_ack_not_covered_by_steer_gate(monkeypatch, tmp_path):
    """Gate off suppresses steer/redirect echoes only — an interrupt that did
    NOT redirect still gets its ack (the run is genuinely being aborted)."""
    import gateway.run as _gr

    monkeypatch.delenv("HERMES_GATEWAY_BUSY_STEER_ACK_ENABLED", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", raising=False)
    monkeypatch.setattr(_gr, "_load_gateway_config", lambda *a, **k: _display(platform_off=True))
    monkeypatch.setattr(_gr, "_hermes_home", tmp_path)

    runner = _make_runner(_display(platform_off=True))
    agent = _make_redirect_agent()
    agent._supports_active_turn_redirect = False  # runtime without redirect ⇒ plain interrupt
    handled, adapter, _event, _sk = await _drive_redirect(runner, agent)

    assert handled is True
    agent.interrupt.assert_called_once()
    adapter._send_with_retry.assert_called_once()  # interrupt ack still delivered
