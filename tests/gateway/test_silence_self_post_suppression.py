"""Out-of-band same-thread self-post suppresses the unexpected-silence notice (est-ro981).

A human turn that ends on a bare silence marker owes a visible notice — UNLESS the agent already
answered that same trigger in-thread through another door (final-answers-only display policy kept
the turn silent while the answer shipped via CLI/API). The receipt is the optional adapter
capability ``has_self_post_since``; per the sibling est-2ek.1.503 ruling the suppression may fire
ONLY on a positively-delivered bot post newer than the trigger, never on failure-with-nothing-
delivered. These tests pin all three delivery-shaped verdicts across the normal lane, the queued
twin, and the crash-left mirror — with fake adapters only, no network.
"""

import time
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.run_turn import _unexpected_silence_reply
from gateway.platforms.base import self_post_answered_since
from gateway.platforms.event import MessageEvent
from gateway.session import SessionEntry, SessionSource
from gateway.response_filters import is_intentional_silence_response

TRIGGER_ID = "900000000000000001"


class _SelfPostAdapter:
    """Minimal delivery adapter carrying the optional capability, recording its calls."""

    def __init__(self, *, verdict=False, raises=None):
        self._verdict = verdict
        self._raises = raises
        self.calls = []
        self.sent = []

    async def has_self_post_since(self, *, chat_id, thread_id=None, after_id=None, after_ts=None):
        self.calls.append(dict(chat_id=chat_id, thread_id=thread_id, after_id=after_id, after_ts=after_ts))
        if self._raises is not None:
            raise self._raises
        return self._verdict

    async def send(self, chat_id, content, *, metadata=None, **_kw):
        self.sent.append(content)
        return SimpleNamespace(success=True, message_id="m-1")


class _PlainAdapter:
    """Adapter WITHOUT the capability — the default-False surface every other platform ships."""

    def __init__(self):
        self.sent = []

    async def send(self, chat_id, content, *, metadata=None, **_kw):
        self.sent.append(content)
        return SimpleNamespace(success=True, message_id="m-1")


def _source(*, message_id=TRIGGER_ID):
    return SessionSource(
        platform=Platform.DISCORD, chat_id="c-1", chat_type="channel", user_id="u-1",
        message_id=message_id,
    )


def _runner(monkeypatch, tmp_path, adapter):
    runner = gateway_run.GatewayRunner(GatewayConfig())
    runner.adapters = {Platform.DISCORD: adapter}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._handle_active_session_busy_message = AsyncMock(return_value=False)
    runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _gen: True
    runner._reply_anchor_for_event = lambda _event: None
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_a, **_kw: False
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:discord:channel:c-1:u-1", session_id="sess-oor",
        created_at=datetime.now(), updated_at=datetime.now(),
        platform=Platform.DISCORD, chat_type="channel",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length", lambda *_a, **_kw: 100_000)
    return runner


def _silent_result():
    return {
        "final_response": "[SILENT]",
        "messages": [{"role": "user", "content": "question"},
                     {"role": "assistant", "content": "[SILENT]"}],
        "tools": [], "history_offset": 0, "last_prompt_tokens": 0,
        "api_calls": 1, "failed": False,
    }


async def _drive_human_turn(monkeypatch, tmp_path, adapter):
    runner = _runner(monkeypatch, tmp_path, adapter)
    runner._run_agent = AsyncMock(return_value=_silent_result())
    event = MessageEvent(
        text="question", source=_source(), message_id=TRIGGER_ID, reply_expected=True)
    response = await runner._handle_message_with_agent(
        event, _source(), "agent:main:discord:channel:c-1:u-1", 1)
    return runner, response


def _warnings(caplog):
    return [r for r in caplog.records
            if r.levelname == "WARNING" and "silence marker" in r.message]


@pytest.mark.asyncio
async def test_answered_turn_suppresses_the_notice(monkeypatch, tmp_path, caplog):
    """Self-post newer than the trigger in the same thread = an answered turn: no delivery,
    no warning — the substantive answer already reached the human."""
    adapter = _SelfPostAdapter(verdict=True)
    with caplog.at_level("DEBUG"):
        _runner_unused, response = await _drive_human_turn(monkeypatch, tmp_path, adapter)
    assert response == ""
    assert _unexpected_silence_reply() not in adapter.sent  # the notice never reaches the chat
    assert not _warnings(caplog)
    assert any("answered out-of-band" in r.message for r in caplog.records)
    call = adapter.calls[-1]
    assert call["chat_id"] == "c-1"
    assert call["after_id"] == TRIGGER_ID
    assert isinstance(call["after_ts"], float)


@pytest.mark.asyncio
async def test_unanswered_turn_keeps_the_notice(monkeypatch, tmp_path, caplog):
    """Capability present, no post found: the human-facing warning is unchanged."""
    adapter = _SelfPostAdapter(verdict=False)
    with caplog.at_level("DEBUG"):
        _runner_unused, response = await _drive_human_turn(monkeypatch, tmp_path, adapter)
    assert response == _unexpected_silence_reply()
    assert _warnings(caplog)


@pytest.mark.asyncio
async def test_adapter_without_capability_keeps_the_notice(monkeypatch, tmp_path, caplog):
    """Default False everywhere: a platform that never implements the capability behaves exactly
    as before this fix — the warning fires on a silent human turn."""
    adapter = _PlainAdapter()
    with caplog.at_level("DEBUG"):
        _runner_unused, response = await _drive_human_turn(monkeypatch, tmp_path, adapter)
    assert response == _unexpected_silence_reply()
    assert _warnings(caplog)


@pytest.mark.asyncio
async def test_capability_error_keeps_the_notice(monkeypatch, tmp_path, caplog):
    """A failing receipt lookup is not evidence of delivery — fail closed to the warning."""
    adapter = _SelfPostAdapter(raises=RuntimeError("REST 500"))
    with caplog.at_level("DEBUG"):
        _runner_unused, response = await _drive_human_turn(monkeypatch, tmp_path, adapter)
    assert response == _unexpected_silence_reply()
    assert _warnings(caplog)


# ---------------------------------------------------------------- queued-lane twin


def _turn_ctx(**over):
    base = dict(
        mute_notification_reply=False, session_key="agent:main:discord:channel:c-1:u-1",
        stream_consumer_holder=[None], source=_source(), persist_user_display_kind=None,
        reply_expected=True, inbound_message_id=TRIGGER_ID, persist_user_timestamp=1000.0,
        _status_thread_metadata=None, event_message_id=TRIGGER_ID, run_generation=1,
        session_id="sess-oor", history=None, interrupt_depth=0,
    )
    base.update(over)
    return SimpleNamespace(**base)


@pytest.mark.asyncio
async def test_queued_lane_suppresses_on_self_post(monkeypatch, tmp_path):
    adapter = _SelfPostAdapter(verdict=True)
    runner = _runner(monkeypatch, tmp_path, adapter)
    runner._deliver_queued_first_response = AsyncMock()
    await runner._run_agent_deliver_first_response(
        _turn_ctx(), adapter, {"final_response": "[SILENT]", "failed": False}, None, None)
    runner._deliver_queued_first_response.assert_not_awaited()
    call = adapter.calls[-1]
    assert call["after_id"] == TRIGGER_ID
    assert call["after_ts"] == 1000.0


@pytest.mark.asyncio
async def test_queued_lane_warns_without_self_post(monkeypatch, tmp_path):
    adapter = _SelfPostAdapter(verdict=False)
    runner = _runner(monkeypatch, tmp_path, adapter)
    runner._deliver_queued_first_response = AsyncMock(return_value=True)
    await runner._run_agent_deliver_first_response(
        _turn_ctx(), adapter, {"final_response": "[SILENT]", "failed": False}, None, None)
    runner._deliver_queued_first_response.assert_awaited_once()
    assert runner._deliver_queued_first_response.await_args.args[0] == \
        _unexpected_silence_reply()


@pytest.mark.asyncio
async def test_queued_lane_plain_adapter_warns(monkeypatch, tmp_path):
    adapter = _PlainAdapter()
    runner = _runner(monkeypatch, tmp_path, adapter)
    runner._deliver_queued_first_response = AsyncMock(return_value=True)
    await runner._run_agent_deliver_first_response(
        _turn_ctx(), adapter, {"final_response": "[SILENT]", "failed": False}, None, None)
    runner._deliver_queued_first_response.assert_awaited_once()


# ---------------------------------------------------------------- crash-left mirror


def _bare_runner(adapter):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {Platform.DISCORD: adapter} if adapter is not None else {}
    return runner


def _crash_history(reply="NO_REPLY"):
    now = time.time()
    return ([{"role": "user", "content": "question", "timestamp": now - 5},
            {"role": "assistant", "content": reply, "timestamp": now - 1}], now - 10)


@pytest.mark.asyncio
async def test_crash_left_mirror_suppresses_on_self_post():
    """Same scenario at recovery: with a connected adapter confirming the post, the crash-left
    pass owes nothing — parity with the live suppression."""
    adapter = _SelfPostAdapter(verdict=True)
    runner = _bare_runner(adapter)
    history, started = _crash_history()
    text = await runner._crash_left_reply(history, started, _source())
    assert text == ""
    call = adapter.calls[-1]
    assert call["after_ts"] == started  # recovery floor: turn-marker start, no message id


@pytest.mark.asyncio
async def test_crash_left_mirror_keeps_notice_without_adapter():
    """Recovery usually runs before the adapter connects: fail closed, notice stands."""
    runner = _bare_runner(None)
    history, started = _crash_history()
    text = await runner._crash_left_reply(history, started, _source())
    assert text == _unexpected_silence_reply()


@pytest.mark.asyncio
async def test_crash_left_mirror_plain_adapter_keeps_notice():
    runner = _bare_runner(_PlainAdapter())
    history, started = _crash_history()
    text = await runner._crash_left_reply(history, started, _source())
    assert text == _unexpected_silence_reply()


# ---------------------------------------------------------------- capability helper


@pytest.mark.asyncio
async def test_helper_false_without_adapter_or_floor():
    assert await self_post_answered_since(None, _source(), after_id="1", after_ts=1.0) is False
    assert await self_post_answered_since(_PlainAdapter(), _source(), after_id="1") is False
    adapter = _SelfPostAdapter(verdict=True)
    # No floor supplied: no causal bound exists to prove "newer than the trigger" — False.
    assert await self_post_answered_since(adapter, _source()) is False
    # Source without a chat: nothing to scan.
    assert await self_post_answered_since(
        adapter, SimpleNamespace(chat_id=None, thread_id=None), after_ts=1.0) is False


@pytest.mark.asyncio
async def test_helper_accepts_sync_capability():
    class _Sync:
        def has_self_post_since(self, **_kw):
            return True
    assert await self_post_answered_since(_Sync(), _source(), after_ts=1.0) is True
