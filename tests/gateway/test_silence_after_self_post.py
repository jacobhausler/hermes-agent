"""Unexpected-silence warning is suppressed when the bot already answered in-thread."""

from unittest.mock import AsyncMock

import pytest

from gateway.response_filters import is_intentional_silence_response
from tests.gateway.test_gateway_silence_tokens import _event, _runner, _source

_SESSION_KEY = "agent:main:telegram:group:-1001:12345"


class _SelfPostAdapter:
    """Adapter stand-in exposing only the read-back capability under test."""

    def __init__(self, answer: bool):
        self.answer = answer
        self.calls = []

    async def has_self_post_since(self, chat_id, thread_id, since_ts):
        self.calls.append((chat_id, thread_id, since_ts))
        return self.answer


class _NoCapabilityAdapter:
    """Adapter stand-in without ``has_self_post_since`` at all."""


def _silent_result():
    return {
        "final_response": "[SILENT]",
        "messages": [
            {"role": "user", "content": "side chatter"},
            {"role": "assistant", "content": "[SILENT]"},
        ],
        "tools": [],
        "history_offset": 0,
        "last_prompt_tokens": 0,
        "api_calls": 1,
        "failed": False,
    }


async def _run(monkeypatch, tmp_path, caplog, adapter):
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value=_silent_result())
    runner._adapter_for_source = lambda source: adapter
    with caplog.at_level("DEBUG"):
        response = await runner._handle_message_with_agent(
            _event(reply_expected=True), _source(), _SESSION_KEY, 1
        )
    warned = any(
        record.levelname == "WARNING" and "silence marker rejected" in record.message
        for record in caplog.records
    )
    return response, warned


@pytest.mark.asyncio
async def test_self_post_confirmed_suppresses_warning(monkeypatch, tmp_path, caplog):
    adapter = _SelfPostAdapter(True)
    response, warned = await _run(monkeypatch, tmp_path, caplog, adapter)
    assert adapter.calls, "the guard must consult the adapter"
    chat_id, _thread_id, since_ts = adapter.calls[0]
    assert chat_id == "-1001" and since_ts > 0
    assert not warned
    assert not response or is_intentional_silence_response(response)


@pytest.mark.asyncio
async def test_no_self_post_keeps_warning(monkeypatch, tmp_path, caplog):
    adapter = _SelfPostAdapter(False)
    response, warned = await _run(monkeypatch, tmp_path, caplog, adapter)
    assert adapter.calls
    assert warned
    assert response and not is_intentional_silence_response(response)


@pytest.mark.asyncio
async def test_missing_capability_keeps_warning(monkeypatch, tmp_path, caplog):
    response, warned = await _run(monkeypatch, tmp_path, caplog, _NoCapabilityAdapter())
    assert warned
    assert response and not is_intentional_silence_response(response)
