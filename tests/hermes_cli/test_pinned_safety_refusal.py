"""Explicit CLI model+provider pins must not re-bill a safety refusal on ambient fallbacks.

Regression for estate feedback 5ceb2abab8312004 (est-2ek.1.929).
"""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock
import sqlite3

import pytest

FALLBACK = [{"provider": "openrouter", "model": "other-model"}]


@pytest.mark.parametrize("model,provider,pinned", [
    ("gpt-test", "openai-codex", True),
    ("gpt-test", None, False),
    (None, "openai-codex", False),
    (None, None, False),
])
def test_cli_pin_cannot_borrow_or_hot_reload_ambient_chain(monkeypatch, model, provider, pinned):
    import cli
    from hermes_cli.config import get_config_path

    cfg = deepcopy(cli.CLI_CONFIG)
    cfg["model"] = {"default": "gpt-test", "provider": "openai-codex"}
    cfg["fallback_providers"] = FALLBACK
    monkeypatch.setattr(cli, "CLI_CONFIG", cfg)
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fallback_providers:\n  - provider: openrouter\n    model: other-model\n")
    shell = cli.HermesCLI(model=model, provider=provider, compact=True, max_turns=1)
    expected = [] if pinned else FALLBACK
    assert (shell._fallback_model or []) == expected
    agent = SimpleNamespace(
        _fallback_chain=[], _fallback_model=None, _fallback_index=0,
        _fallback_activated=False, _rate_limited_until=0, _unavailable_fallback_keys=set(),
    )
    shell._sync_fallback_chain_with_config(agent)
    assert agent._fallback_chain == expected
    assert (shell._fallback_model or []) == expected


@pytest.mark.parametrize("failure", ["soft", "incomplete", "exception"])
def test_pinned_oneshot_refusal_stops_once_and_keeps_session_evidence(tmp_path, monkeypatch, failure):
    import httpx
    import openai
    import run_agent
    from hermes_cli import oneshot
    from hermes_state import SessionDB

    db_path = tmp_path / "state.db"
    cfg = {
        "model": {"default": "ambient-model", "provider": "openrouter"},
        "fallback_providers": FALLBACK,
        "fallback_model": {"provider": "anthropic", "model": "legacy-model"},
    }
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    monkeypatch.setattr(oneshot, "_create_session_db_for_oneshot", lambda: SessionDB(db_path))
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_: [])
    monkeypatch.setattr("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build", lambda **_: None)
    monkeypatch.setattr(oneshot, "_linger_for_background_completions", lambda: None)
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **_: [])
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda **_: {})
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", MagicMock())
    monkeypatch.setattr(run_agent.AIAgent, "_create_request_openai_client", lambda *_, **__: MagicMock())
    resolved = []

    def resolve(**kw):
        resolved.append(kw["requested"])
        return {
            "provider": kw["requested"], "requested_provider": kw["requested"],
            "api_mode": "codex_responses", "api_key": "test-key-1234567890",
            "base_url": "https://chatgpt.com/backend-api/codex", "credential_pool": None,
        }

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve)
    agents = []
    request_count = []
    refusal = "Your request was rejected by our safety system."

    def stream(agent, *_a, **kw):
        agents.append(agent)
        request_count.append(kw)
        # Make a forbidden second attempt finish promptly, rather than burn a backoff ladder.
        if len(request_count) > 1:
            return SimpleNamespace(
                status="completed", incomplete_details=None, usage=None, model="gpt-test",
                output=[SimpleNamespace(type="message", status="completed", content=[
                    SimpleNamespace(type="output_text", text="Unexpected second attempt"),
                ])],
            )
        if failure == "exception":
            raise openai.BadRequestError(
                refusal,
                response=httpx.Response(400, request=httpx.Request("POST", "https://chatgpt.com/backend-api/codex")),
                body={"error": {"code": "content_policy_violation", "message": refusal}},
            )
        return SimpleNamespace(
            status="failed" if failure == "soft" else "incomplete", output=[], output_text="",
            model="gpt-test", usage=None,
            error=SimpleNamespace(code="content_policy_violation", message=refusal) if failure == "soft" else None,
            incomplete_details=SimpleNamespace(reason="content_filter") if failure == "incomplete" else None,
        )

    monkeypatch.setattr(run_agent.AIAgent, "_run_codex_stream", stream)
    text, result = oneshot._run_agent(
        "Reply briefly.", model="gpt-test", provider="openai-codex", toolsets=[], use_config_toolsets=False,
    )
    assert len(request_count) == 1
    assert resolved == ["openai-codex"]
    assert agents[0]._fallback_chain == []
    assert result["failed"] is True and result["failure_retryable"] is False
    assert result["failure_reason"] == "content_policy_blocked"
    assert "safety filter refused" in text
    if failure != "incomplete":
        assert refusal in text
    with sqlite3.connect(db_path) as db:
        row = db.execute(
            "SELECT model,billing_provider,last_activity_provenance,last_activity_description FROM sessions"
        ).fetchone()
    assert row[:3] == ("gpt-test", "openai-codex", "agent.content_policy_blocked")
    assert "safety refusal" in row[3].lower()
    assert cfg["fallback_providers"] == FALLBACK  # pinning is run-local, not a config mutation
