"""Turn-wide TOTAL bounds on the provider-outage recovery loop, and true totals in the banner.

The post-exhaustion auto-recovery ladder caps ``agent.auto_recovery_cycles``, but that counter
lives on the per-attempt ``TurnRetryState``: any steer/re-entry that re-enters the loop resets
it, so a dead provider can burn tens of minutes and dozens of attempts while the failure banner
reports the LAST cycle's count ("3 attempts" over an observed 1652.2s / 37-attempt outage).

Contracts here:
* ``agent.auto_recovery_max_wall_seconds`` / ``agent.auto_recovery_max_attempts`` (env
  HERMES_AUTO_RECOVERY_MAX_WALL_S / HERMES_AUTO_RECOVERY_MAX_ATTEMPTS) bound the loop by
  TOTALS accumulated across ALL recovery/steer/re-entry cycles of one turn;
* the terminal banner reports those true totals, never the last cycle's count;
* an always-timeout fake provider stops at the bound without touching anything else;
* a healthy fake provider passes through untouched by the bounds.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.error_classifier import classify_api_error
from agent.turn_api_error import settle_unrecovered_error
from agent.turn_recovery import max_retries_exhausted_result
from agent.turn_retry_state import TurnRetryState


class _Err(Exception):
    def __init__(self, status_code, message, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.response = SimpleNamespace(headers=headers or {})
        self.body = None


class _Agent:
    """Real seams: fallback state, the status/wait callbacks, the streamed-text gate; the
    recovery bounds and ledger attrs the ladder reads are set explicitly (no lambda defaults)."""
    log_prefix = ""
    verbose = False
    provider = "custom"
    platform = "cli"
    _credential_pool = None
    _current_streamed_assistant_text = ""
    _interrupt_requested = False
    sleep_outcome = None

    def __init__(self, *, cycles=99, max_wall_s=600.0, max_attempts=10.0):
        self.statuses, self.waits, self.activated = [], [], []
        self._auto_recovery_cycles = cycles
        self._auto_recovery_max_wall_s = max_wall_s
        self._auto_recovery_max_attempts = max_attempts
        self._recovery_totals = None

    def _has_pending_fallback(self):
        return False

    def _try_activate_fallback(self, **kwargs):
        self.activated.append(True)
        return False

    def _try_recover_primary_transport(self, *a, **k):
        return False

    def _has_content_after_think_block(self, text):
        return bool(text.strip())

    def _emit_diagnostic_status(self, text):
        self.statuses.append(str(text))

    def _emit_diagnostic_wait(self, text):
        self.waits.append(str(text))

    def _summarize_api_error(self, error):
        return str(error)

    def _client_log_context(self):
        return ""

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _settle(agent, err, retry, result_kwargs):
    """Drive one exhaustion through the REAL error-settling path; record the terminal
    result-builder kwargs; simulate the loop's per-iteration TurnRetryState reset by having
    the caller pass a FRESH retry state per cycle (the steer/re-entry shape)."""
    classified = classify_api_error(err, provider=agent.provider)

    def _capture(*args, **kwargs):
        result_kwargs.append(kwargs)
        return {"failed": True}

    with patch("agent.conversation_loop._is_copilot_provider", lambda a: False), \
            patch("agent.conversation_loop._arm_fallback_restart", lambda a, m, s, r: s), \
            patch("agent.turn_api_error.max_retries_exhausted_result", _capture), \
            patch("agent.turn_recovery.interruptible_backoff_sleep", lambda *a, **k: None):
        return settle_unrecovered_error(
            agent, api_error=err, classified=classified, _retry=retry, status_code=err.status_code,
            error_msg=str(err).lower(),
            is_context_length_error=False, is_rate_limited=False,
            _is_zai_coding_overload=False, _provider=agent.provider, _base="http://fake", _model="m",
            messages=[], api_messages=[], api_kwargs=None, active_system_prompt=None,
            conversation_history=None, approx_tokens=0, retry_count=3, max_retries=3,
            compression_attempts=0, api_call_count=1,
        )


def _clock(monkeypatch, step_s):
    """Deterministic monotonic clock for the recovery ledger: one tick per call."""
    from agent import turn_recovery_autorecover as ladder
    ticks = iter(range(0, 10_000_000, step_s))
    monkeypatch.setattr(ladder, "_now", lambda: float(next(ticks)))


def _run_outage(agent, monkeypatch, *, step_s, cycles_max):
    """Repeat the exhausted-503 gate until it stops asking to continue (each pass gets a
    fresh TurnRetryState, exactly what a steer/re-entry restart gives the loop). Returns the
    terminal result kwargs and the number of recovery cycles that were announced."""
    monkeypatch.setattr("agent.retry_utils.jittered_backoff", lambda *a, **k: 0.0)
    _clock(monkeypatch, step_s)
    result_kwargs: list[dict] = []
    verdicts = []
    for _ in range(cycles_max):
        verdict = _settle(agent, _Err(503, "Service temporarily unavailable"), TurnRetryState(
            primary_recovery_attempted=True), result_kwargs)
        verdicts.append(verdict.action)
        if verdict.action != "continue":
            break
    return result_kwargs, verdicts


def test_attempt_bound_stops_loop_across_steer_reentries(monkeypatch):
    """3 attempts per exhaustion x 3 exhaustions = 9 >= the 9-attempt bound: the loop must
    stop at gate 3 even though cycles (99) and wall (600s) still allow far more."""
    agent = _Agent(cycles=99, max_wall_s=600.0, max_attempts=9.0)
    result_kwargs, verdicts = _run_outage(agent, monkeypatch, step_s=100, cycles_max=25)

    assert verdicts == ["continue", "continue", "return"]
    assert len(result_kwargs) == 1
    assert result_kwargs[0]["recovery_attempts"] == 9, "banner must carry the TRUE total"
    assert result_kwargs[0]["recovery_wall_s"] == 200.0, "gate-to-gate wall across 3 exhaustion gates"
    give_up = [s for s in agent.statuses if "gave up" in s]
    assert give_up and "9 attempts / 200s" in give_up[0], agent.statuses


def test_wall_bound_stops_loop_even_with_cycles_and_attempts_remaining(monkeypatch):
    """A 150s wall bound with cycles and attempts still in reserve must end the loop."""
    agent = _Agent(cycles=99, max_wall_s=150.0, max_attempts=1000.0)
    result_kwargs, verdicts = _run_outage(agent, monkeypatch, step_s=100, cycles_max=25)

    assert verdicts == ["continue", "continue", "return"]
    assert result_kwargs[0]["recovery_attempts"] == 9
    assert result_kwargs[0]["recovery_wall_s"] == 200.0
    give_up = [s for s in agent.statuses if "gave up" in s]
    assert give_up and "150s" in give_up[0], agent.statuses


class _SurfaceAgent(_Agent):
    """Enough real surface for the untouched terminal-result builder."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.model = "m"
        self.base_url = "http://fake"

    def _flush_status_buffer(self):
        pass

    def _persist_session(self, *a, **k):
        pass


def test_failure_banner_reports_true_totals_not_the_last_cycle_count():
    """The observed incident shape: 37 attempts over 1652.2s must never surface as '3 retries'."""
    agent = _SurfaceAgent()
    err = _Err(503, "Service temporarily unavailable")
    result = max_retries_exhausted_result(
        agent, err, classify_api_error(err, provider=agent.provider),
        attempts=3, is_rate_limited=False, error_msg="service temporarily unavailable",
        api_kwargs=None, api_messages=[], messages=[], conversation_history=None,
        api_call_count=1, approx_tokens=10, provider="custom", base_url="http://fake",
        model="m", recovery_attempts=37, recovery_wall_s=1652.2,
    )
    banner = [s for s in agent.statuses if s.startswith("❌")]
    assert banner and "after 37 retries over 1652.2s" in banner[0], agent.statuses
    assert "after 3 retries" not in "".join(agent.statuses)
    text = result["final_response"]
    assert "37 attempts" in text and "1652s" in text, text


def test_no_recovery_run_keeps_the_banner_unchanged():
    """Control at the copy level: without a recovery run the terminal banner is byte-for-byte
    the legacy line — the bounds must not touch any other failure path."""
    agent = _SurfaceAgent()
    err = _Err(503, "Service temporarily unavailable")
    result = max_retries_exhausted_result(
        agent, err, classify_api_error(err, provider=agent.provider),
        attempts=3, is_rate_limited=False, error_msg="service temporarily unavailable",
        api_kwargs=None, api_messages=[], messages=[], conversation_history=None,
        api_call_count=1, approx_tokens=10, provider="custom", base_url="http://fake",
        model="m",
    )
    banner = [s for s in agent.statuses if s.startswith("❌")]
    assert banner == ["❌ API failed after 3 retries — Service temporarily unavailable"]
    assert "recovery loop" not in result["final_response"]


# --------------------------------------------------------------------------- loop level

from tests.e2e.core.history._helpers import NO_BACKGROUND_REVIEW, OFFLINE_CONFIG
from tests.fakes.fake_llm_provider import Error, FakeLLMServer, Text, write_hermes_home

_ALWAYS_TIMEOUT = lambda record: Error(503, "simulated provider outage, always")


def _agent_turn(script, extra_config: str, session_id: str, agent_block: str | None = None):
    """One in-process AIAgent turn against the loopback fake. ``agent_block`` replaces the
    writer's stock ``agent:`` section (appending a second one would duplicate the YAML key
    and silently drop the whole file back to defaults)."""
    from run_agent import AIAgent
    home = Path(os.environ["HERMES_HOME"])
    with FakeLLMServer(script) as srv:
        write_hermes_home(home, srv.base_url, extra_config=extra_config)
        if agent_block:
            cfgp = home / "config.yaml"
            cfgp.write_text(
                cfgp.read_text(encoding="utf-8").replace(
                    "agent:\n  api_max_retries: 1\n", f"agent:\n{agent_block}"),
                encoding="utf-8")
        agent = AIAgent(provider="custom", base_url=srv.base_url, api_key="sk-fake",
                        model="fake-model", session_id=session_id, quiet_mode=True,
                        platform="cli", skip_memory=True)
        try:
            result = agent.run_conversation("hi", conversation_history=[], task_id="t")
            totals = getattr(agent, "_recovery_totals", None)
        finally:
            agent.close()
        requests = srv.main_requests()
    return result, requests, totals


def test_healthy_fake_provider_passes_through_untouched():
    """Control: a healthy fake provider answers normally with the bounds configured."""
    result, requests, totals = _agent_turn(
        [Text("ALL-GOOD")], OFFLINE_CONFIG + NO_BACKGROUND_REVIEW, "bound-healthy",
        agent_block="  api_max_retries: 1\n  auto_recovery_max_wall_seconds: 600\n"
                    "  auto_recovery_max_attempts: 10\n")
    assert result.get("completed") and result.get("final_response") == "ALL-GOOD", result
    assert len(requests) == 1, "healthy provider must see exactly one request"
    assert not totals, f"no recovery ledger on a healthy turn: {totals}"


def test_always_timeout_provider_stops_at_the_configured_bound():
    """Inject an always-503 fake provider (loopback, no network) with a 4-attempt total
    bound and cycles that would allow 50: the loop must stop at EXACTLY 4 requests and the
    failure copy must carry the true attempt count."""
    result, requests, totals = _agent_turn(
        _ALWAYS_TIMEOUT, OFFLINE_CONFIG + NO_BACKGROUND_REVIEW, "bound-timeout",
        agent_block="  api_max_retries: 1\n  auto_recovery_cycles: 50\n"
                    "  auto_recovery_max_attempts: 4\n")
    assert result.get("failed"), result
    assert totals is not None and int(totals["attempts"]) == 4, totals
    assert "4 attempts" in (result.get("final_response") or ""), result.get("final_response")
    # The OpenAI client adds its own connection retries inside one Hermes attempt, so the
    # recorded HTTP count can exceed the ledger by a few; an unbounded loop (the base bug:
    # 50 cycles) records 50+.
    assert 4 <= len(requests) <= 12, f"loop must stop AT the bound, saw {len(requests)} requests"


def test_bound_config_and_env_override_parse():
    """Config keys surface on the agent; HERMES_AUTO_RECOVERY_MAX_* env values win."""
    from run_agent import AIAgent
    cfg = {"agent": {"auto_recovery_max_wall_seconds": 420, "auto_recovery_max_attempts": 6}}

    def _make(env=None):
        env_patch = {k: str(v) for k, v in (env or {}).items()}
        with patch("agent.process_bootstrap.OpenAI"), \
                patch("hermes_cli.config.load_config", return_value=cfg), \
                patch("hermes_cli.config.load_config_readonly", return_value=cfg), \
                patch.dict(os.environ, env_patch):
            return AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1",
                           model="test/model", quiet_mode=True, skip_context_files=True,
                           skip_memory=True)

    agent = _make()
    assert agent._auto_recovery_max_wall_s == 420.0
    assert agent._auto_recovery_max_attempts == 6.0

    agent = _make({"HERMES_AUTO_RECOVERY_MAX_WALL_S": "90",
                   "HERMES_AUTO_RECOVERY_MAX_ATTEMPTS": "3"})
    assert agent._auto_recovery_max_wall_s == 90.0
    assert agent._auto_recovery_max_attempts == 3.0
