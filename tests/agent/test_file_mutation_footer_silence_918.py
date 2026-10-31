"""est-2ek.1.918 — the verifier footer must never decorate a bare silence marker.

Regression: a turn whose whole final response is the intentional-silence marker
(``[SILENT]`` / ``NO_REPLY`` / ...) while the turn also carries failed file
mutations used to ship as ``"[SILENT]\\n\\n<footer>"``. The gateway drops a
response only when the WHOLE answer is a marker
(``gateway.response_filters.is_intentional_silence_response``), so the footer
defeated the silence check and the turn reached the channel.

The guard must reuse the gateway's canonical matcher (late import) — a locally
re-implemented marker list would drift from the delivery side, which is exactly
the bug class this test exists to keep closed.
"""

from __future__ import annotations

import logging

from agent.turn_finalizer import _append_file_mutation_footer
from gateway.response_filters import is_intentional_silence_response

LOG = logging.getLogger("test")


class _StubAgent:
    """Only the four attributes the footer gate reads, nothing else."""

    def __init__(self, failed=None, enabled=True):
        self._turn_failed_file_mutations = failed if failed is not None else {"/tmp/x.md": "boom"}
        self._enabled = enabled

    def _file_mutation_verifier_enabled(self):
        return self._enabled

    def _file_mutations_still_failed(self, failed):
        return failed

    def _format_file_mutation_failure_footer(self, failed):
        return "⚠️ File-mutation verifier: 1 file did not change."


def test_bare_silent_marker_is_not_decorated():
    out = _append_file_mutation_footer(_StubAgent(), "[SILENT]", LOG)
    assert out == "[SILENT]"
    assert is_intentional_silence_response(out)


def test_bare_no_reply_marker_is_not_decorated():
    out = _append_file_mutation_footer(_StubAgent(), "NO_REPLY", LOG)
    assert out == "NO_REPLY"
    assert is_intentional_silence_response(out)


def test_marker_with_edge_punctuation_still_excluded():
    # The gateway matcher tolerates ``*[SILENT]*``; the guard must not be
    # stricter than the matcher, or the same drift ships for punctuated turns.
    out = _append_file_mutation_footer(_StubAgent(), "*[SILENT]*", LOG)
    assert out == "*[SILENT]*"
    assert is_intentional_silence_response(out)


def test_real_prose_still_gets_the_footer():
    body = "I edited three files this turn."
    out = _append_file_mutation_footer(_StubAgent(), body, LOG)
    assert out.startswith(body)
    assert "File-mutation verifier" in out


def test_silence_with_prose_note_is_still_decorated():
    # Only the BARE marker is the delivery-silence shape; "[SILENT] and then
    # some prose" already fails the exact matcher and ships anyway, so the
    # footer's anti-over-claim duty still applies.
    body = "[SILENT] noted, but see the summary below"
    out = _append_file_mutation_footer(_StubAgent(), body, LOG)
    assert "File-mutation verifier" in out


def test_no_failures_never_decorates_anything():
    agent = _StubAgent(failed={})
    assert _append_file_mutation_footer(agent, "[SILENT]", LOG) == "[SILENT]"
    assert _append_file_mutation_footer(agent, "plain", LOG) == "plain"


def test_guard_shares_the_delivery_side_marker_set():
    # Single source of truth: every canonical marker the gateway treats as
    # silence must survive the footer gate undecorated. If a marker is added to
    # LIVE_GATEWAY_SILENT_MARKERS and the guard grows its own list, this row
    # fails — which is the drift this test polices.
    from gateway.response_filters import LIVE_GATEWAY_SILENT_MARKERS

    agent = _StubAgent()
    for marker in LIVE_GATEWAY_SILENT_MARKERS:
        assert _append_file_mutation_footer(agent, marker, LOG) == marker
