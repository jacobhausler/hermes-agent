"""The destructive-slash gate detail for ``/clear`` and ``/new`` must state honestly
what the command does.

Both are session *rotations*, not destructions: ``new_session()`` flushes the
in-flight turn to the old session before rotating (the #47202 guard),
``end_session()`` only stamps ``ended_at``/``end_reason``, and the old session
stays resummable via ``/resume``. The detail previously claimed the history
"will be discarded", which is false — the confirmation ceremony may stay, but
the copy must not over-claim destruction (feedback key 115b383911c95c2d).
"""

from __future__ import annotations

from types import SimpleNamespace


def _bound_cancel_handler(command: str):
    """Bind the real ``_cmd_clear``/``_cmd_new`` against a stand-in whose
    ``_confirm_destructive_slash`` records the detail and returns ``None``
    (cancelled). Both handlers must bail out at that point, so nothing else
    needs stubbing."""
    from cli import HermesCLI

    captured: dict = {}

    def _confirm(cmd, detail, cmd_original=None):
        captured["command"] = cmd
        captured["detail"] = detail
        return None  # cancel — caller must stop before rotating

    self_ = SimpleNamespace(
        _split_destructive_skip=HermesCLI._split_destructive_skip,
        _confirm_destructive_slash=_confirm,
    )
    method = HermesCLI._cmd_clear if command == "clear" else HermesCLI._cmd_new
    return method.__get__(self_, type(self_)), captured


def test_clear_gate_detail_does_not_claim_history_is_discarded():
    """``/clear`` must tell the user the conversation survives via /resume."""
    handler, cap = _bound_cancel_handler("clear")
    assert handler("/clear") is True  # cancelled => handled, REPL stays alive
    assert cap["command"] == "clear"
    detail = cap["detail"].lower()
    assert "discarded" not in detail
    assert "resume" in detail


def test_new_gate_detail_does_not_claim_history_is_discarded():
    """``/new``/``/reset`` share ``_cmd_new`` and rotate exactly like /clear."""
    handler, cap = _bound_cancel_handler("new")
    assert handler("/new") is True
    assert cap["command"] == "new"
    detail = cap["detail"].lower()
    assert "discarded" not in detail
    assert "resume" in detail
