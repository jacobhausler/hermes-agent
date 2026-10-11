"""The /clear confirm modal must not over-claim destruction.

/clear is a session rotation + screen wipe, not history destruction:
``new_session()`` flushes the current turn to the session DB before rotating
(#47202), ``end_session`` only stamps ``ended_at``/``end_reason``
(``hermes_state_sessions.py`` — ``reopen_session()`` to re-end), messages stay
in SQLite and the old session stays ``/resume``-able; the only deletion is
empty-row pruning (``delete_session_if_empty`` drops rows with no messages,
no title and no children). Contrast ``/exit --delete``, which actually removes
transcripts + SQLite. So the confirmation detail may not claim the history
"will be discarded" — it must state where the old session went.
"""

from unittest.mock import patch

from tests.hermes_cli.test_cli_init import _make_cli


def _clear_confirm_detail():
    """Drive /clear through the real dispatch and capture the detail shown on
    the destructive-confirm modal; the spy cancels so the rotation never runs —
    the assertion is about the text, not the mechanics (covered in
    test_cli_new_session.py)."""
    cli = _make_cli()
    seen = {}

    def _spy_confirm(command, detail, cmd_original=None):
        seen["command"] = command
        seen["detail"] = detail
        # implicit None — cancel: _cmd_clear short-circuits, REPL stays alive

    with patch.object(cli, "_confirm_destructive_slash", _spy_confirm):
        result = cli.process_command("/clear")

    assert result is True  # cancelled command is handled, keep REPL alive
    assert seen.get("command") == "clear", "/clear did not reach the confirm gate"
    return seen["detail"]


def test_clear_confirm_detail_does_not_claim_history_discarded():
    """>/clear's detail must not promise destruction the handler does not perform."""
    detail = _clear_confirm_detail()
    assert "discarded" not in detail.lower()
    assert "destroy" not in detail.lower()


def test_clear_confirm_detail_points_at_resume():
    """The accurate framing: the old session is rotated out, not deleted —
    the modal must tell the user where it went."""
    detail = _clear_confirm_detail()
    assert "/resume" in detail
