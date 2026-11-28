"""Observed in production (2026-11-28): a Desktop prompt to a chat whose ledger queue held work
never reached the owner's FIFO: the relay's ``session.resume`` answer carried the full history
(3 MB on the affected chat), one frame over the client's 1 MiB cap, so the socket died before
``prompt.submit`` and the Desktop surfaced the transient ``SESSION_OWNER_BUSY`` refusal on a queue
that would never drain on its own. The relay needs only the attach and the submit — it must not
fetch the transcript.

Real: the ``SessionDB`` ledger, the legacy ``prompt.submit`` and its ledger gate, and the owner
(``SessionAuthority`` + ``AuthorityConnection``: resume, admission, publication). Replaced: the
owner's discovery and dial (``owner_forward.discover_owner`` / ``connect_owner``) and the transport
itself — a full-history ``session.resume`` over the capped wire answers exactly what the incident
observed: a lost connection and no answer."""
import asyncio
import contextlib
from types import SimpleNamespace

from hermes_state import SessionDB
from hermes_state_runtime import admit_session_input, begin_runtime_epoch, list_session_admissions
from tui_gateway import server


class _FrameCappedOwner:
    """``GatewayClient``'s surface (``rpc`` + ``events``) over a real ``AuthorityConnection``, with
    the capped wire: a ``session.resume`` that carries the history is one oversized frame, so the
    connection dies mid-answer and the client raises the loss — never "nothing was sent"."""

    def __init__(self, authority):
        from gateway.session_controls import AuthorityConnection
        self.loop, self.events = asyncio.get_running_loop(), asyncio.Queue()
        self.conn = AuthorityConnection(authority, self, {"user_id": "desktop"})

    def write(self, frame):  # the owner's fanout to this socket
        self.loop.call_soon_threadsafe(self.events.put_nowait, frame)
        return True

    def close(self):
        return None

    async def rpc(self, method, **params):
        if method == "session.resume" and not params.get("omit_messages"):
            from hermes_cli.gateway_client import DISCONNECTED, GatewayClientError
            raise GatewayClientError(DISCONNECTED)
        reply = await self.conn.dispatch({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
        if "error" in reply:
            from hermes_cli.gateway_client import rpc_error
            raise rpc_error(reply["error"])
        return reply["result"]


def test_forward_admits_behind_a_queued_backlog_on_a_chat_too_large_to_resume(monkeypatch, tmp_path):
    from tui_gateway import owner_forward
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = None
    try:
        monkeypatch.setattr(server, "_get_db", lambda: db)
        for name in ("_schedule_agent_build", "_schedule_session_cap_enforcement", "_register_session_cwd",
                     "_start_agent_build"):
            monkeypatch.setattr(server, name, lambda *_a, **_k: None)
        monkeypatch.setattr(server, "_restart_completed_failed_agent_build", lambda *_a: False)
        monkeypatch.setattr(server, "_emit", lambda *_a, **_k: True)
        created = server.handle_request({"id": "c", "method": "session.create", "params": {"source": "desktop"}})
        sid, key = created["result"]["session_id"], created["result"]["stored_session_id"]
        db.create_session(key, "test")
        epoch = begin_runtime_epoch(db, instance_id="gateway")
        from gateway.session_authority import LiveSession, SessionAuthority
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id="owned",
                                     instance_id="gateway", db=db, epoch=epoch)
        authority.sessions[key] = LiveSession(SimpleNamespace(platform=None, user_id="desktop"), "route")
        monkeypatch.setattr(authority, "_schedule", lambda ref: None)
        # The queue already holds an unclaimed wake: the slot refuses and forwards, and the relay
        # meets the capped wire on its way to the owner's FIFO.
        first = admit_session_input(db, epoch=epoch, principal_id="automation", session_id=key,
                                    request_id="wake-1", payload={"text": "a workflow wake"})
        assert first["status"] == "queued"

        @contextlib.asynccontextmanager
        async def connect(_home, endpoint, scope=None, max_size=2 ** 20):
            assert endpoint == "ready-owner"
            yield _FrameCappedOwner(authority)
        monkeypatch.setattr(owner_forward, "discover_owner", lambda _home: "ready-owner")
        monkeypatch.setattr(owner_forward, "connect_owner", connect)

        reply = server.handle_request({"id": "p", "method": "prompt.submit",
                                       "params": {"session_id": sid, "text": "fix the queue"}})

        assert "error" not in reply, reply  # base: 4090 SESSION_OWNER_BUSY, the wire died pre-submit
        assert reply["result"]["status"] == "queued" and reply["result"]["forwarded"] is True
        rows = list_session_admissions(db, session_id=key)
        assert [r["status"] for r in rows] == ["queued", "queued"]  # base: the user's prompt never lands
        assert [r["payload"]["text"] for r in rows] == ["a workflow wake", "fix the queue"]
    finally:
        if sid:
            server._sessions.pop(sid, {}).setdefault("_finalized", True)  # retire the relay thread
        db.close()
