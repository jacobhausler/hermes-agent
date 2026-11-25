"""A producer policy can drop a completion before it enters the local FIFO."""
import pytest

from gateway.session_contract import Submission
from gateway.session_controls import AuthorityConnection
from gateway.session_local import create_local_session
from gateway.session_local_title import title_new_session
from hermes_cli.plugins import get_plugin_manager
from tests.gateway.test_automation_notification_category import _authority
from tools.process_registry import ProcessSession


@pytest.mark.asyncio
@pytest.mark.usefixtures('authority_cwd_in_tmp')
async def test_stopped_owner_completion_never_enters_fifo(tmp_path, monkeypatch):
    runner, authority = await _authority(tmp_path, monkeypatch)
    connection = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    stop = tmp_path / 'stop.request'
    blocked_title = 'stopped worker'
    def veto(session_title):
        if session_title == blocked_title and stop.exists():
            return {'action': 'drop'}
        return None

    manager = get_plugin_manager()
    monkeypatch.setattr(manager, '_discovered', True)
    monkeypatch.setitem(manager._hooks, 'pre_process_notification', [veto])

    async def finish(title):
        process, ref = await new_process(title)
        event = runner._build_process_completion_event({'session_key': process.session_key}, process, process.id)
        return ref, await runner._inject_watch_notification('[process finished]', event)

    async def new_process(title):
        ref = create_local_session(authority, connection.actor, {
            'request_id': title, 'source': 'cli', 'model': 'fixture', 'toolsets': []})
        title_new_session(authority, ref, title)
        live = authority.sessions[ref.session_id]
        process = ProcessSession(id='proc-' + title, command='fixture', task_id='fixture',
                                 session_key=live.route, parent_session_id=ref.session_id,
                                 started_at=1.0, exit_code=0, exited=True)
        return process, ref

    try:
        process, stopped = await new_process(blocked_title)
        stop.touch()
        event = runner._build_process_completion_event({'session_key': process.session_key}, process, process.id)
        assert await runner._inject_watch_notification('[process finished]', event) is None
        assert not authority.persistence.admissions.list(session_id=stopped.session_id, pending_only=False)

        ordinary, delivered = await finish('ordinary chat')
        assert delivered is True
        ordinary_rows = authority.persistence.admissions.list(session_id=ordinary.session_id, pending_only=False)
        assert any('local_automation_v1' in row.payload for row in ordinary_rows)

        reply = await connection.dispatch({'id': 1, 'method': 'session.resume',
                                           'params': {'session_id': stopped.session_id}})
        assert reply['result']['session_id'] == stopped.session_id
        await authority.submit(connection.actor, Submission('manual', stopped, {'text': 'continue'}, 'queue'))
        resumed_rows = authority.persistence.admissions.list(session_id=stopped.session_id, pending_only=False)
        assert [row.request_id for row in resumed_rows] == ['manual']
    finally:
        await connection.close()
        runner.session_store.close_all_db_handles()
