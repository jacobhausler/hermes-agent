"""Plugin policy for producer wakes, separate from human input admission."""
import asyncio
import logging

from gateway.session_authorities import active_authority
from hermes_cli.lifecycle import ainvoke_hook, has_hook
from persistence import STORE_FAILURES

logger = logging.getLogger(__name__)
_HOOK = 'pre_process_notification'
_PROCESS_NOTICES = frozenset({'completion', 'watch_match', 'watch_disabled', 'heartbeat'})


async def drop_process_notification(runner, event):
    """Read the pinned conversation, not whatever now occupies its route."""
    kind = event.get('type')
    pinned = str(event.get('parent_session_id') or '')
    if kind not in _PROCESS_NOTICES or not pinned or not has_hook(_HOOK):
        return False
    authority = active_authority(runner)
    if authority is None:
        return False
    try:
        session_id = await asyncio.to_thread(authority.logical_owner, pinned)
        row = await authority.persistence.aio.sessions.get(session_id)
    except STORE_FAILURES:
        logger.warning('Process notification policy could not read owner %s', pinned, exc_info=True)
        return False
    if row is None:
        return False
    results = await ainvoke_hook(_HOOK, session_id=session_id, session_title=row.title or '',
                                 session_source=row.source, notification_type=kind)
    return any(isinstance(result, dict) and result.get('action') == 'drop' for result in results)
