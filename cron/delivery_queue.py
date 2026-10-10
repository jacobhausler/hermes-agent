"""Profile-local durable handoff for cron delivery through live gateway adapters.

A restart-safe cron worker executes outside the gateway cgroup.  It cannot own
relay/E2EE adapter objects, so it queues the final send here.  A gateway claims
each row at most once.  If that gateway dies after claiming, the outcome is
marked unknown and never retried: losing a delivery is safer than duplicating a
possibly-completed send.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from agent.redact import redact_sensitive_text
from cron.executions import _owner_is_live, _process_start_time
from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now

logger = logging.getLogger(__name__)

DELIVERY_DB: Optional[Path] = None
_PROCESS_ID = uuid.uuid4().hex
_lock = threading.RLock()
_ACTIVE_DELIVERIES: set[str] = set()
_TERMINAL = ("delivered", "failed", "unknown", "suppressed")
MAX_TERMINAL_DELIVERIES = 1000
DEFAULT_DELIVERY_WAIT_TIMEOUT_SECONDS = 300.0


def _prune_terminal_unlocked(conn: sqlite3.Connection) -> None:
    """Redact terminal payloads and retain only bounded outcome metadata."""
    conn.execute(
        """UPDATE deliveries SET job_json='{}', content=''
           WHERE status IN ('delivered','failed','unknown','suppressed')
             AND (job_json != '{}' OR content != '')"""
    )
    keep = max(0, int(MAX_TERMINAL_DELIVERIES))
    terminal_count = int(
        conn.execute(
            "SELECT COUNT(*) FROM deliveries "
            "WHERE status IN ('delivered','failed','unknown','suppressed')"
        ).fetchone()[0]
    )
    excess = terminal_count - keep
    if excess > 0:
        conn.execute(
            """INSERT OR IGNORE INTO delivery_tombstones
               (execution_id, terminal_status, finished_at)
               SELECT execution_id, status, finished_at FROM deliveries
               WHERE status IN ('delivered','failed','unknown','suppressed')
               ORDER BY julianday(finished_at), finished_at,
                        julianday(created_at), created_at, execution_id
               LIMIT ?""",
            (excess,),
        )
        conn.execute(
            """DELETE FROM deliveries WHERE execution_id IN (
                 SELECT execution_id FROM deliveries
                 WHERE status IN ('delivered','failed','unknown','suppressed')
                 ORDER BY julianday(finished_at), finished_at,
                          julianday(created_at), created_at, execution_id
                 LIMIT ?
               )""",
            (excess,),
        )


def queue_path(home: Optional[Path] = None) -> Path:
    """The queue file of ``home`` (the active home when None); a test override wins."""
    if DELIVERY_DB is not None:
        return DELIVERY_DB
    root = Path(home) if home is not None else get_hermes_home()
    return root.resolve() / "cron" / "deliveries.db"


def _path() -> Path:
    return queue_path()


def _initialize_schema(conn: sqlite3.Connection) -> None:
    from hermes_cli.sqlite_util import add_column_if_missing

    # SQLite cannot widen a CHECK in place. Preserve all old rows atomically,
    # including claimed sends, while admitting a distinct never-sent disposition.
    for table in ("deliveries", "delivery_tombstones"):
        row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
        if row and "'suppressed'" not in row[0]:
            conn.execute("SAVEPOINT notification_disposition")
            try:
                conn.execute(f"ALTER TABLE {table} RENAME TO {table}_old")
                conn.execute(row[0].replace("'unknown'", "'unknown','suppressed'"))
                conn.execute(f"INSERT INTO {table} SELECT * FROM {table}_old")
                conn.execute(f"DROP TABLE {table}_old")
                conn.execute("RELEASE notification_disposition")
            except BaseException:
                conn.execute("ROLLBACK TO notification_disposition")
                conn.execute("RELEASE notification_disposition")
                raise

    conn.execute(
        """CREATE TABLE IF NOT EXISTS deliveries (
             execution_id TEXT PRIMARY KEY,
             job_json TEXT NOT NULL,
             content TEXT NOT NULL,
             for_failure INTEGER NOT NULL DEFAULT 0,
             status TEXT NOT NULL CHECK(status IN
               ('pending','delivering','delivered','failed','unknown','suppressed')),
             owner_process_id TEXT,
             owner_pid INTEGER,
             owner_started_at INTEGER,
             created_at TEXT NOT NULL,
             finished_at TEXT,
             error TEXT
           )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS delivery_tombstones (
             execution_id TEXT PRIMARY KEY,
             terminal_status TEXT NOT NULL CHECK(terminal_status IN
               ('delivered','failed','unknown','suppressed')),
             finished_at TEXT
           )"""
    )
    add_column_if_missing(
        conn, "deliveries", "for_failure",
        "for_failure INTEGER NOT NULL DEFAULT 0",
    )


def _connect() -> sqlite3.Connection:
    # Late imports: a scheduler daemon that outlives an on-disk upgrade already has the OLD
    # ``hermes_cli.sqlite_util`` / ``cron.jobs`` cached, so new names must be resolved at call time,
    # not at import time (the guarantee cron/ledger.py used to carry, see e24c8499).
    from hermes_cli.sqlite_util import open_db

    path = _path()
    conn = open_db(path, db_label="cron/deliveries.db", synchronous_full=True, initialize=_initialize_schema)
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return conn


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    # Pruning is done explicitly by the paths that create terminal
    # rows (_finish / recover_abandoned / _terminalize_wait_timeout);
    # read-only polls must not pay for a full-table UPDATE + COUNT.
    from hermes_cli.sqlite_util import transaction

    with _lock, transaction(_connect()) as conn:
        yield conn


def enqueue(
    execution_id: str,
    job: dict,
    content: str,
    *,
    for_failure: bool = False,
) -> dict:
    """Persist one idempotent delivery request before the worker waits."""
    with _transaction() as conn:
        # Serialize the tombstone check and insert with retention in other
        # processes, which can move a terminal delivery into the tombstone table.
        conn.execute("BEGIN IMMEDIATE")
        tombstone = conn.execute(
            "SELECT terminal_status, finished_at FROM delivery_tombstones "
            "WHERE execution_id=?",
            (str(execution_id),),
        ).fetchone()
        if tombstone is not None:
            return {
                "execution_id": str(execution_id),
                "status": tombstone["terminal_status"],
                "finished_at": tombstone["finished_at"],
            }
        conn.execute(
            """INSERT OR IGNORE INTO deliveries
               (execution_id, job_json, content, for_failure, status, created_at)
               VALUES (?, ?, ?, ?, 'pending', ?)""",
            (
                str(execution_id),
                json.dumps(job, ensure_ascii=False, sort_keys=True),
                str(content),
                int(bool(for_failure)),
                _hermes_now().isoformat(),
            ),
        )
        row = conn.execute(
            "SELECT * FROM deliveries WHERE execution_id=?", (str(execution_id),)
        ).fetchone()
    return dict(row)


def get_status(execution_id: str) -> Optional[dict]:
    with _transaction() as conn:
        row = conn.execute(
            "SELECT * FROM deliveries WHERE execution_id=?", (str(execution_id),)
        ).fetchone()
        if row is not None:
            return dict(row)
        tombstone = conn.execute(
            "SELECT execution_id, terminal_status, finished_at "
            "FROM delivery_tombstones WHERE execution_id=?",
            (str(execution_id),),
        ).fetchone()
    if tombstone is None:
        return None
    return {
        "execution_id": tombstone["execution_id"],
        "status": tombstone["terminal_status"],
        "finished_at": tombstone["finished_at"],
        "error": None,
    }


def claim_next() -> Optional[dict]:
    """Atomically claim one pending send before touching the transport."""
    pid = os.getpid()
    started = _process_start_time(pid)
    with _transaction() as conn:
        # created_at carries a DST-varying offset: order by instant, not text.
        row = conn.execute(
            "SELECT execution_id FROM deliveries WHERE status='pending' "
            "ORDER BY julianday(created_at), created_at, execution_id LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        cur = conn.execute(
            """UPDATE deliveries SET status='delivering', owner_process_id=?,
               owner_pid=?, owner_started_at=?
               WHERE execution_id=? AND status='pending'""",
            (_PROCESS_ID, pid, started, row["execution_id"]),
        )
        if cur.rowcount != 1:
            return None
        claimed = conn.execute(
            "SELECT * FROM deliveries WHERE execution_id=?", (row["execution_id"],)
        ).fetchone()
        _ACTIVE_DELIVERIES.add(row["execution_id"])
    result = dict(claimed)
    result["job"] = json.loads(result.pop("job_json"))
    return result


def _finish(execution_id: str, *, error: Optional[str], suppressed: bool = False) -> bool:
    status = "failed" if error else "suppressed" if suppressed else "delivered"
    safe_error = (
        redact_sensitive_text(str(error), force=True, redact_url_credentials=True)
        if error
        else None
    )
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE deliveries SET status=?, finished_at=?, error=?
               WHERE execution_id=? AND status='delivering'
                 AND owner_process_id=? AND owner_pid=?""",
            (
                status,
                _hermes_now().isoformat(),
                safe_error,
                execution_id,
                _PROCESS_ID,
                os.getpid(),
            ),
        )
        _prune_terminal_unlocked(conn)
    return cur.rowcount == 1


def recover_abandoned() -> int:
    """Fence dead delivery owners as unknown; never replay uncertain sends."""
    changed = 0
    with _transaction() as conn:
        rows = conn.execute(
            "SELECT execution_id, owner_process_id, owner_pid, owner_started_at "
            "FROM deliveries WHERE status='delivering'"
        ).fetchall()
        for row in rows:
            same_process = row["owner_process_id"] == _PROCESS_ID
            if same_process:
                with _lock:
                    if row["execution_id"] in _ACTIVE_DELIVERIES:
                        continue
            elif _owner_is_live(int(row["owner_pid"]), row["owner_started_at"]):
                continue
            error = (
                "Gateway finished delivery but could not persist its outcome; "
                "send was not retried."
                if same_process
                else "Gateway exited during delivery; send outcome is unknown and was not retried."
            )
            cur = conn.execute(
                """UPDATE deliveries SET status='unknown', finished_at=?, error=?
                   WHERE execution_id=? AND status='delivering'""",
                (
                    _hermes_now().isoformat(),
                    error,
                    row["execution_id"],
                ),
            )
            changed += cur.rowcount
        _prune_terminal_unlocked(conn)
    return changed


def _superseded_row(conn: sqlite3.Connection, row: sqlite3.Row) -> bool:
    """True when a newer COMPLETED run of the same job finished after this pending row
    was created — the message it carries is already stale ground.

    A pending row can outlive its news: an interrupted/disabled run leaves it queued
    while later fires complete normally, and a later drain replays the old stdout as a
    fresh alert — under a job name the job no longer carries (est-2ek.1.667: a
    pre-conversion truncated stdout re-delivered ~5.5h late). Only a ``completed``
    ledger row supersedes: a crashed run's own undelivered message stays deliverable
    (crash-retry parity), and the comparison is made on the ledger the worker itself
    wrote, not on config mtime (which no consumer can read fail-closed across homes).
    """
    from cron.executions import _connect as _connect_executions

    job_id = str(json.loads(row["job_json"]).get("id") or "")
    if not job_id:
        return False
    econn = None
    try:
        econn = _connect_executions()
        newer = econn.execute(
            """SELECT 1 FROM executions
               WHERE job_id=? AND status='completed'
                 AND julianday(finished_at) > julianday(?)
               LIMIT 1""",
            (job_id, row["created_at"]),
        ).fetchone()
    except sqlite3.Error as exc:  # unreadable ledger: deliver rather than silently drop
        logger.warning("Cron delivery %s: supersede check failed (%s); delivering",
                       row["execution_id"], exc)
        return False
    finally:
        if econn is not None:
            econn.close()
    return newer is not None


def drain(
    send: Callable[[dict, str, bool], Optional[str]], *, limit: int = 20
) -> int:
    """Deliver pending rows through *send*, terminalizing every claimed row."""
    recover_abandoned()
    processed = 0
    for _ in range(max(0, limit)):
        with _lock, _transaction() as conn:
            head = conn.execute(
                "SELECT * FROM deliveries WHERE status='pending' "
                "ORDER BY julianday(created_at), created_at, execution_id LIMIT 1"
            ).fetchone()
            if head is not None and _superseded_row(conn, head):
                conn.execute(
                    """UPDATE deliveries SET status='suppressed', finished_at=?,
                       error='superseded by a newer completed run of the same job'
                       WHERE execution_id=? AND status='pending'""",
                    (_hermes_now().isoformat(), head["execution_id"]),
                )
                _prune_terminal_unlocked(conn)
                logger.info(
                    "Cron delivery %s: job re-ran to completion since this run — "
                    "stale delivery dropped", head["execution_id"])
                continue
        row = claim_next()
        if row is None:
            break
        with _lock:
            _ACTIVE_DELIVERIES.add(row["execution_id"])
        try:
            try:
                error = send(
                    row["job"], row["content"], bool(row["for_failure"])
                )
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
            _finish(row["execution_id"], error=error,
                    suppressed=bool(row["job"].get("_notification_all_targets_suppressed")))
        finally:
            with _lock:
                _ACTIVE_DELIVERIES.discard(row["execution_id"])
        processed += 1
    return processed


def _terminalize_wait_timeout(execution_id: str) -> str:
    """Fence a delivery whose worker can no longer wait for confirmation.

    A row still ``pending`` was provably never attempted, so it is left queued
    for whichever gateway comes up next (a restart that includes an update can
    easily exceed the worker's wait budget).  That is a deferral, not a
    failure: report success so the job is not recorded ``delivery_failed`` for
    a message the drain will still send.  Only a row caught mid-send is
    uncertain and gets fenced ``unknown``.
    """
    now = _hermes_now().isoformat()
    uncertain_error = (
        "timed out while gateway delivery was in progress; outcome is unknown and "
        "was not retried"
    )
    with _transaction() as conn:
        row = conn.execute(
            "SELECT status FROM deliveries WHERE execution_id=?",
            (str(execution_id),),
        ).fetchone()
        if row is not None and row["status"] == "pending":
            logger.warning(
                "Cron delivery %s: no live gateway within the wait budget; "
                "left queued for the next gateway",
                execution_id,
            )
            return ""
        conn.execute(
            """UPDATE deliveries SET status='unknown', finished_at=?, error=?
               WHERE execution_id=? AND status='delivering'""",
            (now, uncertain_error, str(execution_id)),
        )
        row = conn.execute(
            "SELECT status, error FROM deliveries WHERE execution_id=?",
            (str(execution_id),),
        ).fetchone()
        _prune_terminal_unlocked(conn)
    if row is None:
        return "timed out waiting for live gateway delivery"
    if row["status"] in {"delivered", "suppressed"}:
        return ""
    return str(row["error"] or f"delivery {row['status']}")


def enqueue_and_wait(
    execution_id: str,
    job: dict,
    content: str,
    *,
    for_failure: bool = False,
    timeout: Optional[float] = None,
) -> Optional[str]:
    """Queue delivery and wait for a gateway's terminal at-most-once outcome."""
    queued = enqueue(execution_id, job, content, for_failure=for_failure)
    if queued["status"] in _TERMINAL:
        return None if queued["status"] in {"delivered", "suppressed"} else str(
            queued.get("error") or f"delivery {queued['status']}"
        )
    wait_timeout = (
        DEFAULT_DELIVERY_WAIT_TIMEOUT_SECONDS if timeout is None else max(0.0, timeout)
    )
    deadline = time.monotonic() + wait_timeout
    while time.monotonic() < deadline:
        row = get_status(execution_id)
        if row and row["status"] in _TERMINAL:
            return None if row["status"] in {"delivered", "suppressed"} else str(
                row.get("error") or f"delivery {row['status']}"
            )
        time.sleep(1.0)
    return _terminalize_wait_timeout(execution_id) or None
