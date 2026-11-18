"""Async psycopg pool for a FastAPI service on Neon.

One pool, created at startup and closed at shutdown. Callers *borrow* a connection through
Database.transaction(), which also begins/commits/rolls back the transaction, so nothing
leaks and nobody forgets to commit.
"""

from __future__ import annotations
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from uuid import UUID
import psycopg
from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool
from settings import Settings

log = logging.getLogger(__name__)

async def _configure(conn: psycopg.AsyncConnection) -> None:
    """Runs once on every NEW physical connection the pool opens.

    The pool requires the connection to be left idle (not inside a transaction); with
    autocommit=True below that is automatic. Needs `CREATE EXTENSION vector` to have run
    already (see setup_db.py), otherwise pool startup fails with a clear error.
    """
    await register_vector_async(conn)

def create_pool(settings:Settings) -> AsyncConnectionPool:
    return AsyncConnectionPool(
        conninfo=settings.database_url.get_secret_value(),
        min_size=settings.db_pool_min_size,
        max_size=settings.db_pool_max_size,
        timeout=settings.db_pool_timeout_s,
        max_idle=settings.db_max_idle_s,
        max_lifetime=settings.db_max_lifetime_s,
        check=AsyncConnectionPool.check_connection,
        configure=_configure,
        kwargs={
            "connect_timeout": settings.db_connect_timeout_s,
            "autocommit": True,
            "prepare_threshold": None,
            "application_name": "api",
            "keepalives":1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
        },
        open=False,
        name="api-pool",
    )

async def _apply_local_settings(
        conn: psycopg.AsyncConnection, statement_timeout_ms: int, tenant_id: UUID | str | None
) -> None:
    # set_config(name, value, true) is SET LOCAL, but accepts bind parameters (SET LOCAL does not).
    # LOCAL = lasts until COMMIT/ROLLBACK, so it is safe with PgBouncer transaction pooling.
    sql = "SELECT set_config('statement_timeout', %s, true)"
    params: list[str] = [str(statement_timeout_ms)]
    if tenant_id is not None:
        sql += ", set_config('app.tenant_id',%s, true)"
        params.append(str(tenant_id))
    await conn.execute(sql, params)


class Database:
    def __init__(self, settings: Settings) -> None:
        self._default_timeout_ms = settings.db_statement_timeout_ms
        self.pool = create_pool(settings)

    async def open(self) -> None:
        # Waits until min_size connections exist; fails fast if the DB is unreachable.
        # 30s leaves room for a Neon cold start.
        await self.pool.open(wait=True, timeout=30)

    async def close(self) -> None:
        await self.pool.close(timeout=10)

    @asynccontextmanager
    async def transaction(
        self, *, tenant_id: UUID | str | None=None,
        statement_timeout_ms: int | None=None,
    ) -> AsyncGenerator[psycopg.AsyncConnection]:
        """Borrow a connection and run the body in ONE transaction.

        Commits on normal exit, rolls back on any exception, always returns the
        connection to the pool. Sets statement_timeout (and the RLS tenant, if given)
        with SET LOCAL semantics.

        Retry: if the connection turns out to be dead *before* your code runs (stale
        connection, Neon waking up), it takes a fresh connection and tries once more.
        Errors raised by your own body are never retried here.
        """
        timeout_ms = self._default_timeout_ms if statement_timeout_ms is None else statement_timeout_ms
        for attempt in (1,2):
            started = False
            try:
                async with self.pool.connection() as conn:
                    async with conn.transaction():
                        await _apply_local_settings(conn, timeout_ms, tenant_id)
                        started = True
                        yield conn
                return
            except psycopg.OperationalError as exc:
                if started or attempt ==2:
                    raise
                log.warning("DB connection failed before work started (%s); retrying once", type(exc).__name__)

    async def ping(self) -> None:
        async with self.transaction() as conn:
            await conn.execute("SELECT 1")