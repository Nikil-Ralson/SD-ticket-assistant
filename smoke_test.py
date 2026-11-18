"""Phase 1 smoke test: run the pooled-endpoint checks against your real Neon database.

    python smoke_test.py              # checks 1-3, about 15 seconds
    python smoke_test.py --idle 6     # also check 4: wait 6 minutes, then ping again

Reads DATABASE_URL (the pooled endpoint) from .env. Run setup_db.py first so the
`vector` extension exists.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

import numpy as np
import psycopg

from db.database import Database
from settings import get_settings


async def scalar(conn: psycopg.AsyncConnection, sql: str, params: Any = None) -> Any:
    cur = await conn.execute(sql, params)
    row = await cur.fetchone()
    assert row is not None
    return row[0]


async def check_open_and_ping(db: Database) -> str:
    t0 = time.perf_counter()
    await db.open()
    opened = time.perf_counter() - t0
    t1 = time.perf_counter()
    await db.ping()
    pinged = time.perf_counter() - t1
    return f"pool opened in {opened:.2f}s (over ~2s means Neon was waking up), ping {pinged * 1000:.0f} ms"


async def check_vector(db: Database) -> str:
    sent = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    async with db.transaction() as conn:
        got = await scalar(conn, "SELECT %s::vector", (sent,))
        dist = await scalar(conn, "SELECT %s::vector <=> %s::vector", (sent, sent))
    if not isinstance(got, np.ndarray) or got.tolist() != [1.0, 2.0, 3.0]:
        raise AssertionError(f"vector did not round-trip, got {got!r}")
    if abs(float(dist)) > 1e-6:
        raise AssertionError(f"cosine distance of a vector with itself should be 0, got {dist}")
    return "vector round-trips as a numpy array and <=> accepts bound parameters"


async def check_transaction_local(db: Database) -> str:
    tenant = str(uuid4())
    rounds = 20  # PgBouncer may hand a different backend each time, so repeat
    for i in range(rounds):
        async with db.transaction(tenant_id=tenant) as conn:
            inside = await scalar(conn, "SELECT current_setting('app.tenant_id', true)")
        if inside != tenant:
            raise AssertionError(f"round {i}: tenant not visible inside its own transaction ({inside!r})")
        async with db.transaction() as conn:
            after = await scalar(conn, "SELECT current_setting('app.tenant_id', true)")
        if after not in (None, ""):
            raise AssertionError(f"round {i}: tenant leaked into the next transaction ({after!r})")

    try:
        async with db.transaction(statement_timeout_ms=200) as conn:
            await conn.execute("SELECT pg_sleep(2)")
    except psycopg.errors.QueryCanceled:
        pass
    else:
        raise AssertionError("a 200 ms statement_timeout did not cancel pg_sleep(2)")
    async with db.transaction() as conn:
        after_timeout = await scalar(conn, "SHOW statement_timeout")
    if after_timeout == "200ms":
        raise AssertionError("the 200 ms statement_timeout leaked into the next transaction")
    return (
        f"tenant stayed inside its transaction for {rounds} rounds; "
        f"statement_timeout afterwards is {after_timeout}"
    )


async def check_after_idle(db: Database, minutes: float) -> str:
    print(f"      waiting {minutes:g} min so Neon suspends its compute ...", flush=True)
    await asyncio.sleep(minutes * 60)
    t0 = time.perf_counter()
    await db.ping()
    elapsed = time.perf_counter() - t0
    return f"ping worked after {minutes:g} min idle and took {elapsed:.2f}s"


async def run(name: str, check: Callable[[], Awaitable[str]]) -> bool:
    try:
        detail = await check()
    except Exception as exc:  # noqa: BLE001 - a test runner reports every failure the same way
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
        return False
    print(f"[PASS] {name}: {detail}")
    return True


async def main(idle_minutes: float) -> int:
    # WARNING level shows the "retrying once" line if the wake-up retry fires.
    logging.basicConfig(level=logging.WARNING, format="      log: %(name)s: %(message)s")
    db = Database(get_settings())
    try:
        if not await run("1. open + ping", lambda: check_open_and_ping(db)):
            return 1
        results = [
            await run("2. pgvector type", lambda: check_vector(db)),
            await run("3. transaction-local settings", lambda: check_transaction_local(db)),
        ]
        if idle_minutes > 0:
            results.append(await run("4. ping after idle", lambda: check_after_idle(db, idle_minutes)))
        else:
            print("[SKIP] 4. ping after idle: run again with --idle 6")
        return 0 if all(results) else 1
    finally:
        await db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--idle", type=float, default=0.0, metavar="MINUTES",
        help="also wait this many minutes, then ping (use 6 or more)",
    )
    # psycopg's async mode cannot use Windows' default ProactorEventLoop, so ask for the selector loop.
    sys.exit(asyncio.run(main(parser.parse_args().idle), loop_factory=asyncio.SelectorEventLoop))