"""One-off setup over the DIRECT Neon endpoint (no -pooler): extensions, migrations,
checkpointer tables. Run before the app starts, and again after schema changes:

    python setup_db.py
"""

import asyncio
import psycopg
from settings import get_settings

async def main() -> None:
    settings = get_settings()
    url = settings.database_url_direct.get_secret_value()

    async with await psycopg.AsyncConnection.connect(
        url, autocommit=True, connect_timeout=settings.db_connect_timeout_s
    ) as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")


    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError:
        return

    async with AsyncPostgresSaver.from_conn_string(url) as saver:
        await saver.setup()

if __name__ == "__main__":
    asyncio.run(main(), loop_factory=asyncio.SelectorEventLoop)
