"""Postgres access layer: pool, schema, and small helpers.

Uses asyncpg with the pgvector codec. The schema is created idempotently at
startup so a fresh Vercel Postgres store works without manual migration steps.
"""

from __future__ import annotations

import ssl as ssl_module
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import asyncpg
from pgvector.asyncpg import register_vector

from . import config

_pool: asyncpg.Pool | None = None


async def _init_connection(conn: asyncpg.Connection) -> None:
    # The vector extension must exist before the codec can be registered.
    # IF NOT EXISTS makes this a cheap no-op after the first connection.
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await register_vector(conn)


SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id              SERIAL PRIMARY KEY,
    title           TEXT NOT NULL,
    company         TEXT NOT NULL,
    location        TEXT NOT NULL,
    work_mode       TEXT NOT NULL DEFAULT 'on-site',
    experience_min  REAL NOT NULL DEFAULT 0,
    experience_max  REAL NOT NULL DEFAULT 2,
    salary_min_lpa  REAL,
    salary_max_lpa  REAL,
    skills          TEXT[] NOT NULL DEFAULT '{}',
    description     TEXT NOT NULL,
    posted_days_ago INTEGER NOT NULL DEFAULT 7,
    embedding       vector({dim})
);
CREATE INDEX IF NOT EXISTS jobs_skills_gin ON jobs USING gin (skills);
CREATE INDEX IF NOT EXISTS jobs_location_idx ON jobs (location);

CREATE TABLE IF NOT EXISTS sessions (
    id          UUID PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    prefs       JSONB NOT NULL DEFAULT '{{}}'::jsonb
);

CREATE TABLE IF NOT EXISTS messages (
    id          BIGSERIAL PRIMARY KEY,
    session_id  UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS messages_session_idx ON messages (session_id, id);

CREATE TABLE IF NOT EXISTS saved_jobs (
    session_id  UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    job_id      INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    saved_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, job_id)
);
"""


def _normalize_dsn(dsn: str) -> str:
    """Make a provider DSN asyncpg-friendly and force TLS for cloud hosts."""
    parts = urlsplit(dsn)
    query = dict(parse_qsl(parts.query))
    scheme = parts.scheme.replace("postgres://", "postgresql://")
    # asyncpg >= 0.29 understands sslmode in the query string, keep it if set.
    if "sslmode" not in query and parts.hostname and "localhost" not in parts.hostname:
        query["sslmode"] = "require"
    return urlunsplit((scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        if not config.DATABASE_URL:
            raise RuntimeError(
                "No database configured. Set POSTGRES_URL (or DATABASE_URL)."
            )
        _pool = await asyncpg.create_pool(
            _normalize_dsn(config.DATABASE_URL),
            min_size=1,
            max_size=4,
            init=_init_connection,
        )
    return _pool


async def init_schema() -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(SCHEMA.format(dim=config.EMBEDDING_DIM))


async def ensure_vector_index(conn: asyncpg.Connection) -> None:
    """HNSW index after bulk seeding (fast to build, fast to query)."""
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS jobs_embedding_hnsw "
        "ON jobs USING hnsw (embedding vector_cosine_ops)"
    )


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def row_to_job(row: Any) -> dict[str, Any]:
    return {
        "job_id": row["id"],
        "title": row["title"],
        "company": row["company"],
        "location": row["location"],
        "work_mode": row["work_mode"],
        "experience_min": row["experience_min"],
        "experience_max": row["experience_max"],
        "salary_min_lpa": row["salary_min_lpa"],
        "salary_max_lpa": row["salary_max_lpa"],
        "skills": list(row["skills"] or []),
        "description": row["description"],
        "posted_days_ago": row["posted_days_ago"],
    }
