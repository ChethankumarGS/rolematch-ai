"""FastAPI app, served by Vercel's Python runtime from api/index.py.

Routes live under /api/*; everything else is static from public/.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import sys
from pathlib import Path

# Vercel runs api/index.py with the repo root on sys.path for the runtime's
# own installs, but not always for sibling packages - make it explicit.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rolematch import agent, config, db, seeder  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_schema()
    yield
    await db.close_pool()


app = FastAPI(title="RoleMatch AI", version="0.1.0", lifespan=lifespan)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class SaveIn(BaseModel):
    job_id: int


def _session_cookie(response: Response, session_id: uuid.UUID) -> None:
    response.set_cookie(
        "rolematch_session",
        str(session_id),
        max_age=60 * 60 * 24 * 30,
        httponly=True,
        samesite="lax",
        secure=True,
    )


async def _get_or_create_session(request: Request, response: Response) -> uuid.UUID:
    pool = await db.get_pool()
    raw = request.cookies.get("rolematch_session")
    async with pool.acquire() as conn:
        if raw:
            try:
                session_id = uuid.UUID(raw)
            except ValueError:
                session_id = None
            if session_id is not None:
                exists = await conn.fetchval(
                    "SELECT 1 FROM sessions WHERE id = $1", session_id
                )
                if exists:
                    return session_id
        session_id = uuid.uuid4()
        await conn.execute("INSERT INTO sessions (id) VALUES ($1)", session_id)
    _session_cookie(response, session_id)
    return session_id


@app.get("/api/health")
async def health():
    pool = await db.get_pool()
    async with pool.acquire() as conn:
        count = await seeder.jobs_count(conn)
    return {"ok": True, "jobs": count, "provider": config.LLM_PROVIDER}


@app.post("/api/chat")
async def chat(payload: ChatIn, request: Request, response: Response):
    session_id = await _get_or_create_session(request, response)
    pool = await db.get_pool()
    async with pool.acquire() as conn:
        result = await agent.handle_message(conn, session_id, payload.message.strip())
    return result


@app.get("/api/saved")
async def list_saved(request: Request, response: Response):
    session_id = await _get_or_create_session(request, response)
    pool = await db.get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT j.* FROM saved_jobs s JOIN jobs j ON j.id = s.job_id "
            "WHERE s.session_id = $1 ORDER BY s.saved_at DESC",
            session_id,
        )
        return {"count": len(rows), "jobs": [db.row_to_job(r) for r in rows]}


@app.post("/api/save")
async def save(payload: SaveIn, request: Request, response: Response):
    session_id = await _get_or_create_session(request, response)
    pool = await db.get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO saved_jobs (session_id, job_id) VALUES ($1, $2) "
            "ON CONFLICT DO NOTHING",
            session_id,
            payload.job_id,
        )
    return {"saved": True, "job_id": payload.job_id}


@app.delete("/api/save/{job_id}")
async def unsave(job_id: int, request: Request, response: Response):
    session_id = await _get_or_create_session(request, response)
    pool = await db.get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM saved_jobs WHERE session_id = $1 AND job_id = $2",
            session_id,
            job_id,
        )
    return {"removed": True, "job_id": job_id}


@app.post("/api/admin/seed")
async def admin_seed(x_seed_secret: str | None = Header(default=None)):
    if not config.SEED_SECRET or x_seed_secret != config.SEED_SECRET:
        raise HTTPException(status_code=401, detail="bad seed secret")
    pool = await db.get_pool()
    async with pool.acquire() as conn:
        try:
            return await seeder.seed_batch(conn)
        except Exception as exc:  # noqa: BLE001 - surface sanitized cause for debugging
            import re

            detail = re.sub(r"key=[^&\s]+", "key=REDACTED", str(exc))[:1200]
            return JSONResponse(
                status_code=500,
                content={"error": type(exc).__name__, "detail": detail},
            )

import logging as _logging
_logging.getLogger("httpx").setLevel(_logging.WARNING)
_logging.getLogger("google_genai").setLevel(_logging.WARNING)
