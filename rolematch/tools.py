"""Agent tools: schemas exposed to the LLM + their executors.

The agent calls these through function calling. Every tool returns plain
JSON-serialisable dicts. `search_jobs` results carry `position` numbers so the
model can resolve references like "save the second one".
"""

from __future__ import annotations

import json
from typing import Any

from . import config, embeddings, search

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "search_jobs",
        "description": (
            "Semantic + structured search over the job listings. Use it whenever "
            "the user describes a role, stack, location, salary or seniority. "
            "Pass the user's intent as a natural-language query plus any "
            "structured filters they mentioned."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language description of the ideal role.",
                },
                "location": {
                    "type": "string",
                    "description": "City or region, e.g. Bengaluru. Remote roles always match.",
                },
                "work_mode": {
                    "type": "string",
                    "enum": ["remote", "hybrid", "on-site"],
                    "description": "Preferred work mode.",
                },
                "min_salary_lpa": {
                    "type": "number",
                    "description": "Minimum acceptable salary in INR lakhs per year.",
                },
                "experience_years": {
                    "type": "number",
                    "description": "User's years of experience; roles requiring more are excluded.",
                },
                "skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Skills the role should use, e.g. ['Python', 'FastAPI'].",
                },
                "limit": {"type": "integer", "description": "How many results, 1-10. Default 5."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_job_details",
        "description": "Fetch the full description and facts for one job by its id.",
        "parameters": {
            "type": "object",
            "properties": {"job_id": {"type": "integer"}},
            "required": ["job_id"],
        },
    },
    {
        "name": "save_job",
        "description": "Save a job to the user's shortlist so they can revisit it.",
        "parameters": {
            "type": "object",
            "properties": {"job_id": {"type": "integer"}},
            "required": ["job_id"],
        },
    },
    {
        "name": "list_saved_jobs",
        "description": "List everything currently on the user's saved shortlist.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "remove_saved_job",
        "description": "Remove a job from the user's saved shortlist.",
        "parameters": {
            "type": "object",
            "properties": {"job_id": {"type": "integer"}},
            "required": ["job_id"],
        },
    },
    {
        "name": "set_preferences",
        "description": (
            "Remember the user's durable preferences for this session "
            "(location, work mode, minimum salary, skills, experience). "
            "Later searches blend these automatically."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "work_mode": {"type": "string", "enum": ["remote", "hybrid", "on-site"]},
                "min_salary_lpa": {"type": "number"},
                "experience_years": {"type": "number"},
                "skills": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
]


def _clamp_limit(value: Any, default: int = 5) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(10, n))


def _clean_filters(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "location": args.get("location") or None,
        "work_mode": (args.get("work_mode") or "").lower() or None,
        "min_salary_lpa": args.get("min_salary_lpa"),
        "experience_years": args.get("experience_years"),
        "skills": [s for s in (args.get("skills") or []) if s][:8],
    }


async def execute_tool(
    name: str, args: dict[str, Any], conn, session_id, prefs: dict[str, Any]
) -> dict[str, Any]:
    if name == "search_jobs":
        query = (args.get("query") or "").strip()
        if not query:
            return {"error": "query is required"}
        filters = _clean_filters(args)
        vector = await embeddings.embed_query(query)
        jobs = await search.hybrid_search(
            conn, query, vector, filters, prefs, limit=_clamp_limit(args.get("limit"))
        )
        for i, job in enumerate(jobs, start=1):
            job["position"] = i
        return {
            "count": len(jobs),
            "jobs": jobs,
            "note": "jobs carry position numbers; use job_id for details/saves.",
        }

    if name == "get_job_details":
        row = await conn.fetchrow(
            "SELECT * FROM jobs WHERE id = $1", int(args["job_id"])
        )
        if not row:
            return {"error": f"no job with id {args['job_id']}"}
        from .db import row_to_job

        return {"job": row_to_job(row)}

    if name == "save_job":
        job_id = int(args["job_id"])
        exists = await conn.fetchval("SELECT 1 FROM jobs WHERE id = $1", job_id)
        if not exists:
            return {"error": f"no job with id {job_id}"}
        await conn.execute(
            "INSERT INTO saved_jobs (session_id, job_id) VALUES ($1, $2) "
            "ON CONFLICT DO NOTHING",
            session_id,
            job_id,
        )
        title = await conn.fetchval("SELECT title FROM jobs WHERE id = $1", job_id)
        company = await conn.fetchval("SELECT company FROM jobs WHERE id = $1", job_id)
        return {"saved": True, "job_id": job_id, "title": title, "company": company}

    if name == "list_saved_jobs":
        rows = await conn.fetch(
            "SELECT j.* FROM saved_jobs s JOIN jobs j ON j.id = s.job_id "
            "WHERE s.session_id = $1 ORDER BY s.saved_at DESC",
            session_id,
        )
        from .db import row_to_job

        return {"count": len(rows), "jobs": [row_to_job(r) for r in rows]}

    if name == "remove_saved_job":
        await conn.execute(
            "DELETE FROM saved_jobs WHERE session_id = $1 AND job_id = $2",
            session_id,
            int(args["job_id"]),
        )
        return {"removed": True, "job_id": int(args["job_id"])}

    if name == "set_preferences":
        updates = _clean_filters(args)
        merged = {**prefs, **{k: v for k, v in updates.items() if v not in (None, [], "")}}
        await conn.execute(
            "UPDATE sessions SET prefs = $1::jsonb WHERE id = $2",
            json.dumps(merged),
            session_id,
        )
        return {"preferences_saved": True, "preferences": merged}

    return {"error": f"unknown tool {name}"}


def to_gemini_declarations() -> list[dict[str, Any]]:
    """Gemini wants {function_declarations: [...]} with OpenAPI-ish schemas."""
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["parameters"],
        }
        for t in TOOL_SCHEMAS
    ]


def to_openai_tools() -> list[dict[str, Any]]:
    return [{"type": "function", "function": t} for t in TOOL_SCHEMAS]
