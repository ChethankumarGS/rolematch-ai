"""Hybrid job search: pgvector cosine candidates + metadata re-ranking.

Stage 1 (SQL): hard filters + nearest-neighbor candidate fetch by cosine
distance on the stored embedding.
Stage 2 (Python): re-rank candidates with a transparent weighted blend of
semantic similarity and structured metadata fit, and explain each match.
The scoring functions are pure so they are unit-testable without a database.
"""

from __future__ import annotations

from typing import Any

SIM_WEIGHT = 0.65
META_WEIGHT = 0.35

# Metadata sub-weights (sum to 1.0)
W_SKILLS = 0.40
W_LOCATION = 0.20
W_WORK_MODE = 0.15
W_SALARY = 0.15
W_RECENCY = 0.10


def skills_score(job_skills: list[str], wanted: list[str]) -> float:
    if not wanted:
        return 0.5  # neutral when the user gave no skill signal
    job_set = {s.lower() for s in job_skills}
    hits = sum(1 for s in wanted if s.lower() in job_set)
    return hits / len(wanted)


def location_score(job_location: str, wanted: str | None) -> float:
    if not wanted:
        return 0.5
    wanted = wanted.strip().lower()
    if not wanted:
        return 0.5
    job = job_location.lower()
    if wanted in job or job in wanted:
        return 1.0
    if wanted == "remote" and "remote" in job:
        return 1.0
    return 0.0


def work_mode_score(job_mode: str, wanted: str | None) -> float:
    if not wanted:
        return 0.5
    return 1.0 if job_mode.lower() == wanted.lower() else 0.0


def salary_score(min_lpa: float | None, max_lpa: float | None, wanted_min: float | None) -> float:
    if wanted_min is None:
        return 0.5
    if max_lpa is None:
        return 0.5
    if max_lpa >= wanted_min:
        return 1.0
    # partial credit when the band is close to the ask (within 20%)
    gap = (wanted_min - max_lpa) / wanted_min
    return max(0.0, 1.0 - gap * 5)


def recency_score(posted_days_ago: int) -> float:
    return max(0.0, 1.0 - posted_days_ago / 30.0)


def metadata_score(job: dict[str, Any], prefs: dict[str, Any]) -> tuple[float, list[str]]:
    """Blend structured fit signals; also return human-readable reasons."""
    reasons: list[str] = []

    s_score = skills_score(job["skills"], prefs.get("skills") or [])
    if prefs.get("skills") and s_score > 0:
        matched = [
            s for s in prefs["skills"] if s.lower() in {j.lower() for j in job["skills"]}
        ]
        if matched:
            reasons.append("skills: " + ", ".join(matched[:5]))

    l_score = location_score(job["location"], prefs.get("location"))
    if prefs.get("location") and l_score == 1.0:
        reasons.append(f"location: {job['location']}")

    w_score = work_mode_score(job["work_mode"], prefs.get("work_mode"))
    if prefs.get("work_mode") and w_score == 1.0:
        reasons.append(f"{job['work_mode']} role")

    sal_score = salary_score(job["salary_min_lpa"], job["salary_max_lpa"], prefs.get("min_salary_lpa"))
    if prefs.get("min_salary_lpa") and sal_score >= 1.0:
        reasons.append(f"pays >= {prefs['min_salary_lpa']} LPA")

    r_score = recency_score(job.get("posted_days_ago", 7))
    if job.get("posted_days_ago", 7) <= 3:
        reasons.append("posted recently")

    score = (
        W_SKILLS * s_score
        + W_LOCATION * l_score
        + W_WORK_MODE * w_score
        + W_SALARY * sal_score
        + W_RECENCY * r_score
    )
    return score, reasons


def rerank(
    candidates: list[dict[str, Any]], prefs: dict[str, Any], limit: int
) -> list[dict[str, Any]]:
    """Blend cosine similarity with metadata fit and attach explanations."""
    for job in candidates:
        meta, reasons = metadata_score(job, prefs)
        sim = job.pop("_similarity", 0.0)
        job["match_score"] = round(SIM_WEIGHT * sim + META_WEIGHT * meta, 3)
        job["match_reasons"] = reasons
    candidates.sort(key=lambda j: j["match_score"], reverse=True)
    return candidates[:limit]


def build_filter_sql(filters: dict[str, Any], start_idx: int) -> tuple[str, list[Any]]:
    """Hard SQL filters from structured args. Returns (where_clause, params)."""
    clauses: list[str] = []
    params: list[Any] = []
    idx = start_idx

    if filters.get("location"):
        clauses.append(f"(location ILIKE '%' || ${idx} || '%' OR work_mode = 'remote')")
        params.append(filters["location"])
        idx += 1
    if filters.get("work_mode"):
        clauses.append(f"work_mode = ${idx}")
        params.append(filters["work_mode"].lower())
        idx += 1
    if filters.get("min_salary_lpa") is not None:
        clauses.append(f"salary_max_lpa >= ${idx}")
        params.append(float(filters["min_salary_lpa"]))
        idx += 1
    if filters.get("experience_years") is not None:
        clauses.append(f"experience_min <= ${idx}")
        params.append(float(filters["experience_years"]))
        idx += 1
    if filters.get("skills"):
        clauses.append(f"skills && ${idx}::text[]")
        params.append([s for s in filters["skills"]])
        idx += 1

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


async def hybrid_search(
    conn,
    query_text: str,
    query_embedding: list[float],
    filters: dict[str, Any],
    prefs: dict[str, Any],
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Vector candidate fetch, then Python re-rank with explanations."""
    where, params = build_filter_sql(filters, start_idx=2)
    sql = f"""
        SELECT id, title, company, location, work_mode,
               experience_min, experience_max, salary_min_lpa, salary_max_lpa,
               skills, description, posted_days_ago,
               1 - (embedding <=> $1) AS similarity
        FROM jobs
        {where}
        ORDER BY embedding <=> $1
        LIMIT 40
    """
    rows = await conn.fetch(sql, query_embedding, *params)

    from .db import row_to_job

    candidates = []
    for row in rows:
        job = row_to_job(row)
        job["_similarity"] = float(row["similarity"] or 0.0)
        candidates.append(job)

    merged_prefs = {**prefs, **{k: v for k, v in filters.items() if v not in (None, "", [])}}
    return rerank(candidates, merged_prefs, limit)
