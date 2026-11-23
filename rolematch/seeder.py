"""Seed the database in small batches (safe within serverless time limits).

Each call generates SEED_BATCH listings, embeds them, and inserts. Called
repeatedly by the admin endpoint until it reports done=true. Idempotent: it
stops early when the target count is already present.
"""

from __future__ import annotations

import asyncpg

from . import config, db, embeddings, seed_data


async def jobs_count(conn: asyncpg.Connection) -> int:
    return await conn.fetchval("SELECT count(*) FROM jobs")


async def seed_batch(conn: asyncpg.Connection) -> dict:
    """Seed one batch. Returns progress for the caller loop."""
    existing = await jobs_count(conn)
    target = config.SEED_TOTAL
    if existing >= target:
        return {"seeded": 0, "total": existing, "target": target, "done": True}

    batch = min(config.SEED_BATCH, target - existing)
    jobs = seed_data.generate_jobs(batch, offset=existing)
    texts = [seed_data.embed_text_for(j) for j in jobs]
    vectors = await embeddings.embed_texts(texts)

    async with conn.transaction():
        await conn.executemany(
            """
            INSERT INTO jobs (title, company, location, work_mode,
                              experience_min, experience_max,
                              salary_min_lpa, salary_max_lpa,
                              skills, description, posted_days_ago, embedding)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
            """,
            [
                (
                    j["title"], j["company"], j["location"], j["work_mode"],
                    j["experience_min"], j["experience_max"],
                    j["salary_min_lpa"], j["salary_max_lpa"],
                    j["skills"], j["description"], j["posted_days_ago"], v,
                )
                for j, v in zip(jobs, vectors)
            ],
        )

    total = existing + batch
    if total >= target:
        await db.ensure_vector_index(conn)
    return {"seeded": batch, "total": total, "target": target, "done": total >= target}
