"""Unit tests for the pure ranking/scoring logic."""

from rolematch.search import (
    build_filter_sql,
    location_score,
    metadata_score,
    recency_score,
    rerank,
    salary_score,
    skills_score,
    work_mode_score,
)


def make_job(**overrides):
    job = {
        "job_id": 1,
        "title": "Backend Engineer",
        "company": "Zylo Labs",
        "location": "Bengaluru",
        "work_mode": "hybrid",
        "experience_min": 0.0,
        "experience_max": 2.0,
        "salary_min_lpa": 8.0,
        "salary_max_lpa": 14.0,
        "skills": ["Python", "FastAPI", "PostgreSQL"],
        "description": "Build APIs.",
        "posted_days_ago": 2,
        "_similarity": 0.8,
    }
    job.update(overrides)
    return job


def test_skills_score_exact_match():
    assert skills_score(["Python", "FastAPI"], ["python"]) == 1.0


def test_skills_score_partial_and_neutral():
    assert skills_score(["Python"], ["python", "go"]) == 0.5
    assert skills_score(["Python"], []) == 0.5


def test_location_score():
    assert location_score("Bengaluru", "bengaluru") == 1.0
    assert location_score("Remote - India", "remote") == 1.0
    assert location_score("Pune", "bengaluru") == 0.0
    assert location_score("Pune", None) == 0.5


def test_work_mode_score():
    assert work_mode_score("remote", "remote") == 1.0
    assert work_mode_score("on-site", "remote") == 0.0
    assert work_mode_score("on-site", None) == 0.5


def test_salary_score():
    assert salary_score(8, 14, 10) == 1.0
    assert salary_score(8, 9, 10) == 0.5  # 10% short -> partial credit within the 20% band
    assert salary_score(8, 7, 10) == 0.0  # 30% short -> no credit
    assert salary_score(None, None, 10) == 0.5
    assert salary_score(8, 14, None) == 0.5


def test_recency_score():
    assert recency_score(0) == 1.0
    assert recency_score(30) == 0.0
    assert 0 < recency_score(15) < 1


def test_metadata_score_collects_reasons():
    job = make_job()
    score, reasons = metadata_score(
        job,
        {"skills": ["Python"], "location": "Bengaluru", "work_mode": "hybrid",
         "min_salary_lpa": 10},
    )
    assert score > 0.8
    assert any("skills" in r for r in reasons)
    assert any("location" in r for r in reasons)


def test_rerank_prefers_better_blend():
    weak = make_job(job_id=1, _similarity=0.9, skills=["Go"], work_mode="on-site",
                    location="Pune", salary_min_lpa=4, salary_max_lpa=6,
                    posted_days_ago=29)
    strong = make_job(job_id=2, _similarity=0.7)
    prefs = {"skills": ["Python"], "location": "Bengaluru", "work_mode": "hybrid"}
    ranked = rerank([weak, strong], prefs, limit=2)
    assert ranked[0]["job_id"] == 2
    assert all("_similarity" not in j for j in ranked)
    assert len(ranked) == 2


def test_rerank_limit():
    jobs = [make_job(job_id=i) for i in range(10)]
    assert len(rerank(jobs, {}, limit=3)) == 3


def test_build_filter_sql():
    where, params = build_filter_sql(
        {"location": "Bengaluru", "work_mode": "hybrid", "min_salary_lpa": 10,
         "skills": ["Python"]},
        start_idx=2,
    )
    assert "location ILIKE" in where
    assert "work_mode = $3" in where
    assert "salary_max_lpa >= $4" in where
    assert "skills && $5::text[]" in where
    assert params == ["Bengaluru", "hybrid", 10.0, ["Python"]]


def test_build_filter_sql_empty():
    assert build_filter_sql({}, start_idx=2) == ("", [])
