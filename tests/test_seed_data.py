"""Tests for the synthetic listing generator."""

from rolematch.seed_data import generate_jobs, embed_text_for


def test_deterministic():
    a = generate_jobs(20, offset=0)
    b = generate_jobs(20, offset=0)
    assert a == b


def test_offset_changes_output():
    a = generate_jobs(10, offset=0)
    b = generate_jobs(10, offset=10)
    assert a != b


def test_shape_and_ranges():
    jobs = generate_jobs(200)
    for j in jobs:
        assert j["title"] and j["company"] and j["location"]
        assert j["work_mode"] in {"remote", "hybrid", "on-site"}
        assert j["salary_min_lpa"] <= j["salary_max_lpa"]
        assert j["experience_min"] <= j["experience_max"]
        assert 3 <= len(j["skills"]) <= 7
        assert 0 <= j["posted_days_ago"] <= 30
        assert j["work_mode"] == "remote" or not j["location"].startswith("Remote")


def test_embed_text_contains_key_fields():
    job = generate_jobs(1)[0]
    text = embed_text_for(job)
    assert job["title"] in text
    assert job["company"] in text
    for skill in job["skills"]:
        assert skill in text


def test_skill_pool_diversity():
    jobs = generate_jobs(500)
    all_skills = {s for j in jobs for s in j["skills"]}
    assert len(all_skills) > 30
