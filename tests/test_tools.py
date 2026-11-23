"""Tests for tool plumbing (schemas and argument cleaning)."""

from rolematch.tools import (
    TOOL_SCHEMAS,
    _clean_filters,
    _clamp_limit,
    to_gemini_declarations,
    to_openai_tools,
)


def test_clamp_limit():
    assert _clamp_limit(None) == 5
    assert _clamp_limit(3) == 3
    assert _clamp_limit(99) == 10
    assert _clamp_limit(0) == 1
    assert _clamp_limit("abc") == 5


def test_clean_filters():
    out = _clean_filters(
        {"location": " Bengaluru ", "work_mode": "Remote", "skills": ["Python", "", None],
         "min_salary_lpa": 10}
    )
    assert out["work_mode"] == "remote"
    assert out["skills"] == ["Python"]
    assert out["min_salary_lpa"] == 10


def test_clean_filters_empty():
    out = _clean_filters({})
    assert out["location"] is None
    assert out["skills"] == []


def test_schemas_convert():
    gemini = to_gemini_declarations()
    assert len(gemini) == len(TOOL_SCHEMAS)
    names = {t["name"] for t in gemini}
    assert {"search_jobs", "save_job", "list_saved_jobs", "set_preferences"} <= names
    for decl in gemini:
        assert decl["description"]
        assert "parameters" in decl

    openai = to_openai_tools()
    assert all(t["type"] == "function" for t in openai)


def test_search_jobs_requires_query_schema():
    schema = next(t for t in TOOL_SCHEMAS if t["name"] == "search_jobs")
    assert schema["parameters"]["required"] == ["query"]
