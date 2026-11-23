"""Environment-driven configuration. No secrets in code, ever."""

from __future__ import annotations

import os


def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


# Database. Vercel Postgres (Neon) injects POSTGRES_URL when a store is
# connected to the project; DATABASE_URL is the generic fallback.
DATABASE_URL = _env("POSTGRES_URL", "DATABASE_URL", "DATABASE_URL_UNPOOLED")

# LLM provider. "gemini" runs on the free tier; "openai" works with your
# own key for local runs or if you attach one later.
LLM_PROVIDER = _env("LLM_PROVIDER", default="gemini").lower()
GEMINI_API_KEY = _env("GEMINI_API_KEY")
OPENAI_API_KEY = _env("OPENAI_API_KEY")

CHAT_MODEL = _env("CHAT_MODEL", default="gemini-3.1-flash-lite")
OPENAI_CHAT_MODEL = _env("OPENAI_CHAT_MODEL", default="gpt-4o-mini")
# Fallbacks keep the agent responsive when a model is rate-limited.
GEMINI_FALLBACK_MODELS = ["gemini-3.1-flash-lite", "gemini-flash-lite-latest", "gemini-3.8-flash"]

EMBEDDING_MODEL = _env("EMBEDDING_MODEL", default="gemini-embedding-001")
OPENAI_EMBEDDING_MODEL = _env("OPENAI_EMBEDDING_MODEL", default="text-embedding-3-small")

# Vector dimensions must match the embedding model. text-embedding-004 -> 768,
# text-embedding-3-small -> 1536.
EMBEDDING_DIMS = {"gemini": 768, "openai": 1536}
EMBEDDING_DIM = EMBEDDING_DIMS.get(LLM_PROVIDER, 768)

# Protects POST /api/admin/seed.
SEED_SECRET = _env("SEED_SECRET")

# How many listings the seeder generates (synthetic sample dataset).
SEED_TOTAL = int(_env("SEED_TOTAL", default="3000"))
SEED_BATCH = int(_env("SEED_BATCH", default="100"))

# Chat history window sent to the model per turn.
HISTORY_LIMIT = 10
