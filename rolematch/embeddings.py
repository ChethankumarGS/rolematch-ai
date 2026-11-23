"""Embedding clients. Gemini (free tier) by default; OpenAI as an option."""

from __future__ import annotations

import asyncio

import httpx

from . import config

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
OPENAI_BASE = "https://api.openai.com/v1"

# Gemini free tier accepts up to 100 texts per batchEmbedContents call.
BATCH_SIZE = 100
_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


async def _gemini_batch(texts: list[str]) -> list[list[float]]:
    requests = [
        {
            "model": f"models/{config.EMBEDDING_MODEL}",
            "content": {"parts": [{"text": t}]},
        }
        for t in texts
    ]
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(
            f"{GEMINI_BASE}/models/{config.EMBEDDING_MODEL}:batchEmbedContents",
            headers={"x-goog-api-key": config.GEMINI_API_KEY},
            json={"requests": requests},
        )
        resp.raise_for_status()
        data = resp.json()
    return [item["values"] for item in data["embeddings"]]


async def _openai_batch(texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(
            f"{OPENAI_BASE}/embeddings",
            headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
            json={"model": config.OPENAI_EMBEDDING_MODEL, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()
    return [item["embedding"] for item in sorted(data["data"], key=lambda d: d["index"])]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts, chunked to provider batch limits."""
    if not texts:
        return []
    embed = _gemini_batch if config.LLM_PROVIDER == "gemini" else _openai_batch
    out: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        chunk = texts[i : i + BATCH_SIZE]
        # one retry with backoff handles transient free-tier rate limits
        for attempt in (0, 1):
            try:
                out.extend(await embed(chunk))
                break
            except httpx.HTTPStatusError:
                if attempt == 1:
                    raise
                await asyncio.sleep(3)
    return out


async def embed_query(text: str) -> list[float]:
    vectors = await embed_texts([text])
    return vectors[0]
