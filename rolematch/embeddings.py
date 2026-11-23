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
            # gemini-embedding-001 natively returns 3072 dims; pin to the
            # schema dimension so it stays compatible with the vector column.
            "outputDimensionality": config.EMBEDDING_DIM,
        }
        for t in texts
    ]
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(
            f"{GEMINI_BASE}/models/{config.EMBEDDING_MODEL}:batchEmbedContents",
            headers={"x-goog-api-key": config.GEMINI_API_KEY},
            json={"requests": requests},
        )
        if resp.status_code != 200:
            # include Google's error body (quota detail); URL carries no secret
            raise httpx.HTTPStatusError(
                f"{resp.status_code}: {resp.text[:1500]}",
                request=resp.request,
                response=resp,
            )
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
        # free-tier embedding quota is per-minute; back off on 429
        for attempt, wait in enumerate((0, 15, 30)):
            try:
                out.extend(await embed(chunk))
                break
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 429 or attempt == 2:
                    raise
                await asyncio.sleep(wait)
    return out


async def embed_query(text: str) -> list[float]:
    vectors = await embed_texts([text])
    return vectors[0]
