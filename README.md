# RoleMatch AI

A conversational job-discovery agent. You chat with it in plain language -
"hybrid backend roles in Bengaluru, Python, 8 LPA minimum" - and it searches,
ranks, explains and remembers, all through LLM tool calling.

**Live:** https://rolematch-ai-seven.vercel.app


## What it does

- **Conversational agent with tool use** - the LLM (function calling) decides
  when to search, fetch details, save roles or store your preferences.
- **Multi-turn memory** - session history and preferences persist in Postgres,
  so follow-ups like "save the second one" or "what did I save?" work.
- **Hybrid vector + SQL search** - pgvector cosine similarity over listing
  embeddings fetches candidates; hard SQL filters (location, work mode, salary,
  experience, skills) keep results valid.
- **Personalized re-ranking** - candidates are re-scored in Python with a
  transparent blend (0.65 semantic similarity + 0.35 metadata fit across
  skills, location, work mode, salary and recency), and every card explains
  *why* it matched.
- **Shortlist** - save roles from chat or from the cards; they persist per
  session.

> The catalog is **3,000 synthetic sample listings** (deterministically
> generated) so the whole system is exercisable without scraping real job
> boards. The search, agent and ranking are fully working; the listings
> themselves are demo data, not live openings.

## Stack

Python · FastAPI · PostgreSQL (pgvector) · LLM function calling (Gemini by
default, OpenAI-compatible path included) · Docker · Vercel (serverless
deployment). The Dockerfile also runs on any container host, including AWS.

## Architecture

```
public/            chat UI (vanilla JS, no build step)
api/index.py       FastAPI routes: /api/chat, /api/saved, /api/save, /api/admin/seed
rolematch/
  agent.py         multi-turn tool loop, session memory
  tools.py         tool schemas + executors (search, details, save, prefs)
  search.py        hybrid retrieval + re-ranking (pure scoring functions)
  embeddings.py    batch embedding client (Gemini / OpenAI)
  llm.py           chat client with function calling, model fallbacks
  seed_data.py     deterministic synthetic listing generator
  seeder.py        batched, idempotent seeding
```

## Running locally

With Docker (Postgres with pgvector included):

```bash
cp .env.example .env       # add your GEMINI_API_KEY (free tier works)
docker compose up --build
```

Then seed the sample catalog:

```bash
curl -X POST http://localhost:8000/api/admin/seed -H "X-Seed-Secret: dev-secret"
# repeat until {"done": true} (each call seeds one batch)
```

Open http://localhost:8000

Or bare-metal: create a Postgres 16 db with the `vector` extension, export
`POSTGRES_URL`, `pip install -r requirements.txt`,
`uvicorn api.index:app --reload`.

## Configuration

| Variable | Purpose |
| --- | --- |
| `POSTGRES_URL` | Postgres connection string (auto-injected by Vercel Postgres) |
| `LLM_PROVIDER` | `gemini` (default, free tier) or `openai` |
| `GEMINI_API_KEY` | Gemini key for chat + embeddings |
| `OPENAI_API_KEY` | Only when `LLM_PROVIDER=openai` |
| `CHAT_MODEL` / `EMBEDDING_MODEL` | Override default models |
| `SEED_SECRET` | Protects the admin seed endpoint |
| `SEED_TOTAL` / `SEED_BATCH` | Catalog size and per-call batch |

## Deploying

The repo is Vercel-ready: `api/index.py` is the Python serverless entry,
`public/` is served as static assets. Create a Vercel Postgres store, connect
it to the project (it injects `POSTGRES_URL`), set `GEMINI_API_KEY` and
`SEED_SECRET`, deploy, then call the seed endpoint until done.

## Tests

```bash
pytest
```

Covers the ranking blend, filter SQL, seed-generator shape and tool plumbing.

## Honest limitations

- Listings are synthetic sample data; wiring a real board is a data-source
  change behind the same schema.
- Sessions are cookie-based (no accounts), so a shortlist doesn't follow you
  across devices.
- The OpenAI provider path is written but only the Gemini path is exercised in
  the live deployment.
