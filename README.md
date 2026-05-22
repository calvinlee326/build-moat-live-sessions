# 現代系統設計 — Live Session Exercises

Personal fork of the live session exercises from **現代系統設計** by Terry & Bohr.

> Original repo: [bohr109/build-moat-live-sessions](https://github.com/bohr109/build-moat-live-sessions)

Each folder is a standalone system design exercise: a real working prototype you build from scratch or via a guided scaffold, then verify end-to-end.

---

## Exercises

| Folder | What You Build | Key Design Concepts |
|--------|---------------|---------------------|
| [`qr_code_generator/`](./qr_code_generator/) | Dynamic QR code shortener with redirect tracking | Dynamic vs static QR, cache-first redirect, token collision handling |
| [`chatgpt_task/`](./chatgpt_task/) | MCP task scheduler with watcher + worker | Watcher/queue/worker separation, time bucket partitioning, MCP tool registry |
| [`knowledge_base_qa_bot/`](./knowledge_base_qa_bot/) | RAG-powered Q&A bot over Markdown docs | BM25 vs vector retrieval, grounded answers, citation quality |

---

## How Each Exercise Works

Every exercise folder has the same structure:

```
<exercise>/
├── PROMPT.md        # Spec + design questions + verification tests
├── README.md        # Setup, tracks, and bonus challenges
├── scaffold/        # Guided track: fill in the TODOs
└── answers/         # Reference solution
```

**Steps:**
1. Read `PROMPT.md` — answer the design questions before coding
2. Pick a track: **Challenge** (build from scratch) or **Guided** (fill in TODOs in `scaffold/`)
3. Verify your prototype passes the tests in `PROMPT.md`
4. Check `answers/` to compare approaches

---

## Quick Start

```bash
# QR Code Generator
cd qr_code_generator/scaffold
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# ChatGPT Task Scheduler
cd chatgpt_task/scaffold
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m app.mcp_server

# Knowledge Base Q&A Bot
cd knowledge_base_qa_bot/scaffold/vector_rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
uvicorn app.main:app --reload
```

---

## Keeping Up with the Original

This repo is a fork. To pull in new exercises from the original:

```bash
git fetch upstream
git merge upstream/main
git push origin main
```

---

## Attribution

Exercises designed by **Terry & Bohr** as part of the 現代系統設計 live session series.
