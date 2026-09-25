## Zepto Support Assistant

A local, offline-first RAG (Retrieval-Augmented Generation) service for
answering Zepto policy questions. Built with local embeddings, ChromaDB,
LangGraph orchestration, and a FastAPI wrapper.
---

## Project structure

```
M3-Support Assistant/
├── docs/
│   ├── doc_01.txt          # Delivery Policy
│   ├── doc_02.txt          # Returns & Refunds
│   ├── doc_03.txt          # Membership Tiers
│   ├── doc_04.txt          # Order Tracking
│   ├── doc_05.txt          # Order Cancellation Policy
│   ├── doc_06.txt          # Damaged or Missing Items
│   ├── doc_07.txt          # Gift Cards
│   └── doc_08.txt          # Customer Support Hours
├── chroma_db/              # Local vector store, auto-created/rebuilt on startup
├── rag_pipeline.py         # Core pipeline — graded baseline, fully offline/mock
├── real_llm.py             # Optional real-LLM extension (only used if MOCK_LLM=0)
├── main.py                 # FastAPI app exposing POST /ask
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Policy documents

The 8 files in `docs/` cover:

| File | Topic |
|---|---|
| `doc_01.txt` | Delivery Policy — timing, fees, priority delivery |
| `doc_02.txt` | Returns & Refunds |
| `doc_03.txt` | Membership Tiers (Basic, Zepto Pass, Zepto Pass+) |
| `doc_04.txt` | Order Tracking |
| `doc_05.txt` | Order Cancellation Policy |
| `doc_06.txt` | Damaged or Missing Items |
| `doc_07.txt` | Gift Cards |
| `doc_08.txt` | Customer Support Hours |

`build_vector_store()` requires exactly 8 files matching `docs/doc_*.txt`
and raises a `ValueError` otherwise.

---

## How it works

### 1. Indexing

On startup, `build_vector_store()` (in `rag_pipeline.py`) reads the 8
policy documents above, embeds each one locally using
`sentence-transformers/all-MiniLM-L6-v2`, and stores the vectors in
ChromaDB (`chroma_db/`) with cosine similarity. Existing entries are
cleared and rebuilt every time this runs, so repeated starts never
produce duplicate chunks.

### 2. Query flow — LangGraph, 3 nodes

```
        START
          │
   classify_intent
          │
    ┌─────┴─────┐
policy_question   general_question
    │                 │
retrieve_and_answer  direct_answer
    │                 │
    └────────┬────────┘
             END
```

- **`classify_intent`** — matches the query against a fixed keyword list:
  `delivery`, `return`, `refund`, `membership`, `tracking`, `cancel`,
  `gift card`, `support hours`.
- **`retrieve_and_answer`** — embeds the query, retrieves the top-3 most
  similar policy chunks from ChromaDB, and (in mock mode) builds a
  grounded answer directly from the top chunk's text.
- **`direct_answer`** — handles non-policy questions without any
  retrieval step.

### 3. Structured output

Every response is validated against a Pydantic `AnswerResponse` model:

```python
answer: str
sources: list[str]
confidence: float  # 0.0–1.0
```

---

## Modes: mock vs. real LLM

Controlled by the `MOCK_LLM` environment variable.

| `MOCK_LLM` | Behavior |
|---|---|
| unset or `1` (default) | **Graded baseline.** Fully deterministic, offline, rule-based. No API key or network call. |
| `0` | **Optional extension**, defined in `real_llm.py` but not exercised or verified in this build. Would require `GROQ_API_KEY` and `langchain-groq` (listed in `requirements.txt`) if used. |

`real_llm.py` is only imported inside `rag_pipeline.py`'s node functions
when `MOCK_LLM=0` — it has no effect on the offline mock path this
project has been run and tested in.

---

## Dependencies (`requirements.txt`)

```
fastapi
uvicorn
pydantic
langgraph
chromadb
sentence-transformers
langchain-groq
```

---

## Setup

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Running

### Run the pipeline directly (CLI demo)

```powershell
py rag_pipeline.py
```

Builds the vector store and prints the active `MOCK_LLM` value, then two
example answers (one policy question, one general question) as JSON.

### Run the FastAPI service

```powershell
uvicorn main:app --reload
```

Open **http://127.0.0.1:8000/docs** for the interactive Swagger UI, or:

```powershell
curl -X POST http://127.0.0.1:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"What is the delivery fee?\"}"
```

---

## Dockerfile (as provided)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

```powershell
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```

The API is served at **http://localhost:7860/docs** once the container
is running. This build/run sequence has not been executed in this
session, so treat it as unverified until you run it yourself.

---

## Status notes

- Verified in this session: the 3 Python files' logic, all 8 `docs/*.txt`
  contents, `requirements.txt` contents, and `Dockerfile` contents.
- Not verified in this session: an actual `uvicorn` run, an actual
  `docker build`/`docker run`, and the `MOCK_LLM=0` / Groq real-LLM path.
- `main.py` defines the endpoint as `POST /ask`, not `/support_assistant/ask`.