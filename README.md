# Project Portfolio

This repository contains three independent modules:

| Module | Folder | What it does |
|---|---|---|
| 1. Book Catalog Data Pipeline | `data_pipeline/` | Scrapes book data, cleans it, loads it into SQLite, and queries it with SQL + pandas |
| 2. Titanic Analytics Pipeline | `analytics/` | EDA, data story, and a full classification/regression modeling pipeline on the Titanic dataset |
| 3. Zepto Support Assistant | `M3-Support Assistant/` | A local, offline-first RAG service (FastAPI + LangGraph + ChromaDB) for answering policy questions |

Each module is self-contained: it has its **own `requirements.txt`** (per-module dependencies, installed into its own environment — see "Setup" below for why), its own scripts/notebooks, and its own module-level `README.md` with full detail. This root README is the entry point: how to set each module up, how to run each one end to end, and a short summary of the design decisions behind each.

---

## Setup

**Dependency strategy: one `requirements.txt` per module (not a single consolidated file).**

The three modules have no shared runtime — Module 1 is a plain scripting/ETL pipeline (`requests`, `beautifulsoup4`, `pandas`), Module 2 is a notebook-based ML pipeline (`scikit-learn`, `imbalanced-learn`, `seaborn`, `joblib`), and Module 3 is a FastAPI/RAG service (`fastapi`, `langgraph`, `chromadb`, `sentence-transformers`). Merging these into one file would pull heavy, unrelated dependencies (e.g. `torch`-backed `sentence-transformers` for someone who only wants to run the data pipeline) into every environment and increase the chance of version conflicts between modules. Keeping them separate lets each module be installed and run in isolation, which also matches how they'd typically be deployed (Module 3 alone ships a `Dockerfile`).

For each module, create a virtual environment and install from that module's own `requirements.txt`:

```bash
# Module 1
cd data_pipeline
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Module 2
cd ../analytics
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Module 3
cd "../M3-Support Assistant"
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## How to run each module end to end

### Module 1 — Book Catalog Data Pipeline (`data_pipeline/`)

Four scripts, run in order — each reads the file(s) produced by the previous step:

```bash
python scrape_pipeline.py     # scrapes books.toscrape.com -> raw_books.csv
python clean_pipeline.py      # cleans raw_books.csv        -> clean_books.csv
python database_pipeline.py   # loads clean_books.csv        -> books.db
python query_pipeline.py      # runs SQL + pandas queries    -> query_outputs.txt
```

`database_pipeline.py` drops and recreates its tables on every run, so it (and every step after it) can be re-run safely at any time. `query_pipeline.py` is the module's most recently changed component — its updates were developed on the branch `feature/update-query-pipeline` and merged into `main`.

### Module 2 — Titanic Analytics Pipeline (`analytics/`)

Two Jupyter notebooks, run in order, each fully ("Run All") before moving to the next:

```bash
jupyter notebook 01_eda.ipynb      # load, profile, clean Titanic data -> titanic.csv (cleaned)
jupyter notebook 02_modeling.ipynb # read cleaned titanic.csv -> train/evaluate models -> best_titanic_pipeline.joblib
```

`01_eda.ipynb` is the only notebook that loads the raw dataset (`sns.load_dataset('titanic')`); `02_modeling.ipynb` only ever reads the cleaned `titanic.csv` that `01_eda.ipynb` produces, so the two notebooks must be run in this order on a first run. `titanic.csv` is also committed as an offline fallback, so `02_modeling.ipynb` can be run on its own if `01_eda.ipynb` isn't re-executed.

### Module 3 — Zepto Support Assistant (`M3-Support Assistant/`)

Either run the pipeline directly, or serve it as an API:

```bash
# Option A: CLI demo — builds the vector store, prints one policy and one general example answer
python rag_pipeline.py

# Option B: FastAPI service
uvicorn main:app --reload
# then open http://127.0.0.1:8000/docs, or:
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the delivery fee?"}'
```

The vector store (`chroma_db/`) is rebuilt from the 8 files in `docs/` every time the service starts (existing entries are cleared first), so it never needs to be pre-populated or committed for the pipeline to work.

By default (`MOCK_LLM` unset or `1`) the service runs fully offline and deterministically — no API key or network call. Setting `MOCK_LLM=0` switches to an optional real-LLM extension (`real_llm.py`, via Groq), which requires `GROQ_API_KEY` and is not exercised or verified as part of this build.

Optionally, via Docker:

```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```

---

## Design decisions

### Module 1 — Book Catalog Data Pipeline

- **Scraping (`scrape_pipeline.py`):** pulls from books.toscrape.com until at least 70 books across at least 3 categories are collected, with a small delay between requests to avoid overloading the server.
- **Cleaning (`clean_pipeline.py`):** prices are parsed from `£xx.xx` text to float; text ratings (`One`…`Five`) are mapped to integers 1–5; availability text is mapped to a boolean, with anything unrecognized treated as **out of stock** (unstated availability is assumed unsafe to sell against); rows missing `title` or `category` are dropped as unrecoverable identifying fields, while missing/invalid `price` and `rating` are **median-imputed** rather than dropped, since a bad price or rating alone doesn't invalidate an otherwise valid book record. GBP is converted to INR using a fixed project-defined rate (1 GBP = 105.50 INR), not a live rate.
- **Database (`database_pipeline.py`):** normalized into two tables, `categories` and `books`, linked by a foreign key on `category_id`. Tables are dropped and recreated on every run so the database can always be regenerated cleanly from `clean_books.csv`.
- **Querying (`query_pipeline.py`):** five SQL queries demonstrate `SELECT`/`WHERE`/`ORDER BY`/`LIMIT`/`DISTINCT`/`IN`/`JOIN`; the JOIN query is also independently reproduced using `pandas.merge()` on the raw tables, and the two results are compared for equality as a correctness check.
- **Workflow:** the query-pipeline update was developed on a feature branch (`feature/update-query-pipeline`) and merged into `main`, keeping that change isolated and reviewable before landing.

### Module 2 — Titanic Analytics Pipeline

- **Missing values:** handled with an explicit threshold rule rather than one blanket strategy — `deck` (77% missing) is dropped outright since imputing three-quarters of a column would be mostly fabricated data; `age` (~20% missing) is median-imputed as a robust central value; `embarked`/`embark_town` (~0.2% missing, same rows) have their few affected rows dropped, since the information loss is negligible.
- **EDA:** univariate checks (IQR outliers, skewness via mean/median/mode) and bivariate survival-rate breakdowns (by sex, by class, and combined) are used to surface the standard "women and higher class survive more" pattern before modeling; a correlation matrix quantifies the two strongest linear relationships (`pclass`↔`fare`, `sibsp`↔`parch`) without claiming causation. Standardization of `age`/`fare` at the EDA stage is exploratory only and doesn't feed into the saved `titanic.csv` or the modeling pipeline, keeping the cleaned handoff file in original units.
- **Modeling:** three classifiers (Logistic Regression, Decision Tree, Random Forest) are compared on an identical stratified 80/20 split; class-imbalance handling (baseline vs. `class_weight='balanced'` vs. SMOTE on the training fold only) is compared separately; Random Forest is tuned with `GridSearchCV` (cross-checked with `oob_score`). **Random Forest is the final recommended model** — chosen for the best F1/recall trade-off, since correctly identifying survivors matters more here than raw accuracy or AUC alone. The fitted pipeline (preprocessing + tuned model) is persisted with `joblib` and was verified to reproduce identical predictions on raw, unprocessed input after reloading.
- **Regression side-task:** a separate fare-prediction regression is included as an auxiliary exercise (not comparable to the classification metrics); residual analysis shows heteroscedasticity, reported honestly rather than glossed over.

### Module 3 — Zepto Support Assistant

- **Retrieval:** each of the 8 policy documents is treated as a single chunk, since each document is already short and topically self-contained — this keeps retrieval simple without needing a text-splitting strategy. Embeddings use a local, free model (`sentence-transformers/all-MiniLM-L6-v2`) and are stored in ChromaDB with cosine similarity, so the whole pipeline runs fully offline with no API key.
- **Orchestration:** a 3-node LangGraph workflow (`classify_intent` → `retrieve_and_answer` or `direct_answer`) keeps policy questions and general questions on separate, simple paths rather than one large prompt trying to handle both.
- **Mock-first design:** the graded baseline (`MOCK_LLM=1`, default) is fully deterministic and rule-based — keyword-matching for intent, and a direct extract from the top retrieved chunk for the answer — so it needs no LLM API key and is reproducible. The optional real-LLM path (`real_llm.py`, Groq) is isolated behind `MOCK_LLM=0` and only imported when explicitly enabled, so it has zero effect on the offline mock path.
- **Structured output:** every response, in either mode, is validated against a Pydantic `AnswerResponse` model (`answer`, `sources`, `confidence`), so the API contract is identical regardless of which mode generated the answer.
- **Idempotent indexing:** `build_vector_store()` clears and rebuilds the ChromaDB collection on every startup rather than appending, so repeated runs/deploys never leave duplicate chunks, and it enforces exactly 8 source documents (raising `ValueError` otherwise) to catch a malformed `docs/` folder early.
- **Verification scope:** the pipeline logic, all 8 policy documents, `requirements.txt`, and the `Dockerfile` contents were verified in development; an actual `uvicorn` run, an actual `docker build`/`docker run`, and the `MOCK_LLM=0` real-LLM path were not exercised and should be treated as unverified until run directly.