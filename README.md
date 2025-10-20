**Overview**

This repository delivers an end-to-end Retrieval-Augmented Generation (RAG) stack for large PDF collections. Ingestion parses documents into text, tables, and images; metadata and assets are stored alongside semantic vectors so the runtime FastAPI service can retrieve relevant snippets and produce grounded Gemini answers. Everything is orchestrated with environment-based configuration and can run either via uv (recommended workflow) or Docker.

**System Architecture**
- **Ingestion (`python -m src.ingest`)**  
  1. Reads PDFs from `domaindata/` plus optional metadata from `domaindata/metadata.jsonl`.  
  2. Uses `unstructured` to partition each PDF into elements (text blocks, tables, figures).  
  3. Chunking logic (`src/chunking.py`) merges text elements with layout awareness; tables/images become separate assets.  
  4. Text chunks are embedded with a sentence-transformer model and written to Chroma (`storage/chroma`).  
  5. Images/tables are persisted in the asset store (`storage/assets/<doc_id>/...` with records in `storage/assets/assets.sqlite`).  
  6. Metadata (industries, countries, dates, asset IDs) is attached to each vector entry for filtering during retrieval.

- **Retrieval (`src/search.py`)**  
  - Given a query, Chroma returns the top-k chunks plus metadata.  
  - Optional metadata filters (industries, country codes, date ranges) are applied client-side.  
  - Lightweight re-ranking combines cosine similarity with token overlap.

- **Serving (`uvicorn src.serve:app`)**  
  - FastAPI UI/API endpoints accept questions and optional filters.  
  - Retrieved chunks and linked assets (tables/images) are assembled into a prompt.  
  - Gemini generates a grounded answer, with fallbacks if quota or API key is unavailable.  
  - Responses include references to chunk IDs and page numbers; uploaded assets are logged for traceability.

**Key Modules**
- `src/config.py` – centralizes paths, models, and environment variables (`DOMAINDATA_DIR`, `storage/`, Gemini settings).  
- `src/ingest.py` – main ingestion loop coordinating partitioning, chunking, embedding, and asset persistence.  
- `src/datastore.py` – SQLite wrapper for assets/failures.  
- `src/vectorstore.py` – Chroma wrapper using sentence-transformer embeddings.  
- `src/interpret.py` – Gemini helpers for on-demand summaries (used elsewhere if desired).  
- `src/search.py` – retrieval, re-ranking, and metadata filtering utilities.  
- `src/serve.py` – FastAPI routes, Gemini generation, and asset attachment.  
- `scripts/inspect_stores.py` – debugging utilities to inspect stored assets/vectors.

**Data Layout**
- `domaindata/` – PDFs plus `metadata.jsonl`. Each metadata entry should include:  
  `uuid`, `industries`, `date`, `country_codes`. UUIDs ideally match PDF filenames.  
- `storage/chroma` – persistent Chroma DB files (created automatically).  
- `storage/assets` – image/table snapshots and `assets.sqlite`.  
- `logs/` – runtime log files (created automatically).

**Environment & Dependencies**
- Python 3.10+. uv is preferred for local development (`uv venv`, `uv pip install -r requirements.txt`), but plain `python -m venv` works.  
- System packages (Linux/macOS): `poppler-utils`, `tesseract-ocr`, `libmagic-dev`, basic X libs for image handling. On Debian/Ubuntu you can install them with:
  ```bash
  sudo apt-get update && sudo apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    libmagic-dev
  ```
  Windows users should install Poppler/Tesseract separately and set `POPPLER_PATH`, `TESSERACT_PATH`.  
- Environment variables (via `.env`):  
  - `GEMINI_API_KEY` **(required)** for Gemini summarization/generation.  
  - Optional overrides (`GEMINI_GENERATE_MODEL`, `SENTENCE_TRANSFORMER_MODEL`, etc.) live in `src/config.py`.

**Workflow with uv**
1. `uv venv` (or `python -m venv .venv`)  
2. `uv pip install -r requirements.txt`  
3. It is assumed that PDFs + `metadata.jsonl` are in `domaindata/` directory in root.  
4. `uv run python -m src.ingest` to populate `storage/`.  
5. `uv run uvicorn src.serve:app --reload` and open `http://127.0.0.1:8000`.

**Running Modules Individually**
- Ingest: `uv run python -m src.ingest`  
- Inspect stores: `uv run python scripts/inspect_stores.py`  
- Serve API: `uv run uvicorn src.serve:app --reload`

**Docker Usage**
1. Build: `docker build -t isi-markets .`  
2. Run ingestion:  
   ```
   docker run --rm \
     -v "$(pwd)/domaindata:/app/domaindata" \
     -v "$(pwd)/storage:/app/storage" \
     --env-file .env \
     isi-markets ingest
   ```
3. Serve:  
   ```
   docker run --rm -p 8000:8000 \
     -v "$(pwd)/domaindata:/app/domaindata" \
     -v "$(pwd)/storage:/app/storage" \
     --env-file .env \
     isi-markets serve
   ```
Container entrypoint defaults to `serve`; pass `ingest` or a custom command as needed.
If you encounter CUDA-related errors, export `CUDA_VISIBLE_DEVICES=-1` before running ingestion or serving to force CPU execution.
GPU users can override this variable (e.g., `-e CUDA_VISIBLE_DEVICES=0`) when launching Docker.

**Important Notes**
- Set `GEMINI_API_KEY` (and any other Gemini config) via `.env` or `-e` when running the app/Docker container. Without it, the system falls back to returning raw context snippets.  
- Ingestion can be resource-intensive; ensure Poppler/Tesseract dependencies are installed and GEMINI quota is sufficient if you re-enable summarization.
- Mount `storage/` as a volume in Docker to persist vector indexes and assets across runs.  
- Logging files accumulate under `logs/`; use them to audit ingestion failures or media attachments.
