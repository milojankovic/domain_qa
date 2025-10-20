from __future__ import annotations

import html
from typing import List, Optional

from google import genai
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse

from .config import GEMINI_API_KEY, GEMINI_GENERATE_MODEL, MAX_ASSET_ATTACH
from .datastore import AssetStore
from .logging_utils import get_logger
from .search import (
    build_where_filters,
    filter_results_by_metadata,
    rerank_results,
    retrieve,
)

logger = get_logger(__name__)

app = FastAPI(title="Domain QA")

_client: Optional[genai.Client] = None


def _get_client() -> Optional[genai.Client]:
    """Return a cached Gemini client if available."""

    global _client
    if not GEMINI_API_KEY:
        return None
    if _client is None:
        try:
            _client = genai.Client(api_key=GEMINI_API_KEY)
        except Exception as exc:
            logger.error(f"Failed to initialize Gemini client: {exc}")
            _client = None
    return _client


def render_home(
    results=None,
    answer: str = "",
    query: str = "",
    industries: str = "",
    countries: str = "",
    date_from: str = "",
    date_to: str = "",
) -> str:
    """Render the simple HTML interface for interactive queries."""

    rows = ""
    if results:
        for r in results:
            m = r["metadata"] or {}
            rows += f"<li><code>{html.escape(r['id'])}</code> [p={html.escape(str(m.get('page')))}] — {html.escape(r['text'][:240])}...</li>"
    body = f"""
    <html><head><title>Domain QA</title>
    <style>
      body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
      textarea {{ width: 100%; height: 100px; }}
      pre {{ background: #f7f7f7; padding: 1rem; white-space: pre-wrap; }}
      .row {{ display: flex; gap: 1rem; align-items: end; }}
      .row > div {{ flex: 1; }}
    </style></head>
    <body>
      <h2>Knowledge Base QA</h2>
      <form method='post' action='/ask'>
        <textarea name='q' placeholder='Ask a question...'>{html.escape(query)}</textarea>
        <br/>
        <div class='row'>
          <div>
            <label>Industries (comma separated)</label><br/>
            <input type='text' name='industries' style='width:100%' value='{html.escape(industries)}' />
          </div>
          <div>
            <label>Country codes (comma separated)</label><br/>
            <input type='text' name='countries' style='width:100%' value='{html.escape(countries)}' />
          </div>
        </div>
        <div class='row'>
          <div>
            <label>Date from (timestamp)</label><br/>
            <input type='text' name='date_from' style='width:100%' value='{html.escape(date_from)}' />
          </div>
          <div>
            <label>Date to (timestamp)</label><br/>
            <input type='text' name='date_to' style='width:100%' value='{html.escape(date_to)}' />
          </div>
        </div>
        <button type='submit'>Ask</button>
      </form>
      {f"<h3>Answer</h3><pre>{html.escape(answer)}</pre>" if answer else ""}
      {"<h3>Top Context</h3><ol>" + rows + "</ol>" if results else ""}
    </body></html>
    """
    return body


@app.get("/", response_class=HTMLResponse)
async def home():
    """Render the landing page for interactive use."""

    return HTMLResponse(render_home())


def build_prompt(contexts: List[dict], query: str) -> str:
    """Construct the textual prompt supplied to Gemini."""

    header = (
        "You are a careful assistant. Answer the user's question using ONLY the provided context snippets.\n"
        "If the answer cannot be found in the context, say you don't know.\n"
        "Cite snippet IDs in brackets like [ID] where applicable.\n\n"
    )
    ctx_lines = []
    for c in contexts:
        ctx_lines.append(f"[ID={c['id']}] {c['text']}")
    ctx_block = "\n".join(ctx_lines)
    return f"{header}Context:\n{ctx_block}\n\nQuestion: {query}\nAnswer:"


def attach_media_parts(
    client: genai.Client, contexts: List[dict], max_assets: int, store: AssetStore
):
    """Collect image and table assets to augment the Gemini prompt."""

    parts = []
    added = 0
    seen_docs = set()
    seen_tables = set()
    for c in contexts:
        m = c.get("metadata") or {}
        doc_id = m.get("doc_id")
        if not doc_id or doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)
        if added >= max_assets:
            break
        for a in store.get_assets_for_doc(doc_id):
            if a.type != "image":
                if a.type == "table":
                    if added >= max_assets:
                        break
                    table_text = ""
                    if a.text_json:
                        table_text = (a.text_json.get("text") or "").strip()
                    if not table_text or a.asset_id in seen_tables:
                        continue
                    logger.info(
                        "Attached table asset %s for doc %s", a.asset_id, doc_id
                    )
                    snippet = table_text[:1500]
                    parts.append({"text": f"[Table {a.asset_id}] {snippet}"})
                    seen_tables.add(a.asset_id)
                    added += 1
                continue
            try:
                if added >= max_assets:
                    break
                upload = client.files.upload(file=a.file_path)
                logger.info("Attached image asset %s for doc %s", a.asset_id, doc_id)
                parts.append(
                    {
                        "file_data": {
                            "file_uri": upload.uri,
                            "mime_type": getattr(upload, "mime_type", None),
                        }
                    }
                )
                added += 1
                if added >= max_assets:
                    break
            except Exception as e:
                logger.warning(f"Failed to upload asset {a.asset_id}: {e}")
        if added >= max_assets:
            break
    return parts


def generate_answer(contexts: List[dict], query: str) -> str:
    """Build a grounded prompt and ask Gemini for an answer."""

    client = _get_client()
    prompt = build_prompt(contexts, query)
    if not client:
        top_context = "\n\n".join(c.get("text", "") for c in contexts[:3])
        return f"(No Gemini API key configured; returning raw context)\n\n{top_context}".strip()

    message_parts = [{"text": prompt}]
    store = AssetStore()
    try:
        media_parts = attach_media_parts(client, contexts, MAX_ASSET_ATTACH, store)
        message_parts.extend(media_parts)
    except Exception as exc:
        logger.warning(f"Failed to attach media parts: {exc}")
    finally:
        try:
            store.close()
        except Exception:
            pass

    contents = [
        {
            "role": "user",
            "parts": message_parts,
        }
    ]
    try:
        resp = client.models.generate_content(
            model=GEMINI_GENERATE_MODEL,
            contents=contents,
        )
        return (resp.text or "").strip()
    except Exception as exc:
        logger.error(f"Gemini generate_content failed: {exc}")
        top_context = "\n\n".join(c.get("text", "") for c in contexts[:3])
        return f"(Gemini generation failed)\n\n{top_context}".strip()


@app.post("/ask", response_class=HTMLResponse)
async def ask_form(
    q: str = Form(...),
    industries: str = Form(""),
    countries: str = Form(""),
    date_from: str = Form(""),
    date_to: str = Form(""),
):
    """Handle HTML form submissions from the web UI."""

    inds = [s.strip() for s in industries.split(",") if s.strip()]
    ccs = [s.strip() for s in countries.split(",") if s.strip()]
    df = int(date_from) if date_from.strip().isdigit() else None
    dt = int(date_to) if date_to.strip().isdigit() else None
    where = build_where_filters(inds or None, ccs or None, df, dt)
    results = retrieve(q, where=where)
    results = filter_results_by_metadata(results, industries=inds, countries=ccs)
    results = rerank_results(q, results)
    answer = generate_answer(results, q)
    return HTMLResponse(
        render_home(
            results=results,
            answer=answer,
            query=q,
            industries=industries,
            countries=countries,
            date_from=date_from,
            date_to=date_to,
        )
    )


@app.post("/api/ask")
async def ask_api(req: Request):
    """JSON API endpoint for programmatic queries."""

    body = await req.json()
    q = (body.get("q") or "").strip()
    if not q:
        return JSONResponse({"error": "Empty query"}, status_code=400)
    inds = body.get("industries") or []
    ccs = body.get("country_codes") or []
    df = body.get("date_from")
    dt = body.get("date_to")
    try:
        df = int(df) if df is not None else None
    except Exception:
        df = None
    try:
        dt = int(dt) if dt is not None else None
    except Exception:
        dt = None
    where = build_where_filters(inds or None, ccs or None, df, dt)
    results = retrieve(q, where=where)
    results = filter_results_by_metadata(results, industries=inds, countries=ccs)
    results = rerank_results(q, results)
    answer = generate_answer(results, q)
    return {"answer": answer, "contexts": results}
