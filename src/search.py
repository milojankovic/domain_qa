from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .config import TOP_K
from .vectorstore import VectorStore


def retrieve(
    query: str,
    vs: Optional[VectorStore] = None,
    where: Optional[Dict[str, Any]] = None,
    top_k: int = TOP_K,
):
    """Fetch top-k candidates from the vector store and normalize the result."""

    vector_store = vs or VectorStore()
    res = vector_store.query(query_text=query, top_k=top_k, where=where)
    out = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for i in range(len(ids)):
        out.append(
            {
                "id": ids[i],
                "text": docs[i],
                "metadata": metas[i],
                "distance": dists[i] if i < len(dists) else None,
            }
        )
    return out


def _tokenize(text: str) -> List[str]:
    """Tokenize text using a lightweight regex split."""

    return [token for token in re.split(r"\W+", (text or "").lower()) if token]


def rerank_results(query: str, results: List[dict], alpha: float = 0.7) -> List[dict]:
    """Combine vector similarity with token overlap for simple re-ranking."""

    q_tokens = set(_tokenize(query))
    ranked = []
    for result in results:
        distance = result.get("distance")
        vec_sim = 1.0 - float(distance) if distance is not None else 0.0
        t_tokens = set(_tokenize(result.get("text", "")))
        inter = len(q_tokens & t_tokens)
        union = len(q_tokens | t_tokens) or 1
        jaccard = inter / union
        score = alpha * vec_sim + (1 - alpha) * jaccard
        enriched = dict(result)
        enriched["_score"] = score
        ranked.append(enriched)
    ranked.sort(key=lambda item: item.get("_score", 0.0), reverse=True)
    return ranked


def build_where_filters(
    industries: Optional[List[str]] = None,
    countries: Optional[List[str]] = None,
    date_from: Optional[int] = None,
    date_to: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Construct Chroma-compatible range filters for metadata queries."""

    clauses: List[Dict[str, Any]] = []
    if date_from is not None or date_to is not None:
        range_clause: Dict[str, Any] = {}
        if date_from is not None:
            range_clause["$gte"] = int(date_from)
        if date_to is not None:
            range_clause["$lte"] = int(date_to)
        if range_clause:
            clauses.append({"date_ts": range_clause})
    if not clauses:
        return None
    return {"$and": clauses} if len(clauses) > 1 else clauses[0]


def _value_matches(meta_val, targets: List[str]) -> bool:
    """Check whether metadata contains any of the target values."""

    if not meta_val:
        return False
    if isinstance(meta_val, (list, tuple, set)):
        haystack = [str(v).lower() for v in meta_val]
        return any(target in haystack for target in targets)
    text = str(meta_val).lower()
    return any(target in text for target in targets)


def filter_results_by_metadata(
    results: List[dict],
    industries: Optional[List[str]] = None,
    countries: Optional[List[str]] = None,
) -> List[dict]:
    """Apply in-memory filtering for industries and country codes."""

    if not industries and not countries:
        return results
    norm_inds = [value.strip().lower() for value in (industries or []) if value.strip()]
    norm_countries = [
        value.strip().lower() for value in (countries or []) if value.strip()
    ]
    filtered: List[dict] = []
    for result in results:
        meta = result.get("metadata") or {}
        if norm_inds and not _value_matches(meta.get("industries"), norm_inds):
            continue
        if norm_countries and not _value_matches(
            meta.get("country_codes"), norm_countries
        ):
            continue
        filtered.append(result)
    return filtered
