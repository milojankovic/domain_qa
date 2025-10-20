from dataclasses import dataclass
import re
from typing import List, Optional, Tuple

from unstructured.documents.elements import Element, Table

from .config import MAX_CHARS_NARRATIVE


@dataclass
class BlockMeta:
    """Positional metadata captured from an unstructured Element."""

    page_number: Optional[int] = None
    y0: Optional[float] = None
    y1: Optional[float] = None
    x0: Optional[float] = None
    x1: Optional[float] = None
    font_size: Optional[float] = None


@dataclass
class Block:
    """Chunk of text or table/image content plus metadata."""

    text: str
    category: str
    metadata: BlockMeta


BOUNDARY_CATS = {"table", "image", "figure", "pagebreak"}
TITLE_CATS = {"title", "subtitle", "header"}


def _get_page(e: Element) -> Optional[int]:
    """Extract the page number from element metadata."""

    try:
        md = getattr(e, "metadata", None)
        pg = getattr(md, "page_number", None) or getattr(md, "page", None)
        return int(pg) if pg is not None else None
    except Exception:
        return None


def _get_coords(
    e: Element,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Return element bounding box coordinates if available."""

    try:
        md = getattr(e, "metadata", None)
        coords = getattr(md, "coordinates", None)
        if coords is None:
            d = getattr(md, "to_dict", lambda: {})()
            coords = d.get("coordinates")
        pts = getattr(coords, "points", None)
        if pts is None and isinstance(coords, dict):
            pts = coords.get("points")
        if not pts:
            return None, None, None, None
        xs = []
        ys = []
        for p in pts:
            if isinstance(p, dict):
                xs.append(float(p.get("x")))
                ys.append(float(p.get("y")))
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                xs.append(float(p[0]))
                ys.append(float(p[1]))
        if not xs or not ys:
            return None, None, None, None
        return min(xs), min(ys), max(xs), max(ys)
    except Exception:
        return None, None, None, None


def _get_font_size(e: Element) -> Optional[float]:
    """Return font size if provided in the element metadata."""

    try:
        md = getattr(e, "metadata", None)
        d = getattr(md, "to_dict", lambda: {})()
        fs = d.get("font_size")
        return float(fs) if fs is not None else None
    except Exception:
        return None


def _sentences(text: str) -> List[str]:
    """Split text into sentences using a lightweight heuristic."""

    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", text)
    return [p.strip() for p in parts if p.strip()]


def _split_by_length(text: str, max_chars: int) -> List[str]:
    """Split text into segments adhering to a maximum character count."""

    if len(text) <= max_chars:
        return [text]
    sents = _sentences(text)
    if not sents:
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
    out: List[str] = []
    cur = ""
    for s in sents:
        if not cur:
            cur = s
        elif len(cur) + 1 + len(s) <= max_chars:
            cur += " " + s
        else:
            out.append(cur)
            cur = s
    if cur:
        out.append(cur)
    return out


def _build_blocks_from_elements(
    elements: List[Element], y_gap_threshold: float = 30.0
) -> List[Block]:
    """Coalesce elements into ordered blocks based on spatial cues."""

    blocks: List[Block] = []
    sortable = []
    for e in elements:
        page = _get_page(e)
        x0, y0, x1, y1 = _get_coords(e)
        sortable.append(
            (
                page or 0,
                y0 if y0 is not None else 0.0,
                x0 if x0 is not None else 0.0,
                e,
                x0,
                y0,
                x1,
                y1,
            )
        )
    sortable.sort(key=lambda t: (t[0], t[1], t[2]))

    cur_text = []
    cur_meta = BlockMeta()
    prev_page: Optional[int] = None
    prev_y1: Optional[float] = None
    prev_font: Optional[float] = None

    def flush_cur():
        nonlocal cur_text, cur_meta
        if cur_text:
            txt = " ".join([t for t in cur_text if t]).strip()
            if txt:
                blocks.append(Block(text=txt, category="text", metadata=cur_meta))
        cur_text = []
        cur_meta = BlockMeta()

    for page, y0, x0, e, ex0, ey0, ex1, ey1 in sortable:
        cat = getattr(e, "category", "").lower()
        text = getattr(e, "text", "") or ""
        fs = _get_font_size(e)

        if cat in BOUNDARY_CATS or isinstance(e, Table):
            flush_cur()
            bmeta = BlockMeta(
                page_number=page, x0=ex0, y0=ey0, x1=ex1, y1=ey1, font_size=fs
            )
            bcat = "table" if isinstance(e, Table) else (cat or "element")
            btext = text if bcat == "table" else (text or bcat)
            blocks.append(Block(text=btext, category=bcat, metadata=bmeta))
            prev_page, prev_y1, prev_font = page, ey1, fs
            continue

        start_new = False
        if prev_page is None or page != prev_page:
            start_new = True
        if cat in TITLE_CATS:
            start_new = True
        if prev_font is not None and fs is not None:
            try:
                if fs - prev_font >= 2.0 or (
                    prev_font and fs / max(prev_font, 1e-6) >= 1.2
                ):
                    start_new = True
            except Exception:
                pass
        if (
            prev_y1 is not None
            and ey0 is not None
            and (ey0 - prev_y1) > y_gap_threshold
        ):
            start_new = True

        if start_new:
            flush_cur()
            cur_meta = BlockMeta(
                page_number=page, x0=ex0, y0=ey0, x1=ex1, y1=ey1, font_size=fs
            )

        cur_text.append(text)
        prev_page = page
        prev_y1 = ey1 if ey1 is not None else prev_y1
        prev_font = fs if fs is not None else prev_font

    flush_cur()
    return blocks


def _fallback_blocks_pymupdf(pdf_path: Optional[str]) -> List[Block]:
    """Fallback chunking using PyMuPDF when unstructured layout data is missing."""

    if not pdf_path:
        return []
    try:
        import fitz  # PyMuPDF
    except Exception:
        return []
    blocks: List[Block] = []
    try:
        doc = fitz.open(pdf_path)
        for pno in range(len(doc)):
            page = doc[pno]
            for b in page.get_text("blocks"):
                if len(b) < 5:
                    continue
                x0, y0, x1, y1, txt = b[0], b[1], b[2], b[3], b[4]
                if not isinstance(txt, str) or not txt.strip():
                    continue
                meta = BlockMeta(
                    page_number=pno + 1,
                    x0=float(x0),
                    y0=float(y0),
                    x1=float(x1),
                    y1=float(y1),
                )
                blocks.append(Block(text=txt.strip(), category="text", metadata=meta))
    except Exception:
        pass
    return blocks


def chunk_elements(
    elements: List[Element], pdf_path: Optional[str] = None
) -> List[Block]:
    """Return layout-aware chunks using visual cues with a PyMuPDF fallback."""

    if not elements:
        return []

    blocks = _build_blocks_from_elements(elements)
    if not blocks:
        blocks = _fallback_blocks_pymupdf(pdf_path)

    out: List[Block] = []
    max_chars = MAX_CHARS_NARRATIVE
    for b in blocks:
        if b.category in BOUNDARY_CATS or b.category == "table":
            if b.category == "table" and b.text.strip():
                for t in _split_by_length(
                    b.text.strip(), max_chars=max(800, int(0.6 * max_chars))
                ):
                    out.append(Block(text=t, category="table", metadata=b.metadata))
            continue
        txt = b.text.strip()
        if not txt:
            continue
        parts = _split_by_length(txt, max_chars=max_chars)
        for p in parts:
            out.append(Block(text=p, category="text", metadata=b.metadata))
    return out
