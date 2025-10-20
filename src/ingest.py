from __future__ import annotations

import json
import re
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from unstructured.documents.elements import Element, Table
from unstructured.partition.pdf import partition_pdf

from src.config import (
    DATA_DIRS,
    HI_RES_STRATEGY,
    METADATA_JSONL,
    OCR_LANGUAGES,
    USE_OCR,
)
from src.chunking import Block, chunk_elements
from src.datastore import Asset, AssetStore
from src.logging_utils import get_logger
from src.vectorstore import VectorStore

logger = get_logger(__name__)

_ALLOWED_META_TYPES = (str, int, float, bool)
MIN_IMAGE_PIXELS = 100 * 100
MIN_IMAGE_BYTES = 4 * 1024
MAX_ASPECT_RATIO = 6.0

try:
    from PIL import Image as PILImage  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    PILImage = None  # type: ignore


def _should_keep_image(path: Path) -> bool:
    """Return True if an extracted image appears meaningful enough to retain."""

    try:
        if path.stat().st_size < MIN_IMAGE_BYTES:
            return False
    except OSError:
        return False
    if PILImage:
        try:
            with PILImage.open(path) as im:
                width, height = im.size
        except Exception:
            return True
        if not width or not height:
            return False
        if (width * height) < MIN_IMAGE_PIXELS:
            return False
        aspect_ratio = max(width / height, height / width)
        if aspect_ratio > MAX_ASPECT_RATIO:
            return False
    return True


def _sanitize_metadata_value(value):
    """Convert metadata values into Chroma-compatible scalars."""

    if value is None or isinstance(value, _ALLOWED_META_TYPES):
        return value
    if isinstance(value, (list, tuple, set)):
        flattened = []
        for item in value:
            cleaned = _sanitize_metadata_value(item)
            if cleaned is None:
                continue
            flattened.append(str(cleaned))
        return ", ".join(flattened) if flattened else None
    if isinstance(value, dict):
        try:
            return json.dumps(value)
        except TypeError:
            safe_dict = {str(k): _sanitize_metadata_value(v) for k, v in value.items()}
            return json.dumps(safe_dict)
    return str(value)


def _sanitize_metadata(metadata: dict) -> dict:
    """Normalize metadata values for storage."""

    return {k: _sanitize_metadata_value(v) for k, v in metadata.items()}


def load_metadata(path: Path) -> Dict[str, dict]:
    """Load document metadata from a JSONL file keyed by UUID."""

    meta: Dict[str, dict] = {}
    if not path.exists():
        logger.warning(f"Metadata file not found at {path}")
        return meta
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                uuid_str = obj.get("uuid")
                if uuid_str:
                    meta[uuid_str] = obj
            except Exception as e:
                logger.warning(f"Failed to parse metadata line: {e}")
    return meta


def guess_doc_id(path: Path) -> str:
    """Return a stable document identifier derived from the file path."""

    stem = path.stem
    if re.fullmatch(r"[0-9a-fA-F-]{32,36}", stem):
        return stem.lower()
    return uuid.uuid5(uuid.NAMESPACE_URL, str(path.resolve()).replace("\\", "/")).hex


def element_page(e: Element) -> Optional[int]:
    """Best-effort extraction of the page number for an element."""

    try:
        pg = getattr(getattr(e, "metadata", None), "page_number", None)
        if pg is None:
            pg = getattr(getattr(e, "metadata", None), "page", None)
        return int(pg) if pg is not None else None
    except Exception:
        return None


def element_image_path(e: Element) -> Optional[Path]:
    """Return the filesystem path embedded in an element's metadata, if any."""

    try:
        p = getattr(getattr(e, "metadata", None), "image_path", None)
        return Path(p) if p else None
    except Exception:
        return None


def save_image_asset_from_element(
    e: Element, store: AssetStore, doc_id: str, pg: Optional[int]
) -> Optional[Asset]:
    """Persist an image element to disk and register it in the asset store."""

    img_path = element_image_path(e)
    if img_path and img_path.exists():
        if not _should_keep_image(img_path):
            return None
        target = store.asset_path(doc_id, pg, img_path.name)
        try:
            target.write_bytes(img_path.read_bytes())
            return Asset(
                asset_id=f"{doc_id}::img::{uuid.uuid4().hex}",
                doc_id=doc_id,
                type="image",
                page=pg,
                file_path=str(target),
                text_json=None,
                extra_json={"category": getattr(e, "category", "image")},
            )
        except Exception:
            pass
    try:
        im = getattr(e, "image", None)
        if im is not None:
            target = store.asset_path(doc_id, pg, f"image-{uuid.uuid4().hex}.png")
            try:
                from PIL import Image

                if isinstance(im, Image.Image):
                    width, height = im.size
                    if width * height < MIN_IMAGE_PIXELS:
                        return None
                    aspect_ratio = (
                        max(width / height, height / width) if width and height else 0
                    )
                    if aspect_ratio > MAX_ASPECT_RATIO:
                        return None
                    im.save(str(target))
                elif isinstance(im, (bytes, bytearray)):
                    Path(target).write_bytes(im)
                else:
                    if hasattr(im, "save"):
                        im.save(str(target))
                    else:
                        raise ValueError("Unsupported image object")
                if not _should_keep_image(target):
                    target.unlink(missing_ok=True)
                    return None
                return Asset(
                    asset_id=f"{doc_id}::img::{uuid.uuid4().hex}",
                    doc_id=doc_id,
                    type="image",
                    page=pg,
                    file_path=str(target),
                    text_json=None,
                    extra_json={"category": getattr(e, "category", "image")},
                )
            except Exception:
                pass
    except Exception:
        pass
    try:
        md = getattr(e, "metadata", None)
        b64 = getattr(md, "image_base64", None) or (
            getattr(md, "to_dict", lambda: {})().get("image_base64")
        )
        if b64:
            import base64

            raw = base64.b64decode(b64)
            target = store.asset_path(doc_id, pg, f"image-{uuid.uuid4().hex}.png")
            Path(target).write_bytes(raw)
            if not _should_keep_image(target):
                target.unlink(missing_ok=True)
                return None
            return Asset(
                asset_id=f"{doc_id}::img::{uuid.uuid4().hex}",
                doc_id=doc_id,
                type="image",
                page=pg,
                file_path=str(target),
                text_json=None,
                extra_json={"category": getattr(e, "category", "image")},
            )
    except Exception:
        pass
    return None


def walk_pdfs(data_dirs: List[Path]) -> List[Path]:
    """Collect all PDF paths beneath the configured data directories."""

    pdfs: List[Path] = []
    for base in data_dirs:
        if not base.exists():
            continue
        for p in base.rglob("*.pdf"):
            pdfs.append(p)
    return sorted(pdfs)


def process_pdf(
    path: Path, meta_map: Dict[str, dict], vs: VectorStore, store: AssetStore
):
    """Process a single PDF into indexed chunks and persisted assets."""

    doc_id = guess_doc_id(path)
    meta = meta_map.get(doc_id, {})
    logger.info(f"Processing {path.name} as doc_id={doc_id}")

    try:
        elements: List[Element] = partition_pdf(
            filename=str(path),
            strategy=HI_RES_STRATEGY,
            extract_images_in_pdf=True,
            infer_table_structure=True,
            ocr_languages=OCR_LANGUAGES if USE_OCR else None,
        )
    except Exception as e:
        logger.error(f"partition_pdf failed for {path}: {e}")
        store.record_failure(doc_id, path, f"partition_pdf error: {e}")
        return

    store.record_document(doc_id, path, title=None, metadata=meta)

    chunk_texts: List[str] = []
    chunk_ids: List[str] = []
    chunk_metas: List[dict] = []

    n_img = 0
    n_tbl = 0
    table_assets_by_element: Dict[int, str] = {}
    image_assets_by_element: Dict[int, str] = {}
    page_table_assets: Dict[int, List[str]] = defaultdict(list)
    page_image_assets: Dict[int, List[str]] = defaultdict(list)
    for e in elements:
        pg = element_page(e)
        cat = getattr(e, "category", "").lower()
        page_key = pg if pg is not None else -1

        if isinstance(e, Table):
            table_text = e.text or ""
            asset_id = f"{doc_id}::tbl::{uuid.uuid4().hex}"
            asset = Asset(
                asset_id=asset_id,
                doc_id=doc_id,
                type="table",
                page=pg,
                file_path="",
                text_json={"text": table_text},
                extra_json={"category": cat},
            )
            store.save_asset(asset)
            table_assets_by_element[id(e)] = asset_id
            page_table_assets[page_key].append(asset_id)
            n_tbl += 1
        elif cat in {"image", "figure"}:
            asset = save_image_asset_from_element(e, store, doc_id, pg)
            if asset:
                store.save_asset(asset)
                image_assets_by_element[id(e)] = asset.asset_id
                page_image_assets[page_key].append(asset.asset_id)
                n_img += 1

    chunks = chunk_elements(elements, pdf_path=str(path))
    if not chunks:
        logger.warning(
            f"No layout chunks for {doc_id}; falling back to naive text grouping"
        )
        page_to_text: Dict[int, List[str]] = {}
        for e in elements:
            cat = getattr(e, "category", "").lower()
            if isinstance(e, Table) or cat in {"image", "figure"}:
                continue
            t = getattr(e, "text", "") or ""
            if not t.strip():
                continue
            pg = element_page(e) or 0
            page_to_text.setdefault(pg, []).append(t)
        for pg, parts in sorted(page_to_text.items()):
            txt = " ".join(parts).strip()
            if txt:
                chunks.append(
                    Block(
                        text=txt,
                        category="text",
                        metadata=type("M", (), {"page_number": pg})(),
                    )
                )

    for ch in chunks:
        if isinstance(ch, Block):
            pg = ch.metadata.page_number
            cat = ch.category.lower()
            text = ch.text
        else:
            pg = element_page(ch)
            cat = getattr(ch, "category", "").lower()
            text = getattr(ch, "text", "") or ""
        ch_id = f"{doc_id}::pg{pg or 0}::{uuid.uuid4().hex}"
        page_key = pg if pg is not None else -1
        asset_id: Optional[str] = None

        if isinstance(ch, Table) or cat == "table":
            raw = (text or "").strip()
            if not raw:
                continue
            document_text = raw
            asset_id = table_assets_by_element.get(id(ch))
        elif cat == "image":
            raw = (text or "").strip()
            if raw:
                document_text = raw
            else:
                document_text = f"Image placeholder for doc {doc_id} page {pg or 0}"
            if not isinstance(ch, Block):
                asset_id = image_assets_by_element.get(id(ch))
        else:
            raw = (text or "").strip()
            if not raw:
                continue
            document_text = raw

        metadata = {
            "doc_id": doc_id,
            "source_path": str(path),
            "page": pg,
            "category": cat
            or (
                getattr(type(ch), "__name__", "element").lower()
                if not isinstance(ch, Block)
                else "text"
            ),
        }
        if asset_id:
            metadata["asset_id"] = asset_id
        if page_table_assets.get(page_key):
            metadata["table_asset_ids"] = ",".join(page_table_assets[page_key])
        if page_image_assets.get(page_key):
            metadata["image_asset_ids"] = ",".join(page_image_assets[page_key])
        for k in ("industries", "date", "country_codes"):
            if k in meta:
                metadata[k] = meta[k]
        if "date" in meta and "date_ts" not in metadata:
            try:
                metadata["date_ts"] = int(str(meta["date"]))
            except Exception:
                pass
        metadata = _sanitize_metadata(metadata)

        chunk_texts.append(document_text)
        chunk_ids.append(ch_id)
        chunk_metas.append(metadata)

    if chunk_texts:
        try:
            vs.upsert(ids=chunk_ids, documents=chunk_texts, metadatas=chunk_metas)
            logger.info(
                f"Indexed {len(chunk_texts)} chunks for {doc_id}; saved {n_tbl} tables, {n_img} images"
            )
        except Exception as e:
            logger.error(f"Vector upsert failed for {doc_id}: {e}")
    else:
        logger.warning(f"No chunks to index for {doc_id}")


def main():
    """Run ingestion over all discovered PDFs."""

    meta_map = load_metadata(METADATA_JSONL)
    paths = walk_pdfs(DATA_DIRS)
    vs = VectorStore()
    store = AssetStore()
    logger.info(f"Found {len(paths)} PDFs to process")
    for p in paths:
        try:
            process_pdf(p, meta_map, vs, store)
        except Exception as e:
            doc_id = guess_doc_id(p)
            logger.exception(f"Unexpected failure for {p}: {e}")
            store.record_failure(doc_id, p, f"unexpected: {e}")


if __name__ == "__main__":
    main()
