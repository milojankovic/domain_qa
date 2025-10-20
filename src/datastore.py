from __future__ import annotations

import json
import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .config import ASSET_DB_PATH, ASSET_DIR, STORAGE_DIR
from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Asset:
    """Serializable representation of a stored document asset."""

    asset_id: str
    doc_id: str
    type: str
    page: Optional[int]
    file_path: str
    text_json: Optional[dict] = None
    extra_json: Optional[dict] = None


class AssetStore:
    """Lightweight wrapper over SQLite for asset persistence."""

    def __init__(self, db_path: Path = ASSET_DB_PATH):
        self.db_path = self._prepare_db_path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.ensure_schema()

    @staticmethod
    def _prepare_db_path(db_path: Path) -> Path:
        """Create directories and migrate the legacy database if present."""

        new_path = Path(db_path)
        old_path = STORAGE_DIR / "assets.sqlite"
        try:
            new_path.parent.mkdir(parents=True, exist_ok=True)
            if (not new_path.exists()) and old_path.exists():
                shutil.move(str(old_path), str(new_path))
        except Exception:
            logger.exception("Failed to prepare asset DB path")
        return new_path

    def close(self) -> None:
        """Close the underlying SQLite connection."""

        self.conn.close()

    def ensure_schema(self) -> None:
        """Create required tables and indexes if they do not exist."""

        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
              doc_id TEXT PRIMARY KEY,
              source_path TEXT,
              title TEXT,
              metadata_json TEXT,
              created_at INTEGER
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS assets (
              asset_id TEXT PRIMARY KEY,
              doc_id TEXT,
              page INTEGER,
              type TEXT,
              file_path TEXT,
              text_json TEXT,
              extra_json TEXT,
              created_at INTEGER
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS failures (
              doc_id TEXT,
              source_path TEXT,
              reason TEXT,
              created_at INTEGER
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_assets_doc ON assets(doc_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_failures_doc ON failures(doc_id)")
        self.conn.commit()

    def record_document(
        self, doc_id: str, source_path: Path, title: Optional[str], metadata: dict
    ) -> None:
        """Insert or update document metadata."""

        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO documents(doc_id, source_path, title, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                str(source_path),
                title or "",
                json.dumps(metadata or {}),
                int(time.time()),
            ),
        )
        self.conn.commit()

    def save_asset(self, asset: Asset) -> None:
        """Persist an asset entry."""

        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO assets(asset_id, doc_id, page, type, file_path, text_json, extra_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                asset.asset_id,
                asset.doc_id,
                asset.page if asset.page is not None else None,
                asset.type,
                asset.file_path,
                json.dumps(asset.text_json or {}),
                json.dumps(asset.extra_json or {}),
                int(time.time()),
            ),
        )
        self.conn.commit()

    def record_failure(self, doc_id: str, source_path: Path, reason: str) -> None:
        """Track ingestion failures for later inspection."""

        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO failures(doc_id, source_path, reason, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (doc_id, str(source_path), reason[:1000], int(time.time())),
        )
        self.conn.commit()

    def get_assets_for_doc(self, doc_id: str) -> List[Asset]:
        """Return all assets associated with a document."""

        cur = self.conn.cursor()
        rows = cur.execute(
            "SELECT asset_id, doc_id, type, page, file_path, text_json, extra_json FROM assets WHERE doc_id=?",
            (doc_id,),
        ).fetchall()
        assets: List[Asset] = []
        for asset_id, doc, typ, page, file_path, text_json, extra_json in rows:
            assets.append(
                Asset(
                    asset_id=asset_id,
                    doc_id=doc,
                    type=typ,
                    page=page,
                    file_path=file_path,
                    text_json=json.loads(text_json) if text_json else None,
                    extra_json=json.loads(extra_json) if extra_json else None,
                )
            )
        return assets

    def asset_path(self, doc_id: str, page: Optional[int], name: str) -> Path:
        """Return the filesystem path where an asset should be stored."""

        base = ASSET_DIR / doc_id
        if page is not None:
            base = base / f"page-{page:04d}"
        base.mkdir(parents=True, exist_ok=True)
        return base / name
