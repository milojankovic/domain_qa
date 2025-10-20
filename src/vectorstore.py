from __future__ import annotations

from typing import Dict, List, Optional

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings, IDs, Metadatas
from sentence_transformers import SentenceTransformer

from .config import CHROMA_DIR, SENTENCE_TRANSFORMER_DEVICE, SENTENCE_TRANSFORMER_MODEL
from .logging_utils import get_logger

logger = get_logger(__name__)


class SentenceTransformerEmbeddingFunction(EmbeddingFunction[str]):
    """Embedding function that delegates to a sentence-transformer model."""

    def __init__(
        self,
        model_name: str = SENTENCE_TRANSFORMER_MODEL,
        device: Optional[str] = SENTENCE_TRANSFORMER_DEVICE,
    ):
        self.model_name = model_name
        self.device = device
        logger.info(
            "Loading sentence-transformer model '%s' (device=%s)",
            model_name,
            device or "auto",
        )
        self.model = SentenceTransformer(model_name, device=device, cache_folder=None)
        if hasattr(self.model, "get_sentence_embedding_dimension"):
            logger.info(
                "Loaded %s; embedding dimension=%d",
                model_name,
                self.model.get_sentence_embedding_dimension(),
            )

    def __call__(self, input: Documents) -> Embeddings:
        """Encode documents into normalized embedding vectors."""

        texts: List[str] = [doc if isinstance(doc, str) else "" for doc in input]
        if not texts:
            return []
        logger.debug("Encoding %d document(s) with %s", len(texts), self.model_name)
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()


class VectorStore:
    """High-level helper for upserting and querying Chroma collections."""

    def __init__(self, collection: str = "kb_chunks"):
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.ef = SentenceTransformerEmbeddingFunction()
        self.c = self.client.get_or_create_collection(
            name=collection,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self, ids: IDs, documents: Documents, metadatas: Optional[Metadatas] = None
    ) -> None:
        """Insert or update vectors, documents, and metadata."""

        self.c.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def query(
        self, query_text: str, top_k: int = 6, where: Optional[Dict[str, object]] = None
    ):
        """Retrieve the top-k nearest chunks for a given query text."""

        return self.c.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where,
        )
