"""Chunks documents and indexes them into Qdrant with dense + sparse embeddings."""

import logging
import re
import uuid
from pathlib import Path
from typing import Any

from qdrant_client.models import PointStruct, SparseVector

from versionrag.clients.embedding import EmbeddingClient
from versionrag.clients.qdrant import QdrantClientWrapper
from versionrag.config import settings
from versionrag.models.schemas import DocumentMetadata
from versionrag.utils import read_file

logger = logging.getLogger(__name__)

_EMBED_BATCH_SIZE = 32
_HEADING_RE = re.compile(r"^#{1,6}\s")
_FENCE_RE = re.compile(r"^\s*```")


def _chunk_lossless(text: str, min_len: int, soft_max: int, hard_max: int) -> list[str]:
    """Lossless line-based chunker — never drops content.

    Guarantee::

        "\\n".join(chunks) == text   (modulo trailing newlines)

    Strategy:
    - Walk line-by-line, accumulating into a buffer.
    - Flush at a heading boundary once the buffer reaches ``soft_max``,
      unless we are inside a fenced code block (``` ``` ``` ``` ``` ```)
      in which case we postpone the split — preserves code-block integrity.
    - Force-flush before crossing ``hard_max`` regardless of position.
    - After the first pass, merge any chunk smaller than ``min_len`` into
      its predecessor so the index isn't full of tiny scraps.
    """
    lines = text.split("\n")
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_size = 0
    in_fence = False

    def flush() -> None:
        nonlocal buffer, buffer_size
        if buffer:
            chunks.append("\n".join(buffer))
            buffer = []
            buffer_size = 0

    for line in lines:
        if _FENCE_RE.match(line):
            in_fence = not in_fence

        added = len(line) + (1 if buffer else 0)

        if buffer and buffer_size + added > hard_max:
            flush()
            added = len(line)

        if (
            not in_fence
            and _HEADING_RE.match(line)
            and buffer_size >= soft_max
        ):
            flush()
            added = len(line)

        buffer.append(line)
        buffer_size += added
    flush()

    merged: list[str] = []
    for c in chunks:
        if merged and len(merged[-1]) < min_len:
            merged[-1] = merged[-1] + "\n" + c
        else:
            merged.append(c)
    return [c for c in merged if c.strip()]


def _chunk_markdown_lib(text: str, min_len: int, soft_max: int, hard_max: int) -> list[str]:
    """Original ``markdown_chunker`` library — block-aware but lossy.

    Library defaults silently drop content via fuzzy dedup + header/footer
    heuristics; on the large_dataset corpus this measured at 5–60 % loss
    per file. Kept here for A/B comparison via the ``CHUNKER`` setting.
    """
    from markdown_chunker import MarkdownChunkingStrategy  # local import

    strategy = MarkdownChunkingStrategy(
        min_chunk_len=min_len,
        soft_max_len=soft_max,
        hard_max_len=hard_max,
    )
    chunks = strategy.chunk_markdown(text)
    return [c.strip() for c in chunks if c.strip()]


class ContentIndexer:
    """Splits document text into chunks and indexes them into Qdrant.

    Each chunk is stored with:
    - A deterministic UUID so re-indexing is idempotent (upsert).
    - Dense vector (BGE-M3 dense embedding).
    - Sparse vector (BGE-M3 lexical weights for BM25-style retrieval).
    - Payload: version, document, category, chunk_index, file_path, doc_type.
    """

    def __init__(self, embedding: EmbeddingClient, qdrant: QdrantClientWrapper) -> None:
        self._embedding = embedding
        self._qdrant = qdrant

    def chunk_text(self, text: str) -> list[str]:
        """Split text into chunks. Strategy is selected by ``settings.chunker``.

        - ``lossless`` (default): custom line-based chunker, never drops
          content, fence-aware (preserves ``` code blocks at soft splits).
        - ``markdown``: original ``markdown_chunker`` library — block-aware
          but silently drops 5–60 % of content via dedup heuristics.
        """
        min_len = settings.chunk_min_len
        soft_max = settings.chunk_soft_max
        hard_max = settings.chunk_hard_max

        choice = settings.chunker.strip().lower()
        if choice == "lossless":
            return _chunk_lossless(text, min_len, soft_max, hard_max)
        if choice == "markdown":
            return _chunk_markdown_lib(text, min_len, soft_max, hard_max)
        raise ValueError(
            f"Unknown CHUNKER={settings.chunker!r}. Expected 'lossless' or 'markdown'."
        )

    @staticmethod
    def generate_chunk_id(document: str, version: str, chunk_index: int) -> str:
        """Generate a deterministic UUID for a chunk.

        The ID is derived from the document name, version, and index so
        repeated indexing of the same content produces the same ID (upsert).

        Args:
            document: Normalised document name.
            version: Version string.
            chunk_index: Zero-based chunk position.

        Returns:
            UUID string.
        """
        id_string = f"{document}_{version}_{chunk_index}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, id_string))

    async def index_document(self, metadata: DocumentMetadata, text: str) -> int:
        """Chunk, embed, and upsert a single document into Qdrant.

        Args:
            metadata: Document metadata (title, version, category_hint, etc.).
            text: Full document text.

        Returns:
            Number of chunks indexed.
        """
        logger.debug(
            "Chunking text for '%s' v%s (%d chars)",
            metadata.title, metadata.version, len(text),
        )
        chunks = self.chunk_text(text)
        if not chunks:
            logger.warning("No chunks generated for '%s' v%s", metadata.title, metadata.version)
            return 0

        avg_len = sum(len(c) for c in chunks) // len(chunks)
        logger.info(
            "Chunked '%s' v%s into %d chunks (avg %d chars)",
            metadata.title, metadata.version, len(chunks), avg_len,
        )

        doc_key = metadata.title.lower()
        cat_key = metadata.category_hint.lower()

        total_indexed = 0

        # Process in batches: embed + upsert immediately per batch
        for batch_start in range(0, len(chunks), _EMBED_BATCH_SIZE):
            batch = chunks[batch_start : batch_start + _EMBED_BATCH_SIZE]
            logger.debug(
                "Embedding batch %d-%d of %d chunks",
                batch_start, batch_start + len(batch), len(chunks),
            )
            embeddings = await self._embedding.encode(batch, purpose="indexing.chunk_embedding")
            use_sparse = self._embedding.supports_sparse()
            use_colbert = self._embedding.supports_colbert()
            sparse_list = embeddings.get("sparse", []) if use_sparse else []
            colbert_list = embeddings.get("colbert", []) if use_colbert else []

            points: list[PointStruct] = []
            for i, dense in enumerate(embeddings["dense"]):
                chunk_index = batch_start + i
                point_id = self.generate_chunk_id(doc_key, metadata.version, chunk_index)

                vector: dict = {"dense": dense.tolist() if hasattr(dense, "tolist") else dense}

                if use_sparse and i < len(sparse_list):
                    sd = sparse_list[i]
                    vector["sparse"] = SparseVector(
                        indices=[int(k) for k in sd.keys()],
                        values=[float(v) for v in sd.values()],
                    )

                if use_colbert and i < len(colbert_list):
                    cb = colbert_list[i]
                    vector["colbert"] = cb.tolist() if hasattr(cb, "tolist") else cb

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "version": metadata.version,
                            "document": doc_key,
                            "category": cat_key,
                            "chunk_index": chunk_index,
                            "content": batch[i],
                            "file_path": metadata.file_path,
                            "doc_type": metadata.doc_type.value,
                        },
                    )
                )

            await self._qdrant.upsert_points(settings.qdrant_collection, points)
            total_indexed += len(points)

        logger.info(
            "Indexed %d chunks for '%s' v%s into Qdrant",
            total_indexed, metadata.title, metadata.version,
        )
        return total_indexed

    async def index_all(self, documents: list[DocumentMetadata]) -> dict[str, Any]:
        """Index all documents in the list.

        Args:
            documents: List of metadata objects (must have ``file_path`` set).

        Returns:
            Dict with ``total_documents``, ``total_chunks``,
            ``per_document`` (title → chunk count).
        """
        total_chunks = 0
        per_document: dict[str, int] = {}

        for metadata in documents:
            try:
                text = read_file(Path(metadata.file_path))
                count = await self.index_document(metadata, text)
                total_chunks += count
                per_document[metadata.title] = per_document.get(metadata.title, 0) + count
            except Exception as exc:
                logger.error(
                    "Failed to index '%s' v%s: %s", metadata.title, metadata.version, exc
                )

        logger.info(
            "Indexed %d documents, %d total chunks", len(documents), total_chunks
        )
        return {
            "total_documents": len(documents),
            "total_chunks": total_chunks,
            "per_document": per_document,
        }
