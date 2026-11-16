"""Vector-based retriever using Qdrant dense or hybrid (dense+sparse) search."""

import logging

from qdrant_client.http.models import FusionQuery
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    MatchAny,
    MatchValue,
    Prefetch,
    SparseVector,
)

from versionrag.clients.embedding import EmbeddingClient
from versionrag.clients.qdrant import QdrantClientWrapper
from versionrag.config import settings
from versionrag.models.enums import QueryType, RetrievalMode
from versionrag.models.schemas import Chunk, ParsedQuery, RetrievalResult

logger = logging.getLogger(__name__)


class VectorRetriever:
    """Retrieves document chunks from Qdrant using dense or hybrid search."""

    def __init__(self, embedding: EmbeddingClient, qdrant: QdrantClientWrapper) -> None:
        self._embedding = embedding
        self._qdrant = qdrant

    def _build_filter(self, parsed: ParsedQuery) -> Filter | None:
        """Build Qdrant payload filter from parsed query fields.

        ``document`` is filtered with MatchValue (exact match) because payload
        values are canonical slugs like "golang_sync" — MatchText would
        token-split and over-match ("go" → "golang_*").

        For CONTENT_VERSION_SPECIFIC, we always emit a version filter when a
        version was parsed. If none was parsed, no version filter is applied —
        the result may be wrong; the parser is responsible for extracting the
        version when one is mentioned in the question.
        """
        filters: list[FieldCondition] = []

        if parsed.query_type == QueryType.CONTENT_VERSION_SPECIFIC and parsed.version:
            filters.append(
                FieldCondition(key="version", match=MatchValue(value=parsed.version))
            )
        # Document filter: prefer the multi-doc list when QueryParser expanded
        # a prefix (legal corpus). Falls back to single MatchValue otherwise —
        # which is the only branch software queries ever hit.
        if parsed.documents:
            filters.append(
                FieldCondition(key="document", match=MatchAny(any=parsed.documents))
            )
        elif parsed.document:
            filters.append(
                FieldCondition(key="document", match=MatchValue(value=parsed.document))
            )
        if parsed.category:
            filters.append(
                FieldCondition(key="category", match=MatchValue(value=parsed.category))
            )

        return Filter(must=filters) if filters else None

    async def retrieve_content(
        self, parsed: ParsedQuery, top_k: int = 15
    ) -> RetrievalResult:
        """Handle CONTENT_RETRIEVAL and CONTENT_VERSION_SPECIFIC queries.

        Uses hybrid search (dense+sparse with RRF) when the embedding client
        supports sparse vectors, otherwise falls back to dense-only search.
        """
        logger.debug("Vector retrieval: question=%s, top_k=%d", parsed.question, top_k)

        query_vectors = await self._embedding.encode_query(parsed.question)
        dense_vec = query_vectors["dense"]
        qdrant_filter = self._build_filter(parsed)

        has_sparse = self._embedding.supports_sparse()
        has_colbert = self._embedding.supports_colbert()

        if has_sparse or has_colbert:
            results = await self._hybrid_search(dense_vec, query_vectors, qdrant_filter, top_k)
        else:
            results = await self._dense_search(dense_vec, qdrant_filter, top_k)

        chunks: list[Chunk] = []
        for point in results.points:
            payload = point.payload or {}
            chunks.append(
                Chunk(
                    content=payload.get("content", ""),
                    document=payload.get("document", ""),
                    version=payload.get("version", ""),
                    category=payload.get("category", ""),
                    chunk_index=payload.get("chunk_index"),
                    score=point.score,
                )
            )

        retrieval_mode = RetrievalMode.HYBRID if qdrant_filter else RetrievalMode.VECTOR

        logger.info(
            "Retrieved %d chunks (mode=%s, sparse=%s)",
            len(chunks), retrieval_mode, self._embedding.supports_sparse(),
        )

        return RetrievalResult(
            chunks=chunks, changes=[], versions=[], retrieval_mode=retrieval_mode,
        )

    async def _dense_search(self, dense_vec, qdrant_filter, top_k):
        """Dense-only vector search."""
        return await self._qdrant.client.query_points(
            collection_name=settings.qdrant_collection,
            query=dense_vec,
            using="dense",
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )

    async def _hybrid_search(self, dense_vec, query_vectors, qdrant_filter, top_k):
        """Hybrid search with Reciprocal Rank Fusion over available vector types."""
        dense_list = dense_vec.tolist() if hasattr(dense_vec, "tolist") else dense_vec
        prefetch_limit = top_k * 4

        prefetch = [
            Prefetch(query=dense_list, using="dense", limit=prefetch_limit),
        ]

        if "sparse" in query_vectors:
            sparse_weights = query_vectors["sparse"]
            sparse_indices = [int(k) for k in sparse_weights.keys()]
            sparse_values = [float(v) for v in sparse_weights.values()]
            sparse_vec = SparseVector(indices=sparse_indices, values=sparse_values)
            prefetch.append(Prefetch(query=sparse_vec, using="sparse", limit=prefetch_limit))

        if "colbert" in query_vectors:
            colbert_vec = query_vectors["colbert"]
            colbert_list = colbert_vec.tolist() if hasattr(colbert_vec, "tolist") else colbert_vec
            prefetch.append(Prefetch(query=colbert_list, using="colbert", limit=prefetch_limit))

        return await self._qdrant.rest_client.query_points(
            collection_name=settings.qdrant_collection,
            prefetch=prefetch,
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )
