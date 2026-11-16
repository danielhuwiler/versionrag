"""Qdrant client wrapper with collection management helpers."""

import logging

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.models import PointStruct

logger = logging.getLogger(__name__)


class QdrantClientWrapper:
    """Wraps the async Qdrant client with lifecycle and collection helpers."""

    def __init__(self, host: str, port: int, grpc_port: int) -> None:
        self._host = host
        self._port = port
        self._grpc_port = grpc_port
        self._client: AsyncQdrantClient | None = None

    @property
    def client(self) -> AsyncQdrantClient:
        """Return the underlying async Qdrant client for direct operations.

        Raises:
            RuntimeError: If the client is not connected.
        """
        if not self._client:
            raise RuntimeError("Qdrant client is not connected. Call connect() first.")
        return self._client

    async def connect(self) -> None:
        """Instantiate the async Qdrant client."""
        self._client = AsyncQdrantClient(
            host=self._host,
            port=self._port,
            grpc_port=self._grpc_port,
            prefer_grpc=True,
        )
        # REST-only client for Fusion queries (gRPC can't serialize Fusion.RRF)
        self._rest_client = AsyncQdrantClient(
            host=self._host,
            port=self._port,
            prefer_grpc=False,
        )
        logger.info("Qdrant client connected: %s:%s", self._host, self._port)

    @property
    def rest_client(self) -> AsyncQdrantClient:
        """Return the REST-only client for Fusion queries."""
        if not self._rest_client:
            raise RuntimeError("Qdrant client is not connected.")
        return self._rest_client

    async def close(self) -> None:
        """Close the underlying HTTP/gRPC connections."""
        if self._client:
            await self._client.close()
        if self._rest_client:
            await self._rest_client.close()
            logger.info("Qdrant client closed")

    async def verify_connectivity(self) -> bool:
        """Check whether the Qdrant instance is reachable.

        Returns:
            True if reachable, False otherwise.
        """
        if not self._client:
            return False
        try:
            await self._client.get_collections()
            return True
        except Exception as exc:
            logger.error("Qdrant connectivity check failed: %s", exc)
            return False

    async def ensure_collection(
        self,
        collection_name: str,
        dimension: int,
        sparse: bool = False,
        colbert: bool = False,
        colbert_dimension: int | None = None,
    ) -> None:
        """Create the collection if it does not already exist.

        If the collection exists, its configured dense vector size is compared
        against ``dimension``. A WARNING is logged if they differ (e.g. because
        the embedding model was changed without recreating the collection).

        Configures:
        - Dense vector (Cosine distance, size=dimension)
        - Sparse vector named ``"sparse"`` for BGE-M3 lexical retrieval
        - Payload indexes on ``version``, ``document``, ``category``

        Args:
            collection_name: Name of the Qdrant collection.
            dimension: Dense vector dimensionality from the loaded embedding model.

        Raises:
            RuntimeError: If the client is not connected.
        """
        if not self._client:
            raise RuntimeError("Qdrant client is not connected. Call connect() first.")

        existing = await self._client.get_collections()
        existing_names = {c.name for c in existing.collections}

        if collection_name in existing_names:
            # Verify that the stored dimension matches the current model
            info = await self._client.get_collection(collection_name)
            vectors_config = info.config.params.vectors
            if isinstance(vectors_config, dict) and "dense" in vectors_config:
                stored_size = vectors_config["dense"].size
                if stored_size != dimension:
                    logger.warning(
                        "Qdrant collection '%s' has dense vector size %d but current "
                        "embedding model produces dimension %d — consider recreating "
                        "the collection after re-indexing all documents.",
                        collection_name,
                        stored_size,
                        dimension,
                    )
            logger.info("Qdrant collection '%s' already exists — skipping", collection_name)
            return

        sparse_config = (
            {"sparse": qmodels.SparseVectorParams(index=qmodels.SparseIndexParams(on_disk=False))}
            if sparse
            else None
        )

        vectors_config: dict[str, qmodels.VectorParams] = {
            "dense": qmodels.VectorParams(
                size=dimension,
                distance=qmodels.Distance.COSINE,
            )
        }

        if colbert:
            cb_dim = colbert_dimension or dimension
            vectors_config["colbert"] = qmodels.VectorParams(
                size=cb_dim,
                distance=qmodels.Distance.COSINE,
                multivector_config=qmodels.MultiVectorConfig(
                    comparator=qmodels.MultiVectorComparator.MAX_SIM,
                ),
            )

        await self._client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_config,
        )
        logger.info(
            "Qdrant collection '%s' created (dense dimension=%d, sparse=%s, colbert=%s)",
            collection_name, dimension, sparse, colbert,
        )

        # Payload indexes for efficient filtered search
        for field in ("version", "document", "category"):
            await self._client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
            logger.debug("Payload index created on field '%s'", field)

    async def upsert_points(
        self,
        collection_name: str,
        points: list[PointStruct],
    ) -> None:
        """Upsert a batch of points into a collection.

        Args:
            collection_name: Target collection name.
            points: List of PointStruct objects (id, vector, payload).

        Raises:
            RuntimeError: If the client is not connected.
        """
        if not self._client:
            raise RuntimeError("Qdrant client is not connected. Call connect() first.")
        await self._client.upsert(collection_name=collection_name, points=points)
        logger.debug("Upserted %d points into '%s'", len(points), collection_name)

    async def delete_collection(self, collection_name: str) -> None:
        """Delete a collection if it exists.

        Args:
            collection_name: Name of the collection to delete.

        Raises:
            RuntimeError: If the client is not connected.
        """
        if not self._client:
            raise RuntimeError("Qdrant client is not connected. Call connect() first.")
        existing = await self._client.get_collections()
        existing_names = {c.name for c in existing.collections}
        if collection_name not in existing_names:
            logger.warning("delete_collection: '%s' does not exist — skipping", collection_name)
            return
        await self._client.delete_collection(collection_name)
        logger.info("Qdrant collection '%s' deleted", collection_name)

    async def scroll_points(
        self,
        collection_name: str,
        scroll_filter,
        with_vectors=False,
        with_payload=True,
        limit: int = 500,
    ):
        """Scroll through points matching a filter.

        Args:
            collection_name: Target collection name.
            scroll_filter: Qdrant Filter to apply.
            with_vectors: Whether to include vectors in the response.
            with_payload: Whether to include payloads.
            limit: Maximum number of points to return.

        Returns:
            List of Record objects.
        """
        if not self._client:
            raise RuntimeError("Qdrant client is not connected. Call connect() first.")
        points, _next_offset = await self._client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            with_vectors=with_vectors,
            with_payload=with_payload,
            limit=limit,
        )
        return points

    async def get_points_count(self, collection_name: str) -> int:
        """Return the number of points stored in a collection.

        Args:
            collection_name: Target collection name.

        Returns:
            Point count, or 0 if the collection does not exist.

        Raises:
            RuntimeError: If the client is not connected.
        """
        if not self._client:
            raise RuntimeError("Qdrant client is not connected. Call connect() first.")
        try:
            info = await self._client.get_collection(collection_name)
            return info.points_count or 0
        except Exception:
            return 0
