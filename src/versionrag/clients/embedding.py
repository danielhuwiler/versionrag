"""Embedding clients: OpenAI API (dense-only) and local BGE-M3 (dense+sparse)."""

import logging
import time
from abc import ABC, abstractmethod

from openai import AsyncOpenAI

from versionrag.clients.usage_tracker import get_tracker

logger = logging.getLogger(__name__)


class EmbeddingClient(ABC):
    """Abstract base for embedding clients."""

    @abstractmethod
    async def load(self) -> None: ...

    @abstractmethod
    async def encode(self, texts: list[str], *, purpose: str = "embedding") -> dict[str, list]: ...

    @abstractmethod
    async def encode_query(self, query: str) -> dict[str, object]: ...

    @abstractmethod
    def get_dimension(self) -> int: ...

    @abstractmethod
    def supports_sparse(self) -> bool: ...

    @abstractmethod
    def supports_colbert(self) -> bool: ...

    def is_loaded(self) -> bool:
        return self.get_dimension() is not None


class OpenAIEmbeddingClient(EmbeddingClient):
    """Dense-only embeddings via the OpenAI API."""

    def __init__(self, model_name: str, api_key: str) -> None:
        self._model_name = model_name
        # max_retries=8: the SDK does exponential backoff honoring the
        # Retry-After header. Indexing embeds hundreds of files in parallel and
        # transiently exceeds the embedding TPM limit; the default of 2 retries
        # let ~11% of files drop out of the vector store silently. 8 retries
        # absorbs those bursts so every chunk lands in Qdrant.
        self._client = AsyncOpenAI(api_key=api_key, max_retries=8)
        self._dimension: int | None = None

    async def load(self) -> None:
        logger.info("Detecting dimension for OpenAI model '%s' ...", self._model_name)
        resp = await self._client.embeddings.create(
            model=self._model_name,
            input=["dimension probe"],
        )
        self._dimension = len(resp.data[0].embedding)
        logger.info("OpenAI embedding '%s' ready, dimension=%d", self._model_name, self._dimension)

    async def encode(self, texts: list[str], *, purpose: str = "embedding") -> dict[str, list]:
        t0 = time.perf_counter()
        resp = await self._client.embeddings.create(model=self._model_name, input=texts)
        duration = time.perf_counter() - t0
        usage = getattr(resp, "usage", None)
        if usage is not None:
            get_tracker().record(
                model=self._model_name, purpose=purpose,
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=0,
                duration_seconds=duration,
            )
        return {"dense": [item.embedding for item in resp.data]}

    async def encode_query(self, query: str) -> dict[str, object]:
        batch = await self.encode([query], purpose="evaluation.query_embedding")
        return {"dense": batch["dense"][0]}

    def get_dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError("OpenAIEmbeddingClient not loaded.")
        return self._dimension

    def supports_sparse(self) -> bool:
        return False

    def supports_colbert(self) -> bool:
        return False


class LocalEmbeddingClient(EmbeddingClient):
    """Dense + sparse embeddings via local BGE-M3 (FlagEmbedding)."""

    def __init__(self, model_name: str, vector_types: set[str] | None = None) -> None:
        self._model_name = model_name
        self._model = None
        self._dimension: int | None = None
        self._vector_types = vector_types or {"dense", "sparse"}

    async def load(self) -> None:
        import numpy as np
        from FlagEmbedding import BGEM3FlagModel

        logger.info("Loading local embedding model '%s' ...", self._model_name)
        self._model = BGEM3FlagModel(self._model_name, use_fp16=True)

        probe = self._model.encode(
            ["dimension probe"], return_sparse=False, return_colbert_vecs=False
        )
        self._dimension = int(np.array(probe["dense_vecs"]).shape[-1])
        logger.info(
            "Local embedding '%s' loaded, dimension=%d", self._model_name, self._dimension
        )

    async def encode(self, texts: list[str], *, purpose: str = "embedding") -> dict[str, list]:
        if not self._model:
            raise RuntimeError("LocalEmbeddingClient not loaded.")
        # Local embeddings are free — we still log call count + duration
        # under the same tracker so the resource report is complete.
        t0 = time.perf_counter()
        _purpose = purpose
        output = self._model.encode(
            texts,
            return_sparse="sparse" in self._vector_types,
            return_colbert_vecs="colbert" in self._vector_types,
        )
        result: dict[str, list] = {"dense": list(output["dense_vecs"])}
        if "sparse" in self._vector_types:
            result["sparse"] = list(output["lexical_weights"])
        if "colbert" in self._vector_types:
            result["colbert"] = list(output["colbert_vecs"])
        duration = time.perf_counter() - t0
        # token count not available locally; record 0 to preserve schema
        get_tracker().record(
            model=self._model_name, purpose=_purpose,
            prompt_tokens=0, completion_tokens=0, duration_seconds=duration,
        )
        return result

    async def encode_query(self, query: str) -> dict[str, object]:
        batch = await self.encode([query], purpose="evaluation.query_embedding")
        result: dict[str, object] = {"dense": batch["dense"][0]}
        if "sparse" in self._vector_types:
            result["sparse"] = batch["sparse"][0]
        if "colbert" in self._vector_types:
            result["colbert"] = batch["colbert"][0]
        return result

    def get_dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError("LocalEmbeddingClient not loaded.")
        return self._dimension

    def supports_sparse(self) -> bool:
        return "sparse" in self._vector_types

    def supports_colbert(self) -> bool:
        return "colbert" in self._vector_types


def create_embedding_client(
    provider: str,
    model_name: str,
    local_model_name: str,
    api_key: str,
    vector_types: set[str] | None = None,
) -> EmbeddingClient:
    """Factory: create the right embedding client based on provider setting."""
    if provider == "local":
        logger.info("Using local embedding provider: %s (vector_types=%s)", local_model_name, vector_types)
        return LocalEmbeddingClient(model_name=local_model_name, vector_types=vector_types)
    logger.info("Using OpenAI embedding provider: %s", model_name)
    return OpenAIEmbeddingClient(model_name=model_name, api_key=api_key)
