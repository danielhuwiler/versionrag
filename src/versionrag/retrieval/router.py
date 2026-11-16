"""Query router that dispatches parsed queries to the appropriate retriever."""

import logging

from versionrag.models.enums import QueryType
from versionrag.models.schemas import ParsedQuery, RetrievalResult
from versionrag.retrieval.graph_retriever import GraphRetriever
from versionrag.retrieval.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)


class RetrievalRouter:
    """Routes parsed queries to the graph or vector retriever based on query type."""

    def __init__(
        self, graph_retriever: GraphRetriever, vector_retriever: VectorRetriever
    ) -> None:
        self._graph = graph_retriever
        self._vector = vector_retriever

    async def route(self, parsed: ParsedQuery) -> RetrievalResult:
        """Route a parsed query to the appropriate retriever.

        Routing logic:
        - VERSION_LISTING → Graph (version nodes)
        - CHANGE_RETRIEVAL_* → Graph (change nodes), fallback to vector if empty
        - CONTENT_VERSION_SPECIFIC → Vector (with version filter)
        - CONTENT_RETRIEVAL → Vector (unfiltered hybrid search)

        Args:
            parsed: The structured parsed query from QueryParser.

        Returns:
            RetrievalResult from the selected retriever.
        """
        logger.debug("Routing query: type=%s, question=%s", parsed.query_type, parsed.question)

        match parsed.query_type:
            case QueryType.VERSION_LISTING:
                result = await self._graph.retrieve_versions(parsed)

            case QueryType.CHANGE_RETRIEVAL_EXPLICIT | QueryType.CHANGE_RETRIEVAL_IMPLICIT:
                result = await self._graph.retrieve_changes(parsed)

                # Fallback to vector search if no changes found in graph
                if not result.changes:
                    logger.info("No graph changes found, falling back to vector search")
                    result = await self._vector.retrieve_content(parsed)

            case QueryType.CONTENT_VERSION_SPECIFIC:
                result = await self._vector.retrieve_content(parsed)

            case QueryType.CONTENT_RETRIEVAL:
                result = await self._vector.retrieve_content(parsed)

            case _:
                logger.warning(
                    "Unknown query type: %s, defaulting to vector search",
                    parsed.query_type,
                )
                result = await self._vector.retrieve_content(parsed)

        logger.info(
            "Routed query type=%s -> mode=%s (chunks=%d, changes=%d, versions=%d)",
            parsed.query_type,
            result.retrieval_mode,
            len(result.chunks),
            len(result.changes),
            len(result.versions),
        )
        return result
