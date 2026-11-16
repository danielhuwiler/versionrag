"""FastAPI application entry point with lifespan management."""

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from versionrag.clients.embedding import create_embedding_client
from versionrag.clients.llm import LLMClient
from versionrag.clients.neo4j import Neo4jClient
from versionrag.clients.qdrant import QdrantClientWrapper
from versionrag.config import settings, setup_logging
from versionrag.generation.generator import AnswerGenerator
from versionrag.indexing.graph_builder import GraphBuilder
from versionrag.indexing.pipeline import IndexingPipeline
from versionrag.models.schemas import HealthStatus
from versionrag.retrieval.graph_retriever import GraphRetriever
from versionrag.retrieval.query_parser import QueryParser, build_document_catalog
from versionrag.retrieval.router import RetrievalRouter
from versionrag.retrieval.vector_retriever import VectorRetriever
from versionrag.utils import version_to_sort_key

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage startup and shutdown of all shared resources.

    Startup order:
    1. Logging
    2. Embedding model → detect dimension
    3. Neo4j connect → ensure schema
    4. Qdrant connect → ensure collection with dynamic dimension

    All clients are stored on ``app.state`` so route handlers can access them
    via ``request.app.state``.
    """
    # ── Startup ───────────────────────────────────────────────────────────────
    setup_logging(settings.log_level)
    logger.info("Starting VersionRAG ...")

    # 1. Embedding model — must load before Qdrant so dimension is known
    vector_types = (
        settings.get_local_vector_types()
        if settings.embedding_provider == "local"
        else None
    )
    embedding = create_embedding_client(
        provider=settings.embedding_provider,
        model_name=settings.embedding_model,
        local_model_name=settings.local_embedding_model,
        # Separate embedding key (institute projects may lack embedding-model
        # access on the main key); falls back to the main key when unset.
        api_key=settings.openai_embedding_api_key or settings.openai_api_key,
        vector_types=vector_types,
    )
    await embedding.load()
    dimension = embedding.get_dimension()

    # 2. Neo4j
    neo4j = Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    await neo4j.connect()
    await neo4j.ensure_schema()

    # 3. Qdrant
    qdrant = QdrantClientWrapper(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        grpc_port=settings.qdrant_grpc_port,
    )
    await qdrant.connect()
    await qdrant.ensure_collection(
        collection_name=settings.qdrant_collection,
        dimension=dimension,
        sparse=embedding.supports_sparse(),
        colbert=embedding.supports_colbert(),
        colbert_dimension=dimension,
    )

    # 4. LLM client
    llm = LLMClient(
        api_key=settings.openai_api_key,
        temperature=settings.llm_temperature,
        default_extract_model=settings.llm_model_extract,
        default_chat_model=settings.llm_model_chat,
        openrouter_api_key=settings.openrouter_api_key,
        openrouter_base_url=settings.openrouter_base_url,
        openrouter_max_concurrent=settings.openrouter_max_concurrent,
    )
    logger.info(
        "LLM client initialized (extract=%s, chat=%s)",
        settings.llm_model_extract,
        settings.llm_model_chat,
    )

    # Expose clients to route handlers via app.state
    app.state.neo4j = neo4j
    app.state.qdrant = qdrant
    app.state.embedding = embedding
    app.state.llm = llm

    logger.info("VersionRAG started successfully")

    yield  # application runs here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    await neo4j.close()
    await qdrant.close()
    logger.info("VersionRAG shutdown complete")


app = FastAPI(
    title="VersionRAG",
    description="Version-aware RAG system for structured document retrieval",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthStatus, summary="System health check")
async def health(request: Request) -> HealthStatus:
    """Return the health status of all system components.

    Queries Neo4j and Qdrant live on every call. Returns HTTP 503 if either
    service is unreachable.

    Returns:
        HealthStatus with per-service status, embedding info, and active constraints.
    """
    neo4j: Neo4jClient = request.app.state.neo4j
    qdrant: QdrantClientWrapper = request.app.state.qdrant
    embedding: EmbeddingClient = request.app.state.embedding

    neo4j_ok = await neo4j.verify_connectivity()
    qdrant_ok = await qdrant.verify_connectivity()

    # Fetch active constraint names from Neo4j
    constraints: list[str] = []
    if neo4j_ok:
        try:
            constraints = await neo4j.get_constraint_names()
        except Exception as exc:
            logger.warning("Could not fetch Neo4j constraints: %s", exc)

    status = HealthStatus(
        status="ok" if (neo4j_ok and qdrant_ok) else "degraded",
        neo4j="ok" if neo4j_ok else "unavailable",
        qdrant="ok" if qdrant_ok else "unavailable",
        embedding_model=settings.embedding_model,
        embedding_dimension=embedding.get_dimension(),
        qdrant_collection=settings.qdrant_collection,
        neo4j_constraints=constraints,
    )

    status_code = 200 if (neo4j_ok and qdrant_ok) else 503
    if status_code != 200:
        return JSONResponse(status_code=status_code, content=status.model_dump())

    return status


# ── Debug / test endpoints ─────────────────────────────────────────────────────
# These endpoints are only intended for development and testing. They allow
# manual inspection and population of the graph without a full indexing run.


@app.post("/debug/test-graph", summary="Populate graph with test data")
async def debug_test_graph(request: Request) -> dict:
    """Insert a representative set of test nodes and relationships.

    Creates categories, documents, versions, NEXT edges, and a handful of
    Change nodes that mirror the structure of the VersionQA dataset.

    Returns:
        Node and relationship counts after insertion.
    """
    neo4j: Neo4jClient = request.app.state.neo4j
    gb = GraphBuilder(neo4j)

    # Categories
    for cat in ("Node.js", "Apache Spark", "CSS Frameworks"):
        await gb.create_category(cat)

    # Documents
    await gb.create_document("assert", "Node.js")
    await gb.create_document("errors", "Node.js")
    await gb.create_document("bootstrap", "CSS Frameworks")

    # Versions — assert (documentation)
    for number in ("14.21.3", "15.14.0", "20.19.0"):
        await gb.create_version("assert", number, version_to_sort_key(number), "documentation", "Node.js")
    await gb.link_versions("assert")

    # Versions — errors (documentation)
    for number in ("15.14.0", "16.20.2", "22.14.0"):
        await gb.create_version("errors", number, version_to_sort_key(number), "documentation", "Node.js")
    await gb.link_versions("errors")

    # Versions — bootstrap (changelog)
    for number in ("5.2.3", "5.3.1", "5.3.3"):
        await gb.create_version("bootstrap", number, version_to_sort_key(number), "changelog", "CSS Frameworks")
    await gb.link_versions("bootstrap")

    # Changes
    await gb.create_change(
        version_from="14.21.3", version_to="15.14.0", document="assert",
        name="CallTracker added",
        description="assert.CallTracker class was added to track function call counts.",
        change_type="implicit", source="deepdiff",
    )
    await gb.create_change(
        version_from="14.21.3", version_to="15.14.0", document="assert",
        name="deepEqual deprecated",
        description="assert.deepEqual was deprecated in favour of assert.deepStrictEqual.",
        change_type="implicit", source="deepdiff",
    )
    await gb.create_change(
        version_from="15.14.0", version_to="16.20.2", document="errors",
        name="ERR_ACCESS_DENIED added",
        description="New error code ERR_ACCESS_DENIED introduced for permission model violations.",
        change_type="implicit", source="deepdiff",
    )
    await gb.create_change(
        version_from="5.2.3", version_to="5.3.1", document="bootstrap",
        name="color-modes-fix",
        description="Fixed color mode switching bug when toggling dark/light theme.",
        change_type="explicit", source="changelog",
    )

    return await gb.count_graph()


@app.get("/debug/graph-stats", summary="Graph node and relationship counts")
async def debug_graph_stats(request: Request) -> dict:
    """Return current node and relationship counts from the graph.

    Returns:
        Dict with ``nodes``, ``relationships``, ``total_nodes``,
        ``total_relationships``.
    """
    neo4j: Neo4jClient = request.app.state.neo4j
    gb = GraphBuilder(neo4j)
    return await gb.count_graph()


@app.get("/debug/search-changes", summary="Full-text search over Change nodes")
async def debug_search_changes(
    request: Request,
    q: str = Query(..., description="Lucene-compatible search term"),
) -> list[dict]:
    """Search Change node names and descriptions using the fulltext index.

    Args:
        q: Search term (e.g. "deprecated", "ERR_*").

    Returns:
        Up to 10 matches ordered by relevance score.
    """
    neo4j: Neo4jClient = request.app.state.neo4j
    gb = GraphBuilder(neo4j)
    return await gb.search_changes(q)


@app.get("/debug/changes", summary="List all changes for a document")
async def debug_changes(
    request: Request,
    document: str = Query(..., description="Document name in the graph"),
) -> list[dict]:
    """Return all Change nodes connected to a document's versions.

    Args:
        document: Document name (e.g. "node.js assert module").

    Returns:
        List of changes with from_version, to_version, name, description,
        change_type, and source.
    """
    neo4j: Neo4jClient = request.app.state.neo4j
    rows = await neo4j.execute_query(
        """
        MATCH (v1:Version)-[:CHANGED_TO]->(c:Change)<-[:CHANGED_FROM]-(v2:Version)
        WHERE v1.document = $document
        RETURN v1.number AS from_version, v2.number AS to_version,
               c.name AS name, c.description AS description,
               c.change_type AS change_type, c.source AS source
        ORDER BY v1.sort_key
        """,
        {"document": document},
    )
    logger.debug("Fetched %d changes for document '%s'", len(rows), document)
    return rows


@app.get("/debug/documents", summary="List all documents and their versions from the graph")
async def debug_documents(request: Request) -> list[dict]:
    """Return all Document nodes with their version counts and version numbers."""
    neo4j: Neo4jClient = request.app.state.neo4j
    rows = await neo4j.execute_query(
        """
        MATCH (d:Document)-[:HAS_VERSION]->(v:Version)
        RETURN d.name AS document, count(v) AS version_count, collect(v.number) AS versions
        ORDER BY d.name
        """
    )
    return rows


@app.get("/debug/qdrant-documents", summary="List unique document values in Qdrant payloads")
async def debug_qdrant_documents(request: Request) -> dict:
    """Scroll through Qdrant points and collect all unique 'document' payload values."""
    qdrant: QdrantClientWrapper = request.app.state.qdrant
    collection = settings.qdrant_collection
    documents: set[str] = set()
    offset = None

    while True:
        results = await qdrant.client.scroll(
            collection_name=collection,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points, next_offset = results
        for point in points:
            doc_val = point.payload.get("document") if point.payload else None
            if doc_val is not None:
                documents.add(doc_val)
        if next_offset is None:
            break
        offset = next_offset

    return {"unique_documents": sorted(documents), "count": len(documents)}


@app.get("/debug/sample-chunks", summary="Sample random chunks from Qdrant")
async def debug_sample_chunks(
    request: Request,
    limit: int = Query(5, description="Number of sample points to return"),
) -> list[dict]:
    """Load a few random points from Qdrant and return their full payloads."""
    qdrant: QdrantClientWrapper = request.app.state.qdrant
    collection = settings.qdrant_collection
    results = await qdrant.client.scroll(
        collection_name=collection,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    points, _ = results
    return [
        {"id": str(p.id), "payload": p.payload}
        for p in points
    ]


@app.delete("/debug/reset-graph", summary="Delete all graph nodes and relationships")
async def debug_reset_graph(request: Request) -> dict:
    """Remove all nodes and relationships from the graph.

    Uses DETACH DELETE so relationships are removed along with nodes.

    Returns:
        Confirmation status dict.
    """
    neo4j: Neo4jClient = request.app.state.neo4j
    await neo4j.execute_query("MATCH (n) DETACH DELETE n")
    logger.info("Graph reset: all nodes and relationships deleted")
    return {"status": "reset"}


# ── Indexing endpoints ─────────────────────────────────────────────────────────


class IndexingRunRequest(BaseModel):
    """Request body for POST /indexing/run."""

    data_dir: str = "data/raw"
    reset: bool = True
    recursive: bool = False


class IndexingAddRequest(BaseModel):
    """Request body for POST /indexing/add."""

    file_paths: list[str]


def _make_pipeline(request: Request) -> IndexingPipeline:
    """Create an IndexingPipeline from the shared app state."""
    return IndexingPipeline(
        neo4j=request.app.state.neo4j,
        qdrant=request.app.state.qdrant,
        embedding=request.app.state.embedding,
        llm=request.app.state.llm,
    )


@app.post("/indexing/run", summary="Run the full indexing pipeline")
async def indexing_run(request: Request, body: IndexingRunRequest) -> dict:
    """Execute the indexing pipeline on a directory of documents.

    Phase A: For each file — extract metadata, create graph nodes, index in Qdrant.
    Phase B: Link versions with NEXT edges, extract changes between consecutive pairs.

    Args:
        body: ``data_dir`` (relative or absolute path) and ``reset`` flag.

    Returns:
        Pipeline summary with per-file results, graph stats, and duration.
    """
    pipeline = _make_pipeline(request)
    return await pipeline.run(
        data_dir=body.data_dir,
        reset=body.reset,
        recursive=body.recursive,
    )


class ReingestRequest(BaseModel):
    file_paths: list[str]
    concurrency: int | None = None


@app.post("/indexing/reingest", summary="Re-run Phase A for specific files")
async def indexing_reingest(request: Request, body: ReingestRequest) -> dict:
    """Repair a partial ingest without re-running change extraction.

    Args:
        body: ``file_paths`` to re-ingest and an optional ``concurrency`` cap.

    Returns:
        Counts of re-ingested files and chunks, plus any files that failed again.
    """
    pipeline = _make_pipeline(request)
    return await pipeline.reingest(body.file_paths, body.concurrency)


@app.post("/indexing/changes", summary="Re-run change extraction only (Phase B)")
async def indexing_changes(request: Request, clear: bool = True) -> dict:
    """Rebuild the change layer over an already-ingested graph.

    Intended for ``INDEX_MODE=parser``, where Phase A is fully deterministic:
    with the embedding model fixed, one ingest is reusable across every
    generator, and only this phase has to be re-paid per model.

    Args:
        clear: Delete existing Change nodes before extracting (default True).

    Returns:
        Graph and change stats, plus the duration of this phase alone.
    """
    pipeline = _make_pipeline(request)
    return await pipeline.rebuild_changes(clear=clear)


@app.post("/indexing/add", summary="Add documents to the existing index")
async def indexing_add(request: Request, body: IndexingAddRequest) -> dict:
    """Add one or more files to the existing graph and index.

    No reset — new documents are merged into the existing graph.
    After all files are ingested, NEXT edges and changes are updated.

    Args:
        body: ``file_paths`` — list of paths to .md or .pdf files.

    Returns:
        Pipeline summary with per-file results, graph stats, and duration.
    """
    pipeline = _make_pipeline(request)
    return await pipeline.add(file_paths=body.file_paths)


@app.get("/indexing/status", summary="Current indexing status")
async def indexing_status(request: Request) -> dict:
    """Return the current state of the graph and Qdrant collection.

    Returns:
        Dict with ``graph`` (node/relationship counts) and
        ``qdrant`` (collection name and point count).
    """
    neo4j: Neo4jClient = request.app.state.neo4j
    qdrant: QdrantClientWrapper = request.app.state.qdrant

    gb = GraphBuilder(neo4j)
    graph_stats = await gb.count_graph()
    points_count = await qdrant.get_points_count(settings.qdrant_collection)

    return {
        "graph": graph_stats,
        "qdrant": {
            "collection": settings.qdrant_collection,
            "points_count": points_count,
        },
    }


# ── Query / Retrieval endpoints ───────────────────────────────────────────────


class QueryRequest(BaseModel):
    """Request body for POST /query and POST /query/retrieve-only."""

    question: str
    top_k: int = 5
    query_type_override: str | None = None


async def _build_retrieval_pipeline(request: Request) -> tuple[
    QueryParser, RetrievalRouter
]:
    """Instantiate the query parser and retrieval router from app state.

    The document catalog is rebuilt from Neo4j on every call so the parser's
    system prompt always reflects the current corpus (no startup cache, no
    refresh hook). Cost is a single aggregation query per request.
    """
    llm: LLMClient = request.app.state.llm
    neo4j_client: Neo4jClient = request.app.state.neo4j
    qdrant_client: QdrantClientWrapper = request.app.state.qdrant
    embedding: EmbeddingClient = request.app.state.embedding

    catalog = await build_document_catalog(neo4j_client)
    # Structured doc-name list for the multi-part prefix expansion path.
    # No-op when settings.multi_part_documents is False (software default),
    # so this extra query is cheap and side-effect-free for the software
    # corpus pipeline.
    doc_rows = await neo4j_client.execute_query(
        "MATCH (d:Document) RETURN d.name AS name"
    )
    document_names = [r["name"] for r in doc_rows]
    parser = QueryParser(llm, document_catalog=catalog, document_names=document_names)
    graph_retriever = GraphRetriever(GraphBuilder(neo4j_client), embedding=embedding)
    vector_retriever = VectorRetriever(embedding, qdrant_client)
    router = RetrievalRouter(graph_retriever, vector_retriever)
    return parser, router


@app.post("/query", summary="Answer a question with retrieval-augmented generation")
async def query(request: Request, body: QueryRequest) -> dict:
    """Full RAG pipeline: parse the question, retrieve context, generate an answer."""
    t_start = time.perf_counter()

    try:
        parser, router = await _build_retrieval_pipeline(request)
        parsed = await parser.parse(body.question)
        if body.query_type_override:
            parsed.query_type = body.query_type_override
        result = await router.route(parsed)

        logger.info(
            "Retrieval complete: type=%s, mode=%s, chunks=%d, changes=%d, versions=%d",
            parsed.query_type, result.retrieval_mode,
            len(result.chunks), len(result.changes), len(result.versions),
        )

        llm: LLMClient = request.app.state.llm
        generator = AnswerGenerator(llm)
        answer = await generator.generate(parsed, result)

        duration = time.perf_counter() - t_start
        logger.info("Query complete: type=%s, mode=%s, duration=%.2fs",
                     parsed.query_type, result.retrieval_mode, duration)

        return {
            "question": answer.question,
            "answer": answer.answer,
            "query_type": answer.query_type,
            "retrieval_mode": answer.retrieval_mode,
            "sources": answer.sources,
            "sources_count": len(answer.sources),
            "parsed_query": parsed.model_dump(),
            "duration_seconds": round(duration, 3),
        }
    except Exception as exc:
        logger.exception("Query failed: %s", exc)
        duration = time.perf_counter() - t_start
        return JSONResponse(
            status_code=500,
            content={"error": type(exc).__name__, "detail": str(exc), "duration": round(duration, 3)},
        )


@app.post(
    "/query/retrieve-only",
    summary="Parse and retrieve context without answer generation",
)
async def query_retrieve_only(request: Request, body: QueryRequest) -> dict:
    """Retrieval-only endpoint: parse the question and return raw retrieval results
    without generating an answer.

    Useful for debugging, evaluation, and inspecting what context the system
    retrieves before the LLM sees it.

    Args:
        body: ``question`` and optional ``top_k``.

    Returns:
        Dict with parsed query metadata and raw retrieval results.
    """
    t_start = time.perf_counter()

    # Parse and retrieve
    parser, router = await _build_retrieval_pipeline(request)
    parsed = await parser.parse(body.question)
    if body.query_type_override:
        parsed.query_type = body.query_type_override
    result = await router.route(parsed)

    duration = time.perf_counter() - t_start
    logger.info(
        "Retrieve-only complete: type=%s, mode=%s, chunks=%d, changes=%d, "
        "versions=%d, duration=%.2fs",
        parsed.query_type,
        result.retrieval_mode,
        len(result.chunks),
        len(result.changes),
        len(result.versions),
        duration,
    )

    return {
        "parsed_query": parsed.model_dump(),
        "query_type": parsed.query_type,
        "retrieval_mode": result.retrieval_mode,
        "chunks": [c.model_dump() for c in result.chunks],
        "changes": [c.model_dump() for c in result.changes],
        "versions": [v.model_dump() for v in result.versions],
        "chunks_count": len(result.chunks),
        "changes_count": len(result.changes),
        "versions_count": len(result.versions),
        "duration_seconds": round(duration, 3),
    }
