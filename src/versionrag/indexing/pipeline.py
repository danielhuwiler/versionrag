"""Orchestrates the full document indexing pipeline."""

import logging
import re
import time
from pathlib import Path
from typing import Any

from versionrag.clients.embedding import EmbeddingClient
from versionrag.clients.llm import LLMClient
from versionrag.clients.neo4j import Neo4jClient
from versionrag.clients.qdrant import QdrantClientWrapper
from versionrag.config import settings
from versionrag.indexing.attribute_extractor import AttributeExtractor
from versionrag.indexing.change_extractor import ChangeExtractor
from versionrag.indexing.content_indexer import ContentIndexer
from versionrag.indexing.deterministic_extractor import DeterministicAttributeExtractor
from versionrag.indexing.graph_builder import GraphBuilder
from versionrag.utils import (
    ALLOWED_CATEGORIES,
    CATEGORY_ALIASES,
    derive_title_from_path,
    normalize_version,
    read_file,
    version_to_sort_key,
)

logger = logging.getLogger(__name__)


class IndexingPipeline:
    """Two-phase pipeline: ingest (per file) → finalize (once).

    Phase A — Per-file ingestion (repeated for each file):
        1. LLM extracts metadata (title, version, doc_type, category_hint).
        2. MERGE Category → Document → Version nodes in Neo4j (idempotent).
        3. Chunk, embed, and upsert into Qdrant.
        4. Free metadata from RAM.

    Phase B — Finalization (once after all files are ingested):
        1. Link consecutive versions with NEXT edges for all documents.
        2. Extract changes between consecutive version pairs (from graph).
    """

    def __init__(
        self,
        neo4j: Neo4jClient,
        qdrant: QdrantClientWrapper,
        embedding: EmbeddingClient,
        llm: LLMClient,
    ) -> None:
        self._neo4j = neo4j
        self._qdrant = qdrant
        self._embedding = embedding
        self._llm = llm

    # ── Phase A: per-file ingestion ───────────────────────────────────────────

    async def _ingest_single(self, file_path: Path) -> dict[str, Any]:
        """Ingest a single file: extract metadata, create graph nodes, index content.

        Args:
            file_path: Path to a .md or .pdf file.

        Returns:
            Dict with ``file``, ``title``, ``version``, ``chunks`` keys.
        """
        logger.info("  Step 1/3: Extracting attributes from '%s'", file_path.name)

        # In the "metadata_filtered" and "parser" baselines, metadata is derived
        # deterministically from the path/filename — no LLM attribute call. The
        # two differ only in Phase B (whether the change layer is built).
        if settings.index_mode in ("metadata_filtered", "parser"):
            extractor: AttributeExtractor | DeterministicAttributeExtractor = (
                DeterministicAttributeExtractor()
            )
        else:
            extractor = AttributeExtractor(self._llm)
        metadata = await extractor.extract_metadata(file_path)

        # ── Programmatic normalization (safety net for LLM inconsistencies) ──
        # Title: lowercase + collapse whitespace to underscores. The new prompt
        # already mandates snake_case lowercase; this is a defensive normalizer
        # for the cases where the LLM returns 'PyTorch Conv2D' or similar.
        # NOTE: we deliberately do NOT strip digit-sequences here — that was
        # the old Bootstrap/Spark cleanup which destroyed valid titles like
        # 'pytorch_conv2d' → 'pytorch_con d'.
        metadata.title = re.sub(r"\s+", "_", metadata.title.strip().lower())

        # For files inside the large_dataset/raw/{changelog,docs}/... layout
        # the leaf-folder name unambiguously identifies the Document family;
        # override the LLM-derived title with that deterministic value to
        # prevent title drift (e.g. 'tensorflow_130', 'pytorch_con d',
        # 'whatsnew_0.13.1' getting their own Document nodes instead of
        # being merged into the parent).
        path_title = derive_title_from_path(file_path)
        if path_title and path_title != metadata.title:
            logger.info(
                "Title override from path: '%s' → '%s' for %s",
                metadata.title, path_title, file_path,
            )
            metadata.title = path_title

        # Centralized version normalization: strips v/go prefixes, decodes
        # 3-digit compact form ("170" → "1.7.0", "120rc3" → "1.2.0rc3"),
        # lowercases ROS codenames. Keeps Neo4j Version.number and Qdrant
        # chunk payloads consistent so Cypher and vector-filter lookups
        # match each other (see versionrag.utils.normalize_version for the
        # full set of rules).
        metadata.version = normalize_version(metadata.version)

        # Recalculate sort_key after version cleanup
        metadata.sort_key = version_to_sort_key(metadata.version)

        # Normalize category_hint: lowercase, strip, spaces→underscores, then
        # map known LLM variants to canonical values, then validate against
        # the closed allowed-list. Anything outside collapses to 'unknown'.
        raw_cat = metadata.category_hint
        norm_cat = raw_cat.strip().lower().replace(" ", "_")
        norm_cat = CATEGORY_ALIASES.get(norm_cat, norm_cat)
        if norm_cat not in ALLOWED_CATEGORIES:
            logger.warning(
                "LLM returned category '%s' (normalized '%s') which is not in "
                "ALLOWED_CATEGORIES — collapsing to 'unknown' for file %s",
                raw_cat, norm_cat, file_path,
            )
            norm_cat = "unknown"
        metadata.category_hint = norm_cat

        logger.info(
            "  Step 1/3 done: title='%s', version='%s', category='%s', doc_type='%s'",
            metadata.title, metadata.version, metadata.category_hint, metadata.doc_type.value,
        )

        # Normalize keys the same way DocumentClusterer did
        cat_key = metadata.category_hint.strip().lower()
        doc_key = metadata.title.strip().lower()

        # MERGE graph nodes (idempotent)
        logger.info("  Step 2/3: Building graph nodes for '%s' v%s", doc_key, metadata.version)
        builder = GraphBuilder(self._neo4j)
        await builder.create_category(cat_key)
        await builder.create_document(doc_key, cat_key)
        await builder.create_version(
            document=doc_key,
            number=metadata.version,
            sort_key=metadata.sort_key,
            doc_type=metadata.doc_type.value,
            category=cat_key,
            file_path=str(file_path),
        )

        # Chunk + embed + upsert into Qdrant
        logger.info("  Step 3/3: Chunking & embedding '%s' v%s into Qdrant", doc_key, metadata.version)
        indexer = ContentIndexer(self._embedding, self._qdrant)
        text = read_file(file_path)
        chunks = await indexer.index_document(metadata, text)

        logger.info(
            "Ingested: %s v%s (%s) — %d chunks",
            metadata.title, metadata.version, metadata.doc_type.value, chunks,
        )

        return {
            "file": str(file_path),
            "title": metadata.title,
            "version": metadata.version,
            "chunks": chunks,
        }

    # ── Phase B: finalization ─────────────────────────────────────────────────

    async def _finalize(self) -> dict[str, Any]:
        """Link all versions and extract changes using Neo4j as source of truth.

        Returns:
            Dict with ``graph_stats`` and ``change_stats``.
        """
        builder = GraphBuilder(self._neo4j)

        # Link consecutive versions with NEXT edges (deterministic — always run)
        logger.info("Phase B: linking all versions with NEXT edges")
        await builder.link_all_versions()

        # Extract changes between consecutive version pairs. Skipped in the
        # "metadata_filtered" ablation baseline — the LLM change-summary layer
        # is exactly what that baseline removes, so no Change nodes are built
        # (change_retrieval queries then fall back to vector search). The
        # "parser" mode deliberately keeps it — that mode drops only the
        # attribute call, so it runs the change layer exactly as "full" does.
        if settings.index_mode == "metadata_filtered":
            logger.info(
                "Phase B: index_mode=metadata_filtered — skipping LLM change extraction"
            )
            change_stats: dict[str, Any] = {
                "skipped": True,
                "total_explicit": 0,
                "total_implicit": 0,
                "total_changes": 0,
            }
        else:
            logger.info("Phase B: extracting changes between consecutive versions")
            change_extractor = ChangeExtractor(self._llm, builder, self._qdrant)
            change_stats = await change_extractor.process_from_graph()

        graph_stats = await builder.count_graph()
        logger.info("Phase B complete: graph=%s, changes=%s", graph_stats, change_stats)
        return {
            "graph_stats": graph_stats,
            "change_stats": change_stats,
        }

    # ── Public entry points ───────────────────────────────────────────────────

    async def reingest(
        self, file_paths: list[str], concurrency: int | None = None
    ) -> dict[str, Any]:
        """Re-run Phase A for specific files. No reset, no Phase B.

        Repairs a partial ingest. ``_ingest_single`` is idempotent — graph nodes
        are MERGEd and chunk point IDs are ``uuid5(document, version, index)``,
        so re-running a file overwrites its chunks in place rather than
        duplicating them.

        Exists because Phase A swallows per-file failures: a 429 from the
        embedding endpoint is logged and the run continues, so the pipeline can
        report success with a fraction of the corpus missing from Qdrant. Change
        extraction reads Neo4j only, so a completed Phase B stays valid and does
        not need re-paying for.

        Args:
            file_paths: Files to re-ingest.
            concurrency: In-flight files. Defaults to a third of the indexing
                cap, since a full-speed retry tends to hit the same rate limit
                that caused the gap.
        """
        import asyncio as _asyncio

        limit = concurrency or max(1, settings.openrouter_max_concurrent // 3)
        sem = _asyncio.Semaphore(limit)
        ok: list[dict[str, Any]] = []
        failed: list[dict[str, str]] = []
        t0 = time.time()

        async def _one(idx: int, path: str) -> None:
            async with sem:
                try:
                    result = await self._ingest_single(Path(path))
                    ok.append(result)
                    logger.info("[%d/%d] REINGEST ok '%s' — %d chunks",
                                idx, len(file_paths), Path(path).name, result["chunks"])
                except Exception as exc:  # noqa: BLE001 — reported to the caller
                    failed.append({"file": path, "error": f"{type(exc).__name__}: {exc}"})
                    logger.error("[%d/%d] REINGEST FAILED '%s': %s",
                                 idx, len(file_paths), Path(path).name, exc)

        await _asyncio.gather(*(_one(i, f) for i, f in enumerate(file_paths, 1)))

        return {
            "requested": len(file_paths),
            "reingested": len(ok),
            "failed": failed,
            "chunks": sum(r["chunks"] for r in ok),
            "concurrency": limit,
            "duration_seconds": round(time.time() - t0, 2),
        }

    async def rebuild_changes(self, clear: bool = True) -> dict[str, Any]:
        """Re-run Phase B only, against a graph that is already ingested.

        In ``parser`` mode Phase A makes no LLM call at all — metadata comes from
        the path parser — so with the embedding model held fixed, the ingested
        graph and the Qdrant chunks are identical no matter which generator is
        configured. Only the change layer is model-dependent. Ingesting once and
        then re-running just this per generator saves the embedding spend and the
        Phase A wall-clock on every run after the first.

        Not valid in ``full`` mode: there Phase A includes the LLM attribute call,
        so the ingested graph is itself a function of the generator.

        Args:
            clear: Delete existing Change nodes first. Leave True unless you are
                deliberately layering a second generator's changes onto the same
                graph — ``process_from_graph`` does not deduplicate.
        """
        if settings.index_mode == "full":
            logger.warning(
                "rebuild_changes called with index_mode=full — Phase A used an LLM "
                "attribute call, so this graph is not generator-independent"
            )
        if clear:
            deleted = await self._neo4j.execute_query(
                "MATCH (c:Change) WITH c, count(*) AS _ DETACH DELETE c RETURN count(*) AS n"
            )
            logger.info("Cleared %s existing Change nodes", deleted[0]["n"] if deleted else 0)
        t0 = time.time()
        result = await self._finalize()
        result["duration_seconds"] = round(time.time() - t0, 2)
        return result

    async def run(
        self,
        data_dir: str | Path,
        reset: bool = True,
        recursive: bool = False,
    ) -> dict[str, Any]:
        """Execute the full indexing pipeline on a directory.

        Args:
            data_dir: Directory containing .md and .pdf files to index.
            reset: If True, wipe the graph and Qdrant collection before indexing.
            recursive: If True, walk subdirectories with rglob; otherwise only top-level files.

        Returns:
            Summary dict with per-file results, graph stats, and duration.
        """
        data_dir = Path(data_dir)
        t0 = time.time()

        # ── Optional reset ────────────────────────────────────────────────
        if reset:
            await self._neo4j.execute_query("MATCH (n) DETACH DELETE n")
            await self._qdrant.delete_collection(settings.qdrant_collection)
            await self._qdrant.ensure_collection(
                collection_name=settings.qdrant_collection,
                dimension=self._embedding.get_dimension(),
                sparse=self._embedding.supports_sparse(),
            )
            logger.info("Reset complete — graph and Qdrant collection cleared")

        # ── Phase A: ingest each file ─────────────────────────────────────
        glob_fn = data_dir.rglob if recursive else data_dir.glob
        files = sorted(list(glob_fn("*.md")) + list(glob_fn("*.pdf")))
        if not files:
            logger.warning("No documents found in '%s' — pipeline aborted", data_dir)
            return {
                "documents_ingested": 0,
                "graph_stats": {},
                "change_stats": {},
                "duration_seconds": round(time.time() - t0, 2),
            }

        # Parallel ingest with a global concurrency cap. Each ``_ingest_single``
        # does: 1 LLM attribute call + chunking + N embedding calls + Neo4j
        # MERGE + Qdrant upserts. MERGE is concurrent-safe; OpenAI / OpenRouter
        # absorb 10 in-flight calls comfortably. Cuts a 882-file Phase A from
        # ~15 min sequential to ~2-3 min parallel.
        import asyncio as _asyncio
        sem = _asyncio.Semaphore(settings.openrouter_max_concurrent)
        ingested: list[dict[str, Any]] = []
        ingested_lock = _asyncio.Lock()

        async def _process_one(idx: int, file_path: Path) -> None:
            async with sem:
                logger.info(
                    "[%d/%d] START ingesting '%s'", idx, len(files), file_path.name,
                )
                try:
                    result = await self._ingest_single(file_path)
                    async with ingested_lock:
                        ingested.append(result)
                    logger.info(
                        "[%d/%d] DONE  '%s' — %d chunks",
                        idx, len(files), file_path.name, result["chunks"],
                    )
                except Exception as exc:
                    logger.error(
                        "[%d/%d] FAILED '%s': %s", idx, len(files), file_path.name, exc,
                    )

        await _asyncio.gather(*(_process_one(i, f) for i, f in enumerate(files, 1)))

        logger.info("Phase A complete: ingested %d / %d files", len(ingested), len(files))

        # ── Phase B: finalize ─────────────────────────────────────────────
        final = await self._finalize()

        duration = round(time.time() - t0, 2)
        logger.info("Pipeline complete in %.1fs", duration)

        return {
            "documents_ingested": len(ingested),
            "ingested": ingested,
            **final,
            "duration_seconds": duration,
        }

    async def add(
        self,
        file_paths: list[str | Path],
    ) -> dict[str, Any]:
        """Add one or more files to the existing graph and index.

        No reset — new documents are merged into the existing graph.

        Args:
            file_paths: List of paths to .md or .pdf files.

        Returns:
            Summary dict with per-file results, graph stats, and duration.
        """
        t0 = time.time()

        # Ensure Qdrant collection exists (no-op if already present)
        await self._qdrant.ensure_collection(
            collection_name=settings.qdrant_collection,
            dimension=self._embedding.get_dimension(),
            sparse=self._embedding.supports_sparse(),
        )

        # ── Phase A: ingest each file ─────────────────────────────────────
        ingested: list[dict[str, Any]] = []
        for fp in file_paths:
            file_path = Path(fp)
            try:
                result = await self._ingest_single(file_path)
                ingested.append(result)
            except Exception as exc:
                logger.error("Failed to ingest '%s': %s", file_path, exc)

        logger.info("Phase A complete: ingested %d / %d files", len(ingested), len(file_paths))

        # ── Phase B: finalize ─────────────────────────────────────────────
        final = await self._finalize()

        duration = round(time.time() - t0, 2)
        logger.info("Add complete in %.1fs", duration)

        return {
            "documents_ingested": len(ingested),
            "ingested": ingested,
            **final,
            "duration_seconds": duration,
        }
