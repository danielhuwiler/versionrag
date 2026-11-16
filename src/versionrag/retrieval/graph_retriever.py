"""Graph-based retriever for version listings and change queries."""

import logging
import math
import re
from typing import Any

from versionrag.clients.embedding import EmbeddingClient
from versionrag.indexing.graph_builder import GraphBuilder
from versionrag.models.enums import ChangeType, DocumentType, RetrievalMode
from versionrag.models.schemas import Change, ParsedQuery, RetrievalResult, VersionInfo

logger = logging.getLogger(__name__)

# Lucene returns up to 20 rows; we keep this many after semantic re-ranking.
# Generation gets a tighter, better-ranked list rather than a long noisy one.
_RERANK_KEEP_TOP_N = 8

_PIN_CITE_BARE_RE = re.compile(
    r"(\d+[A-Za-z]*(?:\([0-9A-Za-z]+\))*)"
)


def _bare_pin_cite(pin: str) -> str:
    """Strip 's.' / 'para.' / 'art.' prefix and return the bare number+chain.

    Examples:
      's. 235(7)(ja)' → '235(7)(ja)'
      'para. 22(1)(b)' → '22(1)(b)'
      'section 47(2)' → '47(2)'
      's.85(7)(ba)' → '85(7)(ba)'
    """
    if not pin:
        return ""
    m = _PIN_CITE_BARE_RE.search(pin)
    return m.group(1).lower() if m else ""


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity for two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class GraphRetriever:
    """Retrieves version and change information from the Neo4j knowledge graph."""

    def __init__(
        self,
        graph_builder: GraphBuilder,
        embedding: EmbeddingClient | None = None,
    ) -> None:
        self._graph = graph_builder
        # Optional — when set, fulltext Change queries are post-filtered by
        # cosine similarity of the question against each candidate's
        # description. Lucene tf-idf is too unscoped for queries like
        # "Blackwell sm103 added to FP8 GEMM" (where 'GEMM' alone matches
        # many unrelated Changes).
        self._embedding = embedding

    async def _semantic_rerank(
        self, question: str, rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Re-rank Lucene candidates by semantic similarity to the question.

        Embeds the question once and each candidate's ``name+description``
        once, sorts by cosine similarity, returns the top ``_RERANK_KEEP_TOP_N``.
        Falls back to the original order on any error (best-effort step).
        """
        if not self._embedding or len(rows) <= _RERANK_KEEP_TOP_N:
            return rows
        try:
            q_vec = (await self._embedding.encode_query(question))["dense"]
            candidate_texts = [
                f"{r.get('name', '')} — {r.get('description', '')}".strip(" —")
                for r in rows
            ]
            embed = await self._embedding.encode(
                candidate_texts, purpose="evaluation.change_rerank"
            )
            cand_vecs = embed["dense"]
            scored = [
                (_cosine(list(q_vec), list(v)), r)
                for r, v in zip(rows, cand_vecs)
            ]
            scored.sort(key=lambda x: -x[0])
            logger.info(
                "Re-ranked %d Lucene candidates → top %d (best_sim=%.3f, "
                "worst_kept_sim=%.3f)",
                len(rows), min(_RERANK_KEEP_TOP_N, len(scored)),
                scored[0][0] if scored else 0.0,
                scored[min(_RERANK_KEEP_TOP_N, len(scored)) - 1][0]
                    if scored else 0.0,
            )
            return [r for _, r in scored[:_RERANK_KEEP_TOP_N]]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Semantic rerank failed (%s) — using Lucene order", exc)
            return rows

    async def retrieve_versions(self, parsed: ParsedQuery) -> RetrievalResult:
        """Handle VERSION_LISTING queries by fetching versions from the graph.

        If a specific document is requested, returns only its versions.
        Otherwise returns all documents with all their versions.

        Args:
            parsed: The structured parsed query.

        Returns:
            RetrievalResult containing version information.
        """
        logger.debug("Retrieving versions for document=%s", parsed.document)

        if parsed.document:
            rows = await self._graph.get_all_versions(parsed.document)
            versions = [
                VersionInfo(
                    number=row["number"],
                    sort_key=row["sort_key"],
                    document=parsed.document,
                    doc_type=DocumentType(row["doc_type"]),
                )
                for row in rows
            ]
        elif parsed.category:
            # Fetch versions for a category, deduplicating across documents
            # (e.g. "node.js assert module" and "node.js errors" share versions)
            rows = await self._graph.get_all_documents_with_versions()
            seen: set[str] = set()
            versions = []
            for row in rows:
                if row["category"].lower() != parsed.category.lower():
                    continue
                if row["version"] not in seen:
                    seen.add(row["version"])
                    versions.append(
                        VersionInfo(
                            number=row["version"],
                            sort_key=row["sort_key"],
                            document=row["document"],
                            doc_type=DocumentType(row["doc_type"]),
                        )
                    )
            versions.sort(key=lambda v: v.sort_key)
        else:
            # Fetch all documents with all versions
            rows = await self._graph.get_all_documents_with_versions()
            versions = [
                VersionInfo(
                    number=row["version"],
                    sort_key=row["sort_key"],
                    document=row["document"],
                    doc_type=DocumentType(row["doc_type"]),
                )
                for row in rows
            ]

        logger.info("Retrieved %d versions (document=%s)", len(versions), parsed.document)

        return RetrievalResult(
            versions=versions,
            chunks=[],
            changes=[],
            retrieval_mode=RetrievalMode.GRAPH,
        )

    async def retrieve_changes(self, parsed: ParsedQuery) -> RetrievalResult:
        """Handle CHANGE_RETRIEVAL_EXPLICIT and CHANGE_RETRIEVAL_IMPLICIT queries.

        Routing priority (most specific → most general). Whichever case fires
        first wins; the global fulltext fallback only runs when no document
        and no category were extracted from the query.

        1. doc + version_from + version_to → exact-pair lookup.
        2. doc + version_to (or bare version) → all changes ending in that version.
        3. doc + version_from → all changes leaving that version.
        4. doc only → fulltext ranked within that document.
        5. category only → fulltext ranked within that category.
        6. nothing → global fulltext (last resort).

        A single ``parsed.version`` is treated as ``version_to`` because
        "what was added in vX?" semantically targets the destination version.

        Args:
            parsed: The structured parsed query.

        Returns:
            RetrievalResult containing change information.
        """
        # A bare `version` means "what changed IN this version" → version_to.
        effective_to = parsed.version_to or parsed.version

        # Multi-part document expansion (legal corpus): if the parser populated
        # ``documents`` (a list, set when one Act maps to many indexed Parts),
        # union the per-doc results. Single-doc software queries hit the
        # ``effective_docs == [parsed.document]`` path → identical behaviour.
        effective_docs: list[str] = (
            list(parsed.documents) if parsed.documents
            else ([parsed.document] if parsed.document else [])
        )

        logger.debug(
            "Retrieving changes: docs=%s, from=%s, to=%s, category=%s",
            effective_docs or "<none>", parsed.version_from, effective_to,
            parsed.category,
        )

        if effective_docs and parsed.version_from and effective_to:
            rows = []
            for d in effective_docs:
                rows.extend(await self._graph.get_changes_between(
                    d, parsed.version_from, effective_to
                ))
            path = "between"
        elif effective_docs and effective_to:
            rows = []
            for d in effective_docs:
                rows.extend(await self._graph.get_changes_into(d, effective_to))
            path = "into"
        elif effective_docs and parsed.version_from:
            rows = []
            for d in effective_docs:
                rows.extend(await self._graph.get_changes_from(
                    d, parsed.version_from
                ))
            path = "from"
        elif effective_docs:
            rows = []
            for d in effective_docs:
                rows.extend(await self._graph.search_changes(
                    parsed.question, document=d
                ))
            rows = await self._semantic_rerank(parsed.question, rows)
            path = "fulltext_doc"
        elif parsed.category:
            rows = await self._graph.search_changes(
                parsed.question, category=parsed.category
            )
            rows = await self._semantic_rerank(parsed.question, rows)
            path = "fulltext_category"
        else:
            rows = await self._graph.search_changes(parsed.question)
            rows = await self._semantic_rerank(parsed.question, rows)
            path = "fulltext_global"

        # NOTE: section-based re-ranking was experimented with (May 2026) but
        # consistently regressed Change(i) by 12-15 points without compensating
        # gain elsewhere. Cause: even as a rerank-only signal (cap 15, no
        # drops), it (a) confused the generator with mixed match/non-match
        # ordering and (b) Mistral's pin-cite extraction was imprecise enough
        # that "matches" sometimes pointed at the wrong subsection. Removed
        # in favour of the existing semantic rerank that was already proven
        # for the software corpus. ``ParsedQuery.section`` remains in the
        # schema as a no-op so the parser prompt that extracts it doesn't
        # break — and downstream consumers can opt in if a smarter ranking
        # strategy is invented later.

        changes = [
            Change(
                name=row["name"],
                description=row["description"],
                change_type=ChangeType(row["change_type"]),
                version_from=row.get("version_from") or "",
                version_to=row.get("version_to") or "",
                source=row.get("source") or "",
            )
            for row in rows
        ]

        logger.info(
            "Retrieved %d changes via %s (document=%s, from=%s, to=%s, category=%s)",
            len(changes), path, parsed.document, parsed.version_from,
            effective_to, parsed.category,
        )

        return RetrievalResult(
            changes=changes,
            chunks=[],
            versions=[],
            retrieval_mode=RetrievalMode.GRAPH,
        )
