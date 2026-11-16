"""Graph construction layer — writes domain entities into Neo4j."""

import logging
from typing import Any

from versionrag.clients.neo4j import Neo4jClient

logger = logging.getLogger(__name__)

# Lucene QueryParser reserved characters that must be backslash-escaped
# when used inside a user-supplied search term passed to
# db.index.fulltext.queryNodes. Source: Lucene docs.
_LUCENE_RESERVED = set(r'+-&|!(){}[]^"~*?:\/')


def _escape_lucene(text: str) -> str:
    """Backslash-escape Lucene QueryParser reserved characters."""
    return "".join("\\" + c if c in _LUCENE_RESERVED else c for c in text)


class GraphBuilder:
    """Builds and queries the version graph in Neo4j.

    All write operations use MERGE where possible so they are idempotent
    and safe to run multiple times. Changes are always CREATEd because each
    detected change is a unique event.
    """

    def __init__(self, neo4j: Neo4jClient) -> None:
        self._neo4j = neo4j

    # ── Write operations ──────────────────────────────────────────────────────

    async def create_category(self, name: str) -> None:
        """Merge a Category node into the graph.

        Args:
            name: Unique category name (e.g. "Node.js").
        """
        await self._neo4j.execute_query(
            "MERGE (:Category {name: $name})",
            {"name": name},
        )
        logger.debug("Created/merged category: %s", name)

    async def create_document(self, name: str, category: str) -> None:
        """Merge a Document node and link it to its Category.

        The MERGE pattern includes ``category`` so the (name, category) UNIQUE
        constraint actually enforces. Without setting the property, two
        Documents with the same name but different categories could be created
        (LLM drift produces cross-category orphans, e.g. 'scipy' under both
        'scipy' and 'numpy'), which then trip the (number, document) UNIQUE
        constraint inside create_version and cause Phase A ingest failures.

        Args:
            name: Document identifier (e.g. "assert").
            category: Name of the parent Category node.
        """
        await self._neo4j.execute_query(
            """
            MERGE (doc:Document {name: $name, category: $category})
            WITH doc
            MATCH (cat:Category {name: $category})
            MERGE (cat)-[:HAS_DOC]->(doc)
            """,
            {"name": name, "category": category},
        )
        logger.debug("Created/merged document: %s in %s", name, category)

    async def create_version(
        self,
        document: str,
        number: str,
        sort_key: int,
        doc_type: str,
        category: str,
        file_path: str = "",
    ) -> None:
        """Merge a Version node and link it to its Document.

        Matching the Document on (name, category) — not just name — keeps
        versions confined to the correct Document when cross-category orphans
        exist and prevents the MATCH from fanning out into multiple Documents.

        Args:
            document: Parent Document name.
            number: Version string (e.g. "15.14.0").
            sort_key: Numeric sort key from ``version_to_sort_key()``.
            doc_type: Document type string (e.g. "documentation").
            category: Parent Document's category (disambiguates same-name docs).
            file_path: Path to the source file on disk.
        """
        await self._neo4j.execute_query(
            """
            MATCH (doc:Document {name: $document, category: $category})
            MERGE (doc)-[:HAS_VERSION]->(v:Version {number: $number, document: $document})
            SET v.sort_key = $sort_key, v.doc_type = $doc_type, v.file_path = $file_path
            """,
            {
                "document": document,
                "category": category,
                "number": number,
                "sort_key": sort_key,
                "doc_type": doc_type,
                "file_path": file_path,
            },
        )
        logger.debug("Created/merged version: %s %s", document, number)

    async def link_versions(self, document: str) -> None:
        """Create NEXT edges between consecutive versions of a document.

        Versions are ordered by sort_key. Existing NEXT edges are merged
        so repeated calls are safe.

        Args:
            document: Document name whose versions should be linked.
        """
        await self._neo4j.execute_query(
            """
            MATCH (doc:Document {name: $document})-[:HAS_VERSION]->(v:Version)
            WITH v ORDER BY v.sort_key
            WITH collect(v) AS versions
            UNWIND range(0, size(versions)-2) AS i
            WITH versions[i] AS v1, versions[i+1] AS v2
            MERGE (v1)-[:NEXT]->(v2)
            """,
            {"document": document},
        )
        logger.debug("Linked versions for document: %s", document)

    async def create_change(
        self,
        version_from: str,
        version_to: str,
        document: str,
        name: str,
        description: str,
        change_type: str,
        source: str,
    ) -> None:
        """Create a Change node and connect it to two Version nodes.

        Uses CREATE (not MERGE) because each change event is unique.

        Args:
            version_from: Source version number.
            version_to: Target version number.
            document: Document both versions belong to.
            name: Short change identifier.
            description: Human-readable change description.
            change_type: "explicit" or "implicit".
            source: Origin of the change (e.g. "deepdiff", "changelog").
        """
        await self._neo4j.execute_query(
            """
            MATCH (v1:Version {number: $version_from, document: $document})
            MATCH (v2:Version {number: $version_to, document: $document})
            CREATE (c:Change {
                name: $name,
                description: $description,
                change_type: $change_type,
                source: $source
            })
            CREATE (v1)-[:CHANGED_TO]->(c)
            CREATE (v2)-[:CHANGED_FROM]->(c)
            """,
            {
                "version_from": version_from,
                "version_to": version_to,
                "document": document,
                "name": name,
                "description": description,
                "change_type": change_type,
                "source": source,
            },
        )
        logger.debug(
            "Created change: %s (%s -> %s)", name, version_from, version_to
        )

    # ── Bulk operations ─────────────────────────────────────────────────────────

    async def link_all_versions(self) -> int:
        """Create NEXT edges for every document in the graph.

        Returns:
            Number of documents whose versions were linked.
        """
        logger.debug("Querying all documents for version linking")
        docs = await self._neo4j.execute_query(
            "MATCH (d:Document) RETURN d.name AS name"
        )
        logger.debug("Found %d documents to link", len(docs))
        for row in docs:
            await self.link_versions(row["name"])
        logger.info("Linked NEXT edges for %d documents", len(docs))
        return len(docs)

    async def get_all_documents_with_versions(self) -> list[dict[str, Any]]:
        """Return all documents with their versions from the graph.

        Returns:
            List of dicts with keys ``category``, ``document``, ``version``,
            ``sort_key``, ``doc_type``, ``file_path``.
        """
        return await self._neo4j.execute_query(
            """
            MATCH (cat:Category)-[:HAS_DOC]->(doc:Document)-[:HAS_VERSION]->(v:Version)
            RETURN cat.name AS category, doc.name AS document,
                   v.number AS version, v.sort_key AS sort_key,
                   v.doc_type AS doc_type, v.file_path AS file_path
            ORDER BY cat.name, doc.name, v.sort_key
            """
        )

    # ── Read operations ───────────────────────────────────────────────────────

    async def get_all_versions(self, document: str) -> list[dict[str, Any]]:
        """Return all versions of a document ordered by sort_key.

        Args:
            document: Document name to query.

        Returns:
            List of dicts with keys ``number``, ``sort_key``, ``doc_type``.
        """
        return await self._neo4j.execute_query(
            """
            MATCH (doc:Document {name: $document})-[:HAS_VERSION]->(v:Version)
            RETURN v.number AS number, v.sort_key AS sort_key, v.doc_type AS doc_type
            ORDER BY v.sort_key
            """,
            {"document": document},
        )

    async def get_changes_between(
        self,
        document: str,
        version_from: str,
        version_to: str,
    ) -> list[dict[str, Any]]:
        """Return all Change nodes between two versions of a document.

        Args:
            document: Document name.
            version_from: Source version number.
            version_to: Target version number.

        Returns:
            List of dicts with keys ``name``, ``description``,
            ``change_type``, ``source``, ``version_from``, ``version_to``.
        """
        return await self._neo4j.execute_query(
            """
            MATCH (v1:Version {number: $version_from, document: $document})
                  -[:CHANGED_TO]->(c:Change)
                  <-[:CHANGED_FROM]-(v2:Version {number: $version_to, document: $document})
            RETURN c.name AS name, c.description AS description,
                   c.change_type AS change_type, c.source AS source,
                   v1.number AS version_from, v2.number AS version_to
            """,
            {
                "document": document,
                "version_from": version_from,
                "version_to": version_to,
            },
        )

    async def get_changes_into(
        self,
        document: str,
        version_to: str,
    ) -> list[dict[str, Any]]:
        """Return all Change nodes whose target version is ``version_to``.

        Used for single-version "what was added/fixed in vX?" queries — the
        Change edges already encode both endpoints, so no fulltext fallback
        is needed when the document and target version are known.
        """
        return await self._neo4j.execute_query(
            """
            MATCH (v1:Version {document: $document})
                  -[:CHANGED_TO]->(c:Change)
                  <-[:CHANGED_FROM]-(v2:Version {number: $version_to, document: $document})
            RETURN c.name AS name, c.description AS description,
                   c.change_type AS change_type, c.source AS source,
                   v1.number AS version_from, v2.number AS version_to
            ORDER BY v1.sort_key DESC
            LIMIT 50
            """,
            {"document": document, "version_to": version_to},
        )

    async def get_changes_from(
        self,
        document: str,
        version_from: str,
    ) -> list[dict[str, Any]]:
        """Return all Change nodes whose source version is ``version_from``.

        Mirror of :meth:`get_changes_into` for "what was removed/changed in
        vX?" or "what came after vX?" style queries.
        """
        return await self._neo4j.execute_query(
            """
            MATCH (v1:Version {number: $version_from, document: $document})
                  -[:CHANGED_TO]->(c:Change)
                  <-[:CHANGED_FROM]-(v2:Version {document: $document})
            RETURN c.name AS name, c.description AS description,
                   c.change_type AS change_type, c.source AS source,
                   v1.number AS version_from, v2.number AS version_to
            ORDER BY v2.sort_key
            LIMIT 50
            """,
            {"document": document, "version_from": version_from},
        )

    async def search_changes(
        self,
        search_term: str,
        document: str | None = None,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """Full-text search across Change node names and descriptions.

        Uses the ``change_fulltext_idx`` index created during schema setup.
        Optional ``document`` / ``category`` filters narrow the result set —
        critical when the corpus contains tens of thousands of Change nodes
        and the global top-20 is dominated by high-volume documents.

        Args:
            search_term: Lucene-compatible search term.
            document: If set, only return Changes whose Version.document matches.
            category: If set, only return Changes whose parent Category matches.

        Returns:
            Up to 20 results ordered by relevance score, with keys
            ``name``, ``description``, ``change_type``, ``source``,
            ``version_from``, ``version_to``, ``document``, ``score``.
        """
        return await self._neo4j.execute_query(
            """
            CALL db.index.fulltext.queryNodes("change_fulltext_idx", $search_term)
            YIELD node, score
            MATCH (v1)-[:CHANGED_TO]->(node)<-[:CHANGED_FROM]-(v2)
            MATCH (cat:Category)-[:HAS_DOC]->(d:Document {name: v1.document})
            WHERE ($document IS NULL OR v1.document = $document)
              AND ($category IS NULL OR cat.name = $category)
            RETURN node.name AS name, node.description AS description,
                   node.change_type AS change_type, node.source AS source,
                   v1.number AS version_from, v2.number AS version_to,
                   v1.document AS document, v1.sort_key AS sort_key_from,
                   score
            ORDER BY score DESC
            LIMIT 40
            """,
            {
                "search_term": _escape_lucene(search_term),
                "document": document,
                "category": category,
            },
        )

    async def count_graph(self) -> dict[str, Any]:
        """Return node and relationship counts grouped by label / type.

        Returns:
            Dict with keys ``nodes`` (label → count), ``relationships``
            (type → count), ``total_nodes``, ``total_relationships``.
        """
        node_rows = await self._neo4j.execute_query(
            "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count"
        )
        rel_rows = await self._neo4j.execute_query(
            "MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count"
        )

        nodes = {row["label"]: row["count"] for row in node_rows if row["label"]}
        relationships = {row["type"]: row["count"] for row in rel_rows}

        return {
            "nodes": nodes,
            "relationships": relationships,
            "total_nodes": sum(nodes.values()),
            "total_relationships": sum(relationships.values()),
        }
