"""Groups extracted document metadata into categories and document clusters."""

import logging
from collections import defaultdict

from versionrag.models.schemas import DocumentMetadata

logger = logging.getLogger(__name__)


class DocumentClusterer:
    """Clusters a flat list of DocumentMetadata into a nested category/document structure.

    Grouping is done by string normalisation (lowercase + strip) of ``category_hint``
    and ``title``. This catches trivial LLM variations like "Node.js" vs "nodejs"
    but does NOT merge semantically different names like "Node.js Assert" and
    "Node.js Errors" — those remain separate documents in the same category.
    """

    @staticmethod
    def _normalize(text: str) -> str:
        """Return a normalised key for grouping: lowercase and stripped."""
        return text.strip().lower()

    def cluster(
        self,
        documents: list[DocumentMetadata],
    ) -> dict[str, dict[str, list[DocumentMetadata]]]:
        """Group documents by normalised category and title.

        Step 1: Normalise ``category_hint`` → group into categories.
        Step 2: Within each category, normalise ``title`` → group into documents.
        Versions of the same document appear as multiple entries in the inner list,
        sorted by ``sort_key``.

        Args:
            documents: Flat list of extracted document metadata.

        Returns:
            Nested dict ``{norm_category: {norm_title: [meta, ...]}}``.
        """
        # outer: norm_category → {norm_title → [meta]}
        clusters: dict[str, dict[str, list[DocumentMetadata]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for meta in documents:
            cat_key = self._normalize(meta.category_hint)
            doc_key = self._normalize(meta.title)
            clusters[cat_key][doc_key].append(meta)

        # Sort versions within each document by sort_key
        for cat in clusters.values():
            for versions in cat.values():
                versions.sort(key=lambda m: m.sort_key)

        # Convert defaultdicts to plain dicts for cleaner output
        result = {cat: dict(docs) for cat, docs in clusters.items()}

        total_categories = len(result)
        total_documents = sum(len(docs) for docs in result.values())
        total_versions = sum(
            len(versions) for docs in result.values() for versions in docs.values()
        )
        logger.info(
            "Clustered into %d categories, %d documents, %d total versions",
            total_categories,
            total_documents,
            total_versions,
        )
        return result
