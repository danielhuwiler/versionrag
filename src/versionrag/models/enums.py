"""Domain enumerations for VersionRAG."""

from enum import StrEnum


class QueryType(StrEnum):
    """Classifies the intent of a user query."""

    CONTENT_RETRIEVAL = "content_retrieval"
    CONTENT_VERSION_SPECIFIC = "content_version_specific"
    VERSION_LISTING = "version_listing"
    CHANGE_RETRIEVAL_EXPLICIT = "change_retrieval_explicit"
    CHANGE_RETRIEVAL_IMPLICIT = "change_retrieval_implicit"


class RetrievalMode(StrEnum):
    """Determines which retrieval strategy is applied."""

    GRAPH = "graph"
    VECTOR = "vector"
    HYBRID = "hybrid"


class DocumentType(StrEnum):
    """Distinguishes between document categories ingested into the system."""

    CHANGELOG = "changelog"
    DOCUMENTATION = "documentation"


class ChangeType(StrEnum):
    """Whether a change was stated explicitly or inferred from content diffs."""

    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
