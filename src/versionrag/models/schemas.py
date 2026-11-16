"""Pydantic data models for VersionRAG."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

from versionrag.models.enums import ChangeType, DocumentType, QueryType, RetrievalMode

# Closed set of software categories the LLM is allowed to return for
# category_hint. Encoded as a Literal type so `instructor` translates it to a
# JSON-schema enum constraint — OpenAI's structured-output mode enforces this
# at decoding time, making LLM drift physically impossible.
CategoryHint = Literal[
    # Software / SemVer corpus (large_dataset/raw/)
    "docker", "java", "mysql", "numpy", "pandas", "postgresql",
    "scipy", "tensorflow", "wireshark", "golang", "kubernetes",
    "nodejs", "nvidia_tensorrt-llm", "python", "pytorch", "ros2",
    "bootstrap", "spark",
    # UK Legal Acts corpus (large_dataset/legal/raw/). Date-based
    # versioning (YYYY-MM-DD). Tokens are derived from the top-level
    # Act directory name. The legal extract prompt (see
    # large_dataset/legal/extract_prompt.txt, loaded via
    # VERSIONRAG_EXTRACT_PROMPT) restricts the LLM to this subset.
    "data_protection", "equality", "finance", "income_tax", "localism",
    # Fallback when nothing else matches.
    "unknown",
]


class ParsedQuery(BaseModel):
    """Structured representation of a user query after LLM parsing."""

    question: str = Field(description="The original question, unchanged")
    query_type: QueryType = Field(
        description=(
            "Classify the query:\n"
            "- content_retrieval: General questions about document content without specific version\n"
            "- content_version_specific: Questions about content IN a specific version "
            "(e.g. 'What is X in version Y?')\n"
            "- version_listing: Questions about available versions, version counts, version existence\n"
            "- change_retrieval_explicit: Questions about documented changes from changelogs\n"
            "- change_retrieval_implicit: Questions about undocumented changes detected between "
            "versions (e.g. 'When was X added/removed?')"
        )
    )
    document: str | None = Field(
        default=None,
        description=(
            "Document name mentioned in query, e.g. 'assert', 'errors', 'bootstrap', 'spark'. "
            "Lowercase."
        ),
    )
    # Populated by QueryParser AFTER the LLM call when ``multi_part_documents``
    # is enabled and the LLM's ``document`` value matches multiple catalog
    # entries by prefix (legal corpus: "Finance Act 2003" → all parts/schedules
    # of that act). When set, downstream retrievers MUST filter by the union of
    # these document names instead of by ``document``. ``SkipJsonSchema`` keeps
    # this field out of the schema instructor sends to the LLM — the LLM
    # therefore cannot populate it (which would shadow the post-call
    # expansion) and we don't pay tokens for an unused field.
    documents: Annotated[list[str] | None, SkipJsonSchema()] = None
    version: str | None = Field(
        default=None,
        description="Specific version if mentioned, e.g. '15.14.0', '5.3.3'. None if no version specified.",
    )
    version_from: str | None = Field(
        default=None,
        description="Start version for range queries, e.g. 'What changed between X and Y'",
    )
    version_to: str | None = Field(
        default=None,
        description="End version for range queries",
    )
    category: str | None = Field(
        default=None,
        description="Category if identifiable: 'node.js', 'apache spark', 'bootstrap'. Lowercase.",
    )


class Chunk(BaseModel):
    """A single text chunk retrieved from the vector store."""

    content: str
    document: str
    version: str
    category: str
    chunk_index: int | None = None
    score: float | None = None


class Change(BaseModel):
    """A detected change between two document versions."""

    name: str
    description: str
    change_type: ChangeType
    version_from: str
    version_to: str
    source: str


class ExtractedChange(BaseModel):
    """A single change extracted from a changelog or diff by the LLM."""

    name: str = Field(
        description="Short identifier for the change, e.g. 'CallTracker added', "
        "'ERR_ACCESS_DENIED introduced'"
    )
    description: str = Field(
        description="Human-readable description of what changed"
    )


class ExtractedChanges(BaseModel):
    """List of changes extracted from a document or diff."""

    changes: list[ExtractedChange] = Field(
        default_factory=list,
        description="All distinct changes found",
    )


class DocumentMetadata(BaseModel):
    """Metadata describing an ingested document.

    Fields with ``Field(description=...)`` are filled by the LLM via structured
    output. ``file_path`` and ``sort_key`` are set programmatically afterwards.
    """

    title: str = Field(description="Document title, e.g. 'Node.js Assert Module'")
    version: str = Field(description="Version string, e.g. '15.14.0', 'v5.3.1'")
    doc_type: DocumentType = Field(
        description="'changelog' for release notes/what's new, 'documentation' for API refs/guides"
    )
    category_hint: CategoryHint = Field(
        description="Software product the document belongs to. MUST be exactly one of the "
        "enum values — the LLM is decoder-constrained to this list. "
        "Documents about the same software MUST share the same category_hint."
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score for this extraction (0.0–1.0)"
    )
    # Set programmatically after the LLM call — not part of the LLM prompt
    file_path: str = ""
    sort_key: int = 0


class VersionInfo(BaseModel):
    """Lightweight version descriptor stored in the graph."""

    number: str
    sort_key: int
    document: str
    doc_type: DocumentType


class RetrievalResult(BaseModel):
    """Aggregated output from the retrieval layer."""

    chunks: list[Chunk]
    changes: list[Change]
    versions: list[VersionInfo]
    retrieval_mode: RetrievalMode


class Answer(BaseModel):
    """Final answer returned to the user, including source attribution."""

    question: str
    answer: str
    sources: list[dict]
    retrieval_mode: RetrievalMode
    query_type: QueryType


class HealthStatus(BaseModel):
    """System health report returned by the /health endpoint."""

    status: str
    neo4j: str
    qdrant: str
    embedding_model: str
    embedding_dimension: int
    qdrant_collection: str
    neo4j_constraints: list[str]
