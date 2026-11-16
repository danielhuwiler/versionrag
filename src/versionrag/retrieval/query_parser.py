"""LLM-based query parser that classifies user questions and extracts parameters."""

import logging
import os
from collections import defaultdict
from pathlib import Path

from versionrag.clients.llm import LLMClient
from versionrag.clients.neo4j import Neo4jClient
from versionrag.config import settings
from versionrag.models.schemas import ParsedQuery
from versionrag.utils import normalize_version

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_TEMPLATE = """\
You are a query classifier for a version-aware document QA system.
Given a user question, classify it and extract relevant parameters.

{document_catalog}

The "document" field MUST match a document name from the catalog above
exactly (case-sensitive). If the question does not clearly map to a
specific document, set document to null. The "category" field must match
a category name from the catalog above, or null.

Classification rules — apply IN ORDER, first match wins:

1. version_listing — the question is ABOUT THE SET OF VERSIONS itself:
     "What is the latest X version?"
     "How many versions of X exist?"
     "Is version 2.0 available?"
     "List all versions of X."

2. change_retrieval_implicit — the question asks WHEN something happened.
   The version is the ANSWER, not a constraint. Strong signals:
     "In which version was X added/removed/introduced/deprecated/changed?"
     "When did Y start raising TypeError?"
     "Which version first supported Z?"
     "In which version did the documentation change to reference W?"
     "When was the typo X corrected?"
     "In what release did Q become the default?"
   This class is the right one whenever the user is asking which version
   is responsible for a documented change — even when they mention a
   library, module, method, or specific identifier (e.g. "Go sync.Map.Clear").
   The MENTION of a function/class/option is NOT what makes a question
   version-specific; the question is version-specific only when the user
   pins a version and asks ABOUT that version's behaviour or content.

3. change_retrieval_explicit — the question is about a changelog/release-
   notes entry by name. Signals: "What does the changelog say about X?",
   "What was fixed in release Y?", "List changes in version Z".

4. content_version_specific — the question pins a SPECIFIC version and
   asks what content/behaviour/options/parameters EXIST in that version.
   The version is a CONSTRAINT, not the answer:
     "What does requestOCSP do in Node.js version 5.x?"
     "What is the default timeout in Kubernetes 1.32?"
     "In TensorRT-LLM v1.2.0rc7, what quantization recipes are listed?"

5. content_retrieval — everything else (general "what is X", "how do I do Y",
   "what does Z mean", "explain why W happens").

Extraction rules:
- Always extract document name (matching catalog) and any version mentioned.
- Version numbers must NOT have a 'v' or 'go' prefix
  (use "5.3.1" not "v5.3.1", "1.23.0" not "go1.23.0").
- ROS 2 codenames are lowercase ("humble", "iron", "jazzy", "kilted").
- All string values must be lowercase.
- Digit-only version slugs that come from filenames must be expanded to the
  dotted form used in the corpus payloads. Specifically:
    "pandas version 15" / "pandas 15" → "1.5"
    "pandas version 20" / "pandas 20" → "2.0"
    "pandas version 21" / "pandas 21" → "2.1"
    "pandas version 22" / "pandas 22" → "2.2"
    "pandas version 30" / "pandas 30" → "3.0"
  More generally, when a version is written as 2 digits XY without a dot for a
  project whose catalog versions use major.minor (pandas, pytorch, kubernetes,
  scipy, numpy, tensorflow, etc.), interpret it as "X.Y". Do NOT apply this
  rule to docker / postgresql / mysql / wireshark / java_jdk where the major
  version itself is a two-digit number.
- For change_retrieval_implicit do NOT lock onto the version the answer
  is supposed to be — leave version_from/version_to null when the user is
  asking which version added/removed something."""


def _load_parser_prompt_template() -> str:
    """Return the parser system-prompt template — env-var override or default.

    Setting ``VERSIONRAG_PARSER_PROMPT`` to a readable file path swaps in a
    corpus-specific parser prompt (e.g. UK legal Acts with ISO-date version
    extraction rules) without touching this module. The file MUST contain
    the literal ``{document_catalog}`` placeholder; it is filled with the
    rendered catalog at QueryParser construction time.

    Unset / empty / missing-file → the default software-corpus template
    declared above is used, preserving bit-identical behaviour for the
    SemVer corpus pipeline.
    """
    path = os.environ.get("VERSIONRAG_PARSER_PROMPT")
    if path and Path(path).is_file():
        template = Path(path).read_text(encoding="utf-8")
        if "{document_catalog}" not in template:
            logger.warning(
                "VERSIONRAG_PARSER_PROMPT file %s does not contain "
                "{document_catalog} placeholder — catalog will not be "
                "injected into the parser system prompt.", path,
            )
        return template
    return _SYSTEM_PROMPT_TEMPLATE


async def build_document_catalog(neo4j: Neo4jClient) -> str:
    """Render a textual document inventory from the current Neo4j graph.

    Output groups documents by category and lists each document's doc_type
    and version count. The result is injected into the QueryParser system
    prompt so the LLM knows the exact set of (category, document) values
    it may emit.
    """
    rows = await neo4j.execute_query(
        """
        MATCH (cat:Category)-[:HAS_DOC]->(doc:Document)-[:HAS_VERSION]->(v:Version)
        WITH cat.name AS category, doc.name AS document, v.doc_type AS doc_type,
             count(v) AS n
        RETURN category, document, doc_type, n
        ORDER BY category, document
        """
    )
    if not rows:
        return "The system currently has no indexed documents."

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    total_docs = sum(len(d) for d in by_cat.values())
    lines = [
        f"The system contains {total_docs} documents across "
        f"{len(by_cat)} categories:",
        "",
    ]
    for cat, docs in by_cat.items():
        lines.append(f'Category "{cat}":')
        for d in docs:
            lines.append(
                f'  - "{d["document"]}" ({d["doc_type"]}, {d["n"]} versions)'
            )
        lines.append("")
    return "\n".join(lines)


class QueryParser:
    """Classifies user queries and extracts structured parameters via LLM.

    The system prompt is built per instance from a document catalog (a
    rendered list of (category, document) pairs currently in the graph)
    so the parser stays in sync with whatever corpus is indexed.
    """

    def __init__(
        self,
        llm: LLMClient,
        document_catalog: str,
        document_names: list[str] | None = None,
    ) -> None:
        self._llm = llm
        template = _load_parser_prompt_template()
        try:
            self._system_prompt = template.format(document_catalog=document_catalog)
        except (KeyError, IndexError):
            # Override file used { } for non-placeholder text — fall back to
            # injecting the catalog verbatim at the top.
            self._system_prompt = f"{document_catalog}\n\n{template}"
        # Canonical document names from the live graph. Used for the
        # multi_part_documents prefix-expansion path; otherwise unused.
        # Default empty so behaviour matches the pre-prefix-feature build.
        self._document_names: list[str] = list(document_names or [])

    async def parse(self, question: str) -> ParsedQuery:
        """Parse a natural-language question into a structured query representation."""
        logger.debug("Parsing query: %s", question)

        result = await self._llm.extract_structured(
            response_model=ParsedQuery,
            prompt=question,
            system_prompt=self._system_prompt,
            model=settings.llm_model_fast,
            purpose="evaluation.query_parsing",
        )

        result.question = question

        # === Version normalisation (always-on, software-safe) ===
        # Normalises spelled-out dates ("1 April 2009" → "2009-04-01")
        # AND existing SemVer surface forms ("v5.3.1" → "5.3.1"). For
        # software-corpus queries the date branches never fire, so the
        # output is bit-identical to the pre-feature behaviour.
        if result.version:
            result.version = normalize_version(result.version)
        if result.version_from:
            result.version_from = normalize_version(result.version_from)
        if result.version_to:
            result.version_to = normalize_version(result.version_to)

        # === Multi-part document prefix expansion (gated) ===
        # Legal corpus stores each Act/Part as its own Document node
        # (e.g. ``finance_act_2003_part_1``), but human queries refer to
        # the Act ("Finance Act 2003"). When the parsed document doesn't
        # match the catalog exactly, expand it to the list of catalog
        # entries whose name starts with the LLM-extracted prefix.
        # Software corpus is unaffected because:
        #   1. ``multi_part_documents`` defaults to False
        #   2. Software doc names are single-token canonical slugs
        #      (e.g. ``nodejs_assert``) — a successful LLM extraction
        #      always hits an exact catalog entry anyway
        if (
            settings.multi_part_documents
            and result.document
            and self._document_names
            and result.document not in self._document_names
        ):
            prefix = result.document.lower()
            matches = [n for n in self._document_names if n.startswith(prefix)]
            if len(matches) == 1:
                logger.info(
                    "Prefix expansion: %r → %r",
                    result.document, matches[0],
                )
                result.document = matches[0]
            elif matches:
                logger.info(
                    "Prefix expansion: %r → %d documents (%s…)",
                    result.document, len(matches), matches[0],
                )
                result.documents = matches
                # Clear single-doc field so downstream filters use the list.
                result.document = None

        logger.info(
            "Parsed query: type=%s, doc=%s, docs=%s, version=%s, category=%s",
            result.query_type,
            result.document,
            len(result.documents) if result.documents else None,
            result.version,
            result.category,
        )
        return result
