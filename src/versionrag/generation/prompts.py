"""Prompt templates for answer generation.

Contains system prompt constants for each query type and a builder function
that assembles the final (system_prompt, user_prompt) pair from retrieval results.
"""

import logging
from versionrag.models.enums import QueryType
from versionrag.models.schemas import ParsedQuery, RetrievalResult

logger = logging.getLogger(__name__)

# ── System prompts per query category ─────────────────────────────────────────

CONTENT_SYSTEM_PROMPT = (
    "You are a technical documentation assistant. Answer the question based on "
    "the provided context. Be precise and concise. If the context contains "
    "relevant information, use it to formulate your answer even if it doesn't "
    "perfectly match the question. Only say 'I don't have enough information' "
    "if the context is completely unrelated to the question. "
    "Do not make up information."
)

VERSION_LISTING_SYSTEM_PROMPT = (
    "You are a version information assistant. Based on the provided version data, "
    "answer the question accurately. List versions in order when asked. Give exact "
    "counts when asked. If asked whether a version exists, answer with a clear Yes or No."
)

CHANGE_RETRIEVAL_SYSTEM_PROMPT = (
    "You are a change tracking assistant. Based on the provided change records, "
    "answer the question accurately.\n\n"
    "Each change record has a version_from and version_to indicating the version "
    "transition where the change occurred. The change_type field tells you the "
    "direction (added / removed / modified / deprecated). Use these rules:\n\n"
    "  • 'When was X added/introduced?' → find the record where X was added; "
    "    answer with its version_to (the version where X first appears).\n"
    "  • 'When was X removed/deleted?' → find the record where X was removed; "
    "    answer with version_to (the version where X is no longer present).\n"
    "  • 'When did X change to Y?' → find the record showing the X→Y transition; "
    "    answer with version_to.\n"
    "  • 'In which version was the typo corrected / wording fixed?' → find the "
    "    record that describes the correction; version_to is the corrected version.\n\n"
    "IMPORTANT — wording mismatches between question and records:\n"
    "  • Records may use slightly different words than the question. A record "
    "    saying 'Removed X' is a valid answer to 'In which version did X disappear?' "
    "    or 'When did X get cleaned up?' even though the question doesn't use the "
    "    word 'removed'. Connect the semantics, not just the words.\n"
    "  • A record saying 'X added' is the answer to 'In which version did Y begin "
    "    using X?' if X and Y are clearly the same feature in different phrasings.\n"
    "  • Do NOT respond with 'no information' just because the change_type word "
    "    in the records (e.g. 'modified') differs from the verb in the question "
    "    (e.g. 'updated', 'changed'). Treat add/introduce/new/first as one family, "
    "    and remove/delete/drop/clean-up as another, and modify/update/change/correct "
    "    as a third family.\n\n"
    "If multiple records mention the same feature, prefer the one whose name or "
    "description most specifically matches the question (highest token overlap with "
    "the question's key terms). If two records describe the same version "
    "transition for the same feature, the one whose description is more specific "
    "wins.\n\n"
    "Reference the EXACT version number in your final sentence "
    "(e.g. '1.32', '2.6', '11.x', 'jazzy'). Be precise and concise.\n\n"
    "REPORT ALL DETAILS — when the question asks 'What … was fixed/added/changed/"
    "introduced/removed in version X?' (i.e. the question takes the version as "
    "given and asks WHAT the change was), reproduce every specific detail from the "
    "matching record's description: CVE identifiers, function / class / parameter "
    "names, error types, version numbers of dependencies, file paths, issue or PR "
    "numbers, exact wording of error messages, and any concrete values mentioned. "
    "Do NOT collapse the answer to a single sentence like 'X was fixed in vY' when "
    "the underlying record description contains specifics — that omits information "
    "the user asked for. Quote identifiers verbatim (e.g. 'CVE-2021-41216', "
    "'np.strings.slice', 'GH 41935').\n\n"
    "Only say 'no matching changes' if NONE of the provided records talks about "
    "the feature in the question — and even then, briefly mention what related "
    "changes ARE present."
)

# Maps query types to the appropriate system prompt
_SYSTEM_PROMPTS: dict[QueryType, str] = {
    QueryType.CONTENT_RETRIEVAL: CONTENT_SYSTEM_PROMPT,
    QueryType.CONTENT_VERSION_SPECIFIC: CONTENT_SYSTEM_PROMPT,
    QueryType.VERSION_LISTING: VERSION_LISTING_SYSTEM_PROMPT,
    QueryType.CHANGE_RETRIEVAL_EXPLICIT: CHANGE_RETRIEVAL_SYSTEM_PROMPT,
    QueryType.CHANGE_RETRIEVAL_IMPLICIT: CHANGE_RETRIEVAL_SYSTEM_PROMPT,
}


def build_generation_prompt(
    parsed: ParsedQuery, result: RetrievalResult
) -> tuple[str, str]:
    """Build the system and user prompts for answer generation.

    Selects the system prompt based on query type, then assembles a user prompt
    that includes the original question and all retrieved context (chunks,
    changes, or versions) formatted for the LLM.

    Args:
        parsed: The structured parsed query with the original question.
        result: The retrieval result containing chunks, changes, and/or versions.

    Returns:
        A (system_prompt, user_prompt) tuple ready for the LLM chat call.
    """
    # Select system prompt based on query type
    system_prompt = _SYSTEM_PROMPTS.get(parsed.query_type, CONTENT_SYSTEM_PROMPT)

    # Build context sections from retrieval results
    context_parts: list[str] = []

    # Format retrieved chunks (content retrieval)
    for chunk in result.chunks:
        context_parts.append(
            f"--- Document: {chunk.document}, Version: {chunk.version} ---\n"
            f"{chunk.content}"
        )

    # Format retrieved changes
    for change in result.changes:
        context_parts.append(
            f"- [{change.change_type}] {change.version_from} → {change.version_to}: "
            f"{change.name} — {change.description}"
        )

    # Format retrieved versions
    for version in result.versions:
        context_parts.append(
            f"- {version.document}: v{version.number} ({version.doc_type})"
        )

    # Fall back to a clear "no context" message
    if not context_parts:
        context_block = "No relevant context was found."
    else:
        context_block = "\n\n".join(context_parts)

    user_prompt = (
        f"Context:\n{context_block}\n\n"
        f"Question: {parsed.question}"
    )

    logger.debug(
        "Built generation prompt: system_prompt_len=%d, user_prompt_len=%d, "
        "context_parts=%d",
        len(system_prompt),
        len(user_prompt),
        len(context_parts),
    )

    return system_prompt, user_prompt
