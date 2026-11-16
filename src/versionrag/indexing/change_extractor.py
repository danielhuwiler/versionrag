"""Change extraction between consecutive document versions.

Supports two modes:
- Explicit: LLM extracts changes from changelog/release-note documents.
- Implicit: File-level DeepDiff between consecutive version files,
  serialised to JSON and sent to the LLM for change extraction.

Implicit detection is intentionally NOT chunk-based — chunk-level cosine
filtering was too coarse to catch subtle edits (typo fixes, URL changes,
single-line additions). The current pipeline diffs the raw markdown
files line-by-line via DeepDiff, which surfaces every textual change;
the LLM then filters trivia (formatting, whitespace) in its prompt.

Explicit changes are only extracted for doc_type="changelog".
Implicit changes are extracted for non-changelog pairs only — changelog
files contain explicit change descriptions, so diffing them would
generate redundant Change nodes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from deepdiff import DeepDiff

from versionrag.clients.llm import LLMClient
from versionrag.clients.qdrant import QdrantClientWrapper
from versionrag.config import settings
from versionrag.indexing.diff_filter import chunk_diff, filter_diff_noise
from versionrag.indexing.graph_builder import GraphBuilder
from versionrag.models.schemas import ExtractedChanges
from versionrag.utils import read_file

logger = logging.getLogger(__name__)


# ── Prompts ───────────────────────────────────────────────────────────────────

_EXPLICIT_SYSTEM = (
    "You are a changelog analyst. "
    "Extract every distinct change (bug fix, new feature, deprecation, breaking change) "
    "from the provided release notes. Return a list of changes, each with a short "
    "identifier (name) and a one-sentence description. "
    "Focus on: new features, bug fixes, dependency updates, breaking changes, deprecations. "
    "Ignore asset listings, download links, and contributor acknowledgements."
)

_IMPLICIT_SYSTEM = (
    "You are a documentation diff analyst. You receive the JSON output of a "
    "line-level DeepDiff between two consecutive versions of the same technical "
    "document.\n\n"
    "Your job: extract EVERY meaningful content change as a separate entry.\n\n"
    "INCLUDE:\n"
    "  - New sections, paragraphs, fields, parameters, options, error codes, "
    "function/class/method names\n"
    "  - Removed sections, fields, parameters, deprecated APIs\n"
    "  - Modified values, defaults, types, descriptions, URLs, code links, "
    "commit hashes, branch references\n"
    "  - Typo corrections that change semantics (e.g. 'programatically' → "
    "'programmatically', 'reduces' → 'reduce')\n"
    "  - Substantive text edits (reworded definitions, updated examples)\n\n"
    "IGNORE:\n"
    "  - Pure whitespace / blank-line / indentation changes\n"
    "  - Punctuation-only or capitalization-only changes\n"
    "  - Markdown syntax noise (heading-anchor IDs, table-cell padding)\n"
    "  - Reordering of unchanged lines\n"
    "  - Page numbers, footers, navigation links\n\n"
    "FORMAT each change as:\n"
    "  - name: short specific identifier — include exact function/class/field "
    "names, error codes, version numbers, or feature names. "
    "Examples: 'sm103 added to FP8 block-wise scaling', 'KVCacheManager link "
    "pinned to commit 560ded5', 'programatically typo corrected', "
    "'writable.writableBuffer reference added'\n"
    "  - description: ONE sentence stating exactly what changed, including "
    "specifics (field names, types, defaults, version numbers, URL fragments). "
    "Do not summarise multiple changes into one entry.\n\n"
    "If the diff contains only trivia, return an empty list."
)


def _load_change_prompts() -> tuple[str, str]:
    """Return (explicit_prompt, implicit_prompt) — env-var overrides honoured.

    ``VERSIONRAG_CHANGE_EXPLICIT_PROMPT`` / ``VERSIONRAG_CHANGE_IMPLICIT_PROMPT``
    each accept a readable file path. When set, its contents replace the
    default software-corpus prompt above. Unset / empty / missing-file →
    the defaults are used, preserving bit-identical behaviour for the
    software-corpus pipeline.
    """
    explicit = _EXPLICIT_SYSTEM
    implicit = _IMPLICIT_SYSTEM
    for var, default in (
        ("VERSIONRAG_CHANGE_EXPLICIT_PROMPT", "explicit"),
        ("VERSIONRAG_CHANGE_IMPLICIT_PROMPT", "implicit"),
    ):
        path = os.environ.get(var)
        if path and Path(path).is_file():
            content = Path(path).read_text(encoding="utf-8")
            if default == "explicit":
                explicit = content
            else:
                implicit = content
    return explicit, implicit


_EXPLICIT_SYSTEM, _IMPLICIT_SYSTEM = _load_change_prompts()


def _file_hash(path: Path) -> str:
    """Stable content fingerprint for skip-if-identical comparison."""
    return hashlib.md5(path.read_bytes()).hexdigest()


class ChangeExtractor:
    """Extracts changes between consecutive versions and writes them to the graph.

    For changelog documents: sends the text to the LLM to extract explicit changes.
    For non-changelog document pairs: runs file-level DeepDiff and sends the
    diff JSON to the LLM for structured change extraction.
    """

    def __init__(
        self,
        llm: LLMClient,
        graph_builder: GraphBuilder,
        qdrant: QdrantClientWrapper,
    ) -> None:
        self._llm = llm
        self._gb = graph_builder
        self._qdrant = qdrant  # kept on the signature for API stability

    # ── Explicit changes (changelogs only) ────────────────────────────────────

    async def extract_explicit_changes(
        self,
        text: str,
        version: str,
        document: str,
    ) -> list[dict[str, str]]:
        """Extract changes from a changelog document using the LLM."""
        max_chars = settings.change_max_changelog_chars
        truncated = text[:max_chars]
        if len(text) > max_chars:
            logger.debug(
                "Truncated changelog text from %d to %d chars for %s v%s",
                len(text), max_chars, document, version,
            )

        prompt = (
            f"Changelog for {document} version {version}:\n\n"
            f"{truncated}"
        )

        result: ExtractedChanges = await self._llm.extract_structured(
            response_model=ExtractedChanges,
            prompt=prompt,
            system_prompt=_EXPLICIT_SYSTEM,
            model=settings.llm_model_change,
            purpose="indexing.change_extraction.changelog",
        )

        logger.info(
            "Extracted %d explicit changes from %s v%s",
            len(result.changes), document, version,
        )
        return [{"name": c.name, "description": c.description} for c in result.changes]

    # ── Implicit changes (file-level DeepDiff → LLM) ──────────────────────────

    async def extract_implicit_changes(
        self,
        document: str,
        version_from: str,
        version_to: str,
        file_path_from: str,
        file_path_to: str,
    ) -> list[dict[str, str]]:
        """Detect implicit changes by diffing the two version files directly.

        Strategy:
          1. MD5-hash both files; if identical, no work to do.
          2. Run DeepDiff on the line-split file contents with
             ``ignore_order=True`` so pure reorderings are filtered.
          3. Serialise the diff to JSON and send to the LLM, which filters
             trivia (whitespace, punctuation) and emits a structured list
             of changes.

        Args:
            document: Document name in the graph.
            version_from: Earlier version number.
            version_to: Later version number.
            file_path_from: On-disk path to the v_from markdown file.
            file_path_to: On-disk path to the v_to markdown file.

        Returns:
            List of dicts with ``name`` and ``description`` keys.
        """
        path_from = Path(file_path_from)
        path_to = Path(file_path_to)

        if not path_from.exists() or not path_to.exists():
            logger.warning(
                "Missing file for %s v%s -> v%s (from=%s exists=%s, to=%s exists=%s)",
                document, version_from, version_to,
                path_from, path_from.exists(), path_to, path_to.exists(),
            )
            return []

        if _file_hash(path_from) == _file_hash(path_to):
            logger.debug(
                "Identical files for %s v%s -> v%s, skipping diff",
                document, version_from, version_to,
            )
            return []

        text_from = path_from.read_text(encoding="utf-8", errors="ignore")
        text_to = path_to.read_text(encoding="utf-8", errors="ignore")

        diff = DeepDiff(
            text_from.splitlines(),
            text_to.splitlines(),
            ignore_order=True,
            verbose_level=2,
        )
        if not diff:
            logger.debug(
                "No textual diff for %s v%s -> v%s",
                document, version_from, version_to,
            )
            return []

        raw_diff_dict = diff.to_dict()
        raw_entries = sum(
            len(v) if isinstance(v, dict) else 1 for v in raw_diff_dict.values()
        )
        raw_chars = len(json.dumps(raw_diff_dict, default=str))

        # Drop deterministic noise (URL bumps, anchor IDs, page footers, ...).
        filtered = filter_diff_noise(raw_diff_dict)
        filtered_entries = sum(
            len(v) if isinstance(v, dict) else 1 for v in filtered.values()
        )

        if not filtered:
            logger.info(
                "All %d diff entries (%d chars) for %s v%s -> v%s were noise — "
                "no LLM call",
                raw_entries, raw_chars, document, version_from, version_to,
            )
            return []

        # Chunk at entry boundaries so a single change is never split, then
        # run one LLM call per chunk and union the results.
        chunks = chunk_diff(
            filtered,
            max_entries=settings.change_diff_chunk_max_entries,
            max_chars=settings.change_diff_chunk_max_chars,
        )
        logger.info(
            "Diff for %s v%s -> v%s: %d entries (%d chars) -> "
            "filter %d entries -> %d chunks",
            document, version_from, version_to,
            raw_entries, raw_chars, filtered_entries, len(chunks),
        )

        async def _extract_one(i: int, chunk: dict) -> list[dict[str, str]]:
            chunk_json = json.dumps(chunk, default=str)
            prompt = (
                f"Document: {document}\n"
                f"Version change: {version_from} → {version_to}\n"
                f"Diff chunk: {i}/{len(chunks)}\n\n"
                f"DeepDiff JSON output:\n{chunk_json}"
            )
            result: ExtractedChanges = await self._llm.extract_structured(
                response_model=ExtractedChanges,
                prompt=prompt,
                system_prompt=_IMPLICIT_SYSTEM,
                model=settings.llm_model_change,
                purpose="indexing.change_extraction.file_diff",
            )
            return [
                {"name": c.name, "description": c.description}
                for c in result.changes
            ]

        # Chunks of the same diff run concurrently; LLMClient's semaphore
        # serialises OpenRouter requests to openrouter_max_concurrent.
        per_chunk = await asyncio.gather(
            *(_extract_one(i, chunk) for i, chunk in enumerate(chunks, 1))
        )
        all_changes: list[dict[str, str]] = [c for batch in per_chunk for c in batch]

        # Dedupe by case-folded, whitespace-normalised name.
        seen: set[str] = set()
        deduped: list[dict[str, str]] = []
        for c in all_changes:
            key = " ".join(c["name"].split()).casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(c)

        logger.info(
            "Extracted %d implicit changes between %s v%s -> v%s "
            "(%d raw across %d chunks, %d after dedupe)",
            len(deduped), document, version_from, version_to,
            len(all_changes), len(chunks), len(deduped),
        )
        return deduped

    # ── Main entry point ──────────────────────────────────────────────────────

    async def process_from_graph(self) -> dict[str, Any]:
        """Query Neo4j for all documents and versions, extract changes.

        For each document with multiple versions, iterates consecutive pairs:
        - If doc_type == "changelog": run explicit extraction on the v_to text.
          (No file diff — changelogs already enumerate their own changes.)
        - Else: run file-level DeepDiff between v_from and v_to source files,
          let the LLM extract structured Change nodes from the diff.

        Returns:
            Summary dict with ``total_explicit``, ``total_implicit``,
            ``total_changes`` counts.
        """
        logger.info("Starting change extraction from graph")

        rows = await self._gb.get_all_documents_with_versions()
        logger.debug("Fetched %d version rows from graph", len(rows))

        # Group by document name (preserves sort order from graph query)
        docs: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            docs[row["document"]].append(row)

        logger.info(
            "Processing changes for %d documents (%d total versions)",
            len(docs), len(rows),
        )

        # Flatten all consecutive (v_from, v_to) pairs across documents.
        # Each pair is independent; we run them via asyncio.gather so the
        # LLMClient semaphore (set to openrouter_max_concurrent) controls
        # the actual fan-out for OpenRouter-bound calls. Direct OpenAI
        # calls remain effectively serial via that same semaphore being
        # bypassed (nullcontext) for non-OpenRouter models.
        all_pairs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        for doc_name, versions in docs.items():
            if len(versions) < 2:
                logger.debug(
                    "Skipping %s — only %d version(s), no pairs to compare",
                    doc_name, len(versions),
                )
                continue
            logger.info(
                "Processing %d version pairs for document '%s'",
                len(versions) - 1, doc_name,
            )
            for i in range(len(versions) - 1):
                all_pairs.append((doc_name, versions[i], versions[i + 1]))

        logger.info(
            "Launching %d pair extractions concurrently", len(all_pairs)
        )
        results = await asyncio.gather(
            *(self._process_pair(*p) for p in all_pairs),
            return_exceptions=False,
        )

        total_explicit = sum(e for e, _ in results)
        total_implicit = sum(i for _, i in results)

        logger.info(
            "Change extraction complete: %d explicit, %d implicit, %d total",
            total_explicit, total_implicit, total_explicit + total_implicit,
        )
        return {
            "total_explicit": total_explicit,
            "total_implicit": total_implicit,
            "total_changes": total_explicit + total_implicit,
        }

    async def _process_pair(
        self,
        doc_name: str,
        v_from: dict[str, Any],
        v_to: dict[str, Any],
    ) -> tuple[int, int]:
        """Extract changes for a single consecutive version pair.

        Returns ``(explicit_count, implicit_count)``. Exceptions are caught
        and logged so a single bad pair doesn't crash the whole gather.
        """
        is_changelog = v_to["doc_type"] == "changelog"
        try:
            if is_changelog:
                text_to = read_file(Path(v_to["file_path"]))
                explicit = await self.extract_explicit_changes(
                    text=text_to,
                    version=v_to["version"],
                    document=doc_name,
                )
                for change in explicit:
                    await self._gb.create_change(
                        version_from=v_from["version"],
                        version_to=v_to["version"],
                        document=doc_name,
                        name=change["name"],
                        description=change["description"],
                        change_type="explicit",
                        source="changelog",
                    )
                return len(explicit), 0
            else:
                implicit = await self.extract_implicit_changes(
                    document=doc_name,
                    version_from=v_from["version"],
                    version_to=v_to["version"],
                    file_path_from=v_from["file_path"],
                    file_path_to=v_to["file_path"],
                )
                for change in implicit:
                    await self._gb.create_change(
                        version_from=v_from["version"],
                        version_to=v_to["version"],
                        document=doc_name,
                        name=change["name"],
                        description=change["description"],
                        change_type="implicit",
                        source="file_diff",
                    )
                return 0, len(implicit)
        except Exception as exc:
            logger.error(
                "Failed to extract changes for %s v%s -> v%s: %s",
                doc_name, v_from["version"], v_to["version"], exc,
            )
            return 0, 0
