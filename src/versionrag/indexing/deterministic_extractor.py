"""Deterministic (LLM-free) document metadata extraction.

This is the metadata layer of the "Metadata-Filtered RAG (Deterministic Version
Control)" ablation baseline. It is a drop-in replacement for
``AttributeExtractor``: same async ``extract_metadata(file_path)`` signature and
same ``DocumentMetadata`` return type, but every field is derived from the file
path/filename and the document's ``Version:`` header — no LLM call.

It is selected in ``IndexingPipeline`` when ``settings.index_mode`` is
"metadata_filtered" or "parser". Combined with the pipeline skipping change
extraction, "metadata_filtered" makes zero generative-LLM calls; "parser" keeps
the change layer and drops only the attribute call.

Two corpus layouts are supported:

Software  (large_dataset/raw/**)
    changelog/<stack>_releases/<file>   → changelog, category from stack
    changelog/<stack>_whatsnew/<file>   → changelog, title keeps _whatsnew
    docs/<vendor>/<topic>/<file>        → documentation, category = vendor

Legal     (large_dataset/legal/raw/**)
    <act_name>/<part_or_schedule>/<part_or_schedule>_<YYYY-MM-DD>.md
    → documentation, title = <act_name>_<part_or_schedule>, category from act

Version rule (identical for both layouts, matching the LLM prompt's instruction
to read the version from the body): prefer the body ``Version:`` header line,
fall back to the filename suffix after the last ``_``. The value is then passed
through :func:`versionrag.utils.normalize_version`, exactly as the LLM path does,
so stored versions are bit-identical to a full VersionRAG run.
"""

import logging
import re
from pathlib import Path

from versionrag.models.enums import DocumentType
from versionrag.models.schemas import DocumentMetadata
from versionrag.utils import (
    ALLOWED_CATEGORIES,
    CATEGORY_ALIASES,
    derive_title_from_path,
    normalize_version,
    read_file,
    version_to_sort_key,
)

logger = logging.getLogger(__name__)

# Software changelog subfolder → canonical category. Docs vendor folders are
# already canonical categories, so only the changelog stacks need a map.
_CHANGELOG_FOLDER_TO_CATEGORY: dict[str, str] = {
    "docker_engine_releases": "docker",
    "docker_compose_releases": "docker",
    "java_jdk_releases": "java",
    "tensorflow_releases": "tensorflow",
    "mysql_releases": "mysql",
    "wireshark_releases": "wireshark",
    "scipy_releases": "scipy",
    "postgresql_releases": "postgresql",
    "numpy_releases": "numpy",
    "pandas_whatsnew": "pandas",
}

# UK legal top-level Act directory → canonical category token. Mirrors the
# mapping in large_dataset/legal/extract_prompt.txt.
_ACT_TO_CATEGORY: dict[str, str] = {
    "data_protection_act_2018": "data_protection",
    "equality_act_2010": "equality",
    "finance_act_2003": "finance",
    "income_tax_earnings_pensions_act_2003": "income_tax",
    "localism_act_2011": "localism",
}

# ``Version: <value>`` header line. Searched only in the document head so a
# stray occurrence deeper in the body can't shadow the real version.
_VERSION_LINE_RE = re.compile(r"^Version:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_VERSION_HEAD_CHARS = 1500


def _canonical_category(raw: str) -> str:
    """Normalize + alias + validate a category token (fallback ``unknown``)."""
    norm = raw.strip().lower().replace(" ", "_")
    norm = CATEGORY_ALIASES.get(norm, norm)
    return norm if norm in ALLOWED_CATEGORIES else "unknown"


def _extract_version(file_path: Path, text: str) -> str:
    """Prefer the body ``Version:`` line, else the filename suffix; normalize."""
    match = _VERSION_LINE_RE.search(text[:_VERSION_HEAD_CHARS])
    if match:
        return normalize_version(match.group(1))
    stem = file_path.stem
    suffix = stem.rsplit("_", 1)[-1] if "_" in stem else stem
    return normalize_version(suffix)


class DeterministicAttributeExtractor:
    """LLM-free metadata extractor — drop-in for ``AttributeExtractor``."""

    async def extract_metadata(self, file_path: Path) -> DocumentMetadata:
        """Derive metadata deterministically from the path and ``Version:`` header.

        Args:
            file_path: Path to a ``.md`` (or ``.pdf``) file.

        Returns:
            A fully-populated ``DocumentMetadata`` (confidence fixed at 1.0).
        """
        text = read_file(file_path)
        version = _extract_version(file_path, text)

        if "legal" in file_path.parts:
            # <act_name>/<part_or_schedule>/<file>_<date>.md
            act = file_path.parent.parent.name
            part = file_path.parent.name
            title = f"{act}_{part}"
            doc_type = DocumentType.DOCUMENTATION
            category = _ACT_TO_CATEGORY.get(act, "unknown")
        else:
            parts = file_path.parts
            is_changelog = "changelog" in parts
            doc_type = (
                DocumentType.CHANGELOG if is_changelog else DocumentType.DOCUMENTATION
            )
            title = derive_title_from_path(file_path) or file_path.stem
            if "docs" in parts:
                # docs/<vendor>/<topic>/... — vendor folder is the category.
                category = _canonical_category(parts[parts.index("docs") + 1])
            elif is_changelog:
                folder = parts[parts.index("changelog") + 1]
                category = _CHANGELOG_FOLDER_TO_CATEGORY.get(
                    folder, _canonical_category(folder)
                )
            else:
                category = "unknown"

        title = re.sub(r"\s+", "_", title.strip().lower())

        metadata = DocumentMetadata(
            title=title,
            version=version,
            doc_type=doc_type,
            category_hint=category,  # type: ignore[arg-type]  # validated against CategoryHint
            confidence=1.0,
        )
        metadata.file_path = str(file_path)
        metadata.sort_key = version_to_sort_key(version)

        logger.debug(
            "Deterministic metadata for %s: title=%s version=%s category=%s doc_type=%s",
            file_path.name, title, version, category, doc_type.value,
        )
        return metadata
