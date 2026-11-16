"""LLM-based document metadata extraction."""

import logging
import os
from pathlib import Path

from versionrag.clients.llm import LLMClient
from versionrag.models.schemas import DocumentMetadata
from versionrag.utils import read_file, version_to_sort_key

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = """You are a document metadata extractor.
The input starts with a `FILEPATH:` line (the workspace-relative path),
followed by a blank line, followed by the leading portion of the document body.
USE the FILEPATH as the primary signal — directory structure encodes the
document family and category deterministically.

CRITICAL RULES:

1. **title** — lowercase snake_case, derived from the path leaf (or filename).
   All versions of the same document MUST share the EXACT same title.

   large_dataset/raw/changelog/<X>_releases/...   → title = <X>          (strip _releases)
   large_dataset/raw/changelog/<X>_whatsnew/...   → title = <X>_whatsnew (keep whatsnew)
   large_dataset/raw/docs/<vendor>/<topic>/...    → title = <vendor>_<topic>

   Examples:
     changelog/docker_compose_releases/...   → 'docker_compose'
     changelog/docker_engine_releases/...    → 'docker_engine'
     changelog/java_jdk_releases/...         → 'java_jdk'
     changelog/pandas_whatsnew/...           → 'pandas_whatsnew'
     changelog/postgresql_releases/...       → 'postgresql'
     docs/golang/errors/...                  → 'golang_errors'
     docs/python/stdtypes/...                → 'python_stdtypes'
     docs/ros2/about-nodes/...               → 'ros2_about-nodes'
     docs/nvidia_tensorrt-llm/attention/...  → 'nvidia_tensorrt-llm_attention'

   For flat layouts (data/raw/*.{md,pdf}) infer the title from the filename:
     Release v5.2.3 · twbs_bootstrap.md      → 'bootstrap_releases'
     assert nodejs 11.15.0.md                → 'nodejs_assert'
     errors nodejs 15.14.0.md                → 'nodejs_errors'
     Spark Release 2.4.7 _ Apache Spark.pdf  → 'spark_releases'

2. **version** — REQUIRED, NEVER empty. Output ONLY the bare version
   value. NEVER include surrounding labels, headings, or punctuation.

   READ THE VERSION FROM THE DOCUMENT BODY (not from the filename). Every
   file in this corpus carries a version near the top. Extract ONLY the
   value, NOT the label/heading text:

     Body text:                            → version field:
     "Version: 1.4.0"                      → '1.4.0'     (NOT 'Version: 1.4.0')
     "# SciPy 1.4.0 Release Notes"         → '1.4.0'     (NOT 'SciPy 1.4.0')
     "# NumPy 2.2.0 Release Notes"         → '2.2.0'
     "# TensorFlow 1.3.0"                  → '1.3.0'
     "# Docker Engine 28.0.1 release..."   → '28.0.1'
     "Compatible With ROS 2 Humble..."     → 'humble'
     "# Go 1.23 Release Notes"             → '1.23'

   Allowed normalisations on the extracted value:
     - strip a leading 'v' or 'go' prefix that sits directly before digits:
         'v5.3.1' → '5.3.1' ;  'go1.23' → '1.23'
     - lowercase ROS 2 codenames:          'Humble' → 'humble'
     - keep ISO dates as-is:               '2026-03-31'

   The filename is ONLY a fallback when the body has no version string at all.
   Filenames in this corpus use COMPACT digit forms (no dots) which are
   ambiguous per software (e.g. `scipy_140.md` = 1.4.0, `numpy_213.md` = 2.1.3,
   `tensorflow_130.md` = 1.3.0, `scipy_1111.md` = 1.11.1) — DO NOT try to
   parse the filename yourself. Always prefer the body.

   NEVER output an empty string, 'unknown', or a value that still contains
   letters other than ROS codenames (no 'Version:', no 'SciPy', no 'v').

3. **doc_type** — 'changelog' if FILEPATH contains 'changelog', 'releases',
   or 'whatsnew'; otherwise 'documentation'.

4. **category_hint** — MUST be EXACTLY one of these lowercase tokens
   (no spaces, no synonyms, no plurals):

     docker, java, mysql, numpy, pandas, postgresql, scipy, tensorflow,
     wireshark, golang, kubernetes, nodejs, nvidia_tensorrt-llm, python,
     pytorch, ros2, bootstrap, spark

   Mapping:
     docker_compose, docker_engine   → 'docker'
     pandas_whatsnew, pandas_*       → 'pandas'
     java_jdk                        → 'java'
     anything under docs/golang/     → 'golang'
     anything under docs/python/     → 'python'
     anything under docs/ros2/       → 'ros2'
     twbs_bootstrap                  → 'bootstrap'
     Apache Spark                    → 'spark'
     Node.js assert/errors           → 'nodejs'

   If genuinely none of the above fits, output 'unknown'.

5. **confidence** — 1.0 when FILEPATH + filename make all fields unambiguous;
   0.5 when you had to fall back to the document body.
"""


def _load_system_prompt() -> str:
    """Return the system prompt — from env-var path if set, else default.

    Setting ``VERSIONRAG_EXTRACT_PROMPT`` to a readable file path swaps in a
    domain-specific prompt (e.g. UK legal Acts) without touching this module.
    Unset/empty/missing-file → default prompt above (Node.js / SemVer corpus).
    """
    path = os.environ.get("VERSIONRAG_EXTRACT_PROMPT")
    if path and Path(path).is_file():
        return Path(path).read_text(encoding="utf-8")
    return _DEFAULT_SYSTEM_PROMPT


_SYSTEM_PROMPT = _load_system_prompt()


class AttributeExtractor:
    """Extracts structured metadata from document files using an LLM.

    Supports Markdown and PDF files. For each file, the first 2000 characters
    are sent to the LLM to extract title, version, document type, and category.
    The ``file_path`` and ``sort_key`` fields are set programmatically after
    the LLM call.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def _get_first_page(self, text: str, max_chars: int = 2000) -> str:
        """Return the leading portion of a document for metadata extraction.

        Args:
            text: Full document text.
            max_chars: Maximum character count to return.

        Returns:
            First ``max_chars`` characters of ``text``.
        """
        return text[:max_chars]

    async def extract_metadata(self, file_path: Path) -> DocumentMetadata:
        """Extract structured metadata from a single file using the LLM.

        Args:
            file_path: Path to a .md or .pdf file.

        Returns:
            Populated ``DocumentMetadata`` instance.

        Raises:
            Exception: Propagates LLM or file-read errors to the caller.
        """
        logger.debug("Reading file for metadata extraction: %s", file_path)
        text = read_file(file_path)
        first_page = self._get_first_page(text)
        # Send the workspace-relative path so the LLM can use the directory
        # structure as a deterministic signal for title + category (e.g.
        # docs/golang/errors/errors_go1230.md unambiguously implies
        # category=golang, title=golang_errors, version=1.23.0).
        try:
            rel_path = file_path.resolve().relative_to(Path.cwd())
        except ValueError:
            rel_path = file_path
        prompt = f"FILEPATH: {rel_path}\n\n{first_page}"
        logger.debug("Sending %d chars to LLM for extraction", len(prompt))

        metadata: DocumentMetadata = await self._llm.extract_structured(
            response_model=DocumentMetadata,
            prompt=prompt,
            system_prompt=_SYSTEM_PROMPT,
            purpose="indexing.attribute_extraction",
        )

        # Set programmatic fields — not part of the LLM response
        metadata.file_path = str(file_path)
        metadata.sort_key = version_to_sort_key(metadata.version)

        logger.info(
            "Extracted: %s v%s (%s, %s) confidence=%.2f",
            metadata.title,
            metadata.version,
            metadata.doc_type,
            metadata.category_hint,
            metadata.confidence,
        )
        return metadata

    async def scan_directory(self, directory: Path) -> list[DocumentMetadata]:
        """Scan a directory for .md and .pdf files and extract metadata from each.

        Files that fail extraction are skipped with an ERROR log entry.

        Args:
            directory: Root directory to scan (non-recursive).

        Returns:
            List of ``DocumentMetadata`` sorted by category_hint, title, sort_key.
        """
        files = sorted(list(directory.glob("*.md")) + list(directory.glob("*.pdf")))
        results: list[DocumentMetadata] = []

        for file_path in files:
            try:
                metadata = await self.extract_metadata(file_path)
                results.append(metadata)
            except Exception as exc:
                logger.error("Failed to extract metadata from '%s': %s", file_path, exc)

        results.sort(key=lambda m: (m.category_hint.lower(), m.title.lower(), m.sort_key))
        logger.info("Scanned %d files, extracted %d documents", len(files), len(results))
        return results
