"""Shared utility functions for VersionRAG."""

import re
from pathlib import Path

# ── Deterministic metadata helpers ─────────────────────────────────────────────
# These live here (rather than in indexing.pipeline) so both the LLM pipeline
# and the deterministic (metadata-filtered) extractor can share them without a
# circular import.

# Closed set of category_hint values the LLM is allowed to produce, and the
# only values the deterministic extractor validates against. Keep in sync with
# the CategoryHint Literal in versionrag.models.schemas and the allowed-list in
# the attribute-extractor system prompt.
ALLOWED_CATEGORIES = frozenset({
    "docker", "java", "mysql", "numpy", "pandas", "postgresql", "scipy",
    "tensorflow", "wireshark", "golang", "kubernetes", "nodejs",
    "nvidia_tensorrt-llm", "python", "pytorch", "ros2",
    "bootstrap", "spark",
    # UK Legal Acts corpus (large_dataset/legal/raw/).
    "data_protection", "equality", "finance", "income_tax", "localism",
    # Legacy single-token from earlier UK iteration (OD/).
    "uk_legislation",
})

# Common LLM surface-form variants → canonical ALLOWED_CATEGORIES value.
# Lookup happens after lowercase + space-to-underscore normalization, so all
# keys here are already in that normalized form.
CATEGORY_ALIASES: dict[str, str] = {
    "nvidia_tensorrt_llm": "nvidia_tensorrt-llm",
    "tensorrt_llm": "nvidia_tensorrt-llm",
    "tensorrt-llm": "nvidia_tensorrt-llm",
    "nvidia": "nvidia_tensorrt-llm",
    "node.js": "nodejs",
    "node_js": "nodejs",
    "go": "golang",
    "k8s": "kubernetes",
    "tf": "tensorflow",
    "apache_spark": "spark",
    "twbs": "bootstrap",
    "twbs_bootstrap": "bootstrap",
    "ros": "ros2",
    "uk-legislation": "uk_legislation",
}


def derive_title_from_path(file_path: Path) -> str | None:
    """Deterministically derive a Document title from the file path layout.

    Works only for the large_dataset/raw/** layout where directory structure
    is meaningful:
        large_dataset/raw/changelog/<stack>_releases/<file>  → <stack>
        large_dataset/raw/changelog/<stack>_whatsnew/<file>  → <stack>_whatsnew
        large_dataset/raw/changelog/<other>/<file>           → <other>
        large_dataset/raw/docs/<vendor>/<topic>/<file>       → <vendor>_<topic>

    For files outside this layout (e.g. data/raw flat files) returns None and
    the caller falls back to the LLM-extracted title.
    """
    parts = file_path.parts
    if "changelog" in parts:
        idx = parts.index("changelog")
        if idx + 1 < len(parts):
            stack = parts[idx + 1]
            if stack.endswith("_releases"):
                return stack[: -len("_releases")]
            return stack
    if "docs" in parts:
        idx = parts.index("docs")
        if idx + 2 < len(parts):
            return f"{parts[idx + 1]}_{parts[idx + 2]}"
    return None


_DATE_VERSION_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATE_SORT_KEY_OFFSET = 1_000_000_000

# Spelled-out date forms commonly produced by humans in legal-style queries.
# Examples:
#   "1 April 2009"          → 2009-04-01
#   "1st April 2009"        → 2009-04-01
#   "April 1, 2009"         → 2009-04-01
#   "01/04/2009" (UK day-first → 2009-04-01)
# Software-version strings (e.g. "5.3.1", "v1.19", "120rc3") never match
# these patterns, so adding them to normalize_version is a no-op on the
# software corpus.
_MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
_DATE_DMY_WORD_RE = re.compile(
    r"^\s*(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\s+(\d{4})\s*$",
)
_DATE_MDY_WORD_RE = re.compile(
    r"^\s*([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\s*$",
)
_DATE_DDMMYYYY_RE = re.compile(r"^\s*(\d{1,2})/(\d{1,2})/(\d{4})\s*$")


def _try_normalize_spelled_date(v: str) -> str | None:
    """Convert spelled-out / slash-separated dates to ISO YYYY-MM-DD.

    Returns ``None`` when ``v`` is not a recognised date form so the caller
    can fall through to the SemVer normalisation path. Only matches when the
    WHOLE string is a date — substrings like "v5.3.1" never collide because
    they don't satisfy the anchors in the patterns above.
    """
    m = _DATE_DMY_WORD_RE.match(v)
    if m:
        day, month, year = m.groups()
        mon = _MONTHS.get(month.lower())
        if mon:
            return f"{year}-{mon:02d}-{int(day):02d}"
    m = _DATE_MDY_WORD_RE.match(v)
    if m:
        month, day, year = m.groups()
        mon = _MONTHS.get(month.lower())
        if mon:
            return f"{year}-{mon:02d}-{int(day):02d}"
    m = _DATE_DDMMYYYY_RE.match(v)
    if m:
        # UK day-first convention (matches legislation.gov.uk and the
        # eval-set format). The DD/MM ambiguity is not solvable from
        # the string alone — if a future US-format query needs the
        # opposite mapping, gate this branch on a setting.
        day, month, year = m.groups()
        if 1 <= int(month) <= 12 and 1 <= int(day) <= 31:
            return f"{year}-{int(month):02d}-{int(day):02d}"
    return None
# Prefixes are stripped ONLY when directly followed by a digit. Without the
# lookahead, "Version: 2.3.5" would lose its leading "V" and become
# "ersion: 2.3.5" — a hard-to-spot data-poisoning bug that broke 369 of
# our Version nodes after a prompt change that let "Version: ..." through.
_GO_PREFIX_RE = re.compile(r"^go(?=\d)", re.IGNORECASE)
_V_PREFIX_RE = re.compile(r"^v(?=\d)", re.IGNORECASE)
_NODEJS_DOT_X_RE = re.compile(r"^(\d+)\.(\d+)\.x$", re.IGNORECASE)
_NODEJS_X_RE = re.compile(r"^(\d+)\.?x$", re.IGNORECASE)
_RC_SUFFIX_RE = re.compile(r"^(.*?)(rc)(\d+)$", re.IGNORECASE)
_COMPACT_DIGITS_RE = re.compile(r"^(\d{3})$")
_COMPACT_3_RC_RE = re.compile(r"^(\d)(\d)(\d)(rc\d+)$", re.IGNORECASE)

# ROS 2 release codenames in chronological order. The release order is
# alphabetical by intent (each ROS 2 distribution starts with the next letter),
# but we encode an explicit map to be robust to future names.
_ROS_CODENAMES = {
    "ardent": 1, "bouncy": 2, "crystal": 3, "dashing": 4, "eloquent": 5,
    "foxy": 6, "galactic": 7, "humble": 8, "iron": 9, "jazzy": 10,
    "kilted": 11, "lyrical": 12, "rolling": 99,
}


def normalize_version(version: str) -> str:
    """Canonicalize a version string for storage in Neo4j and Qdrant.

    The LLM occasionally returns surface forms that don't match the indexed
    canonical form, which breaks Cypher `v.number=$x` lookups and Qdrant
    `version=$x` payload filters. This function fixes the cheap, unambiguous
    cases:

    - Strip "v" prefix:     "v5.3.1"   → "5.3.1"
    - Strip "go" prefix:    "go1.19"   → "1.19"
    - Decode 3-digit compact form (per-character split):
                            "170"      → "1.7.0"
                            "120"      → "1.2.0"
    - Decode 3-digit + RC suffix:
                            "120rc3"   → "1.2.0rc3"
                            "100rc5"   → "1.0.0rc5"
    - ROS 2 codenames:      lowercased ("Humble" → "humble")

    Ambiguous 4-digit compact forms (`8029` for mysql vs `1170` for scipy,
    where the digit-split depends on software convention) are left untouched —
    correct decoding requires per-software knowledge and is better handled
    by an in-place migration script, not by every ingest.

    ISO date versions (YYYY-MM-DD) are returned unchanged.

    Args:
        version: Raw version string as produced by the LLM.

    Returns:
        Canonical version string suitable for graph + vector store.
    """
    v = version.strip()
    if not v:
        return v

    if _DATE_VERSION_RE.match(v):
        return v  # dates are already canonical

    # Spelled-out / slash dates emitted by human-style queries. Returns
    # None unless v matches one of the date patterns; software versions
    # never satisfy these anchors, so this is a no-op for SemVer inputs.
    iso = _try_normalize_spelled_date(v)
    if iso is not None:
        return iso

    # ROS codenames — case-insensitive, lowercase canonical
    if v.lower() in _ROS_CODENAMES:
        return v.lower()

    # Strip prefixes
    v = _GO_PREFIX_RE.sub("", v)
    v = _V_PREFIX_RE.sub("", v)

    # 3-digit compact + RC suffix:  "120rc3" → "1.2.0rc3"
    m = _COMPACT_3_RC_RE.match(v)
    if m:
        return f"{m.group(1)}.{m.group(2)}.{m.group(3)}{m.group(4).lower()}"

    # 3-digit compact (no dots, no suffix):  "120" → "1.2.0"
    if _COMPACT_DIGITS_RE.match(v):
        return f"{v[0]}.{v[1]}.{v[2]}"

    return v


def _semver_to_key(version: str) -> int:
    """Map a plain numeric / dotted version to a sort key.

    Handles the "compact 3-digit" form ("120" → 1.2.0) commonly emitted by
    LLMs that lose the dots, so semver and compact entries interleave
    correctly on sort.
    """
    if _COMPACT_DIGITS_RE.match(version):
        # "120" → 1.2.0, "110" → 1.1.0 — per-character split
        return int(version[0]) * 10_000 + int(version[1]) * 100 + int(version[2])
    parts = version.split(".")
    segments = (parts + ["0", "0", "0"])[:3]
    try:
        major, minor, patch = (int(s) for s in segments)
    except ValueError:
        return 0
    return major * 10_000 + minor * 100 + patch


def version_to_sort_key(version: str) -> int:
    """Convert a version string to a sortable integer.

    Supports the following formats:

    - **SemVer** ("major.minor.patch"): ``major * 10_000 + minor * 100 + patch``.
    - **Compact digits** ("120" → 1.2.0): three-digit form some LLMs return
      without dots — split per character.
    - **ISO date** ("YYYY-MM-DD"): offset by 10⁹ so date-versioned corpora
      (UK Acts of Parliament) sort strictly above SemVer entries.
    - **nodejs Nx / N.x / N.M.x**: "5x"/"5.x" → 5.0.0; "0.12.x" → 0.12.0.
    - **golang goN.M / go1.19**: strip the "go" prefix, then parse as SemVer.
    - **RC suffixes**: "1.2.0rc3" / "120rc3" → 1.2.0 minus a penalty so RCs
      always sort before the corresponding release; among RCs, the rc-number
      determines order.
    - **ROS 2 codenames**: humble, iron, jazzy, kilted, ... mapped to a small
      ordinal so chronological NEXT chains are correct.

    Examples:
        "15.14.0"     → 151_400
        "5.3.1"       → 50_301
        "120"         → 10_200      (compact 1.2.0)
        "5x"          → 50_000      (nodejs major 5)
        "0.12.x"      → 1_200       (nodejs 0.12.0)
        "humble"      → 8           (ROS 2 ordinal)
        "go1.19"      → 11_900
        "1.2.0rc3"    → 10_103      (base 10_200 minus RC penalty)
        "120rc3"      → 10_103      (same release, compact form)
        "2026-03-31"  → 1_020_260_331

    Args:
        version: Version string.

    Returns:
        Integer sort key. Returns 0 for unrecognized/malformed input.
    """
    v = version.strip()

    # ISO date format — keep date corpora strictly above SemVer.
    if _DATE_VERSION_RE.match(v):
        year, month, day = (int(p) for p in v.split("-"))
        return _DATE_SORT_KEY_OFFSET + year * 10_000 + month * 100 + day

    # ROS 2 release codenames.
    lower = v.lower()
    if lower in _ROS_CODENAMES:
        return _ROS_CODENAMES[lower]

    # Strip Golang "go" prefix and continue.
    v_norm = _GO_PREFIX_RE.sub("", v)

    # nodejs "N.M.x" → N.M.0
    m = _NODEJS_DOT_X_RE.match(v_norm)
    if m:
        return int(m.group(1)) * 10_000 + int(m.group(2)) * 100

    # nodejs "Nx" / "N.x" → N.0.0
    m = _NODEJS_X_RE.match(v_norm)
    if m:
        return int(m.group(1)) * 10_000

    # RC suffix: extract base + rc number; RCs sort BEFORE the release.
    rc_match = _RC_SUFFIX_RE.match(v_norm)
    if rc_match:
        base = rc_match.group(1).rstrip(".")
        rc_num = int(rc_match.group(3))
        base_key = _semver_to_key(base)
        # Penalty pushes RC below the base release; rc_num breaks ties between
        # RCs of the same base. Penalty kept small (-10) so RCs don't collide
        # with the previous minor — e.g. for base=1.1.0 (10_100), penalty 10
        # gives RC range 10_090..10_099, which sits cleanly above 1.0.x patches
        # (10_000..10_099 would conflict at high patch counts; in practice
        # libraries don't ship 90+ patches in a single minor).
        return base_key - 10 + rc_num

    return _semver_to_key(v_norm)


def read_file(file_path: Path) -> str:
    """Read a document file and return its text content.

    Supports Markdown/plain text files and PDFs. PDF text is extracted
    page by page using PyMuPDF.

    Args:
        file_path: Path to the file to read.

    Returns:
        Full text content as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file type is not supported.
    """
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        import pymupdf4llm  # PDF → Markdown conversion

        return pymupdf4llm.to_markdown(str(file_path))
    if suffix in (".md", ".txt", ".rst", ""):
        return file_path.read_text(encoding="utf-8")
    raise ValueError(f"Unsupported file type: {suffix}")
