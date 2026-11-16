"""Universal noise filtering and entry-aware chunking for DeepDiff output.

Used by ``change_extractor.extract_implicit_changes`` to shrink the
diff fed into the LLM and to split very large diffs into manageable
windows.

Both helpers are deliberately domain-agnostic:

* ``filter_diff_noise`` drops only entries that are objectively content-
  free across any text corpus — pure whitespace, and modifications whose
  old/new sides differ only in digits, whitespace or punctuation. This
  collapses URL-version bumps (``/v2.0.3/`` → ``/v2.0.4/``), page-number
  footers (``Page 12 of 45`` → ``Page 13 of 45``), heading-anchor IDs
  (``#section-1234`` → ``#section-5678``), date stamps, and article-number
  renumberings in a single rule.

* ``chunk_diff`` splits the diff at entry boundaries (one DeepDiff entry
  = one inserted / removed / modified line) so a single change is never
  cut across two chunks. A chunk closes when *either* the entry count or
  the serialised-character budget is hit.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Strip every digit and every whitespace/punctuation character. What
# remains is the "alphabetic backbone". Two strings with the same
# backbone differ only in noise; the change is not informational.
_BACKBONE_STRIP_RE = re.compile(
    r"[\s\d.,;:!?\-_/\\()\[\]{}<>\"'`#¶~|*+=&%$@^]+"
)


def _backbone(s: str) -> str:
    """Strip digits, whitespace and common punctuation; casefold the rest."""
    if not isinstance(s, str):
        return ""
    return _BACKBONE_STRIP_RE.sub("", s).casefold()


def _is_empty_or_whitespace(line: Any) -> bool:
    return isinstance(line, str) and line.strip() == ""


# Modifications shorter than this stay in even when their backbones match,
# so semantically-loaded short edits like ``Returns int`` → ``Returns int64``
# or ``n=3`` → ``n=5`` are never mistaken for template noise.
_NOISE_MIN_LEN = 25


def _is_pure_noise_modification(old: Any, new: Any) -> bool:
    """True when old and new differ only in digits/whitespace/punctuation
    AND both sides are long enough to plausibly be a "template + numeric
    metadata" pattern (URL with version, anchor with ID, page footer …).

    Examples that match:
        ``Source: https://docs.foo/v2.0.3/api`` vs ``…/v2.0.4/api``
        ``#section-overview-1234``              vs ``#section-overview-5678``
        ``Last updated: 2024-01-15 14:32:00``   vs ``… 2024-01-16 09:11:00``

    Examples that do NOT match (kept as real changes):
        ``Returns int``  vs ``Returns int64``    (too short)
        ``Page 12``      vs ``Page 13``          (too short)
        ``n=3``          vs ``n=5``              (too short)
    """
    if old == new:
        return False
    if not isinstance(old, str) or not isinstance(new, str):
        return False
    if len(old) < _NOISE_MIN_LEN or len(new) < _NOISE_MIN_LEN:
        return False
    return _backbone(old) == _backbone(new)


def filter_diff_noise(diff_dict: dict[str, Any]) -> dict[str, Any]:
    """Drop deterministic noise entries; preserve the dict shape.

    Categories handled:

    * ``iterable_item_added`` / ``iterable_item_removed``: drop entries
      whose value is empty or whitespace-only.
    * ``values_changed``: drop entries whose old/new pair differs only in
      digits / whitespace / punctuation.

    Other categories (set ops, attribute changes, type changes …) are
    passed through unchanged — they are rare on line-level diffs of
    documentation files and we'd rather over-include than over-filter.
    """
    if not diff_dict:
        return diff_dict

    out: dict[str, Any] = {}
    for category, entries in diff_dict.items():
        if category in ("iterable_item_added", "iterable_item_removed") \
                and isinstance(entries, dict):
            kept = {k: v for k, v in entries.items()
                    if not _is_empty_or_whitespace(v)}
            if kept:
                out[category] = kept

        elif category == "values_changed" and isinstance(entries, dict):
            kept = {}
            for k, v in entries.items():
                old = v.get("old_value") if isinstance(v, dict) else None
                new = v.get("new_value") if isinstance(v, dict) else None
                if _is_pure_noise_modification(old, new):
                    continue
                kept[k] = v
            if kept:
                out[category] = kept

        else:
            out[category] = entries

    return out


def _entry_size(category: str, key: Any, value: Any) -> int:
    """Serialised byte cost of a single entry within its category dict."""
    try:
        return len(json.dumps({category: {str(key): value}}, default=str))
    except Exception:
        return len(str(value)) + len(str(key)) + len(category) + 16


def chunk_diff(
    diff_dict: dict[str, Any],
    max_entries: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    """Split a DeepDiff dict into chunks at entry boundaries.

    Each chunk is a DeepDiff-shaped dict containing a subset of entries
    from each original category. A new chunk is started whenever the
    current one would exceed *either* ``max_entries`` entries or
    ``max_chars`` serialised characters.

    Non-dict categories (rare on line-level diffs) are placed alone in
    their own chunk so they never split arithmetic.

    Returns
    -------
    list[dict]
        One dict per chunk. Empty input returns an empty list.
    """
    if not diff_dict:
        return []

    # Flatten dict-categories into (category, key, value, size) tuples.
    flat: list[tuple[str, Any, Any, int]] = []
    passthrough: list[tuple[str, Any]] = []

    for category, entries in diff_dict.items():
        if isinstance(entries, dict):
            for key, value in entries.items():
                flat.append((category, key, value, _entry_size(category, key, value)))
        else:
            # set_item_added / set_item_removed etc. — keep whole.
            passthrough.append((category, entries))

    chunks: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    current_count = 0
    current_size = 0

    def flush() -> None:
        nonlocal current, current_count, current_size
        if current:
            chunks.append(current)
            current = {}
            current_count = 0
            current_size = 0

    for category, key, value, size in flat:
        would_overflow_entries = current_count >= max_entries
        would_overflow_chars = (
            current_count > 0 and current_size + size > max_chars
        )
        if would_overflow_entries or would_overflow_chars:
            flush()
        current.setdefault(category, {})[key] = value
        current_count += 1
        current_size += size

    flush()

    for category, entries in passthrough:
        chunks.append({category: entries})

    return chunks
