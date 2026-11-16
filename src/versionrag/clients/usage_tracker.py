"""Per-call LLM usage tracking — tokens, duration, model, purpose.

A single CSV log is appended to from every LLM call site. Client scripts
(index_large.py / evaluate_large.py / score_large.py) snapshot the log
position at start and slice off the run's own rows at the end to produce
per-run resource reports for thesis documentation.

Environment variable ``VERSIONRAG_USAGE_LOG`` overrides the default path.
"""

from __future__ import annotations

import csv
import datetime as _dt
import os
import threading
from collections import defaultdict
from pathlib import Path

# OpenAI list prices per 1M tokens (Jan 2026). Update when rates change.
# Used only for an at-a-glance cost estimate; not authoritative billing data.
# Note: gpt-5.4 / gpt-5.4-mini prices are placeholders — fill in actual rates
# from OpenAI pricing page when known.
_PRICING_PER_MILLION_USD: dict[str, tuple[float, float]] = {
    "gpt-5.4":                (0.00, 0.00),   # TODO: update with real pricing
    "gpt-5.4-mini":           (0.00, 0.00),   # TODO: update with real pricing
    "gpt-5-nano":             (0.05, 0.40),
    "gpt-4.1-mini":           (0.40, 1.60),
    "gpt-4.1":                (2.00, 8.00),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}

_LOG_HEADER = [
    "timestamp_iso", "model", "purpose",
    "prompt_tokens", "completion_tokens", "total_tokens",
    "duration_seconds", "estimated_cost_usd",
]


def _default_log_path() -> Path:
    return Path(os.environ.get(
        "VERSIONRAG_USAGE_LOG",
        "large_dataset/results/.usage_log_raw.csv",
    ))


class UsageTracker:
    """Append-only CSV logger for LLM API usage."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self._lock = threading.Lock()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with self.log_path.open("w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(_LOG_HEADER)

    @staticmethod
    def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
        rate = _PRICING_PER_MILLION_USD.get(model)
        if not rate:
            return 0.0
        in_rate, out_rate = rate
        return (prompt_tokens / 1_000_000) * in_rate + (completion_tokens / 1_000_000) * out_rate

    def record(
        self,
        model: str,
        purpose: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration_seconds: float,
    ) -> None:
        total = prompt_tokens + completion_tokens
        cost = self._estimate_cost(model, prompt_tokens, completion_tokens)
        row = [
            _dt.datetime.now().isoformat(timespec="seconds"),
            model, purpose,
            prompt_tokens, completion_tokens, total,
            round(duration_seconds, 3), round(cost, 6),
        ]
        with self._lock, self.log_path.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(row)


_INSTANCE: UsageTracker | None = None


def get_tracker() -> UsageTracker:
    """Return the process-wide tracker, creating it on first access."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = UsageTracker(_default_log_path())
    return _INSTANCE


# ── Client-side helpers (used by orchestration scripts to slice the log) ──


def snapshot_position(log_path: Path | None = None) -> int:
    """Return the byte length of the log file — pass to ``rows_since`` later."""
    p = log_path or _default_log_path()
    return p.stat().st_size if p.exists() else 0


def rows_since(start_position: int, log_path: Path | None = None) -> list[dict]:
    """Return all log rows appended after ``start_position`` bytes."""
    p = log_path or _default_log_path()
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        f.seek(max(0, start_position))
        tail = f.read()
    if not tail.strip():
        return []
    import io
    reader = csv.DictReader(io.StringIO(tail), fieldnames=_LOG_HEADER)
    return [r for r in reader if r["timestamp_iso"] != "timestamp_iso"]


def write_resource_report(
    rows: list[dict],
    output_csv: Path,
    summary_txt: Path | None = None,
    title: str = "VersionRAG Resource Report",
) -> None:
    """Write rows + an aggregated summary to disk for the thesis advisors."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_LOG_HEADER)
        w.writeheader()
        w.writerows(rows)

    # Aggregate by (model, purpose)
    agg: dict[tuple[str, str], dict] = defaultdict(lambda: {
        "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
        "total_tokens": 0, "duration_seconds": 0.0, "cost_usd": 0.0,
    })
    for r in rows:
        key = (r["model"], r["purpose"])
        agg[key]["calls"] += 1
        agg[key]["prompt_tokens"] += int(r["prompt_tokens"])
        agg[key]["completion_tokens"] += int(r["completion_tokens"])
        agg[key]["total_tokens"] += int(r["total_tokens"])
        agg[key]["duration_seconds"] += float(r["duration_seconds"])
        agg[key]["cost_usd"] += float(r["estimated_cost_usd"])

    lines: list[str] = [title, "=" * len(title), ""]
    total_calls = sum(a["calls"] for a in agg.values())
    total_prompt = sum(a["prompt_tokens"] for a in agg.values())
    total_completion = sum(a["completion_tokens"] for a in agg.values())
    total_tokens = sum(a["total_tokens"] for a in agg.values())
    total_cost = sum(a["cost_usd"] for a in agg.values())
    total_duration = sum(a["duration_seconds"] for a in agg.values())
    lines += [
        f"Calls:           {total_calls}",
        f"Input tokens:    {total_prompt:,}",
        f"Output tokens:   {total_completion:,}",
        f"Total tokens:    {total_tokens:,}",
        f"Total duration:  {total_duration:.1f} s ({total_duration/60:.1f} min)",
        f"Est. cost (USD): ${total_cost:.4f}",
        "",
        "By (model, purpose):",
        f"  {'model':<25} {'purpose':<35} {'calls':>6} {'in_tok':>10} {'out_tok':>9} {'total_tok':>10} {'dur_s':>8} {'cost_usd':>10}",
        f"  {'-'*25} {'-'*35} {'-'*6} {'-'*10} {'-'*9} {'-'*10} {'-'*8} {'-'*10}",
    ]
    for (model, purpose), a in sorted(agg.items()):
        lines.append(
            f"  {model:<25} {purpose:<35} {a['calls']:>6} "
            f"{a['prompt_tokens']:>10,} {a['completion_tokens']:>9,} "
            f"{a['total_tokens']:>10,} {a['duration_seconds']:>8.1f} "
            f"${a['cost_usd']:>9.4f}"
        )
    report = "\n".join(lines)
    print("\n" + report)
    if summary_txt:
        summary_txt.parent.mkdir(parents=True, exist_ok=True)
        summary_txt.write_text(report + "\n", encoding="utf-8")
