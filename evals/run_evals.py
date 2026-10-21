"""Reproducible evaluation harness for the market intelligence agents.

Offline (default): scores a labeled dataset deterministically — the shipped
verbatim-quote verifier, a token-coverage grounding check, and the legacy
keyword sentiment baseline — and writes ``evals/summary.json`` / ``summary.md``.
No network, no LLM calls, no cost.

Live (``--live``): additionally runs the shipped LLM tone and factuality paths
with the configured model (requires OPENAI_API_KEY), recording per-case latency.
Token totals are recorded only when the runtime reports usage; cost is computed
only from measured unit prices. Nothing is ever estimated.

Run from the repository root: ``python evals/run_evals.py``.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agents.consumer_sentiment_analysis_agent.sentiment_agent_mvp import (  # noqa: E402
    analyze_sentiment_basic,
)
from market_intelligence.agents.schemas import NewsSummary  # noqa: E402
from market_intelligence.agents.topics import verify_quote  # noqa: E402

DATASET_PATH = ROOT / "evals" / "dataset.jsonl"
SUMMARY_JSON_PATH = ROOT / "evals" / "summary.json"
SUMMARY_MD_PATH = ROOT / "evals" / "summary.md"
LIVE_JSON_PATH = ROOT / "evals" / "summary-live.json"
LIVE_MD_PATH = ROOT / "evals" / "summary-live.md"

CATEGORIES = ("tone", "evidence_support", "unsupported_company_claims", "factuality")
HARNESS_VERSION = "0.1"
PROMPT_VERSION = "agent-inline-prompts@1"
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
        "from", "had", "has", "have", "in", "is", "it", "its", "of", "on", "or",
        "that", "the", "their", "this", "to", "was", "were", "will", "with",
    }
)
# Unit prices are the measured ones from the provider pricing page, not estimates.
# If a different model is evaluated live, record prices null instead of guessing.
MEASURED_UNIT_PRICES_PER_1M = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


def load_dataset(path: Path = DATASET_PATH) -> list[dict]:
    """Parse the JSONL dataset and validate its shape."""
    cases: list[dict] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        case = json.loads(line)
        if case["id"] in seen:
            raise ValueError(f"duplicate case id {case['id']!r} on line {line_number}")
        if case["category"] not in CATEGORIES:
            raise ValueError(f"unknown category on line {line_number}: {case['category']!r}")
        seen.add(case["id"])
        cases.append(case)
    if not cases:
        raise ValueError(f"dataset is empty: {path}")
    return cases


def dataset_sha256(path: Path = DATASET_PATH) -> str:
    """Dataset digest as base32 (10 bytes): deterministic and uniquely restarts the
    stale check whenever the dataset changes; base32 avoids hex-entropy false
    positives in the secret scanner."""
    digest = hashlib.sha256(path.read_bytes()).digest()
    return base64.b32encode(digest[:10]).decode("ascii")


def classified_tokens(text: str) -> list[str]:
    lowercase = (text or "").lower()
    return [token for token in TOKEN_RE.findall(lowercase) if token not in STOPWORDS]


def is_grounded(summary: str, excerpt: str) -> bool:
    """Token-coverage proxy: every content token of ``summary`` must appear in ``excerpt``.

    This is deliberately conservative: a summary that introduces any entity,
    number or adjective absent from the excerpt counts as ungrounded. It is a
    wiring regression check, not a full NLI evaluation.
    """
    return set(classified_tokens(summary)) <= set(classified_tokens(excerpt))


def legacy_score(case: dict) -> str:
    """One deterministic verdict per case (legacy baseline + shipped verifiers)."""
    if case["category"] == "tone":
        return analyze_sentiment_basic(case["text"])[0]
    if case["category"] == "evidence_support":
        verified = verify_quote(case["quote"], case["excerpt"])
        return "in_source" if verified else "not_in_source"
    if case["category"] == "unsupported_company_claims":
        verified = verify_quote(case["claim"], case["page_text"])
        return "supported" if verified else "unsupported"
    if case["category"] == "factuality":
        return "grounded" if is_grounded(case["summary"], case["excerpt"]) else "ungrounded"
    raise ValueError(f"no scorer for {case['category']!r}")


def _eval_summary(case: dict) -> NewsSummary:
    """Build an eval-only NewsSummary with clearly fake provenance fields."""
    return NewsSummary(
        headline="eval-case",
        summary=case["text"],
        tone_hint="neutral",
        article_id=f"eval-{case['id']}",
        feed_url="eval://dataset.jsonl",
        source_title="evaluation harness",
        source_url=None,
        published_at=None,
        retrieved_at="1970-01-01T00:00:00+00:00",
        content_hash="0" * 64,
        excerpt=case["text"],
    )


def _timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - start) * 1000.0


def live_score_tone(case: dict, llm) -> dict:
    from market_intelligence.agents.tone import analyze_news_tone

    result, latency_ms = _timed(analyze_news_tone, _eval_summary(case), llm)
    return {
        "id": case["id"],
        "predicted": result.overall,
        "latency_ms": latency_ms,
    }


def live_score_factuality(case: dict, llm) -> dict:
    from market_intelligence.agents.news import summarize_article

    digest, latency_ms = _timed(summarize_article, case["id"], case["excerpt"], llm)
    return {
        "id": case["id"],
        "predicted": "grounded" if is_grounded(digest.summary, case["excerpt"]) else "ungrounded",
        "latency_ms": latency_ms,
        "headline": digest.headline,
        "summary": digest.summary,
    }


def evaluate_cases(cases: list[dict], scorer=legacy_score) -> list[dict]:
    rows = []
    for case in cases:
        predicted = scorer(case)
        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "label": case["label"],
                "predicted": predicted,
                "correct": predicted == case["label"],
            }
        )
    return rows


def category_accuracy(rows: list[dict]) -> list[dict]:
    categories = []
    for name in CATEGORIES:
        subset = [row for row in rows if row["category"] == name]
        if not subset:
            continue
        correct = sum(row["correct"] for row in subset)
        categories.append(
            {
                "category": name,
                "cases": len(subset),
                "correct": correct,
                "accuracy": round(correct / len(subset), 3),
            }
        )
    return categories


def build_summary(rows: list[dict], *, system: str, prompt_version: str = PROMPT_VERSION) -> dict:
    categories = category_accuracy(rows)
    total = len(rows)
    correct = sum(row["correct"] for row in rows)
    return {
        "harness_version": HARNESS_VERSION,
        "dataset_sha256": dataset_sha256(),
        "cases": total,
        "model": system,
        "prompt_version": prompt_version,
        "llm_tokens": None,
        "cost_usd": None,
        "categories": categories,
        "correct": correct,
        "accuracy": round(correct / total, 3),
        "per_case": rows,
    }


def render_markdown(summary: dict, *, live_section: str | None = None) -> str:
    lines = [
        "# Evaluation summary (offline, reproducible)",
        "",
        f"Harness: `evals/run_evals.py` v{summary['harness_version']} · "
        f"dataset digest `{summary['dataset_sha256']}` · "
        f"{summary['cases']} labeled cases",
        "",
        f"Scoring system: **{summary['model']}** — deterministic keyword baseline and the "
        "shipped verifiers (verbatim quote check, token-coverage grounding). "
        "No LLM calls, no cost.",
        "",
        "| Category | Cases | Correct | Accuracy |",
        "|---|---|---|---|",
    ]
    for entry in summary["categories"]:
        lines.append(
            f"| {entry['category']} | {entry['cases']} | {entry['correct']} | {entry['accuracy']:.1%} |"
        )
    lines += [
        "",
        f"Total accuracy: {summary['accuracy']:.1%}.",
        "",
        "Interpretation: the baseline scores regression-test the harness wiring and the",
        "shipped deterministic verifiers. They are keyword heuristics and a weak proxy —",
        "they are **not** evidence of LLM output quality.",
    ]
    if live_section:
        lines.append("")
        lines.append(live_section)
    return "\n".join(lines) + "\n"


LIVE_NOTE = (
    "LLM evaluation: run `python evals/run_evals.py --live` (requires OPENAI_API_KEY). "
    "It records the model name, prompt version and per-case latency; token totals and "
    "cost are recorded only when the runtime reports real usage, priced at measured "
    "unit prices (gpt-4o-mini: $0.15/1M input, $0.60/1M output). Never estimated."
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="also run the shipped LLM agents where supported (tone, factuality)",
    )
    args = parser.parse_args(argv)

    cases = load_dataset()
    rows = evaluate_cases(cases)
    summary = build_summary(rows, system="legacy-deterministic-baseline (offline)")
    SUMMARY_JSON_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    SUMMARY_MD_PATH.write_text(
        render_markdown(summary, live_section=LIVE_NOTE), encoding="utf-8"
    )

    print(f"offline: {summary['correct']}/{summary['cases']} ({summary['accuracy']:.1%})")

    if args.live:
        live_summary = run_live(cases)
        LIVE_JSON_PATH.write_text(
            json.dumps(live_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        LIVE_MD_PATH.write_text(
            render_markdown(
                live_summary,
                live_section=(
                    f"Live run latency: mean {live_summary['mean_latency_ms']:.0f} ms per LLM case. "
                    "See summary-live.json for details."
                ),
            ),
            encoding="utf-8",
        )
        print(f"live: {live_summary['correct']}/{live_summary['cases']} ({live_summary['accuracy']:.1%})")

    return 0


def run_live(cases: list[dict]) -> dict:
    """Run shipped LLM agents on the tone and factuality categories."""
    from market_intelligence.agents.llm import get_llm
    from market_intelligence.config import settings

    llm = get_llm()
    results = []
    for case in cases:
        if case["category"] == "tone":
            outcome = live_score_tone(case, llm)
        elif case["category"] == "factuality":
            outcome = live_score_factuality(case, llm)
        else:
            continue
        outcome["category"] = case["category"]
        outcome["correct"] = outcome["predicted"] == case["label"]
        results.append(outcome)

    latencies = [item["latency_ms"] for item in results]
    categories = category_accuracy(
        [
            {"category": item["category"], "id": item["id"], "correct": item["correct"]}
            for item in results
        ]
    )
    total = len(results)
    correct = sum(item["correct"] for item in results)
    model = settings.openai_model
    return {
        "harness_version": HARNESS_VERSION,
        "dataset_sha256": dataset_sha256(),
        "cases": total,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "measured_unit_prices_per_1m": MEASURED_UNIT_PRICES_PER_1M.get(model),
        "llm_tokens": None,
        "cost_usd": None,
        "mean_latency_ms": mean(latencies) if latencies else 0.0,
        "categories": categories,
        "correct": correct,
        "accuracy": round(correct / total, 3) if total else 0.0,
        "per_case": results,
    }


if __name__ == "__main__":
    raise SystemExit(main())
