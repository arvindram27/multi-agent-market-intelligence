"""Evaluation harness tests: dataset shape, scorer verdicts and reproducible summary."""
import json
from pathlib import Path

from evals.run_evals import (
    CATEGORIES,
    SUMMARY_JSON_PATH,
    SUMMARY_MD_PATH,
    evaluate_cases,
    is_grounded,
    legacy_score,
    load_dataset,
)

CASES = load_dataset()


def test_dataset_has_enough_labeled_cases_across_all_categories():
    assert len(CASES) >= 20
    assert {case["category"] for case in CASES} == set(CATEGORIES)
    assert len({case["id"] for case in CASES}) == len(CASES)


def test_deterministic_verifiers_match_the_authored_labels():
    # evidence_support, unsupported_company_claims and factuality all use the
    # shipped deterministic verifiers, so a correct harness must score 100%.
    names = {"evidence_support", "unsupported_company_claims", "factuality"}
    rows = evaluate_cases([c for c in CASES if c["category"] in names])
    assert all(row["correct"] for row in rows)


def test_legacy_tone_baseline_cannot_represent_mixed_cases():
    # Keyword scoring has no "mixed" verdict, so those cases must stay incorrect.
    # This keeps the committed summary honest about the baseline's limits.
    rows = [row for row in evaluate_cases(CASES) if row["category"] == "tone"]
    assert rows
    mixed = [row for row in rows if row["label"] == "mixed"]
    simple = [row for row in rows if row["label"] != "mixed"]
    assert mixed and all(not row["correct"] for row in mixed)
    assert simple and all(row["correct"] for row in simple)


def test_is_grounded_rejects_introduced_entities():
    excerpt = "Acme raised 40m in Series B funding to expand its copilot across NHS trusts."
    assert is_grounded("Acme raised 40m in Series B funding", excerpt)
    assert is_grounded("Expand its copilot across NHS trusts.", excerpt)
    assert not is_grounded("Acme raised 60m and hired a new CRO", excerpt)


def test_legacy_score_never_crashes_on_any_category():
    for case in CASES:
        predicted = legacy_score(case)
        assert predicted


def test_committed_summary_is_the_offline_reproducible_run():
    summary = json.loads(SUMMARY_JSON_PATH.read_text(encoding="utf-8"))
    assert summary["model"] == "legacy-deterministic-baseline (offline)"
    assert summary["llm_tokens"] is None  # never estimated
    assert summary["cost_usd"] is None  # never estimated
    assert summary["cases"] == len(CASES)
    assert sum(entry["cases"] for entry in summary["categories"]) == len(CASES)

    text = SUMMARY_MD_PATH.read_text(encoding="utf-8")
    assert "Evaluation summary (offline, reproducible)" in text
    assert "| Category |" in text


def test_live_scorers_run_with_a_fake_llm(fake_llm):
    # Prove the --live code path works without network: a fake structured LLM
    # stands in for the real model, and latency/verdicts are still recorded.
    from evals.run_evals import live_score_factuality, live_score_tone
    from market_intelligence.agents.schemas import NewsDigest, NewsToneDraft

    llm = fake_llm(
        {
            NewsToneDraft: NewsToneDraft(overall="positive", score=0.5),
            NewsDigest: NewsDigest(
                headline="h", summary="Acme raised 40m", tone_hint="neutral"
            ),
        }
    )
    tone_case = next(c for c in CASES if c["category"] == "tone")
    fact_case = next(c for c in CASES if c["category"] == "factuality")

    tone_result = live_score_tone(tone_case, llm)
    assert tone_result["predicted"] == "positive"
    assert tone_result["latency_ms"] >= 0.0

    fact_result = live_score_factuality(fact_case, llm)
    assert fact_result["predicted"] in {"grounded", "ungrounded"}
    assert fact_result["latency_ms"] >= 0.0
