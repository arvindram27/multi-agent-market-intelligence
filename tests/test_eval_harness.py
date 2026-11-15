"""Offline tests for the milestone 1a evaluation harness."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval import dataset, grader, runner
from market_intelligence.agents.schemas import NewsDigest

ROOT = Path(__file__).resolve().parent.parent


def _wrong(case_id):
    for case in dataset.load_fixtures("wrong", ROOT):
        if case["case_id"] == case_id:
            return case
    raise AssertionError(f"missing wrong fixture {case_id}")


def _right():
    return dataset.load_fixtures("right", ROOT)


def _facts(*texts):
    return [{"fact_id": f"F{index + 1}", "text": text} for index, text in enumerate(texts)]


def _grade(case):
    return grader.grade_document(
        case["document"], case["expected"], case["articles"], case["pages"]
    )


def _metrics(*texts_generated, expected=None):
    facts = expected if expected is not None else _facts(*texts_generated)
    generated = list(texts_generated)
    matches, diagnostics, sides = grader.match_key_facts(facts, generated)
    pairs = grader.fact_metrics(len(matches), len(facts), len(generated), diagnostics, sides)
    return matches, diagnostics, pairs


def test_dataset_files_parse_and_match_manifest():
    manifest = dataset.verify_manifest(ROOT)
    assert manifest["counts"] == {"dev": 9, "heldout": 3, "total": 12}
    assert manifest["schema_version"] == 1
    dev = dataset.load_jsonl(ROOT / "eval" / "datasets" / "dev.jsonl")
    heldout = dataset.load_jsonl(ROOT / "eval" / "datasets" / "heldout.jsonl")
    assert len(dev) == 9 and len(heldout) == 3
    assert len({item["item_id"] for item in dev + heldout}) == 12


def test_license_and_length_boundaries():
    for item in dataset.load_jsonl(ROOT / "eval" / "datasets" / "dev.jsonl") + dataset.load_jsonl(
        ROOT / "eval" / "datasets" / "heldout.jsonl"
    ):
        for article in item["input"]["articles"]:
            assert article["license"]["kind"] == "self_authored"
            assert article["license"]["redistributable"] is True
            assert len(article["excerpt"]) <= 320
        for page in item["input"]["pages"]:
            assert len(page["text"]) <= 320


def test_every_right_fixture_scores_perfect():
    for case in _right():
        result = _grade(case)
        for key in (
            "news.tone_hint_accuracy",
            "news.headline_containment_rate",
            "news.summary_containment_rate",
            "news.key_fact_lexical_precision",
            "news.key_fact_lexical_recall",
            "topics.topic_alias_accuracy",
            "topics.keyword_f1",
            "topics.evidence_coverage",
            "topics.citation_precision",
            "tones.overall_accuracy",
            "tones.score_within_band_rate",
            "tones.aspect_f1",
            "comparison.field_value_match_rate",
            "comparison.evidence_valid_rate",
        ):
            assert result[key] == 1.0, (case["case_id"], key, result[key])
        assert result["pipeline.invalid_output_rate"] == 0.0
        assert result["comparison.none_evidence_violations"] == 0
        for key in (
            "key_fact_augmented_matches",
            "key_fact_negation_conflicts",
            "key_fact_numeric_conflicts",
            "key_fact_unassessed_generated",
            "key_fact_unassessed_expected",
            "key_fact_unmatched_generated",
            "key_fact_unmatched_expected",
        ):
            assert result["diagnostics"][key] == 0
        assert result["diagnostics"]["key_fact_assessable_coverage_generated"] == 1.0
        assert result["diagnostics"]["key_fact_assessable_coverage_expected"] == 1.0


def test_wrong_classification_is_caught():
    result = _grade(_wrong("wrong-001"))
    assert result["tones.overall_accuracy"] == 0.0
    assert result["tones.score_within_band_rate"] == 0.0


def test_missing_and_extra_facts_change_lexical_precision_and_recall():
    missing = _grade(_wrong("wrong-002"))
    assert missing["news.key_fact_lexical_precision"] is None
    assert missing["news.key_fact_lexical_recall"] == 0.0
    extra = _grade(_wrong("wrong-003"))
    assert extra["news.key_fact_lexical_precision"] == 0.5
    assert extra["news.key_fact_lexical_recall"] == 1.0


def test_negation_with_identical_words_is_rejected():
    _matches, diagnostics, pairs = _metrics(
        "acme never will expand its copilot to three trusts",
        expected=_facts("acme will expand its copilot to three trusts"),
    )
    assert diagnostics["key_fact_negation_conflicts"] == 1
    assert pairs["precision"] == 0.0 and pairs["recall"] == 0.0


def test_exact_numeric_conflict_same_currency_fails():
    _matches, diagnostics, pairs = _metrics(
        "acme raised £41m", expected=_facts("acme raised £40m")
    )
    assert diagnostics["key_fact_numeric_conflicts"] == 1
    assert pairs["precision"] == 0.0


def test_currency_or_unit_mismatch_is_unassessed_not_failed():
    _matches, diagnostics, pairs = _metrics(
        "acme raised $40m", expected=_facts("acme raised £40m")
    )
    assert diagnostics["key_fact_unassessed_generated"] == 1
    assert diagnostics["key_fact_unassessed_expected"] == 1
    assert pairs["precision"] is None and pairs["recall"] is None


def test_safe_unit_conversion_is_applied():
    matches, diagnostics, pairs = _metrics(
        "acme raised 1500m", expected=_facts("acme raised 1.5bn")
    )
    assert len(matches) == 1
    assert pairs["precision"] == 1.0 and pairs["recall"] == 1.0
    assert diagnostics["key_fact_numeric_conflicts"] == 0


def test_augmented_match_is_diagnostic_only():
    result = _grade(_wrong("wrong-016"))
    assert result["news.key_fact_lexical_precision"] == 1.0
    assert result["diagnostics"]["key_fact_augmented_matches"] == 1


def test_one_to_one_tie_break_is_stable():
    matches, diagnostics, _pairs = _metrics(
        "acme acquired beta and gamma",
        expected=_facts("acme acquired beta", "acme acquired gamma"),
    )
    assert matches == [(0, 0)]
    assert diagnostics["key_fact_unmatched_expected"] == 1


def test_key_fact_scores_are_order_independent():
    expected = _facts("alpha beta", "alpha gamma")
    forward = grader.match_key_facts(expected, ["alpha beta", "alpha gamma"])
    reverse = grader.match_key_facts(expected, ["alpha gamma", "alpha beta"])
    forward_ids = sorted(expected[i]["fact_id"] for i, _ in forward[0])
    reverse_ids = sorted(expected[i]["fact_id"] for i, _ in reverse[0])
    assert forward_ids == reverse_ids == ["F1", "F2"]
    assert forward[1] == reverse[1]


def test_unknown_and_non_verbatim_citations_are_unverified():
    unknown = _grade(_wrong("wrong-004"))
    assert unknown["topics.citation_precision"] == 0.0
    assert unknown["topics.evidence_coverage"] == 0.0
    quote = _grade(_wrong("wrong-005"))
    assert quote["topics.citation_precision"] == 0.0
    assert quote["topics.evidence_coverage"] == 0.0


def test_schema_invalid_document_is_counted_not_crashing():
    result = _grade(_wrong("wrong-006"))
    assert result["invalid_output"] is True
    assert result["pipeline.invalid_output_rate"] == 1.0
    assert result["tones.overall_accuracy"] == 0.0


def test_none_and_evidence_invariant_is_enforced():
    result = _grade(_wrong("wrong-007"))
    assert result["comparison.none_evidence_violations"] == 1


def test_comparison_quote_must_appear_in_page_text():
    result = _grade(_wrong("wrong-008"))
    assert result["comparison.evidence_valid_rate"] == 0.5


def test_tone_score_band_is_independent_of_label():
    result = _grade(_wrong("wrong-009"))
    assert result["tones.overall_accuracy"] == 1.0
    assert result["tones.score_within_band_rate"] == 0.0


def test_extra_and_missing_topics_change_precision_and_recall():
    result = _grade(_wrong("wrong-010"))
    assert result["topics.topic_alias_accuracy"] == 0.0
    assert result["topics.keyword_f1"] == 0.0


def test_report_is_deterministic_despite_dev_access_appends():
    work = Path(tempfile.mkdtemp(prefix="mami-eval-det-"))
    first = work / "a.json"
    second = work / "b.json"
    runner.run_context("dev", first, root=ROOT, reports_dir=work)
    runner.run_context("dev", second, root=ROOT, reports_dir=work)
    assert first.read_bytes() == second.read_bytes()


def test_report_labels_fake_tokens_only():
    work = Path(tempfile.mkdtemp(prefix="mami-eval-report-"))
    out = work / "report.json"
    runner.run_fixtures(out, root=ROOT)
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["fake_accounting_only"] is True
    assert report["dataset_sha256"] and report["fixtures_sha256"]
    for key in list(report) + list(report["metrics"]):
        lowered = key.lower()
        assert "cost" not in lowered and "latency" not in lowered
    assert "cost" not in json.dumps(report["metrics"]).lower()
    assert "latency" not in json.dumps(report["metrics"]).lower()
    assert any(grader.LIMITATIONS in note for note in report["notes"])


def test_heldout_is_not_loaded_in_dev_context(monkeypatch):
    opened = []
    real = dataset._read_jsonl

    def spy(path):
        opened.append(str(path))
        return real(path)

    monkeypatch.setattr(dataset, "_read_jsonl", spy)
    items, paths = dataset.load_dataset("dev", ROOT)
    assert len(items) == 9
    assert all("heldout" not in path for path in opened + paths)


def test_contaminated_access_log_is_gated():
    work = Path(tempfile.mkdtemp(prefix="mami-eval-gate-"))
    (work / "access.jsonl").write_text(
        json.dumps(
            {
                "context": "dev",
                "dev_sha256": "x",
                "heldout_sha256": "y",
                "item_ids_sha256": "z",
                "at": "2026-11-15T00:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    records = runner.read_access_log(work / "access.jsonl")
    summary = runner.access_summary(records)
    assert summary["contaminated"] is True
    with pytest.raises(SystemExit):
        runner.run_context(
            "dev", work / "out.json", root=ROOT, reports_dir=work
        )


def test_graph_runner_scores_a_right_item_end_to_end():
    item = dataset.load_jsonl(ROOT / "eval" / "datasets" / "dev.jsonl")[0]
    result = runner.run_single_item(item, root=ROOT)
    assert result["invalid_output"] is False
    assert result["pipeline.invalid_output_rate"] == 0.0
    assert result["news.tone_hint_accuracy"] == 1.0
    assert result["news.headline_containment_rate"] == 1.0
    assert result["news.summary_containment_rate"] == 1.0
    assert result["news.key_fact_lexical_precision"] == 1.0
    assert result["news.key_fact_lexical_recall"] == 1.0
    assert result["tones.overall_accuracy"] == 1.0
    assert result["tones.score_within_band_rate"] == 1.0
    assert result["tones.aspect_f1"] == 1.0
    assert result["comparison.field_value_match_rate"] == 1.0
    assert result["comparison.evidence_valid_rate"] == 1.0
    assert result["comparison.none_evidence_violations"] == 0
    assert result["tokens"]["calls"] == 5
    assert result["tokens_by_schema"]["NewsDigest"]["calls"] == 1
    assert result["tokens_by_schema"]["TopicAnalysis"]["calls"] == 1
    assert result["tokens_by_schema"]["NewsToneDraft"]["calls"] == 1
    assert result["tokens_by_schema"]["SideComparisonDraft"]["calls"] == 2


def test_direction_reversal_with_comparable_number_fails():
    _matches, diagnostics, pairs = _metrics(
        "revenue fell 40%", expected=_facts("revenue grew 40%")
    )
    assert diagnostics["key_fact_numeric_conflicts"] == 1
    assert pairs["precision"] == 0.0


def test_runner_survives_injected_validation_failure_and_counts_it():
    try:
        NewsDigest(headline="h", summary="s", tone_hint="angry")
    except ValidationError as exc:
        injected = exc
    else:  # pragma: no cover
        raise AssertionError("the invalid digest unexpectedly validated")
    item = dataset.load_jsonl(ROOT / "eval" / "datasets" / "dev.jsonl")[0]
    result = runner.run_single_item(item, error=injected, root=ROOT)
    assert result["invalid_output"] is True
    assert result["pipeline.invalid_output_rate"] == 1.0
    assert result["exception_class"] == "ValidationError"
    assert result["attempts"] == 1


def test_missing_currency_on_both_sides_matches_on_value_and_unit():
    matches, diagnostics, pairs = _metrics(
        "acme raised 40m", expected=_facts("acme raised 40 million")
    )
    assert len(matches) == 1
    assert pairs["precision"] == 1.0 and diagnostics["key_fact_numeric_conflicts"] == 0


def test_missing_currency_on_one_side_is_unassessed():
    _matches, diagnostics, pairs = _metrics(
        "acme raised 40m", expected=_facts("acme raised £40m")
    )
    assert diagnostics["key_fact_unassessed_generated"] == 1
    assert diagnostics["key_fact_unassessed_expected"] == 1
    assert pairs["precision"] is None and pairs["recall"] is None


def test_conflict_pair_does_not_consume_valid_matches():
    result = _grade(_wrong("wrong-018"))
    assert result["news.key_fact_lexical_precision"] == 1.0
    assert result["news.key_fact_lexical_recall"] == 1.0
    assert result["diagnostics"]["key_fact_numeric_conflicts"] == 0


def test_matching_is_maximum_cardinality():
    expected = _facts("alpha beta", "alpha beta gamma")
    matches, _diagnostics, pairs = _metrics(
        "alpha beta gamma", "alpha gamma beta delta", expected=expected
    )
    assert len(matches) == 2
    assert pairs["precision"] == 1.0 and pairs["recall"] == 1.0


def test_per_token_punctuation_is_stripped():
    matches, _diagnostics, pairs = _metrics(
        "Acme, acquired Beta.", expected=_facts("acme acquired beta")
    )
    assert len(matches) == 1
    assert pairs["precision"] == 1.0


def test_leftover_numeric_claim_is_unassessed():
    result = _grade(_wrong("wrong-020"))
    assert result["news.key_fact_lexical_precision"] is None
    assert result["news.key_fact_lexical_recall"] is None
    assert result["diagnostics"]["key_fact_unassessed_generated"] == 1


def test_every_wrong_fixture_matches_its_expect_block():
    for case in dataset.load_fixtures("wrong", ROOT):
        result = _grade(case)
        for key, expected in case["expect"].items():
            if key == "diagnostics":
                for name, value in expected.items():
                    assert result["diagnostics"][name] == value, (
                        case["case_id"],
                        name,
                        result["diagnostics"][name],
                        value,
                    )
            else:
                assert result[key] == expected, (case["case_id"], key, result[key], expected)


def _synthetic_item(item_id, split):
    return {
        "item_id": item_id,
        "split": split,
        "input": {
            "articles": [
                {
                    "license": {"kind": "self_authored", "redistributable": True},
                    "excerpt": "synthetic excerpt",
                    "title": "synthetic title",
                }
            ],
            "pages": [],
            "num_articles": 1,
        },
        "fake": {},
        "expected": {},
        "provenance": {
            "label_author": "tests",
            "labeled_at": "2026-11-15",
            "rationale": "synthetic count check",
            "source_reference": "synthetic",
            "review": "synthetic",
        },
    }


def test_dev_context_run_never_deserializes_heldout(monkeypatch):
    work = Path(tempfile.mkdtemp(prefix="mami-eval-iso-"))
    parsed = []
    real = dataset._read_jsonl

    def spy(path):
        parsed.append(str(Path(path).resolve()))
        return real(path)

    monkeypatch.setattr(dataset, "_read_jsonl", spy)
    runner.run_context("dev", work / "report.json", root=ROOT, reports_dir=work)
    target = str((ROOT / "eval" / "datasets" / "heldout.jsonl").resolve())
    assert parsed
    assert all(path != target for path in parsed)


def test_fixtures_run_never_deserializes_heldout(monkeypatch):
    work = Path(tempfile.mkdtemp(prefix="mami-eval-iso-fix-"))
    parsed = []
    real = dataset._read_jsonl

    def spy(path):
        parsed.append(str(Path(path).resolve()))
        return real(path)

    monkeypatch.setattr(dataset, "_read_jsonl", spy)
    runner.run_fixtures(work / "fixtures.json", root=ROOT)
    target = str((ROOT / "eval" / "datasets" / "heldout.jsonl").resolve())
    assert parsed
    assert all(path != target for path in parsed)


def test_dev_context_run_accounts_heldout_raw_byte_reads():
    work = Path(tempfile.mkdtemp(prefix="mami-eval-acct-"))
    out = work / "report.json"
    runner.run_context("dev", out, root=ROOT, reports_dir=work)
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["report_version"] == 2
    assert "heldout_opened" not in report["this_run"]
    assert report["this_run"]["heldout_hashed"] is True
    assert report["this_run"]["heldout_parsed"] is False
    records = runner.read_access_log(work / "access.jsonl")
    assert len(records) == 1
    target = str((ROOT / "eval" / "datasets" / "heldout.jsonl").resolve())
    entries = [entry for entry in records[0]["dataset_reads"] if entry["path"] == target]
    completed = [entry for entry in entries if entry["outcome"] == "completed"]
    assert len(completed) == 2
    assert all(entry["kind"] == "hash" for entry in completed)
    assert any(entry["outcome"] == "attempted" for entry in entries)
    assert not any(
        entry["kind"] == "parse" and entry["outcome"] == "completed" for entry in entries
    )


def test_read_log_records_attempted_and_failed_outcomes():
    work = Path(tempfile.mkdtemp(prefix="mami-eval-outcomes-"))
    missing = (work / "missing.jsonl").resolve()
    bad = work / "bad.jsonl"
    bad.write_text("not json\n", encoding="utf-8")
    undecodable = work / "undecodable.jsonl"
    undecodable.write_bytes(b"\xff\xfe\xfd")
    good = work / "good.jsonl"
    good.write_text('{"item_id": "x"}\n', encoding="utf-8")

    dataset.reset_read_log()
    with pytest.raises(OSError):
        dataset._read_jsonl(missing)
    assert dataset.read_log() == [
        {"path": str(missing), "kind": "parse", "outcome": "attempted"},
        {"path": str(missing), "kind": "parse", "outcome": "open_failed"},
    ]

    dataset.reset_read_log()
    with pytest.raises(dataset.EvalDataError):
        dataset._read_jsonl(bad)
    assert dataset.read_log() == [
        {"path": str(bad.resolve()), "kind": "parse", "outcome": "attempted"},
        {"path": str(bad.resolve()), "kind": "parse", "outcome": "parse_failed"},
    ]

    dataset.reset_read_log()
    with pytest.raises(UnicodeDecodeError):
        dataset._read_jsonl(undecodable)
    assert dataset.read_log() == [
        {"path": str(undecodable.resolve()), "kind": "parse", "outcome": "attempted"},
        {"path": str(undecodable.resolve()), "kind": "parse", "outcome": "decode_failed"},
    ]

    dataset.reset_read_log()
    assert dataset._read_jsonl(good) == [{"item_id": "x"}]
    assert dataset.read_log() == [
        {"path": str(good.resolve()), "kind": "parse", "outcome": "attempted"},
        {"path": str(good.resolve()), "kind": "parse", "outcome": "completed"},
    ]

    dataset.reset_read_log()
    with pytest.raises(OSError):
        dataset._read_bytes(missing)
    assert dataset.read_log() == [
        {"path": str(missing), "kind": "hash", "outcome": "attempted"},
        {"path": str(missing), "kind": "hash", "outcome": "open_failed"},
    ]


def test_final_load_validates_counts_against_manifest():
    work = Path(tempfile.mkdtemp(prefix="mami-eval-counts-"))
    datasets = work / "eval" / "datasets"
    datasets.mkdir(parents=True)
    (datasets / "dev.jsonl").write_text(
        json.dumps(_synthetic_item("synth-dev-1", "dev")) + "\n", encoding="utf-8"
    )
    (datasets / "heldout.jsonl").write_text(
        json.dumps(_synthetic_item("synth-heldout-1", "heldout")) + "\n", encoding="utf-8"
    )
    manifest_path = work / "eval" / "manifest.json"
    manifest_path.write_text(
        json.dumps({"counts": {"dev": 1, "heldout": 2, "total": 3}}) + "\n", encoding="utf-8"
    )
    with pytest.raises(dataset.EvalDataError):
        dataset.load_dataset("final", work)
    manifest_path.write_text(
        json.dumps({"counts": {"dev": 1, "heldout": 1, "total": 2}}) + "\n", encoding="utf-8"
    )
    items, opened = dataset.load_dataset("final", work)
    assert len(items) == 2
