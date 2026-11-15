"""Per-item graph run, audit replay and deterministic report assembly."""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel

import market_intelligence.agents.competitor as competitor_module
import market_intelligence.agents.news as news_module
import market_intelligence.control as control
from market_intelligence.graph import build_graph, checkpoint_serde
from market_intelligence.persistence import NullProfileWriter

from eval import dataset as dataset_module
from eval import fake
from eval import grader

REPORT_NOTES = [
    "fake model: grader plumbing, not model quality",
    "fake token figures are accounting plumbing, not cost or performance evidence",
    "key-fact metrics are lexical matching, not factual accuracy",
    grader.LIMITATIONS,
]

METRIC_KEYS = (
    "news.tone_hint_accuracy",
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
    "comparison.none_evidence_violations",
    "pipeline.invalid_output_rate",
)

COUNTER_KEYS = (
    "key_fact_augmented_matches",
    "key_fact_negation_conflicts",
    "key_fact_numeric_conflicts",
    "key_fact_unassessed_generated",
    "key_fact_unassessed_expected",
    "key_fact_unmatched_generated",
    "key_fact_unmatched_expected",
)

COVERAGE_KEYS = (
    "key_fact_assessable_coverage_generated",
    "key_fact_assessable_coverage_expected",
)


def _values(snapshot):
    values = snapshot.values
    if hasattr(values, "model_dump"):
        return values.model_dump()
    if isinstance(values, dict):
        return {key: _to_plain(val) for key, val in values.items()}
    return values


def _to_plain(value):
    if isinstance(value, BaseModel):
        return {key: _to_plain(val) for key, val in value.model_dump().items()}
    if isinstance(value, dict):
        return {key: _to_plain(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(val) for val in value]
    return value


def _config(thread_id):
    return {"configurable": {"thread_id": thread_id}}


def _empty_metrics():
    return {key: None for key in METRIC_KEYS}


def _empty_diagnostics():
    result = {key: 0 for key in COUNTER_KEYS}
    result.update({key: None for key in COVERAGE_KEYS})
    return result


def _tokens_from_store(store, run_id):
    rows = store._conn.execute("SELECT payload FROM audit_events ORDER BY seq").fetchall()
    events = [
        control.event_from_payload(json.loads(row[0]), f"seq {index}")
        for index, row in enumerate(rows, start=1)
    ]
    schemas = {}
    by_schema = {}
    calls = 0
    for event in events:
        if event.run_id != run_id:
            continue
        if event.event_kind == "decision" and event.operation == "llm":
            if event.decision == "allow":
                calls += 1
                schemas[event.event_id] = event.schema_name
                entry = by_schema.setdefault(
                    event.schema_name, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
                )
                entry["calls"] += 1
    for event in events:
        if event.run_id != run_id or event.event_kind != "outcome" or event.operation != "llm":
            continue
        if event.usage_prompt_tokens is None or event.usage_completion_tokens is None:
            continue
        schema = schemas.get(event.references_event_id)
        if schema is None:
            continue
        entry = by_schema.setdefault(
            schema, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        )
        entry["prompt_tokens"] += event.usage_prompt_tokens
        entry["completion_tokens"] += event.usage_completion_tokens
    totals = {
        "calls": calls,
        "prompt_tokens": sum(item["prompt_tokens"] for item in by_schema.values()),
        "completion_tokens": sum(item["completion_tokens"] for item in by_schema.values()),
    }
    return totals, by_schema


def run_single_item(item, error=None, root=dataset_module.ROOT):
    """Run one dataset item through the real graph with fake effects."""
    fake.apply_settings()
    work = Path(tempfile.mkdtemp(prefix="mami-eval-"))
    store = control.AuditStore(work / "audit.db")
    thread_id = f"eval-{item['item_id']}"
    run_id = item["item_id"]
    clock = fake.Clock()
    gate = fake.make_gate(
        store, fake.policy_for(5), run_id, thread_id, clock=clock
    )
    responses = fake.responses_from_item(item)
    if error is not None:
        responses[next(iter(responses))] = error
    model = fake.UsageLLM(responses)
    fetch = fake.fetch_for_item(item)
    observed = {"exception_class": None, "attempts": 0}
    try:
        with mock.patch.object(news_module, "fetch_bytes", fetch), mock.patch.object(
            competitor_module, "fetch_bytes", fetch
        ):
            graph = build_graph(
                checkpointer=InMemorySaver(serde=checkpoint_serde()),
                llm=model,
                writer=NullProfileWriter(),
                gatekeeper=gate,
                token_counter=fake.fixed_token_counter_for_tests_only,
            )
            state_input = {
                key: item["input"][key]
                for key in (
                    "rss_url",
                    "subject_name",
                    "subject_pages",
                    "competitor_name",
                    "competitor_pages",
                    "num_articles",
                )
            }
            stage = "graph"
            try:
                graph.invoke(state_input, _config(thread_id))
                snapshot = graph.get_state(_config(thread_id))
                if snapshot.next == ("review_profile",):
                    graph.invoke(Command(resume={"approved": True}), _config(thread_id))
                snapshot = graph.get_state(_config(thread_id))
                stage = "grading"
                document = _values(snapshot)
                result = grader.grade_document(
                    document,
                    item["expected"],
                    item["input"]["articles"],
                    item["input"]["pages"],
                )
            except Exception as exc:  # an injected or genuine per-item failure
                observed["exception_class"] = type(exc).__name__
                observed["attempts"] = len(model.calls)
                result = {
                    **_empty_metrics(),
                    "pipeline.invalid_output_rate": 1.0,
                    "diagnostics": _empty_diagnostics(),
                    "invalid_output": True,
                    "invalid_sections": [stage],
                    "exception_class": type(exc).__name__,
                    "attempts": len(model.calls),
                }
    finally:
        totals, by_schema = _tokens_from_store(store, run_id)
        store.close()
    result = dict(result)
    result["item_id"] = item["item_id"]
    result["tokens"] = totals
    result["tokens_by_schema"] = by_schema
    if observed["exception_class"] is not None:
        result["exception_class"] = observed["exception_class"]
        result["attempts"] = observed["attempts"]
    return result


def aggregate(results):
    metrics = {}
    for key in METRIC_KEYS:
        values = [result[key] for result in results if result.get(key) is not None]
        metrics[key] = (sum(values) / len(values)) if values else None
    diagnostics = {}
    for key in COUNTER_KEYS:
        diagnostics[key] = sum(int(result["diagnostics"].get(key, 0)) for result in results)
    for key in COVERAGE_KEYS:
        values = [
            result["diagnostics"][key]
            for result in results
            if result.get("diagnostics", {}).get(key) is not None
        ]
        diagnostics[key] = (sum(values) / len(values)) if values else None
    tokens = {
        "calls": sum(result.get("tokens", {}).get("calls", 0) for result in results),
        "prompt_tokens": sum(
            result.get("tokens", {}).get("prompt_tokens", 0) for result in results
        ),
        "completion_tokens": sum(
            result.get("tokens", {}).get("completion_tokens", 0) for result in results
        ),
    }
    metrics["tokens.fake.calls"] = tokens["calls"]
    metrics["tokens.fake.prompt_tokens"] = tokens["prompt_tokens"]
    metrics["tokens.fake.completion_tokens"] = tokens["completion_tokens"]
    return metrics, diagnostics


def _heldout_entries(entries, root):
    target = str((Path(root) / "eval" / "datasets" / "heldout.jsonl").resolve())
    return [entry for entry in entries if entry["path"] == target]


def build_report(manifest, context, results, dataset_reads, root):
    metrics, diagnostics = aggregate(results)
    heldout_reads = _heldout_entries(dataset_reads, root)
    return {
        "report_version": 2,
        "dataset_sha256": manifest["dataset_sha256"],
        "fixtures_sha256": manifest["fixtures_sha256"]["right"],
        "this_run": {
            "context": context,
            "heldout_hashed": any(
                entry["kind"] == "hash" and entry["outcome"] == "completed"
                for entry in heldout_reads
            ),
            "heldout_parsed": any(
                entry["kind"] == "parse" and entry["outcome"] == "completed"
                for entry in heldout_reads
            ),
        },
        "dataset_reads": [dict(entry) for entry in dataset_reads],
        "metrics": metrics,
        "diagnostics": diagnostics,
        "fake_accounting_only": True,
        "notes": list(REPORT_NOTES),
    }


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")


def read_access_log(path):
    path = Path(path)
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def contaminated(records):
    return any(
        record.get("heldout_sha256") is not None and record.get("context") != "final"
        for record in records
    )


def access_summary(records):
    contexts = {}
    for record in records:
        contexts[record.get("context", "?")] = contexts.get(record.get("context", "?"), 0) + 1
    return {
        "records": len(records),
        "contexts": dict(sorted(contexts.items())),
        "contaminated": contaminated(records),
    }


def append_access(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def run_fixtures(out_path, root=dataset_module.ROOT):
    dataset_module.reset_read_log()
    manifest = dataset_module.verify_manifest(root)
    fixtures = dataset_module.load_fixtures("right", root)
    results = [
        grader.grade_document(
            fixture["document"],
            fixture["expected"],
            fixture["articles"],
            fixture["pages"],
        )
        for fixture in fixtures
    ]
    report = build_report(manifest, "fixtures", results, dataset_module.read_log(), root)
    write_json(out_path, report)
    return report, results


def run_context(context, out_path, root=dataset_module.ROOT, reports_dir=None, diagnostic=False):
    root = Path(root)
    reports_dir = Path(reports_dir) if reports_dir is not None else root / "eval" / "reports"
    dataset_module.reset_read_log()
    manifest = dataset_module.verify_manifest(root)
    access_path = reports_dir / "access.jsonl"
    prior = read_access_log(access_path)
    summary = access_summary(prior)
    write_json(reports_dir / "access_summary.json", summary)
    if summary["contaminated"] and not diagnostic:
        raise SystemExit("held-out contamination recorded in the access log; refusing to measure")
    items, opened = dataset_module.load_dataset(context, root)
    results = [run_single_item(item, root=root) for item in items]
    report = build_report(manifest, context, results, dataset_module.read_log(), root)
    write_json(out_path, report)
    record = {
        "context": context,
        "dev_sha256": manifest["files"]["datasets/dev.jsonl"],
        "heldout_sha256": (
            manifest["files"]["datasets/heldout.jsonl"] if context == "final" else None
        ),
        "item_ids_sha256": dataset_module.sha256_bytes(
            "\n".join(sorted(item["item_id"] for item in items)).encode("utf-8")
        ),
        "dataset_reads": dataset_module.read_log(),
        "at": datetime.now(timezone.utc).isoformat(),
    }
    append_access(access_path, record)
    write_json(reports_dir / "access_summary.json", access_summary(prior + [record]))
    return report, results
