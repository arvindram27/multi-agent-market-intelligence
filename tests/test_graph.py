"""LangGraph workflow tests: fan-out, HITL interrupt/resume, streaming, persistence."""
import sqlite3

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from market_intelligence.agents.schemas import (
    CompetitorProfileDraft,
    NewsDigest,
    NewsToneDraft,
    TopicAnalysis,
    TopicFinding,
)
from market_intelligence.budget import TokenBudgetExceeded
from market_intelligence.graph import build_graph, checkpoint_serde
from market_intelligence.graph.nodes import InvalidDecisionError, parse_decision
from market_intelligence.net import FetchResult
from market_intelligence.persistence import NullProfileWriter

FEED_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Test feed</title>
<item><title>T1</title><description>content one is a sufficiently long RSS excerpt for summarization</description></item>
</channel></rss>"""


def _responses():
    return {
        NewsDigest: NewsDigest(headline="h", summary="s1", tone_hint="neutral"),
        TopicAnalysis: TopicAnalysis(
            topics=[TopicFinding(topic_name="AI", model_confidence=0.8)]
        ),
        NewsToneDraft: NewsToneDraft(overall="positive", score=0.5),
        CompetitorProfileDraft: CompetitorProfileDraft(
            company="Acme", positioning="leader"
        ),
    }


def _mock_io(monkeypatch):
    """Mock the bounded fetcher for RSS and competitor pages (no network)."""

    def fake_fetch(url, **kwargs):
        if "feed" in url:
            return FetchResult(url, FEED_XML, "application/rss+xml", 200)
        return FetchResult(
            url,
            b"<html><body>Acme corp is a leading widget maker.</body></html>",
            "text/html",
            200,
        )

    monkeypatch.setattr("market_intelligence.agents.news.fetch_bytes", fake_fetch)
    monkeypatch.setattr("market_intelligence.agents.competitor.fetch_bytes", fake_fetch)


def _inputs():
    return {
        "rss_url": "http://example.com/feed.xml",
        "competitor_url": "https://acme.com",
        "num_articles": 1,
    }


def _config(thread_id="run-1"):
    return {"configurable": {"thread_id": thread_id}}


def _values(snapshot):
    values = snapshot.values
    return values.model_dump() if hasattr(values, "model_dump") else values


def _memory_saver():
    return InMemorySaver(serde=checkpoint_serde())


class SpyWriter:
    """Records persist calls and can simulate a storage failure."""

    def __init__(self, error: Exception | None = None):
        self.calls: list[tuple[str, object]] = []
        self.error = error

    def persist(self, run_id, state):
        self.calls.append((run_id, state))
        if self.error:
            raise self.error
        return run_id


def test_graph_runs_to_approval_interrupt(fake_llm, monkeypatch):
    _mock_io(monkeypatch)
    graph = build_graph(checkpointer=_memory_saver(), llm=fake_llm(_responses()))

    graph.invoke(_inputs(), _config())

    snap = graph.get_state(_config())
    assert snap.next == ("review_profile",)
    values = _values(snap)
    assert values["status"] == "awaiting_approval"
    assert values["competitor"].company == "Acme"
    assert values["summaries"][0].summary == "s1"
    assert values["topics"][0].topic_name == "AI"
    assert values["tones"][0].overall == "positive"


def test_run_call_ceiling_stops_the_graph(fake_llm, monkeypatch):
    _mock_io(monkeypatch)
    graph = build_graph(
        checkpointer=_memory_saver(),
        llm=fake_llm(_responses()),
        max_run_llm_calls=1,
    )

    with pytest.raises(TokenBudgetExceeded, match="1-call"):
        graph.invoke(_inputs(), _config("budget-run"))

    snap = graph.get_state(_config("budget-run"))
    assert snap.next != ("review_profile",)


def test_graph_completes_on_approval(fake_llm, monkeypatch):
    _mock_io(monkeypatch)
    # Explicit no-op writer: tests must never depend on ambient DATABASE_URL.
    graph = build_graph(
        checkpointer=_memory_saver(),
        llm=fake_llm(_responses()),
        writer=NullProfileWriter(),
    )
    graph.invoke(_inputs(), _config())

    graph.invoke(Command(resume={"approved": True}), _config())

    snap = graph.get_state(_config())
    assert snap.next == ()
    values = _values(snap)
    assert values["approved"] is True
    assert values["status"] == "complete"


def test_graph_marks_rejected_on_rejection(fake_llm, monkeypatch):
    _mock_io(monkeypatch)
    graph = build_graph(checkpointer=_memory_saver(), llm=fake_llm(_responses()))
    graph.invoke(_inputs(), _config())

    graph.invoke(Command(resume={"approved": False}), _config())

    snap = graph.get_state(_config())
    values = _values(snap)
    assert snap.next == ()
    assert values["approved"] is False
    assert values["status"] == "rejected"


def test_rejection_never_reaches_the_writer(fake_llm, monkeypatch):
    _mock_io(monkeypatch)
    writer = SpyWriter()
    graph = build_graph(
        checkpointer=_memory_saver(), llm=fake_llm(_responses()), writer=writer
    )
    graph.invoke(_inputs(), _config())

    graph.invoke(Command(resume={"approved": False}), _config())

    assert writer.calls == []
    assert graph.get_state(_config()).next == ()


def test_approval_writes_once_and_reports_the_run_id(fake_llm, monkeypatch):
    _mock_io(monkeypatch)
    writer = SpyWriter()
    graph = build_graph(
        checkpointer=_memory_saver(), llm=fake_llm(_responses()), writer=writer
    )
    graph.invoke(_inputs(), _config())

    graph.invoke(Command(resume={"approved": True}), _config())

    assert len(writer.calls) == 1
    run_id, persisted_state = writer.calls[0]
    assert run_id == "run-1"
    assert persisted_state.competitor.company == "Acme"
    values = _values(graph.get_state(_config()))
    assert values["status"] == "complete"
    assert values["persisted_run_id"] == "run-1"


def test_writer_failure_never_reports_complete(fake_llm, monkeypatch):
    _mock_io(monkeypatch)
    writer = SpyWriter(error=RuntimeError("database unavailable"))
    graph = build_graph(
        checkpointer=_memory_saver(), llm=fake_llm(_responses()), writer=writer
    )
    graph.invoke(_inputs(), _config())

    graph.invoke(Command(resume={"approved": True}), _config())

    snap = graph.get_state(_config())
    values = _values(snap)
    assert snap.next == ()
    assert values["status"] == "persist_failed"
    assert "database unavailable" in values["error"]
    assert values.get("persisted_run_id") is None
    assert len(writer.calls) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"approved": "false"},
        {"approved": "true"},
        {"approved": 1},
        {"approved": None},
        "yes",
        1,
    ],
)
def test_invalid_decisions_are_not_coerced_to_approval(
    fake_llm, monkeypatch, payload
):
    _mock_io(monkeypatch)
    writer = SpyWriter()
    graph = build_graph(
        checkpointer=_memory_saver(), llm=fake_llm(_responses()), writer=writer
    )
    graph.invoke(_inputs(), _config())

    result = graph.invoke(Command(resume=payload), _config())

    snap = graph.get_state(_config())
    values = _values(snap)
    assert snap.next == ("review_profile",)
    assert values.get("approved") is None
    assert "Approval decision must be a boolean" in values["error"]
    assert "__interrupt__" in result
    assert writer.calls == []

    # The invalid payload must not poison the thread: a valid decision still works.
    graph.invoke(Command(resume={"approved": True}), _config())
    values = _values(graph.get_state(_config()))
    assert values["status"] == "complete"
    assert len(writer.calls) == 1


def test_duplicate_resume_does_not_write_twice(fake_llm, monkeypatch):
    _mock_io(monkeypatch)
    writer = SpyWriter()
    graph = build_graph(
        checkpointer=_memory_saver(), llm=fake_llm(_responses()), writer=writer
    )
    graph.invoke(_inputs(), _config())

    graph.invoke(Command(resume={"approved": True}), _config())
    graph.invoke(Command(resume={"approved": True}), _config())

    assert len(writer.calls) == 1


@pytest.mark.parametrize("decision", [True, False])
def test_parse_decision_accepts_explicit_booleans(decision):
    assert parse_decision(decision) is decision
    assert parse_decision({"approved": decision}) is decision


@pytest.mark.parametrize("decision", ["false", 0, {}, {"approved": "false"}, []])
def test_parse_decision_rejects_ambiguous_payloads(decision):
    with pytest.raises(InvalidDecisionError):
        parse_decision(decision)


def test_graph_streams_node_updates_once_each(fake_llm, monkeypatch):
    _mock_io(monkeypatch)
    graph = build_graph(checkpointer=_memory_saver(), llm=fake_llm(_responses()))

    chunks = list(graph.stream(_inputs(), _config("stream"), stream_mode="updates"))
    names = {name for chunk in chunks for name in chunk}

    assert {"fetch_news", "identify_topics", "analyze_news_tone"} <= names
    # Fan-in: the competitor node must run exactly once after both branches.
    assert sum(1 for chunk in chunks if "research_competitor" in chunk) == 1


def test_sqlite_checkpoint_resumes_in_a_new_graph(fake_llm, monkeypatch, tmp_path):
    _mock_io(monkeypatch)
    db_path = str(tmp_path / "checkpoints.sqlite")

    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        first = build_graph(
            checkpointer=SqliteSaver(conn, serde=checkpoint_serde()),
            llm=fake_llm(_responses()),
            writer=NullProfileWriter(),
        )
        first.invoke(_inputs(), _config("persisted"))
        assert first.get_state(_config("persisted")).next == ("review_profile",)
    finally:
        conn.close()

    # A brand-new graph instance, same database: the interrupt is still resumable.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        second = build_graph(
            checkpointer=SqliteSaver(conn, serde=checkpoint_serde()),
            llm=fake_llm(_responses()),
            writer=NullProfileWriter(),
        )
        second.invoke(Command(resume={"approved": True}), _config("persisted"))
        snap = second.get_state(_config("persisted"))
        assert snap.next == ()
        assert _values(snap)["status"] == "complete"
    finally:
        conn.close()
