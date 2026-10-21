"""LangGraph workflow tests: fan-out, HITL interrupt/resume, streaming, persistence."""
import sqlite3
from types import SimpleNamespace

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from market_intelligence.agents.schemas import (
    CompetitorProfile,
    NewsSummary,
    SentimentResult,
    TrendAnalysis,
    TrendReport,
)
from market_intelligence.graph import build_graph, checkpoint_serde


def _responses():
    return {
        NewsSummary: NewsSummary(headline="h", summary="s1", sentiment_hint="neutral"),
        TrendAnalysis: TrendAnalysis(
            trends=[TrendReport(trend_name="AI", confidence=0.8)]
        ),
        SentimentResult: SentimentResult(overall="positive", score=0.5),
        CompetitorProfile: CompetitorProfile(company="Acme", positioning="leader"),
    }


def _mock_io(monkeypatch):
    """Mock the two external calls (RSS + competitor page)."""
    fake_feed = SimpleNamespace(
        entries=[SimpleNamespace(title="T1", summary="<p>content one</p>")]
    )
    monkeypatch.setattr("feedparser.parse", lambda url: fake_feed)
    fake_resp = SimpleNamespace(
        content=b"<html><body>Acme corp</body></html>",
        raise_for_status=lambda: None,
    )
    monkeypatch.setattr("requests.get", lambda url, **kw: fake_resp)


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
    assert values["trends"][0].trend_name == "AI"
    assert values["sentiments"][0].overall == "positive"


def test_graph_completes_on_approval(fake_llm, monkeypatch):
    _mock_io(monkeypatch)
    graph = build_graph(checkpointer=_memory_saver(), llm=fake_llm(_responses()))
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
    assert values["approved"] is False
    assert values["status"] == "rejected"


def test_graph_streams_node_updates_once_each(fake_llm, monkeypatch):
    _mock_io(monkeypatch)
    graph = build_graph(checkpointer=_memory_saver(), llm=fake_llm(_responses()))

    chunks = list(graph.stream(_inputs(), _config("stream"), stream_mode="updates"))
    names = {name for chunk in chunks for name in chunk}

    assert {"fetch_news", "identify_trends", "analyze_sentiment"} <= names
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
        )
        second.invoke(Command(resume={"approved": True}), _config("persisted"))
        snap = second.get_state(_config("persisted"))
        assert snap.next == ()
        assert _values(snap)["status"] == "complete"
    finally:
        conn.close()
