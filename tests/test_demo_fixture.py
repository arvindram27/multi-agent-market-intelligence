"""The committed demo fixture must stay valid against the agent schemas."""
import json
from pathlib import Path

from market_intelligence.agents.schemas import (
    CompetitorProfile,
    NewsSummary,
    NewsTone,
    TopicReport,
)

FIXTURE = Path(__file__).resolve().parent.parent / "data" / "demo_run.json"


def test_demo_fixture_parses_into_schemas():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    state = payload["state"]

    summaries = [NewsSummary(**item) for item in state["summaries"]]
    topics = [TopicReport(**item) for item in state["topics"]]
    tones = [NewsTone(**item) for item in state["tones"]]
    competitor = CompetitorProfile(**state["competitor"])

    assert payload["captured_at"] == "2026-10-21"
    assert len(summaries) == 3
    assert len(topics) == 3
    assert len(tones) == 3
    assert competitor.company == "Acme Health AI"
    assert all(item.article_id for item in summaries)
    assert all(item.content_kind == "rss_excerpt" for item in summaries)
    assert all(0.0 <= topic.model_confidence <= 1.0 for topic in topics)
    assert all(
        evidence.verified
        for topic in topics
        for evidence in topic.evidence
    )
    assert all(-1.0 <= item.score <= 1.0 for item in tones)
    assert competitor.retrieved_sources
    assert state["status"] == "awaiting_approval"
    assert state["approved"] is None
