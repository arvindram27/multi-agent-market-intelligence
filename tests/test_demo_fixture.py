"""The committed demo fixture must stay valid against the agent schemas."""
import json
from pathlib import Path

from market_intelligence.agents.schemas import (
    CompetitorProfile,
    NewsSummary,
    SentimentResult,
    TrendReport,
)

FIXTURE = Path(__file__).resolve().parent.parent / "data" / "demo_run.json"


def test_demo_fixture_parses_into_schemas():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    state = payload["state"]

    summaries = [NewsSummary(**item) for item in state["summaries"]]
    trends = [TrendReport(**item) for item in state["trends"]]
    sentiments = [SentimentResult(**item) for item in state["sentiments"]]
    competitor = CompetitorProfile(**state["competitor"])

    assert payload["captured_at"] == "2026-10-21"
    assert len(summaries) == 3
    assert len(trends) == 3
    assert len(sentiments) == 3
    assert competitor.company == "Acme Health AI"
    assert all(0.0 <= trend.confidence <= 1.0 for trend in trends)
    assert all(-1.0 <= item.score <= 1.0 for item in sentiments)
    assert state["status"] == "awaiting_approval"
    assert state["approved"] is None
