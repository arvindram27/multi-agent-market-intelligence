"""Typed state for the LangGraph market-intelligence workflow.

The whole state is checkpointed, so a run can be resumed after the human-in-the-loop
interrupt without re-running the agents.
"""
from typing import Literal

from pydantic import BaseModel, Field

from market_intelligence.agents.schemas import (
    CompetitorProfile,
    NewsSummary,
    SentimentResult,
    TrendReport,
)


class MarketIntelligenceState(BaseModel):
    """State shared by every node in the workflow."""

    # Inputs
    rss_url: str
    competitor_url: str
    num_articles: int = 2

    # Agent outputs
    summaries: list[NewsSummary] = Field(default_factory=list)
    trends: list[TrendReport] = Field(default_factory=list)
    sentiments: list[SentimentResult] = Field(default_factory=list)
    competitor: CompetitorProfile | None = None

    # Human-in-the-loop decision and run status
    approved: bool | None = None
    status: Literal["running", "awaiting_approval", "complete", "rejected"] = "running"
