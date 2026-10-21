"""Typed state for the LangGraph market-intelligence workflow.

The whole state is checkpointed, so a run can be resumed after the human-in-the-loop
interrupt without re-running the agents.
"""
from typing import Literal

from pydantic import BaseModel, Field

from market_intelligence.agents.schemas import (
    CompetitorProfile,
    NewsSummary,
    NewsTone,
    TopicReport,
)
from market_intelligence.config import settings


class MarketIntelligenceState(BaseModel):
    """State shared by every node in the workflow."""

    # Inputs
    rss_url: str
    competitor_url: str
    num_articles: int = Field(
        default=settings.num_articles, ge=1, le=settings.max_articles
    )

    # Agent outputs
    summaries: list[NewsSummary] = Field(default_factory=list)
    topics: list[TopicReport] = Field(default_factory=list)
    tones: list[NewsTone] = Field(default_factory=list)
    competitor: CompetitorProfile | None = None
    warnings: list[str] = Field(default_factory=list)

    # Human-in-the-loop decision and run status
    approved: bool | None = None
    status: Literal[
        "running",
        "awaiting_approval",
        "approved_pending_write",
        "complete",
        "rejected",
        "persist_failed",
    ] = "running"
    error: str | None = None
    persisted_run_id: str | None = None
