"""Pydantic structured-output schemas for the market intelligence agents."""
from typing import Literal

from pydantic import BaseModel, Field


class NewsSummary(BaseModel):
    """Structured summary of a single news article."""

    headline: str
    summary: str
    key_facts: list[str] = Field(default_factory=list)
    sentiment_hint: Literal["positive", "negative", "neutral"]


class TrendReport(BaseModel):
    """A single identified market trend."""

    trend_name: str
    keywords: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)


class TrendAnalysis(BaseModel):
    """Container for a list of trends (used for LLM structured output)."""

    trends: list[TrendReport] = Field(default_factory=list)


class SentimentResult(BaseModel):
    """Structured sentiment classification of a text."""

    overall: Literal["positive", "negative", "neutral", "mixed"]
    score: float = Field(ge=-1.0, le=1.0)
    aspects: list[dict] = Field(default_factory=list)


class CompetitorProfile(BaseModel):
    """Structured profile extracted for a competitor."""

    company: str
    positioning: str
    products: list[str] = Field(default_factory=list)
    pricing_model: str | None = None
    funding_stage: str | None = None
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
