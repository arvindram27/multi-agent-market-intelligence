"""Pydantic structured-output schemas for the market intelligence agents.

Two kinds of models live here:

* ``*Draft`` models are sent to the LLM with ``with_structured_output``.
* Enriched models add provenance captured by the application (IDs, URLs,
  timestamps, hashes, verified evidence) so the model cannot invent it.
"""
from typing import Literal

from pydantic import BaseModel, Field


class NewsDigest(BaseModel):
    """LLM-generated summary of one RSS excerpt."""

    headline: str
    summary: str
    key_facts: list[str] = Field(default_factory=list)
    tone_hint: Literal["positive", "negative", "neutral"]


class NewsSummary(NewsDigest):
    """A digest plus the captured source record it came from."""

    article_id: str
    feed_url: str
    source_title: str
    source_url: str | None = None
    published_at: str | None = None
    retrieved_at: str
    content_hash: str
    excerpt: str
    content_kind: Literal["rss_excerpt"] = "rss_excerpt"


class TopicEvidenceQuote(BaseModel):
    """A source reference requested from the LLM."""

    article_id: str
    quote: str


class TopicFinding(BaseModel):
    """LLM-generated topic with cited excerpts."""

    topic_name: str
    keywords: list[str] = Field(default_factory=list)
    model_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="LLM self-rating from 0 to 1; not a calibrated probability.",
    )
    evidence: list[TopicEvidenceQuote] = Field(default_factory=list)


class TopicAnalysis(BaseModel):
    """Container for LLM topic output."""

    topics: list[TopicFinding] = Field(default_factory=list)


class TopicEvidence(BaseModel):
    """An evidence span resolved against a captured RSS excerpt."""

    article_id: str
    quote: str
    source_title: str
    source_url: str | None = None
    published_at: str | None = None
    verified: bool


class TopicReport(BaseModel):
    """A topic extracted from one RSS snapshot.

    ``window_start``/``window_end`` describe the publication window of the
    captured excerpts. This is explicitly not a time-series trend claim.
    """

    topic_name: str
    keywords: list[str] = Field(default_factory=list)
    model_confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[TopicEvidence] = Field(default_factory=list)
    window_start: str | None = None
    window_end: str | None = None
    source_count: int = Field(ge=0)


class ToneAspect(BaseModel):
    """Aspect-level tone observation for one summary."""

    aspect: str
    tone: Literal["positive", "negative", "neutral", "mixed"]
    score: float = Field(
        ge=-1.0,
        le=1.0,
        description=(
            "Signed polarity: -1 very negative, +1 very positive; must be > 0 "
            "when tone is 'positive' and < 0 when tone is 'negative' "
            "(neutral/mixed may take any value in range)."
        ),
    )


class NewsToneDraft(BaseModel):
    """LLM-generated tone classification of a news summary."""

    overall: Literal["positive", "negative", "neutral", "mixed"]
    score: float = Field(
        ge=-1.0,
        le=1.0,
        description=(
            "Signed polarity: -1 very negative, +1 very positive; must be > 0 "
            "when overall is 'positive' and < 0 when overall is 'negative' "
            "(neutral/mixed may take any value in range)."
        ),
    )
    aspects: list[ToneAspect] = Field(default_factory=list)


class NewsTone(NewsToneDraft):
    """Tone result linked back to the article it describes."""

    article_id: str
    headline: str
    source_url: str | None = None
    published_at: str | None = None
    retrieved_at: str


class CompetitorProfileDraft(BaseModel):
    """LLM-extracted competitor fields; sources are added by the application."""

    company: str
    positioning: str
    products: list[str] = Field(default_factory=list)
    pricing_model: str | None = None
    funding_stage: str | None = None
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)


class CompetitorSource(BaseModel):
    """A page actually retrieved by the bounded HTTP client."""

    url: str
    retrieved_at: str
    content_hash: str


class CompetitorProfile(CompetitorProfileDraft):
    """A competitor profile with retrieved source records."""

    sources: list[str] = Field(default_factory=list)
    retrieved_sources: list[CompetitorSource] = Field(default_factory=list)


class PageSource(BaseModel):
    """A page retrieved for one side of a comparison.

    The fetched page text is deliberately not stored: the content hash
    fingerprints it, and quote validation happens during construction against
    the exact capped text that was sent to the model.
    """

    source_id: str
    url: str
    retrieved_at: str
    content_hash: str


class ComparisonEvidence(BaseModel):
    """A model-selected quote tied to one retrieved page."""

    source_id: str
    quote: str


class ComparisonField(BaseModel):
    """One compared attribute.

    ``evidence`` is empty exactly when ``value`` is None: a populated field
    always carries at least one validated citation.
    """

    value: str | None = None
    evidence: list[ComparisonEvidence] = Field(default_factory=list)


class CompanyComparison(BaseModel):
    """One side of the subject-vs-competitor comparison."""

    name: str
    pages: list[PageSource] = Field(default_factory=list)
    product_focus: ComparisonField
    pricing_model: ComparisonField
    positioning: ComparisonField


class Comparison(BaseModel):
    """Subject and competitor compared side by side."""

    subject: CompanyComparison
    competitor: CompanyComparison


class EvidenceCitationDraft(BaseModel):
    """Model output: one quote with the source ID it was copied from."""

    source_id: str = Field(pattern=r"^S[12]$")
    quote: str = Field(min_length=1, max_length=160)


class ComparedFieldDraft(BaseModel):
    """Model output for one field, with at most one citation."""

    value: str | None = Field(default=None, max_length=160)
    evidence: list[EvidenceCitationDraft] = Field(default_factory=list, max_length=1)


class SideComparisonDraft(BaseModel):
    """Model output for one company: the three compared fields."""

    product_focus: ComparedFieldDraft
    pricing_model: ComparedFieldDraft
    positioning: ComparedFieldDraft
