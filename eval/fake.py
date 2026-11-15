"""Fake model and fetch plumbing copied/adapted from the G7 demo tests.

The harness never imports a test module; these helpers are local to eval/.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from xml.sax.saxutils import escape

from market_intelligence.agents.schemas import (
    ComparedFieldDraft,
    EvidenceCitationDraft,
    NewsDigest,
    NewsToneDraft,
    SideComparisonDraft,
    TopicAnalysis,
    TopicFinding,
)
from market_intelligence.admission import PolicyConfig
from market_intelligence.config import settings
from market_intelligence.gatekeeper import Gatekeeper
from market_intelligence.net import FetchResult

USAGE = (10, 5)
COUNTER = 10
RESERVE = 45
MODEL = "gpt-4o-mini"


class Clock:
    def __init__(self):
        self.t = datetime(2026, 11, 15, 0, 0, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.t

    def adv(self, seconds):
        self.t += timedelta(seconds=seconds)


def fixed_token_counter_for_tests_only(prompt):
    return COUNTER


def policy_for(calls: int) -> PolicyConfig:
    return PolicyConfig(
        max_run_tokens=RESERVE * (calls + 2),
        max_run_llm_calls=12,
        max_output_tokens=32,
        max_fetches_per_run=8,
        reservation_margin=0.25,
        provider_overhead_tokens=0,
        allowed_model=MODEL,
        allowed_schemas=frozenset(
            {
                "NewsDigest",
                "TopicAnalysis",
                "NewsToneDraft",
                "SideComparisonDraft",
                "CompetitorProfileDraft",
            }
        ),
    )


def make_gate(store, policy, run_id, thread_id, owner=None, clock=None):
    import uuid

    gate = Gatekeeper(
        store,
        policy,
        run_id=run_id,
        thread_id=thread_id,
        owner_id=owner or str(uuid.uuid4()),
        lease_ttl_seconds=300,
        now=clock or Clock(),
    )
    gate.start_run(str(uuid.uuid4()))
    gate.acquire_lease(str(uuid.uuid4()))
    return gate


class _Runnable:
    def __init__(self, model, schema):
        self._model = model
        self._schema = schema

    def invoke(self, prompt, **kwargs):
        self._model.calls.append(self._schema.__name__)
        value = self._model.next(self._schema)
        if isinstance(value, BaseException):
            raise value
        response = SimpleNamespace(
            llm_output={
                "token_usage": {
                    "prompt_tokens": USAGE[0],
                    "completion_tokens": USAGE[1],
                }
            },
            generations=[],
        )
        for callback in kwargs.get("config", {}).get("callbacks", []):
            callback.on_llm_end(response)
        return value


class UsageLLM:
    """Fake structured model that reports fixed usage through callbacks."""

    def __init__(self, responses):
        self.responses = dict(responses)
        self.calls: list[str] = []

    def next(self, schema):
        value = self.responses[schema]
        if isinstance(value, list):
            return value.pop(0)
        return value

    def with_structured_output(self, schema, **kwargs):
        return _Runnable(self, schema)


def _draft(side: dict) -> SideComparisonDraft:
    def field(name):
        data = side.get(name, {})
        evidence = [
            EvidenceCitationDraft(source_id=item["source_id"], quote=item["quote"])
            for item in data.get("evidence", [])
        ]
        return ComparedFieldDraft(value=data.get("value"), evidence=evidence)

    return SideComparisonDraft(
        product_focus=field("product_focus"),
        pricing_model=field("pricing_model"),
        positioning=field("positioning"),
    )


def responses_from_item(item: dict) -> dict:
    fake = item["fake"]
    news = fake["news"]
    topics = fake["topics"]
    tone = fake["tone"]
    comparison = fake["comparison"]
    return {
        NewsDigest: NewsDigest(**news),
        TopicAnalysis: TopicAnalysis(
            topics=[TopicFinding(**topic) for topic in topics.get("topics", [])]
        ),
        NewsToneDraft: NewsToneDraft(**tone),
        SideComparisonDraft: [_draft(comparison[0]), _draft(comparison[1])],
    }


def responses_from_document(document: dict) -> dict:
    """Build fake drafts from a hand-authored enriched fixture document."""
    summary = document["summaries"][0]
    topic = document["topics"][0] if document.get("topics") else None
    tone = document["tones"][0] if document.get("tones") else None
    comparison = document["comparison"]
    news = {
        "headline": summary["headline"],
        "summary": summary["summary"],
        "key_facts": summary.get("key_facts", []),
        "tone_hint": summary["tone_hint"],
    }
    topics = {"topics": []}
    if topic is not None:
        topics = {
            "topics": [
                {
                    "topic_name": topic["topic_name"],
                    "keywords": topic.get("keywords", []),
                    "model_confidence": topic.get("model_confidence", 0.8),
                    "evidence": [
                        {
                            "article_id": item["article_id"],
                            "quote": item["quote"],
                        }
                        for item in topic.get("evidence", [])
                    ],
                }
            ]
        }
    tone_draft = {
        "overall": tone["overall"],
        "score": tone["score"],
        "aspects": tone.get("aspects", []),
    } if tone is not None else {
        "overall": "neutral",
        "score": 0.0,
        "aspects": [],
    }
    sides = []
    for name in ("subject", "competitor"):
        data = comparison[name]
        sides.append(
            {
                field: {
                    "value": data[field].get("value"),
                    "evidence": data[field].get("evidence", []),
                }
                for field in ("product_focus", "pricing_model", "positioning")
            }
        )
    return {
        NewsDigest: NewsDigest(**news),
        TopicAnalysis: TopicAnalysis(
            topics=[TopicFinding(**topic) for topic in topics["topics"]]
        ),
        NewsToneDraft: NewsToneDraft(**tone_draft),
        SideComparisonDraft: [_draft(sides[0]), _draft(sides[1])],
    }


def feed_xml(article: dict) -> bytes:
    body = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<rss version="2.0"><channel><title>Eval feed</title><item>'
        f"<title>{escape(article['title'])}</title>"
        f"<description>{escape(article['excerpt'])}</description>"
        "</item></channel></rss>"
    )
    return body.encode("utf-8")


def fetch_for_item(item: dict):
    payload = item["input"]
    pages = {page["url"]: page for page in payload["pages"]}

    def _fetch(url, **kwargs):
        if url == payload["rss_url"]:
            return FetchResult(url, feed_xml(payload["articles"][0]), "application/rss+xml", 200)
        page = pages.get(url)
        if page is None:
            raise RuntimeError(f"eval fake fetch has no page for {url!r}")
        body = f"<html><body>{escape(page['text'])}</body></html>".encode("utf-8")
        return FetchResult(url, body, "text/html", 200)

    return _fetch


def apply_settings():
    settings.openai_model = MODEL
    settings.llm_max_output_tokens = 32
