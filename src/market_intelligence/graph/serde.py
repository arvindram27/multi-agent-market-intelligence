"""Checkpoint serializer allowlist for the Pydantic models kept in graph state.

LangGraph warns (and will refuse in a future release) when deserializing msgpack
types that are not explicitly allowlisted. Savers created for this project should
use :func:`checkpoint_serde`.
"""
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from market_intelligence.agents import schemas

ALLOWED_MSGPACK_MODULES = [
    (schemas.__name__, name)
    for name in (
        "NewsSummary",
        "TopicReport",
        "NewsTone",
        "CompetitorProfile",
    )
]


def checkpoint_serde() -> JsonPlusSerializer:
    """Return a serializer that allowlists the agent output models."""
    return JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MSGPACK_MODULES)
