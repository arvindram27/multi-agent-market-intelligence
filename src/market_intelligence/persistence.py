"""Business-persistence seam for approved runs.

The graph treats persistence as an injected dependency so tests can prove that a
rejected run never reaches it. This module defines the contract and an explicit
local placeholder; the durable SQLite writer (and, later, an Oracle adapter) plug
into the same protocol.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from market_intelligence.graph.state import MarketIntelligenceState


class ProfileWriter(Protocol):
    """Persists an approved run and returns its durable identifier."""

    def persist(self, run_id: str, state: "MarketIntelligenceState") -> str:
        """Write the approved profile and related records for ``run_id``."""
        ...


class NullProfileWriter:
    """Explicit placeholder: records nothing and echoes the run id.

    Used by the demo and offline tests until the data layer lands. It exists so
    the graph can require a successful writer call before reporting ``complete``
    without pretending that data was stored.
    """

    def persist(self, run_id: str, state: "MarketIntelligenceState") -> str:
        return run_id
