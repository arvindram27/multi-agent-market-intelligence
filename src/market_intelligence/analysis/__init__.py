"""Read-only SQL analysis of stored runs (landmark I2, milestone M1).

M1 ships the ingestion section: per-source volume and freshness, sweep health
(including dedup/replay and error/disabled rates) and version multiplicity.
The pipeline (topics/tones) section and the ``--dataset`` flag are a later
milestone.

The package is psycopg-free; connections come from
:mod:`market_intelligence.db.analytics`, which opens a server-enforced
read-only session.
"""
from market_intelligence.analysis.report import (
    REPORT_VERSION,
    AnalysisReport,
    PipelineReport,
    build_analysis_report,
    build_report,
    render_text,
)

__all__ = [
    "REPORT_VERSION",
    "AnalysisReport",
    "PipelineReport",
    "build_analysis_report",
    "build_report",
    "render_text",
]
