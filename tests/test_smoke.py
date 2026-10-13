"""Smoke test: confirm the ``-m`` entry point imports cleanly with no network calls."""
import importlib


def test_orchestrator_module_imports():
    module = importlib.import_module("orchestration.main_orchestrator_mvp")
    assert callable(getattr(module, "run_ecosystem_mvp"))
    assert callable(getattr(module, "fetch_and_summarize_news"))
    assert callable(getattr(module, "fetch_competitor_info"))
    assert callable(getattr(module, "analyze_text_for_trends"))
    assert callable(getattr(module, "analyze_sentiment_basic"))
