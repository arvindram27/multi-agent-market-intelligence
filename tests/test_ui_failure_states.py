"""UI failure states: every component renders an explicit empty state, no crash."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from pytest import mark  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

EMPTY_APP = Path(__file__).resolve().parent / "ui_empty_app.py"


@mark.enable_socket  # asyncio's event loop creates a local socketpair; no network I/O
def test_components_render_explicit_empty_states():
    app = AppTest.from_file(str(EMPTY_APP), default_timeout=30)
    app.run()

    assert not app.exception
    messages = [info.value for info in app.info]
    assert any("No articles yet" in message for message in messages)
    assert any("No topics yet" in message for message in messages)
    assert any("No tone results yet" in message for message in messages)
    assert any("No competitor profile yet" in message for message in messages)


@mark.enable_socket
def test_view_from_snapshot_tolerates_empty_graph_state():
    from types import SimpleNamespace

    from ui.render import _describe, view_from_snapshot

    snapshot = SimpleNamespace(values=None, next=())
    view = view_from_snapshot(snapshot)
    assert view["summaries"] == []
    assert view["topics"] == []
    assert view["tones"] == []
    assert view["competitor"] is None
    assert view["pending"] is False

    # Unknown or empty node updates describe themselves neutrally.
    assert _describe("unknown_node", {}) == "done"
    assert _describe("persist", {"persisted_run_id": None}) == "failed"
