"""Headless render test for the Streamlit demo (fixture mode, no network)."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "app.py"


@pytest.mark.enable_socket  # asyncio's event loop creates a local socketpair; no network I/O
def test_demo_app_renders_and_approves(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "1")
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.run()

    assert not app.exception
    assert len(app.tabs) == 4
    assert any(button.label == "Approve profile" for button in app.button)

    next(
        button for button in app.button if button.label == "Approve profile"
    ).click().run()

    assert not app.exception
    assert any("profile approved" in item.value.lower() for item in app.success)
