"""Bounded fetcher tests: SSRF guards, redirects, status, type and size limits."""
import ipaddress

import pytest
import requests

from market_intelligence import net
from market_intelligence.net import (
    PermanentFetchError,
    TransientFetchError,
    fetch_bytes,
    validate_public_url,
)


class FakeResponse:
    def __init__(self, status_code=200, headers=None, body=b"", chunks=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._chunks = list(chunks) if chunks is not None else ([body] if body else [])
        self.closed = False

    def iter_content(self, chunk_size=8192):
        yield from self._chunks

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.closed = False

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    """Resolve example.com publicly and literal IPs without touching DNS."""
    original = net._resolve_public_ips

    def fake(host, port):
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            ip = None
        if ip is not None:
            if not ip.is_global:
                raise PermanentFetchError(
                    f"Refusing non-public address {ip} for host {host!r}."
                )
            return {host}
        if host.endswith("example.com"):
            return {"93.184.216.34"}
        return original(host, port)

    monkeypatch.setattr(net, "_resolve_public_ips", fake)


def _fetch(url="https://example.com/feed", session=None, **kwargs):
    return fetch_bytes(
        url,
        timeout=kwargs.pop("timeout", 5.0),
        max_bytes=kwargs.pop("max_bytes", 1024),
        allowed_content_types=kwargs.pop(
            "allowed_content_types", {"application/rss+xml"}
        ),
        session=session,
        **kwargs,
    )


def test_rejects_non_http_schemes():
    with pytest.raises(PermanentFetchError, match="scheme"):
        _fetch("file:///etc/passwd")
    with pytest.raises(PermanentFetchError, match="scheme"):
        _fetch("ftp://example.com/feed")


def test_rejects_embedded_credentials():
    with pytest.raises(PermanentFetchError, match="credentials"):
        _fetch("https://user:secret@example.com/feed")  # pragma: allowlist secret


def test_rejects_ambiguous_urls():
    for url in (
        "https://example.com\\@127.0.0.1/feed",
        "https://example.com\n@127.0.0.1/feed",
    ):
        with pytest.raises(PermanentFetchError, match="whitespace or backslashes"):
            _fetch(url)


def test_rejects_nonstandard_ports():
    with pytest.raises(PermanentFetchError, match="Port"):
        _fetch("https://example.com:8080/feed")


def test_rejects_loopback_and_link_local_addresses():
    with pytest.raises(PermanentFetchError, match="non-public"):
        validate_public_url("http://127.0.0.1/feed")
    with pytest.raises(PermanentFetchError, match="non-public"):
        validate_public_url("http://[::1]/feed")
    with pytest.raises(PermanentFetchError, match="non-public"):
        validate_public_url("http://169.254.169.254/latest/meta-data")


def test_fetch_applies_timeout_and_no_redirects():
    response = FakeResponse(
        200, {"Content-Type": "application/rss+xml; charset=utf-8"}, b"<rss/>"
    )
    session = FakeSession([response])

    result = _fetch(session=session, timeout=2.5)

    assert result.content == b"<rss/>"
    assert result.content_type == "application/rss+xml"
    assert session.calls[0][1]["timeout"] == 2.5
    assert session.calls[0][1]["allow_redirects"] is False
    assert "User-Agent" in session.calls[0][1]["headers"]
    assert response.closed is True


def test_rejects_missing_or_unexpected_content_type():
    for content_type in (None, "text/html", "application/json"):
        headers = {} if content_type is None else {"Content-Type": content_type}
        session = FakeSession([FakeResponse(200, headers, b"data")])
        with pytest.raises(PermanentFetchError, match="content type"):
            _fetch(session=session)


def test_rejects_declared_oversize_response():
    session = FakeSession(
        [
            FakeResponse(
                200,
                {"Content-Type": "application/rss+xml", "Content-Length": "9999"},
                b"",
            )
        ]
    )
    with pytest.raises(PermanentFetchError, match="9999"):
        _fetch(session=session, max_bytes=100)


def test_rejects_streamed_oversize_response():
    session = FakeSession(
        [
            FakeResponse(
                200,
                {"Content-Type": "application/rss+xml"},
                chunks=[b"a" * 8, b"b" * 8],
            )
        ]
    )
    with pytest.raises(PermanentFetchError, match="exceeded"):
        _fetch(session=session, max_bytes=10)


@pytest.mark.parametrize("status", [429, 500, 503])
def test_rate_limits_and_server_errors_are_transient(status):
    session = FakeSession([FakeResponse(status, {}, b"")])
    with pytest.raises(TransientFetchError, match=str(status)):
        _fetch(session=session)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 410])
def test_client_errors_are_permanent(status):
    session = FakeSession([FakeResponse(status, {}, b"")])
    with pytest.raises(PermanentFetchError, match=str(status)):
        _fetch(session=session)


def test_timeouts_and_connection_errors_are_transient():
    with pytest.raises(TransientFetchError, match="timed out"):
        _fetch(session=FakeSession([requests.Timeout()]))
    with pytest.raises(TransientFetchError, match="Request failed"):
        _fetch(session=FakeSession([requests.ConnectionError("reset")]))


def test_follows_and_validates_redirects():
    redirect = FakeResponse(302, {"Location": "https://cdn.example.com/feed"})
    final = FakeResponse(200, {"Content-Type": "application/rss+xml"}, b"<rss/>")
    session = FakeSession([redirect, final])

    result = _fetch(session=session)

    assert result.url == "https://cdn.example.com/feed"
    assert len(session.calls) == 2
    assert redirect.closed is True


def test_rejects_redirect_to_private_address():
    session = FakeSession(
        [FakeResponse(302, {"Location": "http://127.0.0.1/secret"})]
    )
    with pytest.raises(PermanentFetchError, match="non-public"):
        _fetch(session=session)
    assert len(session.calls) == 1


def test_redirect_without_location_is_permanent():
    session = FakeSession([FakeResponse(302, {})])
    with pytest.raises(PermanentFetchError, match="Location"):
        _fetch(session=session)


def test_redirect_limit_is_enforced():
    responses = [
        FakeResponse(302, {"Location": "https://example.com/again"}) for _ in range(3)
    ]
    session = FakeSession(responses)
    with pytest.raises(PermanentFetchError, match="redirect"):
        _fetch(session=session, max_redirects=2)
