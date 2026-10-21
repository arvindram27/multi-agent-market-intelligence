"""Bounded HTTP fetching with explicit failure classes and SSRF guards.

Every outbound fetch goes through :func:`fetch_bytes`, which:

* allows only ``http``/``https`` URLs without embedded credentials;
* resolves the host and refuses non-public addresses (loopback, private,
  link-local, reserved, multicast);
* re-validates every redirect target instead of letting ``requests`` follow it;
* enforces a wall-clock timeout, redirect limit, content-type allowlist and
  maximum body size before the bytes reach a parser or an LLM.

Residual risk: validation resolves DNS and the HTTP client resolves again, so a
determined DNS-rebinding attack could still swap the address between the two
lookups. Pin the connection to the validated IP before exposing this to the
public internet; the guard here is appropriate for a local/single-tenant demo.
"""
from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import requests

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 MarketIntelligence/0.1"
)
ALLOWED_SCHEMES = frozenset({"http", "https"})
ALLOWED_PORTS = frozenset({None, 80, 443})
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


class FetchError(Exception):
    """Base class for failures raised by the bounded fetcher."""


class PermanentFetchError(FetchError):
    """The input or response is invalid; retrying cannot help."""


class TransientFetchError(FetchError):
    """A timeout, rate limit, or upstream failure that may succeed on retry."""


@dataclass(frozen=True)
class FetchResult:
    """A validated response body and the URL that finally produced it."""

    url: str
    content: bytes
    content_type: str
    status_code: int


def _resolve_public_ips(host: str, port: int) -> set[str]:
    """Resolve ``host`` and reject if any address is not globally routable."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise TransientFetchError(f"Could not resolve host {host!r}.") from exc

    addresses = {info[4][0].split("%", 1)[0] for info in infos}
    if not addresses:
        raise PermanentFetchError(f"Host {host!r} resolved to no addresses.")
    for address in addresses:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError as exc:
            raise PermanentFetchError(
                f"Host {host!r} resolved to an invalid address."
            ) from exc
        if not ip.is_global:
            raise PermanentFetchError(
                f"Refusing non-public address {ip} for host {host!r}."
            )
    return addresses


def validate_public_url(url: str) -> str:
    """Validate scheme, credentials, port and resolved addresses.

    Returns the URL unchanged for convenient chaining. Raises
    :class:`PermanentFetchError` for anything that is not a public HTTP(S) URL.
    """
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise PermanentFetchError(f"Invalid URL {url!r}: {exc}") from exc

    if any(char.isspace() for char in url) or "\\" in url:
        raise PermanentFetchError(
            "URLs containing whitespace or backslashes are not allowed."
        )
    if parts.scheme not in ALLOWED_SCHEMES:
        raise PermanentFetchError(
            f"Unsupported URL scheme {parts.scheme!r}; only http and https are allowed."
        )
    if not parts.hostname:
        raise PermanentFetchError(f"URL {url!r} has no host.")
    if parts.username is not None or parts.password is not None:
        raise PermanentFetchError("URLs with embedded credentials are not allowed.")
    if port not in ALLOWED_PORTS:
        raise PermanentFetchError(
            f"Port {port} is not allowed; use 80, 443, or the scheme default."
        )

    _resolve_public_ips(parts.hostname, port or (443 if parts.scheme == "https" else 80))
    return url


def _media_type(value: str | None) -> str:
    return (value or "").split(";", 1)[0].strip().lower()


def _check_status(status_code: int) -> None:
    if 200 <= status_code < 300:
        return
    if status_code == 429 or 500 <= status_code < 600:
        raise TransientFetchError(f"Upstream returned HTTP {status_code}.")
    raise PermanentFetchError(f"Upstream returned HTTP {status_code}.")


def _read_limited(response, max_bytes: int) -> bytes:
    declared = response.headers.get("Content-Length")
    if declared:
        try:
            if int(declared) > max_bytes:
                raise PermanentFetchError(
                    f"Response declares {declared} bytes, above the {max_bytes}-byte limit."
                )
        except ValueError:
            pass  # a malformed header is handled by the streaming check below

    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=8192):
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            raise PermanentFetchError(
                f"Response exceeded the {max_bytes}-byte limit."
            )
        chunks.append(chunk)
    return b"".join(chunks)


def fetch_bytes(
    url: str,
    *,
    timeout: float,
    max_bytes: int,
    allowed_content_types: set[str] | frozenset[str],
    max_redirects: int = 3,
    session: requests.Session | None = None,
) -> FetchResult:
    """Fetch ``url`` with timeout, size, redirect and SSRF limits applied."""
    if timeout <= 0:
        raise ValueError("timeout must be positive.")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive.")
    if max_redirects < 0:
        raise ValueError("max_redirects must be non-negative.")

    allowed = {item.lower() for item in allowed_content_types}
    owns_session = session is None
    session = session or requests.Session()
    current_url = url

    try:
        for _ in range(max_redirects + 1):
            validate_public_url(current_url)
            try:
                response = session.get(
                    current_url,
                    timeout=timeout,
                    allow_redirects=False,
                    stream=True,
                    headers={"User-Agent": USER_AGENT},
                )
            except requests.Timeout as exc:
                raise TransientFetchError(
                    f"Request timed out after {timeout} seconds."
                ) from exc
            except requests.RequestException as exc:
                raise TransientFetchError(f"Request failed: {exc}") from exc

            try:
                if response.status_code in REDIRECT_STATUSES:
                    location = response.headers.get("Location")
                    if not location:
                        raise PermanentFetchError(
                            f"Redirect {response.status_code} had no Location header."
                        )
                    current_url = urljoin(current_url, location)
                    continue

                _check_status(response.status_code)
                content_type = _media_type(response.headers.get("Content-Type"))
                if content_type not in allowed:
                    raise PermanentFetchError(
                        f"Unsupported content type {content_type or '(missing)'}; "
                        f"expected one of {sorted(allowed)}."
                    )
                content = _read_limited(response, max_bytes)
                return FetchResult(
                    url=current_url,
                    content=content,
                    content_type=content_type,
                    status_code=response.status_code,
                )
            finally:
                response.close()
    finally:
        if owns_session:
            session.close()

    raise PermanentFetchError(f"More than {max_redirects} redirects.")
