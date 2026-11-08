"""Feed source configuration: TOML loading and structural validation.

Validation is structural only — required keys, supported kinds, http(s)
schemes, duplicates, unknown keys and a non-empty list. There is no DNS or
network check at load time: ``net.fetch_bytes`` performs the per-request
validation, so config loading is offline and testable.
"""
from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

ALLOWED_KINDS = frozenset({"rss", "atom"})
ALLOWED_FIELDS = frozenset({"key", "kind", "label", "url", "enabled"})
REQUIRED_FIELDS = ("key", "kind", "label", "url")


class SourceConfigError(ValueError):
    """The feed source configuration is invalid (not a per-source fetch error)."""


@dataclass(frozen=True)
class FeedSource:
    """One configured feed. ``source_key`` is its stable identity."""

    source_key: str
    kind: str
    url: str
    label: str
    enabled: bool = True


def load_sources(path: str | Path) -> list[FeedSource]:
    """Read a TOML source list and validate it, raising SourceConfigError."""
    config_path = Path(path)
    try:
        with open(config_path, "rb") as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise SourceConfigError(
            f"invalid TOML in {config_path}: {exc}"
        ) from exc
    except OSError as exc:
        raise SourceConfigError(
            f"could not read source config {config_path}: {exc}"
        ) from exc
    return validate_sources(data)


def validate_sources(data: Mapping) -> list[FeedSource]:
    """Validate a parsed config mapping and return FeedSource values in order."""
    if not isinstance(data, dict):
        raise SourceConfigError("source config must be a TOML table")
    unknown_top_level = sorted(set(data) - {"sources"})
    if unknown_top_level:
        raise SourceConfigError(f"unknown top-level keys: {unknown_top_level}")

    entries = data.get("sources")
    if not isinstance(entries, list) or not entries:
        raise SourceConfigError("source config must contain a non-empty [[sources]] list")

    sources: list[FeedSource] = []
    seen_keys: set[str] = set()
    seen_urls: set[str] = set()
    for position, raw in enumerate(entries, start=1):
        where = f"sources[{position}]"
        if not isinstance(raw, dict):
            raise SourceConfigError(f"{where} must be a table, got {type(raw).__name__}")

        unknown = sorted(set(raw) - ALLOWED_FIELDS)
        if unknown:
            raise SourceConfigError(f"{where}: unknown keys {unknown}")
        missing = [name for name in REQUIRED_FIELDS if name not in raw]
        if missing:
            raise SourceConfigError(f"{where}: missing required keys {missing}")

        key = raw["key"]
        kind = raw["kind"]
        url = raw["url"]
        label = raw["label"]
        enabled = raw.get("enabled", True)

        for name, value in (
            ("key", key),
            ("kind", kind),
            ("url", url),
            ("label", label),
        ):
            if not isinstance(value, str) or not value.strip():
                raise SourceConfigError(f"{where}: {name} must be a non-empty string")
        if not isinstance(enabled, bool):
            raise SourceConfigError(f"{where}: enabled must be a boolean, got {enabled!r}")
        if kind not in ALLOWED_KINDS:
            raise SourceConfigError(
                f"{where}: kind must be one of {sorted(ALLOWED_KINDS)}, got {kind!r}"
            )
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise SourceConfigError(f"{where}: url must be http(s) with a host, got {url!r}")
        if key in seen_keys:
            raise SourceConfigError(f"{where}: duplicate source_key {key!r}")
        if url in seen_urls:
            raise SourceConfigError(f"{where}: duplicate url {url!r}")

        seen_keys.add(key)
        seen_urls.add(url)
        sources.append(
            FeedSource(
                source_key=key.strip(),
                kind=kind,
                url=url.strip(),
                label=label.strip(),
                enabled=enabled,
            )
        )
    return sources
