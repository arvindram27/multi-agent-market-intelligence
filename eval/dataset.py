"""Dataset and manifest loading for the offline evaluation harness."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"
DATASETS = EVAL_DIR / "datasets"
FIXTURES = EVAL_DIR / "fixtures"
REPORTS = EVAL_DIR / "reports"

ALLOWED_LICENSES = frozenset({"self_authored", "licensed", "permitted_excerpt"})
MAX_TEXT = 320


class EvalDataError(RuntimeError):
    """The dataset, fixtures or manifest are invalid."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(_read_bytes(path))


def dataset_fingerprint(dev_bytes: bytes, heldout_bytes: bytes) -> str:
    return sha256_bytes(dev_bytes + heldout_bytes)


_READ_LOG = []


def reset_read_log() -> None:
    _READ_LOG.clear()


def read_log() -> list:
    return [dict(entry) for entry in _READ_LOG]


def _record_read(path, kind, outcome) -> None:
    _READ_LOG.append({"path": str(path), "kind": kind, "outcome": outcome})


def _read_bytes(path: Path) -> bytes:
    path = Path(path).resolve()
    _record_read(path, "hash", "attempted")
    try:
        data = path.read_bytes()
    except OSError:
        _record_read(path, "hash", "open_failed")
        raise
    _record_read(path, "hash", "completed")
    return data


def _read_jsonl(path: Path) -> list[dict]:
    path = Path(path).resolve()
    _record_read(path, "parse", "attempted")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        _record_read(path, "parse", "open_failed")
        raise
    except UnicodeDecodeError:
        _record_read(path, "parse", "decode_failed")
        raise
    items = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except ValueError as exc:
            _record_read(path, "parse", "parse_failed")
            raise EvalDataError(f"{path.name} line {number} is not JSON: {exc}") from exc
        if not isinstance(value, dict):
            _record_read(path, "parse", "parse_failed")
            raise EvalDataError(f"{path.name} line {number} is not an object")
        items.append(value)
    _record_read(path, "parse", "completed")
    return items


def load_jsonl(path: Path) -> list[dict]:
    return _read_jsonl(Path(path))


def load_manifest(root: Path = ROOT) -> dict:
    path = Path(root) / "eval" / "manifest.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EvalDataError(f"manifest is unreadable: {exc}") from exc


def verify_manifest(root: Path = ROOT) -> dict:
    root = Path(root)
    manifest = load_manifest(root)
    for relative, expected in manifest.get("files", {}).items():
        path = root / "eval" / relative
        if not path.is_file():
            raise EvalDataError(f"missing data file: {relative}")
        actual = sha256_file(path)
        if actual.lower() != expected.lower():
            raise EvalDataError(f"hash mismatch for {relative}: {actual} != {expected}")
    dev = load_jsonl(root / "eval" / "datasets" / "dev.jsonl")
    counts = manifest.get("counts", {})
    if counts.get("dev") != len(dev):
        raise EvalDataError("manifest dev item count does not match the dataset file")
    fingerprint = dataset_fingerprint(
        _read_bytes(root / "eval" / "datasets" / "dev.jsonl"),
        _read_bytes(root / "eval" / "datasets" / "heldout.jsonl"),
    )
    if (manifest.get("dataset_sha256") or "").lower() != fingerprint.lower():
        raise EvalDataError("manifest dataset fingerprint does not match")
    return manifest


def _check_text(value: str, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise EvalDataError(f"{label} must be a non-empty string")
    if len(value) > MAX_TEXT:
        raise EvalDataError(f"{label} exceeds {MAX_TEXT} characters")


def validate_items(items: list[dict], split: str) -> None:
    seen = set()
    for item in items:
        for key in ("item_id", "split", "input", "fake", "expected", "provenance"):
            if key not in item:
                raise EvalDataError(f"{split} item is missing {key}")
        if item["split"] != split:
            raise EvalDataError(f"{item['item_id']} has split {item['split']!r}, expected {split!r}")
        if item["item_id"] in seen:
            raise EvalDataError(f"duplicate item id {item['item_id']}")
        seen.add(item["item_id"])
        payload = item["input"]
        for article in payload.get("articles", []):
            license_info = article.get("license", {})
            if license_info.get("kind") not in ALLOWED_LICENSES:
                raise EvalDataError(f"{item['item_id']} has an unpermitted license kind")
            if license_info.get("redistributable") is not True:
                raise EvalDataError(f"{item['item_id']} article is not redistributable")
            _check_text(article.get("excerpt", ""), f"{item['item_id']} excerpt")
            _check_text(article.get("title", ""), f"{item['item_id']} title")
        for page in payload.get("pages", []):
            _check_text(page.get("text", ""), f"{item['item_id']} page text")
        if len(payload.get("articles", [])) != payload.get("num_articles"):
            raise EvalDataError(f"{item['item_id']} num_articles does not match its articles")
        if not 1 <= len(payload.get("articles", [])) <= 3:
            raise EvalDataError(f"{item['item_id']} must have 1-3 articles")
        provenance = item["provenance"]
        for key in ("label_author", "labeled_at", "rationale", "source_reference", "review"):
            if not provenance.get(key):
                raise EvalDataError(f"{item['item_id']} is missing provenance.{key}")


def load_dataset(context: str = "dev", root: Path = ROOT):
    """Return (items, opened_paths). The held-out file is deserialized only for final."""
    root = Path(root)
    opened = []
    dev_path = root / "eval" / "datasets" / "dev.jsonl"
    dev = _read_jsonl(dev_path)
    opened.append(str(dev_path))
    validate_items(dev, "dev")
    if context == "dev":
        return dev, opened
    if context != "final":
        raise EvalDataError(f"unknown context {context!r}")
    heldout_path = root / "eval" / "datasets" / "heldout.jsonl"
    heldout = _read_jsonl(heldout_path)
    opened.append(str(heldout_path))
    validate_items(heldout, "heldout")
    counts = load_manifest(root).get("counts", {})
    if counts.get("dev") != len(dev) or counts.get("heldout") != len(heldout):
        raise EvalDataError("manifest item counts do not match the dataset files")
    return dev + heldout, opened


def load_fixtures(name: str, root: Path = ROOT) -> list[dict]:
    return _read_jsonl(Path(root) / "eval" / "fixtures" / f"{name}.jsonl")
