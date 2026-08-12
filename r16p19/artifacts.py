"""Deterministic artifact writers and input-integrity helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, body: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp.%d" % (path.name, os.getpid()))
    descriptor = os.open(
        str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    try:
        os.write(descriptor, body.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(str(temporary), str(path))


def write_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    body = "".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    atomic_text(path, body)


def write_sha256sums(root: Path, names: Iterable[str], output_name: str = "SHA256SUMS") -> None:
    root = Path(root)
    lines = []
    for name in sorted(set(names)):
        target = root / name
        if target.name == output_name or not target.is_file():
            continue
        lines.append("%s  %s" % (sha256_file(target), name))
    atomic_text(root / output_name, "\n".join(lines) + "\n")
