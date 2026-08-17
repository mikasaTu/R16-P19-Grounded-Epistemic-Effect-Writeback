"""Append-only hash-chained cold audit store with compact summaries."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass
class AuditStore:
    keep_rows: bool = False
    head: str = "0" * 64
    count: int = 0
    kind_counts: Dict[str, int] = field(default_factory=dict)
    accepted_counts: Dict[str, int] = field(default_factory=dict)
    rows: List[dict] = field(default_factory=list)
    receipt_digests: Dict[str, str] = field(default_factory=dict)

    def receipt_status(self, receipt_id: str, digest: str) -> str:
        previous = self.receipt_digests.get(receipt_id)
        if previous is None:
            self.receipt_digests[receipt_id] = digest
            return "new"
        return "duplicate" if previous == digest else "collision"

    def append(self, row: dict) -> str:
        body = dict(row)
        body["previous_hash"] = self.head
        digest = hashlib.sha256(_canonical(body)).hexdigest()
        body["row_hash"] = digest
        self.head = digest
        self.count += 1
        kind = str(row.get("kind", "UNKNOWN"))
        self.kind_counts[kind] = self.kind_counts.get(kind, 0) + 1
        accepted = str(bool(row.get("accepted", False))).lower()
        self.accepted_counts[accepted] = self.accepted_counts.get(accepted, 0) + 1
        if self.keep_rows:
            self.rows.append(body)
        return digest

    def compact_summary(self) -> dict:
        return {
            "event_count": self.count,
            "chain_head": self.head,
            "kind_counts": dict(sorted(self.kind_counts.items())),
            "accepted_counts": dict(sorted(self.accepted_counts.items())),
        }

    def verify(self) -> bool:
        if not self.keep_rows:
            return True
        previous = "0" * 64
        for row in self.rows:
            body = dict(row)
            claimed = body.pop("row_hash")
            if body.get("previous_hash") != previous:
                return False
            actual = hashlib.sha256(_canonical(body)).hexdigest()
            if actual != claimed:
                return False
            previous = claimed
        return previous == self.head
