"""Typed boundary records for the bounded Phase-5 ASCEL bridge."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class EvidenceReceipt:
    receipt_id: str
    attempt_id: str
    command_id: str
    effect_id: str
    source_id: str
    source_version: int
    observed_at: int
    expires_at: int
    payload_hash: str
    confidence: float
    truth_value: bool
    active_attempt_credit: bool
    capture_id: str = ""
    sensor_source: str = ""
    source_group: str = ""
    correlation_group: str = ""
    frame_digest: str = ""
    timestamp: float = 0.0
    verifier_model_version: str = "oracle-v1"
    calibration_version: str = "frozen-v1"
    parent_ids: Tuple[str, ...] = ()
    realization_kind: str = "attempt"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def valid_at(self, epoch: int) -> bool:
        return self.observed_at <= epoch <= self.expires_at

    def canonical_source(self) -> str:
        return self.sensor_source or self.source_id

    def independent_key(self) -> tuple[str, str, str]:
        return (
            self.correlation_group or self.source_group or self.source_id,
            self.capture_id or self.receipt_id,
            self.frame_digest or self.payload_hash,
        )


@dataclass(frozen=True)
class PolicyRequest:
    observation_hash: str
    history_hash: str
    task_id: str
    effect_id: str
    mode: str
    checkpoint_hash: str
    config_hash: str
    policy_seed: int

    def key(self) -> str:
        import hashlib
        import json

        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LedgerDecision:
    effect_truth: bool
    active_attempt_credit: bool
    accepted: bool
    reason: str
    proof_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
