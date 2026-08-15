"""Typed records for the Phase-4 attempt-scoped causal effect ledger.

These records intentionally do not modify or subclass the frozen Phase-1
``EffectRecord``.  They model physical truth, attempt attribution, and support
validity as separate objects.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class AttemptStatus(str, Enum):
    ACTIVE = "ACTIVE"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    SUPERSEDED = "SUPERSEDED"
    INVALIDATED = "INVALIDATED"


class FactState(str, Enum):
    UNKNOWN = "UNKNOWN"
    OBSERVED = "OBSERVED"
    VERIFIED = "VERIFIED"
    REALIZED = "REALIZED"
    REVOKED = "REVOKED"


class ProofValidity(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    REVOKED = "REVOKED"
    SUPERSEDED = "SUPERSEDED"


class SupportValidityType(str, Enum):
    PERSISTENT = "PERSISTENT"
    UNTIL_CHILD_REALIZED = "UNTIL_CHILD_REALIZED"
    UNTIL_EFFECT_REALIZED = "UNTIL_EFFECT_REALIZED"


class LedgerEventKind(str, Enum):
    REQUEST = "REQUEST"
    COMMAND = "COMMAND"
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    WITNESS = "WITNESS"
    CONTRADICTION = "CONTRADICTION"
    EXTERNAL_REALIZATION = "EXTERNAL_REALIZATION"
    SUPPORT_INVALIDATION = "SUPPORT_INVALIDATION"


@dataclass
class AttemptRecord:
    attempt_id: str
    effect_id: str
    generation: int
    command_event_id: Optional[str]
    start_epoch: int
    end_epoch: Optional[int]
    status: AttemptStatus
    evidence_ids: List[str] = field(default_factory=list)
    evidence_digests: List[str] = field(default_factory=list)
    evidence_sensors: Dict[str, str] = field(default_factory=dict)
    attribution_witness_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        value = asdict(self)
        value["status"] = self.status.value
        return value


@dataclass
class EffectFactRecord:
    effect_id: str
    fact_state: FactState = FactState.UNKNOWN
    fact_epoch: int = 0
    active_attempt_id: Optional[str] = None
    verified_fact_observations: List[str] = field(default_factory=list)
    realization_proof_ids: List[str] = field(default_factory=list)
    invalidation_event_id: Optional[str] = None
    physical_truth_confidence: float = 0.0

    def to_dict(self) -> dict:
        value = asdict(self)
        value["fact_state"] = self.fact_state.value
        return value


@dataclass
class RealizationProof:
    proof_id: str
    effect_id: str
    fact_epoch: int
    attributed_attempt_id: Optional[str]
    evidence_ids: List[str]
    witness_id: str
    support_clause_id: Optional[str]
    validity_status: ProofValidity = ProofValidity.VALID

    def to_dict(self) -> dict:
        value = asdict(self)
        value["validity_status"] = self.validity_status.value
        return value


@dataclass(frozen=True)
class SupportReference:
    parent_proof_id: str
    validity_type: SupportValidityType
    until_effect_id: Optional[str] = None
    discharged: bool = False
    discharge_event_id: Optional[str] = None

    def to_dict(self) -> dict:
        value = asdict(self)
        value["validity_type"] = self.validity_type.value
        return value


@dataclass
class SupportClause:
    clause_id: str
    effect_id: str
    references: List[SupportReference] = field(default_factory=list)
    valid: bool = True

    def to_dict(self) -> dict:
        return {
            "clause_id": self.clause_id,
            "effect_id": self.effect_id,
            "references": [item.to_dict() for item in self.references],
            "valid": bool(self.valid),
        }


@dataclass(frozen=True)
class LedgerEvent:
    event_id: str
    epoch: int
    kind: LedgerEventKind
    effect_id: str
    attempt_id: Optional[str] = None
    command_event_id: Optional[str] = None
    evidence_id: Optional[str] = None
    sensor_id: Optional[str] = None
    evidence_digest: Optional[str] = None
    physical_truth: Optional[bool] = None
    parent_ids: Tuple[str, ...] = field(default_factory=tuple)
    external: bool = False
    payload: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        value = asdict(self)
        value["kind"] = self.kind.value
        value["parent_ids"] = list(self.parent_ids)
        return value

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")


def deterministic_id(namespace: str, *parts: object) -> str:
    material = "|".join([namespace] + [str(part) for part in parts])
    return "%s-%s" % (namespace, hashlib.sha256(material.encode("utf-8")).hexdigest()[:20])


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
