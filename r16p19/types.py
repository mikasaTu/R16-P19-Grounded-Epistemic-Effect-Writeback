"""Typed epistemic state, event, evidence, and decision records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class EpistemicState(str, Enum):
    REQUESTED = "REQUESTED"
    IMAGINED = "IMAGINED"
    OBSERVED = "OBSERVED"
    VERIFIED = "VERIFIED"
    REALIZED = "REALIZED"
    STALLED = "STALLED"
    INVALIDATED_REALIZATION = "INVALIDATED_REALIZATION"


class EventType(str, Enum):
    REQUEST = "REQUEST"
    IMAGINE = "IMAGINE"
    COMMAND = "COMMAND"
    OBSERVE_POSITIVE = "OBSERVE_POSITIVE"
    VERIFY_POSITIVE = "VERIFY_POSITIVE"
    REALIZATION_WITNESS = "REALIZATION_WITNESS"
    OBSERVE_NEGATIVE = "OBSERVE_NEGATIVE"
    CONTRADICTION = "CONTRADICTION"
    TIMEOUT = "TIMEOUT"
    IRRELEVANT = "IRRELEVANT"


class Decision(str, Enum):
    ADVANCE_TO_NEXT_SUBTASK = "ADVANCE_TO_NEXT_SUBTASK"
    RETRY_CURRENT_EFFECT = "RETRY_CURRENT_EFFECT"
    REOBSERVE = "REOBSERVE"
    ROLLBACK_OR_REPLAN = "ROLLBACK_OR_REPLAN"
    SAFE_STOP = "SAFE_STOP"


@dataclass(frozen=True)
class EvidenceReceipt:
    evidence_id: str
    episode_id: str
    event_index: int
    timestamp: float
    sensor_identity: str
    frame_digest: str
    effect_id: str
    evidence_type: str

    def validate(self) -> None:
        required = (
            self.evidence_id,
            self.episode_id,
            self.sensor_identity,
            self.frame_digest,
            self.effect_id,
            self.evidence_type,
        )
        if any(not value for value in required):
            raise ValueError("evidence receipt contains an empty required field")
        if self.event_index < 0 or self.timestamp < 0:
            raise ValueError("evidence receipt index and timestamp must be non-negative")
        if len(self.frame_digest) != 64:
            raise ValueError("frame_digest must be a SHA256 hex digest")
        int(self.frame_digest, 16)

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class Event:
    event_id: str
    episode_id: str
    event_index: int
    timestamp: float
    event_type: EventType
    effect_id: Optional[str]
    parent_ids: Tuple[str, ...] = field(default_factory=tuple)
    receipt: Optional[EvidenceReceipt] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.event_id or not self.episode_id:
            raise ValueError("event identity cannot be empty")
        if self.event_index < 0 or self.timestamp < 0:
            raise ValueError("event index and timestamp must be non-negative")
        if self.event_type != EventType.IRRELEVANT and not self.effect_id:
            raise ValueError("non-irrelevant event requires effect_id")
        if self.receipt is not None:
            self.receipt.validate()
            if self.receipt.effect_id != self.effect_id:
                raise ValueError("event and receipt effect IDs differ")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        value = asdict(self)
        value["event_type"] = self.event_type.value
        if self.receipt is not None:
            value["receipt"] = self.receipt.to_dict()
        return value

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")


@dataclass
class EffectRecord:
    effect_id: str
    state: Optional[EpistemicState] = None
    command_event_id: Optional[str] = None
    evidence_digests: Dict[str, str] = field(default_factory=dict)
    evidence_ids: List[str] = field(default_factory=list)
    recovery_route: List[str] = field(default_factory=list)
    invalidated_by: Optional[str] = None
    blocked_dependents: List[str] = field(default_factory=list)
    duplicate_contradictions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["state"] = self.state.value if self.state is not None else None
        return value


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_stream_sha256(events: List[Event]) -> str:
    digest = hashlib.sha256()
    for event in events:
        digest.update(event.canonical_bytes())
        digest.update(b"\n")
    return digest.hexdigest()

