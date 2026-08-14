"""Shared deterministic receipt/event broker for Phase-3 replay rollouts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

from .types import EvidenceReceipt, Event, EventType, canonical_stream_sha256


def _identifier(*parts: object) -> str:
    return hashlib.sha256("|".join(str(value) for value in parts).encode()).hexdigest()


@dataclass(frozen=True)
class BrokerEventRecord:
    decision_index: int
    event: Event

    def to_dict(self) -> dict:
        raw = self.event.canonical_bytes()
        return {
            "decision_index": self.decision_index,
            "event": self.event.to_dict(),
            "event_bytes_hex": raw.hex(),
            "event_sha256": hashlib.sha256(raw).hexdigest(),
        }


class Phase3EventBroker:
    """Create standardized events without exposing simulator truth to arms."""

    def __init__(self, unit_id: str, episode_id: str):
        self.unit_id = str(unit_id)
        self.episode_id = str(episode_id)
        self.event_index = 0
        self.records: List[BrokerEventRecord] = []
        self.command_event_ids: Dict[str, str] = {}

    def emit(
        self,
        event_type: EventType | str,
        effect_id: str | None,
        decision_index: int,
        *,
        sensor_identity: str | None = None,
        frame_digest: str | None = None,
        parent_ids: Sequence[str] = (),
        payload: Mapping[str, object] | None = None,
    ) -> Event:
        kind = EventType(event_type)
        index = self.event_index
        self.event_index += 1
        event_id = "evt-" + _identifier(
            self.unit_id, index, kind.value, effect_id or "none", sensor_identity or "none"
        )[:32]
        receipt = None
        if sensor_identity is not None:
            if effect_id is None or frame_digest is None:
                raise ValueError("receipt requires effect and frame digest")
            receipt = EvidenceReceipt(
                evidence_id="rcpt-" + _identifier(event_id, sensor_identity)[:32],
                episode_id=self.episode_id,
                event_index=index,
                timestamp=float(index),
                sensor_identity=str(sensor_identity),
                frame_digest=str(frame_digest),
                effect_id=effect_id,
                evidence_type=kind.value,
            )
        safe_payload = dict(payload or {})
        # The high-level tick is public scheduling metadata, not simulator
        # truth.  It lets persistence baselines count one positive tick even
        # when the standardized bundle contains multiple sensor receipts.
        safe_payload.setdefault("decision_index", int(decision_index))
        forbidden = {"physical_truth", "fault_identity", "condition"}.intersection(
            safe_payload
        )
        if forbidden:
            raise ValueError("broker payload leaks forbidden field(s): %s" % sorted(forbidden))
        event = Event(
            event_id=event_id,
            episode_id=self.episode_id,
            event_index=index,
            timestamp=float(index),
            event_type=kind,
            effect_id=effect_id,
            parent_ids=tuple(parent_ids),
            receipt=receipt,
            payload=safe_payload,
        )
        event.validate()
        self.records.append(BrokerEventRecord(int(decision_index), event))
        if kind == EventType.COMMAND and effect_id is not None:
            self.command_event_ids[effect_id] = event_id
        return event

    def request(self, effect_id: str, decision_index: int) -> Event:
        return self.emit(EventType.REQUEST, effect_id, decision_index)

    def imagine_success(self, effect_id: str, decision_index: int) -> Event:
        return self.emit(
            EventType.IMAGINE,
            effect_id,
            decision_index,
            payload={"confidence": "high", "source": "shared_imagination_model"},
        )

    def command(self, effect_id: str, decision_index: int) -> Event:
        return self.emit(EventType.COMMAND, effect_id, decision_index)

    def positive_receipts(
        self,
        effect_id: str,
        decision_index: int,
        frame_digests: Mapping[str, str],
        *,
        include_witness: bool = True,
        witness_linked: bool = True,
    ) -> List[Event]:
        agent = self.emit(
            EventType.OBSERVE_POSITIVE,
            effect_id,
            decision_index,
            sensor_identity="agentview",
            frame_digest=frame_digests["agentview"],
        )
        wrist = self.emit(
            EventType.VERIFY_POSITIVE,
            effect_id,
            decision_index,
            sensor_identity="robot0_eye_in_hand",
            frame_digest=frame_digests["robot0_eye_in_hand"],
        )
        result = [agent, wrist]
        if include_witness:
            command_id = self.command_event_ids.get(effect_id)
            parents = (command_id,) if command_id and witness_linked else ()
            result.append(
                self.emit(
                    EventType.REALIZATION_WITNESS,
                    effect_id,
                    decision_index,
                    sensor_identity="effect_witness",
                    frame_digest=frame_digests["effect_witness"],
                    parent_ids=parents,
                )
            )
        return result

    def single_view_false_positive(
        self, effect_id: str, decision_index: int, frame_digest: str
    ) -> Event:
        return self.emit(
            EventType.OBSERVE_POSITIVE,
            effect_id,
            decision_index,
            sensor_identity="agentview",
            frame_digest=frame_digest,
            payload={"view": "agentview"},
        )

    def negative_receipts(
        self,
        effect_id: str,
        decision_index: int,
        frame_digests: Mapping[str, str],
    ) -> List[Event]:
        return [
            self.emit(
                EventType.OBSERVE_NEGATIVE,
                effect_id,
                decision_index,
                sensor_identity="agentview",
                frame_digest=frame_digests["agentview"],
            ),
            self.emit(
                EventType.OBSERVE_NEGATIVE,
                effect_id,
                decision_index,
                sensor_identity="robot0_eye_in_hand",
                frame_digest=frame_digests["robot0_eye_in_hand"],
            ),
        ]

    def contradiction(
        self, effect_id: str, decision_index: int, frame_digest: str
    ) -> Event:
        return self.emit(
            EventType.CONTRADICTION,
            effect_id,
            decision_index,
            sensor_identity="contradiction_sensor",
            frame_digest=frame_digest,
        )

    def delayed_tick(self, decision_index: int) -> Event:
        return self.emit(
            EventType.IRRELEVANT,
            None,
            decision_index,
            payload={"receipt_pending": True},
        )

    def timeout(self, effect_id: str, decision_index: int) -> Event:
        return self.emit(EventType.TIMEOUT, effect_id, decision_index)

    def stream_hash(self) -> str:
        return canonical_stream_sha256([record.event for record in self.records])

    def event_records(self) -> List[dict]:
        return [record.to_dict() for record in self.records]

    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(
                self.event_records(), sort_keys=True, separators=(",", ":")
            )
            + "\n"
        ).encode("utf-8")
