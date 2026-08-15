"""Deterministic Phase-4 event broker with append-only canonical bytes."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Counter as CounterType
from typing import Dict, List, Optional, Sequence, Tuple

from .phase4_types import (
    LedgerEvent,
    LedgerEventKind,
    canonical_sha256,
    deterministic_id,
)
from .types import EvidenceReceipt, Event, EventType


class Phase4EventBroker:
    def __init__(self, unit_id: str, seed: int) -> None:
        self.unit_id = unit_id
        self.seed = int(seed)
        self.counter = 0
        self.records: List[LedgerEvent] = []
        self.generations: CounterType[str] = Counter()
        self.active_attempts: Dict[str, str] = {}
        self.active_commands: Dict[str, str] = {}

    def _event_id(self, kind: LedgerEventKind, effect_id: str) -> str:
        self.counter += 1
        return deterministic_id(
            "event", self.unit_id, self.seed, self.counter, kind.value, effect_id
        )

    def _append(self, event: LedgerEvent) -> LedgerEvent:
        self.records.append(event)
        return event

    def request(self, effect_id: str) -> LedgerEvent:
        event_id = self._event_id(LedgerEventKind.REQUEST, effect_id)
        self.generations[effect_id] += 1
        attempt_id = deterministic_id(
            "attempt", effect_id, int(self.generations[effect_id]), event_id
        )
        self.active_attempts[effect_id] = attempt_id
        self.active_commands.pop(effect_id, None)
        return self._append(
            LedgerEvent(
                event_id=event_id,
                epoch=self.counter,
                kind=LedgerEventKind.REQUEST,
                effect_id=effect_id,
                attempt_id=attempt_id,
                payload={"generation": int(self.generations[effect_id])},
            )
        )

    def command(self, effect_id: str, attempt_id: Optional[str] = None) -> LedgerEvent:
        event_id = self._event_id(LedgerEventKind.COMMAND, effect_id)
        selected_attempt = attempt_id or self.active_attempts.get(effect_id)
        if selected_attempt == self.active_attempts.get(effect_id):
            self.active_commands[effect_id] = event_id
        return self._append(
            LedgerEvent(
                event_id=event_id,
                epoch=self.counter,
                kind=LedgerEventKind.COMMAND,
                effect_id=effect_id,
                attempt_id=selected_attempt,
            )
        )

    def positive(
        self,
        effect_id: str,
        sensor_id: str,
        attempt_id: Optional[str] = None,
        command_event_id: Optional[str] = None,
        physical_truth: Optional[bool] = None,
        force_verification: Optional[bool] = None,
    ) -> LedgerEvent:
        event_id = self._event_id(LedgerEventKind.POSITIVE, effect_id)
        selected_attempt = attempt_id or self.active_attempts.get(effect_id)
        selected_command = command_event_id or self.active_commands.get(effect_id)
        digest = hashlib.sha256(
            (
                "%s|%s|%s|%s|%s|%d"
                % (
                    self.unit_id,
                    effect_id,
                    sensor_id,
                    selected_attempt,
                    selected_command,
                    self.counter,
                )
            ).encode("utf-8")
        ).hexdigest()
        evidence_id = deterministic_id("evidence", event_id, sensor_id, digest)
        prior_sensor_count = sum(
            record.kind == LedgerEventKind.POSITIVE
            and record.effect_id == effect_id
            and record.attempt_id == selected_attempt
            and record.sensor_id != sensor_id
            for record in self.records
        )
        return self._append(
            LedgerEvent(
                event_id=event_id,
                epoch=self.counter,
                kind=LedgerEventKind.POSITIVE,
                effect_id=effect_id,
                attempt_id=selected_attempt,
                command_event_id=selected_command,
                evidence_id=evidence_id,
                sensor_id=sensor_id,
                evidence_digest=digest,
                physical_truth=physical_truth,
                parent_ids=tuple(
                    value for value in (selected_command,) if value is not None
                ),
                payload={
                    "verification": (
                        bool(prior_sensor_count)
                        if force_verification is None
                        else bool(force_verification)
                    )
                },
            )
        )

    def witness(
        self,
        effect_id: str,
        attempt_id: Optional[str] = None,
        command_event_id: Optional[str] = None,
        physical_truth: Optional[bool] = None,
    ) -> LedgerEvent:
        event_id = self._event_id(LedgerEventKind.WITNESS, effect_id)
        selected_attempt = attempt_id or self.active_attempts.get(effect_id)
        selected_command = command_event_id or self.active_commands.get(effect_id)
        return self._append(
            LedgerEvent(
                event_id=event_id,
                epoch=self.counter,
                kind=LedgerEventKind.WITNESS,
                effect_id=effect_id,
                attempt_id=selected_attempt,
                command_event_id=selected_command,
                physical_truth=physical_truth,
                parent_ids=tuple(
                    value for value in (selected_command,) if value is not None
                ),
            )
        )

    def contradiction(
        self, effect_id: str, physical_truth: Optional[bool] = False
    ) -> LedgerEvent:
        event_id = self._event_id(LedgerEventKind.CONTRADICTION, effect_id)
        return self._append(
            LedgerEvent(
                event_id=event_id,
                epoch=self.counter,
                kind=LedgerEventKind.CONTRADICTION,
                effect_id=effect_id,
                attempt_id=self.active_attempts.get(effect_id),
                command_event_id=self.active_commands.get(effect_id),
                physical_truth=physical_truth,
            )
        )

    def negative(self, effect_id: str) -> LedgerEvent:
        event_id = self._event_id(LedgerEventKind.NEGATIVE, effect_id)
        return self._append(
            LedgerEvent(
                event_id=event_id,
                epoch=self.counter,
                kind=LedgerEventKind.NEGATIVE,
                effect_id=effect_id,
                attempt_id=self.active_attempts.get(effect_id),
                command_event_id=self.active_commands.get(effect_id),
                physical_truth=False,
            )
        )

    def external_realization(self, effect_id: str) -> LedgerEvent:
        event_id = self._event_id(LedgerEventKind.EXTERNAL_REALIZATION, effect_id)
        evidence_ids = [
            deterministic_id("external-evidence", event_id, sensor)
            for sensor in ("sensor_a", "sensor_b")
        ]
        return self._append(
            LedgerEvent(
                event_id=event_id,
                epoch=self.counter,
                kind=LedgerEventKind.EXTERNAL_REALIZATION,
                effect_id=effect_id,
                attempt_id=None,
                command_event_id=None,
                physical_truth=True,
                external=True,
                payload={"evidence_ids": evidence_ids},
            )
        )

    def prefix(self, effect_id: str) -> Tuple[LedgerEvent, LedgerEvent]:
        request = self.request(effect_id)
        command = self.command(effect_id, request.attempt_id)
        return request, command

    def stream_hash(self) -> str:
        return canonical_sha256([record.to_dict() for record in self.records])

    def stream_bytes(self) -> bytes:
        return b"\n".join(record.canonical_bytes() for record in self.records)

    def snapshot(self) -> dict:
        return {
            "unit_id": self.unit_id,
            "seed": self.seed,
            "counter": self.counter,
            "active_attempts": dict(self.active_attempts),
            "active_commands": dict(self.active_commands),
            "records": [record.to_dict() for record in self.records],
            "stream_sha256": self.stream_hash(),
        }


def to_frozen_event(event: LedgerEvent, episode_id: str) -> Event:
    """Map a Phase-4 receipt to the unchanged Phase-1/B6 event interface."""

    event_type = {
        LedgerEventKind.REQUEST: EventType.REQUEST,
        LedgerEventKind.COMMAND: EventType.COMMAND,
        LedgerEventKind.POSITIVE: (
            EventType.VERIFY_POSITIVE
            if bool(event.payload.get("verification"))
            else EventType.OBSERVE_POSITIVE
        ),
        LedgerEventKind.NEGATIVE: EventType.OBSERVE_NEGATIVE,
        LedgerEventKind.WITNESS: EventType.REALIZATION_WITNESS,
        LedgerEventKind.CONTRADICTION: EventType.CONTRADICTION,
        LedgerEventKind.SUPPORT_INVALIDATION: EventType.CONTRADICTION,
        LedgerEventKind.EXTERNAL_REALIZATION: EventType.REALIZATION_WITNESS,
    }[event.kind]
    receipt = None
    if event.kind == LedgerEventKind.POSITIVE:
        assert event.evidence_id is not None
        assert event.sensor_id is not None
        assert event.evidence_digest is not None
        receipt = EvidenceReceipt(
            evidence_id=event.evidence_id,
            episode_id=episode_id,
            event_index=event.epoch,
            timestamp=float(event.epoch),
            sensor_identity=event.sensor_id,
            frame_digest=event.evidence_digest,
            effect_id=event.effect_id,
            evidence_type=event_type.value,
        )
    return Event(
        event_id=event.event_id,
        episode_id=episode_id,
        event_index=event.epoch,
        timestamp=float(event.epoch),
        event_type=event_type,
        effect_id=event.effect_id,
        parent_ids=event.parent_ids,
        receipt=receipt,
        payload={
            "phase4_attempt_id": event.attempt_id,
            "phase4_command_event_id": event.command_event_id,
            "external": event.external,
        },
    )


def canonical_event_sequence_sha256(events: Sequence[LedgerEvent]) -> str:
    return canonical_sha256([event.to_dict() for event in events])
