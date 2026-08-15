"""Deterministic adversarial trace schedule generator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Iterator, Tuple


TRACE_FAMILIES: Tuple[str, ...] = (
    "A1_STALE_WITNESS_AFTER_RETRY",
    "A2_CROSS_ATTEMPT_SENSOR_MIX",
    "A3_SUPERSEDED_COMMAND_WITNESS",
    "A4_VERIFIED_CONTRADICTION_LATE_WITNESS",
    "A5_INCIDENTAL_EFFECT",
    "S1_LIVE_SUPPORT_REVERSAL",
    "S2_DISCHARGED_SUPPORT_REVERSAL",
    "S3_ALTERNATIVE_SUPPORT",
    "S4_BRANCH_LOCAL_INVALIDATION",
    "S5_LATE_DEPENDENT_WITNESS",
)


@dataclass(frozen=True)
class TraceSchedule:
    schedule_id: str
    family: str
    ordinal: int
    seed: int
    sensor_order: Tuple[str, str]
    duplicate_first_receipt: bool

    def to_dict(self) -> dict:
        value = asdict(self)
        value["sensor_order"] = list(self.sensor_order)
        return value


def generate_trace_schedules(
    schedules_per_family: int = 1000,
) -> Iterator[TraceSchedule]:
    if schedules_per_family < 1:
        raise ValueError("schedules_per_family must be positive")
    for family_index, family in enumerate(TRACE_FAMILIES):
        for ordinal in range(schedules_per_family):
            reverse = bool((ordinal + family_index) % 2)
            yield TraceSchedule(
                schedule_id="trace-%02d-%04d" % (family_index, ordinal),
                family=family,
                ordinal=ordinal,
                seed=16190000 + family_index * 1000 + ordinal,
                sensor_order=("sensor_b", "sensor_a") if reverse else ("sensor_a", "sensor_b"),
                duplicate_first_receipt=bool(ordinal % 17 == 0),
            )


def schedule_count(schedules: Iterable[TraceSchedule]) -> int:
    return sum(1 for _ in schedules)
