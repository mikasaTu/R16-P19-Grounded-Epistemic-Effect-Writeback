"""Seeded workload generator for exact live/reference comparison."""

from __future__ import annotations

import hashlib
import random
from dataclasses import replace
from typing import Iterator, Tuple

from .phase5_types import EvidenceReceipt


def workload(events: int, attempts: int, seed: int = 1701) -> Iterator[Tuple[str, object, int]]:
    rng = random.Random(seed)
    effects = [f"effect-{index}" for index in range(4)]
    source_versions = {f"sensor-{index}": 0 for index in range(4)}
    request_period = max(1, events // attempts)
    active = {effect: (f"attempt-{effect}-0", f"command-{effect}-0") for effect in effects}
    last_receipt = {}
    generation = {effect: 0 for effect in effects}
    for epoch in range(events):
        effect = effects[epoch % len(effects)]
        if epoch % request_period == 0:
            generation[effect] += 1
            pair = (f"attempt-{effect}-{generation[effect]}", f"command-{effect}-{generation[effect]}")
            active[effect] = pair
            yield "request", (effect, pair[0], pair[1]), epoch
            continue
        if epoch % 97 == 0:
            yield "revoke", effect, epoch
            continue
        source = f"sensor-{rng.randrange(4)}"
        source_versions[source] += 1
        attempt_id, command_id = active[effect]
        mode = rng.randrange(20)
        if mode == 0:
            attempt_id = f"stale-{attempt_id}"
        if mode == 1:
            command_id = f"stale-{command_id}"
        observed_at = epoch if mode != 2 else max(0, epoch - 100)
        version = source_versions[source] - 2 if mode == 6 else source_versions[source]
        receipt = EvidenceReceipt(
            receipt_id=f"receipt-{epoch}-{hashlib.sha256(str(rng.random()).encode()).hexdigest()[:8]}",
            attempt_id=attempt_id,
            command_id=command_id,
            effect_id=effect,
            source_id=source,
            source_version=max(0, version),
            observed_at=observed_at,
            expires_at=epoch + (10 if mode != 2 else -1),
            payload_hash=hashlib.sha256(f"{seed}:{epoch}".encode()).hexdigest(),
            confidence=0.95,
            truth_value=mode != 3,
            active_attempt_credit=True,
            capture_id=f"capture-{epoch}",
            sensor_source=source,
            source_group=source,
            correlation_group=source,
            frame_digest=hashlib.sha256(f"frame:{seed}:{epoch}".encode()).hexdigest(),
            timestamp=float(epoch) / 20.0,
            verifier_model_version="oracle-v1",
            calibration_version="frozen-v1",
            parent_ids=(command_id,),
        )
        if mode == 4 and effect in last_receipt:
            receipt = last_receipt[effect]
        elif mode == 5 and effect in last_receipt:
            receipt = replace(last_receipt[effect], payload_hash=receipt.payload_hash)
        else:
            last_receipt[effect] = receipt
        yield "receipt", receipt, epoch
