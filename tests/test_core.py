import hashlib

import numpy as np
import torch

from r16p19.actor import INPUT_DIM, TinyChunkMLP
from r16p19.checkpoints import synthetic_retention_test
from r16p19.memory import MemoryArm, ProvenanceLedger
from r16p19.ontology import load_ontology
from r16p19.types import (
    Decision,
    EpistemicState,
    Event,
    EventType,
    EvidenceReceipt,
)


def event(index, kind, effect="STOVE_TURNED_ON", parents=(), digest=None, sensor="agentview"):
    receipt = None
    if digest is not None:
        receipt = EvidenceReceipt(
            evidence_id="r%d" % index,
            episode_id="test",
            event_index=index,
            timestamp=float(index),
            sensor_identity=sensor,
            frame_digest=digest,
            effect_id=effect,
            evidence_type=kind.value,
        )
    return Event(
        event_id="e%d" % index,
        episode_id="test",
        event_index=index,
        timestamp=float(index),
        event_type=kind,
        effect_id=effect,
        parent_ids=tuple(parents),
        receipt=receipt,
    )


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def test_command_and_imagination_never_directly_realize():
    arm = MemoryArm("B6", "stove_moka", load_ontology())
    arm.process(event(0, EventType.REQUEST), "STOVE_TURNED_ON")
    arm.process(event(1, EventType.IMAGINE, parents=("e0",)), "STOVE_TURNED_ON")
    assert arm.records["STOVE_TURNED_ON"].state == EpistemicState.IMAGINED
    arm.process(event(2, EventType.COMMAND, parents=("e1",)), "STOVE_TURNED_ON")
    assert arm.records["STOVE_TURNED_ON"].state != EpistemicState.REALIZED
    assert arm.transition_violations == []


def test_verification_rejects_same_physical_frame_under_new_id():
    arm = MemoryArm("B6", "stove_moka", load_ontology())
    arm.process(event(0, EventType.REQUEST), "STOVE_TURNED_ON")
    arm.process(event(1, EventType.COMMAND, parents=("e0",)), "STOVE_TURNED_ON")
    arm.process(
        event(2, EventType.OBSERVE_POSITIVE, parents=("e1",), digest=digest("same")),
        "STOVE_TURNED_ON",
    )
    arm.process(
        event(
            3,
            EventType.VERIFY_POSITIVE,
            parents=("e2",),
            digest=digest("same"),
            sensor="robot0_eye_in_hand",
        ),
        "STOVE_TURNED_ON",
    )
    assert arm.records["STOVE_TURNED_ON"].state == EpistemicState.OBSERVED
    assert arm.alias_rejections == 1
    assert arm.alias_acceptances == 0


def test_contradiction_invalidates_blocks_dependents_and_is_idempotent():
    arm = MemoryArm("B6", "stove_moka", load_ontology())
    arm.process(event(0, EventType.REQUEST), "STOVE_TURNED_ON")
    arm.process(event(1, EventType.COMMAND, parents=("e0",)), "STOVE_TURNED_ON")
    arm.process(
        event(2, EventType.OBSERVE_POSITIVE, parents=("e1",), digest=digest("a")),
        "STOVE_TURNED_ON",
    )
    arm.process(
        event(
            3,
            EventType.VERIFY_POSITIVE,
            parents=("e2",),
            digest=digest("b"),
            sensor="robot0_eye_in_hand",
        ),
        "STOVE_TURNED_ON",
    )
    arm.process(
        event(4, EventType.REALIZATION_WITNESS, parents=("e1", "e3"), digest=digest("c")),
        "STOVE_TURNED_ON",
    )
    arm.records["MOKA_GRASPED"].state = EpistemicState.REALIZED
    contradiction = event(
        5, EventType.CONTRADICTION, parents=("e4",), digest=digest("negative")
    )
    decision = arm.process(contradiction, "STOVE_TURNED_ON")
    fingerprint = arm.semantic_fingerprint("STOVE_TURNED_ON")
    assert decision == Decision.ROLLBACK_OR_REPLAN
    assert arm.records["STOVE_TURNED_ON"].state == EpistemicState.INVALIDATED_REALIZATION
    assert arm.records["STOVE_TURNED_ON"].recovery_route
    assert "MOKA_GRASPED" in arm.records["STOVE_TURNED_ON"].blocked_dependents
    assert arm.records["MOKA_GRASPED"].state == EpistemicState.INVALIDATED_REALIZATION
    arm.process(contradiction, "STOVE_TURNED_ON")
    assert arm.semantic_fingerprint("STOVE_TURNED_ON") == fingerprint


def test_bounded_resident_cache_keeps_append_only_provenance():
    ledger = ProvenanceLedger()
    parent = ()
    for index in range(50):
        current = Event(
            event_id="e%d" % index,
            episode_id="pressure",
            event_index=index,
            timestamp=float(index),
            event_type=EventType.IRRELEVANT,
            effect_id=None,
            parent_ids=parent,
        )
        ledger.append(current)
        parent = (current.event_id,)
    assert len(ledger.events) == 50
    assert len(ledger.resident) == 32
    assert ledger.max_resident_seen == 32
    assert ledger.dangling_parent_count() == 0


def test_tiny_actor_shape_and_parameter_limit():
    inputs = np.zeros((16, INPUT_DIM), dtype=np.float32)
    chunks = np.zeros((16, 8, 7), dtype=np.float32)
    model = TinyChunkMLP.from_arrays(inputs, chunks)
    output = model(torch.from_numpy(inputs[:2]))
    assert output.shape == (2, 8, 7)
    assert sum(value.numel() for value in model.parameters()) < 10_000_000
    model.loss(torch.from_numpy(inputs), torch.from_numpy(chunks)).backward()


def test_checkpoint_retention_contract():
    result = synthetic_retention_test()
    assert result["status"] == "CHECKPOINT_RETENTION_OK"
    assert 10000 in result["retained_steps"]
    assert result["incomplete_preserved"]
