"""Five frozen Phase-4 arms and four mechanism ablations."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from . import memory as frozen_memory_module
from .memory import MemoryArm
from .phase4_attempt_ledger import AttemptScopedLedger
from .phase4_event_broker import to_frozen_event
from .phase4_support_graph import StaticSupportGraph, SupportProofGraph
from .phase4_types import (
    LedgerEvent,
    LedgerEventKind,
    SupportReference,
    SupportValidityType,
)
from .types import Decision, EpistemicState


PROTECTED_B6_SHA256 = "4992462e105306eca9a777619fa5c7418f90c1289e4e9f6fdde1bdc2fbfce4c5"
MAIN_ARMS = (
    "M0_TYPED_MATCHED",
    "M1_B6_ORIGINAL",
    "M2_ATTEMPT_ONLY",
    "M3_SUPPORT_ONLY",
    "M4_ASCEL_FULL",
)
ABLATION_ARMS = (
    "NO_ATTEMPT_SCOPE",
    "NO_SUPPORT_VALIDITY",
    "NO_ATTRIBUTION_SPLIT",
    "NO_PRE_REALIZATION_REVOCATION",
)


def protected_b6_sha256() -> str:
    path = frozen_memory_module.__file__
    assert path is not None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LedgerBackedArm:
    def __init__(
        self,
        name: str,
        effects: Sequence[str],
        dependencies: Mapping[str, Iterable[str]],
        support_contract: Mapping[str, Sequence[Sequence[Mapping[str, object]]]],
        attempt_scope: bool,
        attribution_split: bool,
        pre_realization_revocation: bool,
        discharge_aware_support: bool,
    ) -> None:
        self.name = name
        self.effects = tuple(effects)
        self.dependencies = {
            effect: tuple(dependencies.get(effect, ())) for effect in self.effects
        }
        self.support_contract = support_contract
        self.ledger = AttemptScopedLedger(
            self.effects,
            attempt_scope=attempt_scope,
            attribution_split=attribution_split,
            pre_realization_revocation=pre_realization_revocation,
        )
        self.discharge_aware_support = bool(discharge_aware_support)
        if self.discharge_aware_support:
            self.support_graph = SupportProofGraph()
        else:
            self.support_graph = StaticSupportGraph(self.dependencies)
        self.decision_counts: Dict[str, int] = {}

    def _support_references(
        self, effect_id: str
    ) -> List[List[SupportReference]]:
        clauses = []
        for raw_clause in self.support_contract.get(effect_id, []):
            references = []
            for raw_reference in raw_clause:
                parent_effect = str(raw_reference["parent"])
                parent_proof = self.support_graph.active_proof(parent_effect)
                if parent_proof is None:
                    parent_proof = "missing:%s" % parent_effect
                validity_type = SupportValidityType(str(raw_reference["type"]))
                until_effect = raw_reference.get("until_effect")
                references.append(
                    SupportReference(
                        parent_proof_id=parent_proof,
                        validity_type=validity_type,
                        until_effect_id=(
                            str(until_effect) if until_effect is not None else None
                        ),
                    )
                )
            clauses.append(references)
        return clauses

    def _register_realization(self, effect_id: str, proof_id: str) -> None:
        if self.discharge_aware_support:
            clauses = self._support_references(effect_id)
            graph_proof = self.support_graph.add_supported_proof(
                effect_id, proof_id, clauses
            )
            if clauses:
                ledger_proof = self.ledger.proofs[proof_id]
                ledger_proof.support_clause_id = "%s:clause:0" % proof_id
            if not graph_proof.valid:
                self.ledger.invalidate_effect_from_support(
                    effect_id, [proof_id], "unsupported:%s" % proof_id, 0
                )
        else:
            self.support_graph.realize(effect_id)

    def _propagate_graph_invalidation(
        self, effect_id: str, direct_proof_ids: Sequence[str], event: LedgerEvent
    ) -> None:
        if self.discharge_aware_support:
            graph_invalidated = []
            for proof_id in direct_proof_ids:
                graph_invalidated.extend(
                    self.support_graph.invalidate_proof(proof_id, event.event_id)
                )
            by_effect: Dict[str, List[str]] = {}
            for proof_id in graph_invalidated:
                graph_proof = self.support_graph.proofs.get(proof_id)
                if graph_proof is not None:
                    by_effect.setdefault(graph_proof.effect_id, []).append(proof_id)
            for invalid_effect, proof_ids in by_effect.items():
                self.ledger.invalidate_effect_from_support(
                    invalid_effect, proof_ids, event.event_id, event.epoch
                )
        else:
            invalid_effects = self.support_graph.invalidate(effect_id, event.event_id)
            for invalid_effect in invalid_effects:
                proof_ids = list(
                    self.ledger.facts[invalid_effect].realization_proof_ids
                )
                self.ledger.invalidate_effect_from_support(
                    invalid_effect, proof_ids, event.event_id, event.epoch
                )

    def process(self, event: LedgerEvent) -> Decision:
        proof = None
        if event.kind == LedgerEventKind.REQUEST:
            self.ledger.request(event)
        elif event.kind == LedgerEventKind.COMMAND:
            self.ledger.command(event)
        elif event.kind == LedgerEventKind.POSITIVE:
            self.ledger.positive(event)
        elif event.kind == LedgerEventKind.WITNESS:
            proof = self.ledger.witness(event)
        elif event.kind == LedgerEventKind.EXTERNAL_REALIZATION:
            proof = self.ledger.external_realization(event)
        elif event.kind in (
            LedgerEventKind.NEGATIVE,
            LedgerEventKind.CONTRADICTION,
            LedgerEventKind.SUPPORT_INVALIDATION,
        ):
            invalidated = self.ledger.negative_or_contradiction(event)
            self._propagate_graph_invalidation(event.effect_id, invalidated, event)
        else:
            raise ValueError("unsupported event kind %s" % event.kind)
        if proof is not None:
            self._register_realization(event.effect_id, proof.proof_id)
        decision = self.decide(event.effect_id)
        self.decision_counts[decision.value] = self.decision_counts.get(decision.value, 0) + 1
        return decision

    def decide(self, effect_id: str) -> Decision:
        if self.effect_fact_verified(effect_id):
            return Decision.ADVANCE_TO_NEXT_SUBTASK
        return self.ledger.decide(effect_id)

    def effect_fact_verified(self, effect_id: str) -> bool:
        if not self.ledger.effect_fact_verified(effect_id):
            return False
        return self.support_graph.effect_valid(effect_id)

    def attempt_attributed_success(self, effect_id: str) -> bool:
        return self.ledger.attempt_attributed_success(effect_id)

    def summary(self) -> dict:
        return {
            "arm": self.name,
            "ledger": self.ledger.summary(),
            "support_graph": self.support_graph.snapshot(),
            "decision_counts": dict(self.decision_counts),
            "effect_fact_verified": {
                effect: self.effect_fact_verified(effect) for effect in self.effects
            },
            "attempt_attributed_success": {
                effect: self.attempt_attributed_success(effect)
                for effect in self.effects
            },
        }


class FrozenB6Arm:
    """Adapter around the byte-protected original ``MemoryArm('B6')``."""

    def __init__(
        self,
        name: str,
        task_id: str,
        effects: Sequence[str],
        dependencies: Mapping[str, Iterable[str]],
        episode_id: str,
    ) -> None:
        if protected_b6_sha256() != PROTECTED_B6_SHA256:
            raise RuntimeError("protected r16p19/memory.py SHA256 mismatch")
        self.name = name
        self.effects = tuple(effects)
        self.episode_id = episode_id
        self.task_key = "phase4_%s" % task_id.lower()
        frozen_memory_module.TASKS[self.task_key] = SimpleNamespace(
            effects=self.effects
        )
        ontology = {
            "tasks": {
                self.task_key: [
                    {
                        "effect_id": effect,
                        "prerequisites": list(dependencies.get(effect, ())),
                    }
                    for effect in self.effects
                ]
            }
        }
        self.memory = MemoryArm("B6", self.task_key, ontology)
        self.processed_events: List[dict] = []

    def process(self, event: LedgerEvent) -> Decision:
        frozen_event = to_frozen_event(event, self.episode_id)
        decision = self.memory.process(frozen_event, event.effect_id)
        self.processed_events.append(event.to_dict())
        return decision

    def decide(self, effect_id: str) -> Decision:
        if self.effect_fact_verified(effect_id):
            return Decision.ADVANCE_TO_NEXT_SUBTASK
        record = self.memory.records[effect_id]
        if record.state == EpistemicState.INVALIDATED_REALIZATION:
            return Decision.ROLLBACK_OR_REPLAN
        if record.state == EpistemicState.STALLED:
            return Decision.RETRY_CURRENT_EFFECT
        return Decision.REOBSERVE

    def effect_fact_verified(self, effect_id: str) -> bool:
        return self.memory.records[effect_id].state == EpistemicState.REALIZED

    def attempt_attributed_success(self, effect_id: str) -> bool:
        record = self.memory.records[effect_id]
        return bool(
            record.state == EpistemicState.REALIZED
            and record.command_event_id is not None
        )

    def summary(self) -> dict:
        value = self.memory.current_summary()
        value.update(
            {
                "arm": self.name,
                "protected_memory_sha256": PROTECTED_B6_SHA256,
                "processed_phase4_events": list(self.processed_events),
                "effect_fact_verified": {
                    effect: self.effect_fact_verified(effect) for effect in self.effects
                },
                "attempt_attributed_success": {
                    effect: self.attempt_attributed_success(effect)
                    for effect in self.effects
                },
                "support_graph": {
                    "mode": "frozen_B6_static_transitive_descendant_invalidation",
                    "blocked_dependents": {
                        effect: list(self.memory.records[effect].blocked_dependents)
                        for effect in self.effects
                    },
                },
            }
        )
        return value


def make_phase4_arm(
    name: str,
    task_id: str,
    effects: Sequence[str],
    dependencies: Mapping[str, Iterable[str]],
    support_contract: Mapping[str, Sequence[Sequence[Mapping[str, object]]]],
    episode_id: str,
):
    if name == "M1_B6_ORIGINAL":
        return FrozenB6Arm(name, task_id, effects, dependencies, episode_id)
    settings = {
        "M0_TYPED_MATCHED": (False, False, True, False),
        "M2_ATTEMPT_ONLY": (True, True, True, False),
        "M3_SUPPORT_ONLY": (False, False, False, True),
        "M4_ASCEL_FULL": (True, True, True, True),
        "NO_ATTEMPT_SCOPE": (False, True, True, True),
        "NO_SUPPORT_VALIDITY": (True, True, True, False),
        "NO_ATTRIBUTION_SPLIT": (True, False, True, True),
        "NO_PRE_REALIZATION_REVOCATION": (True, True, False, True),
    }
    if name not in settings:
        raise ValueError("unknown Phase-4 arm %s" % name)
    attempt_scope, attribution_split, revocation, support = settings[name]
    return LedgerBackedArm(
        name,
        effects,
        dependencies,
        support_contract,
        attempt_scope=attempt_scope,
        attribution_split=attribution_split,
        pre_realization_revocation=revocation,
        discharge_aware_support=support,
    )
