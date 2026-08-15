"""Discharge-aware support proof graph for Phase-4 ASCEL."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace
from typing import DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from .phase4_types import SupportClause, SupportReference, SupportValidityType


@dataclass
class GraphProof:
    proof_id: str
    effect_id: str
    valid: bool = True
    invalidation_event_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "proof_id": self.proof_id,
            "effect_id": self.effect_id,
            "valid": bool(self.valid),
            "invalidation_event_id": self.invalidation_event_id,
        }


class SupportProofGraph:
    """A disjunction of conjunctive support clauses.

    Each realized effect has one or more proof objects.  A proof without a
    support clause is a root.  A dependent proof remains valid when any one of
    its clauses remains valid.  Discharged references no longer follow their
    parent proof's validity.
    """

    def __init__(self) -> None:
        self.proofs: Dict[str, GraphProof] = {}
        self.proofs_by_effect: DefaultDict[str, List[str]] = defaultdict(list)
        self.clauses: Dict[str, SupportClause] = {}
        self.clauses_by_effect: DefaultDict[str, List[str]] = defaultdict(list)
        self.clauses_by_parent: DefaultDict[str, Set[str]] = defaultdict(set)
        self.discharge_events: List[dict] = []
        self.invalidation_paths: List[dict] = []
        self.invariant_violations: List[str] = []

    def _reference_valid(self, reference: SupportReference) -> bool:
        if reference.discharged:
            return True
        parent = self.proofs.get(reference.parent_proof_id)
        return bool(parent is not None and parent.valid)

    def _clause_valid(self, clause: SupportClause) -> bool:
        return all(self._reference_valid(reference) for reference in clause.references)

    def _effect_supported(self, effect_id: str) -> bool:
        clause_ids = self.clauses_by_effect.get(effect_id, [])
        if not clause_ids:
            return True
        return any(self.clauses[clause_id].valid for clause_id in clause_ids)

    def add_root_proof(self, effect_id: str, proof_id: str) -> GraphProof:
        if proof_id in self.proofs:
            return self.proofs[proof_id]
        proof = GraphProof(proof_id=proof_id, effect_id=effect_id)
        self.proofs[proof_id] = proof
        self.proofs_by_effect[effect_id].append(proof_id)
        self.realize_effect(effect_id, "realize:%s" % proof_id)
        return proof

    def add_supported_proof(
        self,
        effect_id: str,
        proof_id: str,
        clause_references: Sequence[Sequence[SupportReference]],
    ) -> GraphProof:
        if proof_id in self.proofs:
            return self.proofs[proof_id]
        if not clause_references:
            return self.add_root_proof(effect_id, proof_id)
        for index, references in enumerate(clause_references):
            clause_id = "%s:clause:%d" % (proof_id, index)
            clause = SupportClause(
                clause_id=clause_id,
                effect_id=effect_id,
                references=list(references),
            )
            clause.valid = self._clause_valid(clause)
            self.clauses[clause_id] = clause
            self.clauses_by_effect[effect_id].append(clause_id)
            for reference in references:
                self.clauses_by_parent[reference.parent_proof_id].add(clause_id)
        proof = GraphProof(
            proof_id=proof_id,
            effect_id=effect_id,
            valid=self._effect_supported(effect_id),
        )
        self.proofs[proof_id] = proof
        self.proofs_by_effect[effect_id].append(proof_id)
        if proof.valid:
            self.realize_effect(effect_id, "realize:%s" % proof_id)
        return proof

    def realize_effect(self, effect_id: str, event_id: str) -> None:
        """Discharge references whose contractual endpoint is now realized."""

        changed = []
        for clause in self.clauses.values():
            updated = []
            clause_changed = False
            for reference in clause.references:
                should_discharge = (
                    reference.validity_type == SupportValidityType.UNTIL_CHILD_REALIZED
                    and clause.effect_id == effect_id
                ) or (
                    reference.validity_type == SupportValidityType.UNTIL_EFFECT_REALIZED
                    and reference.until_effect_id == effect_id
                )
                if should_discharge and not reference.discharged:
                    updated.append(
                        replace(
                            reference,
                            discharged=True,
                            discharge_event_id=event_id,
                        )
                    )
                    clause_changed = True
                else:
                    updated.append(reference)
            if clause_changed:
                clause.references = updated
                clause.valid = self._clause_valid(clause)
                changed.append(clause.clause_id)
        if changed:
            self.discharge_events.append(
                {
                    "event_id": event_id,
                    "effect_id": effect_id,
                    "clause_ids": sorted(changed),
                }
            )

    def invalidate_proof(self, proof_id: str, event_id: str) -> List[str]:
        """Invalidate one proof and propagate only through newly unsupported effects."""

        root = self.proofs.get(proof_id)
        if root is None:
            self.invariant_violations.append("unknown_proof:%s" % proof_id)
            return []
        if not root.valid:
            return []
        root.valid = False
        root.invalidation_event_id = event_id
        newly_invalidated = [proof_id]
        preserved_alternatives: Set[str] = set()
        frontier = deque([proof_id])
        while frontier:
            invalid_parent = frontier.popleft()
            affected_effects: Set[str] = set()
            for clause_id in sorted(self.clauses_by_parent.get(invalid_parent, set())):
                clause = self.clauses[clause_id]
                clause.valid = self._clause_valid(clause)
                affected_effects.add(clause.effect_id)
            for effect_id in sorted(affected_effects):
                if self._effect_supported(effect_id):
                    for candidate in self.proofs_by_effect.get(effect_id, []):
                        if self.proofs[candidate].valid:
                            preserved_alternatives.add(candidate)
                    continue
                for candidate in self.proofs_by_effect.get(effect_id, []):
                    proof = self.proofs[candidate]
                    if not proof.valid:
                        continue
                    proof.valid = False
                    proof.invalidation_event_id = event_id
                    newly_invalidated.append(candidate)
                    frontier.append(candidate)
        self.invalidation_paths.append(
            {
                "event_id": event_id,
                "root_proof_id": proof_id,
                "newly_invalidated_proof_ids": list(newly_invalidated),
                "preserved_alternative_proof_ids": sorted(preserved_alternatives),
            }
        )
        return newly_invalidated

    def effect_valid(self, effect_id: str) -> bool:
        return any(
            self.proofs[proof_id].valid
            for proof_id in self.proofs_by_effect.get(effect_id, [])
        )

    def active_proof(self, effect_id: str) -> Optional[str]:
        for proof_id in reversed(self.proofs_by_effect.get(effect_id, [])):
            if self.proofs[proof_id].valid:
                return proof_id
        return None

    def snapshot(self) -> dict:
        return {
            "proofs": [self.proofs[key].to_dict() for key in sorted(self.proofs)],
            "clauses": [self.clauses[key].to_dict() for key in sorted(self.clauses)],
            "invalidation_paths": list(self.invalidation_paths),
            "discharge_events": list(self.discharge_events),
            "invariant_violations": list(self.invariant_violations),
        }


class StaticSupportGraph:
    """Frozen-style transitive descendant invalidation comparison graph."""

    def __init__(self, dependencies: Mapping[str, Iterable[str]]) -> None:
        self.parents = {key: list(value) for key, value in dependencies.items()}
        self.children: DefaultDict[str, List[str]] = defaultdict(list)
        for child, parents in self.parents.items():
            for parent in parents:
                self.children[parent].append(child)
        self.valid: Dict[str, bool] = {
            effect: False
            for effect in set(self.parents).union(self.children)
        }
        self.invalidation_paths: List[dict] = []

    def realize(self, effect_id: str) -> None:
        self.valid[effect_id] = True

    def invalidate(self, effect_id: str, event_id: str) -> List[str]:
        invalidated = []
        frontier = deque([effect_id])
        seen: Set[str] = set()
        while frontier:
            candidate = frontier.popleft()
            if candidate in seen:
                continue
            seen.add(candidate)
            if self.valid.get(candidate, False):
                self.valid[candidate] = False
                invalidated.append(candidate)
            frontier.extend(self.children.get(candidate, []))
        self.invalidation_paths.append(
            {
                "event_id": event_id,
                "root_effect_id": effect_id,
                "newly_invalidated_effect_ids": invalidated,
            }
        )
        return invalidated

    def effect_valid(self, effect_id: str) -> bool:
        return bool(self.valid.get(effect_id, False))

    def snapshot(self) -> dict:
        return {
            "valid": dict(self.valid),
            "invalidation_paths": list(self.invalidation_paths),
            "mode": "static_transitive_descendant_invalidation",
        }
