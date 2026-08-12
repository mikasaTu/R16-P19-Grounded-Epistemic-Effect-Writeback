"""Effect ontology loading and prerequisite/dependency validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from .config import EXPERIMENT_SOURCE, TASKS


ONTOLOGY_PATH = EXPERIMENT_SOURCE / "effect_ontology.json"


def load_ontology(path: Path = ONTOLOGY_PATH) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        ontology = json.load(handle)
    validate_ontology(ontology)
    return ontology


def validate_ontology(ontology: dict) -> None:
    if ontology.get("schema_version") != 1:
        raise ValueError("unsupported ontology schema")
    tasks = ontology.get("tasks", {})
    if set(tasks) != set(TASKS):
        raise ValueError("ontology task set does not match frozen task set")
    required = {
        "effect_id",
        "prerequisites",
        "incompatible_effects",
        "requested_event",
        "observed_evidence",
        "verification_evidence",
        "realization_witness",
        "invalidation_witness",
        "valid_recovery_actions",
    }
    for task_key, entries in tasks.items():
        observed_ids = [entry.get("effect_id") for entry in entries]
        if observed_ids != list(TASKS[task_key].effects):
            raise ValueError("ontology effect order differs for %s" % task_key)
        known = set(observed_ids)
        for entry in entries:
            if set(entry) != required:
                raise ValueError("ontology fields differ for %s" % entry.get("effect_id"))
            if not set(entry["prerequisites"]).issubset(known):
                raise ValueError("unknown prerequisite for %s" % entry["effect_id"])
            if not entry["valid_recovery_actions"]:
                raise ValueError("empty recovery route for %s" % entry["effect_id"])


def prerequisites_by_task(ontology: dict, task_key: str) -> Dict[str, List[str]]:
    return {
        item["effect_id"]: list(item["prerequisites"])
        for item in ontology["tasks"][task_key]
    }


def dependents(effect_ids: Iterable[str], prerequisites: Dict[str, List[str]]) -> Dict[str, List[str]]:
    result = {effect_id: [] for effect_id in effect_ids}
    for candidate, parents in prerequisites.items():
        for parent in parents:
            result[parent].append(candidate)
    return result

