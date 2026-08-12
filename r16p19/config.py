"""Frozen paths and task contracts for the LIBERO adaptation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_SOURCE = PROJECT_ROOT / "experiments" / "r16p19_libero_phase1"
BENCHMARK_MANIFEST = EXPERIMENT_SOURCE / "benchmark_manifest.json"
LIBERO_ROOT = Path(
    "/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/"
    "LIBERO-r16p19-official-8f1084e"
)
DATA_ROOT = Path(
    "/mnt/cpfs/zbl-cpfs-new/dataset/leon/"
    "embodied_benchmark/datasets/LIBERO/libero_10"
)

TRAIN_EPISODES = tuple("demo_%d" % index for index in range(30))
CALIBRATION_EPISODES = tuple("demo_%d" % index for index in range(30, 40))
TRACE_TEST_EPISODES = tuple("demo_%d" % index for index in range(40, 50))
EVAL_INIT_INDICES = tuple(range(20))

MAX_STATE_DIM = 64
MAX_EFFECTS = 4
EPISTEMIC_STATE_COUNT = 7
MEMORY_SUMMARY_DIM = MAX_EFFECTS * EPISTEMIC_STATE_COUNT
ACTION_DIM = 7
ACTION_HORIZON = 8
RESIDENT_MEMORY_SLOTS = 32


@dataclass(frozen=True)
class TaskSpec:
    key: str
    task_id: str
    instruction: str
    dataset_path: Path
    bddl_path: Path
    init_path: Path
    state_dim: int
    object_name: str
    effects: Tuple[str, ...]
    predicate_specs: Tuple[Tuple[str, ...], ...]


TASKS: Dict[str, TaskSpec] = {
    "stove_moka": TaskSpec(
        key="stove_moka",
        task_id="KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        instruction="turn on the stove and put the moka pot on it",
        dataset_path=DATA_ROOT
        / "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5",
        bddl_path=LIBERO_ROOT
        / "libero/libero/bddl_files/libero_10/"
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl",
        init_path=LIBERO_ROOT
        / "libero/libero/init_files/libero_10/"
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.init",
        state_dim=47,
        object_name="moka_pot_1",
        effects=(
            "STOVE_TURNED_ON",
            "MOKA_GRASPED",
            "MOKA_ON_STOVE",
            "MOKA_RELEASED_ON_STOVE",
        ),
        predicate_specs=(
            ("predicate", "turnon", "flat_stove_1"),
            ("grasp", "moka_pot_1"),
            ("predicate", "on", "moka_pot_1", "flat_stove_1_cook_region"),
            ("released_on", "moka_pot_1", "flat_stove_1_cook_region"),
        ),
    ),
    "bowl_drawer": TaskSpec(
        key="bowl_drawer",
        task_id=(
            "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_"
            "of_the_cabinet_and_close_it"
        ),
        instruction="put the black bowl in the bottom drawer of the cabinet and close it",
        dataset_path=DATA_ROOT
        / (
            "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_"
            "of_the_cabinet_and_close_it_demo.hdf5"
        ),
        bddl_path=LIBERO_ROOT
        / (
            "libero/libero/bddl_files/libero_10/"
            "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_"
            "of_the_cabinet_and_close_it.bddl"
        ),
        init_path=LIBERO_ROOT
        / (
            "libero/libero/init_files/libero_10/"
            "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_"
            "of_the_cabinet_and_close_it.init"
        ),
        state_dim=51,
        object_name="akita_black_bowl_1",
        effects=(
            "BOWL_GRASPED",
            "BOWL_IN_BOTTOM_DRAWER",
            "BOWL_RELEASED_IN_DRAWER",
            "BOTTOM_DRAWER_CLOSED",
        ),
        predicate_specs=(
            ("grasp", "akita_black_bowl_1"),
            (
                "predicate",
                "in",
                "akita_black_bowl_1",
                "white_cabinet_1_bottom_region",
            ),
            (
                "released_in",
                "akita_black_bowl_1",
                "white_cabinet_1_bottom_region",
            ),
            ("predicate", "close", "white_cabinet_1_bottom_region"),
        ),
    ),
}


def task_list() -> List[TaskSpec]:
    return [TASKS["stove_moka"], TASKS["bowl_drawer"]]


def load_benchmark_manifest() -> dict:
    with BENCHMARK_MANIFEST.open("r", encoding="utf-8") as handle:
        return json.load(handle)

