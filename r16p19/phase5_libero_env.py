"""Pinned official LIBERO environment adapter used by the Phase-5 bridge."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


TASKS = {
    0: "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    5: "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
    9: "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
}

TASK_PROMPTS = {
    0: "put both the alphabet soup and the tomato sauce in the basket",
    5: "pick up the book and place it in the back compartment of the caddy",
    9: "put the yellow and white mug in the microwave and close it",
}


def _load_environment_class():
    root = Path(os.environ["R16P19_QPILOTS_ROOT"]).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from qpilots_libero.environment import Task64Environment

    return Task64Environment


def task_config(task_id: int) -> dict[str, Any]:
    if int(task_id) not in TASKS:
        raise ValueError(f"unregistered Phase-5 task: {task_id}")
    # Task64Environment is generic despite its historical name; it consumes only this contract.
    return {
        "task": {
            "suite": "libero_10",
            "task_id": int(task_id),
            "prompt": TASK_PROMPTS[int(task_id)],
            "init_state_count": 50,
            "num_steps_wait": 10,
            "replan_steps": 5,
            "max_steps": 520,
        }
    }


def make_environment(task_id: int, seed: int):
    cls = _load_environment_class()
    environment = cls(task_config(task_id), seed=int(seed))
    expected = TASK_PROMPTS[int(task_id)]
    if str(environment.task.language) != expected:
        environment.close()
        raise RuntimeError(f"task language drifted: {environment.task.language!r} != {expected!r}")
    return environment


def observation_sha256(observation: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in ("observation/image", "observation/wrist_image", "observation/state"):
        value = np.ascontiguousarray(observation[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(value.tobytes())
    digest.update(str(observation["prompt"]).encode("utf-8"))
    return digest.hexdigest()


def compact_features(observation: dict[str, Any], contact_count: int) -> dict[str, np.ndarray]:
    base = np.asarray(observation["observation/image"], dtype=np.uint8)[::8, ::8]
    wrist = np.asarray(observation["observation/wrist_image"], dtype=np.uint8)[::8, ::8]
    return {
        "base_rgb_32": np.ascontiguousarray(base),
        "wrist_rgb_32": np.ascontiguousarray(wrist),
        "proprio": np.asarray(observation["observation/state"], dtype=np.float32),
        "contact_count": np.asarray(contact_count, dtype=np.int32),
    }
