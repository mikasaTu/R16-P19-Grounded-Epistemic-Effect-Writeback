import hashlib

import numpy as np

from r16p19.phase3_snapshot_bank import (
    ReplayedDemo,
    array_sha256,
    first_stable_true_index,
    segment_boundaries,
)


def test_first_stable_true_requires_five_consecutive_steps():
    assert first_stable_true_index([False, True, True, False, True, True, True, True, True]) == 4
    assert first_stable_true_index([True, True, True, True]) is None


def test_later_effect_entry_has_four_action_preroll_but_distinct_effect_start():
    effects = (
        "STOVE_TURNED_ON",
        "MOKA_GRASPED",
        "MOKA_ON_STOVE",
        "MOKA_RELEASED_ON_STOVE",
    )
    stable_starts = dict(zip(effects, (2, 8, 14, 20)))
    length = 26
    post_truths = []
    pre_truths = []
    for index in range(length):
        post_truths.append(
            {effect: index >= stable_starts[effect] for effect in effects}
        )
        pre_truths.append(
            {effect: index - 1 >= stable_starts[effect] for effect in effects}
        )
    states = tuple(np.asarray([index], dtype="<f8") for index in range(length))
    replay = ReplayedDemo(
        task_key="stove_moka",
        episode_id="demo_30",
        actions=np.zeros((length, 7), dtype="<f4"),
        pre_states=states,
        post_states=states,
        pre_observation_sha256=tuple("0" * 64 for _ in range(length)),
        post_observation_sha256=tuple("1" * 64 for _ in range(length)),
        pre_truths=tuple(pre_truths),
        post_truths=tuple(post_truths),
        final_task_success=True,
    )
    boundaries = segment_boundaries(replay)
    second = boundaries["MOKA_GRASPED"]
    assert second["first_action_index_belonging_to_effect"] == 7
    assert second["entry_action_index"] == 3
    assert second["precondition_truth"] == {"STOVE_TURNED_ON": True}
    assert second["target_truth_at_entry"] is False
    assert second["valid"] is True


def test_array_hash_is_little_endian_contiguous_bytes():
    value = np.asarray([[1.25, -2.5]], dtype="<f4")
    assert array_sha256(value, "<f4") == hashlib.sha256(value.tobytes(order="C")).hexdigest()
