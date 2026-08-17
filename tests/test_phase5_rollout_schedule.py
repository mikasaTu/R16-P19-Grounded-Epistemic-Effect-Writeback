from collections import Counter

from r16p19.phase5_rollout import schedule


def test_rollout_schedule_is_frozen_and_disjoint():
    rows = schedule()
    counts = Counter(row.split for row in rows)
    assert counts == {"qualification": 30, "pilot": 15, "natural": 210, "calibration": 30, "formal": 240}
    by_split = {}
    for row in rows:
        by_split.setdefault(row.split, set()).add((row.task_id, row.init_index))
    assert not (by_split["natural"] & by_split["calibration"])
    assert not (by_split["natural"] & by_split["qualification"])
    assert not (by_split["formal"] & by_split["qualification"])
