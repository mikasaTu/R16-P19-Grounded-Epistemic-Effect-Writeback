import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from r16p19.config import TASKS
from r16p19.phase2_executor import (
    ExecutorVariant,
    GeometricSnapshot,
    RetargetedGeometricSkillExecutor,
    executor_input_bytes,
    matrix_to_quaternion_xyzw,
    quaternion_to_matrix_xyzw,
)


def _snapshot(x=0.0, fixture_x=0.0):
    return GeometricSnapshot(
        eef_position=np.asarray((x, 0.0, 0.1), dtype=np.float32),
        eef_quaternion_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32),
        robot_joint_position=np.zeros((7,), dtype=np.float32),
        gripper_joint_position=np.asarray((0.04, -0.04), dtype=np.float32),
        effect_object_position=np.zeros((3,), dtype=np.float32),
        effect_object_quaternion_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32),
        effect_fixture_position=np.asarray((fixture_x, 0.0, 0.0), dtype=np.float32),
        effect_fixture_quaternion_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32),
        effect_fixture_joint_position=np.zeros((1,), dtype=np.float32),
    )


def _write_manifest(root: Path, templates_per_effect=1):
    template_dir = root / "skill_templates"
    template_dir.mkdir()
    rows = []
    for task_key, task in TASKS.items():
        for effect_id in task.effects:
            for template_index in range(templates_per_effect):
                start_x = 0.1 * template_index
                value = {
                    "schema_version": 1,
                    "template_id": "%s__%s__demo_%d"
                    % (task_key, effect_id, template_index),
                    "task_key": task_key,
                    "effect_id": effect_id,
                    "source_episode": "demo_%d" % template_index,
                    "source_segment": {"start": 0, "stop_exclusive": 2},
                    "source_frame_position": [0.0, 0.0, 0.0],
                    "source_frame_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "local_positions": [
                        [start_x, 0.0, 0.1],
                        [start_x + 0.02, 0.0, 0.1],
                    ],
                    "local_quaternions_xyzw": [
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "world_positions": [
                        [start_x, 0.0, 0.1],
                        [start_x + 0.02, 0.0, 0.1],
                    ],
                    "world_quaternions_xyzw": [
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "actions": [
                        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
                    ]
                    * 2,
                    "gripper": [-1.0, 1.0],
                    "descriptor": [0.0] * 14,
                    "waypoint_source_indices": [0, 1],
                }
                raw = (
                    json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
                ).encode()
                name = value["template_id"] + ".json"
                (template_dir / name).write_bytes(raw)
                rows.append(
                    {
                        "task_key": task_key,
                        "effect_id": effect_id,
                        "template_id": value["template_id"],
                        "source_episode": "demo_%d" % template_index,
                        "waypoint_count": 2,
                        "path": "skill_templates/" + name,
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
    manifest = {
        "schema_version": 1,
        "templates": rows,
    }
    path = root / "skill_template_manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_quaternion_matrix_round_trip():
    values = (
        np.asarray((0.0, 0.0, 0.0, 1.0)),
        np.asarray((0.2, -0.3, 0.1, 0.92)),
        np.asarray((0.7, 0.0, 0.0, 0.7)),
    )
    for value in values:
        matrix = quaternion_to_matrix_xyzw(value)
        recovered = matrix_to_quaternion_xyzw(matrix)
        expected = value / np.linalg.norm(value)
        assert abs(float(np.dot(recovered, expected))) > 1.0 - 1e-6


def test_canonical_executor_input_is_deterministic_and_strict():
    history = [_snapshot()]
    first = executor_input_bytes(history, "stove_moka", "STOVE_TURNED_ON", "EXECUTE", 0)
    second = executor_input_bytes(history, "stove_moka", "STOVE_TURNED_ON", "EXECUTE", 0)
    assert first == second
    assert first != executor_input_bytes(
        history, "stove_moka", "STOVE_TURNED_ON", "RETRY", 1
    )
    with pytest.raises(TypeError):
        executor_input_bytes([np.zeros(4)], "stove_moka", "STOVE_TURNED_ON", "EXECUTE", 0)


def test_executor_identical_input_has_identical_action_bytes(tmp_path):
    manifest = _write_manifest(tmp_path)
    executor = RetargetedGeometricSkillExecutor(
        manifest, variant=ExecutorVariant.FROZEN_FULL
    )
    history = [_snapshot()]
    first = executor.action_chunk(
        history, "stove_moka", "STOVE_TURNED_ON", "EXECUTE", 0
    )
    second = executor.action_chunk(
        history, "stove_moka", "STOVE_TURNED_ON", "EXECUTE", 0
    )
    assert first.dtype == np.float32
    assert first.shape == (8, 7)
    assert first.tobytes() == second.tobytes()
    assert np.all(first[:, 0] > 0.0)


def test_demonstration_feedforward_is_an_isolated_component(tmp_path):
    manifest = _write_manifest(tmp_path)
    enabled = RetargetedGeometricSkillExecutor(
        manifest, demonstration_feedforward=True
    ).action_chunk([_snapshot()], "stove_moka", "STOVE_TURNED_ON", "EXECUTE", 0)
    disabled = RetargetedGeometricSkillExecutor(
        manifest, demonstration_feedforward=False
    ).action_chunk([_snapshot()], "stove_moka", "STOVE_TURNED_ON", "EXECUTE", 0)
    assert not np.array_equal(enabled, disabled)


def test_retry_order_is_frozen_from_first_effect_geometry(tmp_path):
    manifest = _write_manifest(tmp_path, templates_per_effect=3)
    executor = RetargetedGeometricSkillExecutor(manifest)
    executor.action_chunk(
        [_snapshot(x=0.0)], "stove_moka", "STOVE_TURNED_ON", "EXECUTE", 0
    )
    first_sha256 = executor.last_trace["template_sha256"]
    executor.action_chunk(
        [_snapshot(x=0.1)], "stove_moka", "STOVE_TURNED_ON", "RETRY", 1
    )
    assert executor.last_trace["template_sha256"] != first_sha256
    assert executor.last_trace["template_rank"] == 1


def test_executor_module_has_no_memory_or_effect_truth_dependency():
    source = (
        Path(__file__).resolve().parents[1] / "r16p19" / "phase2_executor.py"
    ).read_text(encoding="utf-8")
    assert "from .memory" not in source
    assert "import memory" not in source
    assert "effect_truths(" not in source
    assert "fault_identity" not in source.split("class RetargetedGeometricSkillExecutor", 1)[0]


def test_calibration_resolves_relative_manifest_before_project_provenance():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "calibrate_phase2_templates.py"
    ).read_text(encoding="utf-8")
    assert "Path(extraction_manifest_path).resolve()" in source
    assert "extraction_manifest_path.relative_to(PROJECT_ROOT)" in source
