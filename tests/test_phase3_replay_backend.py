import inspect
import json

import numpy as np

import r16p19.phase3_replay_backend as replay_module
from r16p19.phase3_replay_backend import ExecutionMode, FrozenEffectReplayBackend
from r16p19.phase3_snapshot_bank import array_sha256, sha256_file


class FakeSim:
    def __init__(self):
        self.state = np.zeros(2, dtype=np.float64)

    def get_state(self):
        return self

    def flatten(self):
        return self.state.copy()


class FakeEnv:
    def __init__(self):
        self.sim = FakeSim()

    def step(self, action):
        self.sim.state[0] += float(action[0])
        return {"state": self.sim.state.copy()}, 0.0, False, {}

    def close(self):
        pass


def _backend(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshots" / "qualification" / "stove_moka" / "demo_30" / "STOVE_TURNED_ON.npz"
    snapshot.parent.mkdir(parents=True)
    actions = np.zeros((5, 7), dtype="<f4")
    actions[:, 0] = 1.0
    entry = np.zeros(2, dtype="<f8")
    np.savez_compressed(
        snapshot,
        entry_state=entry,
        actions=actions,
        stable_post_state=np.asarray([5.0, 0.0], dtype="<f8"),
        next_effect_entry_state=np.asarray([5.0, 0.0], dtype="<f8"),
        effect_truth_timeline=np.ones(5, dtype=np.bool_),
    )
    manifest = {
        "segments": [
            {
                "split": "qualification",
                "task_key": "stove_moka",
                "source_episode": "demo_30",
                "effect_id": "STOVE_TURNED_ON",
                "snapshot_path": str(snapshot.relative_to(tmp_path)),
                "snapshot_sha256": sha256_file(snapshot),
                "entry_state_sha256": array_sha256(entry, "<f8"),
                "action_segment_sha256": array_sha256(actions, "<f4"),
                "action_count": 5,
                "valid": True,
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    environment = FakeEnv()
    monkeypatch.setattr(replay_module, "make_env", lambda task, camera_obs=False: environment)
    monkeypatch.setattr(
        replay_module,
        "reset_to_state",
        lambda env, state, seed: _fake_reset(env, state),
    )
    monkeypatch.setattr(
        replay_module,
        "effect_truths",
        lambda env, task: {effect: env.sim.state[0] >= 1.0 for effect in task.effects},
    )
    return FrozenEffectReplayBackend(manifest_path, tmp_path), environment


def _fake_reset(env, state):
    env.sim.state = np.asarray(state, dtype=np.float64).copy()
    return {"state": env.sim.state.copy()}


def test_backend_replays_exact_actions_and_reobserve_is_eight_zero_steps(tmp_path, monkeypatch):
    backend, environment = _backend(tmp_path, monkeypatch)
    segment = backend.segment("qualification", "stove_moka", "demo_30", "STOVE_TURNED_ON")
    result = backend.execute_effect(
        "qualification",
        "stove_moka",
        "demo_30",
        "STOVE_TURNED_ON",
        backend.snapshot_id(segment),
        ExecutionMode.EXECUTE,
        seed=1619,
    )
    assert result.action_steps == 5
    assert result.physical_truth_after is True
    assert result.predicate_stability_duration == 5
    state_before = environment.sim.state.copy()
    observed = backend.execute_effect(
        "qualification",
        "stove_moka",
        "demo_30",
        "STOVE_TURNED_ON",
        backend.snapshot_id(segment),
        ExecutionMode.REOBSERVE,
        seed=1619,
    )
    assert observed.action_steps == 8
    np.testing.assert_array_equal(environment.sim.state, state_before)


def test_backend_source_has_no_memory_or_fault_import():
    source = inspect.getsource(replay_module)
    assert "r16p19.memory" not in source
    assert "phase3_baselines" not in source
    assert "fault_identity" not in source.replace("fault_identity_input", "")
