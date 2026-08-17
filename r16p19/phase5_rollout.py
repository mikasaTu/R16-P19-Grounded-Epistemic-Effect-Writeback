"""Resumable real LIBERO trajectory collection for Phase-5."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np

from .phase5_libero_env import TASKS, TASK_PROMPTS, compact_features, make_environment, observation_sha256
from .phase5_policy_broker import FrozenPolicyBroker
from .phase5_policy_server import FrozenPi05PolicyProcess
from .phase5_types import PolicyRequest


@dataclass(frozen=True)
class EpisodeSpec:
    split: str
    task_id: int
    init_index: int
    policy_seed: int
    variant: str = "clean"

    @property
    def episode_id(self) -> str:
        return f"{self.split}-t{self.task_id:02d}-i{self.init_index:02d}-s{self.policy_seed:02d}-{self.variant}"


def schedule() -> List[EpisodeSpec]:
    rows: List[EpisodeSpec] = []
    for task in TASKS:
        rows.extend(EpisodeSpec("qualification", task, init, 0) for init in range(20, 30))
        rows.extend(EpisodeSpec("pilot", task, init, 0, "noop") for init in range(20, 25))
        rows.extend(EpisodeSpec("natural", task, init, seed) for init in range(0, 10) for seed in range(7))
        rows.extend(EpisodeSpec("calibration", task, init, 0) for init in range(10, 20))
        rows.extend(EpisodeSpec("formal", task, init, seed, variant) for init in range(30, 50) for seed in range(2) for variant in ("clean", "noop"))
    return rows


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(prefix=f".{path.stem}.", suffix=".npz", dir=path.parent, delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
        directory_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _first_marker(root: Path, row: dict) -> None:
    destination = root / "FIRST_COMPLETED_ROLLOUT.json"
    try:
        descriptor = os.open(str(destination), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(row, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _seed_from_request(request: PolicyRequest) -> int:
    return int(request.key()[:8], 16)


def collect_one(spec: EpisodeSpec, output_root: Path, policy: FrozenPi05PolicyProcess, pairing_budget: list[int] | None = None) -> dict:
    destination = output_root / "episodes" / spec.split / f"{spec.episode_id}.npz"
    summary_path = destination.with_suffix(".json")
    if destination.is_file() and summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    environment_seed = 500000 + spec.task_id * 10000 + spec.init_index * 100 + spec.policy_seed
    environment = make_environment(spec.task_id, environment_seed)
    observations: dict[str, dict[str, Any]] = {}

    def infer(request: PolicyRequest) -> np.ndarray:
        return policy.infer(observations[request.observation_hash], _seed_from_request(request))

    broker = FrozenPolicyBroker(infer, capacity=1024)
    steps = []
    started = time.time()
    try:
        observation = environment.reset(spec.init_index)
        initial_predicates = environment.official_predicates()
        history = hashlib.sha256(b"phase5-empty-history").hexdigest()
        done = False
        success = False
        chunk_index = 0
        while not done:
            obs_hash = observation_sha256(observation)
            observations[obs_hash] = observation
            request = PolicyRequest(obs_hash, history, str(spec.task_id), "task_goal", "EXECUTE", "pi05-libero-frozen", "official-54cbaee6", spec.policy_seed)
            query_started = time.perf_counter()
            action_chunk = broker.action_chunk(request)
            policy_ms = (time.perf_counter() - query_started) * 1000.0
            executed = action_chunk[:5]
            injected_noop = spec.variant == "noop" and chunk_index == 2
            if injected_noop:
                executed = np.tile(np.asarray([0.0] * 6 + [-1.0], dtype=np.float32), (5, 1))
            prefix_snapshot = environment.capture_snapshot()
            trace = environment.execute_actions(executed)
            observation = environment.raw_observation()
            predicates = environment.official_predicates()
            snapshot = environment.capture_snapshot()
            terminal_hash = hashlib.sha256(snapshot.to_bytes()).hexdigest()
            forced_hashes = [terminal_hash]
            should_pair = spec.split in ("natural", "qualification") and pairing_budget is not None and pairing_budget[0] > 0
            if should_pair:
                for _ in range(4):
                    environment.restore_snapshot(prefix_snapshot)
                    environment.execute_actions(executed)
                    forced_hashes.append(hashlib.sha256(environment.capture_snapshot().to_bytes()).hexdigest())
                environment.restore_snapshot(snapshot)
                pairing_budget[0] -= 1
            controller_hash = hashlib.sha256(pickle.dumps(snapshot.controller_state, protocol=4)).hexdigest()
            rng_hash = hashlib.sha256(pickle.dumps((snapshot.numpy_rng_state, snapshot.python_rng_state), protocol=4)).hexdigest()
            contact = int(trace["contact_count"][-1]) if len(trace["contact_count"]) else 0
            features = compact_features(observation, contact)
            action_hash = hashlib.sha256(np.ascontiguousarray(executed).tobytes()).hexdigest()
            history = hashlib.sha256((history + action_hash).encode("ascii")).hexdigest()
            steps.append({
                "chunk_index": chunk_index,
                "observation_sha256": obs_hash,
                "action_sha256": action_hash,
                "policy_request_key": request.key(),
                "policy_history_sha256": request.history_hash,
                "physics_state_sha256": hashlib.sha256(np.ascontiguousarray(snapshot.sim_state).tobytes()).hexdigest(),
                "prefix_physics_state_sha256": hashlib.sha256(np.ascontiguousarray(prefix_snapshot.sim_state).tobytes()).hexdigest(),
                "prefix_controller_state_sha256": hashlib.sha256(pickle.dumps(prefix_snapshot.controller_state, protocol=4)).hexdigest(),
                "prefix_rng_state_sha256": hashlib.sha256(pickle.dumps((prefix_snapshot.numpy_rng_state, prefix_snapshot.python_rng_state), protocol=4)).hexdigest(),
                "controller_state_sha256": controller_hash,
                "rng_state_sha256": rng_hash,
                "terminal_state_sha256": terminal_hash,
                "forced_identical_terminal_state": len(set(forced_hashes)) == 1 if should_pair else False,
                "pairing_qualified_unit": should_pair,
                "policy_latency_ms": policy_ms,
                "injected_noop": injected_noop,
                "predicate_values": predicates["values"],
                "predicate_fraction": predicates["fraction"],
                "reward": float(trace["raw_reward"]),
                "success": bool(trace["success"]),
                "done": bool(trace["done"]),
                "executed_steps": int(trace["executed_steps"]),
                **features,
            })
            done = bool(trace["done"])
            success = bool(trace["success"])
            chunk_index += 1
            if chunk_index > 104:
                raise RuntimeError("rollout exceeded official 520-step budget")
    finally:
        environment.close()

    arrays = {
        "base_rgb_32": np.stack([row["base_rgb_32"] for row in steps]),
        "wrist_rgb_32": np.stack([row["wrist_rgb_32"] for row in steps]),
        "proprio": np.stack([row["proprio"] for row in steps]),
        "contact_count": np.asarray([row["contact_count"] for row in steps], dtype=np.int32),
        "predicate_values": np.asarray([row["predicate_values"] for row in steps], dtype=np.bool_),
        "predicate_fraction": np.asarray([row["predicate_fraction"] for row in steps], dtype=np.float32),
        "reward": np.asarray([row["reward"] for row in steps], dtype=np.float32),
        "success": np.asarray([row["success"] for row in steps], dtype=np.bool_),
        "done": np.asarray([row["done"] for row in steps], dtype=np.bool_),
        "executed_steps": np.asarray([row["executed_steps"] for row in steps], dtype=np.int32),
        "injected_noop": np.asarray([row["injected_noop"] for row in steps], dtype=np.bool_),
        "policy_latency_ms": np.asarray([row["policy_latency_ms"] for row in steps], dtype=np.float32),
        "observation_sha256": np.asarray([row["observation_sha256"] for row in steps]),
        "action_sha256": np.asarray([row["action_sha256"] for row in steps]),
        "policy_request_key": np.asarray([row["policy_request_key"] for row in steps]),
        "policy_history_sha256": np.asarray([row["policy_history_sha256"] for row in steps]),
        "physics_state_sha256": np.asarray([row["physics_state_sha256"] for row in steps]),
        "prefix_physics_state_sha256": np.asarray([row["prefix_physics_state_sha256"] for row in steps]),
        "prefix_controller_state_sha256": np.asarray([row["prefix_controller_state_sha256"] for row in steps]),
        "prefix_rng_state_sha256": np.asarray([row["prefix_rng_state_sha256"] for row in steps]),
        "controller_state_sha256": np.asarray([row["controller_state_sha256"] for row in steps]),
        "rng_state_sha256": np.asarray([row["rng_state_sha256"] for row in steps]),
        "terminal_state_sha256": np.asarray([row["terminal_state_sha256"] for row in steps]),
        "forced_identical_terminal_state": np.asarray([row["forced_identical_terminal_state"] for row in steps], dtype=np.bool_),
        "pairing_qualified_unit": np.asarray([row["pairing_qualified_unit"] for row in steps], dtype=np.bool_),
        "initial_predicate_values": np.asarray(initial_predicates["values"], dtype=np.bool_),
        "predicate_labels": np.asarray(initial_predicates["labels"]),
    }
    _atomic_npz(destination, **arrays)
    summary = {
        **asdict(spec),
        "episode_id": spec.episode_id,
        "complete": True,
        "success": success,
        "chunks": len(steps),
        "action_steps": int(sum(row["executed_steps"] for row in steps)),
        "backend_errors": 0,
        "policy_inference_count": int(sum(broker.inference_count.values())),
        "policy_latency_ms_mean": float(np.mean(arrays["policy_latency_ms"])),
        "trajectory_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "wall_seconds": time.time() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _first_marker(output_root, summary)
    return summary


def collect(output_root: Path, rank: int, world_size: int, only_split: str | None = None, max_episodes: int = 0) -> dict:
    jobs = [item for item in schedule() if only_split is None or item.split == only_split]
    jobs = [item for index, item in enumerate(jobs) if index % world_size == rank]
    if max_episodes:
        jobs = jobs[:max_episodes]
    checkpoint = os.environ["R16P19_PI05_CHECKPOINT"]
    output_root.mkdir(parents=True, exist_ok=True)
    completed = []
    pairing_budget = [125]
    default_prompt = TASK_PROMPTS[next(iter(TASKS))]
    with FrozenPi05PolicyProcess(checkpoint, default_prompt) as policy:
        contract_path = output_root / f"policy-server-contract-rank-{rank:02d}.json"
        if not contract_path.exists():
            contract_path.write_text(json.dumps(policy.contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for spec in jobs:
            row = collect_one(spec, output_root, policy, pairing_budget)
            completed.append(row)
            print(json.dumps({"event": "completed_rollout", **row}, sort_keys=True), flush=True)
    result = {"rank": rank, "world_size": world_size, "assigned": len(jobs), "completed": len(completed), "successes": sum(row["success"] for row in completed)}
    (output_root / f"worker-{rank:02d}-complete.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    parser.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    parser.add_argument("--only-split", choices=("qualification", "pilot", "natural", "calibration", "formal"))
    parser.add_argument("--max-episodes", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(collect(args.output_root, args.rank, args.world_size, args.only_split, args.max_episodes), sort_keys=True))


if __name__ == "__main__":
    main()
