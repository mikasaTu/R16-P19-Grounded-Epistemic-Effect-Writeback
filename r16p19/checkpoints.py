"""Complete-state PyTorch checkpoints with single-writer fail-closed retention."""

from __future__ import annotations

import fcntl
import json
import os
import random
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch


def capture_rng_state() -> dict:
    value = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        value["torch_cuda"] = torch.cuda.get_rng_state_all()
    return value


def restore_rng_state(value: dict) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"].cpu())
    if torch.cuda.is_available() and "torch_cuda" in value:
        torch.cuda.set_rng_state_all([item.cpu() for item in value["torch_cuda"]])


def _fsync_path(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class CheckpointManager:
    def __init__(self, root: Path, latest_nonmilestones: int = 3, milestone_interval: int = 10000):
        self.root = Path(root)
        self.keep_latest = int(latest_nonmilestones)
        self.milestone_interval = int(milestone_interval)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.root / ".writer.lock"

    @contextmanager
    def writer_lock(self) -> Iterator[None]:
        descriptor = os.open(str(self.lock_path), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _step_dir(self, step: int) -> Path:
        return self.root / ("step_%09d" % int(step))

    def complete_steps(self) -> List[int]:
        result = []
        for path in self.root.glob("step_*"):
            try:
                step = int(path.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            if (path / "COMPLETE").is_file() and (path / "state.pt").is_file():
                result.append(step)
        return sorted(result)

    def incomplete_paths(self) -> List[str]:
        result = []
        for path in sorted(self.root.iterdir()):
            if path.name.startswith("step_") and path.is_dir():
                if not (path / "COMPLETE").is_file() or not (path / "state.pt").is_file():
                    result.append(str(path))
            elif path.name.startswith(".step_") and ".tmp." in path.name:
                result.append(str(path))
        return result

    def save(self, step, model, optimizer, scheduler=None, extra=None) -> Path:
        step = int(step)
        with self.writer_lock():
            final = self._step_dir(step)
            if (final / "COMPLETE").is_file() and (final / "state.pt").is_file():
                return final
            if final.exists():
                raise RuntimeError("incomplete final checkpoint blocks overwrite: %s" % final)
            temporary = Path(tempfile.mkdtemp(prefix=".%s.tmp." % final.name, dir=str(self.root)))
            try:
                payload = {
                    "global_step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "rng": capture_rng_state(),
                    "extra": dict(extra or {}),
                }
                torch.save(payload, str(temporary / "state.pt"))
                _fsync_path(temporary / "state.pt")
                with (temporary / "metadata.json").open("w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "global_step": step,
                            "complete_state": ["model", "optimizer", "scheduler", "rng", "global_step"],
                        },
                        handle,
                        sort_keys=True,
                    )
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                with (temporary / "COMPLETE").open("w", encoding="utf-8") as handle:
                    handle.write("global_step=%d\n" % step)
                    handle.flush()
                    os.fsync(handle.fileno())
                _fsync_path(temporary)
                os.replace(str(temporary), str(final))
                _fsync_path(self.root)
            except Exception:
                # Preserve partial state for diagnosis; startup never treats it as resumable.
                raise
            self._enforce_retention_locked()
            self._verify_retention_locked()
            return final

    def _retained_set(self, steps: List[int]) -> set:
        milestones = {
            step for step in steps if step > 0 and step % self.milestone_interval == 0
        }
        nonmilestones = [step for step in steps if step not in milestones]
        return milestones | set(nonmilestones[-self.keep_latest :])

    def _enforce_retention_locked(self) -> None:
        steps = self.complete_steps()
        keep = self._retained_set(steps)
        for step in steps:
            if step in keep:
                continue
            target = self._step_dir(step)
            if target.parent != self.root or not target.name.startswith("step_"):
                raise RuntimeError("unsafe retention target")
            shutil.rmtree(str(target))
        _fsync_path(self.root)

    def _verify_retention_locked(self) -> None:
        steps = self.complete_steps()
        milestones = [
            step for step in steps if step > 0 and step % self.milestone_interval == 0
        ]
        if any(not (self._step_dir(step) / "COMPLETE").is_file() for step in milestones):
            raise RuntimeError("retention lost a complete milestone")
        nonmilestones = [step for step in steps if step not in milestones]
        if len(nonmilestones) > self.keep_latest:
            raise RuntimeError("retention failed to bound non-milestones")

    def latest(self) -> Optional[Path]:
        steps = self.complete_steps()
        return self._step_dir(steps[-1]) if steps else None

    def load_latest(self, model, optimizer=None, scheduler=None, map_location="cpu") -> Tuple[int, dict]:
        latest = self.latest()
        if latest is None:
            return 0, {}
        payload = torch.load(str(latest / "state.pt"), map_location=map_location)
        model.load_state_dict(payload["model"])
        if optimizer is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        restore_rng_state(payload["rng"])
        return int(payload["global_step"]), dict(payload.get("extra", {}))


def synthetic_retention_test() -> dict:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory(prefix="r16p19-checkpoint-test-") as root:
        manager = CheckpointManager(Path(root))
        for step in range(1000, 15000, 1000):
            manager.save(step, model, optimizer)
        expected = [10000, 12000, 13000, 14000]
        observed = manager.complete_steps()
        if observed != expected:
            raise AssertionError("retention mismatch: %r != %r" % (observed, expected))
        incomplete = Path(root) / "step_000015000"
        incomplete.mkdir()
        (incomplete / "state.pt").write_bytes(b"partial")
        if str(incomplete) not in manager.incomplete_paths():
            raise AssertionError("incomplete checkpoint was not detected")
        if not incomplete.exists():
            raise AssertionError("incomplete checkpoint must be preserved")
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        manager.save(16000, model, optimizer)
        expected_rng = (random.random(), float(np.random.rand()), float(torch.rand(())))
        random.seed(9)
        np.random.seed(9)
        torch.manual_seed(9)
        manager.load_latest(model, optimizer)
        observed_rng = (random.random(), float(np.random.rand()), float(torch.rand(())))
        if not np.allclose(observed_rng, expected_rng):
            raise AssertionError("RNG resume mismatch")
        return {
            "status": "CHECKPOINT_RETENTION_OK",
            "retained_steps": manager.complete_steps(),
            "incomplete_preserved": True,
            "rng_resume_exact": True,
        }
