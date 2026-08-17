"""Separate CUDA process for the frozen official pi0.5 LIBERO policy."""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import traceback
from typing import Any

import numpy as np


def _server(connection, checkpoint: str, default_prompt: str) -> None:
    try:
        qpilots_root = os.environ["R16P19_QPILOTS_ROOT"]
        openpi_root = os.environ["QPILOTS_OPENPI_ROOT"]
        import sys

        sys.path.insert(0, qpilots_root)
        sys.path.insert(0, f"{openpi_root}/src")
        from qpilots_libero.policy import CleanPi05LiberoPolicy

        policy = CleanPi05LiberoPolicy(checkpoint, default_prompt=default_prompt, config_name="pi05_libero")
        connection.send({"ready": True, "checkpoint": policy.checkpoint_identity(), "contract": policy.live_contract})
        while True:
            request = connection.recv()
            if request is None:
                break
            observation, seed = request
            key = policy.jax.random.PRNGKey(int(seed) & 0xFFFFFFFF)
            noise = np.asarray(policy.jax.random.normal(key, (10, 32)), dtype=np.float32)
            actions = policy.infer_official(observation, noise)
            connection.send({"actions": actions, "sha256": hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest()})
    except BaseException as exc:  # process boundary must return the decisive traceback
        connection.send({"error": repr(exc), "traceback": traceback.format_exc()})
    finally:
        connection.close()


class FrozenPi05PolicyProcess:
    def __init__(self, checkpoint: str, default_prompt: str) -> None:
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        self._connection = parent
        self._process = context.Process(target=_server, args=(child, checkpoint, default_prompt), daemon=True)
        self._process.start()
        message = parent.recv()
        if "error" in message:
            raise RuntimeError(f"policy process initialization failed: {message['error']}\n{message['traceback']}")
        self.contract = message

    def infer(self, observation: dict[str, Any], seed: int) -> np.ndarray:
        self._connection.send((observation, int(seed)))
        message = self._connection.recv()
        if "error" in message:
            raise RuntimeError(f"policy inference failed: {message['error']}\n{message['traceback']}")
        actions = np.asarray(message["actions"], dtype=np.float32)
        if actions.shape != (10, 7):
            raise RuntimeError(f"official pi0.5 action shape drifted: {actions.shape}")
        if hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest() != message["sha256"]:
            raise RuntimeError("policy action bytes changed across the process boundary")
        return actions

    def close(self) -> None:
        if getattr(self, "_process", None) is None:
            return
        if self._process.is_alive():
            self._connection.send(None)
            self._process.join(timeout=10)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        self._connection.close()
        self._process = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
