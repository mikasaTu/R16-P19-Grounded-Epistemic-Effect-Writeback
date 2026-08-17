"""Arm-blind, content-addressed broker for frozen policy action chunks."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Callable, Dict, Tuple

import numpy as np

from .phase5_types import PolicyRequest


class PolicyBrokerError(RuntimeError):
    pass


class FrozenPolicyBroker:
    def __init__(self, infer: Callable[[PolicyRequest], np.ndarray], capacity: int = 4096) -> None:
        self._infer = infer
        self.capacity = int(capacity)
        self._cache: "OrderedDict[str, bytes]" = OrderedDict()
        self._shape: Dict[str, Tuple[int, ...]] = {}
        self.inference_count: Dict[str, int] = {}

    def action_chunk(self, request: PolicyRequest) -> np.ndarray:
        key = request.key()
        if key not in self._cache:
            value = np.asarray(self._infer(request), dtype=np.float32)
            if value.ndim != 2 or value.shape[-1] != 7 or not np.all(np.isfinite(value)):
                raise PolicyBrokerError(f"invalid policy action shape/value: {value.shape}")
            raw = np.ascontiguousarray(value).tobytes()
            self._cache[key] = raw
            self._shape[key] = tuple(value.shape)
            self.inference_count[key] = self.inference_count.get(key, 0) + 1
            while len(self._cache) > self.capacity:
                old, _ = self._cache.popitem(last=False)
                self._shape.pop(old, None)
                self.inference_count.pop(old, None)
        return np.frombuffer(self._cache[key], dtype=np.float32).reshape(self._shape[key]).copy()

    def action_sha256(self, request: PolicyRequest) -> str:
        chunk = self.action_chunk(request)
        return hashlib.sha256(np.ascontiguousarray(chunk).tobytes()).hexdigest()

    def cache_manifest(self) -> dict:
        return {
            "keys": list(self._cache),
            "action_sha256": {key: hashlib.sha256(value).hexdigest() for key, value in self._cache.items()},
            "inference_count": dict(self.inference_count),
        }
