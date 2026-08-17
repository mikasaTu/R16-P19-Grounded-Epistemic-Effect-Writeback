"""Exact serialized shared-prefix snapshots for paired arm execution."""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class SharedPrefixSnapshot:
    physics_state: bytes
    controller_state: bytes
    policy_cache: bytes
    numpy_rng: bytes
    python_rng: bytes
    observation: bytes
    event_prefix: bytes
    action_prefix: bytes
    terminal_state: bytes

    @classmethod
    def capture(cls, **fields: Any) -> "SharedPrefixSnapshot":
        expected = set(cls.__dataclass_fields__)
        if set(fields) != expected:
            raise ValueError(f"snapshot fields mismatch: {sorted(set(fields) ^ expected)}")
        return cls(**{key: pickle.dumps(fields[key], protocol=4) for key in expected})

    def sha256(self) -> str:
        digest = hashlib.sha256()
        for key in sorted(self.__dataclass_fields__):
            value = getattr(self, key)
            digest.update(key.encode("utf-8"))
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
        return digest.hexdigest()

    def field_hashes(self) -> Dict[str, str]:
        return {key: hashlib.sha256(getattr(self, key)).hexdigest() for key in sorted(self.__dataclass_fields__)}

    def restore(self) -> Dict[str, Any]:
        return {key: pickle.loads(getattr(self, key)) for key in self.__dataclass_fields__}
