"""Deterministic bounded-container helpers."""

from __future__ import annotations

from collections import OrderedDict, deque
from typing import Deque, Dict, Hashable, MutableMapping, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


def insert_lru(mapping: "OrderedDict[K, V]", key: K, value: V, capacity: int) -> None:
    mapping.pop(key, None)
    mapping[key] = value
    while len(mapping) > capacity:
        mapping.popitem(last=False)


def append_bounded(mapping: MutableMapping[K, Deque[V]], key: K, value: V, capacity: int) -> None:
    bucket = mapping.setdefault(key, deque(maxlen=capacity))
    bucket.append(value)


def bounded_size(mapping: Dict[object, object]) -> int:
    """Portable structural size estimate used by property tests."""
    import sys

    seen = set()

    def walk(value: object) -> int:
        identity = id(value)
        if identity in seen:
            return 0
        seen.add(identity)
        total = sys.getsizeof(value)
        if isinstance(value, dict):
            total += sum(walk(k) + walk(v) for k, v in value.items())
        elif isinstance(value, (list, tuple, set, frozenset, deque, OrderedDict)):
            total += sum(walk(item) for item in value)
        elif hasattr(value, "__dict__"):
            total += walk(vars(value))
        return total

    return walk(mapping)
