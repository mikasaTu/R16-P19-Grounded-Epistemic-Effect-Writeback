"""Leakage-free feature construction for the learned effect verifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


TASK_INDEX = {0: 0, 5: 1, 9: 2}
MAX_EFFECTS = 4


def _feature_rows(data: np.lib.npyio.NpzFile, task_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = data["base_rgb_32"].astype(np.float32) / 255.0
    wrist = data["wrist_rgb_32"].astype(np.float32) / 255.0
    proprio = data["proprio"].astype(np.float32)
    contact = np.log1p(data["contact_count"].astype(np.float32))[:, None]
    predicate = data["predicate_values"].astype(np.float32)
    chunks, effects = predicate.shape
    base_stats = np.concatenate([base.mean(axis=(1, 2)), base.std(axis=(1, 2))], axis=1)
    wrist_stats = np.concatenate([wrist.mean(axis=(1, 2)), wrist.std(axis=(1, 2))], axis=1)
    time = np.linspace(0.0, 1.0, chunks, dtype=np.float32)[:, None]
    previous = np.vstack([base_stats[:1], base_stats[:-1]])
    temporal = base_stats - previous
    shared = np.concatenate([base_stats, wrist_stats, temporal, proprio, contact, time], axis=1)
    task_onehot = np.zeros((chunks, 3), dtype=np.float32)
    task_onehot[:, TASK_INDEX[task_id]] = 1.0
    shared = np.concatenate([shared, task_onehot], axis=1)
    features = []
    labels = []
    effect_ids = []
    for effect in range(effects):
        effect_onehot = np.zeros((chunks, MAX_EFFECTS), dtype=np.float32)
        effect_onehot[:, effect] = 1.0
        features.append(np.concatenate([shared, effect_onehot], axis=1))
        labels.append(predicate[:, effect])
        effect_ids.append(np.full(chunks, effect, dtype=np.int32))
    return np.concatenate(features), np.concatenate(labels), np.concatenate(effect_ids)


def load_split(root: Path, split: str, policy_seeds: Iterable[int] | None = None):
    allowed = set(policy_seeds) if policy_seeds is not None else None
    features = []
    labels = []
    groups = []
    paths = sorted((root / "episodes" / split).glob("*.npz"))
    for path in paths:
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if allowed is not None and int(meta["policy_seed"]) not in allowed:
            continue
        with np.load(path, allow_pickle=False) as data:
            x, y, effects = _feature_rows(data, int(meta["task_id"]))
        features.append(x)
        labels.append(y)
        groups.extend([(int(meta["task_id"]), int(effect)) for effect in effects])
    if not features:
        raise RuntimeError(f"no verifier episodes for split={split}")
    return np.concatenate(features), np.concatenate(labels), np.asarray(groups, dtype=np.int32)


def formal_frame_features(path: Path, index: int) -> np.ndarray:
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as data:
        x, _, effects = _feature_rows(data, int(meta["task_id"]))
        chunks = len(data["predicate_fraction"])
        selected = [x[effect * chunks + min(index, chunks - 1)] for effect in range(len(data["predicate_labels"]))]
    return np.stack(selected)
