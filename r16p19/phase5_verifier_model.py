"""Small, auditable linear and MLP effect verifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .phase5_verifier_data import load_split


def sigmoid(value):
    value = np.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-value))


def auroc(labels, scores) -> float:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    positive = labels == 1
    negative = ~positive
    if not positive.any() or not negative.any():
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    for group, count in enumerate(counts):
        if count > 1:
            mask = inverse == group
            ranks[mask] = ranks[mask].mean()
    n_pos = positive.sum()
    n_neg = negative.sum()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def metrics(labels, scores, threshold: float) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    predicted = scores >= threshold
    positive = labels == 1
    negative = ~positive
    tpr = float(predicted[positive].mean()) if positive.any() else float("nan")
    fpr = float(predicted[negative].mean()) if negative.any() else float("nan")
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (scores >= lo) & (scores < hi if hi < 1 else scores <= hi)
        if mask.any():
            ece += mask.mean() * abs(scores[mask].mean() - labels[mask].mean())
    return {"auroc": auroc(labels, scores), "ece": float(ece), "tpr": tpr, "fpr": fpr, "accuracy": float((predicted == positive).mean())}


class Standardizer:
    def fit(self, x):
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, x):
        return (x - self.mean) / self.std


class LinearVerifier:
    def fit(self, x, y, steps=1200, learning_rate=0.03):
        rng = np.random.default_rng(17)
        self.w = rng.normal(0, 0.01, size=x.shape[1]).astype(np.float32)
        self.b = 0.0
        positive_weight = float((y == 0).sum() / max(1, (y == 1).sum()))
        weights = np.where(y > 0.5, positive_weight, 1.0)
        for _ in range(steps):
            p = sigmoid(x @ self.w + self.b)
            error = (p - y) * weights
            self.w -= learning_rate * ((x.T @ error) / len(x) + 1e-4 * self.w)
            self.b -= learning_rate * float(error.mean())
        return self

    def predict(self, x):
        return sigmoid(x @ self.w + self.b)


class MLPVerifier:
    def fit(self, x, y, steps=1600, learning_rate=0.01, hidden=32):
        rng = np.random.default_rng(29)
        self.w1 = (rng.normal(size=(x.shape[1], hidden)) / np.sqrt(x.shape[1])).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.w2 = (rng.normal(size=hidden) / np.sqrt(hidden)).astype(np.float32)
        self.b2 = 0.0
        positive_weight = float((y == 0).sum() / max(1, (y == 1).sum()))
        weights = np.where(y > 0.5, positive_weight, 1.0)
        for _ in range(steps):
            hpre = x @ self.w1 + self.b1
            h = np.maximum(hpre, 0.0)
            p = sigmoid(h @ self.w2 + self.b2)
            dz = (p - y) * weights / len(x)
            dw2 = h.T @ dz + 1e-4 * self.w2
            db2 = float(dz.sum())
            dh = dz[:, None] * self.w2[None, :]
            dh[hpre <= 0] = 0
            dw1 = x.T @ dh + 1e-4 * self.w1
            db1 = dh.sum(axis=0)
            self.w2 -= learning_rate * dw2
            self.b2 -= learning_rate * db2
            self.w1 -= learning_rate * dw1
            self.b1 -= learning_rate * db1
        return self

    def predict(self, x):
        return sigmoid(np.maximum(x @ self.w1 + self.b1, 0.0) @ self.w2 + self.b2)


def choose_threshold(labels, scores) -> float:
    candidates = np.unique(np.concatenate([[0.0, 1.0], np.asarray(scores)]))
    feasible = []
    ranked = []
    for threshold in candidates:
        value = metrics(labels, scores, float(threshold))
        ranked.append((value["tpr"] - value["fpr"], -threshold, threshold))
        if value["tpr"] >= 0.8 and value["fpr"] <= 0.1:
            feasible.append(threshold)
    return float(max(feasible)) if feasible else float(max(ranked)[2])


def fit_calibrator(labels, scores, steps: int = 1200, learning_rate: float = 0.03):
    labels = np.asarray(labels, dtype=np.float64)
    logits = np.log(np.clip(scores, 1e-6, 1 - 1e-6) / np.clip(1 - scores, 1e-6, 1 - 1e-6))
    scale, bias = 1.0, 0.0
    for _ in range(steps):
        probability = sigmoid(scale * logits + bias)
        error = probability - labels
        scale -= learning_rate * float(np.mean(error * logits))
        bias -= learning_rate * float(np.mean(error))
    return float(scale), float(bias)


def apply_calibrator(scores, scale: float, bias: float):
    logits = np.log(np.clip(scores, 1e-6, 1 - 1e-6) / np.clip(1 - scores, 1e-6, 1 - 1e-6))
    return sigmoid(scale * logits + bias)


def train(root: Path, output: Path) -> dict:
    x_train, y_train, _ = load_split(root, "natural", policy_seeds=range(5))
    x_cal, y_cal, _ = load_split(root, "calibration")
    x_qual, y_qual, qual_groups = load_split(root, "qualification")
    standardizer = Standardizer().fit(x_train)
    train_x = standardizer.transform(x_train)
    cal_x = standardizer.transform(x_cal)
    qual_x = standardizer.transform(x_qual)
    models = {"linear": LinearVerifier().fit(train_x, y_train), "small_mlp": MLPVerifier().fit(train_x, y_train)}
    model_rows = {}
    for name, model in models.items():
        raw_cal_scores = model.predict(cal_x)
        calibration_scale, calibration_bias = fit_calibrator(y_cal, raw_cal_scores)
        cal_scores = apply_calibrator(raw_cal_scores, calibration_scale, calibration_bias)
        threshold = choose_threshold(y_cal, cal_scores)
        qual_scores = apply_calibrator(model.predict(qual_x), calibration_scale, calibration_bias)
        per_group = []
        for task, effect in sorted(set(map(tuple, qual_groups.tolist()))):
            mask = (qual_groups[:, 0] == task) & (qual_groups[:, 1] == effect)
            per_group.append({"task_id": int(task), "effect_index": int(effect), **metrics(y_qual[mask], qual_scores[mask], threshold)})
        finite_aurocs = [row["auroc"] for row in per_group if np.isfinite(row["auroc"])]
        summary = metrics(y_qual, qual_scores, threshold)
        shifts = {
            "occlusion": qual_x.copy(),
            "lighting_shift": qual_x.copy(),
            "camera_corruption": qual_x.copy(),
            "stale_frames": np.roll(qual_x, 1, axis=0),
            "sensor_dropout": qual_x.copy(),
        }
        shifts["occlusion"][:, :6] = 0
        shifts["lighting_shift"][:, :12] *= 0.65
        shifts["camera_corruption"][:, 6:12] = 0
        shifts["sensor_dropout"][:, 18:27] = 0
        shift_metrics = {key: metrics(y_qual, apply_calibrator(model.predict(value), calibration_scale, calibration_bias), threshold) for key, value in shifts.items()}
        summary.update({"threshold": threshold, "calibration_scale": calibration_scale, "calibration_bias": calibration_bias, "macro_auroc": float(np.mean(finite_aurocs)) if finite_aurocs else float("nan"), "min_tpr": float(np.nanmin([row["tpr"] for row in per_group])), "max_fpr": float(np.nanmax([row["fpr"] for row in per_group])), "per_task_effect": per_group, "shift_metrics": shift_metrics})
        summary["qualified"] = summary["macro_auroc"] >= 0.9 and summary["ece"] <= 0.05 and summary["min_tpr"] >= 0.8 and summary["max_fpr"] <= 0.1
        model_rows[name] = summary
    selected = max(model_rows, key=lambda name: (model_rows[name]["qualified"], model_rows[name]["macro_auroc"], -model_rows[name]["ece"], name == "linear"))
    model = models[selected]
    payload = {"model_type": np.asarray(selected), "mean": standardizer.mean, "std": standardizer.std, "threshold": np.asarray(model_rows[selected]["threshold"], dtype=np.float32), "calibration_scale": np.asarray(model_rows[selected]["calibration_scale"], dtype=np.float32), "calibration_bias": np.asarray(model_rows[selected]["calibration_bias"], dtype=np.float32)}
    if selected == "linear":
        payload.update({"w": model.w, "b": np.asarray(model.b, dtype=np.float32)})
    else:
        payload.update({"w1": model.w1, "b1": model.b1, "w2": model.w2, "b2": np.asarray(model.b2, dtype=np.float32)})
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    report = {"schema_version": 1, "training_samples": len(y_train), "calibration_samples": len(y_cal), "qualification_samples": len(y_qual), "formal_samples_accessed": 0, "models": model_rows, "selected": selected, "selected_qualified": model_rows[selected]["qualified"]}
    output.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def predict(checkpoint: Path, x: np.ndarray) -> Tuple[np.ndarray, float]:
    with np.load(checkpoint, allow_pickle=False) as data:
        z = (x - data["mean"]) / data["std"]
        model_type = str(data["model_type"])
        if model_type == "linear":
            scores = sigmoid(z @ data["w"] + float(data["b"]))
        else:
            scores = sigmoid(np.maximum(z @ data["w1"] + data["b1"], 0.0) @ data["w2"] + float(data["b2"]))
        scores = apply_calibrator(scores, float(data["calibration_scale"]), float(data["calibration_bias"]))
        return scores, float(data["threshold"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(train(args.rollout_root, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
