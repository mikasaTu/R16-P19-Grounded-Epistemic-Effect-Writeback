"""Correctness, latency and hot-memory benchmark for the bounded ledger."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from .phase5_ledger_live import BoundedASCELLedger
from .phase5_ledger_reference import ReferenceLedger
from .phase5_property_generator import workload


def _dispatch(ledger, kind, payload, epoch):
    if kind == "request":
        ledger.request(*payload, epoch)
        return None
    if kind == "revoke":
        ledger.revoke(payload, epoch)
        return None
    return ledger.ingest(payload, epoch)


def _quantile(values, q):
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def run(events: int = 100000, attempts: int = 10000, seed: int = 1701) -> dict:
    live = BoundedASCELLedger()
    reference = ReferenceLedger()
    latencies_ms = []
    tick_latencies_ms = []
    mismatches = 0
    for kind, payload, epoch in workload(events, attempts, seed):
        start = time.perf_counter_ns()
        left = _dispatch(live, kind, payload, epoch)
        latencies_ms.append((time.perf_counter_ns() - start) / 1e6)
        right = _dispatch(reference, kind, payload, epoch)
        if (left is None) != (right is None) or (left is not None and left.to_dict() != right.to_dict()):
            mismatches += 1
        if live.visible_state() != reference.visible_state():
            mismatches += 1
        if len(latencies_ms) % 4 == 0:
            tick_latencies_ms.append(sum(latencies_ms[-4:]))
    event_p99 = _quantile(latencies_ms, 0.99)
    p99_at_1k = _quantile(latencies_ms[: min(1000, len(latencies_ms))], 0.99)
    scaling_ratio = event_p99 / max(p99_at_1k, 1e-12)
    tick_p99 = _quantile(tick_latencies_ms, 0.99)
    memory_mb = live.hot_memory_bytes() / (1024.0 * 1024.0)
    summary = {
        "schema_version": 1,
        "events": events,
        "attempts_target": attempts,
        "seed": seed,
        "python": platform.python_version(),
        "event_latency_ms": {"p50": _quantile(latencies_ms, 0.5), "p95": _quantile(latencies_ms, 0.95), "p99": event_p99, "mean": statistics.fmean(latencies_ms)},
        "tick_latency_p99_ms": tick_p99,
        "latency_scaling": {"p99_at_1k_ms": p99_at_1k, "p99_at_100k_ms": event_p99, "ratio": scaling_ratio},
        "hot_memory_mb": memory_mb,
        "exact_reference_mismatches": mismatches,
        "audit_chain_breaks": 0 if live.audit.verify() else 1,
        "audit_summary": live.audit.compact_summary(),
    }
    summary["gates"] = {
        "event_p99_le_1ms": event_p99 <= 1.0,
        "tick_p99_le_2ms": tick_p99 <= 2.0,
        "p99_scaling_ratio_le_1_2": scaling_ratio <= 1.2,
        "hot_memory_le_10mb": memory_mb <= 10.0,
        "reference_mismatch_zero": mismatches == 0,
        "audit_chain_breaks_zero": summary["audit_chain_breaks"] == 0,
    }
    summary["pass_partial"] = all(summary["gates"].values())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=100000)
    parser.add_argument("--attempts", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.events, args.attempts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
