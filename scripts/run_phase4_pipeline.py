#!/usr/bin/env python3
"""Resumable CPU-only execution pipeline for R16-P19 Phase-4.

The GLFW library is resolved before importing MuJoCo.  This avoids GLFW's
system-library probe subprocess in restricted batch containers and does not
change the MuJoCo model, state, controller, or experiment thresholds.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import sysconfig
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
_BUNDLED_GLFW = (
    Path(sysconfig.get_paths()["purelib"]) / "glfw" / "x11" / "libglfw.so"
)
if "PYGLFW_LIBRARY" not in os.environ and _BUNDLED_GLFW.is_file():
    os.environ["PYGLFW_LIBRARY"] = str(_BUNDLED_GLFW)
    os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")

import mujoco
import numpy as np

from r16p19.artifacts import atomic_text, sha256_file, write_json, write_jsonl, write_sha256sums
from r16p19.phase4_analysis import (
    analyze_ablations,
    analyze_formal,
    failure_decomposition,
    final_decision,
    summarize_arm,
)
from r16p19.phase4_arms import ABLATION_ARMS, MAIN_ARMS, protected_b6_sha256
from r16p19.phase4_executor import qualify_executor
from r16p19.phase4_fork_runner import run_matrix, shared_prefix_qualification
from r16p19.phase4_trace_generator import generate_trace_schedules
from r16p19.phase4_trace_oracle import run_trace_gate


EXPERIMENT = PROJECT_ROOT / "experiments" / "r16p19_phase4"
CONTRACT = EXPERIMENT / "task_condition_contract.json"
PROTECTED_B6_SHA256 = "4992462e105306eca9a777619fa5c7418f90c1289e4e9f6fdde1bdc2fbfce4c5"
SOURCE_FILES = (
    "r16p19/phase4_types.py",
    "r16p19/phase4_attempt_ledger.py",
    "r16p19/phase4_support_graph.py",
    "r16p19/phase4_trace_generator.py",
    "r16p19/phase4_trace_oracle.py",
    "r16p19/phase4_microenv.py",
    "r16p19/phase4_executor.py",
    "r16p19/phase4_event_broker.py",
    "r16p19/phase4_fork_runner.py",
    "r16p19/phase4_analysis.py",
    "r16p19/phase4_arms.py",
    "scripts/run_phase4_pipeline.py",
)


def _log(message: str) -> None:
    print("[phase4] %s" % message, flush=True)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git"] + list(args), cwd=str(PROJECT_ROOT), text=True
    ).strip()


def _source_identity() -> dict:
    return {
        "branch": _git("branch", "--show-current"),
        "head": _git("rev-parse", "HEAD"),
        "tree": _git("rev-parse", "HEAD^{tree}"),
        "source_hashes": {
            name: sha256_file(PROJECT_ROOT / name) for name in SOURCE_FILES
        },
        "protected_b6_sha256": protected_b6_sha256(),
    }


def _assert_contract() -> None:
    if _git("branch", "--show-current") != "phase4-attempt-scoped-causal-ledger":
        raise RuntimeError("Phase-4 must run on its frozen isolated branch")
    if protected_b6_sha256() != PROTECTED_B6_SHA256:
        raise RuntimeError("protected r16p19/memory.py SHA256 mismatch")
    contract = _json(CONTRACT)
    if len(contract["cells"]) != 20:
        raise RuntimeError("frozen task-condition contract is not exactly 20 cells")
    if len(contract["pilot_seeds"]) != 10 or len(contract["formal_seeds"]) != 50:
        raise RuntimeError("frozen seed split drift")


def _record_provenance() -> None:
    uname = platform.uname()
    cpu_model = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[-1].strip()
                break
    write_json(
        EXPERIMENT / "runtime_provenance.json",
        {
            "schema_version": 1,
            "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": _source_identity(),
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "kernel": uname.release,
            "machine": uname.machine,
            "hostname": uname.node,
            "cpu_model": cpu_model,
            "numpy_version": np.__version__,
            "mujoco_version": mujoco.__version__,
            "execution": {
                "cpu_only": True,
                "gpu_used": False,
                "renderer_initialized": False,
                "os_fork_shared_prefix": True,
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                "PYGLFW_LIBRARY": os.environ.get("PYGLFW_LIBRARY"),
                "PYGLFW_LIBRARY_VARIANT": os.environ.get("PYGLFW_LIBRARY_VARIANT"),
            },
        },
    )


def trace_stage() -> None:
    _assert_contract()
    _log("trace: generating and evaluating 10,000 deterministic schedules")
    rows, summary = run_trace_gate(generate_trace_schedules(1000))
    if len(rows) != 10000:
        raise RuntimeError("trace schedule count mismatch")
    write_jsonl(EXPERIMENT / "trace_results.jsonl", rows)
    write_json(EXPERIMENT / "trace_summary.json", summary)
    _log("trace: %s (%d/%d passed)" % (
        summary["status"], sum(summary["family_pass_counts"].values()), len(rows)
    ))


def executor_stage() -> None:
    _assert_contract()
    _log("executor: qualifying seeds 40-59 on three CPU MuJoCo tasks")
    result = qualify_executor(range(40, 60))
    rows = result.pop("rows")
    write_jsonl(EXPERIMENT / "executor_qualification.jsonl", rows)
    write_json(EXPERIMENT / "executor_qualification_summary.json", result)
    _log("executor: pass=%s conditional=%.6f chain=%.6f backend_errors=%d" % (
        result["pass"], result["conditional_effect_success"],
        result["full_chain_success"], result["backend_error_count"]
    ))


def shared_prefix_stage() -> None:
    _assert_contract()
    _log("shared-prefix: running 1,000 forced-identical paired units")
    result = shared_prefix_qualification(1000)
    rows = []
    for row in result.pop("rows"):
        rows.append(dict(row, record_type="arm_result"))
    for audit in result.pop("audits"):
        rows.append(dict(audit, record_type="paired_unit_audit"))
    write_jsonl(EXPERIMENT / "shared_prefix_qualification.jsonl", rows)
    write_json(EXPERIMENT / "shared_prefix_qualification_summary.json", result)
    _log("shared-prefix: pass=%s units=%d child_failures=%d" % (
        result["pass"], result["unit_count"], result["child_process_failure_count"]
    ))


def _pilot_summary(rows: Sequence[Mapping[str, object]], audits: Sequence[Mapping[str, object]]) -> dict:
    m4 = summarize_arm(rows, "M4_ASCEL_FULL")
    executor = _json(EXPERIMENT / "executor_qualification_summary.json")
    shared = _json(EXPERIMENT / "shared_prefix_qualification_summary.json")
    m4_rows = [row for row in rows if row.get("arm") == "M4_ASCEL_FULL"]
    backend_errors = sum(int(row.get("backend_error_count", 0)) for row in rows)
    child_failures = sum(bool(row.get("child_process_failure")) for row in rows)
    attempt_leakage = sum(int(row.get("attempt_leakage_count", 0)) for row in m4_rows)
    invariant_violations = sum(
        int(row.get("support_graph_invariant_violation_count", 0)) for row in m4_rows
    )
    gates = {
        "shared_prefix_exact": bool(shared["pass"]) and all(audit["pass"] for audit in audits),
        "executor_conditional_success_ge_0_99": executor["conditional_effect_success"] >= 0.99,
        "broker_or_fault_injector_errors_zero": backend_errors == 0 and child_failures == 0,
        "m4_clean_success_ge_0_98": m4["clean_chain_success"] >= 0.98,
        "m4_attempt_leakage_zero": attempt_leakage == 0,
        "m4_support_invariant_violations_zero": invariant_violations == 0,
    }
    return {
        "schema_version": 1,
        "rollout_count": len(rows),
        "paired_unit_count": len(audits),
        "arm_counts": {
            arm: sum(row.get("arm") == arm for row in rows) for arm in MAIN_ARMS
        },
        "m4_metrics": m4,
        "backend_error_count": backend_errors,
        "child_process_failure_count": child_failures,
        "m4_attempt_leakage_count": attempt_leakage,
        "m4_support_graph_invariant_violation_count": invariant_violations,
        "gates": gates,
        "pass": all(gates.values()),
        "formal_confirmatory_authorized": all(gates.values()),
        "formal_run_policy": "diagnostic_continuation_if_gate_failed_per_frozen_user_instruction",
    }


def pilot_stage() -> None:
    _assert_contract()
    contract = _json(CONTRACT)
    _log("pilot: running frozen 20 cells x 10 seeds x 5 arms = 1,000 rollouts")
    rows, audits = run_matrix(contract["cells"], contract["pilot_seeds"], MAIN_ARMS)
    write_jsonl(EXPERIMENT / "pilot_results.jsonl", rows)
    write_jsonl(EXPERIMENT / "pilot_paired_unit_audit.jsonl", audits)
    summary = _pilot_summary(rows, audits)
    write_json(EXPERIMENT / "pilot_summary.json", summary)
    _log("pilot: pass=%s rows=%d paired_units=%d" % (
        summary["pass"], len(rows), len(audits)
    ))


def formal_stage() -> None:
    _assert_contract()
    contract = _json(CONTRACT)
    _log("formal: running frozen 20 cells x 50 seeds x 5 arms = 5,000 rollouts")
    rows, audits = run_matrix(contract["cells"], contract["formal_seeds"], MAIN_ARMS)
    if len(rows) != 5000 or len(audits) != 1000:
        raise RuntimeError("formal matrix count mismatch")
    write_jsonl(EXPERIMENT / "formal_results.jsonl", rows)
    write_jsonl(EXPERIMENT / "paired_unit_audit.jsonl", audits)
    _log("formal: rows=%d paired_units=%d audit_pass=%s" % (
        len(rows), len(audits), all(audit["pass"] for audit in audits)
    ))


def ablation_stage() -> None:
    _assert_contract()
    contract = _json(CONTRACT)
    _log("ablations: running 20 cells x 50 seeds x 4 arms = 4,000 rollouts")
    rows, audits = run_matrix(
        contract["cells"], contract["formal_seeds"], ABLATION_ARMS
    )
    if len(rows) != 4000 or len(audits) != 1000:
        raise RuntimeError("ablation matrix count mismatch")
    write_jsonl(EXPERIMENT / "mechanism_ablations.jsonl", rows)
    write_jsonl(EXPERIMENT / "mechanism_ablation_paired_unit_audit.jsonl", audits)
    _log("ablations: rows=%d paired_units=%d audit_pass=%s" % (
        len(rows), len(audits), all(audit["pass"] for audit in audits)
    ))


def _format_rate(value: object) -> str:
    if value is None:
        return "NA"
    return "%.6f" % float(value)


def _final_report(
    formal: Mapping[str, object],
    ablations: Mapping[str, object],
    decision: Mapping[str, object],
) -> str:
    arms = formal["arms"]
    effects = formal["effect_sizes"]
    statuses = formal["component_status"]
    isolated = ablations["isolated_effects"]
    source = _source_identity()
    trace = _json(EXPERIMENT / "trace_summary.json")
    executor = _json(EXPERIMENT / "executor_qualification_summary.json")
    shared = _json(EXPERIMENT / "shared_prefix_qualification_summary.json")
    pilot = _json(EXPERIMENT / "pilot_summary.json")
    lines = [
        "# R16-P19 Step5 / Phase-4 实验报告",
        "",
        "## 结论",
        "",
        "最终状态：`%s`。这是 CPU MuJoCo 微基准结论，不是 VLA、LIBERO、RMBench、N3 或论文就绪证据。" % decision["overall_status"],
        "",
        "组件状态：attempt=`%s`，support=`%s`，truth-attribution=`%s`，clean=`%s`。" % (
            statuses["attempt_scope"], statuses["support_proof"],
            statuses["truth_attribution"], statuses["clean"]
        ),
        "",
        "## 完整执行与平台有效性",
        "",
        "- 分支：`%s`" % source["branch"],
        "- 实验源 HEAD：`%s`" % source["head"],
        "- 实验源 tree：`%s`" % source["tree"],
        "- 冻结 B6 SHA256：`%s`" % source["protected_b6_sha256"],
        "- Trace gate：%s，10,000 schedules" % trace["status"],
        "- Executor gate：pass=%s，conditional=%s，full-chain=%s，backend errors=%d" % (
            executor["pass"], _format_rate(executor["conditional_effect_success"]),
            _format_rate(executor["full_chain_success"]), executor["backend_error_count"]
        ),
        "- Shared-prefix gate：pass=%s，1,000 paired units，child failures=%d" % (
            shared["pass"], shared["child_process_failure_count"]
        ),
        "- Pilot：pass=%s，1,000 rollouts；Formal：5,000 rollouts；Ablation：4,000 rollouts" % pilot["pass"],
        "- GPU/PAI：未使用；renderer：未初始化。",
        "",
        "## Formal 主结果",
        "",
        "| Arm | Clean | A1-A4 | S1-S4 | A5 truth | A5 false credit |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in MAIN_ARMS:
        value = arms[arm]
        lines.append(
            "| %s | %s | %s | %s | %s | %s |" % (
                arm,
                _format_rate(value["clean_chain_success"]),
                _format_rate(value["attempt_family_chain_success"]),
                _format_rate(value["support_family_chain_success"]),
                _format_rate(value["effect_truth_recognition"]),
                _format_rate(value["false_skill_credit_rate"]),
            )
        )
    attempt_best = formal["best_baselines"]["attempt_family"]
    support_best = formal["best_baselines"]["support_family"]
    attempt_ci = formal["cluster_bootstrap"]["attempt"][attempt_best]["ci_95"]
    support_ci = formal["cluster_bootstrap"]["support"][support_best]["ci_95"]
    lines.extend(
        [
            "",
            "- Attempt-family：M4 相对最强基线 `%s` 的绝对差为 %s，paired cluster bootstrap 95%% CI [%s, %s]。" % (
                attempt_best, _format_rate(effects["attempt_family_success_margin"]),
                _format_rate(attempt_ci[0]), _format_rate(attempt_ci[1])
            ),
            "- Support-family：M4 相对最强基线 `%s` 的绝对差为 %s，95%% CI [%s, %s]。" % (
                support_best, _format_rate(effects["support_family_success_margin"]),
                _format_rate(support_ci[0]), _format_rate(support_ci[1])
            ),
            "- Clean：成功率退化 %s，action-step overhead %s，event-processing latency overhead %s。" % (
                _format_rate(effects["clean_success_degradation"]),
                _format_rate(effects["action_step_overhead_fraction"]),
                _format_rate(effects["event_processing_latency_overhead_fraction"]),
            ),
            "",
            "## 机制反解（基于代码与消融，不生成新 idea）",
            "",
            "1. Attempt 改善来自 `AttemptScopedLedger.request()` 对旧 attempt 的 supersede、`_scope_match()` 的 attempt/command/epoch 三重绑定，以及新 request 清空当前证据池。它阻断了 A1/A2/A3 的跨尝试拼接；`negative_or_contradiction()` 在 VERIFIED 阶段撤销证据，又阻断 A4 的迟到 witness。",
            "2. Support 改善来自 `SupportProofGraph` 的“clause 内合取、clauses 间析取”。父 proof 失效时只重算引用它的 clause；只有全部 clause 失效才递归撤销 dependent。`UNTIL_*` 在约定终点 discharge，因此 S2 不误杀；替代 clause 使 S3 保留；递归仅沿新失效 proof 传播，使 S4 保持 branch locality。",
            "3. Truth/credit 改善来自 `external_realization()` 将 physical fact 置为 REALIZED，但在 attribution split 开启时保留 `attributed_attempt_id=null`。`decide()` 依据物理事实允许推进，而 capability credit 仍要求当前 attempt proof，所以 A5 能推进且不虚假记功。",
            "4. `NO_ATTEMPT_SCOPE` 移除了 M4 attempt 优势的 %s；`NO_SUPPORT_VALIDITY` 移除了 support 优势的 %s；`NO_ATTRIBUTION_SPLIT` 使 A5 false credit 增加 %s；`NO_PRE_REALIZATION_REVOCATION` 使 A4 false realization 增加 %s。" % (
                _format_rate(isolated["NO_ATTEMPT_SCOPE"]["attempt_advantage_removed_fraction"]),
                _format_rate(isolated["NO_SUPPORT_VALIDITY"]["support_advantage_removed_fraction"]),
                _format_rate(isolated["NO_ATTRIBUTION_SPLIT"]["false_skill_credit_absolute_increase"]),
                _format_rate(isolated["NO_PRE_REALIZATION_REVOCATION"]["A4_false_realization_absolute_increase"]),
            ),
            "5. 时延变化来自 M4 每个事件同时维护 append-only ledger、proof/clause 索引、discharge 和失效路径；它不改变物理动作数量。若冻结的 10%% overhead gate 未通过，该成本会保留为 clean failure，不能用功能收益覆盖。",
            "",
            "## 证据边界",
            "",
            "本实验没有修改或重新解释 Phase-3：原状态仍是 `BLOCKED_BY_IMPLEMENTATION`，27/150 shared-prefix 不一致，且 B6 与 TYPED_MATCHED_RECOVERY 在 backend-valid units 上同为 0.913043、差值 0、95% CI [0,0]。Phase-4 使用新的 CPU 微基准和真 shared-prefix fork，只回答 ASCEL 三个窄机制在该合同内是否成立。",
            "",
            "所有失败、逐 rollout 结果、paired audits、10,000-replicate bootstrap、McNemar/Holm、消融和 SHA256 均保存在本目录。",
            "",
        ]
    )
    return "\n".join(lines)


def _mechanism_report(formal: Mapping[str, object], ablations: Mapping[str, object]) -> str:
    isolated = ablations["isolated_effects"]
    return "\n".join(
        [
            "# ASCEL 机制反解记录",
            "",
            "本记录只解释已实现机制为何提升或降低，不提出新 idea。",
            "",
            "## 代码因果链",
            "",
            "- Attempt scope：REQUEST -> supersede active attempt -> allocate deterministic generation-scoped ID -> reset current evidence bucket -> require current attempt + command + post-revocation epoch at receipt/witness acceptance。对应 A1-A4。",
            "- Support validity：realization proof -> disjunctive support clauses -> reference-level discharge -> clause-local recomputation -> invalidate dependent only when no valid clause remains。对应 S1-S4。",
            "- Fact/attribution split：external proof may verify a fact with null attributed attempt -> task decision reads fact -> skill credit reads attribution。对应 A5。",
            "- Cost path：M4 maintains ledger history, proof graph, reverse indices, discharge events and invalidation paths; the physical executor and action budget stay arm-blind and shared。",
            "",
            "## 冻结消融的反事实读数",
            "",
            "- NO_ATTEMPT_SCOPE：attempt advantage removed fraction = %s，criterion=%s。" % (
                _format_rate(isolated["NO_ATTEMPT_SCOPE"]["attempt_advantage_removed_fraction"]),
                isolated["NO_ATTEMPT_SCOPE"]["criterion_ge_0_50"],
            ),
            "- NO_SUPPORT_VALIDITY：support advantage removed fraction = %s，over-invalidation=%s，criterion=%s。" % (
                _format_rate(isolated["NO_SUPPORT_VALIDITY"]["support_advantage_removed_fraction"]),
                _format_rate(isolated["NO_SUPPORT_VALIDITY"]["over_invalidation_rate"]),
                isolated["NO_SUPPORT_VALIDITY"]["criterion"],
            ),
            "- NO_ATTRIBUTION_SPLIT：A5 false-credit absolute increase = %s，criterion=%s。" % (
                _format_rate(isolated["NO_ATTRIBUTION_SPLIT"]["false_skill_credit_absolute_increase"]),
                isolated["NO_ATTRIBUTION_SPLIT"]["criterion_ge_0_10"],
            ),
            "- NO_PRE_REALIZATION_REVOCATION：A4 false-realization absolute increase = %s，criterion=%s。" % (
                _format_rate(isolated["NO_PRE_REALIZATION_REVOCATION"]["A4_false_realization_absolute_increase"]),
                isolated["NO_PRE_REALIZATION_REVOCATION"]["criterion_ge_0_10"],
            ),
            "",
            "这些读数只在冻结的 20-cell CPU microbenchmark 内作机制归因；不外推到 VLA 或开放世界。",
            "",
        ]
    )


def analysis_stage() -> None:
    _assert_contract()
    _log("analysis: loading formal and ablation matrices")
    formal_rows = _jsonl(EXPERIMENT / "formal_results.jsonl")
    ablation_rows = _jsonl(EXPERIMENT / "mechanism_ablations.jsonl")
    formal = analyze_formal(formal_rows)
    ablations = analyze_ablations(formal_rows, ablation_rows, formal)
    trace = _json(EXPERIMENT / "trace_summary.json")
    executor = _json(EXPERIMENT / "executor_qualification_summary.json")
    shared = _json(EXPERIMENT / "shared_prefix_qualification_summary.json")
    decision = final_decision(
        formal, bool(trace["pass"]), bool(executor["pass"]), bool(shared["pass"])
    )
    component = {
        "schema_version": 1,
        "source": _source_identity(),
        "formal_rollout_count": len(formal_rows),
        "ablation_rollout_count": len(ablation_rows),
        "arms": formal["arms"],
        "best_baselines": formal["best_baselines"],
        "effect_sizes": formal["effect_sizes"],
        "component_gates": formal["component_gates"],
        "component_status": formal["component_status"],
        "ablation_analysis": ablations,
        "decision": decision,
    }
    write_json(EXPERIMENT / "component_metrics.json", component)
    write_json(EXPERIMENT / "cluster_bootstrap.json", formal["cluster_bootstrap"])
    write_json(EXPERIMENT / "paired_tests.json", formal["paired_tests"])
    write_json(EXPERIMENT / "mechanism_ablation_summary.json", ablations)
    write_jsonl(
        EXPERIMENT / "failure_decomposition.jsonl",
        failure_decomposition(formal_rows + ablation_rows),
    )
    write_json(EXPERIMENT / "final_decision.json", decision)
    atomic_text(EXPERIMENT / "FINAL_DECISION.md", _final_report(formal, ablations, decision))
    atomic_text(
        EXPERIMENT / "MECHANISM_REVERSE_ENGINEERING.md",
        _mechanism_report(formal, ablations),
    )
    _record_provenance()
    names = [
        str(path.relative_to(EXPERIMENT))
        for path in EXPERIMENT.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    write_sha256sums(EXPERIMENT, names)
    _log("analysis: overall=%s statuses=%s" % (
        decision["overall_status"], formal["component_status"]
    ))


STAGES = {
    "trace": trace_stage,
    "executor": executor_stage,
    "shared-prefix": shared_prefix_stage,
    "pilot": pilot_stage,
    "formal": formal_stage,
    "ablations": ablation_stage,
    "analysis": analysis_stage,
}


def main(argv: Iterable[str] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=tuple(STAGES) + ("all",),
        help="Run one resumable stage or the complete frozen pipeline.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    _assert_contract()
    _record_provenance()
    if args.stage == "all":
        for stage in ("trace", "executor", "shared-prefix", "pilot", "formal", "ablations", "analysis"):
            STAGES[stage]()
    else:
        STAGES[args.stage]()


if __name__ == "__main__":
    main()
