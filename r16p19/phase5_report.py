"""Generate the human-readable Chinese report and mechanism trace."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value):
    return f"{100 * value:.2f}%"


def run(root: Path) -> None:
    decision = load(root / "final_decision.json")
    bounded = load(root / "bounded_property_results.json")
    oracle = load(root / "oracle_analysis.json")
    learned = load(root / "learned_verifier_analysis.json")
    support = load(root / "support_analysis.json")
    verifier = load(root / "verifier_qualification.json")
    attribution = load(root / "mechanism_attribution.json")
    qualification = load(root / "policy_qualification.json")
    pairing = load(root / "shared_prefix_summary.json")
    lines = [
        "# R16-P19 Phase-5 Bounded ASCEL Embodied Bridge 实验报告",
        "",
        f"最终状态：`{decision['status']}`。所有预注册矩阵均已执行；未通过的 gate 不被改写，后续矩阵按用户明确要求仅作为 diagnostic continuation。",
        "",
        "## 核心结果",
        "",
        f"- bounded ledger：100,000 events，reference mismatch={bounded['exact_reference_mismatches']}，audit break={bounded['audit_chain_breaks']}，event P99={bounded['event_latency_ms']['p99']:.4f} ms，hot memory={bounded['hot_memory_mb']:.4f} MB，systems pass={bounded['pass']}。",
        f"- 冻结策略 qualification：pass={qualification['pass']}；逐任务结果为 `{json.dumps(qualification['tasks'], ensure_ascii=False, sort_keys=True)}`。",
        f"- shared prefix：{pairing['units']} units，全部字段精确一致={pairing['all_fields_exact']}。",
        f"- oracle Core：最强 baseline={oracle['strongest_baseline']}；faulted success 风险差={oracle['paired_bootstrap']['risk_difference']:.4f}，95% CI={oracle['paired_bootstrap']['ci95']}，pass={oracle['pass']}。",
        f"- learned verifier：selected={verifier['selected']}，qualified={verifier['selected_qualified']}；formal 风险差={learned['paired_bootstrap']['risk_difference']:.4f}，95% CI={learned['paired_bootstrap']['ci95']}，pass={learned['pass']}。",
        f"- support proof：Full-Core 风险差={support['paired_bootstrap']['risk_difference']:.4f}，95% CI={support['paired_bootstrap']['ci95']}，cascade precision={support['cascade_precision']:.4f}，recall={support['cascade_recall']:.4f}，pass={support['pass']}。",
        "",
        "## 机制反解（不生成新 idea）",
        "",
        "采用 code-first 的 first-divergence 方法：先固定同一物理/观察/动作前缀，再定位 arm 首次不同的 ledger 状态与 decision。提升只能归因到首次分叉前唯一不同的机制；降低同样按该路径追踪，不按最终分数倒推故事。",
        "",
    ]
    for name, row in attribution["ablations"].items():
        lines.append(f"- `{name}`：target-error 增量={row['target_error_increase']:.4f}，对应优势移除比例={row['advantage_removed_fraction']:.4f}，支持该机制归因={row['supports_mechanism']}。")
    lines += [
        "",
        f"learned 分支的 0 增益来自 verifier 门控而不是 ASCEL ledger 退化：选中的 small MLP threshold={verifier['models']['small_mlp']['threshold']:.4f}，qualification min TPR={verifier['models']['small_mlp']['min_tpr']:.4f}。`phase5_formal_runner.py` 对一条 receipt 使用 `np.all(scores >= threshold)`；任一 effect 漏检就整体不承认真实完成，因此 Core 与 baseline 同时不能推进，formal 风险差精确为 0。该结论与 oracle Core 的正向结果并存，最终按预注册优先级记为 `BLOCKED_BY_VERIFIER`。",
        "",
        "`NO_TRUTH_CREDIT_SPLIT` 的零消融效应有指标边界：本轮 `target_error` 只统计 false advance，A5 的 task success 也没有因 `active_attempt_credit` 单独扣分；因此它只能说明选定 outcome 对 credit 字段不敏感，不能证明 truth 与 credit 在代码语义上等价。",
        "",
        "代码路径上，attempt scope 在 `phase5_ledger_live.py` 的 active-attempt/command 检查处拒绝 stale 与 cross-attempt receipt；pre-realization revocation 通过 revocation epoch 使旧 witness 不再复活事实；truth-credit split 允许 external realization 更新物理事实但不把成功计给 active skill；support graph 只递归失效新近失去全部有效 clause 的 proof，并保留 alternative branch 或已 discharge 的结果。",
        "",
        "## 证据边界",
        "",
        "LIBERO 的策略轨迹是真实官方 simulator + frozen π0.5 推理；arm 级 4,200/1,680 rows 使用一次真实轨迹形成的 shared-prefix 事件/决策反事实，避免为每个 arm 独立重跑前缀。A1 具有真实 no-op 物理轨迹；A2/A3/V1 是绑定真实观察时点的 receipt fault。A4 与 A5 的因果归属主要由预注册事件 broker/归因干预实现，不等价于机器人外力硬件干预，因此即便数值为正，也不能外推为真实机器人证据。support 的 960 cells 来自重力、接触和约束启用的 MuJoCo 物理任务，不是独立 slider。",
        "",
        "本阶段不改变 Phase-1 至 Phase-4 的任何结论，也不把 oracle-only、diagnostic continuation 或 learned-verifier 未过 gate 的结果称为 VLA 改进。",
    ]
    (root / "EXPERIMENT_REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "FINAL_DECISION.md").write_text(f"# Phase-5 Final Decision\n\n`{decision['status']}`\n\nSee `EXPERIMENT_REPORT_ZH.md` and `final_decision.json` for gate-level evidence.\n", encoding="utf-8")
    mechanism = ["# Phase-5 Mechanism Reverse Engineering", "", "No new idea is proposed. This file records code-first causal attribution from frozen shared-prefix divergences.", ""]
    mechanism.extend(f"- {name}: {json.dumps(value, sort_keys=True)}" for name, value in attribution["ablations"].items())
    (root / "MECHANISM_REVERSE_ENGINEERING.md").write_text("\n".join(mechanism) + "\n", encoding="utf-8")
    formal = [json.loads(line) for line in (root / "oracle_formal_results.jsonl").read_text().splitlines() if line]
    failures = [row for row in formal if not row["task_success"]]
    (root / "failure_cases.md").write_text("# Failure cases\n\n" + f"Total formal arm failures: {len(failures)}. Full rows and video references are in `oracle_formal_results.jsonl` and `video_manifest.jsonl`.\n", encoding="utf-8")
    excluded = {"SHA256SUMS"}
    checksums = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name not in excluded):
        checksums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}")
    (root / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.result_root)


if __name__ == "__main__":
    main()
