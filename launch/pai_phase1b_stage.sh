#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

NEW_ROOT=/mnt/cpfs/zbl-cpfs-new
LEON_UID=2254
LEON_GID=2254
LIBERO_ROOT="$NEW_ROOT/USERS/leon/code/LIBERO-r16p19-official-8f1084e"
PYTHON="$NEW_ROOT/USERS/leon/envs/libero-original/bin/python"
OUTPUT_ROOT="${R16P19_PHASE1B_OUTPUT_ROOT:-$NEW_ROOT/USERS/leon/logs/r16p19_libero_phase1b/v1/experiment}"
CHECKPOINT_ROOT="${R16P19_PHASE1B_CHECKPOINT_ROOT:-$NEW_ROOT/CKPT/leon/torch/r16p19_libero_phase1b/v1}"

: "${PAI_CANARY_RUN_ID:?registry run ID is required}"
: "${PAI_CANARY_RUN_DIR:?registry run directory is required}"
: "${PAI_CANARY_NONCE:?registry nonce is required}"
: "${PAI_CANARY_EXPECTED_GPUS:?registry GPU contract is required}"
: "${R16P19_PHASE1B_STAGE:?Phase-1B stage is required}"
: "${R16P19_PHASE1B_CODE_ROOT:?immutable source path is required}"
: "${R16P19_EXPECTED_COMMIT:?expected source commit is required}"
: "${WANDB_API_KEY:?controller-injected W&B key is required}"
: "${WANDB_ENTITY:?controller-injected W&B entity is required}"

fail() {
  echo "FATAL $*" >&2
  exit 2
}

case "$R16P19_PHASE1B_STAGE" in
  train-primary|qualify-primary|train-fallback|qualify-fallback|formal-gate|closed-loop) ;;
  *) fail "unknown Phase-1B stage $R16P19_PHASE1B_STAGE" ;;
esac

[[ "$(id -u):$(id -g)" == "$LEON_UID:$LEON_GID" ]] ||
  fail "workload identity must be $LEON_UID:$LEON_GID"
[[ "$PAI_CANARY_EXPECTED_GPUS" == 2 ]] || fail "job must request two GPUs"
mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#gpu_names[@]}" == 2 ]] || fail "physical GPU inventory must equal two"
[[ "${gpu_names[0]}" == *A800* && "${gpu_names[1]}" == *A800* ]] ||
  fail "formal stages require two A800 GPUs"
active_gpu_uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed -n '1p')"
[[ "$WANDB_ENTITY" == chen_jian-cj-workspace ]] || fail "unexpected W&B entity"
[[ -x "$PYTHON" ]] || fail "pinned LIBERO Python is missing"
[[ -d "$R16P19_PHASE1B_CODE_ROOT" && ! -L "$R16P19_PHASE1B_CODE_ROOT" ]] ||
  fail "immutable Phase-1B source is missing"
[[ "$(git -C "$R16P19_PHASE1B_CODE_ROOT" rev-parse HEAD)" == "$R16P19_EXPECTED_COMMIT" ]] ||
  fail "Phase-1B source commit drift"
[[ -z "$(git -C "$R16P19_PHASE1B_CODE_ROOT" status --porcelain)" ]] ||
  fail "Phase-1B source worktree is dirty"
[[ "$(git -C "$LIBERO_ROOT" rev-parse HEAD)" == 8f1084e3132a39270c3a13ebe37270a43ece2a01 ]] ||
  fail "official LIBERO commit drift"

install -d -m 0700 "$PAI_CANARY_RUN_DIR" "$OUTPUT_ROOT" "$CHECKPOINT_ROOT"
for path in "$PAI_CANARY_RUN_DIR" "$OUTPUT_ROOT" "$CHECKPOINT_ROOT"; do
  [[ "$(stat -c '%u:%g' "$path")" == "$LEON_UID:$LEON_GID" ]] ||
    fail "write root owner mismatch: $path"
done

export PYTHONPATH="$R16P19_PHASE1B_CODE_ROOT:$LIBERO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$R16P19_PHASE1B_CODE_ROOT/experiments/r16p19_libero_phase1/libero_config"
export CUDA_VISIBLE_DEVICES="$active_gpu_uuid"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless
export MUJOCO_EGL_DEVICE_ID=0
export EGL_DEVICE_ID=0
export PAI_AUTOMATIC_FAULT_TOLERANCE=0
export WANDB_PROJECT=r16p19-libero-phase1b
export R16P19_WANDB_REQUIRED=1
export MPLCONFIGDIR="$PAI_CANARY_RUN_DIR/.matplotlib"

contract="$PAI_CANARY_RUN_DIR/phase1b-launch-contract.json"
"$PYTHON" - "$contract" <<'PY'
import json
import os
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
payload = {
    "application_auto_resume": True,
    "checkpoint_contract": ["model", "optimizer", "scheduler", "rng", "global_step"],
    "checkpoint_interval_steps": 2500,
    "expected_source_commit": os.environ["R16P19_EXPECTED_COMMIT"],
    "hardware": "2xA800_gpu0_active_gpu1_reserved_idle",
    "pai_automatic_fault_tolerance": False,
    "platform_restart_limit": 0,
    "run_id": os.environ["PAI_CANARY_RUN_ID"],
    "stage": os.environ["R16P19_PHASE1B_STAGE"],
    "wandb_entity": os.environ["WANDB_ENTITY"],
}
temporary = target.with_name(".%s.tmp.%d" % (target.name, os.getpid()))
with temporary.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(str(temporary), str(target))
PY

exec > >(tee -a "$PAI_CANARY_RUN_DIR/run.log") 2>&1

reserved_telemetry="$PAI_CANARY_RUN_DIR/reserved_gpu1_dmon.log"
nvidia-smi dmon -i 1 -s pucm -d 10 -o DT >"$reserved_telemetry" 2>&1 &
telemetry_pid=$!
stop_telemetry() {
  if kill -0 "$telemetry_pid" 2>/dev/null; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
}
trap stop_telemetry EXIT

cd "$R16P19_PHASE1B_CODE_ROOT"
"$PYTHON" -u phase1b_pipeline.py static-check --expected-commit "$R16P19_EXPECTED_COMMIT"

common=(--output-dir "$OUTPUT_ROOT" --expected-commit "$R16P19_EXPECTED_COMMIT" --require-pai)
case "$R16P19_PHASE1B_STAGE" in
  train-primary)
    "$PYTHON" -u phase1b_pipeline.py train-primary "${common[@]}" --checkpoint-dir "$CHECKPOINT_ROOT"
    test -f "$OUTPUT_ROOT/primary_selected_actor.json"
    ;;
  qualify-primary)
    "$PYTHON" -u phase1b_pipeline.py qualify "${common[@]}" --family primary
    test -f "$OUTPUT_ROOT/qualification_primary_summary.json"
    ;;
  train-fallback)
    "$PYTHON" -u phase1b_pipeline.py train-fallback "${common[@]}" --checkpoint-dir "$CHECKPOINT_ROOT"
    test -f "$OUTPUT_ROOT/fallback_selected_actor.json"
    ;;
  qualify-fallback)
    "$PYTHON" -u phase1b_pipeline.py qualify "${common[@]}" --family fallback
    test -f "$OUTPUT_ROOT/qualification_fallback_summary.json"
    ;;
  formal-gate)
    "$PYTHON" -u phase1b_pipeline.py formal-gate "${common[@]}"
    test -f "$OUTPUT_ROOT/formal_actor_gate_summary.json"
    ;;
  closed-loop)
    "$PYTHON" -u phase1b_pipeline.py closed-loop "${common[@]}"
    test -f "$OUTPUT_ROOT/final_status.json"
    ;;
esac

stop_telemetry
trap - EXIT

completion="$PAI_CANARY_RUN_DIR/RUN_COMPLETE.json"
"$PYTHON" - "$completion" "$OUTPUT_ROOT" <<'PY'
import json
import os
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
payload = {
    "run_complete": True,
    "stage": os.environ["R16P19_PHASE1B_STAGE"],
    "runtime_uid_gid": "%d:%d" % (os.getuid(), os.getgid()),
    "output_root": sys.argv[2],
}
temporary = target.with_name(".%s.tmp.%d" % (target.name, os.getpid()))
with temporary.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(str(temporary), str(target))
PY
echo "PHASE1B_STAGE_COMPLETE stage=$R16P19_PHASE1B_STAGE completion=$completion"
