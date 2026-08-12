#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

NEW_ROOT=/mnt/cpfs/zbl-cpfs-new
LEON_UID=2254
LEON_GID=2254
CODE_ROOT="$NEW_ROOT/USERS/leon/code/r16p19-libero-phase1-20260813"
LIBERO_ROOT="$NEW_ROOT/USERS/leon/code/LIBERO-r16p19-official-8f1084e"
PYTHON="$NEW_ROOT/USERS/leon/envs/libero-original/bin/python"
CHECKPOINT_ROOT="$NEW_ROOT/CKPT/leon/torch/r16p19_libero_phase1/tiny_state_bc_v1"

: "${PAI_CANARY_RUN_ID:?registry run ID is required}"
: "${PAI_CANARY_RUN_DIR:?registry artifact directory is required}"
: "${PAI_CANARY_NONCE:?registry nonce is required}"
: "${PAI_CANARY_EXPECTED_GPUS:?registry GPU count is required}"
: "${R16P19_EXPECTED_COMMIT:?expected source commit is required}"
: "${WANDB_API_KEY:?controller-injected WANDB_API_KEY is required}"
: "${WANDB_ENTITY:?controller-injected WANDB_ENTITY is required}"

fail() {
  echo "FATAL $*" >&2
  exit 2
}

[[ "$(id -u):$(id -g)" == "$LEON_UID:$LEON_GID" ]] ||
  fail "workload identity must be $LEON_UID:$LEON_GID"
[[ "$PAI_CANARY_EXPECTED_GPUS" == 2 ]] || fail "registry contract must request two GPUs"
[[ "$(nvidia-smi -L | wc -l | tr -d ' ')" == 2 ]] || fail "runtime GPU inventory is not two"
mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${gpu_names[0]}" == *A800* && "${gpu_names[1]}" == *A800* ]] ||
  fail "formal run requires two A800 devices: ${gpu_names[*]}"
[[ "${gpu_names[*]}" != *4090* ]] || fail "RTX4090 is forbidden for this formal run"
active_gpu_uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed -n '1p')"
[[ "$WANDB_ENTITY" == chen_jian-cj-workspace ]] ||
  fail "WANDB_ENTITY must remain chen_jian-cj-workspace"
[[ -x "$PYTHON" ]] || fail "missing pinned LIBERO Python"
[[ -d "$CODE_ROOT" && ! -L "$CODE_ROOT" ]] || fail "missing immutable experiment source"
[[ -d "$LIBERO_ROOT" && ! -L "$LIBERO_ROOT" ]] || fail "missing frozen LIBERO source"
[[ "$(git -C "$CODE_ROOT" rev-parse HEAD)" == "$R16P19_EXPECTED_COMMIT" ]] ||
  fail "experiment source commit drifted"
[[ -z "$(git -C "$CODE_ROOT" status --porcelain)" ]] || fail "experiment source is dirty"
[[ "$(git -C "$LIBERO_ROOT" rev-parse HEAD)" == 8f1084e3132a39270c3a13ebe37270a43ece2a01 ]] ||
  fail "LIBERO source commit drifted"
[[ -z "$(git -C "$LIBERO_ROOT" status --porcelain)" ]] || fail "LIBERO source is dirty"

install -d -m 700 "$PAI_CANARY_RUN_DIR" "$CHECKPOINT_ROOT"
[[ "$(stat -c '%u:%g' "$PAI_CANARY_RUN_DIR")" == "$LEON_UID:$LEON_GID" ]] ||
  fail "artifact directory owner mismatch"
[[ "$(stat -c '%u:%g' "$CHECKPOINT_ROOT")" == "$LEON_UID:$LEON_GID" ]] ||
  fail "checkpoint directory owner mismatch"

artifact_probe="$PAI_CANARY_RUN_DIR/.ownership-probe-$PAI_CANARY_NONCE"
checkpoint_probe="$CHECKPOINT_ROOT/.ownership-probe-$PAI_CANARY_NONCE"
: >"$artifact_probe"
: >"$checkpoint_probe"
[[ "$(stat -c '%u:%g' "$artifact_probe")" == "$LEON_UID:$LEON_GID" ]] ||
  fail "artifact write owner mismatch"
[[ "$(stat -c '%u:%g' "$checkpoint_probe")" == "$LEON_UID:$LEON_GID" ]] ||
  fail "checkpoint write owner mismatch"
rm -f -- "$artifact_probe" "$checkpoint_probe"

contract_marker="$PAI_CANARY_RUN_DIR/launch-contract.env"
"$PYTHON" - "$contract_marker" <<PY
import os
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
temporary = target.with_name(".%s.tmp.%d" % (target.name, os.getpid()))
body = """registry_run_id=$PAI_CANARY_RUN_ID
registry_nonce=$PAI_CANARY_NONCE
runtime_uid_gid=$(id -u):$(id -g)
source_commit=$R16P19_EXPECTED_COMMIT
libero_commit=8f1084e3132a39270c3a13ebe37270a43ece2a01
checkpoint_root=$CHECKPOINT_ROOT
gpu_name=${gpu_names[0]}
gpu_count=2
active_gpu_ordinal=0
reserved_idle_gpu_ordinal=1
active_gpu_uuid=$active_gpu_uuid
platform_restart_limit=0
launcher_attempts=1
checkpoint_contract=model_optimizer_scheduler_rng_global_step
checkpoint_interval_steps=1000
checkpoint_retention=all_positive_10000_milestones_plus_latest_3_complete_nonmilestones
wandb_entity=$WANDB_ENTITY
wandb_project=r16p19-libero-phase1
evidence_boundary=actor_free_trace_plus_tiny_state_bc_not_vla
"""
descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
try:
    os.write(descriptor, body.encode())
    os.fsync(descriptor)
finally:
    os.close(descriptor)
os.replace(temporary, target)
PY
[[ "$(stat -c '%u:%g' "$contract_marker")" == "$LEON_UID:$LEON_GID" ]] ||
  fail "first artifact owner mismatch"
echo "FIRST_ARTIFACT_OWNER_OK path=$contract_marker uid_gid=$(stat -c '%u:%g' "$contract_marker")"

export PYTHONPATH="$CODE_ROOT:$LIBERO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$CODE_ROOT/experiments/r16p19_libero_phase1/libero_config"
export CUDA_VISIBLE_DEVICES="$active_gpu_uuid"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export WANDB_PROJECT=r16p19-libero-phase1
export R16P19_WANDB_REQUIRED=1
export MPLCONFIGDIR="$PAI_CANARY_RUN_DIR/.matplotlib"

cd "$CODE_ROOT"
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

"$PYTHON" -u pipeline.py static-check
"$PYTHON" -u pipeline.py retention-test

output_dir="$PAI_CANARY_RUN_DIR/experiments/r16p19_libero_phase1"
"$PYTHON" -u pipeline.py formal-run \
  --output-dir "$output_dir" \
  --checkpoint-dir "$CHECKPOINT_ROOT" \
  --expected-commit "$R16P19_EXPECTED_COMMIT"

for required in \
  final_status.json metrics.json paired_bootstrap.json trace_events.jsonl \
  memory_outputs.jsonl state_bc_results.jsonl failure_cases.md README.md \
  readiness_report.md SHA256SUMS; do
  [[ -f "$output_dir/$required" ]] || fail "required artifact missing: $required"
done

stop_telemetry
trap - EXIT

completion="$PAI_CANARY_RUN_DIR/RUN_COMPLETE.json"
"$PYTHON" - "$completion" "$output_dir" <<'PY'
import json
import os
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
status = json.load(open(output / "final_status.json", encoding="utf-8"))
payload = {
    "run_complete": True,
    "runtime_uid_gid": "%d:%d" % (os.getuid(), os.getgid()),
    "result": status,
    "output_dir": str(output),
}
temporary = target.with_name(".%s.tmp.%d" % (target.name, os.getpid()))
with open(temporary, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, target)
PY
[[ "$(stat -c '%u:%g' "$completion")" == "$LEON_UID:$LEON_GID" ]] ||
  fail "completion artifact owner mismatch"
echo "R16P19_FORMAL_RUN_COMPLETE path=$completion"
