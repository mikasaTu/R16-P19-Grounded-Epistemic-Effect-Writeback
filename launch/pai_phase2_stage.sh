#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

NEW_ROOT=/mnt/cpfs/zbl-cpfs-new
SOURCE_ROOT="$NEW_ROOT/USERS/leon/code/r16p19-phase1b-publication-20260813"
LIBERO_ROOT="$NEW_ROOT/USERS/leon/code/LIBERO-r16p19-official-8f1084e"
PYTHON="$NEW_ROOT/USERS/leon/envs/libero-original/bin/python"
EXPECTED_LIBERO_COMMIT=8f1084e3132a39270c3a13ebe37270a43ece2a01
EXPECTED_LIBERO_TREE=99f4ada3f1d62e026fc9ff2390eb4ff8a1760e60
EXPECTED_PYTHON_SHA=903441582909636b4316667ee22beb1e6e2726991584ea9265c765ac8388abd1
LEON_UID=2254
LEON_GID=2254

RUN_ID=${PAI_CANARY_RUN_ID:?PAI_CANARY_RUN_ID is required}
EVIDENCE_DIR=${PAI_CANARY_RUN_DIR:?PAI_CANARY_RUN_DIR is required}
EXPECTED_GPUS=${PAI_CANARY_EXPECTED_GPUS:?PAI_CANARY_EXPECTED_GPUS is required}
STAGE=${R16P19_PHASE2_STAGE:?R16P19_PHASE2_STAGE is required}
EXPECTED_SOURCE_COMMIT=${R16P19_EXPECTED_COMMIT:?R16P19_EXPECTED_COMMIT is required}
WORK_ROOT=${R16P19_PHASE2_OUTPUT_ROOT:?R16P19_PHASE2_OUTPUT_ROOT is required}

monitor_pid=
on_error() {
  local status=$?
  if test -n "$monitor_pid"; then
    kill "$monitor_pid" 2>/dev/null || true
  fi
  printf 'R16P19_PHASE2_FAILED line=%s status=%s command=%q\n' \
    "${BASH_LINENO[0]:-?}" "$status" "$BASH_COMMAND" >&2
  exit "$status"
}
trap on_error ERR

case "$STAGE" in
  template-calibration|qualification|formal-competence|closed-loop) ;;
  *) printf 'unknown Phase-2 stage: %s\n' "$STAGE" >&2; exit 64 ;;
esac
case "$WORK_ROOT" in
  "$NEW_ROOT/USERS/leon/logs/r16p19_libero_phase2/"*) ;;
  *) printf 'invalid Phase-2 output root: %s\n' "$WORK_ROOT" >&2; exit 65 ;;
esac

for command in git sha256sum nvidia-smi stat realpath find wc sleep kill; do
  command -v "$command" >/dev/null
done
test "$(id -u):$(id -g)" = "$LEON_UID:$LEON_GID"
test "$EXPECTED_GPUS" = 1
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" = 1
test "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" = "$EXPECTED_SOURCE_COMMIT"
test -z "$(git -C "$SOURCE_ROOT" status --porcelain)"
test "$(git -C "$LIBERO_ROOT" rev-parse HEAD)" = "$EXPECTED_LIBERO_COMMIT"
test "$(git -C "$LIBERO_ROOT" rev-parse 'HEAD^{tree}')" = "$EXPECTED_LIBERO_TREE"
test -z "$(git -C "$LIBERO_ROOT" status --porcelain)"
test "$(sha256sum "$(realpath -e "$PYTHON")" | awk '{print $1}')" = "$EXPECTED_PYTHON_SHA"

mkdir -p "$WORK_ROOT" "$WORK_ROOT/tmp" "$WORK_ROOT/cache"
test "$(realpath -e "$WORK_ROOT")" = "$WORK_ROOT"
test "$(stat -c '%u:%g' "$WORK_ROOT")" = "$LEON_UID:$LEON_GID"
test "$(stat -c '%u:%g' "$EVIDENCE_DIR")" = "$LEON_UID:$LEON_GID"

export PYTHONPATH="$SOURCE_ROOT:$LIBERO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless
export MUJOCO_EGL_DEVICE_ID=0
export EGL_DEVICE_ID=0
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
export PAI_AUTOMATIC_FAULT_TOLERANCE=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TMPDIR="$WORK_ROOT/tmp"
export XDG_CACHE_HOME="$WORK_ROOT/cache/xdg"
export PYTHONPYCACHEPREFIX="$WORK_ROOT/cache/pycache"
mkdir -p "$XDG_CACHE_HOME" "$PYTHONPYCACHEPREFIX"

"$PYTHON" - "$EVIDENCE_DIR/phase2-launch-contract.json" <<'PY'
import json
import os
import pathlib
import subprocess
import sys

target = pathlib.Path(sys.argv[1])
gpu = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"],
    text=True,
).strip()
value = {
    "application_resume_unit": "completed_rollout_or_calibration_cell",
    "expected_source_commit": os.environ["R16P19_EXPECTED_COMMIT"],
    "gpu_inventory": gpu,
    "hardware_contract": "exactly_one_rendering_gpu",
    "pai_automatic_fault_tolerance": False,
    "platform_restart_limit": 0,
    "run_id": os.environ["PAI_CANARY_RUN_ID"],
    "stage": os.environ["R16P19_PHASE2_STAGE"],
    "uid_gid": "%d:%d" % (os.getuid(), os.getgid()),
    "work_root": os.environ["R16P19_PHASE2_OUTPUT_ROOT"],
}
temporary = target.with_name(".%s.tmp.%d" % (target.name, os.getpid()))
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
os.replace(str(temporary), str(target))
PY
test "$(stat -c '%u:%g' "$EVIDENCE_DIR/phase2-launch-contract.json")" = "$LEON_UID:$LEON_GID"

exec > >(tee -a "$EVIDENCE_DIR/run.log") 2>&1
cd "$SOURCE_ROOT"
"$PYTHON" -m py_compile \
  r16p19/phase2_executor.py \
  r16p19/phase2_evaluation.py \
  r16p19/phase2_closed_loop.py \
  scripts/calibrate_phase2_templates.py \
  scripts/run_phase2_clean_gate.py \
  scripts/run_phase2_closed_loop.py
"$PYTHON" -m pytest tests/test_phase2.py -q

first_work_pattern=
case "$STAGE" in
  template-calibration)
    first_work_pattern="$WORK_ROOT/template_calibration_results.jsonl"
    ;;
  qualification)
    first_work_pattern="$WORK_ROOT/qualification_rollouts.jsonl"
    ;;
  formal-competence)
    first_work_pattern="$WORK_ROOT/formal_competence_rollouts.jsonl"
    ;;
  closed-loop)
    first_work_pattern="$WORK_ROOT/closed_loop_results.jsonl"
    ;;
esac
(
  while true; do
    if test -s "$first_work_pattern"; then
      "$PYTHON" - "$EVIDENCE_DIR/FIRST_COMPLETED_CELL.json" "$first_work_pattern" <<'PY'
import json
import os
import pathlib
import sys
import time

target = pathlib.Path(sys.argv[1])
value = {
    "first_persisted_path": sys.argv[2],
    "persisted_at_unix": time.time(),
    "uid": os.getuid(),
    "gid": os.getgid(),
}
temporary = target.with_name(".%s.tmp.%d" % (target.name, os.getpid()))
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
os.replace(str(temporary), str(target))
PY
      break
    fi
    sleep 5
  done
) &
monitor_pid=$!

case "$STAGE" in
  template-calibration)
    "$PYTHON" -u scripts/calibrate_phase2_templates.py \
      --extraction-manifest experiments/r16p19_libero_phase2/skill_template_extraction_manifest.json \
      --output-manifest "$WORK_ROOT/skill_template_manifest.json" \
      --output-dir "$WORK_ROOT"
    test -f "$WORK_ROOT/template_calibration_selection.json"
    ;;
  qualification)
    "$PYTHON" -u scripts/run_phase2_clean_gate.py qualification \
      --selected-manifest experiments/r16p19_libero_phase2/selected_executor_manifest.json \
      --output-dir "$WORK_ROOT" \
      --save-failure-videos
    test -f "$WORK_ROOT/qualification_summary.json"
    ;;
  formal-competence)
    qualification_summary=${R16P19_PHASE2_QUALIFICATION_SUMMARY:?qualification summary is required}
    "$PYTHON" -u scripts/run_phase2_clean_gate.py formal_competence \
      --selected-manifest experiments/r16p19_libero_phase2/selected_executor_manifest.json \
      --qualification-summary "$qualification_summary" \
      --output-dir "$WORK_ROOT" \
      --save-failure-videos
    test -f "$WORK_ROOT/formal_competence_summary.json"
    ;;
  closed-loop)
    formal_summary=${R16P19_PHASE2_FORMAL_SUMMARY:?formal summary is required}
    "$PYTHON" -u scripts/run_phase2_closed_loop.py \
      --selected-manifest experiments/r16p19_libero_phase2/selected_executor_manifest.json \
      --formal-summary "$formal_summary" \
      --output-dir "$WORK_ROOT"
    test -f "$WORK_ROOT/final_status.json"
    ;;
esac

wait "$monitor_pid"
monitor_pid=
"$PYTHON" - "$EVIDENCE_DIR/RUN_COMPLETE.json" <<'PY'
import json
import os
import pathlib
import sys
import time

target = pathlib.Path(sys.argv[1])
value = {
    "completed_at_unix": time.time(),
    "run_complete": True,
    "run_id": os.environ["PAI_CANARY_RUN_ID"],
    "stage": os.environ["R16P19_PHASE2_STAGE"],
    "uid": os.getuid(),
    "gid": os.getgid(),
    "work_root": os.environ["R16P19_PHASE2_OUTPUT_ROOT"],
}
temporary = target.with_name(".%s.tmp.%d" % (target.name, os.getpid()))
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
os.replace(str(temporary), str(target))
PY
test "$(stat -c '%u:%g' "$EVIDENCE_DIR/FIRST_COMPLETED_CELL.json")" = "$LEON_UID:$LEON_GID"
test "$(stat -c '%u:%g' "$EVIDENCE_DIR/RUN_COMPLETE.json")" = "$LEON_UID:$LEON_GID"
printf 'R16P19_PHASE2_STAGE_COMPLETE run_id=%s stage=%s\n' "$RUN_ID" "$STAGE"
