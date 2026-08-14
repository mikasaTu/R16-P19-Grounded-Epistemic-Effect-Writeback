#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

NEW_ROOT=/mnt/cpfs/zbl-cpfs-new
SOURCE_ROOT="$NEW_ROOT/USERS/leon/code/r16p19-phase1b-publication-20260813"
EXPERIMENT_ROOT="$SOURCE_ROOT/experiments/r16p19_libero_phase3"
LIBERO_ROOT="$NEW_ROOT/USERS/leon/code/LIBERO-r16p19-official-8f1084e"
PYTHON="$NEW_ROOT/USERS/leon/envs/libero-original/bin/python"
EXPECTED_LIBERO_COMMIT=8f1084e3132a39270c3a13ebe37270a43ece2a01
EXPECTED_LIBERO_TREE=99f4ada3f1d62e026fc9ff2390eb4ff8a1760e60
EXPECTED_PYTHON_SHA=903441582909636b4316667ee22beb1e6e2726991584ea9265c765ac8388abd1
EXPECTED_STOVE_DATASET_SHA=6b30906a52a5741e98ef447d27e7066d6c0be4a5f7acd7ecaf1cb7468aca4aa9
EXPECTED_BOWL_DATASET_SHA=703950f48a3c49dfde61be489ade91527f16e1449b4f29a85f2e51153cef3638
STOVE_DATASET="$NEW_ROOT/dataset/leon/embodied_benchmark/datasets/LIBERO/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5"
BOWL_DATASET="$NEW_ROOT/dataset/leon/embodied_benchmark/datasets/LIBERO/libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo.hdf5"
LEON_UID=2254
LEON_GID=2254

RUN_ID=${PAI_CANARY_RUN_ID:?PAI_CANARY_RUN_ID is required}
EVIDENCE_DIR=${PAI_CANARY_RUN_DIR:?PAI_CANARY_RUN_DIR is required}
EXPECTED_GPUS=${PAI_CANARY_EXPECTED_GPUS:?PAI_CANARY_EXPECTED_GPUS is required}
STAGE=${R16P19_PHASE3_STAGE:?R16P19_PHASE3_STAGE is required}
EXPECTED_SOURCE_COMMIT=${R16P19_PHASE3_EXPECTED_COMMIT:?R16P19_PHASE3_EXPECTED_COMMIT is required}
INCLUDE_VIDEOS=${R16P19_PHASE3_INCLUDE_VIDEOS:-0}

monitor_pid=
on_error() {
  local status=$?
  if test -n "$monitor_pid"; then
    kill "$monitor_pid" 2>/dev/null || true
  fi
  printf 'R16P19_PHASE3_FAILED line=%s status=%s command=%q\n' \
    "${BASH_LINENO[0]:-?}" "$status" "$BASH_COMMAND" >&2
  exit "$status"
}
trap on_error ERR

case "$STAGE" in
  prepare|formal|render) ;;
  *) printf 'unknown Phase-3 stage: %s\n' "$STAGE" >&2; exit 64 ;;
esac
case "$INCLUDE_VIDEOS" in
  0|1) ;;
  *) printf 'invalid INCLUDE_VIDEOS value\n' >&2; exit 65 ;;
esac
for command in git sha256sum nvidia-smi stat realpath find wc sleep kill awk grep sync; do
  command -v "$command" >/dev/null
done
test "$(id -u):$(id -g)" = "$LEON_UID:$LEON_GID"
test "$EXPECTED_GPUS" = 2
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c '^NVIDIA A800')" = 2
test "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" = "$EXPECTED_SOURCE_COMMIT"
test "$(git -C "$LIBERO_ROOT" rev-parse HEAD)" = "$EXPECTED_LIBERO_COMMIT"
test "$(git -C "$LIBERO_ROOT" rev-parse 'HEAD^{tree}')" = "$EXPECTED_LIBERO_TREE"
test -z "$(git -C "$LIBERO_ROOT" status --porcelain)"
test "$(sha256sum "$(realpath -e "$PYTHON")" | awk '{print $1}')" = "$EXPECTED_PYTHON_SHA"
test "$(sha256sum "$STOVE_DATASET" | awk '{print $1}')" = "$EXPECTED_STOVE_DATASET_SHA"
test "$(sha256sum "$BOWL_DATASET" | awk '{print $1}')" = "$EXPECTED_BOWL_DATASET_SHA"
test "$(stat -c '%u:%g' "$EVIDENCE_DIR")" = "$LEON_UID:$LEON_GID"
test "$(stat -c '%u:%g' "$EXPERIMENT_ROOT")" = "$LEON_UID:$LEON_GID"

# Generated experiment artifacts are the only permitted worktree dirt.  This
# permits application-level resume while keeping code and old phases frozen.
while IFS= read -r status_line; do
  test -z "$status_line" && continue
  path=${status_line:3}
  case "$path" in
    experiments/r16p19_libero_phase3/*) ;;
    *) printf 'unexpected source worktree dirt: %s\n' "$status_line" >&2; exit 66 ;;
  esac
done < <(git -C "$SOURCE_ROOT" status --porcelain)

export PYTHONNOUSERSITE=1
export PYTHONPATH="$SOURCE_ROOT:$LIBERO_ROOT"
export LIBERO_CONFIG_PATH="$SOURCE_ROOT/experiments/r16p19_libero_phase1/libero_config"
# The resource carrier allocates two A800s, but the scientific process sees
# and renders on only physical GPU0, honoring the at-most-one-rendering-GPU
# Phase-3 contract.
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless
export MUJOCO_EGL_DEVICE_ID=0
export EGL_DEVICE_ID=0
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
export PAI_AUTOMATIC_FAULT_TOLERANCE=0
export WANDB_MODE=disabled
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TMPDIR="$EVIDENCE_DIR/tmp"
export XDG_CACHE_HOME="$EVIDENCE_DIR/cache/xdg"
export PYTHONPYCACHEPREFIX="$EVIDENCE_DIR/cache/pycache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$PYTHONPYCACHEPREFIX"

"$PYTHON" - "$EVIDENCE_DIR/phase3-launch-contract.json" <<'PY'
import json
import os
import pathlib
import subprocess
import sys

import torch
import libero

target = pathlib.Path(sys.argv[1])
source = pathlib.Path("/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/LIBERO-r16p19-official-8f1084e")
assert os.getuid() == 2254 and os.getgid() == 2254
assert torch.cuda.device_count() == 1
assert pathlib.Path(libero.__file__).resolve().is_relative_to(source)
value = {
    "allocated_gpu_count": 2,
    "application_resume_unit": "one_fsynced_rollout_or_replay_cell",
    "expected_source_commit": os.environ["R16P19_PHASE3_EXPECTED_COMMIT"],
    "gpu_inventory": subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"], text=True
    ).strip().splitlines(),
    "libero_import": libero.__file__,
    "pai_automatic_fault_tolerance": False,
    "pai_probe_created": False,
    "run_id": os.environ["PAI_CANARY_RUN_ID"],
    "scientific_visible_gpu_count": torch.cuda.device_count(),
    "stage": os.environ["R16P19_PHASE3_STAGE"],
    "uid_gid": "%d:%d" % (os.getuid(), os.getgid()),
    "wandb": "disabled_non_neural_evaluation",
}
temporary = target.with_name(".%s.tmp.%d" % (target.name, os.getpid()))
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(str(temporary), str(target))
PY
sync -f "$EVIDENCE_DIR/phase3-launch-contract.json"

exec > >(tee -a "$EVIDENCE_DIR/run.log") 2>&1
cd "$SOURCE_ROOT"
"$PYTHON" -m py_compile r16p19/phase3_*.py scripts/run_phase3_pipeline.py
"$PYTHON" -m pytest -q

case "$STAGE" in
  prepare)
    first_work="$EXPERIMENT_ROOT/cells/development/stove_moka/demo_0.json"
    complete="$EXPERIMENT_ROOT/prepare_stage_complete.json"
    ;;
  formal)
    first_work="$EXPERIMENT_ROOT/formal_access_ledger/stove_moka/demo_40.complete.json"
    complete="$EXPERIMENT_ROOT/formal_stage_complete.json"
    ;;
  render)
    first_work="$EXPERIMENT_ROOT/video_manifest.jsonl"
    complete="$EXPERIMENT_ROOT/video_policy_summary.json"
    ;;
esac
(
  while true; do
    if test -s "$first_work"; then
      "$PYTHON" - "$EVIDENCE_DIR/FIRST_COMPLETED_CELL.json" "$first_work" <<'PY'
import json, os, pathlib, sys, time
target = pathlib.Path(sys.argv[1])
value = {"first_persisted_path": sys.argv[2], "persisted_at_unix": time.time(), "uid": os.getuid(), "gid": os.getgid()}
temporary = target.with_name(".%s.tmp.%d" % (target.name, os.getpid()))
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(str(temporary), str(target))
PY
      sync -f "$EVIDENCE_DIR/FIRST_COMPLETED_CELL.json"
      break
    fi
    sleep 5
  done
) &
monitor_pid=$!

if test "$STAGE" = formal && test "$INCLUDE_VIDEOS" = 1; then
  "$PYTHON" -u scripts/run_phase3_pipeline.py formal --include-videos
else
  "$PYTHON" -u scripts/run_phase3_pipeline.py "$STAGE"
fi
test -s "$complete"
wait "$monitor_pid"
monitor_pid=

"$PYTHON" - "$EVIDENCE_DIR/EVALUATION_COMPLETE.json" "$complete" <<'PY'
import hashlib, json, os, pathlib, sys, time
target = pathlib.Path(sys.argv[1])
complete = pathlib.Path(sys.argv[2])
value = {
    "application_complete_path": str(complete),
    "application_complete_sha256": hashlib.sha256(complete.read_bytes()).hexdigest(),
    "completed_at_unix": time.time(),
    "run_id": os.environ["PAI_CANARY_RUN_ID"],
    "stage": os.environ["R16P19_PHASE3_STAGE"],
    "status": "complete",
    "uid": os.getuid(),
    "gid": os.getgid(),
}
temporary = target.with_name(".%s.tmp.%d" % (target.name, os.getpid()))
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(str(temporary), str(target))
PY
sync -f "$EVIDENCE_DIR/EVALUATION_COMPLETE.json"
test "$(stat -c '%u:%g' "$EVIDENCE_DIR/FIRST_COMPLETED_CELL.json")" = "$LEON_UID:$LEON_GID"
test "$(stat -c '%u:%g' "$EVIDENCE_DIR/EVALUATION_COMPLETE.json")" = "$LEON_UID:$LEON_GID"
printf 'R16P19_PHASE3_STAGE_COMPLETE run_id=%s stage=%s\n' "$RUN_ID" "$STAGE"
