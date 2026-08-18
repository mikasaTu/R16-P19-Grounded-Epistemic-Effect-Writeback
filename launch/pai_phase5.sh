#!/usr/bin/env bash
set -euo pipefail

expected_uid=2254
expected_gid=2254
if [ "$(id -u)" != "$expected_uid" ] || [ "$(id -g)" != "$expected_gid" ]; then
  echo "[phase5][fatal] expected uid:gid 2254:2254, got $(id -u):$(id -g)" >&2
  exit 2
fi

: "${PAI_CANARY_RUN_DIR:?PAI_CANARY_RUN_DIR is injected by pai-job-registry}"
: "${PAI_CANARY_RUN_ID:?PAI_CANARY_RUN_ID is injected by pai-job-registry}"
export ARTIFACT_DIR="${ARTIFACT_DIR:-$PAI_CANARY_RUN_DIR}"
export RUN_ID="${RUN_ID:-$PAI_CANARY_RUN_ID}"

source_root="/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/R16-P19-Grounded-Epistemic-Effect-Writeback-phase5"
qpilots_root="/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/QPILOTS-r16p15-stage1-task64-20260812"
checkpoint="/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/openpi/r16p15/openpi-assets/checkpoints/pi05_libero"
rollout_root="$ARTIFACT_DIR/rollouts"
result_root="$ARTIFACT_DIR/results"
python_bin="/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/openpi_py311/bin/python"

for required in "$source_root/.git" "$qpilots_root/.git" "$checkpoint/params/manifest.ocdbt" "$python_bin"; do
  [ -e "$required" ] || { echo "[phase5][fatal] missing $required" >&2; exit 3; }
done
[ "$(git -C "$qpilots_root" rev-parse HEAD)" = "eacf47b981e3b22357f8a74902f8dad8cfcfa375" ] || exit 4
[ "$(git -C "$qpilots_root/third_party/openpi" rev-parse HEAD)" = "54cbaee6ae0c010a1ed431871cdaa8f4684ac709" ] || exit 4
[ "$(sha256sum "$source_root/r16p19/memory.py" | awk '{print $1}')" = "4992462e105306eca9a777619fa5c7418f90c1289e4e9f6fdde1bdc2fbfce4c5" ] || exit 4

mkdir -p "$rollout_root" "$result_root" "$ARTIFACT_DIR/logs"
probe="$ARTIFACT_DIR/.write-probe"
printf phase5 > "$probe"
[ "$(stat -c '%u:%g' "$probe")" = "2254:2254" ] || exit 5
rm "$probe"

export R16P19_QPILOTS_ROOT="$qpilots_root"
export QPILOTS_OPENPI_ROOT="$qpilots_root/third_party/openpi"
export QPILOTS_LIBERO_SITE="$QPILOTS_OPENPI_ROOT/third_party/libero"
export R16P19_PI05_CHECKPOINT="$checkpoint"
export LIBERO_CONFIG_PATH="/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/libero/r16p15-stage1-task64"
export OPENPI_DATA_HOME="/mnt/cpfs/zbl-cpfs-new/USERS/leon/cache/openpi/r16p15"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$source_root:$QPILOTS_OPENPI_ROOT/src:$qpilots_root:$QPILOTS_OPENPI_ROOT/packages/openpi-client/src"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES="$qpilots_root/configs/r16p15/10_nvidia.json"
export NVIDIA_DRIVER_CAPABILITIES=all
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.72
export WANDB_ENTITY=chen_jian-cj-workspace
export WANDB_PROJECT=r16p19-phase5-bounded-ascel

cd "$source_root"
"$python_bin" -m pytest -q tests/test_phase5_*.py | tee "$ARTIFACT_DIR/logs/tests.log"

pids=()
world_size="${PAI_CANARY_EXPECTED_GPUS:?PAI_CANARY_EXPECTED_GPUS is injected by pai-job-registry}"
for ((rank = 0; rank < world_size; rank++)); do
  CUDA_VISIBLE_DEVICES="$rank" EGL_DEVICE_ID=0 RANK="$rank" WORLD_SIZE="$world_size" \
    "$python_bin" -m r16p19.phase5_rollout --output-root "$rollout_root" --rank "$rank" --world-size "$world_size" \
    > "$ARTIFACT_DIR/logs/rollout-rank-${rank}.log" 2>&1 &
  pids+=("$!")
done

worker_failure=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    worker_failure=1
  fi
done
[ "$worker_failure" = 0 ] || { echo "[phase5][fatal] at least one rollout worker failed" >&2; exit 6; }

"$python_bin" -m r16p19.phase5_runner --rollout-root "$rollout_root" --result-root "$result_root" \
  | tee "$ARTIFACT_DIR/logs/postprocess.log"

"$python_bin" - "$result_root" "$ARTIFACT_DIR" <<'PY'
import json
import os
import sys
import tempfile
from pathlib import Path

result_root = Path(sys.argv[1])
artifact_root = Path(sys.argv[2])
required = [
    result_root / "PIPELINE_COMPLETE.json",
    result_root / "final_decision.json",
    result_root / "oracle_formal_results.jsonl",
    result_root / "learned_verifier_formal_results.jsonl",
    result_root / "support_formal_results.jsonl",
]
for path in required:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"missing terminal artifact {path}")
payload = {
    "schema_version": 1,
    "run_id": os.environ["RUN_ID"],
    "uid": os.getuid(),
    "gid": os.getgid(),
    "result": json.loads((result_root / "final_decision.json").read_text()),
}
destination = artifact_root / "FORMAL_COMPLETE.json"
with tempfile.NamedTemporaryFile("w", dir=artifact_root, prefix=".formal-complete-", delete=False) as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
    temporary = Path(handle.name)
os.replace(temporary, destination)
PY

echo "[phase5] complete: $ARTIFACT_DIR/FORMAL_COMPLETE.json"
