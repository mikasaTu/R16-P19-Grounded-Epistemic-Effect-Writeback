#!/usr/bin/env bash
set -Eeuo pipefail
export R16P19_PHASE1B_STAGE=qualify-fallback
export R16P19_EXPECTED_COMMIT=49e2bfe240c4f7d9dcc425c1ab35426ee95e7301
export R16P19_PHASE1B_CODE_ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r16p19-libero-phase1b-49e2bfe
export R16P19_PHASE1B_OUTPUT_ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16p19_libero_phase1b/v2/experiment
export R16P19_PHASE1B_CHECKPOINT_ROOT=/mnt/cpfs/zbl-cpfs-new/CKPT/leon/torch/r16p19_libero_phase1b/v2
exec /mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r16p19-libero-phase1b-49e2bfe/launch/pai_phase1b_stage.sh "$@"
