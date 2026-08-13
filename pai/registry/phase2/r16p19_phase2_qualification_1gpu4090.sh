#!/usr/bin/env bash
set -Eeuo pipefail

export R16P19_PHASE2_STAGE=qualification
export R16P19_EXPECTED_COMMIT=8963f8cb3b10201095a47c48cec13ce11b0832f0
export R16P19_PHASE2_OUTPUT_ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16p19_libero_phase2/qualification-8963f8c

exec /bin/bash --noprofile --norc \
  /mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r16p19-phase1b-publication-20260813/launch/pai_phase2_stage.sh
