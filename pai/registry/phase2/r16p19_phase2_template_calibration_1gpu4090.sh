#!/usr/bin/env bash
set -Eeuo pipefail

export R16P19_PHASE2_STAGE=template-calibration
export R16P19_EXPECTED_COMMIT=136e8923829c0436ca27755078a609f91bcf75a5
export R16P19_PHASE2_OUTPUT_ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r16p19_libero_phase2/template-calibration-1a96b3b

exec /bin/bash --noprofile --norc \
  /mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r16p19-phase1b-publication-20260813/launch/pai_phase2_stage.sh
