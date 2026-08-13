#!/usr/bin/env bash
set -Eeuo pipefail
export R16P19_PHASE1B_STAGE=train-primary
export R16P19_EXPECTED_COMMIT=bf06ef5b2e8ddeb3a528491ddb27525d28853432
export R16P19_PHASE1B_CODE_ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r16p19-libero-phase1b-bf06ef5
exec /mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r16p19-libero-phase1b-bf06ef5/launch/pai_phase1b_stage.sh "$@"
