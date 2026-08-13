#!/usr/bin/env bash
set -Eeuo pipefail

export R16P19_EXPECTED_COMMIT=ae362efeba68643ab4dd2a99cfd295c72a9cbdcc

exec /mnt/cpfs/zbl-cpfs-new/USERS/leon/code/r16p19-libero-phase1-20260813/launch/pai_formal_phase1.sh "$@"
