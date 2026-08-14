# Phase-3 PAI execution audit

## Frozen execution topology

- Alibaba PAI/DLC resource: `quotakzri8a5wqcp`.
- One worker with a 2×A800 carrier, 12 CPU, and 200 GiB memory.
- Only physical GPU 0 was exposed to the scientific process; GPU 1 remained
  hidden, preserving the preregistered single-renderer-GPU constraint.
- W&B was disabled. Platform automatic restart count was zero.
- The process ran as UID/GID `2254:2254` and wrote directly to the durable CPFS
  evidence root.
- There was no probe job. Formal demo access occurred only in the single
  formal job listed below.

## Prepare attempts

| Run | JobId | Outcome | Evidence boundary |
|---|---|---|---|
| v1 | no job | local preflight failure | durable application root absent; no PAI submission |
| v2 | no job | local preflight failure | PAI output parent absent; no PAI submission |
| v3 | `dlc3r2jgqc8lu2p1` | infrastructure/controller failure | required Phase-3 variables omitted by `env -i`; failed before application cells |
| v4 | `dlc1xieb62snpwdv` | environment compatibility failure | frozen Python 3.8/LIBERO namespace check failed before application cells |
| v5 | `dlc6j1afexrn1am1` | succeeded | 35/35 tests, qualification, K calibration, and prepare evidence completed |

The failed prepare attempts did not access formal demos and did not contribute
rows to any scientific result file.

## Formal run

| Field | Frozen value |
|---|---|
| Run ID | `r16p19-phase3-formal-20260814-v1` |
| JobId | `dlc1o37keilxr3sr` |
| Source commit | `6089341084c9e39bb76b065a3c51fa3aa53ced25` |
| Source tree | `5c8eb5e72c57d2620c0dad395e7d5c631691474c` |
| Formal contract SHA256 | `a78728e6bfe0bc33ae016a974c2bafa956e8eb1c3e744bc4616e276d3a4a438f` |
| Launcher SHA256 | `79595973e25cd5075ccb364a2f28c44758237c641d8796ac49277775eeab8861` |
| First durable formal cell | `formal_access_ledger/stove_moka/demo_40.complete.json` |
| Running interval | `2026-08-14T06:54:51Z` to `2026-08-14T12:01:18Z` |
| PAI duration | `18,485 s` (5 h 8 min 5 s) |
| Terminal state | `Succeeded` / `JobSucceeded` |
| Workload pod | one pod, one PodUid, terminal `Succeeded` |
| Platform restarts | zero; AIMaster disabled |

The formal job ran the replay-only gate and every downstream diagnostic even
after gate failure, as explicitly required by the user before formal access.
It produced the 12-cell smoke, 900-row main grid, 180-row D1 grid, 150 paired
audits, 176 forced-decision rows, 10,000 bootstrap draws, exact paired tests,
90 ablation rows, and the complete preregistered video policy.

One transient `pai-job get` invocation returned local rc=2 during monitoring.
The next exact readback showed the same JobId still `Running`, application
artifacts continued advancing, and the job later ended `Succeeded`; this was
classified as a control-plane/network read blip, not a workload failure or
restart. The workflow remained CLI/OpenAPI-only, so
`browser_not_used=fifo_not_applicable`.

## Superseded service-record cleanup

After successful prepare v5 had a persisted first real cell and terminal
marker, the exact failed v3/v4 predecessors passed `--prepare-only` checks for
local ledger binding, singleton live identity, workspace, ResourceId, nonce,
terminal `Failed`, and `purpose=formal-evaluation`. They were then deleted one
at a time through the pinned `DeleteJob` helper:

| Run | JobId | DeleteJob result | Fresh absence |
|---|---|---|---|
| prepare v3 | `dlc3r2jgqc8lu2p1` | HTTP 200; request `01A0003C-7AD2-589A-B033-676098709DC5` | exact ListJobs empty and GetJob `403 OperationForbidden` |
| prepare v4 | `dlc1xieb62snpwdv` | HTTP 200; request `01A0003C-9AAB-5720-ACE3-B3EEF95D614D` | exact ListJobs empty and GetJob `403 OperationForbidden` |

Only those two PAI service rows were removed. Their registry run directories,
ledger entries, before/after API snapshots, logs, and CPFS evidence remain in
the repository. Successful prepare v5 and formal v1 were not deleted.

## Evidence included in this repository

- `pai/workload/r16p19-phase3-formal-20260814-v1/`: launch contract,
  first-cell marker, complete log, and terminal marker. Runtime caches are
  deliberately omitted.
- `pai/workload/r16p19-phase3-prepare-20260814-v5/`: successful prepare
  application log and terminal markers.
- `pai/control-plane/r16p19-phase3-prepare-20260814-v1` through `v5`, plus the
  matching formal directory: requested/resolved contracts, exact launcher
  payloads, submit output, GetJob readbacks, and exact v3/v4 deletion evidence.
  Secret injection was disabled and the captured password fields are empty.
- `video_manifest.jsonl` and `video_policy_summary.json`: one record for every
  required video-policy request: 748 requests, 496 rendered videos, 252
  deterministic invalid-snapshot errors, and four rendered outcome mismatches.
- `SHA256SUMS`: application-produced byte manifest.
- `BUNDLE_SHA256SUMS`: repository evidence-bundle byte manifest.

No failed job was silently retried into the same run identity, and no formal
result was generated on the development host. No PAI probe was created.
