# Artifacts

`formal/` contains the authoritative successful PAI run. `failed/` contains the
first failed worker and its root-cause traceback. `checkpoints/` contains all
three complete actor checkpoints retained by the run. `local-validation/`
contains checks rerun from the GitHub publication checkout.

Run-level raw outputs are intentionally committed: `trace_events.jsonl`,
`trace_case_results.jsonl`, `memory_outputs.jsonl`, demo labels, metrics,
failure cases, provenance and readiness notes.

Verify all files in this directory with:

```bash
sha256sum -c BUNDLE_SHA256SUMS
```

The formal experiment subdirectory also carries the original worker-generated
`SHA256SUMS`, which covers its 24 preregistered deliverables.
