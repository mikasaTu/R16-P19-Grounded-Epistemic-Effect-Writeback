# Phase-3 validation record

## Tests and immutable inputs

| Check | Result |
|---|---|
| Phase-3 specified test files | `12 passed in 5.77s` |
| Full repository test suite | `35 passed in 4.36s` |
| Same full suite at PAI launch | `35/35 passed` |
| Protected `r16p19/memory.py` SHA256 | `4992462e105306eca9a777619fa5c7418f90c1289e4e9f6fdde1bdc2fbfce4c5` |
| Formal execution contract SHA256 | `a78728e6bfe0bc33ae016a974c2bafa956e8eb1c3e744bc4616e276d3a4a438f` |
| Selected-chain manifest SHA256 | `53c6a9fa448267b3b77a9fa4458552e9383fda8b1a63fcf127acf80fe4904677` |
| Formal source tree | `5c8eb5e72c57d2620c0dad395e7d5c631691474c` |

The local commands used the frozen LIBERO Python environment and did not run
training or formal simulation. The formal PAI test result is retained in
`pai/formal-v1/run.log`.

## Exact artifact row counts

| Artifact | Rows / units |
|---|---:|
| non-formal snapshot cells | 80 |
| extracted non-formal effect segments | 320 |
| qualification replay | 400 |
| persistence K=2/K=4/K=8 | 150 each |
| formal replay-only | 250 |
| smoke matrix | 12 |
| main matrix | 900 |
| delayed-receipt matrix | 180 |
| paired-unit audit | 150 |
| first-divergence forced replays | 176 from 88 eligible units |
| mechanism ablations | 90 |
| cluster-bootstrap draws | 10,000 |
| video-policy requests | 748 unique: 496 rendered, 252 render errors |

## Structural checks

- Exact formal grid: pass (`900/900`, no duplicate key).
- Exact D1 grid: pass (`180/180`, no duplicate key).
- Fault/truth leakage: zero.
- Receipt-broker errors: zero.
- Fault-injector errors: zero.
- B6 dangling parents / transition violations: zero / zero.
- Resident-memory maximum: 27, below the frozen 32-slot capacity.
- Pair-prefix audit: fail (`27/150` non-identical before first decision).
- Rendered-video outcome consistency: four mismatches among 496 rendered
  requests; retained as a diagnostic warning and not used to rewrite primary
  results.
- Application `SHA256SUMS`: pass (`sha256sum --quiet -c SHA256SUMS`).
- Repository `BUNDLE_SHA256SUMS`: generated after final evidence assembly.

The scientific terminal remains `BLOCKED_BY_IMPLEMENTATION`; passing tests and
complete grids do not override the replay-competence and pairing failures.
