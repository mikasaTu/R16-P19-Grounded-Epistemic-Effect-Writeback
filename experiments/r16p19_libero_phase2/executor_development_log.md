# Phase-2 executor development log

This log contains development-only evidence from init 40--59 and demo 30--39.
No qualification init 60--79 or formal init 0--19 was accessed at this stage.

## Reproducibility correction

Early unseeded development results are retained under `artifacts/phase2/` but
are marked `INVALIDATED_FOR_SELECTION_PRE_SEED_FIX`.  A hidden soft-reset RNG
dependency made outcomes depend on earlier rollouts.  The fixed reset rule is
`1619 + 1000 * task_ordinal + init_index`.  Independent repeats of both task
cells at init 40 became byte-identical; see
`artifacts/phase2_seeded/determinism_audit.json`.

## Fixed-seed component ablation

All rows below use both tasks and init 40--59 (40 paired rollouts), the complete
demo 0--29 extraction manifest, position gain 10, orientation gain 2, and a
0.015 m waypoint tolerance.  A0--A2 execute all eight proposed actions; A3
executes four and replans.

| Arm | Added component | Full task | Minimum effect | Repeated loop | Mean steps |
|---|---|---:|---:|---:|---:|
| A0 | world-frame open loop | 0.000 | 0.000 | 1.000 | 375.025 |
| A1 | effect-local retargeting | 0.000 | 0.000 | 1.000 | 365.750 |
| A2 | Cartesian feedback + demonstrated feed-forward | 0.300 | 0.200 | 0.700 | 337.950 |
| A3 | four-action receding-horizon prefix + fixed retry offsets | 0.725 | 0.650 | 0.275 | 192.100 |

Mechanism account:

- **Confirmed:** local-frame retargeting alone did not improve behavior on
  these paired cells (A0=A1=0 full success).
- **Confirmed:** closing the Cartesian loop was the first component to create
  nonzero effect and task success (A1→A2: +0.30 full, +0.20 minimum effect).
- **Confirmed:** shortening the executed prefix from eight to four and
  replanning reduced stale-action repetition (A2→A3: +0.425 full,
  +0.450 minimum effect, -0.425 loop rate, -145.85 mean steps).
- **Confirmed:** disabling only the demonstrated Cartesian feed-forward term
  reduced full success from 0.725 to 0.300, minimum effect from 0.650 to 0.150,
  and raised loop rate from 0.275 to 0.700.  Pose feedback alone reaches free
  space but cannot reliably actuate the stove/drawer contacts; the demonstrated
  command supplies the direction and force-like persistent delta at constrained
  waypoints.  The raw paired probe is under
  `artifacts/phase2_seeded/feedforward_diagnostic/`.

## Reverse checks that lowered performance

The development trace showed a switch path projecting from waypoint 22 back to
18.  A monotonic cursor was therefore tested as a one-factor change.  It
lowered single-attempt full success from 0.725 to 0.475 and minimum effect from
0.650 to 0.400; four-attempt full success was 0.600 with loop rate 0.375.
Trace inspection showed two mediators: the preceding effect terminated in a
less useful endpoint pose, and the cursor could remain attached to a waypoint
that feed-forward contact motion had made unreachable.  The cursor is disabled
in the candidate executor and retained only as a negative ablation.

Retry traces also showed that recomputing template rank after a failed attempt
could select the same template twice.  Freezing the geometric template order
at the first call made retries distinct but did not change aggregate full
success (0.775); with the corrected loop definition it exposed a 0.225 loop
rate.  A pre-grasp re-approach recovered one previously failed stove cell but
made other cells spend most of their budget chasing an object already displaced
by the first failed grasp.  It is therefore a diagnostic, not yet selected.

## Preregistered one-factor controller calibration

The search starts at gain 8, orientation gain 2, and tolerance 0.010 m.  At
each stage selection is minimum per-effect success, then full-task success,
then mean steps; qualification/formal data are forbidden.

| Candidate | Minimum effect | Full task | Mean steps |
|---|---:|---:|---:|
| gain 6, orientation 2, tol 0.010 | 0.150 | 0.400 | 200.775 |
| gain 8, orientation 2, tol 0.010 | 0.150 | 0.325 | 187.750 |
| gain 10, orientation 2, tol 0.010 | 0.050 | 0.350 | 187.800 |
| gain 6, orientation 1.5, tol 0.010 | 0.350 | 0.525 | 184.225 |
| gain 6, orientation 3, tol 0.010 | 0.150 | 0.275 | 229.750 |
| gain 6, orientation 1.5, tol 0.008 | 0.300 | 0.425 | 186.025 |
| gain 6, orientation 1.5, tol 0.015 | 0.400 | 0.450 | 200.300 |

The rule selects gain 6, orientation gain 1.5, and tolerance 0.015 m.  The
larger tolerance improves the worst effect but lowers aggregate task success,
so this is a fairness-oriented bottleneck tradeoff rather than a global optimum.

## Template calibration boundary

The complete 24-template extraction is immutable.  A calibration subset will
be chosen only from demo 30--39 using the already frozen rule in
`template_calibration_rule.json`.  Inferred-transition episodes are excluded
for the affected effect.  The job is cell-resumable and runs on a single PAI
rendering GPU.  Init 40--59 diagnostics do not enter that selection score.
