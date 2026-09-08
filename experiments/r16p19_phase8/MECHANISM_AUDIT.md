# Step9 mechanism audit (retrospective, no new idea)

The earlier completion statements and C4 102/18 substitution are withdrawn. This note explains code-supported mechanisms; it is not a Phase-8 final research report.

## Pooled certification changed the target quantity

`audit_c0_c2.py:select` chooses the largest pooled threshold with the frozen Wilson lower endpoint >=0.90. At the original Platt working threshold 0.6477173811021721, successes are 346/371, 4/14, 7/14, 375/381, and 1/15. Pooled successes are 733/795 and pooled LCB is 0.9012777, yet rare-effect lower endpoints are 0.1172138, 0.2679920, and 0.0118669.

The two common effects supply 752/795 = 94.59% of positives. An objective dominated by them can tolerate failure on rare effects. Pooling only changes the estimand to a frequency-weighted average; it does not provide 795 independent observations about each rare effect. The old `all(v is not None ...)` condition incorrectly treated non-null low bounds as certification. Correct code explicitly requires all five lower endpoints >=0.90.

Even an all-success rule on 14/14/15 positives has Wilson lower endpoints 0.784689/0.784689/0.796117. A deterministic scalar mapping cannot increase these denominators. Thus probability-scale alignment and statistical certification are distinct questions.

## Monotone maps preserve each effect's rank, but not receipt ranking

Positive temperature scaling and positive-slope Platt scaling are increasing maps of the scalar logit. Their inverse maps recover a raw threshold for every transformed threshold. Therefore the feasible per-effect decision sets and effect AUC are preserved, subject to explicitly checked finite-precision ties. They do not train the verifier weights w1/b1/w2/b2.

For a shared threshold after different per-effect maps, each effect instead gets a different inverse raw threshold. This can change which effect bottlenecks `min(scores)`, change receipt ranking and receipt AUC, and change false receipt upgrades. Per-effect AUC invariance alone never implies invariance of min-aggregation AUC. C1 and C3 report both quantities separately.

## Dependence and budget error

The old budget used positive_frames / all_frames as if it were positive_frames / episodes. There are 15 calibration+pilot episodes per task. Rare effects occur in 14, 14, and 15 positive episodes respectively. At the frozen central 95%/99% Wilson endpoints, all-success minimum independent positive-unit counts are 35 and 60. This gives standalone per-task collection extrapolations of 38/38/35 or 65/65/60 episodes. These are not guaranteed yields or a launch request.

The all-positive-frames-detected episode interval has a different estimand from frame-weighted TPR. Both are labeled explicitly; a separate episode bootstrap estimates uncertainty of the frame-ratio statistic. A bootstrap with no observed failures can have zero width and is not certification. Fitting the map and evaluating it on the same labels adds selection dependence that neither pooled Wilson nor Bonferroni removes.

For fresh certification, use independent seeds disjoint from calibration/pilot/qualification/formal; keep the certification allocation untouched by mapping fitting, model choice, and threshold choice. The quoted standalone budget is for the certification allocation, with fitting/selection episodes additional.

## Ledger AND versus independent effect upgrades

C4_AUDITED.json and C4_MECHANISM.md supply actual event-level comparisons at the same fixed diagnostic threshold. The new rule must preserve provenance and witness checks while replacing all-effect acceptance with effect-specific acceptance. Missing another effect must no longer suppress an otherwise witnessed effect. At a single receipt with identical witness/provenance, independent acceptance is a superset of AND acceptance; false positives cannot be assumed to decrease. Ledger concordance and gain assumptions must be measured, not copied from an earlier variant.

## Protocol boundary

Original C0 was incompletely specified and hashed after fitting started. This cannot be repaired retrospectively. The original C0 bytes remain unchanged; the added read-audited selection seal proves this repair process did not read formal inputs, not that the original experiment was properly preregistered. Formal labels appear only in ceiling diagnostics and the frozen Phase-5 reproduction, never a new workpoint evaluation or selection seal.

References for interval definitions: https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm and https://www.itl.nist.gov/div898/handbook/prc/section4/prc463.htm .

## Exact receipt-order decomposition of the improvement and degradation

`MECHANISM_PAIR_AUDIT.json` reconstructs every positive/negative receipt pair. For Platt, estimation improves 565 pairs and worsens 199 of 97,911 (AUC delta +0.0037380887). On qualification it improves 190 but worsens 254 of 42,224 (AUC delta -0.0015157257). The bottleneck effect changes on 381 estimation and 306 qualification receipts. This exactly accounts for the AUC movements; it does not require any change to per-effect ranking or verifier capacity. Temperature improves 480 vs worsens171 pairs on estimation, and126 vs80 on qualification.

The persistent C4 wrapper separately stores detector-gated upgrade state/decision with underlying witness proof ids and revocation status. At identical events, partial detector evidence can now create one REALIZED upgrade while the other stays UNKNOWN; AND formerly vetoed both. This increases correct effect detections from42/190 to108/190, but also false upgrades0/510 to2/510 and reduces full-unit agreement19/60 to18/60. These are nonformal descriptive counts with correlated repeated conditions, not independent certification evidence. Pilot task-success envelopes are computed separately from stored M0/M3 task outcomes; they are not effect-TPR gains.
