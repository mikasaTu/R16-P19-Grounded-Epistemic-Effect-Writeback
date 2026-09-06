# Continued goal audit — original experiments remain incomplete

Authoritative baseline checked live: main b7a13785ad0d6fa9345bcbe1611a97bec44276aa; worktree clean at start. The previous turn made progress and published results; it did not complete all original experiments.

| Requirement | Current evidence | Completion |
| --- | --- | --- |
| B0 preregister before B1; preserve hash | B0_PREREGISTRATION.json, commit8871bf9, SHA2567fd8f1cc3392aea5a375abe293843305178c789d5e38f243e34f9738f04f17ec | Complete |
| B1 fit-source inventory and formal ceiling at .02/.05 | B1_TRAINING_SPLIT_AUDIT.json and B1_CEILING.json: both minTPR1.0, FP10/612 | Complete diagnostic, not claim-eligible |
| B2 merge eligible splits and Wilson thresholds | B2_CALIBRATION.json and B2_SELECTION_SEAL.json: 371/14/14/381/15 positives, three null thresholds | Calculation complete; complete certified vector unavailable |
| B3 qualification per-effect and receipt evaluation | B3_HELDOUT.json: two effects evaluated; three lack thresholds, receipt metrics null | Partial |
| B4 min and sum(log p) complete curves/AUC | B4_SOFT.json, now rendered to B4_COMPLETE_CURVES.png/.pdf/.svg | Complete diagnostic |
| B5 compare both existing models under fixed-FPR criterion; no retraining | B5_MODEL_SELECTION.json: small_mlp minTPR2/3, FP38/2277; linear saved parameters/scores absent in original report | Partial; bounded campaign/source/history recovery audit found no original linear artifact |
| B6 target0.80..0.99 frontier and qualified operating-point seal | B6_FRONTIER.json: all20 targets lack certified vector; no seal | Target feasibility checked, requested numerical frontier/seal incomplete |
| B7 sealed-point replay, gain bounds and candidate units | B7_REPLAY.json and B7_S2_UNITS_STATUS.json: no selected point | Not executed |
| Decision gates, source consistency, publication | B_DECISION files;1680 mismatch0; GitHub main and Feishu readbacks | Decision checkpoint complete, scientific goal incomplete |

## Obstruction independent of implementation

For n successes out of n, the Wilson95% lower bound is n/(n+z²), z=1.959963984540054. This is the largest possible lower bound for a fixed sample count. To reach0.90 requires at least35 positives, all correctly accepted. The existing three rare effects have14/14/15 and maxima0.784689/0.784689/0.796117; they cannot satisfy even the lowest B6 LCB target0.80. Removing the explicit minimum-n35 check would not make B6 feasible.

A broader input audit counted60 natural episodes at seeds5/6 which did not fit the original base model. They remain outside the frozen B2 split definition. Even hypothetically adding them yields only34 tomato and33 book positives: maximum lower bounds0.898485 and0.895730. Thus this reinterpretation would still not certify G1. No such data merge or new threshold selection was performed. The runnable count/proof and input hashes are in SAMPLE_OBSTRUCTION.json and verify_sample_obstruction.py.

Qualification was used for original model selection and shares15 environment seeds with pilot. Its use here is held out only from new threshold fitting. Existing intervals must not be represented as fully independent episode-level guarantees.

## Why missing values cannot be filled with defaults

No B2 threshold exists for three effects. Assigning all missing effects to accept or reject would create a new decision rule. Independent audit found the remaining known-effect filter allows794 qualification receipts, of which765 are oracle-negative. Arbitrary missing-effect assignments can therefore span widely different false-upgrade counts. Neither arbitrary endpoint is the required B3 point estimate.

B7 requires a current selected operating-point seal. Using formal-derived F thresholds, prior S1 thresholds, the B5 ranking-only thresholds or a historical disagreement set would change that requirement. None has been substituted.

The goal remains active and incomplete. A diagnostic-only follow-up under an explicitly new preregistration is an option for user review; it cannot retroactively complete or change the original sealed B0 experiment.
