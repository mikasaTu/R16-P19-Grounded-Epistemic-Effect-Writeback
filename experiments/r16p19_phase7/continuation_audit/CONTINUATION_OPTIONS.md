# Concrete continuation options — proposed, not executed

The user's instruction to continue authorizes searching/recovering inputs and finishing work within the original constraints. It does not identify which of the conflicting explicit constraints should change. B0/B2 remain unchanged. No new training, rollout, PAI or GPU use has been performed.

## Option A — preserve original certification requirements

1. Recover the exact original linear weights or original per-sample scores. If no copy exists, user must supply them; aggregate metrics cannot reconstruct the fixed-FPR frontier.
2. Supply additional valid, nonformal, model-fit-excluded observations with provenance and a separately approved split addition. Under the current calibration+pilot counts, at least21/21/20 additional positive observations for tomato/book/close are necessary merely to make a0.90 Wilson lower bound possible with all positives correctly accepted; these are mathematical lower bounds, not a guarantee of success or an independence claim.
3. Freeze a new protocol version before deriving new thresholds; retain the original failed-certification record. Keep qualification and formal separate and explicitly address prior qualification reuse/pilot environment overlap.
4. Execute full model comparison, calibration, heldout/frontier, then B7 only if a complete eligible point exists. No rule can promise a passing scientific result in advance.

This option preserves the goal of certification. Without recoverable inputs and an approved data supplement, it has no executable next experiment.

## Option B — finish diagnostic questions in a separate preregistered round

Requires explicit permission to depart from the original no-retraining and qualified-point-only conditions:

1. Reconstruct a CPU linear candidate using original Phase-5 code, original natural seed0–4 data, original seed/steps/hyperparameters; label it reconstructed rather than claiming recovered original weights. Save all candidate parameters and per-sample scores. Do not overwrite Phase-5 artifacts.
2. Freeze an empirical-TPR target frontier0.80..0.99 and a diagnostic operating-point selection rule using only calibration+pilot/qualification. Retain sample counts and Wilson intervals, but allow diagnostic evaluation even if certification fails.
3. Selection rule proposal: on calibration+pilot, highest per-effect threshold meeting each empirical-TPR target; on qualification rank complete vectors by min effect TPR subject to receipt FPR<=0.02, then oracle decision agreement, then lower FPR, then original selected model, then higher target. If no vector meets the diagnostic FPR cap, retain and report the full frontier without fabricating a point.
4. Seal any selected diagnostic point before formal evaluation. Evaluate formal once; run decision replay and report adversarial/inherited-oracle/neutral-assumption quantities and current-point disagreement unit IDs. Mark all outputs diagnostic-only; G1 cannot be claimed from this changed rule.
5. Continue CPU only, no GPU/PAI/new rollout/S2 execution. Write new results under experiments/r16p19_phase7/supplement_v1/, preserve old B0 and evidence, publish both positive and negative results to main and the existing step8 report.

This option can answer missing diagnostic questions if the inputs are sufficient; it is not a substitute certification pass and is not authorized/executed yet.
