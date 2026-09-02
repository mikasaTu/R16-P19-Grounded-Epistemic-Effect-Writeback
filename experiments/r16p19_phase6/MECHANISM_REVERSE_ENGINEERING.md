# Mechanism reverse-engineering

This document reverse-explains measured changes from frozen code and data. It does not propose a new idea.

## Load-bearing implementation

Phase-5 computes calibrated per-effect probabilities and then applies one global threshold followed by `numpy.all`. The all-reduction is conjunctive: one missed effect rejects the complete receipt.

## Isolated causal contrast

- B (unchanged global threshold + AND): min effect TPR `0.0000`, false upgrades `0`.
- C (only per-effect thresholds changed; AND retained): min effect TPR `0.8421`, false upgrades `8`.
- D (calibration-only weighted soft rule): min effect TPR `0.0000`, false upgrades `4`.

## Mechanistic conclusion

If C passes G1, detector capacity is not the limiting factor on this substrate: threshold scale mismatch plus AND amplification caused the zero learned gain. If neither C nor D passes, the evidence instead supports insufficient action/effect detection. The decision file records which branch occurred.
