# PAI submission evidence

This directory contains the R16-P19-specific portion of the PAI job workflow:

- `registry/`: the exact two-GPU profile and thin payload binding;
- `controller-patches/`: the three commits that added and updated the
  fail-closed R16-P19 contract in the external registry;
- `control-plane/`: requested/resolved contracts, submission state, redacted
  environment readbacks and GetJob evidence for each attempt.

The full PAI registry repository is not copied because it contains unrelated
experiments. Applying the patches to its recorded parent commit reconstructs
the R16-P19 controller changes.

No credential is included. Secret environment values are literal
`<redacted>` placeholders.
