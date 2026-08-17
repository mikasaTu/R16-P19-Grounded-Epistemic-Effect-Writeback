# R16-P19 Phase-5 preregistration

This directory freezes the Phase-5 Bounded ASCEL embodied bridge experiment before implementation and before any formal-seed access.

The experiment uses three official LIBERO-10 tasks and the frozen official OpenPI `pi05_libero` checkpoint. It does not train a new policy, modify `r16p19/memory.py`, reinterpret prior phases, expose simulator truth to the policy/verifier/memory, or replay prefixes independently across arms.

The original gate order remains confirmatory. Per the explicit continuation instruction, a failed gate does not cancel later matrices: later results are completed as diagnostics, and the failed gate still blocks any pass claim.

The learned verifier and support-proof matrices are therefore allowed to run diagnostically after an upstream failure, but they cannot rescue the final confirmatory status.
