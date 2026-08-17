"""Verifier metric API used by qualification and shift evaluations."""

from .phase5_verifier_model import auroc, metrics

__all__ = ["auroc", "metrics"]
