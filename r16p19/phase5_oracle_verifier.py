"""Oracle broker boundary: truth is translated to signed ledger events only."""

from __future__ import annotations

from .phase4_event_broker import Phase4EventBroker


class OracleEffectVerifierBroker(Phase4EventBroker):
    """Named boundary used by Phase-5; arms never receive raw simulator state."""

    def emit_observation(self, effect_id: str, attempt_id: str, command_id: str, truth: bool):
        if truth:
            return (
                self.positive(effect_id, "base_view", attempt_id, command_id, True),
                self.positive(effect_id, "wrist_view", attempt_id, command_id, True),
                self.witness(effect_id, attempt_id, command_id, True),
            )
        return (self.contradiction(effect_id, False),)


__all__ = ["OracleEffectVerifierBroker"]
