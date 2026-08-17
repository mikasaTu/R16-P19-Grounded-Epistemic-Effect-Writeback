from r16p19.phase5_ledger_live import BoundedASCELLedger
from r16p19.phase5_types import EvidenceReceipt


def receipt(**updates):
    fields = dict(receipt_id="r", attempt_id="a", command_id="c", effect_id="e", source_id="s", source_version=1, observed_at=2, expires_at=5, payload_hash="h", confidence=1.0, truth_value=True, active_attempt_credit=True, capture_id="cap", correlation_group="g", frame_digest="f")
    fields.update(updates)
    return EvidenceReceipt(**fields)


def test_stale_attempt_is_rejected():
    ledger = BoundedASCELLedger()
    ledger.request("e", "a", "c", 1)
    result = ledger.ingest(receipt(attempt_id="old"), 2)
    assert not result.effect_truth
    assert not result.active_attempt_credit
    assert result.reason == "stale_or_cross_attempt"


def test_current_attempt_truth_and_credit_are_separate_fields():
    ledger = BoundedASCELLedger()
    ledger.request("e", "a", "c", 1)
    first = ledger.ingest(receipt(), 2)
    assert first.reason == "accepted_pending_independence"
    result = ledger.ingest(receipt(receipt_id="r2", source_id="s2", source_version=1, capture_id="cap2", correlation_group="g2", frame_digest="f2"), 3)
    assert result.effect_truth and result.active_attempt_credit


def test_external_realization_establishes_truth_without_credit():
    ledger = BoundedASCELLedger()
    ledger.request("e", "a", "c", 1)
    result = ledger.ingest(receipt(attempt_id="external", realization_kind="external", active_attempt_credit=False), 2)
    assert result.effect_truth and not result.active_attempt_credit
