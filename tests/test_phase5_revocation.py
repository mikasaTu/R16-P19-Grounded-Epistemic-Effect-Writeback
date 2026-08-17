from r16p19.phase5_ledger_live import BoundedASCELLedger
from r16p19.phase5_types import EvidenceReceipt


def test_pre_revocation_receipt_is_rejected():
    ledger = BoundedASCELLedger()
    ledger.request("e", "a", "c", 1)
    ledger.revoke("e", 4)
    item = EvidenceReceipt("r", "a", "c", "e", "s", 1, 3, 10, "h", 1.0, True, True)
    result = ledger.ingest(item, 5)
    assert not result.accepted
    assert result.reason == "pre_revocation"
    assert not result.effect_truth
