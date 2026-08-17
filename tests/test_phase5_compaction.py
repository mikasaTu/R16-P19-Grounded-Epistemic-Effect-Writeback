from r16p19.phase5_ledger_live import BoundedASCELLedger
from r16p19.phase5_types import EvidenceReceipt


def test_receipts_and_active_attempts_are_bounded():
    ledger = BoundedASCELLedger(active_capacity=4, receipts_per_source=4)
    for index in range(20):
        effect = f"e{index}"
        ledger.request(effect, f"a{index}", f"c{index}", index * 2)
        ledger.ingest(EvidenceReceipt(f"r{index}", f"a{index}", f"c{index}", effect, "sensor", index, index * 2 + 1, index * 2 + 2, "h", 1.0, True, True, capture_id=f"cap{index}", correlation_group=f"g{index}", frame_digest=f"f{index}"), index * 2 + 1)
    assert len(ledger.active) == 4
    assert sum(len(bucket) for bucket in ledger.hot_receipts.values()) <= 16
    assert len(ledger.recent_effects) <= 4
