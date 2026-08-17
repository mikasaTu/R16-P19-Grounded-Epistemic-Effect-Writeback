from r16p19.phase5_audit_store import AuditStore


def test_audit_chain_detects_mutation():
    store = AuditStore(keep_rows=True)
    store.append({"kind": "A", "accepted": True})
    store.append({"kind": "B", "accepted": False})
    assert store.verify()
    store.rows[0]["kind"] = "X"
    assert not store.verify()
