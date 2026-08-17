from r16p19.phase5_verifier_data import TASK_INDEX


def test_verifier_feature_contract_only_registers_frozen_tasks():
    assert TASK_INDEX == {0: 0, 5: 1, 9: 2}
