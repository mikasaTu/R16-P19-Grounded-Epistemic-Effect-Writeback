import pytest


pytest.importorskip("mujoco")

from r16p19.phase4_executor import ExecutionMode, MacroSkillExecutor, qualify_executor
from r16p19.phase4_microenv import Phase4MicroEnv, TASK_CONTRACTS


@pytest.mark.parametrize("task_id", sorted(TASK_CONTRACTS))
def test_macro_executor_reaches_each_chain_effect_with_ordinary_steps(task_id):
    env = Phase4MicroEnv(task_id, 7)
    executor = MacroSkillExecutor(env)
    for effect_id in env.contract.chain_effects:
        result = executor.execute(task_id, effect_id, ExecutionMode.EXECUTE)
        assert result["action_steps"] == env.CONTROL_STEPS
        assert env.effect_truth(effect_id)
    assert env.all_chain_truth()
    assert env.backend_errors == []


def test_reverse_is_physical_and_changes_state_hash():
    env = Phase4MicroEnv("T2_PERSISTENT_SUPPORT", 11)
    executor = MacroSkillExecutor(env)
    executor.execute("T2_PERSISTENT_SUPPORT", "SUPPORT_PRESENT", ExecutionMode.EXECUTE)
    realized_hash = env.state_sha256()
    executor.execute("T2_PERSISTENT_SUPPORT", "SUPPORT_PRESENT", ExecutionMode.ROLLBACK)
    assert not env.effect_truth("SUPPORT_PRESENT")
    assert env.state_sha256() != realized_hash


def test_executor_qualification_helper_reports_exact_success_on_two_seeds():
    result = qualify_executor([40, 41])
    assert result["conditional_effect_success"] == 1.0
    assert result["full_chain_success"] == 1.0
    assert result["backend_error_count"] == 0
    assert result["pass"] is True
