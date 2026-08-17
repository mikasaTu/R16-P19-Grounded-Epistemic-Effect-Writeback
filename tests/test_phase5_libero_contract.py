from r16p19.phase5_libero_env import TASK_PROMPTS, TASKS, task_config


def test_frozen_official_task_prompts_exclude_scene_metadata():
    assert set(TASKS) == set(TASK_PROMPTS) == {0, 5, 9}
    for task_id, prompt in TASK_PROMPTS.items():
        assert task_config(task_id)["task"]["prompt"] == prompt
        assert "SCENE" not in prompt
        assert prompt == prompt.lower()
