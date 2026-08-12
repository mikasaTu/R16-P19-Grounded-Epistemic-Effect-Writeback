"""Clean actor competence gate and paired memory-driven LIBERO rollouts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch

from .actor import (
    MAX_EFFECTS,
    RetrievalAugmentedActor,
    ideal_memory_summary,
    memory_summary_from_states,
)
from .config import ACTION_HORIZON, EVAL_INIT_INDICES, TASKS, TRAIN_EPISODES
from .memory import MemoryArm
from .ontology import load_ontology
from .simulator import (
    deterministic_target_effect,
    effect_truths,
    load_init_states,
    make_env,
    padded_flat_state,
    reset_to_state,
)
from .types import Decision, EpistemicState, Event, EventType, EvidenceReceipt


KEY_ARMS = ("B2", "B3", "B5", "B6")
CLOSED_LOOP_CONDITIONS = ("C0", "C1", "C2", "C3", "C7")


class PhaseScriptActor:
    """Fallback primitive: replay effect-segment actions from the nearest demo."""

    def __init__(self, labels: Iterable[object]):
        lookup = {(label.task_key, label.episode_id): label for label in labels}
        self.entries: Dict[int, List[dict]] = {0: [], 1: []}
        for task_index, task_key in enumerate(TASKS):
            task = TASKS[task_key]
            with h5py.File(str(task.dataset_path), "r") as handle:
                for episode in TRAIN_EPISODES:
                    label = lookup[(task_key, episode)]
                    demo = handle["data"][episode]
                    states = np.asarray(demo["states"], dtype=np.float32)
                    actions = np.asarray(demo["actions"], dtype=np.float32)
                    transitions = dict(label.stable_transition_indices)
                    transitions.update(
                        {
                            key: value
                            for key, value in label.transition_indices.items()
                            if key not in transitions
                        }
                    )
                    boundaries = [0] + [
                        int(transitions.get(value, len(actions) - 1)) for value in task.effects
                    ]
                    boundaries[-1] = min(len(actions), boundaries[-1] + 8)
                    segments = []
                    for phase in range(MAX_EFFECTS):
                        start = max(0, boundaries[phase] - (4 if phase else 0))
                        stop = max(start + 1, min(len(actions), boundaries[phase + 1] + 4))
                        segments.append(actions[start:stop].copy())
                    self.entries[task_index].append(
                        {
                            "episode": episode,
                            "initial": padded_flat_state(states[0]),
                            "segments": segments,
                        }
                    )
        initials = np.stack(
            [entry["initial"] for values in self.entries.values() for entry in values]
        )
        self.std = np.maximum(initials.std(axis=0), 1e-3)
        self.entry = None
        self.cursors = [0] * MAX_EFFECTS

    def reset(self, state=None, task_index=0) -> None:
        if state is None:
            self.entry = None
            return
        state = padded_flat_state(state)
        choices = self.entries[task_index]
        distance = [np.mean(((entry["initial"] - state) / self.std) ** 2) for entry in choices]
        self.entry = choices[int(np.argmin(distance))]
        self.cursors = [0] * MAX_EFFECTS

    def action_chunk(self, state, task_index, phase, memory_summary):
        del state, task_index, memory_summary
        segment = self.entry["segments"][phase]
        cursor = self.cursors[phase]
        indices = np.minimum(np.arange(cursor, cursor + ACTION_HORIZON), len(segment) - 1)
        self.cursors[phase] = cursor + ACTION_HORIZON
        return segment[indices].copy()


def _reset_actor(actor, state, task_index):
    try:
        actor.reset(state, task_index)
    except TypeError:
        actor.reset()


def run_competence_gate(actor, actor_name: str, max_steps: int = 600) -> Tuple[List[dict], dict]:
    rows: List[dict] = []
    task_keys = list(TASKS)
    for task_index, task_key in enumerate(task_keys):
        task = TASKS[task_key]
        env = make_env(task, camera_obs=False)
        initial_states = load_init_states(task)
        try:
            for seed_index in EVAL_INIT_INDICES:
                observation = reset_to_state(env, initial_states[seed_index])
                for _ in range(5):
                    observation, _, _, _ = env.step(np.zeros((7,), dtype=np.float32))
                _reset_actor(actor, env.get_sim_state(), task_index)
                reached = [False] * MAX_EFFECTS
                phase = 0
                steps = 0
                while steps < max_steps and phase < MAX_EFFECTS:
                    chunk = actor.action_chunk(
                        env.get_sim_state(), task_index, phase, ideal_memory_summary(phase)
                    )
                    effect_reached = False
                    for action in chunk:
                        observation, _, _, _ = env.step(np.asarray(action, dtype=np.float64))
                        steps += 1
                        truths = effect_truths(env, task)
                        if truths[task.effects[phase]]:
                            reached[phase] = True
                            phase += 1
                            effect_reached = True
                            break
                        if steps >= max_steps:
                            break
                    if not effect_reached and isinstance(actor, PhaseScriptActor):
                        # Restart a scripted phase once so a retry is not an infinite final-action loop.
                        actor.cursors[phase] = 0
                rows.append(
                    {
                        "record_type": "actor_competence",
                        "actor": actor_name,
                        "task_key": task_key,
                        "init_index": int(seed_index),
                        "effect_success": {
                            effect: bool(reached[index]) for index, effect in enumerate(task.effects)
                        },
                        "full_task_success": bool(env.check_success() and all(reached)),
                        "action_steps": steps,
                    }
                )
                print(
                    "COMPETENCE_ROLLOUT actor=%s task=%s init=%d effects=%s full=%s steps=%d"
                    % (actor_name, task_key, seed_index, reached, rows[-1]["full_task_success"], steps),
                    flush=True,
                )
        finally:
            env.close()
    per_effect = {}
    for task_key, task in TASKS.items():
        subset = [row for row in rows if row["task_key"] == task_key]
        per_effect[task_key] = {
            effect: float(np.mean([row["effect_success"][effect] for row in subset]))
            for effect in task.effects
        }
    minimum = min(value for task in per_effect.values() for value in task.values())
    summary = {
        "actor": actor_name,
        "rollout_count": len(rows),
        "per_effect_success": per_effect,
        "min_per_effect_success": minimum,
        "full_task_success_rate": float(np.mean([row["full_task_success"] for row in rows])),
        "threshold": 0.80,
        "pass": minimum >= 0.80,
    }
    return rows, summary


def _camera_array(observation: Mapping[str, object], sensor: str) -> np.ndarray:
    candidates = {
        "agentview": ("agentview_image", "agentview_rgb"),
        "robot0_eye_in_hand": ("robot0_eye_in_hand_image", "eye_in_hand_rgb"),
    }[sensor]
    for key in candidates:
        if key in observation:
            return np.asarray(observation[key])
    raise KeyError("camera %s missing; observation keys=%r" % (sensor, sorted(observation)))


class LiveEventFactory:
    def __init__(self, task_key: str, init_index: int, condition: str, arm: str):
        self.episode_id = "%s:init_%02d:%s:%s" % (task_key, init_index, condition, arm)
        self.index = 0

    def make(
        self,
        event_type: EventType,
        effect: str,
        parent_ids: Sequence[str] = (),
        observation: Optional[Mapping[str, object]] = None,
        sensor: str = "agentview",
        payload: Optional[Mapping[str, object]] = None,
    ) -> Event:
        event_id = "%s:e%05d:%s" % (self.episode_id, self.index, event_type.value)
        receipt = None
        if event_type in (
            EventType.OBSERVE_POSITIVE,
            EventType.VERIFY_POSITIVE,
            EventType.REALIZATION_WITNESS,
            EventType.OBSERVE_NEGATIVE,
            EventType.CONTRADICTION,
        ):
            if observation is None:
                raise ValueError("live physical event lacks observation")
            digest = hashlib.sha256(_camera_array(observation, sensor).tobytes()).hexdigest()
            receipt = EvidenceReceipt(
                evidence_id=event_id + ":receipt",
                episode_id=self.episode_id,
                event_index=self.index,
                timestamp=self.index / 20.0,
                sensor_identity=sensor,
                frame_digest=digest,
                effect_id=effect,
                evidence_type=event_type.value,
            )
        event = Event(
            event_id=event_id,
            episode_id=self.episode_id,
            event_index=self.index,
            timestamp=self.index / 20.0,
            event_type=event_type,
            effect_id=effect,
            parent_ids=tuple(parent_ids),
            receipt=receipt,
            payload=dict(payload or {}),
        )
        self.index += 1
        return event


def _process_positive(factory, arm, effect, command_id, observation):
    observe = factory.make(
        EventType.OBSERVE_POSITIVE, effect, (command_id,), observation, "agentview"
    )
    arm.process(observe, effect)
    verify = factory.make(
        EventType.VERIFY_POSITIVE, effect, (observe.event_id,), observation, "robot0_eye_in_hand"
    )
    arm.process(verify, effect)
    witness = factory.make(
        EventType.REALIZATION_WITNESS,
        effect,
        (command_id, verify.event_id),
        observation,
        "agentview",
    )
    return arm.process(witness, effect)


def _arm_memory_summary(arm: MemoryArm) -> np.ndarray:
    return memory_summary_from_states([arm.records[effect].state for effect in arm.effects])


def run_closed_loop(actor, actor_name: str, max_steps: int = 700) -> List[dict]:
    ontology = load_ontology()
    rows: List[dict] = []
    task_keys = list(TASKS)
    for task_index, task_key in enumerate(task_keys):
        task = TASKS[task_key]
        initial_states = load_init_states(task)
        for condition in CLOSED_LOOP_CONDITIONS:
            for arm_name in KEY_ARMS:
                env = make_env(task, camera_obs=True)
                try:
                    for init_index in EVAL_INIT_INDICES:
                        observation = reset_to_state(env, initial_states[init_index])
                        for _ in range(5):
                            observation, _, _, _ = env.step(np.zeros((7,), dtype=np.float32))
                        _reset_actor(actor, env.get_sim_state(), task_index)
                        arm = MemoryArm(arm_name, task_key, ontology)
                        factory = LiveEventFactory(task_key, init_index, condition, arm_name)
                        target_phase = deterministic_target_effect(
                            task_key, "init_%02d" % init_index, condition
                        )
                        delay_chunks = 2 + (init_index % 3)
                        phase = 0
                        steps = 0
                        attempts = [0] * MAX_EFFECTS
                        chunks_in_phase = [0] * MAX_EFFECTS
                        command_ids: List[Optional[str]] = [None] * MAX_EFFECTS
                        need_command = True
                        fault_consumed = False
                        reversed_once = False
                        pre_effect_state = None
                        premature = 0
                        retries = 0
                        recovery_success = False
                        safe_stop = False
                        repeated_loop = False
                        reached = [False] * MAX_EFFECTS
                        while steps < max_steps and phase < MAX_EFFECTS and not safe_stop:
                            effect = task.effects[phase]
                            if need_command:
                                request = factory.make(EventType.REQUEST, effect)
                                arm.process(request, effect)
                                if condition == "C7" and phase == target_phase and not fault_consumed:
                                    imagined = factory.make(
                                        EventType.IMAGINE,
                                        effect,
                                        (request.event_id,),
                                        payload={"predicted_success": True},
                                    )
                                    arm.process(imagined, effect)
                                command = factory.make(
                                    EventType.COMMAND,
                                    effect,
                                    (request.event_id,),
                                    payload={"attempt": attempts[phase]},
                                )
                                arm.process(command, effect)
                                command_ids[phase] = command.event_id
                                attempts[phase] += 1
                                need_command = False
                                if condition == "C3" and phase == target_phase and not reversed_once:
                                    pre_effect_state = np.asarray(env.get_sim_state()).copy()

                            suppress = False
                            if phase == target_phase and not fault_consumed:
                                if condition in ("C1", "C7"):
                                    suppress = True
                                elif condition == "C2" and chunks_in_phase[phase] < delay_chunks:
                                    suppress = True
                            chunk = actor.action_chunk(
                                env.get_sim_state(), task_index, phase, _arm_memory_summary(arm)
                            )
                            chunks_in_phase[phase] += 1
                            truth_observation = None
                            for action in chunk:
                                executed = np.zeros((7,), dtype=np.float64) if suppress else action
                                observation, _, _, _ = env.step(np.asarray(executed, dtype=np.float64))
                                steps += 1
                                if effect_truths(env, task)[effect]:
                                    truth_observation = observation
                                    reached[phase] = True
                                    break
                                if steps >= max_steps:
                                    break

                            if suppress and condition in ("C1", "C7"):
                                negative = factory.make(
                                    EventType.OBSERVE_NEGATIVE,
                                    effect,
                                    (command_ids[phase],),
                                    observation,
                                    "agentview",
                                    {"fault": condition},
                                )
                                arm.process(negative, effect)
                                timeout = factory.make(
                                    EventType.TIMEOUT, effect, (negative.event_id,), payload={"fault": condition}
                                )
                                decision = arm.process(timeout, effect)
                                fault_consumed = True
                            elif truth_observation is not None:
                                decision = _process_positive(
                                    factory, arm, effect, command_ids[phase], truth_observation
                                )
                            else:
                                last_event = list(arm.ledger.events.values())[-1]
                                decision = arm.decide(effect, last_event)

                            if (
                                condition == "C3"
                                and phase == target_phase
                                and truth_observation is not None
                                and not reversed_once
                            ):
                                observation = reset_to_state(env, pre_effect_state)
                                contradiction = factory.make(
                                    EventType.CONTRADICTION,
                                    effect,
                                    (list(arm.ledger.events)[-1],),
                                    observation,
                                    "agentview",
                                    {"fault": "post_realization_reversal"},
                                )
                                decision = arm.process(contradiction, effect)
                                reversed_once = True
                                fault_consumed = True

                            physical_now = effect_truths(env, task)[effect]
                            if decision == Decision.ADVANCE_TO_NEXT_SUBTASK:
                                if not physical_now:
                                    premature += 1
                                # C3 must physically realize before the registered reversal is injected.
                                if condition == "C3" and phase == target_phase and not reversed_once:
                                    continue
                                phase += 1
                                need_command = True
                            elif decision in (
                                Decision.RETRY_CURRENT_EFFECT,
                                Decision.ROLLBACK_OR_REPLAN,
                            ):
                                retries += 1
                                need_command = True
                                if fault_consumed:
                                    recovery_success = True
                                if isinstance(actor, RetrievalAugmentedActor):
                                    actor.reset()
                            elif decision == Decision.SAFE_STOP:
                                safe_stop = True
                            if chunks_in_phase[min(phase, MAX_EFFECTS - 1)] > 30:
                                repeated_loop = True
                                safe_stop = True
                        task_success = bool(env.check_success() and all(reached))
                        row = {
                            "record_type": "closed_loop",
                            "actor": actor_name,
                            "task_key": task_key,
                            "init_index": int(init_index),
                            "condition": condition,
                            "arm": arm_name,
                            "target_effect": task.effects[target_phase],
                            "task_success": task_success,
                            "effects_reached": {
                                effect: reached[index] for index, effect in enumerate(task.effects)
                            },
                            "premature_subtask_transitions": premature,
                            "false_completion": premature > 0,
                            "repeated_action_loop": repeated_loop,
                            "recovery_success": bool(recovery_success and task_success),
                            "retry_count": retries,
                            "action_steps": steps,
                            "safe_stop": safe_stop,
                            "resident_slot_count_max": arm.ledger.max_resident_seen,
                            "dangling_parent_count": arm.ledger.dangling_parent_count(),
                        }
                        rows.append(row)
                        print(
                            "CLOSED_LOOP task=%s init=%d condition=%s arm=%s success=%s premature=%d retries=%d steps=%d"
                            % (
                                task_key,
                                init_index,
                                condition,
                                arm_name,
                                task_success,
                                premature,
                                retries,
                                steps,
                            ),
                            flush=True,
                        )
                finally:
                    env.close()
    return rows


def paired_bootstrap(rows: Sequence[Mapping[str, object]], repetitions=10000, seed=1619) -> dict:
    pairs = []
    for task_key in TASKS:
        for init_index in EVAL_INIT_INDICES:
            values = {}
            for arm in ("B2", "B3", "B6"):
                subset = [
                    row
                    for row in rows
                    if row["task_key"] == task_key
                    and row["init_index"] == init_index
                    and row["condition"] in ("C1", "C3", "C7")
                    and row["arm"] == arm
                ]
                values[arm] = float(np.mean([row["task_success"] for row in subset]))
            pairs.append({"unit": "%s:init_%02d" % (task_key, init_index), **values})
    rng = np.random.RandomState(seed)
    b6_b3 = np.asarray([row["B6"] - row["B3"] for row in pairs], dtype=np.float64)
    b6_b2 = np.asarray([row["B6"] - row["B2"] for row in pairs], dtype=np.float64)

    def interval(values):
        sample = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
        return {
            "estimate": float(values.mean()),
            "ci95": [float(np.percentile(sample, 2.5)), float(np.percentile(sample, 97.5))],
        }

    return {
        "paired_unit": "task_and_init_index",
        "repetitions": repetitions,
        "seed": seed,
        "unit_count": len(pairs),
        "B6_minus_B3_faulted_success": interval(b6_b3),
        "B6_minus_B2_faulted_success": interval(b6_b2),
    }


def summarize_closed_loop(rows: Sequence[Mapping[str, object]], bootstrap: Mapping[str, object]) -> dict:
    arms = {}
    for arm in KEY_ARMS:
        subset = [row for row in rows if row["arm"] == arm]
        clean = [row for row in subset if row["condition"] == "C0"]
        faulted = [row for row in subset if row["condition"] != "C0"]
        target_faults = [row for row in subset if row["condition"] in ("C1", "C3", "C7")]
        arms[arm] = {
            "rollout_count": len(subset),
            "full_task_success": float(np.mean([row["task_success"] for row in subset])),
            "clean_task_success": float(np.mean([row["task_success"] for row in clean])),
            "faulted_task_success": float(np.mean([row["task_success"] for row in faulted])),
            "target_fault_task_success": float(np.mean([row["task_success"] for row in target_faults])),
            "false_completion_rate": float(np.mean([row["false_completion"] for row in subset])),
            "premature_transition_rate": float(
                np.mean([int(row["premature_subtask_transitions"] > 0) for row in subset])
            ),
            "repeated_action_loop_rate": float(np.mean([row["repeated_action_loop"] for row in subset])),
            "recovery_success_rate": float(np.mean([row["recovery_success"] for row in target_faults])),
            "mean_retries": float(np.mean([row["retry_count"] for row in subset])),
            "mean_action_steps": float(np.mean([row["action_steps"] for row in subset])),
        }
    b3_false = arms["B3"]["false_completion_rate"]
    relative_reduction = (
        (b3_false - arms["B6"]["false_completion_rate"]) / b3_false if b3_false else 0.0
    )
    b6_c3 = [row for row in rows if row["arm"] == "B6" and row["condition"] == "C3"]
    contradiction_recovery = float(np.mean([row["recovery_success"] for row in b6_c3]))
    clean_degradation = arms["B3"]["clean_task_success"] - arms["B6"]["clean_task_success"]
    ci_b3 = bootstrap["B6_minus_B3_faulted_success"]["ci95"]
    ci_b2 = bootstrap["B6_minus_B2_faulted_success"]["ci95"]
    gates = {
        "B6_false_completion_relative_reduction_at_least_0_50": relative_reduction >= 0.50,
        "B6_contradiction_recovery_at_least_0_80": contradiction_recovery >= 0.80,
        "B6_fault_success_exceeds_B3": (
            arms["B6"]["target_fault_task_success"] > arms["B3"]["target_fault_task_success"]
        ),
        "B6_fault_success_exceeds_B5": (
            arms["B6"]["target_fault_task_success"] > arms["B5"]["target_fault_task_success"]
        ),
        "paired_bootstrap_B6_minus_B3_lower_gt_0": ci_b3[0] > 0.0,
        "clean_degradation_at_most_0_03": clean_degradation <= 0.03,
        "B6_statistically_distinguishable_from_B2": ci_b2[0] > 0.0 or ci_b2[1] < 0.0,
    }
    return {
        "arms": arms,
        "B6_false_completion_relative_reduction_vs_B3": relative_reduction,
        "B6_contradiction_recovery_recall": contradiction_recovery,
        "clean_success_degradation_vs_B3": clean_degradation,
        "behavior_gates": gates,
        "behavior_gate_pass": all(gates.values()),
    }
