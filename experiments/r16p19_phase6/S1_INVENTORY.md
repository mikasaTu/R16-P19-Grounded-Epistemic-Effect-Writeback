# S1.0 Phase-5 frozen-substrate inventory

Phase-5 is read-only. Raw per-effect continuous scores are present, and the frozen checkpoint can be rerun on frozen episode features. Therefore S1 may continue.

- Raw per-effect scores saved: `true`
- Learned/oracle/support rows: `1680` / `4200` / `960`
- Calibration/formal episodes: `30` / `240`

| Path | Lines | Fields | SHA256 |
| --- | ---: | --- | --- |
| `experiments/r16p19_phase5/artifacts/results/verifier_checkpoint.npz` | binary | b1, b2, calibration_bias, calibration_scale, mean, model_type, std, threshold, w1, w2 | `edba3ff23ae71c5768760af161b601b3d1a9f65d37d1ddeb8304b2510e938005` |
| `experiments/r16p19_phase5/artifacts/results/verifier_checkpoint.metrics.json` | 207 | calibration_samples, formal_samples_accessed, models, qualification_samples, schema_version, selected, selected_qualified, training_samples | `495589fdf47f3909debfdd7d8a8aa7586759c85e17d12fc88b5c5118fae22463` |
| `experiments/r16p19_phase5/artifacts/results/learned_verifier_formal_results.jsonl` | 1680 | arm, clean_degradation, cluster_id, condition, effect_truth_recognized, false_current_attempt_credit, false_grounded_advance, formal_init, policy_seed, task_id, task_success, threshold, trajectory_sha256, verifier, verifier_scores | `7cd40fbd6ff84e6db49e5527b90b67e26f55dcbeca8e0b3013ffa28fca30d999` |
| `experiments/r16p19_phase5/artifacts/results/oracle_formal_results.jsonl` | 4200 | action_steps, active_attempt_credit, arm, backend_errors, clean_degradation, cluster_id, condition, cross_attempt_acceptance, effect_truth_recognized, false_current_attempt_credit, false_grounded_advance, first_divergence, formal_init, late_witness_acceptance, physical_policy_success, policy_seed, recovery_success, semantic_event_count, stale_acceptance, task_id, task_success, trajectory_sha256 | `595337aef08cdc14bb9bcd410ec91196e0dfd94a6182e20a49dfb8a46b37e9be` |
| `experiments/r16p19_phase5/artifacts/results/support_formal_results.jsonl` | 960 | action_steps, arm, backend_errors, cascade_false_negative, cascade_true_positive, cluster_id, condition, contact_count, formal_init, over_invalidation, policy_seed, prefix_sha256, recovery_executed, seed, task_id, task_success, under_invalidation | `747afaaad728a1a30f07bfcafbacdf1c602ed2592161857054320a86227f3208` |
| `experiments/r16p19_phase5/artifacts/results/verifier_dataset_manifest.json` | 19 | counts, episode_id_set_sha256, formal_access_before_freeze, schema_version, split_by_source_episode | `b8be18283d9d149c0a78ac80883aea57d17520cc25ab53f698915ff380f2edf8` |
| `experiments/r16p19_phase5/task_selection_contract.yaml` | 22 | — | `825979f9da9b3611bd35e6d3f160177f2073cd35e1c6e07cc5df30bc63331f72` |
| `experiments/r16p19_phase5/preregistration.yaml` | 53 | — | `959477d10371a43855ab96416c2d36121141296ff9f20e75579400e559bbfd41` |
| `experiments/r16p19_phase5/artifacts/raw_rollout_sha256.txt` | 525 | — | `63aa07a8cf6c7ae0cda5dd07f2dd92cd3baba84ae3ca79333743e38006121ce3` |

## Frozen split inventory

- `calibration`: 30 NPZ + 30 JSON; episode-id-set digest `4170ddae092d24888efd5fe821216adee9f9f60d8fb442ee066ffd2bb059a952`; NPZ-digest-set `c3c96447ed59af5bc0db15c5361392d1e0f3d63359fd98cbb544c8af82f4075d`.
- `formal`: 240 NPZ + 240 JSON; episode-id-set digest `f99da4d9eab51e5619268f76dcf93475075094bfd9cfe3de5ba027808a8600b9`; NPZ-digest-set `49622d2ba20f080469110995a40683a5c92c4a231b83a8dced725e63aa7776c1`.
