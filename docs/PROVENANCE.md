# Provenance and artifact map

## Frozen inputs

| Item | Identifier |
|---|---|
| Experiment source | `ae362efeba68643ab4dd2a99cfd295c72a9cbdcc` |
| LIBERO source | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| Stove demo HDF5 | `6b30906a52a5741e98ef447d27e7066d6c0be4a5f7acd7ecaf1cb7468aca4aa9` |
| Drawer demo HDF5 | `703950f48a3c49dfde61be489ade91527f16e1449b4f29a85f2e51153cef3638` |
| Stove BDDL | `491835cc2eb6956f7ce3d1ee4e266377116168fe453e6529fe105e3b0333635d` |
| Drawer BDDL | `5255fe54d7f25fad4dee8fa30a30033d8cb908f1708d953c91ab609264fb4fb8` |
| Stove init states | `8519d4638868ce661a20d331495d97e0521f8e1535479dd26ba875d5cc06b88f` |
| Drawer init states | `7eba1f68f9d3a553b14e99a437356fbcc91ba1a531ac2475fc858c1e9bcbe2fc` |

Dataset and official LIBERO binaries are not vendored. Paths and hashes are
preserved in both the frozen and runtime manifests.

## PAI run lineage

| Run ID | Job ID | Outcome | Role |
|---|---|---|---|
| `...-011500` | none | preflight sealed | checkpoint parent absent |
| `...-011800` | none | preflight sealed | output parent absent |
| `...-012000` | `dlc1rycl56e4nvac` | Failed | noninteractive config root cause |
| `...-013200` | `dlc6sr1fu466f1g9` | Succeeded | authoritative formal run |

The two preflight records never created PAI jobs. Their requested/resolved
contracts and sealed state are included for full workflow auditability.

## Artifact directories

- `artifacts/formal/.../experiment/`: exact preregistered scientific
  deliverables, including raw event and memory JSONL.
- `artifacts/formal/.../run.log`: training, W&B, actor-free and simulator
  competence stdout/stderr.
- `artifacts/formal/.../reserved_gpu1_dmon.log`: evidence that the second
  contract-reserved A800 stayed idle.
- `artifacts/checkpoints/tiny_state_bc_v1/actor/`: three complete PyTorch
  checkpoints and training metrics.
- `artifacts/failed/dlc1rycl56e4nvac/`: the failed first worker's immutable
  artifacts and traceback.
- `pai/control-plane/`: sanitized controller requests and PAI GetJob responses.
- `pai/controller-patches/`: patches adding the exact R16-P19 profile to the
  external job registry. The full unrelated registry is intentionally not
  copied.

## Credential boundary

No secret value is present. The controller records use `<redacted>` for
`WANDB_API_KEY`; only its environment variable name is retained. W&B local
binary caches and unrelated source-job snapshots are excluded because they are
not scientific deliverables. The W&B run URL and non-secret metadata remain.
