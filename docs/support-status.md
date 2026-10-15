# Support status for 1.2.12

Tests verify contracts and failure handling. They do not prove fast driving on
unseen maps. The [algorithm matrix](../readme/algorithms.md) is authoritative for
supported learner/replay/model combinations.

| Surface | Status and purpose |
| --- | --- |
| RunSpec 2.0, typed contracts, component resolution | Public SDK |
| `init`, `inspect-config`, `validate` | Supported project setup and offline validation |
| `track check`, boundary/trajectory recording, geometry building, `track from-ghost` | Own-map asset workflow; ghost corridors approximate the driven line rather than physical track walls |
| `train`, `resume`, `smoke`, `learner`, `actor` | Supported asynchronous off-policy workflow |
| PPO `train`, `resume`, `smoke`, `benchmark` | Supported local on-policy workflow with telemetry and CNN vision factories. Use local commands, not distributed actor/learner commands |
| SAC, REDQ, TQC, stable discrete SAC | Supported telemetry and camera actor-critic workflows with generated configurations and first-party model factories |
| RGB vision pipeline, frame stacking, CNN encoder | Supported library components, optional desktop capture via `[vision]`, see [vision setup](../readme/vision.md) |
| `benchmark`, optional window recording | Supported checkpoint evaluation, Windows-only recorder |
| `track record-demo`, `offline-pretrain`, `bc-train`, `bc-benchmark` | Supported demonstration and imitation workflows, subject to documented data contracts |
| `track record-recovery` | Guarded, opt-in human intervention collection, requires an attentive human |
| `recovery-finetune` | Supported incident-gated graph adapter training with held-out selection and policy-only checkpoints, not an arbitrary-model trainer |
| `demo-benchmark`, `diagnose expert` | Diagnostic expert replay/policy comparison, distinct from learned-policy benchmarking |
| `dagger-collect` | Supported teacher-labelled student collection, episode archives and subsequent BC training |
| Trajectory tracking/stitching/synthetic recovery/optimisation | Supported map-specific tools. Configure timing and provenance explicitly. Synthetic labels and search results require live evaluation |
| Graph IQN V1–V6, residual GRU, Mamba | Optional model families whose graph layers are required implementation dependencies, not public aliases |
| W&B | Optional projection of authoritative local logs, no account required for default workflow |
| Gemini/Optuna orchestration | Optional experiment scheduling, no speed or reliability guarantee |
| `experiments/sub37/` | Repository-only single-map benchmark reproduction, never a default policy |
| `scripts/` | Maintainer distribution checks, soak verification and analysis, not public Python API |

Historical failed experiments and forensic reports stay in `experiments/` and
`docs/benchmarks/`. Local checkpoints, recordings, databases and generated logs
remain ignored and are not distributed. Preserve them separately if reproducing
an old result. No local training data was deleted for this release cleanup.

Only the documented RunSpec and checkpoint schemas are accepted. Other versions
and private CLI helper aliases are rejected.
Version numbers belonging to other formats (for example JSONL events or geometry)
are independent contracts, not reasons to remove working data support.
