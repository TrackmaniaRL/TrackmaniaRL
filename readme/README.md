# TrackmaniaRL documentation

![TrackmaniaRL logo](../docs/assets/trackmaniarl-logo.png)

Configuration and checkpoint schemas are 2.0. See the [changelog](../CHANGELOG.md)
for package versions.
These numbers describe different contracts. Historical benchmark and review pages
retain the versions and dates of the runs they describe.

## Start here

Choose a section by what you want to do. Some linked guides combine instructions,
reference material and explanations; they may appear in more than one section.

## Tutorials: learn by doing

- [Python quickstart](python-quickstart.md): run a game-free example, train from a
  configuration and plug in a model.
- [Game quickstart](quickstart.md): set up Trackmania, train on your own map and
  evaluate a checkpoint. Read the driver warning before installing on Windows.
- [Python examples](../examples/README.md): explore runnable examples after the quickstart.

## How-to guides: complete a task

- [Trackmania integration](trackmania.md): connect Openplanet, set up controls and
  prepare map geometry.
- [Distributed setup](distributed-setup.md): train with actors on multiple computers.
- [Migration and recovery](recovery.md): preserve checkpoints, journals and actor
  spools, diagnose resume failures and interpret telemetry coverage.
- [Native headless and the full QR profile](headless-full-qr.md): prepare a native
  deployment and choose a model. This mixed guide also describes the runtime contract
  and historical profile; it requires caller-supplied assets and references.
- [Vision](vision.md) and [vision BC](vision-bc.md): use camera observations or
  train with demonstrations.
- [Imitation learning](imitation-learning.md): record demonstrations and use BC,
  DAgger or recovery workflows.
- [Performance](performance.md), [observability](observability.md) and
  [troubleshooting](../docs/troubleshooting.md): investigate training, capture,
  telemetry, controller and device problems.
- [Recording](../docs/recording.md): capture optional video evidence of a benchmark.
- [Development](development.md): add components, run checks and prepare releases.

## Extending and maintaining the library

## Reference: look up details

- [Configuration](configuration.md): components, rollout and replay settings,
  execution device and evaluation rules.
- [Algorithms](algorithms.md): supported algorithms and model combinations. Generated
  `run.yaml` uses IQN/lidar, `run-ppo.yaml` uses PPO/telemetry, and
  `run-ALGORITHM-vision.yaml` selects camera observations for each RL family.
- [Public API](public-api.md) and [SDK guide](sdk.md): extension boundaries,
  component factories, model composition and checkpoint handling.
- [Observability](observability.md): metrics and incident signals.
- [Support matrix](../docs/support-status.md): supported paths and their contracts.
- [Repository layout](../docs/repository-layout.md): where to find the package,
  examples, tests and local experiment data.

## Explanations: understand how it works

- [Library data flow](../docs/library-architecture.md) and
  [runtime architecture](architecture.md): how observations become actions,
  transitions and learning batches.
- [Replay and sequences](replay-and-sequences.md): n-step targets, terminal masks,
  PER, recurrent history and fresh PPO rollouts. It also includes configuration recipes.
- [Reward construction](../docs/reward-function.md): how the Trackmania reward is built.
- [Vision](vision.md): capture, frame stacking, CNNs and PPO behavior.

## From setup to an evaluated policy

For a training run, `validate` performs a synthetic learner update and checkpoint
round trip; `smoke` exercises the game. Resume with a full training checkpoint as
explained in the [SDK guide](sdk.md). Evaluate with `benchmark`, keeping every
attempt, finish rate, finish times and telemetry quality.

Tests establish component and workflow behavior. Racing results require live
experiments. Camera capture uses a visible desktop viewport and does not provide
an atomic image/telemetry snapshot. PPO is supported locally, while distributed
learner/actor execution is an off-policy workflow.
