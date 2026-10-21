# TrackmaniaRL documentation

![TrackmaniaRL logo](../docs/assets/trackmaniarl-logo.png)

Configuration and checkpoint schemas are 2.0. See the [changelog](../CHANGELOG.md)
for package versions.
These numbers describe different contracts. Historical benchmark and review pages
retain the versions and dates of the runs they describe.

## Start here

| Goal | Entry point |
| --- | --- |
| Use the library from Python | [Python quickstart](python-quickstart.md) |
| Connect the game and train an agent | [Game quickstart](quickstart.md) |
| Train with actors on multiple computers | [Distributed setup](distributed-setup.md) |
| Add your own components | [Public API](public-api.md), [examples](../examples/README.md) and [SDK reference](sdk.md) |
| Develop the library | [Development](development.md) and [repository layout](../docs/repository-layout.md) |

## From setup to an evaluated policy

1. Follow [installation and game setup](quickstart.md) and
   [Trackmania integration](trackmania.md) to connect Openplanet, configure control
   input, identify the map and build its geometry.
2. Choose an [algorithm and supported model combination](algorithms.md).
   Generated `run.yaml` uses IQN/lidar, `run-ppo.yaml` uses PPO/telemetry, and
   `run-ALGORITHM-vision.yaml` selects camera observations for each RL family.
   The [vision guide](vision.md) explains capture, frame stacking and CNNs.
   For supervised camera training use [vision BC](vision-bc.md).
3. Use the [configuration reference](configuration.md) to set components, rollout
   or replay settings, execution device and evaluation rules. Run `validate` for a
   synthetic learner update and checkpoint round trip, then `smoke` with the game.
4. Read [library data flow](../docs/library-architecture.md) and
   [runtime architecture](architecture.md) to understand how observations become
   actions, transitions and learning batches. [Replay and sequences](replay-and-sequences.md)
   explains n-step targets, terminal masks, PER, recurrent history and fresh PPO rollouts.
5. Inspect [reward construction](../docs/reward-function.md) and
   [metrics](observability.md) while training. Use [performance](performance.md)
   and [troubleshooting](../docs/troubleshooting.md) to distinguish learning problems
   from capture, telemetry, controller or device problems.
6. Resume with a full training checkpoint using the [checkpoint lifecycle](sdk.md).
   Evaluate with `benchmark`, keeping every attempt, finish rate, finish times and
   telemetry quality. [Recording](../docs/recording.md) explains optional video evidence.

## Extending and maintaining the library

The [Python SDK](sdk.md) documents public contracts, component factories, PyTrees,
model composition and checkpoint handling. [Imitation learning](imitation-learning.md)
covers demonstration recording, BC, DAgger and recovery workflows.
[Development](development.md) covers adding components, tests and release checks.
The [support matrix](../docs/support-status.md) describes supported paths and their contracts.

Tests establish component and workflow behavior. Racing results require live
experiments. Camera capture uses a visible desktop viewport and does not provide
an atomic image/telemetry snapshot. PPO is supported locally, while distributed
learner/actor execution is an off-policy workflow.
