# Train with actors on multiple computers

One learner can train an off-policy model while multiple Trackmania actors collect
rollouts asynchronously on separate computers. This workflow supports the
distributed algorithms in the [algorithm matrix](algorithms.md), including IQN
and SAC. PPO uses [local on-policy training](vision.md).

## Prepare every machine

1. Use Python 3.12 and the same TrackmaniaRL version on the learner and actors.
   The generated Trackmania project includes the `distributed` extra; other
   projects must install `trackmaniarl[distributed]`.
2. Copy the same `run.yaml`, custom model/feature code, map, geometry and any
   configured pace-reference assets to every machine. Asset directories may be
   in different locations, but their contents and the resolved run fingerprint
   must match. Run `uv run trackmaniarl validate run.yaml` on each machine.
3. On each actor machine, complete the [game and map setup](quickstart.md), open
   the configured map and run `uv run trackmaniarl track check --config run.yaml`.
   The learner needs the matching configuration and assets but does not need a
   running game.
4. Generate one token and save the same value in an ignored `.env` file next to
   `run.yaml` on every machine. Keep it private and use at least 32 characters:

   ```powershell
   uv run python -c "print(__import__('secrets').token_urlsafe(32))"
   ```

   Each `.env` file needs `TRACKMANIARL_DISTRIBUTED_TOKEN=YOUR_TOKEN` unless
   `distributed.token_env` in the RunSpec names a different variable.

## Start the learner

On the training computer, run:

```powershell
uv run trackmaniarl learner run.yaml --bind 127.0.0.1:8787
```

The learner owns replay, model updates, policy snapshots and checkpoints. It
accepts only a loopback bind. Port `8787` is the default; choose another port
consistently if it is occupied.

## Connect and start each actor

From **each actor computer**, open a separate terminal and forward its local
port 8787 to the learner through SSH:

```powershell
ssh -N -L 8787:127.0.0.1:8787 USER@LEARNER_HOST
```

The learner host must run an SSH server reachable from the actor. Keep this
terminal open. A WireGuard or equivalent encrypted tunnel is also suitable;
the gRPC bearer token authenticates requests but does not encrypt traffic.

In another terminal on each actor computer, run the actor with a unique, stable
ID (for example `desktop-a` and `desktop-b`):

```powershell
uv run trackmaniarl actor run.yaml --connect 127.0.0.1:8787 --actor-id desktop-a
```

Each actor drives its own local game, persists rollout chunks before sending
them, and periodically pulls the learner's latest policy. Run the same commands
with a different actor ID on each additional computer.

## If an actor cannot connect

- `UNAVAILABLE`: check the SSH tunnel, learner process and matching ports.
- `UNAUTHENTICATED`: check that every machine reads the same token from its
  environment or `.env` beside `run.yaml`.
- `FAILED_PRECONDITION` with a fingerprint mismatch: compare TrackmaniaRL and
  custom code versions, the RunSpec, geometry and pace-reference contents.

See [runtime architecture](architecture.md) for rollout durability and policy
publication, and [configuration](configuration.md#distributed) for refresh,
heartbeat, chunk and actor execution settings.
