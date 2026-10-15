# Reward function

This reference is checked against `trackmaniarl/trackmania/reward.py`,
`reward_step.py`, `reward_components.py`, `reward_config.py` and the environment
configuration. It describes the default library reward, not a reconstruction of
the V108 controller experiment. The optional `neighbors` wrapper changes action
selection and has no reward term.

`TrajectoryReward` is the built-in geometry-based time-trial objective used by
`OpenPlanetEnvironmentFactory`. It combines an explicit task reward with two
potential-based shaping terms. The configured geometry, telemetry units and
learner discount are part of the reward contract. Changing one without the
others changes the objective.

See [configuration](../readme/configuration.md) for the complete RunSpec hierarchy and
[algorithms](../readme/algorithms.md) for learner-specific constraints.

<p align="center">
  <img src="diagrams/reward-decomposition-preview.svg" alt="Trackmania reward decomposition and terminal potential reset" width="900">
</p>

[Editable diagram](diagrams/reward-decomposition.excalidraw) ·
[local preview](diagrams/reward-decomposition-preview.html)

## Reward equation

For one environment transition, the reported scalar is exactly

```text
r = r_time + r_progress-PBRS + r_direct-progress
  + r_projected-velocity + r_projected-speed + r_steering-delta
  + r_terminal + r_time-attack + r_collision + r_pace-PBRS
```

Every term is also exposed in the transition `info` mapping. A disabled term is
zero. The terminal task reward and terminal time-attack adjustment are separate
diagnostics. Neither is counted twice.

The two PBRS terms follow

$$F(s,a,s') = \gamma\Phi(s')-\Phi(s).$$

from [Policy invariance under reward transformations: Theory and application
to reward shaping](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf).
On every terminal reason, TrackmaniaRL sets the next potential to zero, so the
terminal shaping contribution is `-Phi(s)`. Reset establishes the initial
potential for the next episode. `components.environment.kwargs.config.reward_gamma`
must therefore equal `training.gamma`. RunSpec resolution rejects a mismatch
for the built-in environment.

## Components

Let `dt` be the elapsed race-clock time in seconds, capped by
`max_time_delta_s`. `L = max(1 m, path length)` the normalization length. `d` the accepted
monotonic progress. `v_parallel` the velocity projected on the local unit
tangent. And `p = d / L`.

| Component | Formula and units | Activation, sign and bound |
| --- | --- | --- |
| Time | `r_time = -time_penalty_per_second * dt` | Every transition with two valid timestamps. Non-positive, bounded per step by `-scale * max_time_delta_s`. |
| Direct progress | `r_direct-progress = progress_reward_full_lap * Delta(d) / L` | When accepted progress increases. Non-negative and sums to at most the configured full-lap scale before terminal completion adjustment. This is a task reward, not PBRS. |
| Progress PBRS | `Phi_progress = potential_progress_weight * p`, `F = gamma * Phi' - Phi` | Every non-terminal transition, terminal next potential is zero. Potential lies in `[0, potential_progress_weight]`. With `gamma < 1`, waiting at a fixed potential is slightly negative. |
| Projected velocity | `r_projected-velocity = projected_velocity_scale * v_parallel * dt` | Time-scaled. Signed, `v_parallel` is clipped to `[-max_projected_speed_mps, +max_projected_speed_mps]`. Missing velocity yields zero. |
| Positive projected speed | `r_projected-speed = projected_speed_bonus_scale * max(0, v_parallel / max_projected_speed_mps)^2 * dt` | Time-scaled, non-negative and bounded by `scale * dt`. Reversing receives no bonus. |
| Steering delta | `r_steering-delta = -steering_delta_penalty * abs(steer_t - steer_(t-1))` | Applied when steering is supplied. Steering is clipped to `[-1, 1]`, so the term lies in `[-2 * scale, 0]`. It is per decision, not time-normalized. |
| Finish/failure | `+finish_reward` on a valid finish, `-terminal_failure_penalty` on failure | Once, on termination. Failures are `off_track`, `time_limit`, `no_progress` and `slow_progress`. |
| Time attack | `bonus_scale * max(0, target_s - finish_s)^2 + linear_scale * (target_s - finish_s)` | Valid finish only. Lower-bounded by `-finish_reward`, it can reward beating the target and penalize missing it. Both scales require `time_attack_target_s`. |
| Collision | `-collision_penalty` | On a detected collision outside the race-clock cooldown. Non-positive. A collision inside the cooldown is still reported but is not penalized again. |
| Pace PBRS | `Phi_pace = -pace_reward_scale * clipped_time_debt`, `F = gamma * Phi' - Phi` | Requires a reference demonstration and race time. Terminal next potential is zero. The debt is clipped to `[-pace_debt_clip_s, +pace_debt_clip_s]`. |

Direct progress and progress PBRS are intentionally independent. Direct
progress changes the task objective. PBRS redistributes feedback while
preserving the discounted optimal-policy ordering under its assumptions. Set
`progress_reward_full_lap: 0.0` when an experiment needs shaping without the
extra task reward. Do not infer safe weights solely from their names: log the
component returns and compare their episode totals with `finish_reward`.

All fields below are stable built-in environment fields under
`components.environment.kwargs.config`. Every numeric value must be finite.

| Field | Type/default, range | Exact effect and tuning risk |
| --- | --- | --- |
| `crash_distance` | float `25.0` m, `>0` | Maximum local trajectory distance before `off_track`. Too small rejects valid corner cuts, too large admits another nearby branch. |
| `finish_progress` | float `0.995`, `(0,1]` | Minimum accepted metric-lap fraction for a valid finish signal. Lower values tolerate sparse finish geometry but admit larger shortcuts, `1.0` requires the projected path to reach its end. |
| `no_progress_steps` | int `200`, `>=1` | Terminal after this many decisions without a higher accepted index. It is cadence-dependent. |
| `slow_progress_window_steps` | int `80`, `>=2` | Length of the rolling metric-progress terminal window. |
| `minimum_progress_per_window_m` | float `2.0` m, `>=0` | Minimum progress required in that window. Zero disables slow-progress failure in practice. |
| `terminal_failure_penalty` | float `1.0`, `>=0` | Magnitude subtracted once on any natural failure terminal. |
| `collision_penalty` | float `0.05`, `>=0` | Magnitude subtracted for an eligible collision signal. |
| `collision_cooldown_s` | float `0.0` s, `>=0` | Minimum race-clock spacing between penalties. Large values hide repeated impacts, although detection is still counted. |
| `minimum_finish_steps` | int `50`, `>=1` | Earliest decision at which finish UI can terminate successfully. |
| `nearest_forward_points` | int `500`, `>=1` | Local forward projection window in geometry points. Too small loses fast movement, too large makes folded tracks more ambiguous. |
| `nearest_backward_points` | int `10`, `>=0` | Backward search allowance for localization only, accepted progress remains monotonic. |
| `limit_progress_by_kinematics` | bool `true` | Caps accepted reward progress by displacement and an elapsed-time speed bound. Disable only when a custom adapter intentionally teleports, this is separate from the feature-pipeline option with the same name. |
| `time_penalty_per_second` | float `0.1`, `>=0` | Negative reward per accepted race-clock second. |
| `max_time_delta_s` | float `1.0` s, `>0` | Per-step clock cap for time-scaled reward and the no-clock movement allowance. Too small undercounts real stalls, too large makes telemetry gaps dominate. |
| `maximum_race_time_s` | float/null `null`, `>0` | Optional `time_limit` failure terminal in physical seconds. |
| `progress_reward_full_lap` | float `10.0`, `>=0` | Total direct reward for one accepted metric lap. |
| `finish_reward` | float `30.0`, `>=0` | Base valid-finish terminal reward and magnitude floor for a negative time-attack adjustment. |
| `potential_progress_weight` | float `2.0`, `>=0` | Maximum progress potential. Zero disables progress PBRS. |
| `max_projected_speed_mps` | float `100.0` m/s, `>0` | Symmetric projected-velocity clip, positive-speed normalization and movement-speed cap. It must exceed plausible speed without legitimizing teleports. |
| `velocity_to_mps_scale` | float `1.0`, `>0` | Multiplier from native OpenPlanet velocity units to m/s. A unit error rescales velocity rewards by the same factor. |
| `projected_velocity_scale` | float `0.0`, `>=0` | Linear signed velocity reward per metre travelled along the local tangent. |
| `projected_speed_bonus_scale` | float `0.0`, `>=0` | Quadratic positive velocity-ratio bonus per second. |
| `steering_delta_penalty` | float `0.0`, `>=0` | Per-decision action smoothness cost, not time-normalized. |
| `time_attack_target_s` | float/null `null`, `>0` | Finish-time reference required by either time-attack scale. |
| `time_attack_bonus_scale` | float `0.0`, `>=0` | Quadratic reward for seconds faster than target, no quadratic penalty when slower. |
| `time_attack_linear_scale` | float `0.0`, `>=0` | Signed linear seconds-ahead term at finish. |
| `pace_reference_path` | path/null `null` | One explicit compatible demonstration. Requires `geometry_path`. |
| `pace_reward_scale` | float `0.0`, `>=0` | Potential magnitude per second of clipped time debt. Non-zero requires `pace_reference_path`. |
| `pace_debt_clip_s` | float `10.0` s, `>0` | Symmetric debt clip before pace potential construction. |
| `reward_gamma` | float `0.995`, `[0,1]` | Discount inside both PBRS terms, must equal `training.gamma`. |
| `use_racing_line` | bool `false` | Uses the geometry asset racing line instead of its reward centre for progress, tangent and pace projection. |

## Geometry and movement validation

The trajectory must contain at least two finite 3D points. Adjacent duplicates
and opposing neighbouring segments that produce a zero local tangent are
rejected when the geometry or reward is constructed.

Nearest-point search is local: it includes a bounded number of points behind
and ahead of the current monotonic index. Progress never decreases. When
`limit_progress_by_kinematics` is enabled, a candidate advance is also capped by
physical displacement and elapsed time. The bound applies to each step, not to
the lap total: a racing line cuts corners, so a fast lap drives less distance than
the centre line measures and a lap-total cap would stop it short of the finish.

```text
accepted_motion = min(position_displacement,
                      max_projected_speed_mps * time_budget)
advance        <= 2 * accepted_motion   (per step, from the current progress)
```

This prevents a stationary car near a later crossing or hairpin from catching
up over repeated calls and avoids using a global longest-segment allowance on
shorter parts of the map. If race time is absent, the conservative time budget
is `max_time_delta_s`.

`velocity_to_mps_scale` converts the OpenPlanet velocity field to metres per
second before the tangent dot product. The supported plugin already sends m/s, so
the built-in environment uses `1.0`.
Changing the telemetry source requires a measured unit conversion, not a reward
weight adjustment. Uneven geometry sampling affects index density but progress
percentage and direct progress use cumulative metric distance.

The valid-finish gate requires all of the following:

- the game finish UI is active.
- metric progress has reached `finish_progress`.
- at least `minimum_finish_steps` decisions have elapsed.
- the car is within `crash_distance` of the local trajectory.

These checks reject a finish signal at the start and most cross-track or
teleport shortcuts. Geometry still needs to follow the intended driving line
in order and at sufficient density.

## Time, cadence and termination

Race timestamps must be finite, non-negative and monotonic within an episode.
Identical timestamps produce `dt = 0`. A backward timestamp raises an error.
A large forward gap is capped by `max_time_delta_s` for all time-scaled terms
and for the reachable-progress allowance. Reset clears collision cooldown,
steering history, progress windows and both previous potentials.

Time, projected-velocity and projected-speed terms are approximately invariant
to a reasonable decision-cadence change because they integrate over `dt`.
Direct progress sums to the same amount for the same accepted path. Discounted
PBRS telescopes at a fixed discount contract, but keeping the same per-decision
gamma while changing cadence changes the physical-time discount and can change
the objective. Steering delta and step-count termination windows also change:
when changing `action_repeat_frames` or `decision_interval_ms`, convert the
desired real-time stall windows into new `no_progress_steps` and
`slow_progress_window_steps` values and re-evaluate the steering penalty.

Game or telemetry interruption is an environment truncation, not a natural MDP
terminal. Natural reward reasons set `terminated=True`. Replay bootstrapping
stops at a true terminal, while a truncation preserves the learner's configured
bootstrap semantics. N-step sampling never crosses either episode boundary.

## Human pace reference

`pace_reference_path` selects one concrete demonstration archive. TrackmaniaRL
does not search a directory or automatically choose the fastest lap. Record
and retain compatible complete laps, compare their `finish_time_s`, then point
the RunSpec at the fastest retained archive you deliberately selected.

The loader verifies:

- `map_uid` equals the geometry asset map UID.
- `geometry_sha256` equals the geometry asset hash.
- all frames and timestamps are finite.
- race times are strictly increasing.
- exactly the final frame has the finish flag.
- `finish_time_s` agrees with the final frame within 50 ms.
- the monotonic projection reaches the end of the trajectory.

Demonstration positions are projected monotonically onto the geometry. The
first time observed at each visited trajectory index is retained, missing
indices are linearly interpolated and the last profile value is set to the
recorded finish time. Optional speed values are converted with
`velocity_to_mps_scale`, interpolated and smoothed. The reward itself uses the
reference times.

At a non-terminal step,

```text
reference_time = profile[current_geometry_index]
time_debt = clip(race_time - reference_time,
                 -pace_debt_clip_s, +pace_debt_clip_s)
Phi_pace = -pace_reward_scale * time_debt
```

At a valid finish, the reference is the final profile time. On every terminal
reason, the next pace potential is zero. A negative debt means the agent is
ahead. A positive debt means it is behind.

## Configuration fragment

This is a fragment for
`components.environment.kwargs.config`, not a complete RunSpec:

```yaml
geometry_path: assets/my-map.geometry.npz
expected_map_uid: my-map-uid

reward_gamma: 0.995       # must equal training.gamma
time_penalty_per_second: 0.1
max_time_delta_s: 1.0
progress_reward_full_lap: 10.0
potential_progress_weight: 2.0
finish_reward: 30.0
terminal_failure_penalty: 1.0

max_projected_speed_mps: 100.0
velocity_to_mps_scale: 1.0
projected_velocity_scale: 0.0
projected_speed_bonus_scale: 0.0
steering_delta_penalty: 0.0

time_attack_target_s: 40.0
time_attack_bonus_scale: 0.1
time_attack_linear_scale: 0.2

collision_penalty: 0.05
collision_cooldown_s: 0.25

pace_reference_path: demonstrations/fastest-compatible-lap.npz
pace_reward_scale: 0.5
pace_debt_clip_s: 10.0
```

Pair it with:

```yaml
training:
  gamma: 0.995
```

Validate the complete file before touching the game:

```bash
uv run trackmaniarl validate run.yaml
uv run trackmaniarl track check --config run.yaml
uv run trackmaniarl smoke run.yaml --transitions 100
```

The last two commands are live gates and require Trackmania, the prepared map,
controller backend and [TrackmaniaRL Connect](https://openplanet.dev/plugin/sac_getdata).

## Why these terms and recovery on another map

The base reward combines a dense progress signal with an explicit incentive to
finish and a cost for elapsed race time. Defaults give 10 direct progress units
per complete lap, 30 for finishing and -0.1 per second. A valid finish fills any
small remaining accepted progress gap before the terminal terms are evaluated.
The progress potential adds short-horizon feedback. Optional velocity, pace and
time-attack terms are disabled by default. Their weights change the learning
problem and need measured ablations on the user's map.

There is **no separate positive reward for crashing or recovering**. A collision
may subtract its penalty while driving can continue. Recovery can still earn
progress, finish and time-attack reward, subject to the same stall and off-track
limits. Moving backward does not reduce accepted progress. It can receive
negative projected-velocity reward when enabled. Replay/recovery demonstrations
and an incident adapter change training data or policy, not this reward equation.

Collision input comes from the controller's detection capability (gamepad rumble
in the built-in path). A detected signal is not a precise count of wall impacts.
the keyboard backend cannot provide equivalent rumble evidence. Cooldown spaces
penalties in race seconds while preserving raw detections. A small wall contact
can be part of a fast line, so do not optimize the collision flag in isolation.

Start another map with optional terms disabled. Confirm metre/second units,
geometry localization, finish gating and the duration allowed for recovery.
Inspect **episode sums of each term**, including failure episodes. If a car
prefers quitting, compare the discounted finish incentive with accumulated time
and terminal-potential costs. If it farms local speed, reduce optional speed
reward and inspect monotonic progress. If it cannot recover, check the off-track
radius and stall windows before reducing collision penalties. Add one optional
term at a time and evaluate fixed complete trial budgets on the intended map.

For a 50 ms decision interval, 200 no-progress decisions are about 10 seconds,
provided telemetry and control keep pace. This is only an interpretation of a
step-based threshold, not a wall-clock guarantee. Long telemetry gaps are capped
for reward integration, but must remain visible in benchmark quality diagnostics.
