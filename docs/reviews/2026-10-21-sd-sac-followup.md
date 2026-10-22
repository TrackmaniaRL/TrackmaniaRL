# SD-SAC follow-up, 2026-10-21

The entropy-matched variant remains experimental. No new Trackmania finish-rate
improvement is established by this follow-up. No game access, full training run,
checkpoint migration, production runtime edit or headless modification was made.

## Confirmed implementation correction

The old priority `abs((Q1 + Q2)/2 - y)` cancels opposing residuals. With `Q1=2`,
`Q2=-2`, `y=0`, it returned zero despite both squared errors being four.
`priority_td_error: mean_absolute` now returns two. `mean_value` explicitly
retains the historical rule and arithmetic for exact checkpoint continuation.
The checkpoint contract rejects a silent change. This is relevant to prioritized
replay, **not an explanation for the observed uniform-sampler pilot**.

New twin-residual metrics expose cancellation. Centered action disagreement and
mean-greedy twin agreement distinguish disagreement about action preferences
from a constant value offset. Neither proves critic calibration.

The entropy solver now stops widening a bracket when all rows already bracket
the target. An alternating local CPU benchmark of 128 x 78 values (one thread,
12 rounds, five calls per arm) measured median 14.119 ms before and 8.565 ms after,
1.65 times solver throughput, with bitwise equal temperatures and target logs.
This is a solver measurement, not a whole-training speedup or GPU claim.

## Recorded driving and bounded actor study

Source checkpoint: `distributed-update-00033856.pt`, SHA256
`f2b0d4aa942606c1b6c0c058942d7e4e0f816a909ea17733635bf38ab0d40f53`.
It and its saved optimizer were read only. Original model tensors were checked
unchanged. Earlier real driving yielded zero finishes in ten trials; four
entropy-matched trials consistently launched but stalled near 5.4% progress.

For this follow-up, reconstruct feature histories in order from all ten recorded
trials. Use the first occurrence of each actor variant for fitting and the second
for holdout. Hold out whole historical replay episodes separately (2,048 fit,
512 held replay rows). Sample every fifth live state and the final state. There
are 291 held live rows, 211 at step 80 or later.

Each arm starts from the same previously fitted `matched-17` actor and a copy of
the original saved Adam state, then takes 128 actor-only updates at batch 128.
This is an explicit fit experiment, **not exact resumed training**: the actor
copy's weights and historical optimizer are intentionally different histories.
Both arms omit the historical entropy anchor; the comparison isolates inclusion
of live observations at equal update count. The arms differ only in batches:
historical replay alone, or 50/50 historical replay and live-fit observations.
Both use the same frozen approximate mean critic and target entropy 0.8.
Two sampling seeds (17, 29), four disposable copies, 512 updates total.

| Sampling seed | Replay-only held live agreement | Mixed held live agreement | Replay-only held forward KL | Mixed held forward KL |
| --- | ---: | ---: | ---: | ---: |
| 17 | 64.60% | 87.63% | 0.3895 | 0.1817 |
| 29 | 69.42% | 83.51% | 0.3725 | 0.2119 |

Agreement is **with the frozen mean critic**, not correct driving. Deterministic
resets can make distinct held episodes nearly duplicates; this is not a new
independent rollout. Both critics chose the same best action in only 2.405% of
held live rows. Their mean absolute all-action disagreement was 0.03685, while
the mean top-two mean-Q gap was 0.002325. A value offset can contribute to that
absolute disagreement, so the ratio alone is not evidence of incorrect rankings.
Greedy disagreement is direct evidence that their preferred actions differ.

The result supports a coverage/actor-fit issue and the need to evaluate the
critics; it does **not** justify blindly fitting harder to their targets, adding
live traces to a production replay, or declaring the policy fixed.

## Gates before a long run

1. Keep the old model and logs immutable; retain historical priority mode if it
   must be exactly resumed. New options need a new experiment identity.
2. Use a bounded fresh joint actor/critic pilot with a fixed transition budget,
   fixed reward/observation/action contract and a separate held evaluation.
   Separate the temperature change from the backup-entropy intervention.
3. Report coverage over progress and terminal states, twin TD residuals, critic
   greedy agreement, actor agreement, clipping blockage, action distribution and
   actual finish/progress metrics. Actor/critic agreement alone is insufficient.
4. Reject a long campaign if the independent driving assessment still stalls.
   Qualify headless reset/input/observation parity before transferring this pilot.

Local evidence is retained under
`F:/AITrackmania/reports/sd-sac-followup-20261021/`: `audit.py`, `result.json`,
`audit.log`, the exact audited objective snapshot and `solver-benchmark.json`.
These local telemetry artifacts are not committed or uploaded. Unit tests cover
opposing-error priorities, explicit legacy/current checkpoint contracts, centered
rank disagreement and joint entropy-matched actor/critic learning with exact
next-update resume on a delayed-goal MDP. Synthetic success does not qualify
Trackmania driving.

Validation on Windows with Python 3.12 and CPU Torch 2.11.0: 211 SD-SAC tests
passed, including the joint-learning/resume case; the full repository suite
passed 1,261 tests with five skips in 295 seconds. Ruff formatting and lint passed;
strict mypy passed all 273 source files. An initial wider learning-test attempt
hit Windows permissions on the shared system pytest temporary folder; using a
fresh task-owned base directory resolved the fixture setup errors. Original
failed logs are retained. These are local checks; remote CI is a separate gate.

## Fresh pilot and early-stop accounting, 21–22 October

A fresh seed-17 pilot jointly trained the actor and critics with uniform replay,
target entropy 0.8, backup entropy coefficient 0.01 and countdown throttle enabled.
The operator stopped it early after approximately three hours. It collected
184,316 of 262,144 planned transitions, performed 43,060 updates and recorded
111 finishes in 202 training episodes. Its best training lap was 50.78 seconds;
the median among training finishes was 58.76 seconds. Nineteen of the last
twenty training episodes finished. These are exploratory training trajectories,
not independent greedy evaluation or a comparison with QR-DQN.

The saved checkpoint retained 519 earned updates. The accounting conservation
check passed (43,060 performed + 519 pending = 43,579 earned), but the final
drained-credit gate correctly blocked automatic benchmark qualification.
The local launcher's shared immediate-stop signal caused the undrained save;
the separate graceful-shutdown fix in PR #60 stops the actor first and then
drains accepted learner work. The original checkpoint and its pending credit
are preserved; no retrospective optimization or counter edits are permitted.

Checkpoint SHA256:
`f4fb037933d845ef4149fc7cdeccf6387e26a711c72c9392ab2c1e877898df0d`.
Its original run fingerprint and finite learner tensors passed verification.
All 285 frozen source/configuration hashes remained unchanged, all observed
owned processes closed, and the controller mutex was free after shutdown.
An optional subsequent assessment must use this exact saved policy without
training and be labelled as an early-stopped pilot assessment, not a completed
budget benchmark. SD-SAC remains experimental and excluded from the standard
multi-seed campaign; this pilot does not establish competitive driving quality.
