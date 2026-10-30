# Full QR headless measurements

These measurements use the same pinned map, full V5/V6 observations, historical
rewards and 50 ms/150 s native contract. They are local measurements, not promises
for other hardware/maps. The learner used CUDA float32 on an RTX 4090; CPU actors
and the learner used two Torch threads each.

Subsequent PR review found that the per-tick provider restarted brake taps on
each of the five physical ticks. In these recorded runs, the 26 tap action IDs
therefore behaved as full-brake actions. The provider now preserves a single
10 ms pulse per 50 ms decision, but the measurements below predate that fix.
They remain measurements of the preserved deployment, not qualification or
performance evidence for the corrected 78-action behavior. The same review
also fixed omitted reward components and charged-collision counts in actor
metrics; historical component logs must not be interpreted as measured zeros.

| Measurement | Actors | Useful simulation rate |
|---|---:|---:|
| Earlier baseline, two-seed short trials | 2 | about 3.56x real time |
| Final checkpoint-path qualification, two-seed median | 3 | 4.234x |
| Same qualification, two-seed median | 4 | 4.113x |
| Completed fresh seven-hour campaign | 3 | 4.500x |

Final short-trial ranges were 4.192–4.276x for three actors and 4.032–4.193x for
four. Three were preferred within the five-percent near-tie rule. Larger-actor
tests did not justify forcing more workers. Two short repetitions do not
establish statistical certainty or seven-hour stability for every configuration.

Useful rate is `min(transitions/s, updates/s / 0.25) * 0.05`. It counts the
learner budget, not actor simulation that the learner cannot consume. Its
reference is real simulation time, not simultaneous Windows-client training.
Short windows and complete campaigns are different scopes; their percentage
difference is not a controlled comparison of time to a good policy.

## Completed seven-hour campaign

The fresh run completed 25,200.003385115 seconds on `CLOCK_MONOTONIC_RAW`, with
2,278,223 accepted transitions, 567,052 updates and 3,346 training episodes.
It loaded no prior model, replay or demonstrations. All four workers exited
zero. Independent audit read 444 finite online/target/optimizer tensors and
checked 3,368 exact owned identities, with no survivors or forced cleanup.
Final checkpoint SHA-256:
`c11002e46cec56789e8552aa59f49e15f7957963561ea2e9eedede0d5eb13f87`.

The final CPU policy completed 30/30 inference-only repetitions of the same
map and deterministic start. Each actual native Goal time was 35.162 seconds;
its observed physical boundary was 35.200 seconds. These are repeatability tests,
not independent randomized trials or generalization evidence. Evaluation made
zero learner updates and closed all native sessions. The three retained native
archives passed independent SHA readback.

Campaign results describe the preserved laboratory deployment. Package relocation
or type cleanup alone does not inherit its source qualification: the packaged
profile requires its own source/reference and bounded native gates. Raw private
evidence and licensed native assets are not included in the wheel.

## What profiling changed

The initial real-batch CUDA trace showed substantial small-kernel launch
overhead. On matched actual batches, encoder CUDA Graphs reduced the isolated
update median from 28.136 ms to 15.027 ms while preserving checked model, target,
Adam, RNG, losses, gradients and priorities. This 1.87x figure concerns the
isolated update operator, not total training.

A full-production CPU profile sampled device transfer in 236/843 learner-main
observations. Both sampled actors spent their 999-sample windows waiting for
update credit. RPC threads were mostly idle. These observations did not justify
removing validation or adding actors. The existing pinned-transfer protocol was
tested separately and not selected by the production comparison.

An existing asynchronous checkpoint writer still held the GIL while serializing
large NumPy protocol-2 byte strings. The final storage-backed implementation
reduced the measured longest pause from 8.181 s to 0.077 s and save time from
31.066 s to 12.050 s on the inspected full replay snapshot. These are checkpoint
measurements, not a whole-run speed multiplier.

Virtual motor rendering remains 150 FPS and does not reproduce Windows jitter
bitwise. The known 1800/1750 ms right-turn difference remains; no unmeasured
channels, missing time or Finish events are filled in.

## Subsequent corrected-action short trials

Later bounded trials used the corrected 78-action behavior, three CPU actors
with one Torch thread each, and a CUDA float32 learner with two threads. Both
profiles retained batch 256, 64 quantiles, a million-transition replay capacity,
10,000 warmup transitions and 0.25 updates per accepted transition. Each measured
run had at least 5,000 updates and 240 seconds of learning; the steady window was
the final 180 seconds. Replay occupancy was about 40,000, not one million.

| Observation/model profile | Steady useful rate | Whole bounded-run rate |
|---|---:|---:|
| V5/V6 graph/lidar, two fresh seeds | 5.974x | 3.957x |
| Lossless base3 ribbon CNN, two fresh seeds | 7.275x | 4.850x |

Whole-run rates include learner setup, compiler warmup, replay warmup and the
final checkpoint. The image profile uses schematic road ribbons, not game RGB;
these rows compare neither equivalent sensors nor convergence. They are not a
repeat of the historical seven-hour campaign or a Windows-client benchmark.
Independent readback checked 446 finite learner-state tensors per graph
checkpoint and 52 per ribbon checkpoint, matching counters and fingerprints,
with owned processes closed. Native physics and reward guards were unchanged.

Profiling found distinct costs: blocking batch transfer in the graph learner
and scalar prioritized-replay feedback in the image learner. An ordered bulk
feedback candidate preserved all six replay trees, next sample IDs, weights and
RNG bitwise and ran 2.129x faster in an isolated CPU test. Four fresh whole-image
trials in A/B/B/A order nevertheless measured a 4.42% whole-run and 5.29% steady
regression. The candidate was removed; its isolated result is not a training
speedup.

An additional graph transfer-prefetch comparison stopped at the storage reserve
guard before both paired repetitions completed. Its checkpoint was finite and
processes closed, but that incomplete trial supplies no selected speedup.
Transfer prefetch remains optional and off in the selected preset. Increasing
actors, moving data to a RAM disk or changing WSL limits is not supported by
these measurements alone.

## Matched-recipe representation screening (26 October 2026)

Six fresh QR runs compared the full V5/V6 telemetry/geometry profile against
the schematic road-ribbon CNN on the same pinned map. Each had a 1,200-second
budget from learner readiness, including replay warmup and checkpoint pauses;
compiler setup preceded readiness and final checkpoint writing followed the
budget. Both used three CPU actors (one Torch thread each), a CUDA float32
learner (two threads), Uniform sampling without transfer prefetch, and the same
exploration, clipping, target-update interval and QR recipe. Observation content
and model architectures (192 versus 256 features) remained different.

| Profile | Training seed | Evaluation finishes | Native Goal time | Progress | Accepted transitions | Updates |
|---|---:|---:|---:|---:|---:|---:|
| V5/V6 | 17 | 2/2 | 53.851 s | 100% | 128,762 | 29,655 |
| Schematic CNN | 17 | 0/2 | — | 40.13% | 146,255 | 34,063 |
| V5/V6 | 0 | 2/2 | 70.493 s | 100% | 80,062 | 17,417 |
| Schematic CNN | 0 | 0/2 | — | 40.50% | 138,173 | 32,043 |
| V5/V6 | 29 | 2/2 | 56.952 s | 100% | 125,020 | 28,753 |
| Schematic CNN | 29 | 0/2 | — | 0.46% | 132,652 | 30,663 |

The two evaluations per checkpoint repeated the same deterministic reset and
produced identical results; they are repeatability checks, not independent
samples. Schematic seeds 17 and 0 ended with `no_progress`, seed 29 with
`slow_progress`. No finish time is assigned to those policies.

Telemetry seed 0 overlapped user-reported gaming and is excluded from controlled
throughput and equal-load comparisons, leaving two clean matched seed pairs.
Its original supervisor failure is preserved: a post-budget progress-age check
misclassified checkpoint finalization, although all workers exited successfully
and independent checkpoint readback passed. A documented, case-specific audit
allowed evaluation without resuming training. Its missing wall-time and RTF
measurements remain unavailable.

For the two clean pairs, accepted simulation time divided by measured training
wall time (including finalization, excluding compiler setup) was 5.249x/5.110x
for V5/V6 and 5.975x/5.454x for the schematic CNN. These are aggregate collection
rates during training relative to simulated real time, not Windows-client
speedups or time-to-good-model measurements.

This short screening supports retaining V5/V6 as the conservative default and
the schematic CNN as experimental. Faster collection did not yield a finishing
schematic policy within this budget. It establishes neither general convergence
superiority nor game-RGB transfer, cross-map performance or full replay occupancy.
