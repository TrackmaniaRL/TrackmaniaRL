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
