# PPO recovery confirmation — final result, 14 October 2026

The fresh seed29 trial completed training and its planned final evaluation at
11:01:59 UTC (13:01:59 Warsaw). No new training, extra evaluation, checkpoint
selection or full campaign was launched.

| Final CP100 | Seed17 | Seed29 |
|---|---:|---:|
| Completed races | 10/10 | 9/10 |
| Median finish | 51.835 s | 51.980 s |
| Mean of finishes | 51.841 s | 52.131 s |
| Decisions | 10368 | 10297 |
| Intervals above 100 ms | 1 | 1 |
| Maximum interval | 140 ms | 180 ms |
| Quality gate | PASS | PASS |
| Timing / strict gate | FAIL | FAIL |

Seed29 midpoint was 0/10; final was 9/10 after the full 204800 transitions and
100 updates. The one final failure (trial8) ended for no_progress at79.75085%,
maximum interval50ms, without controller/telemetry errors. Recovery has not
eliminated every failure near the hairpin. Two seeds support the final quality
candidate; they do not establish causal benefit or stability over weeks.

## Timing evidence and next decision

Seed17 outlier: trial5 step818 at80.3966%, interval140ms, policy.act wall time
141.424ms, controller0.4395ms,20 skipped telemetry frames; race finished.
Seed29 outlier: trial5 step1028 at96.9556%, interval180ms, policy.act wall time
175.599ms, controller0.5723ms,25 skipped telemetry frames; race finished.
These two outliers are distinct from seed29's failed trial8. Skips are preserved
in the evidence; they are not silently reclassified as runtime errors.

Source evaluation.py:350–354 measures action latency around policy.act only;
environment.step and its telemetry wait follow that timer. Thus the observed
long call is in the action-selection wall-clock path. Existing traces cannot
separate preprocessing, CPU scheduling, GPU queue/synchronization or inference.
Do not claim an OS bug or pure telemetry delay. Before long training, add scoped
stage timings and CUDA event measurements in a new reviewed runtime/plan.
Do not tune PPO blindly or waive the strict100ms threshold. No automatic new job.

## Closure

All21 observed exact PID/creation-time identities, including wrapper and guard,
are closed; controller mutex free; guard normal exit reasonnull. Final checkpoint
finite/fingerprint/full budget/processed/fractional credit verified; both ten-trial
trace suites valid; final root evaluation equals CP100 phase evaluation.
Source/config/plugin/assets, predecessor CP and STOP pins verified; no active STOP.

Final CP100 SHA256:
`22b0709ff65e1a4ad1c1e93ae81cc7c08b0f5233955c90dea05f2ec246eaf0f4`.

[Closure evidence](evidence/ppo-confirm-day-20261014/recovery-s29-final-closure.json)
and [timing analysis](evidence/ppo-confirm-day-20261014/two-seed-timing-analysis.json).
Existing monitor is paused after completion; other automations are unchanged.
Full PPO remains BLOCKED. Campaign configs still use baseV5 and were not replaced
or repinned in place. KL variant remains rejected; SD-SAC remains experimental.
