# Authorized overnight PPO repair

Human authorization: autonomous repairs and fresh bounded pilots for 12 hours from
2026-10-13 21:04:23 UTC, ending 2026-10-14 09:04:23 UTC. Success is not guaranteed.
Reuse the existing PPO heartbeat every 15 minutes; remain quiet during normal progress.
No automatic full campaign, SD-SAC, shared reward/model/GNN edits or used-run restart.

## First decision: transactional KL guard only

The last 50 baseline updates exceeded target KL after the epoch in 43/50 cases.
The first candidate accepts an epoch only if full-rollout empirical KL remains within
the existing .02 target; otherwise model, Adam, scaler and Torch RNG roll back together.
Everything else remains the trend baseline: fresh seed17, 204800 transitions/100 updates,
8192000 LR horizon, ten-trial evaluations after50 and100. This tests update size control,
not a claim that empirical KL guarantees better driving. Record rejected epochs and
skipped updates; reject if learning stalls, final driving regresses or measurements fail.

Existing baseline: midpoint10/10, final7/10, stalls around81%, median51.885→63.14s.
Predecessor final CP100 actual SHA a5927a3187bc2533b99dfdca7a336181dd0bd1f0416aab0ced3aa491c5aa5e60.
Preserve all old checkpoints, traces, STOP files and closure proofs.

## Follow-up decisions

Only after complete closure and written outcome may a new candidate start. If the KL
guard preserves driving but the hairpin still stalls, test the independently prepared
persistent-recovery variant, keeping the original updater. It preserves previously
observed incident memory only while nearly stationary at the edge; it does not teach
an escape maneuver. Do not combine variants before isolated evidence. Use seed29 as
confirmation if an isolated variant succeeds; do not repeat identical failures or grid
search coefficients. Do not begin a four-hour trial with less than four hours plus
closure margin left. At the overnight deadline preserve, close, report and pause monitor.

## Measurement and launch contract

Fresh detached runtime from the exact published source commit, new queue/run and plan,
actual pins, config fingerprint, predecessor SHA, validate-only, one controller mutex,
CUDA, verified foreground/unlockeddesktop/correct UID. Every trial has14400s actual-start
cap, SAVE at13800s and HARD at14400s, no extension. STOP and source mismatch win.
Only confirmed owned PID+creationtime may be closed; capture wrapper and descendants.

The new trial prospectively reports timing separately: a complete, correctly bound
midpoint with one timing outlier can finish the planned trend experiment. It remains
a strict timing FAIL. Missing trials, errors, wrong UID/CP, bad traces or interruptions
remain fatal. Final quality>=8/10 and maxinterval<=100ms are separate strict gates;
no waiver or retroactive reclassification of earlier results. Final CP100 is mandatory,
no checkpoint selection. Compare finish count, median time and hairpin failures against
both midpoint and the preserved baseline. Timing-only failure does not prove model failure.

Operational ledger: BASE/ppo-overnight-20261013/repair-state.json. Current trial uses
pilot_tools.ppo_overnight prepare/verify/launch with an explicit variant and seed.
All source changes only in the editable checkout; no edits to frozen or main H.
