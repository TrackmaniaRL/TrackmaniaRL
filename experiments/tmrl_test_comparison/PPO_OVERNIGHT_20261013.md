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

## Actual launch

First queue `ppo-overnight-kl-s17-20261013` was deliberately stopped during startup
after identifying that Windows venv Python adds a forwarding ancestor to the deadline
guard. It could falsely count itself as a surviving workload at normal closure. No CP
or events were saved. All observed identities closed, mutex free,360pins and predecessor
actual SHA verified; STOP and new closure proof preserved. Old runtime1a148173 unchanged.
The correction excludes the guard's own forwarding ancestry and has a regression test
for normal COMPLETE as well as a hard-deadline test that preserves mismatched identities.

Fresh queue `ppo-overnight-kl-s17-20261013-r2`, run `tmrl-ppo-overnight-kl-s17-r2`,
runtime `ppo-overnight-runtime-20261013-r2` commit87832312. Actual start
2026-10-13T21:18:09.354949UTC; SAVE2026-10-14T01:08:09.354949UTC,
HARD2026-10-14T01:18:09.354949UTC. PreflightREADY10ms/33fields/59reads.
Independent guard ready, monitoring15min. W&B runrwoi180y. No driving improvement
claimed from launch or warmup. This is a fresh run; the stopped attempt is never resumed.

## KL result and next isolated hypothesis,2026-10-14 00:43UTC

KL trial COMPLETE204800/100, final0/10 versus midpoint9/10. Final failures were
two at41.28%, eight at65.59–65.73%, not the original81% hairpin. Accepted final
rollout KL never exceeded.0199611, yet driving collapsed;15 updates entirely skipped.
Reject this variant as a driving fix, no KL coefficient grid or seed29 confirmation.
FinalCP SHA0983f59f797592a117e443c8fdb60a8d6b258c2b7f2d70abb51ee406ee301a19.
Timing separatelyFAIL130ms, including120ms in another trial; eight failed at50ms max,
so timing alone cannot explain all failures. Final CPfinite/fingerprint/budget/processed,
both trace suites, actualSHA/sourcepins and all observed process closure/mutex verified.

The guard generated a post-COMPLETE STOP for a console auxiliary in its own subtree;
all identities are now closed and STOP preserved. New-source guard excludes its entire
subtree; a real Windows subprocess regression test confirms COMPLETE closes without STOP.
No used runtime edited. This cleanup issue happened after the completed evaluation.

Next trial: persistent recovery ONLY, original PPO updater. It tests the independently
supported original-baseline81% stall hypothesis, not a repair of the rejected KLvariant.
Saved baseline traces showed recovery gate disappearing during sustained stalls, while
shadow replay preserved it and left all10 successfulCP50 traces unchanged. Fresh seed17,
same budgets/settings, new runtime/queue/run/fingerprint, predecessor last completedKLCP
for provenance only; no weights resumed and no KL component combined. Rejection: invalid
measurements, final<8/10, median slowdown review or unresolved stalls. No promotion from
training finishes. Pin predecessor STOP actualSHA+mtime; a change stops the new trial.

Recovery trial actually launched2026-10-14T00:46:40.368328UTC, newqueue
`ppo-overnight-recovery-s17-20261013`, run `tmrl-ppo-overnight-recovery-s17`,
new runtime `ppo-overnight-recovery-runtime-20261014` commit1fd74717.
PreflightREADY10ms/33fields/60reads. SAVE04:36:40.368328UTC,
HARD04:46:40.368328UTC.20focusedtests and RuffPASS beforefreeze/launch;
newconfig scope test confirms originalPPOupdater, not combined withKL.
MonitorACTIVE15min; no driving outcome inferred from launch.

## Recovery seed17 result and independent confirmation

Recovery COMPLETE2026-10-14T04:05:10.843792UTC,204800/100,final10/10,
median51.835s,mean51.841s. Midpoint0/10 became10/10; stochastic training progress
must not replace deterministic evaluation. One140ms interval in trial5 out of10368
decisions makes timing strictFAIL; qualityPASS and fullBLOCKED remain separate.
CP100 actualSHA190f539038508e72d6e1e43e67473d57772966ea092ce1ed706423a6e2307215.
Finite/fingerprint/fullbudget/processed, both suites/traces/identity,360sourcepins,
predecessorSTOP+CP and baselineCPactualSHAs verified. All observed processes closed,
mutexfree, guard normalexit with reasonnull; no newSTOP. Proof preserved.

Next justified decision: fresh recovery seed29 with identical settings, rollout budget,
original updater and recovery thresholds, no resume or extra evaluation. This tests
whether finalquality and hairpin handling repeat under another initialization; it is
not causal proof, a timing waiver or multiweek readiness. Freshqueue/runtime/plan,
predecessor recoveryCP17 actualSHA and predecessorSTOP absence pinned (appearance wins).
Reject confirmation if<8/10, unresolved hairpin stalls or material time regression;
report timing separately. After this final bounded confirmation, report honestly and
pause the overnight automation; no new four-hour job may exceed09:04:23UTC window.

Actual seed29 attempt used new runtime `ppo-overnight-confirm-runtime-20261014-r2`
commit6ebf9e9d, queue `ppo-overnight-recovery-s29-20261013`, freshrun identity.
Preflight failed2026-10-14T04:13:56UTC at ready_before_reset: correctUID but plugin
confirm_ready rejected player_not_ready. No reset, telemetry reads, run directory,
training or optimizer updates. Closure/360pins/predecessorCP/STOP verified;
all observed processes closed and mutex free. Seed29 is NOT TESTED, not0/10.
No retry or readiness bypass. Final report and monitor pause:
[PPO_OVERNIGHT_REPORT_20261014.md](PPO_OVERNIGHT_REPORT_20261014.md).
