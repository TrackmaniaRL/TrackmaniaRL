# PPO overnight result,14 October2026

The isolated persistent recovery candidate finished10/10 final trials with median
51.835s and mean51.841s. This is promising driving evidence on one seed, not proven
causality, repeatability or readiness for weeks of training. One140ms decision means
strict timing qualification still FAIL. Full training remains BLOCKED.

| Trial | Midpoint50 | Final100 | Timing | Decision |
|---|---|---|---|---|
| Historical original PPO seed17 |10/10,median51.885s|7/10,median63.14s|130ms midpoint,120ms final|Regression at81% hairpin|
| Transactional KL seed17 |9/10,median64.84s|0/10|140ms midpoint,130ms final|Reject as driving fix|
| Persistent recovery seed17 |0/10|10/10,median51.835s|60ms midpoint,140ms final|Quality candidate; strict timing FAIL|
| Recovery seed29 confirmation |Not run|Not run|No measurements|Preflight player_not_ready;0updates|

The KL trial accepted rollout KL<=.0199611 and skipped15 updates entirely, yet final
driving failed: two stalls41.28%, eight65.59–65.73%. A local empirical KL bound does
not guarantee preservation of driving. No coefficient grid or failed-variant confirmation.

Recovery retains previously detected incident memory only while nearly stationary
at an edge; original PPO updater, shared model/GNN/reward and other settings unchanged.
All successful originalCP50 saved traces had unchanged features under the shadow
control. Fresh training changed its trajectory, so one good final seed does not by
itself isolate which behavior improved. No original81% final stalls in its ten trials.
Its midpoint0/10 also shows why sampled training reward or intermediate results alone
cannot qualify the final policy. Do not select an intermediate checkpoint instead.

## Final artifacts and closure

BASE=`H:/Studia/inzynierskie/inzynierkav2/AITrackmania/artifacts/tmrl-test-comparison`.
Recovery17 CP100: `BASE/tmrl-ppo-overnight-recovery-s17/checkpoints/update-00000100.pt`.
ActualSHA190f539038508e72d6e1e43e67473d57772966ea092ce1ed706423a6e2307215.
Full204800transitions/100updates/processed/fractional0, finite/fingerprint verified.
Both ten-trial suites have matched UID, CPidentity/SHA, counts and complete raw traces.
Final10368decisions; trial5 has max140ms, other trials satisfy100ms maximum.
Skipped telemetry frames are recorded in the original trials and closure proof.

Recovery17 queue `ppo-overnight-recovery-s17-20261013` completed04:05:10UTC.
New post-closure-verification.json confirms all observed exactPID+ctime closed,
mutexfree,360sourcepins/config/plugin/assets, predecessorSTOP+CP and historical
baseline actualCP SHA unchanged. Independent guard exited normally, reasonnull,
no newSTOP. Copies of closure evidence are in `evidence/ppo-overnight-20261013`.
Used frozen runtimes, helpers, oldCP/replay/plans/STOP were preserved.

## Why the independent confirmation did not start

Seed29 queue `ppo-overnight-recovery-s29-20261013`, runtimecommit6ebf9e9d,
actualattempt04:13:53UTC. CorrectmapUID passed, but confirm_ready returned
player_not_ready at ready_before_reset. The reference plugin checks GUIPlayer,
ScriptAPI and ViewingPlayerState; the response does not identify which is absent.
A finish-screen state after ten successful trials is plausible, not directly observed.
The readiness check precedes reset, and was retained. Zero telemetry reads, resets,
training/optimizer updates; the run directory was never created. This is a preflight
block, not a failed driving test or another0/10. Source and priorCP verified, all
observed processes closed, mutexfree, newSTOP and closure proof preserved.

Do not rerun or resume this used queue. Restore a playable race state on the correct
map before a future separately authorized fresh confirmation. Do not remove readiness
checks, modify signed plugins, bypass desktop security or waive timing qualification.
If this finish-to-next-run lifecycle recurs, first add direct non-control diagnostics
for the missing readiness state, then review a guarded normal reset design.

The overnight loop ended at this actionable environment block. Only its heartbeat
is paused; no other automation is changed. No full campaign or new algorithm launched.
