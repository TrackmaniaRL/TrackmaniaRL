# Autonomous shortcut learning from demonstrations

This is the standalone task for beating Jimmy's17715 ms on pinned TMX99162.
It replaces the source-guided residual proposal for this goal. The repaired
pretraining produced five autonomous native finishes of **17.715 s** from the
same deterministic start. This matches the demonstration; it does not beat the
record or establish recovery from disturbances. No online RL was started.

## What the policy actually controls

`AutonomousMacroShortcutFactory` exposes127 discrete motor programs. Each action
contains five native10 ms commands, so the policy still observes and decides at
50 ms while retaining precise control changes inside that interval. The
vocabulary includes all78 ordinary actions, the demonstrated five-tick patterns,
and alternative switch timings for observed command pairs. It is deduplicated
and lexicographically sorted, with no lap-time or action-order array.

At inference the policy selects an action solely from the current causal V5/V6
observation. The provider executes that selected program. There is no default
expert action, reference-time lookup, residual correction or automatic playback.
The factory rejects a `reference_controls` binding. The training-only teacher
labels are separate from the action vocabulary and deployment configuration.

The new127-action head has a different model/run identity. Existing78-action
policies must not be loaded into it. Fixed quantiles64, graph feature192,
dueling, batch256, replay1m, lr1e-4, gamma0.995 and the50 ms/150 s native
observation/event contract remain explicit. Throughput of this larger action
head has not been benchmarked; previous speed multipliers are not guarantees.

## Demonstrations are training data

One bounded native episode supplied355 real transitions and356 full prepared
observations. They include the actual rewards, terminal result and Goal17715 ms;
the last physical observation is17750 ms. No missing ghost telemetry, contacts
or terminal positions were filled in. That episode used external teacher labels
only for collection; the environment did not select its actions.

The non-pickle NPZ importer checks SHA, map binary, reward configuration/clock,
action-vocabulary SHA, full V5 shapes, finite data, chronological continuity and
the genuine final-interval Goal result. The replay store protects demonstration
transitions from FIFO eviction.355 steps are one episode, not355 independent
examples of successful laps.

The initial 5000 offline QR TD updates with demonstration cross-entropy did not
produce a finish: five evaluations stopped at 6.34% progress. Exact action
agreement on the recorded states was 53.52%, with class-balanced agreement 29.18%.

The repaired stage uses `BehaviorCloningPretrainer`, separate from the TD learner:

* cross-entropy over neutral quantile-mean action scores, matching greedy inference;
* inverse demonstrated-class frequency weights, so rare maneuvers retain weight;
* Adam at 3e-4, gradient clip 10, and the full 355-state dataset per update;
* no TD bootstrapping, rewards, next observations, or teacher control selection;
* an empirical fit gate: every state correct with a minimum action margin of 0.5
  on two checks 50 updates apart, followed by separate CPU and live native tests.

The measured repair initialized from the earlier learned weights and used **1850
additional supervised updates**, approximately 72.25 seconds of learner time.
All 355 recorded decisions then matched, including individual CPU inference.
This is a successful continuation of pretraining, not evidence that 1850 updates
from random initialization will suffice. The head's scores are imitation scores;
supervised fitting alone does not calibrate them as return quantiles.

An optional future phase is autonomous experience with exploration and TD learning.
The previously proposed online configuration has the following schedule:
   Online expert-mixture mass falls from25% to5% over500000 transitions; draws
   within pools are uniform (`alpha=beta=0`), using the incremental replay index.
   Expert supervision starts at one-tenth of its offline weight and reaches
   **zero** at250000 online transitions, allowing better actions to supersede
   Jimmy's labels. The uniform pool may also contain demos: mixture mass is not
   a claim of an exact observed demo fraction in every batch.

The inference YAML has no demonstration binding or offline updates. An earlier
untrained greedy policy was driven through the native environment with access
to expert sequences/datasets blocked; it chose its own actions and ended
`off_track`, rather than automatically completing Jimmy's lap. This verifies
independence, not learned driving quality.

The trained model was then evaluated with expert sequence, label and demonstration
file access blocked, using a policy-only export without replay or optimizer data.
The final complete evaluation finished 5/5 trials at actual native Goal 17715 ms
(the physical endpoint is 17750 ms). Independently recomputing all 1775 decisions
on CPU from the recorded live observation tensors and exported weights reproduced
every selected action, without teacher inputs or a step index. Weights stayed
unchanged during evaluation. These identical starts test repeatability, not five
independent trained models or generalization.

A previous evaluation controller failed after four finishes because its disk-quota
scan raced with atomic progress-file replacement. That failure remains preserved.
The repaired scan tolerates only vanished files; oversized quota, low free space,
and permission errors still fail. The complete five-trial result comes from a
separate controller using the same checkpoint, without retraining.

## Reward and record criterion

The separate shortcut trajectory guides reward progress. It does not alter
physical road geometry or provide future expert controls to the policy. This
task uses a25 m reward corridor, kinematic progress checks, real collision
events and a genuine native Goal. This is an explicit task configuration;
the ordinary factory's historical defaults remain unchanged.

The reward uses finish100, total progress50, time cost1/s and a linear final
time incentive10/s. Projected-speed and pace shaping are disabled. The final
time terms use the actual Goal event rather than its rounded50 ms physical
observation. A17714 ms finish is therefore0.011 reward better than17715 ms,
all other terms equal. Reward correction never manufactures a Finish or moves
the observed physics clock. `AutonomousMacroEvaluator` reports actual Goal time.

Success is **a genuine autonomous finish <=17714 ms**, with the matching model,
map and action-vocabulary hashes. Report best, median and finish rate separately.
Repeated identical starts check repeatability, not robustness or generalization.
Before describing a public ranking record, export and independently validate
the new ghost with correct metadata; a native Goal alone is not ranking
acceptance. The old placeholder-validator `IsValid=false` must not be relabelled.

## Private evidence and future launch

The task report directory `target18-setup1019/autonomous1032` contains:

* `train-autonomous1040.yaml`: the historical initial QR/offline/online proposal;
* `inference-autonomous1040.yaml`: the same action contract without demo access;
* the pinned macro vocabulary and native demonstration;
* `preflight1040.json`: actual data import, mixed replay batches, finite
  `(256,64,127)` forwards, guidance loss and unchanged random model weights;
* `pretraining1048`: supervised checkpoint and separate CPU readback/export;
* `evaluation1051`: complete autonomous evaluation and actual policy inputs;
* `pretraining-success1052.json`: checkpoint, decision recomputation and process
  closure audit, with empty remaining-owned, cleanup-error and forced lists.

Preparation used the separate immutable `shortcut-source1039` deployment.
No source-guided residual preset should be used for this task. The old
absolute-action campaign launcher intentionally rejects other factories; do not
weaken its guard. After a future explicit training instruction, use the general
SDK training path with the training YAML and its pinned `--demo` file, supervised
with a chosen wall-clock budget, process ownership, checkpoints and storage
reserves. The historical training YAML does not automatically route to the new
supervised helper; do not reuse its original offline stage as the repaired recipe.
Initialization for future TD learning must explicitly load the audited model
weights and synchronize the target, rather than treating the supervised checkpoint
as a resumable QR optimizer checkpoint. Three CPU actors and one CUDA learner are
a proposed starting allocation, not a measured optimum for the new head.
No online training duration has been scheduled or authorized by this pretraining test.

A trained policy now matches the demonstration autonomously. Beating the reference
remains a separate learning and validation objective.

### Online retention experiment

The first subsequent online attempt did not retain the supervised policy: none
of its evaluated policies finished the course. It was stopped explicitly by the
user, and its final checkpoint (315789 updates) and process closure were audited.
The numerical health of the learner was not evidence of successful driving.

A bounded probe on the actual demonstration reproduced 23 changed decisions out
of 355 after the original recipe's first TD update. An experimental alternative
uses `RetainedExpertObjective`, class-balanced imitation with a constant finite
coefficient, a constant 50% expert replay mixture, and learning rate `1e-5`.
With coefficient 128 the probe retained all demonstrated decisions at its checked
updates through 1000; coefficient 64 did not. The QR loss remains the standard
sum over 64 predicted quantiles. The coefficient is an experimental loss balance,
not a change to rewards, action masking, or inference-time expert control.

This probe is not autonomous native qualification or proof of improvement.
Require repeated full native finishes after online updates before starting a new
long attempt. Stop and report sustained evaluation failure independently of finite
losses, save the best evaluated policy, and preserve the supervised baseline.

### Multiple native demonstrations

`MultiDemonstrationShortcutFactory` accepts a hash-pinned manifest of distinct,
complete native laps. Each member must independently match the map, reward,
action vocabulary and actual Goal event. A downloaded ghost or its recorded
positions alone do not meet this contract. Approximate controls and unsuccessful
re-simulations must be reported separately rather than imported as full telemetry.

The current screening decoded all six available TMX99162 ghosts below20s.
Only Jimmy's17.715s and Aqusorias's18.995s laps completed in the checked native
re-simulation; the other four were excluded. Their735 transitions need132
unordered five-tick actions. Joint BC retained every demonstrated Jimmy decision
and reduced mistakes on Aqusorias's recorded states from174 to51 of380.
Five separate autonomous native evaluations then finished in17.715s each,
with expert file access denied. These are training-fit and deterministic-start
repeatability results, not a new record or a generalization claim.

`PaceAwareMixedSampler` keeps both complete laps available for TD updates while
weighting imitation toward the faster lap. It reads pace through replay's public
accessor so weights remain valid after protected demos move during FIFO eviction.
`ReleasingExpertObjective` can gradually reduce retention to a configured nonzero
floor using actual collected-transition counts. This is an experimental schedule;
its initial native qualification does not prove later-stage stability or improvement.
The regular environment, input preprocessing and historical reward defaults remain
unchanged. A new online attempt still requires independent qualification and closure.
