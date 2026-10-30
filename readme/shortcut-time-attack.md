# Preparing an 18-second shortcut policy

For the requested autonomous learned policy and record attempt, use
[Autonomous shortcut learning](shortcut-autonomous-learning.md). The residual
profile below is retained as a historical source-guided experiment, not the
selected training setup.

## Qualified source-guided profile

`trackmaniarl.experiments.shortcut_residual:SourceGuidedShortcutFactory` is a
separate experimental hybrid task. The supplied reference contributes literal
10 ms commands, while the policy chooses a correction every50 ms. Its78 actions
mean13 additive steering bins and six longitudinal modes: coast, brake,
first-tick brake, inherit, throttle, throttle plus first-tick brake. Action39
preserves the reference. These actions have a different meaning from ordinary
absolute action78; do not reuse a policy checkpoint or report them as equivalent.

Two bounded runs of the final deployment observed genuine native Goal at
**17.715 s**, with the final physical observation at17.750 s. The second also
performed finite full graph V6 / fixed-quantile inference on each real
observation, without selecting its actions or updating weights. These are
reference-controller results, not results of a trained residual policy.

The opt-in `reward_reference` binding supplies a hash-checked trajectory for
progress and corridor distance, separately from the unchanged physical road
geometry. `reference_controls` is independently hash-checked and bound to the
same map binary. Every native tick still needs real vehicle, adherence,
material/contact and event coverage. The original50 ms observations,150 s limit,
normal cleanup and real Goal guards are retained. No snapshots are restored.
The ordinary factory keeps its original reward trajectory and action meaning.

The environment records `actual_goal_race_time_ms` separately from
`observed_physics_time_ms`. Reward timing remains on the qualified physical
decision boundaries; do not call17.750 s the exact Goal timestamp.
`native_tick_actions` records the five-command plan; terminal collection still
uses actual captured physics rather than assuming all planned ticks executed.

After the expert commands end, the explicitly configured feed-forward base is
neutral. The policy can still command steering, throttle and braking until the
normal episode termination. This is a command policy, not extrapolated expert
data. A real late-braking trial crossed the reference end and terminated normally
without a Goal; stopping the car never grants completion.

The private `run-ready1029.yaml` contains the full QR setup with this new task
class path, fresh initialization, no offline updates and no checkpoint loading.
It is prepared for a future explicitly authorized training launch; no training
was started or qualified here. Use the general SDK actor/learner entry points
for this task. `SourceGuidedShortcutEvaluator` was also exercised through the
standard SDK evaluation API, reporting17.715 s from the actual Goal event. It
admits only the pinned map and geometry. The old absolute-action fresh-pilot launcher deliberately rejects
this factory and must not have its guard removed. Do not infer a throughput or
learned-policy success rate from these reference replays.

The remainder documents the earlier ordinary-action preparation and the
remaining requirements for a standalone model, without a reference controller.

Preparation only. No learner, offline pretraining, simulator or game is started
by `scripts.prepare_shortcut_reference`. This is an experimental task recipe,
not a qualified launch configuration.

## Target and evidence

The requested [TMX map 99162](https://trackmania.exchange/mapshow/99162) is
`tmrl-test`, UID `oqIJ5rQDRrNwLPTh9H2p_W4tLof`, matching the current test map.
Matching UID does not replace checking the exact map-file hash and game build.
The supplied private RE archive contains Jimmy's declared 17.715-second ghost
and its literal input export. The earlier [V7 reference experiment](../docs/headless-finish-r5.md)
observed a computed physical finish of 17715 ms with these 10 ms controls.
Its validator reported `IsValid=false`: it is not an accepted ranking ghost or
qualification of the current full V8 training environment.

Static preparation of the pinned export finds:

| Diagnostic | Result |
| --- | ---: |
| Recorded control changes | 68 |
| Changes not aligned to 50 ms decisions | 52 |
| Recorded 10 ms ticks changed by first-sample/50 ms hold | 126 / 1772 |
| Maximum 3D distance from the existing reward centerline | 23.718 m |
| Recorded poses farther than the configured 15 m crash distance | 15 / 355 |

This exposes two qualification questions: can a 20 Hz policy execute the
shortcut, and does the ordinary-route reward reject its trajectory? Geometric
distance alone is not a full reward replay. A changed command does not prove
that a 20 Hz shortcut is impossible. Do not claim either result without a real
native rollout.

The export contains recorded poses through 17700 ms and a declared finish at
17715 ms. Do not synthesize the final 15 ms, terminal observation, contact
events, wheel state, or native Goal. Poses are not restorable simulator states.

## Proposed implementation and qualification order

1. Pin the exact map, ghost, parser, controls and geometry. Keep private source
   assets outside the package. Prepare the source controls and pose corridor
   with the tool below; no training is needed for this step.
2. After authorization to run, qualify the original reference and a 50 ms
   candidate against the current full backend. Require real Goal, contacts,
   materials and normal cleanup. Never reuse the V7 EOF classification as V8
   terminal proof. Optimize timing within the 50 ms action space first.
3. Add a separate shortcut reward profile using the recorded reference route
   for progress and corridor distance. Keep road geometry as road geometry;
   do not move boundaries to make the shortcut appear paved. Test crossings,
   direction, checkpoint ordering and the genuine terminal event. Compare
   returns for the reference, an ordinary lap, a stationary agent and an
   incomplete shortcut. Keep the existing default profile unchanged.
4. Collect complete, aligned native observations and rewards along the
   qualified reference. The ghost pose export cannot be imported directly as
   full V5/V6 training transitions. Implement and qualify a separate
   demonstration-enabled task runner; the fresh-pilot coordinator deliberately
   rejects demonstrations and resume. Do not remove that guard.
5. Train a fresh full QR policy from expert supervision plus TD, followed by
   online refinement. Start with 3 CPU actors / one thread each and a CUDA
   float32 learner / two threads, graph V6 feature192, 78 actions, 64 fixed
   quantiles, dueling, batch256, replay1m, lr1e-4 and gamma0.995 at 50 ms.
   Keep the online update ratio0.25, target interval1000 and clip10. An initial
   proposed expert batch fraction is25%; this is a tuning choice, not a measured
   optimum. Retain expert data explicitly so FIFO eviction cannot remove it.
   Validate supervised-objective support before using the compiled learner.
6. Use neutral-start rollouts first. A later curriculum can hand over control
   after executing a genuine expert prefix from the start. Do not implement
   resets by assigning the ghost's position and velocity. Sampling from the
   middle requires a separate complete-state restoration qualification.
7. Evaluate without exploration or learner updates from the real start on the
   pinned map. Report actual Goal times, finish rate, median and best
   separately. The target is an observed valid finish <=18.000 s, not a reward
   threshold or projected lap time. Thirty identical deterministic starts
   check repeatability; perturbation tests require their own explicit protocol.

If 50 ms control cannot reproduce a competitive shortcut, a 10 ms profile is
a separate backend task. V8 graph frames currently enforce 50 ms. Reward
windows, recovery features, replay warmup and discounting all need time-domain
review; simply changing the driver step would invalidate the model contract.
Do not promise the previous throughput multiplier for that profile.

## Offline preparation

Run from the repository using the SDK's Python environment:

```powershell
python -B -m scripts.prepare_shortcut_reference `
  --archive "<private RE source archive.zip>" `
  --member "Trackmania2020RE/analysis/tmrl-shortcut/export-001/jimmy-17715.json" `
  --sha256 214b54f8b79c40af5db23375bd88ab9ff817d2de98315245de2a9c6792887777 `
  --map-uid oqIJ5rQDRrNwLPTh9H2p_W4tLof `
  --geometry "<pinned current geometry.npz>" `
  --output "<new private preparation directory>"
```

Outputs are two non-pickle NPZ reference files and `reference-audit.json`.
The tool refuses an existing output directory and never produces native
training transitions. Keep all outputs private.

The proposed supervised-plus-TD approach is inspired by
[DQfD](https://ojs.aaai.org/index.php/AAAI/article/view/11757). Existing SDK margin
and cross-entropy objectives are building blocks, not a complete DQfD pipeline.
No training-duration or 18-second success guarantee follows from that paper.
