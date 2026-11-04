# Offline policy stability diagnostics and future evaluation

This protocol is separate from the active seed-17 campaign. Do not alter its
training, evaluation, checkpoint choice or reported result. These diagnostics
cannot establish that an algorithmic change fixes IQN instability.

## Paired state analysis

At a predeclared schedule of accepted-transition counts, preserve complete
checkpoints and record their fingerprints. On an isolated copy, collect a fixed,
versioned set of representative observations, including their preprocessing,
map and simulation identifiers. Include states from successes and failures;
do not select a checkpoint based on the resulting score. Inference for the two
checkpoints must use the same action ordering, preprocessing, quantile
estimation procedure and RNG controls; repeat stochastic estimates over fixed
seeds. Record the policy and return-generation settings, discount, checkpoint
and code hashes alongside the input JSON. Collect discounted returns only from
rollouts starting at the matched state, taking the reported selected action and
following the corresponding policy, with the same reward and terminal rules.
If a matched return is unavailable, omit it rather than using zero.

The analyzer is a standalone **offline analysis of supplied scores**, not an
instrumentation hook or score collector:

```console
python -m tools.headless_campaign_queue.policy_stability paired.json report.json
```

It refuses to overwrite the report. Input example:

```json
{
  "actions": ["left", "straight", "right"],
  "observations": [
    {"id": "state-001", "before": [1.0, 1.1, 0.9],
     "after": [1.2, 1.0, 0.9], "before_return": 0.8, "after_return": 0.7}
  ]
}
```

Each row pairs the same observation for two checkpoints and contains finite
expected action values. The output summarizes action changes, top-two margins,
ties, score changes and, when supplied, selected Q minus discounted return.
The latter is a descriptive error, not a proof of calibration: a single rollout
may be noisy, and off-policy values require matching future policy and reward
definitions. A tie is resolved by the first action in the declared order;
multiple ties can make a change rate sensitive to this convention. The tool does
not verify provenance or produce scores from models. Preserve the input, output,
collection script, fingerprints and configuration for reproducibility.

The prior isolated IQN investigation produced aligned tensors using
`/home/kuba/tmrl-iqn-causal-debug-20261102-001/grid_probe.py`: its `build_run`
constructs a CPU learner and loads each checkpoint's learner state, while
`grid_values` evaluates expected action values on the same `replay-probe.pt`
observations. It writes `grid-values-<checkpoint>.pt` dictionaries. Its paths,
checkpoint set and runtime directories are hard-coded to that investigation;
inspect and adapt a copy in a new isolated directory before use. Never point its
output or native work directory at an active run. CPU inference and checkpoint
loading consume resources; run only when they will not compete with training.

For grids already produced by that script, export one shared grid key, e.g.:

```console
python -m tools.headless_campaign_queue.export_grid_scores \
  grid-values-340.pt grid-values-365.pt paired.json --key 32 \
  --state-set-sha256 HASH --before-checkpoint-sha256 HASH \
  --after-checkpoint-sha256 HASH
python -m tools.headless_campaign_queue.policy_stability paired.json report.json
```

The exporter hashes the two grid files but **cannot verify** that the claimed
state set, action order and checkpoint hashes actually produced them; preserve
the original collection evidence. Index labels are not physical control labels.
Only load trusted tensor files. The prior `bellman_probe.py` calculates one-step
TD quantities from replay; those are not realized on-policy returns and must not
be entered as `before_return` or `after_return`. The existing native backend has
not been demonstrated to restore an arbitrary recorded state for matched
rollouts. Until that capability is proven, leave the return fields absent; full
return calibration and periodic automatic collection remain future work.

The 2026-11-02 isolated IQN diagnosis found update-induced action-ranking
fragility and a contribution from encoder/head updates. Lower learning rate,
freezing the encoder, five-step returns and extra quantiles did not establish
a durable remedy. Do not present those interventions or this tool as fixes.

## Separate post-campaign robustness protocol

Before launching further training, predeclare the methods, seeds, budget,
checkpoint policy, software and map fingerprints, and evaluation seeds. Train
multiple independent seeds per method and preserve every run, including failures.
On fixed final checkpoints, evaluate each seed under identical, explicitly
recorded nominal conditions and a separate set of controlled perturbations
(for example reproducible start-state variations or held-out maps supported by
the backend). Verify that each perturbation is valid and comparable. Keep the
campaign's original 30-trial result separate from this additional protocol.

Report all attempts, finish fraction, termination reasons, and conditional
mean, median, spread and best time among finishes; include failure-aware results
so that faster finishes do not hide failures. Treat a training seed as the unit
of replication: repeated deterministic rollouts from the same state are not
independent training seeds. Compare across seed-level summaries and disclose
uncertainty and missing/failed runs. Do not choose best seeds or checkpoints.

Full analog parity between the native backend and the game is not established.
Test it independently using pinned controls, maps, versions and reproducible
trajectories before interpreting simulator results as equivalent to game results.
