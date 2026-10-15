# TrackmaniaRL 1.2.12: Jakub Szulc — QR, seed 17

This is the first full run of the new campaign, not a pilot or a resumed run.
Library base: release 1.2.12, commit d4aa669e390692a6fc255afbafb18da6e3f3d867.
The research branch adds supervision and the historical batch-shape adapter;
it does not change the released learner, reward, model or game controller code.

## Design fixed before collection

- Operator: Jakub Szulc. Assignments: QR then IQN; seeds 17, 29 and 43 each.
  This launcher starts only QR seed 17. Other people and subsequent runs are not launched.
- Fresh random weights and empty replay. No demonstrations, warm start, pilot
  checkpoint, reference-policy filter, or selected checkpoint.
- Budget: 8,192,000 environment transitions; warmup 10,000; uniform replay
  capacity 1,000,000; batch 256; 0.25 updates per eligible transition; gamma .995.
  Strict accounting requires 2,045,500 updates and drained final update credit.
- Historical V5 graph observations and incident-gated V6 encoder; QR uses
  64 quantiles, 78 discrete controls, dueling head, Adam LR .0001, hard target
  every 1,000 updates. All resolved options are recorded in qr.yaml and manifest.
- Map UID oqIJ5rQDRrNwLPTh9H2p_W4tLof, same map/geometry as previous pilots,
  decision interval 50 ms, episode cap 150 s. No racing-line or pace guidance.
- New campaign uses 278 m/s as the projected-speed limit, consistently for all
  new configurations. Other reward weights are retained from the old campaign.
  This is a protocol change: historical pilot scores are diagnostic context,
  not matched controls for the new experiment. Never pool the old and new runs.
- Train to the fixed budget regardless of score. No performance-based early stop.
  Operational STOP, controller/telemetry errors, source mismatch or low disk stop
  the run and are reported as interruptions, not algorithmic quality failures.
- Final evaluation: last completed training checkpoint, 30 trials on the same
  map, once. Record finishes, conditional finish times, failure progress, errors,
  all decision timing and telemetry skips; no 100 ms exclusion/waiver hiding.
  No retrying evaluation or choosing a checkpoint on those trials.

## Reproducibility and scope of claims

Retain source/configuration/asset hashes, resolved manifest, source snapshot,
environment versions, hardware, OS/driver, game/plugin versions, local steering
calibration, game settings/FPS, all checkpoints retained by policy, JSONL logs,
W&B URL, supervisor identities, STOP history, and final checkpoint SHA256.
Local artifacts are authoritative; W&B is a projection, not the only copy.
At completion verify counters, finite weights, fingerprint, final evaluation
identity and closure before describing the run as complete.

The nominal interaction time is 113.78 hours (4.74 days) at 20 decisions/s;
resets and learning add time. Maximum wall-clock runtime including evaluation
is 336 hours; final 600 seconds are reserved for saving and closing. No extension,
automatic restart, resume or next run. An independent guard supervises the
launcher and exact owned PID/create-time identities. STOP wins over all stages.
Require 100 GiB free before launch, stop below 10 GiB. Keep the desktop unlocked,
game foreground and machine awake; the launcher does not alter OS/security settings.

This single-map, shared-engineering study can support a reproducible case study.
Three seeds are a minimum planned comparison, not a guarantee of statistical
power. Thirty trials of one network are not thirty independent training seeds.
Report each seed and uncertainty across independently trained models, with the
small sample size explicit; do not claim general superiority across unseen maps.
Broader paper claims require prespecified held-out maps and more evidence.
Evaluation guidance: [Agarwal et al., NeurIPS 2021](https://arxiv.org/abs/2108.13264).

## Operation

From a clean frozen checkout with a pinned CUDA Python environment:

```text
python -m experiments.jakub_1212.campaign prepare
python -m experiments.jakub_1212.campaign verify
python -m experiments.jakub_1212.campaign host-check
python -m experiments.jakub_1212.campaign check-game
python -m experiments.jakub_1212.campaign run --launch
```

Calibrate steering locally before preparing the host; never reuse another
computer's curve. Do not edit or update the frozen checkout after starting.
Stop this run by creating artifacts/STOP-qr inside its frozen checkout.
