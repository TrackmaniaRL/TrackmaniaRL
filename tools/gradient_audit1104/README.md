# Frozen checkpoint gradient and finish-time audit

This diagnostic uses a private checkpoint copy on CPU. It does not attach to a
learner, read a live WAL, append transitions, launch physics, apply gradients, or
modify a campaign. It restores the copy's online/target networks, replay and
sampler RNG, then draws twelve batches of256positions at a fixed transition count.
These are diagnostic draws from the saved replay, not recovered logged batches
or independent training runs.

Run with the existing development Python from the repository root:

```text
python -B -m tools.gradient_audit1104.measure --checkpoint PRIVATE/frozen-update25000.pt --spec PRIVATE/original-run.yaml --source //wsl.localhost/Ubuntu/home/kuba/tmrl-full-qr-20261022-011/shortcut-multi-source1094 --study STUDY --output PRIVATE --batches 12
```

The output directory must contain `snapshot.json` identifying a different source
path and the private copy's SHA256 as `copy_sha256`. No package installation is
needed. The saved immutable SDK is imported explicitly and its package path is
checked. The CPU invocation hides CUDA, uses one Torch thread, allows at most
sixteen batches and checks a300second budget between batches. Checkpoint
decompression is bounded to4GiB and occurs beside the private copy. Existing
`result.json` is never overwritten. The checkpoint's optimizer is not resumed.

`ValueUpdater` supplies the actual separate QR and imitation losses. Only its
batch preparation and forward/loss methods are called. `torch.autograd.grad`
measures the TD and already weighted BC gradients on the same forward graph.
The implementation never calls learner update, backward-to-`.grad`, optimizer
step, target synchronization, native factories, or evaluation actors.

Reports include global, encoder and head norms, weighted BC/TD ratios, cosine
similarity and combined gradient norms. Lambda16 rescalings keep this exact
snapshot and batch fixed; they are sensitivity calculations, not forecasts of
future training. All values are before clipping and Adam. A ratio above one
does not prove that imitation blocks a better trajectory. The first batch is
repeated exactly; online/target tensors, `.grad` fields, empty private optimizer
state, checkpoint SHA and absence of a CUDA context are verified afterward.

The reward audit replays the real archived Jimmy prefix using the saved reward
configuration and pinned reward-reference points. It checks agreement with the
saved rewards, then scores cloned final reward states with different Goal times.
The physical observation remains at17750ms. Those changed timestamps are
explicitly counterfactual reward-function probes, not generated finishes or
evidence that a faster policy exists. The physical/native Goal guard is never
modified or bypassed for any actual episode.

The audit requires the prior `ghost-divergence1103/jimmy-headless-final.json` and
qualified Jimmy demonstration under the study directory. Source and report
fingerprints remain private study evidence. Two focused gradient tests check
direction/magnitude, zero-norm handling and preservation of model weights.
