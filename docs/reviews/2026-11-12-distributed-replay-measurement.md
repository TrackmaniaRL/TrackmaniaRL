# Bounded distributed replay measurement

A later [matched-budget measurement](2026-11-12-distributed-replay-matched.md)
uses equal transition and update counts, with its own stated limitations.

This exploratory measurement does not establish production throughput or
performance. The driver uses `ActorRuntime`, local gRPC transport,
the coordinator's SQLite journal, replay, sequence sampler and CPU IQN+GRU
learner. It also writes periodic and final checkpoints. The test environment
produces 8×8 synthetic observations and sleeps 10 ms per step. No Trackmania
process, port, checkpoint, stop file or training output is used.

On 2026-11-12 the baseline was commit `536b42b5` and changed code `859c9b06`.
Four separate processes were run in baseline, changed, changed, baseline order.
Each process selected one available CPU using Windows process affinity, set idle
priority, and limited Torch, OpenMP, MKL and OpenBLAS to one thread. The server
bound `127.0.0.1:0`, so the OS assigned a free port for each run. Each trial
claimed a new isolated output directory, its own journal, spool and checkpoints.
All four completed without a reported actor or coordinator error.

The elapsed value covers server/actor startup, training and actor cleanup; it
excludes initial Python import, config creation and `resolve_run`. The requested
budget was 64 transitions, warmup 8, 0.5 updates per transition, batch 2,
sequence length 4, n-step 1, and checkpoint interval 8 updates. Four raw result
records and counts checked from each journal are in
[the accompanying CSV](2026-11-12-distributed-replay-results.csv). All runs had
one source provenance archive and wrote 24–27 journal receipts. The full local
run directories and stdout logs are retained outside the repository in the
separate benchmark workspace.

| Trial | Version | Seconds | Transitions | Updates | Checkpoints |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | baseline | 21.722 | 72 | 32 | 4 |
| 2 | changed | 16.100 | 75 | 33 | 5 |
| 3 | changed | 18.691 | 80 | 36 | 5 |
| 4 | baseline | 16.319 | 75 | 33 | 5 |

Asynchronous buffering overshot the requested budget by different amounts;
checkpoint counts, startup costs and a shared machine also varied. Two short
runs per version do not support an estimate of the change's training impact.
In particular, these wall times must not be used as a percentage speedup or
regression. A larger matched-budget comparison on a dedicated representative
distributed environment remains necessary for such a claim.

To repeat, choose a new output directory under a parent known to be separate
from all active training resources. Run the driver in a new process with
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` and
`PYTHONPATH` pointing to the checkout being measured:

```text
python path/to/benchmark_distributed_replay.py NEW_ISOLATED_DIRECTORY --transitions 64
```

Run each version separately and verify the emitted `package_path` and actual
transition/update counts. The driver does not know which other directories are
in use; the operator is responsible for selecting an isolated output parent.
