# Matched-budget synthetic distributed replay measurement

On 2026-11-12 a bounded game-free run compared baseline `536b42b5` with
replay change `048de845`. This measures the actor, local gRPC transport,
SQLite journal, replay, sequence sampling, CPU IQN+GRU learner and checkpoint
writes together. It does not run Trackmania, use the active training session,
or measure representative game, GPU or multi-host behavior.

Each run produced exactly 128 synthetic transitions and 60 updates, in the
order baseline, changed, changed, baseline. The environment sleeps 10 ms per
step and terminates every eight steps. On the final step it signals the driver;
if the actor reaches the next reset first, a gate stops further production.
The driver stops only its own actor, waits for its spool to drain, and then lets
the coordinator finish ingest, earned updates and the final checkpoint. The
driver fails if its returned transition or update count differs from the
requested budget. Warmup is eight transitions and the update ratio is 0.5, so
the expected update count is `(128 - 8) * 0.5 = 60`.

The [records](2026-11-12-distributed-replay-matched.csv) were checked against
each run's returned counts, final checkpoint counters, journal receipt count,
and applied journal frontier. Each run wrote eight checkpoints and one source
archive. The checked package path identified the intended checkout. Four
independent processes used a single CPU affinity and idle priority on Windows;
Torch, OpenMP, MKL and OpenBLAS each used one thread. Every run had a new
output directory and a local server bound to port zero, allowing the OS to
choose a free port. The measured interval includes server and actor startup,
training, spool drain and checkpoint completion; it excludes Python imports,
configuration creation and `resolve_run`. The original per-run logs and
directories are retained in the isolated benchmark workspace, not in the
active training output.

| Trial | Version | Seconds | Transitions | Updates | Journal receipts |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | baseline | 39.402 | 128 | 60 | 50 |
| 2 | changed | 32.400 | 128 | 60 | 51 |
| 3 | changed | 24.381 | 128 | 60 | 43 |
| 4 | baseline | 27.384 | 128 | 60 | 45 |

The matched counts eliminate the earlier transition/update budget mismatch,
but these are two short runs per version on a shared machine. Scheduling
changes the number of journal chunks and timing of policy updates; receipt
counts differ. The sample cannot establish a speedup, regression, causal
mechanism or production impact. A representative, longer comparison is
needed before making a performance claim.

To repeat, run `scripts/benchmark_distributed_replay.py` in a fresh process
with the desired checkout on `PYTHONPATH`, `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, an output directory that does
not yet exist, and `--transitions 128`. The output parent must be chosen
outside every active training resource; the driver cannot verify that choice.
Run and verify each version separately. The driver requires Windows for its
affinity and priority controls.
