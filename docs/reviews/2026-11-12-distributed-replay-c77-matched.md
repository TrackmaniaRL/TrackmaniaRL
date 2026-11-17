# Matched-budget synthetic distributed replay measurement at c77a6507

On 2026-11-12, four isolated processes compared baseline `536b42b5` and replay
change `c77a6507`, including the object-column ownership fix. This repeats the
[matched-budget method](2026-11-12-distributed-replay-matched.md): actual actor,
local gRPC, SQLite journal, replay and sequence sampler, CPU IQN+GRU learner,
spool drain and checkpoint writes. The synthetic 8×8 environment sleeps 10 ms
per step. It does not exercise the real game, GPU, multi-host transport or the
active training run.

Trials ran in baseline, changed, changed, baseline order, each in a new output
directory with a dynamically allocated loopback port. Each process used a
single CPU affinity, idle priority and one thread for Torch, OpenMP, MKL and
OpenBLAS. The elapsed interval covers actor/server startup, training, spool
drain and checkpoint completion; it excludes Python import, config generation
and run resolution. A boundary gate ensured exactly 128 transitions, warmup of
8, and 60 learner updates per run.

The [records](2026-11-12-distributed-replay-c77-matched.csv) were checked against
the process output, final checkpoint counters, journal receipts and applied
frontier. All runs completed with zero remaining journal chunks, eight
checkpoints and one source archive. The package path in each output identified
the intended checkout. Raw logs and run directories remain in the separately
created benchmark workspace; no existing training outputs were accessed.

The changed checkout's provenance manifest marked the tree dirty because it
included two untracked `.test-tmp` JSON files (159 bytes each); its tracked
patch was empty. The baseline provenance was clean. Source archives were
2,199,457 bytes for the changed version and 2,177,205 bytes for the baseline,
so source-snapshot work was not identical and may affect elapsed time. These
untracked test files are not part of the compared commit.

| Trial | Version | Seconds | Transitions | Updates | Journal receipts |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | baseline | 38.987 | 128 | 60 | 51 |
| 2 | changed | 21.313 | 128 | 60 | 42 |
| 3 | changed | 26.139 | 128 | 60 | 45 |
| 4 | baseline | 19.521 | 128 | 60 | 34 |

Equal transition and update counts do not make the workloads identical:
journal chunk counts, scheduling and startup costs differed. Two short runs per
version on a shared machine cannot establish a speedup, regression, mechanism
or production impact. A longer representative comparison is needed before a
performance claim. Repeat with `scripts/benchmark_distributed_replay.py` as
described in the earlier report, selecting a fresh output parent separate from
all active training resources.
