# Replay refactor: bounded game-free training measurement

Measured on 2026-11-12, comparing baseline `536b42b5` with `8ccefba5`.
These numbers describe only this small local CPU workload. The benchmark used no
game, existing training port, checkpoint,
stop signal, or output directory from that run. The two code checkouts and all
benchmark output directories were separate. No two benchmark processes ran at
the same time.

## Workload and method

`scripts/benchmark_replay_roadmap.py` uses the local `Trainer`, a synthetic 8x8
image environment and an IQN model with a GRU, prioritized replay, sequence
length 4, batch size 2, 128-entry replay, burn-in 1, and CPU float32. Each
process used seed 42 and ran 32 transitions, 12 learner updates and four
eight-step episodes. The run wrote its own checkpoints at updates 8 and 12.
The timer starts immediately before `resolve_run` and ends after training and
logger close; it includes model setup, collection, sampling, updates and
checkpoint writes, but excludes Python imports and argument parsing.

Each process ran with OMP, MKL, OpenBLAS, PyTorch intra-op and inter-op thread
counts set to one. On Windows, it set its own priority to below normal. The
execution order was base, changed, changed, base, base, changed, changed, base.
Each run used a fresh output directory. The last pair recorded the imported
package paths, confirming the intended checkout for each version. The preceding
six used the same script and `PYTHONPATH` selection; the later addition of the
path to the JSON output did not change the measured code or timer.

To repeat in PowerShell, use the same Python environment for both checkouts,
set the three thread environment variables to `1`, and run sequentially with
distinct output paths. Replace the placeholders with local absolute paths:
The script atomically creates the final output directory and rejects an
existing one. The caller must choose a separate parent outside the active
training's directories; the script cannot identify or validate that boundary.

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = '<checkout under test>'
& '<python.exe>' '<changed checkout>\scripts\benchmark_replay_roadmap.py' '<new isolated output directory>' --transitions 32
```

## Results

| Run order | Version | Elapsed seconds |
| ---: | --- | ---: |
| 1 | baseline | 4.086 |
| 2 | changed | 9.654 |
| 3 | changed | 5.352 |
| 4 | baseline | 3.975 |
| 5 | baseline | 4.006 |
| 6 | changed | 5.479 |
| 7 | changed | 5.502 |
| 8 | baseline | 4.018 |

Medians were 4.012 s baseline and 5.491 s changed, about 37% longer in this
small sample. This is an observed regression on this particular test, not an
attribution to a specific change. One changed run took 9.654 s and the machine
was shared, so scheduling and competing load are uncontrolled. A pilot before
the measured sequence took 16.345 s on the baseline and is not included in the
table. The [recorded result data](2026-11-12-replay-game-free-results.csv)
contains the unrounded elapsed times and completed work counts. The original
logs and independent output directories remain on the measurement host, outside
either checkout; host-specific personal paths are omitted from distribution.

This local run does not exercise the distributed journal or real Trackmania,
larger observations, GPU execution, long-running replay, or representative
checkpoint sizes. It cannot establish end-to-end distributed training impact,
cause of the timing difference, or a performance improvement. A representative
measurement requires separate safely allocated resources and a longer controlled
run; do not repurpose the live training resources for it.
