# Migration and recovery

Use this guide for full training checkpoints and distributed actor rollouts.
It does not guarantee recovery of data that was never saved. A checkpoint does
not restore an in-progress game session.

## Before updating or moving a run

1. Stop the learner and all actors, and confirm they have exited. Keep the
   original library version, configuration and assets available for recovery.
2. Back up the run directory, including `checkpoints/`, the distributed journal
   `distributed/rollouts.sqlite3` and any SQLite `-wal` and `-shm` companions,
   plus every actor's local spool. Take the backup with all processes stopped;
   an actor on another machine has its own local files.
3. Test recovery on a copy before using the updated installation for an
   important run. Do not run old and new actors against the same spool at the
   same time.

The distributed journal upgrades schema 1 to schema 2 on opening. Schema and
version are changed in one transaction, and legacy evaluation delivery state is
migrated on load. Keep the pre-upgrade backup if you may need the old version;
the upgrade is not a guarantee of downgrade compatibility. The journal is tied
to its checkpoint by identity and progress; replacing it with a fresh file is
not equivalent to recovery.

## Stop and resume

Local `train` and `resume` accept `--stop-file PATH`; the path must not exist
when the process starts. Creating it requests a controlled stop. The launcher
stops the actor and drains accepted journal work and earned learner updates.
It saves a final checkpoint only when `training.save_final_checkpoint` is
enabled. On this stop-file shutdown path, actor shutdown has a 60-second grace
period, followed by a 120-second bound for the learner to drain and exit.
When the actor instead exits naturally, the initial wait for learner drain has
no fixed timeout; a stop file can request the bounded shutdown path. Check the
process exit status and checkpoint-completed log before relying on a final
checkpoint. A timeout or actor failure may leave it incomplete. Remove the stop
file before resuming.

```powershell
uv run trackmaniarl resume run.yaml artifacts/RUN/checkpoints/CHECKPOINT.pt
```

Use a full training checkpoint for exact resume. Fingerprint mismatches are
rejected, including changes to tracked library sources, effective configuration
or assets. New checkpoints can report changed hashed sections and relative
source paths; older ones may only report a mismatch. Restore the original code,
configuration and assets and use the matching journal. Do not rewrite the
fingerprint to force a resume. For an intentionally different run, use a
compatible model-initialization checkpoint as described in the
[quickstart](quickstart.md#train-and-resume); it does not restore optimizer or
replay state.

Full resume rejects older checkpoints without recorded consecutive evaluation
progress when evaluation-based early stopping is enabled; those checkpoints
cannot reconstruct the missing progress exactly. Older evaluated-policy leader
checkpoints also cannot provide the distinct online learner and pending
promotion state needed for exact resume. Use a matching full training checkpoint
instead. A compatible evaluated-policy checkpoint can still initialize a new
run's model, subject to the model-initialization rules.

## Actor spool recovery

An actor stores pending rollouts under
`ARTIFACTS_DIR/RUN_ID/actors/actor-<sha256-of-actor-id>/spool/`, resolved from
the config directory. Reuse the same explicit actor ID and compatible run
configuration when restarting a separately launched actor. On startup, valid
numbered `.tmp` files are recovered into `.rollout` files; invalid temporary
files are retained as `.invalid` for inspection. The sender retries retained
rollouts after restart. A file remaining on disk is durable evidence, not proof
that the learner accepted it: inspect the actor and learner logs.

Legacy actor directories used a sanitized actor ID. The new actor migrates
pending files only when their recorded ownership is unambiguous. Invalid legacy
temporary files are retained under
`actors/<sanitized-actor-id>/spool-quarantine/`. Mixed ownership,
an unidentifiable rollout or an existing destination can stop migration with an
error. Preserve both directories and inspect their contents rather than deleting
or merging files blindly. A spool produced with a different run fingerprint may
still be rejected by the learner; moving it does not make versions compatible.

## Telemetry quality

With `--reject-telemetry-skips`, a benchmark requires valid skipped-frame
measurements for every step of every trial as well as zero reported skips.
Missing or invalid measurements fail this gate; older results lacking coverage
information do not establish zero skips. Inspect the per-trial result and
coverage fields. Without the strict gate, incomplete telemetry can still be
reported, so use the coverage fields when interpreting aggregates. The gate
does not remove slow or unsuccessful trials from the benchmark.
