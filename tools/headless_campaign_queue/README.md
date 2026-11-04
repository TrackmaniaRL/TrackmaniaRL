# Queue prototype

This is a general process queue, not a qualified Trackmania campaign launcher.
It accepts a separately prepared JSON manifest containing explicit commands. No
training command is supplied: the five headless configurations and their
continuous-action compatibility remain to be implemented and qualified.

The manifest schema is `{"schema_version": 1, "jobs": [{"id": "...", "cwd":
"/absolute/directory", "argv": ["/absolute/python", "-m", "..."]}]}`. Every
executable and working directory must already exist. Only use a manifest that has
been reviewed and tested against the actual runtime. All jobs are checked before
any start; this structural preflight does not validate game assets, correct
training configuration, process-tree cleanup, model results or output budgets.

`python -m tools.headless_campaign_queue MANIFEST --output OUTPUT` checks only.
Adding `--execute` runs sequentially and stops at the first nonzero exit. Each
command has separate stdout and stderr, with atomic state updates and a manifest
hash. A zero exit alone does not prove training or evaluation completed correctly.

For schema version 2, each job also supplies absolute `receipt` and positive integer
`transitions`. Its trusted training wrapper must write a fresh JSON receipt with
`job_id`, `status` = `completed`, exact `transitions`, absolute `checkpoint`, and
`checkpoint_sha256`, absolute `evaluation`, `evaluation_sha256`, `process_pid`
and `manifest_sha256` after completing the job.
The wrapper must use its own PID and copy the manifest hash from the environment
variable `TRACKMANIARL_QUEUE_MANIFEST_SHA256`; job ID is supplied in
`TRACKMANIARL_QUEUE_JOB_ID`. The queue verifies these against the process it
launched and the manifest, as well as the receipt and
checkpoint hash, then records both hashes in state. A receipt is a claim by its
producer: this check cannot prove that the producer actually trained for that many
transitions, saved a restorable full checkpoint, completed evaluation, or closed
all descendants. Those are separate requirements of the training wrapper.

The experimental strict entry point is `python -m tools.headless_campaign_queue.campaign`.
It accepts schema 3 with a common seed, transition budget, nonnegative
`max_overshoot` acceptance bound and `map_uid`, and jobs
with `id`, absolute `config`, `config_sha256`, `run_fingerprint`, absolute fresh
`receipt` and `stop_file`. It requires all five named algorithms, hashes and
validates all configurations before starting any, then constructs only `job.py`
commands. It also requires a pinned `qualification_report` and
`qualification_sha256`: the report's `kind` must be
`headless-five-qualification-v1`, and each job must match its config SHA and run
fingerprint and record pass for `native_action_ack`, `checkpoint_restore`,
`evaluation`, and `clean_process_shutdown`. A report is a claim, not proof of
its own contents: do not fabricate it. No such qualification has been performed.
Besides the five team methods, the preflight recognizes Q, FQF, discrete SAC and
REDQ. It checks each learner and, for value methods, the head and strategy. Unknown
methods are rejected until their contracts are added. Configuration SHA and run
fingerprint are checked again immediately before each individual job.

Use a fresh output directory and fresh receipt paths. `--execute --resume` is
restricted to schema 2 queues interrupted between jobs: the manifest hash, each
completed receipt and checkpoint are rechecked; all remaining jobs must be
untouched. It never resumes an interrupted or failed training job. An abrupt
termination may leave a lock or a child process running. Inspect all processes
and artifacts before manually removing a stale lock or deciding how to recover.
The queue does not manage sleep,
power failure, process-tree cancellation, checkpoints, or disk-space guarantees.
The shared lock lives in the current user's home directory and prevents two
instances under that user from using different output directories simultaneously.
It does not prevent another OS user or an unrelated launcher from starting work.

`job.py` is an experimental producer that trains from a fresh config and performs
evaluation via the internal API, without imposing a performance threshold. It
selects the last durably logged regular training checkpoint, checks the checkpoint
fingerprint and the actual accepted transition counter, and requires the counter
to lie between the requested budget and budget plus `max_overshoot`. It preserves
and reports the actual count rather than trimming rollouts. For off-policy runs it
also requires a matching journal identity/frontier, no submitted WAL rows beyond
the frontier and no remaining spool files. The check is an acceptance criterion,
not a runtime collection limit; actual physical environment steps can exceed the
accepted transition counter. It has only synthetic tests and is not yet qualified
for any live method. A stopped run or one outside this bound fails closed.
For isolated tests only, `--qualification-trials N` accepts 1–30 matching trials;
its receipt is marked `qualification_only` and cannot complete the campaign.
The queue does not yet ensure all manifest commands use this trusted producer.

`verify_artifacts.py` contains read-only checks of training fingerprint, counter,
learner and replay shape, plus exact evaluation map and trial identities. These
are building blocks and are not yet wired into a qualified producer. They do not
verify restorable optimizer/RNG state or prove that the environment behaved as
intended.

`windows_wsl.ps1` can run a preflight through WSL and, with explicit `-Execute`,
hold a Windows idle-sleep request for the duration of the foreground queue. All
four Linux paths are explicit arguments (`Repository`, `Python`, `Manifest`,
`Output`). This is an experimental entry point; it does not survive termination
of its own Windows process, prevent explicit sleep/reboot, or establish that WSL
descendants have exited. Do not use it for an unattended campaign until those
lifecycle contracts and the real manifest have been qualified.

The Windows entry point sets `WANDB_MODE=offline`: W&B records are local and
can be synced later; the JSONL logger also remains enabled. Before each job and
its evaluation, and every ten seconds while they run, the job checks free space
on the configured output and native volumes. If a reserve is breached it writes
the cooperative stop file. This is not a quota and cannot guarantee graceful
shutdown if the stop file cannot be written or the process is forcibly killed.
