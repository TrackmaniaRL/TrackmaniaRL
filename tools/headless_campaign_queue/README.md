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
`checkpoint_sha256`, `process_pid` and `manifest_sha256` after completing the job.
The wrapper must use its own PID and copy the manifest hash from the environment
variable `TRACKMANIARL_QUEUE_MANIFEST_SHA256`; job ID is supplied in
`TRACKMANIARL_QUEUE_JOB_ID`. The queue verifies these against the process it
launched and the manifest, as well as the receipt and
checkpoint hash, then records both hashes in state. A receipt is a claim by its
producer: this check cannot prove that the producer actually trained for that many
transitions, saved a restorable full checkpoint, completed evaluation, or closed
all descendants. Those are separate requirements of the training wrapper.

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
