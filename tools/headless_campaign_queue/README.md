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

Use a fresh output directory. Existing state, logs or lock block execution; no
automatic restart or resume is attempted. An abrupt termination may leave a lock
or a child process running. Inspect the process and artifacts before manually
removing a stale lock or deciding how to recover. The queue does not manage sleep,
power failure, process-tree cancellation, checkpoints, or disk-space guarantees.
The shared lock lives in the current user's home directory and prevents two
instances under that user from using different output directories simultaneously.
It does not prevent another OS user or an unrelated launcher from starting work.
