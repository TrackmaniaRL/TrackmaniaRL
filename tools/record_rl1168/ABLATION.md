# Retention comparison — not yet executed

The completed campaign did not improve 17601 ms. Parameter updates are not driving improvement.

Compare retention weights 2, 0.2 and 0 using --retention-weight. Keep initializer, seeds, RL update cap, native backend, action space and reward identical. Report actual transition counts and any early stops: the elapsed-time guard can prevent equal update budgets. Use multiple matched seeds before selecting a winner. The default remains 2 until native evidence justifies changing it.

Record separate full-start greedy finishes and times, exploratory finishes and times, prefix outcomes, candidate admissions, accepted RL update count, and failed attempts. Retain every failure in its denominator. A prefix finish is never a full-lap record. Compare a frozen BC policy on the same exploratory seeds; 3/23 alone does not establish regression without this control.

Acceptance: an independently frozen RL checkpoint must produce a full-start NativeGoal below17601 ms without teacher inputs or learning. Retention accuracy and unit tests alone cannot meet this condition. Preserve the last proven actor independently throughout.

The training loop now buffers transitions until episode validation and trace archival succeed. Observation-guard failures discard pending transitions instead of leaving partial trajectories in replay.

Native comparisons require an explicit time budget. Windows/TICK parity and visual exports are separate validations.

## Configuration and scope

These repository-only research tools require an independently qualified native backend and assets referenced by a preparation manifest. They do not include the Trackmania server or claim Windows-client parity. Use an existing absolute Linux directory in `TMRL_CAMPAIGN_PRIVATE_BASE` for campaign-owned runtimes. The launcher requires explicit Windows report paths, Linux Python executable, private base and WSL distribution; it never chooses a user-specific path. Both report and private filesystems retain at least 4 GiB of free space. The launcher enforces a fresh output directory and explicit deadline.

Incomplete, interrupted, or invalid training episodes do not enter replay. A rejected progress measurement during evaluation is counted as a failed trial only after native closure passes; no reward or Goal is fabricated.
