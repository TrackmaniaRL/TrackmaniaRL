# Retention comparison — not yet executed

The completed campaign did not improve 17601 ms. Parameter updates are not driving improvement.

Compare retention weights 2, 0.2 and 0 using --retention-weight. Keep initializer, seeds, total real transitions, RL update cap, native backend, action space and reward identical. Use multiple matched seeds before selecting a winner. The default remains 2 until native evidence justifies changing it.

Record separate full-start greedy finishes and times, exploratory finishes and times, prefix outcomes, candidate admissions, accepted RL update count, and failed attempts. Retain every failure in its denominator. A prefix finish is never a full-lap record. Compare a frozen BC policy on the same exploratory seeds; 3/23 alone does not establish regression without this control.

Acceptance: an independently frozen RL checkpoint must produce a full-start NativeGoal below17601 ms without teacher inputs or learning. Retention accuracy and unit tests alone cannot meet this condition. Preserve the last proven actor independently throughout.

The training loop now buffers transitions until episode validation and trace archival succeed. Observation-guard failures discard pending transitions instead of leaving partial trajectories in replay.

The previous two-hour authorization has ended. No new training campaign or resume was started by this change. Native comparison needs a new explicit total budget; Windows/TICK parity and visual exports remain outstanding.
