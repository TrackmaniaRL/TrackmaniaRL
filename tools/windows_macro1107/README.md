# Windows macro adapter 1107

Experimental CPU-only live inference for the protected 132-action policy which
finished five headless trials in 17.715 s. No successful Windows finish has been
demonstrated: the three actual trials recorded so far all reached the 30-second
test ceiling without finishing. Two earlier attempts failed before driving.

The adapter loads only a policy-only checkpoint and its SHA-bound **unordered**
catalog from the inference config (training-data1091). It uses real 33-field
Openplanet packets, the original V5 pipeline, and SDK DiscreteValuePolicy in
EVALUATION mode. No learner or optimizer is constructed. During live inference
an audit hook denies ghost, teacher, labels and demonstration files. Offline
regression used 64 archived model observations and matched all 64 original
model decisions; those archived action labels are never used by the live loop.

Each model-selected program is converted from five native action78 commands
into GBS Windows controls. Mode2 uses brake1 for that literal tick, matching
native session.advance(command, 1), rather than replaying a five-tick brake tap.
All 78 conversions were checked against action_sgb(command)[0]. The keyboard
backend refuses fractional steering; it never silently quantizes the model.
The gamepad backend sends requested raw stick positions without claiming an
unmeasured analog calibration curve.

The Windows pipeline and native pipeline produced bitwise-identical tensors
on the same 11 real Windows frames. This checks preprocessing on those frames,
not equality of the Windows and headless trajectories.

The first prototype accumulated inference time between 50 ms programs. The
current scheduler uses a fixed host epoch estimated from the actual race clock,
so that inference no longer shifts every later decision. Host send timestamps
and lateness are logged for every tick. **These are not engine acknowledgements
or proof that commands were applied at exact native physics boundaries.**

For example, the second keyboard program requested a brief right steer while
the packet at race100ms still showed InputSteer0; the model first selected a
different next action than the headless baseline at decision2. The final pad
trial also did not finish. Timing/input sampling, reset-state differences and
other runtime differences need paired evidence before assigning a cause.

Execution is explicit via `--live`; importing the module or running tests never
drives the game. Preconditions include the exact map UID, ready player and the
named Windows controller mutex. One invocation performs one recorded trial,
at most30s driving and90s supervised wall time. Reset must yield a genuine zero
race clock. Controls, telemetry and the owned recorder are closed in finally.
`STOP` is confined to that invocation's output directory. Prior output is never
overwritten. Old runs and immutable deployments are not changed or resumed.

Outputs preserve the original MKV, recording metadata, FFmpeg log, each actual
model decision, raw telemetry, requested motor commands and final result. Finish
time is Windows telemetry only; no headless Goal or public record is inferred.
The video includes failures. Successful publication as a 17.715 s autonomous
Windows run requires a real client finish, which is currently absent.

The installed FFmpeg helper and dependencies live only in the report directory;
no packages were installed into the training environment. Paths are extended
only for local artifact access; the recorder executable has a shorter owned
path to satisfy CreateProcess limits.

Checks: six focused tests, Ruff and strict mypy; manual review of recorded video
frames; all78 SGB conversions; policy SHA unchanged after each live attempt.
