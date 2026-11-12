# Map lifecycle observation probe

This is a separate, optional Openplanet developer probe. **It is not part of
Connect or the training runtime**, starts no listener, sends no controls, and
does not start recording until its window's **Start new capture** button is
pressed. Do not install or run it in a game or profile used by any active
training job. It has not been compiled or run in a game.

It samples once per Openplanet coroutine iteration: the active map UID, the
same local-player readiness predicate used by Connect, whether
`Network.ServerPlugin` exists, its `MapLoaded` and `MapUnloadRequested` flags,
and the `MapLoaded`, `MapUnloadRequested` and `MapUnloaded` entries visible in
its `PendingEvents` buffer. It reads `HasBeenProcessed` but never changes it or
consumes events. Each line contains a local sample counter, `Time::Now`, and a
race-time field calculated like Connect's telemetry (zero when unavailable).
The buffer is bounded at 12,000 samples and 2 MiB of JSONL, and records at most
the first 64 pending events per sample. Map UIDs are limited to 256 bytes and a
field indicates truncation. The window shows the last 20 samples. **Copy JSONL**
explicitly places the bounded capture and a status trailer on the clipboard;
nothing is written to a file by the probe. The trailer identifies an active
capture, user stop, sample/byte limit or read error. An error or limit is marked
as truncated; do not treat such a capture as complete.

Samples are observations, not a complete event log. The server-plugin object
may be absent in local play, another component may process or clear events,
and a change can happen between samples. The same pending entry can be observed
in multiple samples and must not be counted as distinct events. The probe does
not prove loaded map bytes, event ordering relative to engine threads, or
continuity across A→B→A. Reading runtime objects during a game transition has
not been validated in the game and may itself fail; use a disposable session.

## Controlled procedure

1. With a **separate**, exclusively controlled game instance and Openplanet
   development profile, arrange the official `VehicleState` dependency and
   load this directory as its own developer plugin. Do not use a profile or
   fixed ports associated with a running training job. Do not use a deployment
   tool that broadcasts to all running games.
2. Compile/load the probe in that instance and record the Openplanet version,
   signature mode, game build and whether compilation succeeds. If it fails,
   stop; the static LSP check is not a substitute.
3. Open the probe window and click **Start new capture**. Manually load trusted
   test map A, then B, then A again, and include a same-map restart and a pause
   or finish screen. Stop the capture before the limit; copy JSONL to an
   isolated test artifact. Record the intended action order separately.
4. Compare UID, readiness, server availability and pending event entries with
   the intended actions. Treat missing, repeated, processed or truncated entries
   as results; do not infer events that were not observed. Repeat cached-map
   transitions and plugin reloads before drawing any conclusion.

The type names and event enum were inspected in local Openplanet 1.29.14 API
metadata. Their presence does not establish availability in a particular game
mode or School Mode, nor that this probe compiles or runs there.
