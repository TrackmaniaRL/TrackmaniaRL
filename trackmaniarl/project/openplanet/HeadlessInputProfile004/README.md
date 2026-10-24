# Temporary input profile 004

This independent plugin listens only on `127.0.0.1:19177`. Loading it starts an
idle server and changes no settings. The supported profile is
`no-artificial-lag/v1`; its only game writes set `FakeInputLagAvg` and
`FakeInputLagVar` to zero, then restore their exact captured values.

The Python `NoArtificialLagLease` context manager in
`scripts/headless_input_profile.py` first owns the existing
`Global\TrackmaniaRL.ComparisonController` mutex. It checks the already focused
game window, reads a fresh baseline hello, binds the actual map UID, and echoes
the baseline in an explicit acquire request. It does not activate a window,
drive, reset or change a map. The maximum TTL is120seconds, without renewal.

The wire protocol is `tmrl.input-profile/v1`, JSON lines limited to8192bytes.
Acquire binds a fresh32lowercasehex nonce, profile, expected map, exact baseline,
TTL, controller-mutex attestation and sequence1. Subsequent `status`/`release`
requests increment the sequence and carry the same nonce. Duplicate/nonsequential
requests, reused nonces, additional keys and invalid values are rejected. The
plugin cannot independently inspect the host mutex; its attestation is supplied
by the Python client that actually owns that mutex.

The plugin checks deadline, focus, map and current settings every yielded frame.
It restores on release, EOF, timeout, protocol/runtime error, disabled/destroyed
callbacks, focus loss or a map change. The current playground map is preferred;
the actual `RootMap` is accepted during editor-validation transitions. Conflicting
nonempty map identifiers fail closed. Restoration attempts both game writes
before reply construction or journal serialization. Cleanup is idempotent.

The plugin writes fresh per-nonce baseline/applied/restored JSON records only in
its own storage. A restored record explicitly states whether readback verified
restoration and whether another restoration attempt remains pending. The client
can also retain its request/response transcript as a separate JSON report.
The profile does not modify `MaxSampleRate`,
`PollingEnabled`, `IgnoreFocusForGamePads`, driving controls or physics.

Use `lease.check()` at case boundaries to confirm the actual remote profile/map.
Use `lease.check_local()` before each action when an additional TCP exchange would
change the timing measurement; it checks only foreground and the local deadline,
and cannot confirm that remote settings remain applied. The plugin watchdog still
runs. Existing queued inputs and physical reader timing require measurement;
zero settings do not by themselves prove zero end-to-end actuation delay.

Example after the operator has explicitly loaded the fresh plugin and focused the
expected local map:

```python
from scripts.headless_input_profile import InputProfileConfig, NoArtificialLagLease

with NoArtificialLagLease(InputProfileConfig(expected_map_uid=known_uid)) as profile:
    profile.check()        # actual remote confirmation at a case boundary
    profile.check_local()  # no network exchange before the controller's action
```

Restoration verification occurs before the context releases its controller lock.
If the normal restore ACK is lost, it closes the socket to trigger EOF rollback,
then reconnects read-only and checks the original baseline. An unverifiable
rollback raises an error. A profile lease is single use; repeat runs use a fresh
nonce and fresh journal filename.

Offline tests use an injected fake loopback server and fake guards. They do not
load Openplanet, control the game, or exercise a real runtime setting.
