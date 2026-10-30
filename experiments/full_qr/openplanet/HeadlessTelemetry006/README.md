# Fresh telemetry diagnostic 006

This optional source was compiled and used for bounded live r4 diagnostics on
Openplanet1.29.14/git6fd4200f and the pinned client. It remains opt-in. Existing
Connect files are unchanged. Its dependencies are the same
`VehicleState` and `PlayerState` helpers used by Connect.

Loading the independent plugin opens only `127.0.0.1:19179`. The plugin reads
telemetry and metadata. It sends no controls, changes no settings, invokes no
reset and installs no code hook. Disabled/destroyed callbacks close its sockets.
The listener can remain idle; each accepted client has a 120-second lifetime,
five-second idle deadline and 4,096-request cap. The coroutine yields every
frame. These checks run when that coroutine resumes, without an independent
operating-system watchdog.

## Protocol

The protocol is `tmrl.fresh-telemetry/v6`, collector version `0.0.6`, field
schema `tmrl.connect33-float32/v2`. On connection the server sends a single
JSON-line `status="hello"` header, with zero payload. It never pushes an
observation without a request. A client sends:

```json
{"protocol":"tmrl.fresh-telemetry/v6","op":"sample","nonce":"0123456789abcdef0123456789abcdef","sequence":1}
```

The actual client generates a new 32-character lowercase hexadecimal nonce per
connection. Exactly those four request keys are accepted. The nonce is bound
on request one, with consecutive integer sequences thereafter. Requests are
limited to 1,024 bytes. A malformed request closes the connection after a
best-effort error header. Nonces distinguish connections; this diagnostic does
not claim persistent replay protection across plugin reloads.

Each valid request builds a new MemoryBuffer using the repository
`TrackmaniaRL_Connect.as` 2.4.0 builder's field order and conversions. Its response
is a JSON-line header followed immediately by exactly **132 binary bytes**:
33 little-endian float32 values. Both parts use `Socket.WriteRaw`, so there is
no implicit string-length prefix. The raw MemoryBuffer bytes pass through
`ReadString(132)` and retain embedded NULs. Live r4 tests received bound132-byte
payloads with retained hashes. Linux tests exercise the matching client
against an independent mock producer.

The success header includes nonce, sequence, exact byte/field counts,
`producer_begin_Time_Now_raw`, `producer_end_Time_Now_raw`, actual map UID and
loaded game/Openplanet versions. The begin/end interval covers prerequisite
checks, metadata access and field construction. It ends before socket writes.
Missing playground/player/vehicle/race-data/map or a failed getter returns
`status="unavailable"`, zero fields and zero payload bytes. It never sends a
zero observation or silently substitutes a previous sample. Connect's defined
pre-start zero race time/checkpoint/lap representation is retained only when
the complete actual observation sources are present.

| Indices | Connect fields |
| --- | --- |
| 0–3 | checkpoints, lap, finished, race time |
| 4–9 | actual API position XYZ, velocity XYZ |
| 10–15 | visual direction XYZ, up XYZ |
| 16–18 | speed, RPM, gear |
| 19–22 | FL/FR/RL/RR slip coefficients |
| 23–26 | FL/FR/RL/RR contact materials |
| 27–29 | wheels skidding, flying duration, adherence |
| 30–32 | observed API steer, gas, braking |

## Python integration

`scripts/headless_telemetry006.py` exposes `Telemetry006Config` and
`Telemetry006Client`. The familiar `connect`, `read`, `read_next`, `read_latest`
and `close` methods are synchronous. Each read requests one new observation;
`skipped_frames` is always zero because this is not a streamed queue. The
returned object is the existing `TelemetryFrame`, with a read-only float32
array and the same finite-vector/speed domain checks as the standard client.

```python
from scripts.headless_telemetry006 import Telemetry006Client, Telemetry006Config

client = Telemetry006Client(Telemetry006Config(timeout_s=2))
try:
    frame = client.read_next()
    packet_metadata = client.last_metadata
    exact_raw33_bytes = client.last_packet
finally:
    client.close()
```

`last_metadata` retains the untouched producer header, host request/receive
`perf_counter_ns` and payload SHA-256. `last_packet` retains exact raw bytes.
Capture them immediately with each frame. On explicit unavailability,
`Telemetry006UnavailableError` is raised and metadata retains that response;
`last_packet` is None. A caller waiting through a reset may retry fresh requests
within its own existing STOP/deadline budget, yielding between attempts. This
client itself does not retry or replay an uncertain failed request. EOF,
protocol error or a trickling partial response fails within a total receive
deadline and closes the connection. A later explicit read may reconnect with a
new nonce. No live integration or controller is launched by this module.

## Interpretation limits

Fresh request binding removes an existing producer-pushed socket queue as a
source of old packets. It does not prove that game getters are current to a
particular physics tick, mutually atomic, or an acknowledgement that issued
inputs were consumed. Dependency helpers and game caches can have their own
update phases. Producer `Time::Now` is separate host metadata; it is not the race
clock and has no measured conversion to the Python host clock. The observed
API input fields remain distinct from issued controls.

Request/response exchange also changes scheduling and must be measured as a
separate diagnostic transport. No TCP/Nagle explanation or end-to-end zero-lag
claim follows from this source. Comparative training remains gated by the
independent evidence requirements.
