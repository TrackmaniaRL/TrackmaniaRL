# Openplanet integration

For a normal installation, open Openplanet's **Plugin Manager**, install the signed
[**TrackmaniaRL Connect**](https://openplanet.dev/plugin/sac_getdata) plugin
(identifier `SAC_GetData`) and verify that it reports version **2.4.0** on
Openplanet **1.26.0 or newer**. Enable Openplanet **School
Mode**, then enter the configured local map with a visible vehicle. School Mode
deliberately prevents online play and official leaderboard submissions while the
plugin is active.

Run the complete local handshake before training:

```powershell
uv run trackmaniarl track check --config run.yaml
```

The command validates three 33-float telemetry frames, session protocol 2, the
active map UID and local-player readiness. The protocol cannot report the installed
package's signature or release number, so those two properties must be checked in
Plugin Manager. The plugin does not load maps or build geometry assets.
Its telemetry endpoint serves one client at a time, so let `track check` finish
before starting smoke or training.

The bundled `.as` source is a developer reference derived from 2.4.0; it also
contains an **experimental, disabled-by-default** protocol 3 map switcher. The
signed 2.4.0 plugin does not contain it. Do not install the loose script alongside
the managed plugin. The reference waits until both player and vehicle-visual state
exist; it never fabricates missing orientation, wheel-slip or contact-material
values with zeroes. Protocol 3 additionally requires the viewed vehicle to belong
to the local GUI player. Protocol 2 retains the earlier behavior, which can mix
local player data with a spectated vehicle's visual state. Real-game protocol-2
compatibility of this reference has not been verified.

Protocol 3 is not a supported multi-map backend yet. A prior source revision
compiled in an isolated game, but its first attempted map switch ended
indeterminate; automatic evaluation and training-map restoration remain
unverified. Later source changes require their own compilation and game tests.
Do not enable it in a game used for training.
It requires explicit enabling in plugin settings, `PlayLocalMap` permission and
exclusive control of a separate game instance. Use only trusted local map files:
the command can leave the current playground, and native map loading may fail or
crash outside the script's exception handling. The socket is localhost-only but
does not authenticate local processes; the epoch is an identity, not a security
boundary. Its ports are fixed by the reference source, and the experimental
Python factory currently requires session endpoint `127.0.0.1:9001`. It cannot
run alongside another Connect instance using those ports on the same host.
The Python factory additionally requires both `enable_experimental_multimap: true`
and `assume_exclusive_no_unobserved_map_changes: true`. The second option accepts
the assumption that this trusted instance makes no autonomous or unobserved map
changes. Sampled checks cannot establish that assumption; do not enable the path
if it does not hold. See the [experimental setup](../../../docs/distributed-map-switch-design.md).

### Preparing a separate development instance

These are preparation steps for a future isolated test, not a verified
installation procedure for this snapshot. Use a separate Windows game and
Openplanet user profile and an isolated network namespace or machine. A
second Windows user alone does **not** isolate localhost ports. Do not change
the plugin, signature mode, or settings of a game used for training. The
reference listens on fixed ports 9000, 9001 and 9002; it cannot share those
ports with another Connect instance in the same network namespace.

1. In the separate Openplanet user folder, make a new plugin directory under
   `Plugins`, for example `OpenplanetNext/Plugins/TrackmaniaRL-Connect-Dev`.
   Copy this repository's `info.toml` and `TrackmaniaRL_Connect.as` into that
   directory. The [Remote Build documentation](https://github.com/skybaks/vscode-openplanet-remotebuild)
   describes the plugin folder and manifest layout; do not run its remote
   build tools against a host with another active game.
2. In that isolated profile only, make sure the managed signed Connect
   (`SAC_GetData`) and any other Connect source are not active. The manifest
   declares `VehicleState` and `PlayerState` dependencies; check that the
   separate Openplanet installation resolves them before attempting a load.
   Do not copy or overwrite a plugin in the training profile.
3. The copied source is unsigned. Follow Openplanet's development signature
   mode instructions for that separate instance; the ordinary signed-plugin
   School Mode setup above is not evidence that this source will load. Whether
   the account and mode permit this source and `PlayLocalMap` needs to be
   established in the isolated test. If not, stop rather than change another
   game or account.
4. With no other Connect bound in the isolated network namespace, load the
   source, inspect Openplanet's compilation output and enable **Enable
   experimental map switching** in that plugin's settings. A visible setting
   or a protocol-2 handshake alone does not establish protocol-3 behavior.
   Then use the separate journal and both Python opt-ins in the
   [prototype setup](../../../docs/distributed-map-switch-design.md#prototype-setup-experimental-assumptions-required).

Do not treat LSP output as game compilation. A real test must separately
establish that the copied source loads, its declared dependencies resolve,
permissions allow local loading, and the game accepts the staged map paths.

The reference was inspected using `openplanet-lsp` 0.8.2 with the local
Openplanet 1.29.14 type database and the school signature profile. The check
exited with an unresolved `VehicleState` dependency. A negative control using
an intentionally nonexistent symbol also produced no diagnostic, so this run
does not establish that the script or its API calls compile. In-game compilation
and controlled runtime testing remain necessary.

A follow-up check supplied the official [VehicleState dependency source](https://github.com/openplanet-nl/vehiclestate/tree/edd4bbb9dd17da89fa7eeef5ee52d4c267665f2f)
in a separate temporary dependency directory. With that dependency resolved,
the same LSP reported no errors, and a separate copy with an intentionally
unknown API call did report the expected unknown-symbol error. This gives a
more useful static check for this particular tool and type database, but is
still not the Openplanet compiler or a test in game.

The client first calls protocol-3 `get_instance` to read the plugin-generated
`instance_epoch`, then durably records that value alongside its own random
lowercase 32-hex `epoch` before sending `claim`. `claim` carries both epochs;
`switch_map` carries both epochs, a fresh 32-hex `operation_id`, `map_path`, and
`expected_map_uid`; `switch_status` carries both epochs and the operation ID.
Responses include `status`, `protocol_version`, and `instance_epoch`;
operation responses also include `epoch`, `operation_id`, `state`, `map_uid`,
string-valued `ready`, and integer `remaining_operations`. Errors do not
disclose lease or operation tokens.
The session server reads each JSONL command incrementally and rejects a line
longer than 16 KiB, including its newline. A client sends one command per
connection. This bound applies to protocol 2 and protocol 3.
States
are `idle`, `accepted`, `leaving`, `loading`, `ready`, and `indeterminate`.
When experimental map switching is enabled, the separate TCP listener on
`127.0.0.1:9002` accepts one JSONL request at a time with `command: "frame"`,
a fresh lowercase 32-hex `request_id`, and boolean `wait_ready`. Only the first
request on a connection may set `wait_ready` to true, allowing a bounded wait
for the local player during reset while checking map and operation identity on
the same socket. A disconnected or mismatched connection must not be retried as
the same observation. It returns a JSONL header followed
by exactly 132 bytes containing the same 33 float32 values as port 9000. The
header identifies `protocol_version`, `request_id`, `instance_epoch`,
`operation_id`, `map_uid`, `state` and `payload_bytes`. Clients must read the
complete pair before requesting another frame. The script checks the switch
identity and local player before
and after collecting each payload without yielding between those checks. This
does not prove the game's fields are an atomic snapshot, identify loaded map
bytes, or authenticate a local peer. A receiver must reject incomplete frames
and identity mismatches. Port 9000 retains the existing protocol-2 format.
An exact retry of the current operation returns its status; a conflicting or
older ID is rejected. The operation history is bounded to 4096 IDs per epoch.
Before a suite, the actor checks that enough IDs remain for every map and one
training-map restore. An older plugin without the capacity field is rejected.

Before sending a switch command, a client must durably record the intent and
its uncertain-dispatch state. After a lost response it must query the same ID;
it must never blindly retry another operation. `ready` means the script has
issued its one PlayMap call and observed the requested UID and ready local
player across several frames, with no further call to PlayMap by that coroutine.
It is not a hash of
the map bytes or proof against other plugins or manual game actions. A timeout,
plugin reload, lost or changed instance epoch, unexpected status or unknown operation requires
durable quarantine. An indeterminate status can report the plugin phase at the
first failure (`accepted`, `leaving`, `loading`, or `ready`) for diagnosis; it
does not prove whether an asynchronous game action has completed. A reload
cannot prove that the game has stopped processing
an earlier request. A client with an unresolved operation must not reclaim the
game after a reload, even if the plugin would accept a fresh claim. Resolve that
uncertainty using a separately verified game
instance before attempting restoration or resuming collection. Protocol 2 map
checks are rejected while a protocol 3 operation is unresolved, but that does
not itself stop a legacy client from sending controls to the game.

Official references:

- [TrackmaniaRL Connect](https://openplanet.dev/plugin/sac_getdata)
- <https://openplanet.dev/docs/school-mode>
- <https://openplanet.dev/docs/tutorials/installing-plugins>
