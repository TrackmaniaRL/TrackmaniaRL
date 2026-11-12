# Experimental distributed map switching

Status: experimental opt-in; real-game validation remains outstanding.
This path assumes exclusive control of a trusted game instance and no
autonomous or unobserved map changes. Continuous map identity is not proven.
The default protocol-2 and single-map paths retain their existing behavior.

## Evidence and constraints

The checked-in Connect source is a reference snapshot of the separately
distributed signed plugin. Its protocol 2 exposes the active UID and player
readiness, but has no map-load command. It uses substring matching for its two
commands; a new protocol must parse JSON instead. No local Openplanet compiler or
isolated game instance has been verified for this work.

The [tm-control-mcp implementation at commit 04e713b](https://github.com/clankercode/tm-control-mcp/blob/04e713bb4c18cca8bd792d2f3185c6ac9688e767/src/McpTools.as)
provides evidence that `ManiaTitleControlScriptAPI.PlayMap` is callable in a
plugin when `Permissions::PlayLocalMap()` is granted. Its `PlayMapCoroutine`
leaves the current playground, waits for menu readiness and then invokes the
API. Its acknowledgement means queued, not completed. This is evidence of a
possible mechanism, not proof of availability in School Mode, of compilation
against our plugin dependencies, or of successful loading on a user's game.

## Required protocol

Implement a new explicitly negotiated plugin protocol alongside protocol 2;
do not change 2's meaning or silently enable switching. Introduce an exclusive
session lease, a random plugin-instance epoch and client-generated operation
IDs. A switch command must bind its ID to the lease, the game-local canonical
map path and expected UID. A duplicate ID with identical arguments returns the
same operation; reuse with different arguments is an error. Exactly one switch
may be active. Concurrent requests, missing permission, unsupported game mode,
or a lost lease fail closed. All requests and responses use parsed JSON and
strict length and field limits.

The plugin owns one state machine and tracks its launch coroutines. It records
the operation before initiating any game side effect,
then exposes `accepted`, `leaving`, `loading`, `ready`, or `indeterminate`.
`ready` requires that the worker can no longer issue a load, the active map UID
matches the request, and the local player is ready. Before `ready`, disconnect,
timeout, cancellation, and an unexpected UID never authorize a second load or
collection. The implementation must establish whether the game API permits a
positive terminal failure; if a request may still execute, report
`indeterminate` and quarantine the session. A status query must remain available
while the worker progresses. Acknowledging queueing is not completion.

The actor persists the operation ID, lease and plugin epoch before sending the
command. After a lost response it queries the same operation without submitting
a new ID. If the epoch changes, the status cannot be established, or a previous
load might still be pending, the actor stops and requires explicit recovery of
the isolated game instance. An epoch change alone cannot prove an old engine
load has stopped. Durable actor records must survive an actor restart and must
not share files or ports with another training session. This protocol assumes
exclusive control of the game instance; manual map changes and other plugins
that launch maps invalidate the lease and must not be treated as verified.

`trackmaniarl.trackmania.map_switch_journal` provides durable storage for the
experimental controller. Its SQLite transactions
serialize intents and persist `dispatching` before a caller sends a command.
Open a previously created journal without `create=True`; a missing or existing
file cannot be silently replaced. The caller must keep its path and instance
identity stable across restarts and must not recreate a missing file for a
possibly active instance. Every controller of one game instance must use the
same database: two different files can each accept an operation, and this
controller additionally binds its fixed endpoint to a journal in a per-user
registry; this is not a physical game-process identity. SQLite EXTRA
synchronous mode requires a local filesystem with working locking and sync.
Process-exit tests do not prove
resilience to power loss or faulty storage.

Before recording a switch intent, the controller creates and flushes a separate
operation marker in the per-user registry. It refuses a new switch if either the
marker or journal is unresolved, and requires matching operation IDs to recover.
This prevents restoring an older journal alone from silently erasing an uncertain
dispatch. A crash after the terminal journal write but before marker removal can
be recovered only by rechecking that operation, instance, UID, player and file.
A corrupt marker, a missing marker with a pending row, or conflicting state stops
the controller. POSIX also syncs the registry directory; Windows currently flushes
the marker file but directory-entry durability after power loss is unverified.
Changing user profile, deleting or jointly rolling back the registry and journal,
or uncooperative local software lies outside this protection.

`resolve_terminal` records evidence supplied by the caller but does not verify
its authenticity or meaning. Only a backend that verifies the exact
operation and epoch and proves its worker cannot issue a later load may call
it. Queue acknowledgements, matching map UID alone, timeouts and plugin epoch
changes are not terminal evidence. Manual or third-party game controls do not
use this journal and are outside its protection.

Before consuming bound telemetry, the experimental actor also writes and
flushes a separate session guard in the instance registry. The owning session
retains it through evaluation, restoration and subsequent training. A clean
shutdown releases controls, closes telemetry, verifies the training map and
removes the guard; an observed error or process exit leaves it in place and
blocks automatic restart or recovery. Explicit recovery requires first
stopping the isolated game and determining the state; a ready status by itself
does not clear this guard. As with the switch marker, Windows directory-entry
durability after power loss is unverified.

## Evaluation and restoration

At admission, resolve each local map and geometry/reward asset and bind their
UIDs and content hashes to the durable assignment. Require an explicit training
map path, UID and matching assets for restoration. Reject missing checksums,
untrusted map locations and unsupported map-dependent feature pipelines.
Map-file hashing and a matching reported UID bind configured inputs, but are
not by themselves cryptographic proof of the bytes loaded by the game; document
that limitation and validate the actual load mechanism in a controlled game.

The current multi-map assignment path records a digest of the suite identity,
map UIDs and declared map/reward/geometry file contents in the durable
assignment. The actor compares its local files before and after the suite;
accepted WAL rows retain the coordinator's assigned digest for recovery. This
is only file-content binding. In particular, it neither proves which bytes the
game loaded nor detects a file replaced and restored between the two actor
checks. The prototype described below has these limitations. Older
single-map assignments and journals retain their existing format.

An experimental OpenPlanet path stages map and reward
assets in a new Windows directory, denies writes and deletes while handles are
open, and compares staged suite contents with the assigned digest. It retains
the staged files after shutdown because an uncertain game load may outlive the
actor. The fixed suite is reused within one actor session; a limit on retained
directories and bytes stops further staging rather than deleting uncertain
files. Explicit cleanup requires confirming that the associated isolated game
instance has stopped and no pending load can still consume those files.

The controller binds the fixed Connect endpoint to one journal path in a
per-user registry and holds an OS lock across the game session. The
experimental factory only accepts `127.0.0.1:9001`, matching the checked-in
plugin. The plugin does not prove an operating-system game-process identity;
other accounts, a modified plugin using another endpoint, or uncooperative local
software can bypass the registry. Any future use of the experimental path
requires a separately owned game, the unverified protocol-3 plugin, and explicit
both `enable_experimental_multimap: true` and
`assume_exclusive_no_unobserved_map_changes: true`. The second option explicitly
accepts the assumption that the trusted instance has no autonomous or
unobserved map changes during the session. If that assumption is false, do not
use this path: matching observations cannot establish continuity.
Windows file locks, source
hashes, and UID/readiness checks do not themselves prove the game loaded the
expected bytes.

The prototype requests each telemetry frame with a fresh request ID and checks
the associated plugin epoch, switch operation and UID before returning it. A
missing, partial or mismatched frame aborts the suite without publishing results.
For the first frame after an episode reset, the client requests a bounded wait
for player readiness on the same connection. An EOF is never retried: it may
also signal an identity failure. Later frame requests do not wait for readiness.
These sampled checks do not detect an A→B→A map change between observations,
and collecting several engine fields without yielding is not proof of an atomic
engine snapshot. Observed mismatches abort collection; the explicit assumption
is necessary for changes that are never observed. This does not establish
support in a real game, and the actual plugin remains to be validated.

The following alternatives were examined but do not establish that guarantee:

- Openplanet 1.29.14's local `OpenplanetNext.json` exposes
  `CGamePlaygroundScript.MapLoaded` as a boolean. The older
  [ManiaScript reference](https://maniaplanet.github.io/maniascript-reference/doc_8h_source.html)
  likewise describes a boolean. Sampling it cannot distinguish an unchanged
  map from an unload and reload between observations; the older reference is
  not evidence of current Trackmania behavior.
- [TMEvents' map tracking](https://github.com/brekekekex001/TMEvents/blob/de65dc4d1c186fb345465fdf6b92ae4ba4da0216/Main.as)
  compares the current UID during `Update`. A→B→A between updates is invisible.
  Its event interface does not turn those observations into continuous proof.
- Openplanet 1.29.14's local `OpenplanetCore.json` describes
  `RegisterLoadCallback` as notification of loaded nods of a given class; an
  [example callback](https://github.com/XertroV/tm-birds/blob/4ca2d9a575d6b2ad79de43eb6687391d2e1d9fe7/src/Main.as#L69-L72)
  describes a nod loaded from a file. We found no guarantee that every map
  activation loads such a nod, or that the callback is ordered with activation
  and telemetry. Background loads and reuse also need investigation. A counter
  of these callbacks therefore cannot attest continuous map identity.

Removing the experimental assumption requires evidence for a signal covering every relevant map
activation, including reuse and A→B→A, with no lost events and an ordering that
binds it to the sampled telemetry and instance; an equivalent verified guarantee
would also suffice. Validate the mechanism and its failure behavior in an
isolated real game, in addition to protocol tests. An observed callback or a
passing simulated A→B→A test alone does not establish that coverage.

The plugin retains at most 4096 operation IDs for its epoch. Before staging or
leaving the training map, the actor checks the authenticated status response
has enough remaining IDs for every evaluation map and the training-map
restore. It fails closed if capacity is missing, invalid, or insufficient.
This relies on exclusive ownership of the plugin instance throughout the suite.

The experimental multi-map actor currently refuses a configured
`actor_stall_timeout_s`: staging and the game switch do not yet expose enough
progress information to distinguish a slow valid operation from a hang. This
restriction must be resolved before enabling the backend for such runs.

### Prototype setup (experimental assumptions required)

Use a separate, exclusively controlled Windows game instance, a development
installation of the protocol-3 Connect source and local, checksum-bound map
and reward assets. Its compilation and runtime have not yet been verified.
Create an empty local staging directory and a parent directory for the journal.
For the CLI, relative paths in the example below are resolved relative to the
directory containing the run configuration. From that same directory, and for
a genuinely new, isolated game instance only, initialize the journal once:

```powershell
Set-Location -LiteralPath 'C:\path\to\directory-containing-run-yaml'
python -c "from pathlib import Path; from trackmaniarl.trackmania.map_switch_journal import MapSwitchJournal; MapSwitchJournal(Path('isolated/switch.sqlite'), 'isolated-game', create=True)"
```

Creation fails if the path already exists. Never recreate or replace a journal
after an uncertain operation: its pending record is the quarantine. Keep the
same absolute-resolving path, endpoint and `instance_id` across actor restarts.
If invoking the SDK with a different `base_dir`, resolve the initialization
path against that `base_dir` instead, or use the same absolute path in both
places. See the [separate development-instance preparation](../trackmaniarl/project/openplanet/README.md#preparing-a-separate-development-instance)
for the unsigned plugin, dependencies and fixed-port constraints.
An example environment component is:

```yaml
components:
  environment:
    class_path: trackmaniarl.trackmania.multimap_environment:OpenPlanetMultiMapEnvironmentFactory
    kwargs:
      config:
        expected_map_uid: TRAINING_MAP_UID
        reward_points_path: assets/training-reward.npz
      training_map_path: assets/training.Map.Gbx
      journal_path: isolated/switch.sqlite
      instance_id: isolated-game
      staging_root: isolated/staging
      enable_experimental_multimap: true
      assume_exclusive_no_unobserved_map_changes: true
```

Replace the example paths and UID with real assets. The rest of the run must
configure the supported value policy, a reviewed telemetry pipeline and an
evaluation suite with explicit map files, UIDs and per-map reward or geometry
assets. `actor_stall_timeout_s` must remain unset. Switching also starts on
initialization to verify the training map, so do not point this at any game
used by another process or person. The staged files are retained after shutdown;
remove them only after ensuring the isolated game has stopped and no load can
still be pending.

`trackmaniarl validate` runs a synthetic learner update; it does not perform
multi-map actor admission or check the switch journal. The actor checks the
opt-ins, assets and supported policy during initialization before it creates
the game session. It opens the previously provisioned journal during session
creation, before dispatching a switch.

Connect allows up to 30 seconds to leave the old map and another 30 seconds
to load the new one. The controller's default 75-second polling budget covers
both phases with some scheduling margin. A shorter explicitly supplied timeout
may quarantine a switch that the plugin is still legitimately processing;
it does not cancel the game's request.

The coordinator assigns one immutable policy snapshot and suite identity. For
every map the actor verifies the assets, switches and confirms readiness, then
constructs a fresh environment and a separately attested feature pipeline.
Each result carries assignment, policy version, map ID, UID, asset digests and
trial index. Coordinator admission checks the exact map-by-trial Cartesian set,
rejecting duplicates, missing or extra trials, mismatched assets and snapshots.
Only a full suite can influence promotions or stop criteria. Preserve the
existing single-map result contract and replay compatibility during migration.

On normal completion or an interruption after a **terminal** switch, release
controls, close evaluation environments, switch back to the training map, and
confirm UID and readiness before the training environment can reset. A failed
or indeterminate switch is not safe to undo with another queued command: stop
the actor without resuming collection and retain the assignment for recovery.
If restoration itself fails, likewise stop without publishing a successful
suite or resuming collection. A retry must use the same assigned policy and
must not count partial trials twice.

For an experimental multi-map assignment, a stop observed after restoration or
while waiting for spool capacity leaves the assignment pending. The last stop
check occurs immediately before entering the durable spool write. A stop racing
that write may still produce a complete result: the stop event and filesystem
commit are not atomic with each other. Training rollout shutdown behavior is
unchanged.

## Verification gates

- Protocol tests: parsed requests, invalid fields, permission refusal, duplicate
  and conflicting IDs, simultaneous clients, lost acknowledgement and status
  polling, cancellation at each state, late loads, restart and epoch change.
- Actor/coordinator integration: interruption and replay, exact trial accounting,
  snapshot and asset mismatch, pipeline isolation, failure during restore,
  and durable quarantine across actor restart.
- Compile and test the actual plugin; publish a compatible signed release or
  explicitly document supported development installation and version negotiation.
- On a separately owned game instance with trusted real maps and isolated ports,
  verify both map transitions, UID/readiness, reward and pipeline binding,
  restoration, and interrupted/failed switching. Preserve evidence; simulated
  tests alone cannot establish real-map support.
