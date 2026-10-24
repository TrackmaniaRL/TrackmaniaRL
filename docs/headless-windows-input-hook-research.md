> Historical home-directory prefixes are shown as `<USER_HOME>` placeholders. Replace them with your explicit local paths; original private receipts are not redistributed.

# Windows simulation-input hook research

The static Openplanet API and installed plugin sources expose mechanisms for
intercepting reflected procedures and native code. They do **not** identify a
qualified Trackmania simulation-input setter or an applied-tick acknowledgement.
The remaining concrete research target is the client function that consumes a
mapped input record and the corresponding physical simulation boundary.

No game control, live plugin edit, focus-policy change or native hook installation
was performed for this inventory.

## Exact local evidence

The inspected files are the existing `<USER_HOME>/OpenplanetNext/Openplanet.h`,
`OpenplanetCore.json` and `OpenplanetNext.json`, dated 19 October 2026. The core
dump identifies `1.29.14 (next, Public, 6fd4200f)`; the game dump identifies
Openplanet `1.29.14` and game reflection version `2026-02-02 17:51:19`. These are
static snapshots, not a newly generated live reflection dump.

The reproducible, read-only inventory command is:

```powershell
python -B scripts/headless_windows_hook_inventory.py `
  --snapshot-root <USER_HOME>/OpenplanetNext `
  --output F:/AITrackmania/reports/headless-training-parity-20261021/windows-static-hook-inventory001.json
```

The output path must be fresh. The inventory records file hashes, modification
times, relevant reflected class members, exact hook API declarations and source
contexts inside the installed `.op` ZIP files. It does not extract or load plugins.

## Available hook mechanisms

`OpenplanetCore.json` contains these exact declarations:

```cpp
Dev::HookInfo@ Dev::Hook(uint64 ptr, int padding, const string&in func,
    int pushRegisters = 0, Dev::FreeRegister freeRegister = FreeRegister::Rax)
void Dev::Unhook(Dev::HookInfo@ hook)
void Dev::InterceptProc(const string&in className, const string&in procName,
    Dev::ProcIntercept@ func)
void Dev::InterceptProc(const string&in className, const string&in procName,
    Dev::ProcInterceptEx@ func)
bool Dev::ProcIntercept(CMwStack&in)
bool Dev::ProcInterceptEx(CMwStack&in, CMwNod@)
```

`Dev::ResetInterceptProc` has matching callback-specific overloads. A new isolated
plugin should unregister its own callback rather than reset all interceptors for
a procedure used by another plugin.

`InterceptProc` observes reflected procedure dispatch through `CMwStack`. The
snapshot does not prove that an engine-internal direct native call goes through
that dispatch path. A device procedure interceptor may therefore receive zero
calls during actual driving; that result would not identify a missing input tick.

The local description for `Hook` says its callback can receive parameters named
for x86 registers, including a node handle named `rcx`. The available enums are
`PushRegisters::Basic = 0`, `PushRegisters::SSE = 1`, and free-register choices
`Rax = 0`, `Rbx = 1`. `HookInfo` exposes no additional public methods or fields.
The callback ABI, instruction boundary and padding for the selected client
function still require verification. `Unhook` must run outside that callback.

`Dev::BaseAddress`, `BaseAddressEnd`, `FindPattern`, `SafeRead*`, `Read*`, `Write*`
and `SafeWrite*` also exist. Their presence does not establish the meaning or
writability of a simulation input record. `CMwStack::CurrentFloat`, `CurrentUint`,
`CurrentBool`, and other accessors return references; they can only help when a
relevant reflected procedure and its argument semantics are actually known.

## Input surfaces and their limits

| Surface | Exact local finding | What it establishes |
| --- | --- | --- |
| `GetApp().InputPort` / `CInputPort` | `ConnectedDevices`, `Script_Pads`, `EventInStoreCount`, device/timing statistics; only exposed procedure is `DeviceHotPlugUpdate()` | Input-port identity and device/queue diagnostics; no public simulation setter or tick callback |
| `CInputDevice::ReadHardwareCurState()` | Exposed procedure, inherited by device classes | A bounded read-only `InterceptProc` candidate for device-poll ordering; application to physics remains unknown |
| `CInputScriptManager::PendingEvents` | Read-only event buffer; `Now`, `Period` and pad state available | Script/device-event observation |
| `CInputScriptEvent` | Sole exposed event enum is `PadButtonPress`; fields include `Pad`, `Button`, `IsAutoRepeat`, `KeyCode`, `KeyName` | No analog record or applied simulation-tick acknowledgement |
| `CInputScriptPad` | `LeftStickX/Y`, `RightStickX/Y`, `L2/R2` and button values are `const` | Device sample observation, not a writable driving API |
| `CSmScriptPlayer` | `InputSteer`, `InputGasPedal`, `InputIsBraking` are `const` | Existing endpoint telemetry |
| `CSceneVehicleVisState` | Input fields are writable in the declaration, alongside visual state | Visual-state fields; excluded as a physics-input actuator or proof |
| `CSmPlayerDriver::Scripted_RequestInput` | Class is documented as the ShootMania bot API; enum concerns actions such as weapons and movement | No established Trackmania steer/gas/brake API |
| `CInputReplay::NbEvents` | Read-only event count | Replay metadata; no live event append interface |

`Time::FrameCount` explicitly counts frames processed by Openplanet and can differ
from the game's frame counter. Neither it nor `Update(float dt)` proves a physical
10 ms boundary. `CMwCmdBufferCore::SimulationTime` is exposed as a read-only clock,
but its binding to the selected input-consumption call remains to be measured.

## Installed plugin source results

- **MLHook 0.5.5**: hooks reflected custom-event procedures in
  `HookManialinkCode.as:15-18`: `CGameManiaApp::LayerCustomEvent`,
  `CGameManiaAppPlayground::SendCustomEvent`,
  `CGameManialinkScriptHandler::SendCustomEvent`, and
  `CGameEditorMainPlugin::SendPluginEvent`. Its callbacks read `CMwStack` fields.
  The input inspector polls `InputMgr.PendingEvents`; it does not hook a physics
  input consumer. No installed source calls `Dev::Hook`.
- **MLFeed: Race Data 0.6.4**: consumes Manialink event feeds. No native hook or
  `InterceptProc` call was found in its installed source archive.
- **PlayerState 1.3.1**: updates state through `Update(float dt)` and race data.
  No native hook or `InterceptProc` call was found.
- **TrackmaniaRL Connect 2.4.0**: serves endpoint samples. Its telemetry loop reads
  `VehicleState::ViewingPlayerState()` and `CSmScriptPlayer` input fields. It
  provides map/readiness commands, not an action-sequence/tick acknowledgement.

The repository's `GamepadController._apply` updates a virtual XInput device and
returns after the device report. The brake-tap implementation uses a host sleep.
Neither operation establishes which simulation ticks consumed the action.

The [upstream MLHook source](https://raw.githubusercontent.com/openplanet-nl/mlhook/master/src/HookManialinkCode.as)
corroborates that its hooks route custom events and poll script event buffers.
The [project README](https://github.com/openplanet-nl/mlhook) describes Manialink
messaging and injection, without promising a physics-tick input interface.
Official `Dev::Hook` documentation returned HTTP 403 during this research; exact
API details above therefore come from the hashed local core dump.

## Bounded candidate plan

1. In an isolated read-only plugin, record device-poll procedure calls and the
   available simulation/race clocks into a fixed-capacity ring buffer. Cap the
   run at five seconds or 1,000 records. Keep socket/file I/O outside callbacks.
   If the engine reaches that reflected procedure, this can establish ordering;
   otherwise record zero calls. Neither outcome acknowledges applied controls.
2. Use the pinned client build and REA analysis to locate the mapped-input
   consumer reached from `CInputPort` and its downstream physics boundary. Bind
   the candidate to the executable hash and a uniquely matching instruction
   signature. Prove original instruction length, register arguments, object
   identity, input layout and clock before installing a native read-only hook.
3. Capture **both** the literal consumed input and the next physical endpoint:
   monotonic sequence, actual simulation timestamp, owning player/vehicle,
   steer/gas/brake float bits and position/velocity float bits. An acknowledgement
   becomes valid only after the requested number of actual consumption/boundary
   pairs is complete; a command receipt or sampled visual field is insufficient.
4. Only after those observations identify a genuine writable pre-consumption
   record should an actuator be implemented. It must bind a fresh action sequence
   to that record and return the measured applied interval count. Brake tap must
   be released by an observed simulation interval rather than host sleep.

No writable input record or applied acknowledgement was qualified by the API
and plugin-source audit above. The subsequent pinned PE analysis found the
physical raw-to-cache copy callback and cached API getter dispatches; their exact
RVAs, bytes, ABI, hook semantics and qualification limits are recorded in
[Physical probe 005](../trackmaniarl/project/openplanet/HeadlessPhysicalProbe005/README.md).
The source remains a bounded diagnostic and does not establish an input-consumed
simulation tick or an actuator.
