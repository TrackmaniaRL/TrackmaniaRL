> Historical home-directory prefixes are shown as `<USER_HOME>` placeholders. Replace them with your explicit local paths; original private receipts are not redistributed.

# R6 local pause / single-physics-step API inventory

No documented local-world pause plus single-physics-tick operation was found in
the inspected Openplanet declarations. The available names do not establish a
way to hold the live physical world while Python computes, then advance exactly
one accepted input interval. This is an inventory result for the pinned local
snapshots, not a proof that no private engine implementation exists.

Only local source/reflection evidence was read. No game was started, no live
process accessed, no procedure invoked, and no setting, clock or thread changed.
No R5 file was edited.

## Sources and reproducible scope

The structured receipt is
`F:/AITrackmania/reports/headless-training-parity-20261022-r6/pause-api-inventory001.json`,
also retained at
`<USER_HOME>/tmrl-headless-parity-20261022-006/pause-api/pause-api-inventory001.json`.
Its SHA256 is
`84ef74189cd1307997d2e02941670941cac875f80140da7db26b2b2ff5b37f2f`.
It records the exact matching expression, declarations, header line numbers,
class IDs, member indices and source hashes. It examined 2,585 game classes
and 16,321 declared members, retaining 51 matching game members, 46 matching
header lines, and four matching core declarations. Broad searches included
pause/resume, freeze, time scale/factor/coefficient, simulation time/speed,
single step, frame advance, replay speed, timeline and chrono operations.

The source files under `<USER_HOME>/OpenplanetNext` identify Openplanet
`1.29.14 (next, Public, 6fd4200f)`:

```text
Openplanet.h        16b50c345edebd9bd139ba596a1401b83f1a2d0932a6a50912bc43bc25ce0332
OpenplanetCore.json c0cbaa483644a29f5151cf40365300d1cf82a4738cfb6161187adc752bfb84c9
OpenplanetNext.json 9d5ce8d9bfeed57f878674efe213fe5abaa04759d2dae0302e3c8e867cdf74df
```

The prior actual `runtime-probe003-live001.json` reflection receipt was also
inspected for matching member names and offsets. This is the existing R3
capture, not a newly generated live dump.

## Closest declarations

| Exact local declaration / owner | Static member index | What the source establishes |
| --- | ---: | --- |
| `const uint CMwCmdBufferCore::SimulationTime` | 4 | Read-only simulation clock declaration; no step operation |
| `const float CMwCmdBufferCore::SimulationRelativeSpeed` | 5 | Read-only speed declaration; no setter |
| `float CSceneVehicleVisState::SimulationTimeCoef` | 62 | Writable declaration in visual vehicle state; physical-world control is unproved |
| `float CSceneVehicleVisState::BulletTimeNormed` | 61 | Visual-state field, range0..1; no pause/step guarantee |
| `bool CGameManiaplanetPlugin::RequestStartUpFreeze` | 41 | Startup/plugin request; source does not promise a local physical tick barrier |
| `float CGameCtnMediaClipViewer::PlaySpeed` | 4 | Media clip viewer state; no live-world step guarantee |
| `float CGameEditorMediaTrackerPluginAPI::PlaySpeed` | 7 | Editor/MediaTracker playback state |
| `void CGamePlaygroundModuleServerChrono::StopChrono()` | 0 | Chrono operation; does not declare stopping physics |
| `void CGamePlaygroundModuleServerChrono::StartChrono()` | 1 | Chrono operation; does not declare one accepted physical interval |

`Openplanet.h:57..70` declares the complete `CMwCmdBufferCore` surface, including
the two const properties. The other mutable properties there are
`PeriodEstimated`, `DeltaSmoothed`, `PeriodSmoothing`, and `DeltaSmoothing`.
Their names and declarations do not authorize setting a physical timestep.

These exact timeline methods are exposed on `CGameEditorTimeLine`, starting at
header line9765:

```cpp
const uint Time;
void TimeStop();
void TimeSlowForward();
void TimeNormalForward();
void TimeFirstFrame();
void TimeLastFrame();
void TimePlayStop();
```

They belong to an editor timeline. The declarations provide no live local
physics-world pause or next-physics-tick contract. Similarly, MediaTracker
exposes `void TimePlay()`, `void TimeStop()`, `void TimeToggle()` and
`void Rewind()` beside playback/camera controls at header lines12023..12033.
First/last frame and rewind are not evidence of one physical input interval.

The mutable `GameTime` on `CGameScriptHandlerPlaygroundInterface` and its
`ReadOnly`-named sibling belongs to the GameModes Manialink interface. A writable
clock-like field is not proof that engine integration stops or advances once.
`CGameCtnReplayRecord::HumanTimeToGameTimeFunc` is a replay-record function field.
Neither is used as a physics actuator here.

## Explicitly rejected false matches

The only core pause methods are `Audio::Voice::Pause()` and
`Audio::Voice::IsPaused()`. The broad search also finds
`UI::TableSetupScrollFreeze(int cols, int rows)` and
`UI::ListClipper::Step()`, whose documented purpose is table/list rendering.
They supply no physical synchronization.

Game `Pause()` declarations belong to `CGameVideoScriptVideo`, `CPlugFileVideo`
and `NGameScriptDebugger_SWatchedVariable`; download types expose const
`IsPaused`. Other matches concern particle camera motion, scene lighting/day
time, animation clips, character frozen flags, input-axis configuration and
camera effects. No inference of world pause is made from those names. Menu
resume callbacks were observed in the header but were not invoked or treated
as a menu-free physical-step API.

## Actual reflection and byte-offset limits

The existing R3 reflection receipt contains these additional `CInputPort`
members, absent from the public header/Next declaration list:

| Reflected name | Actual reflected slot | Reported offset |
| --- | ---: | ---: |
| `InputsReplay_Pause` | 23 | `65535` / `0xffff` |
| `InputsReplay_Resume` | 24 | `65535` / `0xffff` |

They are named for input replay, not world simulation. Their signatures and
effects are not established by the captured name/offset metadata. `0xffff`
is not a usable data-member byte offset and supplies no callable address.
The adjacent `InputsReplay_Replay` field was reported at byte offset320
(`0x140`), which is a different member and cannot stand in for either operation.

The `i` values in `OpenplanetNext.json` are member indices, **not byte offsets**.
The prior R3 requested-type list contains neither `CMwCmdBufferCore` nor
`CSceneVehicleVisState`, so it supplies no actual runtime byte offsets for their
clock/coefficient fields. No offsets are fabricated, calculated from the indices,
or borrowed from unrelated members. All matching static rows explicitly report
their verified byte offset as unavailable.

## Consequence for the Python actor contract

This evidence does not permit replacing the remaining command/consumption work
with an API pause/step loop. A valid pause route would need to prove that the
physical state and consumption count remain fixed while Python computes, that
the script/transport still runs, that exactly the requested physical intervals
occur on resumption, and that pause/resume/termination restore correctly without
altering physics or borrowing a UI/frame clock. None of those properties was
measured or promised by the inspected declarations.

`Time::Now`, UI frames, editor playback time and a changed race/chrono display
remain separate observations. The inventory grants no physical pause, one-tick,
issued-command ACK, or paired-training qualification.
