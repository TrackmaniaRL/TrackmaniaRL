> Historical home-directory prefixes are shown as `<USER_HOME>` placeholders. Replace them with your explicit local paths; original private receipts are not redistributed.

# Physical input-consumer diagnostic 007

This is an independently armed diagnostic source. Compilation, loading and a
successful live measurement must be established by a separate runtime receipt.
The source and Linux socket tests alone do not qualify its live operation.
The plugin listens on `127.0.0.1:19180` and installs no hook until an explicit
nonce-bound arm request acknowledges the temporary executable-code replacement.
It changes no game data, input state, cache, focus or settings.

## What this probe measures

The pinned target is Windows game RVA `0x841734`, the eight-byte instruction
`f3 0f 11 8b 30 14 00 00` (`movss [rbx+0x1430],xmm1`). The original-first
trampoline executes this complete position-independent store before the
callback. The callback therefore observes the filtered-steer store in the
physical vehicle update. It does not claim whole-step PRE or POST state.

The callback retains up to 256 records over five host seconds and at most 4,096
invocations. It performs only integer safe reads and integer host metadata
access, with preallocated arrays. Serialization and disk/network writes occur
outside the callback. Its 20 literal uint32 words per accepted record are:

| Report field | Source and order |
| --- | --- |
| `actual_control_bits_u32` | vehicle `+0x98` gas, `+0x9c` brake, `+0xa0` steer, `+0x1430` filtered steer |
| `vehicle_observed_input_bits_u32` | vehicle `+0x858` steer, `+0x85c` gas, `+0x868` braking integer |
| `participant_cached_input_bits_u32` | participant `+0xeec` steer, `+0xef0` gas, `+0xef4` braking integer |
| `raw_body_bits_u32` | vehicle `+0x12e0`: quaternion4, position3, velocity3 |

Braking in the latter two arrays is an integer, not a float bit pattern. The
separate `R13D_step_argument_raw` is the exact low uint32 simulation-function
argument. Its clock meaning and conversion to race time are **unqualified**.
Neither `Time::Now` nor this argument is substituted for a measured race clock.
Observed controls are not an acknowledgement that a particular issued command
was consumed. No issued-command nonce is present in this game function.

Sequential safe reads are not independently proved cross-thread atomic. This
probe does not write floating-point controls or independently measure their
preservation. The pinned Openplanet SSE save/restore bridge and ABI limits are
documented in the sibling 005 README and retained disassembly below.

## Static source proof

All RVAs refer to the offline copy of the current game executable, SHA-256
`3fc7d8cda542beda131c44306b123f4004d07d7e22f512b46b762afc29f6edda`.
The Openplanet DLL is SHA-256
`f134f405979d2c2641f8f131491e3401fcdd4f4b5266cde9f1a7001a6d3b49ef`.
The caller checks both actual files. The plugin checks game version
`2026-02-02_17_51`, Openplanet `1.29.14`, git commit `6fd4200f`, actual ASLR
bounds, the exact 64-byte prefix and its unique `FindPattern` location.

The complete pinned 64-byte prefix is:

```text
f30f118b301400008b83d8150000440f284c2450440f284424600f287c24700f28b4
24800000008983081600008b83dc15000039830c160000740d89830c1600
```

Function `0x841500` saves original RDX into RBX at `0x841521`, and original
R8D into R13D at `0x841511`. It consumes gas/brake at `0x84169e/0x8416a8`
and invokes the physical update at `0x8416b7`; it reads steer `+0xa0` at
`0x8416da`, filters it and writes the target `+0x1430`. Downstream physical
arithmetic also reads gas/brake at `0x83bd5a/0x83bd65`. One caller at
`0x852109` loads original R8D from its step-context `+0x14`, and XMM3 from
`+0x18`. Other call sites exist; this establishes the raw argument's provenance,
without establishing a global simulation-clock contract.

The local participant is obtained from `CSmScriptPlayer+0x50` only after its
managed node identity equals the actual local `CSmPlayer`. The binding code
at `0x12f2244` copies a 16-byte typed reference into participant `+0x1110`.
The immediately preceding type check uses `0x7c58d0`, whose literal class ID
is `0x0a020000`. The payload pointer is at `+0x1118`. Arm requires that class
and a bounded user-space pointer; every callback rechecks class and payload,
then accepts only `RBX == expected vehicle`. The frame loop also checks the
managed owner, playground and focus. This avoids capturing another vehicle.

The API input cache is produced through `0x134d480`: at `0x134d532` it calls
`0x72a350` with RCX=participant`+0xec0`, RDX=vehicle`+0x848`. Stores
`0x72a47d/0x72a48c/0x72a494` copy source `+0x10/+0x14/+0x20` to destination
`+0x2c/+0x30/+0x34`, yielding the offsets above. Floats pass through
`0x72a330`, which zeros magnitudes below the float32 threshold `0.01` and
otherwise returns the value. It does not explain an above-threshold one-ULP
difference by itself. This producer is an observation path, not a physics ACK.

An earlier direct-displacement candidate at `0xc579aa` was rejected: its class
ID is `0x030c9000`, whereas local `CSmPlayer` is `0x2d008000`. The queued
command dispatcher `0x12e86c0` writes a separate vehicle `+0x1750` structure;
its command enum has not been equated with actual gas/brake/steer.

Private binaries and bounded disassembly remain outside the repository:

`<USER_HOME>/tmrl-headless-parity-20261021-005/windows-input-consumer/`

- `getters003.txt`, `observed-input-cache-writer021.txt`, `cache-writer-caller022.txt`;
- `physical-steer-filter024.txt`, `vehicle-post-state025.txt`, `filter-entry031.txt`;
- `filter-caller-time034.txt`, `owner-vehicle-binding035.txt`, `owner-ref-use036.txt`;
- candidate inventories and `probe-target-xrefs032.json` retain false-positive limits.

The earlier pinned Openplanet builder/bridge slices are in ext4 round003:
`windows-static/op-hook-jit-builder017.txt` and
`windows-static/op-hook-original-first-bridge018.txt`. `padding=3` means
`5+3=8` stolen bytes; SSE flag1 spills/restores XMM0..15. RAX is safe scratch
at this point because the next original instruction overwrites EAX. The plugin
checks the actual eight trampoline bytes and the E9-plus-three-NOP replacement.

## Protocol, cleanup and runner

The separate wire is `tmrl.input-consumer-probe/v7`, incompatible with 005.
Read JSON-line `hello`, then send `arm` with a fresh 32-character nonce and
`acknowledge_temporary_code_hook=true`. Matching `stop` and `status` omit the
acknowledgement field. One arm attempt is allowed per plugin load. The reply
contains actual bounded capture evidence; 007 does not synthesize records.

The five-second limit stops recording in the callback and unhooks when `Main`
next resumes. This is not an OS watchdog. EOF, owner/playground/focus change,
disable/destroy and explicit stop also attempt restoration. Unhook, report
persistence and socket closing are independent guarded phases. Every hook
attempt retains the subsequently observed 64-byte prefix, even after an arm
failure. Success requires an absent hook handle, no unhook error and exact
original bytes before/after. Missing proof fails the Python client.

`scripts/input_consumer_probe.py` exposes `InputConsumerProbeClient`,
`verify_restoration` and `verify_capture`. It reuses the bounded 005 lifetime
and EOF/status recovery, overriding all wire/restoration methods for 007.
`scripts/input_consumer_live.py` supplies an explicit `ProbeRunContract` to
the existing guarded runner. Its default 005 contract remains unchanged.
The 007 driver requests neutral500ms, gas750ms, mildleft750ms using observed
Connect race deltas, preserving any actual overshoot. Inputs, control issue
times and literal observations remain separate. Controls are released and the
environment is closed on failure. Optional 006 snapshots are retained at phase
boundaries with exact payload hex and producer/host metadata; their additional
request scheduling is part of the diagnostic, not an atomic cross-probe join.

Only the root operator starts the prepared command after compiling/loading the
matching plugins and acquiring the controller window:

```powershell
Set-Location <USER_HOME>/.codex/worktrees/headless-training-parity/AITrackmania
$env:PYTHONPATH = (Get-Location).Path
$env:PYTHONDONTWRITEBYTECODE = "1"
& H:/Studia/inzynierskie/inzynierkav2/AITrackmania/.venv/Scripts/python.exe -m scripts.input_consumer_live F:/AITrackmania/reports/headless-training-parity-20261021-r5/input-consumer007-live001 --live --no-artificial-lag --fresh-telemetry006
```

Use a fresh output directory. 004 must be loaded for the optional input-profile
lease, and 006 for fresh telemetry. Outputs are `input-consumer007.json`,
`original-connect.json` (including separate `fresh006`) and `diagnostic.json`.
Linux tests exercise mock sockets and pure validators, not a game or code hook.

The optional `--phase-profile long-mild-left` uses nominal phase goals
150/150/2000ms (2300ms total). The default500/750/750ms remains unchanged.
Both preserve observed duration overshoot and the same256-record/5-second
collector cap; neither guarantees that every issued phase is captured.
Actual runs `input-consumer007-live002` and `live003` captured199 and228
records respectively and restored the original64 bytes, input profile and
controller. The first attempt `live001` refused competing test Python processes
before touching the game. Installed source folders were then archived and all
own probe ports stopped listening. These are observed diagnostics, not a
production issued-command acknowledgement protocol.
