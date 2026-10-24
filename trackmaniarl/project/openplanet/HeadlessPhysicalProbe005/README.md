> Historical home-directory prefixes are shown as `<USER_HOME>` placeholders. Replace them with your explicit local paths; original private receipts are not redistributed.

# Physical copy probe 005

This is source for an isolated, manually armed diagnostic. It was compiled and
run in Openplanet1.29.14/git6fd4200f on the pinned client in the r4 diagnostic.
The bounded run captured195 records, verified original64-byte restoration and
closed its controller. This does not qualify a simulation clock or input ACK.
Loading it only opens
`127.0.0.1:19178`. One explicit arm attempt is allowed per plugin load.

The callback reads the local player's physical source and destination **before
the copy writes its destinations**. It sends no controls and writes no game
data. `Dev::Hook` temporarily replaces seven executable bytes; the arm request
must acknowledge that operation explicitly. The callback retains at most 256
records, at most 4,096 total invocations and at most five seconds of records.
The frame coroutine then unhooks. Disable/destroy, owner/playground change,
focus loss, socket EOF and explicit stop also attempt unhook. Socket and file
failures are handled after the independent restoration attempt.
The time limit stops recording inside the callback and removes the hook on the
next `Main` turn; it is not an independent operating-system watchdog.

## Manual protocol

Connect to loopback port 19178 and read the JSON-line `hello`. After the caller
has acquired the existing comparison controller lease and reset the game,
send one line:

```json
{"protocol":"tmrl.physical-probe/v5","op":"arm","nonce":"0123456789abcdef0123456789abcdef","acknowledge_temporary_code_hook":true}
```

Generate a new 32-character nonce for the actual run. Keep the connection open.
An `armed` response is followed by `complete`, containing `output_path`, the
actual report and `original_64_bytes_restored`. A preflight rejection returns
`failed`. `status` and `stop` use the same protocol and matching nonce, without
the acknowledgement field. A subsequent connection can retrieve the saved
report through `status`. This is a diagnostic interface, not a controller.

## Bounded diagnostic client

`scripts/headless_windows_physical_probe.py` is an opt-in Windows orchestration
command. It acquires the existing global comparison controller lease, refuses
competing Python actors, verifies the actual executable and DLL hashes, uses the
existing focus/map/reset preflight and performs no model or optimizer work.
It resets the actual configured environment, arms 005, issues full gas followed
by the mildest available left steer without braking, and retains the original
Connect33 packets separately. Each phase ends after the observed race clock
advances at least 1,000 ms; this does not prove an exact consumed input duration.
The driving receive/STOP budget is five host seconds, within the overall
90-second preflight/reset budget. Controls are released and the environment is
closed on failure.

The background reader drains 005 while driving. Normal completion, body failure,
EOF and uncertain arm acknowledgement all require independent original-byte
restoration evidence. If the initial connection fails, closing it triggers the
plugin EOF cleanup and a fresh bounded status request attempts to retrieve the
actual report. Missing or mismatched restoration evidence fails the command.
Outputs are `physical005.json`, `original-connect.json` and `diagnostic.json`;
partial original telemetry remains available after interruption. Issued controls,
API observations, physical source bits and PRE-copy cache bits stay separate.

Only after the foreign controller run has ended and the matching plugin has been
compiled and explicitly loaded, the prepared invocation is:

```powershell
Set-Location <USER_HOME>/.codex/worktrees/headless-training-parity/AITrackmania
$env:PYTHONPATH = (Get-Location).Path
$env:PYTHONDONTWRITEBYTECODE = "1"
& H:/Studia/inzynierskie/inzynierkav2/AITrackmania/.venv/Scripts/python.exe -m scripts.headless_windows_physical_probe F:/AITrackmania/reports/headless-training-parity-20261021-r3/physical005-live001 --live --no-artificial-lag
```

These settings select this worktree's source and suppress bytecode writes in the
read-only interpreter environment. Use a fresh output directory. `--no-artificial-lag`
uses the separately guarded 004 lease and requires its matching loaded plugin.
Without that flag the script leaves the existing input profile in place.
No Windows controller, reset, live socket or code hook was exercised by the
Linux socket/pure tests. The client and plugin remain unqualified for live use
until the bounded measurement and restoration proof actually pass.

## Pinned static evidence

The source executable was copied without executing it:

- `D:/Epic Games/Trackmania/Trackmania.exe`, 45,467,720 bytes, SHA-256
  `3fc7d8cda542beda131c44306b123f4004d07d7e22f512b46b762afc29f6edda`.
- `D:/Epic Games/Trackmania/Openplanet.dll`, 10,573,824 bytes, SHA-256
  `f134f405979d2c2641f8f131491e3401fcdd4f4b5266cde9f1a7001a6d3b49ef`.
- Observed runtime version gates: game `2026-02-02_17_51`, Openplanet `1.29.14`,
  Openplanet commit `6fd4200f`. The plugin checks these and the exact 64-byte
  function. It does **not** independently hash the loaded binaries; the caller
  must retain the separate executable/DLL identity checks.

Private binaries and generated analysis are outside the repository, under
`<USER_HOME>/tmrl-headless-parity-20261021-003/windows-static/`:

- `physical-probe005-final-targets015.json`: hashes, byte slices and static xref
  candidates for the physical callback, cached getters and cache producer.
- `physical-probe005-hook-semantics016.json`: original bridge/length byte slices.
- `op-hook-jit-builder017.txt`: bounded objdump disassembly of Openplanet RVA
  `0x16a8f0..0x16adf4`, including SSE save/restore emission.
- `op-hook-original-first-bridge018.txt`: bounded objdump disassembly of
  Openplanet RVA `0x0964e0..0x096629`, including original-byte copy, callback
  bridge, return jump and replacement construction.

The stdlib-only scripts `scripts/headless_windows_native_inventory.py` and
`scripts/headless_windows_native_slice.py` reproduce bounded offline byte/xref
reports. Their xrefs are byte-pattern candidates; the specific instruction
boundaries below were also checked with objdump. They access no live process.

## Game code and ABI

The physical callback is game RVA `0x7d0cb0`, raw file offset `0x7d00b0`, size
64 bytes, with this exact body:

```text
0f1082e0120000410f1100f20f108af0120000f2410f1148108b82f812000041894018
f20f1082fc120000f2410f11018b820413000041894108b801000000c3
```

Its ABI has raw physical body in `RDX`, seven-word pose output in `R8` and
three-word velocity output in `R9`. It copies body `+0x12e0` quaternion (16
bytes), `+0x12f0` position (12 bytes) and `+0x12fc` velocity (12 bytes). There is
no arithmetic in this function. It returns integer one.

The local player cache producer prepares `R8 = participant + 0xe14` and
`R9 = participant + 0xe30` at game RVA `0x12ea8a9`, calling generic dispatch
`0x6c3450` at `0x12ea8c5`. That dispatcher uses callback slot `+0x80` of a
`0x120`-byte descriptor. Registration `0x7c5b00` places callback `0x7d0cb0`
at descriptor `+0x80`; `0x7d5fd0` registers it through `0x6c0520`. The runtime
type index is read by `0x7c5d30` from RVA `0x20540d8`. Before arming, the plugin
requires the corresponding raw callback table slot, using the pointer at RVA
`0x1fab3f0`, to equal the exact selected function address. It rechecks that
identity inside each accepted callback.

CSmScriptPlayer's getter dispatcher is RVA `0x1353c70`. Position member ID
`0x2d02e039` dispatches to `0x1353fcc`, calling getter `0x13536f0`; Velocity ID
`0x2d02e040` dispatches to `0x135468b`, calling getter `0x1353740`. Both receive
participant from `[scriptPlayer+0x50]`. They read participant `+0xe24` and
`+0xe30`, after the availability check `0x12ddd10` (`participant+0x21c != -1`).
The plugin also checks the managed identity of `GetOffsetNod(api, 0x50)` against
the actual local `CSmPlayer` before retaining its raw pointer.

## Hook semantics and preservation limit

Openplanet's registration at RVA `0x12517b` binds `Dev::Hook` to `0x12b320`,
which calls builder `0x16a8f0`. The builder records `padding + 5` bytes at
`0x16a927`. Its lower patcher `0x0964e0` copies those original bytes into the
trampoline at `0x096539`, appends the callback bridge, then appends the jump to
the original address plus the copied byte count. This executes original stolen
instructions **before** the callback; it does not relocate relative addresses.

For this specific function, `padding=2` copies one complete seven-byte
`movups xmm0,[rdx+0x12e0]`. It has no relative address or branch. The callback
therefore runs after that quaternion load and before the first destination
store. It does not claim POST cache state. The source verifies the actual E9
plus two NOP patch and the original seven bytes in the trampoline, then records
the observed complete prefix again after unhook.

`PushRegisters::SSE=1` enables the builder's sixteen XMM spill/restore emissions
at `0x16a95d..0x16a9db` and `0x16ace8..0x16ad63`, preserving loaded XMM0.
A [primary source example](https://raw.githubusercontent.com/XertroV/tm-editor-plus-plus/master/research/HookEditorInputs.asbak)
also uses padding two for a seven-byte original instruction and the SSE flag.

No explicit MXCSR or x87 control save/restore was found in this bridge. The
[Windows x64 ABI](https://learn.microsoft.com/en-us/cpp/build/x64-calling-convention?view=msvc-170)
requires the x87 control word and MXCSR control bits 6–15 to survive calls;
MXCSR status bits 0–5 may change. That ABI requirement is not a measurement of
this callback chain. The callback uses integer safe reads and integer metadata;
FP-control preservation must remain an explicit qualification limit until
separately observed. This collector does not write FP controls.

`Time::Now` is host metadata. Neither its simulation phase nor an input-consumed
tick is proved. PRE-copy cache values must never be presented as a current POST
API observation. Captured source and destination bits require subsequent
analysis; no parity, physical clock or applied-input acknowledgement is declared
by this probe.
The sequential integer reads are not independently proved to form an atomic
snapshot across threads. Their exact code phase and pointer identity are retained
so that qualification can assess that limitation explicitly.

## Actual r4 evidence

`F:/AITrackmania/reports/headless-training-parity-20261021-r4/physical005-live001/`
contains the actual hook report and separately acquired Connect packets.
193/195 raw position/velocity records have exact byte matches with a unique
API race time; two have no retained API match. No missing values were filled.
PRE-copy destinations were still defaults in all195 calls, so they must not be
called a previous-frame API sample. The content join is not a same-call POST
read or an independent timing proof. The collector was unloaded and its installed
source archived after testing; port19178 was confirmed closed.
