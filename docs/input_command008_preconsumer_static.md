> Historical home-directory prefixes are shown as `<USER_HOME>` placeholders. Replace them with your explicit local paths; original private receipts are not redistributed.

# R6: proposed 008 command hook before the Windows input consumer

This is an offline construction plan, not an installed plugin or an applied
command acknowledgement. No game process was opened, modified or controlled.
The proposed PRE point is suitable for a narrowly scoped diagnostic of
`0x841500`. It has not been qualified as the beginning of the entire physics
step. Earlier caller input reads and unresolved callback teardown prevent that
stronger claim.

## Exact PRE point and ABI

The private executable remains SHA256
`3fc7d8cda542beda131c44306b123f4004d07d7e22f512b46b762afc29f6edda`,
45,467,720 bytes, image base `0x140000000`, version
`2026-02-02_17_51`. The candidate is RVA **`0x841553`**, file offset
`0x840953`. Its first instruction is the complete eight-byte sequence:

```text
48 89 84 24 c8 00 00 00    mov qword ptr [rsp+0xc8], rax
```

The literal 64-byte prefix is:

```text
48898424c8000000e8d0cef7ff448bbb800300004c8da3f01c0000f30f103552de4d0133ed450f57c00f57ff8bf54585ff0f84b5000000440f29542440488dbb
```

At `0x841511`, the function saves original R8D into R13D. At `0x841521`, it
saves original RDX into RBX. Thus the callback can bind `uint64 rbx, uint64 r13`
and validate the same participant→typed vehicle reference used by 007. R13D is
still the literal function argument, not a globally qualified race clock. The
selected instruction only stores the already loaded vehicle configuration
pointer into the existing frame; it does not consume an input channel.

No conditional branch or call precedes this point after the function entry.
The first call is `0x84155b→0x7be430`. All paths within the inspected function
therefore pass PRE before its local gas/brake/steer reads and before the wheel
consumer calls. The eight stolen bytes contain neither a relative branch nor a
RIP-relative address. With the pinned Openplanet implementation, `padding=3`
copies exactly eight bytes. The original-first trampoline executes them before
its own pushes/stack adjustment, so this stack-relative store sees original RSP
and RAX. The callback bridge subsequently restores registers and resumes at
`0x84155b`. Common's independent bridge review confirmed this order. Both code
sites must still verify exact original/trampoline/replacement/restored bytes in
any actual construction. The observations of successful 007 restoration do not
prove teardown safety for a future two-hook writer.

## Consumer branches that the receipt must expose

PRE must precede the first call, rather than just the obvious later scalar
loads. At `0x8415b2`, a wheel loop invokes `0x843190`. Its entry preserves
vehicle RCX in R8 (`0x84319d`). The same vehicle's gas is read at
`0x843230`, `0x8432cb` and `0x8434b8`; brake is read at `0x84327a` and
`0x843476`. These reads are already after PRE553. Wheel count `vehicle+0x380`
can bypass the loop. Other vehicle/wheel mode branches can bypass particular
uses, so writing a channel does not prove that every arithmetic branch used it.

The later scalar branch converges as follows:

| Point | Behavior after PRE553 |
| --- | --- |
| `0x841649→0x83db80` | If low two bits of either `vehicle+0x176c` or `vehicle+0x1370` equal2, use float1 instead of a gas/brake load. |
| `0x841686→0x83db50` | Otherwise, if either low pair equals1, use float0. |
| `0x841695` | Otherwise, nonzero `vehicle+0x1600` selects brake at `0x84169e`; zero selects gas at `0x8416a8`. |
| `0x8416b7→0x8447b0` | Receives the selected float; it has additional state-dependent processing. |
| `0x8416c3→0x83dbd0` | Bit8 in either `vehicle+0x176c` or `vehicle+0x1370` selects float0 instead of the steer load. |
| `0x8416da` | Otherwise load raw steer from `vehicle+0xa0`. |
| `0x8416f2..0x841734` | Rate-limit/filter against previous `vehicle+0x1430`; store filtered steer at `0x841734`. |

The helpers use `RCX=vehicle+0x1280`; their local displacements `+0x4ec`
and `+0xf0` therefore resolve to `+0x176c` and `+0x1370`. These flags and
`+0x1600` should be retained with every diagnostic write and completion. A
future writer must not force a mode, patch a branch, or overwrite filtered
steer/cache merely to manufacture an ACK. It writes only the literal three raw
control fields and reports any gating as observed.

## Float brake and other writers

Raw `vehicle+0x98/+0x9c/+0xa0` contain float32 gas/brake/steer, as demonstrated
by the qualified vehicle-based scalar loads and floating arithmetic above. A
raw brake1 must be **`0x3f800000`**, never integer `0x00000001`. Integer zero stores at
`0x83deb3/0x83df00` correctly produce float positive zero. The observed/cache
braking boolean is a separate representation; it must not be used as a raw
brake word.

The concrete brake proof includes vehicle R8 preserved at `0x84319d`,
`MOVSS XMM1,[R8+0x9c]` at `0x84327a`, `COMISS XMM1,XMM9` at `0x843286`,
and `MOVAPS XMM3,XMM1`→`MULSS XMM3,[config+0xd94]` at
`0x8432a9/0x8432ac`. This is float arithmetic on the owner vehicle brake
channel, not the rejected head-track copy. The retained live traces used
brake0; they do not independently qualify a nonzero-braking trajectory.

The writer search exposed an important false-positive base identity:

- `0x7eb2c0` has a function pointer in the table at `0x1bec790`. The table
  starting `0x1bec660` names its class through `0x7eace0`: **`0x0a044000`**.
  Its allocator `0x7ead20` allocates only `0xc8` bytes. Local static
  `OpenplanetNext.json` identifies that class as `Scene::CSceneFxHeadTrack`,
  size200, parent `CSceneFxCompo`→`CSceneFx`. It is not a vehicle input
  controller. Consequently the literal copies at
  `0x7eb47f/0x7eb48a/0x7eb50f/0x7eb518` and clamp `0x7eb52e` operate on a
  head-tracking effect's similarly numbered fields. They must not be cited as
  direct writes to the owner vehicle. The actual normal input→vehicle copy and
  its scheduling order still need to be identified.

Qualified or still-candidate writes outside `0x841500` remain relevant:

- `0x83de60` can replace gas at `0x83de87`, clear brake at `0x83deb3`, and
  clear gas/brake at `0x83dee7/0x83df00` under other vehicle modes.
- In the inspected primary caller, `0x851f9e` calls `0x83de60` before
  `0x852110` calls `0x841500`. The secondary caller likewise calls the clamp
  at `0x869dc0`, then `0x841500` at `0x869e8d`.
- Wide copies at `0x7d4fdd/0x7d4feb` cover the input words as part of a much
  larger vehicle state copy; `0x7d703e/0x7d704c` belong to another broad copy.
  They do not by themselves establish the normal per-tick input producer.

There is a concrete boundary limitation: the primary caller itself reads brake
at `0x851fa3` and gas at `0x851fbb`, caches comparisons in its frame, and calls
`0x841790` at `0x8520fc` before PRE553. A bounded preceding-update slice also
contains further work. Their complete downstream effects have not been traced.
Thus PRE553 is before the identified consumers in `0x841500`, not before every
consumer in the enclosing simulation step. The script deliberately reports
`interprocedural_no_overwrite_proven=false`. Displacement candidates also miss
possible alias/vector writes covering a field, so they are not an exhaustive
writer inventory. Completion must re-read actual fields, and a physical
observation join remains necessary even if those fields match.

A conservative conditional profile that avoids the identified substitutions
requires `vehicle+0x176c & 0x40f == 0`, `vehicle+0x1370 & 0x0f == 0`,
forward selector `vehicle+0x1600 == 0`, signed word `vehicle+0x1768 == 0`,
and the float at `vehicle+0x1374` exactly positive zero. The latter two avoid
the clamp's `0x83dc30` gas-override condition; bit4 avoids its brake clear,
and bit0x400 avoids the known speed-limit branch. These are statically derived
conditional gate requirements, not measured current-map values. They must not
be inferred or relaxed from the first sample of a future run.

An earlier primary-caller candidate is RVA `0x851f96`, two complete stores
totalling eight bytes (`488945504c8965c0`). RDI was loaded from original
R8`+0x8` at `0x851f1f`; RSI preserves that original R8 step context at
`0x851f31`. Its `+0x14` word is later loaded into the original R8D argument of
`0x841500` at `0x852109`. A callback there could read owner RDI and literal
argument `[RSI+0x14]` before the clamp and before the caller's gas/brake reads.
It is not yet a global earliest-latch point: the clamp can overwrite the
requested fields, the owner/step path must be qualified dynamically, and other
callers exist. This point is a read-only qualification candidate, not an
qualified writer. Its literal64 prefix is:

```text
488945504c8965c0e8bdbefefff30f10879c00000033c0f30f103dc7cf4c01498bcc0f2fc7f30f1087980000000f97c04533f60f2fc7894590f20f104628f20f
```

## Proposed opt-in 008 diagnostic, pending source/lifetime qualification

The first construction should use a finite immutable tape loaded and sealed
before either hook is installed. It is distinct from an online actor queue.
Each tape row contains a nonce-bound command ID, exact gas/brake/steer uint32
words, and an explicit relative consumer-argument slot. Limit to256 rows and
five seconds, with preallocated integer evidence arrays. Begin with a separate
read-only PRE/POST qualification before enabling writes.

Once the source/lifetime requirements are met, a PRE callback would validate owner,
vehicle identity and the expected argument progression; retain original raw
SGB, mode flags and raw physical10; write only the three requested literal
words; read them back; and publish a bounded record under its command ID. The
existing proven late site `0x841734` would provide an independent completion
callback, retaining actual raw/filtered controls, flags and physical10. Neither
readback nor this same-function completion is by itself a whole-step physical
ACK. A subsequent actual physical observation must be joined by owner, command
ID, raw argument progression and exact state evidence. Missing, duplicate,
reordered, gated or unmatched observations must fail qualification.

Both hooks must preserve full original instructions and pin executable and
Openplanet identities, both literal64 prefixes, original trampoline bytes and
exact code restoration after every attempted install path. Install the observer
before enabling the writer; disable writes before any teardown. Retain separate
errors and cleanup phases, with no cached getter writes and no synthetic
transition. Do not restore a stale pre-command SGB snapshot over subsequent
engine/controller input; control release and hook-code restoration are distinct
operations requiring an explicit owned handoff.

The current threading review found a shared native VM lock for Main and callback
execution in the pinned Openplanet DLL. This helps serialization but does not
establish safe teardown: the hook dispatcher marks an invocation active before
acquiring that lock, while destruction can restore/free without a demonstrated
wait/join. Therefore neither a mutable interactive queue nor safe two-hook
unload is qualified by this document. Native callback quiescence must be
established before implementing or exercising a writer. The missing lifetime
and whole-step source proofs are why the current deliverable remains offline.
An online common actor
will additionally require bounded request/ACK/backpressure and a qualified
physical duration contract; an immutable diagnostic tape does not supply that.

## Reproducible offline evidence

`scripts/input_command008_static.py` hashes the private executable, checks the
literal64 prefixes, decodes the complete PRE instruction, and retains fifteen bounded
objdump windows. PE `.pdata` can describe chained unwind/prologue fragments,
so the script uses explicit windows and does not treat a fragment's end as a
whole function boundary. The first attempt `probe001` rejected that truncated
fragment before producing a successful report; its partial output is preserved.

```bash
cd /mnt/c<USER_HOME>/.codex/worktrees/headless-training-parity/AITrackmania
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" \
  <USER_HOME>/tmrl-headless-parity-20261021-001/venv/bin/python \
  -m scripts.input_command008_static \
  --binary <USER_HOME>/tmrl-headless-parity-20261021-003/windows-static/Trackmania-20261021.exe \
  --output <USER_HOME>/tmrl-headless-parity-20261022-006/input-command008-static/NEW
```

The final successful report is
`<USER_HOME>/tmrl-headless-parity-20261022-006/input-command008-static/probe005/report.json`,
SHA256 `ce5e8c27faeae47689b33a8139064da29e8ba1cb75dad33965be5608a3454d6a`.
It retains147 decoded instructions for the consumer entry through return, plus
bounded callee/helper/writer windows. GNU objdump is Ubuntu binutils2.42; this
offline run completed in under one second. Ruff check and format pass for the
new script. No native engine, Windows Python, live hook or training was run.
Five pure tests exercise exact instruction parsing, rejection of absent decoded
instructions, bounded disassembly before tool invocation, and rejection of an
unpinned image before creating output. The pinned private-image execution of
the offline script supplies separate actual byte/disassembly evidence. The
combined R6 audit subsequently passed34 tests (29 protocol and5 static), Ruff
check/format for four files, and strict mypy for both new modules.

Supplementary ext4 evidence in the same parent directory:

- `xrefs003.json`, SHA256
  `ea670fa735484213563991b17f8e88a7f787e8e49dfb04b176166ab9e8227771`,
  retains source bytes and bounded pointer/direct-branch candidates for PRE,
  the input writer, clamp and preceding update.
- `caller-primary004.txt`, `caller-secondary005.txt`,
  `preceding-update006.txt`, and `caller-primary007.txt` retain the inspected
  call order and earlier caller reads.
- `higher-entry008.txt`, `writer-exit009.txt`, `writer-vtable010.json`, and
  `writer-class011.txt` retain the earlier primary entry and the rejected
  controller/vehicle base-identity assumption.
- `vehicle-copy-candidate012.txt`, `vehicle-copy-candidate013.txt`,
  `other-copy-candidate014.txt`, and `covering-write-candidates015.json`
  retain the bounded search for wide stores. The final search covers literal
  displacement `+0x90` in RVA `0x7c5800..0x7e0000`, retains all77 byte
  candidates without truncation, and does not qualify their instruction/base
  identity automatically.
- `vehicle-class016.txt`, `vehicle-writer-candidates017.json`,
  `input-address-xrefs018.json`, `input-prefix-candidate019.txt`,
  `input-prefix-candidate020.txt`, `input-struct-init021.txt`,
  `input-struct-registration022.txt`, and `covering-write-candidates023.json`
  preserve the final bounded producer search, including nontruncated raw
  displacement scans in `0x830000..0x870000` and partial/truncated wider
  scans. No genuine normal per-tick input latch was qualified from those
  candidates. The search stopped at that explicit limitation.

The local `OpenplanetNext.json` used solely to name the rejected head-track
class has SHA256
`9d5ce8d9bfeed57f878674efe213fe5abaa04759d2dae0302e3c8e867cdf74df`.
The compact derived receipt is `rejected-headtrack-classification024.json`,
SHA256 `afdca43971215509da7f863e1c852305d5a1b07754028a6e346114b7f837344c`.
The preceding successful `probe004/report.json` is also retained, SHA256
`4db24c0dd881e61ccdf8c84b5bf576a9d73f812de5fcb36b9104644144ff7144`;
probe005 uses the final source with a separately testable pin-validation helper.

The executable and all private disassembly stay outside the repository. The
previous 007 source, runtime pins and reports are unchanged.
