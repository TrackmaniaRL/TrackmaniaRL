> Historical home-directory prefixes are shown as `<USER_HOME>` placeholders. Replace them with your explicit local paths; original private receipts are not redistributed.

# R6: Openplanet script synchronization and command ACK limits

The pinned Openplanet DLL contains a shared host mutex around the inspected
`Main` coroutine execution and `Dev::Hook` callback execution. This is a concrete
static basis for sharing a bounded queue **between these two execution paths**.
It is not a public API guarantee, an observed OS-thread identity, a physical
input ACK, or a proof that native hook removal waits for other threads.

This investigation only reads local declarations and offline binary files. It
does not load plugins, access a live process, issue controls, or execute target
native code. The R5 source and captures are unchanged.

## Retained evidence

The JSON receipts are copied identically to:

- `F:/AITrackmania/reports/headless-training-parity-20261022-r6/threading/`
- `<USER_HOME>/tmrl-headless-parity-20261022-006/openplanet-threading/`

| File | Contents | Full SHA256 |
| --- | --- | --- |
| `scan001.json` | PE import names/IAT slots; bounded relative-call candidates | `351c89a16458d9c3ad270a851effd5e00e22ebb6bddabf6a5ceded32ef7c9ed7` |
| `entry-strings002.json` | `Main`, `Update` and scheduler string references | `668cf1470ce9345bcd059f5dc84d484b7c5b5d3bff748313d3d34a9b115ecf91` |
| `mutex-disassembly003.json` | Fifteen bounded disassembly slices with raw bytes, hashes and exact commands | `9b336c3b7290237a67564304321bbae96d9a83bd03bb4fff7578e68fb3338767` |
| `api-inventory004.json` | Local API hashes, exact declarations and primitive-name search | `507e00c405e0373aad5eb1c56d074ed15be093a9350903466323d93bda939094` |
| `owner-tid005.json` | Exact embedded-mutex pointer chain and current-thread-ID import | `b8f14301bdd7522b0a4e4b1ee333d1c422acd47ceade409ffe1fe73946d06a34` |

Openplanet offline DLL SHA256 is
`f134f405979d2c2641f8f131491e3401fcdd4f4b5266cde9f1a7001a6d3b49ef`.
All Openplanet addresses below are RVAs in this file, whose preferred image base
is `0x180000000`. ASLR must be handled independently during any future use.

The inspected local declarations identify Openplanet
`1.29.14 (next, Public, 6fd4200f)`. Their hashes are:

```text
Openplanet.h        16b50c345edebd9bd139ba596a1401b83f1a2d0932a6a50912bc43bc25ce0332
OpenplanetCore.json c0cbaa483644a29f5151cf40365300d1cf82a4738cfb6161187adc752bfb84c9
OpenplanetNext.json 9d5ce8d9bfeed57f878674efe213fe5abaa04759d2dae0302e3c8e867cdf74df
```

## Public API findings

The structured inventory examined 1,996 core function/class/method/enum
declarations. It found no declaration named for a mutex, atomic, semaphore,
critical section, memory barrier, current thread or thread ID. This is a result
for these local snapshots, rather than a claim about every possible extension.

The available operations include:

```cpp
awaitable@ startnew(CoroutineFunc@ func)
void await(awaitable@)
void yield()
void yield(uint frames)
void sleep(uint64 ms)
awaitable@ Meta::StartWithRunContext(Meta::RunContext, CoroutineFunc@)
Dev::HookInfo@ Dev::Hook(uint64 ptr, int padding, const string&in func,
    int pushRegisters = 0, Dev::FreeRegister freeRegister = FreeRegister::Rax)
void Dev::Unhook(Dev::HookInfo@ hook)
void Dev::SafeWrite(uint64 ptr, uint i)
```

`SafeWrite` also has explicitly listed signed/unsigned integer and other
overloads. Its return type is `void`: returning from it is not a successful
readback, physical consumption, or ACK. The uint overload allows preserving a
float32 bit pattern without requiring a float conversion in the callback; this
does not by itself qualify the target field or actuator.

`startnew` and `StartWithRunContext` are documented as yieldable coroutines.
`Meta::RunContext` enumerates Main0, BeforeScripts1, MainLoop2, GameLoop3,
AfterMainLoop4, NetworkAfterMainLoop5, AfterScripts6 and UpdateSceneEngine7.
Their descriptions identify scheduling locations, without promising a native
thread, mutual exclusion, or physical tick boundary. `yield()` describes
resumption at the next game tick, while its frame overload is explicitly frame
dependent. These descriptions cannot establish one 10 ms physical input step.
`Dev::HookInfo` exposes no public methods or properties. `Unhook` explicitly
prohibits being called from its own callback.

The game declarations' thread-like matches are display `MultiThread` and
`ThreadCountMax`, scene `EnableMultiThread`, and Maniascript playground
`Synchro_AddBarrier`/`Synchro_BarrierReached`. Their types and placement do not
establish script-memory synchronization. This work changes none of them.

The generic [AngelScript multithreading documentation](https://www.angelcode.com/angelscript/sdk/docs/manual/doc_adv_multithread.html)
places synchronization of shared script state on the host/application. The
specific host implementation below supplies the relevant evidence here.

## Pinned shared-host-lock proof

### The same mutex object

The global pointer at Openplanet RVA `0x9bca28` supplies an object whose `+0x308`
field contains the script manager. The wrappers take that same manager
regardless of which plugin is passed as the operation's second argument:

```text
enter 0x1720a0 -> [global 0x9bca28]+0x308 -> 0x163190
leave 0x1720d0 -> [global 0x9bca28]+0x308 -> 0x1630f0
```

`0x163190` acquires the mutex at `manager+0x18` through thunk `0x745945`.
That thunk jumps to IAT RVA `0x7710c0`, whose actual PE import name is
`MSVCP140.dll!_Mtx_lock`. The return code is checked before script-manager
mutation. `0x1630f0` releases the same `manager+0x18` through thunk `0x74594b`
and IAT RVA `0x771098`, bound to `MSVCP140.dll!_Mtx_unlock`. Thus this finding
uses identified call sites and import slots, rather than merely the DLL having
lock-related imports somewhere.

### Hook callback region

The hook JIT builder `0x16a8f0` embeds a call to native dispatcher `0x16a2e0`.
That dispatcher calls enter at `0x16a321`, sets up its AngelScript context and
arguments, calls context execution via `0x167e80` at `0x16a86d`, and calls leave
at `0x16a88e` after execution and context cleanup. The ordinary callback script
therefore executes within the identified shared mutex's region.

The lower patcher `0x0964e0` copies the original `5+padding` bytes to the
**beginning** of the trampoline at `0x096539`. Register pushes and the callback
bridge are appended after those copied bytes. Therefore the proposed complete
eight-byte instruction at game RVA `0x841553`,
`48 89 84 24 c8 00 00 00` (`mov [rsp+0xc8],rax`), would execute with the original
RSP and RAX before the bridge changes the stack. The bridge restores GPR values
on returning to the original continuation. The scratch-register selection must
still be valid at that exact site; relocation and ABI acceptance are separately
owned by the PRE-consumer investigation. This document authorizes no patch.

### Main coroutine region

`0x16ff60` references the literal `void Main()` at `0x7c0198`, obtains that
function, constructs its coroutine through `0x15b230`, and appends it to the
plugin's array `+0x1410` with count `+0x1418`. The coroutine constructor retains
its function at `+0x20` and context at `+0x18`.

Scheduler `0x16e6f0` calls the same enter wrapper at `0x16e75a`. It iterates this
same coroutine array, selects the requested run context using coroutine `+0x58`,
and invokes resume function `0x15b3b0` at `0x16e81a`. Resume calls context
execution `0x167e80` at `0x15b4c9`. The scheduler calls leave at `0x16e836`, after
the coroutine resumes have returned. Main's script execution on this inspected
path is therefore enclosed by the same mutex as the hook script.

This establishes a static shared-lock relationship for these execution paths.
It does not establish their OS thread IDs or extend to arbitrary callbacks,
extensions, direct native writers, or a different Openplanet binary.

## Recursive acquisition and reentrancy

Manager constructor `0x15be10` stores mutex type `0x102` at `manager+0x18`
at `0x15be52`. Its caller `0x98027` constructs the object and stores it into the
owner's `+0x308` field at `0x9802d`.

The local `C:/Windows/System32/msvcp140.dll`, version `14.51.36247.0`, has SHA256
`7c26614e1d733892c2deac7e245ce115504b1d80592dd0a01b08e3e5a55f89ca`.
Its `_Mtx_lock` export at RVA `0x17d50` reads the current thread ID, checks the
stored owner at mutex `+0x48`, increments recursion count `+0x4c`, and explicitly
permits repeated acquisition when mutex type has bit `0x100` at `0x17db0`.
Type `0x102` has that bit. The exact local runtime code and constructor therefore
support interpreting this host lock as recursive. This research did not read a
live loaded-module list; the System32 file is a local runtime implementation
receipt, not proof of which MSVCP file a running process loaded.

A recursive mutex excludes another thread but permits nested execution on its
owner thread. No claim is made that arbitrary game getters cannot synchronously
cause physics work. A candidate queue must therefore:

1. Mutate shared queue/journal state only in the proven Main/hook paths.
2. Use an `EditingQueue` guard around a complete non-yielding publication, with
   fixed preallocated slots, no resizing, serialization, networking, game getter,
   or arbitrary external callback during that publication.
3. Check `EditingQueue` and `InsideCallback` before reading a command or writing
   game fields. A nested/reentrant attempt must record failure or skip injection
   before reading a partially edited slot; it cannot be credited as consumed.
4. Perform socket/file work outside the physical callback and keep all callback
   work bounded. Never wait for Main or a socket while holding the physical call.

The host mutex supplies the exclusion; a plain boolean guard alone is not a
cross-thread synchronization primitive. Publishing an index last is useful for
validation but is not a substitute for the host lock. These conditions are a
design for a bounded prototype, not a measured interactive training result.

## Native teardown and lifecycle remain separate

Hook dispatcher `0x16a315` writes its native `inside` byte at hook-info `+0x4c`
**before** acquiring the script mutex. Hook-info destructor `0x16a1d0` checks
that byte at `0x16a1f3`; the inspected branch logs/asserts on the condition and
then continues toward restoration and deallocation. It does not show a
wait/join for a callback already entered or blocked on the script mutex.

Consequently, holding the Main script mutex and observing a script
`InsideCallback=false` does not establish native trampoline quiescence.
Successful original-code readback also does not prove that no other thread
still holds a trampoline or descriptor pointer. A future prototype must retain
this qualification limit and independently establish safe native teardown.
Neither an arbitrary host sleep nor a chosen number of empty frames is a
documented join primitive in the inspected API.

This investigation does not prove the execution paths of `OnDisabled` or
`OnDestroyed`. They must not mutate the interactive queue/journal under an
assumption that this proof automatically covers them. Restoration, report
persistence and socket cleanup still need independent guarded attempts, and
the final result must report missing restoration evidence as failure. No
restoration claim should depend on serialization or successful network output.

## Future read-only thread-ID observation

The owner field offers a concrete read-only candidate without inventing a
`GetCurrentThreadId` AngelScript API. In the inspected local MSVCP file, the
call at `_Mtx_lock+0x0d` (RVA `0x17d5d`) goes through IAT `0x631c0`, whose exact
PE import is `KERNEL32.dll!GetCurrentThreadId`. That returned ID is compared
with/stored in mutex `+0x48`. The other relevant import `0x63050` is
`KERNEL32.dll!AcquireSRWLockExclusive`. The pointer chain is:

```text
root          = ReadUint64(verifiedOpenplanetBase + 0x9bca28)
manager       = ReadUint64(root + 0x308)
mutexAddress  = manager + 0x18         // address addition, no pointer dereference
type          = ReadUint32(manager + 0x18)  // expected 0x102
ownerTid      = ReadUint32(manager + 0x60)  // mutex + 0x48
recursion     = ReadUint32(manager + 0x64)  // mutex + 0x4c, must be >= 1
```

During the inspected Main/hook script regions, the host already holds this
mutex. With validated module identity, pointer bounds, type and nonzero
recursion, the owner field therefore identifies the thread currently holding
that region. A future diagnostic must pin the **actually loaded** MSVCP path,
hash and relevant bytes; this offline System32 receipt is insufficient to claim
that runtime observation has happened. It must reject zero/sentinel IDs or
failed chain reads and retain the chain/type/count alongside every ID.

The proposed diagnostic records Main and **all** hook invocations before local
vehicle filtering, including callbacks that would later be skipped. Recording
only accepted player rows could hide another thread entering the native hook.
Capacity exhaustion or missing entries fail the thread-topology observation.
No input writes are needed for this measurement.

Equal observed IDs can narrow a bounded same-thread-path hypothesis; they do
not establish a permanent global thread assignment or, by themselves, prove
quiescence at later removal. A callback blocked before reaching script remains
unobserved until the lock is released. Nested execution on the same thread
also remains possible because the mutex is recursive. The absence of such
invocations and the normal-path shutdown ordering need independent runtime
evidence and call-path reasoning before removing the native-teardown limit.
This thread-ID candidate is not a physical input-consumption ACK.

## Minimum command-to-consumption contract

A queue receipt can bind a nonce, command sequence, literal SGB words, requested
bounded interval count and deadline. It confirms only publication. At a proven
PRE-consumer phase the callback may select that exact committed command and
record owner, command sequence, physical call identity, pre-write words and
verified readback after any authorized writes. A readback is still an injection
receipt. Physical acceptance requires a separately qualified downstream
consumption observation with the same owner and call/tick identity, plus the
qualified resulting physical endpoint. Requested intervals must never be
substituted for the number of measured accepted intervals. Partial writes,
unexpected overwrite, skipped/missing ticks, ambiguity or reentrancy fail the
command; they cannot yield an ACK or silently borrow a later successful sample.

Before interactive publication is measured, the smaller experiment is a fully
validated immutable bounded tape loaded before hook installation. The callback
alone owns its cursor and output buffer while armed. Main neither changes nor
reads those mutable buffers until independently established quiescence; the
callback never waits for the client. Every tape entry carries an explicit
sequence and literal words, and every measured acceptance is retained. EOF,
capacity, deadline, owner loss and restoration failures remain explicit.
That experiment can test a prerecorded command-to-consumption association,
but it is not interactive policy training and does not close a paired-training
gate.
