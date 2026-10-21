//! fkshim -- LD_PRELOAD fork server for the TM2020 dedicated server.
//!
//! # What it does
//!
//! The server simulates a ghost's input tape tick by tick. Re-simulating from
//! tick 0 for every candidate wastes the whole shared prefix. This shim stops
//! the process *inside* the simulation, holds that state as a checkpoint, and
//! forks a child per candidate: the child rewrites the still-unread part of the
//! decoded input array and runs only the tail.
//!
//! # The three mechanisms
//!
//! 1. **A deterministic clock.** The engine calls `lroundf` ~25.5 times per
//!    simulated millisecond, and the total for a given (map, ghost) is bit-exact
//!    across runs. Interposing it gives a reproducible cursor into the middle of
//!    a simulation with no debugger, no disassembly and no ptrace.
//!
//! 2. **The decoded input array.** The ghost's bitstream is decoded up front (it
//!    is *not* read during the simulation) into one 32-byte record per 10 ms
//!    tick: `f32 steer = (i8)steer/127`, `f32 gas`, `f32 brake`, then engine
//!    fields. The shim finds it by searching its own address space for the
//!    reference ghost's steer sequence at stride 32.
//!
//! 3. **fork() from the checkpoint.** The engine simulates on the main thread
//!    (the only other thread, `NetPoll`, sleeps), so the fork child is a
//!    complete, self-consistent simulator. The parent never advances.
//!
//! # Protocol
//!
//! Driven over two inherited fds (`FKSHIM_CMD_FD`, `FKSHIM_RES_FD`). Frames are
//! `u32 len` + payload. The shim announces `READY <base> <clock>`, then serves
//! commands until `Q`:
//!   `R <n> [tick steer gas brake]*n`  -- fork, patch, run, return the child's
//!                                        stdout (the validator's JSON block)

mod post_observe;
use std::os::raw::{c_char, c_int, c_void};
use std::sync::atomic::{AtomicI32, AtomicU64, AtomicU8, AtomicUsize, Ordering};

#[path = "../../vendor/trackmania-tas/tools/search/forkoracle/src/pred_core.rs"]
pub mod pred_core;
use pred_core::{
    Eval, Fire, Gate, KeyOp, Pred, RefLine, KEYOP_BYTES, MAXKOPS, MAXP, PRED_BYTES, SUMMARY_BYTES,
};

const RTLD_NEXT: *mut c_void = -1isize as *mut c_void;

extern "C" {
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn getenv(name: *const c_char) -> *const c_char;
    fn atoll(s: *const c_char) -> i64;
    fn raise(sig: c_int) -> c_int;
    fn fork() -> c_int;
    fn _exit(code: c_int) -> !;
    fn pipe(fds: *mut c_int) -> c_int;
    fn dup2(old: c_int, new: c_int) -> c_int;
    fn close(fd: c_int) -> c_int;
    fn kill(pid: c_int, sig: c_int) -> c_int;
    fn waitpid(pid: c_int, status: *mut c_int, opts: c_int) -> c_int;
    fn signal(sig: c_int, h: usize) -> usize;
    fn fflush(f: *mut c_void) -> c_int;
    fn setvbuf(f: *mut c_void, buf: *mut c_char, mode: c_int, size: usize) -> c_int;
    fn poll(fds: *mut PollFd, n: u64, timeout: c_int) -> c_int;
    fn open(path: *const c_char, flags: c_int, ...) -> c_int;
    static mut stdout: *mut c_void;
}

#[repr(C)]
struct PollFd {
    fd: c_int,
    events: i16,
    revents: i16,
}

const SIGTRAP: c_int = 5;
const SIGSTOP: c_int = 19;
const SIGKILL: c_int = 9;
const SIGCHLD: c_int = 17;
const SIG_IGN: usize = 1;
const IONBF: c_int = 2;
const POLLIN: i16 = 0x001;

extern "C" {
    fn socket(domain: c_int, ty: c_int, proto: c_int) -> c_int;
    fn connect(fd: c_int, addr: *const u8, len: u32) -> c_int;
}
const AF_UNIX: c_int = 1;
const SOCK_STREAM: c_int = 1;

/// Connect to the driver's listening unix socket. Returns -1 on failure.
///
/// The path is a filesystem path in the run's own work directory rather than an
/// abstract-namespace name: a work directory is already per-process and already
/// locked (`take_dir_lock`), so the socket inherits that isolation instead of
/// needing its own naming scheme. Two runs cannot collide on it for the same
/// reason two runs cannot collide on their replays.
unsafe fn connect_unix(path: &[u8]) -> c_int {
    let fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if fd < 0 {
        return -1;
    }
    // sockaddr_un: u16 family, then a NUL-terminated path.
    let mut sa = [0u8; 110];
    sa[0] = AF_UNIX as u8;
    sa[1] = 0;
    let n = path.len().min(107);
    std::ptr::copy_nonoverlapping(path.as_ptr(), sa.as_mut_ptr().add(2), n);
    if connect(fd, sa.as_ptr(), (2 + n + 1) as u32) != 0 {
        close(fd);
        return -1;
    }
    fd
}

static N_LROUNDF: AtomicU64 = AtomicU64::new(0);
static STOP_AT: AtomicU64 = AtomicU64::new(u64::MAX);
static CKPT_AT: AtomicU64 = AtomicU64::new(u64::MAX);
static ARMED: AtomicUsize = AtomicUsize::new(0);
static IS_CHILD: AtomicUsize = AtomicUsize::new(0);
static INIT: AtomicUsize = AtomicUsize::new(0);
static REAL_LROUNDF: AtomicUsize = AtomicUsize::new(0);
static REAL_WRITE: AtomicUsize = AtomicUsize::new(0);
static CMD_FD: AtomicI32 = AtomicI32::new(-1);
static RES_FD: AtomicI32 = AtomicI32::new(-1);
/// Load address of the main executable, captured before any signal handler runs.
static MODULE_BASE: AtomicUsize = AtomicUsize::new(0);

// The validator's simulation-binding callback on build 128182. It is reached
// from the /validatepath virtual method before the first simulation tick:
//
//     0x113ade4  call [validator.vtable + 0x238]
//       -> 0x1182b40 validation state machine
//       -> 0x11818aa call 0x118c170
//
// At 0x118c170, rdi is the validation controller and rcx is its simulation
// object. The function itself stores rcx at [rdi + 0x1a70]. Capturing these
// arguments gives the driver a validator-owned root; no memory search chooses a
// car. The build-specific prologue is verified before the one-byte breakpoint is
// installed. That breakpoint is restored before the original instruction is
// re-executed, so the hook is one-shot and behavior-preserving.
const VALIDATOR_SIM_BIND_OFF: usize = 0x118c170;
// Full build-128182 prologue through the controller.sim store. Checking only
// `push rbp` would turn an unsupported server build into a trap at an arbitrary
// function that happened to begin with the same byte.
const VALIDATOR_SIM_BIND_SIGNATURE: [u8; 33] = [
    0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x50, 0x48, 0x89,
    0xcb, 0x41, 0x89, 0xd7, 0x49, 0x89, 0xf6, 0x49, 0x89, 0xfc, 0x48, 0x89, 0x8f, 0x70, 0x1a, 0x00,
    0x00,
];
static VALIDATOR_TRAP_ADDR: AtomicUsize = AtomicUsize::new(0);
static VALIDATOR_TRAP_BYTE: AtomicU8 = AtomicU8::new(0);
static VALIDATOR_CONTROLLER: AtomicUsize = AtomicUsize::new(0);
static VALIDATOR_SIM: AtomicUsize = AtomicUsize::new(0);
static VALIDATOR_PARTICIPANTS_ARG: AtomicUsize = AtomicUsize::new(0);
static VALIDATOR_PARTICIPANT_COUNT_ARG: AtomicUsize = AtomicUsize::new(0);

// --------------------------------------------------------------- the BRANCH
//
// THE SAVESTATE TREE. A fork child normally runs the tape to the end and dies;
// everything it learned along the way dies with it, so every node of a search
// has to be reached by re-simulating its whole prefix from the one checkpoint
// the parent holds.
//
// A branch child instead simulates a FEW ticks and then re-enters the fork
// server itself, becoming a new fork point. The three things that stopped it
// from doing that, all in code we own:
//
//  1. `IS_CHILD` -- set by every child so it can never re-enter. A branch child
//     needs to re-enter exactly once, so `BRANCH_ARMED` licenses that one
//     re-entry and is consumed on the way in.
//  2. `ARMED` -- the one-shot latch on the checkpoint test. Reset in the branch
//     child so the NEXT checkpoint fires.
//  3. **The two inherited fds.** A child that served the parent's `cmd`/`res`
//     pipes would race the parent for every command byte. So a branch child
//     gets FRESH fds: it connects to a unix socket the driver is listening on,
//     and serves the same protocol down that one socket (the frame code does
//     not care that read and write are the same descriptor).
//
// A branch child also caches what the first entry already worked out. The
// entry path Horspool-scans every `rw` mapping of a 150 MB address space to
// find the decoded input array; repeating that per node would put the scan on
// the hot path. The array does not move (it is decoded once, before the
// simulation, and there is exactly one copy), and the child is a fork of the
// process that found it, so the address is still good -- but "still good" is a
// claim, so the branch VERIFIES the cached base against the key rather than
// trusting it, and a mismatch is a hard abort.
static BRANCH_ARMED: AtomicUsize = AtomicUsize::new(0);
// Experimental K: exact raw-clock threshold, independent of lroundf counts.
static CLOCK_STOP_ADDR: AtomicUsize = AtomicUsize::new(0);
static CLOCK_STOP_TARGET: AtomicU64 = AtomicU64::new(0);
static BRANCH_SOCK_LEN: AtomicUsize = AtomicUsize::new(0);
static mut BRANCH_SOCK: [u8; 108] = [0; 108];
/// The input array's base, found once by `locate` and inherited by every
/// descendant. Re-verified, never merely trusted.
static CACHED_BASE: AtomicUsize = AtomicUsize::new(0);
/// The parsed key, leaked on first use so descendants need no file read.
static CACHED_KEY: AtomicUsize = AtomicUsize::new(0);
/// The socket this process is currently serving on (0 = none): a branch child
/// closes its parent's so a dead driver is seen as EOF by exactly one process.
static SERVING_FD: AtomicI32 = AtomicI32::new(-1);
/// The trace file a branch child writes its per-tick state to while it consumes
/// its `k` ticks; closed on re-entry so the driver reads a complete file.
static TRACE_FD: AtomicI32 = AtomicI32::new(-1);

// ------------------------------------------------------------------ sampling
//
// The point of the whole exercise: a forked child does not just produce a
// finish time, it can report the car's state as it simulates. `lroundf` is
// already on the simulation's hot path (~255 calls per 10 ms tick), so the hook
// doubles as a sampling clock: every `SAMPLE_STRIDE` calls, copy a window of
// the vehicle struct out to a dedicated pipe.
//
// Dedup is what makes this tick-accurate without knowing where the engine keeps
// its tick counter: the physics integrator writes the state once per tick, so
// sampling several times per tick and emitting only on change yields exactly
// the distinct states, in order.

static SAMPLE_ADDR: AtomicUsize = AtomicUsize::new(0);
static SAMPLE_LEN: AtomicUsize = AtomicUsize::new(0);
/// Up to 8 (addr, len) segments gathered into one contiguous record. The
/// engine's race clock sits ~8 KB away from the vehicle state, and streaming
/// the 8 KB between them every tick would cost more than the simulation does;
/// gathering two small segments costs nothing.
const MAX_SEG: usize = 8;
static SEG_N: AtomicUsize = AtomicUsize::new(0);
/// A pointer chain the SAMPLER re-walks at every instant, so segment 0 follows
/// a car that the engine reallocates mid-race. 0 = disabled.
/// Is this address readable? A cheap `write`-to-devnull probe: the kernel
/// returns EFAULT rather than delivering a signal, so this cannot crash.
unsafe fn addr_readable(a: usize) -> bool {
    // A RAW SYSCALL, not `write` and not `real_write()`.
    //
    // This shim EXPORTS `write`, and it runs on the sampling path -- going
    // through the C library here risks re-entering the interceptor, and
    // `real_write()`'s dlsym is not something to be doing at every
    // dereference of every sampled instant. `SYS_write` on a probe fd returns
    // EFAULT for an unmapped address instead of delivering a signal, which is
    // the whole point: a bad pointer must come back as `false`, never as a
    // SIGSEGV that kills the server and reads as "arm_chain: no reply".
    // The fd is opened at library init, NOT here. This shim tracks its own
    // descriptors, and taking one lazily -- at arm time, in the middle of a
    // run -- shifts the numbering underneath it.
    let fd = DEVNULL_FD.load(Ordering::Relaxed);
    if fd < 0 {
        return false;
    }
    let r: isize;
    std::arch::asm!(
        "syscall",
        inlateout("rax") 1isize => r,   // SYS_write
        in("rdi") fd as usize,
        in("rsi") a,
        in("rdx") 8usize,
        lateout("rcx") _,
        lateout("r11") _,
        options(nostack)
    );
    r == 8
}
static DEVNULL_FD: AtomicI32 = AtomicI32::new(-1);
static CHAIN_N: AtomicUsize = AtomicUsize::new(0);
/// The chain only rewrites segment 1 once the real gather has begun; see the
/// note in the sampler.
static CHAIN_ARMED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static CHAIN_ROOT: AtomicUsize = AtomicUsize::new(0);
static CHAIN_BACK: AtomicUsize = AtomicUsize::new(0);
static CHAIN_OFF: [AtomicUsize; 8] = [
    AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0),
    AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0),
];
static SEG_ADDR: [AtomicUsize; MAX_SEG] = [
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
];
static SEG_LEN: [AtomicUsize; MAX_SEG] = [
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
];
static SAMPLE_FD: AtomicI32 = AtomicI32::new(-1);
static SAMPLE_STRIDE: AtomicU64 = AtomicU64::new(0);
static SAMPLE_NEXT: AtomicU64 = AtomicU64::new(u64::MAX);
static SAMPLE_LEFT: AtomicU64 = AtomicU64::new(0);
static SAMPLE_DEDUP: AtomicUsize = AtomicUsize::new(0); // key length, 0 = off
static SAMPLE_KEYOFF: AtomicUsize = AtomicUsize::new(0);
/// Absolute `lroundf` count past which the child stops simulating (0 = never).
///
/// WHY: a locate probe wants 6 or 150 TICKS, and without this the child
/// simulates the whole remaining tape -- on a 440 s record that is 43 000 ticks
/// for six ticks of data, which is what made the blind locate cost 5.5 minutes
/// per attempt and put a general fix out of reach on long tapes.
static SAMPLE_DEADLINE: AtomicU64 = AtomicU64::new(0);
/// Exit the child the moment the sample budget is spent (1 = yes).
static SAMPLE_EXIT: AtomicUsize = AtomicUsize::new(0);
/// GRID GATE. Sample only when the engine's race clock is on the record's own
/// 50 ms grid: `(clock_value - GATE_PHASE) % GATE_MOD == 0`.
///
/// WHY: a ghost sample exists only every 50 ms, so a snapshot taken at any
/// other tick can be compared with a recorded value only by tolerating a slop
/// of up to 25 ms -- and at 70 m/s that slop is 1.75 m, which drags the
/// correlation of a slot that IS the position down below the correlation of
/// junk. Gating in the shim makes every snapshot land on an instant the answer
/// key actually has, so the comparison can be exact.
static GATE_ADDR: AtomicUsize = AtomicUsize::new(0);
static GATE_MOD: AtomicU64 = AtomicU64::new(0);
static GATE_PHASE: AtomicU64 = AtomicU64::new(0);
static SAMPLE_PREV: AtomicUsize = AtomicUsize::new(0); // *mut u8, len SAMPLE_LEN
static SAMPLE_BUF: AtomicUsize = AtomicUsize::new(0); // *mut u8, len 8 + SAMPLE_LEN

/// Gather the watched segments out, if the key slice changed. Called from the
/// `lroundf` hook in the child only.
///
/// Dedup is on a *key* slice of the gathered record rather than the whole
/// thing: the record has to be wide enough to catch the neighbouring fields,
/// but the engine touches some of those several times per tick, and only the
/// race clock marks a tick boundary unconditionally.
#[inline(never)]
unsafe fn do_sample(clock: u64) {
    // The simulated-time deadline is checked FIRST and before the budget test
    // below, which parks the hook (`SAMPLE_NEXT = MAX`) and would otherwise
    // stop this from ever being reached again.
    let dl = SAMPLE_DEADLINE.load(Ordering::Relaxed);
    if dl != 0 && clock > dl {
        _exit(0)
    }
    let stride = SAMPLE_STRIDE.load(Ordering::Relaxed);
    let gm = GATE_MOD.load(Ordering::Relaxed);
    if gm != 0 {
        // A REJECTED gate check must re-arm for the NEXT call, not for the next
        // stride: advancing by a whole stride here means each sample needs the
        // one lroundf call `stride` ahead to land on the grid by luck, which
        // turned a request for 64 snapshots into 17.
        let ga = GATE_ADDR.load(Ordering::Relaxed) as *const u32;
        if ga.is_null() {
            SAMPLE_NEXT.store(clock + stride, Ordering::Relaxed);
            return;
        }
        let c = std::ptr::read_volatile(ga) as u64;
        // PHASE u32::MAX means "whatever this process's clock is congruent to".
        // `find_clock2` returns a CLASS of counters and two processes can land
        // on ones with different absolute offsets, so a phase computed in the
        // anchor process can match NOTHING here -- measured as "0 instants
        // sampled" on a third of the corpus.
        if GATE_PHASE.load(Ordering::Relaxed) == u32::MAX as u64 {
            GATE_PHASE.store(c % gm, Ordering::Relaxed);
        }
        if c.wrapping_sub(GATE_PHASE.load(Ordering::Relaxed)) % gm != 0 {
            SAMPLE_NEXT.store(clock + 1, Ordering::Relaxed);
            return;
        }
    }
    SAMPLE_NEXT.store(clock + stride, Ordering::Relaxed);
    let n = SAMPLE_LEFT.load(Ordering::Relaxed);
    if n == 0 {
        if SAMPLE_EXIT.load(Ordering::Relaxed) != 0 {
            _exit(0)
        }
        SAMPLE_STRIDE.store(0, Ordering::Relaxed);
        SAMPLE_NEXT.store(u64::MAX, Ordering::Relaxed);
        return;
    }
    let len = SAMPLE_LEN.load(Ordering::Relaxed);
    let buf = SAMPLE_BUF.load(Ordering::Relaxed) as *mut u8;
    if buf.is_null() {
        return;
    }
    let nseg = SEG_N.load(Ordering::Relaxed);
    // RE-WALK THE CAR'S POINTER CHAIN AT EVERY SAMPLE, when one is armed.
    //
    // A fixed address is only correct for a map that keeps one vehicle object
    // for the whole race. 287431 does not: it spawns the car 646 m up, the
    // 2.13 s fall is a separate entity, and the object a chain resolved at the
    // start is left FROZEN when the driving entity replaces it -- `fk trace`
    // on that address shows y stuck at 20.875 while vy holds -277.794 m/s for
    // two seconds. Resolving once per run cannot see that; only the sampler,
    // which runs inside the simulation, can follow it.
    //
    // CHAIN_N == 0 means no chain is armed and this costs one relaxed load.
    // ONLY ONCE THE REAL GATHER IS RUNNING.
    //
    // `find_clock2` issues its own sampling commands to hunt for the race
    // clock, and it looks for a slot that advances by 10 per tick. Rewriting
    // segment 1 underneath it corrupts exactly what it is measuring, and the
    // clock scan then fails with "no slot advances by a multiple of 10 per
    // tick: race clock not located" -- a failure three layers from its cause.
    //
    // CHAIN_ARMED is set by the 'G' handler, so the chain is inert for every
    // probe and scan that comes before it.
    let cn = if CHAIN_ARMED.load(Ordering::Relaxed) {
        CHAIN_N.load(Ordering::Relaxed)
    } else {
        0
    };
    if cn != 0 {
        let mut a = CHAIN_ROOT.load(Ordering::Relaxed);
        let mut ok = true;
        for i in 0..cn {
            // Checked, for the same reason as at arm time: this runs at every
            // sampled instant, and a fault here takes the whole run with it.
            if a == 0 || a < 0x1000 || !addr_readable(a) {
                ok = false;
                break;
            }
            a = *(a as *const usize);
            if a == 0 || a < 0x1000 {
                ok = false;
                break;
            }
            a = a.wrapping_add(CHAIN_OFF[i].load(Ordering::Relaxed));
        }
        if ok && a != 0 {
            // SEGMENT 1, NOT 0. Segment 0 is the 4-byte race CLOCK -- the grid
            // gate reads it to decide whether this instant is on the sampling
            // grid at all. Overwriting it with the car's address made the gate
            // never match and the run gathered 0 instants. The car window is
            // the segment after it.
            SEG_ADDR[1].store(a.wrapping_sub(CHAIN_BACK.load(Ordering::Relaxed)), Ordering::Relaxed);
        }
    }
    let mut o = 0usize;
    for s in 0..nseg {
        let a = SEG_ADDR[s].load(Ordering::Relaxed) as *const u8;
        let l = SEG_LEN[s].load(Ordering::Relaxed);
        std::ptr::copy_nonoverlapping(a, buf.add(8 + o), l);
        o += l;
    }
    let klen = SAMPLE_DEDUP.load(Ordering::Relaxed);
    let koff = SAMPLE_KEYOFF.load(Ordering::Relaxed);
    let prev = SAMPLE_PREV.load(Ordering::Relaxed) as *mut u8;
    if klen != 0 && !prev.is_null() {
        if std::slice::from_raw_parts(prev, klen)
            == std::slice::from_raw_parts(buf.add(8 + koff), klen)
        {
            return;
        }
        std::ptr::copy_nonoverlapping(buf.add(8 + koff), prev, klen);
    }
    std::ptr::copy_nonoverlapping(clock.to_le_bytes().as_ptr(), buf, 8);
    SAMPLE_LEFT.store(n - 1, Ordering::Relaxed);
    write_all(
        SAMPLE_FD.load(Ordering::Relaxed),
        std::slice::from_raw_parts(buf, 8 + len),
    );
}

// ----------------------------------------------------------------- watchdog
//
// STOP SIMULATING A CANDIDATE THE MOMENT IT IS CLEARLY DEAD.
//
// Cost is linear in simulated time, and a third to a half of the candidates a
// search generates crash, stop or leave the track long before the finish. The
// state needed to notice that is already in hand -- `do_sample` above proves
// the child can read the car's position and velocity every tick for a couple
// of milliseconds -- so the condition costs almost nothing to evaluate and the
// saving is the whole remaining tail.
//
// Three properties are non-negotiable:
//
//  1. A candidate that does NOT trip must produce a bit-identical answer to
//     one run with no watchdog. The evaluator therefore only ever READS the
//     simulation's memory, never allocates (the `Eval` is a `static`, the
//     reference line is memory the parent leaked before the first fork), and
//     never makes a syscall until it decides to abort.
//  2. Per-tick semantics must match `decode_rows` in the driver exactly: the
//     record is deduplicated on its whole content, and a tick is evaluated
//     when the RACE CLOCK moves on, using the last record carrying the old
//     clock. That is the same "last sample per clock value" rule the CSV path
//     uses, which is the rule that was validated to 3.4 mm.
//  3. The parent has to learn what happened even though the child exits
//     without printing anything. A one-page MAP_SHARED region carries the
//     summary out; the child updates it every tick, so the parent can read
//     progress even for candidates that ran to the end.

extern "C" {
    fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        off: i64,
    ) -> *mut c_void;
}
const PROT_RW: c_int = 3;
const MAP_SHARED: c_int = 1;
const MAP_ANONYMOUS: c_int = 0x20;

const MAXREC: usize = 256;

/// Armed configuration, set once by the `A` command in the PARENT, inherited
/// by every child through the fork.
struct WatchCfg {
    np: usize,
    preds: [Pred; MAXP],
    rl: RefLine,
    clock0: i64,
    finish_s: f32,
    fast: u32,
    plane_x: f32,
    plane_axis: usize,
    off_clock: usize,
    off_quat: usize,
    off_pos: usize,
    off_vel: usize,
    rec_len: usize,
    /// THE STATE OBJECTIVE, armed once in the parent and inherited by every
    /// fork child.
    gate: Gate,
    /// THE EVENT, armed once in the parent like the gate.
    fire: Fire,
    nseg: usize,
    seg: [(usize, usize); MAXP],
    /// MAP_SHARED page: the child's report, readable by the parent afterwards.
    out: *mut u8,
}

static mut WCFG: WatchCfg = WatchCfg {
    np: 0,
    preds: [Pred::ZERO; MAXP],
    rl: RefLine::NONE,
    clock0: 0,
    finish_s: 0.0,
    fast: 1,
    plane_x: 0.0,
    plane_axis: 0,
    off_clock: 0,
    off_quat: 4,
    off_pos: 20,
    off_vel: 32,
    rec_len: 44,
    gate: Gate::NONE,
    fire: Fire::NONE,
    nseg: 0,
    seg: [(0, 0); MAXP],
    out: core::ptr::null_mut(),
};

static mut EVAL: Eval = Eval::ZERO;
static mut WREC: [u8; MAXREC] = [0; MAXREC];
static mut WPREV: [u8; MAXREC] = [0; MAXREC];
static WPREV_VALID: AtomicUsize = AtomicUsize::new(0);
static WLAST_CLOCK: AtomicU64 = AtomicU64::new(u64::MAX);
static WATCH_ON: AtomicUsize = AtomicUsize::new(0);

#[inline]
unsafe fn rec_u32(b: *const u8, o: usize) -> u32 {
    let mut v = [0u8; 4];
    std::ptr::copy_nonoverlapping(b.add(o), v.as_mut_ptr(), 4);
    u32::from_le_bytes(v)
}

#[inline]
unsafe fn rec_f32(b: *const u8, o: usize) -> f32 {
    f32::from_bits(rec_u32(b, o))
}

/// Evaluate one finished tick and, if a predicate trips, end the child.
///
/// `clock` labels the tick being judged; `rec` is the gathered record.
#[inline(never)]
unsafe fn watch_eval(rec: *const u8, clock: i64) {
    let cfg = &*core::ptr::addr_of!(WCFG);
    let ev = &mut *core::ptr::addr_of_mut!(EVAL);
    let tick = ((clock - cfg.clock0) / 10) as i32;
    let pos = [
        rec_f32(rec, cfg.off_pos),
        rec_f32(rec, cfg.off_pos + 4),
        rec_f32(rec, cfg.off_pos + 8),
    ];
    let vel = [
        rec_f32(rec, cfg.off_vel),
        rec_f32(rec, cfg.off_vel + 4),
        rec_f32(rec, cfg.off_vel + 8),
    ];
    // The quaternion is read unconditionally and costs four loads: the gate is
    // the only thing that uses it, and a branch here would be a branch on the
    // hot path inside the game server.
    let quat = [
        rec_f32(rec, cfg.off_quat),
        rec_f32(rec, cfg.off_quat + 4),
        rec_f32(rec, cfg.off_quat + 8),
        rec_f32(rec, cfg.off_quat + 12),
    ];
    let trip = ev.feed(tick, pos, vel, quat);
    if !cfg.out.is_null() {
        let mut buf = [0u8; SUMMARY_BYTES];
        ev.sum.encode(&mut buf);
        std::ptr::copy_nonoverlapping(buf.as_ptr(), cfg.out, SUMMARY_BYTES);
    }
    if trip >= 0 {
        // dead candidate: stop paying for it. The parent sees EOF on the JSON
        // pipe and reads the verdict out of the shared page.
        _exit(0)
    }
}

#[inline(never)]
unsafe fn watch_gather(rec: *mut u8) {
    let cfg = &*core::ptr::addr_of!(WCFG);
    let mut o = 0usize;
    for s in 0..cfg.nseg {
        let (a, l) = cfg.seg[s];
        std::ptr::copy_nonoverlapping(a as *const u8, rec.add(o), l);
        o += l;
    }
}

/// The child's per-`lroundf` hook when the watchdog is armed.
///
/// Two modes, and they were measured against each other over hundreds of
/// candidates before the cheap one became the default:
///
/// * **full** -- gather the whole record on every call (255 per tick), dedup on
///   its content, and judge the last record carrying a given clock value. This
///   is exactly the rule `decode_rows` uses on the driver side, which is the
///   rule that was validated to 3.4 mm against ghost telemetry.
/// * **fast** -- read only the 4-byte race clock on every call, and gather the
///   state only when it moves. Measured, not assumed: the engine writes the
///   car's state for tick T *before* it advances the clock to T, so the first
///   call carrying clock T already sees tick T's finished state -- the same
///   bytes the full path judges at the END of clock T's span. (Judging one
///   sample earlier is exactly what makes it worth doing: an abort lands a
///   tick sooner.) It costs a load and a compare instead of a 44-byte gather
///   250 times a tick, which is 10 ms a candidate. `fk pred --mode equiv`
///   checks the two paths field by field.
#[inline(never)]
unsafe fn do_watch(clock: u64) {
    SAMPLE_NEXT.store(
        clock + SAMPLE_STRIDE.load(Ordering::Relaxed),
        Ordering::Relaxed,
    );
    let cfg = &*core::ptr::addr_of!(WCFG);
    let rec = core::ptr::addr_of_mut!(WREC) as *mut u8;
    let prev = core::ptr::addr_of_mut!(WPREV) as *mut u8;
    if cfg.fast != 0 {
        let c_now = *((cfg.seg[0].0) as *const u32) as i64;
        let last = WLAST_CLOCK.load(Ordering::Relaxed) as i64;
        if c_now == last {
            return;
        }
        watch_gather(rec);
        watch_eval(rec, c_now);
        WLAST_CLOCK.store(c_now as u64, Ordering::Relaxed);
        WPREV_VALID.store(1, Ordering::Relaxed);
        return;
    }
    watch_gather(rec);
    let n = cfg.rec_len;
    if WPREV_VALID.load(Ordering::Relaxed) != 0 {
        if std::slice::from_raw_parts(rec, n) == std::slice::from_raw_parts(prev, n) {
            return;
        }
        let cprev = rec_u32(prev, cfg.off_clock) as i64;
        if rec_u32(rec, cfg.off_clock) as i64 != cprev {
            watch_eval(prev, cprev);
        }
    }
    std::ptr::copy_nonoverlapping(rec, prev, n);
    WPREV_VALID.store(1, Ordering::Relaxed);
}

type WriteFn = unsafe extern "C" fn(c_int, *const c_void, usize) -> isize;
type ReadFn = unsafe extern "C" fn(c_int, *mut c_void, usize) -> isize;

unsafe fn real_write() -> WriteFn {
    let mut p = REAL_WRITE.load(Ordering::Relaxed);
    if p == 0 {
        p = dlsym(RTLD_NEXT, b"write\0".as_ptr() as *const c_char) as usize;
        REAL_WRITE.store(p, Ordering::Relaxed);
    }
    std::mem::transmute(p)
}

unsafe fn real_read() -> ReadFn {
    static P: AtomicUsize = AtomicUsize::new(0);
    let mut p = P.load(Ordering::Relaxed);
    if p == 0 {
        p = dlsym(RTLD_NEXT, b"read\0".as_ptr() as *const c_char) as usize;
        P.store(p, Ordering::Relaxed);
    }
    std::mem::transmute(p)
}

fn log(s: &[u8]) {
    unsafe {
        real_write()(2, s.as_ptr() as *const c_void, s.len());
    }
}

fn logn(prefix: &[u8], v: u64) {
    let mut o = Vec::with_capacity(64);
    o.extend_from_slice(prefix);
    utoa(v, &mut o);
    o.push(b'\n');
    log(&o);
}

pub fn utoa(mut v: u64, out: &mut Vec<u8>) {
    let mut d = [0u8; 24];
    let mut k = 0;
    if v == 0 {
        d[0] = b'0';
        k = 1;
    }
    while v > 0 {
        d[k] = b'0' + (v % 10) as u8;
        v /= 10;
        k += 1;
    }
    while k > 0 {
        k -= 1;
        out.push(d[k]);
    }
}

unsafe fn env_i64(name: &[u8]) -> Option<i64> {
    let e = getenv(name.as_ptr() as *const c_char);
    if e.is_null() {
        None
    } else {
        Some(atoll(e))
    }
}

unsafe fn env_str(name: &[u8]) -> Option<String> {
    let e = getenv(name.as_ptr() as *const c_char);
    if e.is_null() {
        return None;
    }
    let mut n = 0;
    while *e.add(n) != 0 {
        n += 1;
    }
    Some(String::from_utf8_lossy(std::slice::from_raw_parts(e as *const u8, n)).into_owned())
}

fn main_module_base() -> usize {
    let exe = match std::fs::read_link("/proc/self/exe") {
        Ok(p) => p.to_string_lossy().into_owned(),
        Err(_) => return 0,
    };
    let maps = match std::fs::read_to_string("/proc/self/maps") {
        Ok(s) => s,
        Err(_) => return 0,
    };
    maps.lines()
        .filter(|line| line.split_whitespace().nth(5) == Some(exe.as_str()))
        .filter_map(|line| {
            let range = line.split_whitespace().next()?;
            usize::from_str_radix(range.split_once('-')?.0, 16).ok()
        })
        .min()
        .unwrap_or(0)
}

/// Install the one-shot validator callback trap before `main` starts. Gating it
/// keeps `forkshim` usable with `shimhost`, whose text obviously has no server
/// instruction at this offset.
unsafe extern "C" fn install_validator_trace() {
    if std::env::var_os("FKSHIM_VALIDATOR_CAR").is_none() {
        return;
    }
    let base = main_module_base();
    MODULE_BASE.store(base, Ordering::SeqCst);
    if base == 0 {
        return;
    }
    post_observe::configure(base);
    let at = base + VALIDATOR_SIM_BIND_OFF;
    let got = std::slice::from_raw_parts(at as *const u8, VALIDATOR_SIM_BIND_SIGNATURE.len());
    if got != VALIDATOR_SIM_BIND_SIGNATURE {
        return;
    }
    let original = got[0];
    let ps = getpagesize() as usize;
    if mprotect((at / ps * ps) as *mut c_void, ps, 7) != 0 {
        return;
    }
    let act = SigactionT {
        handler: validator_trap_handler as *const () as usize,
        mask: [0; 16],
        flags: SA_SIGINFO,
        restorer: 0,
    };
    if sigaction(SIGTRAP, &act, &raw mut VALIDATOR_OLD_SIGTRAP) != 0 {
        mprotect((at / ps * ps) as *mut c_void, ps, PROT_READ_EXEC);
        return;
    }
    VALIDATOR_TRAP_ADDR.store(at, Ordering::SeqCst);
    VALIDATOR_TRAP_BYTE.store(original, Ordering::SeqCst);
    std::ptr::write_volatile(at as *mut u8, 0xcc);
}

#[used]
#[cfg_attr(target_os = "linux", link_section = ".init_array")]
static VALIDATOR_TRACE_INIT: unsafe extern "C" fn() = install_validator_trace;

unsafe extern "C" fn validator_trap_handler(_sig: c_int, _info: *const u8, ctx: *mut c_void) {
    if post_observe::handle(_info, ctx) { return; }
    let at = VALIDATOR_TRAP_ADDR.load(Ordering::Relaxed);
    if at == 0 || ctx.is_null() {
        _exit(94)
    }
    // Linux x86-64 ucontext_t: mcontext starts at byte 40 and gregs[REG_RIP]
    // is slot 16. rdi/rsi/rdx/rcx are slots 8/9/12/14.
    let g = (ctx as *mut usize).add(5);
    let rip = *g.add(16);
    if rip != at + 1 {
        _exit(94)
    }
    VALIDATOR_CONTROLLER.store(*g.add(8), Ordering::SeqCst);
    VALIDATOR_PARTICIPANTS_ARG.store(*g.add(9), Ordering::SeqCst);
    VALIDATOR_PARTICIPANT_COUNT_ARG.store(*g.add(12), Ordering::SeqCst);
    VALIDATOR_SIM.store(*g.add(14), Ordering::SeqCst);
    std::ptr::write_volatile(at as *mut u8, VALIDATOR_TRAP_BYTE.load(Ordering::Relaxed));
    let ps = getpagesize() as usize;
    if mprotect((at / ps * ps) as *mut c_void, ps, PROT_READ_EXEC) != 0 { _exit(94) }
    let persistent = post_observe::activate();
    if !persistent && sigaction(SIGTRAP, &raw const VALIDATOR_OLD_SIGTRAP, std::ptr::null_mut()) != 0 { _exit(94) }
    VALIDATOR_TRAP_ADDR.store(0, Ordering::SeqCst);
    *g.add(16) = at;
}

unsafe fn init() {
    if INIT.swap(1, Ordering::SeqCst) != 0 {
        return;
    }
    MODULE_BASE.store(main_module_base(), Ordering::SeqCst);
    // The chain walker's readability probe writes 8 bytes at a candidate
    // address to /dev/null and reads the errno: EFAULT for an unmapped page,
    // instead of a signal. Its fd is taken HERE, at init, before the engine
    // has opened anything -- taking it lazily mid-run shifts descriptor
    // numbering under code that tracks its own.
    DEVNULL_FD.store(open(b"/dev/null\0".as_ptr() as *const c_char, 1, 0), Ordering::SeqCst);
    if let Some(v) = env_i64(b"FKSHIM_STOP_LROUNDF\0") {
        if v > 0 {
            STOP_AT.store(v as u64, Ordering::SeqCst);
        }
    }
    if let Some(v) = env_i64(b"FKSHIM_CKPT\0") {
        if v > 0 {
            CKPT_AT.store(v as u64, Ordering::SeqCst);
        }
    }
    if let Some(v) = env_i64(b"FKSHIM_CMD_FD\0") {
        CMD_FD.store(v as i32, Ordering::SeqCst);
    }
    if let Some(v) = env_i64(b"FKSHIM_RES_FD\0") {
        RES_FD.store(v as i32, Ordering::SeqCst);
    }
}

// ---------------------------------------------------------------- input array

/// Horspool substring search (see fk/src/procmem.rs for the rationale).
struct Horspool {
    needle: [u8; 4],
    shift: [usize; 256],
}
impl Horspool {
    fn new(needle: [u8; 4]) -> Horspool {
        let mut shift = [4usize; 256];
        for i in 0..3 {
            shift[needle[i] as usize] = 3 - i;
        }
        Horspool { needle, shift }
    }
    fn find_from(&self, hay: &[u8], from: usize) -> Option<usize> {
        let n = 4;
        if hay.len() < n {
            return None;
        }
        let mut i = from;
        while i + n <= hay.len() {
            let c = hay[i + 3];
            if c == self.needle[3] && hay[i..i + 4] == self.needle {
                return Some(i);
            }
            i += self.shift[c as usize];
        }
        None
    }
}

const STRIDE: usize = 32;

/// The reference ghost's steer axis, one f32 per tick, plus the offset of the
/// most distinctive window. Written by the driver, read here.
struct Key {
    t0: usize,
    m: usize,
    steer: Vec<f32>,
}

unsafe fn read_key(path: &str) -> Option<Key> {
    let data = std::fs::read(path).ok()?;
    if data.len() < 12 {
        return None;
    }
    let n = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let t0 = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
    let m = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
    if data.len() < 12 + 4 * n {
        return None;
    }
    let steer = (0..n)
        .map(|i| f32::from_le_bytes(data[12 + 4 * i..16 + 4 * i].try_into().unwrap()))
        .collect();
    Some(Key { t0, m, steer })
}

/// Scan our own address space for the decoded input array.
unsafe fn locate(key: &Key) -> Option<usize> {
    let maps = std::fs::read_to_string("/proc/self/maps").ok()?;
    let want: Vec<[u8; 4]> = key.steer.iter().map(|v| v.to_le_bytes()).collect();
    let hs = Horspool::new(want[key.t0]);
    let n = key.steer.len();
    for line in maps.lines() {
        let mut it = line.split_whitespace();
        let range = it.next()?;
        let perms = it.next().unwrap_or("");
        if !perms.starts_with("rw") {
            continue;
        }
        let path = line.split_whitespace().nth(5).unwrap_or("");
        if path.starts_with("/dev") || path == "[vvar]" || path == "[vsyscall]" || path == "[stack]"
        {
            continue;
        }
        let (a, b) = range.split_once('-')?;
        let start = usize::from_str_radix(a, 16).ok()?;
        let end = usize::from_str_radix(b, 16).ok()?;
        let hay = std::slice::from_raw_parts(start as *const u8, end - start);
        let mut i = 0usize;
        while let Some(p) = hs.find_from(hay, i) {
            i = p + 1;
            let last = key.t0 + key.m - 1;
            if p + (key.m - 1) * STRIDE + 4 > hay.len() {
                continue;
            }
            if hay[p + (key.m - 1) * STRIDE..p + (key.m - 1) * STRIDE + 4] != want[last] {
                continue;
            }
            if p < key.t0 * STRIDE {
                continue;
            }
            let base = p - key.t0 * STRIDE;
            if base + n * STRIDE > hay.len() {
                continue;
            }
            let mut ok = true;
            for t in 0..n {
                if hay[base + t * STRIDE..base + t * STRIDE + 4] != want[t] {
                    ok = false;
                    break;
                }
            }
            if ok {
                return Some(start + base);
            }
        }
    }
    None
}

/// The steer value we believe each record holds, tick for tick.
///
/// **Why a shadow rather than the key.** The branch path verifies its cached
/// base before serving, and the obvious check â€” "does the array still read back
/// as the key?" â€” is wrong the moment anything is patched into it, which is the
/// entire point of a tree. The first depth test written against this fired
/// `ERR basemoved` at generation 2 for exactly that reason.
///
/// So the shim keeps what it believes it wrote: the key at first entry, updated
/// on every patch. A branch then verifies the array against the process's own
/// beliefs, which is a STRONGER control than the key ever was â€” it catches a
/// moved array, a stale base, AND a patch that did not land where it was aimed.
/// It is one f32 per tick (~18 KB on a 4500-tick tape), leaked once, and
/// inherited by every descendant through the fork.
static EXPECT: AtomicUsize = AtomicUsize::new(0);
static EXPECT_N: AtomicUsize = AtomicUsize::new(0);

/// Write one 12-byte input record AND record what we now believe it holds.
///
/// Every patch site goes through here. A patch that updated the array without
/// updating the shadow would make the next branch abort with `ERR basemoved`,
/// and one that updated the shadow without the array would make the check
/// blind â€” so they are the same statement.
#[inline]
unsafe fn apply_patch(base: usize, tick: usize, src: *const u8) {
    std::ptr::copy_nonoverlapping(src, (base + tick * STRIDE) as *mut u8, 12);
    let p = EXPECT.load(Ordering::Relaxed) as *mut f32;
    if !p.is_null() && tick < EXPECT_N.load(Ordering::Relaxed) {
        let mut v = [0u8; 4];
        std::ptr::copy_nonoverlapping(src, v.as_mut_ptr(), 4);
        *p.add(tick) = f32::from_le_bytes(v);
    }
}

/// Does the cached base still hold what this process believes it wrote?
///
/// The branch path does not re-scan: it inherits the address its ancestor
/// found. That is sound (the array is decoded once, before the simulation, and
/// there is exactly one copy) but it is a CLAIM, and this is what turns it back
/// into a measurement. Cost is one strided read of `n * 32` bytes.
///
/// **There is deliberately no rescan fallback here.** A cached base that no
/// longer reads back is an inconsistency about where the engine keeps its
/// inputs, and quietly re-finding the array would hide it while producing a
/// number that looks fine â€” the same comfortable recovery that
/// `tm2020-forkserver.md` bans for the boundary probe. The caller aborts.
unsafe fn base_still_holds(base: usize) -> Option<(usize, f32, f32)> {
    let p = EXPECT.load(Ordering::Relaxed) as *const f32;
    let n = EXPECT_N.load(Ordering::Relaxed);
    if p.is_null() || n == 0 {
        return Some((usize::MAX, 0.0, 0.0));
    }
    for t in 0..n {
        let mut v = [0u8; 4];
        std::ptr::copy_nonoverlapping((base + t * STRIDE) as *const u8, v.as_mut_ptr(), 4);
        let got = f32::from_le_bytes(v);
        if got.to_bits() != (*p.add(t)).to_bits() {
            return Some((t, got, *p.add(t)));
        }
    }
    None
}

// ------------------------------------------------------- boundary tick probe
//
// The one thing that can make a resumed run silently wrong is rewriting a tick
// the simulation has ALREADY read. Guessing that boundary from the clock is not
// good enough -- at a 98.9% checkpoint a four-tick probe mis-read it and two of
// thirty candidates came back 2-3 ms off. So ask the engine instead: fork a
// throwaway child, take away read access to the input array, and see which
// record it faults on next. That address IS the tick the simulation is about to
// consume, so every tick from there on is safe to rewrite.

extern "C" {
    fn mprotect(addr: *mut c_void, len: usize, prot: c_int) -> c_int;
    fn sigaction(sig: c_int, act: *const SigactionT, old: *mut SigactionT) -> c_int;
    fn getpagesize() -> c_int;
    fn getpid() -> c_int;
}

#[repr(C)]
struct SigactionT {
    handler: usize,
    mask: [u64; 16],
    flags: c_int,
    restorer: usize,
}

static mut VALIDATOR_OLD_SIGTRAP: SigactionT = SigactionT {
    handler: 0,
    mask: [0; 16],
    flags: 0,
    restorer: 0,
};

const SA_SIGINFO: c_int = 4;
const SIGSEGV: c_int = 11;
const PROT_NONE: c_int = 0;
const PROT_READ_WRITE: c_int = 3;
const PROT_READ_EXEC: c_int = 5;

static PROBE_FD: AtomicI32 = AtomicI32::new(-1);
static PROBE_BASE: AtomicUsize = AtomicUsize::new(0);
static PROBE_END: AtomicUsize = AtomicUsize::new(0);

unsafe extern "C" fn segv_handler(_sig: c_int, info: *const u8, ctx: *mut c_void) {
    let addr = *(info.add(16) as *const usize);
    let base = PROBE_BASE.load(Ordering::Relaxed);
    let end = PROBE_END.load(Ordering::Relaxed);
    let fd = PROBE_FD.load(Ordering::Relaxed);
    if addr < base || addr >= end {
        // a neighbour sharing one of the two edge pages: give that page back
        // and let the instruction retry, so the probe keeps waiting for a real
        // input-array read.
        let ps = getpagesize() as usize;
        mprotect((addr / ps * ps) as *mut c_void, ps, PROT_READ_WRITE);
        return;
    }
    // Linux x86-64 ucontext_t stores mcontext.gregs at byte 40. Record the
    // faulting instruction and register arguments before exiting the disposable
    // probe child. This makes the decoded-input reader a concrete static entry
    // into the validator call graph rather than another inferred clock marker.
    // The driver still receives the unchanged `TICK n` reply on its pipe; this
    // evidence goes only to the server's stderr log.
    #[cfg(target_arch = "x86_64")]
    if !ctx.is_null() {
        let g = (ctx as *const usize).add(5);
        for (name, index) in [
            (b" rip " as &[u8], 16usize),
            (b" rdi ", 8),
            (b" rsi ", 9),
            (b" rdx ", 12),
            (b" rcx ", 14),
            (b" r8 ", 0),
            (b" r9 ", 1),
            (b" rax ", 13),
            (b" rbx ", 11),
            (b" r12 ", 4),
            (b" r13 ", 5),
            (b" r14 ", 6),
            (b" r15 ", 7),
            (b" rbp ", 10),
            (b" rsp ", 15),
        ] {
            let mut line = Vec::with_capacity(48);
            line.extend_from_slice(b"FKSHIM input_fault");
            line.extend_from_slice(name);
            utoa(*g.add(index) as u64, &mut line);
            line.push(b'\n');
            log(&line);
        }
        logn(
            b"FKSHIM input_fault module_base ",
            MODULE_BASE.load(Ordering::Relaxed) as u64,
        );
        let sp = *g.add(15);
        let mut fp = *g.add(10) as *const usize;
        for depth in 0..12u64 {
            let at = fp as usize;
            if at < sp || at.saturating_sub(sp) > (1 << 20) || at & 7 != 0 {
                break;
            }
            let next = *fp;
            let ret = *fp.add(1);
            let mut line = Vec::with_capacity(64);
            line.extend_from_slice(b"FKSHIM input_fault frame ");
            utoa(depth, &mut line);
            line.push(b' ');
            utoa(ret as u64, &mut line);
            line.push(b'\n');
            log(&line);
            if next <= at {
                break;
            }
            fp = next as *const usize;
        }
    }
    let mut o = Vec::with_capacity(64);
    o.extend_from_slice(b"TICK ");
    // Root bounded fix: GNU objdump 2.42 decoded the saved input_fault
    // RVA 0x119f169 as movups xmm0,[rdx+rcx]. The vector includes the
    // preceding 4-byte record header; flooring its address underreports
    // the decoded input-channel index by one. Bind to this exact reader,
    // its register arithmetic and the caller-verified server SHA.
    if ctx.is_null() { write_all(fd, b"ERR no register context\n"); _exit(0); }
    let g = (ctx as *const usize).add(5);
    let module = MODULE_BASE.load(Ordering::Relaxed);
    let rip = *g.add(16);
    let rdx = *g.add(12);
    let rcx = *g.add(14);
    let index = *g.add(9); // rsi: independently decoded native index
    let count = (end - base) / STRIDE;
    if module == 0 || rip != module + 0x119f169 || rdx.checked_add(4) != Some(base)
        || index >= count || index.checked_mul(STRIDE) != Some(rcx)
        || rdx.checked_add(rcx) != Some(addr) {
        write_all(fd, b"ERR unsupported input reader; no index guess\n");
        _exit(0);
    }
    logn(b"FKSHIM proven_input_index ", index as u64);
    utoa(index as u64, &mut o);
    o.push(b'\n');
    write_all(fd, &o);
    _exit(0)
}

/// In a forked child: revoke read access to every page the input array touches
/// (including the two partial edge pages -- the tail of the array lives in the
/// last one, and a checkpoint late in the run reads only there) and let the
/// simulation walk into it.
unsafe fn arm_probe(base: usize, n: usize, fd: c_int) {
    let ps = getpagesize() as usize;
    let lo = base / ps * ps;
    let hi = (base + n * STRIDE + ps - 1) / ps * ps;
    PROBE_FD.store(fd, Ordering::SeqCst);
    PROBE_BASE.store(base, Ordering::SeqCst);
    PROBE_END.store(base + n * STRIDE, Ordering::SeqCst);
    let act = SigactionT {
        handler: segv_handler as *const () as usize,
        mask: [0; 16],
        flags: SA_SIGINFO,
        restorer: 0,
    };
    sigaction(SIGSEGV, &act, std::ptr::null_mut());
    if hi > lo {
        mprotect(lo as *mut c_void, hi - lo, PROT_NONE);
    }
}

// ------------------------------------------------------------------- protocol

unsafe fn read_exact(fd: c_int, buf: &mut [u8]) -> bool {
    let mut got = 0usize;
    while got < buf.len() {
        let r = real_read()(
            fd,
            buf.as_mut_ptr().add(got) as *mut c_void,
            buf.len() - got,
        );
        if r <= 0 {
            return false;
        }
        got += r as usize;
    }
    true
}

unsafe fn write_all(fd: c_int, buf: &[u8]) -> bool {
    let mut sent = 0usize;
    while sent < buf.len() {
        let r = real_write()(
            fd,
            buf.as_ptr().add(sent) as *const c_void,
            buf.len() - sent,
        );
        if r <= 0 {
            return false;
        }
        sent += r as usize;
    }
    true
}

unsafe fn send_frame(fd: c_int, payload: &[u8]) {
    let mut hdr = (payload.len() as u32).to_le_bytes().to_vec();
    hdr.extend_from_slice(payload);
    write_all(fd, &hdr);
}

/// The fork server. Entered once, in the parent, at the checkpoint; never
/// returns there. Returns in the *child*, so the simulation resumes with the
/// patched tail.
/// Parse an `A` (arm) payload into `WCFG`, and return `(predicates,
/// reference points)`.
///
/// Extracted from the command loop so it can be TESTED against the driver that
/// writes it: `tmoracle::pred::Watch::arm_payload` is the only producer of
/// these bytes, this is the only consumer, and `tests` in this crate feed one
/// to the other. The wire format is:
///
/// ```text
/// 'A' | u32 np | np * Pred
///     | i64 clock0 | u32 off_clock off_pos off_vel rec_len
///     | u32 nseg | nseg * (u64 addr, u32 len)
///     | f32 corridor | i32 ahead | i32 back | f32 finish_s | u32 fast
///     | u32 nref | nref * 3 f32 xyz | nref * f32 arclength
///     | f32 plane_x        (trailing: an older shim ignores it)
///     | u32 off_quat | u32 gate_armed | 6 f32 bounds | f32 minspeed
///     | u32 nkops | nkops * KeyOp
/// ```
///
/// It is parsed ONCE, in the parent, so every later fork inherits it for free.
/// That matters most for the reference line: it is 30 KB and must not cross the
/// pipe per candidate.
///
/// The gate block is trailing like `plane_x`, but unlike `plane_x` a shim that
/// silently ignored it would score every candidate "never reached the gate" --
/// a perfectly plausible answer that is a lie. So the count of key operations
/// installed goes back in the ARM ack and the driver refuses a mismatch.
unsafe fn parse_arm(payload: &[u8]) -> (usize, usize, usize) {
    let cfg = &mut *core::ptr::addr_of_mut!(WCFG);
    let mut o = 1usize;
    let g4 = |o: usize| u32::from_le_bytes(payload[o..o + 4].try_into().unwrap());
    let np = (g4(o) as usize).min(MAXP);
    o += 4;
    for i in 0..np {
        cfg.preds[i] = Pred::decode(&payload[o..o + PRED_BYTES]);
        o += PRED_BYTES;
    }
    cfg.np = np;
    cfg.clock0 = i64::from_le_bytes(payload[o..o + 8].try_into().unwrap());
    o += 8;
    cfg.off_clock = g4(o) as usize;
    cfg.off_pos = g4(o + 4) as usize;
    cfg.off_vel = g4(o + 8) as usize;
    cfg.rec_len = (g4(o + 12) as usize).min(MAXREC);
    o += 16;
    let nseg = (g4(o) as usize).min(MAXP);
    o += 4;
    cfg.nseg = nseg;
    for s in 0..nseg {
        cfg.seg[s] = (
            u64::from_le_bytes(payload[o..o + 8].try_into().unwrap()) as usize,
            g4(o + 8) as usize,
        );
        o += 12;
    }
    let corridor = f32::from_bits(g4(o));
    let ahead = g4(o + 4) as i32;
    let back = g4(o + 8) as i32;
    cfg.finish_s = f32::from_bits(g4(o + 12));
    cfg.fast = g4(o + 16);
    o += 20;
    let nref = g4(o) as usize;
    o += 4;
    if nref > 0 {
        let mut xyz: Vec<f32> = Vec::with_capacity(3 * nref);
        for i in 0..3 * nref {
            xyz.push(f32::from_bits(g4(o + 4 * i)));
        }
        o += 12 * nref;
        let mut s: Vec<f32> = Vec::with_capacity(nref);
        for i in 0..nref {
            s.push(f32::from_bits(g4(o + 4 * i)));
        }
        o += 4 * nref;
        cfg.rl = RefLine {
            n: nref,
            xyz: xyz.as_ptr(),
            s: s.as_ptr(),
            corridor,
            ahead,
            back,
        };
        // leaked on purpose: every child must see it at the same address, and
        // it is written once per server
        std::mem::forget(xyz);
        std::mem::forget(s);
    } else {
        cfg.rl = RefLine::NONE;
    }
    cfg.plane_x = if payload.len() >= o + 4 {
        f32::from_bits(g4(o))
    } else {
        0.0
    };
    o += 4;
    let mut nk = 0usize;
    if payload.len() >= o + 8 {
        cfg.off_quat = g4(o) as usize;
        let armed = g4(o + 4) != 0;
        o += 8;
        let mut gate = Gate::NONE;
        gate.armed = armed;
        for i in 0..6 {
            gate.bounds[i] = f32::from_bits(g4(o + 4 * i));
        }
        o += 24;
        gate.minspeed = f32::from_bits(g4(o));
        o += 4;
        nk = (g4(o) as usize).min(MAXKOPS - 1);
        o += 4;
        for i in 0..nk {
            gate.prog[i] = KeyOp::decode(&payload[o..o + KEYOP_BYTES]);
            o += KEYOP_BYTES;
        }
        cfg.gate = gate;
        // THE EVENT CLAUSE, trailing behind the gate.
        if payload.len() >= o + 20 {
            let mut fire = Fire::NONE;
            fire.armed = g4(o) != 0;
            fire.at = f32::from_bits(g4(o + 4));
            fire.need = g4(o + 8).max(1);
            fire.after_ticks = g4(o + 12);
            fire.after_from_end = g4(o + 16) != 0;
            o += 20;
            fire.where_box.armed = g4(o) != 0;
            o += 4;
            for i in 0..6 {
                fire.where_box.bounds[i] = f32::from_bits(g4(o + 4 * i));
            }
            o += 24;
            for which in 0..2 {
                let n = (g4(o) as usize).min(MAXKOPS - 1);
                o += 4;
                for i in 0..n {
                    let k = KeyOp::decode(&payload[o..o + KEYOP_BYTES]);
                    if which == 0 {
                        fire.cond[i] = k;
                    } else {
                        fire.after[i] = k;
                    }
                    o += KEYOP_BYTES;
                }
                nk += n;
            }
            cfg.fire = fire;
            // THE PLANE AXIS, trailing behind the event; absent means x.
            cfg.plane_axis = if payload.len() >= o + 4 { (g4(o) as usize).min(2) } else { 0 };
            o += 4;
        } else {
            cfg.fire = Fire::NONE;
            cfg.plane_axis = 0;
        }
    } else {
        cfg.gate = Gate::NONE;
        cfg.fire = Fire::NONE;
        cfg.plane_axis = 0;
    }
    let _ = o;
    (np, nref, nk)
}

// K's new wire has a strict bounded validator before legacy B parsing.
fn validate_clock_frame(p: &[u8], input_count: usize) -> Option<()> {
    fn take<'a>(p: &'a [u8], o: &mut usize, n: usize) -> Option<&'a [u8]> {
        let end = o.checked_add(n)?; let part = p.get(*o..end)?; *o=end; Some(part)
    }
    fn u32at(p: &[u8], o: &mut usize) -> Option<u32> { Some(u32::from_le_bytes(take(p,o,4)?.try_into().ok()?)) }
    fn u64at(p: &[u8], o: &mut usize) -> Option<u64> { Some(u64::from_le_bytes(take(p,o,8)?.try_into().ok()?)) }
    let mut o=1usize; let np=u32at(p,&mut o)? as usize;
    if np>100_000 || np>input_count { return None; }
    let target=u64at(p,&mut o)?; if target>u32::MAX as u64 {return None;}
    let slen=u32at(p,&mut o)? as usize; if slen==0 || slen>107 {return None;}
    let sock=take(p,&mut o,slen)?; if sock.first()!=Some(&b'/') || sock.contains(&0) {return None;}
    let tlen=u32at(p,&mut o)? as usize; if tlen==0 || tlen>1024 {return None;}
    let trace=take(p,&mut o,tlen)?; if trace.first()!=Some(&b'/') || trace.contains(&0) {return None;}
    if u32at(p,&mut o)?!=2 {return None;}
    for expected_len in [4u32,40] {
        let address=u64at(p,&mut o)?; let length=u32at(p,&mut o)?;
        if address<4096 || address%4!=0 || length!=expected_len || address.checked_add(length as u64).is_none() {return None;}
    }
    if u64at(p,&mut o)?!=1 {return None;}
    let max=u32at(p,&mut o)?; if max==0 || max>200_000 {return None;}
    if u32at(p,&mut o)?!=44 || u32at(p,&mut o)?!=0 {return None;}
    let mut previous=None;
    for _ in 0..np {
        let tick=u32at(p,&mut o)? as usize;
        if tick>=input_count || previous.map_or(false, |x| tick<=x) {return None;}
        previous=Some(tick);
        for axis in 0..3 {
            let value=f32::from_bits(u32at(p,&mut o)?);
            if !value.is_finite() || value>1. || value<if axis==0 {-1.} else {0.} {return None;}
        }
    }
    if o!=p.len() {return None;} Some(())
}

fn clock_range_readable(address: usize, length: usize) -> bool {
    let end=match address.checked_add(length) {Some(v)=>v,None=>return false};
    let maps=match std::fs::read_to_string("/proc/self/maps") {Ok(v)=>v,Err(_)=>return false};
    maps.lines().any(|line| {
        let mut parts=line.split_whitespace();
        let range=match parts.next(){Some(v)=>v,None=>return false};
        let perm=match parts.next(){Some(v)=>v,None=>return false};
        let (lo,hi)=match range.split_once('-'){Some(v)=>v,None=>return false};
        match (usize::from_str_radix(lo,16),usize::from_str_radix(hi,16)) {
            (Ok(lo),Ok(hi))=>perm.starts_with('r') && lo<=address && end<=hi,
            _=>false
        }
    })
}

unsafe fn forkserver() {
    // A BRANCH RE-ENTRY consumes its licence on the way in, so a child that
    // re-enters once cannot do it twice by accident.
    let branch = BRANCH_ARMED.swap(0, Ordering::SeqCst) != 0;
    CLOCK_STOP_ADDR.store(0, Ordering::SeqCst);
    CLOCK_STOP_TARGET.store(0, Ordering::SeqCst);

    // The trace file this child wrote while it consumed its k ticks. Closed
    // BEFORE the handshake, so the driver that reads it after READY reads a
    // complete file rather than a racing one.
    let tfd = TRACE_FD.swap(-1, Ordering::SeqCst);
    if tfd >= 0 {
        SAMPLE_STRIDE.store(0, Ordering::SeqCst);
        SAMPLE_NEXT.store(u64::MAX, Ordering::SeqCst);
        SAMPLE_FD.store(-1, Ordering::SeqCst);
        close(tfd);
    }

    let (cmd, res) = if branch {
        // FRESH FDS. Serving the inherited command pipe would put two processes
        // in a race for every command byte, and the loser would execute a
        // command assembled from the middle of somebody else's patch payload.
        let path = &*core::ptr::addr_of!(BRANCH_SOCK);
        let n = BRANCH_SOCK_LEN.load(Ordering::SeqCst);
        let fd = connect_unix(&path[..n]);
        if fd < 0 {
            log(b"FKSHIM: branch could not reach the driver socket\n");
            _exit(98)
        }
        // Drop everything this child inherited that belongs to an ancestor: the
        // root server's command pipe, and the socket its parent branch is
        // serving on. Otherwise a dead driver is never seen as EOF, which is
        // exactly the orphan-holds-the-pipe failure the CLOEXEC work fixed on
        // the driver side.
        let a = CMD_FD.swap(-1, Ordering::SeqCst);
        let b = RES_FD.swap(-1, Ordering::SeqCst);
        if a >= 0 {
            close(a);
        }
        if b >= 0 && b != a {
            close(b);
        }
        let s = SERVING_FD.swap(fd, Ordering::SeqCst);
        if s >= 0 && s != fd {
            close(s);
        }
        (fd, fd)
    } else {
        let c = CMD_FD.load(Ordering::SeqCst);
        let r = RES_FD.load(Ordering::SeqCst);
        SERVING_FD.store(r, Ordering::SeqCst);
        (c, r)
    };

    // The key, parsed once per process tree and leaked so a branch pays no
    // file read for it.
    let key: &Key = {
        let p = CACHED_KEY.load(Ordering::SeqCst);
        if p != 0 {
            &*(p as *const Key)
        } else {
            let keypath = match env_str(b"FKSHIM_KEY\0") {
                Some(p) => p,
                None => {
                    log(b"FKSHIM: no FKSHIM_KEY\n");
                    _exit(97)
                }
            };
            let k = match read_key(&keypath) {
                Some(k) => k,
                None => {
                    log(b"FKSHIM: bad key file\n");
                    _exit(97)
                }
            };
            let leaked: &'static Key = Box::leak(Box::new(k));
            CACHED_KEY.store(leaked as *const Key as usize, Ordering::SeqCst);
            leaked
        }
    };

    let base = if branch {
        let b = CACHED_BASE.load(Ordering::SeqCst);
        // HARD ABORT, never a rescan. See `base_still_holds`.
        if b == 0 {
            log(b"FKSHIM: branch has no cached base -- aborting\n");
            send_frame(res, b"ERR basemoved no-cache");
            _exit(95)
        }
        if let Some((t, got, want)) = base_still_holds(b) {
            // Say WHAT it saw. A control that only says "no" cannot tell a
            // moved array from a patch that did not land where it was aimed,
            // and those want different fixes.
            let mut m = Vec::new();
            m.extend_from_slice(b"ERR basemoved tick ");
            utoa(t as u64, &mut m);
            m.extend_from_slice(b" got ");
            utoa(got.to_bits() as u64, &mut m);
            m.extend_from_slice(b" want ");
            utoa(want.to_bits() as u64, &mut m);
            log(b"FKSHIM: branch base no longer holds what we wrote -- aborting\n");
            log(&m);
            log(b"\n");
            send_frame(res, &m);
            _exit(95)
        }
        b
    } else {
        match locate(key) {
            Some(b) => {
                CACHED_BASE.store(b, Ordering::SeqCst);
                // The shadow starts as the key: at first entry nothing has been
                // patched, so what we believe the array holds IS the reference.
                let mut e: Vec<f32> = key.steer.clone();
                EXPECT_N.store(e.len(), Ordering::SeqCst);
                EXPECT.store(e.as_mut_ptr() as usize, Ordering::SeqCst);
                std::mem::forget(e);
                b
            }
            None => {
                log(b"FKSHIM: input array NOT FOUND\n");
                send_frame(res, b"ERR notfound");
                _exit(96)
            }
        }
    };
    let mut hello = Vec::new();
    signal(SIGCHLD, SIG_IGN);
    hello.extend_from_slice(b"READY ");
    utoa(base as u64, &mut hello);
    hello.push(b' ');
    utoa(N_LROUNDF.load(Ordering::Relaxed), &mut hello);
    // The branch's own pid, so the driver that now owns this node can end it.
    // A node the driver cannot kill is an orphan holding a 150 MB address
    // space, and a beam of them is how a box dies.
    hello.push(b' ');
    utoa(getpid() as u64, &mut hello);
    // The validator-owned root captured at 0x118c170. Both values are sent so
    // the driver can prove the callback argument agrees with the object's own
    // +0x1a70 field before following the rest of the chain.
    hello.push(b' ');
    utoa(
        VALIDATOR_CONTROLLER.load(Ordering::SeqCst) as u64,
        &mut hello,
    );
    hello.push(b' ');
    utoa(VALIDATOR_SIM.load(Ordering::SeqCst) as u64, &mut hello);
    send_frame(res, &hello);

    loop {
        let mut lenb = [0u8; 4];
        if !read_exact(cmd, &mut lenb) {
            _exit(0);
        }
        let len = u32::from_le_bytes(lenb) as usize;
        let mut payload = vec![0u8; len];
        if len > 0 && !read_exact(cmd, &mut payload) {
            _exit(0);
        }
        if len == 0 || payload[0] == b'Q' {
            _exit(0);
        }
        if payload[0] == b'N' {
            // null fork: how much of the per-candidate cost is fork + child
            // startup alone, with no simulation at all?
            let t_start = now_us();
            let mut fds = [0i32; 2];
            if pipe(fds.as_mut_ptr()) != 0 {
                send_frame(res, b"ERR pipe");
                continue;
            }
            let pid = fork();
            if pid < 0 {
                close(fds[0]);
                close(fds[1]);
                send_frame(res, b"ERR fork");
                continue;
            }
            if pid == 0 {
                close(fds[0]);
                write_all(fds[1], b"x");
                _exit(0);
            }
            let t_forked = now_us();
            close(fds[1]);
            let mut b = [0u8; 8];
            real_read()(fds[0], b.as_mut_ptr() as *mut c_void, 8);
            let t_got = now_us();
            kill(pid, SIGKILL);
            close(fds[0]);
            let mut o = Vec::new();
            o.extend_from_slice(b"NULLFORK fork_us ");
            utoa(t_forked - t_start, &mut o);
            o.extend_from_slice(b" roundtrip_us ");
            utoa(t_got - t_start, &mut o);
            o.push(b'\n');
            send_frame(res, &o);
            continue;
        }
        if payload[0] == b'P' {
            // boundary probe
            fflush(std::ptr::null_mut());
            let mut fds = [0i32; 2];
            if pipe(fds.as_mut_ptr()) != 0 {
                send_frame(res, b"ERR pipe");
                continue;
            }
            let pid = fork();
            // A FAILED FORK MUST NOT BECOME kill(-1). `fork()` returns -1 under
            // EAGAIN, and the handlers below all end in `kill(pid, SIGKILL)`:
            // with pid = -1 that is "signal every process this user can reach",
            // which on a box running five searches means killing all of them.
            if pid < 0 {
                close(fds[0]);
                close(fds[1]);
                send_frame(res, b"ERR fork");
                continue;
            }
            if pid == 0 {
                IS_CHILD.store(1, Ordering::SeqCst);
                close(fds[0]);
                dup2(fds[1], 1);
                arm_probe(base, key.steer.len(), fds[1]);
                return;
            }
            close(fds[1]);
            let mut out: Vec<u8> = Vec::new();
            let mut buf = [0u8; 1024];
            loop {
                let mut pfd = PollFd {
                    fd: fds[0],
                    events: POLLIN,
                    revents: 0,
                };
                if poll(&mut pfd, 1, 20000) <= 0 {
                    out.extend_from_slice(b"PROBE-TIMEOUT");
                    break;
                }
                let r = real_read()(fds[0], buf.as_mut_ptr() as *mut c_void, buf.len());
                if r <= 0 {
                    break;
                }
                out.extend_from_slice(&buf[..r as usize]);
                if out.windows(5).any(|w| w == b"TICK ") {
                    break;
                }
            }
            kill(pid, SIGKILL);
            let mut st = 0i32;
            waitpid(pid, &mut st, 0);
            close(fds[0]);
            send_frame(res, &out);
            continue;
        }
        if payload[0] == b'B' || payload[0] == b'K' {
            let clock_mode = payload[0] == b'K';
            if clock_mode && validate_clock_frame(&payload, key.steer.len()).is_none() {
                send_frame(res, b"ERR malformed-clock-frame"); continue;
            }
            // THE BRANCH. Fork a child that patches its tail, consumes a fixed
            // number of `lroundf` calls, and then re-enters this same loop on a
            // socket of its own -- a new fork point, a node of a savestate
            // tree, rather than a candidate that runs to the end and dies.
            //
            //   'B' | u32 n_patch | u64 stop_after_lroundf
            //       | u32 sock_len | sock_path
            //       | u32 trace_len | trace_path        (0 = no state trace)
            //       | u32 nseg | nseg * (u64 addr, u32 len)
            //       | u64 sstride | u32 smax | u32 sdedup | u32 skeyoff
            //       | n_patch * (u32 tick, f32 steer, f32 gas, f32 brake)
            //
            // The parent does NOT wait and does NOT kill: it answers
            // `BRANCHED <pid>` at once and goes back to serving. The node
            // announces itself when it arrives, down its own socket.
            let np = u32::from_le_bytes(payload[1..5].try_into().unwrap()) as usize;
            let stop_after = u64::from_le_bytes(payload[5..13].try_into().unwrap());
            let mut o = 13usize;
            let slen = u32::from_le_bytes(payload[o..o + 4].try_into().unwrap()) as usize;
            o += 4;
            let sock = payload[o..o + slen].to_vec();
            o += slen;
            let tlen = u32::from_le_bytes(payload[o..o + 4].try_into().unwrap()) as usize;
            o += 4;
            let mut tracep: Vec<u8> = payload[o..o + tlen].to_vec();
            tracep.push(0);
            o += tlen;
            let nseg =
                (u32::from_le_bytes(payload[o..o + 4].try_into().unwrap()) as usize).min(MAX_SEG);
            o += 4;
            let mut segs = [(0usize, 0usize); MAX_SEG];
            let mut sblen = 0usize;
            for s in 0..nseg {
                segs[s] = (
                    u64::from_le_bytes(payload[o..o + 8].try_into().unwrap()) as usize,
                    u32::from_le_bytes(payload[o + 8..o + 12].try_into().unwrap()) as usize,
                );
                sblen += segs[s].1;
                o += 12;
            }
            let sstride = u64::from_le_bytes(payload[o..o + 8].try_into().unwrap());
            let smax = u32::from_le_bytes(payload[o + 8..o + 12].try_into().unwrap()) as u64;
            let sdedup = u32::from_le_bytes(payload[o + 12..o + 16].try_into().unwrap()) as usize;
            let skeyoff = u32::from_le_bytes(payload[o + 16..o + 20].try_into().unwrap()) as usize;
            o += 20;
            let poff = o;
            if slen == 0 || slen > 107 {
                send_frame(res, b"ERR socklen");
                continue;
            }
            if clock_mode {
                if nseg != 2 || segs[0].1 != 4 || segs[0].0 & 3 != 0
                    || !clock_range_readable(segs[0].0,4) || !clock_range_readable(segs[1].0,40)
                    || stop_after > u32::MAX as u64 {
                    send_frame(res, b"ERR clock-address-or-target"); continue;
                }
                let now = core::ptr::read_volatile(segs[0].0 as *const u32) as u64;
                if stop_after <= now || stop_after - now > 10_000 || (stop_after - now) % 10 != 0 {
                    send_frame(res, b"ERR clock-target-range-or-phase"); continue;
                }
            }
            fflush(std::ptr::null_mut());
            let pid = fork();
            if pid < 0 {
                send_frame(res, b"ERR fork");
                continue;
            }
            if pid > 0 {
                let mut o = Vec::new();
                o.extend_from_slice(b"BRANCHED ");
                utoa(pid as u64, &mut o);
                send_frame(res, &o);
                continue;
            }
            // ---- child: becomes a node of the tree
            IS_CHILD.store(1, Ordering::SeqCst);
            if clock_mode {
                WATCH_ON.store(0, Ordering::SeqCst);
                CHAIN_N.store(0, Ordering::SeqCst);
                CHAIN_ARMED.store(false, Ordering::SeqCst);
                GATE_ADDR.store(0, Ordering::SeqCst);
                GATE_MOD.store(0, Ordering::SeqCst);
                GATE_PHASE.store(0, Ordering::SeqCst);
                SAMPLE_DEADLINE.store(0, Ordering::SeqCst);
                SAMPLE_EXIT.store(0, Ordering::SeqCst);
            }
            for i in 0..np {
                let q = poff + i * 16;
                let tick = u32::from_le_bytes(payload[q..q + 4].try_into().unwrap()) as usize;
                apply_patch(base, tick, payload.as_ptr().add(q + 4));
            }
            {
                let b = &mut *core::ptr::addr_of_mut!(BRANCH_SOCK);
                b[..slen].copy_from_slice(&sock);
                BRANCH_SOCK_LEN.store(slen, Ordering::SeqCst);
            }
            // The state trace, if the driver asked for one. Same sampler the
            // 'S' path uses, aimed at a file instead of a pipe: nobody is
            // reading the other end yet, and a pipe that fills would stall the
            // simulation we are trying to time.
            if nseg > 0 && tlen > 0 {
                let fd = open(tracep.as_ptr() as *const c_char, 577, 0o644);
                if fd < 0 {
                    log(b"FKSHIM: branch could not open its trace file\n");
                    _exit(94)
                }
                let mut bb = vec![0u8; 8 + sblen];
                let mut pv = vec![0xFFu8; sblen.max(1)];
                SAMPLE_BUF.store(bb.as_mut_ptr() as usize, Ordering::SeqCst);
                SAMPLE_PREV.store(pv.as_mut_ptr() as usize, Ordering::SeqCst);
                std::mem::forget(bb);
                std::mem::forget(pv);
                SAMPLE_ADDR.store(segs[0].0, Ordering::SeqCst);
                SAMPLE_LEN.store(sblen, Ordering::SeqCst);
                SEG_N.store(nseg, Ordering::SeqCst);
                for s in 0..nseg {
                    SEG_ADDR[s].store(segs[s].0, Ordering::SeqCst);
                    SEG_LEN[s].store(segs[s].1, Ordering::SeqCst);
                }
                SAMPLE_FD.store(fd, Ordering::SeqCst);
                TRACE_FD.store(fd, Ordering::SeqCst);
                SAMPLE_LEFT.store(smax, Ordering::SeqCst);
                SAMPLE_EXIT.store(0, Ordering::SeqCst);
                SAMPLE_DEADLINE.store(0, Ordering::SeqCst);
                SAMPLE_DEDUP.store(sdedup, Ordering::SeqCst);
                SAMPLE_KEYOFF.store(skeyoff, Ordering::SeqCst);
                GATE_ADDR.store(0, Ordering::SeqCst);
                GATE_MOD.store(0, Ordering::SeqCst);
                SAMPLE_STRIDE.store(sstride.max(1), Ordering::SeqCst);
                SAMPLE_NEXT.store(0, Ordering::SeqCst);
            } else {
                SAMPLE_STRIDE.store(0, Ordering::SeqCst);
                SAMPLE_NEXT.store(u64::MAX, Ordering::SeqCst);
            }
            // Re-arm the checkpoint: `ARMED` is the one-shot latch that stops a
            // process entering the fork server twice, and `BRANCH_ARMED` is the
            // one-use licence that lets THIS child past the `IS_CHILD` guard.
            if clock_mode {
                CLOCK_STOP_TARGET.store(stop_after, Ordering::SeqCst);
                CLOCK_STOP_ADDR.store(segs[0].0, Ordering::SeqCst);
                CKPT_AT.store(u64::MAX, Ordering::SeqCst);
            } else {
            CKPT_AT.store(
                N_LROUNDF
                    .load(Ordering::Relaxed)
                    .saturating_add(stop_after.max(1)),
                Ordering::SeqCst,
            );
            }
            ARMED.store(0, Ordering::SeqCst);
            BRANCH_ARMED.store(1, Ordering::SeqCst);
            return; // resume the simulation; re-enter in `stop_after` calls
        }
        if payload[0] == b'S' {
            // AN 'S' FORK MUST NOT SEE THE CHAIN. `find_clock2` hunts the race
            // clock with 'S' commands, and a forked child inherits CHAIN_N --
            // so the chain rewrote segment 1 inside the very scan that was
            // trying to measure it, and the scan failed with "no slot advances
            // by a multiple of 10 per tick". The chain belongs to the parent's
            // own 'G' run and nothing else. Proven by control: with a
            // zero-hop chain armed, the same scan succeeds.
            CHAIN_ARMED.store(false, Ordering::SeqCst);
            // Sample-and-run: like 'R', but the child also gathers segments of
            // memory out on a second pipe as it simulates. Two frames come
            // back: the validator's JSON, then the raw sample blob.
            //   u32 n_patch | u32 n_seg | u64 stride | u32 max
            //   | u32 key_len | u32 key_off
            //   | n_seg * (u64 addr, u32 len)
            //   | n_patch * (u32 tick, f32 steer, f32 gas, f32 brake)
            let np = u32::from_le_bytes(payload[1..5].try_into().unwrap()) as usize;
            let nseg = u32::from_le_bytes(payload[5..9].try_into().unwrap()) as usize;
            let sstride = u64::from_le_bytes(payload[9..17].try_into().unwrap());
            let smax_raw = u32::from_le_bytes(payload[17..21].try_into().unwrap());
            let sexit = (smax_raw & 0x8000_0000) != 0;
            let smax = (smax_raw & 0x7fff_ffff) as u64;
            let sdedup = u32::from_le_bytes(payload[21..25].try_into().unwrap()) as usize;
            let skeyoff = u32::from_le_bytes(payload[25..29].try_into().unwrap()) as usize;
            let mut segs = [(0usize, 0usize); MAX_SEG];
            let mut slen = 0usize;
            for s in 0..nseg.min(MAX_SEG) {
                let o = 29 + s * 12;
                segs[s] = (
                    u64::from_le_bytes(payload[o..o + 8].try_into().unwrap()) as usize,
                    u32::from_le_bytes(payload[o + 8..o + 12].try_into().unwrap()) as usize,
                );
                slen += segs[s].1;
            }
            let poff = 29 + nseg * 12;
            // OPTIONAL trailing u32 (older drivers do not send it): how many
            // more `lroundf` calls this child may simulate before it exits.
            // ~255 calls to the tick, so 6 ticks is ~1530.
            let sbudget = {
                let o = poff + np * 16;
                if payload.len() >= o + 4 {
                    u32::from_le_bytes(payload[o..o + 4].try_into().unwrap()) as u64
                } else {
                    0
                }
            };

            // OPTIONAL trailing (u64 gate_addr, u32 gate_mod, u32 gate_phase)
            let (gaddr, gmod, gphase) = {
                let o = poff + np * 16 + 4;
                if payload.len() >= o + 16 {
                    (
                        u64::from_le_bytes(payload[o..o + 8].try_into().unwrap()) as usize,
                        u32::from_le_bytes(payload[o + 8..o + 12].try_into().unwrap()) as u64,
                        u32::from_le_bytes(payload[o + 12..o + 16].try_into().unwrap()) as u64,
                    )
                } else {
                    (0, 0, 0)
                }
            };
            fflush(std::ptr::null_mut());
            let t_start = now_us();
            let mut fds = [0i32; 2];
            let mut sfds = [0i32; 2];
            if pipe(fds.as_mut_ptr()) != 0 || pipe(sfds.as_mut_ptr()) != 0 {
                // TWO frames, because that is what the driver reads for 'S'.
                // Answering with one desynchronises the stream permanently:
                // every later response is read as the answer to the previous
                // command, and the server eventually executes a command byte
                // taken from the middle of a patch payload.
                send_frame(res, b"ERR pipe");
                send_frame(res, b"");
                continue;
            }
            let pid = fork();
            if pid < 0 {
                close(fds[0]);
                close(fds[1]);
                close(sfds[0]);
                close(sfds[1]);
                send_frame(res, b"ERR fork");
                send_frame(res, b"");
                continue;
            }
            if pid == 0 {
                IS_CHILD.store(1, Ordering::SeqCst);
                close(fds[0]);
                close(sfds[0]);
                dup2(fds[1], 1);
                close(fds[1]);
                setvbuf(stdout, std::ptr::null_mut(), IONBF, 0);
                for i in 0..np {
                    let o = poff + i * 16;
                    let tick = u32::from_le_bytes(payload[o..o + 4].try_into().unwrap()) as usize;
                    apply_patch(base, tick, payload.as_ptr().add(o + 4));
                }
                // buffers are allocated BEFORE the hook can fire, so the hot
                // path never allocates
                let mut b = vec![0u8; 8 + slen];
                let mut p = vec![0xFFu8; slen.max(1)];
                SAMPLE_BUF.store(b.as_mut_ptr() as usize, Ordering::SeqCst);
                SAMPLE_PREV.store(p.as_mut_ptr() as usize, Ordering::SeqCst);
                std::mem::forget(b);
                std::mem::forget(p);
                SAMPLE_ADDR.store(segs[0].0, Ordering::SeqCst);
                SAMPLE_LEN.store(slen, Ordering::SeqCst);
                SEG_N.store(nseg.min(MAX_SEG), Ordering::SeqCst);
                for s in 0..nseg.min(MAX_SEG) {
                    SEG_ADDR[s].store(segs[s].0, Ordering::SeqCst);
                    SEG_LEN[s].store(segs[s].1, Ordering::SeqCst);
                }
                SAMPLE_FD.store(sfds[1], Ordering::SeqCst);
                SAMPLE_LEFT.store(smax, Ordering::SeqCst);
                SAMPLE_EXIT.store(if sexit { 1 } else { 0 }, Ordering::SeqCst);
                SAMPLE_DEADLINE.store(
                    if sbudget == 0 {
                        0
                    } else {
                        N_LROUNDF.load(Ordering::Relaxed) + sbudget
                    },
                    Ordering::SeqCst,
                );
                SAMPLE_DEDUP.store(sdedup, Ordering::SeqCst);
                SAMPLE_KEYOFF.store(skeyoff, Ordering::SeqCst);
                GATE_ADDR.store(gaddr, Ordering::SeqCst);
                GATE_MOD.store(gmod, Ordering::SeqCst);
                GATE_PHASE.store(gphase, Ordering::SeqCst);
                SAMPLE_STRIDE.store(sstride.max(1), Ordering::SeqCst);
                SAMPLE_NEXT.store(0, Ordering::SeqCst);
                return;
            }
            let t_forked = now_us();
            close(fds[1]);
            close(sfds[1]);
            let mut out: Vec<u8> = Vec::with_capacity(4096);
            let mut samples: Vec<u8> = Vec::with_capacity(1 << 20);
            let mut buf = [0u8; 65536];
            let mut json_done = false;
            let mut samples_eof = false;
            let mut t_first = 0u64;
            loop {
                let mut pfds = [
                    PollFd {
                        fd: if json_done { -1 } else { fds[0] },
                        events: POLLIN,
                        revents: 0,
                    },
                    PollFd {
                        fd: if samples_eof { -1 } else { sfds[0] },
                        events: POLLIN,
                        revents: 0,
                    },
                ];
                if json_done && samples_eof {
                    break;
                }
                let pr = poll(pfds.as_mut_ptr(), 2, 60000);
                if pr <= 0 {
                    out.extend_from_slice(b"\nFKSHIM-TIMEOUT\n");
                    break;
                }
                if !samples_eof && pfds[1].revents != 0 {
                    let r = real_read()(sfds[0], buf.as_mut_ptr() as *mut c_void, buf.len());
                    if r <= 0 {
                        samples_eof = true;
                    } else {
                        samples.extend_from_slice(&buf[..r as usize]);
                    }
                }
                if !json_done && pfds[0].revents != 0 {
                    let r = real_read()(fds[0], buf.as_mut_ptr() as *mut c_void, buf.len());
                    if r <= 0 {
                        json_done = true;
                    } else {
                        if t_first == 0 {
                            t_first = now_us();
                        }
                        out.extend_from_slice(&buf[..r as usize]);
                        // Bound collection to the COMPLETE outer object, with
                        // quoted/escaped braces ignored. Host still parses JSON
                        // strictly; this is a framing detector, not a validator.
                        let mut depth = 0usize;
                        let mut started = false;
                        let mut quoted = false;
                        let mut escaped = false;
                        for &byte in &out {
                            if quoted {
                                if escaped { escaped = false; }
                                else if byte == b'\\' { escaped = true; }
                                else if byte == b'"' { quoted = false; }
                                continue;
                            }
                            if byte == b'"' && started { quoted = true; }
                            else if byte == b'{' { depth += 1; started = true; }
                            else if byte == b'}' && started {
                                depth = depth.saturating_sub(1);
                                if depth == 0 {
                                    json_done = true;
                                    kill(pid, SIGKILL);
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            if !samples_eof {
                // drain whatever the pipe still holds
                loop {
                    let mut pfd = PollFd {
                        fd: sfds[0],
                        events: POLLIN,
                        revents: 0,
                    };
                    if poll(&mut pfd, 1, 200) <= 0 {
                        break;
                    }
                    let r = real_read()(sfds[0], buf.as_mut_ptr() as *mut c_void, buf.len());
                    if r <= 0 {
                        break;
                    }
                    samples.extend_from_slice(&buf[..r as usize]);
                }
            }
            let complete = json_done && samples_eof;
            if !complete { kill(pid, SIGKILL); }
            close(fds[0]);
            close(sfds[0]);
            out.extend_from_slice(b"\nFKCOMPLETE stdout_record_or_eof ");
            utoa(if json_done { 1 } else { 0 }, &mut out);
            out.extend_from_slice(b" samples_eof ");
            utoa(if samples_eof { 1 } else { 0 }, &mut out);
            out.push(b'\n');
            out.extend_from_slice(b"\nFKTIME fork_us ");
            utoa(t_forked - t_start, &mut out);
            out.extend_from_slice(b" first_us ");
            utoa(t_first.saturating_sub(t_forked), &mut out);
            out.extend_from_slice(b" done_us ");
            utoa(now_us() - t_start, &mut out);
            out.push(b'\n');
            send_frame(res, &out);
            send_frame(res, &samples);
            continue;
        }
        if payload[0] == b'C' {
            // REPLY FIRST. Separates "the handler never runs" from "the
            // handler runs and dies before it answers" -- both look like
            // "arm_chain: no reply" to the caller.
            if std::env::var("FK_CHAIN_PING").is_ok() {
                send_frame(res, b"CHAIN at 0");
                continue;
            }
            // ARM THE SAMPLER'S POINTER CHAIN.
            //
            //   u64 root | u64 back | u32 n | n x u64 off
            //
            // Segment 0's address is then recomputed at every sampled instant
            // as (walk(root, offs) - back), so a car the engine reallocates
            // mid-race is followed instead of going stale. n = 0 disarms.
            if payload.len() >= 21 {
                let root = u64::from_le_bytes(payload[1..9].try_into().unwrap()) as usize;
                let back = u64::from_le_bytes(payload[9..17].try_into().unwrap()) as usize;
                let n = u32::from_le_bytes(payload[17..21].try_into().unwrap()) as usize;
                let n = n.min(8);
                for i in 0..n {
                    let o = 21 + i * 8;
                    if o + 8 <= payload.len() {
                        CHAIN_OFF[i].store(
                            u64::from_le_bytes(payload[o..o + 8].try_into().unwrap()) as usize,
                            Ordering::SeqCst,
                        );
                    }
                }
                CHAIN_ROOT.store(root, Ordering::SeqCst);
                CHAIN_BACK.store(back, Ordering::SeqCst);
                CHAIN_N.store(n, Ordering::SeqCst);
                // Bisect point 2: parsed and stored, nothing walked yet.
                if std::env::var("FK_CHAIN_PING2").is_ok() {
                    send_frame(res, b"CHAIN at 2");
                    continue;
                }
                // Walk it ONCE here and report where it lands, so the caller
                // can compare against an address it resolved by other means in
                // this same process. Without this, a chain that walks into
                // nothing is indistinguishable from a chain that walks
                // somewhere fine and is sampled wrongly -- both show up only
                // as "0 instants sampled", three layers away.
                // EVERY DEREFERENCE IS CHECKED. An unchecked walk SIGSEGVs the
                // server, and the caller sees "arm_chain: no reply" -- which is
                // how the first version of this failed. A chain that walks into
                // nothing must report that, not kill the process.
                // ONE HOP AT A TIME, reporting which one broke, so a bad
                // chain names its own bad hop instead of dying anonymously.
                // Record every hop so the reply can say where the walk went,
                // not merely that it failed.
                let mut hops = [0usize; 8];
                let mut a = root;
                let mut ok = true;
                let mut broke_at = 99usize;
                for i in 0..n {
                    if a == 0 || a < 0x1000 || !addr_readable(a) {
                        ok = false;
                        broke_at = i;
                        break;
                    }
                    a = *(a as *const usize);
                    if a == 0 || a < 0x1000 { ok = false; break; }
                    a = a.wrapping_add(CHAIN_OFF[i].load(Ordering::Relaxed));
                    if i < 8 { hops[i] = a; }
                }
                let mut msg = [0u8; 200];
                let m = if ok {
                    let mut w = 0usize;
                    for (i, b) in b"CHAIN at ".iter().enumerate() { msg[i] = *b; w = i + 1; }
                    // the landing address, then every hop that led to it
                    fn put(msg: &mut [u8], w: &mut usize, v0: usize) {
                        let mut v = v0;
                        let mut d = [0u8; 20];
                        let mut dn = 0;
                        if v == 0 { d[0] = b'0'; dn = 1; }
                        while v > 0 { d[dn] = b'0' + (v % 10) as u8; v /= 10; dn += 1; }
                        while dn > 0 { dn -= 1; msg[*w] = d[dn]; *w += 1; }
                    }
                    put(&mut msg, &mut w, a);
                    for b in b" hops" { msg[w] = *b; w += 1; }
                    for i in 0..n.min(8) {
                        msg[w] = b' '; w += 1;
                        let h = hops[i];
                        put(&mut msg, &mut w, h);
                    }
                    &msg[..w]
                } else {
                    let mut w = 0usize;
                    for (i, b) in b"CHAIN broke at hop ".iter().enumerate() { msg[i] = *b; w = i + 1; }
                    msg[w] = b'0' + (broke_at.min(9) as u8);
                    &msg[..w + 1]
                };
                send_frame(res, m);
            } else {
                send_frame(res, b"ERR chain");
            }
            continue;
        }
        if payload[0] == b'G' {
            // GO: sample the rest of THIS process's run and return to the
            // simulation. No fork, no resume, no input patching.
            //
            // WHY THIS EXISTS: a forked child is NOT the clean run. Measured on
            // this engine (banked as `hl_RESULT_v13`, and reproduced here on
            // 208024): the same tape resumed from two different checkpoints
            // agrees on 0 of 522 ticks and diverges by up to 2.9 m, and against
            // a human ghost's own recorded path a resume at race -30 ms is
            // 5.578 m out where a resume at race 140 ms is 0.0055 m. Telemetry
            // regenerated from a resumed child would therefore describe a
            // slightly different run from the one the oracle validates. The
            // parent, on the other hand, is an ordinary /validatepath run: it
            // has forked children (which cannot touch its memory) and is
            // otherwise untouched.
            //
            //   u32 n_seg | u64 stride | u32 max | u32 key_len | u32 key_off
            //   | u64 gate_addr | u32 gate_mod | u32 gate_phase
            //   | u32 path_len | path | n_seg * (u64 addr, u32 len)
            let nseg = u32::from_le_bytes(payload[1..5].try_into().unwrap()) as usize;
            let sstride = u64::from_le_bytes(payload[5..13].try_into().unwrap());
            let smax = u32::from_le_bytes(payload[13..17].try_into().unwrap()) as u64;
            let sdedup = u32::from_le_bytes(payload[17..21].try_into().unwrap()) as usize;
            let skeyoff = u32::from_le_bytes(payload[21..25].try_into().unwrap()) as usize;
            let gaddr = u64::from_le_bytes(payload[25..33].try_into().unwrap()) as usize;
            let gmod = u32::from_le_bytes(payload[33..37].try_into().unwrap()) as u64;
            let gphase = u32::from_le_bytes(payload[37..41].try_into().unwrap()) as u64;
            let plen = u32::from_le_bytes(payload[41..45].try_into().unwrap()) as usize;
            let mut path: Vec<u8> = payload[45..45 + plen].to_vec();
            path.push(0);
            let mut segs = [(0usize, 0usize); MAX_SEG];
            let mut slen = 0usize;
            for s in 0..nseg.min(MAX_SEG) {
                let o = 45 + plen + s * 12;
                segs[s] = (
                    u64::from_le_bytes(payload[o..o + 8].try_into().unwrap()) as usize,
                    u32::from_le_bytes(payload[o + 8..o + 12].try_into().unwrap()) as usize,
                );
                slen += segs[s].1;
            }
            let fd = open(path.as_ptr() as *const c_char, 577, 0o644);
            if fd < 0 {
                send_frame(res, b"ERR open");
                continue;
            }
            let mut b = vec![0u8; 8 + slen];
            let mut pv = vec![0xFFu8; slen.max(1)];
            SAMPLE_BUF.store(b.as_mut_ptr() as usize, Ordering::SeqCst);
            SAMPLE_PREV.store(pv.as_mut_ptr() as usize, Ordering::SeqCst);
            std::mem::forget(b);
            std::mem::forget(pv);
            SAMPLE_ADDR.store(segs[0].0, Ordering::SeqCst);
            SAMPLE_LEN.store(slen, Ordering::SeqCst);
            SEG_N.store(nseg.min(MAX_SEG), Ordering::SeqCst);
            for s in 0..nseg.min(MAX_SEG) {
                SEG_ADDR[s].store(segs[s].0, Ordering::SeqCst);
                SEG_LEN[s].store(segs[s].1, Ordering::SeqCst);
            }
            SAMPLE_FD.store(fd, Ordering::SeqCst);
            SAMPLE_LEFT.store(smax as u64, Ordering::SeqCst);
            SAMPLE_EXIT.store(0, Ordering::SeqCst);
            SAMPLE_DEADLINE.store(0, Ordering::SeqCst);
            SAMPLE_DEDUP.store(sdedup, Ordering::SeqCst);
            SAMPLE_KEYOFF.store(skeyoff, Ordering::SeqCst);
            GATE_ADDR.store(gaddr, Ordering::SeqCst);
            GATE_MOD.store(gmod, Ordering::SeqCst);
            GATE_PHASE.store(gphase, Ordering::SeqCst);
            SAMPLE_STRIDE.store(sstride.max(1), Ordering::SeqCst);
            SAMPLE_NEXT.store(0, Ordering::SeqCst);
            CHAIN_ARMED.store(true, Ordering::SeqCst);
            post_observe::prepare_action();
            send_frame(res, b"GO");
            return;
        }
        if payload[0] == b'A' {
            let (np, nref, nk) = parse_arm(&payload);
            let cfg = &mut *core::ptr::addr_of_mut!(WCFG);
            if cfg.out.is_null() {
                let p = mmap(
                    std::ptr::null_mut(),
                    4096,
                    PROT_RW,
                    MAP_SHARED | MAP_ANONYMOUS,
                    -1,
                    0,
                );
                if p as isize == -1 {
                    send_frame(res, b"ERR mmap");
                    continue;
                }
                cfg.out = p as *mut u8;
            }
            let mut ack = Vec::new();
            ack.extend_from_slice(b"ARMED ");
            utoa(np as u64, &mut ack);
            ack.extend_from_slice(b" preds, ");
            utoa(nref as u64, &mut ack);
            ack.extend_from_slice(b" reference points, ");
            utoa(nk as u64, &mut ack);
            ack.extend_from_slice(b" key ops");
            send_frame(res, &ack);
            continue;
        }
        if payload[0] == b'W' {
            // Run one candidate with the watchdog armed. Same wire shape as
            // 'R'; two frames come back: the validator's JSON (empty if the
            // child was aborted) and the fixed-size summary.
            let n = u32::from_le_bytes(payload[1..5].try_into().unwrap()) as usize;
            let cfg = &mut *core::ptr::addr_of_mut!(WCFG);
            if cfg.out.is_null() || cfg.nseg == 0 {
                send_frame(res, b"ERR not armed");
                send_frame(res, b"");
                continue;
            }
            // clear the shared report before the fork, so a child that dies
            // without writing cannot be mistaken for one that reported
            std::ptr::write_bytes(cfg.out, 0, SUMMARY_BYTES);
            fflush(std::ptr::null_mut());
            let t_start = now_us();
            let mut fds = [0i32; 2];
            if pipe(fds.as_mut_ptr()) != 0 {
                send_frame(res, b"ERR pipe");
                send_frame(res, b"");
                continue;
            }
            let pid = fork();
            if pid < 0 {
                close(fds[0]);
                close(fds[1]);
                send_frame(res, b"ERR fork");
                send_frame(res, b"");
                continue;
            }
            if pid == 0 {
                IS_CHILD.store(1, Ordering::SeqCst);
                close(fds[0]);
                dup2(fds[1], 1);
                close(fds[1]);
                setvbuf(stdout, std::ptr::null_mut(), IONBF, 0);
                for i in 0..n {
                    let o = 5 + i * 16;
                    let tick = u32::from_le_bytes(payload[o..o + 4].try_into().unwrap()) as usize;
                    apply_patch(base, tick, payload.as_ptr().add(o + 4));
                }
                let ev = &mut *core::ptr::addr_of_mut!(EVAL);
                ev.reset();
                ev.np = cfg.np;
                ev.preds = cfg.preds;
                ev.rl = cfg.rl;
                ev.finish_s = cfg.finish_s;
                ev.plane_x = cfg.plane_x;
                ev.plane_axis = cfg.plane_axis;
                ev.gate = cfg.gate;
                ev.fire = cfg.fire;
                WPREV_VALID.store(0, Ordering::SeqCst);
                WLAST_CLOCK.store(u64::MAX, Ordering::SeqCst);
                WATCH_ON.store(1, Ordering::SeqCst);
                SAMPLE_STRIDE.store(1, Ordering::SeqCst);
                SAMPLE_NEXT.store(0, Ordering::SeqCst);
                return; // resume the simulation, watched
            }
            let t_forked = now_us();
            let mut t_first = 0u64;
            close(fds[1]);
            let mut out: Vec<u8> = Vec::with_capacity(4096);
            let mut buf = [0u8; 4096];
            let mut done = false;
            while !done {
                let mut pfd = PollFd {
                    fd: fds[0],
                    events: POLLIN,
                    revents: 0,
                };
                let pr = poll(&mut pfd, 1, 20000);
                if pr <= 0 {
                    out.extend_from_slice(b"\nFKSHIM-TIMEOUT\n");
                    break;
                }
                let r = real_read()(fds[0], buf.as_mut_ptr() as *mut c_void, buf.len());
                if r <= 0 {
                    break; // the child aborted itself, or finished and closed
                }
                if t_first == 0 {
                    t_first = now_us();
                }
                out.extend_from_slice(&buf[..r as usize]);
                if out.windows(9).any(|w| w == b"\"IsValid\"") {
                    done = true;
                }
            }
            kill(pid, SIGKILL);
            close(fds[0]);
            out.extend_from_slice(b"\nFKTIME fork_us ");
            utoa(t_forked - t_start, &mut out);
            out.extend_from_slice(b" first_us ");
            utoa(t_first.saturating_sub(t_forked), &mut out);
            out.extend_from_slice(b" done_us ");
            utoa(now_us() - t_start, &mut out);
            out.push(b'\n');
            let mut sum = [0u8; SUMMARY_BYTES];
            std::ptr::copy_nonoverlapping(cfg.out, sum.as_mut_ptr(), SUMMARY_BYTES);
            send_frame(res, &out);
            send_frame(res, &sum);
            continue;
        }
        // 'R' u32 n, then n * (u32 tick, f32 steer, f32 gas, f32 brake)
        let n = u32::from_le_bytes(payload[1..5].try_into().unwrap()) as usize;
        fflush(std::ptr::null_mut()); // no half-written stdio in the child
        let t_start = now_us();
        let mut fds = [0i32; 2];
        if pipe(fds.as_mut_ptr()) != 0 {
            send_frame(res, b"ERR pipe");
            continue;
        }
        let pid = fork();
        if pid < 0 {
            close(fds[0]);
            close(fds[1]);
            send_frame(res, b"ERR fork");
            continue;
        }
        if pid == 0 {
            // ---- child: becomes the candidate's simulator
            IS_CHILD.store(1, Ordering::SeqCst);
            close(fds[0]);
            dup2(fds[1], 1);
            close(fds[1]);
            setvbuf(stdout, std::ptr::null_mut(), IONBF, 0);
            for i in 0..n {
                let o = 5 + i * 16;
                let tick = u32::from_le_bytes(payload[o..o + 4].try_into().unwrap()) as usize;
                apply_patch(base, tick, payload.as_ptr().add(o + 4));
            }
            return; // resume the simulation, with the tail rewritten
        }
        // ---- parent: collect the child's JSON, then stop it dead
        let t_forked = now_us();
        let mut t_first = 0u64;
        close(fds[1]);
        let mut out: Vec<u8> = Vec::with_capacity(4096);
        let mut buf = [0u8; 4096];
        let mut done = false;
        while !done {
            let mut pfd = PollFd {
                fd: fds[0],
                events: POLLIN,
                revents: 0,
            };
            let pr = poll(&mut pfd, 1, 20000);
            if pr <= 0 {
                out.extend_from_slice(b"\nFKSHIM-TIMEOUT\n");
                break;
            }
            let r = real_read()(fds[0], buf.as_mut_ptr() as *mut c_void, buf.len());
            if r <= 0 {
                break;
            }
            if t_first == 0 {
                t_first = now_us();
            }
            out.extend_from_slice(&buf[..r as usize]);
            // everything we need is in ValidatedResult/Desc, which precede IsValid;
            // stopping here skips the DeclaredResult block and the Inputs RLE
            if out.windows(9).any(|w| w == b"\"IsValid\"") {
                done = true;
            }
        }
        if pid > 0 {
            // SIGKILL and walk away: tearing down a 150 MB address space costs
            // milliseconds, and SIGCHLD=SIG_IGN makes the kernel reap for us, so
            // the next candidate does not have to wait for it.
            kill(pid, SIGKILL);
        }
        close(fds[0]);
        out.extend_from_slice(b"\nFKTIME fork_us ");
        utoa(t_forked - t_start, &mut out);
        out.extend_from_slice(b" first_us ");
        utoa(t_first.saturating_sub(t_forked), &mut out);
        out.extend_from_slice(b" done_us ");
        utoa(now_us() - t_start, &mut out);
        out.push(b'\n');
        send_frame(res, &out);
    }
}

fn now_us() -> u64 {
    #[repr(C)]
    struct Ts {
        s: i64,
        ns: i64,
    }
    extern "C" {
        fn clock_gettime(id: c_int, t: *mut Ts) -> c_int;
    }
    let mut t = Ts { s: 0, ns: 0 };
    unsafe {
        clock_gettime(1, &mut t);
    }
    (t.s as u64) * 1_000_000 + (t.ns as u64) / 1000
}

// ---------------------------------------------------------------------- hooks

#[no_mangle]
pub unsafe extern "C" fn lroundf(x: f32) -> i64 {
    let mut p = REAL_LROUNDF.load(Ordering::Relaxed);
    if p == 0 {
        init();
        p = dlsym(RTLD_NEXT, b"lroundf\0".as_ptr() as *const c_char) as usize;
        REAL_LROUNDF.store(p, Ordering::Relaxed);
    }
    let n = N_LROUNDF.fetch_add(1, Ordering::Relaxed) + 1;
    if n >= SAMPLE_NEXT.load(Ordering::Relaxed) {
        if WATCH_ON.load(Ordering::Relaxed) != 0 {
            do_watch(n);
        } else {
            do_sample(n);
        }
    }
    let clock_addr = CLOCK_STOP_ADDR.load(Ordering::SeqCst);
    if clock_addr != 0 && BRANCH_ARMED.load(Ordering::SeqCst) != 0 {
        let now = core::ptr::read_volatile(clock_addr as *const u32) as u64;
        let target = CLOCK_STOP_TARGET.load(Ordering::SeqCst);
        if now > target { log(b"FKSTEP clock overshoot; rejected\n"); _exit(93); }
        if now == target && ARMED.swap(1, Ordering::SeqCst) == 0 {
            logn(b"FKSTEP raw_clock_stop ", now);
            forkserver();
        }
    }
    if n >= STOP_AT.load(Ordering::Relaxed) && ARMED.swap(1, Ordering::SeqCst) == 0 {
        log(b"FKSHIM stop\n");
        raise(SIGSTOP);
    }
    if n >= CKPT_AT.load(Ordering::Relaxed)
        && (IS_CHILD.load(Ordering::Relaxed) == 0 || BRANCH_ARMED.load(Ordering::Relaxed) != 0)
        && ARMED.swap(1, Ordering::SeqCst) == 0
    {
        forkserver();
    }
    let f: unsafe extern "C" fn(f32) -> i64 = std::mem::transmute(p);
    f(x)
}

fn emit() {
    if IS_CHILD.load(Ordering::Relaxed) == 0 {
        logn(b"FKSHIM lroundf_total ", N_LROUNDF.load(Ordering::Relaxed));
    }
}

#[used]
#[link_section = ".fini_array"]
static FINI: extern "C" fn() = {
    extern "C" fn f() {
        emit()
    }
    f
};

// ---------------------------------------------------------------------- tests
//
// The shim runs inside another process's address space and cannot be exercised
// by ordinary means. What CAN be tested here is the thing most likely to break
// silently: the wire format between the driver that arms the watchdog and the
// child that evaluates it. `forkoracle::pred::Watch::arm_payload` is the only
// producer of those bytes and `parse_arm` above is the only consumer, so the
// test below runs one against the other. If a field is added on one side and
// not the other, this fails instead of a search quietly watching for the wrong
// thing.

#[cfg(test)]
mod tests {
    use super::*;
    use forkoracle::pred::{parse_fire, parse_gate, parse_spec, RefLineData, Watch};

    /// `WCFG` is a `static mut` -- the child holds it in `.bss` and never
    /// allocates -- so two tests calling `parse_arm` in parallel threads write
    /// over each other. They did, and the failure looked like a wire-format bug
    /// rather than a test-harness one. One lock, and each test sees its own
    /// payload.
    static ARM: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn a_watch() -> Watch {
        let mut w = Watch::new();
        w.corridor = 40.0;
        w.ahead = 24;
        w.back = 8;
        w.finish_s = 1397.0;
        w.fast = 1;
        w.plane_x = 123.5;
        w.refline = RefLineData::from_points(
            &(0..64)
                .map(|i| [i as f64 * 1.5, 2.0, -i as f64])
                .collect::<Vec<_>>(),
        );
        for s in [
            "crash:speeddrop:frac=0.5,win=50,minpeak=15,after=200",
            "stuck:floor:speed=3,need=50,after=250",
        ] {
            w.preds.push(parse_spec(s).unwrap());
        }
        w.gate = parse_gate(
            "xmin=56,xmax=136,ymin=48,ymax=54,zmin=704,zmax=713,minspeed=60",
            "min(abs(bodyright), 5*(-vz)) * nose(0.888,0.451,-0.086)",
        )
        .unwrap();
        w.fire = parse_fire(
            "dspeed",
            10.0,
            3,
            "xmin=56,xmax=80,ymin=48,ymax=54,zmin=704,zmax=713",
            "-dist(366,50,736)",
            7,
            true,
        )
        .unwrap();
        w
    }

    #[test]
    fn the_arm_payload_survives_the_crossing_into_the_child() {
        let _g = ARM.lock().unwrap_or_else(|e| e.into_inner());
        let w = a_watch();
        let segs = [(0x7f00_1000u64, 4u32), (0x7f00_2000, 40)];
        let payload = w.arm_payload(-1580, 0, 4, 20, 32, 44, &segs);

        let (np, nref, nk) = unsafe { parse_arm(&payload) };
        assert_eq!(np, 2);
        assert_eq!(nref, 64);
        assert_eq!(nk, w.nkops(), "the key program did not survive the pipe");

        let cfg = unsafe { &*core::ptr::addr_of!(WCFG) };
        assert_eq!(cfg.np, 2);
        assert_eq!(cfg.clock0, -1580);
        assert_eq!(cfg.off_clock, 0);
        assert_eq!(cfg.off_quat, 4);
        assert_eq!(cfg.off_pos, 20);
        assert_eq!(cfg.off_vel, 32);
        assert_eq!(cfg.rec_len, 44);
        assert_eq!(cfg.nseg, 2);
        assert_eq!(cfg.seg[0], (0x7f00_1000, 4));
        assert_eq!(cfg.seg[1], (0x7f00_2000, 40));
        assert_eq!(cfg.finish_s, 1397.0);
        assert_eq!(cfg.fast, 1);
        assert_eq!(cfg.plane_x, 123.5, "the trailing timing plane was lost");
        assert_eq!(cfg.rl.n, 64);
        assert_eq!(cfg.rl.corridor, 40.0);
        assert_eq!(cfg.rl.ahead, 24);
        assert_eq!(cfg.rl.back, 8);

        // the reference line itself, point for point
        for i in 0..64 {
            let x = unsafe { *cfg.rl.xyz.add(3 * i) };
            let y = unsafe { *cfg.rl.xyz.add(3 * i + 1) };
            let z = unsafe { *cfg.rl.xyz.add(3 * i + 2) };
            assert_eq!(
                (x, y, z),
                (i as f32 * 1.5, 2.0, -(i as f32)),
                "reference point {}",
                i
            );
        }
        let last = unsafe { *cfg.rl.s.add(63) };
        assert!(
            (last - w.refline.s[63]).abs() < 1e-3,
            "arclength did not survive"
        );

        // THE GATE, box and program, byte for byte
        assert!(cfg.gate.armed, "the gate did not survive the pipe");
        assert_eq!(cfg.gate.bounds, w.gate.bounds);
        assert_eq!(cfg.gate.minspeed, w.gate.minspeed);
        for i in 0..w.nkops() {
            let (mut a, mut b) = ([0u8; KEYOP_BYTES], [0u8; KEYOP_BYTES]);
            w.gate.prog[i].encode(&mut a);
            cfg.gate.prog[i].encode(&mut b);
            assert_eq!(a, b, "key operation {} changed crossing the pipe", i);
        }
        // THE EVENT, box and both programs
        assert!(cfg.fire.armed, "the event clause did not survive the pipe");
        assert_eq!(cfg.fire.at, w.fire.at);
        assert_eq!(
            cfg.fire.need, w.fire.need,
            "the event's need count did not cross the pipe"
        );
        assert_eq!(
            cfg.fire.after_ticks, w.fire.after_ticks,
            "the after window did not cross"
        );
        assert_eq!(
            cfg.fire.after_from_end, w.fire.after_from_end,
            "which end the after window opens at did not cross the pipe"
        );
        assert!(cfg.fire.where_box.armed);
        assert_eq!(cfg.fire.where_box.bounds, w.fire.where_box.bounds);
        for (a, b) in [
            (&cfg.fire.cond, &w.fire.cond),
            (&cfg.fire.after, &w.fire.after),
        ] {
            for i in 0..forkoracle::pred::prog_len(b) {
                let (mut x, mut y) = ([0u8; KEYOP_BYTES], [0u8; KEYOP_BYTES]);
                a[i].encode(&mut x);
                b[i].encode(&mut y);
                assert_eq!(x, y, "event key operation {} changed crossing the pipe", i);
            }
        }
        // and it computes the same number on both sides of the wire
        let (p, v, q) = (
            [70.2, 50.4, 708.9],
            [-60.0, -20.0, -80.0],
            [0.7, 0.1, 0.7, 0.05],
        );
        assert_eq!(
            pred_core::key_eval(&cfg.gate.prog, pred_core::St::at(p, v, q)),
            forkoracle::pred_core::key_eval(&w.gate.prog, forkoracle::pred_core::St::at(p, v, q))
        );

        // and the predicates, by re-encoding what the child now holds
        for (i, np) in w.preds.iter().enumerate() {
            let mut a = [0u8; PRED_BYTES];
            let mut b = [0u8; PRED_BYTES];
            np.pred.encode(&mut a);
            cfg.preds[i].encode(&mut b);
            assert_eq!(a, b, "predicate {} changed crossing the pipe", i);
        }
    }

    /// The timing plane, the gate and the event are all written after the
    /// fields an older shim knows about, so a shim built before any of them
    /// existed simply ignores what it does not recognise. That is only true if
    /// a short payload is safe at every one of those boundaries.
    #[test]
    fn a_payload_cut_at_any_trailing_boundary_is_still_read() {
        let _g = ARM.lock().unwrap_or_else(|e| e.into_inner());
        let w = a_watch();
        let full = w.arm_payload(0, 0, 4, 20, 32, 44, &[(0x1000, 4)]);

        // 1. cut before the EVENT block: gate armed, event not.
        let fire_bytes = 4
            + 4
            + 4
            + 4
            + 4
            + 4
            + 24
            + 4
            + KEYOP_BYTES * forkoracle::pred::prog_len(&w.fire.cond)
            + 4
            + KEYOP_BYTES * forkoracle::pred::prog_len(&w.fire.after);
        // arm_payload also appends plane_axis after the event block. Remove
        // that trailing u32 when testing a legacy frame without the event.
        let cut = full.len() - fire_bytes - std::mem::size_of::<u32>();
        let (np, nref, nk) = unsafe { parse_arm(&full[..cut]) };
        assert_eq!((np, nref), (2, 64));
        assert_eq!(
            nk,
            w.gate_kops(),
            "the gate's own key did not survive the cut"
        );
        let cfg = unsafe { &*core::ptr::addr_of!(WCFG) };
        assert!(cfg.gate.armed, "the gate was lost by a cut that is past it");
        assert!(
            !cfg.fire.armed,
            "a missing event clause must read as disarmed"
        );

        // 2. cut before the GATE block too: both disarmed, reference intact.
        let cut = cut - 4 - 8 - 24 - 4 - 4 - KEYOP_BYTES * w.gate_kops();
        let (np, nref, nk) = unsafe { parse_arm(&full[..cut]) };
        assert_eq!((np, nref, nk), (2, 64, 0));
        let cfg = unsafe { &*core::ptr::addr_of!(WCFG) };
        assert_eq!(
            cfg.plane_x, 0.0,
            "a missing timing plane must read as disabled"
        );
        assert!(!cfg.gate.armed, "a missing gate must read as disarmed");
        assert!(
            !cfg.fire.armed,
            "a missing event clause must read as disarmed"
        );
        assert_eq!(
            cfg.rl.n, 64,
            "the reference line was damaged by the short read"
        );
    }
}

// THE BOOT IS NOT THE SLEEPING, THOUGH IT LOOKS LIKE IT. Measured, and left
// here so it is not re-derived:
//
//   strace -c on a bare boot: clock_nanosleep 69.96% of syscall time, 273
//   calls, and summing the wall time of each gives 2.74 s inside a 2.24 s
//   boot -- more than the boot itself, because they are concurrent IDLE
//   threads, not the critical path.
//
// An LD_PRELOAD interceptor for `nanosleep` (the symbol the binary actually
// imports -- `objdump -T` shows `nanosleep`, while strace reports both under
// one name) DOES bind: with the shim loaded, strace counts ZERO sleep calls.
// The boot still takes 2.24 s. So the sleeps cost nothing and the boot's real
// cost is elsewhere -- most likely the 27,174 FAILING stat calls and the
// engine's own start-up work. Interception was the right instinct and the
// wrong target.
