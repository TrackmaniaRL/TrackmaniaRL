//! `shimhost` — a stand-in engine, so the savestate tree can be tested with no
//! game, no map and no `.Ghost.Gbx` at all.
//!
//! # Why this exists
//!
//! The fork shim is an `LD_PRELOAD` interposer on `lroundf`. Everything it does
//! — count calls, stop at a checkpoint, fork, patch the decoded input array,
//! probe the consumed boundary, re-enter as a branch node on a fresh socket —
//! is about **process mechanics and one array in memory**. None of it is about
//! Trackmania.
//!
//! So the mechanism can be exercised against a program that merely *behaves
//! like* the engine in the three ways the shim depends on:
//!
//! 1. it calls `lroundf` a fixed number of times per simulated tick;
//! 2. it holds one decoded input array, 32 bytes per tick, in an `rw` heap
//!    mapping, and reads it strictly **in tick order, one record per tick** —
//!    which is what makes the page-fault probe meaningful;
//! 3. it prints a verdict containing `"IsValid"` when it finishes.
//!
//! That is enough to test the whole tree end to end, including the thing that
//! matters most: **a record already consumed cannot be un-consumed.** The
//! host's verdict is a hash of the records it actually consumed, so a write
//! that lands above the boundary changes the answer and a write that lands
//! below it does not — the exact signature of the defect the forward-only rule
//! exists for, reproducible in milliseconds with no engine.
//!
//! # What it does NOT establish
//!
//! Nothing about cost. This host has a ~1 MB address space and no physics; the
//! Q1 numbers are about forking a ~150 MB engine and simulating real ticks, and
//! they can only be measured on the real thing. It also says nothing about
//! whether the real engine reads its input array the way this one does — that
//! is what the page-fault probe measures on the real engine, and it is why the
//! probe is asked of the engine rather than assumed.

use std::io::Write;

const STRIDE: usize = 32;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let n: usize = a.first().and_then(|v| v.parse().ok()).unwrap_or(2000);
    let per_tick: usize = a.get(1).and_then(|v| v.parse().ok()).unwrap_or(255);
    let key_path = std::env::var("FKSHIM_KEY").expect("shimhost needs FKSHIM_KEY");

    // The decoded input array, exactly the shape the engine holds: one 32-byte
    // record per tick, `f32 steer, f32 gas, f32 brake`, in tick order, one
    // allocation, in an anonymous rw mapping.
    let steer = key_steer(&key_path, n);
    let mut arr = vec![0u8; n * STRIDE];
    for t in 0..n {
        arr[t * STRIDE..t * STRIDE + 4].copy_from_slice(&steer[t].to_le_bytes());
        arr[t * STRIDE + 4..t * STRIDE + 8].copy_from_slice(&1.0f32.to_le_bytes());
        arr[t * STRIDE + 8..t * STRIDE + 12].copy_from_slice(&0.0f32.to_le_bytes());
        // A distinguishable tail so a stray 12-byte patch is visible.
        arr[t * STRIDE + 16..t * STRIDE + 20].copy_from_slice(&(t as u32).to_le_bytes());
    }

    // The "simulation". The order inside a tick is the load-bearing part: the
    // clock advances FIRST and the record is read AFTER, so a checkpoint that
    // fires inside tick `t` leaves record `t` unconsumed -- which is exactly
    // what the probe should report.
    let mut hash: u64 = 1469598103934665603;
    for t in 0..n {
        for i in 0..per_tick {
            // A value whose rounding depends on the tick, so nothing can be
            // constant-folded away.
            unsafe { lroundf((t as f32) * 0.001 + i as f32 * 1e-6) };
        }
        let rec = unsafe {
            std::ptr::read_volatile(arr.as_ptr().add(t * STRIDE) as *const [u8; 12])
        };
        for b in rec {
            hash ^= b as u64;
            hash = hash.wrapping_mul(1099511628211);
        }
    }

    // The verdict, in the shape `forkoracle::forksrv::parse_result` reads.
    let mut out = std::io::stdout().lock();
    let _ = write!(
        out,
        "{{\n  \"ValidatedResult\": {{\n    \"Time\": {},\n    \"NbCheckpoints\": {}\n  }},\n  \"IsValid\": true\n}}\n",
        hash % 1_000_000,
        n
    );
    let _ = out.flush();
    // Keep the array alive to the very end; without this LLVM is free to drop
    // the allocation once the loop is done and the shim's cached base would be
    // verifying freed memory.
    std::hint::black_box(&arr);
}

extern "C" {
    fn lroundf(x: f32) -> i64;
}

/// Read the steer sequence the shim will search for out of the key file the
/// driver wrote, so the host's array and the shim's key agree by construction.
fn key_steer(path: &str, n: usize) -> Vec<f32> {
    let d = std::fs::read(path).expect("key file");
    let m = u32::from_le_bytes(d[0..4].try_into().unwrap()) as usize;
    assert_eq!(m, n, "the key file and the host disagree about the tape length");
    (0..n)
        .map(|i| f32::from_le_bytes(d[12 + 4 * i..16 + 4 * i].try_into().unwrap()))
        .collect()
}
