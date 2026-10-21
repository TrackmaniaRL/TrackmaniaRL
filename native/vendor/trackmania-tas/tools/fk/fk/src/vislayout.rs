//! `vislayout` — the game's own `CSceneVehicleVisState` -> 116-byte record
//! sample writer, transcribed instruction for instruction from the dedicated
//! server's disassembly (2026-05-15 / 128182, md5
//! `0f0f4b25f31f80c60c81404366c95e68`).
//!
//! This is NOT a fit. Every offset, coefficient, clamp and bit position below
//! is read out of the binary, and the module exists so that reading can be
//! SCORED against recordings the game wrote (`fk carrier layout`). Nothing here
//! may be tuned against a recording; if a byte disagrees, the disagreement is
//! the result.
//!
//! # Where it lives in the binary
//!
//! ```text
//!   0x9cfed0  the versioned archiver for class 0x0A018000 (CSceneVehicleVis)
//!             version 33 == the `u03` field of the ghost's EntRecordDesc
//!     0xaca280 -> 0xac9e20 -> 0xacb110 -> 0xacb230 -> 0xacb520
//!             fill a flat 0x55 = 85-byte block (sample bytes 0..84), which
//!             the archiver then emits with one raw 85-byte write
//!     the remaining 31 bytes are emitted field by field, version-gated
//! ```
//!
//! The version ladder is the arithmetic control that says the reading is
//! complete: v30 -> 85+4+4+8+1+1 = **103** bytes, v31 -> 107, v32 -> 112,
//! v33 -> **116**. 116 is what TM2020 ghosts carry and 103 is the floor the
//! project's own decoder documents (`gbx::record`, "sampleSize >= 103").
//!
//! # The anchor
//!
//! Offsets here are into `CSceneVehicleVisState`, whose size is 864 (0x360) —
//! the stride of the array of copies `CARRIER.md` measured, and the `u01` field
//! of the ghost's EntRecordDesc. `Loc.translation` is at **0x50**, so
//! `CARRIER.md`'s `car` anchor is `state + 0x50` and every `car+N` in that
//! document is `state + 0x50 + N`.

/// `Loc.translation`'s offset inside `CSceneVehicleVisState`. The position
/// triple every other module anchors on is `state + POS_IN_STATE`, so a
/// pointer to the struct is `car - POS_IN_STATE`.
pub const POS_IN_STATE: i64 = 0x50;

/// `sizeof(CSceneVehicleVisState)`, from the class descriptor at 0x9d2ea0 —
/// and the stride of the array of copies `CARRIER.md` measured, and the `u01`
/// field of the ghost's EntRecordDesc. Three independent witnesses to 864.
pub const STATE_SIZE: i64 = 0x360;

/// The wheel records: four of them, `WHEEL_STRIDE` apart, starting at
/// `WHEEL0` into the state, with the rotation float at `+WHEEL_ROT`.
pub const WHEEL0: i64 = 0xa8;
pub const WHEEL_STRIDE: i64 = 44;
pub const WHEEL_ROT: i64 = 4;

/// The four wheel-rotation slots, RELATIVE TO THE POSITION TRIPLE — `[92, 136,
/// 180, 224]`. Four live floats is a car; four constants is a bare position
/// copy, and a regeneration anchored on one writes zeroed wheels into a file
/// that passes every acceptance test there is.
///
/// DERIVED, and that is the point. These four numbers were stated
/// independently in five places — as a const array in `fk ptr`, as a second
/// const array in `fk carrier`, as `92 + 44k` computed inline a few hundred
/// lines further down the same file, as `0xac + 44k` here, and as
/// `WHEEL0 + WHEEL_STRIDE*k + WHEEL_ROT` against a different anchor entirely in
/// `fk liveness`. Deriving them from the class descriptor's `0x88` and
/// subtracting `POS_IN_STATE` puts them 32 bytes low, which scores 3 of 4
/// against neighbouring live floats: the near-miss that looks like a result.
pub fn wheel_rot_rel() -> [i64; 4] {
    std::array::from_fn(|k| WHEEL0 + WHEEL_STRIDE * k as i64 + WHEEL_ROT - POS_IN_STATE)
}

/// One byte of the sample, and where it comes from.
pub struct ByteDoc {
    pub byte: usize,
    /// The `CSceneVehicleVisState` member, named from the engine's own
    /// reflection table at 0x9d2ea0 where it names one.
    pub field: &'static str,
    pub encoding: &'static str,
}

/// Reads one instant of a gathered `CSceneVehicleVisState`.
pub trait State {
    /// The f32 at `off` bytes into the state.
    fn f32(&self, off: usize) -> f32;
    /// The u32 at `off`.
    fn u32(&self, off: usize) -> u32;
    /// The byte at `off`.
    fn u8(&self, off: usize) -> u8;
    /// Are all `STATE_SIZE` bytes of the state actually readable?
    ///
    /// Every implementation over a gathered window answers 0 for a byte it does
    /// not have, because panicking mid-transcription would be worse. But a
    /// window that stops short does not produce a partial sample — it produces
    /// a CONFIDENT ZERO in every byte past the edge, and a zero looks measured.
    /// Measured 2026-08-23: a pointer window sized 124 bytes short transcribed
    /// the tail of the struct as zeros, every acceptance test passed, and the
    /// published clip had no wheel effects.
    ///
    /// Defaults to `true` so a test fixture that is the whole state need not
    /// say so. Real gathers override it, and `pack_checked` is what asks.
    fn covers_state(&self) -> bool {
        true
    }
}

/// `cvttss2si` after the game's clamp idiom: negative (or NaN) becomes 0, the
/// top is `hi`, and the conversion TRUNCATES — it is never a round.
fn clamp_trunc(v: f32, hi: f32) -> u32 {
    let v = if v.is_nan() || v < 0.0 { 0.0 } else { v };
    let v = if v > hi { hi } else { v };
    v as u32
}

/// The game's four-way quantiser for a normalised [0,1] float: it appears
/// nine times in a row for the fields at 0x244..0x264 and once more for the
/// gas pedal.
fn code2(v: f32) -> u32 {
    if v < 1e-5 {
        0
    } else if v < 0.5 {
        1
    } else if v < 0.99 {
        2
    } else {
        3
    }
}

/// The tri-state a signed component of `ReactorAirControl` becomes: 0
/// negative, 1 zero, 2 positive.
fn tri(v: f32) -> u32 {
    if v.abs() < 1e-5 {
        1
    } else if v >= 0.0 {
        2
    } else {
        0
    }
}

/// `0xcdcd` is the debug fill for uninitialised heap and the writer refuses to
/// emit it, bumping each wheel-rotation word to `0xcdce`. Transcribed because
/// it is a real, observable byte difference — not because it is likely.
fn no_cdcd(v: u32) -> u32 {
    if v == 0xcdcd {
        0xcdce
    } else {
        v
    }
}

/// The engine's 4-byte "direction and log speed" pack (`0x1ab32d0`): an i16
/// `1000*ln|v|` then a heading and a pitch byte. Used twice — for `WorldVel`
/// and for the unnamed vector at state+0x68.
fn dirpack(x: f32, y: f32, z: f32) -> [u8; 4] {
    let n = (x * x + y * y + z * z).sqrt();
    if n < 1e-5 {
        // the writer emits 0x8000 and then a heading/pitch of a zero vector
        return [0x00, 0x80, 0, 0];
    }
    let inv = 1.0f32 / n;
    let (ux, uy, uz) = (x * inv, y * inv, z * inv);
    let s = (1000.0f32 * n.ln()).max(-32768.0).min(32767.0) as i32 as i16;
    let pitch = if uz < -0.999999 {
        -std::f32::consts::FRAC_PI_2
    } else if uz > 0.999999 {
        std::f32::consts::FRAC_PI_2
    } else {
        uz.asin()
    };
    let heading = uy.atan2(ux);
    let mut o = [0u8; 4];
    o[0..2].copy_from_slice(&s.to_le_bytes());
    o[2] = (heading * 127.0 / std::f32::consts::PI) as i32 as i8 as u8;
    o[3] = (pitch / std::f32::consts::FRAC_PI_2 * 127.0) as i32 as i8 as u8;
    o
}

/// Which bytes this transcription predicts. The rest are the orientation words
/// (59..64), which need the matrix-to-quaternion step, and the countdown at
/// 108..111, which needs the archiver's caller-supplied timestamp.
///
/// STATED IN `gbx::sample`, not here. `ghost` judges the files this writes and
/// cannot depend on `fk`, so both crates once kept their own copy of this list
/// and the copies drifted for months without failing anything. One statement,
/// re-exported.
pub use gbx::sample::UNPREDICTED;

/// Build the 116 bytes the game would write for this state, REFUSING a state
/// the reader cannot see all of. This is what production should call; `pack`
/// is for a fixture that is the whole struct by construction.
pub fn pack_checked(s: &dyn State) -> Result<[u8; 116], String> {
    if !s.covers_state() {
        return Err(format!(
            "the gathered window does not hold all {} bytes of the vehicle state, so the \
             transcription would write a confident zero into every byte past its edge -- \
             which passes every acceptance test this project has",
            STATE_SIZE
        ));
    }
    Ok(pack(s))
}

/// Build the 116 bytes the game would write for this state.
pub fn pack(s: &dyn State) -> [u8; 116] {
    let mut o = [0u8; 116];
    let put16 = |o: &mut [u8; 116], at: usize, v: u32| {
        o[at] = v as u8;
        o[at + 1] = (v >> 8) as u8;
    };
    let put32 = |o: &mut [u8; 116], at: usize, v: u32| {
        o[at..at + 4].copy_from_slice(&v.to_le_bytes());
    };

    // --- bytes 0..33: 0xacb520 -----------------------------------------
    // FrontSpeed, over [-1000, 10000] on the u16
    let v = s.f32(0x74);
    let x = if v < -1000.0 {
        0.0
    } else {
        (v.min(10000.0) + 1000.0) / 11000.0 * 65535.0
    };
    put16(&mut o, 0, clamp_trunc(x, 65535.0));
    // the unnamed lateral speed at state+0x78, over [-1000, 1000]
    let v = s.f32(0x78);
    let x = if v < -1000.0 {
        0.0
    } else {
        (v.min(1000.0) + 1000.0) / 2000.0 * 65535.0
    };
    put16(&mut o, 2, clamp_trunc(x, 65535.0));
    // rpm at state+0x198, over [0, 30000]
    put16(&mut o, 4, clamp_trunc(s.f32(0x198) / 30000.0 * 65535.0, 65535.0));
    // the four wheel rotations, over [0, 2*pi*256]
    for k in 0..4 {
        let v = s.f32((WHEEL0 + WHEEL_ROT + WHEEL_STRIDE * k as i64) as usize) / 1608.4955 * 65535.0;
        put16(&mut o, 6 + 2 * k, no_cdcd(clamp_trunc(v, 65535.0)));
    }
    o[14] = clamp_trunc((s.f32(0x10) + 1.0) * 0.5 * 255.0, 255.0) as u8; // InputSteer
    let braking = s.u32(0x20) != 0; // InputIsBraking
    let pedal = clamp_trunc(s.f32(0x14) * 255.0, 255.0) as u8; // InputGasPedal
    o[15] = if braking { 0 } else { pedal };
    // bytes 16, 17 are written as a literal zero word
    o[18] = if braking { pedal } else { 0 };
    o[19] = clamp_trunc((s.f32(0x228) + 1.0) * 0.5 * 255.0, 255.0) as u8;
    o[20] = clamp_trunc((s.f32(0x22c) + 1.0) * 0.5 * 255.0, 255.0) as u8;
    o[21] = clamp_trunc(s.f32(0x1ac) * 255.0, 255.0) as u8; // TurboTime
    // Wheels[0].SteerAngle over [-pi, pi] -- 255/(2*pi), NOT the wheel constant
    o[22] = clamp_trunc(
        (s.f32(0xb4) + std::f32::consts::PI) / (2.0 * std::f32::consts::PI) * 255.0,
        255.0,
    ) as u8;
    // DamperLength over [-2, 2], and the ground material, per wheel
    for k in 0..4 {
        let w = 0xa8 + 44 * k;
        let damp = clamp_trunc((s.f32(w) + 2.0) * 0.25 * 255.0, 255.0) as u8;
        // the sample stores the raw material byte unless the wheel's flag word
        // says no contact, when it stores 13. The script accessor has a THIRD
        // case (80) that the sample does not.
        let mat = if s.u8(w + 0x28) & 2 != 0 { 13 } else { s.u8(w + 0x10) };
        o[23 + 2 * k] = damp;
        o[24 + 2 * k] = mat;
    }
    o[31] = ((s.u32(0x19c) & 7) | (((s.u8(0x8b) & 1) as u32) << 7)) as u8;
    let slip = |k: usize| (s.f32(0xbc + 44 * k) > 0.1) as u32;
    let wflag2 = |k: usize| (s.u32(0xd0 + 44 * k) >> 2) & 1;
    o[32] = ((slip(0) << 6) | (wflag2(0) << 7)) as u8;
    o[33] = (slip(1)
        | (wflag2(1) << 1)
        | (slip(2) << 2)
        | (wflag2(2) << 3)
        | (slip(3) << 4)
        | (wflag2(3) << 5)
        | (((s.f32(0x1a0) > 0.0) as u32) << 6)
        | (((s.u32(0x88) >> 5) & 1) << 7)) as u8;

    // --- bytes 34..41: 0xacb230 ----------------------------------------
    o[34] = clamp_trunc(s.f32(0x224) * 255.0, 255.0) as u8;
    // bytes 35..38 are written as a literal zero dword
    let mut w = 0u32;
    for i in 0..8 {
        w |= code2(s.f32(0x244 + 4 * i)) << (2 * i);
    }
    put16(&mut o, 39, w);
    o[41] = (((s.f32(0x84) * 7.0).round() as i32 as u8) << 5) | code2(s.f32(0x264)) as u8;

    // --- bytes 42, 43: 0xacb110 ----------------------------------------
    o[42] = ((s.u8(0x30c) & 1)
        | ((s.u8(0x30d) & 1) << 1)
        | ((s.u8(0x30e) & 1) << 2)
        | ((s.u8(0x30f) & 1) << 3)
        | (((s.u32(0x310) & 1) as u8) << 4)) as u8;
    o[43] = (code2(s.f32(0x14))
        | (((s.u32(0x1bc) == 1) as u32) << 2)
        | (((s.u32(0x24) != 0) as u32) << 3)
        | (((s.u8(0xa) & 0xf) as u32) << 4)) as u8;

    // --- bytes 44..73: 0xac9e20 ----------------------------------------
    o[44] = clamp_trunc(s.f32(0x7c) / 5.0 * 255.0, 255.0) as u8;
    o[45] = ((s.u32(0x88) >> 12) & 1) as u8;
    o[46] = clamp_trunc(s.f32(0x1c0) * 255.0, 255.0) as u8;
    for k in 0..3 {
        o[47 + 4 * k..51 + 4 * k].copy_from_slice(&s.f32(0x50 + 4 * k).to_le_bytes());
    }
    // 59..64: the orientation words, from the 3x3 rotation stored at state+0x2c
    o[65..69].copy_from_slice(&dirpack(s.f32(0x5c), s.f32(0x60), s.f32(0x64)));
    o[69..73].copy_from_slice(&dirpack(s.f32(0x68), s.f32(0x6c), s.f32(0x70)));
    o[73] = (s.u8(0x1bc) & 0xf) | (s.u8(0x8) << 4);

    // --- bytes 74..84: 0xaca280 ----------------------------------------
    o[74] = clamp_trunc(s.f32(0x158) / (2.0 * std::f32::consts::PI) * 255.0, 255.0) as u8;
    o[75] = s.u8(0x8);
    let fl = s.u32(0x88);
    o[76] = (((fl >> 10) & 1)
        | (((fl >> 9) & 1) << 1)
        | (((fl >> 8) & 1) << 2)
        | (((fl >> 7) & 1) << 3)
        | (((s.u32(0x178) != 0) as u32) << 4)
        | (((fl >> 4) & 1) << 5)
        | (((fl >> 17) & 1) << 6)
        | (((fl >> 6) & 1) << 7)) as u8;
    put32(&mut o, 77, s.u32(0x338));
    for k in 0..4 {
        o[81 + k] = clamp_trunc(s.f32(0xc4 + 44 * k) * 255.0, 255.0) as u8; // Icing01
    }

    // --- bytes 85..115: emitted field by field by the archiver ----------
    put32(&mut o, 85, s.u32(0x15c));
    let bits = ((fl >> 20) & 1)                     // IsGroundContact
        | (((fl >> 19) & 1) << 1)                   // IsReactorGroundMode
        | (((fl >> 18) & 1) << 2)                   // ReactorInputsX
        | ((s.u32(0x178) & 3) << 3)                 // ReactorBoostType
        | ((s.u32(0x174) & 3) << 5)                 // ReactorBoostLvl
        | (((s.u8(0x1b8) & 3) as u32) << 7)
        | ((s.u32(0x170) & 7) << 9)
        | (tri(s.f32(0x180)) << 12)                 // ReactorAirControl.x
        | (tri(s.f32(0x184)) << 14)                 // .y
        | (tri(s.f32(0x188)) << 16)                 // .z
        | ((s.u32(0x1a4) & 0xf) << 18); // CurGear
    put32(&mut o, 89, bits);
    for k in 0..4 {
        o[93 + 2 * k] = clamp_trunc(s.f32(0xc0 + 44 * k) * 255.0, 255.0) as u8;
        o[94 + 2 * k] = clamp_trunc(s.f32(0xc8 + 44 * k) * 255.0, 255.0) as u8; // TireWear01
    }
    o[101] = clamp_trunc(s.f32(0x328) * 255.0, 255.0) as u8; // WetnessValue01
    o[102] = clamp_trunc(s.f32(0x230) * 255.0, 255.0) as u8; // SimulationTimeCoef
    put32(&mut o, 103, s.u32(0x80));
    o[107] = s.u8(0x344);
    // 108..111: -2 - min(now - state[0x340], 3000), needs the archiver's clock
    put32(&mut o, 112, s.u32(0x348));
    o
}

/// One quantity inside a packed byte or dword of the sample.
///
/// A byte-level score hides which of six quantities carries the agreement, and
/// for the reactor that is the whole question: byte 89 is 100 % on a run with no
/// reactor in it because `IsGroundContact` is 100 %. These are scored
/// separately so a reactor bit that no key exercises is reported as untested
/// rather than hidden inside a byte that passes.
pub struct BitFieldDoc {
    pub name: &'static str,
    /// First sample byte of the little-endian word this field sits in.
    pub byte: usize,
    /// How many bytes that word is.
    pub width: usize,
    pub shift: u32,
    pub bits: u32,
}

pub const BITFIELDS: &[BitFieldDoc] = &[
    BitFieldDoc { name: "state+0x19c enum", byte: 31, width: 1, shift: 0, bits: 3 },
    BitFieldDoc { name: "IsTurbo (flag 24)", byte: 31, width: 1, shift: 7, bits: 1 },
    BitFieldDoc { name: "Wheels[0].SlipCoef>0.1", byte: 32, width: 1, shift: 6, bits: 1 },
    BitFieldDoc { name: "Wheels[0] flag bit 2", byte: 32, width: 1, shift: 7, bits: 1 },
    BitFieldDoc { name: "IsWheelsBurning (flag 5)", byte: 33, width: 1, shift: 7, bits: 1 },
    BitFieldDoc { name: "InputGasPedal 2-bit code", byte: 43, width: 1, shift: 0, bits: 2 },
    BitFieldDoc { name: "state+0x1bc == 1", byte: 43, width: 1, shift: 2, bits: 1 },
    BitFieldDoc { name: "state+0x24 != 0", byte: 43, width: 1, shift: 3, bits: 1 },
    BitFieldDoc { name: "DiscontinuityCount", byte: 43, width: 1, shift: 4, bits: 4 },
    BitFieldDoc { name: "IsTopContact (flag 4)", byte: 76, width: 1, shift: 5, bits: 1 },
    BitFieldDoc { name: "ReactorBoostType != 0", byte: 76, width: 1, shift: 4, bits: 1 },
    BitFieldDoc { name: "IsGroundContact (flag 20)", byte: 89, width: 4, shift: 0, bits: 1 },
    BitFieldDoc { name: "IsReactorGroundMode (flag 19)", byte: 89, width: 4, shift: 1, bits: 1 },
    BitFieldDoc { name: "ReactorInputsX (flag 18)", byte: 89, width: 4, shift: 2, bits: 1 },
    BitFieldDoc { name: "ReactorBoostType", byte: 89, width: 4, shift: 3, bits: 2 },
    BitFieldDoc { name: "ReactorBoostLvl", byte: 89, width: 4, shift: 5, bits: 2 },
    BitFieldDoc { name: "state+0x1b8", byte: 89, width: 4, shift: 7, bits: 2 },
    BitFieldDoc { name: "state+0x170", byte: 89, width: 4, shift: 9, bits: 3 },
    BitFieldDoc { name: "ReactorAirControl.x", byte: 89, width: 4, shift: 12, bits: 2 },
    BitFieldDoc { name: "ReactorAirControl.y", byte: 89, width: 4, shift: 14, bits: 2 },
    BitFieldDoc { name: "ReactorAirControl.z", byte: 89, width: 4, shift: 16, bits: 2 },
    BitFieldDoc { name: "CurGear", byte: 89, width: 4, shift: 18, bits: 4 },
];

impl BitFieldDoc {
    pub fn read(&self, sample: &[u8]) -> u32 {
        let mut w = 0u32;
        for k in 0..self.width {
            w |= (sample[self.byte + k] as u32) << (8 * k);
        }
        (w >> self.shift) & ((1u32 << self.bits) - 1)
    }
}

/// What each byte is, for the report. Kept beside [`pack`] so the two cannot
/// drift: a test asserts every byte appears exactly once.
pub const DOC: &[ByteDoc] = &[
    ByteDoc { byte: 0, field: "FrontSpeed", encoding: "u16 lo: (min(v,10000)+1000)/11000*65535" },
    ByteDoc { byte: 1, field: "FrontSpeed", encoding: "u16 hi" },
    ByteDoc { byte: 2, field: "state+0x78 (lateral speed)", encoding: "u16 lo: (min(v,1000)+1000)/2000*65535" },
    ByteDoc { byte: 3, field: "state+0x78", encoding: "u16 hi" },
    ByteDoc { byte: 4, field: "state+0x198 (rpm)", encoding: "u16 lo: v/30000*65535" },
    ByteDoc { byte: 5, field: "state+0x198 (rpm)", encoding: "u16 hi" },
    ByteDoc { byte: 6, field: "Wheels[0].Rot", encoding: "u16 lo: v/(2pi*256)*65535" },
    ByteDoc { byte: 7, field: "Wheels[0].Rot", encoding: "u16 hi" },
    ByteDoc { byte: 8, field: "Wheels[1].Rot", encoding: "u16 lo" },
    ByteDoc { byte: 9, field: "Wheels[1].Rot", encoding: "u16 hi" },
    ByteDoc { byte: 10, field: "Wheels[2].Rot", encoding: "u16 lo" },
    ByteDoc { byte: 11, field: "Wheels[2].Rot", encoding: "u16 hi" },
    ByteDoc { byte: 12, field: "Wheels[3].Rot", encoding: "u16 lo" },
    ByteDoc { byte: 13, field: "Wheels[3].Rot", encoding: "u16 hi" },
    ByteDoc { byte: 14, field: "InputSteer", encoding: "(v+1)/2*255" },
    ByteDoc { byte: 15, field: "InputGasPedal", encoding: "v*255, zero while InputIsBraking" },
    ByteDoc { byte: 16, field: "-", encoding: "literal 0" },
    ByteDoc { byte: 17, field: "-", encoding: "literal 0" },
    ByteDoc { byte: 18, field: "InputGasPedal", encoding: "v*255, only while InputIsBraking" },
    ByteDoc { byte: 19, field: "state+0x228", encoding: "(v+1)/2*255" },
    ByteDoc { byte: 20, field: "state+0x22c", encoding: "(v+1)/2*255" },
    ByteDoc { byte: 21, field: "TurboTime", encoding: "v*255" },
    ByteDoc { byte: 22, field: "Wheels[0].SteerAngle", encoding: "(v+pi)/(2pi)*255" },
    ByteDoc { byte: 23, field: "Wheels[0].DamperLength", encoding: "(v+2)/4*255" },
    ByteDoc { byte: 24, field: "Wheels[0] ground material", encoding: "raw byte, or 13 when flags&2" },
    ByteDoc { byte: 25, field: "Wheels[1].DamperLength", encoding: "(v+2)/4*255" },
    ByteDoc { byte: 26, field: "Wheels[1] ground material", encoding: "raw byte, or 13" },
    ByteDoc { byte: 27, field: "Wheels[2].DamperLength", encoding: "(v+2)/4*255" },
    ByteDoc { byte: 28, field: "Wheels[2] ground material", encoding: "raw byte, or 13" },
    ByteDoc { byte: 29, field: "Wheels[3].DamperLength", encoding: "(v+2)/4*255" },
    ByteDoc { byte: 30, field: "Wheels[3] ground material", encoding: "raw byte, or 13" },
    ByteDoc { byte: 31, field: "state+0x19c / IsTurbo", encoding: "bits0-2 = enum&7; bit7 = flags bit 24 (IsTurbo)" },
    ByteDoc { byte: 32, field: "Wheels[0].SlipCoef, wheel flags", encoding: "bit6 = SlipCoef>0.1; bit7 = flags bit 2" },
    ByteDoc { byte: 33, field: "wheels 1-3 slip/flags, IsWheelsBurning", encoding: "8 bits, see DOC in vislayout.rs" },
    ByteDoc { byte: 34, field: "state+0x224", encoding: "v*255" },
    ByteDoc { byte: 35, field: "-", encoding: "literal 0" },
    ByteDoc { byte: 36, field: "-", encoding: "literal 0" },
    ByteDoc { byte: 37, field: "-", encoding: "literal 0" },
    ByteDoc { byte: 38, field: "-", encoding: "literal 0" },
    ByteDoc { byte: 39, field: "state+0x244..0x254", encoding: "four 2-bit codes (0 / <0.5 / <0.99 / else)" },
    ByteDoc { byte: 40, field: "state+0x258..0x260", encoding: "four 2-bit codes" },
    ByteDoc { byte: 41, field: "state+0x264, state+0x84", encoding: "bits0-1 = 2-bit code; bits5-7 = round(v*7)" },
    ByteDoc { byte: 42, field: "state+0x30c..0x310", encoding: "five bools" },
    ByteDoc { byte: 43, field: "InputGasPedal, state+0x1bc, +0x24, DiscontinuityCount", encoding: "2+1+1+4 bits" },
    ByteDoc { byte: 44, field: "state+0x7c", encoding: "v/5*255" },
    ByteDoc { byte: 45, field: "flags bit 12", encoding: "bool" },
    ByteDoc { byte: 46, field: "state+0x1c0", encoding: "v*255" },
    ByteDoc { byte: 47, field: "Loc.translation.x", encoding: "f32" },
    ByteDoc { byte: 48, field: "Loc.translation.x", encoding: "f32" },
    ByteDoc { byte: 49, field: "Loc.translation.x", encoding: "f32" },
    ByteDoc { byte: 50, field: "Loc.translation.x", encoding: "f32" },
    ByteDoc { byte: 51, field: "Loc.translation.y", encoding: "f32" },
    ByteDoc { byte: 52, field: "Loc.translation.y", encoding: "f32" },
    ByteDoc { byte: 53, field: "Loc.translation.y", encoding: "f32" },
    ByteDoc { byte: 54, field: "Loc.translation.y", encoding: "f32" },
    ByteDoc { byte: 55, field: "Loc.translation.z", encoding: "f32" },
    ByteDoc { byte: 56, field: "Loc.translation.z", encoding: "f32" },
    ByteDoc { byte: 57, field: "Loc.translation.z", encoding: "f32" },
    ByteDoc { byte: 58, field: "Loc.translation.z", encoding: "f32" },
    ByteDoc { byte: 59, field: "Loc rotation (state+0x2c)", encoding: "u16 angle = acos(qw)/pi*65535, TRUNCATED" },
    ByteDoc { byte: 60, field: "Loc rotation", encoding: "u16 angle hi" },
    ByteDoc { byte: 61, field: "Loc rotation", encoding: "i16 axis heading" },
    ByteDoc { byte: 62, field: "Loc rotation", encoding: "i16 axis heading hi" },
    ByteDoc { byte: 63, field: "Loc rotation", encoding: "i16 axis pitch" },
    ByteDoc { byte: 64, field: "Loc rotation", encoding: "i16 axis pitch hi" },
    ByteDoc { byte: 65, field: "WorldVel", encoding: "i16 1000*ln|v|, TRUNCATED" },
    ByteDoc { byte: 66, field: "WorldVel", encoding: "i16 hi" },
    ByteDoc { byte: 67, field: "WorldVel", encoding: "i8 heading" },
    ByteDoc { byte: 68, field: "WorldVel", encoding: "i8 pitch" },
    ByteDoc { byte: 69, field: "state+0x68 (a second vector)", encoding: "i16 1000*ln|v|" },
    ByteDoc { byte: 70, field: "state+0x68", encoding: "i16 hi" },
    ByteDoc { byte: 71, field: "state+0x68", encoding: "i8 heading" },
    ByteDoc { byte: 72, field: "state+0x68", encoding: "i8 pitch" },
    ByteDoc { byte: 73, field: "state+0x1bc, state+0x8", encoding: "two nibbles" },
    ByteDoc { byte: 74, field: "state+0x158", encoding: "v/(2pi)*255" },
    ByteDoc { byte: 75, field: "state+0x8", encoding: "raw byte" },
    ByteDoc { byte: 76, field: "flags 4,6,7,8,9,10,17 + ReactorBoostType!=0", encoding: "8 bits" },
    ByteDoc { byte: 77, field: "state+0x338", encoding: "u32" },
    ByteDoc { byte: 78, field: "state+0x338", encoding: "u32" },
    ByteDoc { byte: 79, field: "state+0x338", encoding: "u32" },
    ByteDoc { byte: 80, field: "state+0x338", encoding: "u32" },
    ByteDoc { byte: 81, field: "Wheels[0].Icing01", encoding: "v*255" },
    ByteDoc { byte: 82, field: "Wheels[1].Icing01", encoding: "v*255" },
    ByteDoc { byte: 83, field: "Wheels[2].Icing01", encoding: "v*255" },
    ByteDoc { byte: 84, field: "Wheels[3].Icing01", encoding: "v*255" },
    ByteDoc { byte: 85, field: "state+0x15c", encoding: "f32, verbatim" },
    ByteDoc { byte: 86, field: "state+0x15c", encoding: "f32" },
    ByteDoc { byte: 87, field: "state+0x15c", encoding: "f32" },
    ByteDoc { byte: 88, field: "state+0x15c", encoding: "f32" },
    ByteDoc { byte: 89, field: "IsGroundContact, IsReactorGroundMode, ReactorInputsX, ReactorBoostType, ReactorBoostLvl", encoding: "bit0/1/2, bits3-4, bits5-6" },
    ByteDoc { byte: 90, field: "state+0x1b8, state+0x170, ReactorAirControl.x/.y", encoding: "bits0, 1-3, 4-5, 6-7" },
    ByteDoc { byte: 91, field: "ReactorAirControl.z, CurGear", encoding: "bits0-1 tri-state, bits2-5 gear" },
    ByteDoc { byte: 92, field: "-", encoding: "always 0 (top of the u32 at 89)" },
    ByteDoc { byte: 93, field: "Wheels[0]+0x18 (the dirt slot)", encoding: "v*255" },
    ByteDoc { byte: 94, field: "Wheels[0].TireWear01", encoding: "v*255" },
    ByteDoc { byte: 95, field: "Wheels[1]+0x18", encoding: "v*255" },
    ByteDoc { byte: 96, field: "Wheels[1].TireWear01", encoding: "v*255" },
    ByteDoc { byte: 97, field: "Wheels[2]+0x18", encoding: "v*255" },
    ByteDoc { byte: 98, field: "Wheels[2].TireWear01", encoding: "v*255" },
    ByteDoc { byte: 99, field: "Wheels[3]+0x18", encoding: "v*255" },
    ByteDoc { byte: 100, field: "Wheels[3].TireWear01", encoding: "v*255" },
    ByteDoc { byte: 101, field: "WetnessValue01", encoding: "v*255" },
    ByteDoc { byte: 102, field: "SimulationTimeCoef", encoding: "v*255" },
    ByteDoc { byte: 103, field: "state+0x80", encoding: "f32, verbatim" },
    ByteDoc { byte: 104, field: "state+0x80", encoding: "f32" },
    ByteDoc { byte: 105, field: "state+0x80", encoding: "f32" },
    ByteDoc { byte: 106, field: "state+0x80", encoding: "f32" },
    ByteDoc { byte: 107, field: "state+0x344", encoding: "raw byte" },
    ByteDoc { byte: 108, field: "state+0x340", encoding: "i32 countdown, needs the archiver clock" },
    ByteDoc { byte: 109, field: "state+0x340", encoding: "i32" },
    ByteDoc { byte: 110, field: "state+0x340", encoding: "i32" },
    ByteDoc { byte: 111, field: "state+0x340", encoding: "i32" },
    ByteDoc { byte: 112, field: "state+0x348", encoding: "u32, verbatim" },
    ByteDoc { byte: 113, field: "state+0x348", encoding: "u32" },
    ByteDoc { byte: 114, field: "state+0x348", encoding: "u32" },
    ByteDoc { byte: 115, field: "state+0x348", encoding: "u32" },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_doc_covers_every_byte_exactly_once() {
        let mut seen = [false; 116];
        for d in DOC {
            assert!(!seen[d.byte], "byte {} documented twice", d.byte);
            seen[d.byte] = true;
        }
        assert!(seen.iter().all(|&x| x), "a sample byte is undocumented");
    }

    /// The version ladder from the archiver at 0x9cfed0 is the arithmetic that
    /// says the reading is complete: the emitted sizes must land on 116, and on
    /// the 103 the project's own decoder documents as the floor.
    #[test]
    fn the_field_sizes_add_up_to_the_sample_size() {
        let blob = 0x55; // one raw write of the packed block
        let v30 = blob + 4 + 4 + 4 * 2 + 1 + 1;
        assert_eq!(v30, 103);
        assert_eq!(v30 + 4, 107); // v31 adds state+0x80
        assert_eq!(v30 + 4 + 1 + 4, 112); // v32 adds state+0x344 and the countdown
        assert_eq!(v30 + 4 + 1 + 4 + 4, 116); // v33 adds state+0x348
    }
}

/// Bytes whose SOURCE SLOT is identically zero in the dedicated server while a
/// game-written recording's own values move.
///
/// This is a fourth verdict, distinct from "wrong offset" and from "could not
/// be tested": the transcription is right, the slot is real, and *this binary
/// never populates it*. Byte 34 is `floor(state[0x224] · 255)` and takes 137
/// distinct values in the client's own recording of a nine-boost-gate map — and
/// reads 0 in every instant the server produces. Same for 19, 20 and the four
/// dirt slots.
///
/// They are listed rather than written because writing them would replace a
/// real value with a confident zero, and nothing downstream would catch it: none
/// of these bytes affects the simulation, so the plain oracle and the whole
/// `ghost verify` gate pass on a file full of them. That is exactly the failure
/// that put zeroed wheels into published clips.
///
/// To source them, run the CLIENT rather than the server. That is a task, not a
/// conclusion.
///
/// STATED IN `gbx::sample` for the reason `UNPREDICTED` is; re-exported.
pub use gbx::sample::DEAD_IN_SERVER;

#[cfg(test)]
mod geometry_tests {
    use super::*;

    /// THE DERIVED OFFSETS MUST BE THE ONES THAT WERE MEASURED.
    ///
    /// `[92, 136, 180, 224]` is what `CARRIER.md` confirmed on eight keys and
    /// what `fk ptr check` grades 4-of-4 against on three maps. Deriving them
    /// is only an improvement if the derivation reproduces them exactly; a
    /// derivation that quietly disagrees is how the 32-byte-low version scored
    /// 3 of 4 and looked like a result.
    #[test]
    fn the_wheel_slots_derive_to_the_offsets_that_were_measured() {
        assert_eq!(wheel_rot_rel(), [92, 136, 180, 224]);
    }

    /// The wheel block is inside the struct. If it is not, the window sized
    /// from `STATE_SIZE` does not reach the bytes the liveness rule reads, and
    /// the rule answers "dead" for a reason that has nothing to do with the car.
    #[test]
    fn the_wheel_block_lies_inside_the_state() {
        for r in wheel_rot_rel() {
            let off = r + POS_IN_STATE;
            assert!(off >= 0 && off + 4 <= STATE_SIZE, "wheel slot at state+{off} is outside");
        }
    }
}
