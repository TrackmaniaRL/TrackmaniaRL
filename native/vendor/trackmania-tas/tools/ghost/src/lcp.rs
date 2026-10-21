//! `ghost lcp FILE` -- the editor's launched-checkpoints cache, decoded.
//!
//! WHY (2026-09-09): vjeux test-drove the tiny maps in the editor and did not
//! finish 20, 21 or 22, so no MediaTrackerCache ghost exists for them. But the
//! client keeps a second per-map file, written on every checkpoint crossing in
//! test mode so the player can "start from checkpoint":
//! `C:\ProgramData\Trackmania\LaunchedCheckpointsCache\<map name>.LaunchedCP.gbx`.
//!
//! For every checkpoint reached it stores the FULL car state at the crossing and
//! the last ~1.5 s of the approach as vehicle samples -- the route skeleton of an
//! unfinished run: which checkpoints, in which order, where the car was and how
//! fast, and what the hands were doing on the way in.
//!
//! LAYOUT, measured on the tiny 13/20/21/22 files (class 0x03262000
//! `CGameSaveLaunchedCheckpoints`, one chunk, uncompressed body):
//!
//! ```text
//!     u32 chunk id 0x03262000
//!     u32 version = 7
//!     u32 33                         (constant on every file seen)
//!     u32 n                          number of entries
//!     n x ENTRY (663 B)              one per checkpoint reached, in the order reached
//!     u32 total                      number of approach samples that follow
//!     total x { 116 B CSceneVehicleVis sample ; u32 t_ms }
//!                                    t counts up inside each entry's window;
//!                                    the last sample of a window is the frame
//!                                    before the crossing
//!     n x u32                        samples per entry (sums to total)
//!     n x { u32 0x0FF00000 ; MwId ident }   the vehicle model (CarSport / 10003 / Nadeo)
//!     u32 0xFACADE01
//!
//!     ENTRY:
//!       +0x00 u32  checkpoint landmark index      +0x04 u32 0
//!       +0x08 u32  time at the crossing, ms        (the test-mode clock: race time on a
//!                                                  first run, session time after respawns)
//!       +0x0c f32x3 checkpoint position            +0x18 u32 0x0A020000 (state format tag)
//!       +0x1c u32  landmark to LAUNCH from (the group's, for a linked checkpoint; else the same)
//!       +0x20 f32x4 orientation quaternion x y z w
//!       +0x30 f32x3 car position   +0x3c f32x3 velocity m/s   +0x48 f32x3 angular velocity rad/s
//!       +0x152, +0x195, +0x1d8, +0x21b: four 67-byte WHEEL blocks (FL FR RR RL by the
//!                steer field: the first two carry the steering angle, the rear two 0):
//!                +0 f32, +4 u32 1, +8 u32, +22 f32 rotation, +26 f32 steer angle rad
//!       +0x26b f32 signed forward speed m/s (negative = crossed in reverse)
//!       the rest: timers and -1 ids, kept as raw hex in the JSON for whoever needs them
//! ```
//!
//! The 116-byte approach samples are exactly the ghost telemetry sample
//! (`gbx::record::decode_vehicle_sample`): position, quaternion, velocity, gear,
//! rpm, wheel state -- and the steer / gas / brake bytes, so the last 1.5 s of
//! INPUTS before each crossing are in here too, at the render frame rate (~53 ms).

use crate::cli::{die, flag, has};
use gbx::container::Container;
use gbx::record::{decode_vehicle_sample, Sample};

pub const CLASS_LAUNCHED_CP: u32 = 0x0326_2000;
const ENTRY_LEN: usize = 663;
const SAMPLE_LEN: usize = 116;
const WHEEL_OFFSETS: [usize; 4] = [0x152, 0x195, 0x1d8, 0x21b];
const FACADE: u32 = 0xFACA_DE01;

#[derive(Debug, Clone)]
pub struct Wheel {
    pub rotation: f32,
    pub steer: f32,
}

#[derive(Debug, Clone)]
pub struct Entry {
    /// the checkpoint crossed (its index in `tmmaps waypoints MAP`)
    pub landmark: u32,
    /// the checkpoint the game launches from for it -- the same, except for a
    /// LINKED checkpoint group where every member reports the group's landmark
    /// (11: landmark 4 launches as 5)
    pub launch_landmark: u32,
    pub time_ms: u32,
    pub cp_pos: [f32; 3],
    pub quat: [f32; 4],
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    pub angvel: [f32; 3],
    /// signed forward speed, m/s (negative: crossed in reverse -- 21 landmark 3)
    pub speed: f32,
    pub wheels: [Wheel; 4],
    pub raw: Vec<u8>,
    /// (t in the window, decoded sample, raw 116 bytes)
    pub samples: Vec<(u32, Sample, Vec<u8>)>,
}

#[derive(Debug, Clone)]
pub struct LaunchedCheckpoints {
    pub version: u32,
    pub word2: u32,
    pub entries: Vec<Entry>,
    pub tail: Vec<u8>,
}

fn u32_at(b: &[u8], o: usize) -> Result<u32, String> {
    b.get(o..o + 4).map(|s| u32::from_le_bytes(s.try_into().unwrap())).ok_or_else(|| format!("truncated at {o}"))
}
fn f32_at(b: &[u8], o: usize) -> Result<f32, String> {
    Ok(f32::from_bits(u32_at(b, o)?))
}
fn f32x3(b: &[u8], o: usize) -> Result<[f32; 3], String> {
    Ok([f32_at(b, o)?, f32_at(b, o + 4)?, f32_at(b, o + 8)?])
}

pub fn parse(body: &[u8]) -> Result<LaunchedCheckpoints, String> {
    let id = u32_at(body, 0)?;
    if id != CLASS_LAUNCHED_CP {
        return Err(format!("body does not open with chunk 0x03262000 (found 0x{id:08X})"));
    }
    let version = u32_at(body, 4)?;
    if version != 7 {
        return Err(format!("CGameSaveLaunchedCheckpoints version {version}: this decoder was measured on version 7 only"));
    }
    let word2 = u32_at(body, 8)?;
    let n = u32_at(body, 12)? as usize;
    if n > 512 {
        return Err(format!("{n} entries: not a plausible count"));
    }
    let mut o = 16;
    let mut entries = Vec::with_capacity(n);
    for k in 0..n {
        let e = body.get(o..o + ENTRY_LEN).ok_or_else(|| format!("entry {k}: truncated"))?;
        let landmark = u32_at(e, 0)?;
        let tag = u32_at(e, 0x18)?;
        if tag != 0x0A02_0000 {
            return Err(format!("entry {k}: state tag 0x{tag:08X} is not the 0x0A020000 every measured entry carries -- the layout has moved"));
        }
        let mut wheels = Vec::with_capacity(4);
        for w in WHEEL_OFFSETS {
            wheels.push(Wheel { rotation: f32_at(e, w + 22)?, steer: f32_at(e, w + 26)? });
        }
        entries.push(Entry {
            landmark,
            launch_landmark: u32_at(e, 0x1c)?,
            time_ms: u32_at(e, 8)?,
            cp_pos: f32x3(e, 0x0c)?,
            quat: [f32_at(e, 0x20)?, f32_at(e, 0x24)?, f32_at(e, 0x28)?, f32_at(e, 0x2c)?],
            pos: f32x3(e, 0x30)?,
            vel: f32x3(e, 0x3c)?,
            angvel: f32x3(e, 0x48)?,
            speed: f32_at(e, 0x26b)?,
            wheels: wheels.try_into().unwrap(),
            raw: e.to_vec(),
            samples: Vec::new(),
        });
        o += ENTRY_LEN;
    }
    let total = u32_at(body, o)? as usize;
    o += 4;
    let mut flat = Vec::with_capacity(total);
    for i in 0..total {
        let s = body.get(o..o + SAMPLE_LEN).ok_or_else(|| format!("sample {i}: truncated"))?;
        let t = u32_at(body, o + SAMPLE_LEN)?;
        flat.push((t, decode_vehicle_sample(s), s.to_vec()));
        o += SAMPLE_LEN + 4;
    }
    let mut counts = Vec::with_capacity(n);
    for _ in 0..n {
        counts.push(u32_at(body, o)? as usize);
        o += 4;
    }
    if counts.iter().sum::<usize>() != total {
        return Err(format!("per-entry sample counts {counts:?} do not sum to the {total} samples stored"));
    }
    let mut it = flat.into_iter();
    for (e, c) in entries.iter_mut().zip(counts) {
        e.samples = it.by_ref().take(c).collect();
    }
    let tail = body.get(o..).unwrap_or(&[]).to_vec();
    if tail.len() < 4 || u32::from_le_bytes(tail[tail.len() - 4..].try_into().unwrap()) != FACADE {
        return Err("the body does not end with 0xFACADE01".into());
    }
    Ok(LaunchedCheckpoints { version, word2, entries, tail })
}

pub fn load(path: &str) -> Result<LaunchedCheckpoints, String> {
    let c = Container::load(path)?;
    if c.gbx.class_id != CLASS_LAUNCHED_CP {
        return Err(format!("{path}: class 0x{:08X} is not CGameSaveLaunchedCheckpoints (0x03262000)", c.gbx.class_id));
    }
    parse(c.body()).map_err(|e| format!("{path}: {e}"))
}

fn secs(ms: u32) -> String {
    gbx::container::secs(ms as i64)
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn f3(v: [f32; 3]) -> String {
    format!("[{:.3},{:.3},{:.3}]", v[0], v[1], v[2])
}

pub fn to_json(l: &LaunchedCheckpoints, path: &str) -> String {
    let mut s = String::new();
    s.push_str(&format!("{{\"source\":\"{}\",\"version\":{},\"word2\":{},\"entries\":[", path.replace('\\', "/"), l.version, l.word2));
    for (k, e) in l.entries.iter().enumerate() {
        if k > 0 {
            s.push(',');
        }
        s.push_str(&format!(
            "{{\"index\":{k},\"landmark\":{},\"launch_landmark\":{},\"time_ms\":{},\"cp_pos\":{},\"quat\":[{:.6},{:.6},{:.6},{:.6}],\"pos\":{},\"vel\":{},\"angvel\":{},\"speed_fwd\":{:.3},\"wheels\":[{}],\"raw_hex\":\"{}\",\"samples\":[",
            e.landmark,
            e.launch_landmark,
            e.time_ms,
            f3(e.cp_pos),
            e.quat[0], e.quat[1], e.quat[2], e.quat[3],
            f3(e.pos),
            f3(e.vel),
            f3(e.angvel),
            e.speed,
            e.wheels.iter().map(|w| format!("{{\"rotation\":{:.4},\"steer\":{:.4}}}", w.rotation, w.steer)).collect::<Vec<_>>().join(","),
            hex(&e.raw)
        ));
        for (i, (t, sm, raw)) in e.samples.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!(
                "{{\"t_ms\":{t},\"pos\":[{:.3},{:.3},{:.3}],\"quat\":[{:.5},{:.5},{:.5},{:.5}],\"vel\":[{:.3},{:.3},{:.3}],\"speed\":{:.3},\"steer\":{:.3},\"gas\":{:.0},\"brake\":{:.0},\"gear\":{},\"rpm_raw\":{},\"ground\":{},\"raw_hex\":\"{}\"}}",
                sm.x, sm.y, sm.z, sm.qx, sm.qy, sm.qz, sm.qw, sm.vx, sm.vy, sm.vz, sm.speed_ms, sm.steer, sm.gas.min(1.0), sm.brake, sm.gear_raw, sm.rpm_raw, sm.is_ground_contact as u8, hex(raw)
            ));
        }
        s.push_str("]}");
    }
    s.push_str(&format!("],\"tail_hex\":\"{}\"}}\n", hex(&l.tail)));
    s
}

pub fn to_csv(l: &LaunchedCheckpoints) -> String {
    let mut s = String::from("entry,landmark,kind,time_ms,t_window_ms,x,y,z,qx,qy,qz,qw,vx,vy,vz,speed_fwd_ms,steer,gas,brake\n");
    for (k, e) in l.entries.iter().enumerate() {
        s.push_str(&format!(
            "{k},{},crossing,{},,{:.3},{:.3},{:.3},{:.6},{:.6},{:.6},{:.6},{:.3},{:.3},{:.3},{:.3},{:.4},,\n",
            e.landmark, e.time_ms, e.pos[0], e.pos[1], e.pos[2], e.quat[0], e.quat[1], e.quat[2], e.quat[3], e.vel[0], e.vel[1], e.vel[2], e.speed, e.wheels[0].steer
        ));
        for (t, sm, _) in &e.samples {
            s.push_str(&format!(
                "{k},{},approach,{},{t},{:.3},{:.3},{:.3},{:.5},{:.5},{:.5},{:.5},{:.3},{:.3},{:.3},{:.3},{:.3},{:.0},{:.0}\n",
                e.landmark, e.time_ms, sm.x, sm.y, sm.z, sm.qx, sm.qy, sm.qz, sm.qw, sm.vx, sm.vy, sm.vz, sm.speed_ms, sm.steer, sm.gas.min(1.0), sm.brake
            ));
        }
    }
    s
}

pub fn cmd(a: &[String]) {
    let Some(path) = a.first() else {
        die("ghost lcp FILE.LaunchedCP.gbx [--json OUT.json] [--csv OUT.csv] [--samples]");
    };
    let l = load(path).unwrap_or_else(|e| die(e));
    println!(
        "{path}: CGameSaveLaunchedCheckpoints v{} -- {} checkpoint(s) reached, {} approach samples",
        l.version,
        l.entries.len(),
        l.entries.iter().map(|e| e.samples.len()).sum::<usize>()
    );
    for (k, e) in l.entries.iter().enumerate() {
        let (yaw, _pitch, _roll) = gbx::record::quat_to_ypr([e.quat[0] as f64, e.quat[1] as f64, e.quat[2] as f64, e.quat[3] as f64]);
        println!(
            "  #{k:<2} landmark {:<3}{} at {}  cp {}  car {}  v {} = {:.1} m/s ({:.0} km/h)  yaw {:.2}  steer {:+.3}  {} samples over {} ms",
            e.landmark,
            if e.launch_landmark != e.landmark { format!(" (launch {})", e.launch_landmark) } else { String::new() },
            secs(e.time_ms),
            f3(e.cp_pos),
            f3(e.pos),
            f3(e.vel),
            e.speed,
            e.speed * 3.6,
            yaw,
            e.wheels[0].steer,
            e.samples.len(),
            e.samples.last().map(|s| s.0).unwrap_or(0)
        );
        if has(a, "--samples") {
            for (t, sm, _) in &e.samples {
                println!(
                    "        t+{t:>5}  [{:8.2},{:7.2},{:8.2}]  {:5.1} m/s  steer {:+.2} gas {:.0} brake {:.0} gear {} rpm {}",
                    sm.x, sm.y, sm.z, sm.speed_ms, sm.steer, sm.gas.min(1.0), sm.brake, sm.gear_raw, sm.rpm_raw
                );
            }
        }
    }
    if let Some(out) = flag(a, "--json") {
        std::fs::write(out, to_json(&l, path)).unwrap_or_else(|e| die(format!("{out}: {e}")));
        println!("wrote {out}");
    }
    if let Some(out) = flag(a, "--csv") {
        std::fs::write(out, to_csv(&l)).unwrap_or_else(|e| die(format!("{out}: {e}")));
        println!("wrote {out}");
    }
}
