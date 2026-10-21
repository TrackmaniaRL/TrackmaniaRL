//! Rendering a decoded run as text: the 29-column CSV and the two JSON shapes.
//!
//! These reproduce, byte for byte, the artefacts the retired Python
//! (`entrec.py`, `decode_all.py`) wrote — `%.6g` floats, `True`/`False` bools,
//! CRLF line endings, `repr(float)` in JSON. That byte-identity is not
//! nostalgia: it is the strongest available check that the Rust decode is the
//! same decode, and `tests/golden_decode.rs` asserts it over the 51 reference
//! trajectories. See `crate::pyfmt` for the three CPython behaviours it needs.
//!
//! This lives in `tmtraj`, not `gbx`: `gbx` says what the bytes are, this says
//! how we choose to print them.

use gbx::record;
use record::{Decoded, Sample, FIELD_CONFIDENCE};

// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Val {
    /// `f32`, not `f64`, because `Sample`'s floats are f32 -- see the note on
    /// `gbx::record::Sample`. Printing a WIDENED f32 is the trap here: the
    /// shortest string that round-trips as f64 is long and ugly
    /// (`0.10000000149011612`), while the same value printed as f32 is `0.1`.
    /// Keeping the narrow type all the way to the formatter keeps the output
    /// both short and exact.
    F(f32),
    I(i64),
    B(bool),
}

impl Val {
    pub fn as_f64(self) -> f64 {
        match self {
            Val::F(v) => v as f64,
            Val::I(v) => v as f64,
            Val::B(v) => {
                if v {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
    /// Rendered exactly as `csv.writer` renders the Python value.
    pub fn csv(self) -> String {
        match self {
            Val::F(v) => crate::json::fmt_g6(v as f64),
            Val::I(v) => v.to_string(),
            Val::B(v) => (if v { "True" } else { "False" }).to_string(),
        }
    }
}

/// Read any documented field of a sample by name.
///
/// A trait, not an inherent method, because `Sample` belongs to `gbx` and this
/// name-keyed view is a presentation concern. `tests` assert every name in
/// `FIELD_CONFIDENCE` resolves here, so the table and the accessor cannot drift.
pub trait SampleFields {
    fn field(&self, name: &str) -> Val;
}

impl SampleFields for Sample {
    fn field(&self, name: &str) -> Val {
        use Val::*;
        match name {
            "time_ms" => I(self.time_ms as i64),
            "x" => F(self.x),
            "y" => F(self.y),
            "z" => F(self.z),
            "speed_ms" => F(self.speed_ms),
            "speed_kmh" => F(self.speed_kmh),
            "vx" => F(self.vx),
            "vy" => F(self.vy),
            "vz" => F(self.vz),
            "qx" => F(self.qx),
            "qy" => F(self.qy),
            "qz" => F(self.qz),
            "qw" => F(self.qw),
            "yaw" => F(self.yaw),
            "pitch" => F(self.pitch),
            "roll" => F(self.roll),
            "gear" => F(self.gear),
            "rpm_raw" => I(self.rpm_raw as i64),
            "side_speed" => F(self.side_speed),
            "steer" => F(self.steer),
            "brake" => F(self.brake),
            "gas" => F(self.gas),
            "is_turbo" => B(self.is_turbo),
            "turbo_time" => F(self.turbo_time),
            "is_ground_contact" => B(self.is_ground_contact),
            "is_top_contact" => B(self.is_top_contact),
            "wetness" => F(self.wetness),
            "sim_time_coef" => F(self.sim_time_coef),
            "fl_wheel_rot" => F(self.fl_wheel_rot),
            "fr_wheel_rot" => F(self.fr_wheel_rot),
            "rr_wheel_rot" => F(self.rr_wheel_rot),
            "rl_wheel_rot" => F(self.rl_wheel_rot),
            "fl_dampen" => F(self.fl_dampen),
            "fr_dampen" => F(self.fr_dampen),
            "rr_dampen" => F(self.rr_dampen),
            "rl_dampen" => F(self.rl_dampen),
            "fl_ice" => F(self.fl_ice),
            "fr_ice" => F(self.fr_ice),
            "rr_ice" => F(self.rr_ice),
            "rl_ice" => F(self.rl_ice),
            "fl_dirt" => F(self.fl_dirt),
            "fr_dirt" => F(self.fr_dirt),
            "rr_dirt" => F(self.rr_dirt),
            "rl_dirt" => F(self.rl_dirt),
            "ground_mode_raw" => I(self.ground_mode_raw as i64),
            "booster_air_control_raw" => I(self.booster_air_control_raw as i64),
            "vehicle_state_raw" => I(self.vehicle_state_raw as i64),
            "gear_raw" => I(self.gear_raw as i64),
            other => panic!("unknown sample field {:?}", other),
        }
    }
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

pub const CSV_COLUMNS: &[&str] = &[
    "time_ms", "x", "y", "z", "speed_kmh", "speed_ms", "vx", "vy", "vz", "yaw", "pitch", "roll",
    "qx", "qy", "qz", "qw", "gear", "rpm_raw", "steer", "gas", "brake", "side_speed", "is_turbo",
    "is_ground_contact", "turbo_time", "fl_dampen", "fr_dampen", "rr_dampen", "rl_dampen",
];

/// Byte-for-byte the file `entrec.write_csv` produces (`%.6g` floats,
/// `True`/`False` bools, `\r\n` line endings from `csv.writer`).
pub fn csv_string(dec: &Decoded) -> String {
    let mut out = String::with_capacity(dec.samples.len() * 220);
    out.push_str(&CSV_COLUMNS.join(","));
    out.push_str("\r\n");
    for s in &dec.samples {
        let mut first = true;
        for c in CSV_COLUMNS {
            if !first {
                out.push(',');
            }
            first = false;
            out.push_str(&s.field(c).csv());
        }
        out.push_str("\r\n");
    }
    out
}

/// Byte-for-byte the file `decode_all.py` writes into `paths/`:
/// `{"name","time_ms","checkpoints_ms","sample_period_ms","samples":[{t,x,y,z,speed,gear,yaw}]}`
/// with `round(,4)` on the positions/speed, `round(,6)` on yaw and CPython
/// `repr` float rendering.
pub fn path_json_string(dec: &Decoded) -> String {
    use crate::json::{py_repr, py_round};
    let mut o = String::with_capacity(dec.samples.len() * 110);
    o.push_str("{\"name\": \"");
    o.push_str(&dec.name);
    o.push_str("\", \"time_ms\": ");
    match dec.race_time_ms {
        Some(t) => o.push_str(&t.to_string()),
        None => o.push_str("null"),
    }
    o.push_str(", \"checkpoints_ms\": [");
    for (i, c) in dec.checkpoints_ms.iter().enumerate() {
        if i > 0 {
            o.push_str(", ");
        }
        o.push_str(&c.to_string());
    }
    o.push_str("], \"sample_period_ms\": ");
    match dec.sample_period_ms {
        Some(p) => o.push_str(&p.to_string()),
        None => o.push_str("null"),
    }
    o.push_str(", \"samples\": [");
    for (i, s) in dec.samples.iter().enumerate() {
        if i > 0 {
            o.push_str(", ");
        }
        o.push_str("{\"t\": ");
        o.push_str(&s.time_ms.to_string());
        o.push_str(", \"x\": ");
        o.push_str(&py_repr(py_round(s.x as f64, 4)));
        o.push_str(", \"y\": ");
        o.push_str(&py_repr(py_round(s.y as f64, 4)));
        o.push_str(", \"z\": ");
        o.push_str(&py_repr(py_round(s.z as f64, 4)));
        o.push_str(", \"speed\": ");
        o.push_str(&py_repr(py_round(s.speed_kmh as f64, 4)));
        o.push_str(", \"gear\": ");
        o.push_str(&py_repr(s.gear as f64));
        o.push_str(", \"yaw\": ");
        o.push_str(&py_repr(py_round(s.yaw as f64, 6)));
        o.push('}');
    }
    o.push_str("]}");
    o
}

/// The full decode as JSON -- every field of every sample, plus the record
/// header. (`entrec.py --json` wrote this shape.)
pub fn full_json_string(dec: &Decoded) -> String {
    use crate::json::py_repr;
    let mut o = String::new();
    o.push('{');
    o.push_str(&format!("\"path\": \"{}\", ", dec.path));
    o.push_str(&format!("\"version\": {}, ", dec.version));
    o.push_str(&format!("\"start_ms\": {}, ", dec.start_ms));
    o.push_str(&format!("\"end_ms\": {}, ", dec.end_ms));
    o.push_str(&format!(
        "\"sample_period_ms\": {}, ",
        dec.sample_period_ms
            .map_or("null".to_string(), |v| v.to_string())
    ));
    o.push_str(&format!("\"sample_size\": {}, ", dec.sample_size));
    o.push_str(&format!(
        "\"race_time_ms\": {}, ",
        dec.race_time_ms.map_or("null".to_string(), |v| v.to_string())
    ));
    o.push_str("\"checkpoints_ms\": [");
    for (i, c) in dec.checkpoints_ms.iter().enumerate() {
        if i > 0 {
            o.push_str(", ");
        }
        o.push_str(&c.to_string());
    }
    o.push_str("], ");
    o.push_str(&format!(
        "\"bytes_consumed\": {}, \"bytes_total\": {}, ",
        dec.bytes_consumed, dec.bytes_total
    ));
    o.push_str("\"ents\": [");
    for (i, e) in dec.ents.iter().enumerate() {
        if i > 0 {
            o.push_str(", ");
        }
        o.push_str(&format!(
            "{{\"type\": {}, \"classId\": {}, \"n_samples\": {}, \"sample_size\": {}}}",
            e.type_,
            e.class_id.map_or("null".to_string(), |c| c.to_string()),
            e.n_samples,
            e.sample_size
        ));
    }
    o.push_str("], \"samples\": [");
    for (i, s) in dec.samples.iter().enumerate() {
        if i > 0 {
            o.push_str(", ");
        }
        o.push('{');
        let mut first = true;
        for f in FIELD_CONFIDENCE {
            if !first {
                o.push_str(", ");
            }
            first = false;
            o.push_str(&format!("\"{}\": ", f.name));
            match s.field(f.name) {
                Val::F(v) => o.push_str(&crate::json::py_repr_f32(v)),
                Val::I(v) => o.push_str(&v.to_string()),
                Val::B(v) => o.push_str(if v { "true" } else { "false" }),
            }
        }
        o.push('}');
    }
    o.push_str("]}");
    o
}
