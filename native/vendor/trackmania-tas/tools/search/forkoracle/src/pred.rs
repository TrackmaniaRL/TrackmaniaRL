//! Driver side of the watchdog: the condition language, the reference line,
//! and the wire format that arms them inside the fork server.
//!
//! # The condition language
//!
//! One predicate per `--pred` flag, in three colon-separated parts:
//!
//! ```text
//!     NAME:KIND:key=value,key=value,...
//! ```
//!
//! `NAME` is free text and is what the search logs when the predicate fires.
//! `KIND` is one of:
//!
//! | kind | fires when | keys (defaults) |
//! |---|---|---|
//! | `speeddrop` | speed falls below `frac` of its peak over the last `win` ticks -- what a crash looks like | `frac=0.5`, `win=50`, `minpeak=8` (m/s, below which it is not armed), `need=1`, `after=0`, `until=` |
//! | `floor` | speed below `speed` for `need` consecutive ticks -- stopped, stuck, or facing a wall | `speed=3`, `need=30` |
//! | `box` | the car leaves an axis-aligned region -- off the track, off the map | `xmin= xmax= ymin= ymax= zmin= zmax=`, `need=1` |
//! | `offref` | the car is further than `dist` metres from the reference line | `dist=12`, `need=5` |
//! | `noprog` | net displacement over the last `win` ticks is under `dist` metres | `dist=5`, `win=100` |
//!
//! Common keys: `need` (consecutive ticks required), `after` / `until` (tape
//! tick range in which the predicate is live -- `after` is what keeps a
//! predicate off the standing start, where the car is legitimately stationary).
//!
//! Several may be armed at once; they are evaluated in the order given and the
//! first to trip wins, which is why each one is named.
//!
//! # The reference line and `progress`
//!
//! `offref` and the progress measure both use a reference trajectory: one
//! position per tape tick, from `fk btraj` (which measures ANY tape, including
//! a search incumbent with no recorded telemetry). Deviation is measured to the
//! nearest point of the line within a window of the last match, not to the
//! reference's position at the same millisecond: a candidate that is simply
//! 100 ms ahead of the incumbent has not left the line.
//!
//! `progress` is the arclength of that nearest point, maximised over the run
//! and only counted while inside `corridor` metres of the line. It is the
//! measure the search scores aborted candidates by, so it has to mean the same
//! thing for an aborted and a completed run -- which it does, because the child
//! computes it identically in both cases.

use crate::pred_core::{
    key_eval, Fire, Gate, KeyOp, Pred, Summary, KEYOP_BYTES, KOP_ABS, KOP_ADD, KOP_ALONG, KOP_AXISDOT,
    KOP_BODYVEL, KOP_CONST, KOP_DIST, KOP_DIV, KOP_DSPEED, KOP_MAX, KOP_MIN, KOP_MUL, KOP_NEG,
    KOP_DOMEGA, KOP_OMEGA, KOP_OMEGAMAG, KOP_POS, KOP_SPEED, KOP_SUB, KOP_VDIST, KOP_VEL,
    MAXKOPS, PRED_BYTES,
};

/// One armed, named condition.
#[derive(Clone)]
pub struct NamedPred {
    pub name: String,
    pub pred: Pred,
}

/// A reference line resampled onto tape ticks, with its cumulative arclength.
#[derive(Clone, Default)]
pub struct RefLineData {
    pub n: usize,
    /// 3 * n
    pub xyz: Vec<f32>,
    /// n
    pub s: Vec<f32>,
    /// Race ms of the first and last reference sample this line was built
    /// from, and how many of its ticks run PAST the candidate tape's own last
    /// tick. All zero for a line built straight from points.
    ///
    /// These exist because the line's length used to depend silently on the
    /// candidate tape's length; see [`RefLineData::from_samples`].
    pub first_ms: i64,
    pub last_ms: i64,
    pub past_tape: usize,
}

impl RefLineData {
    pub fn from_points(pts: &[[f64; 3]]) -> RefLineData {
        let n = pts.len();
        let mut xyz = Vec::with_capacity(3 * n);
        let mut s = Vec::with_capacity(n);
        let mut acc = 0.0f64;
        for (i, p) in pts.iter().enumerate() {
            if i > 0 {
                let q = pts[i - 1];
                acc += ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2))
                    .sqrt();
            }
            xyz.push(p[0] as f32);
            xyz.push(p[1] as f32);
            xyz.push(p[2] as f32);
            s.push(acc as f32);
        }
        RefLineData { n, xyz, s, first_ms: 0, last_ms: 0, past_tape: 0 }
    }

    /// Arclength at a tape tick, for turning a checkpoint time into a progress
    /// threshold.
    pub fn s_at_tick(&self, tick: usize) -> f32 {
        if self.n == 0 {
            0.0
        } else {
            self.s[tick.min(self.n - 1)]
        }
    }

    /// Nearest point on the line, searched over the whole line: the offline
    /// twin of what the child does with a moving window.
    pub fn nearest(&self, p: [f64; 3]) -> (usize, f64) {
        let mut bi = 0;
        let mut bd = f64::MAX;
        for i in 0..self.n {
            let d = ((self.xyz[3 * i] as f64 - p[0]).powi(2)
                + (self.xyz[3 * i + 1] as f64 - p[1]).powi(2)
                + (self.xyz[3 * i + 2] as f64 - p[2]).powi(2))
            .sqrt();
            if d < bd {
                bd = d;
                bi = i;
            }
        }
        (bi, bd)
    }
}

/// Everything needed to arm a fork server.
#[derive(Clone, Default)]
pub struct Watch {
    pub preds: Vec<NamedPred>,
    pub refline: RefLineData,
    pub corridor: f32,
    pub ahead: i32,
    pub back: i32,
    /// Arclength of the reference's finish; predicates are disarmed past it.
    pub finish_s: f32,
    /// 1 = the cheap clock-gated sampling path in the child.
    pub fast: u32,
    /// World coordinate of the sub-tick timing plane; 0 disables it.
    pub plane_x: f32,
    /// The axis that plane cuts: 0 = x, 1 = y, 2 = z. Trailing on the wire.
    pub plane_axis: u32,
    /// The state objective. Disarmed by default, and then the child does not
    /// evaluate a single instruction of it.
    pub gate: Gate,
    /// The event clause: a thing that happens, and what to score after it.
    pub fire: Fire,
}

fn getf(kv: &[(String, String)], k: &str, d: f32) -> f32 {
    kv.iter()
        .find(|(a, _)| a == k)
        .and_then(|(_, v)| v.parse::<f32>().ok())
        .unwrap_or(d)
}
fn geti(kv: &[(String, String)], k: &str, d: i32) -> i32 {
    kv.iter()
        .find(|(a, _)| a == k)
        .and_then(|(_, v)| v.parse::<i32>().ok())
        .unwrap_or(d)
}

/// Parse one `NAME:KIND:k=v,...` spec. Unknown keys are an error rather than a
/// shrug: a typo in a watchdog's threshold is exactly the kind of mistake that
/// silently kills good candidates.
pub fn parse_spec(spec: &str) -> Result<NamedPred, String> {
    let parts: Vec<&str> = spec.splitn(3, ':').collect();
    if parts.len() < 2 {
        return Err(format!("bad predicate {:?}: want NAME:KIND[:k=v,...]", spec));
    }
    let name = parts[0].to_string();
    let kind_s = parts[1];
    let kv: Vec<(String, String)> = if parts.len() == 3 && !parts[2].is_empty() {
        let mut v = Vec::new();
        for item in parts[2].split(',') {
            let (a, b) = item
                .split_once('=')
                .ok_or_else(|| format!("bad key=value {:?} in {:?}", item, spec))?;
            v.push((a.trim().to_string(), b.trim().to_string()));
        }
        v
    } else {
        Vec::new()
    };
    let allowed: &[&str] = match kind_s {
        "speeddrop" => &["frac", "win", "minpeak", "need", "after", "until"],
        "floor" => &["speed", "need", "after", "until"],
        "box" => &["xmin", "xmax", "ymin", "ymax", "zmin", "zmax", "need", "after", "until"],
        "offref" => &["dist", "need", "after", "until"],
        "noprog" => &["dist", "win", "need", "after", "until"],
        k => return Err(format!("unknown predicate kind {:?}", k)),
    };
    for (k, _) in &kv {
        if !allowed.contains(&k.as_str()) {
            return Err(format!(
                "predicate {} ({}): unknown key {:?}; allowed: {:?}",
                name, kind_s, k, allowed
            ));
        }
    }
    let mut p = Pred::ZERO;
    // ONE table of predicate names, in `pred_core`, which is the file the shim
    // compiles into the child. Each arm below used to set `p.kind` itself, so
    // the language had two name-to-kind maps in one crate and adding a
    // predicate meant editing both.
    p.kind = crate::pred_core::kind_of(kind_s)
        .ok_or_else(|| format!("unknown predicate kind {:?}", kind_s))?;
    p.after = geti(&kv, "after", 0);
    p.until = geti(&kv, "until", i32::MAX);
    match kind_s {
        "speeddrop" => {
            p.win = geti(&kv, "win", 50).max(1) as u32;
            p.need = geti(&kv, "need", 1).max(1) as u32;
            p.p[0] = getf(&kv, "frac", 0.5);
            p.p[1] = getf(&kv, "minpeak", 8.0);
        }
        "floor" => {
            p.need = geti(&kv, "need", 30).max(1) as u32;
            p.p[0] = getf(&kv, "speed", 3.0);
        }
        "box" => {
            p.need = geti(&kv, "need", 1).max(1) as u32;
            p.p[0] = getf(&kv, "xmin", f32::NEG_INFINITY);
            p.p[1] = getf(&kv, "xmax", f32::INFINITY);
            p.p[2] = getf(&kv, "ymin", f32::NEG_INFINITY);
            p.p[3] = getf(&kv, "ymax", f32::INFINITY);
            p.p[4] = getf(&kv, "zmin", f32::NEG_INFINITY);
            p.p[5] = getf(&kv, "zmax", f32::INFINITY);
        }
        "offref" => {
            p.need = geti(&kv, "need", 5).max(1) as u32;
            p.p[0] = getf(&kv, "dist", 12.0);
        }
        "noprog" => {
            p.win = geti(&kv, "win", 100).max(1) as u32;
            p.need = geti(&kv, "need", 1).max(1) as u32;
            p.p[0] = getf(&kv, "dist", 5.0);
        }
        _ => unreachable!(),
    }
    if p.win as usize >= crate::pred_core::RINGW {
        return Err(format!(
            "predicate {}: win={} exceeds the child's ring buffer ({} ticks)",
            name,
            p.win,
            crate::pred_core::RINGW
        ));
    }
    Ok(NamedPred { name, pred: p })
}

// -------------------------------------------------------------- the gate
//
// THE STATE OBJECTIVE'S TWO HALVES, and why they are parsed here.
//
// `--gate` is a box. `--gate-key` is an expression over the car's whole state
// inside it. Both are compiled in this process, where a mistake is a message;
// the child only ever executes a fixed-size program.

/// Parse `xmin=..,xmax=..,ymin=..,ymax=..,zmin=..,zmax=..[,minspeed=..]`.
///
/// All six bounds are required. A gate with a defaulted-open side is a gate
/// that fires somewhere else on the map, and the one thing this feature must
/// not do is measure the wrong place convincingly.
pub fn parse_gate(spec: &str, key: &str) -> Result<Gate, String> {
    let mut kv: Vec<(String, String)> = Vec::new();
    for item in spec.split(',').filter(|s| !s.trim().is_empty()) {
        let (a, b) = item
            .split_once('=')
            .ok_or_else(|| format!("bad key=value {:?} in --gate {:?}", item, spec))?;
        kv.push((a.trim().to_string(), b.trim().to_string()));
    }
    let allowed = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax", "minspeed"];
    for (k, _) in &kv {
        if !allowed.contains(&k.as_str()) {
            return Err(format!("--gate: unknown key {:?}; allowed: {:?}", k, allowed));
        }
    }
    let need = |k: &str| -> Result<f32, String> {
        kv.iter()
            .find(|(a, _)| a == k)
            .ok_or_else(|| {
                format!(
                    "--gate: {} is required. All six bounds must be given: a side left open is a \
                     box that also contains somewhere else on the map.",
                    k
                )
            })
            .and_then(|(_, v)| {
                v.parse::<f32>().map_err(|_| format!("--gate {}: {:?} is not a number", k, v))
            })
    };
    let b = [need("xmin")?, need("xmax")?, need("ymin")?, need("ymax")?, need("zmin")?, need("zmax")?];
    for (i, ax) in ["x", "y", "z"].iter().enumerate() {
        if b[2 * i + 1] <= b[2 * i] {
            return Err(format!(
                "--gate: {}max ({}) must be greater than {}min ({})",
                ax, b[2 * i + 1], ax, b[2 * i]
            ));
        }
    }
    let minspeed = match kv.iter().find(|(a, _)| a == "minspeed") {
        Some((_, v)) => v.parse::<f32>().map_err(|_| format!("--gate minspeed: {:?}", v))?,
        None => 0.0,
    };
    Ok(Gate { armed: true, bounds: b, minspeed, prog: parse_key(key)? })
}

/// Compile a key expression into the child's postfix program.
///
/// ```text
///   expr    := term (('+'|'-') term)*
///   term    := unary (('*'|'/') unary)*
///   unary   := '-' unary | atom
///   atom    := NUMBER | NAME | NAME '(' expr,... ')' | '(' expr ')'
/// ```
///
/// | name | value |
/// |---|---|
/// | `speed` | the car's speed, m/s |
/// | `dspeed` | the ONE-TICK rise in speed, m/s per 10 ms -- the launch detector |
/// | `omega` `omegax` `omegay` `omegaz` | BODY-FRAME angular rate, deg/s |
/// | `domega` | the change in that rate per tick -- the LOAD detector: a free rigid body holds omega exactly constant |
/// | `vx` `vy` `vz` | world velocity components |
/// | `px` `py` `pz` | world position components |
/// | `bodyright` `bodyup` `bodyfwd` | velocity resolved in the CAR's own frame -- `bodyright` is the ghost format's `side_speed` |
/// | `along(x,y,z)` | speed along a world direction |
/// | `nose(x,y,z)` `roof(x,y,z)` `flank(x,y,z)` | how well the car's forward / up / right axis points along a world direction, -1..1 |
/// | `dist(x,y,z)` | metres from a world point |
/// | `vdist(x,y,z)` | m/s from a target velocity |
/// | `abs(e)` `min(a,b)` `max(a,b)` | |
///
/// Bigger is always better, so a quantity to be MINIMISED is negated by the
/// person writing the expression, in the open, where it can be read:
/// `-(dist(70.2,50.4,708.9) + vdist(...)/5)`.
pub fn parse_key(src: &str) -> Result<[KeyOp; MAXKOPS], String> {
    let toks = lex(src)?;
    let mut p = KeyParser { t: &toks, i: 0, out: Vec::new(), src };
    p.expr()?;
    if p.i != p.t.len() {
        return Err(format!("--gate-key {:?}: trailing {:?}", src, p.t[p.i]));
    }
    if p.out.is_empty() {
        return Err(format!("--gate-key {:?}: empty", src));
    }
    if p.out.len() > MAXKOPS - 1 {
        return Err(format!(
            "--gate-key {:?}: {} operations, the child holds {}",
            src,
            p.out.len(),
            MAXKOPS - 1
        ));
    }
    let mut prog = [KeyOp::END; MAXKOPS];
    for (i, k) in p.out.iter().enumerate() {
        prog[i] = *k;
    }
    // A compiled key that cannot produce a number on a plausible state is a
    // parser bug, and it would show up in the child as every candidate scoring
    // nothing. Cheaper to find it here.
    let probe = key_eval(&prog, crate::pred_core::St::at([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 0.0, 0.0, 0.0]));
    if !probe.is_finite() {
        return Err(format!("--gate-key {:?}: compiled to a program that does not evaluate", src));
    }
    Ok(prog)
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Num(f32),
    Name(String),
    Punct(char),
}

fn lex(src: &str) -> Result<Vec<Tok>, String> {
    let b: Vec<char> = src.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() || (c == '.' && i + 1 < b.len() && b[i + 1].is_ascii_digit()) {
            let s = i;
            while i < b.len() && (b[i].is_ascii_digit() || b[i] == '.') {
                i += 1;
            }
            // an exponent, so 1e-3 is a number and not `1e` minus `3`
            if i < b.len() && (b[i] == 'e' || b[i] == 'E') {
                let save = i;
                i += 1;
                if i < b.len() && (b[i] == '+' || b[i] == '-') {
                    i += 1;
                }
                if i < b.len() && b[i].is_ascii_digit() {
                    while i < b.len() && b[i].is_ascii_digit() {
                        i += 1;
                    }
                } else {
                    i = save;
                }
            }
            let t: String = b[s..i].iter().collect();
            out.push(Tok::Num(t.parse().map_err(|_| format!("bad number {:?}", t))?));
        } else if c.is_ascii_alphabetic() || c == '_' {
            let s = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == '_') {
                i += 1;
            }
            out.push(Tok::Name(b[s..i].iter().collect()));
        } else if "+-*/(),".contains(c) {
            out.push(Tok::Punct(c));
            i += 1;
        } else {
            return Err(format!("--gate-key: unexpected {:?} in {:?}", c, src));
        }
    }
    Ok(out)
}

struct KeyParser<'a> {
    t: &'a [Tok],
    i: usize,
    out: Vec<KeyOp>,
    src: &'a str,
}

impl<'a> KeyParser<'a> {
    fn peek(&self) -> Option<&Tok> {
        self.t.get(self.i)
    }
    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(&Tok::Punct(c)) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, c: char) -> Result<(), String> {
        if self.eat(c) {
            Ok(())
        } else {
            Err(format!("--gate-key {:?}: expected {:?}, got {:?}", self.src, c, self.peek()))
        }
    }
    fn emit(&mut self, op: u32, axis: u32, a: [f32; 3]) {
        self.out.push(KeyOp { op, axis, a });
    }

    fn expr(&mut self) -> Result<(), String> {
        self.term()?;
        loop {
            if self.eat('+') {
                self.term()?;
                self.emit(KOP_ADD, 0, [0.0; 3]);
            } else if self.eat('-') {
                self.term()?;
                self.emit(KOP_SUB, 0, [0.0; 3]);
            } else {
                return Ok(());
            }
        }
    }
    fn term(&mut self) -> Result<(), String> {
        self.unary()?;
        loop {
            if self.eat('*') {
                self.unary()?;
                self.emit(KOP_MUL, 0, [0.0; 3]);
            } else if self.eat('/') {
                self.unary()?;
                self.emit(KOP_DIV, 0, [0.0; 3]);
            } else {
                return Ok(());
            }
        }
    }
    fn unary(&mut self) -> Result<(), String> {
        if self.eat('-') {
            self.unary()?;
            self.emit(KOP_NEG, 0, [0.0; 3]);
            return Ok(());
        }
        self.atom()
    }
    /// Three floats in parentheses, for the terms that take a direction, a
    /// point or a target velocity.
    fn vec3(&mut self, name: &str) -> Result<[f32; 3], String> {
        self.expect('(')?;
        let mut v = [0.0f32; 3];
        for (i, slot) in v.iter_mut().enumerate() {
            let mut sign = 1.0;
            while self.eat('-') {
                sign = -sign;
            }
            match self.peek() {
                Some(Tok::Num(n)) => {
                    *slot = sign * *n;
                    self.i += 1;
                }
                other => {
                    return Err(format!(
                        "--gate-key: {}() wants three plain numbers, argument {} is {:?}",
                        name,
                        i + 1,
                        other
                    ))
                }
            }
            if i < 2 {
                self.expect(',')?;
            }
        }
        self.expect(')')?;
        Ok(v)
    }
    fn unit(&mut self, name: &str) -> Result<[f32; 3], String> {
        let v = self.vec3(name)?;
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if !(n > 0.0) {
            return Err(format!("--gate-key: {}() needs a direction, not (0,0,0)", name));
        }
        Ok([v[0] / n, v[1] / n, v[2] / n])
    }
    fn atom(&mut self) -> Result<(), String> {
        match self.peek().cloned() {
            Some(Tok::Num(n)) => {
                self.i += 1;
                self.emit(KOP_CONST, 0, [n, 0.0, 0.0]);
                Ok(())
            }
            Some(Tok::Punct('(')) => {
                self.i += 1;
                self.expr()?;
                self.expect(')')
            }
            Some(Tok::Name(n)) => {
                self.i += 1;
                match n.as_str() {
                    "speed" => self.emit(KOP_SPEED, 0, [0.0; 3]),
                    "dspeed" => self.emit(KOP_DSPEED, 0, [0.0; 3]),
                    "omegax" => self.emit(KOP_OMEGA, 0, [0.0; 3]),
                    "omegay" => self.emit(KOP_OMEGA, 1, [0.0; 3]),
                    "omegaz" => self.emit(KOP_OMEGA, 2, [0.0; 3]),
                    "omega" => self.emit(KOP_OMEGAMAG, 0, [0.0; 3]),
                    "domega" => self.emit(KOP_DOMEGA, 0, [0.0; 3]),
                    "vx" => self.emit(KOP_VEL, 0, [0.0; 3]),
                    "vy" => self.emit(KOP_VEL, 1, [0.0; 3]),
                    "vz" => self.emit(KOP_VEL, 2, [0.0; 3]),
                    "px" => self.emit(KOP_POS, 0, [0.0; 3]),
                    "py" => self.emit(KOP_POS, 1, [0.0; 3]),
                    "pz" => self.emit(KOP_POS, 2, [0.0; 3]),
                    "bodyright" => self.emit(KOP_BODYVEL, 0, [0.0; 3]),
                    "bodyup" => self.emit(KOP_BODYVEL, 1, [0.0; 3]),
                    "bodyfwd" => self.emit(KOP_BODYVEL, 2, [0.0; 3]),
                    "along" => {
                        let d = self.unit("along")?;
                        self.emit(KOP_ALONG, 0, d);
                    }
                    "flank" => {
                        let d = self.unit("flank")?;
                        self.emit(KOP_AXISDOT, 0, d);
                    }
                    "roof" => {
                        let d = self.unit("roof")?;
                        self.emit(KOP_AXISDOT, 1, d);
                    }
                    "nose" => {
                        let d = self.unit("nose")?;
                        self.emit(KOP_AXISDOT, 2, d);
                    }
                    "dist" => {
                        let p = self.vec3("dist")?;
                        self.emit(KOP_DIST, 0, p);
                    }
                    "vdist" => {
                        let p = self.vec3("vdist")?;
                        self.emit(KOP_VDIST, 0, p);
                    }
                    "abs" => {
                        self.expect('(')?;
                        self.expr()?;
                        self.expect(')')?;
                        self.emit(KOP_ABS, 0, [0.0; 3]);
                    }
                    "min" | "max" => {
                        self.expect('(')?;
                        self.expr()?;
                        self.expect(',')?;
                        self.expr()?;
                        self.expect(')')?;
                        self.emit(if n == "min" { KOP_MIN } else { KOP_MAX }, 0, [0.0; 3]);
                    }
                    other => {
                        return Err(format!(
                            "--gate-key {:?}: unknown term {:?}. Known: speed, vx vy vz, px py pz, \
                             bodyright bodyup bodyfwd, along(x,y,z), nose/roof/flank(x,y,z), \
                             dspeed, omega, omegax/y/z, domega, dist(x,y,z), \
                             vdist(vx,vy,vz), abs(), min(), max()",
                            self.src, other
                        ))
                    }
                }
                Ok(())
            }
            other => Err(format!("--gate-key {:?}: expected a term, got {:?}", self.src, other)),
        }
    }
}

/// Parse the event clause: a condition, the value it must reach, an optional
/// box it must happen inside, and what to score afterwards.
///
/// `where_spec` empty means anywhere; `after` empty means the event itself is
/// the whole objective and everything that fires ties.
#[allow(clippy::too_many_arguments)]
pub fn parse_fire(
    cond: &str,
    at: f32,
    need: u32,
    where_spec: &str,
    after: &str,
    after_ticks: u32,
    after_from_end: bool,
) -> Result<Fire, String> {
    let where_box = if where_spec.is_empty() {
        Gate::NONE
    } else {
        // the same six-bound box as `--gate`, and the same refusal of a side
        // left open: a half-open region for a launch to happen in is a region
        // that also contains somewhere else on the map.
        parse_gate(where_spec, "speed")?
    };
    Ok(Fire {
        armed: true,
        cond: parse_key(cond)?,
        at,
        need: need.max(1),
        after_ticks,
        after_from_end,
        where_box,
        after: if after.is_empty() { [KeyOp::END; MAXKOPS] } else { parse_key(after)? },
    })
}

impl Watch {
    pub fn new() -> Watch {
        Watch {
            preds: Vec::new(),
            refline: RefLineData::default(),
            corridor: 40.0,
            ahead: 24,
            back: 8,
            finish_s: 0.0,
            fast: 1,
            plane_x: 0.0,
            plane_axis: 0,
            gate: Gate::NONE,
            fire: Fire::NONE,
        }
    }

    pub fn describe(&self) -> String {
        let mut s = String::new();
        if self.gate.armed {
            let b = &self.gate.bounds;
            s.push_str(&format!(
                "  gate  x {}..{}  y {}..{}  z {}..{}  minspeed {} m/s, key in {} ops\n",
                b[0], b[1], b[2], b[3], b[4], b[5], self.gate.minspeed, self.gate_kops()
            ));
        }
        if self.fire.armed {
            s.push_str(&format!(
                "  fire  when a {}-op condition reaches {}{}, then {}\n",
                prog_len(&self.fire.cond),
                self.fire.at,
                if self.fire.where_box.armed {
                    let w = &self.fire.where_box.bounds;
                    format!(
                        " inside x {}..{} y {}..{} z {}..{}",
                        w[0], w[1], w[2], w[3], w[4], w[5]
                    )
                } else {
                    " anywhere".into()
                },
                if prog_len(&self.fire.after) > 0 {
                    format!("maximise a {}-op key after it", prog_len(&self.fire.after))
                } else {
                    "everything that fires ties".into()
                }
            ));
        }
        for (i, np) in self.preds.iter().enumerate() {
            let p = &np.pred;
            s.push_str(&format!(
                "  [{}] {:<10} {:<10} need={} after={} p={:?}\n",
                i,
                np.name,
                crate::pred_core::kind_name(p.kind),
                p.need,
                p.after,
                &p.p[..2]
            ));
        }
        s
    }

    pub fn name_of(&self, i: i32) -> &str {
        if i < 0 || i as usize >= self.preds.len() {
            "-"
        } else {
            &self.preds[i as usize].name
        }
    }

    /// The `A` frame: predicates, record layout, watched segments, reference,
    /// and the state objective.
    #[allow(clippy::too_many_arguments)]
    pub fn arm_payload(
        &self,
        clock0: i64,
        off_clock: u32,
        off_quat: u32,
        off_pos: u32,
        off_vel: u32,
        rec_len: u32,
        segs: &[(u64, u32)],
    ) -> Vec<u8> {
        let mut v = Vec::with_capacity(64 + 16 * self.refline.n);
        v.push(b'A');
        v.extend_from_slice(&(self.preds.len() as u32).to_le_bytes());
        let mut buf = [0u8; PRED_BYTES];
        for np in &self.preds {
            np.pred.encode(&mut buf);
            v.extend_from_slice(&buf);
        }
        v.extend_from_slice(&clock0.to_le_bytes());
        v.extend_from_slice(&off_clock.to_le_bytes());
        v.extend_from_slice(&off_pos.to_le_bytes());
        v.extend_from_slice(&off_vel.to_le_bytes());
        v.extend_from_slice(&rec_len.to_le_bytes());
        v.extend_from_slice(&(segs.len() as u32).to_le_bytes());
        for (a, l) in segs {
            v.extend_from_slice(&a.to_le_bytes());
            v.extend_from_slice(&l.to_le_bytes());
        }
        v.extend_from_slice(&self.corridor.to_le_bytes());
        v.extend_from_slice(&self.ahead.to_le_bytes());
        v.extend_from_slice(&self.back.to_le_bytes());
        v.extend_from_slice(&self.finish_s.to_le_bytes());
        v.extend_from_slice(&self.fast.to_le_bytes());
        v.extend_from_slice(&(self.refline.n as u32).to_le_bytes());
        for f in &self.refline.xyz {
            v.extend_from_slice(&f.to_le_bytes());
        }
        for f in &self.refline.s {
            v.extend_from_slice(&f.to_le_bytes());
        }
        // trailing, so an older shim simply ignores it
        v.extend_from_slice(&self.plane_x.to_le_bytes());
        // THE STATE OBJECTIVE, also trailing -- but a shim that ignores it is
        // a shim that scores every candidate as a miss, so the ARM ack reports
        // how many key operations it took and `ForkEval` refuses a mismatch.
        v.extend_from_slice(&off_quat.to_le_bytes());
        v.extend_from_slice(&(self.gate.armed as u32).to_le_bytes());
        for f in &self.gate.bounds {
            v.extend_from_slice(&f.to_le_bytes());
        }
        v.extend_from_slice(&self.gate.minspeed.to_le_bytes());
        v.extend_from_slice(&(self.gate_kops() as u32).to_le_bytes());
        let mut kb = [0u8; KEYOP_BYTES];
        for k in self.gate.prog.iter().take(self.gate_kops()) {
            k.encode(&mut kb);
            v.extend_from_slice(&kb);
        }
        // THE EVENT CLAUSE, trailing behind the gate for the same reason and
        // covered by the same ack: the ARM reply reports the total number of
        // key operations installed across all three programs.
        v.extend_from_slice(&(self.fire.armed as u32).to_le_bytes());
        v.extend_from_slice(&self.fire.at.to_le_bytes());
        v.extend_from_slice(&self.fire.need.to_le_bytes());
        v.extend_from_slice(&self.fire.after_ticks.to_le_bytes());
        v.extend_from_slice(&(self.fire.after_from_end as u32).to_le_bytes());
        v.extend_from_slice(&(self.fire.where_box.armed as u32).to_le_bytes());
        for f in &self.fire.where_box.bounds {
            v.extend_from_slice(&f.to_le_bytes());
        }
        for prog in [&self.fire.cond, &self.fire.after] {
            let n = prog_len(prog);
            v.extend_from_slice(&(n as u32).to_le_bytes());
            for k in prog.iter().take(n) {
                k.encode(&mut kb);
                v.extend_from_slice(&kb);
            }
        }
        // THE PLANE AXIS, trailing behind the event. A shim without it reads
        // x, which is what every plane before this field meant.
        v.extend_from_slice(&self.plane_axis.to_le_bytes());
        v
    }

    /// Every key operation this watch will install, across all three programs.
    /// The ARM ack reports it back and `ForkEval` refuses a mismatch, so a shim
    /// older than the driver arming it is an abort and not a silent zero.
    pub fn nkops(&self) -> usize {
        prog_len(&self.gate.prog) + prog_len(&self.fire.cond) + prog_len(&self.fire.after)
    }

    /// How many instructions the gate's own key uses.
    pub fn gate_kops(&self) -> usize {
        prog_len(&self.gate.prog)
    }

}

/// The car's WHOLE state where the gate scored: position, velocity and
/// attitude, plus the key and the tick.
///
/// It carries the quaternion because the map this feature was proven on
/// ignored position and velocity and triggered on which way the car pointed,
/// and because it is what makes the seed identity control decisive.
#[derive(Clone, Copy, Debug)]
pub struct GateRecord {
    pub tick: i32,
    pub key: f32,
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    /// `(qw, qx, qy, qz)`
    pub quat: [f32; 4],
}

impl GateRecord {
    pub fn speed(&self) -> f32 {
        (self.vel[0] * self.vel[0] + self.vel[1] * self.vel[1] + self.vel[2] * self.vel[2]).sqrt()
    }
    /// Velocity in the car's own frame: right, up, forward.
    pub fn body_vel(&self) -> [f32; 3] {
        let a = crate::pred_core::body_axes(self.quat);
        [
            a[0][0] * self.vel[0] + a[0][1] * self.vel[1] + a[0][2] * self.vel[2],
            a[1][0] * self.vel[0] + a[1][1] * self.vel[1] + a[1][2] * self.vel[2],
            a[2][0] * self.vel[0] + a[2][1] * self.vel[1] + a[2][2] * self.vel[2],
        ]
    }
}

impl std::fmt::Display for GateRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let b = self.body_vel();
        write!(
            f,
            "tick {} at ({:.2}, {:.2}, {:.2}) v ({:.2}, {:.2}, {:.2}) |v| {:.2} \
             body(right,up,fwd) ({:.2}, {:.2}, {:.2}) q ({:.4}, {:.4}, {:.4}, {:.4})",
            self.tick,
            self.pos[0],
            self.pos[1],
            self.pos[2],
            self.vel[0],
            self.vel[1],
            self.vel[2],
            self.speed(),
            b[0],
            b[1],
            b[2],
            self.quat[0],
            self.quat[1],
            self.quat[2],
            self.quat[3]
        )
    }
}

/// What one watched candidate did.
#[derive(Clone, Debug)]
pub struct Outcome {    /// Finish time in ms, if it finished.
    pub time: Option<i64>,
    /// Checkpoints reached, when the validator reported them (i.e. when the
    /// candidate was NOT aborted).
    pub cps: Option<u32>,
    pub sum: Option<Summary>,
}

impl Outcome {
    pub fn tripped(&self) -> Option<(i32, i32, f32)> {
        match &self.sum {
            Some(s) if s.trip_pred >= 0 => Some((s.trip_pred, s.trip_tick, s.trip_value)),
            _ => None,
        }
    }
    pub fn progress(&self) -> f32 {
        self.sum.map(|s| s.progress).unwrap_or(0.0)
    }
    pub fn travelled(&self) -> f32 {
        self.sum.map(|s| s.travelled).unwrap_or(0.0)
    }
    /// THE STATE OBJECTIVE's answer: the whole state at the tick that scored
    /// best inside the gate, or `None` if the run never qualified.
    ///
    /// `None` and `Some` are the two bands, and the caller cannot collapse
    /// them onto one number by accident because there is no number here to
    /// collapse: see `tmsearch::score::GateState`.
    pub fn gate(&self) -> Option<GateRecord> {
        match &self.sum {
            Some(s) if s.gate_tick >= 0 => Some(GateRecord {
                tick: s.gate_tick,
                key: s.gate_key,
                pos: s.gate_pos,
                vel: s.gate_vel,
                quat: s.gate_quat,
            }),
            _ => None,
        }
    }
    /// What the event clause saw. `armed` is the driver's own knowledge: a
    /// silent clause and no clause at all look identical in the summary and
    /// mean opposite things to the ranking.
    /// `after_key_armed` says whether `--after-key` was given, and it is
    /// load-bearing rather than cosmetic.
    ///
    /// `after_tick < 0` conflates two situations the ranking must treat as
    /// opposites: **no after-key was asked for** (every candidate is equal
    /// there, and 0 is the right flat value), and **an after-key was asked for
    /// and the window was EMPTY** because the run ended on the firing tick.
    /// Reporting 0 for the second is a decoy the instrument builds: a
    /// documented after-key like `-dist(x,y,z)` is never positive, so **0 is
    /// the best score obtainable**, and firing on the very last tick of the run
    /// beats every candidate that fired earlier and then actually did the
    /// thing. Measured on 267460, with a watchdog that aborts every candidate
    /// at a fixed tick: the search climbed to `GATE FIRED, after +0.0000`
    /// whose firing tick WAS the abort tick, and stopped improving -- it had
    /// found the empty window and there is nothing above it.
    ///
    /// An empty window is therefore the WORST measured value, not the best.
    pub fn event(&self, armed: bool, after_key_armed: bool) -> crate::EventSeen {
        if !armed {
            return crate::EventSeen::Unarmed;
        }
        match &self.sum {
            Some(s) if s.fire_tick >= 0 => crate::EventSeen::Fired {
                tick: s.fire_tick,
                value: s.fire_value,
                pos: s.fire_pos,
                after: match (s.after_tick >= 0, after_key_armed) {
                    (true, _) => s.after_key,
                    (false, true) => f32::NEG_INFINITY,
                    (false, false) => 0.0,
                },
                after_tick: s.after_tick,
            },
            _ => crate::EventSeen::Silent,
        }
    }
    /// Closest approach to the gate box in metres, for a run that never got
    /// inside. `None` when nothing was measured at all (no summary came back).
    pub fn gate_miss(&self) -> Option<f32> {
        match &self.sum {
            Some(s) if s.gate_tick < 0 && s.gate_miss.is_finite() => Some(s.gate_miss),
            _ => None,
        }
    }
    pub fn last_tick(&self) -> i32 {
        self.sum.map(|s| s.last_tick).unwrap_or(-1)
    }
    /// Continuous arrival time at the armed timing plane, in tape ticks
    /// (fractional). `None` when no plane was armed or the run never crossed
    /// it. Multiply by 10 and add the tape's `start_offset_ms` for race ms.
    pub fn cross(&self) -> Option<f64> {
        match &self.sum {
            Some(s) if s.cross_tick >= 0 => Some(s.cross_tick as f64 + s.cross_frac as f64),
            _ => None,
        }
    }
}

/// Turn the two frames of a `W` reply into an outcome.
pub fn outcome(json: &str, blob: &[u8]) -> Outcome {
    let (time, cps) = crate::forksrv::parse_result(json);
    Outcome {
        time,
        cps,
        sum: Summary::decode(blob),
    }
}

impl RefLineData {
    /// Read the first four columns (`time_ms,x,y,z`) of a trajectory CSV --
    /// the format `fk btraj` and `tmtraj decode --csv` both write -- and index
    /// it by tape tick. `fk btraj` emits one row per 10 ms tick, so this is a
    /// re-index rather than a resample; ticks before the first row (the
    /// standing start) clamp to it, and interior holes interpolate.
    pub fn from_csv(
        path: &str,
        start_offset_ms: i32,
        nticks: usize,
    ) -> Result<RefLineData, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
        let mut rows: Vec<(i64, f64, f64, f64)> = Vec::new();
        for line in text.lines().skip(1) {
            let f: Vec<&str> = line.trim().split(',').collect();
            if f.len() < 4 {
                continue;
            }
            if let (Ok(a), Ok(b), Ok(c), Ok(d)) = (
                f[0].parse::<i64>(),
                f[1].parse::<f64>(),
                f[2].parse::<f64>(),
                f[3].parse::<f64>(),
            ) {
                rows.push((a, b, c, d));
            }
        }
        Self::from_samples(&rows, start_offset_ms, nticks).map_err(|e| format!("{}: {}", path, e))
    }

    /// The same thing from samples already in hand: `(race_ms, x, y, z)`,
    /// resampled onto tape ticks and linearly interpolated across the gaps.
    ///
    /// Interpolation is not a nicety: telemetry is on a 50 ms grid and ticks
    /// are 10 ms, so at 100 km/h the live position is up to 0.7 m from the
    /// nearest recorded sample.
    ///
    /// # `nticks` is the tape's length and it must NOT decide the line's
    ///
    /// This used to drop every reference sample at or past `nticks`, so **the
    /// reference line stopped where the CANDIDATE TAPE stopped** -- and a
    /// candidate tape is routinely shorter than the reference, because a graft
    /// that reaches the same place sooner is shorter by exactly the time it
    /// saved. The line then ends short of the finish, its last index is
    /// reachable without crossing anything, and the arclength ladder tops out
    /// at 100% on a tape that cannot finish. Measured on 267460: the incumbent
    /// line is 560 m, and grafted tapes of 2131 and 2129 ticks got lines of
    /// 471 m and 470 m -- the length tracking the tape's tick count, which is
    /// the fingerprint. Two searches then spent 500 000 evaluations sitting at
    /// "100%" 63 m short of the flag.
    ///
    /// So the grid is long enough for every sample the tape's clock can
    /// address, whatever the tape's own length, and how far it runs past the
    /// tape is reported rather than silently applied.
    pub fn from_samples(
        rows: &[(i64, f64, f64, f64)],
        start_offset_ms: i32,
        nticks: usize,
    ) -> Result<RefLineData, String> {
        let idx = |ms: i64| (ms - start_offset_ms as i64) / 10;
        let need = rows
            .iter()
            .map(|&(ms, ..)| idx(ms))
            .filter(|&t| t >= 0)
            .max()
            .map(|t| t as usize + 1)
            .unwrap_or(0);
        let grid = nticks.max(need);
        let mut pts: Vec<Option<[f64; 3]>> = vec![None; grid];
        let mut nrow = 0;
        let (mut first_ms, mut last_ms) = (i64::MAX, i64::MIN);
        for &(ms, x, y, z) in rows {
            let t = idx(ms);
            if t >= 0 && (t as usize) < grid {
                pts[t as usize] = Some([x, y, z]);
                nrow += 1;
                first_ms = first_ms.min(ms);
                last_ms = last_ms.max(ms);
            }
        }
        if nrow < 10 {
            return Err(format!("only {} usable rows", nrow));
        }
        let first = pts.iter().position(|p| p.is_some()).unwrap();
        let last = pts.iter().rposition(|p| p.is_some()).unwrap();
        for i in 0..first {
            pts[i] = pts[first];
        }
        for i in last + 1..grid {
            pts[i] = pts[last];
        }
        let mut i = first;
        while i <= last {
            if pts[i].is_some() {
                i += 1;
                continue;
            }
            let a = i - 1;
            let mut b = i;
            while pts[b].is_none() {
                b += 1;
            }
            let (pa, pb) = (pts[a].unwrap(), pts[b].unwrap());
            for k in a + 1..b {
                let u = (k - a) as f64 / (b - a) as f64;
                pts[k] = Some([
                    pa[0] + u * (pb[0] - pa[0]),
                    pa[1] + u * (pb[1] - pa[1]),
                    pa[2] + u * (pb[2] - pa[2]),
                ]);
            }
            i = b + 1;
        }
        let flat: Vec<[f64; 3]> = pts.into_iter().map(|p| p.unwrap()).collect();
        let mut line = RefLineData::from_points(&flat);
        line.first_ms = first_ms;
        line.last_ms = last_ms;
        line.past_tape = grid.saturating_sub(nticks);
        Ok(line)
    }
}

/// How many instructions a compiled program actually uses.
pub fn prog_len(prog: &[KeyOp; MAXKOPS]) -> usize {
    prog.iter().position(|k| k.op == crate::pred_core::KOP_END).unwrap_or(MAXKOPS)
}
