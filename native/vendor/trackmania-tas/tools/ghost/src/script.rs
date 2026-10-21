//! Build an input tape from an event script.
//!
//! A tool-assisted run is written, and read, as a list of events: at this race
//! time, press this key; at that one, release it. `vidread ktevents` recovers
//! exactly that shape off a video, and this is the other end of the pipe — it
//! turns such a list into a tape the oracle can simulate.
//!
//! The script is line-based and ignores blank lines and `#` comments:
//!
//! ```text
//! 0     press gas
//! 1230  press left
//! 1310  release left
//! 2400  press brake
//! ```
//!
//! Keys are `gas`, `brake`, `left`, `right`. Anything `vidread` prints that is
//! not an event (`… record starts`, `… gap until …`) is skipped, so a recovered
//! record can be fed in unedited — but a gap is a hole in the OBSERVATION, and
//! this fills it by holding the last state, which is an assumption and not a
//! reading. Where that matters, split the script at the gap.

/// Steering is digital here: full lock or nothing, which is what a keyboard
/// TAS produces. The magnitude is the tape's own full-lock value.
const FULL: i32 = 127;

#[derive(Clone, Copy, Default, PartialEq)]
pub struct Keys {
    pub gas: bool,
    pub brake: bool,
    pub left: bool,
    pub right: bool,
}

impl Keys {
    pub fn steer(&self) -> i32 {
        match (self.left, self.right) {
            (true, false) => -FULL,
            (false, true) => FULL,
            _ => 0,
        }
    }
}

pub struct Event {
    pub race_ms: i64,
    pub press: bool,
    pub key: String,
}

pub fn parse_events(txt: &str) -> Result<Vec<Event>, String> {
    let mut v = Vec::new();
    for (n, line) in txt.lines().enumerate() {
        let line = line.split('#').next().unwrap().trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 3 {
            continue; // "… record starts", "… gap until …" and friends
        }
        let press = match f[1] {
            "press" => true,
            "release" => false,
            _ => continue,
        };
        let key = match f[2] {
            "gas" | "up" => "gas",
            "brake" | "brake2" | "down" => "brake",
            "left" => "left",
            "right" => "right",
            other => return Err(format!("line {}: unknown key {other}", n + 1)),
        };
        let race_ms: i64 =
            f[0].parse().map_err(|_| format!("line {}: {} is not a race time", n + 1, f[0]))?;
        v.push(Event { race_ms, press, key: key.into() });
    }
    v.sort_by_key(|e| e.race_ms);
    Ok(v)
}

/// Rewrite every tick of `base` (the text of a `ghost tape extract`) at or
/// after race time 0 to the state the script implies. Ticks before race 0 are
/// left exactly as they are: the pre-start packets are the container's own and
/// nothing in a script is about them.
pub fn apply_from(base: &str, events: &[Event], keep_before: bool) -> Result<String, String> {
    let start_offset: i64 = base
        .lines()
        .find(|l| l.starts_with("@archive "))
        .and_then(|l| {
            l.split_whitespace()
                .find_map(|w| w.strip_prefix("start_offset_ms="))
                .and_then(|v| v.parse().ok())
        })
        .ok_or("no @archive line with start_offset_ms")?;

    let first_ms = if keep_before { events.first().map(|e| e.race_ms).unwrap_or(0) } else { 0 };
    let mut out = String::with_capacity(base.len());
    let mut keys = Keys::default();
    let mut ei = 0usize;
    for line in base.lines() {
        if !line.starts_with("t=") {
            out.push_str(line);
            out.push('\n');
            continue;
        }
        let tick: i64 = line[2..]
            .split_whitespace()
            .next()
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| format!("bad tick line: {line}"))?;
        let race = tick * 10 + start_offset;
        if race < first_ms || race < 0 || !line.contains("mode=2") {
            out.push_str(line);
            out.push('\n');
            continue;
        }
        while ei < events.len() && events[ei].race_ms <= race {
            let e = &events[ei];
            let f = match e.key.as_str() {
                "gas" => &mut keys.gas,
                "brake" => &mut keys.brake,
                "left" => &mut keys.left,
                _ => &mut keys.right,
            };
            *f = e.press;
            ei += 1;
        }
        let mut s = String::new();
        for w in line.split_whitespace() {
            let rep = match w.split_once('=') {
                Some(("steer", _)) => format!("steer={}", keys.steer()),
                Some(("accel", _)) => format!("accel={}", keys.gas as u8),
                Some(("brake", _)) => format!("brake={}", keys.brake as u8),
                Some(("vsame", _)) => "vsame=0".to_string(),
                _ => w.to_string(),
            };
            if !s.is_empty() {
                s.push(' ');
            }
            s.push_str(&rep);
        }
        out.push_str(&s);
        out.push('\n');
    }
    if ei < events.len() {
        return Err(format!(
            "{} events are past the end of the tape (first at race {} ms)",
            events.len() - ei,
            events[ei].race_ms
        ));
    }
    Ok(out)
}

/// Write a 40-tick steer signature into the tape at `race_ms`.
///
/// `fk`'s input-array locator keys on the most distinctive window of 24
/// consecutive steer values. A scripted tape steers with three values and
/// holds each for a long time, so it has no distinctive window at all and the
/// locator lands on the wrong array — which shows up as `TAPE MISMATCH` on
/// every tick, including ticks the script never touched. Forty pseudorandom
/// ticks placed PAST the finish give it something to key on and change no
/// physics that anyone is measuring.
pub fn signature(base: &str, race_ms: i64) -> Result<String, String> {
    let start_offset: i64 = base
        .lines()
        .find(|l| l.starts_with("@archive "))
        .and_then(|l| {
            l.split_whitespace()
                .find_map(|w| w.strip_prefix("start_offset_ms="))
                .and_then(|v| v.parse().ok())
        })
        .ok_or("no @archive line with start_offset_ms")?;
    let first = (race_ms - start_offset) / 10;
    let mut out = String::with_capacity(base.len());
    let mut written = 0;
    for line in base.lines() {
        if !line.starts_with("t=") {
            out.push_str(line);
            out.push('\n');
            continue;
        }
        let tick: i64 = line[2..].split_whitespace().next().and_then(|v| v.parse().ok()).unwrap_or(-1);
        let k = tick - first;
        if !(0..40).contains(&k) || !line.contains("mode=2") {
            out.push_str(line);
            out.push('\n');
            continue;
        }
        // A fixed, arbitrary, high-entropy pattern; the values matter only in
        // that no 24 of them repeat anywhere else in the tape.
        let v = (((k * 37 + 11) % 61) - 30) as i32 * 4;
        let mut s = String::new();
        for w in line.split_whitespace() {
            let rep = match w.split_once('=') {
                Some(("steer", _)) => format!("steer={}", v.clamp(-127, 127)),
                Some(("vsame", _)) => "vsame=0".to_string(),
                _ => w.to_string(),
            };
            if !s.is_empty() {
                s.push(' ');
            }
            s.push_str(&rep);
        }
        out.push_str(&s);
        out.push('\n');
        written += 1;
    }
    if written != 40 {
        return Err(format!("signature at race {race_ms} ms lands on {written} ticks, not 40"));
    }
    Ok(out)
}
