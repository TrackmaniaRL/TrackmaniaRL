//! A very small JSON reader plus the two float formatters the port needs to be
//! byte-compatible with the Python it replaces.
//!
//! Why the formatters matter: the reference artefacts in `/tmp/entrec` were
//! written by `json.dump` (CPython `repr` of a float) and by
//! `csv.writer` fed with `"%.6g" % value`. Reproducing those files exactly is
//! the strongest available check that the decode is bit-identical, so both
//! formats are re-implemented here rather than approximated.
//!
//! `parse` also accepts the bare `NaN`, `Infinity` and `-Infinity` tokens that
//! `json.dump` emits (and that `JSON.parse` rejects) -- none of the 51
//! reference files actually contain one, but a decoder that met a non-finite
//! sample would have produced them.

use std::collections::BTreeMap;
use std::fmt::Write as _;

#[derive(Debug, Clone, PartialEq)]
pub enum J {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
}

impl J {
    pub fn get(&self, key: &str) -> Option<&J> {
        match self {
            J::Obj(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn num(&self) -> f64 {
        match self {
            J::Num(v) => *v,
            _ => panic!("not a number: {:?}", self),
        }
    }
    pub fn int(&self) -> i64 {
        self.num() as i64
    }
    pub fn str(&self) -> &str {
        match self {
            J::Str(s) => s,
            _ => panic!("not a string: {:?}", self),
        }
    }
    pub fn arr(&self) -> &[J] {
        match self {
            J::Arr(a) => a,
            _ => panic!("not an array: {:?}", self),
        }
    }
    /// Convenience for the `{"a": {...}}` maps in the reference cluster dump.
    pub fn obj_map(&self) -> BTreeMap<&str, &J> {
        match self {
            J::Obj(kv) => kv.iter().map(|(k, v)| (k.as_str(), v)).collect(),
            _ => panic!("not an object"),
        }
    }
}

pub fn parse(s: &str) -> Result<J, String> {
    let b = s.as_bytes();
    let mut p = 0usize;
    let v = parse_value(b, &mut p)?;
    skip_ws(b, &mut p);
    if p != b.len() {
        return Err(format!("trailing junk at byte {}", p));
    }
    Ok(v)
}

fn skip_ws(b: &[u8], p: &mut usize) {
    while *p < b.len() && matches!(b[*p], b' ' | b'\t' | b'\n' | b'\r') {
        *p += 1;
    }
}

fn lit(b: &[u8], p: &mut usize, word: &str) -> bool {
    if b[*p..].starts_with(word.as_bytes()) {
        *p += word.len();
        true
    } else {
        false
    }
}

fn parse_value(b: &[u8], p: &mut usize) -> Result<J, String> {
    skip_ws(b, p);
    if *p >= b.len() {
        return Err("unexpected end".into());
    }
    match b[*p] {
        b'{' => {
            *p += 1;
            let mut out = Vec::new();
            skip_ws(b, p);
            if b[*p] == b'}' {
                *p += 1;
                return Ok(J::Obj(out));
            }
            loop {
                skip_ws(b, p);
                let k = match parse_value(b, p)? {
                    J::Str(s) => s,
                    other => return Err(format!("object key not a string: {:?}", other)),
                };
                skip_ws(b, p);
                if b[*p] != b':' {
                    return Err(format!("expected ':' at {}", p));
                }
                *p += 1;
                let v = parse_value(b, p)?;
                out.push((k, v));
                skip_ws(b, p);
                match b[*p] {
                    b',' => *p += 1,
                    b'}' => {
                        *p += 1;
                        return Ok(J::Obj(out));
                    }
                    c => return Err(format!("expected ',' or '}}' got {}", c as char)),
                }
            }
        }
        b'[' => {
            *p += 1;
            let mut out = Vec::new();
            skip_ws(b, p);
            if b[*p] == b']' {
                *p += 1;
                return Ok(J::Arr(out));
            }
            loop {
                out.push(parse_value(b, p)?);
                skip_ws(b, p);
                match b[*p] {
                    b',' => *p += 1,
                    b']' => {
                        *p += 1;
                        return Ok(J::Arr(out));
                    }
                    c => return Err(format!("expected ',' or ']' got {}", c as char)),
                }
            }
        }
        b'"' => {
            *p += 1;
            let mut s = String::new();
            loop {
                let c = b[*p];
                *p += 1;
                match c {
                    b'"' => return Ok(J::Str(s)),
                    b'\\' => {
                        let e = b[*p];
                        *p += 1;
                        match e {
                            b'n' => s.push('\n'),
                            b't' => s.push('\t'),
                            b'r' => s.push('\r'),
                            b'b' => s.push('\u{8}'),
                            b'f' => s.push('\u{c}'),
                            b'u' => {
                                let h = std::str::from_utf8(&b[*p..*p + 4]).map_err(|e| e.to_string())?;
                                let cp = u32::from_str_radix(h, 16).map_err(|e| e.to_string())?;
                                *p += 4;
                                s.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                            }
                            other => s.push(other as char),
                        }
                    }
                    _ => {
                        // copy the whole UTF-8 sequence
                        let len = utf8_len(c);
                        s.push_str(std::str::from_utf8(&b[*p - 1..*p - 1 + len]).map_err(|e| e.to_string())?);
                        *p += len - 1;
                    }
                }
            }
        }
        b't' if lit(b, p, "true") => Ok(J::Bool(true)),
        b'f' if lit(b, p, "false") => Ok(J::Bool(false)),
        b'n' if lit(b, p, "null") => Ok(J::Null),
        b'N' if lit(b, p, "NaN") => Ok(J::Num(f64::NAN)),
        b'I' if lit(b, p, "Infinity") => Ok(J::Num(f64::INFINITY)),
        b'-' if b[*p..].starts_with(b"-Infinity") => {
            *p += 9;
            Ok(J::Num(f64::NEG_INFINITY))
        }
        _ => {
            let start = *p;
            if b[*p] == b'-' || b[*p] == b'+' {
                *p += 1;
            }
            while *p < b.len()
                && matches!(b[*p], b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-')
            {
                *p += 1;
            }
            std::str::from_utf8(&b[start..*p])
                .unwrap()
                .parse::<f64>()
                .map(J::Num)
                .map_err(|e| format!("bad number at {}: {}", start, e))
        }
    }
}

fn utf8_len(c: u8) -> usize {
    if c < 0x80 {
        1
    } else if c >> 5 == 0b110 {
        2
    } else if c >> 4 == 0b1110 {
        3
    } else {
        4
    }
}

// ---------------------------------------------------------------------------
// Python-compatible float formatting
// ---------------------------------------------------------------------------

/// `round(x, n)` exactly as CPython does it: correctly round the binary double
/// to `n` decimal places and convert that decimal string back to a double.
pub fn py_round(x: f64, n: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{:.*}", n, x).parse::<f64>().unwrap()
}

/// CPython's `repr(float)` -- which is what `json.dump` writes: the shortest
/// decimal string that round-trips, in fixed notation when the decimal point
/// lands in `(-4, 16]`, else exponential with a signed two-digit exponent, and
/// always with a `.0` on an integral value.
pub fn py_repr(v: f64) -> String {
    if v.is_nan() {
        return "NaN".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    py_repr_exp(&format!("{:e}", v))
}

/// The same rendering for an `f32`, WITHOUT widening it first.
///
/// `py_repr(v as f64)` would be wrong here, not merely wasteful: widening asks
/// for the shortest string that round-trips as an f64, and for a value that is
/// really an f32 that is the full seventeen-digit expansion --
/// `0.1f32 as f64` prints `0.10000000149011612`. Formatting the f32 directly
/// asks for the shortest string that round-trips as an f32, which is `0.1`.
/// Same number, and only this one is readable and stable.
pub fn py_repr_f32(v: f32) -> String {
    if v.is_nan() {
        return "NaN".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    py_repr_exp(&format!("{:e}", v))
}

/// Shared tail of `py_repr` / `py_repr_f32`: turn Rust's `{:e}` form (already
/// the shortest round-tripping mantissa for whichever width produced it, and
/// exactly the digit string CPython's dtoa mode 0 produces) into Python's
/// `repr` layout.
fn py_repr_exp(s: &str) -> String {
    let (mant, exp) = s.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let neg = mant.starts_with('-');
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    let digits = digits.trim_end_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };
    // decpt: value = 0.<digits> * 10^decpt
    let decpt = exp + 1;
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    if digits == "0" {
        out.push_str("0.0");
        return out;
    }
    if decpt <= -4 || decpt > 16 {
        out.push_str(&digits[0..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let e = decpt - 1;
        let _ = write!(out, "e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs());
    } else if decpt <= 0 {
        out.push_str("0.");
        for _ in 0..(-decpt) {
            out.push('0');
        }
        out.push_str(digits);
    } else if (decpt as usize) >= digits.len() {
        out.push_str(digits);
        for _ in 0..(decpt as usize - digits.len()) {
            out.push('0');
        }
        out.push_str(".0");
    } else {
        out.push_str(&digits[..decpt as usize]);
        out.push('.');
        out.push_str(&digits[decpt as usize..]);
    }
    out
}

/// C's `%.6g`, which is what `entrec.write_csv` uses for every float column.
pub fn fmt_g6(v: f64) -> String {
    fmt_g(v, 6)
}

pub fn fmt_g(v: f64, prec: usize) -> String {
    if v.is_nan() {
        return "nan".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let p = prec.max(1);
    if v == 0.0 {
        // preserves the sign of negative zero, like printf
        return if v.is_sign_negative() { "-0".into() } else { "0".into() };
    }
    // %e with p-1 fractional digits fixes both the rounded digits and the
    // exponent that decides which style %g uses.
    let sci = format!("{:.*e}", p - 1, v);
    let (mant, exp) = sci.split_once('e').unwrap();
    let x: i32 = exp.parse().unwrap();
    if x < -4 || x >= p as i32 {
        let mant = trim_zeros(mant);
        format!("{}e{}{:02}", mant, if x < 0 { '-' } else { '+' }, x.abs())
    } else {
        let dec = (p as i32 - 1 - x).max(0) as usize;
        trim_zeros(&format!("{:.*}", dec, v))
    }
}

fn trim_zeros(s: &str) -> String {
    if !s.contains('.') {
        return s.to_string();
    }
    let t = s.trim_end_matches('0');
    t.trim_end_matches('.').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repr_matches_cpython() {
        // left-hand side is literally what CPython's repr() prints
        for (want, v) in [
            ("1.0", 1.0f64),
            ("-0.0", -0.0f64),
            ("0.0", 0.0f64),
            ("1e-06", 1e-6f64),
            ("-1e-06", -1e-6f64),
            ("0.0001", 1e-4f64),
            ("1e-05", 1e-5f64),
            ("1584.0001", 1584.0001f64),
            ("18.002", 18.002f64),
            ("2.7758", 2.7758f64),
            ("1e+16", 1e16f64),
            ("1000000000000000.0", 1e15f64),
            ("-1.2345e-07", -1.2345e-7f64),
            ("3.141592653589793", std::f64::consts::PI),
        ] {
            assert_eq!(py_repr(v), want, "repr({})", v);
        }
        assert_eq!(py_repr(f64::NAN), "NaN");
        assert_eq!(py_repr(f64::INFINITY), "Infinity");
        assert_eq!(py_repr(f64::NEG_INFINITY), "-Infinity");
    }

    #[test]
    fn g6_matches_printf() {
        for (want, v) in [
            ("0", 0.0f64),
            ("-0", -0.0f64),
            ("1584", 1584.0f64),
            ("18.002", 18.002f64),
            ("0.810531", 0.8105307f64),
            ("0.225147", 0.22514742f64),
            ("0.00278466", 0.002784659f64),
            ("-0.00392157", -0.00392156862745098f64),
            ("-0.0305176", -0.030517578125f64),
            ("0.00784314", 0.00784313725490196f64),
            ("1.23457e-07", 1.234567e-7f64),
            ("1.23457e+08", 123456789.0f64),
            ("100000", 100000.0f64),
            ("1e+06", 1000000.0f64),
        ] {
            assert_eq!(fmt_g6(v), want, "%.6g of {}", v);
        }
    }

    #[test]
    fn round_matches_cpython() {
        assert_eq!(py_round(1584.00012345, 4), 1584.0001);
        assert_eq!(py_round(-1e-9, 6), -0.0);
        assert_eq!(py_repr(py_round(-1e-9, 6)), "-0.0");
        assert_eq!(py_round(0.5, 0), 0.0); // ties to even
        assert_eq!(py_round(1.5, 0), 2.0);
    }

    #[test]
    fn parses_python_json_extensions() {
        let v = parse("{\"a\": [NaN, Infinity, -Infinity, 1, -2.5e3], \"b\": \"x\\ty\"}").unwrap();
        let a = v.get("a").unwrap().arr();
        assert!(a[0].num().is_nan());
        assert_eq!(a[1].num(), f64::INFINITY);
        assert_eq!(a[2].num(), f64::NEG_INFINITY);
        assert_eq!(a[3].num(), 1.0);
        assert_eq!(a[4].num(), -2500.0);
        assert_eq!(v.get("b").unwrap().str(), "x\ty");
    }
}
