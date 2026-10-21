//! How this project prints a time.
//!
//! **Times are SECONDS with a decimal — `36.049`, never `36049`.** A standing
//! rule across the toolchain, because long raw-millisecond integers are hard to
//! read and easy to misread by a factor of ten. It applies to prose, tables,
//! anything published, and every line these tools print.
//!
//! Tick indices stay integers (a tick is a count, not a duration). Sub-
//! millisecond values keep their extra digits: a 6595.20 ms measurement is
//! `6.59520`, not `6.595`.

/// A race time in milliseconds, as seconds: `22730` -> `"22.730"`.
pub fn secs(ms: i64) -> String {
    let neg = ms < 0;
    let a = ms.unsigned_abs();
    format!("{}{}.{:03}", if neg { "-" } else { "" }, a / 1000, a % 1000)
}


/// A difference, which reads naturally with a sign: `-21` -> `"-0.021"`.
pub fn delta(ms: i64) -> String {
    format!("{}{}", if ms > 0 { "+" } else { "" }, secs(ms))
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seconds_with_a_decimal() {
        assert_eq!(secs(36049), "36.049");
        assert_eq!(secs(22730), "22.730");
        assert_eq!(secs(7241), "7.241");
        assert_eq!(secs(999), "0.999");
        assert_eq!(secs(1000), "1.000");
        assert_eq!(secs(0), "0.000");
        // a negative offset is a real value here: countdown-prefixed tapes
        // start at about -1.560 s, and printing that as -1560 is the bug.
        assert_eq!(secs(-1560), "-1.560");
        assert_eq!(delta(-21), "-0.021");
        assert_eq!(delta(21), "+0.021");


    }
}
