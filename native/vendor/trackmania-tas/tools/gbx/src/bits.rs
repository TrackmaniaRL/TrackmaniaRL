//! LSB-first bit reader/writer, matching the TM2020 ghost input bitstream.

pub struct BitReader<'a> {
    pub d: &'a [u8],
    pub pos: usize,
    /// Set when a read ran past the end of the data. See `bits()`.
    overran: bool,
}

impl<'a> BitReader<'a> {
    pub fn new(d: &'a [u8]) -> Self {
        BitReader { d, pos: 0, overran: false }
    }

    /// True if any read ran past the end of the bitstream, i.e. the data was
    /// shorter than its own header claimed. Zeros were returned for the
    /// missing bits.
    pub fn ended_early(&self) -> bool {
        self.overran
    }

    /// Read `n` bits. **Past the end returns zeros rather than panicking.**
    ///
    /// This asserted `byi < self.d.len()`, so a single flipped bit in a
    /// packet-count or length field walked the reader off the end and took the
    /// process down (exit 101) — found by flipping one bit at offset 20000 of
    /// a real ghost. A corrupt tape is *input*, and input must be judged, not
    /// fatal: the caller compares what it decoded against the record and
    /// refuses honestly.
    ///
    /// `ended_early()` reports whether this happened, so a caller that wants
    /// to distinguish "short tape" from "tape of zeros" still can.
    pub fn bits(&mut self, n: usize) -> u64 {
        let mut v: u64 = 0;
        let mut shift = 0;
        let mut left = n;
        while left > 0 {
            let byi = self.pos >> 3;
            if byi >= self.d.len() {
                self.overran = true;
                return v;
            }
            let bit = self.pos & 7;
            let take = left.min(8 - bit);
            let chunk = ((self.d[byi] >> bit) as u64) & ((1u64 << take) - 1);
            v |= chunk << shift;
            self.pos += take;
            shift += take;
            left -= take;
        }
        v
    }

    pub fn bit(&mut self) -> u64 {
        self.bits(1)
    }
}

#[derive(Default)]
pub struct BitWriter {
    pub buf: Vec<u8>,
    pub pos: usize,
}

impl BitWriter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn bits(&mut self, v: u64, n: usize) {
        let mut v = if n >= 64 { v } else { v & ((1u64 << n) - 1) };
        let mut left = n;
        while left > 0 {
            let byi = self.pos >> 3;
            if byi >= self.buf.len() {
                self.buf.push(0);
            }
            let bit = self.pos & 7;
            let put = left.min(8 - bit);
            let mask = (1u64 << put) - 1;
            self.buf[byi] |= (((v & mask) << bit) & 0xFF) as u8;
            v >>= put;
            self.pos += put;
            left -= put;
        }
    }

    pub fn bit(&mut self, v: u64) {
        self.bits(v, 1);
    }
}

