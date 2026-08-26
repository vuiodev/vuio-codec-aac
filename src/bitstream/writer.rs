//! High-Performance Dynamic Bitstream Writer
//!
//! Provides fast bit packing, alignment byte padding, and buffer serialization.

/// Dynamic bitstream writer for accumulating arbitrary bit sequences into byte buffers.
#[derive(Debug, Clone, Default)]
pub struct BitWriter {
    buffer: Vec<u8>,
    accumulator: u64,
    bits_in_acc: usize,
    total_bits: usize,
}

impl BitWriter {
    /// Create a new empty `BitWriter`.
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(1024),
            accumulator: 0,
            bits_in_acc: 0,
            total_bits: 0,
        }
    }

    /// Create a new `BitWriter` with pre-allocated capacity in bytes.
    pub fn with_capacity(capacity_bytes: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity_bytes),
            accumulator: 0,
            bits_in_acc: 0,
            total_bits: 0,
        }
    }

    /// Total number of bits written to the stream so far.
    #[inline(always)]
    pub const fn bits_written(&self) -> usize {
        self.total_bits
    }

    /// Total number of full bytes written to the underlying buffer.
    #[inline(always)]
    pub fn bytes_written(&self) -> usize {
        self.total_bits.div_ceil(8)
    }

    /// Whether the writer is currently byte-aligned.
    #[inline(always)]
    pub const fn is_byte_aligned(&self) -> bool {
        self.total_bits.is_multiple_of(8)
    }

    /// Write up to 64 bits to the bitstream.
    pub fn write_bits(&mut self, val: u64, n: usize) {
        if n == 0 {
            return;
        }
        assert!(n <= 64, "Cannot write more than 64 bits at once");
        let mask = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
        let masked_val = val & mask;

        self.total_bits += n;

        // If fits in accumulator
        if self.bits_in_acc + n < 64 {
            self.accumulator = (self.accumulator << n) | masked_val;
            self.bits_in_acc += n;
            self.flush_full_bytes();
        } else {
            let space_left = 64 - self.bits_in_acc;
            let first_part = masked_val >> (n - space_left);
            self.accumulator = (self.accumulator << space_left) | first_part;
            self.bits_in_acc = 64;
            self.flush_full_bytes();

            let remaining_bits = n - space_left;
            let rem_mask = (1u64 << remaining_bits) - 1;
            self.accumulator = masked_val & rem_mask;
            self.bits_in_acc = remaining_bits;
            self.flush_full_bytes();
        }
    }

    /// Write a single boolean bit (`1` if true, `0` if false).
    #[inline(always)]
    pub fn write_bit(&mut self, bit: bool) {
        self.write_bits(if bit { 1 } else { 0 }, 1);
    }

    /// Write an unsigned integer up to 8 bits.
    #[inline(always)]
    pub fn write_u8(&mut self, val: u8, n: usize) {
        assert!(n <= 8, "Bit width must be <= 8");
        self.write_bits(val as u64, n);
    }

    /// Write an unsigned integer up to 16 bits.
    #[inline(always)]
    pub fn write_u16(&mut self, val: u16, n: usize) {
        assert!(n <= 16, "Bit width must be <= 16");
        self.write_bits(val as u64, n);
    }

    /// Write an unsigned integer up to 32 bits.
    #[inline(always)]
    pub fn write_u32(&mut self, val: u32, n: usize) {
        assert!(n <= 32, "Bit width must be <= 32");
        self.write_bits(val as u64, n);
    }

    /// Write an aligned slice of bytes directly to the bitstream.
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u8(b, 8);
        }
    }

    /// Align to the next byte boundary by padding with zeros.
    pub fn byte_align_zero(&mut self) {
        let rem = self.total_bits % 8;
        if rem != 0 {
            self.write_bits(0, 8 - rem);
        }
    }

    /// Align to the next byte boundary by padding with ones.
    pub fn byte_align_one(&mut self) {
        let rem = self.total_bits % 8;
        if rem != 0 {
            let padding_len = 8 - rem;
            let ones = (1u64 << padding_len) - 1;
            self.write_bits(ones, padding_len);
        }
    }

    /// Flush completed full 8-bit bytes from accumulator to output buffer.
    #[inline]
    fn flush_full_bytes(&mut self) {
        while self.bits_in_acc >= 8 {
            let shift = self.bits_in_acc - 8;
            let byte = ((self.accumulator >> shift) & 0xFF) as u8;
            self.buffer.push(byte);
            self.bits_in_acc -= 8;
            self.accumulator &= if self.bits_in_acc == 0 {
                0
            } else {
                (1u64 << self.bits_in_acc) - 1
            };
        }
    }

    /// Finalize writer and flush any sub-byte bits (padded with zeros) to return the byte slice.
    pub fn finalize(&mut self) -> &[u8] {
        self.byte_align_zero();
        &self.buffer
    }

    /// Consume writer and return the completed byte vector.
    pub fn into_bytes(mut self) -> Vec<u8> {
        self.byte_align_zero();
        self.buffer
    }

    /// Get reference to currently written byte buffer.
    pub fn as_bytes(&self) -> &[u8] {
        &self.buffer
    }

    /// Reset writer to empty state without deallocating buffer.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.accumulator = 0;
        self.bits_in_acc = 0;
        self.total_bits = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_writer_basic() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b101, 3);
        writer.write_bits(0b10010, 5);
        writer.write_u8(0xFF, 8);

        let bytes = writer.finalize();
        assert_eq!(bytes, &[0b1011_0010, 0xFF]);
    }
}
