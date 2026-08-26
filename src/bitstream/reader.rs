//! Zero-Copy High-Performance Bitstream Reader
//!
//! Provides bounds-checked, zero-copy, branchless bit extraction from byte slices
//! with support for arbitrary bit widths (1..64), fast 64-bit word peeking, and byte alignment.

use crate::error::{BitstreamError, Result};

/// Zero-copy bitstream reader operating directly over a contiguous byte slice.
#[derive(Debug, Clone)]
pub struct BitReader<'a> {
    buffer: &'a [u8],
    bit_pos: usize,
    total_bits: usize,
}

impl<'a> BitReader<'a> {
    /// Create a new `BitReader` over a byte slice.
    pub fn new(buffer: &'a [u8]) -> Self {
        let total_bits = buffer.len().saturating_mul(8);
        Self {
            buffer,
            bit_pos: 0,
            total_bits,
        }
    }

    /// Number of unread bits remaining in the bitstream.
    #[inline(always)]
    pub fn bits_remaining(&self) -> usize {
        self.total_bits.saturating_sub(self.bit_pos)
    }

    /// Current bit read position (0-indexed).
    #[inline(always)]
    pub fn bit_position(&self) -> usize {
        self.bit_pos
    }

    /// Total number of bytes in the underlying buffer.
    #[inline(always)]
    pub fn total_bytes(&self) -> usize {
        self.buffer.len()
    }

    /// Current byte position (rounded up to the next byte boundary).
    #[inline(always)]
    pub fn byte_position(&self) -> usize {
        self.bit_pos.div_ceil(8)
    }

    /// Number of fully consumed whole bytes.
    #[inline(always)]
    pub fn consumed_bytes(&self) -> usize {
        self.bit_pos / 8
    }

    /// Whether the current read position is byte-aligned.
    #[inline(always)]
    pub fn is_byte_aligned(&self) -> bool {
        self.bit_pos.is_multiple_of(8)
    }

    /// Align reader to the next byte boundary, discarding any remaining sub-byte bits.
    #[inline]
    pub fn byte_align(&mut self) {
        let rem = self.bit_pos % 8;
        if rem != 0 {
            self.bit_pos += 8 - rem;
        }
    }

    /// Fast branchless bit peeker extracting up to 64 bits without advancing position.
    #[inline]
    pub fn peek_bits(&self, n: usize) -> Result<u64> {
        if n == 0 {
            return Ok(0);
        }
        if n > 64 {
            return Err(BitstreamError::InvalidBitCount(n).into());
        }
        let available = self.bits_remaining();
        if available < n {
            return Err(BitstreamError::UnexpectedEof {
                needed_bits: n,
                available_bits: available,
            }
            .into());
        }

        let byte_idx = self.bit_pos / 8;
        let bit_in_byte = self.bit_pos % 8;

        // Fast path: 8-byte word available in buffer
        if byte_idx + 8 <= self.buffer.len() {
            let chunk = u64::from_be_bytes(self.buffer[byte_idx..byte_idx + 8].try_into().unwrap());
            let val = (chunk << bit_in_byte) >> (64 - n);
            return Ok(val);
        }

        // Tail path: buffer edge case
        let mut val: u64 = 0;
        let mut curr_pos = self.bit_pos;
        let mut bits_left = n;

        while bits_left > 0 {
            let b_idx = curr_pos / 8;
            let b_bit = curr_pos % 8;
            let take = (8 - b_bit).min(bits_left);
            let byte = self.buffer[b_idx];
            let mask = ((1u16 << take) - 1) as u8;
            let shift = 8 - b_bit - take;
            let extracted = ((byte >> shift) & mask) as u64;

            val = (val << take) | extracted;
            curr_pos += take;
            bits_left -= take;
        }

        Ok(val)
    }

    /// Peek the next 32 bits, left-aligned, zero-padding past the end of the buffer.
    ///
    /// Huffman decoding needs a fixed-width lookahead window and only consumes the
    /// bits the decoded codeword actually occupies, so reading past the end is
    /// normal on the final codeword of a frame. Padding with zeros (rather than
    /// erroring) matches the reference decoder, which reads from a buffer whose
    /// tail is zero-filled.
    #[inline(always)]
    pub fn peek32_padded(&self) -> u32 {
        let byte_idx = self.bit_pos / 8;
        let bit_in_byte = (self.bit_pos % 8) as u32;

        // Fast path: eight bytes in hand covers any 32-bit window at any bit offset.
        if byte_idx + 8 <= self.buffer.len() {
            let chunk = u64::from_be_bytes(
                self.buffer[byte_idx..byte_idx + 8].try_into().unwrap(),
            );
            return ((chunk << bit_in_byte) >> 32) as u32;
        }

        // Tail path: assemble from whatever bytes remain, zero-padding the rest.
        let mut chunk: u64 = 0;
        for i in 0..8 {
            let b = self.buffer.get(byte_idx + i).copied().unwrap_or(0);
            chunk = (chunk << 8) | b as u64;
        }
        ((chunk << bit_in_byte) >> 32) as u32
    }

    /// Read up to 64 bits from the bitstream and advance the position.
    #[inline(always)]
    pub fn read_bits(&mut self, n: usize) -> Result<u64> {
        let val = self.peek_bits(n)?;
        self.bit_pos += n;
        Ok(val)
    }

    /// Read a single boolean bit (`true` if 1, `false` if 0).
    #[inline(always)]
    pub fn read_bit(&mut self) -> Result<bool> {
        let bit_idx = self.bit_pos;
        if bit_idx >= self.total_bits {
            return Err(BitstreamError::UnexpectedEof {
                needed_bits: 1,
                available_bits: 0,
            }
            .into());
        }
        let byte = self.buffer[bit_idx / 8];
        let shift = 7 - (bit_idx % 8);
        self.bit_pos += 1;
        Ok(((byte >> shift) & 1) == 1)
    }

    /// Read an unsigned integer up to 8 bits.
    #[inline(always)]
    pub fn read_u8(&mut self, n: usize) -> Result<u8> {
        assert!(n <= 8, "Bit width must be <= 8");
        Ok(self.read_bits(n)? as u8)
    }

    /// Read an unsigned integer up to 16 bits.
    #[inline(always)]
    pub fn read_u16(&mut self, n: usize) -> Result<u16> {
        assert!(n <= 16, "Bit width must be <= 16");
        Ok(self.read_bits(n)? as u16)
    }

    /// Read an unsigned integer up to 32 bits.
    #[inline(always)]
    pub fn read_u32(&mut self, n: usize) -> Result<u32> {
        assert!(n <= 32, "Bit width must be <= 32");
        Ok(self.read_bits(n)? as u32)
    }

    /// Skip `n` bits forward in the stream.
    #[inline(always)]
    pub fn skip_bits(&mut self, n: usize) -> Result<()> {
        let available = self.bits_remaining();
        if available < n {
            return Err(BitstreamError::UnexpectedEof {
                needed_bits: n,
                available_bits: available,
            }
            .into());
        }
        self.bit_pos += n;
        Ok(())
    }

    /// Get slice of remaining unread bytes starting from the current byte boundary.
    pub fn get_remaining_bytes(&self) -> &'a [u8] {
        let byte_pos = self.byte_position();
        if byte_pos >= self.buffer.len() {
            &[]
        } else {
            &self.buffer[byte_pos..]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_reader_basic() {
        let data = [0b10110010, 0b11110000, 0b10101010, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06];
        let mut reader = BitReader::new(&data);

        assert_eq!(reader.bits_remaining(), data.len() * 8);
        assert!(reader.read_bit().unwrap());
        assert!(!reader.read_bit().unwrap());
        assert!(reader.read_bit().unwrap());
        assert!(reader.read_bit().unwrap());

        assert_eq!(reader.read_u8(4).unwrap(), 0b0010);
        assert_eq!(reader.read_u8(8).unwrap(), 0b11110000);
        assert_eq!(reader.read_u16(16).unwrap(), 0b1010101000000001);
    }

    #[test]
    fn test_bit_reader_eof() {
        let data = [0xFF];
        let mut reader = BitReader::new(&data);
        assert!(reader.read_bits(8).is_ok());
        assert!(reader.read_bit().is_err());
    }
}
