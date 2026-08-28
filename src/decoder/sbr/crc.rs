//! SBR's own CRC-10 (`sbr_crc_check()`, ISO/IEC 14496-3 clause 4.6.18.4.3).
//!
//! Ported from `c/libxaac/decoder/ixheaacd_sbr_crc.c`
//! (`ixheaacd_calc_chk_sum`, `ixheaacd_sbr_crc`, `ixheaacd_sbr_crccheck`).
//!
//! # Why this checks bits that are never consumed
//!
//! An SBR payload can optionally be preceded by a 10-bit checksum covering the
//! payload bits that follow it. Checking it must not disturb the real
//! bitstream position: the caller still needs to *parse* those same bits
//! afterward through the normal SBR data path, so the checksum is verified
//! against a disposable copy of the reader
//! ([`BitReader::clone`](crate::bitstream::BitReader)) and the real reader is
//! left positioned right after the 10-bit checksum field, exactly as
//! `ixheaacd_sbr_crccheck` copies `*it_bit_buff` into a scratch
//! `it_bit_buff_local` before reading through it.
//!
//! [`SbrDecoder::decode_extension`](crate::decoder::sbr::SbrDecoder::decode_extension)
//! calls [`sbr_crc_check`] whenever its `with_crc` flag is set (the fill
//! element's extension type distinguishes `SBR_DATA` from `SBR_DATA_CRC`,
//! decoded in [`crate::decoder::engine`]), and rejects the payload with a
//! [`crate::error::DecodeError::CorruptedFrame`] on a mismatch, exactly the
//! same as any other malformed SBR payload -- the frame's core signal still
//! decodes, just without that frame's band replication.

use crate::bitstream::BitReader;
use crate::error::Result;

/// CRC-10 polynomial SBR uses (`SBR_CRC_POLY`), applied MSB-first over one
/// 10-bit shift register (`crc_mask` tests bit 9).
const CRC_POLY: u16 = 0x0233;
/// Width of the transmitted checksum field (`SBR_CYC_REDCY_CHK_BITS`).
pub const CRC_BITS: usize = 10;

/// Feed `num_bits` bits of `data` (MSB-first, as read straight off the
/// bitstream) through the running CRC-10 register (`ixheaacd_calc_chk_sum`).
fn update(state: u16, data: u32, num_bits: u32) -> u16 {
    let mut state = state;
    for i in (0..num_bits).rev() {
        let data_bit = (data >> i) & 1 != 0;
        let reg_bit = (state & (1 << 9)) != 0;
        state <<= 1;
        if reg_bit ^ data_bit {
            state ^= CRC_POLY;
        }
    }
    state
}

/// Compute the CRC-10 over the next `num_crc_bits` bits of `reader`, byte at a
/// time with a final partial byte, exactly as `ixheaacd_sbr_crc` does
/// (`reader` is consumed; pass a clone to leave the original untouched).
pub(crate) fn checksum(reader: &mut BitReader, num_crc_bits: usize) -> Result<u16> {
    let mut state = 0u16;
    let full_bytes = num_crc_bits / 8;
    let rem_bits = num_crc_bits % 8;

    for _ in 0..full_bytes {
        state = update(state, reader.read_u32(8)?, 8);
    }
    if rem_bits > 0 {
        state = update(state, reader.read_u32(rem_bits)?, rem_bits as u32);
    }
    Ok(state & 0x03FF)
}

/// Verify an SBR payload's CRC. Reads the 10-bit transmitted checksum for
/// real, then checks it against `crc_bits_len` bits of whatever follows
/// *without* consuming them, so the caller can still parse that data
/// normally afterward.
///
/// `crc_bits_len` is the payload length the standard specifies for this SBR
/// header configuration; if fewer bits remain in the stream than that (a
/// truncated frame), only what remains is checked, matching the reference's
/// own clamp.
pub fn sbr_crc_check(reader: &mut BitReader, crc_bits_len: usize) -> Result<bool> {
    let transmitted = reader.read_u32(CRC_BITS)?;

    let available = reader.bits_remaining();
    if available == 0 {
        return Ok(false);
    }
    let num_crc_bits = crc_bits_len.min(available);

    let mut scratch = reader.clone();
    let computed = checksum(&mut scratch, num_crc_bits)?;

    Ok(computed as u32 == transmitted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::BitWriter;

    /// A checksum computed here and re-verified here must agree with itself,
    /// for both a byte-aligned and a non-byte-aligned payload length.
    #[test]
    fn a_freshly_computed_checksum_verifies_against_its_own_payload() {
        for payload_bits in [8usize, 13, 24, 37] {
            let mut w = BitWriter::new();
            // Placeholder for the checksum field, filled in below.
            w.write_bits(0, CRC_BITS);
            for i in 0..payload_bits {
                w.write_bit(i % 3 == 0);
            }
            let mut bytes = w.finalize().to_vec();

            // Compute the real checksum over the payload region and patch it
            // into the first 10 bits of the buffer.
            let mut payload_reader = BitReader::new(&bytes);
            payload_reader.skip_bits(CRC_BITS).unwrap();
            let crc = checksum(&mut payload_reader, payload_bits).unwrap();
            bytes[0] = (crc >> 2) as u8;
            bytes[1] = (bytes[1] & 0x3F) | (((crc & 0b11) as u8) << 6);

            let mut reader = BitReader::new(&bytes);
            assert!(
                sbr_crc_check(&mut reader, payload_bits).unwrap(),
                "payload_bits={payload_bits}: a self-computed checksum must verify"
            );
            // The real reader must be left right after the 10-bit field,
            // ready to parse the payload it just checked without consuming it.
            assert_eq!(reader.bit_position(), CRC_BITS);
        }
    }

    /// Flipping any single payload bit must be caught -- this is the whole
    /// point of a CRC, and a false-negative here would defeat it silently.
    #[test]
    fn a_single_corrupted_bit_is_detected() {
        let payload_bits = 32usize;
        let mut w = BitWriter::new();
        w.write_bits(0, CRC_BITS);
        for i in 0..payload_bits {
            w.write_bit((i * 7) % 5 < 2);
        }
        let mut bytes = w.finalize().to_vec();

        let mut payload_reader = BitReader::new(&bytes);
        payload_reader.skip_bits(CRC_BITS).unwrap();
        let crc = checksum(&mut payload_reader, payload_bits).unwrap();
        bytes[0] = (crc >> 2) as u8;
        bytes[1] = (bytes[1] & 0x3F) | (((crc & 0b11) as u8) << 6);

        // Flip one bit well inside the payload region (bit CRC_BITS + 5).
        let flip_bit = CRC_BITS + 5;
        bytes[flip_bit / 8] ^= 1 << (7 - flip_bit % 8);

        let mut reader = BitReader::new(&bytes);
        assert!(!sbr_crc_check(&mut reader, payload_bits).unwrap(), "a corrupted payload must fail its CRC");
    }

    /// Checking must not consume the payload bits it verifies -- a caller
    /// parsing SBR data right after this call must see it untouched.
    #[test]
    fn checking_does_not_consume_the_payload() {
        let mut w = BitWriter::new();
        w.write_bits(0, CRC_BITS);
        w.write_bits(0xABCDu64, 16);
        let bytes = w.finalize().to_vec();

        let mut reader = BitReader::new(&bytes);
        let _ = sbr_crc_check(&mut reader, 16).unwrap();
        assert_eq!(reader.read_u32(16).unwrap(), 0xABCD, "payload bits must still be there to read");
    }

    /// A truncated stream (fewer bits available than the declared payload
    /// length) must be handled by the documented clamp, not panic or error.
    #[test]
    fn a_truncated_stream_clamps_rather_than_panicking() {
        let mut w = BitWriter::new();
        w.write_bits(0, CRC_BITS);
        w.write_bits(0b101, 3);
        let bytes = w.finalize().to_vec();
        let mut reader = BitReader::new(&bytes);
        // Ask for far more payload bits than actually remain.
        assert!(sbr_crc_check(&mut reader, 1000).is_ok());
    }
}
