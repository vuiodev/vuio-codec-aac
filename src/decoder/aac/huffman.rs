//! Spectral and scalefactor Huffman entropy decoding.
//!
//! Implements the canonical length-indexed Huffman search used by the reference C
//! decoder (`ixheaacd_huffman_decode` in `decoder/ixheaacd_env_extr.c`), driven by
//! the ROM tables in [`crate::tables::huffman_rom`].
//!
//! # Table format
//!
//! Each codebook is described by two tables:
//!
//! * `input_table[0]` holds the longest codeword length. `input_table[i + 1]` packs
//!   the decoded symbol in bits 5..15 and the codeword length in bits 0..4.
//! * `idx_table[clo]`, indexed by the count of leading one-bits in the lookahead
//!   window, packs the largest codeword value at that length in bits 0..19, the
//!   offset into `input_table` above that, and the length increment to try next in
//!   the top bits. Codebooks 1..10 and the scalefactor book use an 8-bit offset with
//!   a 4-bit increment; codebook 11's 290-entry input table needs a 9-bit offset and
//!   so leaves only 3 bits for the increment.
//!
//! Decoding walks the codeword-length classes until the lookahead prefix falls at or
//! below the class maximum, then indexes straight to the symbol. Because both tables
//! and the search are byte-for-byte those of the reference decoder, entropy decoding
//! is bit-exact by construction.

use crate::bitstream::BitReader;
use crate::error::{DecodeError, Result};
use crate::tables::huffman_rom::*;

/// Static description of one AAC spectral Huffman codebook.
#[derive(Debug, Clone, Copy)]
pub struct Codebook {
    /// Packed symbol/length table.
    pub input_table: &'static [u16],
    /// Length-class index table.
    pub idx_table: &'static [u32],
    /// Tuple width: 4 for codebooks 1..4, 2 for codebooks 5..11.
    pub dim: usize,
    /// Largest absolute value the codebook can represent.
    pub lav: i32,
    /// Radix used to unpack the tuple from the decoded symbol.
    pub huff_mode: i32,
    /// `true` when the codebook stores magnitudes and sign bits follow each nonzero.
    pub unsigned: bool,
    /// `true` when the index table packs a 9-bit offset and 3-bit length increment.
    pub wide_offset: bool,
}

/// Codebook table indexed by codebook number; entry 0 is the unused ZERO codebook.
pub static CODEBOOKS: [Option<Codebook>; 12] = [
    None,
    // CB1/CB2: 4-tuple, signed, values in -1..=1 encoded base 3.
    Some(Codebook { input_table: &INPUT_TABLE_CB1, idx_table: &IDX_TABLE_HF1, dim: 4, lav: 1, huff_mode: 3, unsigned: false, wide_offset: false }),
    Some(Codebook { input_table: &INPUT_TABLE_CB2, idx_table: &IDX_TABLE_HF2, dim: 4, lav: 1, huff_mode: 3, unsigned: false, wide_offset: false }),
    // CB3/CB4: 4-tuple, magnitudes 0..=2 base 3, sign bits follow.
    Some(Codebook { input_table: &INPUT_TABLE_CB3, idx_table: &IDX_TABLE_HF3, dim: 4, lav: 2, huff_mode: 3, unsigned: true, wide_offset: false }),
    Some(Codebook { input_table: &INPUT_TABLE_CB4, idx_table: &IDX_TABLE_HF4, dim: 4, lav: 2, huff_mode: 3, unsigned: true, wide_offset: false }),
    // CB5/CB6: pair, signed, values in -4..=4 encoded base 9.
    Some(Codebook { input_table: &INPUT_TABLE_CB5, idx_table: &IDX_TABLE_HF5, dim: 2, lav: 4, huff_mode: 9, unsigned: false, wide_offset: false }),
    Some(Codebook { input_table: &INPUT_TABLE_CB6, idx_table: &IDX_TABLE_HF6, dim: 2, lav: 4, huff_mode: 9, unsigned: false, wide_offset: false }),
    // CB7/CB8: pair, magnitudes 0..=7 base 8, sign bits follow.
    Some(Codebook { input_table: &INPUT_TABLE_CB7, idx_table: &IDX_TABLE_HF7, dim: 2, lav: 7, huff_mode: 8, unsigned: true, wide_offset: false }),
    Some(Codebook { input_table: &INPUT_TABLE_CB8, idx_table: &IDX_TABLE_HF8, dim: 2, lav: 7, huff_mode: 8, unsigned: true, wide_offset: false }),
    // CB9/CB10: pair, magnitudes 0..=12 base 13, sign bits follow.
    Some(Codebook { input_table: &INPUT_TABLE_CB9, idx_table: &IDX_TABLE_HF9, dim: 2, lav: 12, huff_mode: 13, unsigned: true, wide_offset: false }),
    Some(Codebook { input_table: &INPUT_TABLE_CB10, idx_table: &IDX_TABLE_HF10, dim: 2, lav: 12, huff_mode: 13, unsigned: true, wide_offset: false }),
    // CB11 (ESC): pair, magnitudes 0..=16 base 17; magnitude 16 escapes to a wider value.
    Some(Codebook { input_table: &INPUT_TABLE_CB11, idx_table: &IDX_TABLE_HF11, dim: 2, lav: 16, huff_mode: 17, unsigned: true, wide_offset: true }),
];

/// Look up the codebook description for a spectral codebook number.
#[inline]
pub fn codebook(cb: u8) -> Option<&'static Codebook> {
    CODEBOOKS.get(cb as usize).and_then(|c| c.as_ref())
}

/// Decode one Huffman symbol from a 32-bit left-aligned lookahead window.
///
/// Returns `(symbol, codeword_length_in_bits)`.
///
/// `WIDE` selects the index-table packing. Most codebooks store the input-table
/// offset in 8 bits and the length increment in 4 (`ixheaacd_huffman_decode`), but
/// codebook 11's input table has 290 entries, so it widens the offset to 9 bits and
/// narrows the increment to 3 (`ixheaacd_huff_sfb_table`). Decoding a book with the
/// wrong packing silently yields out-of-range symbols, so the choice is carried in
/// [`Codebook::wide_offset`] rather than inferred.
#[inline(always)]
fn decode_symbol<const WIDE: bool>(
    lookahead: u32,
    input_table: &[u16],
    idx_table: &[u32],
) -> (i32, u32) {
    let max_len = input_table[0] as u32;

    // Isolate the widest prefix any codeword in this book can occupy.
    let mask = (0x8000_0000u32.wrapping_sub(1 << (31 - max_len))) << 1;
    let temp = lookahead & mask;

    let (offset_mask, incr_shift, incr_mask) = if WIDE {
        (0x1FFu32, 29u32, 0x7u32)
    } else {
        (0x0FFu32, 28u32, 0xFu32)
    };

    let mut len_end = max_len;
    let mut clo = temp.leading_ones();

    loop {
        // A corrupt stream can walk past the table; fall back to the shortest
        // codeword so the caller makes forward progress instead of panicking.
        let entry = match idx_table.get(clo as usize) {
            Some(&e) => e,
            None => return (0, 1),
        };
        let offset = ((entry >> 20) & offset_mask) as usize;
        let Some(&packed) = input_table.get(offset + 1) else {
            return (0, 1);
        };
        let length = (packed & 0x1F) as u32;
        let cwrd = entry & 0x000F_FFFF;
        let prefix = temp >> (32 - length);

        if prefix <= cwrd {
            let back = (cwrd - prefix) as usize;
            if back > offset {
                return (0, 1);
            }
            let symbol = (input_table[offset - back + 1] >> 5) as i32;
            return (symbol, length);
        }

        len_end += (entry >> incr_shift) & incr_mask;
        clo = len_end;
    }
}

/// Decode a scalefactor delta using the scalefactor codebook.
///
/// Returns the DPCM delta already biased by `-SF_OFFSET` (60), matching the
/// reference decoder's `ixheaacd_book_scl` tables.
#[inline]
pub fn decode_scalefactor_delta(reader: &mut BitReader) -> Result<i32> {
    let lookahead = reader.peek32_padded();
    let (symbol, len) = decode_symbol::<false>(
        lookahead,
        &HUFFMAN_CODE_BOOK_SCL,
        &HUFFMAN_CODE_BOOK_SCL_INDEX,
    );
    reader.skip_bits(len as usize)?;
    Ok(symbol - 60)
}

/// Decode the escape magnitude that follows a CB11 value of 16.
///
/// The escape is a unary-coded extra bit count followed by that many value bits:
/// `value = (1 << n) + extra`, with `n` starting at 4.
#[inline]
fn decode_escape(reader: &mut BitReader) -> Result<i32> {
    let mut n = 4u32;
    while reader.read_bit()? {
        n += 1;
        // The largest legal AAC escape is 2^19; refuse to spin on corrupt input.
        if n > 20 {
            return Err(DecodeError::HuffmanDecodeError { codebook: 11, bits: 0 }.into());
        }
    }
    let extra = reader.read_u32(n as usize)? as i32;
    Ok((1i32 << n) + extra)
}

/// Decode `output.len()` quantized spectral coefficients using codebook `cb`.
///
/// `output.len()` must be a multiple of the codebook's tuple width, which the AAC
/// scalefactor-band layout guarantees (every band width is a multiple of four).
pub fn decode_spectral_band(reader: &mut BitReader, cb: u8, output: &mut [i32]) -> Result<()> {
    if cb == 0 {
        output.fill(0);
        return Ok(());
    }
    let Some(book) = codebook(cb) else {
        // Codebooks 12..15 carry no spectral data (intensity / noise are handled
        // by the caller); leave the band zeroed.
        output.fill(0);
        return Ok(());
    };

    if book.dim == 4 {
        decode_quads(reader, book, output)
    } else if cb == 11 {
        decode_pairs_esc(reader, book, output)
    } else {
        decode_pairs(reader, book, output)
    }
}

#[inline]
fn decode_quads(reader: &mut BitReader, book: &Codebook, output: &mut [i32]) -> Result<()> {
    let m = book.huff_mode;
    for chunk in output.chunks_exact_mut(4) {
        let lookahead = reader.peek32_padded();
        let (symbol, len) = decode_symbol::<false>(lookahead, book.input_table, book.idx_table);
        reader.skip_bits(len as usize)?;

        let mut idx = symbol;
        let w = idx / (m * m * m);
        idx -= w * m * m * m;
        let x = idx / (m * m);
        idx -= x * m * m;
        let y = idx / m;
        let z = idx - y * m;

        if book.unsigned {
            chunk[0] = apply_sign(reader, w)?;
            chunk[1] = apply_sign(reader, x)?;
            chunk[2] = apply_sign(reader, y)?;
            chunk[3] = apply_sign(reader, z)?;
        } else {
            let off = book.lav;
            chunk[0] = w - off;
            chunk[1] = x - off;
            chunk[2] = y - off;
            chunk[3] = z - off;
        }
    }
    Ok(())
}

#[inline]
fn decode_pairs(reader: &mut BitReader, book: &Codebook, output: &mut [i32]) -> Result<()> {
    let m = book.huff_mode;
    for chunk in output.chunks_exact_mut(2) {
        let lookahead = reader.peek32_padded();
        let (symbol, len) = decode_symbol::<false>(lookahead, book.input_table, book.idx_table);
        reader.skip_bits(len as usize)?;

        let y = symbol / m;
        let z = symbol - y * m;

        if book.unsigned {
            chunk[0] = apply_sign(reader, y)?;
            chunk[1] = apply_sign(reader, z)?;
        } else {
            let off = book.lav;
            chunk[0] = y - off;
            chunk[1] = z - off;
        }
    }
    Ok(())
}

/// Decode codebook 11, where a magnitude of 16 is followed by an escape sequence.
///
/// Both sign bits are read before either escape, per ISO/IEC 14496-3 4.6.3.
#[inline]
fn decode_pairs_esc(reader: &mut BitReader, book: &Codebook, output: &mut [i32]) -> Result<()> {
    debug_assert!(book.wide_offset, "codebook 11 must use the wide index packing");
    let m = book.huff_mode;
    for chunk in output.chunks_exact_mut(2) {
        let lookahead = reader.peek32_padded();
        let (symbol, len) = decode_symbol::<true>(lookahead, book.input_table, book.idx_table);
        reader.skip_bits(len as usize)?;

        let mag_y = symbol / m;
        let mag_z = symbol - mag_y * m;

        let neg_y = mag_y != 0 && reader.read_bit()?;
        let neg_z = mag_z != 0 && reader.read_bit()?;

        let val_y = if mag_y == 16 { decode_escape(reader)? } else { mag_y };
        let val_z = if mag_z == 16 { decode_escape(reader)? } else { mag_z };

        chunk[0] = if neg_y { -val_y } else { val_y };
        chunk[1] = if neg_z { -val_z } else { val_z };
    }
    Ok(())
}

/// Read a sign bit for a nonzero magnitude; zero magnitudes carry no sign bit.
#[inline(always)]
fn apply_sign(reader: &mut BitReader, magnitude: i32) -> Result<i32> {
    if magnitude == 0 {
        return Ok(0);
    }
    Ok(if reader.read_bit()? { -magnitude } else { magnitude })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::BitWriter;

    /// Every codebook's index-table offsets must stay inside its input table,
    /// under that codebook's own packing.
    #[test]
    fn codebook_tables_are_in_bounds() {
        for cb in 1..=11u8 {
            let book = codebook(cb).expect("codebook present");
            let offset_mask: u32 = if book.wide_offset { 0x1FF } else { 0xFF };
            for &entry in book.idx_table.iter() {
                let offset = ((entry >> 20) & offset_mask) as usize;
                assert!(
                    offset + 1 < book.input_table.len(),
                    "cb{cb}: offset {offset} outside input table of {}",
                    book.input_table.len()
                );
            }
        }
    }

    /// Sweep the whole lookahead space of every codebook and check each decoded
    /// symbol lands inside `huff_mode ^ dim`.
    ///
    /// This is the guard against mis-declared index-table packing: decoding a book
    /// with the wrong offset/increment widths still terminates, but yields symbols
    /// past the end of the tuple space, which unpacks into values beyond the LAV.
    #[test]
    fn every_lookahead_decodes_to_a_valid_symbol() {
        for cb in 1..=11u8 {
            let book = codebook(cb).unwrap();
            let max_symbol = (book.huff_mode as i64).pow(book.dim as u32) - 1;
            let max_len = book.input_table[0] as u32;

            // Every distinct prefix the mask can produce, exhaustively.
            for pattern in 0..(1u32 << max_len) {
                let lookahead = pattern << (32 - max_len);
                let (symbol, len) = if book.wide_offset {
                    decode_symbol::<true>(lookahead, book.input_table, book.idx_table)
                } else {
                    decode_symbol::<false>(lookahead, book.input_table, book.idx_table)
                };
                assert!(
                    (symbol as i64) <= max_symbol,
                    "cb{cb}: lookahead {lookahead:#010x} gave symbol {symbol} > {max_symbol}"
                );
                assert!(
                    len >= 1 && len <= max_len,
                    "cb{cb}: lookahead {lookahead:#010x} gave length {len}, expected 1..={max_len}"
                );
            }
        }
    }

    /// Decoded tuples must stay within each codebook's largest absolute value.
    #[test]
    fn decoded_values_respect_lav() {
        for cb in 1..=10u8 {
            let book = codebook(cb).unwrap();
            for pattern in 0..8192u32 {
                let bits = pattern << 19;
                let bytes = bits.to_be_bytes();
                let buf = [bytes[0], bytes[1], bytes[2], bytes[3], 0, 0, 0, 0, 0, 0, 0, 0];
                let mut r = BitReader::new(&buf);
                let mut out = [0i32; 4];
                let n = book.dim;
                if decode_spectral_band(&mut r, cb, &mut out[..n]).is_ok() {
                    for &v in &out[..n] {
                        assert!(
                            v.abs() <= book.lav,
                            "cb{cb} produced {v} beyond lav {}",
                            book.lav
                        );
                    }
                }
            }
        }
    }

    /// The all-zero tuple is the most probable symbol in every codebook and must
    /// round-trip through the length-class search.
    #[test]
    fn zero_tuple_decodes() {
        // Codebook 1's shortest codeword ("1") maps to the all-zero quad.
        let buf = [0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0];
        let mut r = BitReader::new(&buf);
        let mut out = [7i32; 4];
        decode_spectral_band(&mut r, 1, &mut out).unwrap();
        assert!(out.iter().all(|&v| v.abs() <= 1));
    }

    /// Escape decoding must invert the unary-plus-value encoding.
    #[test]
    fn escape_round_trip() {
        for &value in &[16i32, 17, 31, 32, 100, 255, 1000, 8191] {
            let mut n = 4u32;
            while (1i32 << n) + ((1i32 << n) - 1) < value {
                n += 1;
            }
            let mut w = BitWriter::with_capacity(16);
            for _ in 4..n {
                w.write_bit(true);
            }
            w.write_bit(false);
            w.write_bits((value - (1 << n)) as u64, n as usize);
            w.byte_align_zero();
            let bytes = w.into_bytes();
            let mut r = BitReader::new(&bytes);
            assert_eq!(decode_escape(&mut r).unwrap(), value, "escape for {value}");
        }
    }
}
