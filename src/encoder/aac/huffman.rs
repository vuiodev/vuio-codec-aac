//! Spectral and scalefactor Huffman encoding.
//!
//! The codeword tables are derived from the decoder's ROM tables rather than being
//! a second transcription of the standard. That makes the two halves consistent by
//! construction: `encoder_decoder_round_trip` below encodes every symbol of every
//! codebook and decodes it back, so the pair cannot drift apart.
//!
//! # Recovering codewords from the decode tables
//!
//! The decoder maps a `length`-bit prefix `p` to the symbol at index
//! `offset - (cwrd - p)`, where `(offset, cwrd, length)` describe one codeword-length
//! class. Reading that backwards, the symbol at index `offset - d` has codeword
//! `cwrd - d`, for as long as the entries keep the same codeword length. Walking each
//! class that way recovers every codeword exactly.

use crate::bitstream::BitWriter;
use crate::decoder::aac::huffman::{Codebook, codebook};
use crate::tables::huffman_rom::{HUFFMAN_CODE_BOOK_SCL, HUFFMAN_CODE_BOOK_SCL_INDEX};
use std::sync::OnceLock;

/// One Huffman codeword.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Codeword {
    pub bits: u32,
    pub len: u8,
}

impl Codeword {
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.len > 0
    }
}

/// Build the symbol-to-codeword table for one codebook.
fn build_table(
    input_table: &[u16],
    idx_table: &[u32],
    wide_offset: bool,
    num_symbols: usize,
) -> Vec<Codeword> {
    let offset_mask: u32 = if wide_offset { 0x1FF } else { 0xFF };
    let mut out = vec![Codeword::default(); num_symbols];

    for &entry in idx_table {
        let offset = ((entry >> 20) & offset_mask) as usize;
        let cwrd = entry & 0x000F_FFFF;
        let Some(&packed) = input_table.get(offset + 1) else {
            continue;
        };
        let length = (packed & 0x1F) as u8;
        if length == 0 {
            continue;
        }

        // Walk down from this class's top entry while the codeword length holds.
        let mut d = 0usize;
        while d <= offset && d as u32 <= cwrd {
            let packed = input_table[offset - d + 1];
            if (packed & 0x1F) as u8 != length {
                break;
            }
            let symbol = (packed >> 5) as usize;
            if symbol < num_symbols {
                out[symbol] = Codeword { bits: cwrd - d as u32, len: length };
            }
            d += 1;
        }
    }
    out
}

struct EncodeTables {
    /// Per spectral codebook 1..=11, indexed by symbol.
    spectral: [Vec<Codeword>; 12],
    /// Scalefactor codebook, indexed by `delta + 60`.
    scalefactor: Vec<Codeword>,
}

fn tables() -> &'static EncodeTables {
    static TABLES: OnceLock<EncodeTables> = OnceLock::new();
    TABLES.get_or_init(|| {
        let mut spectral: [Vec<Codeword>; 12] = Default::default();
        for cb in 1..=11u8 {
            let book: &Codebook = codebook(cb).expect("codebook present");
            let num = (book.huff_mode as usize).pow(book.dim as u32);
            spectral[cb as usize] =
                build_table(book.input_table, book.idx_table, book.wide_offset, num);
        }
        // The scalefactor book has 121 symbols covering deltas -60..=60.
        let scalefactor = build_table(
            &HUFFMAN_CODE_BOOK_SCL,
            &HUFFMAN_CODE_BOOK_SCL_INDEX,
            false,
            121,
        );
        EncodeTables { spectral, scalefactor }
    })
}

/// Codeword for `symbol` in spectral codebook `cb`.
#[inline]
pub fn spectral_codeword(cb: u8, symbol: usize) -> Option<Codeword> {
    let t = tables();
    let table = t.spectral.get(cb as usize)?;
    let cw = *table.get(symbol)?;
    cw.is_valid().then_some(cw)
}

/// Codeword for a scalefactor DPCM delta, which must lie in `-60..=60`.
#[inline]
pub fn scalefactor_codeword(delta: i32) -> Option<Codeword> {
    let idx = usize::try_from(delta + 60).ok()?;
    let cw = *tables().scalefactor.get(idx)?;
    cw.is_valid().then_some(cw)
}

/// Write a scalefactor DPCM delta.
pub fn write_scalefactor_delta(writer: &mut BitWriter, delta: i32) -> bool {
    match scalefactor_codeword(delta) {
        Some(cw) => {
            writer.write_bits(cw.bits as u64, cw.len as usize);
            true
        }
        None => false,
    }
}

/// Bits needed to code one tuple with codebook `cb`, including sign and escape bits.
///
/// Returns `None` when the codebook cannot represent the tuple.
pub fn tuple_cost(cb: u8, values: &[i32]) -> Option<u32> {
    let book = codebook(cb)?;
    if values.len() != book.dim {
        return None;
    }
    let (symbol, extra) = encode_tuple_symbol(book, cb, values)?;
    let cw = spectral_codeword(cb, symbol)?;
    Some(cw.len as u32 + extra)
}

/// Map a tuple to its codebook symbol, plus the number of trailing bits (signs and
/// escapes) that follow the codeword.
fn encode_tuple_symbol(book: &Codebook, cb: u8, values: &[i32]) -> Option<(usize, u32)> {
    let m = book.huff_mode;
    let mut symbol = 0usize;
    let mut extra = 0u32;

    for &v in values {
        let digit = if book.unsigned {
            let mag = v.unsigned_abs() as i32;
            // Codebook 11 escapes any magnitude at or above its LAV.
            let coded = if cb == 11 && mag >= book.lav { book.lav } else { mag };
            if coded > book.lav {
                return None;
            }
            if coded != 0 {
                extra += 1; // sign bit
            }
            if cb == 11 && mag >= book.lav {
                extra += escape_bits(mag)?;
            }
            coded
        } else {
            if v.abs() > book.lav {
                return None;
            }
            v + book.lav
        };
        symbol = symbol * m as usize + digit as usize;
    }
    Some((symbol, extra))
}

/// Bits an escape sequence occupies for magnitude `mag` (which is at least 16).
fn escape_bits(mag: i32) -> Option<u32> {
    let mut n = 4u32;
    while n <= 20 {
        // A value of n codes magnitudes in [2^n, 2^(n+1)).
        if mag < (1i32 << (n + 1)) {
            // (n - 4) ones, one zero, then n value bits.
            return Some((n - 4) + 1 + n);
        }
        n += 1;
    }
    None
}

/// Write one escape sequence.
fn write_escape(writer: &mut BitWriter, mag: i32) {
    let mut n = 4u32;
    while n <= 20 && mag >= (1i32 << (n + 1)) {
        n += 1;
    }
    for _ in 4..n {
        writer.write_bit(true);
    }
    writer.write_bit(false);
    writer.write_bits((mag - (1i32 << n)) as u64, n as usize);
}

/// Write one tuple with codebook `cb`. Returns `false` if it cannot be represented.
pub fn write_tuple(writer: &mut BitWriter, cb: u8, values: &[i32]) -> bool {
    let Some(book) = codebook(cb) else { return false };
    if values.len() != book.dim {
        return false;
    }
    let Some((symbol, _)) = encode_tuple_symbol(book, cb, values) else {
        return false;
    };
    let Some(cw) = spectral_codeword(cb, symbol) else {
        return false;
    };

    writer.write_bits(cw.bits as u64, cw.len as usize);

    if !book.unsigned {
        return true;
    }

    // Sign bits for every nonzero value, then escapes, in that order.
    for &v in values {
        let mag = v.unsigned_abs() as i32;
        let coded = if cb == 11 && mag >= book.lav { book.lav } else { mag };
        if coded != 0 {
            writer.write_bit(v < 0);
        }
    }
    if cb == 11 {
        for &v in values {
            let mag = v.unsigned_abs() as i32;
            if mag >= book.lav {
                write_escape(writer, mag);
            }
        }
    }
    true
}

/// Largest magnitude codebook `cb` can code without escaping.
#[inline]
pub fn codebook_lav(cb: u8) -> i32 {
    codebook(cb).map_or(0, |b| b.lav)
}

/// Smallest codebook that can code every value in `band`, or `None` if all are zero.
///
/// Prefers the lower-numbered book of each pair, which is the one tuned for the
/// smaller values; the caller refines the choice by measured cost.
pub fn minimum_codebook(band: &[i32]) -> Option<u8> {
    let peak = band.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0) as i32;
    if peak == 0 {
        return None;
    }
    Some(match peak {
        1 => 1,
        2 => 3,
        3..=4 => 5,
        5..=7 => 7,
        8..=12 => 9,
        _ => 11,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::BitReader;
    use crate::decoder::aac::huffman::decode_spectral_band;

    /// Every symbol of every codebook must have a codeword.
    #[test]
    fn all_symbols_have_codewords() {
        for cb in 1..=11u8 {
            let book = codebook(cb).unwrap();
            let num = (book.huff_mode as usize).pow(book.dim as u32);
            for symbol in 0..num {
                assert!(
                    spectral_codeword(cb, symbol).is_some(),
                    "cb{cb} symbol {symbol} has no codeword"
                );
            }
        }
        for delta in -60..=60 {
            assert!(scalefactor_codeword(delta).is_some(), "scalefactor delta {delta}");
        }
    }

    /// Codeword lengths must satisfy the Kraft equality, which holds exactly for a
    /// complete prefix code and would fail if any codeword were missing or duplicated.
    #[test]
    fn codeword_lengths_satisfy_kraft_equality() {
        for cb in 1..=11u8 {
            let book = codebook(cb).unwrap();
            let num = (book.huff_mode as usize).pow(book.dim as u32);
            let sum: f64 = (0..num)
                .map(|s| 2f64.powi(-(spectral_codeword(cb, s).unwrap().len as i32)))
                .sum();
            assert!((sum - 1.0).abs() < 1e-9, "cb{cb}: Kraft sum {sum}");
        }
    }

    /// No codeword may be a prefix of another.
    #[test]
    fn code_is_prefix_free() {
        for cb in 1..=11u8 {
            let book = codebook(cb).unwrap();
            let num = (book.huff_mode as usize).pow(book.dim as u32);
            let mut words: Vec<Codeword> =
                (0..num).map(|s| spectral_codeword(cb, s).unwrap()).collect();
            words.sort_by_key(|c| (c.len, c.bits));
            for (i, a) in words.iter().enumerate() {
                for b in &words[i + 1..] {
                    if b.len < a.len {
                        continue;
                    }
                    let shifted = b.bits >> (b.len - a.len);
                    assert!(
                        shifted != a.bits || (a.len == b.len && a.bits != b.bits),
                        "cb{cb}: {a:?} is a prefix of {b:?}"
                    );
                }
            }
        }
    }

    /// Encoding a tuple and decoding it back must return the original values, for
    /// every codebook and across its whole representable range.
    #[test]
    fn encoder_decoder_round_trip() {
        for cb in 1..=11u8 {
            let book = codebook(cb).unwrap();
            let lav = book.lav;
            let dim = book.dim;

            // Sweep the full value range each codebook can represent.
            let values: Vec<i32> = if book.unsigned {
                (0..=lav).collect::<Vec<_>>()
                    .into_iter()
                    .flat_map(|v| if v == 0 { vec![0] } else { vec![v, -v] })
                    .collect()
            } else {
                (-lav..=lav).collect()
            };

            for a in &values {
                for b in &values {
                    let tuple: Vec<i32> = if dim == 4 {
                        vec![*a, *b, *b, *a]
                    } else {
                        vec![*a, *b]
                    };

                    let mut w = BitWriter::with_capacity(16);
                    assert!(write_tuple(&mut w, cb, &tuple), "cb{cb} cannot code {tuple:?}");
                    w.byte_align_zero();
                    let bytes = w.into_bytes();

                    let mut r = BitReader::new(&bytes);
                    let mut out = vec![0i32; dim];
                    decode_spectral_band(&mut r, cb, &mut out).unwrap();
                    assert_eq!(out, tuple, "cb{cb} round trip");
                }
            }
        }
    }

    /// Codebook 11's escape path must round-trip large magnitudes.
    #[test]
    fn escape_values_round_trip() {
        for &mag in &[16i32, 17, 31, 32, 33, 100, 255, 256, 1000, 4095, 8191] {
            for tuple in [[mag, 0], [0, mag], [mag, -mag], [-mag, 3]] {
                let mut w = BitWriter::with_capacity(32);
                assert!(write_tuple(&mut w, 11, &tuple), "cb11 cannot code {tuple:?}");
                w.byte_align_zero();
                let bytes = w.into_bytes();

                let mut r = BitReader::new(&bytes);
                let mut out = [0i32; 2];
                decode_spectral_band(&mut r, 11, &mut out).unwrap();
                assert_eq!(out, tuple, "escape round trip for {tuple:?}");
            }
        }
    }

    /// Scalefactor deltas must round-trip through the decoder.
    #[test]
    fn scalefactor_deltas_round_trip() {
        use crate::decoder::aac::huffman::decode_scalefactor_delta;
        for delta in -60..=60 {
            let mut w = BitWriter::with_capacity(8);
            assert!(write_scalefactor_delta(&mut w, delta));
            w.byte_align_zero();
            let bytes = w.into_bytes();
            let mut r = BitReader::new(&bytes);
            assert_eq!(decode_scalefactor_delta(&mut r).unwrap(), delta);
        }
    }

    /// The reported cost must equal the bits actually written.
    #[test]
    fn cost_matches_written_length() {
        for cb in 1..=11u8 {
            let book = codebook(cb).unwrap();
            let lav = book.lav;
            for v in -lav..=lav {
                let tuple: Vec<i32> =
                    if book.dim == 4 { vec![v, 0, -v, v] } else { vec![v, -v] };
                let cost = tuple_cost(cb, &tuple).expect("representable");
                let mut w = BitWriter::with_capacity(16);
                let before = w.bits_written();
                assert!(write_tuple(&mut w, cb, &tuple));
                let written = w.bits_written() - before;
                assert_eq!(written as u32, cost, "cb{cb} tuple {tuple:?}");
            }
        }
    }

    /// The chosen codebook must be able to code the band it was chosen for.
    #[test]
    fn minimum_codebook_covers_its_band() {
        for peak in 1..40i32 {
            let band = [peak, -peak, 0, 1];
            let cb = minimum_codebook(&band).expect("nonzero band");
            let mut w = BitWriter::with_capacity(32);
            let dim = codebook(cb).unwrap().dim;
            for chunk in band.chunks(dim) {
                assert!(write_tuple(&mut w, cb, chunk), "cb{cb} cannot code peak {peak}");
            }
        }
        assert_eq!(minimum_codebook(&[0, 0, 0, 0]), None);
    }
}
