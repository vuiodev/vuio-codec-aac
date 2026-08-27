//! USAC spectral arithmetic encoding.
//!
//! Ported from `iusace_arith_encode_level2` in
//! `c/libxaac/encoder/iusace_arith_enc.c`, the mirror of
//! [`crate::decoder::usac::arith::decode_pairs`] — see that module and
//! [`crate::tables::usac_arith`] for the context model and coder this shares
//! with the decoder.
//!
//! One simplification versus the reference: the reference encoder can end a
//! block early with an escape+zero sentinel when everything from some position
//! to the end of the block is zero, trading an extra trial encode (it tries
//! both ways and keeps whichever is shorter) for fewer bits on a long trailing
//! silence. This encoder always codes every pair explicitly. The two are
//! wire-compatible in both directions — [`decode_pairs`](crate::decoder::usac::arith::decode_pairs)
//! still recognises the sentinel if it appears, it just never has to here — so
//! this only costs a few bits on heavily-silent tails, not correctness.

use crate::bitstream::BitWriter;
use crate::tables::usac_arith::{ARI_CF_M, ARI_CF_R, Contexts, ESCAPE, context_pk};

/// The range encoder's interval, carried across symbols within one block.
struct RangeEncoder {
    low: u32,
    high: u32,
    /// Count of pending "opposite" bits an E3 (mid-range) renormalisation has
    /// deferred, resolved the next time a bit is actually emitted.
    pending: u32,
}

impl RangeEncoder {
    fn new() -> Self {
        Self { low: 0, high: 65535, pending: 0 }
    }

    /// Emit `bit`, followed by `self.pending` copies of its complement — the
    /// standard bit-plus-follow resolution of deferred E3 renormalisations.
    fn emit(&mut self, writer: &mut BitWriter, bit: bool) {
        writer.write_bit(bit);
        for _ in 0..self.pending {
            writer.write_bit(!bit);
        }
        self.pending = 0;
    }

    /// Encode `symbol` against `cum_freq`, a descending cumulative-frequency
    /// table zero-terminated at `cum_freq[cum_freq.len() - 1]`.
    fn encode_symbol(&mut self, writer: &mut BitWriter, symbol: i32, cum_freq: &[u16]) {
        let range = (self.high - self.low + 1) as i64;
        if symbol > 0 {
            self.high =
                self.low + ((range * cum_freq[symbol as usize - 1] as i64) >> 14) as u32 - 1;
        }
        self.low += ((range * cum_freq[symbol as usize] as i64) >> 14) as u32;

        loop {
            if self.high < 32768 {
                self.emit(writer, false);
            } else if self.low >= 32768 {
                self.emit(writer, true);
                self.low -= 32768;
                self.high -= 32768;
            } else if self.low >= 16384 && self.high < 49152 {
                self.pending += 1;
                self.low -= 16384;
                self.high -= 16384;
            } else {
                break;
            }
            self.low *= 2;
            self.high = 2 * self.high + 1;
        }
    }

    /// Flush the final interval so the decoder can disambiguate it, ending the
    /// block's arithmetic-coded payload.
    fn finish(&mut self, writer: &mut BitWriter) {
        self.pending += 1;
        let bit = self.low >= 16384;
        self.emit(writer, bit);
    }
}

/// Encode `n` coefficient pairs (`2n` spectral lines) from `quant[0..2n]`,
/// advancing `contexts` for the next block.
///
/// `pres_len` is the block's full coefficient-pair budget (see
/// [`crate::decoder::usac::arith::decode_pairs`] for why it can exceed `n`);
/// positions from `n` to `pres_len` are reset to the context's "coded zero"
/// state to match what the decoder does for the same range.
pub fn encode_pairs(
    writer: &mut BitWriter,
    contexts: &mut Contexts,
    quant: &[i32],
    n: usize,
    pres_len: usize,
) {
    let mut coder = RangeEncoder::new();
    let mut state = (contexts.prev(0) as u32) << 12;

    for i in 0..n {
        let context = contexts.get_context(i, &mut state);

        let mut a = quant[2 * i].unsigned_abs() as i32;
        let mut b = quant[2 * i + 1].unsigned_abs() as i32;
        contexts.set_pres(i as isize, (a + b + 1).min(0xF));

        let mut planes = [0i32; 32];
        let mut lev = 0usize;
        let mut esc_nb = 0usize;
        while a > 3 || b > 3 {
            let pki = context_pk(context.wrapping_add((esc_nb as u32) << 17));
            coder.encode_symbol(writer, ESCAPE, &ARI_CF_M[pki]);
            planes[lev] = (a & 1) | ((b & 1) << 1);
            lev += 1;
            a >>= 1;
            b >>= 1;
            esc_nb = (esc_nb + 1).min(7);
        }
        let symbol = a + (b << 2);
        let pki = context_pk(context.wrapping_add((esc_nb as u32) << 17));
        coder.encode_symbol(writer, symbol, &ARI_CF_M[pki]);

        for &plane in planes[..lev].iter().rev() {
            let lsbidx = if a == 0 {
                1
            } else if b == 0 {
                0
            } else {
                2
            };
            coder.encode_symbol(writer, plane, &ARI_CF_R[lsbidx]);
            a = (a << 1) | (plane & 1);
            b = (b << 1) | ((plane >> 1) & 1);
        }
    }
    coder.finish(writer);

    for i in n..pres_len {
        contexts.set_pres(i as isize, 1);
    }

    // Sign is not arithmetic-coded: one raw bit per nonzero coefficient,
    // 1 = positive, immediately following the coded block.
    for i in 0..n {
        if quant[2 * i] != 0 {
            writer.write_bit(quant[2 * i] > 0);
        }
        if quant[2 * i + 1] != 0 {
            writer.write_bit(quant[2 * i + 1] > 0);
        }
    }

    contexts.advance(pres_len);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::usac::arith::decode_pairs;

    /// Escape planes must be replayed most-significant-bit-first even though
    /// they were peeled off least-significant-bit-first — get this backwards
    /// and small values still round-trip (they never escape) while anything
    /// past magnitude 3 silently corrupts, which is exactly the kind of bug a
    /// narrow test would miss.
    #[test]
    fn escape_plane_order_reconstructs_large_magnitudes() {
        let quant = vec![8191, -8191, 4096, -4096];
        let n = quant.len() / 2;
        let mut writer = BitWriter::with_capacity(64);
        let mut ctx = Contexts::new();
        encode_pairs(&mut writer, &mut ctx, &quant, n, n);
        writer.byte_align_zero();
        let bytes = writer.into_bytes();

        let mut reader = crate::bitstream::BitReader::new(&bytes);
        let mut dec_ctx = Contexts::new();
        let mut out = vec![0i32; 2 * n];
        decode_pairs(&mut reader, &mut dec_ctx, n, n, &mut out);
        assert_eq!(out, quant);
    }
}
