//! USAC spectral arithmetic decoding.
//!
//! Ported from `ixheaacd_arth_decoding_level2` and `ixheaacd_esc_iquant` in
//! `c/libxaac/decoder/ixheaacd_arith_dec.c`. See
//! [`crate::tables::usac_arith`] for the context model and cumulative-frequency
//! tables this drives.
//!
//! This covers the coder itself — decoding one block's coefficient-pair sequence
//! into signed magnitudes, and turning those into linear coefficients. It does
//! not cover scalefactor-band bookkeeping, noise-filling's start-offset rule, or
//! the surrounding `UsacFrame()` syntax: those belong to a full frame decoder
//! built on top of this, not to the coder.

use crate::bitstream::BitReader;
use crate::tables::usac_arith::{
    ARI_CF_M, ARI_CF_R, Contexts, ESCAPE, POW_TABLE_Q13_USAC, context_pk,
};

/// The range decoder's interval, carried across symbols within one block.
struct RangeDecoder {
    low: u32,
    high: u32,
    value: u32,
    /// Bits pulled from the stream so far, including the 16-bit priming read.
    bits_consumed: u32,
}

/// A bit past end of stream reads as zero, matching the reference decoder's
/// `cnt_bits > 0 ? read : 0`: the coder is fed a few bits beyond what a
/// well-formed block strictly needs, and both sides agree those trailing reads
/// are zero rather than an error.
fn next_bit(reader: &mut BitReader) -> bool {
    reader.read_bit().unwrap_or(false)
}

impl RangeDecoder {
    fn prime(reader: &mut BitReader) -> Self {
        let mut value = 0u32;
        for _ in 0..16 {
            value = (value << 1) | next_bit(reader) as u32;
        }
        Self { low: 0, high: 65535, value, bits_consumed: 16 }
    }

    /// Decode one symbol against `cum_freq`, a descending cumulative-frequency
    /// table zero-terminated at `cum_freq[cum_freq.len() - 1]`.
    fn decode_symbol(&mut self, reader: &mut BitReader, cum_freq: &[u16]) -> i32 {
        let range = (self.high - self.low + 1) as i64;
        let cumulative = (((self.value as i64 - self.low as i64 + 1) << 14) - 1) / range;
        let symbol = find_symbol(cum_freq, cumulative as i32);

        if symbol != 0 {
            self.high =
                self.low + ((range * cum_freq[symbol as usize - 1] as i64) >> 14) as u32 - 1;
        }
        self.low += ((range * cum_freq[symbol as usize] as i64) >> 14) as u32;

        loop {
            if self.high < 32768 {
            } else if self.low >= 32768 {
                self.value -= 32768;
                self.low -= 32768;
                self.high -= 32768;
            } else if self.low >= 16384 && self.high < 49152 {
                self.value -= 16384;
                self.low -= 16384;
                self.high -= 16384;
            } else {
                break;
            }
            self.low *= 2;
            self.high = 2 * self.high + 1;
            self.value = (self.value << 1) | next_bit(reader) as u32;
            self.bits_consumed += 1;
        }
        symbol
    }
}

/// Locate the symbol whose cumulative-frequency interval contains `cumulative`,
/// via the reference's odd-length binary search (the `cfl` table lengths here
/// are 17 and 4, neither a power of two, hence the `cfl += 1` correction rather
/// than a textbook power-of-two bisection).
fn find_symbol(cum_freq: &[u16], cumulative: i32) -> i32 {
    let mut cfl = cum_freq.len() as i32;
    let mut p: i32 = -1;
    loop {
        let q = p + (cfl >> 1);
        debug_assert!(q >= 0, "cumulative-frequency search stepped out of bounds");
        if cum_freq[q as usize] as i32 > cumulative {
            p = q;
            cfl += 1;
        }
        cfl >>= 1;
        if cfl <= 1 {
            break;
        }
    }
    p + 1
}

/// Decode `n` coefficient pairs (`2n` spectral lines) into `quant[0..2n]`,
/// advancing `contexts` for the next block.
///
/// `pres_len` is the block's full coefficient-pair budget; positions from `n`
/// to `pres_len` get the context's "coded zero" treatment without being read
/// from the bitstream (the reference decoder's `max_spec_coefficients <
/// arith_pres_n` case, e.g. a band table that stops short of the transform
/// size). `quant` must hold at least `2 * n` entries.
///
/// Decoding runs against a private clone of `reader` because this coder reads
/// a few bits further ahead than it ends up needing (an artifact of priming
/// with a full 16-bit lookahead) — the real amount consumed is only known once
/// decoding finishes, at which point `reader` is advanced by exactly that many
/// bits. This mirrors the reference decoder's own temp-buffer-then-rewind
/// approach; the offset (16 bits primed, minus the 14-bit cumulative-frequency
/// precision) is a boundary condition of this specific coder variant, not a
/// tunable.
pub fn decode_pairs(
    reader: &mut BitReader,
    contexts: &mut Contexts,
    n: usize,
    pres_len: usize,
    quant: &mut [i32],
) {
    contexts.advance(pres_len);

    let mut scratch = reader.clone();
    let mut coder = RangeDecoder::prime(&mut scratch);
    let mut state = (contexts.prev(0) as u32) << 12;

    let mut coded = n;
    for i in 0..n {
        let context = contexts.get_context(i, &mut state);

        let mut lev = 0usize;
        let mut esc_nb = 0usize;
        let symbol = loop {
            let pki = context_pk(context.wrapping_add((esc_nb as u32) << 17));
            let m = coder.decode_symbol(&mut scratch, &ARI_CF_M[pki]);
            if m < ESCAPE {
                break m;
            }
            lev += 1;
            esc_nb = lev.min(7);
        };

        if symbol == 0 {
            if esc_nb > 0 {
                // The encoder's early-stop sentinel: everything from here on is
                // zero and was never transmitted.
                coded = i;
                break;
            }
            quant[2 * i] = 0;
            quant[2 * i + 1] = 0;
            contexts.set_pres(i as isize, 1);
        } else {
            let mut b = symbol >> 2;
            let mut a = symbol & 3;
            for _ in 0..lev {
                let lsbidx = if a == 0 {
                    1
                } else if b == 0 {
                    0
                } else {
                    2
                };
                let m = coder.decode_symbol(&mut scratch, &ARI_CF_R[lsbidx]);
                a = (a << 1) | (m & 1);
                b = (b << 1) | ((m >> 1) & 1);
            }
            quant[2 * i] = a;
            quant[2 * i + 1] = b;
            contexts.set_pres(i as isize, (a + b + 1).min(0xF));
        }
    }
    for i in coded..n {
        quant[2 * i] = 0;
        quant[2 * i + 1] = 0;
    }
    for i in n..pres_len {
        contexts.set_pres(i as isize, 1);
    }

    let true_bits = coder.bits_consumed - 14;
    reader.skip_bits(true_bits as usize).expect("block must carry as many bits as it coded");

    // Sign is not arithmetic-coded: one raw bit per nonzero coefficient,
    // 1 = positive, immediately following the coded block.
    for i in 0..coded {
        if quant[2 * i] != 0 && !next_bit(reader) {
            quant[2 * i] = -quant[2 * i];
        }
        if quant[2 * i + 1] != 0 && !next_bit(reader) {
            quant[2 * i + 1] = -quant[2 * i + 1];
        }
    }
}

/// Advance the noise generator and return its sign, `+1` or `-1`.
///
/// The multiplier and increment are the reference's raw LCG constants
/// (`ixheaacd_randomsign_fix`) — this exists to reproduce the reference's noise
/// sequence bit-for-bit, not as a general-purpose PRNG, so the constants are
/// load-bearing rather than arbitrary.
pub fn random_sign(seed: &mut u32) -> i32 {
    *seed = seed.wrapping_mul(69069).wrapping_add(5);
    if *seed & 0x10000 != 0 { -1 } else { 1 }
}

/// Turn one block's signed coded magnitudes into linear coefficients.
///
/// `fac_fix` is the band's fixed-point scale (see
/// [`crate::tables::usac_arith::TABLE_EXP`] and
/// [`crate::tables::usac_arith::TABLE_FRAC`] for how a scalefactor turns into
/// one); `noise_level_fixed` is a noise-fill level already looked up through
/// [`crate::tables::usac_arith::POW_14_3`]. When `with_noise` is set, a zero
/// input coefficient is synthesized from `seed` instead of decoding to silence
/// — USAC noise-fills spectral gaps that would otherwise cost bits to code as
/// exact zero, since the ear doesn't need them exact.
pub fn dequantize(
    quant: &[i32],
    coef: &mut [i32],
    noise_level_fixed: i32,
    with_noise: bool,
    seed: &mut u32,
    fac_fix: i64,
) {
    for (i, &q) in quant.iter().enumerate() {
        if with_noise && q == 0 {
            let level = random_sign(seed) * noise_level_fixed;
            coef[i] = ((fac_fix * level as i64) >> 25) as i32;
            continue;
        }

        let sign = if q < 0 { -1 } else { 1 };
        let magnitude = (q.unsigned_abs() as i32).min(8191);
        let linear = if magnitude < 1024 {
            POW_TABLE_Q13_USAC[magnitude as usize]
        } else {
            let q1 = (magnitude >> 3) as usize;
            let interp = magnitude - ((q1 as i32) << 3);
            let step = POW_TABLE_Q13_USAC[q1 + 1] - POW_TABLE_Q13_USAC[q1];
            (step * interp + (POW_TABLE_Q13_USAC[q1] << 3)) * 2
        };

        let scaled = sign * linear;
        coef[i] = ((fac_fix * scaled as i64) >> 22) as i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::BitWriter;
    use crate::encoder::usac::arith::encode_pairs;

    fn round_trip(quant: &[i32]) -> Vec<i32> {
        let n = quant.len() / 2;
        let mut writer = BitWriter::with_capacity(256);
        let mut enc_ctx = Contexts::new();
        encode_pairs(&mut writer, &mut enc_ctx, quant, n, n);
        writer.byte_align_zero();
        let bytes = writer.into_bytes();

        let mut reader = BitReader::new(&bytes);
        let mut dec_ctx = Contexts::new();
        let mut out = vec![0i32; 2 * n];
        decode_pairs(&mut reader, &mut dec_ctx, n, n, &mut out);
        out
    }

    #[test]
    fn all_zero_round_trips() {
        let quant = vec![0i32; 16];
        assert_eq!(round_trip(&quant), quant);
    }

    #[test]
    fn small_magnitudes_round_trip_without_escaping() {
        // Every |value| here is <= 3, so no coefficient forces the escape path.
        let quant = vec![1, -2, 0, 3, -1, 1, 2, -3, 0, 0, -3, -3, 3, 3, 1, -1];
        assert_eq!(round_trip(&quant), quant);
    }

    #[test]
    fn large_magnitudes_force_escape_coding() {
        let quant = vec![120, -85, 4000, -4000, 8191, -8191, 500, -1, 0, 0, 7, -7, 63, -63, 9, -9];
        assert_eq!(round_trip(&quant), quant);
    }

    #[test]
    fn context_carries_across_successive_blocks() {
        // A second block must decode correctly against the context left behind
        // by the first, not a freshly reset one -- this is what actually
        // exercises `prev` (as opposed to a single round trip, where `prev`
        // stays all-zero throughout).
        let first = vec![5, -5, 6, -6, 0, 0, 2, -2];
        let second = vec![1, -1, 0, 0, 9, -9, 3, -3];
        let n = first.len() / 2;

        let mut writer = BitWriter::with_capacity(256);
        let mut enc_ctx = Contexts::new();
        encode_pairs(&mut writer, &mut enc_ctx, &first, n, n);
        encode_pairs(&mut writer, &mut enc_ctx, &second, n, n);
        writer.byte_align_zero();
        let bytes = writer.into_bytes();

        let mut reader = BitReader::new(&bytes);
        let mut dec_ctx = Contexts::new();
        let mut out1 = vec![0i32; 2 * n];
        let mut out2 = vec![0i32; 2 * n];
        decode_pairs(&mut reader, &mut dec_ctx, n, n, &mut out1);
        decode_pairs(&mut reader, &mut dec_ctx, n, n, &mut out2);

        assert_eq!(out1, first);
        assert_eq!(out2, second);
    }

    /// Hand-traced against the reference algorithm for the first few symbols:
    /// the very first context, before any history exists, must be exactly
    /// `c_prev[0] << 12` folded through the mask/add — a wrong context-update
    /// formula can still round-trip (encoder and decoder agreeing with each
    /// other while disagreeing with the spec), so this pins the value itself.
    #[test]
    fn first_context_matches_a_hand_trace() {
        let contexts = Contexts::new();
        let mut state = (contexts.prev(0) as u32) << 12;
        // i = 0: tmp = prev(1) = 0; c = (state>>4) + 0 = 0; c = (0 & 0xFFF0) + pres(-1) = 0.
        let c0 = contexts.get_context(0, &mut state);
        assert_eq!(c0, 0);
        assert_eq!(state, 0);
    }

    #[test]
    fn unit_scale_dequantizes_to_the_pow_table_directly() {
        // fac_fix = 1<<22 makes dequantize's final >>22 a no-op, isolating the
        // magnitude curve from the scalefactor math.
        let quant = [0, 1, -1, 100];
        let mut coef = [0i32; 4];
        let mut seed = 1u32;
        dequantize(&quant, &mut coef, 0, false, &mut seed, 1 << 22);
        assert_eq!(coef[0], 0);
        assert_eq!(coef[1], POW_TABLE_Q13_USAC[1]);
        assert_eq!(coef[2], -POW_TABLE_Q13_USAC[1]);
        assert_eq!(coef[3], POW_TABLE_Q13_USAC[100]);
    }

    #[test]
    fn noise_filling_only_touches_zero_positions() {
        let quant = [0, 5, 0, -3];
        let mut coef = [0i32; 4];
        let mut seed = 42u32;
        dequantize(&quant, &mut coef, 1000, true, &mut seed, 1 << 22);
        assert_ne!(coef[0], 0, "a noise-filled position must not stay silent");
        assert_eq!(coef[1], POW_TABLE_Q13_USAC[5]);
        assert_ne!(coef[2], 0);
        assert_eq!(coef[3], -POW_TABLE_Q13_USAC[3]);
    }

    #[test]
    fn random_sign_matches_the_reference_recurrence() {
        let mut seed = 0u32;
        let s1 = random_sign(&mut seed);
        assert_eq!(seed, 5);
        assert_eq!(s1, 1);
        let s2 = random_sign(&mut seed);
        assert_eq!(seed, 5u32.wrapping_mul(69069).wrapping_add(5));
        assert!(s2 == 1 || s2 == -1);
    }
}
