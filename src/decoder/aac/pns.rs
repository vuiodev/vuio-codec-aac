//! Perceptual Noise Substitution (PNS).
//!
//! Bands the encoder judged noise-like are not transmitted at all. Instead the
//! bitstream carries only their energy, and the decoder substitutes locally
//! generated noise at that energy. See ISO/IEC 14496-3 clause 4.6.13.
//!
//! Because the substituted signal is synthesised rather than transmitted, decoders
//! agree on its energy but not on its samples. This generator uses the same linear
//! congruential sequence as the reference decoder (`ixheaacd_gen_rand_vec`) so that
//! decoded PNS bands track the reference as closely as a float pipeline can.

use crate::decoder::aac::ics::{ChannelData, NOISE_HCB};

/// How the noise generator is seeded.
///
/// Only the *energy* of a noise-substituted band is normative; the samples are
/// synthesised locally, so the choice here is a genuine trade-off rather than a
/// question of correctness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NoiseMode {
    /// Thread one generator through the stream, as the reference decoder does.
    ///
    /// Tracks the reference decoder's noise closely, which is what a fidelity
    /// comparison against it measures. The cost is that a frame's noise depends on
    /// every frame before it, so decoding from a seek point or in parallel chunks
    /// gives different samples in those bands.
    #[default]
    Sequential,
    /// Seed from the frame index and channel.
    ///
    /// A frame decodes to the same noise wherever decoding started, which makes
    /// seeking and chunk-parallel decoding byte-exact against a sequential decode.
    /// The cost is that the noise no longer tracks the reference decoder's.
    PerFrame,
}

/// Linear congruential generator matching `ixheaacd_gen_rand_vec`.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoiseRng {
    seed: i32,
}

impl NoiseRng {
    /// Start a generator from an explicit seed.
    pub const fn with_seed(seed: i32) -> Self {
        Self { seed }
    }

    /// Seed for a given frame and channel.
    ///
    /// Mixes the two with an odd multiplier so adjacent frames and channels start
    /// far apart in the sequence rather than at neighbouring values.
    pub const fn for_frame(frame: u64, channel: usize) -> Self {
        let mixed = (frame as i64).wrapping_mul(0x9E37_79B9) ^ ((channel as i64) << 40);
        Self { seed: mixed as i32 }
    }

    /// Current seed, so a channel pair can share a correlated sequence.
    #[inline]
    pub const fn seed(&self) -> i32 {
        self.seed
    }

    /// Advance the generator and return the next sample.
    ///
    /// The reference takes `seed >> 3`, keeping the value inside 28 bits.
    #[inline]
    pub fn next_sample(&mut self) -> f32 {
        self.seed = self.seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.seed >> 3) as f32
    }

    /// Fill `band` with noise normalised to unit energy, then scaled by
    /// `2^(energy / 4)`.
    pub fn fill_band(&mut self, band: &mut [f32], energy: i16) {
        if band.is_empty() {
            return;
        }
        let mut power = 0.0f64;
        for slot in band.iter_mut() {
            let v = self.next_sample();
            *slot = v;
            power += (v as f64) * (v as f64);
        }
        if power <= 0.0 {
            return;
        }
        let target = (energy as f64 * 0.25).exp2();
        let scale = (target / power.sqrt()) as f32;
        for slot in band.iter_mut() {
            *slot *= scale;
        }
    }
}

/// Fill every noise-substituted band of a channel.
///
/// `rng` is threaded across channels and frames so that the sequence advances the
/// way the reference decoder's does.
pub fn apply_pns(ch: &mut ChannelData, rng: &mut NoiseRng) {
    let ics = ch.ics.clone();

    for g in 0..ics.num_window_groups {
        let group_base = ics.group_base(g);
        let group_len = ics.window_group_length[g];
        for sfb in 0..ics.max_sfb {
            if ch.sfb_cb[g][sfb] != NOISE_HCB {
                continue;
            }
            let start = group_base + ics.grouped_offset(g, sfb);
            let width = (ics.swb_offset[sfb + 1] - ics.swb_offset[sfb]) as usize * group_len;
            let end = (start + width).min(ch.spec.len());
            if start >= end {
                continue;
            }
            let energy = ch.scale_factors[g][sfb];
            rng.fill_band(&mut ch.spec[start..end], energy);
        }
    }
}

/// Legacy alias retained for the encoder-side tests.
pub type PnsGenerator = NoiseRng;

#[cfg(test)]
mod tests {
    use super::*;

    /// The generator must reproduce the reference LCG exactly.
    #[test]
    fn lcg_matches_the_reference_recurrence() {
        let mut rng = NoiseRng::default();
        let mut expect: i32 = 0;
        for step in 0..64 {
            expect = expect.wrapping_mul(1664525).wrapping_add(1013904223);
            let got = rng.next_sample();
            assert_eq!(got, (expect >> 3) as f32, "step {step}");
        }
    }

    /// A filled band must carry exactly the requested energy.
    #[test]
    fn filled_band_has_the_requested_energy() {
        for energy in [-40i16, -8, 0, 8, 40, 100] {
            let mut rng = NoiseRng::default();
            let mut band = vec![0.0f32; 32];
            rng.fill_band(&mut band, energy);

            let power: f64 = band.iter().map(|&v| (v as f64) * (v as f64)).sum();
            let want = (energy as f64 * 0.25).exp2().powi(2);
            let ratio = power / want;
            assert!((ratio - 1.0).abs() < 1e-3, "energy {energy}: ratio {ratio}");
        }
    }

    /// Successive bands must get different noise, not a repeated block.
    #[test]
    fn successive_bands_differ() {
        let mut rng = NoiseRng::default();
        let mut a = vec![0.0f32; 16];
        let mut b = vec![0.0f32; 16];
        rng.fill_band(&mut a, 40);
        rng.fill_band(&mut b, 40);
        assert!(a.iter().zip(b.iter()).any(|(x, y)| x != y));
    }

    /// The same seed must give the same noise, so a channel pair can correlate.
    #[test]
    fn equal_seeds_give_equal_noise() {
        let mut a = NoiseRng::with_seed(12345);
        let mut b = NoiseRng::with_seed(12345);
        let mut x = vec![0.0f32; 24];
        let mut y = vec![0.0f32; 24];
        a.fill_band(&mut x, 20);
        b.fill_band(&mut y, 20);
        assert_eq!(x, y);
    }

    /// An empty band must not advance the generator or panic.
    #[test]
    fn empty_band_is_a_no_op() {
        let mut rng = NoiseRng::default();
        rng.fill_band(&mut [], 10);
        assert_eq!(rng.seed(), 0);
    }
}
