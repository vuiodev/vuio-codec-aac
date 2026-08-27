//! The hybrid filterbank parametric stereo layers on top of the QMF bank.
//!
//! A 64-band QMF bank spaces its bands linearly, which is far coarser than the ear
//! at low frequency: at 44.1 kHz the first band alone spans 0–345 Hz, where several
//! critical bands live. Parametric stereo therefore splits the lowest few QMF bands
//! again, with a short complex-modulated FIR, before it measures or applies anything.
//!
//! Two configurations exist. The coarse one splits three QMF bands into 8 + 2 + 2
//! and folds the 8 down to 6, giving ten sub-bands; the fine one splits five QMF
//! bands into 12 + 8 + 4 + 4 + 4, giving thirty-two. Which is in force follows from
//! the parameter resolution the bitstream asks for.
//!
//! # Alignment
//!
//! Every prototype here is 13 taps and symmetric, so its output is centred six QMF
//! slots into its window. Rather than delay the bands that are *not* split to match,
//! the filterbank reads six slots past the end of the frame: the split bands all sit
//! below the band-replication crossover, where the grid already holds the core
//! signal unaltered, so the look-ahead costs nothing and parametric stereo adds no
//! delay of its own.

use crate::dsp::fft::Complex32;
use crate::tables::ps::{
    HYBRID_P2_20, HYBRID_P4_34, HYBRID_P8_20, HYBRID_P8_34, HYBRID_P12_34, HYBRID_RES_20,
    HYBRID_RES_34, HYBRID_TAPS, MOD_2, MOD_4, MOD_8, MOD_12, SUB_QMF_20, SUB_QMF_34,
};

/// Time slots one parametric stereo frame carries.
pub const SLOTS: usize = 32;
/// Input slots a 13-tap filter needs before the first output slot.
const HISTORY: usize = HYBRID_TAPS - 1;
/// Slots the symmetric 13-tap prototypes delay their input by.
pub const HYBRID_DELAY_SLOTS: usize = HYBRID_TAPS / 2;

/// Which of the two hybrid splits is in force.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resolution {
    /// Three QMF bands into ten sub-bands, for 20 parameter bins.
    Coarse,
    /// Five QMF bands into thirty-two sub-bands, for 34 parameter bins.
    Fine,
}

impl Resolution {
    /// Sub-bands the split produces.
    #[inline]
    pub const fn sub_bands(self) -> usize {
        match self {
            Self::Coarse => SUB_QMF_20,
            Self::Fine => SUB_QMF_34,
        }
    }

    /// QMF bands the split consumes; the rest pass through.
    #[inline]
    pub const fn qmf_bands(self) -> usize {
        match self {
            Self::Coarse => HYBRID_RES_20.len(),
            Self::Fine => HYBRID_RES_34.len(),
        }
    }
}

/// One QMF band's split: its filter and the input history that filter needs.
struct Band {
    /// `coeff[q][n]`: prototype tap `n` already multiplied by sub-band `q`'s modulation.
    coeff: Vec<[Complex32; HYBRID_TAPS]>,
    /// The twelve input slots preceding the frame.
    history: [Complex32; HISTORY],
    /// Where this band's sub-bands start in the hybrid layout.
    offset: usize,
}

/// The analysis and synthesis halves of one hybrid configuration.
///
/// Both configurations are kept alive by [`super::PsDecoder`] even when only one is
/// in use, so that a stream switching resolution mid-way finds warm history rather
/// than a click.
pub struct HybridFilterbank {
    resolution: Resolution,
    bands: Vec<Band>,
}

impl HybridFilterbank {
    /// Build a filterbank with cleared history.
    pub fn new(resolution: Resolution) -> Self {
        let splits: &[usize] =
            if resolution == Resolution::Coarse { &HYBRID_RES_20 } else { &HYBRID_RES_34 };

        let mut bands = Vec::with_capacity(splits.len());
        let mut offset = 0;
        for &width in splits {
            let prototype: &[f32; HYBRID_TAPS] = match (resolution, width) {
                (Resolution::Coarse, 8) => &HYBRID_P8_20,
                (Resolution::Coarse, _) => &HYBRID_P2_20,
                (Resolution::Fine, 12) => &HYBRID_P12_34,
                (Resolution::Fine, 8) => &HYBRID_P8_34,
                (Resolution::Fine, _) => &HYBRID_P4_34,
            };

            let mut coeff = Vec::with_capacity(width);
            for q in 0..width {
                let mut taps = [Complex32::default(); HYBRID_TAPS];
                for (n, tap) in taps.iter_mut().enumerate() {
                    let p = prototype[n];
                    // The two-band split is modulated by a real +-1 sequence; the
                    // wider ones by a complex exponential. Folding the prototype
                    // into the modulation here turns the run-time filter into a
                    // plain complex FIR.
                    *tap = match width {
                        2 => Complex32::new(p * MOD_2[q][n], 0.0),
                        4 => scale(MOD_4[q][n], p),
                        8 => scale(MOD_8[q][n], p),
                        _ => scale(MOD_12[q][n], p),
                    };
                }
                coeff.push(taps);
            }
            bands.push(Band { coeff, history: [Complex32::default(); HISTORY], offset });
            offset += width;
        }

        Self { resolution, bands }
    }

    /// The split this bank implements.
    #[inline]
    pub const fn resolution(&self) -> Resolution {
        self.resolution
    }

    /// Forget all history, as after a seek.
    pub fn reset(&mut self) {
        for band in &mut self.bands {
            band.history = [Complex32::default(); HISTORY];
        }
    }

    /// Split the lowest QMF bands of one frame into hybrid sub-bands.
    ///
    /// `frame[slot][band]` is one frame of QMF subband samples and `ahead` the
    /// [`HYBRID_DELAY_SLOTS`] slots that follow it; `output[slot][q]` receives the
    /// sub-bands, contiguous across the QMF bands that were split. Output slot `i`
    /// is centred on `frame[i]`, so the split introduces no delay.
    pub fn analyse(
        &mut self,
        frame: &[[Complex32; 64]],
        ahead: &[[Complex32; 64]],
        output: &mut [[Complex32; SUB_QMF_34]],
    ) {
        debug_assert_eq!(frame.len(), SLOTS);
        debug_assert_eq!(ahead.len(), HYBRID_DELAY_SLOTS);
        debug_assert_eq!(output.len(), SLOTS);

        // The filter reads a sliding window of 13 slots, so lay history and frame
        // out contiguously once rather than wrapping a ring buffer 32 times. The
        // window starts six slots into the frame, which is what centres the output.
        let mut work = [Complex32::default(); HISTORY + SLOTS];

        for (b, band) in self.bands.iter_mut().enumerate() {
            work[..HISTORY].copy_from_slice(&band.history);
            for (n, w) in work[HISTORY..].iter_mut().enumerate() {
                let source = n + HYBRID_DELAY_SLOTS;
                *w = if source < SLOTS { frame[source][b] } else { ahead[source - SLOTS][b] };
            }
            band.history.copy_from_slice(&work[SLOTS..]);

            for (q, taps) in band.coeff.iter().enumerate() {
                let out_band = band.offset + q;
                for (i, out) in output.iter_mut().enumerate() {
                    out[out_band] = dot(&work[i..i + HYBRID_TAPS], taps);
                }
            }
        }

        if self.resolution == Resolution::Coarse {
            fold_coarse(output);
        }
    }

    /// Sum hybrid sub-bands back into the QMF bands they came from.
    ///
    /// The split is critically sampled, so this is a plain sum: no filtering, and
    /// no further delay.
    pub fn synthesise(&self, hybrid: &[[Complex32; SUB_QMF_34]], output: &mut [[Complex32; 64]]) {
        debug_assert_eq!(hybrid.len(), output.len());

        for (out, src) in output.iter_mut().zip(hybrid.iter()) {
            for (b, band) in self.bands.iter().enumerate() {
                let mut sum = Complex32::default();
                for v in &src[band.offset..band.offset + band.coeff.len()] {
                    sum.re += v.re;
                    sum.im += v.im;
                }
                out[b] = sum;
            }
        }
    }
}

/// Fold the eight sub-bands of the coarse split's first QMF band down to six.
///
/// The coarse 8-band prototype is built so that two of its outputs are empty: the
/// pass-bands that would have held them are already folded onto sub-bands 3 and 2,
/// which respond to both signs of their centre frequency. Summing the pairs and
/// clearing the empty two makes that explicit, and leaves the six sub-bands the
/// parameter grid is defined on.
fn fold_coarse(hybrid: &mut [[Complex32; SUB_QMF_34]]) {
    for slot in hybrid {
        slot[3].re += slot[4].re;
        slot[3].im += slot[4].im;
        slot[2].re += slot[5].re;
        slot[2].im += slot[5].im;
        slot[4] = Complex32::default();
        slot[5] = Complex32::default();
    }
}

#[inline]
fn scale(v: Complex32, s: f32) -> Complex32 {
    Complex32::new(v.re * s, v.im * s)
}

/// Complex dot product of a 13-slot window with 13 taps.
#[inline]
fn dot(window: &[Complex32], taps: &[Complex32; HYBRID_TAPS]) -> Complex32 {
    let mut re = 0.0f32;
    let mut im = 0.0f32;
    for (w, t) in window.iter().zip(taps.iter()) {
        re += w.re * t.re - w.im * t.im;
        im += w.im * t.re + w.re * t.im;
    }
    Complex32::new(re, im)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pseudo-random complex QMF frame, deterministic across runs.
    fn frame(seed: u32) -> Vec<[Complex32; 64]> {
        let mut state = seed | 1;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1 << 23) as f32 - 1.0
        };
        (0..SLOTS)
            .map(|_| {
                let mut slot = [Complex32::default(); 64];
                for band in slot.iter_mut() {
                    *band = Complex32::new(next(), next());
                }
                slot
            })
            .collect()
    }

    /// The split is critically sampled: summing the sub-bands must give back the
    /// QMF band it came from, six slots late and otherwise untouched.
    fn reconstructs(resolution: Resolution) {
        let mut bank = HybridFilterbank::new(resolution);
        let mut hybrid = vec![[Complex32::default(); SUB_QMF_34]; SLOTS];
        let mut got = vec![[Complex32::default(); 64]; SLOTS];

        // Feed a continuous signal split into frames, so the look-ahead of one
        // frame is the head of the next and the filter never sees a discontinuity.
        let a = frame(1);
        let b = frame(2);
        let c = frame(3);
        bank.analyse(&a, &b[..HYBRID_DELAY_SLOTS], &mut hybrid);
        bank.synthesise(&hybrid, &mut got);
        bank.analyse(&b, &c[..HYBRID_DELAY_SLOTS], &mut hybrid);
        bank.synthesise(&hybrid, &mut got);

        for slot in 0..SLOTS {
            for band in 0..resolution.qmf_bands() {
                let want = b[slot][band];
                let have = got[slot][band];
                assert!(
                    (want.re - have.re).abs() < 1e-5 && (want.im - have.im).abs() < 1e-5,
                    "{resolution:?} slot {slot} band {band}: want {want:?}, got {have:?}"
                );
            }
        }
    }

    #[test]
    fn coarse_split_reconstructs_its_input() {
        reconstructs(Resolution::Coarse);
    }

    #[test]
    fn fine_split_reconstructs_its_input() {
        reconstructs(Resolution::Fine);
    }

    /// The sub-bands of a split must actually separate frequencies: a tone placed
    /// in one QMF band's lower half must not land in the sub-band covering its
    /// upper half.
    #[test]
    fn the_split_separates_frequencies() {
        let mut bank = HybridFilterbank::new(Resolution::Coarse);
        let mut hybrid = vec![[Complex32::default(); SUB_QMF_34]; SLOTS];

        // A slot-to-slot rotation of exp(i*2*pi*f) inside QMF band 0.
        let make = |f: f32| -> Vec<[Complex32; 64]> {
            (0..SLOTS)
                .map(|slot| {
                    let mut s = [Complex32::default(); 64];
                    let a = std::f32::consts::TAU * f * slot as f32;
                    s[0] = Complex32::new(a.cos(), a.sin());
                    s
                })
                .collect()
        };

        let mut energy = |f: f32| -> [f32; 6] {
            bank.reset();
            let ahead = make(f);
            for _ in 0..3 {
                let input = make(f);
                bank.analyse(&input, &ahead[..HYBRID_DELAY_SLOTS], &mut hybrid);
            }
            // The fold leaves six live sub-bands in the first QMF band; 4 and 5
            // have been added into 3 and 2 and zeroed.
            const LIVE: [usize; 6] = [0, 1, 2, 3, 6, 7];
            let mut e = [0.0f32; 6];
            for slot in &hybrid[HYBRID_DELAY_SLOTS..] {
                for (acc, &q) in e.iter_mut().zip(LIVE.iter()) {
                    *acc += slot[q].re * slot[q].re + slot[q].im * slot[q].im;
                }
            }
            e
        };

        // Sub-band centres are 1/8 apart, so these two tones sit four apart.
        let low = energy(0.0625);
        let high = energy(-0.3125);
        let low_peak = low.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        let high_peak = high.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        assert_ne!(low_peak, high_peak, "both tones landed in sub-band {low_peak}");
    }
}
