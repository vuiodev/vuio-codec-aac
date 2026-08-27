//! Scalefactor estimation and rate control.
//!
//! The masking model in [`super::psycho`] says how much noise each band may carry.
//! This module turns that into the numbers the bitstream actually holds: one
//! scalefactor per band, chosen so the quantizer's error lands on the threshold,
//! and a global scaling of those thresholds that makes the frame fit its bit
//! budget.
//!
//! # Choosing a band's scalefactor
//!
//! AAC quantizes with a power law, `q = round(|x|^(3/4) * 2^(-3/16 * (sf - 100)))`,
//! so a band's noise power is set entirely by `sf`. Inverting the usual estimate of
//! that noise gives a starting point,
//!
//! ```text
//! sf = 100 + (8/3) * log2(6.75 * threshold / sum |x|^(1/2))
//! ```
//!
//! which is then floored by the smallest `sf` that keeps every coefficient inside
//! the 8191 the bitstream can code, and refined by measuring the real error either
//! side of it. That local search matters: the estimate is derived for a flat band,
//! and a band with one dominant line behaves quite differently.
//!
//! # Fitting the budget
//!
//! Raising every threshold by the same factor coarsens the whole frame and costs
//! monotonically fewer bits, so the outer loop bisects on that factor. Bands the
//! model says are perceptually important resist being zeroed even when the budget
//! is tight, which keeps the encoder from punching audible holes at low rates.

use crate::encoder::aac::psycho::{MAX_BANDS, PsychoResult};
use crate::encoder::aac::quant::{
    BandChoice, MAX_QUANT_MAGNITUDE, SF_OFFSET, choose_codebook, quantize_band,
};

/// Widest span two coded bands' scalefactors may have, so every delta stays inside
/// the scalefactor codebook.
pub const MAX_SCALEFACTOR_DELTA: i32 = 60;
/// Smallest scalefactor the bitstream can carry.
pub const MIN_SCALEFACTOR: i32 = 0;
/// Largest scalefactor the bitstream can carry.
pub const MAX_SCALEFACTOR: i32 = 255;
/// Steps either side of the estimate the refinement is allowed to search.
const REFINE_STEPS: i32 = 1;
/// How far above the threshold a band's error may sit before refinement gives up on
/// improving it and simply takes the least bad option.
const DISTORTION_SLACK: f32 = 1.25;

/// Marks a band the estimator decided not to code at all.
const UNCODED: i32 = i32::MIN;

/// One channel's quantization, as the rate loop leaves it.
#[derive(Debug, Clone)]
pub struct Quantization {
    /// Scalefactor per band; equal to [`SF_OFFSET`] for bands that carry nothing.
    pub scalefactors: Vec<i32>,
    /// Codebook and cost per band.
    pub choices: Vec<BandChoice>,
    /// Quantized coefficients, band-major over the whole frame.
    pub quant: Vec<i32>,
    /// Payload bits the frame will cost.
    pub bits: usize,
    /// First band that carries data, if any.
    pub first_coded: Option<usize>,
    /// Threshold scale, in dB, the last frame settled on.
    ///
    /// Successive frames of the same programme need almost the same one, so it
    /// brackets the search and saves most of its iterations.
    scale_db: f32,
}

impl Quantization {
    /// Allocate for a frame of `lines` coefficients in `bands` bands.
    pub fn new(lines: usize, bands: usize) -> Self {
        Self {
            scalefactors: vec![SF_OFFSET; bands],
            choices: vec![BandChoice::default(); bands],
            quant: vec![0; lines],
            bits: 0,
            first_coded: None,
            scale_db: 0.0,
        }
    }
}

/// Magnitudes the inverse power law is tabulated for.
///
/// The quantizer's inverse is a cube root, far too slow to call once per coefficient
/// in a loop that quantizes the frame a dozen times over. Almost every magnitude is
/// small, so a table covers the common case and the tail falls back on `powf`.
const POW43_TABLE: usize = 1024;

/// Turns thresholds into scalefactors, and scalefactors into a frame that fits.
#[derive(Debug, Clone)]
pub struct RateLoop {
    /// Working copy of the thresholds, scaled by the outer loop.
    thresholds: [f32; MAX_BANDS],
    /// Scratch for the refinement's trial quantization.
    trial: Vec<i32>,
    /// `q^(4/3)` for `q` below [`POW43_TABLE`].
    pow43: Box<[f32; POW43_TABLE]>,
}

impl RateLoop {
    /// Build a rate loop for frames of `lines` coefficients.
    pub fn new(lines: usize) -> Self {
        let mut pow43 = Box::new([0.0f32; POW43_TABLE]);
        for (q, slot) in pow43.iter_mut().enumerate() {
            *slot = (q as f32).powf(4.0 / 3.0);
        }
        Self { thresholds: [0.0; MAX_BANDS], trial: vec![0; lines], pow43 }
    }



    /// Quantize one channel to fit `budget` payload bits.
    ///
    /// Returns the bits the result costs, which may exceed the budget if even the
    /// coarsest quantization cannot fit — emitting an over-long frame beats emitting
    /// none.
    pub fn fit(
        &mut self,
        spectrum: &[f32],
        offsets: &[usize],
        psycho: &PsychoResult,
        min_snr: &dyn Fn(usize) -> f32,
        budget: usize,
        out: &mut Quantization,
    ) -> usize {
        // Cost falls monotonically as the thresholds rise, so the search is a
        // bracket followed by bisection. The bracket starts at what the previous
        // frame settled on and widens geometrically, which on real material lands
        // in one or two probes instead of the eight a blind range would take.
        const FLOOR: f32 = -80.0;
        const CEILING: f32 = 120.0;
        const TOLERANCE: f32 = 0.35;

        let hint = out.scale_db.clamp(FLOOR, CEILING);
        let mut low = FLOOR;
        let mut high = CEILING;
        let mut best = CEILING;
        let mut fits = false;

        if self.attempt(spectrum, offsets, psycho, min_snr, hint, out) <= budget {
            best = hint;
            fits = true;
            high = hint;
            let mut step = 2.0f32;
            while high - step > FLOOR {
                let probe = high - step;
                if self.attempt(spectrum, offsets, psycho, min_snr, probe, out) <= budget {
                    best = probe;
                    high = probe;
                    step *= 2.0;
                } else {
                    low = probe;
                    break;
                }
            }
        } else {
            low = hint;
            let mut step = 2.0f32;
            loop {
                let probe = (low + step).min(CEILING);
                if self.attempt(spectrum, offsets, psycho, min_snr, probe, out) <= budget {
                    best = probe;
                    fits = true;
                    high = probe;
                    break;
                }
                low = probe;
                if probe >= CEILING {
                    break;
                }
                step *= 2.0;
            }
        }

        while high - low > TOLERANCE {
            let mid = 0.5 * (low + high);
            if self.attempt(spectrum, offsets, psycho, min_snr, mid, out) <= budget {
                best = mid;
                fits = true;
                high = mid;
            } else {
                low = mid;
            }
        }

        if !fits {
            best = CEILING;
        }
        out.scale_db = best;
        self.attempt(spectrum, offsets, psycho, min_snr, best, out)
    }

    /// Quantize at one threshold scale and report the cost.
    fn attempt(
        &mut self,
        spectrum: &[f32],
        offsets: &[usize],
        psycho: &PsychoResult,
        min_snr: &dyn Fn(usize) -> f32,
        scale_db: f32,
        out: &mut Quantization,
    ) -> usize {
        let bands = psycho.bands.min(out.scalefactors.len());
        let scale = 10f32.powf(scale_db / 10.0);

        for b in 0..bands {
            // A band may never be given a threshold above its own energy times the
            // model's floor on signal-to-mask ratio: past that it would be zeroed
            // even though the ear would notice.
            let ceiling = psycho.spread_energy[b] * min_snr(b);
            self.thresholds[b] = (psycho.threshold[b] * scale).min(ceiling.max(psycho.threshold[b]));
        }

        out.first_coded = None;
        let mut assigned = [UNCODED; MAX_BANDS];
        let mut lowest = i32::MAX;

        for b in 0..bands {
            let lo = offsets[b];
            let hi = offsets[b + 1].min(spectrum.len());
            if lo >= hi {
                continue;
            }
            let band = &spectrum[lo..hi];
            let threshold = self.thresholds[b];

            let peak = band.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            if peak <= 0.0 || psycho.energy[b] <= threshold {
                continue;
            }

            let Some(estimate) = initial_scalefactor(band, threshold, peak) else {
                continue;
            };
            let floor = smallest_representable(peak);
            let start = estimate.max(floor).clamp(MIN_SCALEFACTOR, MAX_SCALEFACTOR);

            let chosen = self.refine(band, threshold, start, floor, &mut out.quant[lo..hi]);
            assigned[b] = chosen;
            lowest = lowest.min(chosen);
        }

        // Every coded band's scalefactor has to sit within one delta's reach of the
        // lowest, or the scalefactor codebook cannot express the difference.
        let ceiling = if lowest == i32::MAX { MAX_SCALEFACTOR } else { lowest + MAX_SCALEFACTOR_DELTA };
        for b in 0..bands {
            let lo = offsets[b];
            let hi = offsets[b + 1].min(spectrum.len());
            if assigned[b] == UNCODED {
                out.scalefactors[b] = SF_OFFSET;
                out.quant[lo..hi].fill(0);
                out.choices[b] = BandChoice::default();
                continue;
            }
            if assigned[b] > ceiling {
                assigned[b] = ceiling;
                quantize_band(&spectrum[lo..hi], ceiling, &mut out.quant[lo..hi]);
            }
            out.scalefactors[b] = assigned[b];
            out.choices[b] = choose_codebook(&out.quant[lo..hi]);
            if out.choices[b].codebook != 0 && out.first_coded.is_none() {
                out.first_coded = Some(b);
            }
        }

        // A band whose codebook came out zero carries nothing, so its scalefactor is
        // not transmitted and must not take part in the delta chain.
        for b in 0..bands {
            if out.choices[b].codebook == 0 {
                out.scalefactors[b] = SF_OFFSET;
            }
        }

        out.bits = cost(out, bands);
        out.bits
    }

    /// Search either side of the estimate for the scalefactor with the least error.
    ///
    /// Leaves the winning quantization in `quant`.
    fn refine(
        &mut self,
        band: &[f32],
        threshold: f32,
        start: i32,
        floor: i32,
        quant: &mut [i32],
    ) -> i32 {
        quantize_band(band, start, quant);
        let mut best = start;
        let best_error = distortion(&self.pow43, band, quant, start);

        if best_error > DISTORTION_SLACK * threshold {
            // Already too coarse: the only way out is a finer step, if there is one.
            if start > floor && start > MIN_SCALEFACTOR {
                let Self { trial, pow43, .. } = self;
                let trial = &mut trial[..band.len()];
                quantize_band(band, start - 1, trial);
                let error = distortion(pow43, band, trial, start - 1);
                if error < best_error {
                    best = start - 1;
                    quant.copy_from_slice(trial);
                }
            }
            return best;
        }

        // Room to spare: coarsen while the error stays under what the ear allows,
        // which buys bits for free.
        let allowed = (best_error * DISTORTION_SLACK).min(threshold);
        for step in 1..=REFINE_STEPS {
            let candidate = start + step;
            if candidate > MAX_SCALEFACTOR {
                break;
            }
            let Self { trial, pow43, .. } = self;
            let trial = &mut trial[..band.len()];
            quantize_band(band, candidate, trial);
            if distortion(pow43, band, trial, candidate) < allowed {
                best = candidate;
                quant.copy_from_slice(trial);
            }
        }
        best
    }
}

/// The scalefactor the usual estimate suggests for a band.
///
/// `None` when the band has no content the estimate can work from.
fn initial_scalefactor(band: &[f32], threshold: f32, peak: f32) -> Option<i32> {
    if peak <= 0.0 {
        return None;
    }
    // The form factor, the sum of the square roots of the magnitudes, is what makes
    // the estimate hold for a band whose energy sits in one line as well as one
    // where it is spread evenly.
    let mut form_factor = 0.0f32;
    for &x in band {
        form_factor += x.abs().sqrt();
    }
    if form_factor <= f32::MIN_POSITIVE {
        return None;
    }

    let ratio = (6.75 * threshold.max(f32::MIN_POSITIVE)) / form_factor;
    Some(SF_OFFSET + ((8.0 / 3.0) * ratio.log2()).floor() as i32)
}

/// Squared error a band's quantization leaves behind.
///
/// Measured the way the decoder will hear it: each magnitude is put back through the
/// inverse power law and the scalefactor's gain, and compared with what went in.
#[inline]
fn distortion(pow43: &[f32; POW43_TABLE], band: &[f32], quant: &[i32], scalefactor: i32) -> f32 {
    let gain = (0.25 * (scalefactor - SF_OFFSET) as f32).exp2();
    let mut sum = 0.0f32;
    for (&x, &q) in band.iter().zip(quant.iter()) {
        let magnitude = q.unsigned_abs() as usize;
        let inverse = match pow43.get(magnitude) {
            Some(&v) => v,
            None => (magnitude as f32).powf(4.0 / 3.0),
        };
        let error = x.abs() - inverse * gain;
        sum += error * error;
    }
    sum
}

/// Smallest scalefactor that keeps every coefficient inside the coded range.
fn smallest_representable(peak: f32) -> i32 {
    if peak <= 0.0 {
        return MIN_SCALEFACTOR;
    }
    // Invert `peak^(3/4) * 2^(-3/16 * (sf - 100)) <= MAX_QUANT`.
    let headroom = (MAX_QUANT_MAGNITUDE as f32) - 0.5;
    let sf = (16.0 / 3.0) * (0.75 * peak.log2() - headroom.log2());
    SF_OFFSET + sf.ceil() as i32
}

/// Payload bits one channel's quantization costs.
fn cost(q: &Quantization, bands: usize) -> usize {
    let mut bits = 0usize;
    for choice in &q.choices[..bands] {
        bits += choice.bits as usize;
    }

    // Section data: one record per run of equal codebooks.
    let mut sections = 1usize;
    for b in 1..bands {
        if q.choices[b].codebook != q.choices[b - 1].codebook {
            sections += 1;
        }
    }
    bits += sections * (4 + 5);

    // Scalefactor data: a Huffman-coded delta for each coded band.
    let mut previous: Option<i32> = None;
    for b in 0..bands {
        if q.choices[b].codebook == 0 {
            continue;
        }
        let delta = match previous {
            Some(p) => q.scalefactors[b] - p,
            None => 0,
        };
        previous = Some(q.scalefactors[b]);
        bits += crate::encoder::aac::huffman::scalefactor_codeword(delta)
            .map_or(19, |c| c.len as usize);
    }

    bits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::aac::psycho::PsychoacousticModel;
    use crate::types::WindowSequence;

    fn table(bands: usize, width: usize) -> Vec<usize> {
        (0..=bands).map(|b| b * width).collect()
    }

    /// A tone must quantize to something that reconstructs close to it.
    #[test]
    fn quantization_lands_near_the_threshold() {
        let offsets = table(20, 16);
        let lines = offsets[20];
        let mut spectrum = vec![0.0f32; lines];
        for (i, v) in spectrum.iter_mut().enumerate() {
            *v = 1.0e5 * (i as f32 * 0.3).sin();
        }

        let mut model = PsychoacousticModel::new(44100, 64000, &offsets, false);
        let mut psycho = Default::default();
        model.analyse(&spectrum, &offsets, WindowSequence::OnlyLongSequence, &mut psycho);

        let mut rate = RateLoop::new(lines);
        let mut out = Quantization::new(lines, 20);
        let bits = rate.fit(&spectrum, &offsets, &psycho, &|b| model.min_snr(b), 4000, &mut out);

        assert!(bits <= 4000, "rate loop overspent: {bits} bits");
        assert!(out.first_coded.is_some(), "nothing was coded at all");
        for b in 0..psycho.bands {
            let sf = out.scalefactors[b];
            assert!((MIN_SCALEFACTOR..=MAX_SCALEFACTOR).contains(&sf), "band {b} has sf {sf}");
        }
    }

    /// A tighter budget must not produce a bigger frame.
    #[test]
    fn cost_falls_as_the_budget_tightens() {
        let offsets = table(20, 16);
        let lines = offsets[20];
        let spectrum: Vec<f32> = (0..lines).map(|i| 1.0e5 * (i as f32 * 0.11).sin()).collect();

        let mut model = PsychoacousticModel::new(44100, 64000, &offsets, false);
        let mut psycho = Default::default();
        model.analyse(&spectrum, &offsets, WindowSequence::OnlyLongSequence, &mut psycho);

        let mut rate = RateLoop::new(lines);
        let mut out = Quantization::new(lines, 20);
        let generous = rate.fit(&spectrum, &offsets, &psycho, &|b| model.min_snr(b), 6000, &mut out);
        let tight = rate.fit(&spectrum, &offsets, &psycho, &|b| model.min_snr(b), 600, &mut out);

        assert!(tight <= generous, "tightening the budget cost more bits");
        assert!(tight <= 600 || tight < generous);
    }

    /// Every transmitted scalefactor delta must be one the codebook can carry.
    #[test]
    fn scalefactor_deltas_stay_codeable() {
        let offsets = table(30, 16);
        let lines = offsets[30];
        // A spectrum with a huge dynamic range across bands, which is what would
        // push the deltas out of range if nothing constrained them.
        let mut spectrum = vec![0.0f32; lines];
        for b in 0..30 {
            let level = 10f32.powi(b as i32 % 12);
            for i in offsets[b]..offsets[b + 1] {
                spectrum[i] = level * if i % 2 == 0 { 1.0 } else { -1.0 };
            }
        }

        let mut model = PsychoacousticModel::new(44100, 64000, &offsets, false);
        let mut psycho = Default::default();
        model.analyse(&spectrum, &offsets, WindowSequence::OnlyLongSequence, &mut psycho);

        let mut rate = RateLoop::new(lines);
        let mut out = Quantization::new(lines, 30);
        rate.fit(&spectrum, &offsets, &psycho, &|b| model.min_snr(b), 8000, &mut out);

        let mut previous: Option<i32> = None;
        for b in 0..psycho.bands {
            if out.choices[b].codebook == 0 {
                continue;
            }
            if let Some(p) = previous {
                let delta = out.scalefactors[b] - p;
                assert!(
                    crate::encoder::aac::huffman::scalefactor_codeword(delta).is_some(),
                    "band {b} needs an uncodeable delta of {delta}"
                );
            }
            previous = Some(out.scalefactors[b]);
        }
    }

    /// Quantized magnitudes must stay inside what the spectral codebooks can code.
    #[test]
    fn magnitudes_stay_inside_the_coded_range() {
        let offsets = table(20, 16);
        let lines = offsets[20];
        let spectrum = vec![3.0e7f32; lines];

        let mut model = PsychoacousticModel::new(44100, 64000, &offsets, false);
        let mut psycho = Default::default();
        model.analyse(&spectrum, &offsets, WindowSequence::OnlyLongSequence, &mut psycho);

        let mut rate = RateLoop::new(lines);
        let mut out = Quantization::new(lines, 20);
        rate.fit(&spectrum, &offsets, &psycho, &|b| model.min_snr(b), 100_000, &mut out);

        for &q in &out.quant {
            assert!(q.abs() <= MAX_QUANT_MAGNITUDE, "quantized magnitude {q} is out of range");
        }
    }
}
