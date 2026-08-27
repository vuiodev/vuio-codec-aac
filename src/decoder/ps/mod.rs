//! Parametric stereo.
//!
//! Parametric stereo is what makes HE-AAC v2: the encoder mixes its two channels
//! down to one, codes that mono signal with AAC and SBR, and spends a few hundred
//! bits per frame describing the stereo image it threw away. The decoder rebuilds
//! the image by mixing the downmix with a *decorrelated* copy of itself.
//!
//! The chain, per frame, all of it in the QMF domain that SBR already produced:
//!
//! 1. [`hybrid`] splits the lowest QMF bands finer, so the parameter grid can follow
//!    the ear at low frequency;
//! 2. [`Decorrelator`] runs the signal through a chain of frequency-dependent
//!    all-pass sections, giving a second signal with the same spectrum but no
//!    correlation to the first;
//! 3. a 2x2 matrix built from the transmitted level difference and coherence mixes
//!    the two into left and right, interpolated across each envelope;
//! 4. the hybrid split is summed back, leaving two ordinary QMF frames.
//!
//! # Transient handling
//!
//! An all-pass chain smears energy in time, which on a transient is heard as a
//! doubled attack. Step 2 therefore tracks a fast decaying peak per parameter band
//! and, where the signal is rising faster than the peak follows, fades the
//! decorrelated copy out — see [`TransientDetector`].
//!
//! # Delay
//!
//! None of its own. The hybrid filterbank is centred, so it reads six QMF slots past
//! the end of the frame rather than delaying everything that passes it by; see
//! [`hybrid`]. Parametric stereo therefore leaves the decoder's delay exactly where
//! band replication left it.

pub mod data;
pub mod hybrid;

use crate::bitstream::BitReader;
use crate::dsp::fft::Complex32;
use crate::error::Result;
use crate::tables::ps::{
    AP_LEAD_DELAY, AP_LINKS, AP_LINK_DECAY, AP_LINK_DELAY, BINS_20, BINS_34, BIN_OF_GROUP_20,
    BIN_OF_GROUP_34, FIRST_DELAY_BAND, GROUP_BORDERS_20, GROUP_BORDERS_34, HYBRID_DELAY, ICC_ALPHA,
    ICC_RHO, IID_SCALE, IID_SCALE_FINE, IID_STEPS, IID_STEPS_DB, IID_STEPS_DB_FINE, IID_STEPS_FINE,
    MAX_AP_DELAY, MAX_QMF_DELAY, PHI_FRACT_QMF, PHI_FRACT_SUB_20, PHI_FRACT_SUB_34, QMF_BANDS,
    QMF_DELAY_LEN, Q_FRACT_QMF, Q_FRACT_SUB_20, Q_FRACT_SUB_34, SUBQMF_GROUPS_20,
    SUBQMF_GROUPS_34, SUB_QMF_34,
};

pub use data::{PsData, PsParser};
pub use hybrid::{HybridFilterbank, Resolution, SLOTS};

/// One frame of QMF subband samples: [`SLOTS`] time slots of [`QMF_BANDS`] subbands.
pub type QmfSlot = [Complex32; QMF_BANDS];
/// One frame of hybrid subband samples.
type HybridSlot = [Complex32; SUB_QMF_34];

/// How fast the transient detector's peak follower decays, per slot.
const PEAK_DECAY: f32 = 0.765_928_3;
/// Weight of each new slot in the detector's two smoothing filters.
const SMOOTH: f32 = 0.25;
/// How far the detector lets the peak run ahead of the average before it fades the
/// decorrelated copy out.
const TRANSIENT_MARGIN: f32 = 1.5;
/// Highest QMF subband whose all-pass chain runs at full strength.
const DECAY_CUTOFF_COARSE: f32 = 3.0;
/// Same, for the fine configuration, whose hybrid split reaches higher.
const DECAY_CUTOFF_FINE: f32 = 5.0;
/// How fast the all-pass chain is faded out above the cutoff, per subband.
const DECAY_SLOPE: f32 = 0.05;

/// One parameter band's 2x2 mixing matrix.
///
/// Real, because the phase parameters the standard's extension can carry are not
/// transmitted by any encoder this port follows; see [`data`].
#[derive(Debug, Clone, Copy, PartialEq)]
struct Mix {
    /// Downmix into left.
    h11: f32,
    /// Downmix into right.
    h12: f32,
    /// Decorrelated signal into left.
    h21: f32,
    /// Decorrelated signal into right.
    h22: f32,
}

impl Default for Mix {
    /// The matrix a decoder starts from: both outputs are the downmix, unaltered.
    fn default() -> Self {
        Self { h11: 1.0, h12: 1.0, h21: 0.0, h22: 0.0 }
    }
}

impl Mix {
    /// Step every coefficient by one slot's worth of the way to `target`.
    #[inline]
    fn step(&mut self, delta: &Mix) {
        self.h11 += delta.h11;
        self.h12 += delta.h12;
        self.h21 += delta.h21;
        self.h22 += delta.h22;
    }

    /// The per-slot increment that reaches `self` from `from` in `slots` steps.
    #[inline]
    fn ramp_from(&self, from: &Mix, slots: usize) -> Mix {
        let n = slots.max(1) as f32;
        Mix {
            h11: (self.h11 - from.h11) / n,
            h12: (self.h12 - from.h12) / n,
            h21: (self.h21 - from.h21) / n,
            h22: (self.h22 - from.h22) / n,
        }
    }
}

/// Tracks how transient each parameter band is, slot by slot.
///
/// The decorrelated signal is a smeared version of the downmix, so on an attack it
/// arrives late and is heard as a second, softer strike. Comparing a fast-decaying
/// peak against a smoothed average finds those moments; where the peak is well
/// ahead, the gain returned falls below one and the copy is faded out.
struct TransientDetector {
    peak: [f32; BINS_34],
    energy: [f32; BINS_34],
    peak_excess: [f32; BINS_34],
}

impl TransientDetector {
    fn new() -> Self {
        Self { peak: [0.0; BINS_34], energy: [0.0; BINS_34], peak_excess: [0.0; BINS_34] }
    }

    fn reset(&mut self) {
        *self = Self::new();
    }

    /// Advance one slot of one band and return the gain the decorrelator should use.
    #[inline]
    fn gain(&mut self, bin: usize, power: f32) -> f32 {
        let peak = (self.peak[bin] * PEAK_DECAY).max(power);
        self.peak[bin] = peak;

        let excess = self.peak_excess[bin] + SMOOTH * (peak - power - self.peak_excess[bin]);
        self.peak_excess[bin] = excess;

        let energy = self.energy[bin] + SMOOTH * (power - self.energy[bin]);
        self.energy[bin] = energy;

        let threshold = TRANSIENT_MARGIN * excess;
        if threshold <= energy { 1.0 } else { energy / threshold }
    }
}

/// The all-pass chain and delay lines that turn the downmix into a signal
/// uncorrelated with it.
///
/// Low subbands get a cascade of fractional-delay all-pass sections, which scramble
/// phase without touching the magnitude spectrum. Higher subbands, where the ear
/// judges coherence by envelope rather than phase, get a plain delay instead —
/// fourteen slots up to subband 35 and one above it.
struct Decorrelator {
    /// Leading section's delay line, hybrid subbands.
    sub_lead: [HybridSlot; AP_LEAD_DELAY],
    /// Leading section's delay line, QMF subbands.
    qmf_lead: [QmfSlot; AP_LEAD_DELAY],
    /// Serial sections' delay lines, hybrid subbands.
    sub_links: [[HybridSlot; MAX_AP_DELAY]; AP_LINKS],
    /// Serial sections' delay lines, QMF subbands.
    qmf_links: [[QmfSlot; MAX_AP_DELAY]; AP_LINKS],
    /// Plain delay line for the subbands above the all-pass range.
    plain: [QmfSlot; MAX_QMF_DELAY],
    /// Read cursor into [`Self::plain`], one per subband because the lengths differ.
    plain_cursor: [usize; QMF_BANDS],
    lead_cursor: usize,
    link_cursor: [usize; AP_LINKS],
}

impl Decorrelator {
    fn new() -> Self {
        Self {
            sub_lead: [[Complex32::default(); SUB_QMF_34]; AP_LEAD_DELAY],
            qmf_lead: [[Complex32::default(); QMF_BANDS]; AP_LEAD_DELAY],
            sub_links: [[[Complex32::default(); SUB_QMF_34]; MAX_AP_DELAY]; AP_LINKS],
            qmf_links: [[[Complex32::default(); QMF_BANDS]; MAX_AP_DELAY]; AP_LINKS],
            plain: [[Complex32::default(); QMF_BANDS]; MAX_QMF_DELAY],
            plain_cursor: [0; QMF_BANDS],
            lead_cursor: 0,
            link_cursor: [0; AP_LINKS],
        }
    }

    fn reset(&mut self) {
        *self = Self::new();
    }
}

/// One all-pass section: read the delay line, rotate, mix against the input.
///
/// `line` is the section's delay slot for this subband, updated in place.
#[inline]
fn all_pass(input: Complex32, line: &mut Complex32, phase: Complex32, decay: f32) -> Complex32 {
    let delayed = rotate(*line, phase);
    let out = Complex32::new(delayed.re - decay * input.re, delayed.im - decay * input.im);
    *line = Complex32::new(input.re + decay * out.re, input.im + decay * out.im);
    out
}

#[inline]
fn rotate(v: Complex32, phase: Complex32) -> Complex32 {
    Complex32::new(v.re * phase.re - v.im * phase.im, v.re * phase.im + v.im * phase.re)
}

#[inline]
fn power(v: Complex32) -> f32 {
    v.re * v.re + v.im * v.im
}

/// Parametric stereo for one channel element.
pub struct PsDecoder {
    parser: PsParser,
    data: PsData,
    /// Whether a payload has been parsed for the frame about to be processed.
    have_data: bool,
    coarse: HybridFilterbank,
    fine: HybridFilterbank,
    decorrelator: Decorrelator,
    transient: TransientDetector,
    /// Mixing matrix each bin ended the previous envelope on.
    previous: [Mix; BINS_34],
    /// Resolution the previous frame used, so a change can reset what depends on it.
    previous_resolution: Resolution,
    left: Vec<HybridSlot>,
    right: Vec<HybridSlot>,
    /// Transient gain per slot and bin, filled by the decorrelator.
    gains: Vec<[f32; BINS_34]>,
    /// Band energies the transient detector works from.
    powers: Vec<[f32; BINS_34]>,
}

impl Default for PsDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl PsDecoder {
    /// Build a decoder that has seen no payload.
    pub fn new() -> Self {
        Self {
            parser: PsParser::new(),
            data: PsData::default(),
            have_data: false,
            coarse: HybridFilterbank::new(Resolution::Coarse),
            fine: HybridFilterbank::new(Resolution::Fine),
            decorrelator: Decorrelator::new(),
            transient: TransientDetector::new(),
            previous: [Mix::default(); BINS_34],
            previous_resolution: Resolution::Coarse,
            left: vec![[Complex32::default(); SUB_QMF_34]; SLOTS],
            right: vec![[Complex32::default(); SUB_QMF_34]; SLOTS],
            gains: vec![[0.0; BINS_34]; SLOTS],
            powers: vec![[0.0; BINS_34]; SLOTS],
        }
    }

    /// Whether a payload has ever been parsed, so that [`Self::process`] will do
    /// more than pass the downmix through.
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.parser.is_ready()
    }

    /// Forget all inter-frame state, as after a seek.
    pub fn reset(&mut self) {
        self.parser.reset();
        self.data = PsData::default();
        self.have_data = false;
        self.coarse.reset();
        self.fine.reset();
        self.decorrelator.reset();
        self.transient.reset();
        self.previous = [Mix::default(); BINS_34];
    }

    /// Parse one `ps_data()` payload from an SBR extension field.
    ///
    /// `available_bits` is what remains of the extension, which bounds the
    /// length-prefixed sub-extension the payload may end with.
    pub fn parse(&mut self, reader: &mut BitReader, available_bits: usize) -> Result<()> {
        self.data = self.parser.parse(reader, available_bits)?;
        self.have_data = true;
        Ok(())
    }

    /// Rebuild a stereo pair from one frame of the mono downmix.
    ///
    /// `left` arrives holding the downmix and leaves holding the left channel;
    /// `right` is written. `ahead` is the [`HYBRID_DELAY`] slots that follow the
    /// frame, which the hybrid filterbank needs to stay centred; only the lowest
    /// few subbands of it are read.
    pub fn process(&mut self, left: &mut [QmfSlot], ahead: &[QmfSlot], right: &mut [QmfSlot]) {
        debug_assert_eq!(left.len(), SLOTS);
        debug_assert_eq!(ahead.len(), HYBRID_DELAY);
        debug_assert_eq!(right.len(), SLOTS);

        if !self.have_data {
            // No payload this frame: hold the last envelope's parameters across the
            // whole of it, rather than replaying the previous frame's envelope
            // structure at the wrong times.
            let last = self.data.envelopes - 1;
            self.data.iid[0] = self.data.iid[last];
            self.data.icc[0] = self.data.icc[last];
            self.data.envelopes = 1;
            self.data.borders[0] = 0;
            self.data.borders[1] = SLOTS;
        }

        let resolution = if self.have_data { self.data.resolution } else { self.previous_resolution };
        if resolution != self.previous_resolution {
            self.adopt_resolution(resolution);
        }

        let bank = match resolution {
            Resolution::Coarse => &mut self.coarse,
            Resolution::Fine => &mut self.fine,
        };
        let split_bands = bank.resolution().qmf_bands();
        let sub_bands = bank.resolution().sub_bands();

        bank.analyse(left, ahead, &mut self.left);

        self.decorrelate(resolution, left, right, sub_bands, split_bands);
        self.remix(resolution, left, right, sub_bands, split_bands);

        let bank = match resolution {
            Resolution::Coarse => &self.coarse,
            Resolution::Fine => &self.fine,
        };
        bank.synthesise(&self.left, left);
        bank.synthesise(&self.right, right);

        self.have_data = false;
    }

    /// Adopt a new hybrid split, carrying across what can be carried.
    ///
    /// The delay lines belong to the old band layout and cannot be reinterpreted, so
    /// they start again; the mixing matrices are per parameter bin and are resampled,
    /// which keeps the stereo image continuous across the change.
    fn adopt_resolution(&mut self, resolution: Resolution) {
        match resolution {
            Resolution::Fine => {
                self.fine.reset();
                widen_mix(&mut self.previous);
            }
            Resolution::Coarse => {
                self.coarse.reset();
                narrow_mix(&mut self.previous);
            }
        }
        self.decorrelator.reset();
        self.transient.reset();
        self.previous_resolution = resolution;
    }

    /// Build the decorrelated copy of the downmix, in both domains.
    fn decorrelate(
        &mut self,
        resolution: Resolution,
        qmf: &[QmfSlot],
        qmf_out: &mut [QmfSlot],
        sub_bands: usize,
        split_bands: usize,
    ) {
        let (borders, bins, sub_groups) = grid(resolution);
        let bin_of = bin_map(resolution);

        self.measure_transients(qmf, borders, bins, sub_groups, bin_of, split_bands);

        let (phi_sub, q_sub): (&[Complex32], &[[Complex32; AP_LINKS]]) = match resolution {
            Resolution::Coarse => (&PHI_FRACT_SUB_20, &Q_FRACT_SUB_20),
            Resolution::Fine => (&PHI_FRACT_SUB_34, &Q_FRACT_SUB_34),
        };

        // Hybrid subbands: the whole all-pass chain, at full strength.
        for group in 0..sub_groups {
            let band = borders[group];
            if band >= sub_bands {
                continue;
            }
            let bin = bin_of[group];
            let mut lead = self.decorrelator.lead_cursor;
            let mut links = self.decorrelator.link_cursor;

            for slot in 0..SLOTS {
                let mut v = all_pass(
                    self.left[slot][band],
                    &mut self.decorrelator.sub_lead[lead][band],
                    phi_sub[band],
                    0.0,
                );
                for m in 0..AP_LINKS {
                    v = all_pass(
                        v,
                        &mut self.decorrelator.sub_links[m][links[m]][band],
                        q_sub[band][m],
                        AP_LINK_DECAY[m],
                    );
                }
                let g = self.gains[slot][bin];
                self.right[slot][band] = Complex32::new(v.re * g, v.im * g);
                advance(&mut lead, AP_LEAD_DELAY, &mut links);
            }
        }

        // QMF subbands: the all-pass chain fades out with frequency, and above
        // subband 23 gives way to a plain delay.
        let cutoff = match resolution {
            Resolution::Coarse => DECAY_CUTOFF_COARSE,
            Resolution::Fine => DECAY_CUTOFF_FINE,
        };
        for group in sub_groups..borders.len() - 1 {
            let bin = bin_of[group];
            for band in borders[group].max(split_bands)..borders[group + 1].min(QMF_BANDS) {
                let decay_scale = if (band as f32) <= cutoff {
                    1.0
                } else {
                    (1.0 + cutoff * DECAY_SLOPE - DECAY_SLOPE * band as f32).max(0.0)
                };
                let mut lead = self.decorrelator.lead_cursor;
                let mut links = self.decorrelator.link_cursor;
                let mut plain = self.decorrelator.plain_cursor[band];
                let plain_len = QMF_DELAY_LEN[band].max(1);

                for slot in 0..SLOTS {
                    let input = qmf[slot][band];
                    let v = if band >= FIRST_DELAY_BAND {
                        let out = self.decorrelator.plain[plain][band];
                        self.decorrelator.plain[plain][band] = input;
                        plain = (plain + 1) % plain_len;
                        out
                    } else {
                        let mut v = all_pass(
                            input,
                            &mut self.decorrelator.qmf_lead[lead][band],
                            PHI_FRACT_QMF[band],
                            0.0,
                        );
                        for m in 0..AP_LINKS {
                            v = all_pass(
                                v,
                                &mut self.decorrelator.qmf_links[m][links[m]][band],
                                Q_FRACT_QMF[band][m],
                                decay_scale * AP_LINK_DECAY[m],
                            );
                        }
                        v
                    };
                    let g = self.gains[slot][bin];
                    qmf_out[slot][band] = Complex32::new(v.re * g, v.im * g);
                    advance(&mut lead, AP_LEAD_DELAY, &mut links);
                }
                self.decorrelator.plain_cursor[band] = plain;
            }
        }

        // Every subband ran the same number of slots, so the cursors all land in
        // the same place; step the shared ones on once rather than per band.
        self.decorrelator.lead_cursor = (self.decorrelator.lead_cursor + SLOTS) % AP_LEAD_DELAY;
        for (cursor, len) in self.decorrelator.link_cursor.iter_mut().zip(AP_LINK_DELAY.iter()) {
            *cursor = (*cursor + SLOTS) % len;
        }
    }

    /// Fill [`Self::gains`] with the transient gain of every slot and bin.
    fn measure_transients(
        &mut self,
        qmf: &[QmfSlot],
        borders: &[usize],
        bins: usize,
        sub_groups: usize,
        bin_of: &[usize],
        split_bands: usize,
    ) {
        for slot in self.powers.iter_mut() {
            slot.fill(0.0);
        }

        for group in 0..sub_groups {
            let band = borders[group];
            let bin = bin_of[group];
            for slot in 0..SLOTS {
                self.powers[slot][bin] += power(self.left[slot][band]);
            }
        }
        for group in sub_groups..borders.len() - 1 {
            let bin = bin_of[group];
            for band in borders[group].max(split_bands)..borders[group + 1].min(QMF_BANDS) {
                for slot in 0..SLOTS {
                    self.powers[slot][bin] += power(qmf[slot][band]);
                }
            }
        }

        for bin in 0..bins {
            for slot in 0..SLOTS {
                self.gains[slot][bin] = self.transient.gain(bin, self.powers[slot][bin]);
            }
        }
    }

    /// Apply the mixing matrix of each envelope, interpolated across its slots.
    fn remix(
        &mut self,
        resolution: Resolution,
        qmf_left: &mut [QmfSlot],
        qmf_right: &mut [QmfSlot],
        sub_bands: usize,
        split_bands: usize,
    ) {
        let (borders, bins, sub_groups) = grid(resolution);
        let bin_of = bin_map(resolution);
        let mut target = [Mix::default(); BINS_34];

        for env in 0..self.data.envelopes {
            for bin in 0..bins {
                target[bin] = self.mix_for(env, bin);
            }

            let from = self.data.borders[env];
            let to = self.data.borders[env + 1].min(SLOTS);
            if to <= from {
                continue;
            }
            let span = to - from;

            for group in 0..sub_groups {
                let band = borders[group];
                if band >= sub_bands {
                    continue;
                }
                let bin = bin_of[group];
                let mut mix = self.previous[bin];
                let delta = target[bin].ramp_from(&mix, span);
                for slot in from..to {
                    mix.step(&delta);
                    let (l, r) = (self.left[slot][band], self.right[slot][band]);
                    self.left[slot][band] = combine(mix.h11, l, mix.h21, r);
                    self.right[slot][band] = combine(mix.h12, l, mix.h22, r);
                }
            }

            for group in sub_groups..borders.len() - 1 {
                let bin = bin_of[group];
                let lo = borders[group].max(split_bands);
                let hi = borders[group + 1].min(QMF_BANDS);
                if lo >= hi {
                    continue;
                }
                let mut mix = self.previous[bin];
                let delta = target[bin].ramp_from(&mix, span);
                for slot in from..to {
                    mix.step(&delta);
                    for band in lo..hi {
                        let (l, r) = (qmf_left[slot][band], qmf_right[slot][band]);
                        qmf_left[slot][band] = combine(mix.h11, l, mix.h21, r);
                        qmf_right[slot][band] = combine(mix.h12, l, mix.h22, r);
                    }
                }
            }

            self.previous[..bins].copy_from_slice(&target[..bins]);
        }
    }

    /// The matrix one envelope and bin asks for.
    fn mix_for(&self, env: usize, bin: usize) -> Mix {
        let iid = self.data.iid[env][bin] as i32;
        let icc = (self.data.icc[env][bin] as usize).min(ICC_ALPHA.len() - 1);

        if self.data.pca_rotation {
            return principal_component_mix(iid, icc, self.data.fine_iid);
        }

        let (steps, table): (i32, &[f32]) = if self.data.fine_iid {
            (IID_STEPS_FINE as i32, &IID_SCALE_FINE)
        } else {
            (IID_STEPS as i32, &IID_SCALE)
        };
        let right = table[(steps + iid).clamp(0, steps * 2) as usize];
        let left = table[(steps - iid).clamp(0, steps * 2) as usize];

        let alpha = ICC_ALPHA[icc];
        let beta = alpha * (right - left) / std::f32::consts::SQRT_2;
        Mix {
            h11: left * (beta + alpha).cos(),
            h12: right * (beta - alpha).cos(),
            h21: left * (beta + alpha).sin(),
            h22: right * (beta - alpha).sin(),
        }
    }
}

/// Mix one output sample from the downmix and its decorrelated copy.
#[inline]
fn combine(a: f32, x: Complex32, b: f32, y: Complex32) -> Complex32 {
    Complex32::new(a * x.re + b * y.re, a * x.im + b * y.im)
}

/// Step the all-pass cursors on by one slot.
#[inline]
fn advance(lead: &mut usize, lead_len: usize, links: &mut [usize; AP_LINKS]) {
    *lead = (*lead + 1) % lead_len;
    for (cursor, len) in links.iter_mut().zip(AP_LINK_DELAY.iter()) {
        *cursor = (*cursor + 1) % len;
    }
}

/// Subband borders, parameter bins and sub-QMF group count of a configuration.
fn grid(resolution: Resolution) -> (&'static [usize], usize, usize) {
    match resolution {
        Resolution::Coarse => (&GROUP_BORDERS_20, BINS_20, SUBQMF_GROUPS_20),
        Resolution::Fine => (&GROUP_BORDERS_34, BINS_34, SUBQMF_GROUPS_34),
    }
}

/// Parameter bin each group of a configuration reads.
fn bin_map(resolution: Resolution) -> &'static [usize] {
    match resolution {
        Resolution::Coarse => &BIN_INDEX_20,
        Resolution::Fine => &BIN_INDEX_34,
    }
}

/// [`BIN_OF_GROUP_20`] with the phase flag dropped, which nothing here reads.
static BIN_INDEX_20: [usize; BIN_OF_GROUP_20.len()] = {
    let mut t = [0usize; BIN_OF_GROUP_20.len()];
    let mut i = 0;
    while i < t.len() {
        t[i] = BIN_OF_GROUP_20[i].bin;
        i += 1;
    }
    t
};

/// [`BIN_OF_GROUP_34`] with the phase flag dropped.
static BIN_INDEX_34: [usize; BIN_OF_GROUP_34.len()] = {
    let mut t = [0usize; BIN_OF_GROUP_34.len()];
    let mut i = 0;
    while i < t.len() {
        t[i] = BIN_OF_GROUP_34[i].bin;
        i += 1;
    }
    t
};

/// The matrix a principal-component-rotation payload asks for.
///
/// Modes 3 to 5 of `bs_icc_mode` describe the same image by the angle of its
/// principal component and how far the residual spreads, rather than by a level
/// difference and a coherence. The two parameterisations meet at the same 2x2
/// matrix; only the route there differs.
fn principal_component_mix(iid: i32, icc: usize, fine: bool) -> Mix {
    let steps: &[i32] = if fine { &IID_STEPS_DB_FINE } else { &IID_STEPS_DB };
    let db = if iid == 0 {
        0.0
    } else {
        let magnitude = steps[((iid.unsigned_abs() as usize) - 1).min(steps.len() - 1)] as f32;
        if iid > 0 { magnitude } else { -magnitude }
    };
    let c = 10f32.powf(db / 20.0);
    let rho = ICC_RHO[icc].max(0.05);

    let alpha = if (c - 1.0).abs() < f32::EPSILON && rho == 0.0 {
        std::f32::consts::FRAC_PI_4
    } else {
        let a = 0.5 * (2.0 * c * rho / (c * c - 1.0)).atan();
        if a < 0.0 { a + std::f32::consts::FRAC_PI_2 } else { a }
    };

    let sum = c + 1.0 / c;
    let mu = 1.0 + (4.0 * rho * rho - 4.0) / (sum * sum);
    let root = mu.max(0.0).sqrt();
    let gamma = ((1.0 - root).max(0.0) / (1.0 + root)).sqrt().atan();

    let scale = std::f32::consts::SQRT_2;
    Mix {
        h11: scale * alpha.cos() * gamma.cos(),
        h12: scale * alpha.sin() * gamma.cos(),
        h21: -scale * alpha.sin() * gamma.sin(),
        h22: scale * alpha.cos() * gamma.sin(),
    }
}

/// Resample per-bin mixing matrices from the 20-bin grid onto the 34-bin one.
fn widen_mix(mix: &mut [Mix; BINS_34]) {
    let s = *mix;
    const SOURCE: [usize; BINS_34] = [
        0, 0, 1, 2, 2, 3, 4, 4, 5, 5, 6, 7, 8, 8, 9, 9, 10, 11, 12, 13, 14, 14, 15, 15, 16, 16, 17,
        17, 18, 18, 18, 18, 19, 19,
    ];
    for (bin, out) in mix.iter_mut().enumerate() {
        *out = match bin {
            1 => mean(s[0], s[1]),
            4 => mean(s[2], s[3]),
            _ => s[SOURCE[bin]],
        };
    }
}

/// Resample per-bin mixing matrices from the 34-bin grid onto the 20-bin one.
fn narrow_mix(mix: &mut [Mix; BINS_34]) {
    let s = *mix;
    let mut out = [Mix::default(); BINS_34];
    out[0] = blend(s[0], 2.0, s[1], 1.0);
    out[1] = blend(s[1], 1.0, s[2], 2.0);
    out[2] = blend(s[3], 2.0, s[4], 1.0);
    out[3] = blend(s[4], 1.0, s[5], 2.0);
    out[4] = mean(s[6], s[7]);
    out[5] = mean(s[8], s[9]);
    out[6] = s[10];
    out[7] = s[11];
    out[8] = mean(s[12], s[13]);
    out[9] = mean(s[14], s[15]);
    out[10] = s[16];
    out[11] = s[17];
    out[12] = s[18];
    out[13] = s[19];
    out[14] = mean(s[20], s[21]);
    out[15] = mean(s[22], s[23]);
    out[16] = mean(s[24], s[25]);
    out[17] = mean(s[26], s[27]);
    out[18] = mean(mean(s[28], s[29]), mean(s[30], s[31]));
    out[19] = mean(s[32], s[33]);
    *mix = out;
}

#[inline]
fn mean(a: Mix, b: Mix) -> Mix {
    blend(a, 1.0, b, 1.0)
}

#[inline]
fn blend(a: Mix, wa: f32, b: Mix, wb: f32) -> Mix {
    let n = wa + wb;
    Mix {
        h11: (wa * a.h11 + wb * b.h11) / n,
        h12: (wa * a.h12 + wb * b.h12) / n,
        h21: (wa * a.h21 + wb * b.h21) / n,
        h22: (wa * a.h22 + wb * b.h22) / n,
    }
}
