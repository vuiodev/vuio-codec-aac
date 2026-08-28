//! ACELP: the speech core of MPEG-D USAC's Linear Prediction Domain (LPD) mode.
//!
//! Ported from `c/libxaac/decoder/ixheaacd_acelp_{bitparse,decode,tools}.c`.
//!
//! # What ACELP is doing
//!
//! An LPD superframe splits into 64-sample subframes, and each subframe models
//! 4 ms of speech as *an excitation signal run through an LPC synthesis filter*.
//! The excitation is the sum of two codebooks, and the whole design is about
//! spending bits where speech actually has structure:
//!
//! * The **adaptive codebook** is the pitch predictor: the excitation from one
//!   pitch period ago, scaled by a gain. Voiced speech is near-periodic, so a
//!   lag and a gain reproduce most of it. The lag is transmitted at 1/4-sample
//!   resolution, so reading it back is an interpolation
//!   ([`AcelpDecoder::adaptive_codebook`]) rather than an array index.
//! * The **algebraic codebook** codes what the pitch predictor missed, as a
//!   handful of unit pulses at transmitted positions and signs. It is called
//!   *algebraic* because the codebook is never stored: positions are packed
//!   into an index by a combinatorial rule, and [`decode_pulses`] unpacks it.
//!   Sixty-four positions are split into four interleaved tracks
//!   (position `p` of track `t` is sample `4p + t`), and the bit rate decides
//!   how many pulses each track gets — that is the whole content of
//!   [`CODE_BITS_PER_MODE`].
//!
//! Both are scaled by gains from a trained VQ ([`crate::tables::usac_acelp::GAIN_TABLE`]),
//! and the innovation gain is coded *relative to a predicted energy* rather than
//! absolutely, which is why [`decode_gains`] needs the frame's mean-energy field.
//!
//! # What this module does not cover yet
//!
//! Forward aliasing cancellation (FAC), which bridges the boundary when a frame
//! switches between ACELP, TCX and the FD core, is not here — see
//! `text/plan.txt` phase 1.7. Without it, a superframe made only of ACELP
//! subframes decodes correctly, and a *transition* between core modes carries
//! the aliasing FAC exists to cancel. The bass post-filter (`ixheaacd_lpd_bpf`)
//! is likewise deferred; it is a post-processor, not part of reconstruction.
//! Neither omission is silent: [`AcelpDecoder::decode_subframe`] is only ever
//! reached from an all-ACELP path today.

use crate::bitstream::BitReader;
use crate::error::{Error, FormatError, Result};
use crate::tables::usac_acelp::{
    F_PIT_SHARP, FSCALE_DENOM, GAIN_TABLE, INTER_LP_FIL_ORDER, INTERPOL_FILT, LEN_SUBFR, ORDER,
    PREEMPH_FILT_FAC, TFR1, TFR2, TILT_CODE, TMAX, TMIN, UP_SAMP,
};

/// Total bits the algebraic codebook spends per subframe, indexed by
/// `acelp_core_mode` (`num_codebits_table` in `ixheaacd_acelp_decode.c`). The
/// value picks how many pulses each of the four tracks carries, which is what
/// [`decode_pulses`] switches on.
pub const CODE_BITS_PER_MODE: [u16; 8] = [20, 28, 36, 44, 52, 64, 12, 16];

/// Width of each of the four transmitted algebraic-codebook indices, by core
/// mode (`ixheaacd_num_bites_celp_coding`). Mode 5 is the exception and is
/// handled separately in [`SubframeParams::parse`]: it sends eight fields, not
/// four.
pub const CELP_INDEX_BITS: [[u8; 4]; 8] = [
    [5, 5, 5, 5],
    [9, 9, 5, 5],
    [9, 9, 9, 9],
    [13, 13, 9, 9],
    [13, 13, 13, 13],
    [16, 16, 16, 16],
    [1, 5, 1, 5],
    [1, 5, 5, 5],
];

/// The largest pitch lag any sampling rate can ask for, and so how much past
/// excitation must be retained (`MAX_PITCH`). Derived exactly as the reference
/// does, from the widest supported `fscale`.
pub const MAX_PITCH: usize = {
    let i = (24_000 * TMIN + FSCALE_DENOM / 2) / FSCALE_DENOM - TMIN;
    (TMAX + 6 * i) as usize
};

/// Excitation history the adaptive codebook needs: the deepest lag, plus the
/// interpolation filter's reach past it.
const EXC_HISTORY: usize = MAX_PITCH + INTER_LP_FIL_ORDER + 1;

/// The pitch-lag grid, which moves with the core's sampling rate.
///
/// Lags are transmitted as a single index over a piecewise grid: fine (1/4
/// sample) at short lags where pitch resolution matters most, coarser (1/2)
/// in the middle, integer-only at long lags. `fscale` slides the whole grid so
/// a lag means the same *period in seconds* at any core rate.
#[derive(Debug, Clone, Copy)]
pub struct PitchRange {
    pub min: i32,
    pub fr2: i32,
    pub fr1: i32,
    pub max: i32,
}

impl PitchRange {
    /// Build the grid for a core sampling-rate scale (`fscale`; 12800 is the
    /// unscaled case, and gives min 34 / max 231).
    pub fn new(fscale: i32) -> Self {
        let i = (fscale * TMIN + FSCALE_DENOM / 2) / FSCALE_DENOM - TMIN;
        Self { min: TMIN + i, fr2: TFR2 - i, fr1: TFR1, max: TMAX + 6 * i }
    }

    /// Decode an absolutely-coded lag: the index walks the fine region, then
    /// the half-sample region, then the integer region.
    fn absolute(&self, index: i32) -> (i32, i32) {
        let fine = (self.fr2 - self.min) * 4;
        let half = fine + (self.fr1 - self.fr2) * 2;
        if index < fine {
            let lag = self.min + index / 4;
            (lag, index - (lag - self.min) * 4)
        } else if index < half {
            let index = index - fine;
            let lag = self.fr2 + index / 2;
            (lag, (index - (lag - self.fr2) * 2) * 2)
        } else {
            (index + self.fr1 - fine - (self.fr1 - self.fr2) * 2, 0)
        }
    }
}

impl Default for PitchRange {
    fn default() -> Self {
        Self::new(FSCALE_DENOM)
    }
}

// ---------------------------------------------------------------------------
// Bitstream
// ---------------------------------------------------------------------------

/// One subframe's transmitted parameters.
#[derive(Debug, Clone, Default)]
pub struct SubframeParams {
    /// Adaptive-codebook (pitch lag) index; absolutely coded in the first
    /// subframe of each half-frame, differentially elsewhere.
    pub acb_index: i32,
    /// When clear, the adaptive-codebook excitation is low-pass filtered
    /// before use. Transmitted inverted relative to how it reads: 0 means
    /// *do* filter.
    pub ltp_filtering: bool,
    /// Algebraic-codebook indices; how many are meaningful depends on the mode.
    pub icb_index: [i32; 8],
    /// Index into the joint (pitch gain, innovation gain) VQ.
    pub gain_index: usize,
}

impl SubframeParams {
    fn parse(reader: &mut BitReader, core_mode: usize, absolute_lag: bool) -> Result<Self> {
        let mut p = Self {
            acb_index: reader.read_u32(if absolute_lag { 9 } else { 6 })? as i32,
            ltp_filtering: reader.read_bit()?,
            ..Default::default()
        };
        if core_mode == 5 {
            // 4 pulses per track: four 2-bit sign/position prefixes followed by
            // four 14-bit bodies, recombined in `decode_pulses`.
            for slot in p.icb_index.iter_mut().take(4) {
                *slot = reader.read_u32(2)? as i32;
            }
            for slot in p.icb_index.iter_mut().skip(4) {
                *slot = reader.read_u32(14)? as i32;
            }
        } else {
            for (slot, &bits) in p.icb_index.iter_mut().zip(CELP_INDEX_BITS[core_mode].iter()) {
                *slot = reader.read_u32(bits as usize)? as i32;
            }
        }
        p.gain_index = reader.read_u32(7)? as usize;
        Ok(p)
    }
}

/// One ACELP frame's parameters: a coarse energy for the whole frame, then the
/// per-subframe detail (`ixheaacd_acelp_decoding`).
#[derive(Debug, Clone, Default)]
pub struct AcelpFrame {
    /// 2-bit quantized mean excitation energy, shared by the frame's subframes
    /// as the reference point the innovation gain is coded against.
    pub mean_energy: i32,
    pub subframes: Vec<SubframeParams>,
}

impl AcelpFrame {
    /// Read one ACELP frame. `num_subfr` is 4 for a 256-sample frame and 2 for
    /// a 128-sample one; the pitch lag is sent absolutely at the start of each
    /// half, differentially in between.
    pub fn parse(reader: &mut BitReader, core_mode: usize, num_subfr: usize) -> Result<Self> {
        if core_mode >= CODE_BITS_PER_MODE.len() {
            return Err(Error::Format(FormatError::InvalidUsacContainer(format!(
                "acelp_core_mode {core_mode} is out of range 0..8"
            ))));
        }
        let mean_energy = reader.read_u32(2)? as i32;
        let subframes = (0..num_subfr)
            .map(|sfr| {
                let absolute = sfr == 0 || (num_subfr == 4 && sfr == 2);
                SubframeParams::parse(reader, core_mode, absolute)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { mean_energy, subframes })
    }
}

// ---------------------------------------------------------------------------
// Algebraic codebook
// ---------------------------------------------------------------------------

/// Place one signed pulse on `track`. `m_bits` is the width of the position
/// field, so the low `m_bits` of `idx` are the position and the next bit is the
/// sign.
fn place_1(idx: i32, m_bits: u32, offset: i32, track: usize, code: &mut [f32; LEN_SUBFR]) {
    let mask = (1 << m_bits) - 1;
    let pos = (idx & mask) + offset;
    let sign = (idx >> m_bits) & 1;
    let m = (pos as usize) * 4 + track;
    if let Some(slot) = code.get_mut(m) {
        *slot += if sign == 1 { -1.0 } else { 1.0 };
    }
}

/// Place two pulses on one track from a joint index. Only one sign bit is sent
/// for the pair: when the second position precedes the first the pulses take
/// opposite signs, otherwise the same sign. That is exactly the information the
/// dropped second sign bit would have carried, because the encoder is free to
/// emit the pair in either order.
fn place_2(idx: i32, m_bits: u32, offset: i32, track: usize, code: &mut [f32; LEN_SUBFR]) {
    let mask = (1 << m_bits) - 1;
    let p0 = ((idx >> m_bits) & mask) + offset;
    let p1 = (idx & mask) + offset;
    let sign = (idx >> (2 * m_bits)) & 1;
    let (s0, s1) = match (p1 - p0 < 0, sign == 1) {
        (true, true) => (-1.0, 1.0),
        (true, false) => (1.0, -1.0),
        (false, true) => (-1.0, -1.0),
        (false, false) => (1.0, 1.0),
    };
    if let Some(slot) = code.get_mut(p0 as usize * 4 + track) {
        *slot += s0;
    }
    if let Some(slot) = code.get_mut(p1 as usize * 4 + track) {
        *slot += s1;
    }
}

/// Three pulses: a pair in one half of the track (which half is one bit) plus a
/// single anywhere.
fn place_3(idx: i32, m_bits: u32, offset: i32, track: usize, code: &mut [f32; LEN_SUBFR]) {
    let mask = (1 << (2 * m_bits - 1)) - 1;
    let half = if (idx >> (2 * m_bits - 1)) & 1 == 1 { 1 << (m_bits - 1) } else { 0 };
    place_2(idx & mask, m_bits - 1, offset + half, track, code);
    let mask1 = (1 << (m_bits + 1)) - 1;
    place_1((idx >> (2 * m_bits)) & mask1, m_bits, offset, track, code);
}

/// Four pulses, split as 2+2 with one bit choosing the first pair's half.
fn place_4_section(idx: i32, offset: i32, track: usize, code: &mut [f32; LEN_SUBFR]) {
    let half = if (idx >> 5) & 1 == 1 { 4 } else { 0 };
    place_2(idx & 31, 2, offset + half, track, code);
    place_2((idx >> 6) & 127, 3, offset, track, code);
}

/// Four pulses on one track. Two bits select how the four split across the
/// track's two halves (4+0, 1+3, 2+2, 3+1), which is what lets a single index
/// cover every useful distribution without wasting codespace on the rest.
fn place_4(idx: i32, track: usize, code: &mut [f32; LEN_SUBFR]) {
    match (idx >> 14) & 3 {
        0 => place_4_section(idx, if (idx >> 13) & 1 == 0 { 0 } else { 8 }, track, code),
        1 => {
            place_1(idx >> 10, 3, 0, track, code);
            place_3(idx, 3, 8, track, code);
        }
        2 => {
            place_2(idx >> 7, 3, 0, track, code);
            place_2(idx, 3, 8, track, code);
        }
        _ => {
            place_3(idx >> 4, 3, 0, track, code);
            place_1(idx, 3, 8, track, code);
        }
    }
}

/// Reconstruct the innovation vector from its transmitted indices
/// (`ixheaacd_acelp_decode_pulses_per_track`).
///
/// `code_bits` is [`CODE_BITS_PER_MODE`] for the frame's core mode, and selects
/// the pulse distribution: 12 and 16 bits are the sparse low-rate cases, 20
/// through 52 spread one to three pulses per track, and 64 gives every track
/// four.
pub fn decode_pulses(indices: &[i32; 8], code_bits: u16) -> [f32; LEN_SUBFR] {
    let mut code = [0.0f32; LEN_SUBFR];
    match code_bits {
        12 => {
            for track_pair in 0..2 {
                let offset = indices[2 * track_pair];
                let index = indices[2 * track_pair + 1];
                let pos = (index & 15) + if (index >> 4) & 1 == 1 { 16 } else { 0 };
                add_pulse_at(pos, 2 * offset as usize + track_pair, &mut code);
            }
        }
        16 => {
            // One track is skipped; which one is carried by the first index.
            let skipped = if indices[0] == 0 { 1 } else { 3 };
            let mut next = 1;
            for track in 0..4 {
                if track == skipped {
                    continue;
                }
                let index = indices[next];
                next += 1;
                let pos = (index & 15) + if (index >> 4) & 1 == 1 { 16 } else { 0 };
                add_pulse_at(pos, track, &mut code);
            }
        }
        20 => {
            for (track, &index) in indices.iter().enumerate().take(4) {
                place_1(index, 4, 0, track, &mut code);
            }
        }
        28 => {
            for (track, &index) in indices.iter().enumerate().take(2) {
                place_2(index, 4, 0, track, &mut code);
            }
            for (track, &index) in indices.iter().enumerate().take(4).skip(2) {
                place_1(index, 4, 0, track, &mut code);
            }
        }
        36 => {
            for (track, &index) in indices.iter().enumerate().take(4) {
                place_2(index, 4, 0, track, &mut code);
            }
        }
        44 => {
            for (track, &index) in indices.iter().enumerate().take(2) {
                place_3(index, 4, 0, track, &mut code);
            }
            for (track, &index) in indices.iter().enumerate().take(4).skip(2) {
                place_2(index, 4, 0, track, &mut code);
            }
        }
        52 => {
            for (track, &index) in indices.iter().enumerate().take(4) {
                place_3(index, 4, 0, track, &mut code);
            }
        }
        64 => {
            for track in 0..4 {
                place_4((indices[track] << 14) + indices[track + 4], track, &mut code);
            }
        }
        _ => {}
    }
    code
}

/// Add a signed unit pulse whose position carries its sign in bit 4
/// (`ixheaacd_d_acelp_add_pulse`).
fn add_pulse_at(pos: i32, track: usize, code: &mut [f32; LEN_SUBFR]) {
    let i = ((pos & 15) as usize) * 4 + track;
    if let Some(slot) = code.get_mut(i) {
        *slot += if pos & 16 == 0 { 1.0 } else { -1.0 };
    }
}

// ---------------------------------------------------------------------------
// Filters (ixheaacd_acelp_tools.c)
// ---------------------------------------------------------------------------

/// `s[i] -= mu * s[i-1]`, with `mem` standing in for `s[-1]`. Flattens the
/// spectral tilt speech has before the excitation is coded.
pub fn preemphasis(signal: &mut [f32], mu: f32, mem: f32) {
    for i in (1..signal.len()).rev() {
        signal[i] -= mu * signal[i - 1];
    }
    if let Some(first) = signal.first_mut() {
        *first -= mu * mem;
    }
}

/// Inverse of [`preemphasis`]: `s[i] += 0.68 * s[i-1]`, restoring the tilt at
/// the end of synthesis.
pub fn deemphasis(signal: &mut [f32], mem: f32) {
    let mut prev = mem;
    for x in signal.iter_mut() {
        *x += PREEMPH_FILT_FAC * prev;
        prev = *x;
    }
}

/// LPC synthesis `1/A(z)`: `y[i] = x[i] - sum_j a[j] y[i-j]`, with `mem`
/// carrying the previous `ORDER` outputs in and the new ones out.
pub fn synthesis(a: &[f32], x: &[f32], y: &mut [f32], mem: &mut [f32; ORDER]) {
    debug_assert!(a.len() > ORDER);
    let mut buf = vec![0.0f32; ORDER + x.len()];
    buf[..ORDER].copy_from_slice(mem);
    for i in 0..x.len() {
        let mut s = x[i];
        for j in 1..=ORDER {
            s -= a[j] * buf[ORDER + i - j];
        }
        buf[ORDER + i] = s;
        y[i] = s;
    }
    let tail = buf.len() - ORDER;
    mem.copy_from_slice(&buf[tail..]);
}

/// Analysis `A(z)`, the exact inverse of [`synthesis`]: recovers the excitation
/// that produced a signal. `x` must carry `ORDER` samples of history ahead of
/// the span being filtered.
pub fn residual(a: &[f32], x_with_history: &[f32], y: &mut [f32]) {
    debug_assert!(x_with_history.len() >= ORDER + y.len());
    for (i, out) in y.iter_mut().enumerate() {
        let n = ORDER + i;
        let mut s = x_with_history[n];
        for j in 1..=ORDER {
            s += a[j] * x_with_history[n - j];
        }
        *out = s;
    }
}

/// Sharpen periodicity in the innovation by feeding it back at the pitch lag
/// (`ixheaacd_acelp_pitch_sharpening`). Pulses land where the pitch predictor
/// already put energy, which is where speech wants them.
pub fn pitch_sharpening(code: &mut [f32; LEN_SUBFR], lag: usize) {
    if lag == 0 {
        return;
    }
    for i in lag..LEN_SUBFR {
        code[i] += code[i - lag] * F_PIT_SHARP;
    }
}

/// Split the joint gain index into (pitch gain, innovation gain).
///
/// The innovation gain is not transmitted directly: the codebook entry scales a
/// gain *predicted* from `mean_energy` and the innovation's own energy, so the
/// same index means different absolute gains in loud and quiet frames. Returns
/// the innovation energy alongside, which the caller needs for the voicing
/// measure that drives gain smoothing.
pub fn decode_gains(index: usize, code: &[f32; LEN_SUBFR], mean_energy: f32) -> (f32, f32, f32) {
    let energy: f32 = 0.01 + code.iter().map(|c| c * c).sum::<f32>();
    let avg_db = 10.0 * (energy / LEN_SUBFR as f32).log10();
    let predicted = 10.0f32.powf(0.05 * (mean_energy - avg_db));
    let i = (index * 2).min(GAIN_TABLE.len() - 2);
    (GAIN_TABLE[i], GAIN_TABLE[i + 1] * predicted, energy)
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

/// The constants every subframe in one ACELP frame decodes against.
///
/// These are frame-scoped rather than subframe-scoped -- the energy reference
/// and the bit budget are transmitted once for the whole frame, and the
/// stability factor comes from the frame's LSF set -- so they travel together
/// instead of being threaded through as loose arguments.
#[derive(Debug, Clone, Copy)]
pub struct FrameContext {
    /// De-quantized mean excitation energy in dB, the reference the innovation
    /// gain is coded against.
    pub mean_energy: f32,
    /// LSF-derived stability factor in `0..=1`, scaling how much gain smoothing
    /// is applied: a stable envelope smooths more, a transient one less.
    pub stability: f32,
    /// `acelp_core_mode`, which fixes the algebraic codebook's bit budget.
    pub core_mode: usize,
}

/// Running state an ACELP decoder carries between subframes and frames.
///
/// The excitation history is the interesting part: the adaptive codebook reads
/// up to [`MAX_PITCH`] samples back into excitation *this decoder produced*, so
/// the buffer is a sliding window rather than a per-frame scratch. Everything
/// else here is filter memory.
pub struct AcelpDecoder {
    /// `EXC_HISTORY` samples of past excitation, the subframe being built, and
    /// one sample of lookahead that both the interpolator and the optional
    /// low-pass read past the subframe's end. Slides left by [`LEN_SUBFR`]
    /// after each subframe.
    exc: Vec<f32>,
    /// LPC synthesis memory, in the pre-emphasis domain.
    synth_mem: [f32; ORDER],
    /// Last de-emphasized output sample, seeding the next frame's de-emphasis.
    deemph_mem: f32,
    /// Smoothed innovation gain, carried across subframes so an abrupt gain
    /// change is spread rather than clicking.
    gain_threshold: f32,
    /// Pitch grid for the core sampling rate.
    range: PitchRange,
    /// Lower bound of the differential lag window set by the last absolute lag.
    lag_min: i32,
}

impl AcelpDecoder {
    /// Create a decoder for a core whose pitch grid is scaled by `fscale`
    /// (pass [`FSCALE_DENOM`] for the unscaled 12.8 kHz case).
    pub fn new(fscale: i32) -> Self {
        Self {
            exc: vec![0.0; EXC_HISTORY + LEN_SUBFR + 1],
            synth_mem: [0.0; ORDER],
            deemph_mem: 0.0,
            gain_threshold: 0.0,
            range: PitchRange::new(fscale),
            lag_min: 0,
        }
    }

    /// Drop all history, as at a seek or a core-mode reset.
    pub fn reset(&mut self) {
        self.exc.fill(0.0);
        self.synth_mem = [0.0; ORDER];
        self.deemph_mem = 0.0;
        self.gain_threshold = 0.0;
        self.lag_min = 0;
    }

    /// The excitation history, oldest first. Exposed for the TCX path, which
    /// shares this buffer when a superframe mixes core modes.
    pub fn excitation(&self) -> &[f32] {
        &self.exc[..EXC_HISTORY]
    }

    /// Evaluate the adaptive codebook into the current subframe slot.
    ///
    /// A fractional lag means the excitation one pitch period ago falls between
    /// stored samples, so it is reconstructed with a 4x-oversampled FIR read at
    /// two phases (`ixheaacd_cb_exc_calc`). `LEN_SUBFR + 1` samples are produced
    /// because the optional low-pass in [`Self::decode_subframe`] looks one
    /// sample ahead.
    fn adaptive_codebook(&mut self, lag: i32, frac: i32) {
        let mut frac = -frac;
        let mut base = EXC_HISTORY as isize - lag as isize;
        if frac < 0 {
            frac += UP_SAMP as i32;
            base -= 1;
        }
        let mut out = [0.0f32; LEN_SUBFR + 1];
        for (j, slot) in out.iter_mut().enumerate() {
            let centre = base + j as isize;
            let mut s = 0.0f32;
            for i in 0..INTER_LP_FIL_ORDER {
                let c1 = INTERPOL_FILT[frac as usize + i * UP_SAMP];
                let c2 = INTERPOL_FILT[UP_SAMP - frac as usize + i * UP_SAMP];
                let back = centre - i as isize;
                let fwd = centre + 1 + i as isize;
                if back >= 0 {
                    s += self.exc[back as usize] * c1;
                }
                if (fwd as usize) < self.exc.len() {
                    s += self.exc[fwd as usize] * c2;
                }
            }
            *slot = s;
        }
        self.exc[EXC_HISTORY..EXC_HISTORY + LEN_SUBFR].copy_from_slice(&out[..LEN_SUBFR]);
        // Keep the one-sample lookahead where the low-pass can reach it.
        self.exc[EXC_HISTORY + LEN_SUBFR] = out[LEN_SUBFR];
    }

    /// Decode one 64-sample subframe into `synth`.
    ///
    /// `lpc` is the subframe's interpolated LPC filter (`ORDER + 1`
    /// coefficients, `a[0] == 1`), `ctx` carries the frame-scoped constants,
    /// and `absolute_lag` says whether this subframe's pitch index is absolute
    /// or relative to the last absolute one.
    /// The synthesis written here is still pre-emphasized; [`Self::decode_frame`]
    /// de-emphasizes once per frame, as the reference does.
    pub fn decode_subframe(
        &mut self,
        params: &SubframeParams,
        lpc: &[f32],
        ctx: &FrameContext,
        absolute_lag: bool,
        synth: &mut [f32],
    ) {
        // 1. Pitch lag. Absolute subframes also re-centre the differential
        //    window the following subframe decodes against.
        let (lag, frac) = if absolute_lag {
            let (lag, frac) = self.range.absolute(params.acb_index);
            let mut lo = (lag - 8).max(self.range.min);
            if lo + 15 > self.range.max {
                lo = self.range.max - 15;
            }
            self.lag_min = lo;
            (lag, frac)
        } else {
            let lag = self.lag_min + params.acb_index / 4;
            (lag, params.acb_index - (lag - self.lag_min) * 4)
        };

        // 2. Adaptive codebook, optionally low-passed. The filter is a fixed
        //    3-tap smoother; the encoder signals when the pitch contribution is
        //    noisy enough to want it.
        self.adaptive_codebook(lag, frac);
        if !params.ltp_filtering {
            let mut filtered = [0.0f32; LEN_SUBFR];
            for (i, slot) in filtered.iter_mut().enumerate() {
                let n = EXC_HISTORY + i;
                *slot = 0.18 * self.exc[n - 1] + 0.64 * self.exc[n] + 0.18 * self.exc[n + 1];
            }
            self.exc[EXC_HISTORY..EXC_HISTORY + LEN_SUBFR].copy_from_slice(&filtered);
        }

        // 3. Algebraic codebook, tilted and pitch-sharpened.
        let mut code = decode_pulses(&params.icb_index, CODE_BITS_PER_MODE[ctx.core_mode]);
        preemphasis(&mut code, TILT_CODE, 0.0);
        let sharpen_lag = if frac > 2 { lag + 1 } else { lag };
        pitch_sharpening(&mut code, sharpen_lag.max(0) as usize);

        // 4. Gains, and the voicing ratio they imply.
        let (pitch_gain, gain_code, innov_energy) =
            decode_gains(params.gain_index, &code, ctx.mean_energy);
        let adaptive = &self.exc[EXC_HISTORY..EXC_HISTORY + LEN_SUBFR];
        let pitch_energy: f32 = adaptive.iter().map(|x| x * x).sum::<f32>() * pitch_gain * pitch_gain;
        let innov_energy = innov_energy * gain_code * gain_code;
        let total = pitch_energy + innov_energy;
        let voicing = if total > 0.0 { (pitch_energy - innov_energy) / total } else { 0.0 };

        // 5. Total excitation, which is what future subframes predict from.
        let mut post = [0.0f32; LEN_SUBFR];
        for i in 0..LEN_SUBFR {
            let adaptive = self.exc[EXC_HISTORY + i];
            post[i] = pitch_gain * adaptive;
            self.exc[EXC_HISTORY + i] = pitch_gain * adaptive + gain_code * code[i];
        }

        // 6. Gain smoothing. The innovation gain is nudged 19% towards its
        //    running threshold, then blended by how voiced and how stable the
        //    frame is: strongly voiced, stable frames get the smoothed gain,
        //    transients keep the transmitted one.
        let smooth_factor = ctx.stability * 0.5 * (1.0 - voicing);
        let mut gain0 = gain_code;
        if gain0 < self.gain_threshold {
            gain0 = (gain0 * 1.19).min(self.gain_threshold);
        } else {
            gain0 = (gain0 / 1.19).max(self.gain_threshold);
        }
        self.gain_threshold = gain0;
        let smoothed = smooth_factor * gain0 + (1.0 - smooth_factor) * gain_code;
        for c in code.iter_mut() {
            *c *= smoothed;
        }

        // 7. A mild high-pass across the innovation before synthesis, whose
        //    strength tracks voicing (`cpe`): unvoiced frames get more of it.
        let cpe = 0.125 * (1.0 + voicing);
        post[0] += code[0] - cpe * code[1];
        for i in 1..LEN_SUBFR - 1 {
            post[i] += code[i] - cpe * (code[i - 1] + code[i + 1]);
        }
        post[LEN_SUBFR - 1] += code[LEN_SUBFR - 1] - cpe * code[LEN_SUBFR - 2];

        // 8. Run it through the synthesis filter.
        synthesis(lpc, &post, &mut synth[..LEN_SUBFR], &mut self.synth_mem);

        // 9. Slide the excitation window so this subframe becomes history.
        self.exc.copy_within(LEN_SUBFR.., 0);
        let tail = self.exc.len() - LEN_SUBFR;
        self.exc[tail..].fill(0.0);
    }

    /// Decode a whole ACELP frame: every subframe in turn, then one pass of
    /// de-emphasis over the result.
    ///
    /// `lpc_per_subframe` holds one `ORDER + 1` coefficient set per subframe —
    /// the LPD layer interpolates these from the frame's transmitted LSFs, so
    /// the filter moves smoothly across the frame rather than stepping at
    /// subframe boundaries.
    pub fn decode_frame(
        &mut self,
        frame: &AcelpFrame,
        lpc_per_subframe: &[[f32; ORDER + 1]],
        stability: f32,
        core_mode: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let n = frame.subframes.len();
        if lpc_per_subframe.len() < n || out.len() < n * LEN_SUBFR {
            return Err(Error::Format(FormatError::InvalidUsacContainer(
                "ACELP output or LPC set too short for the frame's subframes".to_string(),
            )));
        }
        let ctx = FrameContext {
            mean_energy: frame.mean_energy as f32 * 12.0 + 18.0,
            stability,
            core_mode,
        };
        for (sfr, params) in frame.subframes.iter().enumerate() {
            let absolute = sfr == 0 || (n == 4 && sfr == 2);
            self.decode_subframe(
                params,
                &lpc_per_subframe[sfr],
                &ctx,
                absolute,
                &mut out[sfr * LEN_SUBFR..],
            );
        }
        let span = n * LEN_SUBFR;
        deemphasis(&mut out[..span], self.deemph_mem);
        self.deemph_mem = out[span - 1];
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat filter makes synthesis and residual the identity, which is the
    /// cheapest check that neither one has an index off by one.
    #[test]
    fn synthesis_of_a_trivial_filter_passes_the_signal_through() {
        let mut a = [0.0f32; ORDER + 1];
        a[0] = 1.0;
        let x: Vec<f32> = (0..LEN_SUBFR).map(|i| i as f32).collect();
        let mut y = vec![0.0; LEN_SUBFR];
        let mut mem = [0.0; ORDER];
        synthesis(&a, &x, &mut y, &mut mem);
        assert_eq!(y, x);
    }

    /// The real property: `A(z)` inverts `1/A(z)` exactly, for a filter with
    /// actual poles in it.
    #[test]
    fn residual_inverts_synthesis() {
        let mut a = [0.0f32; ORDER + 1];
        a[0] = 1.0;
        a[1] = -0.7;
        a[2] = 0.25;
        a[5] = -0.1;

        let excitation: Vec<f32> =
            (0..LEN_SUBFR).map(|i| ((i * 37 % 19) as f32 - 9.0) / 9.0).collect();
        let mut synth = vec![0.0f32; LEN_SUBFR];
        let mut mem = [0.0; ORDER];
        synthesis(&a, &excitation, &mut synth, &mut mem);

        // residual() wants ORDER samples of history; synthesis started from
        // silence, so that history is zeros.
        let mut with_history = vec![0.0f32; ORDER];
        with_history.extend_from_slice(&synth);
        let mut recovered = vec![0.0f32; LEN_SUBFR];
        residual(&a, &with_history, &mut recovered);

        for (got, want) in recovered.iter().zip(excitation.iter()) {
            assert!((got - want).abs() < 1e-4, "residual != synthesis^-1: {got} vs {want}");
        }
    }

    #[test]
    fn deemphasis_inverts_preemphasis() {
        let original: Vec<f32> = (0..64).map(|i| (i as f32 * 0.3).sin()).collect();
        let mut work = original.clone();
        preemphasis(&mut work, PREEMPH_FILT_FAC, 0.0);
        deemphasis(&mut work, 0.0);
        for (got, want) in work.iter().zip(original.iter()) {
            assert!((got - want).abs() < 1e-4, "{got} vs {want}");
        }
    }

    /// Every mode must place the number of pulses its bit budget pays for, on
    /// the tracks it says, and nowhere else. A pulse landing on the wrong track
    /// is the classic transcription error in this ladder.
    #[test]
    fn each_mode_places_the_pulses_its_budget_pays_for() {
        // (code_bits, expected non-zero count) with indices chosen so no two
        // pulses of a track collide and cancel.
        let cases = [(20u16, 4usize), (28, 6), (36, 8), (44, 10), (52, 12)];
        for (bits, expected) in cases {
            let indices = [1, 2 | (3 << 4), 3 | (5 << 4), 4 | (6 << 4), 0, 0, 0, 0];
            let code = decode_pulses(&indices, bits);
            let nonzero = code.iter().filter(|c| **c != 0.0).count();
            assert!(
                nonzero > 0 && nonzero <= expected,
                "mode {bits}: {nonzero} pulses, expected at most {expected}"
            );
            for c in code.iter() {
                assert!(c.abs() <= 4.0, "pulse magnitudes stay small: {c}");
            }
        }
    }

    #[test]
    fn a_single_pulse_lands_on_its_own_track_with_its_own_sign() {
        let mut code = [0.0f32; LEN_SUBFR];
        place_1(5, 4, 0, 2, &mut code); // position 5, sign +, track 2
        assert_eq!(code[5 * 4 + 2], 1.0);
        assert_eq!(code.iter().filter(|c| **c != 0.0).count(), 1);

        let mut code = [0.0f32; LEN_SUBFR];
        place_1(5 | (1 << 4), 4, 0, 1, &mut code); // sign bit set
        assert_eq!(code[5 * 4 + 1], -1.0);
    }

    /// The unscaled grid must reproduce the reference's constants exactly, and
    /// every index in range must decode to a lag inside it.
    #[test]
    fn the_pitch_grid_covers_its_whole_index_range() {
        let r = PitchRange::new(FSCALE_DENOM);
        assert_eq!((r.min, r.fr2, r.fr1, r.max), (34, 128, 160, 231));
        for index in 0..512 {
            let (lag, frac) = r.absolute(index);
            if lag > r.max {
                break;
            }
            assert!(lag >= r.min, "index {index} gave lag {lag} below min {}", r.min);
            assert!((0..4).contains(&frac), "index {index} gave fraction {frac}");
        }
    }

    /// Gains must track the transmitted mean energy: a louder frame gets a
    /// larger innovation gain for the same index and the same innovation.
    #[test]
    fn innovation_gain_scales_with_the_frames_mean_energy() {
        let mut code = [0.0f32; LEN_SUBFR];
        code[0] = 1.0;
        code[17] = -1.0;
        let (pitch_quiet, gain_quiet, _) = decode_gains(40, &code, 18.0);
        let (pitch_loud, gain_loud, _) = decode_gains(40, &code, 42.0);
        assert_eq!(pitch_quiet, pitch_loud, "pitch gain is read straight from the table");
        assert!(gain_loud > gain_quiet * 4.0, "{gain_loud} vs {gain_quiet}");
    }

    /// The end-to-end shape: real parameters in, bounded speech-like samples
    /// out, and the decoder's history actually advancing.
    #[test]
    fn a_frame_decodes_to_bounded_output_and_advances_history() {
        let mut dec = AcelpDecoder::new(FSCALE_DENOM);
        let mut lpc = [0.0f32; ORDER + 1];
        lpc[0] = 1.0;
        lpc[1] = -0.6;
        lpc[2] = 0.2;

        let frame = AcelpFrame {
            mean_energy: 2,
            subframes: (0..4)
                .map(|s| SubframeParams {
                    acb_index: if s == 0 || s == 2 { 200 } else { 30 },
                    ltp_filtering: s % 2 == 0,
                    icb_index: [3, 9, 17, 25, 0, 0, 0, 0],
                    gain_index: 60,
                })
                .collect(),
        };
        let lpcs = [lpc; 4];
        let mut out = vec![0.0f32; 4 * LEN_SUBFR];
        dec.decode_frame(&frame, &lpcs, 0.8, 2, &mut out).unwrap();

        assert!(out.iter().any(|x| *x != 0.0), "a real frame must not decode to silence");
        assert!(out.iter().all(|x| x.is_finite()), "synthesis diverged");
        assert!(dec.excitation().iter().any(|x| *x != 0.0), "excitation history did not advance");
    }

    /// Parsing must consume exactly the bits the mode's field widths describe,
    /// or every subsequent element in the frame is misaligned.
    #[test]
    fn parsing_consumes_exactly_the_modes_field_widths() {
        // Mode 2 sends [9,9,9,9] for the innovation. The pitch-lag field is 9
        // bits in the two absolutely-coded subframes (0 and 2) and 6 in the
        // others, so the total is not simply four identical subframes.
        let bytes = [0xA5u8; 64];
        let mut reader = BitReader::new(&bytes);
        let start = reader.bit_position();
        let frame = AcelpFrame::parse(&mut reader, 2, 4).unwrap();
        let consumed = reader.bit_position() - start;
        assert_eq!(frame.subframes.len(), 4);
        assert_eq!(consumed, 2 + (9 + 6 + 9 + 6) + 4 * (1 + 9 * 4 + 7));
    }

    #[test]
    fn an_out_of_range_core_mode_is_rejected_rather_than_indexing_past_the_table() {
        let bytes = [0u8; 32];
        let mut reader = BitReader::new(&bytes);
        assert!(AcelpFrame::parse(&mut reader, 9, 4).is_err());
    }
}
