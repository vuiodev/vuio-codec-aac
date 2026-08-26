//! Spectral band replication.
//!
//! SBR lets an encoder throw away the top half of the spectrum and send, in its
//! place, a few hundred bits per frame describing what used to be there. The
//! decoder rebuilds that range by copying the surviving low band upwards and then
//! reshaping the copy to the transmitted description: its energy envelope over
//! time and frequency, how noise-like each part of it should be, and where pure
//! tones the copy could not supply have to be added.
//!
//! The chain, per channel:
//!
//! 1. the core signal goes through a 32-band complex QMF analysis ([`crate::dsp::qmf`]),
//! 2. [`hf::generate`] fills subbands above the crossover from the ones below it,
//! 3. [`hf::adjust`] scales, noises and tones that range to the transmitted envelope,
//! 4. a 64-band synthesis returns time samples at twice the core rate.
//!
//! Everything the bitstream carries is parsed in [`header`], [`grid`] and [`data`].
//!
//! # Delay
//!
//! Steps 2 and 3 both reach backwards in time, so the reconstruction of a frame is
//! finished only once part of the following frame has been analysed. The decoder
//! therefore emits, for each core frame, a window that ends six QMF slots before
//! that frame does; see [`hf`] for the buffer layout that implements it.

pub mod data;
pub mod grid;
pub mod hf;
pub mod header;

use crate::bitstream::BitReader;
use crate::dsp::fft::Complex32;
use crate::dsp::qmf::{QmfAnalysis, QmfSynthesis, SynthesisWidth};
use crate::error::{DecodeError, Result};

use data::{ChannelHistory, SbrChannelData};
use grid::FrameGrid;
use hf::{GRID_BANDS, HfState, SLOTS_PER_FRAME};
use header::{BandLayout, SbrHeader};

/// Core samples one SBR frame covers.
pub const SBR_CORE_FRAME: usize = SLOTS_PER_FRAME * 32;

/// Which syntactic element the SBR payload belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SbrElement {
    /// A single channel: the payload describes one channel.
    Single,
    /// A channel pair: the payload describes two, possibly coupled.
    Pair,
}

impl SbrElement {
    /// Channels this element carries.
    #[inline]
    pub const fn channels(self) -> usize {
        match self {
            Self::Single => 1,
            Self::Pair => 2,
        }
    }
}

/// One channel's inter-frame state.
struct SbrChannel {
    analysis: QmfAnalysis,
    synthesis: QmfSynthesis,
    hf: HfState,
    history: ChannelHistory,
    data: SbrChannelData,
    /// Whether this channel has a payload to apply this frame.
    have_data: bool,
}

impl SbrChannel {
    fn new(width: SynthesisWidth) -> Self {
        Self {
            analysis: QmfAnalysis::new(),
            synthesis: QmfSynthesis::new(width),
            hf: HfState::new(),
            history: ChannelHistory::default(),
            data: SbrChannelData::default(),
            have_data: false,
        }
    }

    fn reset(&mut self) {
        self.analysis.reset();
        self.synthesis.reset();
        self.hf.reset();
        self.history.reset();
        self.have_data = false;
    }
}

/// Spectral band replication for one channel element.
pub struct SbrDecoder {
    header: Option<SbrHeader>,
    layout: BandLayout,
    channels: Vec<SbrChannel>,
    /// Sampling rate of the SBR output, twice the core rate.
    output_rate_hz: u32,
    /// Whether to synthesise at the core rate instead of doubling.
    downsampled: bool,
    /// Set until a header has been seen and a layout derived.
    ready: bool,
}

impl SbrDecoder {
    /// Create a decoder for `channels` channels whose core runs at `core_rate_hz`.
    pub fn new(channels: usize, core_rate_hz: u32, downsampled: bool) -> Self {
        let width = if downsampled { SynthesisWidth::Downsampled } else { SynthesisWidth::Full };
        Self {
            header: None,
            layout: BandLayout::default(),
            channels: (0..channels.max(1)).map(|_| SbrChannel::new(width)).collect(),
            output_rate_hz: core_rate_hz.saturating_mul(2),
            downsampled,
            ready: false,
        }
    }

    /// Samples per channel this decoder produces per core frame.
    #[inline]
    pub fn output_frame_len(&self) -> usize {
        if self.downsampled { SBR_CORE_FRAME } else { 2 * SBR_CORE_FRAME }
    }

    /// Whether a header has been seen, so that [`Self::process`] will do anything.
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.ready
    }

    /// The band layout the current header implies.
    #[inline]
    pub fn layout(&self) -> &BandLayout {
        &self.layout
    }

    /// Forget all inter-frame state, as after a seek.
    pub fn reset(&mut self) {
        for ch in &mut self.channels {
            ch.reset();
        }
    }

    /// Adopt a new core sampling rate, rebuilding the band layout.
    pub fn set_core_rate(&mut self, core_rate_hz: u32) {
        let output = core_rate_hz.saturating_mul(2);
        if output == self.output_rate_hz {
            return;
        }
        self.output_rate_hz = output;
        self.ready = false;
        if let Some(header) = self.header.clone() {
            self.adopt_header(header);
        }
    }

    /// Parse `sbr_extension_data()` from a fill element's payload.
    ///
    /// A payload that cannot be parsed leaves the previous frame's state alone and
    /// reports the error; the caller may then either drop the frame or let the core
    /// signal through unreplicated.
    pub fn decode_extension(
        &mut self,
        reader: &mut BitReader,
        element: SbrElement,
        with_crc: bool,
    ) -> Result<()> {
        for ch in &mut self.channels {
            ch.have_data = false;
        }

        if with_crc {
            // The CRC covers the payload that follows. Nothing here acts on a
            // mismatch, so it is read only to stay aligned.
            let _crc = reader.read_u16(10)?;
        }

        if reader.read_bit()? {
            let header = SbrHeader::parse(reader)?;
            self.adopt_header(header);
        }
        if !self.ready {
            return Err(DecodeError::CorruptedFrame(
                "SBR payload arrived before any SBR header".into(),
            )
            .into());
        }

        match element {
            SbrElement::Single => self.decode_single(reader)?,
            SbrElement::Pair => self.decode_pair(reader)?,
        }
        Ok(())
    }

    /// Install a header and derive everything that depends on it.
    fn adopt_header(&mut self, header: SbrHeader) {
        let unchanged = self.header.as_ref().is_some_and(|h| h.same_layout_as(&header));
        match BandLayout::derive(&header, self.output_rate_hz) {
            Ok(layout) => {
                self.layout = layout;
                self.ready = true;
                if !unchanged {
                    // A different band layout invalidates every delta reference and
                    // every gain the smoothing filter is holding.
                    for ch in &mut self.channels {
                        ch.history.reset();
                        ch.hf.reset();
                    }
                }
            }
            Err(_) => self.ready = false,
        }
        if std::env::var_os("AAC_TRACE_SBR").is_some() {
            eprintln!("sbr header {header:?}");
            eprintln!(
                "  layout k_x {} k_high {} master {:?} low {:?} high {:?} noise {:?} lim {:?}",
                self.layout.k_x,
                self.layout.k_high,
                self.layout.master,
                self.layout.low,
                self.layout.high,
                self.layout.noise,
                self.layout.limiter
            );
            eprintln!("  patches {:?}", self.layout.patches);
        }
        self.header = Some(header);
    }

    /// Parse `sbr_single_channel_element()`.
    fn decode_single(&mut self, reader: &mut BitReader) -> Result<()> {
        let header = self.header.clone().expect("ready implies a header");
        let layout = std::mem::take(&mut self.layout);

        // `bs_data_extra` gates a reserved field that no profile defines.
        if reader.read_bit()? {
            let _reserved = reader.read_u8(4)?;
        }

        let grid = FrameGrid::parse(reader)?;
        let amp_res = data::effective_resolution(&grid, header.amp_res);
        let (df_env, df_noise) = data::read_direction_flags(reader, &grid)?;
        let invf = data::read_invf(reader, layout.noise_band_count())?;

        let ch = &mut self.channels[0];
        let envelope_q = data::read_envelopes(
            reader,
            &layout,
            &grid,
            amp_res,
            false,
            &df_env,
            &mut ch.history,
        )?;
        let noise_q =
            data::read_noise_floors(reader, &layout, amp_res, false, &df_noise, &mut ch.history)?;

        let add_harmonic = if reader.read_bit()? {
            data::read_added_sinusoids(reader, layout.sfb_count(true))?
        } else {
            vec![false; layout.sfb_count(true)]
        };
        skip_extended_data(reader)?;

        ch.data = SbrChannelData {
            grid,
            amp_res,
            invf,
            envelope: Vec::new(),
            noise: Vec::new(),
            add_harmonic,
            envelope_q,
            noise_q,
        };
        data::dequantize(&mut ch.data);
        ch.have_data = true;

        self.layout = layout;
        Ok(())
    }

    /// Parse `sbr_channel_pair_element()`.
    fn decode_pair(&mut self, reader: &mut BitReader) -> Result<()> {
        if self.channels.len() < 2 {
            return Err(DecodeError::CorruptedFrame(
                "SBR channel pair arrived for a single-channel element".into(),
            )
            .into());
        }
        let header = self.header.clone().expect("ready implies a header");
        let layout = std::mem::take(&mut self.layout);
        let high_bands = layout.sfb_count(true);

        // `bs_data_extra` gates two reserved fields, one per channel.
        if reader.read_bit()? {
            let _reserved_left = reader.read_u8(4)?;
            let _reserved_right = reader.read_u8(4)?;
        }
        let coupled = reader.read_bit()?;

        let (left, right) = self.channels.split_at_mut(1);
        let left = &mut left[0];
        let right = &mut right[0];

        let (mut left_data, mut right_data) = if coupled {
            // One grid, one set of inverse filtering modes, and envelopes that code
            // the pair's total and its balance rather than two independent levels.
            let grid = FrameGrid::parse(reader)?;
            let amp_res = data::effective_resolution(&grid, header.amp_res);
            let (df_env_l, df_noise_l) = data::read_direction_flags(reader, &grid)?;
            let (df_env_r, df_noise_r) = data::read_direction_flags(reader, &grid)?;
            let invf = data::read_invf(reader, layout.noise_band_count())?;

            let env_l = data::read_envelopes(
                reader, &layout, &grid, amp_res, false, &df_env_l, &mut left.history,
            )?;
            let noise_l = data::read_noise_floors(
                reader, &layout, amp_res, false, &df_noise_l, &mut left.history,
            )?;
            let env_r = data::read_envelopes(
                reader, &layout, &grid, amp_res, true, &df_env_r, &mut right.history,
            )?;
            let noise_r = data::read_noise_floors(
                reader, &layout, amp_res, true, &df_noise_r, &mut right.history,
            )?;

            (
                SbrChannelData {
                    grid: grid.clone(),
                    amp_res,
                    invf: invf.clone(),
                    add_harmonic: vec![false; high_bands],
                    envelope_q: env_l,
                    noise_q: noise_l,
                    ..Default::default()
                },
                SbrChannelData {
                    grid,
                    amp_res,
                    invf,
                    add_harmonic: vec![false; high_bands],
                    envelope_q: env_r,
                    noise_q: noise_r,
                    ..Default::default()
                },
            )
        } else {
            let grid_l = FrameGrid::parse(reader)?;
            let grid_r = FrameGrid::parse(reader)?;
            let amp_l = data::effective_resolution(&grid_l, header.amp_res);
            let amp_r = data::effective_resolution(&grid_r, header.amp_res);
            let (df_env_l, df_noise_l) = data::read_direction_flags(reader, &grid_l)?;
            let (df_env_r, df_noise_r) = data::read_direction_flags(reader, &grid_r)?;
            let invf_l = data::read_invf(reader, layout.noise_band_count())?;
            let invf_r = data::read_invf(reader, layout.noise_band_count())?;

            let env_l = data::read_envelopes(
                reader, &layout, &grid_l, amp_l, false, &df_env_l, &mut left.history,
            )?;
            let env_r = data::read_envelopes(
                reader, &layout, &grid_r, amp_r, false, &df_env_r, &mut right.history,
            )?;
            let noise_l = data::read_noise_floors(
                reader, &layout, amp_l, false, &df_noise_l, &mut left.history,
            )?;
            let noise_r = data::read_noise_floors(
                reader, &layout, amp_r, false, &df_noise_r, &mut right.history,
            )?;

            (
                SbrChannelData {
                    grid: grid_l,
                    amp_res: amp_l,
                    invf: invf_l,
                    add_harmonic: vec![false; high_bands],
                    envelope_q: env_l,
                    noise_q: noise_l,
                    ..Default::default()
                },
                SbrChannelData {
                    grid: grid_r,
                    amp_res: amp_r,
                    invf: invf_r,
                    add_harmonic: vec![false; high_bands],
                    envelope_q: env_r,
                    noise_q: noise_r,
                    ..Default::default()
                },
            )
        };

        if reader.read_bit()? {
            left_data.add_harmonic = data::read_added_sinusoids(reader, high_bands)?;
        }
        if reader.read_bit()? {
            right_data.add_harmonic = data::read_added_sinusoids(reader, high_bands)?;
        }
        skip_extended_data(reader)?;

        if coupled {
            data::dequantize_coupled(&mut left_data, &mut right_data);
        } else {
            data::dequantize(&mut left_data);
            data::dequantize(&mut right_data);
        }

        left.data = left_data;
        right.data = right_data;
        left.have_data = true;
        right.have_data = true;

        self.layout = layout;
        Ok(())
    }

    /// Run the reconstruction for one channel and write the result.
    ///
    /// `core` is one frame of the AAC core's output. `out` receives
    /// [`Self::output_frame_len`] samples. Channels with no payload this frame are
    /// still filtered through the QMF pair, so that the output keeps the same
    /// delay and the core signal is not interrupted.
    pub fn process_channel(&mut self, channel: usize, core: &[f32], out: &mut [f32]) -> Result<()> {
        if channel >= self.channels.len() {
            return Err(DecodeError::CorruptedFrame("SBR channel index out of range".into()).into());
        }
        if core.len() < SBR_CORE_FRAME || out.len() < self.output_frame_len() {
            return Err(DecodeError::CorruptedFrame("SBR frame buffers are the wrong size".into())
                .into());
        }

        let layout = std::mem::take(&mut self.layout);
        let header = self.header.clone();
        let ch = &mut self.channels[channel];

        ch.hf.advance_frame();
        let mut bands = [Complex32::default(); 32];
        for slot in 0..SLOTS_PER_FRAME {
            ch.analysis.process_slot(&core[slot * 32..slot * 32 + 32], &mut bands);
            ch.hf.store_slot(slot, &bands);
        }

        if ch.have_data && self.ready {
            if std::env::var_os("AAC_TRACE_SBR").is_some() && channel == 0 {
                eprintln!(
                    "  frame: class {:?} env {} borders {:?} noise_borders {:?} res {:?} trans {:?} invf {:?}",
                    ch.data.grid.class,
                    ch.data.grid.envelopes(),
                    ch.data.grid.borders,
                    ch.data.grid.noise_borders,
                    ch.data.grid.high_res,
                    ch.data.grid.transient_envelope,
                    ch.data.invf
                );
                eprintln!("    env {:?}", ch.data.envelope);
                eprintln!("    noise {:?}", ch.data.noise);
                eprintln!("    sines {:?}", ch.data.add_harmonic);
            }
            let header = header.as_ref().expect("ready implies a header");
            hf::generate(&mut ch.hf, &layout, &ch.data);
            hf::adjust(&mut ch.hf, &layout, header, &ch.data);
        }

        let width = ch.synthesis.bands();
        let mut slot_bands = [Complex32::default(); GRID_BANDS];
        for slot in 0..SLOTS_PER_FRAME {
            ch.hf.output_slot(slot, &mut slot_bands);
            ch.synthesis
                .process_slot(&slot_bands[..width], &mut out[slot * width..(slot + 1) * width]);
        }

        self.layout = layout;
        Ok(())
    }
}

/// Step over `bs_extended_data`, whose payload this decoder does not use.
///
/// Parametric stereo travels here; a decoder that wants it reads the payload
/// before calling this.
fn skip_extended_data(reader: &mut BitReader) -> Result<()> {
    if !reader.read_bit()? {
        return Ok(());
    }
    let mut count = reader.read_u8(4)? as usize;
    if count == 15 {
        count += reader.read_u8(8)? as usize;
    }
    reader.skip_bits(count * 8)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A decoder with no header must refuse a payload rather than guess a layout.
    #[test]
    fn payload_without_a_header_is_rejected() {
        let mut sbr = SbrDecoder::new(1, 22050, false);
        // A single zero bit: no header follows.
        let bytes = [0u8; 8];
        let mut reader = BitReader::new(&bytes);
        assert!(sbr.decode_extension(&mut reader, SbrElement::Single, false).is_err());
        assert!(!sbr.is_ready());
    }

    /// With no payload the chain must still pass the core signal through, delayed
    /// and doubled in rate, rather than fall silent.
    #[test]
    fn core_passes_through_without_a_payload() {
        let mut sbr = SbrDecoder::new(1, 22050, false);
        let mut out = vec![0.0f32; sbr.output_frame_len()];
        let mut reconstructed = Vec::new();
        let mut source = Vec::new();

        for frame in 0..8 {
            let core: Vec<f32> = (0..SBR_CORE_FRAME)
                .map(|i| {
                    let t = (frame * SBR_CORE_FRAME + i) as f32;
                    0.4 * (t * 0.011).sin()
                })
                .collect();
            sbr.process_channel(0, &core, &mut out).unwrap();
            source.extend_from_slice(&core);
            reconstructed.extend_from_slice(&out);
        }

        // 577 samples for the filterbank, plus six QMF slots of SBR look-back.
        let delay = 577 + 6 * 64;
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in 400..(reconstructed.len() - delay) / 2 - 200 {
            let want = source[i] as f64;
            let got = reconstructed[2 * i + delay] as f64;
            num += (got - want) * (got - want);
            den += want * want;
        }
        let snr = 10.0 * (den / num.max(1e-30)).log10();
        assert!(snr > 40.0, "core passthrough reconstructs at only {snr:.1} dB");
    }

    /// Arbitrary payload bytes must never panic and must never leave the decoder in
    /// a state where processing panics either.
    #[test]
    fn arbitrary_payloads_do_not_panic() {
        let mut sbr = SbrDecoder::new(2, 22050, false);
        let core = vec![0.25f32; SBR_CORE_FRAME];
        let mut out = vec![0.0f32; sbr.output_frame_len()];

        for seed in 0..2048u32 {
            let bytes: Vec<u8> = (0..24)
                .map(|i| ((seed.wrapping_mul(2246822519).wrapping_add(i)) >> 9) as u8)
                .collect();
            for element in [SbrElement::Single, SbrElement::Pair] {
                let mut reader = BitReader::new(&bytes);
                let _ = sbr.decode_extension(&mut reader, element, seed % 3 == 0);
                for ch in 0..2 {
                    let _ = sbr.process_channel(ch, &core, &mut out);
                    assert!(out.iter().all(|v| v.is_finite()), "seed {seed} produced non-finite output");
                }
            }
        }
    }

    /// Downsampled mode must produce core-rate output.
    #[test]
    fn downsampled_mode_keeps_the_core_rate() {
        let sbr = SbrDecoder::new(1, 22050, true);
        assert_eq!(sbr.output_frame_len(), SBR_CORE_FRAME);
        let sbr = SbrDecoder::new(1, 22050, false);
        assert_eq!(sbr.output_frame_len(), 2 * SBR_CORE_FRAME);
    }
}
