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

pub mod crc;
pub mod data;
pub mod grid;
pub mod hf;
pub mod header;

use crate::bitstream::BitReader;
use crate::decoder::ps::{PsDecoder, QmfSlot};
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
    /// Parametric stereo, present once an extension field has carried a payload.
    ///
    /// It turns a single-channel element into a stereo pair, so it also owns the
    /// synthesis bank the second channel needs.
    ps: Option<Box<ParametricStereo>>,
}

/// The parametric stereo decoder and the buffers its extra channel needs.
struct ParametricStereo {
    decoder: PsDecoder,
    /// Synthesis bank for the channel parametric stereo invents.
    synthesis: QmfSynthesis,
    left: Vec<QmfSlot>,
    right: Vec<QmfSlot>,
    /// The slots past the frame the hybrid filterbank reads.
    ahead: Vec<QmfSlot>,
}

impl ParametricStereo {
    fn new(width: SynthesisWidth) -> Self {
        Self {
            decoder: PsDecoder::new(),
            synthesis: QmfSynthesis::new(width),
            left: vec![[Complex32::default(); GRID_BANDS]; SLOTS_PER_FRAME],
            right: vec![[Complex32::default(); GRID_BANDS]; SLOTS_PER_FRAME],
            ahead: vec![[Complex32::default(); GRID_BANDS]; hf::LOOKAHEAD],
        }
    }
}

impl SbrDecoder {
    /// Create a decoder for `channels` channels whose core runs at `core_rate_hz`.
    pub fn new(channels: usize, core_rate_hz: u32, downsampled: bool) -> Self {
        let width = synthesis_width(downsampled);
        Self {
            header: None,
            layout: BandLayout::default(),
            channels: (0..channels.max(1)).map(|_| SbrChannel::new(width)).collect(),
            output_rate_hz: core_rate_hz.saturating_mul(2),
            downsampled,
            ready: false,
            ps: None,
        }
    }

    /// Channels this element produces, which parametric stereo doubles.
    #[inline]
    pub fn output_channels(&self) -> usize {
        if self.ps.is_some() { 2 } else { self.channels.len() }
    }

    /// Whether parametric stereo is reconstructing a second channel.
    #[inline]
    pub fn parametric_stereo(&self) -> bool {
        self.ps.is_some()
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
        if let Some(ps) = self.ps.as_mut() {
            ps.decoder.reset();
            ps.synthesis.reset();
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
    /// `payload_bits_remaining` is how many bits of this fill element's SBR
    /// payload are left when this is called (its declared byte count, less
    /// the 4-bit extension-type nibble the caller already consumed to learn
    /// `with_crc`) -- needed only to size the CRC's protected span correctly;
    /// callers with no payload-length tracking may pass `usize::MAX` and rely
    /// on `sbr_crc_check`'s own clamp to "however much of the reader remains",
    /// which is correct as long as this fill element is the last thing in the
    /// buffer the caller hands in.
    pub fn decode_extension(
        &mut self,
        reader: &mut BitReader,
        element: SbrElement,
        with_crc: bool,
        payload_bits_remaining: usize,
    ) -> Result<()> {
        for ch in &mut self.channels {
            ch.have_data = false;
        }

        if with_crc {
            let protected_bits = payload_bits_remaining.saturating_sub(crc::CRC_BITS);
            if !crc::sbr_crc_check(reader, protected_bits)? {
                return Err(DecodeError::CorruptedFrame(
                    "SBR payload failed its CRC-10 check".into(),
                )
                .into());
            }
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
        // Splitting the borrow keeps `self.layout` in place: an early `?` used to
        // leave a taken-out layout behind as an empty default, which the next frame
        // then indexed as though it were valid.
        let Self { header, layout, channels, ps, downsampled, .. } = self;
        let width = synthesis_width(*downsampled);
        let header = header.clone().expect("ready implies a header");

        // `bs_data_extra` gates a reserved field that no profile defines.
        if reader.read_bit()? {
            let _reserved = reader.read_u8(4)?;
        }

        let grid = FrameGrid::parse(reader)?;
        let amp_res = data::effective_resolution(&grid, header.amp_res);
        let (df_env, df_noise) = data::read_direction_flags(reader, &grid)?;
        let invf = data::read_invf(reader, layout.noise_band_count())?;

        let ch = &mut channels[0];
        let envelope_q = data::read_envelopes(
            reader,
            layout,
            &grid,
            amp_res,
            false,
            &df_env,
            &mut ch.history,
        )?;
        let noise_q =
            data::read_noise_floors(reader, layout, amp_res, false, &df_noise, &mut ch.history)?;

        let add_harmonic = if reader.read_bit()? {
            data::read_added_sinusoids(reader, layout.sfb_count(true))?
        } else {
            vec![false; layout.sfb_count(true)]
        };
        read_extended_data(reader, ps, width)?;

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
        let Self { header, layout, channels, ps, downsampled, .. } = self;
        let width = synthesis_width(*downsampled);
        let header = header.clone().expect("ready implies a header");
        let high_bands = layout.sfb_count(true);

        // `bs_data_extra` gates two reserved fields, one per channel.
        if reader.read_bit()? {
            let _reserved_left = reader.read_u8(4)?;
            let _reserved_right = reader.read_u8(4)?;
        }
        let coupled = reader.read_bit()?;

        let (left, right) = channels.split_at_mut(1);
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
        read_extended_data(reader, ps, width)?;

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

        Ok(())
    }

    /// Run the reconstruction for one channel and write the result.
    ///
    /// `core` is one frame of the AAC core's output. `out` receives
    /// [`Self::output_frame_len`] samples. Channels with no payload this frame are
    /// still filtered through the QMF pair, so that the output keeps the same
    /// delay and the core signal is not interrupted.
    pub fn process_channel(&mut self, channel: usize, core: &[f32], out: &mut [f32]) -> Result<()> {
        self.check_buffers(channel, core, out)?;
        self.reconstruct(channel, core);

        let ch = &mut self.channels[channel];
        let width = ch.synthesis.bands();
        let mut slot_bands = [Complex32::default(); GRID_BANDS];
        for slot in 0..SLOTS_PER_FRAME {
            ch.hf.output_slot(slot, &mut slot_bands);
            ch.synthesis
                .process_slot(&slot_bands[..width], &mut out[slot * width..(slot + 1) * width]);
        }
        Ok(())
    }

    /// Run the reconstruction for a parametric stereo element, writing both channels.
    ///
    /// The element carries one coded channel; the second is invented from the
    /// transmitted stereo image. Both come out six QMF slots later than
    /// [`Self::process_channel`] would deliver the same core signal, which is what
    /// the hybrid filterbank costs.
    pub fn process_parametric(
        &mut self,
        core: &[f32],
        left: &mut [f32],
        right: &mut [f32],
    ) -> Result<()> {
        self.check_buffers(0, core, left)?;
        self.check_buffers(0, core, right)?;
        let Some(mut ps) = self.ps.take() else {
            return Err(DecodeError::PsError("no parametric stereo payload for this element".into())
                .into());
        };

        self.reconstruct(0, core);

        let ch = &mut self.channels[0];
        for (slot, bands) in ps.left.iter_mut().enumerate() {
            ch.hf.output_slot(slot, bands);
        }
        for (offset, bands) in ps.ahead.iter_mut().enumerate() {
            ch.hf.output_slot(SLOTS_PER_FRAME + offset, bands);
        }
        ps.decoder.process(&mut ps.left, &ps.ahead, &mut ps.right);

        let width = ch.synthesis.bands();
        for slot in 0..SLOTS_PER_FRAME {
            let span = slot * width..(slot + 1) * width;
            ch.synthesis.process_slot(&ps.left[slot][..width], &mut left[span.clone()]);
            ps.synthesis.process_slot(&ps.right[slot][..width], &mut right[span]);
        }

        self.ps = Some(ps);
        Ok(())
    }

    /// Reject a call whose buffers are not the sizes the chain needs.
    fn check_buffers(&self, channel: usize, core: &[f32], out: &[f32]) -> Result<()> {
        if channel >= self.channels.len() {
            return Err(DecodeError::CorruptedFrame("SBR channel index out of range".into()).into());
        }
        if core.len() < SBR_CORE_FRAME || out.len() < self.output_frame_len() {
            return Err(
                DecodeError::CorruptedFrame("SBR frame buffers are the wrong size".into()).into()
            );
        }
        Ok(())
    }

    /// Analyse one core frame and fill the high band from the transmitted envelope.
    fn reconstruct(&mut self, channel: usize, core: &[f32]) {
        let Self { header, layout, channels, ready, .. } = self;
        let ready = *ready;
        let ch = &mut channels[channel];

        ch.hf.advance_frame();
        let mut bands = [Complex32::default(); 32];
        for slot in 0..SLOTS_PER_FRAME {
            ch.analysis.process_slot(&core[slot * 32..slot * 32 + 32], &mut bands);
            ch.hf.store_slot(slot, &bands);
        }

        if ch.have_data && ready {
            if channel == 0 && std::env::var_os("AAC_TRACE_SBR").is_some() {
                let d = &ch.data;
                eprintln!(
                    "  frame: class {:?} env {} borders {:?} noise_borders {:?} res {:?} trans {:?} invf {:?}",
                    d.grid.class,
                    d.grid.envelopes(),
                    d.grid.borders,
                    d.grid.noise_borders,
                    d.grid.high_res,
                    d.grid.transient_envelope,
                    d.invf
                );
                eprintln!("    env {:?}", d.envelope);
                eprintln!("    noise {:?}", d.noise);
                eprintln!("    sines {:?}", d.add_harmonic);
            }
            let header = header.as_ref().expect("ready implies a header");
            hf::generate(&mut ch.hf, layout, &ch.data);
            hf::adjust(&mut ch.hf, layout, header, &ch.data);
        }
    }
}

/// The synthesis bank width core-rate or doubled output calls for.
#[inline]
const fn synthesis_width(downsampled: bool) -> SynthesisWidth {
    if downsampled { SynthesisWidth::Downsampled } else { SynthesisWidth::Full }
}

/// Extension field identifiers `bs_extended_data` can carry.
mod extension_id {
    /// Parametric stereo, the only one this decoder acts on.
    pub const PS: u8 = 2;
}

/// Read `bs_extended_data`, decoding any parametric stereo payload it carries.
///
/// The field is byte-counted and each extension inside it is length-implicit, so
/// the reader is stepped to the declared end whatever was found: an extension this
/// decoder does not know, or one that fails to parse, must not desynchronise the
/// rest of the frame.
fn read_extended_data(
    reader: &mut BitReader,
    ps: &mut Option<Box<ParametricStereo>>,
    width: SynthesisWidth,
) -> Result<()> {
    if !reader.read_bit()? {
        return Ok(());
    }
    let mut count = reader.read_u8(4)? as usize;
    if count == 15 {
        count += reader.read_u8(8)? as usize;
    }
    let payload_bits = count * 8;
    if reader.bits_remaining() < payload_bits {
        return Err(DecodeError::CorruptedFrame(
            "SBR extension field runs past the end of the payload".into(),
        )
        .into());
    }
    let end = reader.bit_position() + payload_bits;

    while end.saturating_sub(reader.bit_position()) > 7 {
        let id = reader.read_u8(2)?;
        if id != extension_id::PS {
            break;
        }
        let stereo = ps.get_or_insert_with(|| Box::new(ParametricStereo::new(width)));
        let room = end - reader.bit_position();
        if stereo.decoder.parse(reader, room).is_err() {
            // Keep the element rather than the payload: the downmix is still a
            // perfectly good mono signal, and the previous frame's image holds.
            break;
        }
    }

    let position = reader.bit_position();
    if position < end {
        reader.skip_bits(end - position)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A payload whose transmitted CRC does not match its content must be
    /// refused with the CRC-specific error, not silently accepted or
    /// rejected for some unrelated reason -- this is the actual wiring
    /// [`crc::sbr_crc_check`] exists to be used from.
    #[test]
    fn a_payload_with_a_mismatched_crc_is_refused() {
        use crate::bitstream::BitWriter;

        let mut sbr = SbrDecoder::new(1, 22050, false);
        let mut w = BitWriter::new();
        w.write_bits(0, crc::CRC_BITS); // transmitted checksum: wrong for what follows
        w.write_bit(false); // header_present -- would fail differently if reached
        for _ in 0..32 {
            w.write_bit(true);
        }
        let bytes = w.finalize().to_vec();
        let mut reader = BitReader::new(&bytes);

        let err = sbr
            .decode_extension(&mut reader, SbrElement::Single, true, bytes.len() * 8)
            .unwrap_err();
        assert!(format!("{err}").contains("CRC"), "unexpected error: {err}");
    }

    /// The mirror case: a correctly computed CRC must not be rejected by the
    /// CRC check itself, whatever happens afterward while parsing the (here,
    /// header-less) payload that follows it.
    #[test]
    fn a_payload_with_a_matching_crc_passes_the_crc_check() {
        use crate::bitstream::BitWriter;

        let mut sbr = SbrDecoder::new(1, 22050, false);
        let payload_bits = 33usize; // header_present=0 plus 32 filler bits
        let mut w = BitWriter::new();
        w.write_bits(0, crc::CRC_BITS); // placeholder, patched below
        w.write_bit(false);
        for _ in 0..32 {
            w.write_bit(true);
        }
        let mut bytes = w.finalize().to_vec();

        let mut payload_reader = BitReader::new(&bytes);
        payload_reader.skip_bits(crc::CRC_BITS).unwrap();
        let real_crc = crc::checksum(&mut payload_reader, payload_bits).unwrap();
        bytes[0] = (real_crc >> 2) as u8;
        bytes[1] = (bytes[1] & 0x3F) | (((real_crc & 0b11) as u8) << 6);

        // Pass the true intended payload length (33 bits: header_present +
        // 32 filler), not the byte-padded buffer length -- exactly as the
        // real fill-element caller passes the declared byte count's bits,
        // not however many the underlying buffer happens to round up to.
        let mut reader = BitReader::new(&bytes);
        let err = sbr
            .decode_extension(&mut reader, SbrElement::Single, true, crc::CRC_BITS + payload_bits)
            .unwrap_err();
        assert!(!format!("{err}").contains("CRC"), "a valid CRC must not fail the CRC check: {err}");
    }

    /// A decoder with no header must refuse a payload rather than guess a layout.
    #[test]
    fn payload_without_a_header_is_rejected() {
        let mut sbr = SbrDecoder::new(1, 22050, false);
        // A single zero bit: no header follows.
        let bytes = [0u8; 8];
        let mut reader = BitReader::new(&bytes);
        assert!(sbr.decode_extension(&mut reader, SbrElement::Single, false, usize::MAX).is_err());
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
                let _ = sbr.decode_extension(&mut reader, element, seed % 3 == 0, bytes.len() * 8);
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
