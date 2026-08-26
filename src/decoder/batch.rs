//! Parallel batch decoding of a complete ADTS stream.
//!
//! Frames are not independent: each frame's output overlap-adds the tail of the
//! previous one, and its rising window edge takes the previous frame's window shape.
//! Both dependencies reach back exactly one frame, so a worker that starts one frame
//! early and discards that frame's output produces byte-identical results to a
//! sequential decode of the whole stream.
//!
//! That priming frame is the only redundant work, so with `k` chunks the overhead is
//! `k` extra frames regardless of stream length.
//!
//! The third piece of cross-frame state is the noise-substitution generator, which
//! by default is threaded through the whole stream and so cannot be reconstructed
//! from a priming frame. These entry points therefore select
//! [`NoiseMode::PerFrame`], which seeds it from the frame position instead. Both
//! functions below use that mode, so they agree with each other byte for byte; a
//! decode driven frame-by-frame through [`Decoder`] with its default
//! [`NoiseMode::Sequential`] will differ from them, but only inside
//! noise-substituted bands, and only in samples no two decoders agree on anyway.

use crate::bitstream::BitReader;
use crate::decoder::aac::pns::NoiseMode;
use crate::decoder::engine::Decoder;
use crate::error::Result;
use crate::syntax::adts::AdtsHeader;
use crate::syntax::asc::AudioSpecificConfig;

/// Byte range of one ADTS frame within a stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameSpan {
    pub start: usize,
    pub end: usize,
}

/// Locate every ADTS frame in `stream`.
///
/// Resynchronizes by scanning forward one byte at a time when a header does not
/// parse, which is what lets a stream with leading garbage or a torn frame still
/// decode the frames that follow.
pub fn scan_adts_frames(stream: &[u8]) -> Vec<FrameSpan> {
    let mut spans = Vec::new();
    let mut offset = 0usize;

    while offset + 7 <= stream.len() {
        if stream[offset] == 0xFF && (stream[offset + 1] & 0xF0) == 0xF0 {
            let mut reader = BitReader::new(&stream[offset..]);
            if let Ok(header) = AdtsHeader::parse(&mut reader) {
                let len = header.frame_length as usize;
                // A frame shorter than its own header is a false sync.
                if len >= 7 && offset + len <= stream.len() {
                    spans.push(FrameSpan { start: offset, end: offset + len });
                    offset += len;
                    continue;
                }
            }
        }
        offset += 1;
    }
    spans
}

/// Decoded PCM for a run of frames.
pub struct DecodedAudio {
    /// Interleaved 16-bit samples.
    pub samples: Vec<i16>,
    pub channels: usize,
    pub sample_rate: u32,
    /// Frames that decoded successfully.
    pub frames: usize,
}

impl DecodedAudio {
    /// Samples per channel.
    pub fn samples_per_channel(&self) -> usize {
        if self.channels == 0 { 0 } else { self.samples.len() / self.channels }
    }

    /// Duration in seconds.
    pub fn duration_secs(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.samples_per_channel() as f64 / self.sample_rate as f64
    }
}

/// Decode a contiguous run of frames, optionally priming on earlier frames whose
/// output is discarded.
///
/// `prime` frames before `spans[0]` are decoded to warm the overlap buffer and the
/// window-shape history; they must be the frames immediately preceding this run.
fn decode_run(
    stream: &[u8],
    prime: &[FrameSpan],
    spans: &[FrameSpan],
    config: &AudioSpecificConfig,
    first_frame_index: u64,
) -> DecodedAudio {
    let mut decoder = Decoder::new(config.clone());
    // Position-independent noise seeding is what lets a chunk start anywhere.
    decoder.set_noise_mode(NoiseMode::PerFrame);
    // Frame indices drive that seeding, so a chunk must number its frames the way a
    // whole-stream decode would.
    decoder.set_frame_index(first_frame_index.saturating_sub(prime.len() as u64));

    for span in prime {
        let _ = decoder.decode_frame(&stream[span.start..span.end]);
    }

    let mut samples = Vec::with_capacity(spans.len() * 2048);
    let mut frames = 0usize;
    let mut channels = decoder.channels();

    for span in spans {
        if let Ok(pcm) = decoder.decode_frame(&stream[span.start..span.end]) {
            channels = pcm.channels();
            let len = pcm.samples_per_channel();
            for s in 0..len {
                for c in 0..channels {
                    samples.push(pcm.channel(c)[s]);
                }
            }
            frames += 1;
        }
    }

    DecodedAudio { samples, channels, sample_rate: decoder.sample_rate_hz(), frames }
}

/// Decode a whole ADTS stream on one thread, using [`NoiseMode::PerFrame`].
pub fn decode_stream(stream: &[u8], config: &AudioSpecificConfig) -> Result<DecodedAudio> {
    let spans = scan_adts_frames(stream);
    Ok(decode_run(stream, &[], &spans, config, 0))
}

/// Decode a whole ADTS stream across a rayon thread pool.
///
/// Produces exactly the same samples as [`decode_stream`].
#[cfg(feature = "rayon")]
pub fn decode_stream_parallel(
    stream: &[u8],
    config: &AudioSpecificConfig,
) -> Result<DecodedAudio> {
    use rayon::prelude::*;

    let spans = scan_adts_frames(stream);
    if spans.is_empty() {
        return Ok(DecodedAudio {
            samples: Vec::new(),
            channels: 0,
            sample_rate: config.sampling_rate.hz(),
            frames: 0,
        });
    }

    // One chunk per worker, but never so small that the priming frame dominates.
    let threads = rayon::current_num_threads().max(1);
    let min_chunk = 32usize;
    let chunk = (spans.len().div_ceil(threads)).max(min_chunk);
    if chunk >= spans.len() {
        return Ok(decode_run(stream, &[], &spans, config, 0));
    }

    let parts: Vec<DecodedAudio> = spans
        .par_chunks(chunk)
        .enumerate()
        .map(|(i, part)| {
            let start = i * chunk;
            // One preceding frame is enough: overlap-add and window-shape history
            // both reach back exactly one frame.
            let prime = if start == 0 { &spans[0..0] } else { &spans[start - 1..start] };
            decode_run(stream, prime, part, config, start as u64)
        })
        .collect();

    let channels = parts.iter().map(|p| p.channels).max().unwrap_or(0);
    let sample_rate = parts.first().map_or(config.sampling_rate.hz(), |p| p.sample_rate);
    let total: usize = parts.iter().map(|p| p.samples.len()).sum();
    let frames = parts.iter().map(|p| p.frames).sum();

    let mut samples = Vec::with_capacity(total);
    for p in parts {
        samples.extend_from_slice(&p.samples);
    }

    Ok(DecodedAudio { samples, channels, sample_rate, frames })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::syntax::adts::AdtsHeader;
    use crate::types::{AudioObjectType, ChannelConfiguration, SamplingRate};

    /// Scanning must find every frame of a well-formed stream and nothing else.
    #[test]
    fn scans_well_formed_frames() {
        let mut stream = Vec::new();
        let mut expected = Vec::new();
        for len in [64usize, 128, 96] {
            let header = AdtsHeader {
                mpeg_id: 0,
                layer: 0,
                protection_absent: true,
                audio_object_type: AudioObjectType::AacLc,
                sampling_rate: SamplingRate::Hz44100,
                channel_config: ChannelConfiguration::Stereo,
                frame_length: len,
                buffer_fullness: 0x7FF,
                num_raw_data_blocks: 0,
                crc: None,
            };
            let mut w = crate::bitstream::BitWriter::with_capacity(len);
            header.write(&mut w);
            let mut bytes = w.into_bytes();
            bytes.resize(len, 0);
            expected.push(FrameSpan { start: stream.len(), end: stream.len() + len });
            stream.extend_from_slice(&bytes);
        }

        assert_eq!(scan_adts_frames(&stream), expected);
    }

    /// Leading garbage must not prevent the real frames from being found.
    #[test]
    fn resynchronizes_past_garbage() {
        let header = AdtsHeader {
            mpeg_id: 0,
            layer: 0,
            protection_absent: true,
            audio_object_type: AudioObjectType::AacLc,
            sampling_rate: SamplingRate::Hz44100,
            channel_config: ChannelConfiguration::Stereo,
            frame_length: 64usize,
            buffer_fullness: 0x7FF,
            num_raw_data_blocks: 0,
            crc: None,
        };
        let mut w = crate::bitstream::BitWriter::with_capacity(64);
        header.write(&mut w);
        let mut frame = w.into_bytes();
        frame.resize(64, 0);

        let mut stream = vec![0x12, 0x34, 0x56, 0xFF, 0x00];
        let start = stream.len();
        stream.extend_from_slice(&frame);

        let spans = scan_adts_frames(&stream);
        assert_eq!(spans, vec![FrameSpan { start, end: start + 64 }]);
    }

    /// An empty or truncated stream yields no frames rather than panicking.
    #[test]
    fn handles_degenerate_streams() {
        assert!(scan_adts_frames(&[]).is_empty());
        assert!(scan_adts_frames(&[0xFF]).is_empty());
        assert!(scan_adts_frames(&[0xFF, 0xF1, 0x50, 0x80]).is_empty());
    }
}
