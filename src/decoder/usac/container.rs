//! Reads back what [`crate::encoder::usac::container`] writes — see that module's
//! docs for why this container exists and is not ISO/IEC 23003-3's real framing.

use crate::bitstream::BitReader;
use crate::decoder::usac::fd::{FRAME_LEN, UsacFdDecoder, UsacFdStereoDecoder};
use crate::encoder::usac::container::{HEADER_LEN, MAGIC, VERSION};
use crate::error::{Error, FormatError, Result};
use crate::types::SamplingRate;

/// One decoded stream: interleaved 16-bit PCM plus the channel count and sample
/// rate the container's header carried.
pub struct DecodedUsac {
    pub samples: Vec<i16>,
    pub channels: usize,
    pub sample_rate_hz: u32,
}

/// True if `bytes` starts with this container's magic — the caller's cue to route
/// here instead of the ADTS/AAC-LC path, before trying to parse anything else.
pub fn is_usac_container(bytes: &[u8]) -> bool {
    bytes.len() >= HEADER_LEN && bytes[..4] == MAGIC
}

/// Decode a whole container produced by [`crate::encoder::usac::container::encode`].
pub fn decode(bytes: &[u8]) -> Result<DecodedUsac> {
    if !is_usac_container(bytes) {
        return Err(Error::Format(FormatError::InvalidUsacContainer(
            "missing VUSC magic".to_string(),
        )));
    }
    if bytes[4] != VERSION {
        return Err(Error::Format(FormatError::InvalidUsacContainer(format!(
            "unsupported container version {}, expected {VERSION}",
            bytes[4]
        ))));
    }
    let channels = bytes[5] as usize;
    let sample_rate_hz = SamplingRate::from_index(bytes[6])
        .ok_or_else(|| {
            Error::Format(FormatError::InvalidUsacContainer(format!(
                "invalid sample-rate index {}",
                bytes[6]
            )))
        })?
        .hz();

    let mut samples = Vec::new();
    let mut offset = HEADER_LEN;

    match channels {
        1 => {
            let mut decoder = UsacFdDecoder::new();
            while let Some((frame, next)) = read_frame(bytes, offset)? {
                let mut reader = BitReader::new(frame);
                let pcm = decoder.decode_frame(&mut reader)?;
                extend_with_channel(&mut samples, &pcm);
                offset = next;
            }
        }
        2 => {
            let mut decoder = UsacFdStereoDecoder::new();
            while let Some((frame, next)) = read_frame(bytes, offset)? {
                let mut reader = BitReader::new(frame);
                let (left, right) = decoder.decode_frame(&mut reader)?;
                for i in 0..FRAME_LEN {
                    samples.push(clamp_to_i16(left[i]));
                    samples.push(clamp_to_i16(right[i]));
                }
                offset = next;
            }
        }
        other => {
            return Err(Error::Format(FormatError::InvalidUsacContainer(format!(
                "USAC FD path supports 1 or 2 channels, got {other}"
            ))));
        }
    }

    Ok(DecodedUsac { samples, channels, sample_rate_hz })
}

/// Push one mono frame's samples, saturating to `i16`.
fn extend_with_channel(samples: &mut Vec<i16>, pcm: &[f32]) {
    samples.extend(pcm.iter().map(|&v| clamp_to_i16(v)));
}

/// Same saturating rounding [`crate::decoder::engine`] uses for its own PCM
/// output, duplicated here rather than made `pub` there — this container is a
/// small, separate path, the same convention `encoder::usac::fd`/`decoder::usac::fd`
/// already established for this pair of modules.
fn clamp_to_i16(v: f32) -> i16 {
    v.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

/// Read one `(length, bytes)` frame at `offset`, returning the frame's payload and
/// the offset just past it, or `None` once the stream is exhausted.
fn read_frame(bytes: &[u8], offset: usize) -> Result<Option<(&[u8], usize)>> {
    if offset == bytes.len() {
        return Ok(None);
    }
    if offset + 4 > bytes.len() {
        return Err(Error::Format(FormatError::InvalidUsacContainer(
            "truncated frame length prefix".to_string(),
        )));
    }
    let len = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
    let start = offset + 4;
    let end = start + len;
    if end > bytes.len() {
        return Err(Error::Format(FormatError::InvalidUsacContainer(format!(
            "frame at offset {offset} declares {len} bytes, only {} remain",
            bytes.len() - start
        ))));
    }
    Ok(Some((&bytes[start..end], end)))
}
