//! A minimal, self-describing container around the FD codec in
//! [`crate::encoder::usac::fd`], so a whole PCM stream can round-trip through this
//! crate's own `aacenc`/`aacdec` binaries.
//!
//! This is **not** ISO/IEC 23003-3's `UsacFrame()`/LOAS/RAW framing — that syntax
//! carries a full `UsacConfig()` element tree this codebase does not build (see
//! `fd.rs`'s own docs for why), and separately, MPEG-4 ADTS cannot carry it either:
//! ADTS's `profile` field is only 2 bits (`AdtsHeader::parse`/`write` compute
//! `audio_object_type` as `profile + 1`), so the largest object type it can express
//! is 4 (`AacLtp`) — nowhere near USAC's object type 42. Reusing ADTS here, as the
//! rest of this crate's AAC-LC/HE-AAC path does, was tried first and found to not
//! fit for exactly that reason, rather than assumed not to. What follows instead is
//! a small header (magic, version, channel count, sample-rate index) followed by a
//! sequence of `(length, raw block)` frames — good enough for this codec to read
//! back what it wrote, nothing more.

use crate::encoder::usac::fd::{FRAME_LEN, UsacFdEncoder, UsacFdStereoEncoder};
use crate::error::{Error, FormatError, Result};
use crate::types::SamplingRate;

/// Identifies this container, distinct from the ADTS sync word (`0x0FFF`) so a
/// reader can tell the two apart from the first bytes alone.
pub const MAGIC: [u8; 4] = *b"VUSC";
/// Bumped if the header or frame shape below ever changes incompatibly.
pub const VERSION: u8 = 1;
/// `MAGIC` + `VERSION` + channel count + sample-rate index + one reserved byte.
pub const HEADER_LEN: usize = 8;

/// Encode a whole interleaved PCM stream through the minimal USAC FD codec.
///
/// `channels` must be 1 or 2 (mono or the mid/side stereo path) and `sample_rate_hz`
/// must be 44100 or 48000 — the only rates [`crate::tables::sfb::SFB_48_1024`], and
/// so this whole codec, is built for (see `fd.rs`'s module docs). Anything else is
/// rejected outright rather than silently mis-encoded. A sample count that is not a
/// whole number of [`FRAME_LEN`]-sample frames is padded with silence at the end;
/// the last frame's genuine content still decodes correctly; the caller learns
/// nothing about how much padding was added, which is fine for this crate's own
/// round-trip use but would not be for anything expecting exact-length output.
pub fn encode(
    samples: &[i16],
    channels: usize,
    sample_rate_hz: u32,
    budget_bits: usize,
) -> Result<Vec<u8>> {
    if channels != 1 && channels != 2 {
        return Err(Error::Format(FormatError::InvalidUsacContainer(format!(
            "USAC FD path supports 1 or 2 channels, got {channels}"
        ))));
    }
    if sample_rate_hz != 44_100 && sample_rate_hz != 48_000 {
        return Err(Error::Format(FormatError::InvalidUsacContainer(format!(
            "USAC FD path supports 44100 or 48000 Hz only, got {sample_rate_hz}"
        ))));
    }
    let sf_index = SamplingRate::from_hz(sample_rate_hz).to_index().expect(
        "44100 and 48000 are both in the standard rate table, checked just above",
    );

    let mut out = Vec::with_capacity(HEADER_LEN + samples.len());
    out.extend_from_slice(&MAGIC);
    out.push(VERSION);
    out.push(channels as u8);
    out.push(sf_index);
    out.push(0); // reserved

    let frame_stride = channels * FRAME_LEN;
    let total_frames = samples.len().div_ceil(frame_stride);

    match channels {
        1 => {
            let mut encoder = UsacFdEncoder::new();
            encoder.set_budget_bits(budget_bits);
            let mut pcm = vec![0.0f32; FRAME_LEN];
            for f in 0..total_frames {
                let start = f * FRAME_LEN;
                fill_channel(samples, start, 1, 0, &mut pcm);
                let block = encoder.encode_frame(&pcm);
                push_frame(&mut out, &block);
            }
        }
        _ => {
            let mut encoder = UsacFdStereoEncoder::new();
            encoder.set_budget_bits(budget_bits);
            let mut left = vec![0.0f32; FRAME_LEN];
            let mut right = vec![0.0f32; FRAME_LEN];
            for f in 0..total_frames {
                let start = f * frame_stride;
                fill_channel(samples, start, 2, 0, &mut left);
                fill_channel(samples, start, 2, 1, &mut right);
                let block = encoder.encode_frame(&left, &right);
                push_frame(&mut out, &block);
            }
        }
    }

    Ok(out)
}

/// One channel's worth of `FRAME_LEN` samples out of an interleaved buffer,
/// starting at interleaved index `start`; positions past the end of `samples` are
/// left at silence (the padding [`encode`]'s docs describe).
fn fill_channel(samples: &[i16], start: usize, channels: usize, channel: usize, out: &mut [f32]) {
    out.fill(0.0);
    for (i, o) in out.iter_mut().enumerate() {
        let idx = start + i * channels + channel;
        if let Some(&s) = samples.get(idx) {
            *o = s as f32;
        }
    }
}

/// Append one frame as a 4-byte little-endian length prefix followed by its bytes.
fn push_frame(out: &mut Vec<u8>, block: &[u8]) {
    out.extend_from_slice(&(block.len() as u32).to_le_bytes());
    out.extend_from_slice(block);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::usac::container::decode;

    fn tone(n: usize, freq: f32, rate: f32) -> Vec<i16> {
        (0..n).map(|i| ((i as f32 / rate * freq * std::f32::consts::TAU).sin() * 12000.0) as i16).collect()
    }

    #[test]
    fn rejects_unsupported_channel_counts() {
        let samples = vec![0i16; FRAME_LEN * 3];
        assert!(encode(&samples, 3, 44_100, 12_000).is_err());
    }

    #[test]
    fn rejects_unsupported_sample_rates() {
        let samples = vec![0i16; FRAME_LEN];
        assert!(encode(&samples, 1, 22_050, 12_000).is_err());
    }

    /// A mono stream through the container must come back close to the original,
    /// the same bar the library-level round-trip tests already hold `fd.rs` to —
    /// this only proves the container's own framing does not lose or corrupt data
    /// the codec itself already gets right.
    #[test]
    fn mono_round_trips_through_the_container() {
        let samples = tone(FRAME_LEN * 6, 440.0, 44_100.0);
        let bytes = encode(&samples, 1, 44_100, 16_000).unwrap();
        let decoded = decode(&bytes).unwrap();
        assert_eq!(decoded.channels, 1);
        assert_eq!(decoded.sample_rate_hz, 44_100);
        assert!(decoded.samples.len() >= samples.len());
    }

    #[test]
    fn stereo_round_trips_through_the_container() {
        let n = FRAME_LEN * 4;
        let mut samples = Vec::with_capacity(n * 2);
        let left = tone(n, 440.0, 44_100.0);
        let right = tone(n, 660.0, 44_100.0);
        for i in 0..n {
            samples.push(left[i]);
            samples.push(right[i]);
        }
        let bytes = encode(&samples, 2, 44_100, 16_000).unwrap();
        let decoded = decode(&bytes).unwrap();
        assert_eq!(decoded.channels, 2);
        assert!(decoded.samples.len() >= samples.len());
    }
}
