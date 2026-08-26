//! Parallel batch decoding must be byte-identical to sequential decoding.
//!
//! The parallel path primes each chunk on the frame before it, which is only exact
//! if every piece of cross-frame state reaches back at most one frame. This test is
//! what holds that property: if a future change adds longer-lived state (a predictor
//! history, a stream-global generator), these comparisons break.

use vuiocodecaac::decoder::aac::pns::NoiseMode;
use vuiocodecaac::decoder::batch::{decode_stream, scan_adts_frames};
use vuiocodecaac::syntax::asc::AudioSpecificConfig;
use vuiocodecaac::syntax::adts::AdtsHeader;
use vuiocodecaac::bitstream::BitReader;
use vuiocodecaac::types::{
    AudioObjectType, ChannelConfiguration, FrameLength, SamplingRate,
};

/// Locate a corpus AAC file produced by `tools/verify_corpus.sh`, if present.
fn corpus_file() -> Option<Vec<u8>> {
    let dir = std::env::var("AAC_CORPUS").ok()?;
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "aac"))
        .collect();
    entries.sort();
    let path = entries.first()?;
    std::fs::read(path).ok()
}

fn config_from(stream: &[u8]) -> AudioSpecificConfig {
    let mut base = AudioSpecificConfig {
        audio_object_type: AudioObjectType::AacLc,
        sampling_rate: SamplingRate::Hz44100,
        channel_config: ChannelConfiguration::Stereo,
        frame_length: FrameLength::Samples1024,
        depends_on_core_coder: false,
        core_coder_delay: 0,
        extension_audio_object_type: None,
        extension_sampling_rate: None,
        sbr_present: false,
        ps_present: false,
    };
    if let Some(span) = scan_adts_frames(stream).first() {
        let mut r = BitReader::new(&stream[span.start..span.end]);
        if let Ok(h) = AdtsHeader::parse(&mut r) {
            base.sampling_rate = h.sampling_rate;
            base.channel_config = h.channel_config;
            base.audio_object_type = h.audio_object_type;
        }
    }
    base
}

#[cfg(feature = "rayon")]
#[test]
fn parallel_matches_sequential() {
    use vuiocodecaac::decoder::batch::decode_stream_parallel;

    let Some(stream) = corpus_file() else {
        eprintln!("AAC_CORPUS not set; skipping");
        return;
    };
    let config = config_from(&stream);

    let serial = decode_stream(&stream, &config).expect("sequential decode");
    let parallel = decode_stream_parallel(&stream, &config).expect("parallel decode");

    assert_eq!(serial.frames, parallel.frames, "frame counts differ");
    assert_eq!(serial.channels, parallel.channels, "channel counts differ");
    assert_eq!(
        serial.samples.len(),
        parallel.samples.len(),
        "sample counts differ"
    );
    let mismatches = serial
        .samples
        .iter()
        .zip(parallel.samples.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(mismatches, 0, "{mismatches} samples differ between serial and parallel decode");
}

/// Decoding a run of frames starting mid-stream, primed on the preceding frame,
/// must match what a decode from the start produces for those same frames.
///
/// This is the property that makes seeking exact, and it holds only in
/// [`NoiseMode::PerFrame`]: the default sequential generator's state depends on
/// every frame before the seek point.
#[test]
fn seeking_with_one_primed_frame_matches() {
    let Some(stream) = corpus_file() else {
        eprintln!("AAC_CORPUS not set; skipping");
        return;
    };
    let config = config_from(&stream);
    let spans = scan_adts_frames(&stream);
    if spans.len() < 20 {
        eprintln!("corpus file too short; skipping");
        return;
    }

    let full = decode_stream(&stream, &config).expect("full decode");
    let ch = full.channels.max(1);
    let per_frame = full.samples_per_channel() / full.frames.max(1);

    // Decode frames 10.. after priming on frame 9, and compare against the same
    // region of the full decode.
    let mut decoder = vuiocodecaac::decoder::Decoder::new(config.clone());
    decoder.set_noise_mode(NoiseMode::PerFrame);
    decoder.set_frame_index(9);
    let _ = decoder.decode_frame(&stream[spans[9].start..spans[9].end]);

    let mut got = Vec::new();
    for span in &spans[10..20] {
        if let Ok(pcm) = decoder.decode_frame(&stream[span.start..span.end]) {
            for s in 0..pcm.samples_per_channel() {
                for c in 0..pcm.channels() {
                    got.push(pcm.channel(c)[s]);
                }
            }
        }
    }

    let start = 10 * per_frame * ch;
    let want = &full.samples[start..start + got.len()];
    let mismatches = got.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
    assert_eq!(mismatches, 0, "{mismatches} samples differ after seeking");
}

/// The frame scanner must partition the stream without gaps or overlaps.
#[test]
fn frame_spans_do_not_overlap() {
    let Some(stream) = corpus_file() else {
        eprintln!("AAC_CORPUS not set; skipping");
        return;
    };
    let spans = scan_adts_frames(&stream);
    assert!(!spans.is_empty(), "no frames found in corpus file");
    for pair in spans.windows(2) {
        assert!(pair[0].end <= pair[1].start, "frames {:?} and {:?} overlap", pair[0], pair[1]);
        assert!(pair[0].start < pair[0].end, "empty span {:?}", pair[0]);
    }
}
