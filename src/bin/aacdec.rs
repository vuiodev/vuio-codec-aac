//! Command-line AAC Audio Decoder Tool (`aacdec`)
//!
//! Decodes MPEG-4 AAC / HE-AAC bitstreams (ADTS/RAW) to uncompressed 16-bit PCM WAV.

use clap::Parser;
use hound::{WavSpec, WavWriter};
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;
use vuiocodecaac::bitstream::BitReader;
use vuiocodecaac::prelude::*;
use vuiocodecaac::syntax::adts::AdtsHeader;
use vuiocodecaac::syntax::asc::AudioSpecificConfig;

#[derive(Parser, Debug)]
#[command(
    name = "aacdec",
    author = "Vuio AAC",
    version,
    about = "High-performance MPEG AAC / HE-AAC / USAC Audio Decoder in pure Rust (2024)",
    long_about = None
)]
struct Args {
    /// Input AAC / ADTS audio file path
    #[arg(required = true)]
    input: PathBuf,

    /// Output WAV audio file path
    #[arg(required = true)]
    output: PathBuf,

    /// Decode without writing output, and report decode throughput alone.
    #[arg(long)]
    bench: bool,

    /// Repeat the decode this many times (implies --bench).
    #[arg(long, default_value_t = 1)]
    repeat: u32,

    /// Decode across a thread pool.
    ///
    /// Uses position-independent noise seeding, so output matches a single-threaded
    /// run of the same mode exactly; it differs from the default sequential mode
    /// only inside noise-substituted bands.
    #[arg(long)]
    parallel: bool,
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    let args = Args::parse();

    let mut file = File::open(&args.input)?;
    let mut bitstream = Vec::new();
    file.read_to_end(&mut bitstream)?;

    let bench = args.bench || args.repeat > 1;

    // The USAC FD path uses its own small non-ADTS container (see
    // `decoder::usac::container`'s docs) since real ADTS cannot carry USAC's
    // audio object type at all (its `profile` field is only 2 bits) — check for
    // it before anything else, since its magic bytes never look like ADTS sync.
    if vuiocodecaac::decoder::usac::container::is_usac_container(&bitstream) {
        return decode_usac_container(&args, &bitstream, bench);
    }

    if args.parallel {
        return run_parallel(&args, &bitstream, bench);
    }

    let mut decoder = Decoder::new_default();
    let mut all_pcm_samples: Vec<i16> = Vec::new();
    let mut offset;
    let mut frame_count = 0;

    // Decode-only timing, excluding file reading and WAV encoding.
    let decode_start = Instant::now();
    for pass in 0..args.repeat.max(1) {
        let keep = !bench && pass == 0;
        decoder.reset();
        all_pcm_samples.clear();
        offset = 0;
        frame_count = 0;

        while offset + 7 <= bitstream.len() {
            if bitstream[offset] == 0xFF && (bitstream[offset + 1] & 0xF0) == 0xF0 {
                let mut reader = BitReader::new(&bitstream[offset..]);
                if let Ok(adts) = AdtsHeader::parse(&mut reader) {
                    let frame_len = adts.frame_length as usize;
                    if offset + frame_len <= bitstream.len() {
                        let frame_data = &bitstream[offset..offset + frame_len];
                        if let Ok(pcm_frame) = decoder.decode_frame(frame_data) {
                            let ch = pcm_frame.channels();
                            let len = pcm_frame.samples_per_channel();
                            if keep {
                                for s in 0..len {
                                    for c in 0..ch {
                                        all_pcm_samples.push(pcm_frame.channel(c)[s]);
                                    }
                                }
                            } else {
                                // Keep the decode observable so it is not optimized away.
                                std::hint::black_box(pcm_frame.channel(0)[0]);
                            }
                            frame_count += 1;
                        }
                        offset += frame_len;
                        continue;
                    }
                }
            }
            offset += 1;
        }
    }
    let decode_elapsed = decode_start.elapsed().as_secs_f64() / args.repeat.max(1) as f64;

    let frames_decoded = frame_count;
    let audio_secs =
        frames_decoded as f64 * decoder.frame_length() as f64 / decoder.sample_rate_hz() as f64;
    println!(
        "decode only: {:.2}s audio in {:.4}s = {:.1}x real-time ({:.2} us/frame)",
        audio_secs,
        decode_elapsed,
        audio_secs / decode_elapsed.max(1e-9),
        decode_elapsed * 1e6 / frames_decoded.max(1) as f64,
    );

    if bench {
        return Ok(());
    }

    let spec = WavSpec {
        channels: decoder.channels() as u16,
        sample_rate: decoder.sample_rate_hz(),
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut wav_writer = WavWriter::create(&args.output, spec)?;
    for &sample in &all_pcm_samples {
        wav_writer.write_sample(sample)?;
    }
    wav_writer.finalize()?;

    let elapsed = start_time.elapsed().as_secs_f64();
    let num_ch = decoder.channels().max(1) as f64;
    let total_audio_samples = all_pcm_samples.len() as f64 / num_ch;
    let audio_duration = total_audio_samples / decoder.sample_rate_hz().max(1) as f64;
    let speed = audio_duration / elapsed.max(1e-6);

    println!(
        "Decoded {} frames ({:.2}s audio in {:.3}s, speed={:.1}x real-time) -> {:?}",
        frame_count, audio_duration, elapsed, speed, args.output
    );
    Ok(())
}

/// Decode a stream written by `encoder::usac::container::encode` — the minimal
/// USAC FD codec's own container, not standard ADTS (see that module's docs).
fn decode_usac_container(
    args: &Args,
    bitstream: &[u8],
    bench: bool,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    let decoded = vuiocodecaac::decoder::usac::container::decode(bitstream)?;

    let frames = decoded.samples.len() / decoded.channels.max(1);
    let audio_duration = frames as f64 / decoded.sample_rate_hz as f64;
    println!(
        "decode only: {:.2}s audio ({} channel(s) @ {} Hz, non-standard USAC FD container)",
        audio_duration, decoded.channels, decoded.sample_rate_hz
    );

    if bench {
        return Ok(());
    }

    let spec = hound::WavSpec {
        channels: decoded.channels as u16,
        sample_rate: decoded.sample_rate_hz,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut wav_writer = WavWriter::create(&args.output, spec)?;
    for &sample in &decoded.samples {
        wav_writer.write_sample(sample)?;
    }
    wav_writer.finalize()?;

    let elapsed = start_time.elapsed().as_secs_f64();
    println!(
        "Decoded {} frames ({:.2}s audio in {:.3}s) -> {:?}",
        frames, audio_duration, elapsed, args.output
    );
    Ok(())
}

/// Decode the whole stream across a thread pool.
fn run_parallel(
    args: &Args,
    bitstream: &[u8],
    bench: bool,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    use vuiocodecaac::decoder::batch::{decode_stream_parallel, scan_adts_frames};
    use vuiocodecaac::types::{AudioObjectType, ChannelConfiguration, FrameLength, SamplingRate};

    // Take the stream parameters from the first frame's ADTS header.
    let mut config = AudioSpecificConfig {
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
    if let Some(span) = scan_adts_frames(bitstream).first() {
        let mut r = BitReader::new(&bitstream[span.start..span.end]);
        if let Ok(h) = AdtsHeader::parse(&mut r) {
            config.sampling_rate = h.sampling_rate;
            config.channel_config = h.channel_config;
            config.audio_object_type = h.audio_object_type;
        }
    }

    let start = Instant::now();
    let mut decoded = decode_stream_parallel(bitstream, &config)?;
    for _ in 1..args.repeat.max(1) {
        decoded = decode_stream_parallel(bitstream, &config)?;
    }
    let elapsed = start.elapsed().as_secs_f64() / args.repeat.max(1) as f64;

    println!(
        "decode only: {:.2}s audio in {:.4}s = {:.1}x real-time ({:.2} us/frame, {} threads)",
        decoded.duration_secs(),
        elapsed,
        decoded.duration_secs() / elapsed.max(1e-9),
        elapsed * 1e6 / decoded.frames.max(1) as f64,
        rayon::current_num_threads(),
    );

    if bench {
        return Ok(());
    }

    let spec = WavSpec {
        channels: decoded.channels as u16,
        sample_rate: decoded.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut wav = WavWriter::create(&args.output, spec)?;
    for &s in &decoded.samples {
        wav.write_sample(s)?;
    }
    wav.finalize()?;
    println!("Decoded {} frames -> {:?}", decoded.frames, args.output);
    Ok(())
}
