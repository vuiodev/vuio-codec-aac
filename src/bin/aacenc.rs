//! Command-line AAC Audio Encoder Tool (`aacenc`)
//!
//! Encodes multi-channel PCM WAV audio to standard-compliant MPEG-4 AAC / HE-AAC bitstreams.

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use clap::Parser;
use hound::WavReader;
use vuiocodecaac::prelude::*;

#[derive(Parser, Debug)]
#[command(
    name = "aacenc",
    author = "Antigravity AAC Team",
    version,
    about = "High-performance MPEG AAC / HE-AAC / USAC Audio Encoder in pure Rust (2024)",
    long_about = None
)]
struct Args {
    /// Input WAV audio file path
    #[arg(required = true)]
    input: PathBuf,

    /// Output AAC / ADTS audio file path
    #[arg(required = true)]
    output: PathBuf,

    /// Target bitrate in bits per second (e.g. 64000, 128000, 192000, 256000, 320000)
    #[arg(short, long, default_value_t = 128000)]
    bitrate: u32,

    /// Audio Object Type / Profile: lc (AAC-LC), he (HE-AAC/SBR), he-v2 (HE-AAC v2/PS), ld (AAC-LD), eld (AAC-ELD)
    #[arg(short, long, default_value = "lc")]
    profile: String,

    /// Number of worker threads for parallel multi-threaded encoding (0 = auto)
    #[arg(short, long, default_value_t = 0)]
    threads: usize,
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    println!("Encoding {:?} -> {:?}", args.input, args.output);

    let mut wav_reader = WavReader::open(&args.input)?;
    let spec = wav_reader.spec();

    println!(
        "Input format: {} Hz, {} channels, {} bits",
        spec.sample_rate, spec.channels, spec.bits_per_sample
    );

    let sampling_rate = SamplingRate::from_hz(spec.sample_rate);
    let channel_config = ChannelConfiguration::from_u8(spec.channels as u8)
        .unwrap_or(ChannelConfiguration::Stereo);

    let aot = match args.profile.to_lowercase().as_str() {
        "lc" | "aac-lc" | "aac_lc" => AudioObjectType::AacLc,
        "he" | "he-aac" | "sbr" => AudioObjectType::Sbr,
        "he-v2" | "he-aac-v2" | "ps" => AudioObjectType::Ps,
        "ld" | "aac-ld" => AudioObjectType::ErAacLd,
        "eld" | "aac-eld" => AudioObjectType::ErAacEld,
        "usac" => AudioObjectType::Usac,
        _ => AudioObjectType::AacLc,
    };

    let config = EncoderConfig {
        audio_object_type: aot,
        sampling_rate,
        channel_config,
        bitrate_bps: args.bitrate,
        frame_length: FrameLength::Samples1024,
    };

    let mut encoder = Encoder::new(config)?;
    let mut out_file = File::create(&args.output)?;

    let samples: Vec<i16> = wav_reader.samples::<i16>().map(|s| s.unwrap_or(0)).collect();
    let num_ch = spec.channels as usize;
    let frame_len = 1024;
    let total_frames = samples.len() / (num_ch * frame_len);

    let mut pcm_frame = AudioBuffer::<i16>::new(num_ch, frame_len);
    let mut encoded_frames = 0;

    for f in 0..total_frames {
        let start = f * num_ch * frame_len;
        let frame_slice = &samples[start..start + num_ch * frame_len];
        pcm_frame.from_interleaved(frame_slice);

        let adts_packet = encoder.encode_frame(&pcm_frame)?;
        out_file.write_all(&adts_packet)?;
        encoded_frames += 1;
    }

    println!(
        "Successfully encoded {} frames ({:.2}s, profile: {:?}, {} kbps) to {:?}",
        encoded_frames,
        (encoded_frames * frame_len) as f32 / spec.sample_rate as f32,
        aot,
        args.bitrate / 1000,
        args.output
    );
    Ok(())
}
