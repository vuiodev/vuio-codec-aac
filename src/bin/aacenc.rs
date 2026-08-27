//! Command-line AAC Audio Encoder Tool (`aacenc`)
//!
//! Encodes multi-channel PCM WAV audio to standard-compliant MPEG-4 AAC / HE-AAC bitstreams
//! with optional Rayon multi-threaded parallel batch execution.

use clap::Parser;
use hound::WavReader;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;
use vuiocodecaac::prelude::*;

#[derive(Parser, Debug)]
#[command(
    name = "aacenc",
    author = "Vuio AAC",
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
    let start_time = Instant::now();
    let args = Args::parse();

    let mut wav_reader = WavReader::open(&args.input)?;
    let spec = wav_reader.spec();

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

    let samples: Vec<i16> = wav_reader.samples::<i16>().map(|s| s.unwrap_or(0)).collect();
    let num_ch = spec.channels as usize;
    let frame_len = 1024;
    let frame_stride = num_ch * frame_len;
    let total_frames = samples.len() / frame_stride;

    let encoded_packets: Vec<Vec<u8>> = if args.threads == 1 {
        let mut encoder = Encoder::new(config)?;
        let mut pcm_frame = AudioBuffer::<i16>::new(num_ch, frame_len);
        let mut packets = Vec::with_capacity(total_frames);
        for f in 0..total_frames {
            let start = f * frame_stride;
            pcm_frame.from_interleaved(&samples[start..start + frame_stride]);
            packets.push(encoder.encode_frame(&pcm_frame)?);
        }
        packets.push(encoder.flush()?);
        packets
    } else {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            let base_encoder = Encoder::new(config.clone())?;
            let num_threads = if args.threads > 0 { args.threads } else { rayon::current_num_threads() };
            let chunk_frames = (total_frames / num_threads.max(1)).max(1);
            let chunk_samples = chunk_frames * frame_stride;

            samples
                .par_chunks(chunk_samples)
                .flat_map_iter(|chunk| {
                    let mut enc = base_encoder.clone();
                    let mut pcm = AudioBuffer::<i16>::new(num_ch, frame_len);
                    let n_frames = chunk.len() / frame_stride;
                    let mut local_packets = Vec::with_capacity(n_frames);
                    for f in 0..n_frames {
                        let start = f * frame_stride;
                        pcm.from_interleaved(&chunk[start..start + frame_stride]);
                        local_packets.push(enc.encode_frame(&pcm).unwrap());
                    }
                    // Each chunk runs its own encoder with its own lookahead, so
                    // without this the chunk's last frame stays held back forever
                    // and is silently dropped instead of landing in the next chunk.
                    local_packets.push(enc.flush().unwrap());
                    local_packets
                })
                .collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            let mut encoder = Encoder::new(config)?;
            let mut pcm_frame = AudioBuffer::<i16>::new(num_ch, frame_len);
            let mut packets = Vec::with_capacity(total_frames);
            for f in 0..total_frames {
                let start = f * frame_stride;
                pcm_frame.from_interleaved(&samples[start..start + frame_stride]);
                packets.push(encoder.encode_frame(&pcm_frame)?);
            }
            packets
        }
    };

    let out_file = File::create(&args.output)?;
    let mut buf_writer = BufWriter::with_capacity(65536, out_file);
    for packet in &encoded_packets {
        buf_writer.write_all(packet)?;
    }
    buf_writer.flush()?;

    let elapsed = start_time.elapsed().as_secs_f64();
    let audio_duration = (total_frames * frame_len) as f64 / spec.sample_rate as f64;
    let speed = audio_duration / elapsed.max(1e-6);

    println!(
        "Encoded {} frames ({:.2}s audio in {:.3}s, speed={:.1}x real-time, profile: {:?}, {} kbps) -> {:?}",
        total_frames,
        audio_duration,
        elapsed,
        speed,
        aot,
        args.bitrate / 1000,
        args.output
    );
    Ok(())
}
