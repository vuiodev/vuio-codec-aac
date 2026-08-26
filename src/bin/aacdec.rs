//! Command-line AAC Audio Decoder Tool

use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use clap::Parser;
use hound::{WavSpec, WavWriter};
use xaac::prelude::*;

#[derive(Parser, Debug)]
#[command(name = "aacdec", author, version, about = "High-performance MPEG AAC / USAC / DRC Audio Decoder in pure Rust", long_about = None)]
struct Args {
    /// Input AAC / ADTS / MP4 audio file path
    #[arg(required = true)]
    input: PathBuf,

    /// Output WAV audio file path
    #[arg(required = true)]
    output: PathBuf,

    /// Number of worker threads for multi-threaded decoding
    #[arg(short, long, default_value_t = 0)]
    threads: usize,
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    println!("Decoding {:?} -> {:?}", args.input, args.output);

    let mut input_file = File::open(&args.input)?;
    let mut encoded_data = Vec::new();
    input_file.read_to_end(&mut encoded_data)?;

    let mut decoder = Decoder::new_default();
    let mut all_pcm_samples = Vec::new();

    // Parse ADTS stream frames
    let mut offset = 0;
    let mut frame_count = 0;

    while offset + 7 <= encoded_data.len() {
        if encoded_data[offset] == 0xFF && (encoded_data[offset + 1] & 0xF0) == 0xF0 {
            let mut reader = BitReader::new(&encoded_data[offset..]);
            if let Ok(header) = AdtsHeader::parse(&mut reader) {
                let frame_len = header.frame_length;
                if offset + frame_len <= encoded_data.len() {
                    let frame_bytes = &encoded_data[offset..offset + frame_len];
                    match decoder.decode_frame(frame_bytes) {
                        Ok(pcm) => {
                            let mut interleaved = vec![0i16; pcm.total_samples()];
                            pcm.to_interleaved(&mut interleaved);
                            all_pcm_samples.extend_from_slice(&interleaved);
                            frame_count += 1;
                        }
                        Err(e) => {
                            eprintln!("Error decoding frame {}: {:?}", frame_count, e);
                        }
                    }
                    offset += frame_len;
                    continue;
                }
            }
        }
        offset += 1;
    }

    println!("Decoded {} frames ({} total interleaved samples)", frame_count, all_pcm_samples.len());

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

    println!("Successfully wrote WAV to {:?}", args.output);
    Ok(())
}
