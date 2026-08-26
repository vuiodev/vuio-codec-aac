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
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    let args = Args::parse();

    let mut file = File::open(&args.input)?;
    let mut bitstream = Vec::new();
    file.read_to_end(&mut bitstream)?;

    let mut decoder = Decoder::new_default();
    let mut all_pcm_samples: Vec<i16> = Vec::new();
    let mut offset = 0;
    let mut frame_count = 0;

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
                        for s in 0..len {
                            for c in 0..ch {
                                all_pcm_samples.push(pcm_frame.channel(c)[s]);
                            }
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
