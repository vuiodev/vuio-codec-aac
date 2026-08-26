//! Multi-Mode Performance Benchmark Matrix: Reference C vs Pure Rust (vuiocodecaac)
//!
//! Evaluates and displays side-by-side real-time speed factors (`x real-time`)
//! across varied sample rates (8k..96k), channels (Mono, Stereo, 5.1ch), and profiles (LC, HE, HE-v2, LD, ELD).

use hound::{SampleFormat, WavSpec, WavWriter};
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

struct ModeConfig {
    name: &'static str,
    sample_rate: u32,
    channels: u16,
    profile: &'static str,
    bitrate: u32,
    c_aot: u32,
}

const BENCH_MODES: &[ModeConfig] = &[
    ModeConfig { name: "48kHz Stereo LC 128k", sample_rate: 48000, channels: 2, profile: "lc", bitrate: 128000, c_aot: 2 },
    ModeConfig { name: "48kHz Stereo LC 192k", sample_rate: 48000, channels: 2, profile: "lc", bitrate: 192000, c_aot: 2 },
    ModeConfig { name: "48kHz Stereo LC 320k", sample_rate: 48000, channels: 2, profile: "lc", bitrate: 320000, c_aot: 2 },
    ModeConfig { name: "48kHz Stereo LC 64k",  sample_rate: 48000, channels: 2, profile: "lc", bitrate: 64000,  c_aot: 2 },
    ModeConfig { name: "44.1kHz Stereo LC 128k", sample_rate: 44100, channels: 2, profile: "lc", bitrate: 128000, c_aot: 2 },
    ModeConfig { name: "44.1kHz Mono LC 64k", sample_rate: 44100, channels: 1, profile: "lc", bitrate: 64000,  c_aot: 2 },
    ModeConfig { name: "48kHz Mono LC 32k",   sample_rate: 48000, channels: 1, profile: "lc", bitrate: 32000,  c_aot: 2 },
    ModeConfig { name: "44.1kHz 5.1ch LC 320k", sample_rate: 44100, channels: 6, profile: "lc", bitrate: 320000, c_aot: 2 },
    ModeConfig { name: "44.1kHz Stereo HE-v1 128k", sample_rate: 44100, channels: 2, profile: "he", bitrate: 128000, c_aot: 5 },
    ModeConfig { name: "44.1kHz Stereo HE-v2 128k", sample_rate: 44100, channels: 2, profile: "he-v2", bitrate: 128000, c_aot: 29 },
    ModeConfig { name: "44.1kHz Stereo LD 128k", sample_rate: 44100, channels: 2, profile: "ld", bitrate: 128000, c_aot: 23 },
    ModeConfig { name: "44.1kHz Stereo ELD 128k", sample_rate: 44100, channels: 2, profile: "eld", bitrate: 128000, c_aot: 39 },
    ModeConfig { name: "8kHz Stereo LC 128k", sample_rate: 8000, channels: 2, profile: "lc", bitrate: 128000, c_aot: 2 },
    ModeConfig { name: "16kHz Stereo LC 128k", sample_rate: 16000, channels: 2, profile: "lc", bitrate: 128000, c_aot: 2 },
    ModeConfig { name: "24kHz Stereo LC 128k", sample_rate: 24000, channels: 2, profile: "lc", bitrate: 128000, c_aot: 2 },
    ModeConfig { name: "32kHz Stereo LC 128k", sample_rate: 32000, channels: 2, profile: "lc", bitrate: 128000, c_aot: 2 },
    ModeConfig { name: "64kHz Stereo LC 128k", sample_rate: 64000, channels: 2, profile: "lc", bitrate: 128000, c_aot: 2 },
    ModeConfig { name: "96kHz Stereo LC 128k", sample_rate: 96000, channels: 2, profile: "lc", bitrate: 128000, c_aot: 2 },
];

fn generate_mode_wav(path: &str, sample_rate: u32, channels: u16, duration_secs: f32) {
    let spec = WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec).expect("Failed to create WAV");
    let total_frames = (sample_rate as f32 * duration_secs) as usize;

    for i in 0..total_frames {
        let t = i as f32 / sample_rate as f32;
        for ch in 0..channels {
            let freq = 440.0 * (ch as f32 + 1.0);
            let s = ((2.0 * std::f32::consts::PI * freq * t).sin() * 12000.0) as i16;
            writer.write_sample(s).unwrap();
        }
    }
    writer.finalize().unwrap();
}

fn main() {
    println!("========================================================================================================================================");
    println!("                   MULTI-MODE SPEED MATRIX: REFERENCE C (libxaac/ffmpeg) vs PURE RUST (vuiocodecaac 2024)                              ");
    println!("========================================================================================================================================");
    println!("{:<3} | {:<28} | {:<16} | {:<16} | {:<16} | {:<16} | {:<16}", "No.", "Audio Mode / Configuration", "Ref C Enc (1T)", "Rust Enc (1T)", "Rust Enc (Multi)", "Ref C Dec", "Rust Dec ⚡");
    println!("{:-<3}-|-{:-<28}-|-{:-<16}-|-{:-<16}-|-{:-<16}-|-{:-<16}-|-{:-<16}", "", "", "", "", "", "", "");

    let dir = "test_vectors/bench_matrix";
    fs::create_dir_all(dir).unwrap();

    let duration_secs = 5.0; // 5 seconds per mode benchmark
    let c_enc_bin = "c/libxaac/build/xaacenc";
    let rust_enc_bin = "target/release/aacenc";
    let rust_dec_bin = "target/release/aacdec";

    for (idx, mode) in BENCH_MODES.iter().enumerate() {
        let in_wav = format!("{}/in_{}_{}ch_{}.wav", dir, mode.sample_rate, mode.channels, mode.profile);
        let c_aac = format!("{}/c_out_{}.aac", dir, idx);
        let rust_aac = format!("{}/rust_out_{}.aac", dir, idx);
        let rust_mt_aac = format!("{}/rust_mt_out_{}.aac", dir, idx);
        let c_dec_wav = format!("{}/c_dec_{}.wav", dir, idx);
        let rust_dec_wav = format!("{}/rust_dec_{}.wav", dir, idx);

        generate_mode_wav(&in_wav, mode.sample_rate, mode.channels, duration_secs);

        // 1. Reference C Encoder Speed
        let ref_enc_speed_str = if Path::new(c_enc_bin).exists() && mode.channels <= 2 {
            let start = Instant::now();
            let res = Command::new(c_enc_bin)
                .args([
                    &format!("-ifile:{}", in_wav),
                    &format!("-ofile:{}", c_aac),
                    &format!("-br:{}", mode.bitrate),
                    &format!("-aot:{}", mode.c_aot),
                    "-adts:1",
                ])
                .output();
            let elapsed = start.elapsed().as_secs_f64();
            if res.is_ok() && res.unwrap().status.success() {
                format!("{:>6.1}x real-time", duration_secs as f64 / elapsed.max(1e-6))
            } else {
                "-".to_string()
            }
        } else {
            // FFmpeg fallback for multi-channel / complex modes
            let start = Instant::now();
            let res = Command::new("ffmpeg")
                .args(["-y", "-i", &in_wav, "-c:a", "aac", "-b:a", &format!("{}", mode.bitrate), &c_aac])
                .output();
            let elapsed = start.elapsed().as_secs_f64();
            if res.is_ok() && res.unwrap().status.success() {
                format!("{:>6.1}x real-time", duration_secs as f64 / elapsed.max(1e-6))
            } else {
                "-".to_string()
            }
        };

        // 2. Pure Rust Single-Thread Encoder Speed
        let start_rust_enc = Instant::now();
        let _ = Command::new(rust_enc_bin)
            .args([
                &in_wav,
                &rust_aac,
                "--bitrate",
                &format!("{}", mode.bitrate),
                "--profile",
                mode.profile,
                "--threads",
                "1",
            ])
            .output();
        let elapsed_rust_enc = start_rust_enc.elapsed().as_secs_f64();
        let rust_enc_speed_str = format!("{:>6.1}x real-time", duration_secs as f64 / elapsed_rust_enc.max(1e-6));

        // 3. Pure Rust Multi-Core Rayon Encoder Speed
        let start_rust_mt = Instant::now();
        let _ = Command::new(rust_enc_bin)
            .args([
                &in_wav,
                &rust_mt_aac,
                "--bitrate",
                &format!("{}", mode.bitrate),
                "--profile",
                mode.profile,
                "--threads",
                "0",
            ])
            .output();
        let elapsed_rust_mt = start_rust_mt.elapsed().as_secs_f64();
        let rust_mt_speed_str = format!("{:>6.1}x real-time", duration_secs as f64 / elapsed_rust_mt.max(1e-6));

        // 4. Reference C Decoder (FFmpeg) Speed
        let start_ref_dec = Instant::now();
        let _ = Command::new("ffmpeg")
            .args(["-y", "-i", &rust_aac, &c_dec_wav])
            .output();
        let elapsed_ref_dec = start_ref_dec.elapsed().as_secs_f64();
        let ref_dec_speed_str = format!("{:>6.1}x real-time", duration_secs as f64 / elapsed_ref_dec.max(1e-6));

        // 5. Pure Rust Decoder Speed
        let start_rust_dec = Instant::now();
        let _ = Command::new(rust_dec_bin)
            .args([&rust_aac, &rust_dec_wav])
            .output();
        let elapsed_rust_dec = start_rust_dec.elapsed().as_secs_f64();
        let rust_dec_speed_str = format!("{:>6.1}x real-time", duration_secs as f64 / elapsed_rust_dec.max(1e-6));

        println!(
            "{:>2}. | {:<28} | {:<16} | {:<16} | {:<16} | {:<16} | {:<16}",
            idx + 1,
            mode.name,
            ref_enc_speed_str,
            rust_enc_speed_str,
            rust_mt_speed_str,
            ref_dec_speed_str,
            rust_dec_speed_str
        );
    }

    println!("========================================================================================================================================");
    println!(" All speeds displayed in standard audio speed factor (Duration of Audio / Processing Time = Xx real-time).");
    println!("========================================================================================================================================");
}
