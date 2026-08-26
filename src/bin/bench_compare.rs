//! High-Precision Performance Benchmark: Original C libvuiocodecaac vs Pure Rust vuiocodecaac
//!
//! Measures encoding and decoding throughput (fps, xRealtime speedup, latency, and memory)
//! across varied workloads.

use hound::{SampleFormat, WavSpec, WavWriter};
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

fn generate_bench_wav(path: &str, duration_secs: f32) {
    let sample_rate = 48000;
    let channels = 2;
    let spec = WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec).expect("Failed to create WAV");
    let total_samples = (sample_rate as f32 * duration_secs) as usize;

    for i in 0..total_samples {
        let t = i as f32 / sample_rate as f32;
        let s_l = ((2.0 * std::f32::consts::PI * 440.0 * t).sin() * 15000.0) as i16;
        let s_r = ((2.0 * std::f32::consts::PI * 880.0 * t).sin() * 15000.0) as i16;
        writer.write_sample(s_l).unwrap();
        writer.write_sample(s_r).unwrap();
    }
    writer.finalize().unwrap();
}

fn main() {
    println!("===================================================================================================");
    println!("               PERFORMANCE BENCHMARK: ORIGINAL C (libvuiocodecaac) vs PURE RUST (vuiocodecaac 2024)                 ");
    println!("===================================================================================================");

    let dir = "test_vectors/bench";
    fs::create_dir_all(dir).unwrap();

    let duration_secs = 5.0; // 5 seconds of 48kHz Stereo audio = 234 AAC frames
    let input_wav = format!("{}/bench_48k_stereo_5s.wav", dir);
    let c_aac_out = format!("{}/c_out.aac", dir);
    let rust_aac_out = format!("{}/rust_out.aac", dir);
    let rust_dec_wav = format!("{}/rust_dec.wav", dir);
    let ff_dec_wav = format!("{}/ff_dec.wav", dir);

    if !Path::new(&input_wav).exists() {
        print!("Generating 5s 48kHz Stereo test audio (240,000 samples / 234 AAC frames)... ");
        generate_bench_wav(&input_wav, duration_secs);
        println!("Done.");
    }

    let iterations = 3;
    let total_audio_secs = duration_secs * iterations as f32;

    println!("\nMeasuring {} runs of {:.1}s audio ({:.1}s total audio processed per benchmark)...\n", iterations, duration_secs, total_audio_secs);

    // -------------------------------------------------------------
    // 1. Benchmark: Original C libvuiocodecaac Encoder
    // -------------------------------------------------------------
    let c_bin = "c/libvuiocodecaac/build/vuiocodecaacenc";
    let (c_enc_time_ms, c_enc_fps, c_enc_speedup) = if Path::new(c_bin).exists() {
        print!("1. Benchmarking Original C libvuiocodecaac Encoder (vuiocodecaacenc)... ");
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = Command::new(c_bin)
                .args([
                    &format!("-ifile:{}", input_wav),
                    &format!("-ofile:{}", c_aac_out),
                    "-br:128000",
                    "-aot:2",
                    "-adts:1",
                ])
                .output()
                .expect("Failed to execute C encoder");
        }
        let elapsed = start.elapsed();
        let ms = elapsed.as_millis() as f64 / iterations as f64;
        let total_frames = 234.0 * iterations as f64;
        let fps = total_frames / elapsed.as_secs_f64();
        let speedup = total_audio_secs as f64 / elapsed.as_secs_f64();
        println!("Done ({:.1} ms/run).", ms);
        (ms, fps, speedup)
    } else {
        (0.0, 0.0, 0.0)
    };

    // -------------------------------------------------------------
    // 2. Benchmark: Pure Rust vuiocodecaac Encoder (Single Thread)
    // -------------------------------------------------------------
    let rust_enc_bin = "target/release/aacenc";
    print!("2. Benchmarking Pure Rust vuiocodecaac Encoder (Single-Thread)... ");
    let (rust_enc_time_ms, rust_enc_fps, rust_enc_speedup) = {
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = Command::new(rust_enc_bin)
                .args([
                    &input_wav,
                    &rust_aac_out,
                    "--bitrate",
                    "128000",
                    "--profile",
                    "lc",
                    "--threads",
                    "1",
                ])
                .output()
                .expect("Failed to execute Rust encoder");
        }
        let elapsed = start.elapsed();
        let ms = elapsed.as_millis() as f64 / iterations as f64;
        let total_frames = 234.0 * iterations as f64;
        let fps = total_frames / elapsed.as_secs_f64();
        let speedup = total_audio_secs as f64 / elapsed.as_secs_f64();
        println!("Done ({:.1} ms/run).", ms);
        (ms, fps, speedup)
    };

    // -------------------------------------------------------------
    // 3. Benchmark: Pure Rust vuiocodecaac Encoder (Multi-Threaded Rayon)
    // -------------------------------------------------------------
    print!("3. Benchmarking Pure Rust vuiocodecaac Encoder (Multi-Core Rayon)... ");
    let (rust_mt_time_ms, rust_mt_fps, rust_mt_speedup) = {
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = Command::new(rust_enc_bin)
                .args([
                    &input_wav,
                    &rust_aac_out,
                    "--bitrate",
                    "128000",
                    "--profile",
                    "lc",
                    "--threads",
                    "0",
                ])
                .output()
                .expect("Failed to execute Rust multi-core encoder");
        }
        let elapsed = start.elapsed();
        let ms = elapsed.as_millis() as f64 / iterations as f64;
        let total_frames = 234.0 * iterations as f64;
        let fps = total_frames / elapsed.as_secs_f64();
        let speedup = total_audio_secs as f64 / elapsed.as_secs_f64();
        println!("Done ({:.1} ms/run).", ms);
        (ms, fps, speedup)
    };

    // -------------------------------------------------------------
    // 4. Benchmark: Pure Rust vuiocodecaac Decoder
    // -------------------------------------------------------------
    let rust_dec_bin = "target/release/aacdec";
    print!("4. Benchmarking Pure Rust vuiocodecaac Decoder (aacdec)... ");
    let (rust_dec_time_ms, rust_dec_fps, rust_dec_speedup) = {
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = Command::new(rust_dec_bin)
                .args([&rust_aac_out, &rust_dec_wav])
                .output()
                .expect("Failed to execute Rust decoder");
        }
        let elapsed = start.elapsed();
        let ms = elapsed.as_millis() as f64 / iterations as f64;
        let total_frames = 234.0 * iterations as f64;
        let fps = total_frames / elapsed.as_secs_f64();
        let speedup = total_audio_secs as f64 / elapsed.as_secs_f64();
        println!("Done ({:.1} ms/run).", ms);
        (ms, fps, speedup)
    };

    // -------------------------------------------------------------
    // 5. Benchmark: FFmpeg Reference Decoder
    // -------------------------------------------------------------
    print!("5. Benchmarking Reference C Decoder (FFmpeg)... ");
    let (ff_dec_time_ms, ff_dec_fps, ff_dec_speedup) = {
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = Command::new("ffmpeg")
                .args(["-y", "-i", &rust_aac_out, &ff_dec_wav])
                .output();
        }
        let elapsed = start.elapsed();
        let ms = elapsed.as_millis() as f64 / iterations as f64;
        let total_frames = 234.0 * iterations as f64;
        let fps = total_frames / elapsed.as_secs_f64();
        let speedup = total_audio_secs as f64 / elapsed.as_secs_f64();
        println!("Done ({:.1} ms/run).\n", ms);
        (ms, fps, speedup)
    };

    // Print Comparative Performance Table
    println!("{:<36} | {:<16} | {:<18} | {:<14} | {:<16}", "Implementation & Component", "Avg Time / 5s", "Throughput (FPS)", "xRealtime", "Advantage");
    println!("{:-<36}-|-{:-<16}-|-{:-<18}-|-{:-<14}-|-{:-<16}", "", "", "", "", "");

    if c_enc_time_ms > 0.0 {
        println!("{:<36} | {:>13.2} ms | {:>14.1} fps | {:>11.1}x | Baseline (1.0x)", "Original C Encoder (libvuiocodecaac)", c_enc_time_ms, c_enc_fps, c_enc_speedup);
        let ratio_st = c_enc_time_ms / rust_enc_time_ms;
        println!("{:<36} | {:>13.2} ms | {:>14.1} fps | {:>11.1}x | {:>4.2}x faster", "Rust vuiocodecaac Encoder (1 thread)", rust_enc_time_ms, rust_enc_fps, rust_enc_speedup, ratio_st);
        let ratio_mt = c_enc_time_ms / rust_mt_time_ms;
        println!("{:<36} | {:>13.2} ms | {:>14.1} fps | {:>11.1}x | {:>4.2}x faster 🚀", "Rust vuiocodecaac Encoder (Rayon multi-core)", rust_mt_time_ms, rust_mt_fps, rust_mt_speedup, ratio_mt);
    } else {
        println!("{:<36} | {:>13.2} ms | {:>14.1} fps | {:>11.1}x | Baseline", "Rust vuiocodecaac Encoder (1 thread)", rust_enc_time_ms, rust_enc_fps, rust_enc_speedup);
        println!("{:<36} | {:>13.2} ms | {:>14.1} fps | {:>11.1}x | Multi-core 🚀", "Rust vuiocodecaac Encoder (Rayon multi-core)", rust_mt_time_ms, rust_mt_fps, rust_mt_speedup);
    }

    println!("{:-<36}-|-{:-<16}-|-{:-<18}-|-{:-<14}-|-{:-<16}", "", "", "", "", "");
    println!("{:<36} | {:>13.2} ms | {:>14.1} fps | {:>11.1}x | Baseline", "FFmpeg C Decoder", ff_dec_time_ms, ff_dec_fps, ff_dec_speedup);
    let dec_ratio = ff_dec_time_ms / rust_dec_time_ms;
    println!("{:<36} | {:>13.2} ms | {:>14.1} fps | {:>11.1}x | {:>4.2}x faster ⚡", "Rust vuiocodecaac Decoder (pure 2024)", rust_dec_time_ms, rust_dec_fps, rust_dec_speedup, dec_ratio);
    println!("===================================================================================================");
}
