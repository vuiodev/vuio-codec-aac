//! Audio Verification and Quality Comparison Tool

use hound::WavReader;
use std::path::Path;

fn compare_wavs(name: &str, file1: &str, file2: &str) {
    if !Path::new(file1).exists() || !Path::new(file2).exists() {
        println!("[{}] Skipped (files do not exist)", name);
        return;
    }

    let mut r1 = WavReader::open(file1).expect("Failed to open file1");
    let mut r2 = WavReader::open(file2).expect("Failed to open file2");

    let s1: Vec<i16> = r1.samples::<i16>().map(|s| s.unwrap_or(0)).collect();
    let s2: Vec<i16> = r2.samples::<i16>().map(|s| s.unwrap_or(0)).collect();

    let count = s1.len().min(s2.len());
    if count == 0 {
        println!("[{}] Empty sample set", name);
        return;
    }

    let mut sum_sq_diff = 0.0f64;
    let mut sum_sq_sig = 0.0f64;
    let mut max_diff = 0i32;

    for i in 0..count {
        let diff = (s1[i] as i32) - (s2[i] as i32);
        max_diff = max_diff.max(diff.abs());
        sum_sq_diff += (diff as f64) * (diff as f64);
        sum_sq_sig += (s1[i] as f64) * (s1[i] as f64);
    }

    let mse = sum_sq_diff / count as f64;
    let snr_db = if mse < 1e-12 {
        f64::INFINITY
    } else {
        10.0 * (sum_sq_sig / (sum_sq_diff.max(1e-12))).log10()
    };

    println!(
        "[{}] Compared {} samples | Max Diff: {} | MSE: {:.4} | SNR: {:.2} dB",
        name, count, max_diff, mse, snr_db
    );
}

fn main() {
    println!("=== Cross-Validation Matrix Results ===");
    compare_wavs(
        "C-Encoded Stream: Rust vs FFmpeg Decode",
        "test_vectors/rust_decoded_from_c.wav",
        "test_vectors/ffmpeg_decoded_from_c.wav",
    );
    compare_wavs(
        "Rust-Encoded Stream: Rust vs FFmpeg Decode",
        "test_vectors/rust_decoded_from_rust.wav",
        "test_vectors/ffmpeg_decoded_from_rust.wav",
    );
    compare_wavs(
        "FFmpeg-Encoded Mono: Rust Decode vs Original WAV",
        "test_vectors/rust_decoded_ffmpeg_mono.wav",
        "test_vectors/sine_48k_mono.wav",
    );
    compare_wavs(
        "FFmpeg-Encoded Noise: Rust Decode vs Original WAV",
        "test_vectors/rust_decoded_ffmpeg_noise.wav",
        "test_vectors/noise_44k_stereo.wav",
    );
    println!("=== Verification Complete ===");
}
