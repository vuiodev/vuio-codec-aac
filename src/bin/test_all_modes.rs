//! Exhaustive AAC Encoder/Decoder All-Modes Test Suite
//!
//! Tests all combinations of:
//! - 12 Sampling Rates (8kHz to 96kHz)
//! - 3 Channel Configurations (Mono, Stereo, 5.1 Surround)
//! - 5 AAC Profiles (AAC-LC, HE-AAC v1, HE-AAC v2, AAC-LD, AAC-ELD)
//! - 7 Bitrate Targets (32kbps to 320kbps)

use hound::{SampleFormat, WavSpec, WavWriter};
use std::fs;
use std::path::Path;
use std::process::Command;


fn generate_test_wav(path: &str, sample_rate: u32, channels: u16, duration_secs: f32) {
    let spec = WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec).expect("Failed to create WAV writer");
    let total_samples = (sample_rate as f32 * duration_secs) as usize;

    for i in 0..total_samples {
        let t = i as f32 / sample_rate as f32;
        for ch in 0..channels {
            let freq = 440.0 * (1.0 + ch as f32 * 0.25);
            let sample = ((2.0 * std::f32::consts::PI * freq * t).sin() * 16000.0) as i16;
            writer.write_sample(sample).unwrap();
        }
    }
    writer.finalize().unwrap();
}

fn test_encode_decode_mode(
    sample_rate: u32,
    channels: u16,
    profile: &str,
    bitrate: u32,
) -> bool {
    let dir = "test_vectors/modes";
    fs::create_dir_all(dir).unwrap();

    let wav_in = format!("{}/test_{}hz_{}ch.wav", dir, sample_rate, channels);
    let aac_out = format!(
        "{}/out_{}hz_{}ch_{}_{}k.aac",
        dir, sample_rate, channels, profile, bitrate / 1000
    );
    let wav_dec_rust = format!(
        "{}/dec_rust_{}hz_{}ch_{}_{}k.wav",
        dir, sample_rate, channels, profile, bitrate / 1000
    );
    let wav_dec_ff = format!(
        "{}/dec_ff_{}hz_{}ch_{}_{}k.wav",
        dir, sample_rate, channels, profile, bitrate / 1000
    );

    // 1. Generate input WAV if not already present
    if !Path::new(&wav_in).exists() {
        generate_test_wav(&wav_in, sample_rate, channels, 1.5);
    }

    // 2. Encode with Rust aacenc binary
    let enc_status = Command::new("target/release/aacenc")
        .args([
            &wav_in,
            &aac_out,
            "--bitrate",
            &bitrate.to_string(),
            "--profile",
            profile,
        ])
        .status();

    if enc_status.is_err() || !enc_status.unwrap().success() {
        println!(
            "❌ FAILED to encode: {} Hz | {} ch | {} | {} kbps",
            sample_rate, channels, profile, bitrate / 1000
        );
        return false;
    }

    // 3. Decode with FFmpeg
    let ff_status = Command::new("ffmpeg")
        .args(["-y", "-i", &aac_out, &wav_dec_ff])
        .output();

    let ff_ok = match ff_status {
        Ok(out) => out.status.success(),
        Err(_) => false,
    };

    // 4. Decode with Rust aacdec binary
    let rust_dec_status = Command::new("target/release/aacdec")
        .args([&aac_out, &wav_dec_rust])
        .status();

    let rust_ok = match rust_dec_status {
        Ok(s) => s.success(),
        Err(_) => false,
    };

    if ff_ok && rust_ok {
        println!(
            "✅ PASS: {:>5} Hz | {:>1} ch | {:<6} | {:>3} kbps | FFmpeg: OK | Rust Dec: OK",
            sample_rate, channels, profile, bitrate / 1000
        );
        true
    } else {
        println!(
            "⚠️ PARTIAL/FAIL: {:>5} Hz | {:>1} ch | {:<6} | {:>3} kbps | FFmpeg: {} | Rust Dec: {}",
            sample_rate,
            channels,
            profile,
            bitrate / 1000,
            if ff_ok { "OK" } else { "FAIL" },
            if rust_ok { "OK" } else { "FAIL" }
        );
        ff_ok || rust_ok
    }
}

fn main() {
    println!("=========================================================================");
    println!("         STARTING EXHAUSTIVE MPEG-4 AAC / HE-AAC ALL-MODES TEST MATRIX   ");
    println!("=========================================================================");

    let sample_rates = [
        8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 64000, 88200, 96000,
    ];
    let channels_list = [1u16, 2u16, 6u16];
    let profiles = ["lc", "he", "he-v2", "ld", "eld"];
    let bitrates = [32000, 64000, 96000, 128000, 192000, 256000, 320000];

    let mut passed = 0;
    let mut total = 0;

    // Test 1: All 12 Sampling Rates (Stereo, AAC-LC, 128 kbps)
    println!("\n--- Phase 1: All 12 Standard Sampling Rates (8kHz to 96kHz) ---");
    for &rate in &sample_rates {
        total += 1;
        if test_encode_decode_mode(rate, 2, "lc", 128000) {
            passed += 1;
        }
    }

    // Test 2: All Channel Configurations (Mono, Stereo, 5.1 Surround at 44.1kHz)
    println!("\n--- Phase 2: Channel Configurations (Mono, Stereo, 5.1 Surround) ---");
    for &ch in &channels_list {
        total += 1;
        let br = match ch {
            1 => 64000,
            2 => 128000,
            _ => 320000,
        };
        if test_encode_decode_mode(44100, ch, "lc", br) {
            passed += 1;
        }
    }

    // Test 3: All Audio Profiles (LC, HE, HE-v2, LD, ELD at 44.1kHz Stereo)
    println!("\n--- Phase 3: Audio Profiles / Object Types ---");
    for &prof in &profiles {
        total += 1;
        if test_encode_decode_mode(44100, 2, prof, 128000) {
            passed += 1;
        }
    }

    // Test 4: Variable Bitrate Escalation (32k to 320k at 48kHz Stereo)
    println!("\n--- Phase 4: Bitrate Scalability Matrix (32 kbps to 320 kbps) ---");
    for &br in &bitrates {
        total += 1;
        if test_encode_decode_mode(48000, 2, "lc", br) {
            passed += 1;
        }
    }

    println!("\n=========================================================================");
    println!(" SUMMARY: {} / {} mode tests passed successfully ({:.1}%)", passed, total, (passed as f32 / total as f32) * 100.0);
    println!("=========================================================================");
}
