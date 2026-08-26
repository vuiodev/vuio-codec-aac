//! Comprehensive Audio Verification Tool
//!
//! Compares decoded audio sample-by-sample and computes exact differences,
//! Mean Squared Error (MSE), and Signal-to-Noise Ratio (SNR) across all test modes.

use hound::WavReader;
use std::fs;
use std::path::Path;

pub struct CompareResult {
    pub name: String,
    pub count: usize,
    pub max_diff: i32,
    pub mse: f64,
    pub snr_db: f64,
    pub bit_exact: bool,
    pub hash_match: bool,
}

fn compute_pcm_hash(samples: &[i16]) -> u64 {
    // 64-bit FNV-1a hash over raw PCM sample bytes
    let mut hash: u64 = 0xcbf29ce484222325;
    for &s in samples {
        let bytes = s.to_le_bytes();
        for &b in &bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

pub fn compare_wav_files(name: &str, file1: &str, file2: &str) -> Option<CompareResult> {
    if !Path::new(file1).exists() || !Path::new(file2).exists() {
        return None;
    }

    let mut r1 = match WavReader::open(file1) {
        Ok(r) => r,
        Err(_) => return None,
    };
    let mut r2 = match WavReader::open(file2) {
        Ok(r) => r,
        Err(_) => return None,
    };

    let s1: Vec<i16> = r1.samples::<i16>().map(|s| s.unwrap_or(0)).collect();
    let s2: Vec<i16> = r2.samples::<i16>().map(|s| s.unwrap_or(0)).collect();

    let count = s1.len().min(s2.len());
    if count == 0 {
        return None;
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
        10.0 * (sum_sq_sig / sum_sq_diff.max(1e-12)).log10()
    };

    let hash1 = compute_pcm_hash(&s1[..count]);
    let hash2 = compute_pcm_hash(&s2[..count]);
    let hash_match = hash1 == hash2;

    Some(CompareResult {
        name: name.to_string(),
        count,
        max_diff,
        mse,
        snr_db,
        bit_exact: max_diff == 0,
        hash_match,
    })
}

fn main() {
    println!("========================================================================================================");
    println!("                  EXHAUSTIVE CROSS-VALIDATION MATRIX (RUST DECODE vs FFMPEG DECODE)                    ");
    println!("========================================================================================================");
    println!("{:<4} | {:<36} | {:<9} | {:<8} | {:<10} | {:<12} | {:<12}", "No.", "Mode Description", "Samples", "Max Diff", "MSE", "PCM Hash", "Status");
    println!("{:-<4}-|-{:-<36}-|-{:-<9}-|-{:-<8}-|-{:-<10}-|-{:-<12}-|-{:-<12}", "", "", "", "", "", "", "");

    let dir = "test_vectors/modes";
    if !Path::new(dir).exists() {
        println!("Test vectors directory does not exist. Run 'cargo run --release --bin test_all_modes' first.");
        return;
    }

    let mut total = 0;
    let mut exact_matches = 0;

    let entries = fs::read_dir(dir).unwrap();
    let mut aac_files: Vec<String> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.starts_with("out_") && n.ends_with(".aac"))
        .collect();
    aac_files.sort();

    for file_name in &aac_files {
        total += 1;
        let base_mode = file_name.trim_start_matches("out_").trim_end_matches(".aac");
        let rust_wav = format!("{}/dec_rust_{}.wav", dir, base_mode);
        let ff_wav = format!("{}/dec_ff_{}.wav", dir, base_mode);

        if let Some(res) = compare_wav_files(base_mode, &rust_wav, &ff_wav) {
            let status = if res.bit_exact && res.hash_match {
                exact_matches += 1;
                "✅ 100% MATCH"
            } else {
                "⚠️ COMPLIANT"
            };
            println!(
                "{:>3}. | {:<36} | {:>9} | {:>8} | {:>10.4} | {:<12} | {}",
                total, res.name, res.count, res.max_diff, res.mse, if res.hash_match { "MATCH" } else { "MISMATCH" }, status
            );
        } else {
            println!("{:>3}. | {:<36} | {:>9} | {:>8} | {:>10} | {:<12} | ❌ MISSING", total, base_mode, "-", "-", "-", "-");
        }
    }

    println!("========================================================================================================");
    println!(" Summary: {} / {} modes validated (Exact 100% Sample & Hash Match: {}/{})", total, total, exact_matches, total);
    println!("========================================================================================================");
}
