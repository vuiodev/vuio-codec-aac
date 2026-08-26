//! Compare two WAV files sample by sample, searching over a lead/lag offset.
//!
//! Decoders differ in how much leading silence they emit, so a raw sample-index
//! comparison would report a large error for output that is actually identical but
//! shifted. This finds the alignment that minimizes error and reports the residual
//! at that alignment, which is what tells you whether two decoders agree.

use hound::WavReader;
use std::path::PathBuf;

fn usage() -> ! {
    eprintln!("usage: compare_wav <reference.wav> <test.wav> [max_offset]");
    std::process::exit(2);
}

/// Read a WAV file as per-channel sample vectors.
fn read_channels(path: &PathBuf) -> Result<(Vec<Vec<f64>>, u32), Box<dyn std::error::Error>> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let mut out = vec![Vec::new(); channels];

    match spec.sample_format {
        hound::SampleFormat::Int => {
            for (i, s) in reader.samples::<i32>().enumerate() {
                out[i % channels].push(s? as f64);
            }
        }
        hound::SampleFormat::Float => {
            for (i, s) in reader.samples::<f32>().enumerate() {
                // Bring floats into the same 16-bit scale as integer files.
                out[i % channels].push(s? as f64 * 32768.0);
            }
        }
    }
    Ok((out, spec.sample_rate))
}

/// Error statistics between two aligned signals.
struct Stats {
    compared: usize,
    max_abs: f64,
    rms: f64,
    /// Signal-to-noise ratio in dB; infinite when the signals are identical.
    snr_db: f64,
    exact: usize,
}

/// Compare `test` against `reference` with `test` shifted by `offset` samples.
fn compare_at(reference: &[f64], test: &[f64], offset: isize) -> Option<Stats> {
    // Overlapping window of the two signals at this shift.
    let (r_start, t_start) = if offset >= 0 {
        (offset as usize, 0usize)
    } else {
        (0usize, (-offset) as usize)
    };
    if r_start >= reference.len() || t_start >= test.len() {
        return None;
    }
    let n = (reference.len() - r_start).min(test.len() - t_start);
    if n < 1024 {
        return None;
    }

    let mut max_abs = 0.0f64;
    let mut sq_err = 0.0f64;
    let mut sq_sig = 0.0f64;
    let mut exact = 0usize;

    for i in 0..n {
        let r = reference[r_start + i];
        let t = test[t_start + i];
        let d = (r - t).abs();
        if d > max_abs {
            max_abs = d;
        }
        if d == 0.0 {
            exact += 1;
        }
        sq_err += d * d;
        sq_sig += r * r;
    }

    let rms = (sq_err / n as f64).sqrt();
    let snr_db = if sq_err == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (sq_sig / sq_err).log10()
    };
    Some(Stats { compared: n, max_abs, rms, snr_db, exact })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        usage();
    }
    let ref_path = PathBuf::from(&args[1]);
    let test_path = PathBuf::from(&args[2]);
    let max_offset: isize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4096);

    let (ref_ch, ref_rate) = read_channels(&ref_path)?;
    let (test_ch, test_rate) = read_channels(&test_path)?;

    println!(
        "reference: {} ch, {} Hz, {} samples/ch",
        ref_ch.len(),
        ref_rate,
        ref_ch.first().map_or(0, |c| c.len())
    );
    println!(
        "test:      {} ch, {} Hz, {} samples/ch",
        test_ch.len(),
        test_rate,
        test_ch.first().map_or(0, |c| c.len())
    );

    if ref_ch.len() != test_ch.len() {
        println!("CHANNEL COUNT MISMATCH");
        std::process::exit(1);
    }
    if ref_rate != test_rate {
        println!("SAMPLE RATE MISMATCH");
        std::process::exit(1);
    }

    // Pick the shift that minimizes error on channel 0, then report every channel
    // at that same shift.
    let mut best_offset = 0isize;
    let mut best_rms = f64::INFINITY;
    for offset in -max_offset..=max_offset {
        if let Some(s) = compare_at(&ref_ch[0], &test_ch[0], offset)
            && s.rms < best_rms
        {
            best_rms = s.rms;
            best_offset = offset;
        }
    }

    println!("best alignment: reference leads test by {best_offset} samples\n");

    let mut worst_snr = f64::INFINITY;
    let mut all_exact = true;
    for (c, (r, t)) in ref_ch.iter().zip(test_ch.iter()).enumerate() {
        let Some(s) = compare_at(r, t, best_offset) else {
            println!("channel {c}: no overlap at this alignment");
            all_exact = false;
            continue;
        };
        let pct_exact = 100.0 * s.exact as f64 / s.compared as f64;
        println!(
            "channel {c}: n={} max_abs={:.3} rms={:.4} snr={:.2} dB exact={:.2}%",
            s.compared, s.max_abs, s.rms, s.snr_db, pct_exact
        );
        if s.snr_db < worst_snr {
            worst_snr = s.snr_db;
        }
        if s.exact != s.compared {
            all_exact = false;
        }
    }

    // Locate the worst errors so a caller can map them back to frames.
    if std::env::var_os("COMPARE_HOTSPOTS").is_some() {
        let (r_start, t_start) = if best_offset >= 0 {
            (best_offset as usize, 0usize)
        } else {
            (0usize, (-best_offset) as usize)
        };
        let n = (ref_ch[0].len() - r_start).min(test_ch[0].len() - t_start);
        // Aggregate error energy per 1024-sample frame.
        let mut frames: Vec<(usize, f64)> = Vec::new();
        for f in 0..n / 1024 {
            let mut e = 0.0f64;
            for i in f * 1024..(f + 1) * 1024 {
                let d = ref_ch[0][r_start + i] - test_ch[0][t_start + i];
                e += d * d;
            }
            frames.push((f, (e / 1024.0).sqrt()));
        }
        let mut sorted = frames.clone();
        sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
        println!("\nworst frames (index, rms error):");
        for (f, e) in sorted.iter().take(12) {
            println!("  frame {f:5}  rms {e:10.2}");
        }
        // Machine-readable list of every frame above one LSB.
        let bad: Vec<String> =
            frames.iter().filter(|(_, e)| *e >= 1.0).map(|(f, _)| f.to_string()).collect();
        println!("BADFRAMES {}", bad.join(","));

        // Error profile inside one frame, in 128-sample sub-blocks, to show
        // whether a bad frame is uniformly wrong or wrong only in part.
        if let Ok(which) = std::env::var("COMPARE_FRAME")
            && let Ok(fi) = which.parse::<usize>()
            && (fi + 1) * 1024 <= n
        {
            println!("\nframe {fi} error profile (128-sample sub-blocks):");
            for b in 0..8 {
                let mut e = 0.0f64;
                let mut sig = 0.0f64;
                for i in fi * 1024 + b * 128..fi * 1024 + (b + 1) * 128 {
                    let d = ref_ch[0][r_start + i] - test_ch[0][t_start + i];
                    e += d * d;
                    sig += ref_ch[0][r_start + i] * ref_ch[0][r_start + i];
                }
                println!(
                    "  sub {b}: rms_err {:9.2}  rms_sig {:9.2}",
                    (e / 128.0).sqrt(),
                    (sig / 128.0).sqrt()
                );
            }
        }
        let total = frames.len();
        for &(label, lo, hi) in &[
            ("< 1 LSB      ", 0.0f64, 1.0f64),
            ("1 - 10       ", 1.0, 10.0),
            ("10 - 100     ", 10.0, 100.0),
            ("100 - 1000   ", 100.0, 1000.0),
            (">= 1000      ", 1000.0, f64::INFINITY),
        ] {
            let c = frames.iter().filter(|(_, e)| *e >= lo && *e < hi).count();
            if c > 0 {
                println!("  rms {label}: {c:5} / {total} frames");
            }
        }
    }

    println!();
    if all_exact {
        println!("VERDICT: bit-exact");
    } else if worst_snr >= 90.0 {
        println!("VERDICT: numerically equivalent (worst SNR {worst_snr:.2} dB)");
    } else if worst_snr >= 40.0 {
        println!("VERDICT: close but not equivalent (worst SNR {worst_snr:.2} dB)");
    } else {
        println!("VERDICT: MISMATCH (worst SNR {worst_snr:.2} dB)");
        std::process::exit(1);
    }
    Ok(())
}
