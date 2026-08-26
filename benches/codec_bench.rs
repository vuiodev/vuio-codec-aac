//! Component benchmarks for the decode hot path.
//!
//! The whole-frame benchmark sets the budget; the component benchmarks show where
//! that budget goes, so optimization effort can be aimed rather than guessed.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use vuiocodecaac::decoder::aac::dequant::inverse_quantize_band;
use vuiocodecaac::decoder::aac::tns::{ar_filter, parcor_to_lpc};
use vuiocodecaac::dsp::fft::{Complex32, FftContext};
use vuiocodecaac::dsp::filterbank::Filterbank;
use vuiocodecaac::dsp::imdct::ImdctContext;
use vuiocodecaac::types::{WindowSequence, WindowShape};

fn spectrum(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f32;
            (t * 0.017).sin() * 3000.0 + (t * 0.11).cos() * 900.0
        })
        .collect()
}

fn bench_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft");
    for &n in &[64usize, 128, 512, 1024] {
        let ctx = FftContext::new(n);
        let data: Vec<Complex32> = (0..n)
            .map(|i| Complex32::new((i as f32 * 0.1).sin(), (i as f32 * 0.2).cos()))
            .collect();
        group.bench_function(format!("forward_{n}"), |b| {
            let mut buf = data.clone();
            b.iter(|| {
                buf.copy_from_slice(&data);
                ctx.forward(black_box(&mut buf));
                black_box(buf[0]);
            })
        });
    }
    group.finish();
}

fn bench_imdct(c: &mut Criterion) {
    let mut group = c.benchmark_group("imdct");
    for &n in &[128usize, 1024] {
        let ctx = ImdctContext::new(n);
        let spec = spectrum(n);
        let mut out = vec![0.0f32; 2 * n];
        let mut scratch = vec![Complex32::default(); n];
        group.bench_function(format!("imdct_{n}"), |b| {
            b.iter(|| {
                ctx.imdct(black_box(&spec), &mut out, &mut scratch);
                black_box(out[0]);
            })
        });
    }
    group.finish();
}

fn bench_filterbank(c: &mut Criterion) {
    let mut group = c.benchmark_group("filterbank");
    let n = 1024;
    let spec = spectrum(n);

    for (name, seq) in [
        ("long", WindowSequence::OnlyLongSequence),
        ("eight_short", WindowSequence::EightShortSequence),
    ] {
        let mut fb = Filterbank::new(n);
        let mut overlap = vec![0.0f32; n];
        let mut out = vec![0.0f32; n];
        group.bench_function(name, |b| {
            b.iter(|| {
                fb.synthesize(
                    black_box(&spec),
                    seq,
                    WindowShape::Kbd,
                    WindowShape::Kbd,
                    &mut overlap,
                    &mut out,
                );
                black_box(out[0]);
            })
        });
    }
    group.finish();
}

fn bench_dequant(c: &mut Criterion) {
    let mut group = c.benchmark_group("dequant");
    // A realistic band mix: many small values, a few large ones.
    let quant: Vec<i32> = (0..1024)
        .map(|i| {
            let m = (i * 7919) % 97;
            if m < 60 { 0 } else if m < 90 { (m as i32) - 75 } else { (m as i32 - 89) * 400 }
        })
        .collect();
    let mut out = vec![0.0f32; 1024];
    group.bench_function("band_1024", |b| {
        b.iter(|| {
            inverse_quantize_band(black_box(&quant), 120, &mut out);
            black_box(out[0]);
        })
    });
    group.finish();
}

fn bench_tns(c: &mut Criterion) {
    let mut group = c.benchmark_group("tns");
    let parcor = [0.62f32, -0.41, 0.28, -0.19, 0.12, -0.08, 0.05, -0.03];
    let mut lpc = [0.0f32; 21];
    parcor_to_lpc(&parcor, &mut lpc);
    let base = spectrum(768);
    let mut spec = base.clone();
    group.bench_function("ar_filter_order8_768", |b| {
        b.iter(|| {
            spec.copy_from_slice(&base);
            ar_filter(black_box(&mut spec), &lpc, 8, false);
            black_box(spec[0]);
        })
    });
    group.finish();
}

criterion_group!(benches, bench_fft, bench_imdct, bench_filterbank, bench_dequant, bench_tns);
criterion_main!(benches);
