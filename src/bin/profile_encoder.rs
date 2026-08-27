use std::time::Instant;
use std::hint::black_box;
use vuiocodecaac::dsp::mdct::MdctContext;
use vuiocodecaac::dsp::fft::Complex32;

fn main() {
    let n = 1024;
    
    // Generate test data
    let time_2n: Vec<f32> = (0..2*n).map(|i| (i as f32 * 0.01).sin()).collect();
    let mut spec_n = vec![0.0f32; n];
    let mdct = MdctContext::new(n);
    let mut scratch_fft = vec![Complex32::new(0.0, 0.0); mdct.scratch_len()];
    
    let iterations = 500;
    
    // Warmup
    for _ in 0..10 {
        mdct.forward(black_box(&time_2n), black_box(&mut spec_n), black_box(&mut scratch_fft));
    }
    
    // Benchmark FFT-based forward MDCT
    let start = Instant::now();
    for _ in 0..iterations {
        mdct.forward(black_box(&time_2n), black_box(&mut spec_n), black_box(&mut scratch_fft));
        black_box(&spec_n);
    }
    let elapsed_mdct = start.elapsed();
    let mdct_us = elapsed_mdct.as_secs_f64() * 1e6 / iterations as f64;
    println!("forward_mdct_fft (O(N log N)): {:>10.1} µs/call  ({} iters)", mdct_us, iterations);
    
    // Benchmark quantize_band
    let spectrum: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin() * 100.0).collect();
    let mut quantized = vec![0i32; n];
    for _ in 0..10 {
        vuiocodecaac::encoder::aac::quant::quantize_band(black_box(&spectrum), 100, black_box(&mut quantized));
    }
    let start = Instant::now();
    for _ in 0..iterations {
        vuiocodecaac::encoder::aac::quant::quantize_band(black_box(&spectrum), black_box(100), black_box(&mut quantized));
        black_box(&quantized);
    }
    let elapsed_quant = start.elapsed();
    let quant_us = elapsed_quant.as_secs_f64() * 1e6 / iterations as f64;
    println!("quantize_band:                 {:>10.1} µs/call", quant_us);
    
    // Benchmark codebook selection, which the rate loop runs once per band per
    // bisection step.
    let start = Instant::now();
    for _ in 0..iterations {
        let c = vuiocodecaac::encoder::aac::quant::choose_codebook(black_box(&quantized[..64]));
        black_box(c.bits);
    }
    let elapsed_gain = start.elapsed();
    let gain_us = elapsed_gain.as_secs_f64() * 1e6 / iterations as f64;
    println!("choose_codebook (64 lines):    {:>10.1} µs/call", gain_us);
    
    // Benchmark psychoacoustic analyze
    let sfb_offsets: Vec<usize> = (0..=49).map(|i| (i * 21).min(n)).collect();
    let mut psycho = vuiocodecaac::encoder::aac::psycho::PsychoacousticModel::new(
        44100, 64000, &sfb_offsets, false,
    );
    let mut psycho_out = vuiocodecaac::encoder::aac::psycho::PsychoResult::default();
    let sequence = vuiocodecaac::types::WindowSequence::OnlyLongSequence;
    for _ in 0..10 {
        psycho.analyse(black_box(&spectrum), black_box(&sfb_offsets), sequence, &mut psycho_out);
    }
    let start = Instant::now();
    for _ in 0..iterations {
        psycho.analyse(black_box(&spectrum), black_box(&sfb_offsets), sequence, &mut psycho_out);
        black_box(psycho_out.perceptual_entropy);
    }
    let elapsed_psycho = start.elapsed();
    let psycho_us = elapsed_psycho.as_secs_f64() * 1e6 / iterations as f64;
    println!("psycho.analyse:                {:>10.1} µs/call", psycho_us);
    
    // Benchmark window application
    let window: Vec<f32> = (0..2*n).map(|i| {
        let x = (i as f32 + 0.5) * std::f32::consts::PI / (2.0 * n as f32);
        x.sin()
    }).collect();
    let combined: Vec<f32> = (0..2*n).map(|i| (i as f32 * 0.01).sin()).collect();
    let mut windowed = vec![0.0f32; 2 * n];
    let start = Instant::now();
    for _ in 0..iterations {
        for i in 0..2 * n {
            windowed[i] = black_box(combined[i]) * window[i];
        }
        black_box(&windowed);
    }
    let elapsed_win = start.elapsed();
    let win_us = elapsed_win.as_secs_f64() * 1e6 / iterations as f64;
    println!("window application:            {:>10.1} µs/call", win_us);

    // Benchmark ADTS framing  
    let raw_payload = vec![0u8; 200];
    let start = Instant::now();
    for _ in 0..iterations {
        let frame = vuiocodecaac::encoder::aac::bitstream::finalize_adts_frame(
            black_box(&raw_payload),
            vuiocodecaac::types::AudioObjectType::AacLc,
            vuiocodecaac::types::SamplingRate::Hz48000,
            vuiocodecaac::types::ChannelConfiguration::Stereo,
        );
        black_box(frame);
    }
    let elapsed_adts = start.elapsed();
    let adts_us = elapsed_adts.as_secs_f64() * 1e6 / iterations as f64;
    println!("finalize_adts_frame:           {:>10.1} µs/call", adts_us);

    // Summary
    let total_per_ch = mdct_us + quant_us + gain_us + psycho_us + win_us;
    let total_stereo = total_per_ch * 2.0 + adts_us;
    let audio_per_frame_us = 1024.0 / 48000.0 * 1e6;
    
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  ENCODER PER-FRAME COST BREAKDOWN (1024 samples, 48kHz)    ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  forward_mdct_fft:        {:>8.1} µs  ({:>5.1}%)              ║", mdct_us * 2.0, mdct_us * 2.0 / total_stereo * 100.0);
    println!("║  quantize_band:           {:>8.1} µs  ({:>5.1}%)              ║", quant_us * 2.0, quant_us * 2.0 / total_stereo * 100.0);
    println!("║  estimate_global_gain:    {:>8.1} µs  ({:>5.1}%)              ║", gain_us * 2.0, gain_us * 2.0 / total_stereo * 100.0);
    println!("║  psycho.analyze:          {:>8.1} µs  ({:>5.1}%)              ║", psycho_us * 2.0, psycho_us * 2.0 / total_stereo * 100.0);
    println!("║  window application:      {:>8.1} µs  ({:>5.1}%)              ║", win_us * 2.0, win_us * 2.0 / total_stereo * 100.0);
    println!("║  finalize_adts_frame:     {:>8.1} µs  ({:>5.1}%)              ║", adts_us, adts_us / total_stereo * 100.0);
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  TOTAL per stereo frame:  {:>8.1} µs                        ║", total_stereo);
    println!("║  Audio per frame (48kHz): {:>8.1} µs                        ║", audio_per_frame_us);
    println!("║  Encoder speed:           {:>8.1}x real-time                ║", audio_per_frame_us / total_stereo);
    println!("╚══════════════════════════════════════════════════════════════╝");
}
