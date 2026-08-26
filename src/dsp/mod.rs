//! Digital Signal Processing (DSP) Subsystems
//!
//! Provides optimized fixed-point integer math, complex FFT/IFFT,
//! forward & inverse MDCT with TDAC windowing, 32/64 band QMF filterbanks,
//! LPC analysis/synthesis, and polyphase audio resampling.

pub mod fft;
pub mod filterbank;
pub mod lpc;
pub mod math;
pub mod imdct;
pub mod mdct;
pub mod qmf;
pub mod resampler;
pub mod simd;
pub mod window;

pub use fft::{Complex32, FftContext};
pub use lpc::{levinson_durbin, lpc_analysis_filter, lpc_synthesis_filter};
pub use math::*;
pub use mdct::MdctContext;
pub use qmf::{QmfAnalysis, QmfSynthesis, SynthesisWidth};
pub use resampler::Resampler;
pub use window::{generate_kbd_window_f32, generate_low_delay_window_f32, generate_sine_window_f32};
