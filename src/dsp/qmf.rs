//! Quadrature Mirror Filterbank (QMF) Subsystem
//!
//! Provides 32-band Analysis and 64-band Synthesis QMF filterbanks for SBR and MPS.

use std::f32::consts::PI;
use crate::dsp::fft::Complex32;

/// 32-subband Analysis QMF Filterbank.
#[derive(Debug, Clone)]
pub struct QmfAnalysis32 {
    history: Vec<f32>,
    c_proto: Vec<f32>,
}

impl Default for QmfAnalysis32 {
    fn default() -> Self {
        Self::new()
    }
}

impl QmfAnalysis32 {
    /// Create new 32-band QMF analysis filterbank.
    pub fn new() -> Self {
        let mut c_proto = Vec::with_capacity(320);
        for i in 0..320 {
            let val = ((i as f32 + 0.5) * PI / 320.0).sin();
            c_proto.push(val);
        }
        Self {
            history: vec![0.0f32; 320],
            c_proto,
        }
    }

    /// Process 32 time-domain input samples into 32 complex subband outputs.
    pub fn process_timeslot(&mut self, input_32: &[f32], output_subbands: &mut [Complex32]) {
        assert_eq!(input_32.len(), 32);
        assert_eq!(output_subbands.len(), 32);

        // Shift history buffer by 32 samples
        self.history.copy_within(32..320, 0);
        for (i, &sample) in input_32.iter().enumerate() {
            self.history[320 - 32 + i] = sample;
        }

        // Window with prototype filter
        let mut u = [0.0f32; 64];
        for (k, u_k) in u.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for j in 0..5 {
                let idx = k + j * 64;
                sum += self.history[idx] * self.c_proto[idx];
            }
            *u_k = sum;
        }

        // Modulate into 32 subbands
        for (k, out) in output_subbands.iter_mut().enumerate().take(32) {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for (n, &u_n) in u.iter().enumerate() {
                let angle = (2.0 * k as f32 + 1.0) * (n as f32 - 16.0) * (PI / 64.0);
                re += u_n * angle.cos();
                im += u_n * angle.sin();
            }
            *out = Complex32::new(re, im);
        }
    }
}

/// 64-subband Synthesis QMF Filterbank.
#[derive(Debug, Clone)]
pub struct QmfSynthesis64 {
    history: Vec<f32>,
    c_proto: Vec<f32>,
}

impl Default for QmfSynthesis64 {
    fn default() -> Self {
        Self::new()
    }
}

impl QmfSynthesis64 {
    /// Create new 64-band QMF synthesis filterbank.
    pub fn new() -> Self {
        let mut c_proto = Vec::with_capacity(640);
        for i in 0..640 {
            let val = ((i as f32 + 0.5) * PI / 640.0).sin();
            c_proto.push(val);
        }
        Self {
            history: vec![0.0f32; 640],
            c_proto,
        }
    }

    /// Synthesize 64 complex subband inputs into 64 time-domain output samples.
    pub fn process_timeslot(&mut self, subbands: &[Complex32], output_64: &mut [f32]) {
        assert_eq!(subbands.len(), 64);
        assert_eq!(output_64.len(), 64);

        // Subband synthesis modulation into 128 time samples
        let mut v = [0.0f32; 128];
        for (n, v_n) in v.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (k, sub) in subbands.iter().enumerate().take(64) {
                let angle = (2.0 * k as f32 + 1.0) * (n as f32) * (PI / 128.0);
                sum += sub.re * angle.cos() - sub.im * angle.sin();
            }
            *v_n = sum / 64.0;
        }

        // Shift history buffer by 128 samples
        self.history.copy_within(128..640, 0);
        for (i, &v_i) in v.iter().enumerate() {
            self.history[640 - 128 + i] = v_i;
        }

        // Window synthesis output
        for (i, out) in output_64.iter_mut().enumerate().take(64) {
            let mut sum = 0.0f32;
            for j in 0..5 {
                let idx = i + j * 128;
                sum += self.history[idx] * self.c_proto[idx];
            }
            *out = sum;
        }
    }
}
