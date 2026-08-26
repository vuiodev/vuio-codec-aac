//! Linear Predictive Coding (LPC) and Spectral Analysis Subsystem
//!
//! Implements autocorrelation, Levinson-Durbin recursion, reflection coefficients,
//! Line Spectral Frequencies (LSF) / Immittance Spectral Pairs (ISP) conversion,
//! and LPC all-pole synthesis/analysis filtering.

use crate::error::{DspError, Result};

/// Levinson-Durbin recursion to compute LPC filter coefficients from autocorrelation.
pub fn levinson_durbin(autocorr: &[f32], order: usize, lpc_out: &mut [f32], rc_out: &mut [f32]) -> Result<f32> {
    if autocorr.len() <= order || lpc_out.len() <= order || rc_out.len() < order {
        return Err(DspError::InvalidTransformSize {
            size: order,
            expected: autocorr.len(),
        }.into());
    }

    let mut error = autocorr[0];
    if error <= 0.0 {
        lpc_out[0] = 1.0;
        for c in &mut lpc_out[1..=order] {
            *c = 0.0;
        }
        for r in &mut rc_out[..order] {
            *r = 0.0;
        }
        return Ok(0.0);
    }

    lpc_out[0] = 1.0;
    let mut tmp = vec![0.0f32; order + 1];

    for i in 1..=order {
        let mut sum = 0.0f32;
        for j in 1..i {
            sum += lpc_out[j] * autocorr[i - j];
        }

        let k = -(autocorr[i] + sum) / error;
        rc_out[i - 1] = k;

        tmp[i] = k;
        for j in 1..i {
            tmp[j] = lpc_out[j] + k * lpc_out[i - j];
        }

        lpc_out[1..=i].copy_from_slice(&tmp[1..=i]);
        error *= 1.0 - k * k;
        if error <= 0.0 {
            error = 1e-12;
        }
    }

    Ok(error)
}

/// Apply LPC all-pole synthesis filter: H(z) = 1 / (1 + sum(a_k * z^-k)).
pub fn lpc_synthesis_filter(
    lpc_coeffs: &[f32],
    excitation: &[f32],
    state: &mut [f32],
    output: &mut [f32],
) {
    let order = lpc_coeffs.len().saturating_sub(1);
    assert!(state.len() >= order);
    assert_eq!(excitation.len(), output.len());

    for (out, &exc) in output.iter_mut().zip(excitation.iter()) {
        let mut acc = exc;
        for (k, &coeff) in lpc_coeffs.iter().enumerate().skip(1) {
            acc -= coeff * state[k - 1];
        }

        // Shift state line
        for k in (1..order).rev() {
            state[k] = state[k - 1];
        }
        if order > 0 {
            state[0] = acc;
        }
        *out = acc;
    }
}

/// Apply LPC all-zero analysis filter (inverse filter): A(z) = 1 + sum(a_k * z^-k).
pub fn lpc_analysis_filter(
    lpc_coeffs: &[f32],
    signal: &[f32],
    state: &mut [f32],
    residual: &mut [f32],
) {
    let order = lpc_coeffs.len().saturating_sub(1);
    assert!(state.len() >= order);
    assert_eq!(signal.len(), residual.len());

    for (res, &sig) in residual.iter_mut().zip(signal.iter()) {
        let mut acc = sig;
        for (k, &coeff) in lpc_coeffs.iter().enumerate().skip(1) {
            acc += coeff * state[k - 1];
        }

        // Shift state line
        for k in (1..order).rev() {
            state[k] = state[k - 1];
        }
        if order > 0 {
            state[0] = sig;
        }
        *res = acc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levinson_durbin_and_lpc_filtering() {
        let autocorr = [1.0f32, 0.8, 0.5, 0.2];
        let mut lpc = [0.0f32; 4];
        let mut rc = [0.0f32; 3];

        let error = levinson_durbin(&autocorr, 3, &mut lpc, &mut rc).unwrap();
        assert!(error > 0.0);
        assert_eq!(lpc[0], 1.0);

        let signal = [1.0f32, 0.5, -0.2, 0.8, 0.1];
        let mut state_an = [0.0f32; 3];
        let mut residual = [0.0f32; 5];
        lpc_analysis_filter(&lpc, &signal, &mut state_an, &mut residual);

        let mut state_syn = [0.0f32; 3];
        let mut reconstructed = [0.0f32; 5];
        lpc_synthesis_filter(&lpc, &residual, &mut state_syn, &mut reconstructed);

        for (a, b) in signal.iter().zip(reconstructed.iter()) {
            assert!((a - b).abs() < 1e-4);
        }
    }
}
