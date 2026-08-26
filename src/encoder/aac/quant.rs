//! Rate-Distortion Quantization and Scalefactor Estimation
//!
//! Performs two-loop rate-distortion quantization to fit spectral coefficients
//! within the allocated frame bit budget while keeping distortion below masking thresholds:
//!
//! $$q = \text{sign}(x) \cdot \lfloor (|x| \cdot 2^{-(\text{sf}-100)/4})^{3/4} + 0.4054 \rfloor$$

/// Quantize spectral coefficients using given scalefactor.
pub fn quantize_band(spectrum: &[f32], scalefactor: i16, quantized: &mut [i32]) {
    assert_eq!(spectrum.len(), quantized.len());

    let sf_shift = -(scalefactor as f32 - 100.0) * 0.25;
    let inv_sf_scale = 2.0f32.powf(sf_shift);

    for (i, &x) in spectrum.iter().enumerate() {
        if x.abs() < 1e-9 {
            quantized[i] = 0;
        } else {
            let sign = if x < 0.0 { -1 } else { 1 };
            let scaled_abs = x.abs() * inv_sf_scale;
            let val = (scaled_abs.powf(0.75) + 0.4054).floor() as i32;
            quantized[i] = sign * val;
        }
    }
}

/// Estimate initial global gain from spectral RMS energy.
pub fn estimate_global_gain(spectrum: &[f32], target_bits: usize) -> i16 {
    let mut energy = 0.0f32;
    for &s in spectrum {
        energy += s * s;
    }
    let rms = (energy / spectrum.len().max(1) as f32).sqrt().max(1e-6);
    let estimated_gain = (16.0 / 3.0 * (4.0 * rms.log2() - (target_bits as f32 / spectrum.len() as f32))).round() as i16;
    estimated_gain.clamp(0, 255)
}
