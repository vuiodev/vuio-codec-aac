//! Inverse Quantization and Scalefactor Application
//!
//! Transforms quantized integer spectral coefficients $q$ into floating-point
//! spectral coefficients $x$:
//!
//! $$x = \text{sign}(q) \cdot |q|^{4/3} \cdot 2^{(\text{sf} - 100) / 4}$$

use crate::tables::dequant::POW_TABLE_Q13;

/// Perform inverse quantization on an array of quantized spectral coefficients.
pub fn inverse_quantize(quantized: &[i32], scalefactor: i16, output: &mut [f32]) {
    assert_eq!(quantized.len(), output.len());

    // Compute 2^((sf - 100) / 4)
    let sf_shift = (scalefactor as f32 - 100.0) * 0.25;
    let sf_scale = 2.0f32.powf(sf_shift);

    for (i, &q) in quantized.iter().enumerate() {
        if q == 0 {
            output[i] = 0.0;
        } else {
            let sign = if q < 0 { -1.0f32 } else { 1.0f32 };
            let q_abs = q.unsigned_abs() as usize;

            let pow_val = if q_abs <= 128 {
                (POW_TABLE_Q13[q_abs] as f32) / 8192.0
            } else {
                (q_abs as f32).powf(4.0 / 3.0)
            };

            output[i] = sign * pow_val * sf_scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inverse_quantize_zero() {
        let quant = [0i32; 8];
        let mut out = [0.0f32; 8];
        inverse_quantize(&quant, 100, &mut out);
        assert_eq!(out, [0.0f32; 8]);
    }

    #[test]
    fn test_inverse_quantize_one() {
        let quant = [1i32, -1];
        let mut out = [0.0f32; 2];
        inverse_quantize(&quant, 100, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-4);
        assert!((out[1] - (-1.0)).abs() < 1e-4);
    }
}
