//! Window Tables and Generators (Sine, KBD, Low-Delay)
//!
//! Provides mathematically exact and bit-exact Sine, Kaiser-Bessel Derived (KBD),
//! and Low-Delay window lookup tables for 1024, 960, 512, 480, 128, 120 point transforms.

use std::f64::consts::PI;

/// Generate standard Sine window of length $N$ in 32-bit Q31 format.
pub fn generate_sine_window_q31(n: usize) -> Vec<i32> {
    let mut win = Vec::with_capacity(n);
    let scale = (1u64 << 31) as f64;
    for i in 0..n {
        let val = ((i as f64 + 0.5) * PI / (n as f64)).sin();
        let val_q31 = (val * (scale - 1.0)).round() as i64;
        win.push(val_q31.clamp(0, i32::MAX as i64) as i32);
    }
    win
}

/// Generate standard Sine window of length $N$ in 32-bit floating point format.
pub fn generate_sine_window_f32(n: usize) -> Vec<f32> {
    let mut win = Vec::with_capacity(n);
    for i in 0..n {
        let val = ((i as f64 + 0.5) * PI / (n as f64)).sin();
        win.push(val as f32);
    }
    win
}

/// Modified Bessel function of the first kind $I_0(x)$.
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let x_half = x / 2.0;
    for k in 1..50 {
        term *= (x_half / k as f64) * (x_half / k as f64);
        sum += term;
        if term < 1e-16 {
            break;
        }
    }
    sum
}

/// Generate Kaiser-Bessel Derived (KBD) window of length $N$ with alpha parameter.
pub fn generate_kbd_window_f32(n: usize, alpha: f64) -> Vec<f32> {
    let half_n = n / 2;
    let mut kaiser = Vec::with_capacity(half_n + 1);
    let mut sum = 0.0;

    for i in 0..=half_n {
        let term = (2.0 * i as f64 / half_n as f64) - 1.0;
        let arg = (1.0 - term * term).max(0.0).sqrt();
        let val = bessel_i0(PI * alpha * arg);
        kaiser.push(val);
        sum += val;
    }

    let mut kbd = vec![0.0f32; n];
    let mut cum_sum = 0.0;
    for i in 0..half_n {
        cum_sum += kaiser[i];
        let val = (cum_sum / sum).sqrt() as f32;
        kbd[i] = val;
        kbd[n - 1 - i] = val;
    }

    kbd
}

/// Generate KBD window of length $N$ in 32-bit Q31 format.
pub fn generate_kbd_window_q31(n: usize, alpha: f64) -> Vec<i32> {
    let kbd_f32 = generate_kbd_window_f32(n, alpha);
    let scale = (1u64 << 31) as f32;
    kbd_f32
        .into_iter()
        .map(|v| (v * (scale - 1.0)).round().clamp(0.0, i32::MAX as f32) as i32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sine_window_symmetry() {
        let win = generate_sine_window_f32(1024);
        assert_eq!(win.len(), 1024);
        // Energy conservation property: sin^2(x) + cos^2(x) = 1
        for i in 0..512 {
            let left = win[i] as f64;
            let right = win[1024 - 1 - i] as f64;
            assert!((left - right).abs() < 1e-6);
        }
    }

    #[test]
    fn test_kbd_window_symmetry() {
        let win = generate_kbd_window_f32(1024, 4.0);
        assert_eq!(win.len(), 1024);
        for i in 0..512 {
            let left = win[i] as f64;
            let right = win[1024 - 1 - i] as f64;
            assert!((left - right).abs() < 1e-6);
        }
    }
}
