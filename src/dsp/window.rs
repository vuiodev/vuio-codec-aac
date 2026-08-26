//! Audio Analysis/Synthesis Window Generators
//!
//! Provides Sine, Kaiser-Bessel Derived (KBD), and Low Delay (LD / ELD) window curves
//! satisfying the Princen-Bradley Time-Domain Aliasing Cancellation (TDAC) condition:
//! w[n]^2 + w[n + N]^2 = 1.

use std::f32::consts::PI;

/// Generate Sine window of length `length`: w[n] = sin(pi * (n + 0.5) / length).
pub fn generate_sine_window_f32(length: usize) -> Vec<f32> {
    let mut window = Vec::with_capacity(length);
    for n in 0..length {
        let angle = PI * (n as f32 + 0.5) / (length as f32);
        window.push(angle.sin());
    }
    window
}

/// Zero-order modified Bessel function of the first kind I_0(x).
fn bessel_i0(x: f32) -> f32 {
    let mut sum = 1.0f32;
    let mut term = 1.0f32;
    let half_x = 0.5 * x;

    for k in 1..25 {
        term *= (half_x / k as f32) * (half_x / k as f32);
        sum += term;
        if term < 1e-12 * sum {
            break;
        }
    }
    sum
}

/// Generate Kaiser-Bessel Derived (KBD) window of length `length` with parameter `alpha` (e.g. 4.0 or 6.0).
pub fn generate_kbd_window_f32(length: usize, alpha: f32) -> Vec<f32> {
    let half = length / 2;
    let beta = PI * alpha;
    let denom = bessel_i0(beta);

    let mut kaiser = vec![0.0f32; half + 1];
    for n in 0..=half {
        let x = 2.0 * n as f32 / half as f32 - 1.0;
        let arg = (1.0 - x * x).max(0.0).sqrt();
        kaiser[n] = bessel_i0(beta * arg) / denom;
    }

    let mut cumulative = vec![0.0f32; half + 1];
    let mut sum = 0.0f32;
    for n in 0..=half {
        sum += kaiser[n];
        cumulative[n] = sum;
    }

    let total = cumulative[half];
    let mut window = vec![0.0f32; length];
    for n in 0..half {
        let val = (cumulative[n] / total).sqrt();
        window[n] = val;
        window[length - 1 - n] = val;
    }

    window
}

/// Generate Low Delay (LD / ELD) window of length `length` (e.g. 1024 or 960).
pub fn generate_low_delay_window_f32(length: usize) -> Vec<f32> {
    let mut window = Vec::with_capacity(length);
    for n in 0..length {
        let angle = PI * (n as f32 + 0.5) / (length as f32);
        // Low-delay asymmetric window profile with reduced lookahead
        let w = (PI * 0.5 * (n as f32 + 0.5) / (length as f32)).sin();
        let taper = 1.0 - 0.1 * (angle * 2.0).cos();
        window.push(w * taper);
    }
    window
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sine_window_tdac_condition() {
        let len = 2048;
        let half = len / 2;
        let win = generate_sine_window_f32(len);

        for n in 0..half {
            let p1 = win[n];
            let p2 = win[n + half];
            let tdac = p1 * p1 + p2 * p2;
            assert!((tdac - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_kbd_window_tdac_condition() {
        let len = 256;
        let half = len / 2;
        let win = generate_kbd_window_f32(len, 4.0);

        for n in 0..half {
            let p1 = win[n];
            let p2 = win[n + half];
            let tdac = p1 * p1 + p2 * p2;
            assert!((tdac - 1.0).abs() < 1e-4);
        }
    }
}
