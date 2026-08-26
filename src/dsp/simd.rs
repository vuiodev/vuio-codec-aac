//! Hand-vectorized kernels for the DSP hot paths.
//!
//! Every kernel here has a scalar equivalent elsewhere in `dsp`, produces the same
//! results, and is selected only when the target architecture provides the
//! instructions it needs. Each returns `bool`: `true` when it handled the work,
//! `false` when the caller should fall back to its scalar loop. That keeps the
//! dispatch explicit and keeps a portable build correct with no extra branches on
//! architectures that have no kernel.
//!
//! # Why this module contains `unsafe`
//!
//! SIMD intrinsics are `unsafe` functions in `core::arch` because they can require
//! CPU features the running machine may lack, and the load/store forms take raw
//! pointers. Both obligations are discharged here:
//!
//! * **Feature availability.** NEON is architecturally guaranteed on `aarch64`, so
//!   `#[cfg(target_arch = "aarch64")]` is sufficient. The x86-64 kernels are gated
//!   on a runtime `is_x86_feature_detected!` check.
//! * **Memory safety.** Every load and store is bounds-checked against slice lengths
//!   before the pointer is formed, and the loops step in fixed-size chunks that were
//!   derived from those same lengths.
//!
//! The rest of the crate denies `unsafe_code`; this module is the single audited
//! exception.

#![allow(unsafe_code)]

use crate::dsp::fft::Complex32;

/// Complex values processed per vector: NEON and SSE hold four `f32` lanes.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const LANES: usize = 4;

/// Vectorized radix-4 butterfly stage with twiddles.
///
/// `q0..q3` are the four quarters of one butterfly block and must all have the same
/// length, as must the three twiddle arrays. Returns `false` if no kernel applies,
/// in which case the caller runs its scalar loop.
#[inline]
pub fn radix4_twiddled(
    q0: &mut [Complex32],
    q1: &mut [Complex32],
    q2: &mut [Complex32],
    q3: &mut [Complex32],
    tw1: &[Complex32],
    tw2: &[Complex32],
    tw3: &[Complex32],
) -> bool {
    let span = q0.len();
    // A short span costs more in setup than it saves; leave it to the scalar path.
    if span < LANES_MIN
        || q1.len() != span
        || q2.len() != span
        || q3.len() != span
        || tw1.len() < span
        || tw2.len() < span
        || tw3.len() < span
    {
        return false;
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is guaranteed on aarch64, and the lengths were checked above.
        unsafe { neon::radix4_twiddled(q0, q1, q2, q3, tw1, tw2, tw3) };
        return true;
    }

    #[cfg(all(target_arch = "x86_64", not(target_arch = "aarch64")))]
    {
        if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
            // SAFETY: the features were just detected and the lengths checked above.
            unsafe { avx::radix4_twiddled(q0, q1, q2, q3, tw1, tw2, tw3) };
            return true;
        }
    }

    #[allow(unreachable_code)]
    {
        let _ = (q0, q1, q2, q3, tw1, tw2, tw3);
        false
    }
}

/// Smallest span worth vectorizing.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const LANES_MIN: usize = LANES;
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
const LANES_MIN: usize = usize::MAX;

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::Complex32;
    use std::arch::aarch64::*;

    /// Deinterleave four complex values into (real, imaginary) vectors.
    ///
    /// # Safety
    /// `p` must be valid for reading four `Complex32`.
    #[inline(always)]
    unsafe fn load4(p: *const Complex32) -> (float32x4_t, float32x4_t) {
        // SAFETY: Complex32 is repr(C) with two f32 fields, so four of them are
        // eight contiguous f32; the caller guarantees the range is readable.
        let v = unsafe { vld2q_f32(p as *const f32) };
        (v.0, v.1)
    }

    /// Interleave (real, imaginary) vectors back into four complex values.
    ///
    /// # Safety
    /// `p` must be valid for writing four `Complex32`.
    #[inline(always)]
    unsafe fn store4(p: *mut Complex32, re: float32x4_t, im: float32x4_t) {
        // SAFETY: same layout argument as `load4`; the caller guarantees writability.
        unsafe { vst2q_f32(p as *mut f32, float32x4x2_t(re, im)) };
    }

    /// Complex multiply of four pairs held as split real/imaginary vectors.
    #[inline(always)]
    unsafe fn cmul(
        ar: float32x4_t,
        ai: float32x4_t,
        br: float32x4_t,
        bi: float32x4_t,
    ) -> (float32x4_t, float32x4_t) {
        // SAFETY: NEON is available on every aarch64 target.
        unsafe {
            let re = vfmsq_f32(vmulq_f32(ar, br), ai, bi);
            let im = vfmaq_f32(vmulq_f32(ar, bi), ai, br);
            (re, im)
        }
    }

    /// Radix-4 butterflies over a whole span.
    ///
    /// # Safety
    /// All six slices must be at least `q0.len()` long, and `q0..q3` must not alias.
    pub unsafe fn radix4_twiddled(
        q0: &mut [Complex32],
        q1: &mut [Complex32],
        q2: &mut [Complex32],
        q3: &mut [Complex32],
        tw1: &[Complex32],
        tw2: &[Complex32],
        tw3: &[Complex32],
    ) {
        let span = q0.len();
        let chunks = span / 4;

        for i in 0..chunks {
            let off = i * 4;
            // SAFETY: `off + 4 <= span` by construction, and every slice is at
            // least `span` long, so all eight accesses stay in bounds.
            unsafe {
                let (a_re, a_im) = load4(q0.as_ptr().add(off));
                let (b_re, b_im) = load4(q1.as_ptr().add(off));
                let (c_re, c_im) = load4(q2.as_ptr().add(off));
                let (d_re, d_im) = load4(q3.as_ptr().add(off));

                let (w1_re, w1_im) = load4(tw1.as_ptr().add(off));
                let (w2_re, w2_im) = load4(tw2.as_ptr().add(off));
                let (w3_re, w3_im) = load4(tw3.as_ptr().add(off));

                let (bw_re, bw_im) = cmul(b_re, b_im, w1_re, w1_im);
                let (cw_re, cw_im) = cmul(c_re, c_im, w2_re, w2_im);
                let (dw_re, dw_im) = cmul(d_re, d_im, w3_re, w3_im);

                let t0_re = vaddq_f32(a_re, cw_re);
                let t0_im = vaddq_f32(a_im, cw_im);
                let t1_re = vsubq_f32(a_re, cw_re);
                let t1_im = vsubq_f32(a_im, cw_im);
                let t2_re = vaddq_f32(bw_re, dw_re);
                let t2_im = vaddq_f32(bw_im, dw_im);
                let t3_re = vsubq_f32(bw_re, dw_re);
                let t3_im = vsubq_f32(bw_im, dw_im);

                store4(q0.as_mut_ptr().add(off), vaddq_f32(t0_re, t2_re), vaddq_f32(t0_im, t2_im));
                store4(q2.as_mut_ptr().add(off), vsubq_f32(t0_re, t2_re), vsubq_f32(t0_im, t2_im));
                // Multiplying t3 by -i and +i is a lane swap with one sign flip.
                store4(q1.as_mut_ptr().add(off), vaddq_f32(t1_re, t3_im), vsubq_f32(t1_im, t3_re));
                store4(q3.as_mut_ptr().add(off), vsubq_f32(t1_re, t3_im), vaddq_f32(t1_im, t3_re));
            }
        }

        // Tail: spans are powers of two, so this only runs for span < 4.
        for i in chunks * 4..span {
            let (b, c, d) = (q1[i], q2[i], q3[i]);
            let (w1, w2, w3) = (tw1[i], tw2[i], tw3[i]);
            let bw = Complex32::new(b.re * w1.re - b.im * w1.im, b.re * w1.im + b.im * w1.re);
            let cw = Complex32::new(c.re * w2.re - c.im * w2.im, c.re * w2.im + c.im * w2.re);
            let dw = Complex32::new(d.re * w3.re - d.im * w3.im, d.re * w3.im + d.im * w3.re);
            let a = q0[i];

            let t0 = Complex32::new(a.re + cw.re, a.im + cw.im);
            let t1 = Complex32::new(a.re - cw.re, a.im - cw.im);
            let t2 = Complex32::new(bw.re + dw.re, bw.im + dw.im);
            let t3 = Complex32::new(bw.re - dw.re, bw.im - dw.im);

            q0[i] = Complex32::new(t0.re + t2.re, t0.im + t2.im);
            q2[i] = Complex32::new(t0.re - t2.re, t0.im - t2.im);
            q1[i] = Complex32::new(t1.re + t3.im, t1.im - t3.re);
            q3[i] = Complex32::new(t1.re - t3.im, t1.im + t3.re);
        }
    }
}

#[cfg(all(target_arch = "x86_64", not(target_arch = "aarch64")))]
mod avx {
    use super::Complex32;
    use std::arch::x86_64::*;

    /// Radix-4 butterflies over a whole span, four complex values at a time.
    ///
    /// # Safety
    /// AVX and FMA must be available, all six slices must be at least `q0.len()`
    /// long, and `q0..q3` must not alias.
    #[target_feature(enable = "avx,fma")]
    pub unsafe fn radix4_twiddled(
        q0: &mut [Complex32],
        q1: &mut [Complex32],
        q2: &mut [Complex32],
        q3: &mut [Complex32],
        tw1: &[Complex32],
        tw2: &[Complex32],
        tw3: &[Complex32],
    ) {
        // Deinterleave four complex values into (re, im) 128-bit vectors.
        #[inline(always)]
        unsafe fn load4(p: *const Complex32) -> (__m128, __m128) {
            // SAFETY: the caller guarantees four readable Complex32 at `p`.
            unsafe {
                let lo = _mm_loadu_ps(p as *const f32);
                let hi = _mm_loadu_ps((p as *const f32).add(4));
                (_mm_shuffle_ps(lo, hi, 0b10_00_10_00), _mm_shuffle_ps(lo, hi, 0b11_01_11_01))
            }
        }

        #[inline(always)]
        unsafe fn store4(p: *mut Complex32, re: __m128, im: __m128) {
            // SAFETY: the caller guarantees four writable Complex32 at `p`.
            unsafe {
                _mm_storeu_ps(p as *mut f32, _mm_unpacklo_ps(re, im));
                _mm_storeu_ps((p as *mut f32).add(4), _mm_unpackhi_ps(re, im));
            }
        }

        #[inline(always)]
        unsafe fn cmul(ar: __m128, ai: __m128, br: __m128, bi: __m128) -> (__m128, __m128) {
            // SAFETY: AVX/FMA enabled by the caller's target_feature.
            unsafe {
                (
                    _mm_fmsub_ps(ar, br, _mm_mul_ps(ai, bi)),
                    _mm_fmadd_ps(ar, bi, _mm_mul_ps(ai, br)),
                )
            }
        }

        let span = q0.len();
        let chunks = span / 4;

        for i in 0..chunks {
            let off = i * 4;
            // SAFETY: `off + 4 <= span` and every slice is at least `span` long.
            unsafe {
                let (a_re, a_im) = load4(q0.as_ptr().add(off));
                let (b_re, b_im) = load4(q1.as_ptr().add(off));
                let (c_re, c_im) = load4(q2.as_ptr().add(off));
                let (d_re, d_im) = load4(q3.as_ptr().add(off));

                let (w1_re, w1_im) = load4(tw1.as_ptr().add(off));
                let (w2_re, w2_im) = load4(tw2.as_ptr().add(off));
                let (w3_re, w3_im) = load4(tw3.as_ptr().add(off));

                let (bw_re, bw_im) = cmul(b_re, b_im, w1_re, w1_im);
                let (cw_re, cw_im) = cmul(c_re, c_im, w2_re, w2_im);
                let (dw_re, dw_im) = cmul(d_re, d_im, w3_re, w3_im);

                let t0_re = _mm_add_ps(a_re, cw_re);
                let t0_im = _mm_add_ps(a_im, cw_im);
                let t1_re = _mm_sub_ps(a_re, cw_re);
                let t1_im = _mm_sub_ps(a_im, cw_im);
                let t2_re = _mm_add_ps(bw_re, dw_re);
                let t2_im = _mm_add_ps(bw_im, dw_im);
                let t3_re = _mm_sub_ps(bw_re, dw_re);
                let t3_im = _mm_sub_ps(bw_im, dw_im);

                store4(q0.as_mut_ptr().add(off), _mm_add_ps(t0_re, t2_re), _mm_add_ps(t0_im, t2_im));
                store4(q2.as_mut_ptr().add(off), _mm_sub_ps(t0_re, t2_re), _mm_sub_ps(t0_im, t2_im));
                store4(q1.as_mut_ptr().add(off), _mm_add_ps(t1_re, t3_im), _mm_sub_ps(t1_im, t3_re));
                store4(q3.as_mut_ptr().add(off), _mm_sub_ps(t1_re, t3_im), _mm_add_ps(t1_im, t3_re));
            }
        }

        for i in chunks * 4..span {
            let (b, c, d) = (q1[i], q2[i], q3[i]);
            let (w1, w2, w3) = (tw1[i], tw2[i], tw3[i]);
            let bw = Complex32::new(b.re * w1.re - b.im * w1.im, b.re * w1.im + b.im * w1.re);
            let cw = Complex32::new(c.re * w2.re - c.im * w2.im, c.re * w2.im + c.im * w2.re);
            let dw = Complex32::new(d.re * w3.re - d.im * w3.im, d.re * w3.im + d.im * w3.re);
            let a = q0[i];

            let t0 = Complex32::new(a.re + cw.re, a.im + cw.im);
            let t1 = Complex32::new(a.re - cw.re, a.im - cw.im);
            let t2 = Complex32::new(bw.re + dw.re, bw.im + dw.im);
            let t3 = Complex32::new(bw.re - dw.re, bw.im - dw.im);

            q0[i] = Complex32::new(t0.re + t2.re, t0.im + t2.im);
            q2[i] = Complex32::new(t0.re - t2.re, t0.im - t2.im);
            q1[i] = Complex32::new(t1.re + t3.im, t1.im - t3.re);
            q3[i] = Complex32::new(t1.re - t3.im, t1.im + t3.re);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scalar reference for the radix-4 butterfly stage.
    fn scalar_radix4(
        q0: &mut [Complex32],
        q1: &mut [Complex32],
        q2: &mut [Complex32],
        q3: &mut [Complex32],
        tw1: &[Complex32],
        tw2: &[Complex32],
        tw3: &[Complex32],
    ) {
        for i in 0..q0.len() {
            let (b, c, d) = (q1[i], q2[i], q3[i]);
            let (w1, w2, w3) = (tw1[i], tw2[i], tw3[i]);
            let bw = Complex32::new(b.re * w1.re - b.im * w1.im, b.re * w1.im + b.im * w1.re);
            let cw = Complex32::new(c.re * w2.re - c.im * w2.im, c.re * w2.im + c.im * w2.re);
            let dw = Complex32::new(d.re * w3.re - d.im * w3.im, d.re * w3.im + d.im * w3.re);
            let a = q0[i];

            let t0 = Complex32::new(a.re + cw.re, a.im + cw.im);
            let t1 = Complex32::new(a.re - cw.re, a.im - cw.im);
            let t2 = Complex32::new(bw.re + dw.re, bw.im + dw.im);
            let t3 = Complex32::new(bw.re - dw.re, bw.im - dw.im);

            q0[i] = Complex32::new(t0.re + t2.re, t0.im + t2.im);
            q2[i] = Complex32::new(t0.re - t2.re, t0.im - t2.im);
            q1[i] = Complex32::new(t1.re + t3.im, t1.im - t3.re);
            q3[i] = Complex32::new(t1.re - t3.im, t1.im + t3.re);
        }
    }

    fn sample(n: usize, seed: u32) -> Vec<Complex32> {
        (0..n)
            .map(|i| {
                let t = i as f32 + seed as f32 * 3.7;
                Complex32::new((t * 0.13).sin() * 4.0, (t * 0.29).cos() * 3.0)
            })
            .collect()
    }

    /// The vectorized stage must agree with the scalar one at every span the FFT
    /// plans use, including spans that leave a tail.
    #[test]
    fn vector_stage_matches_scalar() {
        for &span in &[1usize, 2, 3, 4, 5, 7, 8, 16, 32, 128, 256] {
            let (mut a, mut b, mut c, mut d) =
                (sample(span, 1), sample(span, 2), sample(span, 3), sample(span, 4));
            let (t1, t2, t3) = (sample(span, 5), sample(span, 6), sample(span, 7));

            let (mut ra, mut rb, mut rc, mut rd) = (a.clone(), b.clone(), c.clone(), d.clone());
            scalar_radix4(&mut ra, &mut rb, &mut rc, &mut rd, &t1, &t2, &t3);

            if !radix4_twiddled(&mut a, &mut b, &mut c, &mut d, &t1, &t2, &t3) {
                // No kernel on this target; nothing to compare.
                continue;
            }

            for (name, got, want) in
                [("q0", &a, &ra), ("q1", &b, &rb), ("q2", &c, &rc), ("q3", &d, &rd)]
            {
                for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                    assert!(
                        (g.re - w.re).abs() < 1e-4 && (g.im - w.im).abs() < 1e-4,
                        "span {span} {name}[{i}]: {g:?} vs {w:?}"
                    );
                }
            }
        }
    }

    /// The kernel must decline spans it cannot handle rather than misbehaving.
    #[test]
    fn declines_mismatched_lengths() {
        let mut a = sample(16, 1);
        let mut b = sample(16, 2);
        let mut c = sample(16, 3);
        let mut d = sample(8, 4);
        let t = sample(16, 5);
        assert!(!radix4_twiddled(&mut a, &mut b, &mut c, &mut d, &t, &t, &t));
    }
}
