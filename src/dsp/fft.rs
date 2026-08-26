//! Complex FFT for power-of-two lengths.
//!
//! Decimation-in-time, radix-4 wherever the remaining length allows and radix-2 for
//! the single leftover stage when the length is an odd power of two. Radix-4 does
//! the same work as two radix-2 stages with three quarters of the multiplies,
//! because the twiddle by `i` inside the butterfly is a swap and a negation.
//!
//! Two things keep the inner loops fast:
//!
//! * Twiddles are stored per stage in the order the butterflies read them, so the
//!   hot loops walk memory linearly instead of striding by a per-stage step.
//! * The input permutation is precomputed, so the reordering pass is a gather with
//!   no branches or index arithmetic.

use std::f64::consts::PI;
use std::ops::{Add, Mul, Sub};

/// Complex number with a contiguous two-float layout.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[repr(C)]
pub struct Complex32 {
    pub re: f32,
    pub im: f32,
}

impl Complex32 {
    #[inline(always)]
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }
}

impl Add for Complex32 {
    type Output = Self;
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        Self { re: self.re + o.re, im: self.im + o.im }
    }
}

impl Sub for Complex32 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        Self { re: self.re - o.re, im: self.im - o.im }
    }
}

impl Mul for Complex32 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        Self {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }
}

/// One butterfly stage: its radix and the twiddles its butterflies consume.
///
/// The `radix - 1` twiddle factors are kept in separate contiguous arrays rather
/// than interleaved per butterfly, so a vectorized stage can load four consecutive
/// `w^j` with one instruction instead of gathering at stride three.
#[derive(Debug, Clone)]
pub(crate) struct Stage {
    pub(crate) radix: usize,
    /// Sub-transform length entering this stage.
    pub(crate) span: usize,
    /// `w^j` for `j` in `0..span`. Empty when the stage's twiddles are all 1.
    pub(crate) tw1: Vec<Complex32>,
    /// `w^(2j)`; empty for radix-2 stages and for unit-twiddle stages.
    pub(crate) tw2: Vec<Complex32>,
    /// `w^(3j)`; empty for radix-2 stages and for unit-twiddle stages.
    pub(crate) tw3: Vec<Complex32>,
}

/// Precomputed plan for one transform length.
#[derive(Debug, Clone)]
pub struct FftContext {
    pub length: usize,
    /// `permutation[k]` is the input index that belongs at position `k`.
    permutation: Vec<u32>,
    stages: Vec<Stage>,
    /// Kept for API compatibility with callers that inspected it.
    pub twiddles: Vec<Complex32>,
}

/// Order in which inputs must be arranged for the decimation-in-time stages.
///
/// Splits the index set by residue mod the stage radix, recursively, which is the
/// generalization of bit reversal to mixed radix.
fn digit_reverse(n: usize) -> Vec<u32> {
    fn rec(indices: &[u32], out: &mut Vec<u32>) {
        if indices.len() == 1 {
            out.push(indices[0]);
            return;
        }
        let radix = if indices.len().is_multiple_of(4) { 4 } else { 2 };
        for r in 0..radix {
            let sub: Vec<u32> = indices.iter().skip(r).step_by(radix).copied().collect();
            rec(&sub, out);
        }
    }
    let all: Vec<u32> = (0..n as u32).collect();
    let mut out = Vec::with_capacity(n);
    rec(&all, &mut out);
    out
}

/// Radices the stages use, innermost first.
fn stage_radices(n: usize) -> Vec<usize> {
    // Mirrors the recursion in `digit_reverse`: the innermost split is the deepest
    // one, so collect from the bottom up.
    let mut lengths = Vec::new();
    let mut len = n;
    while len > 1 {
        let radix = if len.is_multiple_of(4) { 4 } else { 2 };
        lengths.push(radix);
        len /= radix;
    }
    lengths.reverse();
    lengths
}

impl FftContext {
    /// Build a plan for transform length `n`, which must be a power of two.
    pub fn new(length: usize) -> Self {
        assert!(length.is_power_of_two(), "FFT length must be a power of two");

        let permutation = digit_reverse(length);
        let mut stages = Vec::new();
        let mut span = 1usize;

        for radix in stage_radices(length) {
            let size = span * radix;
            let factor = |r: usize| -> Vec<Complex32> {
                (0..span)
                    .map(|j| {
                        let angle = -2.0 * PI * (r * j) as f64 / size as f64;
                        Complex32::new(angle.cos() as f32, angle.sin() as f32)
                    })
                    .collect()
            };
            // A stage entered with span 1 has only unit twiddles.
            let (tw1, tw2, tw3) = if span == 1 {
                (Vec::new(), Vec::new(), Vec::new())
            } else if radix == 4 {
                (factor(1), factor(2), factor(3))
            } else {
                (factor(1), Vec::new(), Vec::new())
            };
            stages.push(Stage { radix, span, tw1, tw2, tw3 });
            span = size;
        }

        // Legacy table: W_N^k for k in 0..n/2.
        let twiddles = (0..length / 2)
            .map(|k| {
                let a = -2.0 * PI * k as f64 / length as f64;
                Complex32::new(a.cos() as f32, a.sin() as f32)
            })
            .collect();

        Self { length, permutation, stages, twiddles }
    }

    /// In-place forward transform.
    ///
    /// Allocates a scratch buffer per call; [`Self::forward_into`] avoids that and
    /// is what the decode path uses.
    pub fn forward(&self, buffer: &mut [Complex32]) {
        assert_eq!(buffer.len(), self.length);
        self.run(buffer, false);
    }

    /// Forward transform from `input` into `output`, without allocating.
    ///
    /// The decimation-in-time reordering is folded into the copy, so this costs one
    /// gather pass rather than a copy plus an in-place permutation.
    pub fn forward_into(&self, input: &[Complex32], output: &mut [Complex32]) {
        debug_assert_eq!(input.len(), self.length);
        debug_assert_eq!(output.len(), self.length);

        for (dst, &p) in output.iter_mut().zip(self.permutation.iter()) {
            *dst = input[p as usize];
        }
        for stage in &self.stages {
            match stage.radix {
                4 => radix4_stage(output, stage),
                _ => radix2_stage(output, stage),
            }
        }
    }

    /// The decimation-in-time input order, for callers that fold the reordering
    /// into whatever pass produces the FFT input.
    #[inline]
    pub fn permutation(&self) -> &[u32] {
        &self.permutation
    }

    /// In-place inverse transform, normalized by `1/n`.
    pub fn inverse(&self, buffer: &mut [Complex32]) {
        assert_eq!(buffer.len(), self.length);
        self.run(buffer, true);
        let scale = 1.0 / self.length as f32;
        for c in buffer.iter_mut() {
            c.re *= scale;
            c.im *= scale;
        }
    }

    fn run(&self, buffer: &mut [Complex32], inverse: bool) {
        let n = self.length;
        if n == 1 {
            return;
        }

        // Reorder into decimation-in-time order. Conjugating on the way in and out
        // turns the forward transform into the inverse, which avoids a second set
        // of twiddle tables and a branch in every butterfly.
        let mut work: Vec<Complex32> = Vec::with_capacity(n);
        if inverse {
            work.extend(self.permutation.iter().map(|&p| {
                let c = buffer[p as usize];
                Complex32::new(c.re, -c.im)
            }));
        } else {
            work.extend(self.permutation.iter().map(|&p| buffer[p as usize]));
        }

        for stage in &self.stages {
            match stage.radix {
                4 => radix4_stage(&mut work, stage),
                _ => radix2_stage(&mut work, stage),
            }
        }

        if inverse {
            for (dst, src) in buffer.iter_mut().zip(work.iter()) {
                *dst = Complex32::new(src.re, -src.im);
            }
        } else {
            buffer.copy_from_slice(&work);
        }
    }
}

/// Apply every radix-2 butterfly of one stage.
///
/// The two halves of each block are split into disjoint slices so the loop is a
/// straight zip with no bounds checks, which lets the vectorizer fuse the
/// add/subtract pairs.
#[inline]
fn radix2_stage(data: &mut [Complex32], stage: &Stage) {
    let span = stage.span;
    let size = span * 2;

    for block in data.chunks_exact_mut(size) {
        let (lo, hi) = block.split_at_mut(span);
        if stage.tw1.is_empty() {
            for (a, b) in lo.iter_mut().zip(hi.iter_mut()) {
                let x = *a;
                let t = *b;
                *a = Complex32::new(x.re + t.re, x.im + t.im);
                *b = Complex32::new(x.re - t.re, x.im - t.im);
            }
        } else {
            for ((a, b), w) in lo.iter_mut().zip(hi.iter_mut()).zip(stage.tw1.iter()) {
                let x = *a;
                let y = *b;
                let tr = y.re * w.re - y.im * w.im;
                let ti = y.re * w.im + y.im * w.re;
                *a = Complex32::new(x.re + tr, x.im + ti);
                *b = Complex32::new(x.re - tr, x.im - ti);
            }
        }
    }
}

/// Apply every radix-4 butterfly of one stage.
///
/// The `-i` and `+i` twiddles inside the butterfly are applied as a real/imaginary
/// swap with a sign flip rather than a multiply, which is where radix-4 saves work
/// over two radix-2 stages.
#[inline]
fn radix4_stage(data: &mut [Complex32], stage: &Stage) {
    let span = stage.span;
    let size = span * 4;

    for block in data.chunks_exact_mut(size) {
        let (q0, rest) = block.split_at_mut(span);
        let (q1, rest) = rest.split_at_mut(span);
        let (q2, q3) = rest.split_at_mut(span);

        if stage.tw1.is_empty() {
            for (((a, b), c), d) in
                q0.iter_mut().zip(q1.iter_mut()).zip(q2.iter_mut()).zip(q3.iter_mut())
            {
                butterfly4(a, b, c, d, *b, *c, *d);
            }
        } else if crate::dsp::simd::radix4_twiddled(q0, q1, q2, q3, &stage.tw1, &stage.tw2, &stage.tw3)
        {
            // Handled by the vectorized kernel.
        } else {
            for (((((a, b), c), d), w1), (w2, w3)) in q0
                .iter_mut()
                .zip(q1.iter_mut())
                .zip(q2.iter_mut())
                .zip(q3.iter_mut())
                .zip(stage.tw1.iter())
                .zip(stage.tw2.iter().zip(stage.tw3.iter()))
            {
                let bw = Complex32::new(b.re * w1.re - b.im * w1.im, b.re * w1.im + b.im * w1.re);
                let cw = Complex32::new(c.re * w2.re - c.im * w2.im, c.re * w2.im + c.im * w2.re);
                let dw = Complex32::new(d.re * w3.re - d.im * w3.im, d.re * w3.im + d.im * w3.re);
                butterfly4(a, b, c, d, bw, cw, dw);
            }
        }
    }
}

/// The radix-4 butterfly proper, given already-twiddled inputs.
#[inline(always)]
fn butterfly4(
    a: &mut Complex32,
    b: &mut Complex32,
    c: &mut Complex32,
    d: &mut Complex32,
    bw: Complex32,
    cw: Complex32,
    dw: Complex32,
) {
    let t0 = Complex32::new(a.re + cw.re, a.im + cw.im);
    let t1 = Complex32::new(a.re - cw.re, a.im - cw.im);
    let t2 = Complex32::new(bw.re + dw.re, bw.im + dw.im);
    let t3 = Complex32::new(bw.re - dw.re, bw.im - dw.im);

    *a = Complex32::new(t0.re + t2.re, t0.im + t2.im);
    *c = Complex32::new(t0.re - t2.re, t0.im - t2.im);
    *b = Complex32::new(t1.re + t3.im, t1.im - t3.re);
    *d = Complex32::new(t1.re - t3.im, t1.im + t3.re);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Direct DFT, for validating the fast transform.
    fn dft(input: &[Complex32]) -> Vec<Complex32> {
        let n = input.len();
        (0..n)
            .map(|k| {
                let mut re = 0.0f64;
                let mut im = 0.0f64;
                for (j, x) in input.iter().enumerate() {
                    let a = -2.0 * PI * (k * j) as f64 / n as f64;
                    let (s, c) = a.sin_cos();
                    re += x.re as f64 * c - x.im as f64 * s;
                    im += x.re as f64 * s + x.im as f64 * c;
                }
                Complex32::new(re as f32, im as f32)
            })
            .collect()
    }

    fn sample(n: usize, seed: u32) -> Vec<Complex32> {
        (0..n)
            .map(|i| {
                let t = (i as f32) + seed as f32 * 0.37;
                Complex32::new((t * 0.1).sin() + (t * 0.7).cos(), (t * 0.2).cos() - (t * 1.3).sin())
            })
            .collect()
    }

    /// The fast transform must match a direct DFT at every length the codec uses,
    /// covering both even and odd powers of two so the mixed-radix path is exercised.
    #[test]
    fn matches_direct_dft() {
        for &n in &[1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
            let input = sample(n, 1);
            let want = dft(&input);
            let mut got = input.clone();
            FftContext::new(n).forward(&mut got);

            let peak = want.iter().fold(0.0f32, |m, c| m.max(c.re.abs()).max(c.im.abs()));
            for (k, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                let e = (g.re - w.re).abs().max((g.im - w.im).abs());
                assert!(e <= peak * 2e-5 + 1e-4, "n={n} k={k}: {g:?} vs {w:?}");
            }
        }
    }

    /// Forward then inverse must return the original.
    #[test]
    fn round_trips() {
        for &n in &[2usize, 8, 64, 128, 512, 1024] {
            for seed in 0..3 {
                let original = sample(n, seed);
                let mut data = original.clone();
                let ctx = FftContext::new(n);
                ctx.forward(&mut data);
                ctx.inverse(&mut data);
                for (a, b) in original.iter().zip(data.iter()) {
                    assert!((a.re - b.re).abs() < 1e-4, "n={n}: {a:?} vs {b:?}");
                    assert!((a.im - b.im).abs() < 1e-4, "n={n}: {a:?} vs {b:?}");
                }
            }
        }
    }

    /// A DC input must transform to a single non-zero bin.
    #[test]
    fn dc_maps_to_bin_zero() {
        let n = 256;
        let mut data = vec![Complex32::new(1.0, 0.0); n];
        FftContext::new(n).forward(&mut data);
        assert!((data[0].re - n as f32).abs() < 1e-3);
        assert!(data[0].im.abs() < 1e-3);
        for c in &data[1..] {
            assert!(c.re.abs() < 1e-3 && c.im.abs() < 1e-3);
        }
    }

    /// The transform is linear.
    #[test]
    fn is_linear() {
        let n = 128;
        let ctx = FftContext::new(n);
        let a = sample(n, 2);
        let b = sample(n, 5);
        let sum: Vec<Complex32> = a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect();

        let (mut fa, mut fb, mut fs) = (a.clone(), b.clone(), sum.clone());
        ctx.forward(&mut fa);
        ctx.forward(&mut fb);
        ctx.forward(&mut fs);

        for k in 0..n {
            assert!((fa[k].re + fb[k].re - fs[k].re).abs() < 1e-2);
            assert!((fa[k].im + fb[k].im - fs[k].im).abs() < 1e-2);
        }
    }

    /// The permutation must be a genuine permutation of `0..n`.
    #[test]
    fn permutation_is_a_bijection() {
        for &n in &[2usize, 8, 32, 128, 512, 1024] {
            let p = digit_reverse(n);
            assert_eq!(p.len(), n);
            let mut seen = vec![false; n];
            for &i in &p {
                assert!(!seen[i as usize], "n={n}: index {i} appears twice");
                seen[i as usize] = true;
            }
        }
    }

    /// The stage radices must multiply back to the transform length.
    #[test]
    fn stage_radices_factor_the_length() {
        for &n in &[2usize, 4, 8, 64, 512, 1024] {
            let product: usize = stage_radices(n).iter().product();
            assert_eq!(product, n, "radices for {n} do not multiply to it");
        }
    }
}
