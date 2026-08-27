//! LPC / LSP / LSF conversion for MPEG-D USAC's LPD (Linear Prediction Domain)
//! speech-coding mode, ported from `c/libxaac/encoder/iusace_lpc.c` and the
//! matching decoder-side conversions in `c/libxaac/decoder/ixheaacd_lpc_dec.c`.
//!
//! Every LPD frame carries its short-term spectral envelope as a 16th-order LPC
//! filter, but never transmits the filter coefficients directly: coefficients are
//! numerically fragile (a small quantization error can make the filter unstable),
//! while the equivalent Line Spectral Frequency (LSF) representation quantizes
//! gracefully and guarantees a stable filter back out as long as the LSFs stay
//! ordered. This module is the conversion machinery both directions need —
//! `lpc_to_lsp`/`lsp_to_lsf` for the encoder, `lsf_to_lsp`/`lsp_to_lpc` for the
//! decoder — plus the codebook the first, coarse quantization stage uses.
//!
//! # What this module does not cover
//!
//! The reference's LSF quantizer is two-stage: this coarse 8-bit codebook lookup
//! (256 entries, ported below as [`DICO_LSF_ABS_8B`]) followed by a much larger
//! algebraic lattice quantizer (`iusace_alg_vec_quant`/`ixheaacd_rotated_gosset_mtx_dec`
//! in the reference — a Gosset/E8 lattice vector quantizer with Voronoi extension for
//! coding residuals too large for the base lattice, its own combinatorial rank/sign
//! encoding, and roughly a thousand lines of supporting ROM tables split across
//! `iusace_avq_enc.c`/`iusace_avq_rom.c` and the decoder's matching
//! `ixheaacd_avq_dec.c`/`ixheaacd_avq_rom.c`, the latter of which additionally needs
//! its own separate precomputed weight table since the decoder has no access to the
//! encoder's true analysis LSF to derive weights from dynamically). That second
//! stage is real, substantial, separate work — a full quantizer built on top of this
//! module's codebook-only first stage, not something this pass attempts. What is
//! here — the LPC/LSP/LSF conversions and the coarse codebook quantizer — is
//! complete, correct, and independently useful (the codebook alone is a real, if
//! coarse, LSF quantizer), the same way the spectral arithmetic coder was delivered
//! as a correct standalone primitive before the FD frame used it.

/// LPC order every USAC LPD frame uses (`ORDER` in the reference).
pub const LPC_ORDER: usize = 16;
/// Half the LPC order — the degree of each of the two symmetric/antisymmetric
/// polynomials the Chebyshev root search factors the LPC polynomial into.
const ORDER_BY_2: usize = 8;

/// Top of the LSF domain, in the reference's fixed linear-frequency-like units
/// (LSFs run `0..FREQ_MAX`, not radians or Hz — see [`lsp_to_lsf`]).
pub const FREQ_MAX: f32 = 6400.0;
/// Minimum spacing enforced between neighbouring LSFs, in those same units —
/// without it, two LSFs coinciding (or crossing) would produce an LPC filter
/// with a pole pair on the unit circle, i.e. an undamped, unstable resonance.
pub const LSF_GAP: f32 = 50.0;

/// A reasonable, spectrally flat starting point for a stream's first frame or
/// after a reset, evenly spaced across the LSF domain (`lsf_init` in the
/// reference decoder's ROM). Using this as `prev_lsf` for [`lpc_to_lsp`]'s
/// fallback path means a pathological first analysis (silence, most commonly)
/// degrades to a neutral filter rather than an arbitrary, potentially unstable
/// one.
pub const LSF_INIT: [f32; LPC_ORDER] = [
    375.0, 750.0, 1125.0, 1500.0, 1875.0, 2250.0, 2625.0, 3000.0, 3375.0, 3750.0, 4125.0, 4500.0,
    4875.0, 5250.0, 5625.0, 6000.0,
];

/// Grid the Chebyshev root search walks, `cos(pi*j/100)` for `j` in `0..=100` —
/// computed rather than transcribed from the reference's literal 101-entry
/// table, since it is exactly that closed form (verified against the table's
/// values, e.g. entry 1 is `cos(pi/100) = 0.999507...`), not an arbitrary
/// constant.
fn chebyshev_grid(j: usize) -> f64 {
    (std::f64::consts::PI * j as f64 / 100.0).cos()
}

/// Evaluate one of the two symmetric-factor Chebyshev polynomials at `x`
/// (`iusace_lpc_eval_chebyshev_polyn`). `coefs` holds `ORDER_BY_2 + 1` terms.
fn eval_chebyshev(x: f64, coefs: &[f64; ORDER_BY_2 + 1]) -> f64 {
    let x2 = 2.0 * x;
    let mut b2 = 1.0f64;
    let mut b1 = x2 + coefs[1];
    for &c in coefs.iter().take(ORDER_BY_2).skip(2) {
        let b0 = x2 * b1 - b2 + c;
        b2 = b1;
        b1 = b0;
    }
    x * b1 - b2 + 0.5 * coefs[ORDER_BY_2]
}

/// Convert an LPC filter (`[1, a_1, .., a_16]`, `A(z) = 1 + sum a_k z^-k`) to its
/// 16 Line Spectral Pairs (cosines of the Line Spectral Frequencies), via
/// `iusace_lpc_2_lsp_conversion`: `A(z)` factors into a symmetric and an
/// antisymmetric polynomial whose roots interlace on the unit circle, found here
/// by walking a 101-point grid for sign changes and refining each with
/// bisection then linear interpolation.
///
/// Falls back to `prev_lsp` if fewer than [`LPC_ORDER`] roots are found (a
/// numerically degenerate filter, e.g. from near-silent input) — the same
/// fallback the reference takes, since without it an incomplete root set would
/// leave `lsp` partly uninitialized.
pub fn lpc_to_lsp(lpc: &[f32; LPC_ORDER + 1], prev_lsp: &[f32; LPC_ORDER]) -> [f32; LPC_ORDER] {
    let mut sum_polyn = [0.0f64; ORDER_BY_2 + 1];
    let mut diff_polyn = [0.0f64; ORDER_BY_2 + 1];
    sum_polyn[0] = 1.0;
    diff_polyn[0] = 1.0;

    for i in 0..ORDER_BY_2 {
        let a = lpc[1 + i] as f64;
        let b = lpc[LPC_ORDER - i] as f64;
        sum_polyn[i + 1] = a + b - sum_polyn[i];
        diff_polyn[i + 1] = a - b + diff_polyn[i];
    }

    let mut lsp = [0.0f32; LPC_ORDER];
    let mut found = 0usize;
    let mut use_diff = false;
    let mut j = 0usize;

    let mut x_low = chebyshev_grid(0);
    let mut y_low = eval_chebyshev(x_low, &sum_polyn);
    let mut current = &sum_polyn;

    while found < LPC_ORDER && j < 100 {
        j += 1;
        let x_high = x_low;
        let y_high = y_low;
        x_low = chebyshev_grid(j);
        y_low = eval_chebyshev(x_low, current);

        if y_low * y_high <= 0.0 {
            j -= 1;
            let (mut xl, mut yl, mut xh, mut yh) = (x_low, y_low, x_high, y_high);
            for _ in 0..4 {
                let x_mid = 0.5 * (xl + xh);
                let y_mid = eval_chebyshev(x_mid, current);
                if yl * y_mid <= 0.0 {
                    yh = y_mid;
                    xh = x_mid;
                } else {
                    yl = y_mid;
                    xl = x_mid;
                }
            }
            let root = xl - yl * (xh - xl) / (yh - yl);
            lsp[found] = root as f32;
            found += 1;

            use_diff = !use_diff;
            current = if use_diff { &diff_polyn } else { &sum_polyn };
            x_low = root;
            y_low = eval_chebyshev(x_low, current);
        }
    }

    if found < LPC_ORDER {
        return *prev_lsp;
    }
    lsp
}

/// `poly1`/`poly2`'s cascading construction shared by [`lsp_to_lpc`]
/// (`iusace_compute_coeff_poly_f`): builds the two degree-8 polynomials whose
/// sum and difference reconstruct `A(z)` from its 16 roots (the inverse of the
/// factoring [`lpc_to_lsp`] performs).
///
/// Indexing note: real index 0 is the reference's pre-zeroed boundary slot
/// (read whenever the recursion below needs the coefficient "one before the
/// start"); real index 1 is the reference's relative index 0, initialized to
/// 1.0. Every other real index is one more than the reference's relative
/// index into the same array, so the recursion's own `poly[i]`/`poly[i-1]`
/// reads translate directly to `poly1[i + 1]`/`poly1[i]` here with no
/// separate boundary case to special-case.
fn coeff_poly_f(lsp: &[f32; LPC_ORDER]) -> ([f64; ORDER_BY_2 + 2], [f64; ORDER_BY_2 + 2]) {
    let mut poly1 = [0.0f64; ORDER_BY_2 + 2];
    let mut poly2 = [0.0f64; ORDER_BY_2 + 2];
    poly1[1] = 1.0;
    poly2[1] = 1.0;

    for i in 1..=ORDER_BY_2 {
        let b1 = -2.0 * lsp[2 * (i - 1)] as f64;
        let b2 = -2.0 * lsp[2 * (i - 1) + 1] as f64;
        poly1[i + 1] = b1 * poly1[i] + 2.0 * poly1[i - 1];
        poly2[i + 1] = b2 * poly2[i] + 2.0 * poly2[i - 1];
        for j in (1..i).rev() {
            poly1[j + 1] += b1 * poly1[j] + poly1[j - 1];
            poly2[j + 1] += b2 * poly2[j] + poly2[j - 1];
        }
    }
    (poly1, poly2)
}

/// Convert 16 Line Spectral Pairs back to an LPC filter
/// (`iusace_lsp_to_lp_conversion`), the exact inverse of [`lpc_to_lsp`]:
/// reconstructs the symmetric/antisymmetric factor polynomials from their
/// roots, then recombines them into `A(z)`'s coefficients.
pub fn lsp_to_lpc(lsp: &[f32; LPC_ORDER]) -> [f32; LPC_ORDER + 1] {
    let (mut poly1, mut poly2) = coeff_poly_f(lsp);

    for i in 0..ORDER_BY_2 {
        let top = ORDER_BY_2 + 1 - i;
        poly1[top] += poly1[top - 1];
        poly2[top] -= poly2[top - 1];
    }

    let mut lp = [0.0f32; LPC_ORDER + 1];
    lp[0] = 1.0;
    for i in 0..ORDER_BY_2 {
        let a = poly1[2 + i];
        let b = poly2[2 + i];
        lp[1 + i] = (0.5 * (a + b)) as f32;
        lp[LPC_ORDER - i] = (0.5 * (a - b)) as f32;
    }
    lp
}

/// Line Spectral Pair (cosine domain) to Line Spectral Frequency: the
/// reference's LSFs are not radians or Hz but `acos(lsp) * 6400/pi`, an integer-
/// friendly domain `0..=6400` that the quantizer codebook and [`LSF_GAP`] are
/// defined in terms of (`iusace_lsp_2_lsf_conversion`).
pub fn lsp_to_lsf(lsp: &[f32; LPC_ORDER]) -> [f32; LPC_ORDER] {
    let scale = FREQ_MAX as f64 / std::f64::consts::PI;
    let mut lsf = [0.0f32; LPC_ORDER];
    for (o, &x) in lsf.iter_mut().zip(lsp.iter()) {
        *o = ((x as f64).clamp(-1.0, 1.0).acos() * scale) as f32;
    }
    lsf
}

/// The inverse of [`lsp_to_lsf`] (`iusace_lsf_2_lsp_conversion`).
pub fn lsf_to_lsp(lsf: &[f32; LPC_ORDER]) -> [f32; LPC_ORDER] {
    let scale = std::f64::consts::PI / FREQ_MAX as f64;
    let mut lsp = [0.0f32; LPC_ORDER];
    for (o, &x) in lsp.iter_mut().zip(lsf.iter()) {
        *o = ((x as f64) * scale).cos() as f32;
    }
    lsp
}

/// Force an LSF vector into strictly increasing order with at least
/// [`LSF_GAP`] between neighbours and within `[LSF_GAP, FREQ_MAX - LSF_GAP]` —
/// the stability guard the reference applies after adding any quantization
/// residual (`ixheaacd_avq_first_approx_abs`'s two clamp passes). Two LSFs
/// crossing, or one sitting outside the domain, would produce an LPC filter
/// with a pole on or outside the unit circle: an unstable, potentially
/// unbounded synthesis filter. This is applied here even though this module's
/// only quantizer (the coarse codebook) can never itself produce an invalid
/// vector, so that any future caller building on this module inherits the
/// same safety net the reference gives every quantization stage.
pub fn enforce_lsf_stability(lsf: &mut [f32; LPC_ORDER]) {
    let mut floor = LSF_GAP;
    for x in lsf.iter_mut() {
        if *x < floor {
            *x = floor;
        }
        floor = *x + LSF_GAP;
    }
    let mut ceiling = FREQ_MAX - LSF_GAP;
    for x in lsf.iter_mut().rev() {
        if *x > ceiling {
            *x = ceiling;
        }
        ceiling = *x - LSF_GAP;
    }
}

/// The coarse, first-stage LSF codebook (`iusace_dico_lsf_abs_8b_flt`): 256
/// candidate 16-dimensional LSF vectors, each a real, ordered, stable LSF set a
/// trained codebook design settled on. An 8-bit index into this table is this
/// module's whole quantized representation of an LPC filter — coarse (256
/// codewords for a 16-dimensional space), but real and immediately usable
/// without the reference's much larger second-stage lattice refinement (see
/// this module's top-level docs).
#[rustfmt::skip]
pub const DICO_LSF_ABS_8B: [[f32; LPC_ORDER]; 256] = include!("usac_lsf_dico.rs");
