//! RE8 (Gosset) lattice vector decode: `rotated_gosset_mtx_dec`, the shared
//! primitive [`crate::tables::usac_avq`]'s tables exist to serve.
//!
//! Ported from `c/libxaac/decoder/ixheaacd_avq_dec.c`. See
//! [`crate::tables::usac_avq`]'s module docs for the algorithm in plain terms
//! (absolute leader classes, sign leaders, permutation ranking); this module
//! is the decode side of that: transmitted index in, 8-dimensional lattice
//! point out.
//!
//! Two callers in the reference need this: FAC coefficient decode
//! (`ixheaacd_acelp_bitparse.c`, still open — see `text/plan.txt` phase 1.7)
//! and USAC's LSF second-stage residual refinement (`ixheaacd_avq_dec.c`
//! itself, called from the LPD LSF path — also not yet wired to a caller).
//! Landing the primitive itself first, verified on its own, is what makes
//! both of those a smaller, safer step afterward.
//!
//! # What is ported, and what is not
//!
//! [`decode_base_index`] and [`rank_of_permutation`] cover every index with
//! `qn <= 4` — the base RE8 lattice, no extension needed. This is already a
//! real, independently useful, and independently tested primitive.
//! `qn > 4` additionally needs the outer *Voronoi extension* (nearest-neighbor
//! rounding plus a two-candidate search, `ixheaacd_nearest_neighbor_2d` /
//! `ixheaacd_voronoi_search` / `ixheaacd_voronoi_idx_dec`) for residuals too
//! large for the base lattice; that layer is not ported yet, and
//! [`rotated_gosset_mtx_dec`] returns [`crate::error::Error::Unimplemented`]
//! for `qn > 4` rather than guessing.

use crate::error::{Error, Result};
use crate::tables::usac_avq::{
    ABSOLUTE_LEADER_TAB, CARDINALITY_OFFSET_TABLE_I3, CARDINALITY_OFFSET_TAB_I4, FACTORIAL_TABLE,
    ISO_CODE_DATA_TABLE, ISO_CODE_INDEX_TABLE, ISO_CODE_NUM_TABLE, POS_ABS_LEADERS_A3,
    POS_ABS_LEADERS_A4, SIGNED_LEADER_IS,
};

/// Find the bucket a code index falls into within a table of cumulative
/// index-space boundaries (`ixheaacd_get_abs_leader_tbl`): the largest `i`
/// such that `table[i] <= code_book_ind`, searched four entries at a time
/// (matching the reference's unrolled stride, not merely its result, since
/// the exact stopping point is what the caller's `+ 4` step assumes).
fn get_abs_leader_tbl(table: &[u32], code_book_ind: u32) -> usize {
    let size = table.len();
    let mut i = 4usize;
    while i < size {
        if code_book_ind < table[i] {
            break;
        }
        i += 4;
    }
    if i > size {
        i = size;
    }
    if code_book_ind < table[i - 2] {
        i -= 2;
    }
    if code_book_ind < table[i - 1] {
        i -= 1;
    }
    i - 1
}

/// Un-rank a permutation of a multiset (`ixheaacd_gosset_rank_of_permutation`):
/// `xs` comes in holding the class template's values in descending order
/// (ties grouped together, as every [`ABSOLUTE_LEADER_TAB`] row is), and
/// `rank` selects which of the `8! / (multiplicities!)` distinct orderings to
/// produce, written back into `xs` in place.
///
/// This is the standard factorial-number-system technique for ranking
/// permutations, adapted for repeated values: at each output position, `rank`
/// (scaled by how many arrangements the *remaining* multiset has) is reduced
/// against the weighted count of arrangements starting with each remaining
/// distinct value, in descending order, until it selects one.
fn rank_of_permutation(rank: i64, xs: &mut [i32; 8]) {
    let mut a = [0i32; 8];
    let mut w = [0i64; 8];
    let mut j = 0usize;
    w[0] = 1;
    a[0] = xs[0];
    let mut base: i64 = 1;
    for i in 1..8 {
        if xs[i] != xs[i - 1] {
            j += 1;
            w[j] = 1;
            a[j] = xs[i];
        } else {
            w[j] += 1;
            base *= w[j];
        }
    }

    if w[0] == 8 {
        xs.fill(a[0]);
        return;
    }

    let mut target = rank * base;
    let mut fac_b: i64 = 1;
    for i in 0..8 {
        let fac = fac_b * FACTORIAL_TABLE[i];
        let mut jj: usize = 0;
        loop {
            target -= w[jj] * fac;
            if target < 0 {
                break;
            }
            jj += 1;
        }
        xs[i] = a[jj];
        target += w[jj] * fac;
        fac_b *= w[jj];
        w[jj] -= 1;
    }
}

/// Decode a base (non-Voronoi-extended) RE8 point: `n` is the reduced pulse
/// count (2, 3 or 4 -- 0 and 1 both mean "the zero point", handled directly),
/// and `code_book_ind` is the transmitted index within that count's index
/// space (`ixheaacd_gosset_decode_base_index`).
///
/// The three steps mirror the three things the index space is built from,
/// outermost first: which absolute leader class (found via the cumulative
/// boundary tables, [`CARDINALITY_OFFSET_TABLE_I3`]/[`CARDINALITY_OFFSET_TAB_I4`]),
/// then which sign pattern within that class (the same kind of boundary
/// search, over the class's own slice of [`SIGNED_LEADER_IS`]), then which
/// permutation of the signed template ([`rank_of_permutation`]) the remainder
/// of the index names.
pub fn decode_base_index(n: i32, code_book_ind: u32) -> [i32; 8] {
    if n < 2 {
        return [0; 8];
    }

    let idx = match n {
        2 | 3 => {
            let i = get_abs_leader_tbl(&CARDINALITY_OFFSET_TABLE_I3, code_book_ind);
            POS_ABS_LEADERS_A3[i]
        }
        _ => {
            let i = get_abs_leader_tbl(&CARDINALITY_OFFSET_TAB_I4, code_book_ind);
            POS_ABS_LEADERS_A4[i]
        }
    };
    debug_assert!(n == 4 || n == 2 || n == 3, "n must be 0..=4; qn > 4 needs Voronoi extension");

    let mut ya = [0i32; 8];
    for (slot, &v) in ya.iter_mut().zip(ABSOLUTE_LEADER_TAB[idx].iter()) {
        *slot = v as i32;
    }

    let t = ISO_CODE_INDEX_TABLE[idx];
    let im = ISO_CODE_NUM_TABLE[idx];
    let ks = get_abs_leader_tbl(&SIGNED_LEADER_IS[t..t + im], code_book_ind);

    let mut sign_code: u32 = 2 * ISO_CODE_DATA_TABLE[t + ks] as u32;
    for slot in ya.iter_mut().rev() {
        *slot *= 1 - (sign_code & 2) as i32;
        sign_code >>= 1;
    }

    let rank = code_book_ind as i64 - SIGNED_LEADER_IS[t + ks] as i64;
    rank_of_permutation(rank, &mut ya);
    ya
}

/// Decode one transmitted algebraic-vector index into its 8-dimensional
/// lattice point (`ixheaacd_rotated_gosset_mtx_dec`).
///
/// `qn` is the pulse-count parameter the bitstream carries alongside the
/// index (its unary-coded prefix in both the FAC and LSF-refinement callers);
/// `kv` is the accompanying per-dimension sub-index the Voronoi-extension path
/// needs for `qn > 4` (unused, and may be all zero, for `qn <= 4`).
pub fn rotated_gosset_mtx_dec(qn: i32, code_book_idx: u32, _kv: &[i32; 8]) -> Result<[i32; 8]> {
    if qn <= 4 {
        Ok(decode_base_index(qn, code_book_idx))
    } else {
        Err(Error::Unimplemented {
            tool: "RE8 lattice Voronoi extension (qn > 4)",
            detail: "text/plan.txt phase 1.7/6; ixheaacd_voronoi_idx_dec not ported yet",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The zero point must decode to the zero point for both `n` values the
    /// reference treats as trivial.
    #[test]
    fn n_below_2_always_decodes_to_the_zero_point() {
        assert_eq!(decode_base_index(0, 0), [0; 8]);
        assert_eq!(decode_base_index(1, 12345), [0; 8]);
    }

    /// [`rank_of_permutation`] must be a real bijection: every rank in
    /// `0..distinct_permutations` must produce a distinct output, and every
    /// output must be some permutation of the input multiset -- checked
    /// directly against a brute-force list of that multiset's permutations,
    /// independent of the real RE8 tables.
    #[test]
    fn rank_of_permutation_is_a_bijection_over_a_synthetic_multiset() {
        // [3, 1, 1, 1, 1, 1, 1, 1] has 8!/7! = 8 distinct permutations.
        let template = [3, 1, 1, 1, 1, 1, 1, 1];
        let mut seen = std::collections::HashSet::new();
        for rank in 0..8 {
            let mut xs = template;
            rank_of_permutation(rank, &mut xs);
            let mut sorted = xs;
            sorted.sort_unstable();
            let mut want_sorted = template;
            want_sorted.sort_unstable();
            assert_eq!(sorted, want_sorted, "rank {rank} is not a permutation of the input");
            assert!(seen.insert(xs), "rank {rank} collided with an earlier rank: {xs:?}");
        }
    }

    /// The same check on a multiset with a repeated non-trivial value: `[2, 2,
    /// 1, 1, 1, 1, 1, 1]` has 8!/(2!6!) = 28 distinct permutations.
    #[test]
    fn rank_of_permutation_handles_repeated_values_correctly() {
        let template = [2, 2, 1, 1, 1, 1, 1, 1];
        let mut seen = std::collections::HashSet::new();
        for rank in 0..28 {
            let mut xs = template;
            rank_of_permutation(rank, &mut xs);
            seen.insert(xs);
        }
        assert_eq!(seen.len(), 28, "expected all 28 distinct permutations to be reachable");
    }

    /// An all-equal multiset (the `w[0] == 8` shortcut) must decode to itself
    /// regardless of rank -- there is only one permutation.
    #[test]
    fn an_all_equal_template_ignores_the_rank() {
        for rank in [0i64, 1, 100] {
            let mut xs = [1i32; 8];
            rank_of_permutation(rank, &mut xs);
            assert_eq!(xs, [1; 8]);
        }
    }

    /// Every decoded base-index point must be a signed permutation of a real
    /// [`ABSOLUTE_LEADER_TAB`] row: same multiset of absolute values. This is
    /// checked across every class's full index range rather than a few
    /// samples, so a boundary error in [`get_abs_leader_tbl`] cannot hide
    /// between spot checks.
    #[test]
    fn every_decoded_point_is_a_signed_permutation_of_a_real_leader_row() {
        for n in [2, 3, 4] {
            let top = if n == 4 {
                *CARDINALITY_OFFSET_TAB_I4.last().unwrap()
            } else {
                *CARDINALITY_OFFSET_TABLE_I3.last().unwrap()
            };
            for code in 0..top {
                let y = decode_base_index(n, code);
                let mut abs_sorted: Vec<i32> = y.iter().map(|v| v.abs()).collect();
                abs_sorted.sort_unstable();

                let matches_some_row = ABSOLUTE_LEADER_TAB.iter().any(|row| {
                    let mut row_sorted: Vec<i32> = row.iter().map(|&v| v as i32).collect();
                    row_sorted.sort_unstable();
                    row_sorted == abs_sorted
                });
                assert!(matches_some_row, "n={n} code={code}: {y:?} matches no leader row");
            }
        }
    }

    /// `rotated_gosset_mtx_dec` must dispatch qn<=4 to the base decode and
    /// refuse qn>4 outright rather than silently returning a wrong point.
    #[test]
    fn dispatch_refuses_qn_above_four_rather_than_guessing() {
        let kv = [0i32; 8];
        assert!(rotated_gosset_mtx_dec(4, 0, &kv).is_ok());
        assert!(matches!(
            rotated_gosset_mtx_dec(5, 0, &kv),
            Err(Error::Unimplemented { .. })
        ));
    }
}
