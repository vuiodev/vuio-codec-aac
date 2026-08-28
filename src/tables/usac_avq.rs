//! Tables for the RE8 (Gosset) lattice vector quantizer that refines USAC's
//! coarse LSF codebook match, ported from `c/libxaac/encoder/iusace_avq_enc.c` /
//! `iusace_avq_rom.c` and `c/libxaac/decoder/ixheaacd_avq_dec.c` / `ixheaacd_avq_rom.c`.
//!
//! # The idea in plain terms
//!
//! [`crate::encoder::usac::lsf::quantize_lsf_abs`] picks the nearest of 256 trained
//! codewords for an LSF vector — coarse, since 256 codewords cannot cover a
//! 16-dimensional space finely. This module quantizes what that first pass leaves
//! over: the *residual* (true LSF minus the coarse codeword, weighted so
//! perceptually important directions count more).
//!
//! The residual is coded 8 dimensions at a time (twice, for the 16-dimensional LSF
//! vector) against the **RE8 lattice**, a specific, dense packing of points in
//! 8-space (also called the Gosset lattice) — the nearest RE8 point to the residual
//! is found directly by rounding, then that point is identified by an index
//! instead of transmitting all 8 coordinates:
//!
//! * Every RE8 point belongs to one of 37 **absolute leader classes** — a class is
//!   the set of points that are permutations-with-signs of one canonical
//!   nonnegative, sorted template vector (e.g. every permutation and sign choice of
//!   `[3, 1, 1, 1, 1, 1, 1, 1]` is one class), listed here as [`ABSOLUTE_LEADER_TAB`].
//! * Within a class, [`crate::decoder::usac::avq::rank_of_permutation`] is a
//!   combinatorial bijection between "which permutation of the template, with
//!   which signs" and a single integer — the classic technique of ranking
//!   permutations of a multiset, extended to also rank the sign pattern via
//!   [`ISO_CODE_DATA_TABLE`] (not every sign pattern is distinct once permutations
//!   are accounted for, so this table lists only the *representative* sign
//!   patterns — "sign leaders" — each class actually needs).
//! * [`SIGNED_LEADER_IS`] gives each (absolute leader, sign leader) pair a
//!   contiguous block of index space, so "leader class + rank + sign" collapses to
//!   one final integer: the transmitted index.
//!
//! A residual too large for the base lattice (its nearest point falls in a class
//! needing more than 4 extra bits) is coded instead by an outer *Voronoi extension*:
//! finding a scaled-down, coarser version of the same lattice that does contain it,
//! coding the difference against the base lattice, and transmitting which coarse
//! cell it fell into (`ixheaacd_voronoi_idx_dec` in the reference).
//!
//! # What is ported, here and in `decoder::usac::avq`, and what is not
//!
//! This file is tables only; the decode algorithm itself —
//! [`crate::decoder::usac::avq::decode_base_index`] and
//! [`crate::decoder::usac::avq::rotated_gosset_mtx_dec`] — lives in
//! `decoder::usac::avq`, ported from `ixheaacd_avq_dec.c`, and covers every
//! index with `qn <= 4` (the base lattice, no extension needed), with its own
//! tests verifying every decoded point is a genuine signed permutation of one
//! of the 37 [`ABSOLUTE_LEADER_TAB`] rows across each class's full index
//! range. The Voronoi extension for `qn > 4`, and the whole *encode* direction
//! (nearest-lattice-point search, rank computation, mode selection across a
//! superframe in `iusace_quantize_lpc_avq`) are not ported — see
//! `text/plan.txt` phases 1.7/3.6/6 for what remains, and
//! `decoder::usac::avq`'s own module docs for exactly where the ported half
//! stops.

/// Squared-norm ceiling (`>> 3`, i.e. sum of squares divided by 8) up to which
/// [`find_absolute_leader`] can classify a point directly from [`DA_ID`]; beyond
/// this a point needs Voronoi extension first.
pub const NB_SPHERE: usize = 32;
/// Absolute leader classes RE8 points are classified into (`LEN_ABS_LEADER`),
/// indices `0..37`; index `37` itself denotes the all-zero point and index `38`
/// (used only as `DA_NQ`'s last entry) is the "too large, needs Voronoi
/// extension" sentinel — see [`DA_NQ`].
pub const LEN_ABS_LEADER: usize = 37;
/// Total sign-leader entries across every absolute leader class (`LEN_SIGN_LEADER`).
pub const LEN_SIGN_LEADER: usize = 226;
/// Cardinality classes for the decoder's `n = 2` and `n = 3` extra-bit cases,
/// binned together (`LEN_I3`).
pub const LEN_I3: usize = 9;
/// Cardinality classes for the decoder's `n = 4` extra-bit case (`LEN_I4`).
pub const LEN_I4: usize = 28;

/// `2^i` for `i` in `7..=0` descending — used to fold 8 sign bits into one
/// integer, most-significant coordinate first (`iusace_pow2_table`).
pub const POW2_TABLE: [i32; 8] = [128, 64, 32, 16, 8, 4, 2, 1];
/// `(7-i)!` for `i` in `0..8` (`iusace_factorial_table` / `ixheaacd_factorial_7` —
/// identical between encoder and decoder, confirmed against both sources).
pub const FACTORIAL_TABLE: [i64; 8] = [5040, 720, 120, 24, 6, 2, 1, 1];
/// Perceptual-weighting divisor per refinement mode; mode `0` (`60.0`) is the only
/// one this module's single-vector, non-predictive path uses (`iusace_wlsf_factor_table`).
pub const WLSF_FACTOR_TABLE: [f32; 4] = [60.0, 65.0, 64.0, 63.0];

/// Index into [`DA_ID`]/[`DA_NUM_BITS`] where each squared-norm class `s` (`1..=32`,
/// so `DA_POS[s - 1]`) starts (`iusace_da_pos`).
pub const DA_POS: [usize; NB_SPHERE] = [
    0, 2, 5, 8, 13, 18, 20, 22, 23, 25, 26, 27, 27, 28, 28, 28, 29, 30, 31, 31, 32, 32, 32, 32, 32,
    34, 35, 35, 35, 35, 35, 35,
];
/// How many distinct absolute leader classes share squared-norm class `s`
/// (`iusace_da_num_bits`; the name is the reference's, not a bit count here).
pub const DA_NUM_BITS: [usize; NB_SPHERE] = [
    2, 3, 3, 5, 5, 2, 2, 1, 2, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 1,
];
/// Extra bits (beyond the base rank/sign index) each absolute leader class needs;
/// index `37` (all-zero point) needs none, and index `38` is the sentinel
/// [`find_absolute_leader`] returns for a norm too large to classify directly —
/// `100` is deliberately far past the `> 4` threshold that triggers Voronoi
/// extension, not a real bit count (`iusace_da_nq`).
pub const DA_NQ: [i32; LEN_ABS_LEADER + 2] = [
    2, 2, 3, 3, 2, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 0, 100,
];
/// Fourth-power-sum identifier (`sum(y_i^4) >> 3`) that disambiguates the leader
/// classes sharing a squared-norm class, searched via [`DA_POS`]/[`DA_NUM_BITS`]
/// (`iusace_da_id`).
pub const DA_ID: [u32; LEN_ABS_LEADER] = [
    0x0001, 0x0004, 0x0008, 0x000B, 0x0020, 0x000C, 0x0015, 0x0024, 0x0010, 0x001F, 0x0028, 0x0040,
    0x004F, 0x0029, 0x002C, 0x0044, 0x0059, 0x00A4, 0x0060, 0x00A8, 0x00C4, 0x012D, 0x0200, 0x0144,
    0x0204, 0x0220, 0x0335, 0x04E4, 0x0400, 0x0584, 0x0A20, 0x0A40, 0x09C4, 0x12C4, 0x0C20, 0x2000,
    0x4E20,
];

/// The 37 absolute leader classes' canonical (nonnegative, descending)
/// coordinate templates (`ixheaacd_absolute_leader_tab_da`), only needed by the
/// decoder: unlike the encoder, which derives a class from an actual point it
/// already has, the decoder must look the template up from the transmitted index
/// alone.
pub const ABSOLUTE_LEADER_TAB: [[u8; 8]; LEN_ABS_LEADER] = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [2, 2, 0, 0, 0, 0, 0, 0],
    [2, 2, 2, 2, 0, 0, 0, 0],
    [3, 1, 1, 1, 1, 1, 1, 1],
    [4, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 2, 2, 2, 2, 0, 0],
    [3, 3, 1, 1, 1, 1, 1, 1],
    [4, 2, 2, 0, 0, 0, 0, 0],
    [2, 2, 2, 2, 2, 2, 2, 2],
    [3, 3, 3, 1, 1, 1, 1, 1],
    [4, 2, 2, 2, 2, 0, 0, 0],
    [4, 4, 0, 0, 0, 0, 0, 0],
    [5, 1, 1, 1, 1, 1, 1, 1],
    [3, 3, 3, 3, 1, 1, 1, 1],
    [4, 2, 2, 2, 2, 2, 2, 0],
    [4, 4, 2, 2, 0, 0, 0, 0],
    [5, 3, 1, 1, 1, 1, 1, 1],
    [6, 2, 0, 0, 0, 0, 0, 0],
    [4, 4, 4, 0, 0, 0, 0, 0],
    [6, 2, 2, 2, 0, 0, 0, 0],
    [6, 4, 2, 0, 0, 0, 0, 0],
    [7, 1, 1, 1, 1, 1, 1, 1],
    [8, 0, 0, 0, 0, 0, 0, 0],
    [6, 6, 0, 0, 0, 0, 0, 0],
    [8, 2, 2, 0, 0, 0, 0, 0],
    [8, 4, 0, 0, 0, 0, 0, 0],
    [9, 1, 1, 1, 1, 1, 1, 1],
    [10, 2, 0, 0, 0, 0, 0, 0],
    [8, 8, 0, 0, 0, 0, 0, 0],
    [10, 6, 0, 0, 0, 0, 0, 0],
    [12, 0, 0, 0, 0, 0, 0, 0],
    [12, 4, 0, 0, 0, 0, 0, 0],
    [10, 10, 0, 0, 0, 0, 0, 0],
    [14, 2, 0, 0, 0, 0, 0, 0],
    [12, 8, 0, 0, 0, 0, 0, 0],
    [16, 0, 0, 0, 0, 0, 0, 0],
    [20, 0, 0, 0, 0, 0, 0, 0],
];

/// How many distinct sign patterns ("sign leaders") each absolute leader class
/// needs (`ixheaacd_iso_code_num_table`; decoder-only, the encoder computes this
/// implicitly by searching [`ISO_CODE_DATA_TABLE`] directly).
pub const ISO_CODE_NUM_TABLE: [usize; LEN_ABS_LEADER] = [
    5, 3, 5, 8, 2, 7, 11, 6, 9, 12, 10, 3, 8, 13, 14, 9, 14, 4, 4, 8, 8, 8, 2, 3, 6, 4, 8, 4, 3, 4,
    2, 4, 3, 4, 4, 2, 2,
];
/// Start offset into [`ISO_CODE_DATA_TABLE`]/[`SIGNED_LEADER_IS`] for each
/// absolute leader class's sign leaders (`iusace_iso_code_index_table` /
/// `ixheaacd_iso_code_index_table` — identical between encoder and decoder,
/// confirmed against both sources).
pub const ISO_CODE_INDEX_TABLE: [usize; LEN_ABS_LEADER] = [
    0, 5, 8, 13, 21, 23, 30, 41, 47, 56, 68, 78, 81, 89, 102, 116, 125, 139, 143, 147, 155, 163,
    171, 173, 176, 182, 186, 194, 198, 201, 205, 207, 211, 214, 218, 222, 224,
];
/// Every sign leader's sign pattern, packed as an 8-bit code (`iusace_iso_code_data_table`
/// / `ixheaacd_iso_code_data_table` — identical between encoder and decoder).
pub const ISO_CODE_DATA_TABLE: [u8; LEN_SIGN_LEADER] = [
    0, 3, 15, 63, 255, 0, 64, 192, 0, 16, 48, 112, 240, 1, 7, 31, 127, 128, 131, 143, 191, 0, 128,
    0, 4, 12, 28, 60, 124, 252, 0, 3, 15, 63, 65, 71, 95, 192, 195, 207, 255, 0, 32, 96, 128, 160,
    224, 0, 1, 3, 7, 15, 31, 63, 127, 255, 1, 7, 31, 32, 35, 47, 97, 103, 127, 224, 227, 239, 0, 8,
    24, 56, 120, 128, 136, 152, 184, 248, 0, 64, 192, 0, 3, 15, 63, 129, 135, 159, 255, 0, 3, 15,
    17, 23, 48, 51, 63, 113, 119, 240, 243, 255, 0, 2, 6, 14, 30, 62, 126, 128, 130, 134, 142, 158,
    190, 254, 0, 16, 48, 64, 80, 112, 192, 208, 240, 1, 7, 31, 64, 67, 79, 127, 128, 131, 143, 191,
    193, 199, 223, 0, 64, 128, 192, 0, 32, 96, 224, 0, 16, 48, 112, 128, 144, 176, 240, 0, 32, 64,
    96, 128, 160, 192, 224, 1, 7, 31, 127, 128, 131, 143, 191, 0, 128, 0, 64, 192, 0, 32, 96, 128,
    160, 224, 0, 64, 128, 192, 0, 3, 15, 63, 129, 135, 159, 255, 0, 64, 128, 192, 0, 64, 192, 0, 64,
    128, 192, 0, 128, 0, 64, 128, 192, 0, 64, 192, 0, 64, 128, 192, 0, 64, 128, 192, 0, 128, 0, 128,
];
/// Base transmitted-index offset for each (absolute leader, sign leader) pair —
/// adding a within-class permutation rank to the matching entry here gives the
/// final index (`iusace_signed_leader_is` / `ixheaacd_signed_leader_is` —
/// identical between encoder and decoder).
pub const SIGNED_LEADER_IS: [u32; LEN_SIGN_LEADER] = [
    0, 1, 29, 99, 127, 128, 156, 212, 256, 326, 606, 1026, 1306, 1376, 1432, 1712, 1880, 1888, 1896,
    2064, 2344, 240, 248, 0, 28, 196, 616, 1176, 1596, 1764, 1792, 1820, 2240, 2660, 2688, 3024,
    4144, 4480, 4508, 4928, 5348, 2400, 2568, 2904, 3072, 3240, 3576, 5376, 5377, 5385, 5413, 5469,
    5539, 5595, 5623, 5631, 5632, 5912, 6472, 6528, 6696, 8376, 9216, 10056, 11736, 11904, 11960,
    12520, 12800, 13080, 14200, 15880, 17000, 17280, 17560, 18680, 20360, 21480, 3744, 3772, 3828,
    21760, 21768, 21936, 22216, 22272, 22328, 22608, 22776, 22784, 22854, 23274, 23344, 24464,
    25584, 26004, 28524, 28944, 30064, 31184, 31254, 31674, 31744, 31800, 32136, 32976, 34096,
    34936, 35272, 35328, 35384, 35720, 36560, 37680, 38520, 38856, 38912, 39332, 40172, 40592,
    41432, 43112, 43952, 44372, 45212, 45632, 45968, 47088, 47424, 47480, 48320, 49160, 49216,
    49272, 50112, 50952, 51008, 51344, 52464, 3856, 3912, 3968, 4024, 52800, 52856, 53024, 53192,
    53248, 53528, 54368, 55208, 55488, 55768, 56608, 57448, 57728, 58064, 58400, 58736, 59072,
    59408, 59744, 60080, 60416, 60472, 60752, 60920, 60928, 60936, 61104, 61384, 4080, 4088, 61440,
    61468, 61524, 61552, 61720, 62056, 62224, 62392, 62728, 62896, 62952, 63008, 63064, 63120,
    63128, 63296, 63576, 63632, 63688, 63968, 64136, 64144, 64200, 64256, 64312, 64368, 64396,
    64452, 64480, 64536, 64592, 64648, 64704, 64712, 64720, 64776, 64832, 64888, 64944, 64972,
    65028, 65056, 65112, 65168, 65224, 65280, 65336, 65392, 65448, 65504, 65512, 65520, 65528,
];

/// Decoder-only: which absolute leader class each `n = 2`/`n = 3` codebook-index
/// bucket names (`ixheaacd_pos_abs_leaders_a3`).
pub const POS_ABS_LEADERS_A3: [usize; LEN_I3] = [0, 1, 4, 2, 3, 7, 11, 17, 22];
/// Decoder-only: the same for `n = 4` (`ixheaacd_pos_abs_leaders_a4`).
pub const POS_ABS_LEADERS_A4: [usize; LEN_I4] = [
    5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    34, 35, 36,
];
/// Decoder-only: cumulative index-space boundary each `n = 2`/`n = 3` bucket
/// starts at, searched by [`crate::decoder::usac::lsf::get_abs_leader_tbl`]
/// (`ixheaacd_cardinality_offset_table_i3`).
pub const CARDINALITY_OFFSET_TABLE_I3: [u32; LEN_I3] = [0, 128, 240, 256, 1376, 2400, 3744, 3856, 4080];
/// Decoder-only: the same for `n = 4` (`ixheaacd_cardinality_offset_tab_i4`).
pub const CARDINALITY_OFFSET_TAB_I4: [u32; LEN_I4] = [
    0, 1792, 5376, 5632, 12800, 21760, 22784, 31744, 38912, 45632, 52800, 53248, 57728, 60416,
    61440, 61552, 62896, 63120, 64144, 64368, 64480, 64704, 64720, 64944, 65056, 65280, 65504,
    65520,
];
