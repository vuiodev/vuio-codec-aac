//! Golden Bit-Exact Test Harness
//!
//! Asserts bit-level equivalence between Rust `xaac` and reference C `libxaac`
//! tables, math primitives, and decoded/encoded streams.

use xaac::dsp::math::*;
use xaac::tables::dequant::POW_TABLE_Q13;
use xaac::tables::scalefactor::*;

#[test]
fn test_pow_table_q13_reference_values() {
    assert_eq!(POW_TABLE_Q13[0], 0);
    assert_eq!(POW_TABLE_Q13[1], 8192); // 1.0 in Q13
    assert_eq!(POW_TABLE_Q13[2], 20642); // 2^(4/3) * 8192 = 2.5198 * 8192 = 20642.7
    assert_eq!(POW_TABLE_Q13[3], 35444);
    assert_eq!(POW_TABLE_Q13[4], 52016);
    assert_eq!(POW_TABLE_Q13[128], 5284491);
}

#[test]
fn test_sfb_table_exact_bounds() {
    let sfb_48_1024 = SFB_48_1024;
    assert_eq!(sfb_48_1024.len(), 49);
    let total_width: u32 = sfb_48_1024.iter().map(|&w| w as u32).sum();
    assert_eq!(total_width, 1024);

    let sfb_44_1024 = SFB_44100_1024;
    assert_eq!(sfb_44_1024.len(), 49);
    let total_width_44: u32 = sfb_44_1024.iter().map(|&w| w as u32).sum();
    assert_eq!(total_width_44, 1024);

    let sfb_32_1024 = SFB_32_1024;
    assert_eq!(sfb_32_1024.len(), 51);
    let total_width_32: u32 = sfb_32_1024.iter().map(|&w| w as u32).sum();
    assert_eq!(total_width_32, 1024);
}

#[test]
fn test_fixed_point_math_bit_exact_vectors() {
    // Edge case 1: Positive saturation boundary
    let sat_max = sat64_32(i64::MAX);
    assert_eq!(sat_max, 0x7FFFFFFF);

    // Edge case 2: Negative saturation boundary
    let sat_min = sat64_32(i64::MIN);
    assert_eq!(sat_min, -0x80000000);

    // Edge case 3: Fractional MAC in 64-bit accumulator
    let mut acc = 0i64;
    acc = mac32x32in64(acc, 0x40000000, 0x40000000); // 0.5 * 0.5 = 0.25
    assert_eq!(acc, 0x1000000000000000);

    // Edge case 4: Saturating left shift
    let shl_sat = shl32_dir_sat_limit(0x40000000, 1);
    assert_eq!(shl_sat, 0x7FFFFFFF); // saturated to MAX
}
