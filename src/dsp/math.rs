//! Exact Fixed-Point Q-Format DSP Arithmetic Engine
//!
//! Provides bit-exact saturating, normalized, and fractional arithmetic identical
//! to MPEG `libxaac` for sample-accurate bit-perfect decoding and encoding.

pub const MAX_32: i32 = i32::MAX; // 0x7FFF_FFFF
pub const MIN_32: i32 = i32::MIN; // -0x8000_0000
pub const MAX_16: i16 = i16::MAX; // 0x7FFF
pub const MIN_16: i16 = i16::MIN; // -0x8000
pub const MAX_64: i64 = i64::MAX;
pub const MIN_64: i64 = i64::MIN;

#[inline(always)]
pub const fn min32(a: i32, b: i32) -> i32 {
    if a < b { a } else { b }
}

#[inline(always)]
pub const fn max32(a: i32, b: i32) -> i32 {
    if a > b { a } else { b }
}

#[inline(always)]
pub const fn extract16h(var: i32) -> i16 {
    (var >> 16) as i16
}

#[inline(always)]
pub const fn extract16l(var: i32) -> i16 {
    var as i16
}

#[inline(always)]
pub const fn deposit16h_in32(var: i16) -> i32 {
    (var as i32) << 16
}

#[inline(always)]
pub const fn deposit16l_in32(var: i16) -> i32 {
    var as i32
}

#[inline(always)]
pub const fn shl32(a: i32, b: i32) -> i32 {
    let shift = (b << 24) as u32 >> 24;
    if shift > 31 {
        0
    } else {
        a << shift
    }
}

#[inline(always)]
pub const fn shr32(a: i32, b: i32) -> i32 {
    let shift = (b << 24) as u32 >> 24;
    if shift >= 31 {
        if a < 0 { -1 } else { 0 }
    } else {
        a >> shift
    }
}

#[inline(always)]
pub const fn shl32_sat(a: i32, b: i32) -> i32 {
    if b <= 0 {
        return a;
    }
    if b >= 31 {
        return if a > 0 {
            MAX_32
        } else if a < 0 {
            MIN_32
        } else {
            0
        };
    }
    if a > (MAX_32 >> b) {
        MAX_32
    } else if a < (MIN_32 >> b) {
        MIN_32
    } else {
        a << b
    }
}

#[inline(always)]
pub const fn shl32_dir(a: i32, b: i32) -> i32 {
    if b < 0 {
        shr32(a, -b)
    } else {
        shl32(a, b)
    }
}

#[inline(always)]
pub const fn shl32_dir_sat(a: i32, b: i32) -> i32 {
    if b < 0 {
        shr32(a, -b)
    } else {
        shl32_sat(a, b)
    }
}

#[inline(always)]
pub const fn shr32_dir(a: i32, b: i32) -> i32 {
    if b < 0 {
        shl32(a, -b)
    } else {
        shr32(a, b)
    }
}

#[inline(always)]
pub const fn shr32_dir_sat(a: i32, b: i32) -> i32 {
    if b < 0 {
        shl32_sat(a, -b)
    } else {
        shr32(a, b)
    }
}

#[inline(always)]
pub const fn shr32_dir_sat_limit(a: i32, b: i32) -> i32 {
    if b < 0 {
        shl32_sat(a, -b)
    } else {
        let b = min32(b, 31);
        shr32(a, b)
    }
}

#[inline(always)]
pub const fn shl32_dir_sat_limit(a: i32, b: i32) -> i32 {
    if b < 0 {
        let b = min32(-b, 31);
        shr32(a, b)
    } else {
        shl32_sat(a, b)
    }
}

#[inline(always)]
pub const fn add32(a: i32, b: i32) -> i32 {
    a.wrapping_add(b)
}

#[inline(always)]
pub const fn sub32(a: i32, b: i32) -> i32 {
    a.wrapping_sub(b)
}

#[inline(always)]
pub const fn add32_sat(a: i32, b: i32) -> i32 {
    a.saturating_add(b)
}

#[inline(always)]
pub const fn add32_sat3(a: i32, b: i32, c: i32) -> i32 {
    let sum = (a as i64) + (b as i64) + (c as i64);
    if sum > MAX_32 as i64 {
        MAX_32
    } else if sum < MIN_32 as i64 {
        MIN_32
    } else {
        sum as i32
    }
}

#[inline(always)]
pub const fn sub32_sat(a: i32, b: i32) -> i32 {
    a.saturating_sub(b)
}

#[inline(always)]
pub const fn norm32(mut a: i32) -> i32 {
    if a == 0 || a == -1 {
        31
    } else {
        if a < 0 {
            a = !a;
        }
        let mut norm = 0;
        while a < 0x4000_0000 {
            a <<= 1;
            norm += 1;
        }
        norm
    }
}

#[inline(always)]
pub const fn pnorm32(mut a: i32) -> i32 {
    if a == 0 {
        31
    } else {
        let mut norm = 0;
        while a < 0x4000_0000 {
            a <<= 1;
            norm += 1;
        }
        norm
    }
}

#[inline(always)]
pub const fn abs32(a: i32) -> i32 {
    if a < 0 { -a } else { a }
}

#[inline(always)]
pub const fn abs32_sat(a: i32) -> i32 {
    if a == MIN_32 {
        MAX_32
    } else if a < 0 {
        -a
    } else {
        a
    }
}

#[inline(always)]
pub const fn negate32(a: i32) -> i32 {
    -a
}

#[inline(always)]
pub const fn negate32_sat(a: i32) -> i32 {
    if a == MIN_32 {
        MAX_32
    } else {
        -a
    }
}

#[inline(always)]
pub const fn mult16x16in32(a: i16, b: i16) -> i32 {
    (a as i32) * (b as i32)
}

#[inline(always)]
pub const fn mult16x16in32_shl(a: i16, b: i16) -> i32 {
    shl32(mult16x16in32(a, b), 1)
}

#[inline(always)]
pub const fn mult16x16in32_shl_sat(a: i16, b: i16) -> i32 {
    let product = (a as i32) * (b as i32);
    if product != 0x4000_0000 {
        shl32(product, 1)
    } else {
        MAX_32
    }
}

#[inline(always)]
pub const fn mult32x16in32(a: i32, b: i16) -> i32 {
    let temp = (a as i64) * (b as i64);
    (temp >> 16) as i32
}

#[inline(always)]
pub const fn mult32x16in32_shl(a: i32, b: i16) -> i32 {
    let temp = (a as i64) * (b as i64);
    ((temp >> 16) as i32) << 1
}

#[inline(always)]
pub const fn mult32x16in32_sat(a: i32, b: i16) -> i32 {
    let temp = (a as i64) * (b as i64);
    if temp < MIN_32 as i64 {
        MIN_32
    } else if temp > MAX_32 as i64 {
        MAX_32
    } else {
        temp as i32
    }
}

#[inline(always)]
pub const fn mult32x16in32_shl_sat(a: i32, b: i16) -> i32 {
    if a == MIN_32 && b == MIN_16 {
        MAX_32
    } else {
        mult32x16in32_shl(a, b)
    }
}

#[inline(always)]
pub const fn mult32x16hin32(a: i32, b: i32) -> i32 {
    let temp = (a as i64) * ((b >> 16) as i64);
    (temp >> 16) as i32
}

#[inline(always)]
pub const fn mult32x16hin32_shl(a: i32, b: i32) -> i32 {
    let temp = (a as i64) * ((b >> 16) as i64);
    ((temp >> 16) as i32) << 1
}

#[inline(always)]
pub const fn mult32x16h_in32_shl_sat(a: i32, b: i32) -> i32 {
    if a == MIN_32 && extract16h(b) == MIN_16 {
        MAX_32
    } else {
        mult32x16in32_shl(a, extract16h(b))
    }
}

#[inline(always)]
pub const fn mult32(a: i32, b: i32) -> i32 {
    let temp = (a as i64) * (b as i64);
    (temp >> 32) as i32
}

#[inline(always)]
pub const fn mult32_shl(a: i32, b: i32) -> i32 {
    let temp = (a as i64) * (b as i64);
    ((temp >> 32) as i32) << 1
}

#[inline(always)]
pub const fn mult32_shl_sat(a: i32, b: i32) -> i32 {
    if a == MIN_32 && b == MIN_32 {
        MAX_32
    } else {
        mult32_shl(a, b)
    }
}

#[inline(always)]
pub const fn mul32_sh(a: i32, b: i32, shift: u8) -> i32 {
    let temp = (a as i64) * (b as i64);
    (temp >> shift) as i32
}

#[inline(always)]
pub const fn mac16x16in32_sat(a: i32, b: i16, c: i16) -> i32 {
    add32_sat(a, mult16x16in32(b, c))
}

#[inline(always)]
pub const fn mac16x16in32_shl(a: i32, b: i16, c: i16) -> i32 {
    add32(a, mult16x16in32_shl(b, c))
}

#[inline(always)]
pub const fn mac16x16in32_shl_sat(a: i32, b: i16, c: i16) -> i32 {
    add32_sat(a, mult16x16in32_shl_sat(b, c))
}

#[inline(always)]
pub const fn msu16x16in32(a: i32, b: i16, c: i16) -> i32 {
    sub32(a, mult16x16in32(b, c))
}

#[inline(always)]
pub const fn mac32x16in32(a: i32, b: i32, c: i16) -> i32 {
    a.wrapping_add(mult32x16in32(b, c))
}

#[inline(always)]
pub const fn mac32x16in32_shl(a: i32, b: i32, c: i16) -> i32 {
    a.wrapping_add(mult32x16in32_shl(b, c))
}

#[inline(always)]
pub const fn mac32x16in32_shl_sat(a: i32, b: i32, c: i16) -> i32 {
    add32_sat(a, mult32x16in32_shl_sat(b, c))
}

#[inline(always)]
pub const fn mult32x32in64(a: i32, b: i32) -> i64 {
    (a as i64) * (b as i64)
}

#[inline(always)]
pub const fn mac32x32in64(sum: i64, a: i32, b: i32) -> i64 {
    sum.wrapping_add((a as i64) * (b as i64))
}

#[inline(always)]
pub const fn mac32x32in64_dual(a: i32, b: i32, c: i64) -> i64 {
    let temp = (a as i64) * (b as i64);
    add64_sat(c, temp)
}

#[inline(always)]
pub const fn add64(a: i64, b: i64) -> i64 {
    a.wrapping_add(b)
}

#[inline(always)]
pub const fn sub64(a: i64, b: i64) -> i64 {
    a.wrapping_sub(b)
}

#[inline(always)]
pub const fn add64_sat(a: i64, b: i64) -> i64 {
    a.saturating_add(b)
}

#[inline(always)]
pub const fn sub64_sat(a: i64, b: i64) -> i64 {
    a.saturating_sub(b)
}

#[inline(always)]
pub const fn sat64_32(a: i64) -> i32 {
    if a >= MAX_32 as i64 {
        MAX_32
    } else if a <= MIN_32 as i64 {
        MIN_32
    } else {
        a as i32
    }
}

#[inline(always)]
pub const fn div32_pos_normb(a: i32, b: i32) -> i32 {
    if a == b {
        MAX_32
    } else {
        let mut quotient: i32 = 0;
        let mut mantissa_nr = a as u32;
        let mantissa_dr = b as u32;

        let mut i = 0;
        while i < 32 {
            quotient <<= 1;
            if mantissa_nr >= mantissa_dr {
                mantissa_nr -= mantissa_dr;
                quotient += 1;
            }
            mantissa_nr <<= 1;
            i += 1;
        }
        quotient
    }
}

pub fn div32(mut a: i32, mut b: i32, q_format: &mut i32) -> i32 {
    let mut sign: i16 = 0;

    if a < 0 && b != 0 {
        a = -a;
        sign ^= -1;
    }

    if b < 0 {
        b = -b;
        sign ^= -1;
    }

    if b == 0 {
        *q_format = 0;
        return a;
    }

    let q_nr = norm32(a);
    let mut mantissa_nr = (a as u32) << q_nr;
    let q_dr = norm32(b);
    let mantissa_dr = (b as u32) << q_dr;
    *q_format = 30 + q_nr - q_dr;

    let mut quotient: i32 = 0;
    for _ in 0..31 {
        quotient <<= 1;
        if mantissa_nr >= mantissa_dr {
            mantissa_nr -= mantissa_dr;
            quotient += 1;
        }
        mantissa_nr <<= 1;
    }

    if sign < 0 {
        quotient = -quotient;
    }

    quotient
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_saturating_arithmetic() {
        assert_eq!(add32_sat(MAX_32, 1), MAX_32);
        assert_eq!(add32_sat(MIN_32, -1), MIN_32);
        assert_eq!(sub32_sat(MIN_32, 1), MIN_32);
        assert_eq!(sub32_sat(MAX_32, -1), MAX_32);
    }

    #[test]
    fn test_fractional_multiplication() {
        assert_eq!(mult32x16in32_shl_sat(MIN_32, MIN_16), MAX_32);
        assert_eq!(mult32_shl_sat(MIN_32, MIN_32), MAX_32);
        assert_eq!(mult16x16in32_shl_sat(MIN_16, MIN_16), MAX_32);
    }

    #[test]
    fn test_normalization() {
        assert_eq!(norm32(0), 31);
        assert_eq!(norm32(-1), 31);
        assert_eq!(norm32(0x4000_0000), 0);
        assert_eq!(norm32(0x2000_0000), 1);
        assert_eq!(norm32(1), 30);
    }

    #[test]
    fn test_div32_pos_normb() {
        assert_eq!(div32_pos_normb(100, 100), MAX_32);
        let q = div32_pos_normb(50, 100);
        // 50/100 = 0.5 in Q31 = 0x4000_0000 = 1073741824
        assert_eq!(q, 0x4000_0000);
    }
}
