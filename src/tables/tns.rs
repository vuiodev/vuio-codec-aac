//! Temporal Noise Shaping (TNS) Tables
//!
//! Maximum active bands and reflection coefficients for 3-bit and 4-bit precision.

/// Maximum number of TNS bands per sample rate index (for Long and Short windows).
pub static TNS_MAX_BANDS: &[[u8; 2]; 12] = &[
    [31, 9],  // 96000 Hz
    [31, 9],  // 88200 Hz
    [34, 10], // 64000 Hz
    [40, 14], // 48000 Hz
    [42, 14], // 44100 Hz
    [51, 14], // 32000 Hz
    [46, 14], // 24000 Hz
    [46, 14], // 22050 Hz
    [42, 14], // 16000 Hz
    [42, 14], // 12000 Hz
    [42, 14], // 11025 Hz
    [39, 14], //  8000 Hz
];

/// 3-bit resolution TNS reflection coefficients in 16-bit Q15 format.
pub static TNS_COEFF3_16: &[i16; 8] = &[
    -0x7e0e, -0x6eda, -0x5247, -0x2bc7, 0x0000, 0x378a, 0x6413, 0x7cca,
];

/// 4-bit resolution TNS reflection coefficients in 16-bit Q15 format.
pub static TNS_COEFF4_16: &[i16; 16] = &[
    -0x7f74, -0x7b1d, -0x7295, -0x6625, -0x563c, -0x4362, -0x2e3d, -0x1785, 0x0000, 0x1a9d,
    0x3410, 0x4b3d, 0x5f1f, 0x6eda, 0x79bc, 0x7f4c,
];

/// 3-bit resolution TNS reflection coefficients in 32-bit Q31 format.
pub static TNS_COEFF3_32: &[i32; 8] = &[
    -0x7e0e2e31,
    -0x6ed9eba0,
    -0x5246dd48,
    -0x2bc750e8,
    0x00000000,
    0x3789809a,
    0x64130dd3,
    0x7cca7014,
];

/// 4-bit resolution TNS reflection coefficients in 32-bit Q31 format.
pub static TNS_COEFF4_32: &[i32; 16] = &[
    -0x7f7437ac,
    -0x7b1d1a47,
    -0x7294b5f1,
    -0x66256db1,
    -0x563ba8a9,
    -0x4362210d,
    -0x2e3d2aba,
    -0x17851aac,
    0x00000000,
    0x1a9cd9d0,
    0x34101e4a,
    0x4b3c8b41,
    0x5f1f235d,
    0x6ed9eba0,
    0x79bc38e4,
    0x7f4c7e52,
];
