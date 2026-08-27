//! Precomputed Standard and Mathematical Codebooks
//!
//! Provides static tables for:
//! - Spectral and Scalefactor Huffman codebooks (`huffman`)
//! - Scalefactor band definitions across all sampling rates (`scalefactor`)
//! - Window lookup tables and generators (`window`)
//! - Inverse Quantization and Dequantization tables (`dequant`)
//! - Temporal Noise Shaping tables (`tns`)

pub mod dequant;
pub mod huffman;
pub mod huffman_rom;
pub mod ps;
pub mod qmf;
pub mod sbr;
pub mod scalefactor;
pub mod sfb;
pub mod tns;
pub mod usac_arith;
pub mod window;
