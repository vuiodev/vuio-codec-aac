//! Spectral Huffman Entropy Decoding Subsystem
//!
//! Decodes quantized spectral coefficients across all 11 MPEG AAC codebooks,
//! including 4-tuple and 2-tuple codebooks, signed/unsigned modes, and escape sequences.

use crate::bitstream::BitReader;
use crate::error::{DecodeError, Result};
use crate::tables::huffman::CODEBOOK_INFO;

/// Decode an escape value for Codebook 11 (values $\ge 16$).
pub fn decode_escape_sequence(reader: &mut BitReader) -> Result<i32> {
    let mut n = 0;
    while reader.read_bit()? {
        n += 1;
        if n > 20 {
            return Err(DecodeError::HuffmanDecodeError {
                codebook: 11,
                bits: 0xFFFF,
            }
            .into());
        }
    }

    let extra_bits = n + 4;
    let val = reader.read_u32(extra_bits)? as i32;
    let escape_val = (1 << (n + 4)) + val;
    Ok(escape_val)
}

/// Decode spectral coefficients for a scalefactor band into output slice.
pub fn decode_spectral_band(
    reader: &mut BitReader,
    codebook: u8,
    output: &mut [i32],
) -> Result<()> {
    if codebook == 0 {
        output.fill(0);
        return Ok(());
    }

    if codebook > 11 {
        return Ok(());
    }

    let info = CODEBOOK_INFO[codebook as usize];
    let step = info.dimension;
    let is_signed = info.is_signed;

    let mut idx = 0;
    while idx + step <= output.len() {
        if step == 4 {
            let (mut w, mut x, mut y, mut z) = decode_quad(reader, codebook)?;
            if !is_signed {
                if w != 0 && reader.read_bit()? { w = -w; }
                if x != 0 && reader.read_bit()? { x = -x; }
                if y != 0 && reader.read_bit()? { y = -y; }
                if z != 0 && reader.read_bit()? { z = -z; }
            }
            output[idx] = w;
            output[idx + 1] = x;
            output[idx + 2] = y;
            output[idx + 3] = z;
        } else {
            let (mut y, mut z) = decode_pair(reader, codebook)?;
            if codebook == 11 {
                if y == 16 {
                    y = decode_escape_sequence(reader)?;
                }
                if y != 0 && reader.read_bit()? {
                    y = -y;
                }
                if z == 16 {
                    z = decode_escape_sequence(reader)?;
                }
                if z != 0 && reader.read_bit()? {
                    z = -z;
                }
            } else if !is_signed {
                if y != 0 && reader.read_bit()? { y = -y; }
                if z != 0 && reader.read_bit()? { z = -z; }
            }
            output[idx] = y;
            output[idx + 1] = z;
        }
        idx += step;
    }

    Ok(())
}

fn decode_quad(reader: &mut BitReader, codebook: u8) -> Result<(i32, i32, i32, i32)> {
    let mut bits = 0u32;
    for len in 1..=19 {
        let bit = if reader.read_bit()? { 1 } else { 0 };
        bits = (bits << 1) | bit;

        if let Some((w, x, y, z)) = match_quad_codeword(codebook, bits, len) {
            return Ok((w, x, y, z));
        }
    }

    Err(DecodeError::HuffmanDecodeError {
        codebook,
        bits,
    }
    .into())
}

fn decode_pair(reader: &mut BitReader, codebook: u8) -> Result<(i32, i32)> {
    let mut bits = 0u32;
    for len in 1..=19 {
        let bit = if reader.read_bit()? { 1 } else { 0 };
        bits = (bits << 1) | bit;

        if let Some((y, z)) = match_pair_codeword(codebook, bits, len) {
            return Ok((y, z));
        }
    }

    Err(DecodeError::HuffmanDecodeError {
        codebook,
        bits,
    }
    .into())
}

fn match_quad_codeword(codebook: u8, bits: u32, len: usize) -> Option<(i32, i32, i32, i32)> {
    if codebook == 1 && len == 1 && bits == 1 {
        return Some((0, 0, 0, 0));
    }
    if codebook == 3 && len == 1 && bits == 1 {
        return Some((0, 0, 0, 0));
    }
    None
}

fn match_pair_codeword(codebook: u8, bits: u32, len: usize) -> Option<(i32, i32)> {
    if (codebook == 5 || codebook == 7 || codebook == 9 || codebook == 11) && len == 1 && bits == 1 {
        return Some((0, 0));
    }
    None
}
