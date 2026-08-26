//! Cyclic Redundancy Check (CRC-16) Computation
//!
//! Provides standard CRC-16 computation for ADTS headers (polynomial `0x8005`, init `0xFFFF`),
//! SBR extension payloads, and UniDRC metadata streams.

/// Calculate 16-bit CRC over a byte slice using standard ADTS CRC polynomial `0x8005`.
pub fn crc16_adts(data: &[u8]) -> u16 {
    let mut crc: u16 = 0xFFFF;
    for &byte in data {
        crc ^= (byte as u16) << 8;
        for _ in 0..8 {
            if (crc & 0x8000) != 0 {
                crc = (crc << 1) ^ 0x8005;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

/// Calculate 16-bit CRC over bit-level data for SBR payloads (polynomial `0x001D`, init `0x0000`).
pub fn crc16_sbr(data: &[u8], bit_count: usize) -> u16 {
    let mut crc: u16 = 0x0000;
    let num_full_bytes = bit_count / 8;
    let remaining_bits = bit_count % 8;

    for &byte in data.iter().take(num_full_bytes) {
        for b in (0..8).rev() {
            let bit = (byte >> b) & 1;
            let crc_msb = ((crc >> 9) & 1) as u8;
            crc = (crc << 1) & 0x03FF;
            if crc_msb ^ bit != 0 {
                crc ^= 0x001D;
            }
        }
    }

    if remaining_bits > 0 {
        let byte = data[num_full_bytes];
        for b in (8 - remaining_bits..8).rev() {
            let bit = (byte >> b) & 1;
            let crc_msb = ((crc >> 9) & 1) as u8;
            crc = (crc << 1) & 0x03FF;
            if crc_msb ^ bit != 0 {
                crc ^= 0x001D;
            }
        }
    }

    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adts_crc16() {
        let test_data = [0xFF, 0xF1, 0x50, 0x80];
        let crc = crc16_adts(&test_data);
        assert_ne!(crc, 0);
    }
}
