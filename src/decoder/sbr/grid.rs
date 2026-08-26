//! The SBR time grid: where a frame's envelopes and noise floors begin and end.
//!
//! A frame carries between one and eight envelopes. Their borders may be fixed at
//! the frame edges, or one or both edges may float into the neighbouring frame, so
//! that an envelope boundary can be placed exactly on a transient. Which of the
//! four arrangements a frame uses is its *frame class*, and the class decides how
//! the borders are coded.
//!
//! Everything downstream reads only [`FrameGrid`]: the parsing differences between
//! the classes end here.

use crate::bitstream::BitReader;
use crate::error::{DecodeError, Result};

/// QMF time slots in one SBR frame at the usual 1024-sample frame length.
pub const SBR_TIME_SLOTS: i32 = 16;
/// Most envelopes a frame may carry.
pub const MAX_ENVELOPES: usize = 8;
/// Most noise floors a frame may carry.
pub const MAX_NOISE_ENVELOPES: usize = 2;

/// How a frame's envelope borders are anchored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameClass {
    /// Both edges fixed, borders evenly spaced.
    FixFix,
    /// Leading edge fixed, trailing edge and inner borders transmitted.
    FixVar,
    /// Leading edge and inner borders transmitted, trailing edge fixed.
    VarFix,
    /// Both edges transmitted.
    VarVar,
}

impl FrameClass {
    fn from_bits(bits: u8) -> Self {
        match bits {
            0 => Self::FixFix,
            1 => Self::FixVar,
            2 => Self::VarFix,
            _ => Self::VarVar,
        }
    }
}

/// A frame's envelope and noise floor time grid.
#[derive(Debug, Clone)]
pub struct FrameGrid {
    /// How the borders were coded.
    pub class: FrameClass,
    /// Envelope borders, in QMF time slots; `envelopes() + 1` entries.
    pub borders: Vec<i32>,
    /// Noise floor borders; `noise_envelopes() + 1` entries.
    pub noise_borders: Vec<i32>,
    /// Whether each envelope uses the high frequency resolution.
    pub high_res: Vec<bool>,
    /// Envelope holding the transient, or `None` when the frame has none.
    ///
    /// Noise is suppressed in the transient envelope, and gain smoothing is
    /// switched off across it, so that a sharp attack is not smeared.
    pub transient_envelope: Option<usize>,
}

impl Default for FrameGrid {
    fn default() -> Self {
        Self {
            class: FrameClass::FixFix,
            borders: vec![0, SBR_TIME_SLOTS],
            noise_borders: vec![0, SBR_TIME_SLOTS],
            high_res: vec![true],
            transient_envelope: None,
        }
    }
}

impl FrameGrid {
    /// Envelopes in this frame.
    #[inline]
    pub fn envelopes(&self) -> usize {
        self.borders.len() - 1
    }

    /// Noise floors in this frame.
    #[inline]
    pub fn noise_envelopes(&self) -> usize {
        self.noise_borders.len() - 1
    }

    /// Where the frame's first envelope starts, which may be before slot zero for
    /// a class whose leading edge floats.
    #[inline]
    pub fn start_slot(&self) -> i32 {
        self.borders[0]
    }

    /// Where the frame's last envelope ends, which may be past the frame.
    #[inline]
    pub fn end_slot(&self) -> i32 {
        *self.borders.last().unwrap()
    }

    /// Parse `sbr_grid()`.
    pub fn parse(reader: &mut BitReader) -> Result<Self> {
        /// Width of `bs_pointer`, by the number of relative borders coded.
        const POINTER_BITS: [usize; 8] = [1, 2, 2, 3, 3, 3, 3, 3];

        let class = FrameClass::from_bits(reader.read_u8(2)?);
        let mut grid = Self {
            class,
            borders: Vec::with_capacity(MAX_ENVELOPES + 1),
            noise_borders: Vec::with_capacity(MAX_NOISE_ENVELOPES + 1),
            high_res: Vec::with_capacity(MAX_ENVELOPES),
            transient_envelope: None,
        };
        let mut pointer = 0usize;

        match class {
            FrameClass::FixFix => {
                let count = 1usize << reader.read_u8(2)?;
                let high_res = reader.read_bit()?;
                // Evenly spaced, rounding so the borders stay monotone.
                let step = (SBR_TIME_SLOTS + count as i32 / 2) / count as i32;
                grid.borders.push(0);
                for i in 1..count {
                    grid.borders.push(i as i32 * step);
                }
                grid.borders.push(SBR_TIME_SLOTS);
                grid.high_res = vec![high_res; count];
            }

            FrameClass::FixVar => {
                let var_border = reader.read_u8(2)? as i32;
                let relative = reader.read_u8(2)? as usize;
                let count = relative + 1;

                grid.borders = vec![0; count + 1];
                let mut border = SBR_TIME_SLOTS + var_border;
                grid.borders[count] = border;
                for k in (1..=relative).rev() {
                    border -= 2 * reader.read_u8(2)? as i32 + 2;
                    grid.borders[k] = border.max(0);
                }

                pointer = reader.read_bits(POINTER_BITS[relative])? as usize;
                if pointer > count {
                    return Err(corrupt("SBR transient pointer is past the last envelope"));
                }
                grid.high_res = vec![false; count];
                for k in (0..count).rev() {
                    grid.high_res[k] = reader.read_bit()?;
                }
                if pointer > 0 {
                    grid.transient_envelope = Some(count + 1 - pointer);
                }
            }

            FrameClass::VarFix => {
                let var_border = reader.read_u8(2)? as i32;
                let relative = reader.read_u8(2)? as usize;
                let count = relative + 1;

                grid.borders = vec![0; count + 1];
                let mut border = var_border;
                grid.borders[0] = border;
                for k in 1..=relative {
                    border = (border + 2 * reader.read_u8(2)? as i32 + 2).min(SBR_TIME_SLOTS);
                    grid.borders[k] = border;
                }
                grid.borders[count] = SBR_TIME_SLOTS;

                pointer = reader.read_bits(POINTER_BITS[relative])? as usize;
                if pointer > count {
                    return Err(corrupt("SBR transient pointer is past the last envelope"));
                }
                grid.high_res = (0..count).map(|_| reader.read_bit()).collect::<Result<_>>()?;
                if pointer > 1 {
                    grid.transient_envelope = Some(pointer - 1);
                }
            }

            FrameClass::VarVar => {
                let lead_border = reader.read_u8(2)? as i32;
                let trail_border = SBR_TIME_SLOTS + reader.read_u8(2)? as i32;
                let relative_lead = reader.read_u8(2)? as usize;
                let relative_trail = reader.read_u8(2)? as usize;
                let count = relative_lead + relative_trail + 1;
                if count > MAX_ENVELOPES {
                    return Err(corrupt("SBR frame asks for more envelopes than allowed"));
                }

                grid.borders = vec![0; count + 1];
                let mut border = lead_border;
                grid.borders[0] = border;
                for k in 1..=relative_lead {
                    border += 2 * reader.read_u8(2)? as i32 + 2;
                    grid.borders[k] = border;
                }
                let mut border = trail_border;
                grid.borders[count] = border;
                for k in 0..relative_trail {
                    border -= 2 * reader.read_u8(2)? as i32 + 2;
                    grid.borders[count - 1 - k] = border;
                }

                pointer = reader.read_bits(POINTER_BITS[relative_lead + relative_trail])? as usize;
                if pointer > count {
                    return Err(corrupt("SBR transient pointer is past the last envelope"));
                }
                grid.high_res = (0..count).map(|_| reader.read_bit()).collect::<Result<_>>()?;
                if pointer > 0 {
                    grid.transient_envelope = Some(count + 1 - pointer);
                }
            }
        }

        grid.place_noise_borders(pointer);
        grid.validate()?;
        Ok(grid)
    }

    /// Place the noise floor borders, which are a subset of the envelope borders.
    ///
    /// A frame with one envelope carries one noise floor spanning it. Otherwise it
    /// carries two, and the border between them sits at the transient where there
    /// is one, so that the noise level can change with the attack.
    fn place_noise_borders(&mut self, pointer: usize) {
        let count = self.envelopes();
        self.noise_borders.clear();
        self.noise_borders.push(self.borders[0]);
        if count > 1 {
            let middle = match self.class {
                FrameClass::FixFix => count / 2,
                FrameClass::FixVar | FrameClass::VarVar => {
                    count - pointer.saturating_sub(1).max(1)
                }
                FrameClass::VarFix => match pointer {
                    0 => 1,
                    1 => count - 1,
                    p => p - 1,
                },
            };
            self.noise_borders.push(self.borders[middle.clamp(1, count - 1)]);
        }
        self.noise_borders.push(self.borders[count]);
    }

    /// Reject grids whose borders do not increase, which would make an envelope
    /// span a negative number of slots.
    fn validate(&self) -> Result<()> {
        if self.envelopes() == 0 || self.envelopes() > MAX_ENVELOPES {
            return Err(corrupt("SBR frame has an impossible envelope count"));
        }
        if self.borders.windows(2).any(|w| w[1] <= w[0]) {
            return Err(corrupt("SBR envelope borders do not increase"));
        }
        if self.noise_borders.windows(2).any(|w| w[1] <= w[0]) {
            return Err(corrupt("SBR noise borders do not increase"));
        }
        if let Some(t) = self.transient_envelope
            && t > self.envelopes()
        {
            return Err(corrupt("SBR transient envelope is out of range"));
        }
        Ok(())
    }

    /// Total envelope scalefactors this grid implies, given the band counts.
    pub fn envelope_scalefactors(&self, low_bands: usize, high_bands: usize) -> usize {
        self.high_res.iter().map(|&hi| if hi { high_bands } else { low_bands }).sum()
    }
}

fn corrupt(message: &str) -> crate::error::Error {
    DecodeError::CorruptedFrame(message.into()).into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::BitWriter;

    fn parse(bits: &[(u64, usize)]) -> Result<FrameGrid> {
        let mut w = BitWriter::new();
        for &(v, n) in bits {
            w.write_bits(v, n);
        }
        w.write_bits(0, 32);
        let bytes = w.into_bytes();
        let mut r = BitReader::new(&bytes);
        FrameGrid::parse(&mut r)
    }

    /// The evenly spaced class must divide the frame exactly.
    #[test]
    fn fixfix_borders_are_even() {
        for (raw, count) in [(0u64, 1usize), (1, 2), (2, 4), (3, 8)] {
            let grid = parse(&[(0, 2), (raw, 2), (1, 1)]).unwrap();
            assert_eq!(grid.envelopes(), count);
            assert_eq!(grid.borders[0], 0);
            assert_eq!(*grid.borders.last().unwrap(), SBR_TIME_SLOTS);
            let step = SBR_TIME_SLOTS / count as i32;
            for (i, &b) in grid.borders.iter().enumerate() {
                assert_eq!(b, i as i32 * step);
            }
            assert!(grid.high_res.iter().all(|&r| r));
            assert_eq!(grid.transient_envelope, None);
            assert_eq!(grid.noise_envelopes(), if count == 1 { 1 } else { 2 });
        }
    }

    /// A trailing border past the frame is what lets the next frame's transient be
    /// captured, so it must be preserved rather than clamped.
    #[test]
    fn fixvar_trailing_border_may_overshoot() {
        // class FIXVAR, var_bord 3, one relative border of 2*1+2 = 4, pointer 0,
        // then two freq_res bits.
        let grid = parse(&[(1, 2), (3, 2), (1, 2), (1, 2), (0, 2), (1, 1), (0, 1)]).unwrap();
        assert_eq!(grid.class, FrameClass::FixVar);
        assert_eq!(grid.envelopes(), 2);
        assert_eq!(grid.borders, vec![0, 15, 19]);
        assert_eq!(grid.transient_envelope, None);
    }

    /// The transient pointer must select an envelope, and the noise border must
    /// land on it.
    #[test]
    fn transient_pointer_selects_an_envelope() {
        // FIXVAR, var_bord 0, 2 relative borders, pointer 2.
        let grid = parse(&[
            (1, 2),
            (0, 2),
            (2, 2),
            (0, 2),
            (0, 2),
            (2, 2),
            (1, 1),
            (1, 1),
            (1, 1),
        ])
        .unwrap();
        assert_eq!(grid.envelopes(), 3);
        assert_eq!(grid.transient_envelope, Some(2));
        assert_eq!(grid.noise_borders[1], grid.borders[2]);
    }

    /// Parsing must not panic on arbitrary bits, and must reject rather than
    /// return a grid whose borders run backwards.
    #[test]
    fn arbitrary_bits_never_panic() {
        for seed in 0..4096u32 {
            let bytes: Vec<u8> = (0..8)
                .map(|i| ((seed.wrapping_mul(2654435761).wrapping_add(i)) >> 7) as u8)
                .collect();
            let mut r = BitReader::new(&bytes);
            if let Ok(grid) = FrameGrid::parse(&mut r) {
                assert!(grid.borders.windows(2).all(|w| w[0] < w[1]));
                assert!(grid.noise_borders.windows(2).all(|w| w[0] < w[1]));
                assert_eq!(grid.high_res.len(), grid.envelopes());
            }
        }
    }
}
