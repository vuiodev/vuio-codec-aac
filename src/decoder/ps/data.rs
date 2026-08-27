//! Parametric stereo bitstream syntax.
//!
//! A parametric stereo payload rides inside SBR's extension field and describes,
//! for each of one to four *envelopes* covering the frame, how the two original
//! channels differed in each of 10, 20 or 34 parameter bands:
//!
//! * **IID**, the level difference, quantised coarsely (15 steps) or finely (31),
//! * **ICC**, how coherent the two channels were, in 8 steps.
//!
//! Both are Huffman-coded as differences, either against the band below (`Df`) or
//! against the same band of the previous envelope (`Dt`), so a decoder that has
//! lost sync must wait for a frequency-differential envelope before its parameters
//! mean anything.
//!
//! ISO/IEC 14496-3 also defines an extension carrying phase differences (IPD/OPD).
//! The reference implementation this port follows neither writes nor reads it, so
//! the extension is stepped over here and the reconstruction stays phase-neutral.

use crate::bitstream::BitReader;
use crate::error::{DecodeError, Result};
use crate::tables::ps::{
    BINS_20, BINS_34, HUFF_ICC_DF, HUFF_ICC_DT, HUFF_IID_DF, HUFF_IID_DF_FINE, HUFF_IID_DT,
    HUFF_IID_DT_FINE, ICC_LEVELS, IID_STEPS, IID_STEPS_FINE, MAX_ENVELOPES,
};

use super::hybrid::{Resolution, SLOTS};

/// Parameter bands each `bs_iid_mode` / `bs_icc_mode` value asks for.
const BANDS_PER_MODE: [usize; 3] = [10, BINS_20, BINS_34];

/// One frame's worth of parametric stereo parameters, delta-decoded and resampled
/// onto the working resolution.
#[derive(Clone)]
pub struct PsData {
    /// Envelopes this frame carries; at least one.
    pub envelopes: usize,
    /// Slot each envelope starts at, plus a final entry at [`SLOTS`].
    pub borders: [usize; MAX_ENVELOPES + 2],
    /// Level difference index per envelope and bin, in `-steps..=steps`.
    pub iid: [[i8; BINS_34]; MAX_ENVELOPES + 2],
    /// Coherence index per envelope and bin, in `0..ICC_LEVELS`.
    pub icc: [[u8; BINS_34]; MAX_ENVELOPES + 2],
    /// Whether the level differences use the fine quantiser.
    pub fine_iid: bool,
    /// Whether the mixing matrix is built by principal-component rotation.
    pub pca_rotation: bool,
    /// Bins the parameters resolve to, and hence which hybrid split to use.
    pub resolution: Resolution,
}

impl Default for PsData {
    fn default() -> Self {
        Self {
            envelopes: 1,
            borders: [0, SLOTS, SLOTS, SLOTS, SLOTS, SLOTS, SLOTS],
            iid: [[0; BINS_34]; MAX_ENVELOPES + 2],
            icc: [[0; BINS_34]; MAX_ENVELOPES + 2],
            fine_iid: false,
            pca_rotation: false,
            resolution: Resolution::Coarse,
        }
    }
}

impl PsData {
    /// Parameter bins the frame resolves to.
    #[inline]
    pub const fn bins(&self) -> usize {
        match self.resolution {
            Resolution::Coarse => BINS_20,
            Resolution::Fine => BINS_34,
        }
    }
}

/// Everything a payload can leave standing for the frames that follow it.
///
/// A payload may omit the header, in which case the previous one's modes stay in
/// force, and may code its parameters against the previous frame's, so the last
/// envelope's values have to survive the frame boundary too.
#[derive(Clone)]
pub struct PsParser {
    iid_enabled: bool,
    icc_enabled: bool,
    /// `bs_iid_mode` with the fine-quantiser offset removed.
    iid_mode: usize,
    /// `bs_icc_mode` with the rotation offset removed.
    icc_mode: usize,
    fine_iid: bool,
    pca_rotation: bool,
    /// Whether a `ps_extension()` field follows the parameters.
    extension: bool,
    /// Last envelope of the previous frame, the reference for a time-differential
    /// first envelope.
    prev_iid: [i8; BINS_34],
    prev_icc: [u8; BINS_34],
    /// Set once a payload has been parsed, so the first frame of a stream that
    /// opens mid-way is not reconstructed from nothing.
    seen: bool,
}

impl Default for PsParser {
    fn default() -> Self {
        Self::new()
    }
}

impl PsParser {
    /// A parser that has seen no payload.
    pub const fn new() -> Self {
        Self {
            iid_enabled: false,
            icc_enabled: false,
            iid_mode: 0,
            icc_mode: 0,
            fine_iid: false,
            pca_rotation: false,
            extension: false,
            prev_iid: [0; BINS_34],
            prev_icc: [0; BINS_34],
            seen: false,
        }
    }

    /// Forget everything, as after a seek.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Whether any payload has been parsed.
    #[inline]
    pub const fn is_ready(&self) -> bool {
        self.seen
    }

    /// Parse one `ps_data()` element and resolve it to absolute parameters.
    ///
    /// `available_bits` bounds the extension field so that a truncated or hostile
    /// payload cannot walk off the end of the frame.
    pub fn parse(&mut self, reader: &mut BitReader, available_bits: usize) -> Result<PsData> {
        let start = reader.bit_position();

        if reader.read_bit()? {
            self.parse_header(reader)?;
        } else if !self.seen {
            return Err(bad("parametric stereo payload arrived before any header"));
        }

        let mut raw = RawFrame::read(reader, self)?;
        let data = self.resolve(&mut raw);

        if self.extension {
            // `ps_extension()` carries the phase parameters this decoder does not
            // apply. It is byte-counted, so stepping over it keeps the reader
            // aligned whatever it holds.
            let mut count = reader.read_u8(4)? as usize;
            if count == 15 {
                count += reader.read_u8(8)? as usize;
            }
            let consumed = reader.bit_position() - start;
            let room = available_bits.saturating_sub(consumed);
            reader.skip_bits((count * 8).min(room))?;
        }

        if std::env::var_os("AAC_TRACE_PS").is_some() {
            eprintln!(
                "ps: iid {} mode {} fine {} | icc {} mode {} | ext {} | env {} borders {:?} res {:?}",
                self.iid_enabled,
                self.iid_mode,
                self.fine_iid,
                self.icc_enabled,
                self.icc_mode,
                self.extension,
                data.envelopes,
                &data.borders[..=data.envelopes],
                data.resolution
            );
            for e in 0..data.envelopes {
                eprintln!("    iid[{e}] {:?}", &data.iid[e][..data.bins()]);
                eprintln!("    icc[{e}] {:?}", &data.icc[e][..data.bins()]);
            }
        }

        self.seen = true;
        Ok(data)
    }

    /// Parse `ps_header()`, which every field of is optional in later frames.
    fn parse_header(&mut self, reader: &mut BitReader) -> Result<()> {
        self.iid_enabled = reader.read_bit()?;
        if self.iid_enabled {
            let mode = reader.read_u8(3)? as usize;
            // Modes 3 to 5 are modes 0 to 2 with the fine quantiser.
            self.fine_iid = mode > 2;
            self.iid_mode = if self.fine_iid { mode - 3 } else { mode };
            if self.iid_mode > 2 {
                return Err(bad("parametric stereo header asks for a reserved IID mode"));
            }
        }

        self.icc_enabled = reader.read_bit()?;
        if self.icc_enabled {
            let mode = reader.read_u8(3)? as usize;
            // Modes 3 to 5 select the principal-component form of the same grids.
            self.pca_rotation = mode > 2;
            self.icc_mode = if self.pca_rotation { mode - 3 } else { mode };
            if self.icc_mode > 2 {
                return Err(bad("parametric stereo header asks for a reserved ICC mode"));
            }
        }

        self.extension = reader.read_bit()?;
        Ok(())
    }

    /// The hybrid split the current modes call for.
    fn resolution(&self) -> Resolution {
        let fine = (self.iid_enabled && self.iid_mode == 2) || (self.icc_enabled && self.icc_mode == 2);
        if fine { Resolution::Fine } else { Resolution::Coarse }
    }

    /// Turn differences into absolute indices, then resample onto the working grid.
    fn resolve(&mut self, raw: &mut RawFrame) -> PsData {
        let resolution = self.resolution();
        let mut data = PsData {
            envelopes: raw.envelopes.max(1),
            borders: raw.borders,
            iid: [[0; BINS_34]; MAX_ENVELOPES + 2],
            icc: [[0; BINS_34]; MAX_ENVELOPES + 2],
            fine_iid: self.fine_iid,
            pca_rotation: self.pca_rotation,
            resolution,
        };

        let iid_bands = BANDS_PER_MODE[self.iid_mode];
        let icc_bands = BANDS_PER_MODE[self.icc_mode];
        let iid_limit = if self.fine_iid { IID_STEPS_FINE } else { IID_STEPS } as i32;

        // A 10-band grid is stored at 20-band stride, so a time-differential
        // envelope reads its reference every second bin.
        let iid_stride = if iid_bands == 10 { 2 } else { 1 };
        let icc_stride = if icc_bands == 10 { 2 } else { 1 };

        for e in 0..raw.envelopes {
            if self.iid_enabled {
                let prev: [i8; BINS_34] =
                    if e == 0 { self.prev_iid } else { data.iid[e - 1] };
                accumulate(
                    &mut data.iid[e],
                    &raw.iid[e],
                    iid_bands,
                    if raw.iid_time[e] { Some((&prev, iid_stride)) } else { None },
                    -iid_limit,
                    iid_limit,
                );
                if iid_stride == 2 {
                    stretch(&mut data.iid[e], iid_bands);
                }
            }

            if self.icc_enabled {
                let prev: [u8; BINS_34] =
                    if e == 0 { self.prev_icc } else { data.icc[e - 1] };
                let prev_signed = prev.map(|v| v as i8);
                let mut out = [0i8; BINS_34];
                accumulate(
                    &mut out,
                    &raw.icc[e],
                    icc_bands,
                    if raw.icc_time[e] { Some((&prev_signed, icc_stride)) } else { None },
                    0,
                    ICC_LEVELS as i32 - 1,
                );
                if icc_stride == 2 {
                    stretch(&mut out, icc_bands);
                }
                data.icc[e] = out.map(|v| v as u8);
            }
        }

        if raw.envelopes == 0 {
            // A payload with no envelope holds the previous frame's parameters for
            // the whole frame rather than muting the stereo image.
            data.envelopes = 1;
            data.iid[0] = if self.iid_enabled { self.prev_iid } else { [0; BINS_34] };
            data.icc[0] = if self.icc_enabled { self.prev_icc } else { [0; BINS_34] };
        }

        self.prev_iid = data.iid[data.envelopes - 1];
        self.prev_icc = data.icc[data.envelopes - 1];

        place_borders(&mut data, raw.variable_borders);

        // The parameter grid and the hybrid split have to agree, so whichever of the
        // two the bitstream chose, resample onto the other.
        let bins = data.bins();
        for e in 0..data.envelopes {
            resample(&mut data.iid[e], bands_of(self.iid_mode), bins);
            let mut icc = data.icc[e].map(|v| v as i8);
            resample(&mut icc, bands_of(self.icc_mode), bins);
            data.icc[e] = icc.map(|v| v.max(0) as u8);
        }
        data
    }
}

/// Bands a mode codes, after any 10-to-20 stretch.
const fn bands_of(mode: usize) -> usize {
    match mode {
        0 | 1 => BINS_20,
        _ => BINS_34,
    }
}

/// A payload as read, before differences are resolved.
struct RawFrame {
    envelopes: usize,
    borders: [usize; MAX_ENVELOPES + 2],
    variable_borders: bool,
    iid: [[i32; BINS_34]; MAX_ENVELOPES + 2],
    icc: [[i32; BINS_34]; MAX_ENVELOPES + 2],
    iid_time: [bool; MAX_ENVELOPES + 2],
    icc_time: [bool; MAX_ENVELOPES + 2],
}

impl RawFrame {
    fn read(reader: &mut BitReader, parser: &PsParser) -> Result<Self> {
        const FIXED_ENVELOPES: [usize; 4] = [0, 1, 2, 4];

        let mut frame = Self {
            envelopes: 0,
            borders: [0; MAX_ENVELOPES + 2],
            variable_borders: false,
            iid: [[0; BINS_34]; MAX_ENVELOPES + 2],
            icc: [[0; BINS_34]; MAX_ENVELOPES + 2],
            iid_time: [false; MAX_ENVELOPES + 2],
            icc_time: [false; MAX_ENVELOPES + 2],
        };

        frame.variable_borders = reader.read_bit()?;
        let selector = reader.read_u8(2)? as usize;
        if frame.variable_borders {
            frame.envelopes = selector + 1;
            for e in 1..=frame.envelopes {
                frame.borders[e] = reader.read_u8(5)? as usize + 1;
            }
        } else {
            frame.envelopes = FIXED_ENVELOPES[selector];
        }

        if parser.iid_enabled {
            let bands = BANDS_PER_MODE[parser.iid_mode];
            let (df, dt): (&[i16], &[i16]) = if parser.fine_iid {
                (&HUFF_IID_DF_FINE, &HUFF_IID_DT_FINE)
            } else {
                (&HUFF_IID_DF, &HUFF_IID_DT)
            };
            for e in 0..frame.envelopes {
                frame.iid_time[e] = reader.read_bit()?;
                let table = if frame.iid_time[e] { dt } else { df };
                for b in 0..bands {
                    frame.iid[e][b] = huffman(reader, table)?;
                }
            }
        }

        if parser.icc_enabled {
            let bands = BANDS_PER_MODE[parser.icc_mode];
            for e in 0..frame.envelopes {
                frame.icc_time[e] = reader.read_bit()?;
                let table: &[i16] = if frame.icc_time[e] { &HUFF_ICC_DT } else { &HUFF_ICC_DF };
                for b in 0..bands {
                    frame.icc[e][b] = huffman(reader, table)?;
                }
            }
        }

        Ok(frame)
    }
}

/// Turn one envelope's differences into absolute indices, clamped to the grid.
///
/// With `previous` set the differences are against the same band one envelope back,
/// read at `stride` because a 10-band grid is held at 20-band spacing; without it
/// they are against the band below, and the first band is already absolute.
fn accumulate(
    out: &mut [i8; BINS_34],
    deltas: &[i32; BINS_34],
    bands: usize,
    previous: Option<(&[i8; BINS_34], usize)>,
    min: i32,
    max: i32,
) {
    match previous {
        Some((prev, stride)) => {
            for b in 0..bands {
                let reference = prev[(b * stride).min(BINS_34 - 1)] as i32;
                out[b] = (reference + deltas[b]).clamp(min, max) as i8;
            }
        }
        None => {
            let mut running = deltas[0].clamp(min, max);
            out[0] = running as i8;
            for b in 1..bands {
                running = (running + deltas[b]).clamp(min, max);
                out[b] = running as i8;
            }
        }
    }
}

/// Spread a 10-band grid over 20 bins, each band covering two.
fn stretch(values: &mut [i8; BINS_34], bands: usize) {
    for b in (1..bands * 2).rev() {
        values[b] = values[b / 2];
    }
}

/// Move parameters between the 20- and 34-bin grids.
///
/// Widening repeats or splits bins; narrowing averages them. Both directions are
/// fixed by the standard rather than derived, because the two grids are not nested.
fn resample(values: &mut [i8; BINS_34], from: usize, to: usize) {
    match (from, to) {
        (BINS_20, BINS_34) => {
            const SOURCE: [usize; BINS_34] = [
                0, 0, 1, 2, 2, 3, 4, 4, 5, 5, 6, 7, 8, 8, 9, 9, 10, 11, 12, 13, 14, 14, 15, 15, 16,
                16, 17, 17, 18, 18, 18, 18, 19, 19,
            ];
            // Bins 1, 4 and their kind sit between two coarse bins and take the mean.
            const BLEND: [Option<(usize, usize)>; BINS_34] = {
                let mut t = [None; BINS_34];
                t[1] = Some((0, 1));
                t[4] = Some((2, 3));
                t
            };
            let source = *values;
            for b in 0..BINS_34 {
                values[b] = match BLEND[b] {
                    Some((lo, hi)) => average2(source[lo], source[hi]),
                    None => source[SOURCE[b]],
                };
            }
        }
        (BINS_34, BINS_20) => {
            let s = *values;
            let mut out = [0i8; BINS_34];
            out[0] = third(2 * s[0] as i32 + s[1] as i32);
            out[1] = third(s[1] as i32 + 2 * s[2] as i32);
            out[2] = third(2 * s[3] as i32 + s[4] as i32);
            out[3] = third(s[4] as i32 + 2 * s[5] as i32);
            out[4] = average2(s[6], s[7]);
            out[5] = average2(s[8], s[9]);
            out[6] = s[10];
            out[7] = s[11];
            out[8] = average2(s[12], s[13]);
            out[9] = average2(s[14], s[15]);
            out[10] = s[16];
            out[11] = s[17];
            out[12] = s[18];
            out[13] = s[19];
            out[14] = average2(s[20], s[21]);
            out[15] = average2(s[22], s[23]);
            out[16] = average2(s[24], s[25]);
            out[17] = average2(s[26], s[27]);
            out[18] = halve(halve(
                s[28] as i32 + s[29] as i32 + s[30] as i32 + s[31] as i32,
            ) as i32);
            out[19] = average2(s[32], s[33]);
            *values = out;
        }
        _ => {}
    }
}

/// Halve towards zero, as the standard's integer parameter mapping does.
#[inline]
fn halve(v: i32) -> i8 {
    (v.abs() / 2 * v.signum()) as i8
}

/// A third, towards zero.
#[inline]
fn third(v: i32) -> i8 {
    (v.abs() / 3 * v.signum()) as i8
}

#[inline]
fn average2(a: i8, b: i8) -> i8 {
    halve(a as i32 + b as i32)
}

/// Fix up envelope borders, which the two frame classes express differently.
///
/// A fixed-class frame divides the frame evenly; a variable-class frame transmits
/// each border, which then has to be forced into order and inside the frame.
fn place_borders(data: &mut PsData, variable: bool) {
    data.borders[0] = 0;

    if !variable {
        for e in 1..data.envelopes {
            data.borders[e] = e * SLOTS / data.envelopes;
        }
        data.borders[data.envelopes] = SLOTS;
        return;
    }

    if data.borders[data.envelopes] < SLOTS {
        // A frame whose last border stops short gets one more envelope, repeating
        // the last one's parameters, rather than leaving a gap.
        if data.envelopes < MAX_ENVELOPES + 1 {
            data.envelopes += 1;
            data.iid[data.envelopes - 1] = data.iid[data.envelopes - 2];
            data.icc[data.envelopes - 1] = data.icc[data.envelopes - 2];
        }
        data.borders[data.envelopes] = SLOTS;
    }

    for e in 1..data.envelopes {
        let ceiling = SLOTS - (data.envelopes - e);
        let floor = data.borders[e - 1] + 1;
        data.borders[e] = data.borders[e].clamp(floor.min(ceiling), ceiling);
    }
    data.borders[data.envelopes] = SLOTS;
}

/// Decode one Huffman-coded parameter difference.
///
/// The tables are binary trees packed one node to an `i16`: the top byte is the
/// node to move to on a zero bit and the bottom seven bits, sign-extended, the node
/// to move to on a one. A negative node is a leaf holding its value less 64.
fn huffman(reader: &mut BitReader, table: &[i16]) -> Result<i32> {
    let mut node: i32 = 0;
    // The deepest code in any of these tables is well under this; the bound only
    // exists so a corrupt table could not spin here forever.
    for _ in 0..32 {
        let Some(&word) = table.get(node as usize) else {
            return Err(bad("parametric stereo Huffman code left its table"));
        };
        let word = word as i32;
        node = if reader.read_bit()? {
            if word & 0x80 != 0 { word | !0x7f } else { word & 0x7f }
        } else {
            word >> 8
        };
        if node < 0 {
            return Ok(node + 64);
        }
    }
    Err(bad("parametric stereo Huffman code did not terminate"))
}

fn bad(message: &str) -> crate::error::Error {
    DecodeError::PsError(message.into()).into()
}
