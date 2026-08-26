//! AAC Syntactic Channel Element Demuxing and Frame Orchestration
//!
//! Parses SCE, CPE, CCE, LFE, DSE, and FIL syntactic elements from MPEG bitstreams.

use crate::bitstream::BitReader;
use crate::error::Result;
use crate::types::{WindowSequence, WindowShape};

/// Syntactic element identifiers (3-bit ID in bitstream).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ElementType {
    Sce = 0, // Single Channel Element
    Cpe = 1, // Channel Pair Element
    Cce = 2, // Channel Coupling Element
    Lfe = 3, // Low Frequency Element
    Dse = 4, // Data Stream Element
    Pce = 5, // Program Config Element
    Fil = 6, // Fill Element
    End = 7, // Frame Terminator
}

impl ElementType {
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Sce),
            1 => Some(Self::Cpe),
            2 => Some(Self::Cce),
            3 => Some(Self::Lfe),
            4 => Some(Self::Dse),
            5 => Some(Self::Pce),
            6 => Some(Self::Fil),
            7 => Some(Self::End),
            _ => None,
        }
    }
}

/// Single channel individual stream info (window sequence, shape, max sfb).
#[derive(Debug, Clone)]
pub struct IcsInfo {
    pub window_sequence: WindowSequence,
    pub window_shape: WindowShape,
    pub max_sfb: u8,
    pub scale_factor_grouping: u8,
}

impl IcsInfo {
    /// Parse Individual Channel Stream info from bitstream.
    pub fn parse(reader: &mut BitReader, _common_window: bool) -> Result<Self> {
        let _ics_reserved = reader.read_bit()?;
        let win_seq_id = reader.read_u8(2)?;
        let window_sequence = WindowSequence::from_u8(win_seq_id).unwrap_or(WindowSequence::OnlyLongSequence);
        let win_shape_id = reader.read_u8(1)?;
        let window_shape = WindowShape::from_u8(win_shape_id).unwrap_or(WindowShape::Sine);

        let max_sfb = if window_sequence.is_eight_short() {
            reader.read_u8(4)?
        } else {
            let sfb = reader.read_u8(6)?;
            let _predictor_data_present = reader.read_bit()?;
            sfb
        };

        let scale_factor_grouping = if window_sequence.is_eight_short() {
            reader.read_u8(7)?
        } else {
            0
        };

        Ok(Self {
            window_sequence,
            window_shape,
            max_sfb,
            scale_factor_grouping,
        })
    }
}

/// Parsed Single Channel Element (SCE).
#[derive(Debug, Clone)]
pub struct SingleChannelElement {
    pub element_instance_tag: u8,
    pub ics_info: IcsInfo,
    pub spectral_coefficients: Vec<f32>,
}

/// Parsed Channel Pair Element (CPE).
#[derive(Debug, Clone)]
pub struct ChannelPairElement {
    pub element_instance_tag: u8,
    pub common_window: bool,
    pub ms_mask_present: u8,
    pub ics_info_left: IcsInfo,
    pub ics_info_right: Option<IcsInfo>,
    pub spectral_left: Vec<f32>,
    pub spectral_right: Vec<f32>,
}
