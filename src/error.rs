//! Structured Error Hierarchy for vuiocodecaac Audio Codec
//!
//! Provides comprehensive, strongly typed errors for bitstream demuxing, decoding,
//! encoding, DSP transforms, and container parsing using `thiserror`.

use thiserror::Error;

/// Root error type for all operations within the `vuiocodecaac` crate.
#[derive(Debug, Error)]
pub enum Error {
    /// Bitstream parsing and extraction errors.
    #[error("Bitstream error: {0}")]
    Bitstream(#[from] BitstreamError),

    /// Audio decoding pipeline errors.
    #[error("Decode error: {0}")]
    Decode(#[from] DecodeError),

    /// Audio encoding pipeline errors.
    #[error("Encode error: {0}")]
    Encode(#[from] EncodeError),

    /// Container / format parsing errors.
    #[error("Format error: {0}")]
    Format(#[from] FormatError),

    /// DSP mathematical or transform errors.
    #[error("DSP error: {0}")]
    Dsp(#[from] DspError),

    /// Underlying standard I/O errors.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A codec tool that exists in the reference but is not implemented here.
    ///
    /// This is deliberately a hard error rather than a silent approximation:
    /// a wrong answer that looks right is the worst failure mode a codec has,
    /// so a tool this port has not reached yet refuses the work instead of
    /// fabricating output. `text/plan.txt` tracks which tools these are and
    /// which phase implements each.
    #[error("{tool} is not implemented in this port yet ({detail})")]
    Unimplemented {
        /// The tool the caller asked for, e.g. "MPEG Surround decode".
        tool: &'static str,
        /// Where to look: usually the plan phase that covers it.
        detail: &'static str,
    },
}

/// Errors occurring during bitstream bit-level extraction and writing.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BitstreamError {
    #[error("Unexpected end of bitstream (needed {needed_bits} bits, available {available_bits})")]
    UnexpectedEof {
        needed_bits: usize,
        available_bits: usize,
    },

    #[error("Invalid syncword: expected {expected:#06x}, found {found:#06x}")]
    InvalidSyncword { expected: u16, found: u16 },

    #[error("CRC-16 mismatch: expected {expected:#06x}, calculated {calculated:#06x}")]
    CrcMismatch { expected: u16, calculated: u16 },

    #[error("Invalid bit count requested: {0} bits (maximum supported in single read is 64)")]
    InvalidBitCount(usize),

    #[error("Byte alignment error: stream is not properly aligned")]
    UnalignedAccess,
}

/// Errors occurring during AAC, SBR, PS, USAC, or DRC decoding.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DecodeError {
    #[error("Invalid Audio Object Type (AOT): {0}")]
    InvalidAudioObjectType(u8),

    #[error("Unsupported Audio Object Type: {0:?}")]
    UnsupportedAudioObjectType(String),

    #[error("Invalid sampling frequency index: {0}")]
    InvalidSamplingFrequencyIndex(u8),

    #[error("Invalid channel configuration: {0}")]
    InvalidChannelConfiguration(u8),

    #[error("Huffman decoding error: invalid codeword for codebook {codebook} (bits: {bits:#x})")]
    HuffmanDecodeError { codebook: u8, bits: u32 },

    #[error("Scalefactor out of valid range: {0} (valid: 0..=255)")]
    ScalefactorOutOfRange(i32),

    #[error("max_sfb {max_sfb} exceeds the {num_swb} bands available at this rate")]
    InvalidMaxSfb { max_sfb: u8, num_swb: u8 },

    #[error("section_data declared a zero-length section")]
    InvalidSectionLength,

    #[error("TNS filter order {0} exceeds the maximum for this window sequence")]
    InvalidTnsOrder(u8),

    #[error("Invalid window sequence transition from {previous:?} to {current:?}")]
    InvalidWindowTransition {
        previous: String,
        current: String,
    },

    #[error("TNS (Temporal Noise Shaping) order exceeds maximum: {order} > {max_order}")]
    TnsOrderExceeded { order: usize, max_order: usize },

    #[error("SBR payload error: {0}")]
    SbrError(String),

    #[error("Parametric Stereo (PS) error: {0}")]
    PsError(String),

    #[error("MPEG Surround (MPS) error: {0}")]
    MpsError(String),

    #[error("USAC decoding error: {0}")]
    UsacError(String),

    #[error("UniDRC decoding error: {0}")]
    DrcError(String),

    #[error("Corrupted frame payload: {0}")]
    CorruptedFrame(String),
}

/// Errors occurring during audio encoding.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum EncodeError {
    #[error("Unsupported sampling rate: {0} Hz")]
    UnsupportedSampleRate(u32),

    #[error("Unsupported channel count: {0}")]
    UnsupportedChannels(usize),

    #[error("Target bitrate {bitrate} bps out of valid range ({min_bitrate}..={max_bitrate} bps)")]
    BitrateOutOfRange {
        bitrate: u32,
        min_bitrate: u32,
        max_bitrate: u32,
    },

    #[error("Input buffer size {provided} does not match required frame length {required}")]
    InvalidInputSize { provided: usize, required: usize },

    #[error("Quantizer failed to converge within bit reservoir constraints")]
    QuantizerConvergenceFailure,

    #[error("Encoder configuration error: {0}")]
    InvalidConfig(String),
}

/// Errors occurring during container and framing parsing (ADTS, ADIF, LATM).
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum FormatError {
    #[error("Invalid ADTS header: {0}")]
    InvalidAdts(String),

    #[error("Invalid ADIF header: {0}")]
    InvalidAdif(String),

    #[error("Invalid LATM/LOAS payload: {0}")]
    InvalidLatm(String),

    #[error("Invalid AudioSpecificConfig (ASC): {0}")]
    InvalidAsc(String),

    #[error("Invalid Program Config Element (PCE): {0}")]
    InvalidPce(String),

    #[error("Unrecognized stream container format")]
    UnrecognizedContainer,

    #[error("Invalid USAC FD container: {0}")]
    InvalidUsacContainer(String),
}

/// Errors occurring in DSP transforms or filterbanks.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DspError {
    #[error("Invalid transform length {0} (must be power-of-two or supported mixed-radix size)")]
    InvalidTransformLength(usize),

    #[error("Buffer length mismatch: expected {expected}, got {actual}")]
    BufferLengthMismatch { expected: usize, actual: usize },

    #[error("Filterbank initialization failure: {0}")]
    FilterbankError(String),
}

/// Result alias for `vuiocodecaac` operations.
pub type Result<T> = std::result::Result<T, Error>;
