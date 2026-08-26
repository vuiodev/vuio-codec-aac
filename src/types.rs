//! Core Domain Types, Enumerations, and Constants for MPEG Audio Codecs

use std::fmt;

/// MPEG Audio Object Types (AOT) defined in ISO/IEC 14496-3 and extended standards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AudioObjectType {
    Null = 0,
    AacMain = 1,
    AacLc = 2,
    AacSsr = 3,
    AacLtp = 4,
    Sbr = 5,
    AacScalable = 6,
    TwinVq = 7,
    Celp = 8,
    Hvxc = 9,
    Ttsi = 12,
    MainSynthetic = 13,
    Wavetable = 14,
    GeneralMidi = 15,
    Algorithmic = 16,
    ErAacLc = 17,
    ErAacLtp = 19,
    ErAacScalable = 20,
    ErTwinVq = 21,
    ErBsac = 22,
    ErAacLd = 23,
    ErCelp = 24,
    ErHvxc = 25,
    ErHiln = 26,
    ErParametric = 27,
    Ssc = 28,
    Ps = 29,
    MpegSurround = 30,
    Layer1 = 32,
    Layer2 = 33,
    Layer3 = 34,
    Dst = 35,
    Als = 36,
    Sls = 37,
    SlsNonCore = 38,
    ErAacEld = 39,
    SmrSimple = 40,
    SmrMain = 41,
    Usac = 42,
    Saoc = 43,
    LdMpegSurround = 44,
    UsacDrc = 45,
}

impl AudioObjectType {
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Null),
            1 => Some(Self::AacMain),
            2 => Some(Self::AacLc),
            3 => Some(Self::AacSsr),
            4 => Some(Self::AacLtp),
            5 => Some(Self::Sbr),
            6 => Some(Self::AacScalable),
            7 => Some(Self::TwinVq),
            8 => Some(Self::Celp),
            9 => Some(Self::Hvxc),
            12 => Some(Self::Ttsi),
            13 => Some(Self::MainSynthetic),
            14 => Some(Self::Wavetable),
            15 => Some(Self::GeneralMidi),
            16 => Some(Self::Algorithmic),
            17 => Some(Self::ErAacLc),
            19 => Some(Self::ErAacLtp),
            20 => Some(Self::ErAacScalable),
            21 => Some(Self::ErTwinVq),
            22 => Some(Self::ErBsac),
            23 => Some(Self::ErAacLd),
            24 => Some(Self::ErCelp),
            25 => Some(Self::ErHvxc),
            26 => Some(Self::ErHiln),
            27 => Some(Self::ErParametric),
            28 => Some(Self::Ssc),
            29 => Some(Self::Ps),
            30 => Some(Self::MpegSurround),
            32 => Some(Self::Layer1),
            33 => Some(Self::Layer2),
            34 => Some(Self::Layer3),
            35 => Some(Self::Dst),
            36 => Some(Self::Als),
            37 => Some(Self::Sls),
            38 => Some(Self::SlsNonCore),
            39 => Some(Self::ErAacEld),
            40 => Some(Self::SmrSimple),
            41 => Some(Self::SmrMain),
            42 => Some(Self::Usac),
            43 => Some(Self::Saoc),
            44 => Some(Self::LdMpegSurround),
            45 => Some(Self::UsacDrc),
            _ => None,
        }
    }

    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// Whether this Audio Object Type belongs to the Low Delay family (AAC-LD / AAC-ELD).
    pub const fn is_low_delay(self) -> bool {
        matches!(self, Self::ErAacLd | Self::ErAacEld)
    }

    /// Whether this Audio Object Type is USAC (Unified Speech and Audio Coding).
    pub const fn is_usac(self) -> bool {
        matches!(self, Self::Usac | Self::UsacDrc)
    }
}

/// Standard MPEG sampling rates and their table indices (0..12).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SamplingRate {
    Hz96000,
    Hz88200,
    Hz64000,
    Hz48000,
    Hz44100,
    Hz32000,
    Hz24000,
    Hz22050,
    Hz16000,
    Hz12000,
    Hz11025,
    Hz8000,
    Hz7350,
    Custom(u32),
}

impl SamplingRate {
    /// Standard sampling rate table in descending order.
    pub const RATES: [u32; 13] = [
        96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350,
    ];

    /// Convert from standard 4-bit sampling frequency index (0..12).
    pub const fn from_index(index: u8) -> Option<Self> {
        match index {
            0 => Some(Self::Hz96000),
            1 => Some(Self::Hz88200),
            2 => Some(Self::Hz64000),
            3 => Some(Self::Hz48000),
            4 => Some(Self::Hz44100),
            5 => Some(Self::Hz32000),
            6 => Some(Self::Hz24000),
            7 => Some(Self::Hz22050),
            8 => Some(Self::Hz16000),
            9 => Some(Self::Hz12000),
            10 => Some(Self::Hz11025),
            11 => Some(Self::Hz8000),
            12 => Some(Self::Hz7350),
            _ => None,
        }
    }

    /// Return the standard 4-bit index if this rate is one of the standard 13 rates.
    pub const fn to_index(self) -> Option<u8> {
        match self {
            Self::Hz96000 => Some(0),
            Self::Hz88200 => Some(1),
            Self::Hz64000 => Some(2),
            Self::Hz48000 => Some(3),
            Self::Hz44100 => Some(4),
            Self::Hz32000 => Some(5),
            Self::Hz24000 => Some(6),
            Self::Hz22050 => Some(7),
            Self::Hz16000 => Some(8),
            Self::Hz12000 => Some(9),
            Self::Hz11025 => Some(10),
            Self::Hz8000 => Some(11),
            Self::Hz7350 => Some(12),
            Self::Custom(rate) => {
                let mut i = 0;
                while i < 13 {
                    if Self::RATES[i] == rate {
                        return Some(i as u8);
                    }
                    i += 1;
                }
                None
            }
        }
    }

    /// Convert from integer Hertz value.
    pub const fn from_hz(hz: u32) -> Self {
        match hz {
            96000 => Self::Hz96000,
            88200 => Self::Hz88200,
            64000 => Self::Hz64000,
            48000 => Self::Hz48000,
            44100 => Self::Hz44100,
            32000 => Self::Hz32000,
            24000 => Self::Hz24000,
            22050 => Self::Hz22050,
            16000 => Self::Hz16000,
            12000 => Self::Hz12000,
            11025 => Self::Hz11025,
            8000 => Self::Hz8000,
            7350 => Self::Hz7350,
            other => Self::Custom(other),
        }
    }

    /// Return sample rate in Hertz as `u32`.
    pub const fn hz(self) -> u32 {
        match self {
            Self::Hz96000 => 96000,
            Self::Hz88200 => 88200,
            Self::Hz64000 => 64000,
            Self::Hz48000 => 48000,
            Self::Hz44100 => 44100,
            Self::Hz32000 => 32000,
            Self::Hz24000 => 24000,
            Self::Hz22050 => 22050,
            Self::Hz16000 => 16000,
            Self::Hz12000 => 12000,
            Self::Hz11025 => 11025,
            Self::Hz8000 => 8000,
            Self::Hz7350 => 7350,
            Self::Custom(rate) => rate,
        }
    }
}

impl fmt::Display for SamplingRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} Hz", self.hz())
    }
}

/// Standard MPEG channel configurations (0..7, 11..13).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ChannelConfiguration {
    CustomPce = 0,
    Mono = 1,
    Stereo = 2,
    ThreeChannel = 3,
    FourChannel = 4,
    FiveChannel = 5,
    FivePointOne = 6,
    SevenPointOne = 7,
    SixPointOne = 11,
    SevenPointOneTop = 12,
    TwentyTwoPointTwo = 13,
}

impl ChannelConfiguration {
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::CustomPce),
            1 => Some(Self::Mono),
            2 => Some(Self::Stereo),
            3 => Some(Self::ThreeChannel),
            4 => Some(Self::FourChannel),
            5 => Some(Self::FiveChannel),
            6 => Some(Self::FivePointOne),
            7 => Some(Self::SevenPointOne),
            11 => Some(Self::SixPointOne),
            12 => Some(Self::SevenPointOneTop),
            13 => Some(Self::TwentyTwoPointTwo),
            _ => None,
        }
    }

    pub const fn channels(self) -> usize {
        match self {
            Self::CustomPce => 0,
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::ThreeChannel => 3,
            Self::FourChannel => 4,
            Self::FiveChannel => 5,
            Self::FivePointOne => 6,
            Self::SevenPointOne => 8,
            Self::SixPointOne => 7,
            Self::SevenPointOneTop => 8,
            Self::TwentyTwoPointTwo => 24,
        }
    }
}

/// Window Sequence transitions defined in MPEG AAC (ISO/IEC 14496-3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum WindowSequence {
    #[default]
    OnlyLongSequence = 0,
    LongStartSequence = 1,
    EightShortSequence = 2,
    LongStopSequence = 3,
}

impl WindowSequence {
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::OnlyLongSequence),
            1 => Some(Self::LongStartSequence),
            2 => Some(Self::EightShortSequence),
            3 => Some(Self::LongStopSequence),
            _ => None,
        }
    }

    /// Number of window transforms in this frame (1 for long/start/stop, 8 for short).
    pub const fn num_windows(self) -> usize {
        match self {
            Self::EightShortSequence => 8,
            _ => 1,
        }
    }

    /// Whether this sequence is eight short windows.
    pub const fn is_eight_short(self) -> bool {
        matches!(self, Self::EightShortSequence)
    }
}

/// Window shape selection: Sine window vs Kaiser-Bessel Derived (KBD) window vs Low Delay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum WindowShape {
    #[default]
    Sine = 0,
    Kbd = 1,
    LowDelay = 2,
}

impl WindowShape {
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Sine),
            1 => Some(Self::Kbd),
            2 => Some(Self::LowDelay),
            _ => None,
        }
    }
}

/// Frame length in samples per channel (1024, 960, 512, 480, 768).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrameLength {
    Samples1024,
    Samples960,
    Samples512,
    Samples480,
    Samples768,
}

impl FrameLength {
    pub const fn samples(self) -> usize {
        match self {
            Self::Samples1024 => 1024,
            Self::Samples960 => 960,
            Self::Samples512 => 512,
            Self::Samples480 => 480,
            Self::Samples768 => 768,
        }
    }

    pub const fn short_samples(self) -> usize {
        match self {
            Self::Samples1024 => 128,
            Self::Samples960 => 120,
            Self::Samples512 => 64,
            Self::Samples480 => 60,
            Self::Samples768 => 96,
        }
    }
}

/// Audio container / transport stream format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BitstreamFormat {
    Raw,
    Adts,
    Adif,
    Latm,
}
