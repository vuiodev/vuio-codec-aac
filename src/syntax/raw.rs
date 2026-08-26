//! Raw AAC Bitstream Container
//!
//! Provides bare raw AAC frame payload wrapping with AudioSpecificConfig metadata.

use crate::syntax::asc::AudioSpecificConfig;

/// Raw AAC Audio Payload with associated AudioSpecificConfig.
#[derive(Debug, Clone, PartialEq)]
pub struct RawAudioFrame {
    pub config: AudioSpecificConfig,
    pub raw_data: Vec<u8>,
}

impl RawAudioFrame {
    pub fn new(config: AudioSpecificConfig, raw_data: Vec<u8>) -> Self {
        Self { config, raw_data }
    }

    pub fn data(&self) -> &[u8] {
        &self.raw_data
    }

    pub fn config(&self) -> &AudioSpecificConfig {
        &self.config
    }
}
