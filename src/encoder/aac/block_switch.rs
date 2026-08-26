//! Block Switching and Transient Detection Subsystem
//!
//! Determines window sequence switching (OnlyLong, LongStart, EightShort, LongStop)
//! based on high-frequency energy ratio and perceptual attack thresholds.

use crate::types::{WindowSequence, WindowShape};

/// State tracker for transient detection and window sequence state machine.
#[derive(Debug, Clone, Default)]
pub struct BlockSwitching {
    prev_energy: f32,
    current_sequence: WindowSequence,
    current_shape: WindowShape,
}

impl BlockSwitching {
    /// Create a new block switching instance.
    pub fn new() -> Self {
        Self {
            prev_energy: 0.0,
            current_sequence: WindowSequence::OnlyLongSequence,
            current_shape: WindowShape::Sine,
        }
    }

    /// Analyze incoming time domain audio frame and determine optimal window sequence.
    pub fn analyze(&mut self, time_samples: &[f32]) -> (WindowSequence, WindowShape) {
        assert_eq!(time_samples.len(), 1024);

        // Partition 1024 samples into 8 sub-blocks of 128 samples
        let mut sub_energies = [0.0f32; 8];
        for (b, sub_energy) in sub_energies.iter_mut().enumerate() {
            let chunk = &time_samples[b * 128..(b + 1) * 128];
            let energy: f32 = chunk.iter().map(|&x| x * x).sum();
            *sub_energy = energy;
        }

        // Transient detection: attack ratio between consecutive sub-blocks
        let mut is_attack = false;
        for i in 1..8 {
            if sub_energies[i] > 3.0 * sub_energies[i - 1] + 1e-4 {
                is_attack = true;
                break;
            }
        }

        // Transition state machine
        self.current_sequence = match self.current_sequence {
            WindowSequence::OnlyLongSequence => {
                if is_attack {
                    WindowSequence::LongStartSequence
                } else {
                    WindowSequence::OnlyLongSequence
                }
            }
            WindowSequence::LongStartSequence => WindowSequence::EightShortSequence,
            WindowSequence::EightShortSequence => {
                if is_attack {
                    WindowSequence::EightShortSequence
                } else {
                    WindowSequence::LongStopSequence
                }
            }
            WindowSequence::LongStopSequence => {
                if is_attack {
                    WindowSequence::LongStartSequence
                } else {
                    WindowSequence::OnlyLongSequence
                }
            }
        };

        self.prev_energy = sub_energies[7];
        (self.current_sequence, self.current_shape)
    }
}
