//! Deciding when to switch to short windows.
//!
//! A 1024-line transform buys frequency resolution at the cost of spreading its
//! quantization noise over 2048 samples. On a sudden attack that noise is audible
//! *before* the attack, in the quiet that preceded it — a pre-echo. Eight 128-line
//! transforms confine the damage to an eighth of the frame, at the cost of the
//! resolution that makes steady material cheap to code.
//!
//! # Finding an attack
//!
//! The energy of a frame's eight sub-blocks is compared against a running average of
//! what came before, after a high-pass filter that removes the low-frequency energy a
//! sustained note carries and leaves the edges. A sub-block whose filtered energy
//! stands far enough above that average is an attack. A floor on the absolute energy
//! keeps the ratio test from firing on the noise at the start of a fade-in.
//!
//! # Why a frame of lookahead
//!
//! Short windows cannot follow a long one directly: the two window halves that meet
//! at the boundary have to be the same shape or the transform's aliasing does not
//! cancel. A frame of short windows therefore has to be announced by a *start*
//! window in the frame before it, which means the decision needs to see one frame
//! further than it codes. [`BlockSwitch::decide`] takes both, and the encoder holds
//! a frame back to supply them.

use crate::types::WindowSequence;

/// Sub-blocks a frame is divided into, which is also the number of short windows.
pub const SUB_BLOCKS: usize = 8;
/// Groups the short-window grouping table produces.
pub const MAX_GROUPS: usize = 4;

/// High-pass filter the detector measures through: `y[n] = b*(x[n] - x[n-1]) - a*y[n-1]`.
const HIGHPASS: [f32; 2] = [-0.5095, 0.7548];
/// Weight the running average gives the newest sub-block.
const AVERAGE_WEIGHT: f32 = 0.3;
/// How far a sub-block must stand above the running average to count as an attack,
/// as its reciprocal, at a generous bitrate.
const ATTACK_RATIO_HIGH_RATE: f32 = 0.1;
/// The same at a tight one, where short windows cost proportionally more.
const ATTACK_RATIO_LOW_RATE: f32 = 0.056;
/// Bitrate per channel above which the looser ratio applies.
const HIGH_RATE_PER_CHANNEL: u32 = 16_000;
/// Absolute energy a sub-block needs before the ratio test is trusted at all.
const MIN_ATTACK_ENERGY: f32 = 1.0e6;

/// How the eight short windows are grouped, by where the attack landed.
///
/// The group carrying the attack is kept short so its scalefactors can follow the
/// transient, while the windows either side are pooled to keep the side information
/// down.
const GROUPING: [[usize; MAX_GROUPS]; SUB_BLOCKS] = [
    [1, 3, 3, 1],
    [1, 1, 3, 3],
    [2, 1, 3, 2],
    [3, 1, 3, 1],
    [3, 1, 1, 3],
    [3, 2, 1, 2],
    [3, 3, 1, 1],
    [3, 3, 1, 1],
];

/// What the detector found in one frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transient {
    /// Whether the frame carries an attack.
    pub attack: bool,
    /// Which sub-block it landed in.
    pub sub_block: usize,
}

/// The window sequence for a frame, and how its short windows are grouped.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockDecision {
    pub sequence: WindowSequence,
    /// Windows in each group; one group of one for anything but eight short.
    pub groups: [usize; SUB_BLOCKS],
    /// Groups in use.
    pub group_count: usize,
}

impl BlockDecision {
    /// The `scale_factor_grouping` field, seven bits saying where groups begin.
    ///
    /// Bit `6 - i` is clear when short window `i + 1` starts a new group.
    pub fn grouping_bits(&self) -> u8 {
        if self.sequence != WindowSequence::EightShortSequence {
            return 0;
        }
        let mut bits = 0u8;
        let mut index = 0usize;
        for g in 0..self.group_count {
            for w in 0..self.groups[g] {
                if index > 0 && w > 0 {
                    bits |= 1 << (7 - index);
                }
                index += 1;
            }
        }
        bits
    }
}

/// Finds attacks and turns them into window sequences.
#[derive(Debug, Clone)]
pub struct BlockSwitch {
    /// High-pass filter state: the last input and the last output.
    filter: [f32; 2],
    /// Running average of the filtered sub-block energies.
    average: f32,
    /// Filtered energy of the frame's last sub-block, which the next frame's first
    /// sub-block is compared against.
    previous_energy: f32,
    /// Reciprocal of the ratio an attack has to beat.
    inverse_ratio: f32,
    /// Sequence the previous frame used.
    previous: WindowSequence,
    /// Whether the previous frame carried an attack, so that one landing on the very
    /// last sub-block extends into this frame.
    trailing_attack: bool,
    /// Where the last attack landed.
    last_sub_block: usize,
}

impl BlockSwitch {
    /// Build a detector for a given bitrate per channel.
    pub fn new(bitrate_per_channel_bps: u32) -> Self {
        Self {
            filter: [0.0; 2],
            average: 0.0,
            previous_energy: 0.0,
            inverse_ratio: if bitrate_per_channel_bps > HIGH_RATE_PER_CHANNEL {
                ATTACK_RATIO_HIGH_RATE
            } else {
                ATTACK_RATIO_LOW_RATE
            },
            previous: WindowSequence::OnlyLongSequence,
            trailing_attack: false,
            last_sub_block: 0,
        }
    }

    /// Forget everything, as at the start of a stream.
    pub fn reset(&mut self) {
        let ratio = self.inverse_ratio;
        *self = Self::new(0);
        self.inverse_ratio = ratio;
    }

    /// Measure one frame and say whether it carries an attack.
    ///
    /// The frame's samples are consumed in order, so the filter state carries across
    /// frame boundaries and an attack straddling one is still found.
    pub fn analyse(&mut self, samples: &[f32]) -> Transient {
        let length = (samples.len() / SUB_BLOCKS).max(1);
        let mut energies = [0.0f32; SUB_BLOCKS];
        let mut peak = 0.0f32;

        for (w, energy) in energies.iter_mut().enumerate() {
            let lo = w * length;
            let hi = ((w + 1) * length).min(samples.len());
            let mut sum = 0.0f32;
            for &x in &samples[lo..hi] {
                let filtered = self.highpass(x);
                sum += filtered * filtered;
            }
            *energy = sum;
            peak = peak.max(sum);
        }

        let mut attack = false;
        let mut sub_block = self.last_sub_block;
        let mut previous = self.previous_energy;
        for (w, &energy) in energies.iter().enumerate() {
            self.average = (1.0 - AVERAGE_WEIGHT) * self.average + AVERAGE_WEIGHT * previous;
            if energy * self.inverse_ratio > self.average {
                attack = true;
                sub_block = w;
            }
            previous = energy;
        }
        self.previous_energy = previous;

        // A ratio test on near-silence finds an attack in every fade-in; the floor
        // is what stops that.
        if peak < MIN_ATTACK_ENERGY {
            attack = false;
        }

        // An attack on the very last sub-block is still rising when the frame ends,
        // so the frame after it is treated as carrying one too.
        if !attack && self.trailing_attack && self.last_sub_block == SUB_BLOCKS - 1 {
            attack = true;
            sub_block = 0;
        }
        self.trailing_attack = attack;
        self.last_sub_block = sub_block;

        Transient { attack, sub_block }
    }

    /// Choose the sequence for the frame now being coded.
    ///
    /// `here` describes that frame and `next` the one after it, which is what lets a
    /// start window be placed before short windows rather than after them.
    pub fn decide(&mut self, here: Transient, next: Transient) -> BlockDecision {
        use WindowSequence::*;
        let sequence = match self.previous {
            // A start window has promised short windows; the promise has to be kept
            // whatever the detector now says, or the aliasing will not cancel.
            LongStartSequence => EightShortSequence,
            EightShortSequence if here.attack => EightShortSequence,
            EightShortSequence => LongStopSequence,
            _ if next.attack => LongStartSequence,
            _ => OnlyLongSequence,
        };
        self.previous = sequence;

        let mut groups = [1usize; SUB_BLOCKS];
        let mut group_count = 1;
        if sequence == EightShortSequence {
            let table = GROUPING[here.sub_block.min(SUB_BLOCKS - 1)];
            groups = [0; SUB_BLOCKS];
            groups[..MAX_GROUPS].copy_from_slice(&table);
            group_count = MAX_GROUPS;
        }

        BlockDecision { sequence, groups, group_count }
    }

    /// The sequence the previous frame used.
    #[inline]
    pub fn previous(&self) -> WindowSequence {
        self.previous
    }

    /// One sample through the high-pass filter.
    #[inline]
    fn highpass(&mut self, input: f32) -> f32 {
        let out = HIGHPASS[1] * (input - self.filter[0]) - HIGHPASS[0] * self.filter[1];
        self.filter[0] = input;
        self.filter[1] = out;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn steady(n: usize) -> Vec<f32> {
        (0..n).map(|i| 8000.0 * (i as f32 * 0.05).sin()).collect()
    }

    fn attack_at(n: usize, position: usize) -> Vec<f32> {
        (0..n)
            .map(|i| if i < position { 10.0 } else { 20000.0 * (i as f32 * 0.7).sin() })
            .collect()
    }

    /// Steady material must never ask for short windows.
    #[test]
    fn steady_material_stays_long() {
        let mut switcher = BlockSwitch::new(64000);
        let frame = steady(1024);
        // The first frame always looks like an attack against a silent history.
        switcher.analyse(&frame);
        for _ in 0..8 {
            let t = switcher.analyse(&frame);
            assert!(!t.attack, "steady material was called an attack");
        }
    }

    /// A sudden onset must be found, and in the right sub-block.
    #[test]
    fn an_onset_is_found_where_it_happens() {
        let mut switcher = BlockSwitch::new(64000);
        switcher.analyse(&vec![10.0f32; 1024]);
        switcher.analyse(&vec![10.0f32; 1024]);
        let t = switcher.analyse(&attack_at(1024, 512));
        assert!(t.attack, "a step of 2000x was not detected");
        assert_eq!(t.sub_block, 4, "the attack was placed in the wrong sub-block");
    }

    /// Silence must not be mistaken for a transient.
    #[test]
    fn silence_is_not_an_attack() {
        let mut switcher = BlockSwitch::new(64000);
        for _ in 0..4 {
            let t = switcher.analyse(&vec![0.0f32; 1024]);
            assert!(!t.attack);
        }
    }

    /// Every sequence a run of decisions produces must be one the next can follow.
    #[test]
    fn sequences_always_join_up() {
        use WindowSequence::*;
        let mut switcher = BlockSwitch::new(64000);
        let quiet = Transient { attack: false, sub_block: 0 };
        let loud = Transient { attack: true, sub_block: 3 };

        // Every pattern of attacks over eight frames.
        for pattern in 0u32..256 {
            let mut switcher = switcher.clone();
            switcher.previous = OnlyLongSequence;
            let attacks: Vec<Transient> =
                (0..9).map(|i| if pattern >> i & 1 == 1 { loud } else { quiet }).collect();

            let mut previous = OnlyLongSequence;
            for i in 0..8 {
                let decision = switcher.decide(attacks[i], attacks[i + 1]);
                let ok = match previous {
                    OnlyLongSequence | LongStopSequence => {
                        matches!(decision.sequence, OnlyLongSequence | LongStartSequence)
                    }
                    LongStartSequence | EightShortSequence => {
                        matches!(decision.sequence, EightShortSequence | LongStopSequence)
                    }
                };
                assert!(ok, "pattern {pattern:08b}: {previous:?} cannot be followed by {:?}", decision.sequence);
                previous = decision.sequence;
            }
        }
        let _ = switcher.analyse(&steady(1024));
    }

    /// Grouping must always cover all eight windows.
    #[test]
    fn groups_cover_every_window() {
        for position in 0..SUB_BLOCKS {
            let total: usize = GROUPING[position].iter().sum();
            assert_eq!(total, SUB_BLOCKS, "grouping for sub-block {position} covers {total}");
        }
    }

    /// The grouping field must say exactly where the groups begin.
    #[test]
    fn grouping_bits_describe_the_groups() {
        for position in 0..SUB_BLOCKS {
            let mut groups = [0usize; SUB_BLOCKS];
            groups[..MAX_GROUPS].copy_from_slice(&GROUPING[position]);
            let decision = BlockDecision {
                sequence: WindowSequence::EightShortSequence,
                groups,
                group_count: MAX_GROUPS,
            };
            let bits = decision.grouping_bits();

            // Decode the field the way the decoder does and compare.
            let mut lengths = Vec::new();
            let mut run = 1usize;
            for i in 0..7 {
                if bits >> (6 - i) & 1 == 1 {
                    run += 1;
                } else {
                    lengths.push(run);
                    run = 1;
                }
            }
            lengths.push(run);
            assert_eq!(lengths, GROUPING[position].to_vec(), "sub-block {position}");
        }
    }
}
