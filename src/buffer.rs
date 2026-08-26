//! High-Performance Cache-Line Aligned Audio Buffer Abstractions
//!
//! Provides zero-allocation, 64-byte aligned buffers matching AVX-512 and CPU cache lines
//! for seamless SIMD auto-vectorization and high-throughput multi-channel DSP.

use aligned_vec::AVec;

/// 64-byte aligned vector suitable for AVX-512, AVX2, and ARM NEON SIMD operations.
pub type AlignedVec<T> = AVec<T>;

/// Create a 64-byte aligned buffer with pre-allocated capacity.
pub fn alloc_aligned<T: Default + Clone>(len: usize) -> AlignedVec<T> {
    let mut vec = AlignedVec::new(64);
    vec.resize(len, T::default());
    vec
}

/// Multi-channel planar audio buffer stored in contiguous 64-byte aligned memory.
#[derive(Debug, Clone)]
pub struct AudioBuffer<T: Clone + Default> {
    channels: usize,
    samples_per_channel: usize,
    data: AlignedVec<T>,
}

impl<T: Clone + Default + Copy> AudioBuffer<T> {
    /// Create a new multi-channel buffer with pre-allocated memory.
    pub fn new(channels: usize, samples_per_channel: usize) -> Self {
        let total_samples = channels * samples_per_channel;
        let data = alloc_aligned(total_samples);
        Self {
            channels,
            samples_per_channel,
            data,
        }
    }

    /// Number of audio channels in this buffer.
    #[inline(always)]
    pub const fn channels(&self) -> usize {
        self.channels
    }

    /// Number of audio samples per channel.
    #[inline(always)]
    pub const fn samples_per_channel(&self) -> usize {
        self.samples_per_channel
    }

    /// Total number of samples across all channels (`channels * samples_per_channel`).
    #[inline(always)]
    pub const fn total_samples(&self) -> usize {
        self.channels * self.samples_per_channel
    }

    /// Get immutable slice for a specific channel.
    #[inline(always)]
    pub fn channel(&self, ch: usize) -> &[T] {
        assert!(ch < self.channels, "Channel index out of bounds");
        let start = ch * self.samples_per_channel;
        let end = start + self.samples_per_channel;
        &self.data[start..end]
    }

    /// Get mutable slice for a specific channel.
    #[inline(always)]
    pub fn channel_mut(&mut self, ch: usize) -> &mut [T] {
        assert!(ch < self.channels, "Channel index out of bounds");
        let start = ch * self.samples_per_channel;
        let end = start + self.samples_per_channel;
        &mut self.data[start..end]
    }

    /// Clear all channel buffers to default (zero).
    #[inline]
    pub fn clear(&mut self) {
        self.data.fill(T::default());
    }

    /// Resize buffer capacity (reusing allocation if possible).
    pub fn resize(&mut self, channels: usize, samples_per_channel: usize) {
        self.channels = channels;
        self.samples_per_channel = samples_per_channel;
        let total = channels * samples_per_channel;
        self.data.resize(total, T::default());
    }
}

impl AudioBuffer<i16> {
    /// Convert planar channel audio to interleaved 16-bit PCM output.
    pub fn to_interleaved(&self, output: &mut [i16]) {
        assert!(
            output.len() >= self.total_samples(),
            "Output buffer too small for interleaved PCM"
        );
        let num_ch = self.channels;
        let num_samples = self.samples_per_channel;

        for s in 0..num_samples {
            for c in 0..num_ch {
                output[s * num_ch + c] = self.data[c * num_samples + s];
            }
        }
    }

    /// Fill planar channel audio from interleaved 16-bit PCM input.
    pub fn from_interleaved(&mut self, input: &[i16]) {
        assert!(
            input.len() >= self.total_samples(),
            "Input buffer too small for interleaved PCM"
        );
        let num_ch = self.channels;
        let num_samples = self.samples_per_channel;

        for s in 0..num_samples {
            for c in 0..num_ch {
                self.data[c * num_samples + s] = input[s * num_ch + c];
            }
        }
    }
}

impl AudioBuffer<f32> {
    /// Convert 32-bit floating point planar audio to interleaved 16-bit signed integer PCM.
    pub fn to_interleaved_i16(&self, output: &mut [i16]) {
        assert!(
            output.len() >= self.total_samples(),
            "Output buffer too small for interleaved PCM"
        );
        let num_ch = self.channels;
        let num_samples = self.samples_per_channel;

        for s in 0..num_samples {
            for c in 0..num_ch {
                let sample_f32 = self.data[c * num_samples + s];
                // Scale normalized [-1.0, 1.0] to [-32768, 32767] with saturation
                let scaled = (sample_f32 * 32768.0).round();
                let clamped = scaled.clamp(-32768.0, 32767.0) as i16;
                output[s * num_ch + c] = clamped;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_buffer_allocation() {
        let buf: AlignedVec<i32> = alloc_aligned(1024);
        assert_eq!(buf.len(), 1024);
        assert_eq!(buf.as_ptr() as usize % 64, 0, "Buffer must be 64-byte aligned");
    }

    #[test]
    fn test_audio_buffer_planar_and_interleaved() {
        let mut audio_buf = AudioBuffer::<i16>::new(2, 4);
        let left = audio_buf.channel_mut(0);
        left.copy_from_slice(&[10, 20, 30, 40]);
        let right = audio_buf.channel_mut(1);
        right.copy_from_slice(&[-10, -20, -30, -40]);

        let mut interleaved = [0i16; 8];
        audio_buf.to_interleaved(&mut interleaved);
        assert_eq!(interleaved, [10, -10, 20, -20, 30, -30, 40, -40]);

        let mut reconstructed = AudioBuffer::<i16>::new(2, 4);
        reconstructed.from_interleaved(&interleaved);
        assert_eq!(reconstructed.channel(0), &[10, 20, 30, 40]);
        assert_eq!(reconstructed.channel(1), &[-10, -20, -30, -40]);
    }
}
