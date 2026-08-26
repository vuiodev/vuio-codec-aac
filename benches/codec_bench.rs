//! Benchmarks for XAAC Codec Transforms and Pipeline

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use xaac::prelude::*;
use xaac::dsp::fft::FftContext;
use xaac::dsp::mdct::MdctContext;
use xaac::dsp::Complex32;

fn bench_fft(c: &mut Criterion) {
    let fft = FftContext::new(512);
    let mut data = vec![Complex32::default(); 512];

    c.bench_function("fft_512_forward", |b| {
        b.iter(|| {
            fft.forward(black_box(&mut data));
        })
    });
}

fn bench_mdct(c: &mut Criterion) {
    let mdct = MdctContext::new(1024);
    let input = vec![1.0f32; 1024];
    let mut output = vec![0.0f32; 2048];

    c.bench_function("imdct_1024", |b| {
        b.iter(|| {
            mdct.imdct(black_box(&input), black_box(&mut output));
        })
    });
}

fn bench_decoder(c: &mut Criterion) {
    let mut decoder = Decoder::new_default();

    // Construct valid ADTS frame with END element (7)
    let mut payload_writer = BitWriter::new();
    payload_writer.write_u8(7, 3); // END element
    let payload = payload_writer.into_bytes();

    let header = AdtsHeader {
        mpeg_id: 0,
        layer: 0,
        protection_absent: true,
        audio_object_type: AudioObjectType::AacLc,
        sampling_rate: SamplingRate::Hz44100,
        channel_config: ChannelConfiguration::Stereo,
        frame_length: 7 + payload.len(),
        buffer_fullness: 0x7FF,
        num_raw_data_blocks: 0,
        crc: None,
    };

    let mut adts_writer = BitWriter::new();
    header.write(&mut adts_writer);
    adts_writer.write_bytes(&payload);
    let adts_frame = adts_writer.into_bytes();

    c.bench_function("decode_frame_adts_stereo", |b| {
        b.iter(|| {
            let _ = decoder.decode_frame(black_box(&adts_frame));
        })
    });
}

fn bench_encoder(c: &mut Criterion) {
    let mut encoder = Encoder::new(EncoderConfig::default()).unwrap();
    let pcm = AudioBuffer::<i16>::new(2, 1024);

    c.bench_function("encode_frame_adts_stereo", |b| {
        b.iter(|| {
            let _ = encoder.encode_frame(black_box(&pcm));
        })
    });
}

criterion_group!(benches, bench_fft, bench_mdct, bench_decoder, bench_encoder);
criterion_main!(benches);
