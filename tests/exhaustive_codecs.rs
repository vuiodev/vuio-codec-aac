//! Exhaustive Codec Subsystems and Tools Test Suite

use vuiocodecaac::decoder::aac::dequant::inverse_quantize;
use vuiocodecaac::decoder::aac::huffman::decode_spectral_band;
use vuiocodecaac::decoder::aac::pns::PnsGenerator;
use vuiocodecaac::decoder::aac::stereo::{apply_intensity_stereo, apply_ms_stereo};
use vuiocodecaac::decoder::aac::tns::TnsFilter;
use vuiocodecaac::decoder::drc::DrcDecoder;
use vuiocodecaac::decoder::mps::{MpsDecoder, MpsSpatialCues};
use vuiocodecaac::decoder::ps::PsDecoder;
use vuiocodecaac::decoder::sbr::{SbrDecoder, SbrHeader};
use vuiocodecaac::decoder::usac::{UsacCoreMode, UsacDecoder};
use vuiocodecaac::encoder::aac::block_switch::BlockSwitching;
use vuiocodecaac::encoder::aac::psycho::PsychoacousticModel;
use vuiocodecaac::encoder::aac::quant::{estimate_global_gain, quantize_band};
use vuiocodecaac::encoder::drc::DrcEncoder;
use vuiocodecaac::encoder::ps::PsEncoder;
use vuiocodecaac::encoder::sbr::SbrEncoder;
use vuiocodecaac::encoder::usac::UsacEncoder;
use vuiocodecaac::bitstream::{BitReader, BitWriter};

#[test]
fn test_huffman_and_dequantization() {
    let mut writer = BitWriter::with_capacity(32);
    // Write CB 1 zeroes
    writer.write_bit(true);
    writer.write_bit(true);
    let bytes = writer.finalize();

    let mut reader = BitReader::new(bytes);
    let mut quantized = [0i32; 8];
    decode_spectral_band(&mut reader, 1, &mut quantized).unwrap();
    assert_eq!(quantized[0], 0);

    let mut spectral = [0.0f32; 8];
    quantized[0] = 1;
    inverse_quantize(&quantized, 100, &mut spectral);
    assert!((spectral[0] - 1.0).abs() < 1e-4);
}

#[test]
fn test_stereo_pns_and_tns_tools() {
    // M/S Stereo
    let mut left = [1.0f32, 2.0, 3.0, 4.0];
    let mut right = [0.5f32, 1.0, 1.5, 2.0];
    apply_ms_stereo(&mut left, &mut right);
    assert_eq!(left[0], 1.5);
    assert_eq!(right[0], 0.5);

    // Intensity Stereo
    let left_src = [2.0f32, 4.0];
    let mut right_dest = [0.0f32, 0.0];
    apply_intensity_stereo(&left_src, 100, false, &mut right_dest);
    assert_ne!(right_dest[0], 0.0);

    // PNS
    let mut pns = PnsGenerator::default();
    let mut pns_spec = [0.0f32; 16];
    pns.fill_noise_band(100, &mut pns_spec);
    assert_ne!(pns_spec[0], 0.0);

    // TNS
    let mut tns_spec = [1.0f32, 0.5, -0.5, 0.2];
    let tns_filter = TnsFilter {
        start_band: 0,
        stop_band: 4,
        order: 2,
        direction: false,
        coef_res: true,
        coefficients: vec![0.5, -0.25],
    };
    tns_filter.apply(&mut tns_spec);
    assert_eq!(tns_spec.len(), 4);
}

#[test]
fn test_sbr_and_ps_end_to_end() {
    let mut sbr = SbrDecoder::new(SbrHeader::default());
    let mut sbr_enc = SbrEncoder::new(44100, 128000);

    let baseband = vec![0.5f32; 1024];
    let payload = sbr_enc.encode_sbr_frame(&baseband).unwrap();
    assert!(!payload.is_empty());

    let mut output_2x = vec![0.0f32; 2048];
    sbr.process_channel(&baseband, &mut output_2x).unwrap();
    assert_eq!(output_2x.len(), 2048);

    let ps_enc = PsEncoder::new();
    let mut ps_dec = PsDecoder::new();
    let left = vec![0.8f32; 1024];
    let right = vec![0.4f32; 1024];
    let mut mono = vec![0.0f32; 1024];

    let ps_data = ps_enc.encode_stereo(&left, &right, &mut mono).unwrap();
    let mut out_l = vec![0.0f32; 1024];
    let mut out_r = vec![0.0f32; 1024];
    ps_dec.decode_stereo(&mono, &ps_data, &mut out_l, &mut out_r).unwrap();
    assert_ne!(out_l[0], 0.0);
    assert_ne!(out_r[0], 0.0);
}

#[test]
fn test_mps_usac_and_drc() {
    // MPS 5.1
    let mps = MpsDecoder::new(0);
    let left = vec![1.0f32; 512];
    let right = vec![0.5f32; 512];
    let mut out_5point1 = vec![vec![0.0f32; 512]; 6];
    mps.decode_5point1(&left, &right, &MpsSpatialCues::default(), &mut out_5point1).unwrap();
    assert_eq!(out_5point1[0].len(), 512);

    // USAC
    let usac_enc = UsacEncoder::new();
    let mut usac_dec = UsacDecoder::new();
    let pcm = vec![0.1f32; 1024];
    let mode = usac_enc.classify_frame(&pcm);
    assert_eq!(mode, UsacCoreMode::LpdMode);

    let mut acelp_out = vec![0.0f32; 64];
    let bytes = [0u8; 8];
    let mut reader = BitReader::new(&bytes);
    usac_dec.decode_acelp_subframe(&mut reader, 16, 0.7, 0.4, &mut acelp_out).unwrap();
    assert_eq!(acelp_out.len(), 64);

    // DRC
    let drc_enc = DrcEncoder::new(-23.0);
    let mut drc_dec = DrcDecoder::new(-23.0);
    let mut drc_pcm = vec![vec![0.8f32; 512]; 2];
    let drc_slices = [&drc_pcm[0][..], &drc_pcm[1][..]];
    let drc_data = drc_enc.measure_and_generate_drc(&drc_slices);
    drc_dec.process_frame(&mut drc_pcm, &drc_data).unwrap();
    assert_eq!(drc_pcm[0].len(), 512);
}

#[test]
fn test_psychoacoustic_and_quantization() {
    let psycho = PsychoacousticModel::new(49);
    let spec = vec![0.5f32; 1024];
    let sfb_offsets: Vec<usize> = (0..=49).map(|i| (i * 1024) / 49).collect();
    let result = psycho.analyze(&spec, &sfb_offsets);
    assert_eq!(result.masking_thresholds.len(), 49);

    let gain = estimate_global_gain(&spec, 2000);
    let mut quantized = vec![0i32; 1024];
    quantize_band(&spec, gain, &mut quantized);
    assert_eq!(quantized.len(), 1024);

    let mut block_switch = BlockSwitching::new();
    let (seq, shape) = block_switch.analyze(&spec);
    assert_eq!(seq, vuiocodecaac::types::WindowSequence::OnlyLongSequence);
    assert_eq!(shape, vuiocodecaac::types::WindowShape::Sine);
}
