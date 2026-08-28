//! Exhaustive Codec Subsystems and Tools Test Suite

use vuiocodecaac::decoder::aac::dequant::inverse_quantize;
use vuiocodecaac::decoder::aac::huffman::decode_spectral_band;
use vuiocodecaac::decoder::aac::pns::{NoiseRng, apply_pns};
use vuiocodecaac::decoder::aac::ics::{ChannelData, IcsInfo, INTENSITY_HCB, NOISE_HCB};
use vuiocodecaac::decoder::aac::stereo::{MsMask, apply_intensity_stereo, apply_ms_stereo};
use vuiocodecaac::decoder::aac::tns::{ar_filter, ma_filter, parcor_to_lpc};
use vuiocodecaac::tables::scalefactor::{MAX_SFB_LONG, compute_sfb_offsets};
use vuiocodecaac::decoder::drc::{DrcDecoder, DrcSettings};
use vuiocodecaac::decoder::mps::{MpsDecoder, MpsSpatialCues};
use vuiocodecaac::decoder::ps::{PsDecoder, SLOTS as PS_SLOTS};
use vuiocodecaac::dsp::fft::Complex32;
use vuiocodecaac::decoder::sbr::{SBR_CORE_FRAME, SbrDecoder};
use vuiocodecaac::decoder::usac::{UsacCoreMode, UsacDecoder};
use vuiocodecaac::encoder::aac::block_switch::BlockSwitch;
use vuiocodecaac::encoder::aac::psycho::PsychoacousticModel;
use vuiocodecaac::encoder::aac::quant::{choose_codebook, quantize_band};
use vuiocodecaac::encoder::drc::DrcEncoder;
use vuiocodecaac::encoder::sbr::SbrEncoder;
use vuiocodecaac::encoder::usac::UsacEncoder;
use vuiocodecaac::bitstream::{BitReader, BitWriter};

#[test]
fn test_huffman_and_dequantization() {
    // The all-zero 4-tuple is the most probable symbol in codebook 1, so it holds
    // the single-bit codeword `0`.
    let mut writer = BitWriter::with_capacity(32);
    writer.write_bit(false);
    writer.byte_align_zero();
    let bytes = writer.finalize().to_vec();

    let mut reader = BitReader::new(&bytes);
    let mut quantized = [7i32; 4];
    decode_spectral_band(&mut reader, 1, &mut quantized).unwrap();
    assert_eq!(quantized, [0, 0, 0, 0]);
    assert_eq!(reader.bit_position(), 1, "the zero quad must cost one bit");

    // A scalefactor of 100 is unity gain, so a quantized 1 dequantizes to 1.0.
    let mut spectral = [0.0f32; 4];
    quantized[0] = 1;
    inverse_quantize(&quantized, 100, &mut spectral);
    assert!((spectral[0] - 1.0).abs() < 1e-4);
}

/// Build a long-window channel with the 48 kHz band layout.
fn long_channel() -> ChannelData {
    let mut ch = ChannelData::new(1024);
    let mut offsets = [0usize; MAX_SFB_LONG + 1];
    let count = compute_sfb_offsets(vuiocodecaac::tables::sfb::SFB_48_1024, &mut offsets);
    let mut ics = IcsInfo { window_length: 1024, ..Default::default() };
    ics.num_swb = count - 1;
    ics.max_sfb = ics.num_swb;
    for (d, &sv) in ics.swb_offset.iter_mut().zip(offsets.iter()) {
        *d = sv as u16;
    }
    ch.ics = ics;
    ch
}

#[test]
fn test_stereo_pns_and_tns_tools() {
    // M/S stereo: left carries mid, right carries side.
    let mut left = long_channel();
    let mut right = long_channel();
    left.spec.fill(1.0);
    right.spec.fill(0.5);
    apply_ms_stereo(&mut left, &mut right, &MsMask { kind: 2, ..Default::default() });
    assert_eq!(left.spec[0], 1.5);
    assert_eq!(right.spec[0], 0.5);

    // Intensity stereo: the right channel is rebuilt from the left.
    let mut src = long_channel();
    let mut dest = long_channel();
    src.spec.fill(2.0);
    dest.sfb_cb[0][0] = INTENSITY_HCB;
    dest.scale_factors[0][0] = 4; // one octave down
    apply_intensity_stereo(&src, &mut dest, &MsMask::default());
    assert!((dest.spec[0] - 1.0).abs() < 1e-5);

    // PNS: a noise band gets the energy its scalefactor asks for.
    let mut noisy = long_channel();
    noisy.sfb_cb[0][0] = NOISE_HCB;
    noisy.scale_factors[0][0] = 40;
    let mut rng = NoiseRng::default();
    apply_pns(&mut noisy, &mut rng);
    let width = noisy.ics.swb_offset[1] as usize;
    let power: f64 = noisy.spec[..width].iter().map(|&v| (v as f64) * (v as f64)).sum();
    let want = (40.0f64 * 0.25).exp2().powi(2);
    assert!((power / want - 1.0).abs() < 1e-3, "PNS energy {power} vs {want}");

    // TNS: the analysis and synthesis filters invert each other.
    let parcor = [0.5f32, -0.25];
    let mut lpc = [0.0f32; 21];
    parcor_to_lpc(&parcor, &mut lpc);
    let original = [1.0f32, 0.5, -0.5, 0.2, 3.0, -1.5];
    let mut work = original;
    ma_filter(&mut work, &lpc, 2, false);
    ar_filter(&mut work, &lpc, 2, false);
    for (a, b) in original.iter().zip(work.iter()) {
        assert!((a - b).abs() < 1e-4, "TNS round trip {a} vs {b}");
    }
}

#[test]
fn test_sbr_and_ps_end_to_end() {
    let mut sbr = SbrDecoder::new(1, 22050, false);
    let mut sbr_enc = SbrEncoder::new(44100, 128000);

    let baseband: Vec<f32> = (0..SBR_CORE_FRAME)
        .map(|i| 0.5 * ((i as f32) * 0.05).sin())
        .collect();
    // SBR *encode* is not implemented; the contract is that it says so rather
    // than emitting a payload no decoder can read. SBR decode below is real.
    let sbr_enc_err = sbr_enc.encode_sbr_frame(&baseband).unwrap_err();
    assert!(matches!(sbr_enc_err, vuiocodecaac::error::Error::Unimplemented { .. }));

    let mut output_2x = vec![0.0f32; sbr.output_frame_len()];
    sbr.process_channel(0, &baseband, &mut output_2x).unwrap();
    assert_eq!(output_2x.len(), 2 * SBR_CORE_FRAME);

    // Parametric stereo turns one QMF frame into two. With no payload seen the
    // matrix is the identity one both channels start from, so the two outputs
    // carry the downmix and stay finite.
    let mut ps_dec = PsDecoder::new();
    let mut ps_left = vec![[Complex32::default(); 64]; PS_SLOTS];
    let mut ps_right = vec![[Complex32::default(); 64]; PS_SLOTS];
    for (slot, bands) in ps_left.iter_mut().enumerate() {
        for (band, v) in bands.iter_mut().enumerate().take(20) {
            let a = 0.11 * (slot * 20 + band) as f32;
            *v = Complex32::new(a.sin(), a.cos());
        }
    }
    let source = ps_left.clone();
    let ahead = vec![[Complex32::default(); 64]; 6];
    ps_dec.process(&mut ps_left, &ahead, &mut ps_right);

    let mut energy = 0.0f64;
    for (slot, bands) in ps_left.iter().enumerate() {
        for (band, v) in bands.iter().enumerate() {
            assert!(v.re.is_finite() && v.im.is_finite());
            // The three lowest bands go through the hybrid filterbank, whose
            // history and look-ahead this single frame does not supply; only the
            // pass-through range can be checked sample for sample.
            if band < 3 {
                continue;
            }
            let want = source[slot][band];
            energy += ((v.re - want.re) as f64).powi(2) + ((v.im - want.im) as f64).powi(2);
        }
    }
    assert!(energy < 1e-4, "left channel drifted from the downmix by {energy}");
}

#[test]
fn test_mps_usac_and_drc() {
    // MPS is not implemented; the contract is that it says so rather than
    // returning a fabricated upmix.
    let mps = MpsDecoder::new(0);
    let left = vec![1.0f32; 512];
    let right = vec![0.5f32; 512];
    let mut out_5point1 = vec![vec![0.0f32; 512]; 6];
    let mps_err = mps
        .decode(&[&left, &right], &MpsSpatialCues::default(), &mut out_5point1)
        .unwrap_err();
    assert!(matches!(mps_err, vuiocodecaac::error::Error::Unimplemented { .. }));

    // USAC
    let usac_enc = UsacEncoder::new();
    let mut usac_dec = UsacDecoder::new();
    let pcm = vec![0.1f32; 1024];
    let mode = usac_enc.classify_frame(&pcm);
    assert_eq!(mode, UsacCoreMode::LpdMode);

    // One all-ACELP LPD frame, straight off a bitstream: core_mode 2, lpd_mode
    // 0, then the subframe payload.
    let mut bytes = vec![0x40u8, 0x00];
    bytes.extend(std::iter::repeat_n(0x9Cu8, 62));
    let mut reader = BitReader::new(&bytes);
    let lsf = vuiocodecaac::decoder::usac::lsf::dequantize_lsf_abs(120);
    let mut acelp_out = vec![0.0f32; UsacDecoder::frame_len()];
    usac_dec.decode_lpd_frame(&mut reader, &lsf, 0.8, &mut acelp_out).unwrap();
    assert_eq!(acelp_out.len(), 256);
    assert!(acelp_out.iter().all(|x| x.is_finite()));
    assert!(acelp_out.iter().any(|x| *x != 0.0), "a real ACELP frame is not silence");

    // DRC: measure a frame, write its metadata, and read it back through the
    // decoder, which must then attenuate a loud programme.
    let mut drc_enc = DrcEncoder::new(44100, 2, -23.0);
    let loud: Vec<f32> = (0..44100)
        .map(|i| 0.9 * (std::f32::consts::TAU * 1000.0 * i as f32 / 44100.0).sin())
        .collect();
    let info = drc_enc.analyse(&[&loud, &loud]);
    assert!(info.gain[0] < 0, "a loud programme should ask to be turned down");

    let mut drc_dec = DrcDecoder::new(DrcSettings::full_compression());
    drc_dec.accept(info);
    let mut spectrum = vec![1.0f32; 512];
    drc_dec.apply_to_spectrum(0, &mut spectrum);
    assert!(spectrum[0] < 1.0, "the decoder ignored the metadata");
}

#[test]
fn test_psychoacoustic_and_quantization() {
    let sfb_offsets: Vec<usize> = (0..=49).map(|i| (i * 1024) / 49).collect();
    let mut psycho = PsychoacousticModel::new(44100, 64000, &sfb_offsets, false);
    let spec = vec![0.5f32; 1024];
    let mut result = vuiocodecaac::encoder::aac::psycho::PsychoResult::default();
    psycho.analyse(&spec, &sfb_offsets, vuiocodecaac::types::WindowSequence::OnlyLongSequence, &mut result);
    assert_eq!(result.bands, 49);
    for b in 0..result.bands {
        assert!(result.threshold[b] > 0.0);
    }

    // Quantize at unity scale, then confirm the rate estimator picks a codebook
    // that can actually represent the result.
    let mut quantized = vec![0i32; 1024];
    quantize_band(&spec, 100, &mut quantized);
    assert_eq!(quantized.len(), 1024);
    let choice = choose_codebook(&quantized[..64]);
    assert_ne!(choice.codebook, 0, "a nonzero band must get a codebook");
    assert!(choice.bits > 0);

    // Block switching: a steady tone must never ask for short windows.
    let mut block_switch = BlockSwitch::new(64000);
    block_switch.analyse(&spec);
    let here = block_switch.analyse(&spec);
    let next = block_switch.analyse(&spec);
    assert!(!here.attack && !next.attack);
    let decision = block_switch.decide(here, next);
    assert_eq!(decision.sequence, vuiocodecaac::types::WindowSequence::OnlyLongSequence);
}
