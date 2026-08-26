//! Exhaustive Syntax Headers and Containers Test Suite

use xaac::bitstream::{BitReader, BitWriter};
use xaac::syntax::*;
use xaac::types::{AudioObjectType, ChannelConfiguration, FrameLength, SamplingRate};

#[test]
fn test_adts_header_exhaustive_sampling_rates() {
    let rates = [
        SamplingRate::Hz96000,
        SamplingRate::Hz88200,
        SamplingRate::Hz64000,
        SamplingRate::Hz48000,
        SamplingRate::Hz44100,
        SamplingRate::Hz32000,
        SamplingRate::Hz24000,
        SamplingRate::Hz22050,
        SamplingRate::Hz16000,
        SamplingRate::Hz12000,
        SamplingRate::Hz11025,
        SamplingRate::Hz8000,
    ];

    for rate in rates {
        let header = AdtsHeader {
            mpeg_id: 0,
            layer: 0,
            protection_absent: true,
            audio_object_type: AudioObjectType::AacLc,
            sampling_rate: rate,
            channel_config: ChannelConfiguration::Stereo,
            frame_length: 512,
            buffer_fullness: 0x7FF,
            num_raw_data_blocks: 0,
            crc: None,
        };

        let mut writer = BitWriter::with_capacity(7);
        header.write(&mut writer);
        let bytes = writer.finalize();

        let mut reader = BitReader::new(bytes);
        let parsed = AdtsHeader::parse(&mut reader).unwrap();

        assert_eq!(parsed.sampling_rate, rate);
        assert_eq!(parsed.frame_length, 512);
    }
}

#[test]
fn test_asc_and_pce_syntax() {
    let asc = AudioSpecificConfig {
        audio_object_type: AudioObjectType::AacLc,
        sampling_rate: SamplingRate::Hz44100,
        channel_config: ChannelConfiguration::Stereo,
        frame_length: FrameLength::Samples1024,
        sbr_present: false,
        ps_present: false,
    };

    let mut writer = BitWriter::with_capacity(8);
    asc.write(&mut writer);
    let bytes = writer.finalize();

    let mut reader = BitReader::new(bytes);
    let parsed_asc = AudioSpecificConfig::parse(&mut reader).unwrap();
    assert_eq!(parsed_asc.audio_object_type, AudioObjectType::AacLc);

    let pce = ProgramConfigElement {
        element_instance_tag: 1,
        object_type: AudioObjectType::AacLc,
        sampling_rate: SamplingRate::Hz48000,
        num_front_channel_elements: 1,
        num_side_channel_elements: 0,
        num_back_channel_elements: 0,
        num_lfe_channel_elements: 1,
        num_assoc_data_elements: 0,
        num_valid_cc_elements: 0,
        mono_mixdown_present: false,
        mono_mixdown_element_number: 0,
        stereo_mixdown_present: false,
        stereo_mixdown_element_number: 0,
        matrix_mixdown_idx_present: false,
        matrix_mixdown_idx: 0,
        pseudo_surround_enable: false,
        front_elements: vec![(true, 0)],
        side_elements: vec![],
        back_elements: vec![],
        lfe_element_tags: vec![0],
    };

    let mut writer = BitWriter::with_capacity(32);
    pce.write(&mut writer);
    let bytes = writer.finalize();

    let mut reader = BitReader::new(bytes);
    let parsed_pce = ProgramConfigElement::parse(&mut reader).unwrap();
    assert_eq!(parsed_pce.element_instance_tag, 1);
}

#[test]
fn test_adif_and_latm_syntax() {
    let asc = AudioSpecificConfig {
        audio_object_type: AudioObjectType::AacLc,
        sampling_rate: SamplingRate::Hz32000,
        channel_config: ChannelConfiguration::Mono,
        frame_length: FrameLength::Samples1024,
        sbr_present: false,
        ps_present: false,
    };

    let adif = AdifHeader {
        copyright_id_present: false,
        copyright_id: [0u8; 9],
        original_copy: true,
        home: false,
        bitstream_type: true,
        bitrate: 64000,
        num_program_config_elements: 1,
        buffer_fullness: 0,
        configs: vec![asc.clone()],
    };

    let mut writer = BitWriter::with_capacity(64);
    adif.write(&mut writer);
    let bytes = writer.finalize();

    let mut reader = BitReader::new(bytes);
    let parsed_adif = AdifHeader::parse(&mut reader).unwrap();
    assert_eq!(parsed_adif.bitrate, 64000);

    let latm_elem = AudioMuxElement {
        mux_config_present: true,
        stream_mux_config: Some(StreamMuxConfig {
            audio_mux_version: 0,
            all_streams_same_time_framing: true,
            num_sub_frames: 1,
            num_programs: 1,
            num_layers: 1,
            asc: asc.clone(),
        }),
        payload_bytes: vec![0xDE, 0xAD, 0xBE, 0xEF],
    };

    let mut writer = BitWriter::with_capacity(64);
    latm_elem.write_loas(&mut writer);
    let bytes = writer.finalize();

    let mut reader = BitReader::new(bytes);
    let parsed_latm = AudioMuxElement::parse_loas(&mut reader).unwrap();
    assert_eq!(parsed_latm.payload_bytes, vec![0xDE, 0xAD, 0xBE, 0xEF]);
}
