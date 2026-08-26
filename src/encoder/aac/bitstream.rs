//! AAC Bitstream Framing and Serializer Subsystem
//!
//! Emits standard-compliant MPEG-4 AAC ADTS frame headers and syntactic channel elements:
//! - SCE (Single Channel Element, ID = 0)
//! - CPE (Channel Pair Element, ID = 1)
//! - LFE (Low Frequency Effects Element, ID = 3)
//! - FIL (Fill Element / SBR Extension Payload, ID = 6)
//! - END (Terminator Element, ID = 7)

use crate::bitstream::BitWriter;
use crate::syntax::adts::AdtsHeader;
use crate::types::{AudioObjectType, ChannelConfiguration, SamplingRate};

/// Encode a Single Channel Element (SCE, ID = 0).
pub fn write_sce(
    writer: &mut BitWriter,
    instance_tag: u8,
    global_gain: i16,
    max_sfb: u8,
    _quantized: &[i32],
) {
    // 1. Element ID = 0 (SCE)
    writer.write_u8(0, 3);
    // 2. Element instance tag (4 bits)
    writer.write_u8(instance_tag & 0x0F, 4);

    // 3. Individual Channel Stream (ICS):
    encode_ic_stream_header_and_body(writer, global_gain, max_sfb);
}

/// Encode a Channel Pair Element (CPE, ID = 1) for stereo channel pairs.
pub fn write_cpe(
    writer: &mut BitWriter,
    instance_tag: u8,
    global_gain_l: i16,
    global_gain_r: i16,
    max_sfb: u8,
    _quantized_left: &[i32],
    _quantized_right: &[i32],
) {
    // 1. Element ID = 1 (CPE)
    writer.write_u8(1, 3);
    // 2. Element instance tag (4 bits)
    writer.write_u8(instance_tag & 0x0F, 4);

    // 3. common_window = 1
    writer.write_bit(true);

    // 4. ICS Info (shared for common window):
    writer.write_bit(false); // ics_reserved_bit (1 bit = 0)
    writer.write_u8(0, 2);   // window_sequence: ONLY_LONG_SEQUENCE (2 bits = 0)
    writer.write_u8(0, 1);   // window_shape: Sine (1 bit = 0)
    writer.write_u8(max_sfb, 6); // max_sfb (6 bits)
    writer.write_bit(false); // predictor_data_present (1 bit = 0)

    // 5. ms_mask_present = 0 (no M/S stereo mask)
    writer.write_u8(0, 2);

    // 6. Left Individual Channel Stream
    encode_ic_stream_body(writer, global_gain_l, max_sfb);

    // 7. Right Individual Channel Stream
    encode_ic_stream_body(writer, global_gain_r, max_sfb);
}

/// Encode a Low Frequency Enhancement / Subwoofer Element (LFE, ID = 3).
pub fn write_lfe(
    writer: &mut BitWriter,
    instance_tag: u8,
    global_gain: i16,
    max_sfb: u8,
    _quantized: &[i32],
) {
    // 1. Element ID = 3 (LFE)
    writer.write_u8(3, 3);
    // 2. Element instance tag (4 bits)
    writer.write_u8(instance_tag & 0x0F, 4);

    // 3. LFE Individual Channel Stream
    encode_ic_stream_header_and_body(writer, global_gain, max_sfb.min(12));
}

/// Encode a Fill Element (FIL, ID = 6) with SBR extension payload (EXT_SBR_DATA = 13 or 14).
pub fn write_fill_sbr(writer: &mut BitWriter, sbr_payload: &[u8]) {
    // 1. Element ID = 6 (FIL)
    writer.write_u8(6, 3);

    let count = sbr_payload.len() + 1; // +1 for extension_type (4 bits)
    if count < 15 {
        writer.write_u8(count as u8, 4);
    } else {
        writer.write_u8(15, 4);
        let mut extra = count - 14;
        while extra >= 255 {
            writer.write_u8(255, 8);
            extra -= 255;
        }
        writer.write_u8(extra as u8, 8);
    }

    // Extension type: EXT_SBR_DATA = 13 (0b1101)
    writer.write_u8(13, 4);

    for &byte in sbr_payload {
        writer.write_u8(byte, 8);
    }
}

/// Helper to serialize multi-channel stream elements matching standard channel configurations.
pub fn write_multichannel_elements(
    writer: &mut BitWriter,
    channel_config: ChannelConfiguration,
    global_gains: &[i16],
    max_sfb: u8,
    quantized: &[Vec<i32>],
    sbr_payload: Option<&[u8]>,
) {
    match channel_config {
        ChannelConfiguration::Mono => {
            write_sce(writer, 0, global_gains[0], max_sfb, &quantized[0]);
        }
        ChannelConfiguration::Stereo => {
            write_cpe(
                writer,
                0,
                global_gains[0],
                global_gains[1],
                max_sfb,
                &quantized[0],
                &quantized[1],
            );
        }
        ChannelConfiguration::ThreeChannel => {
            // 3.0: 1 Center (SCE 0) + 1 Stereo Pair Left/Right (CPE 0)
            write_sce(writer, 0, global_gains[0], max_sfb, &quantized[0]);
            write_cpe(
                writer,
                0,
                global_gains[1],
                global_gains[2],
                max_sfb,
                &quantized[1],
                &quantized[2],
            );
        }
        ChannelConfiguration::FourChannel => {
            // 4.0: 1 Center (SCE 0) + 1 Front Pair (CPE 0) + 1 Rear (SCE 1)
            write_sce(writer, 0, global_gains[0], max_sfb, &quantized[0]);
            write_cpe(
                writer,
                0,
                global_gains[1],
                global_gains[2],
                max_sfb,
                &quantized[1],
                &quantized[2],
            );
            write_sce(writer, 1, global_gains[3], max_sfb, &quantized[3]);
        }
        ChannelConfiguration::FiveChannel => {
            // 5.0: 1 Center (SCE 0) + 1 Front Pair (CPE 0) + 1 Surround Pair (CPE 1)
            write_sce(writer, 0, global_gains[0], max_sfb, &quantized[0]);
            write_cpe(
                writer,
                0,
                global_gains[1],
                global_gains[2],
                max_sfb,
                &quantized[1],
                &quantized[2],
            );
            write_cpe(
                writer,
                1,
                global_gains[3],
                global_gains[4],
                max_sfb,
                &quantized[3],
                &quantized[4],
            );
        }
        ChannelConfiguration::FivePointOne => {
            // 5.1: 1 Center (SCE 0) + 1 Front Pair (CPE 0) + 1 Surround Pair (CPE 1) + 1 LFE (LFE 0)
            write_sce(writer, 0, global_gains[0], max_sfb, &quantized[0]);
            write_cpe(
                writer,
                0,
                global_gains[1],
                global_gains[2],
                max_sfb,
                &quantized[1],
                &quantized[2],
            );
            write_cpe(
                writer,
                1,
                global_gains[3],
                global_gains[4],
                max_sfb,
                &quantized[3],
                &quantized[4],
            );
            write_lfe(writer, 0, global_gains[5], max_sfb, &quantized[5]);
        }
        ChannelConfiguration::SevenPointOne | ChannelConfiguration::SevenPointOneTop => {
            // 7.1: 1 Center (SCE 0) + 1 Front Pair (CPE 0) + 1 Side Pair (CPE 1) + 1 Rear Pair (CPE 2) + 1 LFE (LFE 0)
            write_sce(writer, 0, global_gains[0], max_sfb, &quantized[0]);
            write_cpe(
                writer,
                0,
                global_gains[1],
                global_gains[2],
                max_sfb,
                &quantized[1],
                &quantized[2],
            );
            write_cpe(
                writer,
                1,
                global_gains[3],
                global_gains[4],
                max_sfb,
                &quantized[3],
                &quantized[4],
            );
            write_cpe(
                writer,
                2,
                global_gains[5],
                global_gains[6],
                max_sfb,
                &quantized[5],
                &quantized[6],
            );
            write_lfe(writer, 0, global_gains[7], max_sfb, &quantized[7]);
        }
        _ => {
            // Fallback for custom layouts
            for (ch, quant) in quantized.iter().enumerate() {
                write_sce(writer, (ch & 0x0F) as u8, global_gains[ch], max_sfb, quant);
            }
        }
    }

    // Optional SBR Fill Element
    if let Some(sbr_data) = sbr_payload {
        write_fill_sbr(writer, sbr_data);
    }
}

fn encode_ic_stream_header_and_body(
    writer: &mut BitWriter,
    global_gain: i16,
    max_sfb: u8,
) {
    // Global Gain (8 bits)
    writer.write_u8((global_gain & 0xFF) as u8, 8);

    // ICS Info (for single channel):
    writer.write_bit(false); // ics_reserved_bit (1 bit = 0)
    writer.write_u8(0, 2);   // window_sequence: ONLY_LONG_SEQUENCE (2 bits = 0)
    writer.write_u8(0, 1);   // window_shape: Sine (1 bit = 0)
    writer.write_u8(max_sfb, 6); // max_sfb (6 bits)
    writer.write_bit(false); // predictor_data_present (1 bit = 0)

    // Body
    encode_section_and_tool_data(writer, max_sfb);
}

fn encode_ic_stream_body(
    writer: &mut BitWriter,
    global_gain: i16,
    max_sfb: u8,
) {
    // Global Gain (8 bits)
    writer.write_u8((global_gain & 0xFF) as u8, 8);

    // Body
    encode_section_and_tool_data(writer, max_sfb);
}

fn encode_section_and_tool_data(writer: &mut BitWriter, max_sfb: u8) {
    // Section Data: Single section using Zero codebook (sect_cb = 0)
    writer.write_u8(0, 4);

    // Length in 5-bit increments (31 while remaining >= 31, then remainder)
    let mut remaining = max_sfb as usize;
    while remaining >= 31 {
        writer.write_u8(31, 5);
        remaining -= 31;
    }
    writer.write_u8(remaining as u8, 5);

    // Tool Flags (pulse = 0, tns = 0, gain_control = 0)
    writer.write_bit(false); // pulse_data_present = 0
    writer.write_bit(false); // tns_data_present = 0
    writer.write_bit(false); // gain_control_data_present = 0
}

/// Finalize and package encoded bitstream with ADTS framing.
pub fn finalize_adts_frame(
    raw_payload: &[u8],
    aot: AudioObjectType,
    sampling_rate: SamplingRate,
    channel_config: ChannelConfiguration,
) -> Vec<u8> {
    let frame_length = raw_payload.len() + 7;
    let header = AdtsHeader {
        mpeg_id: 0,
        layer: 0,
        protection_absent: true,
        audio_object_type: aot,
        sampling_rate,
        channel_config,
        frame_length,
        buffer_fullness: 0x7FF,
        num_raw_data_blocks: 0,
        crc: None,
    };

    let mut writer = BitWriter::with_capacity(frame_length);
    header.write(&mut writer);
    writer.write_bytes(raw_payload);
    writer.into_bytes()
}
