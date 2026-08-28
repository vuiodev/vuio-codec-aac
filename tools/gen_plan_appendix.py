#!/usr/bin/env python3
"""Regenerate appendix A of text/plan.txt: every c/libxaac .c file, its Rust
target and its port status. Run from the repo root:

    head -<N> text/plan.txt > /tmp/p && mv /tmp/p text/plan.txt   # drop old appendix
    python3 tools/gen_plan_appendix.py >> text/plan.txt

Statuses live in the M dict below and are hand-maintained -- update them as
phases land. Line counts and the file list come from the tree, so a reference
update shows up as new [TODO] rows rather than silently going missing.
"""
import glob, os
BASE='c/libxaac/'
def loc(p): return sum(1 for _ in open(p, errors='ignore'))

# (status, rust target / note)
M = {
# ---------------- DECODER: AAC core ----------------
'decoder/ixheaacd_block.c':            ('DONE','decoder/aac/{ics,huffman,dequant}.rs'),
'decoder/ixheaacd_longblock.c':        ('DONE','decoder/aac/ics.rs (section + scalefactor data)'),
'decoder/ixheaacd_channel.c':          ('DONE','decoder/aac/channel.rs, decoder/engine.rs'),
'decoder/ixheaacd_aacdecoder.c':       ('DONE','decoder/engine.rs'),
'decoder/ixheaacd_decode_main.c':      ('DONE','decoder/engine.rs'),
'decoder/ixheaacd_stereo.c':           ('DONE','decoder/aac/stereo.rs (M/S + intensity)'),
'decoder/ixheaacd_tns.c':              ('DONE','decoder/aac/tns.rs'),
'decoder/ixheaacd_aac_tns.c':          ('DONE','decoder/aac/tns.rs'),
'decoder/ixheaacd_pns_js_thumb.c':     ('DONE','decoder/aac/pns.rs'),
'decoder/ixheaacd_aac_rom.c':          ('DONE','tables/huffman.rs (tools/extract_rom.py)'),
'decoder/ixheaacd_hufftables.c':       ('DONE','tables/huffman_rom.rs'),
'decoder/ixheaacd_huff_tools.c':       ('DONE','decoder/aac/huffman.rs'),
'decoder/ixheaacd_basic_ops.c':        ('DONE','dsp/math.rs'),
'decoder/ixheaacd_basic_funcs.c':      ('DONE','dsp/math.rs'),
'decoder/ixheaacd_common_lpfuncs.c':   ('DONE','decoder/engine.rs (DSE / fill / element tags)'),
'decoder/ixheaacd_lpfuncs.c':          ('DONE','dsp/{filterbank,window}.rs (float path)'),
'decoder/ixheaacd_rom.c':              ('PART','tables/* -- AAC ROM done; tables/usac_acelp.rs::GAIN_TABLE extracted (Phase 1.1 done)'),
'decoder/ixheaacd_common_rom.c':       ('PART','tables/* -- subset'),
'decoder/ixheaacd_lt_predict.c':       ('SKIP','ics.rs::skip_ltp_data parses, never applies -- Phase 7.4'),
'decoder/ixheaacd_multichannel.c':     ('PART','decoder::engine::CouplingChannelElement decodes real header/targets/ICS/gains (fixed a bit-alignment bug too); decoder::aac::downmix ports the fixed 5.0/5.1/7.0/7.1 downmix matrix; neither is wired to a caller (CCE target mixing, PCE multichannel slot mapping) -- Phase 7.5/7.6'),
'decoder/ixheaacd_huff_code_reorder.c':('TODO','HCR -- Phase 7.2'),
'decoder/ixheaacd_rev_vlc.c':          ('TODO','RVLC -- Phase 7.3'),
'decoder/ixheaacd_aac_ec.c':           ('TODO','error concealment -- Phase 7.1'),
'decoder/ixheaacd_ec_rom.c':           ('TODO','error concealment tables -- Phase 7.1'),
'decoder/ixheaacd_peak_limiter.c':     ('DONE','dsp::peak_limiter::PeakLimiter, wired into Decoder as an opt-in stage (enable_peak_limiter) -- Phase 7.7'),
# ---------------- DECODER: framing / config / API ----------------
'decoder/ixheaacd_headerdecode.c':     ('DONE','syntax/{adts,asc,pce,adif,latm}.rs, all wired into Decoder::decode_frame'),
'decoder/ixheaacd_adts_crc_check.c':   ('DONE','bitstream/crc.rs'),
'decoder/ixheaacd_bitbuffer.c':        ('DONE','bitstream/{reader,writer}.rs'),
'decoder/ixheaacd_init_config.c':      ('PART','syntax/asc.rs -- AAC/SBR/PS config only, no USAC/MPS config'),
'decoder/ixheaacd_latmdemux.c':        ('DONE','syntax/latm.rs + Decoder::decode_frame auto-detects LOAS sync and unwraps it -- Phase 0.5'),
'decoder/ixheaacd_aacpluscheck.c':     ('PART','decoder/engine.rs SBR presence detection'),
'decoder/ixheaacd_api.c':              ('PART','decoder/engine.rs + typed config (deliberately not the C state machine, D2)'),
'decoder/ixheaacd_create.c':           ('N/A','C handle/scratch allocation -- Rust constructors'),
'decoder/ixheaacd_initfuncs.c':        ('N/A','C init plumbing -- Rust constructors'),
'decoder/ixheaacd_common_initfuncs.c': ('N/A','C init plumbing -- Rust constructors'),
# ---------------- DECODER: DSP ----------------
'decoder/ixheaacd_fft.c':              ('DONE','dsp/fft.rs (+ dsp/simd.rs kernels)'),
'decoder/ixheaacd_aac_imdct.c':        ('DONE','dsp/imdct.rs (quarter-length FFT)'),
'decoder/ixheaacd_imdct.c':            ('DONE','dsp/imdct.rs'),
'decoder/ixheaacd_qmf_dec.c':          ('DONE','dsp/qmf.rs'),
'decoder/ixheaacd_Windowing.c':        ('DONE','dsp/window.rs'),
'decoder/ixheaacd_fft_ifft_32x32.c':   ('N/A','fixed-point FFT twin (R2)'),
'decoder/ixheaacd_dsp_fft32x32s.c':    ('N/A','fixed-point FFT twin (R2)'),
# ---------------- DECODER: SBR / eSBR ----------------
'decoder/ixheaacd_env_extr.c':         ('DONE','decoder/sbr/{mod,data,header,grid}.rs'),
'decoder/ixheaacd_env_dec.c':          ('DONE','decoder/sbr/data.rs (dequantise, delta decode)'),
'decoder/ixheaacd_env_calc.c':         ('DONE','decoder/sbr/hf.rs::adjust'),
'decoder/ixheaacd_lpp_tran.c':         ('DONE','decoder/sbr/hf.rs::generate (LPP transposer)'),
'decoder/ixheaacd_freq_sca.c':         ('DONE','decoder/sbr/header.rs::BandLayout::derive'),
'decoder/ixheaacd_sbrdec_initfuncs.c': ('DONE','decoder/sbr/header.rs'),
'decoder/ixheaacd_sbr_dec.c':          ('DONE','decoder/sbr/mod.rs'),
'decoder/ixheaacd_sbrdecoder.c':       ('DONE','decoder/sbr/mod.rs'),
'decoder/ixheaacd_sbrdec_lpfuncs.c':   ('PART','decoder/sbr/mod.rs -- float subset'),
'decoder/ixheaacd_sbr_rom.c':          ('PART','tables/sbr.rs -- v1 tables only'),
'decoder/ixheaacd_hbe_trans.c':        ('TODO','eSBR harmonic transposer, time domain -- Phase 5.1'),
'decoder/ixheaacd_hbe_dft_trans.c':    ('TODO','eSBR harmonic transposer, DFT -- Phase 5.2'),
'decoder/ixheaacd_pred_vec_block.c':   ('TODO','eSBR PVC -- Phase 5.3'),
'decoder/ixheaacd_pvc_rom.c':          ('TODO','eSBR PVC tables -- Phase 5.3'),
'decoder/ixheaacd_esbr_envcal.c':      ('TODO','eSBR envelope calc -- Phase 5.4'),
'decoder/ixheaacd_esbr_polyphase.c':   ('TODO','eSBR polyphase -- Phase 5.4'),
'decoder/ixheaacd_sbr_crc.c':          ('DONE','decoder/sbr/crc.rs ports the CRC-10 check, wired into SbrDecoder::decode_extension (rejects a payload on mismatch) -- Phase 5.5'),
# ---------------- DECODER: PS ----------------
'decoder/ixheaacd_ps_bitdec.c':        ('DONE','decoder/ps/data.rs'),
'decoder/ixheaacd_ps_dec_flt.c':       ('DONE','decoder/ps/mod.rs'),
'decoder/ixheaacd_hybrid.c':           ('DONE','decoder/ps/hybrid.rs'),
'decoder/ixheaacd_ps_dec.c':           ('N/A','fixed-point twin of ps_dec_flt.c (R2)'),
'decoder/ixheaacd_thumb_ps_dec.c':     ('TODO','low-power PS variant (optional)'),
# ---------------- DECODER: USAC ----------------
'decoder/ixheaacd_arith_dec.c':        ('DONE','decoder/usac/arith.rs + tables/usac_arith.rs'),
'decoder/ixheaacd_lpc_dec.c':          ('DONE','decoder/usac/lsf.rs + tables/usac_lsf*.rs'),
'decoder/ixheaacd_avq_rom.c':          ('PART','tables/usac_avq.rs -- wired into tables/mod.rs (Phase 0.3 done); usac_acelp.rs::INTERPOL_FILT extracted too'),
'decoder/ixheaacd_avq_dec.c':          ('PART','decoder/usac/avq.rs ports rotated_gosset_mtx_dec for qn<=4 (base lattice, no Voronoi ext.), verified structurally against every ABSOLUTE_LEADER_TAB row; qn>4 Voronoi extension and the FAC/LSF-refinement callers are still open'),
'decoder/ixheaacd_spectrum_dec.c':     ('PART','decoder/usac/fd.rs -- own framing, not ISO FD channel stream (Phase 2.2)'),
'decoder/ixheaacd_acelp_decode.c':     ('DONE','decoder/usac/acelp.rs -- real adaptive+algebraic codebooks, gains, post-processing (Phase 1.3/1.4/1.5 done)'),
'decoder/ixheaacd_acelp_bitparse.c':   ('DONE','decoder/usac/acelp.rs::{SubframeParams,AcelpFrame}::parse (Phase 1.6 done)'),
'decoder/ixheaacd_acelp_tools.c':      ('DONE','decoder/usac/acelp.rs -- preemph/deemph/residual/synthesis/pitch_sharpening (Phase 1.2 done; rand_gen still unported, needed only for error concealment)'),
'decoder/ixheaacd_acelp_mdct.c':       ('TODO','ACELP-side MDCT for FAC -- Phase 1.7'),
'decoder/ixheaacd_tcx_fwd_mdct.c':     ('PART','decoder/usac/tcx.rs -- weight_lpc/spectral_envelope_gains ported (ixheaacd_lpc_coeff_wt_apply/ixheaacd_lpc_to_td); this file also has ixheaacd_lsp_to_lp_conversion, already covered by tables/usac_lsf.rs::lsp_to_lpc; ixheaacd_lpc_coef_gen/ixheaacd_interpolation_lsp_params (per-subframe LPC interpolation) covered by decoder/usac/mod.rs::interpolate_lpc'),
'decoder/ixheaacd_tcx_fwd_alcnx.c':    ('PART','decoder/usac/tcx.rs -- noise_shape/low_frequency_deemphasis ported; the asymmetric-overlap MDCT and mode_prev/exc_prev state machine in this file are not'),
'decoder/ixheaacd_fwd_alias_cnx.c':    ('TODO','FAC across mode switches -- Phase 1.7'),
'decoder/ixheaacd_lpc.c':              ('TODO','LPD superframe top level + bass post filter -- Phase 1.8'),
'decoder/ixheaacd_ext_ch_ele.c':       ('TODO','USAC complex prediction stereo -- Phase 2.3'),
'decoder/ixheaacd_process.c':          ('TODO','usac_process + eSBR parse -- Phase 2.1/2.2'),
'decoder/ixheaacd_usac_ec.c':          ('TODO','USAC error concealment -- Phase 7.1'),
# ---------------- DECODER: uniDRC ----------------
'decoder/ixheaacd_drc_freq_dec.c':     ('PART','decoder/drc/mod.rs -- legacy dynamic_range_info() only'),
}
for p in glob.glob(BASE+'decoder/drc_src/*.c'):
    M[p[len(BASE):]] = ('TODO','MPEG-D uniDRC decode -- Phase 6')
M['decoder/drc_src/impd_drc_peak_limiter.c'] = ('TODO','uniDRC peak limiter -- Phase 6.5')
M['decoder/drc_src/impd_drc_bitbuffer.c']    = ('N/A','C bit buffer -- bitstream/reader.rs covers it')
M['decoder/drc_src/impd_drc_static_payload.c'] = ('PART','decoder/drc/{loudness_info,channel_layout,downmix_instructions,gain_modifiers,gain_set_params}.rs port impd_parse_loudness_info/impd_parse_loudness_measure/impd_dec_method_value/impd_parse_ch_layout/impd_parse_dwnmix_instructions/impd_parse_gain_set_params_characteristics/impd_dec_gain_modifiers/impd_parse_gain_set_params; the surrounding uniDrcConfig() (DRC sets, EQ) in this same file is still TODO -- Phase 6.1')

# MPS decode: everything TODO, module itself FAKE
for p in glob.glob(BASE+'decoder/*.c'):
    b = p[len(BASE):]
    n = os.path.basename(b)
    if n.startswith('ixheaacd_mps_') or n.startswith('ixheaacd_ld_mps'):
        M[b] = ('TODO','MPEG Surround -- Phase 9 (decoder/mps/mod.rs is FAKE today)')

# ---------------- ENCODER: AAC core ----------------
M.update({
'encoder/ixheaace_psy_mod.c':            ('DONE','encoder/aac/psycho.rs'),
'encoder/ixheaace_psy_configuration.c':  ('DONE','encoder/aac/psycho.rs'),
'encoder/ixheaace_psy_utils.c':          ('DONE','encoder/aac/psycho.rs'),
'encoder/ixheaace_psy_utils_spreading.c':('DONE','encoder/aac/psycho.rs'),
'encoder/ixheaace_adjust_threshold.c':   ('DONE','encoder/aac/psycho.rs + rate.rs'),
'encoder/ixheaace_fd_qc_adjthr.c':       ('PART','encoder/aac/rate.rs -- single rate strategy'),
'encoder/ixheaace_fd_qc_util.c':         ('PART','encoder/aac/rate.rs'),
'encoder/ixheaace_qc_util.c':            ('PART','encoder/aac/rate.rs'),
'encoder/ixheaace_sf_estimation.c':      ('DONE','encoder/aac/rate.rs (warm-started search)'),
'encoder/ixheaace_fd_quant.c':           ('DONE','encoder/aac/quant.rs'),
'encoder/ixheaace_quant.c':              ('DONE','encoder/aac/quant.rs'),
'encoder/ixheaace_bits_count.c':         ('DONE','encoder/aac/huffman.rs::tuple_cost'),
'encoder/ixheaace_dynamic_bits.c':       ('DONE','encoder/aac/huffman.rs + bitstream.rs'),
'encoder/ixheaace_static_bits.c':        ('DONE','encoder/aac/bitstream.rs'),
'encoder/ixheaace_huffman_rom.c':        ('DONE','tables/huffman.rs'),
'encoder/ixheaace_write_bitstream.c':    ('DONE','encoder/aac/bitstream.rs'),
'encoder/ixheaace_block_switch.c':       ('DONE','encoder/aac/block_switch.rs'),
'encoder/ixheaace_group_data.c':         ('DONE','encoder/engine.rs (short-window grouping)'),
'encoder/ixheaace_tns.c':                ('DONE','encoder/aac/tns.rs'),
'encoder/ixheaace_tns_init.c':           ('DONE','encoder/aac/tns.rs'),
'encoder/ixheaace_tns_params.c':         ('DONE','tables/tns.rs'),
'encoder/ixheaace_ms_stereo.c':          ('DONE','encoder/engine.rs (M/S decision)'),
'encoder/ixheaace_calc_ms_band_energy.c':('DONE','encoder/engine.rs'),
'encoder/ixheaace_fd_enc.c':             ('DONE','encoder/engine.rs'),
'encoder/ixheaace_enc_main.c':           ('DONE','encoder/engine.rs'),
'encoder/ixheaace_basic_ops.c':          ('DONE','dsp/math.rs'),
'encoder/ixheaace_rom.c':                ('PART','tables/* -- AAC-LC subset'),
'encoder/ixheaace_common_rom.c':         ('PART','tables/* -- AAC-LC subset'),
'encoder/ixheaace_nf.c':                 ('PART','encoder/usac/fd.rs -- USAC noise filling, simplified (documented)'),
'encoder/ixheaace_stereo_preproc.c':     ('TODO','stereo pre-processing -- Phase 2.9 backlog'),
'encoder/ixheaace_cplx_pred.c':          ('TODO','complex prediction encode -- Phase 2.3 (encoder half)'),
'encoder/ixheaace_qc_main_hp.c':         ('N/A','high-performance fixed-point QC twin (R2)'),
'encoder/ixheaace_tns_hp.c':             ('N/A','high-performance fixed-point TNS twin (R2)'),
# ENCODER: config / framing / API
'encoder/ixheaace_api.c':                ('PART','encoder/engine.rs + EncoderConfig (not the C state machine, D2)'),
'encoder/ixheaace_enc_init.c':           ('DONE','Encoder::new'),
'encoder/ixheaace_bitbuffer.c':          ('DONE','bitstream/writer.rs'),
'encoder/ixheaace_write_adts_adif.c':    ('PART','encoder/aac/bitstream.rs -- ADTS done, ADIF TODO'),
'encoder/ixheaace_asc_write.c':          ('PART','syntax/asc.rs -- AAC subset'),
'encoder/ixheaace_channel_map.c':        ('PART','encoder/aac/bitstream.rs::write_multichannel_elements'),
'encoder/ixheaace_bitbuffer_hp.c':       ('N/A','fixed-point bit buffer twin (R2)'),
'encoder/ixheaace_interface.c':          ('N/A','C plumbing'),
# ENCODER: DSP
'encoder/ixheaace_fft.c':                ('DONE','dsp/fft.rs'),
'encoder/ixheaace_fd_mdct.c':            ('DONE','dsp/mdct.rs'),
'encoder/ixheaace_radix2_fft.c':         ('DONE','dsp/fft.rs'),
'encoder/ixheaace_resampler.c':          ('DONE','dsp/resampler.rs'),
'encoder/ixheaace_resampler_init.c':     ('DONE','dsp/resampler.rs'),
'encoder/ixheaace_mdct_480.c':           ('TODO','480/512 transform for LD/ELD + 960 frames -- Phase 8.1'),
# ENCODER: PS  (module is FAKE)
'encoder/ixheaace_ps_enc.c':             ('TODO','encoder/ps/mod.rs now refuses (Err(Unimplemented)) rather than faking -- Phase 4.6'),
'encoder/ixheaace_ps_bitenc.c':          ('TODO','PS payload writer -- Phase 4.6'),
'encoder/ixheaace_ps_enc_init.c':        ('TODO','Phase 4.6'),
'encoder/ixheaace_hybrid.c':             ('TODO','encoder-side hybrid filterbank -- Phase 4.6'),
'encoder/ixheaace_hybrid_init.c':        ('TODO','Phase 4.6'),
# ENCODER: uniDRC
'encoder/ixheaace_loudness_measurement.c':('DONE','encoder/drc/mod.rs (BS.1770 K-weighting + gating)'),
# ENCODER: USAC
'encoder/iusace_arith_enc.c':            ('DONE','encoder/usac/arith.rs'),
'encoder/iusace_tns_usac.c':             ('DONE','encoder/usac/tns.rs'),
'encoder/iusace_lpc.c':                  ('DONE','encoder/usac/lsf.rs'),
'encoder/iusace_ms.c':                   ('DONE','encoder/usac/fd.rs (M/S)'),
'encoder/iusace_bitbuffer.c':            ('DONE','bitstream/writer.rs'),
'encoder/iusace_fft.c':                  ('DONE','dsp/fft.rs'),
'encoder/iusace_enc_main.c':             ('PART','encoder/usac/fd.rs -- FD half only'),
'encoder/iusace_rom.c':                  ('PART','tables/usac_{arith,lsf,lsf_dico}.rs -- 15,083 C lines, small fraction ported (R3)'),
'encoder/iusace_avq_rom.c':              ('PART','tables/usac_avq.rs (orphan -- Phase 0.3)'),
'encoder/iusace_avq_enc.c':              ('TODO','AVQ lattice search + index coding -- Phase 3.6'),
'encoder/iusace_lpc_avq.c':              ('TODO','LSF stage-2 AVQ quantisation -- Phase 3.6'),
'encoder/iusace_psy_mod.c':              ('PART','encoder/usac/fd.rs -- simplified'),
'encoder/iusace_psy_utils.c':            ('PART','encoder/usac/fd.rs'),
'encoder/iusace_psy_rom.c':              ('PART','tables/*'),
'encoder/iusace_block_switch.c':         ('PART','encoder/aac/block_switch.rs reused'),
'encoder/iusace_windowing.c':            ('PART','dsp/window.rs'),
'encoder/iusace_write_bitstream.c':      ('TODO','ISO USAC framing (VUSC container today, D3) -- Phase 2.5'),
'encoder/ixheaace_signal_classifier.c':  ('FAKE','usac/mod.rs::classify_frame is a heuristic stand-in -- Phase 3.1'),
'encoder/ixheaace_signal_classifier_rom.c':('TODO','classifier tables -- Phase 3.1'),
'encoder/iusace_acelp_enc.c':            ('TODO','ACELP analysis-by-synthesis -- Phase 3.2/3.3'),
'encoder/iusace_acelp_tools.c':          ('TODO','encoder ACELP tools -- Phase 3.2'),
'encoder/iusace_acelp_rom.c':            ('TODO','ACELP tables -- Phase 3.2'),
'encoder/iusace_tcx_enc.c':              ('TODO','TCX encode -- Phase 3.4'),
'encoder/iusace_tcx_mdct.c':             ('TODO','TCX MDCT -- Phase 3.4'),
'encoder/iusace_enc_fac.c':              ('TODO','FAC generation -- Phase 3.4'),
'encoder/iusace_fd_fac.c':               ('TODO','FD-side FAC -- Phase 3.4'),
'encoder/iusace_lpd_enc.c':              ('TODO','LPD mode decision -- Phase 3.5'),
'encoder/iusace_lpd_utils.c':            ('TODO','LPD helpers -- Phase 3.5'),
'encoder/iusace_lpd_rom.c':              ('TODO','LPD tables -- Phase 3.5'),
# ENCODER: eSBR pieces that live under iusace_
'encoder/iusace_esbr_inter_tes.c':       ('TODO','eSBR inter-TES -- Phase 5.4'),
'encoder/iusace_esbr_pvc.c':             ('TODO','eSBR PVC encode -- Phase 5.3'),
'encoder/iusace_esbr_pvc_rom.c':         ('TODO','eSBR PVC tables -- Phase 5.3'),
'encoder/iusace_esbr_rom.c':             ('TODO','eSBR tables -- Phase 5'),
# ---------------- COMMON ----------------
'common/ixheaac_esbr_fft.c':             ('TODO','eSBR FFT sizes -- Phase 5.2'),
'common/ixheaac_esbr_rom.c':             ('TODO','eSBR ROM -- Phase 5'),
'common/ixheaac_fft_ifft_32x32_rom.c':   ('N/A','fixed-point FFT twiddles (R2)'),
})
# ENCODER SBR (ixheaace_sbr_*) -> FAKE for the module, TODO per file
for p in glob.glob(BASE+'encoder/ixheaace_sbr_*.c'):
    M[p[len(BASE):]] = ('TODO','SBR encode -- Phase 4 (encoder/sbr/mod.rs is FAKE today)')
M['encoder/ixheaace_sbr_env_est.c'] = ('TODO','encoder/sbr/mod.rs now refuses (Err(Unimplemented)) rather than faking -- Phase 4.2')
for n in ('hbe_trans','hbe_dft_trans','hbe_fft_ifft_32x32','hbe_polyphase'):
    k='encoder/ixheaace_sbr_%s.c'%n
    if k in M: M[k] = ('TODO','eSBR encode transposer -- Phase 5 / 4')
# ENCODER MPS
for p in glob.glob(BASE+'encoder/ixheaace_mps_*.c'):
    M[p[len(BASE):]] = ('TODO','MPEG Surround encode -- Phase 9.8')
# ENCODER uniDRC
for p in glob.glob(BASE+'encoder/drc_src/*.c'):
    M[p[len(BASE):]] = ('TODO','MPEG-D uniDRC encode -- Phase 6.7')


DEC_SBR={'ixheaacd_env_extr.c','ixheaacd_env_dec.c','ixheaacd_env_calc.c','ixheaacd_lpp_tran.c',
 'ixheaacd_freq_sca.c','ixheaacd_sbrdec_initfuncs.c','ixheaacd_sbrdec_lpfuncs.c','ixheaacd_sbr_dec.c',
 'ixheaacd_sbrdecoder.c','ixheaacd_sbr_rom.c','ixheaacd_sbr_crc.c','ixheaacd_hbe_trans.c',
 'ixheaacd_hbe_dft_trans.c','ixheaacd_esbr_envcal.c','ixheaacd_esbr_polyphase.c','ixheaacd_pvc_rom.c',
 'ixheaacd_pred_vec_block.c'}
DEC_PS={'ixheaacd_ps_bitdec.c','ixheaacd_ps_dec.c','ixheaacd_ps_dec_flt.c','ixheaacd_hybrid.c',
 'ixheaacd_thumb_ps_dec.c'}
DEC_USAC={'ixheaacd_arith_dec.c','ixheaacd_avq_dec.c','ixheaacd_avq_rom.c','ixheaacd_acelp_bitparse.c',
 'ixheaacd_acelp_decode.c','ixheaacd_acelp_mdct.c','ixheaacd_acelp_tools.c','ixheaacd_tcx_fwd_alcnx.c',
 'ixheaacd_tcx_fwd_mdct.c','ixheaacd_fwd_alias_cnx.c','ixheaacd_lpc.c','ixheaacd_lpc_dec.c',
 'ixheaacd_ext_ch_ele.c','ixheaacd_spectrum_dec.c','ixheaacd_process.c','ixheaacd_usac_ec.c'}
DEC_DSP={'ixheaacd_fft.c','ixheaacd_fft_ifft_32x32.c','ixheaacd_dsp_fft32x32s.c','ixheaacd_aac_imdct.c',
 'ixheaacd_imdct.c','ixheaacd_qmf_dec.c','ixheaacd_Windowing.c'}
DEC_CFG={'ixheaacd_api.c','ixheaacd_create.c','ixheaacd_initfuncs.c','ixheaacd_common_initfuncs.c',
 'ixheaacd_headerdecode.c','ixheaacd_init_config.c','ixheaacd_bitbuffer.c','ixheaacd_latmdemux.c',
 'ixheaacd_aacpluscheck.c','ixheaacd_adts_crc_check.c'}
ENC_PS={'ixheaace_ps_enc.c','ixheaace_ps_bitenc.c','ixheaace_ps_enc_init.c','ixheaace_hybrid.c',
 'ixheaace_hybrid_init.c'}
ENC_DSP={'ixheaace_fft.c','ixheaace_radix2_fft.c','ixheaace_fd_mdct.c','ixheaace_mdct_480.c',
 'ixheaace_resampler.c','ixheaace_resampler_init.c'}
ENC_CFG={'ixheaace_api.c','ixheaace_enc_init.c','ixheaace_bitbuffer.c','ixheaace_bitbuffer_hp.c',
 'ixheaace_write_adts_adif.c','ixheaace_asc_write.c','ixheaace_channel_map.c','ixheaace_interface.c'}

def group_of(b):
    n=os.path.basename(b); d=b.startswith('decoder/')
    side='DECODER' if d else 'ENCODER'
    if '/drc_src/' in b or 'drc' in n or 'loudness' in n: return side+' / uniDRC'
    if '_mps_' in n or n.startswith('ixheaacd_mps') or 'ld_mps' in n: return side+' / MPEG SURROUND'
    if b.startswith('common/'): return 'COMMON'
    if d:
        if n in DEC_SBR:  return 'DECODER / SBR + eSBR'
        if n in DEC_PS:   return 'DECODER / PARAMETRIC STEREO'
        if n in DEC_USAC: return 'DECODER / USAC'
        if n in DEC_DSP:  return 'DECODER / DSP'
        if n in DEC_CFG:  return 'DECODER / FRAMING, CONFIG, API'
        return 'DECODER / AAC-LC CORE'
    if 'sbr' in n:            return 'ENCODER / SBR + eSBR'
    if n in ENC_PS:           return 'ENCODER / PARAMETRIC STEREO'
    if n.startswith('iusace_') or 'signal_classifier' in n: return 'ENCODER / USAC'
    if n in ENC_DSP:          return 'ENCODER / DSP'
    if n in ENC_CFG:          return 'ENCODER / FRAMING, CONFIG, API'
    return 'ENCODER / AAC CORE'

ORDER_NAMES=['DECODER / AAC-LC CORE','DECODER / FRAMING, CONFIG, API','DECODER / DSP',
 'DECODER / SBR + eSBR','DECODER / PARAMETRIC STEREO','DECODER / USAC','DECODER / uniDRC',
 'DECODER / MPEG SURROUND','ENCODER / AAC CORE','ENCODER / FRAMING, CONFIG, API','ENCODER / DSP',
 'ENCODER / SBR + eSBR','ENCODER / PARAMETRIC STEREO','ENCODER / USAC','ENCODER / uniDRC',
 'ENCODER / MPEG SURROUND','COMMON']
ORDER=[(t,(lambda t: (lambda b: group_of(b)==t))(t)) for t in ORDER_NAMES]
allf = sorted(p[len(BASE):] for p in
    glob.glob(BASE+'decoder/*.c')+glob.glob(BASE+'decoder/drc_src/*.c')+
    glob.glob(BASE+'encoder/*.c')+glob.glob(BASE+'encoder/drc_src/*.c')+glob.glob(BASE+'common/*.c'))
used=set(); out=[]
out.append('='*80)
out.append(' APPENDIX A -- EVERY C FILE, ITS RUST TARGET, ITS STATUS')
out.append('='*80)
out.append('')
out.append(' All 283 .c files in c/libxaac (decoder/, decoder/drc_src/, encoder/,')
out.append(' encoder/drc_src/, common/). Architecture-specific selector files under')
out.append(' decoder/{armv7,armv8,x86,x86_64,generic}/ and the test/ harness are excluded')
out.append(' -- see the [N/A] definition in section 0.')
out.append('')
out.append(' LOC   STATUS  C FILE                                  RUST TARGET / NOTE')
for title, pred in ORDER:
    grp=[b for b in allf if b not in used and pred(b)]
    for b in grp: used.add(b)
    if not grp: continue
    grp.sort(key=lambda b:-loc(BASE+b))
    out.append('')
    out.append('-'*80)
    out.append(' '+title+'   ('+str(len(grp))+' files, '+format(sum(loc(BASE+b) for b in grp),',')+' lines)')
    out.append('-'*80)
    for b in grp:
        st,tg = M.get(b, ('TODO','(unclassified -- audit me)'))
        out.append('%6d  [%s] %-38s %s' % (loc(BASE+b), st, os.path.basename(b), tg))
left=[b for b in allf if b not in used]
if left:
    out.append('')
    out.append(' UNGROUPED (fix the appendix generator):')
    for b in left: out.append('        %s'%b)
# tally
from collections import Counter
c=Counter(M.get(b,('TODO',''))[0] for b in allf)
lc=Counter(); 
for b in allf: lc[M.get(b,('TODO',''))[0]] += loc(BASE+b)
out.append('')
out.append('-'*80)
out.append(' TALLY')
out.append('-'*80)
for k in ('DONE','PART','SKIP','FAKE','TODO','N/A'):
    out.append('   %-6s %3d files   %9s C lines' % (k, c.get(k,0), format(lc.get(k,0),',')))
out.append('   %-6s %3d files   %9s C lines' % ('TOTAL', len(allf), format(sum(lc.values()),',')))
out.append('')
out.append('='*80)
print('\n'.join(out))
