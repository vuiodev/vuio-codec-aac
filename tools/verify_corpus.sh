#!/usr/bin/env bash
# Decode a corpus of AAC files with the Rust decoder and compare against the
# reference C decoder and ffmpeg.
#
# Every encoder produces a different bitstream, so the corpus is encoded twice
# (ffmpeg and libxaac) to exercise different tool combinations: ffmpeg's encoder
# uses PNS, M/S and intensity stereo, libxaac's uses TNS.
#
# Perceptual Noise Substitution is the one tool that cannot match sample for
# sample: the bitstream carries only a band's energy and each decoder synthesises
# its own noise to fill it. The PNS column reports the share of spectral energy in
# those bands, and the pass threshold is relaxed accordingly -- a stream that is
# mostly PNS cannot and should not be expected to match bit for bit.
#
# usage: verify_corpus.sh <corpus_dir> <xaacdec_binary>
set -uo pipefail

CORPUS="${1:?corpus directory required}"
XAACDEC="${2:?path to reference xaacdec required}"
RUSTDEC="./target/release/aacdec"
COMPARE="./target/release/compare_wav"

pass=0
fail=0
declare -a failures=()

printf '%-30s %-4s %8s %8s %7s %6s\n' "STREAM" "VS" "SNR(dB)" "MAX_ABS" "EXACT%" "PNS"
printf '%s\n' "--------------------------------------------------------------------------------"

for aac in "$CORPUS"/*.aac; do
  [ -e "$aac" ] || continue
  base="$(basename "$aac" .aac)"

  rust_wav="$CORPUS/${base}.rust.wav"
  c_wav="$CORPUS/${base}.c.wav"
  ff_wav="$CORPUS/${base}.ff.wav"

  if ! "$RUSTDEC" "$aac" "$rust_wav" >/dev/null 2>&1; then
    printf '%-34s %-10s %10s\n' "$base" "rust" "DECODE FAILED"
    fail=$((fail + 1)); failures+=("$base: rust decoder failed")
    continue
  fi

  "$XAACDEC" -ifile:"$aac" -ofile:"$c_wav" >/dev/null 2>&1
  ffmpeg -v error -i "$aac" -c:a pcm_s16le -y "$ff_wav" >/dev/null 2>&1

  # Share of spectral energy in noise-substituted bands, averaged over frames.
  pns="$(AAC_TRACE=1 "$RUSTDEC" "$aac" /dev/null 2>&1 |
    awk '/pns_frac/ { for (i = 1; i <= NF; i++) if ($i == "pns_frac") { s += $(i+1); n++ } }
         END { if (n) printf "%.3f", s / n; else printf "0.000" }')"

  for ref_kind in c ff; do
    ref_wav="$CORPUS/${base}.${ref_kind}.wav"
    [ -s "$ref_wav" ] || continue

    out="$("$COMPARE" "$ref_wav" "$rust_wav" 2>&1)"
    # Report the worst channel. Fields look like:
    #   channel 0: n=90112 max_abs=2.000 rms=0.5552 snr=71.23 dB exact=69.38%
    read -r snr mx ex <<<"$(printf '%s' "$out" | awk '
      /^channel/ {
        for (i = 1; i <= NF; i++) {
          split($i, kv, "=")
          if (kv[1] == "snr")     { v = kv[2] + 0; if (!have_snr || v < snr) { snr = v; have_snr = 1 } }
          if (kv[1] == "max_abs") { v = kv[2] + 0; if (v > mx) mx = v }
          if (kv[1] == "exact")   { gsub("%", "", kv[2]); v = kv[2] + 0; if (!have_ex || v < ex) { ex = v; have_ex = 1 } }
        }
      }
      END { if (have_snr) printf "%.2f %.0f %.1f", snr, mx, ex; else printf "NaN 0 0" }')"

    # Without PNS, two conformant decoders agree to well under an LSB, so 60 dB is
    # a generous floor. With PNS, the synthesised bands dominate the residual, so
    # the floor scales with how much of the signal they account for.
    floor="$(awk -v p="$pns" 'BEGIN { printf "%.1f", (p > 0.01) ? 20 : 60 }')"
    verdict="ok"
    if [ -z "$snr" ] || awk "BEGIN{exit !($snr < $floor)}"; then
      verdict="LOW"
      fail=$((fail + 1)); failures+=("$base vs $ref_kind: SNR ${snr} dB (floor ${floor}, PNS ${pns})")
    else
      pass=$((pass + 1))
    fi
    printf '%-30s %-4s %8s %8s %7s %6s  %s\n' "$base" "$ref_kind" "$snr" "$mx" "$ex" "$pns" "$verdict"
  done
done

printf '%s\n' "--------------------------------------------------------------------------------"
printf 'comparisons: %d ok, %d below threshold\n' "$pass" "$fail"
if [ "${#failures[@]}" -gt 0 ]; then
  printf '\nbelow threshold:\n'
  printf '  %s\n' "${failures[@]}"
  exit 1
fi
