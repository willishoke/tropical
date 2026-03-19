"""
test_compressor.py — Compressor + sidechain bass drum demo patch.

Architecture
------------
  Melody clock  (2 Hz) ──► AD envelope ──► gate VCO ──► compressor input
  Bass clock    (1 Hz) ──► bass drum ───────────────────► compressor sidechain
                                     └────────────────────► mix out

The bass drum punches through the compressor (sidechain ducking) every beat,
then the compressed VCOs fill back in between hits.

Controls
--------
  VCO_FREQ     : two VCO pitches (Hz)
  BD_FREQ      : bass drum fundamental pitch (Hz)
  THRESHOLD    : compressor threshold (dBFS)
  RATIO        : compressor ratio
  ATTACK_MS    : compressor attack time (ms)
  RELEASE_MS   : compressor release / ducking recovery time (ms)
  ENV_ATTACK   : VCO envelope attack (s)
  ENV_DECAY    : VCO envelope decay (s)

Run:
    python test/test_compressor.py
"""

import egress as eg
import egress.module_library as modlib

SAMPLE_RATE = 44100

# --- Patch parameters ---
VCO_FREQS  = [220.0, 330.0]   # two melody oscillators (A3, E4)
BD_FREQ    = 60.0              # bass drum fundamental

ENV_ATTACK = 0.003             # envelope attack: snappy (3 ms)
ENV_DECAY  = 0.18              # envelope decay: 180 ms

THRESHOLD  = -9.0              # compressor threshold (dBFS)
RATIO      = 6.0               # 6:1 — strong ducking
ATTACK_MS  = 3.0               # fast attack so the BD transient ducks quickly
RELEASE_MS = 120.0             # moderate release for pumping feel

MELODY_HZ  = 2.0              # melody trigger rate (notes per second)
BD_HZ      = 1.0              # bass drum rate (hits per second)

BD_PUNCH   = 0.65
BD_DECAY   = 0.38
BD_TONE    = 10.0


def build_patch():
    graph = eg.Graph()

    # ------------------------------------------------------------------ #
    # 1. Bass drum clock + voice                                           #
    # ------------------------------------------------------------------ #
    bd_clock_type = modlib.clock("BDClock")
    bd_clock = bd_clock_type(graph=graph)
    bd_clock.freq = BD_HZ
    bd_clock.ratios_in = eg.array([1.0])

    bd_type = modlib.bass_drum("BassDrum")
    bd = bd_type(graph=graph)
    bd.gate  = 1.0 * bd_clock.output   # rising edge fires drum
    bd.freq  = BD_FREQ
    bd.punch = BD_PUNCH
    bd.decay = BD_DECAY
    bd.tone  = BD_TONE

    # ------------------------------------------------------------------ #
    # 2. Melody clock → AD envelope → gates two VCOs                      #
    # ------------------------------------------------------------------ #
    mel_clock_type = modlib.clock("MelClock")
    mel_clock = mel_clock_type(graph=graph)
    mel_clock.freq = MELODY_HZ
    mel_clock.ratios_in = eg.array([1.0])

    vcos = []
    for i, freq in enumerate(VCO_FREQS):
        env_type = modlib.ad_envelope(f"Env{i}")
        env = env_type(graph=graph)
        env.gate   = 1.0 * mel_clock.output
        env.attack = ENV_ATTACK
        env.decay  = ENV_DECAY

        vco_type = modlib.vco(f"VCO{i}")
        vco = vco_type(graph=graph)
        vco.freq = freq

        vcos.append((env, vco))

    # Mix VCO outputs, scaled by their envelopes (coerce OutputPorts to SignalExpr via 1.0 *)
    vco_mix = sum(
        (1.0 * env.env) * (1.0 * vco.sin) for env, vco in vcos
    )

    # ------------------------------------------------------------------ #
    # 3. Compressor: VCO mix in, bass drum on sidechain                   #
    # ------------------------------------------------------------------ #
    cmp_type = modlib.compressor("Comp")
    cmp = cmp_type(graph=graph)
    cmp.input      = vco_mix
    cmp.sidechain  = 1.0 * bd.output   # BD ducks the melody
    cmp.threshold  = THRESHOLD
    cmp.ratio      = RATIO
    cmp.attack_ms  = ATTACK_MS
    cmp.release_ms = RELEASE_MS
    cmp.makeup     = 2.0               # recover perceived loudness

    # ------------------------------------------------------------------ #
    # 4. Final mix: compressed melody + bass drum (stereo)                #
    # ------------------------------------------------------------------ #
    eg.add_output(cmp.output, graph=graph)   # left:  compressed melody
    eg.add_output(bd.output,  graph=graph)   # left:  + bass drum
    eg.add_output(cmp.output, graph=graph)   # right: compressed melody
    eg.add_output(bd.output,  graph=graph)   # right: + bass drum

    return graph, bd_clock, mel_clock, bd, cmp


def main():
    graph, bd_clock, mel_clock, bd, cmp = build_patch()

    print("Compressor / sidechain bass drum test")
    print(f"  Melody:  {VCO_FREQS} Hz VCOs @ {MELODY_HZ} Hz clock, "
          f"AD {ENV_ATTACK*1000:.0f}/{ENV_DECAY*1000:.0f} ms")
    print(f"  Bass:    {BD_FREQ} Hz drum @ {BD_HZ} Hz clock")
    print(f"  Comp:    threshold {THRESHOLD} dBFS, {RATIO}:1, "
          f"A={ATTACK_MS} ms / R={RELEASE_MS} ms")
    print()
    print("You should hear: BD thump every second, melody notes twice per")
    print("second ducking under the kick and pumping back in between.")
    print()

    dac = eg.DAC(graph, sample_rate=SAMPLE_RATE, channels=2)
    dac.start()
    try:
        input("Playing... press Enter to stop.\n")
    finally:
        dac.stop()

    stats = dac.callback_stats()
    print(f"DAC stats: {stats}")


if __name__ == "__main__":
    main()
