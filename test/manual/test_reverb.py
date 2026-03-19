"""
test_reverb.py — Freeverb-inspired reverb smoke test.

Drives the reverb with 1 Hz clock pulses so you can clearly hear
the tail decay without a continuous source masking it.

Controls
--------
  mix   : wet/dry blend (0 = dry, 1 = 100 % wet)
  decay : feedback gain — higher ⟹ longer tail (try 0.8 – 0.95)
  damp  : high-freq damping — 0 = bright, 1 = dark/warm

Run:
    python test/test_reverb.py
"""

import egress as eg
import module_library as modlib

SAMPLE_RATE = 44100

# --- Reverb settings to audition ---
MIX   = 0.5    # 50/50 wet/dry
DECAY = 0.88   # moderately long tail
DAMP  = 0.45   # slightly warm


def build_patch():
    graph = eg.Graph()

    # 1 Hz clock — fires a brief pulse once per second
    clock_type = modlib.clock("PulseClock")
    clock = clock_type(graph=graph)
    clock.freq = 1.0                 # 1 pulse per second
    clock.ratios_in = eg.array([1.0])

    # Reverb module
    reverb_type = modlib.reverb("MainReverb")
    reverb = reverb_type(graph=graph)

    reverb.input = 1.0 * clock.output  # coerce OutputPort → SignalExpr
    reverb.mix   = MIX
    reverb.decay = DECAY
    reverb.damp  = DAMP

    # Mono reverb output to both stereo channels
    eg.add_output(reverb.output, graph=graph)
    eg.add_output(reverb.output, graph=graph)

    return graph, clock, reverb


def main():
    graph, clock, reverb = build_patch()

    print(
        f"Reverb test — 1 Hz pulses into reverb "
        f"(mix={MIX}, decay={DECAY}, damp={DAMP})"
    )
    print("You should hear a click once per second with a decaying reverb tail.")

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
