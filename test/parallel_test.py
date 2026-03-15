import egress as eg
import module_library as modlib


VOICE_COUNT = 1
WORKER_COUNT = 8
SAMPLE_RATE = 44100


def build_patch():
    graph = eg.graph()
    #graph.set_worker_count(WORKER_COUNT)

    voices = []
    for index in range(VOICE_COUNT):
        clock_type = modlib.clock(f"ClockChime4_{index}")
        topo_type = modlib.topo_waveguide(4, 4, f"TopoChime4_{index}")

        clock = clock_type()
        waveguide = topo_type()

        clock.freq = 0.06 + (0.005 * index)
        clock.ratios_in = [1.0, 1.333, 1.5, 2.0]

        waveguide.g = 0.035
        waveguide.decay = 0.4
        waveguide.brightness = 0.92

        fc = getattr(waveguide, "fc")
        fc[0] = 42.0 + (3.0 * index)
        fc[5] = 63.0 + (4.0 * index)
        fc[10] = 94.0 + (5.0 * index)
        fc[15] = 141.0 + (6.0 * index)

        signal_input = getattr(waveguide, "input")
        signal_input[0] = 2.5 * clock.ratios_out[0]
        signal_input[5] = 1.9 * clock.ratios_out[1]
        signal_input[10] = 1.4 * clock.ratios_out[2]
        signal_input[15] = 1.1 * clock.ratios_out[3]

        eg.add_output(waveguide.out[0])
        eg.add_output(waveguide.out[15])
        voices.append((clock, waveguide))

    return graph, voices


def main():
    graph, voices = build_patch()
    dac = eg.DAC(sample_rate=SAMPLE_RATE, channels=2)
    dac.start()
    try:
        input("playing... press Enter to stop ")
    finally:
        dac.stop()

    print(
        {
            "voice_count": len(voices),
            "worker_count": graph.worker_count(),
            "graph": graph.profile_stats(),
            "dac": dac.callback_timing_stats(),
        }
    )


if __name__ == "__main__":
    main()
