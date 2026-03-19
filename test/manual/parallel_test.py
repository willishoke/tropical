import egress as eg
import module_library as modlib


VOICE_COUNT = 1
WORKER_COUNT = 8
SAMPLE_RATE = 44100
NODE_COUNT = 16  # 4x4 waveguide


def build_patch():
    graph = eg.Graph()
    #graph.worker_count = WORKER_COUNT

    voices = []
    for index in range(VOICE_COUNT):
        clock_type = modlib.clock(f"ClockChime4_{index}")
        topo_type = modlib.topo_waveguide(4, 4, f"TopoChime4_{index}")

        clock = clock_type(graph=graph)
        waveguide = topo_type(graph=graph)

        clock.freq = 0.06 + (0.005 * index)
        clock.ratios_in = eg.array([1.0, 1.333, 1.5, 2.0])

        waveguide.g = 0.035
        waveguide.decay = 0.4
        waveguide.brightness = 0.92

        # Partially override the fc array (other nodes keep waveguide defaults).
        fc_expr = waveguide.fc.expr
        fc_expr = eg.array_set(fc_expr, 0,  42.0 + (3.0 * index))
        fc_expr = eg.array_set(fc_expr, 5,  63.0 + (4.0 * index))
        fc_expr = eg.array_set(fc_expr, 10, 94.0 + (5.0 * index))
        fc_expr = eg.array_set(fc_expr, 15, 141.0 + (6.0 * index))
        waveguide.fc = fc_expr

        # Build signal-input array: index into clock's ratios_out array output.
        input_expr = eg.array([0.0] * NODE_COUNT)
        input_expr = eg.array_set(input_expr, 0,  2.5 * clock.ratios_out[0])
        input_expr = eg.array_set(input_expr, 5,  1.9 * clock.ratios_out[1])
        input_expr = eg.array_set(input_expr, 10, 1.4 * clock.ratios_out[2])
        input_expr = eg.array_set(input_expr, 15, 1.1 * clock.ratios_out[3])
        waveguide.input = input_expr

        eg.add_output(waveguide.out[0], graph=graph)
        eg.add_output(waveguide.out[15], graph=graph)
        voices.append((clock, waveguide))

    return graph, voices


def main():
    graph, voices = build_patch()
    dac = eg.DAC(graph, sample_rate=SAMPLE_RATE, channels=2)
    dac.start()
    try:
        input("playing... press Enter to stop ")
    finally:
        dac.stop()

    print(
        {
            "voice_count": len(voices),
            "worker_count": graph.worker_count,
            "dac": dac.callback_stats(),
        }
    )


if __name__ == "__main__":
    main()
