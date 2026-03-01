import egress as eg


def main():
    graph = eg.Graph(buffer_length=512)
    dac = eg.DAC(graph, sample_rate=44100, channels=2)

    _osc = eg.VCO(graph, "vco1", 440)
    _idx = eg.CONST(graph, "idx", 3.0)
    graph.connect("idx", eg.CONSTOut.OUT, "vco1", eg.VCOIn.FM_INDEX)
    graph.add_output("vco1", eg.VCOOut.SIN)

    # Manual single-buffer render is still available:
    graph.process()
    print("first 8 samples:", graph.output_buffer()[:8])

    # Realtime output is controlled separately from the graph.
    dac.start()
    input("playing... press Enter to stop")
    dac.stop()

    graph.disconnect("idx", eg.CONSTOut.OUT, "vco1", eg.VCOIn.FM_INDEX)
    graph.destroy_module("idx")


if __name__ == "__main__":
    main()
