import egress as eg


def main():
    osc = eg.VCO(440)
    idx = eg.CONST(3.0)
    eg.connect(idx.out, osc.fm_index)
    eg.add_output(osc.sin)
    dac = eg.DAC(sample_rate=44100, channels=2)

    # Manual single-buffer render is still available:
    eg.graph().process()
    print("first 8 samples:", eg.graph().output_buffer()[:8])

    # Realtime output is controlled separately from the graph.
    dac.start()
    input("playing... press Enter to stop")
    dac.stop()
    eg.disconnect(idx.out, osc.fm_index)
    eg.graph().destroy_module(idx.name)


if __name__ == "__main__":
    main()
