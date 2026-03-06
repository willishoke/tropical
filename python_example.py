import egress as eg


def main():
    osc = eg.VCO(440)
    lfo = eg.VCO(2.0)
    lfo2 = eg.VCO(0.37)
    lfo3 = eg.VCO(0.19)

    osc.fm_index = 3.0
    osc.fm = 0.3 * lfo.sin + 0.2 * lfo2.tri * lfo3.saw
    eg.add_output(osc.sin)
    dac = eg.DAC(sample_rate=44100, channels=2)

    # Manual single-buffer render is still available:
    eg.graph().process()
    print("first 8 samples:", eg.graph().output_buffer()[:8])

    # Realtime output is controlled separately from the graph.
    dac.start()
    input("playing... press Enter to stop")
    dac.stop()
    osc.fm = None


if __name__ == "__main__":
    main()
