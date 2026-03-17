import egress as eg
import egress.module_library as modlib


VCO = modlib.vco()
Phaser16 = modlib.phaser16()


def main():
    osc = VCO()
    phaser = Phaser16()

    osc.freq = 220.0
    osc.fm = 0.0
    osc.fm_index = 5.0

    phaser.input = osc.sin
    phaser.feedback = 0.65
    phaser.lfo_speed = 0.2

    eg.add_output(phaser.output)
    dac = eg.DAC(sample_rate=44100, channels=2)

    # Manual single-buffer render is still available:
    eg.graph().process()
    print("first 8 samples:", eg.graph().output_buffer()[:8])

    # Realtime output is controlled separately from the graph.
    dac.start()
    input("playing... press Enter to stop")
    dac.stop()


if __name__ == "__main__":
    main()
