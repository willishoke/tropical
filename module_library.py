import egress as eg


_TWO_PI = 6.283185307179586


def phaser16(name="Phaser16"):
    stage_count = 16
    x_regs = [f"x{i}" for i in range(stage_count)]
    y_regs = [f"y{i}" for i in range(stage_count)]

    regs = {"fb": 0.0}
    for reg_name in x_regs + y_regs:
        regs[reg_name] = 0.0

    def process(inp, reg):
        lfo = eg.sin(_TWO_PI * eg.sample_index() * inp["lfo_speed"] / eg.sample_rate())

        # Keep the one-pole allpass coefficient in a stable range.
        a = 0.6 + 0.35 * lfo
        stage_input = inp["input"] + inp["feedback"] * reg["fb"]

        next_regs = {}
        for idx in range(stage_count):
            x_prev = reg[x_regs[idx]]
            y_prev = reg[y_regs[idx]]
            stage_output = -a * stage_input + x_prev + a * y_prev
            next_regs[x_regs[idx]] = stage_input
            next_regs[y_regs[idx]] = stage_output
            stage_input = stage_output

        next_regs["fb"] = stage_input

        return (
            {
                "output": 0.5 * inp["input"] + 0.5 * stage_input,
                "lfo": lfo,
            },
            next_regs,
        )

    return eg.define_stateful_module(
        name=name,
        inputs=["input", "feedback", "lfo_speed"],
        outputs=["output", "lfo"],
        regs=regs,
        process=process,
        sample_rate=44100.0,
    )
