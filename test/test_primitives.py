import math

import egress as eg


PrimitiveProbe = eg.define_module(
    name="PrimitiveProbe",
    inputs=["x", "lo", "hi", "z"],
    outputs=["abs_out", "clamp_out", "log_out"],
    regs={},
    process=lambda inp, reg: (
        {
            "abs_out": abs(inp["x"]),
            "clamp_out": eg.clamp(inp["x"], inp["lo"], inp["hi"]),
            "log_out": eg.log(inp["z"]),
        },
        {},
    ),
)


def main():
    probe = PrimitiveProbe()
    probe.x = -3.5
    probe.lo = -1.0
    probe.hi = 1.0
    probe.z = math.e ** 2.0

    eg.add_output(probe.abs_out)
    eg.add_output(probe.clamp_out)
    eg.add_output(probe.log_out)

    eg.graph().process()
    buf = eg.graph().output_buffer()
    expected = (3.5 + -1.0 + 2.0) / 20.0

    assert len(buf) == 1024
    assert all(math.isclose(sample, expected, rel_tol=1e-9, abs_tol=1e-9) for sample in buf[:8])

    stats = probe.compile_stats
    assert stats["instruction_count"] > 0
    assert stats["jit_status"] == "numeric JIT active"
    assert stats["numeric_jit_instruction_count"] > 0

    print("primitive-ok", round(buf[0], 6), stats["numeric_jit_instruction_count"], stats["jit_status"])


if __name__ == "__main__":
    main()