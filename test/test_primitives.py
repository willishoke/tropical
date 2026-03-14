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


AllpassStage = eg.define_stateful_function(
    inputs=["x", "a"],
    outputs=["y"],
    regs={"x_prev": 0.0, "y_prev": 0.0},
    process=lambda inp, reg: (
        {
            "y": -inp["a"] * inp["x"] + reg["x_prev"] + inp["a"] * reg["y_prev"],
        },
        {
            "x_prev": inp["x"],
            "y_prev": -inp["a"] * inp["x"] + reg["x_prev"] + inp["a"] * reg["y_prev"],
        },
    ),
)


StatefulProbe = eg.define_module(
    name="StatefulProbe",
    inputs=["x", "a"],
    outputs=["single", "cascade"],
    regs={},
    process=lambda inp, reg: (
        {
            "single": AllpassStage(inp["x"], inp["a"]),
            "cascade": AllpassStage(AllpassStage(inp["x"], inp["a"]), inp["a"]),
        },
        {},
    ),
)


ArrayInputProbe = eg.define_module(
    name="ArrayInputProbe",
    inputs=["x", "weights"],
    outputs=["mixed"],
    regs={},
    process=lambda inp, reg: (
        {
            "mixed": (inp["weights"] * inp["x"])[0]
            + (inp["weights"] * inp["x"])[1]
            + (inp["weights"] * inp["x"])[2],
        },
        {},
    ),
)


IndexedArraySource = eg.define_module(
    name="IndexedArraySource",
    inputs=["x"],
    outputs=["pair"],
    regs={},
    process=lambda inp, reg: (
        {"pair": eg.array([inp["x"], inp["x"] * 2.0, inp["x"] * 3.0])},
        {},
    ),
)


IndexedArraySink = eg.define_module(
    name="IndexedArraySink",
    inputs=["weights"],
    outputs=["lane0", "lane1", "lane2"],
    regs={},
    process=lambda inp, reg: (
        {
            "lane0": inp["weights"][0],
            "lane1": inp["weights"][1],
            "lane2": inp["weights"][2],
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

    stateful = StatefulProbe()
    stateful.x = 1.0
    stateful.a = 0.5

    eg.add_output(stateful.single)
    eg.add_output(stateful.cascade)
    eg.graph().process()

    stateful_buf = eg.graph().output_buffer()
    expected_stateful = expected + (-0.25 / 20.0)
    assert math.isclose(stateful_buf[0], expected_stateful, rel_tol=1e-9, abs_tol=1e-9)

    eg.graph().process()
    stateful_buf = eg.graph().output_buffer()
    expected_mix = expected + (2.0 / 20.0)
    assert math.isclose(stateful_buf[0], expected_mix, rel_tol=1e-9, abs_tol=1e-9)

    array_probe = ArrayInputProbe()
    array_probe.x = 0.5
    array_probe.weights = [2.0, 4.0, 6.0]

    eg.add_output(array_probe.mixed)
    eg.graph().process()

    array_buf = eg.graph().output_buffer()
    expected_array_mix = expected_mix + (6.0 / 20.0)
    assert math.isclose(array_buf[0], expected_array_mix, rel_tol=1e-9, abs_tol=1e-9)

    array_stats = array_probe.compile_stats
    assert array_stats["jit_status"] == "numeric JIT active"
    assert array_stats["numeric_jit_instruction_count"] > 0

    array_probe.x = 1.0
    array_probe.weights = [1.0, 1.0, 1.0, 10.0]

    eg.graph().process()
    array_buf = eg.graph().output_buffer()
    expected_array_mix = expected_mix + (3.0 / 20.0)
    assert math.isclose(array_buf[0], expected_array_mix, rel_tol=1e-9, abs_tol=1e-9)

    array_stats = array_probe.compile_stats
    assert array_stats["jit_status"] == "numeric JIT active"

    eg.graph().destroy_module(probe.name)
    eg.graph().destroy_module(stateful.name)
    eg.graph().destroy_module(array_probe.name)

    source = IndexedArraySource()
    source.x = 1.5

    source_stats = source.compile_stats
    assert source_stats["jit_status"] == "numeric JIT active"
    assert source_stats["numeric_jit_instruction_count"] > 0

    sink = IndexedArraySink()
    sink.weights = [0.0, 4.0, 5.0]
    sink.weights[0] = source.pair[1]

    eg.add_output(sink.lane0)
    eg.add_output(sink.lane1)
    eg.add_output(sink.lane2)
    eg.graph().process()

    indexed_buf = eg.graph().output_buffer()
    assert math.isclose(indexed_buf[0], 9.0 / 20.0, rel_tol=1e-9, abs_tol=1e-9)

    eg.graph().process()
    indexed_buf = eg.graph().output_buffer()
    assert math.isclose(indexed_buf[0], 12.0 / 20.0, rel_tol=1e-9, abs_tol=1e-9)

    eg.graph().destroy_module(source.name)
    eg.graph().destroy_module(sink.name)

    direct_source = IndexedArraySource()
    direct_source.x = 2.0
    eg.add_output(direct_source.pair[2])
    eg.graph().process()
    direct_buf = eg.graph().output_buffer()
    assert math.isclose(direct_buf[0], 6.0 / 20.0, rel_tol=1e-9, abs_tol=1e-9)

    eg.graph().process()
    direct_buf = eg.graph().output_buffer()
    assert math.isclose(direct_buf[0], 6.0 / 20.0, rel_tol=1e-9, abs_tol=1e-9)


if __name__ == "__main__":
    main()