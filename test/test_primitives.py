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


AllpassStage = eg.define_module(
    name="AllpassStage",
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


ModuleComposeProbe = eg.define_module(
    name="ModuleComposeProbe",
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


Gain2 = eg.define_module(
    name="Gain2",
    inputs=["x"],
    outputs=["y"],
    regs={},
    process=lambda inp, reg: (
        {"y": inp["x"] * 2.0},
        {},
    ),
)


DelayComposeProbe = eg.define_module(
    name="DelayComposeProbe",
    inputs=["x"],
    outputs=["current", "delayed", "cascade"],
    regs={},
    process=lambda inp, reg: (
        {
            "current": Gain2(inp["x"]),
            "delayed": eg.delay(Gain2(inp["x"])),
            "cascade": Gain2(eg.delay(Gain2(inp["x"]))),
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


SampleIndexProbe = eg.define_module(
    name="SampleIndexProbe",
    inputs=[],
    outputs=["idx"],
    regs={},
    process=lambda inp, reg: (
        {"idx": eg.sample_index() / 1024.0},
        {},
    ),
)


def main():
    graph = eg.graph()
    assert graph.worker_count() == 1

    compose_stats = ModuleComposeProbe.composition_stats
    assert compose_stats["node_count"] >= 4
    assert compose_stats["same_tick_edge_count"] >= 3
    assert compose_stats["delayed_edge_count"] == 0
    assert compose_stats["same_tick_schedule_size"] == compose_stats["node_count"]
    assert compose_stats["delayed_node_count"] == 0
    assert compose_stats["scheduled_node_count"] == compose_stats["node_count"]
    assert compose_stats["delayed_state_count"] == 0

    delay_stats = DelayComposeProbe.composition_stats
    assert delay_stats["node_count"] >= 5
    assert delay_stats["same_tick_edge_count"] >= 3
    assert delay_stats["delayed_edge_count"] >= 1
    assert delay_stats["same_tick_schedule_size"] == delay_stats["node_count"]
    assert delay_stats["delayed_node_count"] >= 1
    assert delay_stats["scheduled_node_count"] == delay_stats["node_count"]
    assert delay_stats["delayed_state_count"] >= 1

    probe = PrimitiveProbe()
    probe.x = -3.5
    probe.lo = -1.0
    probe.hi = 1.0
    probe.z = math.e ** 2.0

    graph.add_output(probe.abs_out.module_name, probe.abs_out.output_id)
    graph.add_output(probe.clamp_out.module_name, probe.clamp_out.output_id)
    graph.add_output(probe.log_out.module_name, probe.log_out.output_id)

    graph.process()
    buf = graph.output_buffer()
    expected = (3.5 + -1.0 + 2.0) / 20.0

    assert len(buf) == 1024
    assert all(math.isclose(sample, expected, rel_tol=1e-9, abs_tol=1e-9) for sample in buf[:8])

    stats = probe.compile_stats
    assert stats["instruction_count"] > 0
    assert stats["jit_status"] == "numeric JIT active"
    assert stats["numeric_jit_instruction_count"] > 0
    assert "composite_update_count" not in stats

    print("primitive-ok", round(buf[0], 6), stats["numeric_jit_instruction_count"], stats["jit_status"])

    stateful = StatefulProbe()
    stateful.x = 1.0
    stateful.a = 0.5

    graph.add_output(stateful.single.module_name, stateful.single.output_id)
    graph.add_output(stateful.cascade.module_name, stateful.cascade.output_id)
    graph.process()

    stateful_buf = graph.output_buffer()
    expected_stateful = expected + (-0.25 / 20.0)
    assert math.isclose(stateful_buf[0], expected_stateful, rel_tol=1e-9, abs_tol=1e-9)

    graph.process()
    stateful_buf = graph.output_buffer()
    expected_mix = expected + (2.0 / 20.0)
    assert math.isclose(stateful_buf[0], expected_mix, rel_tol=1e-9, abs_tol=1e-9)

    array_probe = ArrayInputProbe()
    array_probe.x = 0.5
    array_probe.weights = [2.0, 4.0, 6.0]

    graph.add_output(array_probe.mixed.module_name, array_probe.mixed.output_id)
    graph.process()

    array_buf = graph.output_buffer()
    expected_array_mix = expected_mix + (6.0 / 20.0)
    assert math.isclose(array_buf[0], expected_array_mix, rel_tol=1e-9, abs_tol=1e-9)

    array_stats = array_probe.compile_stats
    assert array_stats["jit_status"] == "numeric JIT active"
    assert array_stats["numeric_jit_instruction_count"] > 0

    array_probe.x = 1.0
    array_probe.weights = [1.0, 1.0, 1.0, 10.0]

    graph.process()
    array_buf = graph.output_buffer()
    expected_array_mix = expected_mix + (3.0 / 20.0)
    assert math.isclose(array_buf[0], expected_array_mix, rel_tol=1e-9, abs_tol=1e-9)

    array_stats = array_probe.compile_stats
    assert array_stats["jit_status"] == "numeric JIT active"

    graph.destroy_module(probe.name)
    graph.destroy_module(stateful.name)
    graph.destroy_module(array_probe.name)

    module_compose = ModuleComposeProbe()
    module_compose.x = 1.0
    module_compose.a = 0.5

    graph.add_output(module_compose.single.module_name, module_compose.single.output_id)
    graph.add_output(module_compose.cascade.module_name, module_compose.cascade.output_id)
    graph.process()

    module_buf = graph.output_buffer()
    assert math.isclose(module_buf[0], -0.25 / 20.0, rel_tol=1e-9, abs_tol=1e-9)

    module_stats = module_compose.compile_stats
    assert module_stats["jit_status"] == "numeric JIT active"
    assert module_stats["nested_module_count"] >= 2
    assert module_stats["numeric_jit_instruction_count"] > 0
    assert "composite_update_count" not in module_stats

    graph.process()
    module_buf = graph.output_buffer()
    assert math.isclose(module_buf[0], 2.0 / 20.0, rel_tol=1e-9, abs_tol=1e-9)

    graph.destroy_module(module_compose.name)

    delay_probe = DelayComposeProbe()
    delay_probe.x = 2.0

    graph.add_output(delay_probe.current.module_name, delay_probe.current.output_id)
    graph.add_output(delay_probe.delayed.module_name, delay_probe.delayed.output_id)
    graph.add_output(delay_probe.cascade.module_name, delay_probe.cascade.output_id)
    graph.process()

    delay_buf = graph.output_buffer()
    assert math.isclose(delay_buf[0], 4.0 / 20.0, rel_tol=1e-9, abs_tol=1e-9)

    delay_compile_stats = delay_probe.compile_stats
    assert delay_compile_stats["jit_status"] == "numeric JIT active"
    assert delay_compile_stats["numeric_jit_instruction_count"] >= 9
    assert "composite_update_count" not in delay_compile_stats

    graph.process()
    delay_buf = graph.output_buffer()
    assert math.isclose(delay_buf[0], 16.0 / 20.0, rel_tol=1e-9, abs_tol=1e-9)

    graph.destroy_module(delay_probe.name)

    source = IndexedArraySource()
    source.x = 1.5

    source_stats = source.compile_stats
    assert source_stats["jit_status"] == "numeric JIT active"
    assert source_stats["numeric_jit_instruction_count"] > 0

    sink = IndexedArraySink()
    sink.weights = [0.0, 4.0, 5.0]
    sink.weights[0] = source.pair[1]

    graph.add_output(sink.lane0.module_name, sink.lane0.output_id)
    graph.add_output(sink.lane1.module_name, sink.lane1.output_id)
    graph.add_output(sink.lane2.module_name, sink.lane2.output_id)
    graph.process()

    indexed_buf = graph.output_buffer()
    assert math.isclose(indexed_buf[0], 9.0 / 20.0, rel_tol=1e-9, abs_tol=1e-9)

    graph.process()
    indexed_buf = graph.output_buffer()
    assert math.isclose(indexed_buf[0], 12.0 / 20.0, rel_tol=1e-9, abs_tol=1e-9)

    graph.destroy_module(source.name)
    graph.destroy_module(sink.name)

    direct_source = IndexedArraySource()
    direct_source.x = 2.0
    eg.add_output(direct_source.pair[2])
    graph.process()
    direct_buf = graph.output_buffer()
    assert math.isclose(direct_buf[0], 6.0 / 20.0, rel_tol=1e-9, abs_tol=1e-9)

    graph.process()
    direct_buf = graph.output_buffer()
    assert math.isclose(direct_buf[0], 6.0 / 20.0, rel_tol=1e-9, abs_tol=1e-9)

    graph.destroy_module(direct_source.name)

    graph.prime_numeric_jit()

    graph.set_worker_count(2)
    assert graph.worker_count() == 2

    index_a = SampleIndexProbe()
    index_b = SampleIndexProbe()
    tap_a = graph.add_output_tap(index_a.idx.module_name, index_a.idx.output_id)
    tap_b = graph.add_output_tap(index_b.idx.module_name, index_b.idx.output_id)
    graph.process()
    index_a_buf = graph.output_tap_buffer(tap_a)
    index_b_buf = graph.output_tap_buffer(tap_b)
    assert all(math.isclose(index_a_buf[i], float(i) / 1024.0, rel_tol=1e-9, abs_tol=1e-9) for i in range(8))
    assert index_a_buf[:8] == index_b_buf[:8]

    graph.process()
    index_a_buf = graph.output_tap_buffer(tap_a)
    assert all(math.isclose(index_a_buf[i], 1.0 + (float(i) / 1024.0), rel_tol=1e-9, abs_tol=1e-9) for i in range(8))
    graph.remove_output_tap(tap_a)
    graph.remove_output_tap(tap_b)
    graph.destroy_module(index_a.name)
    graph.destroy_module(index_b.name)

    parallel_source = IndexedArraySource()
    parallel_source.x = 2.0
    parallel_sink = IndexedArraySink()
    parallel_sink.weights = [0.0, 0.0, 0.0]
    parallel_sink.weights[0] = parallel_source.pair[2]
    sink_tap = graph.add_output_tap(parallel_sink.lane0.module_name, parallel_sink.lane0.output_id)
    graph.process()
    sink_buf = graph.output_tap_buffer(sink_tap)
    assert math.isclose(sink_buf[0], 0.0, rel_tol=1e-9, abs_tol=1e-9)
    graph.process()
    sink_buf = graph.output_tap_buffer(sink_tap)
    assert math.isclose(sink_buf[0], 6.0, rel_tol=1e-9, abs_tol=1e-9)
    graph.remove_output_tap(sink_tap)
    graph.destroy_module(parallel_source.name)
    graph.destroy_module(parallel_sink.name)
    graph.set_worker_count(1)
    assert graph.worker_count() == 1

    assert graph.fusion_enabled() is False
    graph.set_fusion_enabled(True)
    assert graph.fusion_enabled() is True

    fused_source = Gain2()
    fused_source.x = 2.0
    fused_sink = Gain2()
    fused_sink.x = fused_source.y
    graph.add_output(fused_sink.y.module_name, fused_sink.y.output_id)
    graph.process()
    fused_buf = graph.output_buffer()
    assert math.isclose(fused_buf[0], 0.0, rel_tol=1e-9, abs_tol=1e-9)
    graph.process()
    fused_buf = graph.output_buffer()
    assert math.isclose(fused_buf[0], 8.0 / 20.0, rel_tol=1e-9, abs_tol=1e-9)
    graph.destroy_module(fused_source.name)
    graph.destroy_module(fused_sink.name)

    mixed_fusion_source = Gain2()
    mixed_fusion_source.x = 2.0
    mixed_fusion_sink = DelayComposeProbe()
    mixed_fusion_sink.x = mixed_fusion_source.y
    graph.add_output(mixed_fusion_sink.current.module_name, mixed_fusion_sink.current.output_id)
    graph.process()
    mixed_buf = graph.output_buffer()
    assert math.isclose(mixed_buf[0], 0.0, rel_tol=1e-9, abs_tol=1e-9)
    graph.process()
    mixed_buf = graph.output_buffer()
    assert math.isclose(mixed_buf[0], 8.0 / 20.0, rel_tol=1e-9, abs_tol=1e-9)
    graph.destroy_module(mixed_fusion_source.name)
    graph.destroy_module(mixed_fusion_sink.name)

    fusion_source = IndexedArraySource()
    fusion_source.x = 2.0
    fusion_sink = IndexedArraySink()
    fusion_sink.weights = [0.0, 4.0, 5.0]
    fusion_sink.weights[0] = fusion_source.pair[1]
    graph.add_output(fusion_sink.lane0.module_name, fusion_sink.lane0.output_id)
    graph.process()
    fusion_buf = graph.output_buffer()
    assert math.isclose(fusion_buf[0], 0.0, rel_tol=1e-9, abs_tol=1e-9)
    graph.process()
    fusion_buf = graph.output_buffer()
    assert math.isclose(fusion_buf[0], 4.0 / 20.0, rel_tol=1e-9, abs_tol=1e-9)
    graph.destroy_module(fusion_source.name)
    graph.destroy_module(fusion_sink.name)

    mix_source = IndexedArraySource()
    mix_source.x = 3.0
    eg.add_output(mix_source.pair[1] + 1.0)
    graph.process()
    mix_buf = graph.output_buffer()
    assert math.isclose(mix_buf[0], 7.0 / 20.0, rel_tol=1e-9, abs_tol=1e-9)
    graph.destroy_module(mix_source.name)

    graph.reset_profile_stats()
    graph.set_fusion_enabled(True)
    profile_source = IndexedArraySource()
    profile_source.x = 2.0
    profile_sink = IndexedArraySink()
    profile_sink.weights = [0.0, 0.0, 0.0]
    profile_sink.weights[0] = profile_source.pair[1]
    profile_tap = graph.add_output_tap(profile_sink.lane0.module_name, profile_sink.lane0.output_id)
    graph.prime_numeric_jit()
    graph.process()
    graph.process()
    boxing = graph.profile_stats()["boxing"]
    assert boxing["fused_current_output_sync"]["call_count"] > 0
    assert boxing["fused_prev_output_sync"]["call_count"] > 0
    source_runtime = profile_source.runtime_stats
    sink_runtime = profile_sink.runtime_stats
    assert source_runtime["numeric_output_materialize_call_count"] > 0
    assert source_runtime["materialized_array_outputs"] > 0
    assert sink_runtime["numeric_input_sync_call_count"] > 0
    assert sink_runtime["numeric_output_materialize_call_count"] > 0
    assert sink_runtime["materialized_scalar_outputs"] == 0
    graph.remove_output_tap(profile_tap)
    graph.destroy_module(profile_source.name)
    graph.destroy_module(profile_sink.name)


if __name__ == "__main__":
    main()
