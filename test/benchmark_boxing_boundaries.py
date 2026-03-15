import json
import time

import egress as eg


ITERATIONS = 400


Gain2 = eg.define_module(
    name="BenchGain2",
    inputs=["x"],
    outputs=["y"],
    regs={},
    process=lambda inp, reg: (
        {"y": inp["x"] * 2.0},
        {},
    ),
)


IndexedArraySource = eg.define_module(
    name="BenchIndexedArraySource",
    inputs=["x"],
    outputs=["pair"],
    regs={},
    process=lambda inp, reg: (
        {"pair": eg.array([inp["x"], inp["x"] * 2.0, inp["x"] * 3.0])},
        {},
    ),
)


IndexedArraySink = eg.define_module(
    name="BenchIndexedArraySink",
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


def run_process_loop(graph, iterations):
    start = time.perf_counter()
    for _ in range(iterations):
        graph.process()
    return (time.perf_counter() - start) * 1000.0


def run_scalar_chain(graph, iterations, fusion_enabled):
    graph.set_fusion_enabled(fusion_enabled)
    source = Gain2()
    sink = Gain2()
    tap_id = graph.add_output_tap(sink.y.module_name, sink.y.output_id)
    source.x = 2.0
    sink.x = source.y
    try:
        graph.prime_numeric_jit()
        graph.reset_profile_stats()
        elapsed_ms = run_process_loop(graph, iterations)
        return {
            "elapsed_ms": elapsed_ms,
            "source_compile": source.compile_stats,
            "source_runtime": source.runtime_stats,
            "sink_compile": sink.compile_stats,
            "sink_runtime": sink.runtime_stats,
            "graph_profile": graph.profile_stats(),
        }
    finally:
        graph.remove_output_tap(tap_id)
        graph.destroy_module(source.name)
        graph.destroy_module(sink.name)


def run_array_chain(graph, iterations, fusion_enabled):
    graph.set_fusion_enabled(fusion_enabled)
    source = IndexedArraySource()
    sink = IndexedArraySink()
    tap_id = graph.add_output_tap(sink.lane0.module_name, sink.lane0.output_id)
    source.x = 2.0
    sink.weights = [0.0, 0.0, 0.0]
    sink.weights[0] = source.pair[1]
    try:
        graph.prime_numeric_jit()
        graph.reset_profile_stats()
        elapsed_ms = run_process_loop(graph, iterations)
        return {
            "elapsed_ms": elapsed_ms,
            "source_compile": source.compile_stats,
            "source_runtime": source.runtime_stats,
            "sink_compile": sink.compile_stats,
            "sink_runtime": sink.runtime_stats,
            "graph_profile": graph.profile_stats(),
        }
    finally:
        graph.remove_output_tap(tap_id)
        graph.destroy_module(source.name)
        graph.destroy_module(sink.name)


def main():
    graph = eg.graph()
    results = {
        "iterations": ITERATIONS,
        "scalar_chain": {
            "fusion_off": run_scalar_chain(graph, ITERATIONS, False),
            "fusion_on": run_scalar_chain(graph, ITERATIONS, True),
        },
        "array_chain": {
            "fusion_off": run_array_chain(graph, ITERATIONS, False),
            "fusion_on": run_array_chain(graph, ITERATIONS, True),
        },
    }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
