# egress

## Intro

`egress` is a C++ library with a Python frontend for realtime audio synthesis. Modules can be defined in Python using built-in or user-defined operations, and connections defined using straightforward expression syntax. JIT compilation for module definitions ensures blazing-fast realtime execution with native support for audio playback. It is built to be fast and portable, although it is currently only tested on macOS.

![demo](./img/testchaos.png)

## Build

Build the Python extension with `make`:

```bash
make build
make repl
```

`make build` configures and builds the extension. `make repl` launches Python with `build/` on `PYTHONPATH`, so `import egress` works immediately.

Then import from Python:

```python
import egress as eg

Osc = eg.define_module(
    name="Osc",
    inputs=["freq", "amp"],
    outputs=["out"],
    regs={},
    process=lambda inp, reg: (
        {
            "out": inp["amp"]
            * eg.sin(6.283185307179586 * eg.sample_index() * inp["freq"] / eg.sample_rate())
        },
        {},
    ),
)

osc = Osc()
osc.freq = 440.0
osc.amp = 5.0
eg.add_output(osc.out)
dac = eg.DAC(sample_rate=44100, channels=2)

dac.start()
# mutate graph live from a REPL while audio is running
dac.stop()
```

Python uses a single process-wide default graph. Module instances auto-register into that graph when they are created.

Lifecycle methods:
- `connect(output_port, input_port)`
- `disconnect(output_port, input_port)`
- `add_output(output_port)`
- `graph().prime_numeric_jit()` (prewarms current numeric JIT kernels using the graph's current input wiring before realtime start)
- `graph().set_worker_count(n)` (opt-in graph-level parallelism; `1` keeps single-threaded execution)
- `graph().set_fusion_enabled(True/False)` (opt-in graph-level fused input/mix expression kernels; still off by default because input-kernel-only fusion can slow small graphs, but full primitive-body fusion may still activate automatically when the whole graph can run as one numeric fused body)
- `graph().destroy_module(name)` (advanced/manual control on the singleton graph)

Input assignment shortcuts:
- `module.input = output_port`
- `module.input = 3.0`
- `module.input = 0.3 * lfo.sin + 0.2 * lfo2.tri * lfo3.saw`
- `module.input += expr`
- `module.input = None` clears that input's current routing

User-defined modules can be declared from Python by returning output expressions and next-register expressions from a pure definition function:

```python
import egress as eg

Delay1 = eg.define_module(
    name="Delay1",
    inputs=["input"],
    outputs=["output"],
    regs={"z": 0.0},
    process=lambda inp, reg: (
        {"output": reg["z"]},
        {"z": inp["input"]},
    ),
)

delay = Delay1()
delay.input = lfo.sin
eg.add_output(delay.output)
```

`reg[...]` reads the current register bank for the sample. Returned register assignments become visible on the next sample, so the example above behaves as a one-sample delay. Built-in symbolic values are available as `eg.sample_index()` and `eg.sample_rate()`.

Expression trees now preserve simple scalar types: `bool`, `int`, and `float`. Arithmetic follows Python-style numeric promotion for those scalar types, so `3 + 3.0` produces a floating-point result. Comparisons return symbolic booleans that render as `1.0` / `0.0` when fed into module inputs or outputs. Python's `not` operator is not overloadable for symbolic expressions, so use `eg.logical_not(expr)` instead.

Exponentiation is now a first-class symbolic operation as well. Use either `lhs ** rhs` or `eg.pow(lhs, rhs)` inside pure functions and module definitions.

For reusable stateful building blocks inside a module definition, define an ordinary `eg.define_module(...)` and call it directly from another module body. The call does not create a top-level runtime graph node; instead it becomes a local child module inside the enclosing module. User-declared `regs={...}` remain module-owned state, while composition-introduced state such as explicit delays is stored in compiler-managed composite state.

```python
Allpass = eg.define_module(
    name="Allpass",
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
```

Composite modules built from nested module calls and explicit delays now participate in the numeric JIT when their state is scalar or static 1-D array based. Dynamic `array_state(...)` registers are still not supported inside same-tick module calls inside another module body.

For an explicit one-sample boundary inside a module body, use `eg.delay(expr, init=...)`. This creates compiler-managed composite delay state for that connection and returns the previous sample's value, which makes delayed composition explicit even when chaining module calls inline.

```python
Diff = eg.define_module(
    name="Diff",
    inputs=["x"],
    outputs=["dx"],
    regs={},
    process=lambda inp, reg: (
        {"dx": inp["x"] - eg.delay(inp["x"], init=0.0)},
        {},
    ),
)
```

Module registers may also hold static 1-D arrays of scalar values. Use Python lists / tuples for register initialization and `eg.array([...])` when constructing a new array expression inside `process`. Arithmetic between arrays and scalars broadcasts elementwise, and array indexing is explicit:

```python
Shift = eg.define_module(
    name="Shift",
    inputs=["input"],
    outputs=["tap"],
    regs={"buf": [0.0, 0.0, 0.0]},
    process=lambda inp, reg: (
        {"tap": reg["buf"][0]},
        {"buf": eg.array([reg["buf"][1], reg["buf"][2], inp["input"]])},
    ),
)
```

Array-valued module inputs are supported, including on the numeric JIT path. Graph outputs remain scalar.

Array expressions also support indexed reads and updates. On the expression side you can write `eg.array_set(arr, idx, value)`, and on graph inputs you can use Python indexing sugar such as `wg.in[0] = clock.out[0]` to rebuild a single array lane without replacing the entire input expression manually.

You can also collect reusable user-defined modules in ordinary Python source files. For example, [module_library.py](/Users/willishoke/egress/module_library.py) defines a polyBLEP-style VCO plus both compact and 16-stage phasers:

```python
import egress as eg
import module_library as modlib

Osc = modlib.vco()
osc = Osc()
osc.freq = 220.0
osc.fm = 0.0
osc.fm_index = 5.0
Phaser16 = modlib.phaser16()
phaser = Phaser16()
phaser.input = osc.sin
phaser.feedback = 0.65
phaser.lfo_speed = 0.2
eg.add_output(phaser.output)
```

Each input stores a single canonical expression tree. `input = ...` replaces that tree. `input += expr` and `connect(...)` append additional signal into the input sum.

Realtime output methods:
- `dac = DAC(sample_rate=44100, channels=2)`
- `dac.start()`
- `dac.stop()`

`graph().process()` still renders a single buffer for offline inspection or testing; realtime playback uses `DAC.start()` and `DAC.stop()`. `DAC.start()` now prewarms numeric JIT kernels against the current graph inputs before opening the audio stream so first-use ORC compilation stays off the realtime callback path.

## Graph

`Graph` stores modules, per-input expression trees, output taps, and the output buffer. Each input is represented by a single expression tree whose leaves are literals or references to module outputs. `Graph::process()` evaluates those input expressions sample-by-sample, processes modules, and mixes selected outputs into the output buffer. Top-level module references always read previous-sample outputs, so connected modules still have a single-sample boundary even when graph-level worker threads are enabled via `graph().set_worker_count(...)`.

Modules expose named inputs and outputs plus an optional register bank. After each sample, the runtime applies the register updates returned by `process(...)` and resets inputs to their default values for the next sample. Output signals are clipped to the range `[-10.0, 10.0]`; in practice most patches are expected to stay in the bipolar `[-5.0, 5.0]` range.

## Testing

Tests can be compiled with `make debug`. The `test` directory contains Python scripts for visualizing test outputs.

## License

Free use of `egress` is permitted under the terms of the MIT License.
