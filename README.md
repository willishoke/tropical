# egress

### Willis Hoke

## Intro

`egress` is a C++ library with a Python frontend for realtime signal-graph synthesis. The current Python API is centered on user-defined modules: you describe a module in Python by returning symbolic output expressions and next-register values, and `egress` evaluates that graph sample-by-sample. It is built to be lean and portable, although it is currently only tested on macOS.

![demo](./img/testchaos.png)


This project owes a giant technical debt to Andrew Belt's `VCVRack` project. Although the architecture is different, it served as an inspiration throughout the development process.

## Build

Build the Python extension with `make`:

### Python frontend (`pybind11`)

A Python extension module (`egress`) is available via `pybind11`.

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

Arrays are currently limited to module expressions and registers. Graph inputs and outputs remain scalar.

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

`graph().process()` still renders a single buffer for offline inspection or testing; realtime playback uses `DAC.start()` and `DAC.stop()`.

## Graph

`Graph` stores modules, per-input expression trees, output taps, and the output buffer. Each input is represented by a single expression tree whose leaves are literals or references to module outputs. `Graph::process()` evaluates those input expressions sample-by-sample, processes modules in dependency order, and mixes selected outputs into the output buffer. Since evaluation and processing proceed sequentially, the latency between connected modules is a single sample.

Modules expose named inputs and outputs plus an optional register bank. After each sample, the runtime applies the register updates returned by `process(...)` and resets inputs to their default values for the next sample. Output signals are clipped to the range `[-10.0, 10.0]`; in practice most patches are expected to stay in the bipolar `[-5.0, 5.0]` range.

## Testing

Tests can be compiled with `make debug`. The `test` directory contains Python scripts for visualizing test outputs.

## Next Steps

Several methods are still incomplete, including those to remove modules or connections. More validation checks should be added to ensure module names are valid, etc. While the syntax for declaring modules is fairly terse, it doesn't seem right to have the caller be managing memory. This should probably be accomplished by factory functions in the Graph class. Better test automation would be nice -- should be able to write a script to run all tests sequentially.
 
It would be worthwhile to test alternative implementations for storing modules and their connections. Even just using vectors might end up being more efficient due to the linear memory layout. This libarary lacks any front end implementation, but it would be relatively straightforward to augment it with a simple parser to process user input for interactive use.

## License

Free use of `egress` is permitted under the terms of the MIT License.
