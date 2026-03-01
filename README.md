# egress

### Willis Hoke

## Intro

`egress` is a an object-oriented C++ library for realtime emulation of analog synthesis. It is built to be lean, efficient, and portable, although it is currently only tested on MacOS. No external libraries are required. Sample rates down to 64 samples / second have been tested. Sample implementations are provided for voltage controlled oscillators (`VCO`), four-quadrant multipliers (`VCA`), and two-way analog multiplexers (`MUX`). A full suite of tests demonstrates functionality of each of the modules, showing waveform outputs for each of the VCO outputs and demonstrating basic exponential FM and linear AM.

![demo](./img/testchaos.png)


This project owes a giant technical debt to Andrew Belt's `VCVRack` project. Although the architecture is different, it served as an inspiration throughout the development process.

## Build

Build the Python extension with CMake:

### Python frontend (`pybind11`)

A Python extension module (`egress`) is available via `pybind11`.

```bash
cmake -S . -B build -DEGRESS_BUILD_PYTHON=ON
cmake --build build
```

Then import from Python:

```python
import egress as eg

graph = eg.Graph(2048)
dac = eg.DAC(graph, sample_rate=44100, channels=2)

osc = eg.VCO(graph, "osc", 440)
graph.add_output("osc", eg.VCOOut.SIN)

dac.start()
# mutate graph live from a REPL while audio is running
dac.stop()
```

Supported module wrappers (constructor auto-registers into graph):
- `VCO(graph, name, freq_hz)`
- `MUX(graph, name)`
- `VCA(graph, name)`
- `ENV(graph, name, rise_ms, fall_ms)`
- `DELAY(graph, name, buffer_size_samples)`
- `CONST(graph, name, value)`

Lifecycle methods:
- `destroy_module(name)`
- `connect(from_module, from_output_id, to_module, to_input_id)`
- `disconnect(from_module, from_output_id, to_module, to_input_id)`
- `add_output(module_name, output_id)`

Realtime output methods:
- `dac = DAC(graph, sample_rate=44100, channels=2)`
- `dac.start()`
- `dac.stop()`

`Graph.process()` still renders a single buffer for offline inspection or testing; realtime playback uses `DAC.start()` and `DAC.stop()`.

## Graph

`Graph` is responsible for storing modules and managing connections between them. It also manages mixing and stores the output buffer. Racks store modules using an associative map with a unique name as key and `unique_ptr` to a module object as value. This allows for efficient lookup and constant-time iteration through the module list. Connections are stored using an associative map, with an output module's name and output ID as keys and module name and input ID as values. `Graph` has a `process` method that will iterate through the computation graph, sending output values from one module to another. It then calls each module's `process` method and stores this value in the output buffer. Since the buffer updates occur sequentially, the connection latency between any two modules is only a single sample. 

## Modules

`Module` is a base class that manages inputs, outputs, and postprocessing. Each time a module has finished processing, it should invoke the base class `postprocess` method to reset to default input values. All outputs are clipped to the range [-10.0, 10.0]. It is assumed that most signals are bipolar and in range [-5.0, 5.0]. Each subclass of `Module` should declare an enumeration specifying its inputs and outputs.

### VCO
#### A voltage controlled saw-core FM oscillator

`VCO` is a standard oscillator, with outputs for `saw`, `tri`, `sin`, and `sqr` waves. The oscillator follows the typical 1V / octave standard, so a value of `1.0` present at the `FM` input will result in a tone exactly 1 octave above the fundamental. An optional `FM_INDEX` parameter allows dynamic scaling of FM values. The constructor for `VCO` takes a single value specifying the intial frequency. All tests were run with a wave at 440hz for 2048 samples.

Inputs: `FM`, `FM_INDEX`

Outputs: 

`SIN`

![sine](./img/testsin.png)

`SQR`

![square](./img/testsqr.png)

`SAW`

![sawtooth](./img/testsaw.png)

`TRI`

![triangle](./img/testtri.png)

#### Frequency Modulation

FM was tested with a base freqency of 1000hz, a modulation frequency of 200hz, and an FM index of 3.

![fm](./img/testfm.png)


### VCA
#### Voltage controlled amplifier

`VCA` takes two inputs, `IN1` and `IN2`, and outputs `OUT`. The output voltage is downscaled by a factor of 5 to accomodate +/-5V control signals.

Amplitude modulation was tested with a base frequency of 1000hz and a modulation frequency of 200hz at 2048 samples.

![vca](./img/testam.png)


### MUX
#### Voltage controlled analog multiplexer

`MUX` takes three inputs, `IN1`, `IN2`, and `CTRL`. The input presented at `OUT1` is chosen based on the polarity of the control signal.

`MUX` was tested by generating three waveforms, one at 1000hz, one at 2000hz, and a third at 100hz. The third waveform was used as the control input to switch between the two outputs.

![MUX](./img/testmux.png)


### CONST
#### Constant value

`CONST` takes no inputs. It can be used to provide an offset value -- i.e., for changing an FM modulation index.


### DELAY
#### Under development

## Testing

Tests can be compiled with `make debug`. The `test` directory contains Python scripts for visualizing test outputs.

## Next Steps

Several methods are still incomplete, including those to remove modules or connections. More validation checks should be added to ensure module names are valid, etc. While the syntax for declaring modules is fairly terse, it doesn't seem right to have the caller be managing memory. This should probably be accomplished by factory functions in the Graph class. Better test automation would be nice -- should be able to write a script to run all tests sequentially.
 
It would be worthwhile to test alternative implementations for storing modules and their connections. Even just using vectors might end up being more efficient due to the linear memory layout. This libarary lacks any front end implementation, but it would be relatively straightforward to augment it with a simple parser to process user input for interactive use.

## License

Free use of `egress` is permitted under the terms of the MIT License.
