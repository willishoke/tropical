# bananas
### An esoteric programming language


**bananas** was built with the goal of providing a language with minimal syntax for simulation of analog computation. The name comes from the banana cables used to make connections between modules in early examples of analog computer hardware. 

Is **bananas** Turing complete? It's likely, given that it can simulate a recurrant neural network. However, its output is so unconventional that it would be exceedingly inefficient to simulate build something like a Universal Turing Machine.

**bananas** is strongly typed. There are a number of different modules, including:

- `INT` Integrator
- `DIF` Differentiator
- `ADD` Adder
- `MUL` Multiplier
- `INV` Inverter
- `DEL` Delay
- `CLK` Clock generator
- `SIN` Sinusoidal waveform generator

A module must be declared before use. Declarations look like this:

`out1 : OUT`

This declares module `out1` with type `OUT`. All module names may contain any numeric values. All types are 3 letters long and uppercase. If more than one module is declared with the same name, the parser will throw an error. 

To connect two modules `mult1` and `out1`, you can write:

`mult1 out`

The parser is written with `Attoparsec`, a minimal and efficient monadic parsing library. Since "programs" for analog computers often contain feedback loops, the parser constructs not an AST (abstract syntax tree), but an ASG (abstract syntax graph).

To compile, use the command

`stack run <input.nana> <output.wav> <num_samples>`

This will compile the input file and output a `.wav` file in the `waveforms` folder. This can be analyzed in any audio editing program, and in many cases produces interesting sonic artifacts.
