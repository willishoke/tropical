import Tropical.Engine.Core
import Tropical.Engine.Compile
import Tropical.Engine.Crud
import Tropical.Engine.Wire
import Tropical.Engine.ProgramIO
import Tropical.Engine.Audio
import Tropical.Engine.Front

/-!
# The tropical IR engine — tool semantics, in Lean

The engine is the session's stateful shell: it holds the `SessionSt`
and the host port, runs the pure session semantics (`Tropical.Session.*`:
registration, ingest, lowering) against them, and publishes the result.
There is no compiler-service subprocess.

Every structural graph mutation ends in `syncCompile`: the mirror becomes one
resolved root directly, passes the Core check, partitions, and hot-swaps its
plan into the Lean-owned runtime. Parameter writes use live slots and do not
take this path. A failed structural compile leaves the mutated graph in place;
the previous kernel keeps playing and the error is recoverable.

The implementation is split by concern; this module re-exports the whole
surface so `import Tropical.Engine` sees it unchanged:

* `Engine.Core` — env, tool-arg access, lookup vocabulary
* `Engine.Compile` — mirror → resolved root → partition → hot-swap
* `Engine.Crud` — instance/program lifecycle handlers
* `Engine.Wire` — the wiring tools
* `Engine.ProgramIO` — save / export / ingest
* `Engine.Audio` — audio lifecycle and param control
* `Engine.Front` — tool dispatcher and boot
-/
