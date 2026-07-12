import Tropical.Engine.Core
import Tropical.Engine.Compile
import Tropical.Engine.Registry
import Tropical.Engine.Crud
import Tropical.Engine.Wire
import Tropical.Engine.ProgramIO
import Tropical.Engine.Audio
import Tropical.Engine.Front

/-!
# The tropical IR engine — tool semantics, in Lean

The engine is the whole stack: the session, the native runtime (FFI),
registration (raise + elaborate + strata + entry rendering), the compiler
(Core downcast + partition + plan assembly), the v2 ingest (load/merge), and
save/export. There is no compiler-service subprocess.

Every graph mutation ends in `syncCompile`: the mirror lowers, elaborates,
downcasts, partitions, and the plan hot-swaps into the Lean-owned runtime.
State mutations precede the compile — a failed compile leaves the mutated
graph in place; the previous kernel keeps playing and the error is recoverable.

The implementation is split by concern; this module re-exports the whole
surface so `import Tropical.Engine` sees it unchanged:

* `Engine.Core` — env, tool-arg access, lookup vocabulary
* `Engine.Compile` — mirror → elaborate → partition → hot-swap
* `Engine.Registry` — program registration
* `Engine.Crud` — instance/program lifecycle handlers
* `Engine.Wire` — the wiring tools
* `Engine.ProgramIO` — save / export / ingest
* `Engine.Audio` — audio lifecycle and param control
* `Engine.Front` — tool dispatcher and boot
-/
