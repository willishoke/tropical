import Tropical.Engine

/-!
# EngineMirror — harness-only session recompiles (test support, not production)

The `compileMirror*` family: rebuild a `tropical_plan_5` from the engine's
current session mirror at an arbitrary compilation mode, WITHOUT loading it
or touching the runtime. The production path compiles AND loads through
`syncCompile`; these entry points exist so a harness can hold the compiled
artifact in hand — `diffcli compile`/`emit-ir` print it, the tropicaltest
golden and equivalence gates hash and cross-compare it (elaborate-path vs
direct-root, plan vs typed stage blocks). Consumed ONLY by `tropicaltest`
and the `diffcli` verbs — nothing on the production compile path imports
this module.

Everything here stays in `namespace Tropical.Engine` so call sites read
identically to the production entry points they mirror.
-/

namespace Tropical.Engine

/-- Harness-only (diffcli `compile`): rebuild the plan from the current
    mirror at an arbitrary compilation mode, WITHOUT loading it or
    touching the service. Mirrors the `compile_patch.ts` contract's
    final `compileSession(session, options)`: by load time the wires
    are lifted and every type is registered, so the mode only reaches
    the plan's `compilation_mode` field (the TS auto-flip for
    `microkernel-deep` lands after every registration — the recorded
    known limitation — so plan structure is mode-independent there
    too). Collapses into `syncCompile` at 6f when the sync payload
    retires. -/
def buildSessionInput (env : Env) (mode : Tropical.Plan.CompilationMode) :
    EngineM Tropical.Compile.SessionInput :=
  -- The shared prologue lives in `Engine.Compile.buildSessionInputVia`.
  buildSessionInputVia env "compileMirrorPlan" mode

def compileMirrorFlatPlan (env : Env) (mode : Tropical.Plan.CompilationMode) :
    EngineM Tropical.Plan.FlatPlan := do
  let input ← buildSessionInput env mode
  match Tropical.Compile.compileSession input with
  | .error msg => internalError msg
  | .ok p => pure p

/-- The stage-differential entry: the session input, the compiled plan,
    and the typed per-instruction stages in emit order. -/
def compileMirrorStaged (env : Env) (mode : Tropical.Plan.CompilationMode) :
    EngineM (Tropical.Compile.SessionInput × Tropical.Plan.FlatPlan
      × Array (Array (Option Tropical.Ir.Stage))) := do
  let input ← buildSessionInput env mode
  match Tropical.Compile.compileSessionStaged input with
  | .error msg => internalError msg
  | .ok (p, blocks) => pure (input, p, blocks)

def compileMirrorPlan (env : Env) (mode : Tropical.Plan.CompilationMode) :
    EngineM String := do
  let plan ← compileMirrorFlatPlan env mode
  match plan.toWire with
  | .error msg => internalError msg
  | .ok j => pure j.compress

end Tropical.Engine
