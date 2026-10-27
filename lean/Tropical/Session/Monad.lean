import Tropical.Errors
import Tropical.Session

/-!
# SessionM — session logic as a pure state transformer

The session's semantics (registration, ingest, lifting, the compile
prologue) are functions of `SessionSt` that can fail with an envelope.
`SessionM` is that and nothing more: no IO, no runtime. The engine runs a
`SessionM` against its `IO.Ref` (`Engine.Env.session`); a no-runtime
compiler runs it against a value.

State written before a failure survives it. That is the semantics the
engine always had, with each step writing the ref as it went: a failed
load or compile leaves the mutated graph in place, and the caller reports
the error against that graph.
-/

namespace Tropical.Session

abbrev SessionM := ExceptT Failure (StateM SessionSt)

/-- Run against a state: the outcome and the state after, failure included. -/
def SessionM.runOn {α : Type} (m : SessionM α) (st : SessionSt) :
    Except Failure α × SessionSt :=
  Id.run ((ExceptT.run m).run st)

end Tropical.Session
