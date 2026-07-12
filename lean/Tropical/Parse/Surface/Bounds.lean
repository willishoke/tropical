import Tropical.Parse.Nodes
import Tropical.Parse.BoundLower

/-!
# Bounds lowering (port of `compiler/parse/lower_bounds.ts`)

`in [lo, hi]` and the built-in port-type aliases (`signal`, `bipolar`,
`unipolar`, `phase`, `freq`) are parse-time notation, not IR. This pass
folds them into explicit `clamp`/`select` calls on input defaults and on
the matching output assigns, leaving no `bounds` anywhere (the Lean AST has
no field for it). Explicit `in [...]` overrides the alias bounds.

  [lo, hi]     → clamp(expr, lo, hi)
  [lo, null]   → select(expr > lo, expr, lo)   (max)
  [null, hi]   → select(expr < hi, expr, hi)   (min)
  [null, null] → expr

Each program lowers its own bounds (nested programs were already lowered
when parsed), so no tree recursion is needed here.
-/

namespace Tropical.Parse.Surface

open Tropical.Parse
  (ParsedExpr Program ProgramPorts ProgramPort ProgramPortSpec PortTypeDecl Block
   wrapWithBound)
open Lean (JsonNumber)

/-- An explicit `in [lo, hi]` bound; each side `none` for the `null` sentinel. -/
abbrev BoundPair := Option JsonNumber × Option JsonNumber

private def ji (n : Int) : JsonNumber := { mantissa := n, exponent := 0 }

def portName : ProgramPort → String
  | .bare n => n
  | .spec s => s.name

/-- Built-in port-type aliases with implicit bounds. -/
def builtinPortBounds : String → Option BoundPair
  | "signal"   => some (some (ji (-1)), some (ji 1))
  | "bipolar"  => some (some (ji (-1)), some (ji 1))
  | "unipolar" => some (some (ji 0), some (ji 1))
  | "phase"    => some (some (ji 0), some (ji 1))
  | "freq"     => some (some (ji 0), none)
  | _          => none

/-- Alias bounds from a port type (only a bare scalar name can be a builtin). -/
def aliasBounds : Option PortTypeDecl → Option BoundPair
  | some (.scalar name) => builtinPortBounds name
  | _                   => none

-- `wrapWithBound` (the idempotent bounds → clamp/select fold, shared with the
-- JSON raise path) comes from `Parse/BoundLower.lean`.

private def lookupBound (arr : Array (String × BoundPair)) (nm : String) : Option BoundPair :=
  (arr.find? (·.1 == nm)).map (·.2)

/-- Effective bounds for a port: explicit `in [...]` (from the parser sidecar)
    if present, else the alias bounds from its type. -/
private def effFor (explicit : Array (String × BoundPair)) (p : ProgramPort) : Option BoundPair :=
  match lookupBound explicit (portName p) with
  | some b => some b
  | none => match p with | .spec s => aliasBounds s.type? | .bare _ => none

/-- Fold explicit + alias bounds into `clamp`/`select` on input defaults and
    matching output assigns. `inputBounds`/`outputBounds` are the explicit
    `in [...]` sidecars collected by the port parser. -/
def lowerBounds (prog : Program) (inputBounds outputBounds : Array (String × BoundPair)) :
    Program :=
  match prog with
  | .mk name tp none body breaks => .mk name tp none body breaks
  | .mk name tp (some ports) body breaks =>
    let inputs' := ports.inputs.map fun ins => ins.map fun p =>
      match p with
      | .spec s =>
        match effFor inputBounds p, s.default? with
        | some b, some d => .spec { s with default? := some (wrapWithBound d b) }
        | _, _ => p
      | .bare _ => p
    let outBounds : Array (String × BoundPair) :=
      (ports.outputs.getD #[]).filterMap fun p => (effFor outputBounds p).map fun b => (portName p, b)
    let assigns' := body.assigns.map fun a =>
      match a with
      | .output nm e =>
        match (outBounds.find? (·.1 == nm)).map (·.2) with
        | some b => .output nm (wrapWithBound e b)
        | none => .output nm e
    .mk name tp (some { ports with inputs := inputs' }) (Block.mk body.decls assigns') breaks

end Tropical.Parse.Surface
