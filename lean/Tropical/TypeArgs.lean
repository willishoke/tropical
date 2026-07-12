import Lean.Data.Json
import Tropical.Expr

/-!
# Type-arg resolution

The boundary at which raw `type_args` (as supplied by MCP
`add_instance`/`replicate`) are resolved against a generic template's
declared type params to produce concrete integers the strata
specializer can consume, plus the `Type<N=8>` specialization cache
key.

The TS original also resolves `{op:'typeParam',name}` outer-frame
forwarding (patch-file instance decls inside generic programs); the
MCP path never has an outer frame, so that arm is reachable only as
its error ("no outer frame provides it") — ported byte-exact, like
every other message here (they surface verbatim in
`invalid_type_args` envelopes).

Raw-arg iteration order: TS walks `Object.keys` (insertion order);
this side walks the Lean `Json.obj` map (key-sorted). The two already
disagreed before this port — the engine relayed `type_args` to the
service as key-sorted compressed JSON — so sorted iteration *is* the
shipped behavior being preserved. Observable only when several
unknown keys race for the first error.
-/

namespace Tropical.TypeArgs

open Lean (Json JsonNumber)
open Tropical.Expr (opOf? getStrField?)

/-- JS `Number.isInteger` over the parsed double. -/
private def isIntegerValued (n : JsonNumber) : Bool :=
  let f := n.toFloat
  f == f.floor

/-- Port of `resolveValue`: numeric literals pass through;
    `{op:'typeParam',name}` has no outer frame on the MCP path, so it
    resolves to the TS unresolved-ref error. -/
private def resolveValue (v : Json) (context : String) :
    Except String JsonNumber :=
  match v with
  | .num n => .ok n
  | _ =>
    if opOf? v == some "typeParam" then
      let tpName := (getStrField? v "name").getD "undefined"
      .error s!"{context}: unresolved type_param '{tpName}' (no outer frame provides it)"
    else
      .error s!"{context}: type_arg value must be a number or \{ op: 'typeParam', name }, got {v.compress}"

/-- Port of `resolveTypeArgs` (outer frame fixed to `undefined`):
    validate raw args against the declared params — unknown names
    first, then per declared param (declaration order): explicit arg →
    integer check, else default, else missing-arg. The result is in
    declaration order and total over the declared params. -/
def resolve (raw? : Option Json)
    (typeParams : Array (String × Option JsonNumber))
    (contextName : String) :
    Except String (Array (String × JsonNumber)) := do
  let rawFields : Array (String × Json) := match raw? with
    | some (.obj m) => m.toList.toArray
    | _ => #[]
  for (key, _) in rawFields do
    unless typeParams.any (·.1 == key) do
      let declared := ", ".intercalate (typeParams.toList.map (·.1))
      throw s!"{contextName}: unknown type_arg '{key}'. Declared: {if declared.isEmpty then "(none)" else declared}"
  let mut resolved : Array (String × JsonNumber) := #[]
  for (name, default?) in typeParams do
    match rawFields.find? (·.1 == name) with
    | some (_, v) =>
      let n ← resolveValue v s!"{contextName}.type_args.{name}"
      unless isIntegerValued n do
        throw s!"{contextName}: type_arg '{name}' must be an integer, got {n}"
      resolved := resolved.push (name, n)
    | none =>
      match default? with
      | some d => resolved := resolved.push (name, d)
      | none =>
        throw s!"{contextName}: missing required type_arg '{name}' (no default)"
  return resolved

/-- Port of `specializationCacheKey`: sorted `k=v` pairs in angle
    brackets (`Delay<N=8>`). -/
def cacheKey (typeName : String) (args : Array (String × JsonNumber)) : String :=
  let sorted := (args.qsort (fun a b => a.1 < b.1)).toList.map fun (k, v) => s!"{k}={v}"
  s!"{typeName}<{",".intercalate sorted}>"

end Tropical.TypeArgs
