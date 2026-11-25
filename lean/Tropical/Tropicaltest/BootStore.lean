import Tropical.Session.Registry
import Tropical.Tropicaltest.Gates

/-!
# Boot-store gate (core tier)

The stdlib boot adopts each post-strata program into the session store where
strata left it. Registration used to copy it through the `tropical_resolved_1`
JSON codec instead; the copy changed nothing, but the encoder's preconditions
(a child-descending expression arena and program pool) were checked on every
start. They are statements about construction, so they are checked here, once,
along with the store's indexing: every catalog name resolves to a program of
that name that passes the resolved-shape check.
-/

namespace Tropical.Tropicaltest.BootStore

def runBootStore : IO Bool := do
  let label := "boot-store"
  let st ← match Tropical.Session.bootRegistry with
    | .error f => return ← failGate label s!"stdlib boot failed: {f.toJson.compress}"
    | .ok st => pure st
  if !st.arena.exprs.wf then
    return ← failGate label "expression arena is not child-descending"
  if !Tropical.Ir.progPoolWf st.arena.programs then
    return ← failGate label "program pool is not child-descending"
  if st.catalogOrder.size != Tropical.EmitArrow.stdlibBuilders.size then
    return ← failGate label
      s!"catalog has {st.catalogOrder.size} programs, the stdlib {Tropical.EmitArrow.stdlibBuilders.size}"
  for name in st.catalogOrder do
    let some idx := st.templateByName.get? name
      | return ← failGate label s!"{name}: not in templateByName"
    if st.resolvedByName.get? name != some idx then
      return ← failGate label s!"{name}: resolvedByName disagrees with templateByName"
    let some p := st.arena.program? idx
      | return ← failGate label s!"{name}: index {idx.idx} out of range"
    if p.name != name then
      return ← failGate label s!"{name}: stored program is named '{p.name}'"
    if let .error e := Tropical.Ir.checkResolvedArena st.arena idx then
      return ← failGate label s!"{name}: {firstLine e}"
  passGate label
    s!"{st.catalogOrder.size} programs adopted in place ({st.arena.programs.size} in the pool)"

end Tropical.Tropicaltest.BootStore
