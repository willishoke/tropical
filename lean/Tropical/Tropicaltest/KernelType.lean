import Tropical.Session.Compile
import Tropical.Kernel
import Tropical.Tropicaltest.Gates

/-!
# Typed-kernel gate (core tier)

`design/typed-kernel.md` slice T1: every patch in the repo's corpora derives
a `tropical_kernel_1` type, that type survives a wire round trip, and its
shape says what the document promises — `tau` first, one `param` input per
`param:` slot (and nothing else), one `out` of shape `[frames, channels]`.
A statement about the compiler alone: no runtime is linked.
-/

namespace Tropical.Tropicaltest.KernelType

open Tropical.Kernel

/-- The corpora the parity checker also walks. Paths are relative to the
    repo root, where every test binary runs. -/
def corpora : List String := ["patches", "web/patches"]

private def listPatches : IO (Array String) := do
  let mut out := #[]
  for dir in corpora do
    for entry in ← System.FilePath.readDir dir do
      let name := entry.fileName
      if name.endsWith ".json" && name != "manifest.json" then
        out := out.push s!"{dir}/{name}"
  return out.qsort (· < ·)

private def frames : Nat := 256

/-- Everything the type must say about a plan, as one failure string. -/
private def checkShape (plan : Tropical.Plan.FlatPlan) (k : KernelType) :
    Option String := Id.run do
  let some tau := k.inputs[0]? | return some "no inputs"
  if tau.name != tauName || tau.elem != .i64q32 || tau.shape != #[frames]
      || tau.role != some .signal then
    return some s!"tau port is {tau.describe}"
  let paramSlots := plan.slotNames.filter (·.startsWith paramPrefix)
  let paramPorts := k.inputs.extract 1 k.inputs.size
  if paramPorts.map (·.name) != paramSlots then
    return some s!"param inputs {paramPorts.map (·.name)} ≠ param slots {paramSlots}"
  for p in paramPorts do
    if p.elem != .f64 || p.shape != #[] || p.role != some .param then
      return some s!"param port is {p.describe}"
  for d in plan.paramDisciplines do
    match paramPorts.find? (·.name == paramPrefix ++ d.name) with
    | none => return some s!"discipline '{d.name}' names no param input"
    | some p =>
      if (p.discipline.map (·.discipline)) != some d.discipline then
        return some s!"param '{d.name}' carries discipline {(p.discipline.map (·.discipline)).getD "none"}, plan says {d.discipline}"
  let some out := k.outputs[0]? | return some "no outputs"
  if k.outputs.size != 1 || out.name != outName || out.elem != .f64
      || out.shape != #[frames, plan.outputChannelCount] then
    return some s!"out port is {out.describe}"
  if k.rate != plan.sampleRate then return some "rate differs from the plan's"
  none

def runKernelType : IO Bool := do
  let label := "typed-kernel"
  let registry ← match Tropical.Session.bootRegistry with
    | .error f => return ← failGate label s!"stdlib boot failed: {f.toJson.compress}"
    | .ok st => pure st
  let patches ← listPatches
  if patches.isEmpty then return ← failGate label "no patches found"
  let mut checked := 0
  for path in patches do
    let text ← IO.FS.readFile path
    let plan ← match (Tropical.Session.compilePatch text).runOn registry with
      | (.error f, _) => return ← failGate label s!"{path}: compile: {f.toJson.compress}"
      | (.ok (plan, _), _) => pure plan
    let k ← match ofPlan plan frames with
      | .error e => return ← failGate label s!"{path}: ofPlan: {e}"
      | .ok k => pure k
    if let some why := checkShape plan k then
      return ← failGate label s!"{path}: {why}"
    let wire := k.toWire.compress
    let back ← match Lean.Json.parse wire >>= KernelType.ofWire with
      | .error e => return ← failGate label s!"{path}: ofWire: {e}"
      | .ok back => pure back
    if back.toWire.compress != wire then
      return ← failGate label s!"{path}: wire form is not a fixed point of ofWire ∘ toWire"
    if let some why := checkShape plan back then
      return ← failGate label s!"{path}: after round trip: {why}"
    checked := checked + 1
  passGate label s!"{checked} patches: tau/params/out shape, {schema} round trip"

end Tropical.Tropicaltest.KernelType
