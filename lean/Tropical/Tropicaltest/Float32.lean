import Tropical.Tropicaltest.Int32
import Tropical.Kernel
import Tropical.StagedLoad
import Tropical.Ir.EmitLlvm
import Tropical.Ir.EmitCommon
import Tropical.Playground

/-!
# The binary32 value lane (`--float f32`)

A binary32 artifact realizes the `float` kind as IEEE binary32 on the LLVM
path (`Kernel.FloatFormat`, `EmitLlvm`'s "The value lane's format"). The plan
is the same; the emitter, the kernel type and both hosts change. The gates:

* **literals** — `binary32BitsOfDecimal?` rounds a decimal once, ties to
  even, subnormals included, and refuses one past the binary32 range; a
  decimal just above a binary32 tie rounds up even though its binary64
  rounding is the tie (one rounding, not two).
* **rounding-fixture** — `tests/fixtures/rounding_f32.ll` is the text of
  `EmitLlvm.roundingF32Fixture`, the inline floor/ceil/round sequences that
  `current_rounding_f32` (ctest) runs on all 2³² binary32 inputs.
* **jit** — a two-oscillator ring modulator and a float param through
  `SoftClip`, rendered by the JIT at binary32: every sample is a binary32,
  the render stays within 2⁻¹⁶ of the binary64 render, the kernel type
  declares `f32` ports and a binary32 default, a `set_param` write equals a
  fresh compile at the written value bit for bit, and `renderTau` equals
  `process`.
* **refusals** — a non-raw discipline at binary32 (`Kernel.ofPlan`), a
  literal past the binary32 range (`EmitLlvm`), an unknown float format in
  a manifest (`Kernel.Realization.ofWire` and the native host), and a float
  port declared at another format than the artifact's (the native host).
-/

namespace Tropical.Tropicaltest.Float32

open Tropical
open Tropical.Plan

private def compile (text : String) : Except String (FlatPlan × Array (Array (Option Ir.Stage))) := do
  let st ← Session.bootRegistry.mapError (·.toJson.compress)
  match (Session.compilePatch text).runOn st with
  | (.ok r, _) => .ok r
  | (.error f, _) => .error f.toJson.compress

private def render (plan : FlatPlan) (float : Kernel.FloatFormat) (start n : Nat)
    (writes : List (String × Float) := []) : IO (Array Float) := do
  let rt ← Ffi.Runtime.new (UInt32.ofNat n)
  StagedLoad.load rt plan float
  for (name, v) in writes do
    match ← rt.setParam name v with
    | some _ => pure ()
    | none => throw <| IO.userError s!"set_param {name}: unknown"
  rt.setSampleIndex start.toUInt64
  rt.process
  pure (decodeF64LE (← rt.outputBytes))

private def firstMismatch (got want : Array Float) : Option Nat := Id.run do
  if got.size != want.size then return some (min got.size want.size)
  for k in [0:got.size] do
    if got[k]!.toBits != want[k]!.toBits then return some k
  return none

private def isBinary32 (x : Float) : Bool := x.toFloat32.toFloat.toBits == x.toBits

private def patch (decls : String) (out : String) : String :=
  "{\"schema\":\"tropical_program_2\",\"name\":\"f32\",\"body\":{\"op\":\"block\",\"decls\":[" ++
    decls ++ "],\"assigns\":[{\"op\":\"outputAssign\",\"name\":\"dac.out\",\"expr\":" ++ out ++ "}]}}"

/-- Two `FixedSinOsc`s through a `VCA`: the DDS phase, the sine and a product. -/
def ringMod : String :=
  patch ("{\"op\":\"instanceDecl\",\"name\":\"a\",\"program\":\"FixedSinOsc\",\"inputs\":{\"freq\":220}}," ++
    "{\"op\":\"instanceDecl\",\"name\":\"b\",\"program\":\"FixedSinOsc\",\"inputs\":{\"freq\":331}}," ++
    "{\"op\":\"instanceDecl\",\"name\":\"m\",\"program\":\"VCA\",\"inputs\":{\"audio\":{\"op\":\"ref\",\"instance\":\"a\",\"output\":\"sine\"},\"cv\":{\"op\":\"ref\",\"instance\":\"b\",\"output\":\"sine\"}}}")
    "{\"op\":\"ref\",\"instance\":\"m\",\"output\":\"out\"}"

/-- A float param `drive` into `SoftClip` after a 440 Hz `FixedSinOsc`. -/
def drivePatch (drive : String) : String :=
  patch ("{\"op\":\"paramDecl\",\"name\":\"drive\",\"value\":" ++ drive ++ "}," ++
    "{\"op\":\"instanceDecl\",\"name\":\"o\",\"program\":\"FixedSinOsc\",\"inputs\":{\"freq\":440}}," ++
    "{\"op\":\"instanceDecl\",\"name\":\"s\",\"program\":\"SoftClip\",\"inputs\":{\"input\":{\"op\":\"ref\",\"instance\":\"o\",\"output\":\"sine\"},\"drive\":{\"op\":\"param\",\"name\":\"drive\"}}}")
    "{\"op\":\"ref\",\"instance\":\"s\",\"output\":\"out\"}"

def runLiterals : IO Bool := do
  let cases : List (String × Lean.JsonNumber × Option UInt32) := [
    ("0.1", ⟨1, 1⟩, some 0x3DCCCCCD),
    ("0.9", ⟨9, 1⟩, some 0x3F666666),
    ("0", ⟨0, 0⟩, some 0),
    ("-2.5", ⟨-25, 1⟩, some 0xC0200000),
    ("16777217 (a tie, to even)", ⟨16777217, 0⟩, some 0x4B800000),
    ("16777219 (a tie, to even, up)", ⟨16777219, 0⟩, some 0x4B800002),
    -- This decimal lies about 2⁻⁶⁰ above the binary32 tie 1 + 2⁻²⁴; its
    -- binary64 rounding is the tie itself, which would round to even (1.0).
    -- One rounding gives 1 + 2⁻²³.
    ("1.000000059604644776257921720677, just above the tie 1 + 2^-24",
      ⟨1000000059604644776257921720677, 30⟩, some 0x3F800001),
    ("largest finite", ⟨340282346638528859811704183484516925440, 0⟩, some 0x7F7FFFFF),
    ("past the largest finite", ⟨340282356779733661637539395458142568448, 0⟩, none),
    ("smallest subnormal 2^-149", ⟨1401298464324817070923729583289916131280, 84⟩, some 0x00000001),
    ("just below half the smallest subnormal", ⟨7006492321624085354618647916449580656401, 85⟩, some 0),
    ("1e-46 underflows to 0", ⟨1, 46⟩, some 0)]
  let mut bad : Array String := #[]
  for (label, n, want) in cases do
    let got := Ir.EmitCommon.binary32BitsOfDecimal? n
    if got != want then
      bad := bad.push s!"{label}: got {got.map (·.toNat)} want {want.map (·.toNat)}"
  -- the manifest's decimal of a binary32 is exact: it parses back to the bits
  for bits in [0x3DCCCCCD, 0x00000001, 0x7F7FFFFF, 0x80800000, 0x3F800001] do
    let d := Kernel.binary32Decimal (UInt32.ofNat bits)
    if Ir.EmitCommon.binary32BitsOfDecimal? d != some (UInt32.ofNat bits) then
      bad := bad.push s!"binary32Decimal {bits} = {d} does not read back"
  if bad.isEmpty then
    passGate "f32-literals" s!"{cases.length} decimals rounded once to binary32 (ties to even, subnormals, the range refusal, a decimal above a tie that binary64 rounds onto it); binary32Decimal reads back exactly"
  else failGate "f32-literals" (String.intercalate "; " bad.toList)

def fixturePath : String := "tests/fixtures/rounding_f32.ll"

def runRoundingFixture (writeMode : Bool) : IO Bool := do
  match Ir.EmitLlvm.roundingF32Fixture with
  | .error e => failGate "f32-rounding-fixture" s!"emit: {e}"
  | .ok text =>
    if writeMode then
      IO.FS.writeFile fixturePath text
      passGate "f32-rounding-fixture" s!"wrote {fixturePath}"
    else
      let have_ ← try some <$> IO.FS.readFile fixturePath catch _ => pure none
      match have_ with
      | none => failGate "f32-rounding-fixture" s!"{fixturePath} is missing (run with --write)"
      | some t =>
        if t == text then
          passGate "f32-rounding-fixture" s!"{fixturePath} is the emitter's inline floor/ceil/round (the ctest current_rounding_f32 runs it on all 2^32 inputs)"
        else failGate "f32-rounding-fixture" s!"{fixturePath} differs from EmitLlvm.roundingF32Fixture (regenerate with --write, then rerun current_rounding_f32)"

def runJit : IO Bool := do
  let mut bad : Array String := #[]
  let n := 4096
  match compile ringMod, compile (drivePatch "3"), compile (drivePatch "0.1") with
  | .error e, _, _ | _, .error e, _ | _, _, .error e => failGate "f32-jit" s!"compile: {e}"
  | .ok (ring, _), .ok (drive3, _), .ok (drive01, _) =>
    -- the ring modulator at binary32 against binary64
    let r64 ← render ring .f64 0 n
    let r32 ← render ring .f32 0 n
    let mut maxErr : Float := 0
    for k in [0:n] do
      let a := r32[k]!
      let b := r64[k]!
      if !isBinary32 a then bad := bad.push s!"ring frame {k}: {a} is not a binary32"; break
      maxErr := max maxErr (a - b).abs
    if !(maxErr < 1.52587890625e-5) then bad := bad.push s!"ring: max |f32 - f64| = {maxErr} ≥ 2^-16"
    if !(r32.any (·.abs > 0.5)) then bad := bad.push "ring: f32 render is near silent"
    -- the kernel type: f32 ports, a binary32 default, the realization
    match Kernel.ofPlan drive01 16 .f32 with
    | .error e => bad := bad.push s!"manifest: {e}"
    | .ok k =>
      match k.inputs.find? (·.name == "param:drive") with
      | some p =>
        unless p.elem == .f32 && p.default.map (·.toFloat) == some (0.1 : Float).toFloat32.toFloat do
          bad := bad.push s!"param:drive is {p.elem.wire} default {(p.default.map (·.toString)).getD "-"}, wanted f32 at binary32(0.1)"
      | none => bad := bad.push "manifest: no param:drive"
      if (k.outputs[0]?.map (·.elem)) != some .f32 then bad := bad.push "manifest: output is not f32"
      if k.realization.float != .f32 then bad := bad.push "manifest: realization.float is not f32"
      match k.toWire with
      | .error e => bad := bad.push s!"manifest wire: {e}"
      | .ok j =>
        match Kernel.KernelType.ofWire j with
        | .error e => bad := bad.push s!"manifest decode: {e}"
        | .ok k' =>
          if k'.realization.float != .f32 then bad := bad.push "manifest: the wire round trip lost float"
    -- set_param on an f32 cell ≡ a fresh compile at that value
    let written ← render drive3 .f32 1000 512 [("drive", 0.1)]
    let fresh ← render drive01 .f32 1000 512
    if let some k := firstMismatch written fresh then
      bad := bad.push s!"set_param 0.1: frame {k}: {written[k]?} ≠ fresh compile {fresh[k]?}"
    -- the observation path renders the same binary32 output
    let rt ← Ffi.Runtime.new 512
    StagedLoad.load rt drive01 .f32
    let observed := decodeF64LE (← rt.renderTau
      ((Array.range 512).map fun k => (UInt64.ofNat (1000 + k)) <<< 16) #[0])
    if let some k := firstMismatch observed fresh then
      bad := bad.push s!"renderTau: frame {k}: {observed[k]?} ≠ process {fresh[k]?}"
    if bad.isEmpty then
      passGate "f32-jit" s!"ring modulator at binary32: every sample a binary32, max |f32 - f64| {maxErr * 1e6}e-6 over {n} frames; manifest f32 ports, binary32(0.1) default, realization round trip; set_param ≡ fresh compile; renderTau ≡ process"
    else failGate "f32-jit" (String.intercalate "; " bad.toList)

private def expectErr {α : Type} (label : String) (r : Except String α) (needle : String) : Option String :=
  match r with
  | .ok _ => some s!"{label}: accepted"
  | .error e => if (e.splitOn needle).length > 1 then none else some s!"{label}: '{e}' lacks '{needle}'"

def runRefusals : IO Bool := do
  let mut bad : Array String := #[]
  -- a discipline at binary32: the playground master clock's velocity
  match ← (try some <$> IO.FS.readFile "web/patches/tz-saw.json" catch _ => pure none) with
  | none => bad := bad.push "tz-saw: web/patches/tz-saw.json is unreadable"
  | some text =>
    match Lean.Json.parse text with
    | .error e => bad := bad.push s!"tz-saw parse: {e}"
    | .ok j =>
      match Playground.compilePlan j Plan.defaultSampleRate with
      | .error e => bad := bad.push s!"tz-saw compile: {e}"
      | .ok compiled =>
        let plan := compiled.plan
        if let some m := expectErr "velocity at f32" (Kernel.ofPlan plan 16 .f32)
            "a binary32 artifact's params are written raw" then bad := bad.push m
        if let .error e := Kernel.ofPlan plan 16 .f64 then bad := bad.push s!"tz-saw at f64: {e}"
  -- a literal past binary32
  match compile (patch "{\"op\":\"instanceDecl\",\"name\":\"v\",\"program\":\"VCA\",\"inputs\":{\"audio\":1,\"cv\":4e38}}"
      "{\"op\":\"ref\",\"instance\":\"v\",\"output\":\"out\"}") with
  | .error e => bad := bad.push s!"4e38 compile: {e}"
  | .ok (plan, _) =>
    if let some m := expectErr "4e38 at f32" (Ir.EmitLlvm.emitKernel ({ audio := plan, coeff? := none } : Ir.Stage0.Split) 16 .f32)
        "outside the binary32 range" then bad := bad.push m
  -- an unknown float format, and a port at another format, in a manifest
  match compile (drivePatch "3") with
  | .error e => bad := bad.push s!"drive compile: {e}"
  | .ok (plan, _) =>
    match Kernel.ofPlan plan 16 .f32, Ir.EmitLlvm.emitKernel ({ audio := plan, coeff? := none } : Ir.Stage0.Split) 16 .f32 with
    | .ok k, .ok ir =>
      match k.toWire with
      | .error e => bad := bad.push s!"wire: {e}"
      | .ok j =>
        let realization := (j.getObjValD "realization").setObjVal! "float" "f16"
        let unknown := j.setObjVal! "realization" realization
        if let some m := expectErr "Realization.ofWire f16" (Kernel.KernelType.ofWire unknown)
            "unknown float format 'f16'" then bad := bad.push m
        let hostRefuses (label : String) (manifest : Lean.Json) (needle : String) : IO (Option String) := do
          let rt ← Ffi.Runtime.new 16
          try
            rt.load ir "" "" manifest.compress
            pure (some s!"{label}: the native host loaded it")
          catch e =>
            let msg := toString e
            pure (if (msg.splitOn needle).length > 1 then none else some s!"{label}: '{msg}' lacks '{needle}'")
        if let some m ← hostRefuses "native host, float f16" unknown
            "unknown float format 'f16'" then bad := bad.push m
        let f64Out := match j.getObjVal? "outputs" with
          | .ok (.arr outs) => j.setObjVal! "outputs"
              (Lean.Json.arr (outs.map fun o => o.setObjVal! "elem" "f64"))
          | _ => j
        if let some m ← hostRefuses "native host, an f64 output in an f32 artifact" f64Out
            "whose float format is f32" then bad := bad.push m
    | .error e, _ | _, .error e => bad := bad.push s!"drive at f32: {e}"
  if bad.isEmpty then
    passGate "f32-refusals" "a velocity discipline at binary32, a literal past binary32, an unknown float format (Kernel decode and the native host), an f64 port in an f32 artifact (the native host) — each refused by name"
  else failGate "f32-refusals" (String.intercalate "; " bad.toList)

end Tropical.Tropicaltest.Float32
