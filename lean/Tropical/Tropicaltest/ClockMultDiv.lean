import Tropical.Tropicaltest.TypedPorts
import Tropical.Testing.ClockMultDivLaws

/-!
# ClockMultDivQ15 against its big-integer model

`ClockMultDivQ15` (`EmitArrow/Stdlib.lean`) is rendered by the JIT and
compared, word for word, to `Testing.ClockMultDivLaws` — a model written over
`Int`/`Nat` with no IR in it. Every case is a live param write on one loaded
runtime (`set_param` of `freq`, `nᵢ`, `dᵢ`, `offsetᵢ`), so "a write gives the
phase the model predicts" is what each case checks. The gates:

* **phase** — the `phase` ports (the top word of `m·Ψ + o` on ℤ/2⁶⁴) equal
  the model's at random clocks: whole and fractional samples, negative, and
  out to ±2⁶²; random master frequencies over the whole Q16.16 word
  (negative ones run backward), knob words outside `[0, 15]` (read as their low nibble), and
  random offsets. The audio artifact (`process`) agrees with the observation
  artifact (`renderTau`), and a written runtime with a fresh compile at the
  written values.
* **gate/sine** — the gate is 1.0 (Q17.15) exactly in the first half cycle,
  0 in the second, 0 at `d = 0`; the sine is 0 at `d = 0` and otherwise
  within 3.2 LSB of `32768·sin(2π·phase)` (the documented bound of
  `fixedSinQ15Sig`).
* **recurrence** — at a master frequency whose increment `I` has 2-adic
  valuation `v ≥ 18`, Ψ wraps after exactly `2^(64−v)` samples; the kernel
  repeats every output word there, and every phase is at its offset at
  `τ = 0` and at `τ = 2^(64−v)` samples (Ψ ≡ 0).
-/

namespace Tropical.Tropicaltest.ClockMultDiv

open Tropical
open Tropical.Plan
open Tropical.Testing.ClockMultDivLaws

private def label := "clock-mult-div"

private def j (n : Nat) : Lean.Json := Lean.toJson n

/-- An exact decimal `word · 2^-fracBits`. -/
private def exactJson (word : Int) (fracBits : Nat) : Lean.Json :=
  -- `JsonNumber` prints a zero mantissa at a negative exponent as `0.e-22`
  if word == 0 then Lean.toJson (0 : Nat)
  else Lean.Json.num ⟨word * (5 : Int) ^ fracBits, fracBits⟩

structure Knobs where
  freq : Int
  n : Array Int
  d : Array Int
  offset : Array Int
deriving Inhabited

def defaultKnobs : Knobs :=
  { freq := 65536, n := #[1, 1, 1, 1], d := #[1, 1, 1, 1], offset := #[0, 0, 0, 0] }

private def waves : List Nat := [1, 2, 3, 4]

/-- The params of a patch, with `knobs` as their declared values. -/
private def paramDecls (k : Knobs) : List Lean.Json :=
  let decl (name : String) (word : Int) (fracBits : Nat) : Lean.Json :=
    Lean.Json.mkObj [("op", "paramDecl"), ("name", Lean.Json.str name),
      ("type", "int32"), ("frac_bits", j fracBits), ("value", exactJson word fracBits)]
  [decl "freq" k.freq 16] ++
    waves.map (fun i => decl s!"n{i}" k.n[i - 1]! 0) ++
    waves.map (fun i => decl s!"d{i}" k.d[i - 1]! 0) ++
    waves.map (fun i => decl s!"offset{i}" k.offset[i - 1]! 32)

/-- One `ClockMultDivQ15` with every knob a param, its `outs` on the DAC. -/
def patch (k : Knobs) (outs : List String) : String :=
  let param (name : String) := Lean.Json.mkObj [("op", "param"), ("name", Lean.Json.str name)]
  let names := ["freq"] ++ waves.map (s!"n{·}") ++ waves.map (s!"d{·}") ++
    waves.map (s!"offset{·}")
  let inst := Lean.Json.mkObj [("op", "instanceDecl"), ("name", "c"),
    ("program", "ClockMultDivQ15"),
    ("inputs", Lean.Json.mkObj (names.map fun n => (n, param n)))]
  let assigns := (outs.zip (List.range outs.length)).map fun (o, ch) =>
    Lean.Json.mkObj [("op", "outputAssign"), ("name", "dac.out"),
      ("expr", Lean.Json.mkObj [("op", "ref"), ("instance", "c"), ("output", Lean.Json.str o)]),
      ("channel", j ch)]
  (Lean.Json.mkObj [("schema", "tropical_program_2"), ("name", "clock_mult_div"),
    ("body", Lean.Json.mkObj [("op", "block"),
      ("decls", Lean.Json.arr (paramDecls k ++ [inst]).toArray),
      ("assigns", Lean.Json.arr assigns.toArray)])]).compress

def phaseOuts : List String := waves.map (s!"phase{·}")
def gateSineOuts : List String := waves.map (s!"gate{·}") ++ waves.map (s!"sine{·}")

private def compile (text : String) :
    Except String (FlatPlan × Array (Array (Option Ir.Stage))) := do
  let st ← Session.bootRegistry.mapError (·.toJson.compress)
  match (Session.compilePatch text).runOn st with
  | (.ok r, _) => .ok r
  | (.error f, _) => .error f.toJson.compress

private def load (plan : FlatPlan) (blocks : Array (Array (Option Ir.Stage))) :
    IO Ffi.Runtime := do
  let rt ← Ffi.Runtime.new 256
  StagedLoad.loadTyped rt plan blocks
  pure rt

private def write (rt : Ffi.Runtime) (k : Knobs) : IO Unit := do
  let set (name : String) (v : Float) : IO Unit := do
    match ← rt.setParam name v with
    | some _ => pure ()
    | none => throw <| IO.userError s!"set_param {name}: unknown"
  set "freq" (Float.ofInt k.freq / 65536.0)
  for i in waves do
    set s!"n{i}" (Float.ofInt k.n[i - 1]!)
    set s!"d{i}" (Float.ofInt k.d[i - 1]!)
    set s!"offset{i}" (Float.ofInt k.offset[i - 1]! / 4294967296.0)

/-- A τ as the i64 bit pattern the port reads. -/
private def tauBits (t : Int) : UInt64 := UInt64.ofInt t

/-- The kernel's words at clocks `taus`: `result[c][k]` is channel `c` at
    `taus[k]`, `word = value · 2^fracBits` (exact: an `int32` port widens
    to f64 without loss). -/
def renderWords (rt : Ffi.Runtime) (taus : Array Int) (channels fracBits : Nat) :
    IO (Array (Array Int)) := do
  let xs := decodeF64LE (← rt.renderTau (taus.map tauBits)
    ((Array.range channels).map UInt32.ofNat))
  let scale : Float := (2.0 : Float) ^ fracBits.toFloat
  pure <| (Array.range channels).map fun c =>
    (Array.range taus.size).map fun k => (xs[c * taus.size + k]! * scale).toInt64.toInt

/-- A small deterministic generator. -/
structure Rng where
  s : UInt64

def Rng.next (g : Rng) : UInt64 × Rng :=
  let s := g.s * 6364136223846793005 + 1442695040888963407
  ((s >>> 11) ^^^ (s <<< 7), ⟨s⟩)

/-- A signed value in `[lo, hi]`. -/
def Rng.range (g : Rng) (lo hi : Int) : Int × Rng :=
  let (x, g) := g.next
  (lo + (x.toNat % (hi - lo + 1).toNat : Nat), g)

private def int32Range : Int × Int := (-2147483648, 2147483647)

def randomKnobs (g : Rng) : Knobs × Rng := Id.run do
  let mut g := g
  -- master: the whole Q16.16 word, or an audio/LFO range most of the time
  let (pick, g1) := g.range 0 3; g := g1
  let (freq, g2) := if pick == 0 then g.range int32Range.1 int32Range.2
    else g.range (-65536 * 20000) (65536 * 20000)
  g := g2
  let mut n := #[]; let mut d := #[]; let mut o := #[]
  for _ in [0:4] do
    let (a, g3) := g.range (-2) 17; g := g3
    let (b, g4) := g.range (-1) 16; g := g4
    let (c, g5) := g.range int32Range.1 int32Range.2; g := g5
    n := n.push a; d := d.push b; o := o.push c
  return ({ freq, n, d, offset := o }, g)

def randomTaus (g : Rng) (count : Nat) : Array Int × Rng := Id.run do
  let mut g := g
  let mut out : Array Int := #[0, 65536, 32768, -65536, 1, (2 : Int) ^ 62, -((2 : Int) ^ 62)]
  for k in [0:count] do
    let (bits, g1) := g.range 1 62; g := g1
    let (t, g2) := g.range (-((2 : Int) ^ bits.toNat)) ((2 : Int) ^ bits.toNat); g := g2
    -- every fourth a whole sample, the rest fractional
    out := out.push (if k % 4 == 0 then t / 65536 * 65536 else t)
  return (out, g)

private def modelPhase (rate : Nat) (k : Knobs) (i : Nat) (tau : Int) : Nat :=
  phase rate k.freq tau (knob k.n[i]!) (knob k.d[i]!) k.offset[i]!

/-- Compare the phase ports at `taus` to the model; the first mismatch. -/
private def checkPhases (rt : Ffi.Runtime) (rate : Nat) (k : Knobs) (taus : Array Int) :
    IO (Option String) := do
  let words ← renderWords rt taus 4 32
  for i in [0:4] do
    for t in [0:taus.size] do
      let want := topWord (modelPhase rate k i taus[t]!)
      let got := words[i]![t]!
      if got != want then
        return some s!"phase{i + 1} at τ={taus[t]!} (freq={k.freq}, n={k.n[i]!}, d={k.d[i]!}, o={k.offset[i]!}): kernel {got} ≠ model {want}"
  return none

private def rateOf (plan : FlatPlan) : Except String Nat :=
  if plan.sampleRate.exponent == 0 && plan.sampleRate.mantissa > 0 then
    .ok plan.sampleRate.mantissa.toNat
  else .error s!"sample rate {plan.sampleRate} is not a whole number"

/-- The 2-adic valuation of a nonzero integer. -/
private def v2 (x : Int) : Nat := Id.run do
  let mut x := x.natAbs
  let mut v := 0
  while x != 0 && x % 2 == 0 do
    x := x / 2
    v := v + 1
  return v

def runPhase : IO Bool := do
  let mut bad : Array String := #[]
  let cases := 300
  match compile (patch defaultKnobs phaseOuts) with
  | .error e => return ← failGate s!"{label}-phase" s!"compile: {e}"
  | .ok (plan, blocks) =>
    let rate ← IO.ofExcept (rateOf plan)
    let rt ← load plan blocks
    let mut g : Rng := ⟨0x5eed⟩
    let mut checked := 0
    for _ in [0:cases] do
      let (k, g1) := randomKnobs g
      let (taus, g2) := randomTaus g1 48
      g := g2
      write rt k
      if let some e ← checkPhases rt rate k taus then
        bad := bad.push e
        break
      checked := checked + taus.size * 4
    -- the audio artifact agrees with the observation artifact
    let (k, _) := randomKnobs ⟨77⟩
    write rt k
    let start : Nat := 123456789
    rt.setSampleIndex start.toUInt64
    rt.process
    let frames := decodeF64LE (← rt.outputInterleavedBytes)
    let taus := (Array.range 256).map fun f => ((start + f : Nat) : Int) * 65536
    let observed ← renderWords rt taus 4 32
    for f in [0:256] do
      for c in [0:4] do
        let w := (frames[f * 4 + c]! * 4294967296.0).toInt64.toInt
        if w != observed[c]![f]! then
          bad := bad.push s!"process frame {f} ch {c}: {w} ≠ renderTau {observed[c]![f]!}"
    if let some e ← checkPhases rt rate k taus then bad := bad.push s!"process window: {e}"
    -- a written runtime ≡ a fresh compile at the written values
    match compile (patch k phaseOuts) with
    | .error e => bad := bad.push s!"fresh compile: {e}"
    | .ok (p2, b2) =>
      let rt2 ← load p2 b2
      let (ts, _) := randomTaus ⟨99⟩ 64
      let a ← renderWords rt ts 4 32
      let b ← renderWords rt2 ts 4 32
      if a != b then bad := bad.push "set_param ≠ fresh compile at the written values"
    if bad.isEmpty then
      passGate s!"{label}-phase" s!"{checked} phase words ≡ the big-integer model over {cases} live writes (freq over the whole Q16.16 word, knobs outside [0,15], τ to ±2^62, fractional and negative); process ≡ renderTau; set_param ≡ fresh compile (rate {rate})"
    else failGate s!"{label}-phase" (String.intercalate "; " bad.toList)

def runGateSine : IO Bool := do
  let mut bad : Array String := #[]
  match compile (patch defaultKnobs gateSineOuts) with
  | .error e => return ← failGate s!"{label}-gate-sine" s!"compile: {e}"
  | .ok (plan, blocks) =>
    let rate ← IO.ofExcept (rateOf plan)
    let rt ← load plan blocks
    let mut g : Rng := ⟨0xc10c⟩
    let mut worst : Float := 0
    let mut highs := 0
    let mut held := 0
    for _ in [0:120] do
      let (k, g1) := randomKnobs g
      let (taus, g2) := randomTaus g1 48
      g := g2
      write rt k
      let words ← renderWords rt taus 8 15
      for i in [0:4] do
        let d := knob k.d[i]!
        for t in [0:taus.size] do
          let p := modelPhase rate k i taus[t]!
          let gate := words[i]![t]!
          let sine := words[4 + i]![t]!
          if gate != gateWord d p then
            bad := bad.push s!"gate{i + 1} at τ={taus[t]!}: {gate} ≠ model {gateWord d p}"
          if gate == 32768 then highs := highs + 1
          if d == 0 then
            held := held + 1
            if sine != 0 then bad := bad.push s!"sine{i + 1} at d=0: {sine} ≠ 0"
          else
            let x := Float.ofInt (topWord p) / 4294967296.0
            let want := 32768.0 * Float.sin (2.0 * 3.141592653589793 * x)
            let err := (Float.ofInt sine - want).abs
            if err > worst then worst := err
            if err > 3.2 then
              bad := bad.push s!"sine{i + 1} at τ={taus[t]!}: {sine} vs {want}"
      if bad.size > 4 then break
    if highs == 0 || held == 0 then bad := bad.push s!"coverage: {highs} high gates, {held} held"
    if bad.isEmpty then
      passGate s!"{label}-gate-sine" s!"gates ≡ the model's first half-cycle (and 0 at d=0, {held} held words); sines 0 at d=0 and within {worst} LSB of 32768·sin(2π·phase) elsewhere"
    else failGate s!"{label}-gate-sine" (String.intercalate "; " (bad.toList.take 5))

def runRecurrence : IO Bool := do
  let mut bad : Array String := #[]
  match compile (patch defaultKnobs phaseOuts) with
  | .error e => return ← failGate s!"{label}-recurrence" s!"compile: {e}"
  | .ok (plan, blocks) =>
    let rate ← IO.ofExcept (rateOf plan)
    -- a master frequency whose increment has 2-adic valuation ≥ 18
    let mut f : Int := 65536
    while v2 (increment rate f) < 18 do f := f + 1
    let inc := increment rate f
    let v := v2 inc
    -- Ψ advances by (inc/2^v)·2⁶⁴ ≡ 0 every 2^(64−v) samples
    let delta : Int := (2 : Int) ^ (80 - v)
    let rt ← load plan blocks
    let mut g : Rng := ⟨0xbeef⟩
    for _ in [0:20] do
      let (k0, g1) := randomKnobs g
      g := g1
      let k := { k0 with freq := f }
      write rt k
      let (taus, g2) := randomTaus g 24
      g := g2
      let base := taus.filter fun t => t.natAbs < (2 : Int) ^ 61
      let a ← renderWords rt base 4 32
      let b ← renderWords rt (base.map (· + delta)) 4 32
      if a != b then bad := bad.push s!"freq {f}: the words at τ and τ + 2^{80 - v} differ"
      -- Ψ ≡ 0 at τ = 0 and at τ = Δ: every output is at its offset
      let z ← renderWords rt #[0, delta] 4 32
      for i in [0:4] do
        let want := topWord (Int.emod (k.offset[i]! * (2 : Int) ^ 32) ringSize).toNat
        if z[i]![0]! != want || z[i]![1]! != want then
          bad := bad.push s!"phase{i + 1} at Ψ ≡ 0: {z[i]![0]!}, {z[i]![1]!} ≠ offset {want}"
      if position rate f delta != 0 then bad := bad.push "model: Ψ(Δ) ≠ 0"
    if bad.isEmpty then
      passGate s!"{label}-recurrence" s!"freq word {f} (I = {inc}, v₂ = {v}): every phase word repeats after 2^{64 - v} samples, and every output sits at its offset at τ = 0 and τ = 2^{64 - v} samples (Ψ ≡ 0), 20 knob settings"
    else failGate s!"{label}-recurrence" (String.intercalate "; " (bad.toList.take 5))

def runAll : IO Bool := do
  let a ← runPhase
  let b ← runGateSine
  let c ← runRecurrence
  pure (a && b && c)

end Tropical.Tropicaltest.ClockMultDiv
