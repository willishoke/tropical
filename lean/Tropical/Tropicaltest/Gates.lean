import Lean.Data.Json

/-!
# Tropical.Tropicaltest.Gates

The verdict substrate every suite draws on, with no runtime behind it: the
gate combinators (`passGate`/`failGate`/`hashGate`), the sample arithmetic
(`bitDiffCount`/`energyOf`) and `sha256Hex`. Suites that never render import
this and nothing of the host; the render and compile drivers live in
`Harness`, which imports this.
-/

def firstLine (s : String) : String := (s.splitOn "\n").headD ""

/-- SHA256 hex via `shasum -a 256` — the same digest node crypto and
    `diff_render.sh` produce over the same bytes. -/
def sha256Hex (bytes : ByteArray) : IO String := do
  let tmp := "/tmp/tropicaltest-render.bin"
  IO.FS.writeBinFile tmp bytes
  -- `shasum -a 256` on macOS, `sha256sum` on Linux; both print "<hex>  <file>".
  let out ← (try IO.Process.run { cmd := "shasum", args := #["-a", "256", tmp] }
             catch _ => IO.Process.run { cmd := "sha256sum", args := #[tmp] })
  pure (out.splitOn " " |>.headD "")

-- ── Gate combinators ─────────────────────────────────────────────────────────
-- Every gate's verdict funnels through these two, so the protocol line the
-- runner (and the label-diff check) depends on — `  PASS  <label>  <detail>` /
-- `  FAIL  <label>  <detail>` — is typed in exactly one place.

/-- Print the gate's PASS line and return `true`. -/
def passGate (label msg : String) : IO Bool := do
  IO.println s!"  PASS  {label}  {msg}"
  pure true

/-- Print the gate's FAIL line and return `false`. -/
def failGate (label msg : String) : IO Bool := do
  IO.println s!"  FAIL  {label}  {msg}"
  pure false

/-- Bit-differing sample count over the two arrays' common prefix
    (IEEE `!=` — the pointwise-equality reading the byte-gates use). -/
def bitDiffCount (a b : Array Float) : Nat := Id.run do
  let mut d := 0
  for i in [0:min a.size b.size] do
    if a[i]! != b[i]! then d := d + 1
  return d

/-- Σx² over the whole array. -/
def energyOf (xs : Array Float) : Float :=
  xs.foldl (fun acc s => acc + s * s) 0.0

/-- The hash-equivalence epilogue: sha256 both renders and PASS iff they
    agree, with the gate's own verdict texts built from the hash(es). -/
def hashGate (label : String) (lhs rhs : ByteArray)
    (passMsg : String → String) (failMsg : String → String → String) : IO Bool := do
  let lhsHash ← sha256Hex lhs
  let rhsHash ← sha256Hex rhs
  if lhsHash == rhsHash then passGate label (passMsg lhsHash)
  else failGate label (failMsg lhsHash rhsHash)
