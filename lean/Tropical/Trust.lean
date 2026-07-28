import Tropical.Semantics

/-!
# Tropical trusted-boundary ledger

This file is the machine-readable source of truth.  Runtime claims remain
ordinary data unless an actual Lean theorem symbol is recorded; the ledger
does not manufacture propositions for facts established by tests, inspection,
numeric tolerance, unsafe optimization, or external dependencies.
-/

namespace Tropical.Trust

inductive EvidenceKind where
  | theorem
  | executableGate
  | differential
  | golden
  | inspection
  | numericTolerance
  | unsafeOptimization
  | externalDependency
deriving BEq, Repr, Inhabited

inductive Status where
  | proved
  | evidenceBacked
  | open
  | external
  | scoped
deriving BEq, Repr, Inhabited

inductive Priority where
  | critical | high | medium | low
deriving BEq, Repr, Inhabited

structure Obligation where
  id : String
  statement : String
  evidence : Array EvidenceKind
  implementationPaths : Array String
  gateNames : Array String
  owner : String
  status : Status
  theoremSymbol : Option String := none
  limitation : String
  priority : Priority
deriving Repr, Inhabited

structure TrustSite where
  id : String
  path : String
  symbol : String
  target : Option String := none
  obligationId : String
  kind : String
deriving BEq, Repr, Inhabited

def knownGates : Array String := #[
  "lake-build:Tropical.Semantics",
  "semantics-production-fixtures",
  "semantics-pointer-differential",
  "clock-algebra-theorems",
  "reduce-coverage",
  "msl-goldens",
  "msl-column-guard",
  "exact-carrier",
  "patch-goldens",
  "migration-goldens",
  "native-mode-equivalence",
  "wasm-vs-jit",
  "web-plans-vs-jit",
  "metal-vs-jit",
  "metal-ctest",
  "mcp-protocol",
  "patch-bay-refusal",
  "production-non-emission",
  "compat_legacy_plan4_manifest"
]

/-- Exactly the current production Lean trust escape surface.  The two unsafe
    definitions and their `implemented_by` marker are three sites supporting
    one optimization/refinement obligation, not three independent assumptions. -/
def productionTrustSites : Array TrustSite := #[
  { id := "LOWER_SIG_PTR_GO_UNSAFE"
    path := "lean/Tropical/EmitArrow/Sig.lean"
    symbol := "lowerSigPtrGo"
    obligationId := "LOWER_SIG_PTR_REFINES_TREE"
    kind := "unsafe definition" },
  { id := "LOWER_SIG_PTR_UNSAFE"
    path := "lean/Tropical/EmitArrow/Sig.lean"
    symbol := "lowerSigPtr"
    obligationId := "LOWER_SIG_PTR_REFINES_TREE"
    kind := "unsafe definition" },
  { id := "LOWER_SIG_IMPLEMENTED_BY"
    path := "lean/Tropical/EmitArrow/Sig.lean"
    symbol := "lowerSig"
    target := some "lowerSigPtr"
    obligationId := "LOWER_SIG_PTR_REFINES_TREE"
    kind := "implemented_by marker" }
]

def obligations : Array Obligation := #[
  { id := "LOWER_SIG_TREE_RELATIONAL"
    statement := "lowerSigTree produces the all-constructor LowersTo relation over production Sig, ENode, ExprArena, and eintern."
    evidence := #[.theorem]
    implementationPaths := #["lean/Tropical/Semantics/LowerSig.lean"]
    gateNames := #["lake-build:Tropical.Semantics", "semantics-production-fixtures"]
    owner := "Lean semantics"
    status := .proved
    theoremSymbol := some "Tropical.Semantics.lowerSigTree_lowersTo"
    limitation := "Approved fallback 1: this is structural/relational lowering, not denotational preservation. DedupSound preservation is the next prerequisite."
    priority := .critical },
  { id := "CLOCK_RAIL_IS_EXACT"
    statement := "Per sample, runtime integer rail operations implement the Int denotation modulo the documented i64 image and shift/mask headroom rules."
    evidence := #[.theorem, .inspection, .golden]
    implementationPaths := #["lean/Tropical/EmitArrow/ClockAlgebra.lean", "lean/Tropical/Ir/EmitLlvm.lean", "lean/Tropical/Ir/EmitMsl.lean"]
    gateNames := #["clock-algebra-theorems", "patch-goldens"]
    owner := "Backend correctness"
    status := .open
    limitation := "Front-end algebra is proved; correspondence to emitted LLVM/MSL integer execution is inspected, not proved."
    priority := .critical },
  { id := "REDUCE_REGION_EXECUTES_IN_ARRAY_ORDER"
    statement := "JIT, wasm, and MSL execute bank bodies at increasing indices with a scalar left fold and a dynamic count clamped to the static capacity."
    evidence := #[.theorem, .executableGate, .inspection]
    implementationPaths := #["lean/Tropical/EmitArrow/BankOrder.lean", "lean/Tropical/Ir/EmitBankLaws.lean", "engine/jit/OrcJitEngine.cpp", "lean/Tropical/Ir/EmitMsl.lean"]
    gateNames := #["reduce-coverage", "msl-column-guard", "manual:backend reduce-loop inspection"]
    owner := "Backend correctness"
    status := .open
    limitation := "Tree order, region shape, and prefix clamp are proved; backend execution of the region remains a named runtime assumption."
    priority := .critical },
  { id := "LOWER_SIG_PTR_REFINES_TREE"
    statement := "For immutable Sig, pointer-memoized lowerSigPtr returns the same id and expression-arena result as structural lowerSigTree."
    evidence := #[.unsafeOptimization, .differential, .inspection]
    implementationPaths := #["lean/Tropical/EmitArrow/Sig.lean", "lean/Tropical/Testing/Semantics.lean"]
    gateNames := #["semantics-pointer-differential", "manual:pointer memo code review"]
    owner := "Lean semantics"
    status := .open
    limitation := "Differential fixtures cover representative sharing, arrays, nested banks, and re-intern every emitted node to observe its dedup hit; this remains finite evidence, and ptrAddrUnsafe identity is intentionally not modeled as a theorem."
    priority := .critical },
  { id := "LLVM_TEXT_EXECUTES_PLAN"
    statement := "Generated LLVM implements tropical_plan_5 instruction, source, instance, and sink semantics."
    evidence := #[.executableGate, .golden, .inspection]
    implementationPaths := #["lean/Tropical/Ir/EmitLlvm.lean", "engine/jit/OrcJitEngine.cpp", "engine/runtime/NumericProgramParser.hpp"]
    gateNames := #["patch-goldens", "native-mode-equivalence", "manual:LLVM emitter review"]
    owner := "JIT backend"
    status := .open
    limitation := "Broad executable evidence exists, but neither LLVM text nor LLVM's optimizer has a mechanized refinement proof."
    priority := .critical },
  { id := "MSL_EXECUTES_PLAN_WITHIN_NUMERIC_CONTRACT"
    statement := "Generated MSL implements the plan within the documented f32 SNR contract."
    evidence := #[.numericTolerance, .differential, .golden, .inspection]
    implementationPaths := #["lean/Tropical/Ir/EmitMsl.lean", "engine/metal/MetalKernel.mm", "tests/web/metal_vs_jit.test.ts"]
    gateNames := #["metal-vs-jit", "metal-ctest", "msl-goldens"]
    owner := "Metal backend"
    status := .evidenceBacked
    limitation := "Evidence is tolerance-based and hardware-dependent; it is not exact equality or a proof of Metal compiler behavior."
    priority := .high },
  { id := "WASM_SHARES_LLVM_SEMANTICS"
    statement := "The wasm artifact is emitted from the same LLVM instruction semantics and remains sample-equivalent to native JIT on the gated corpus."
    evidence := #[.differential, .executableGate]
    implementationPaths := #["engine/jit/OrcJitEngine.cpp", "tests/web/wasm_vs_jit.test.ts", "tests/web/web_plans_vs_jit.test.ts"]
    gateNames := #["wasm-vs-jit", "web-plans-vs-jit"]
    owner := "Wasm backend"
    status := .evidenceBacked
    limitation := "Same producer narrows implementation drift; target-specific LLVM and linker behavior remain external."
    priority := .high },
  { id := "EXACT_FORMULA_APPROXIMATION"
    statement := "The Exact carrier bounds rounding of implemented formulas and shipped constants."
    evidence := #[.theorem, .executableGate]
    implementationPaths := #["lean/Tropical/Exact", "lean/Tropical/Tropicaltest/Exact.lean"]
    gateNames := #["exact-carrier"]
    owner := "Numeric semantics"
    status := .scoped
    limitation := "The interval results do not yet bound distance from the true transcendental functions over their complete domains."
    priority := .high },
  { id := "HOST_PARAM_DISCIPLINES"
    statement := "Lean and socket hosts apply the same re-anchoring formulas, slot order, and by-name control reapplication discipline."
    evidence := #[.differential, .executableGate, .inspection]
    implementationPaths := #["lean/Tropical/Engine", "engine/runtime/FlatRuntime.cpp", "mcp"]
    gateNames := #["mcp-protocol", "manual:host re-anchor inspection"]
    owner := "Host integration"
    status := .open
    limitation := "The runtime deliberately does not transfer slot values kernel-to-kernel; hosts must reapply them by name."
    priority := .high },
  { id := "CACHE_KEY_COHERENCE"
    statement := "Every semantic execution change invalidates cached native kernels."
    evidence := #[.inspection, .executableGate]
    implementationPaths := #["engine/jit/OrcJitEngine.cpp", "engine/runtime/FlatRuntime.cpp"]
    gateNames := #["manual:cache key and build-id inspection"]
    owner := "JIT runtime"
    status := .open
    limitation := "The plan hash plus dylib build id does not automatically protect a runtime-only behavior change when a stale dylib remains installed."
    priority := .high },
  { id := "EXTERNAL_TOOLCHAIN"
    statement := "Lean core arithmetic, LLVM/lld, Metal, RtAudio, and Turnstile are distinct external trusted dependencies."
    evidence := #[.externalDependency, .executableGate]
    implementationPaths := #["lean/lean-toolchain", "lean/lakefile.toml", "CMakeLists.txt", "lib/rtaudio"]
    gateNames := #["patch-goldens", "wasm-vs-jit", "metal-ctest", "mcp-protocol"]
    owner := "Build and release"
    status := .external
    limitation := "Pinning and tests constrain versions; they do not verify the implementations of external compilers, frameworks, drivers, or hardware."
    priority := .high },
  { id := "LEGACY_PLAN_4_IS_NOT_SOURCE_SEMANTICS"
    statement := "Plan-4 compatibility is a parser/runtime lift only and cannot expand the production source language."
    evidence := #[.inspection, .executableGate]
    implementationPaths := #["engine/runtime/NumericProgramParser.hpp", "lean/Tropical/PlanDecode.lean", "lean/Tropical/Parse/Raise.lean", "lean/Tropical/WireExpr.lean"]
    gateNames := #["patch-bay-refusal", "production-non-emission",
      "compat_legacy_plan4_manifest", "manual:plan-4 compatibility review"]
    owner := "Compatibility"
    status := .scoped
    limitation := "Hand-authored plan-4 test fixtures exercise runtime compatibility but are outside source-level semantics. FlatPlan.ofWire does not itself validate the top-level schema tag; production emitters construct plan 5 and FlatRuntime validates native loads, so any new direct ofWire caller must preserve that precondition."
    priority := .medium },
  { id := "FROZEN_AUDIO_GOLDENS_ANCHOR_CORRECTNESS"
    statement := "Frozen audio hashes are the independent behavioral anchor after retirement of the TypeScript compiler."
    evidence := #[.golden, .executableGate]
    implementationPaths := #["tests/golden", "lean/Tropicaltest.lean"]
    gateNames := #["patch-goldens", "migration-goldens"]
    owner := "Audio correctness"
    status := .evidenceBacked
    limitation := "Goldens sample a finite corpus and preserve historical output; they do not prove untested programs correct."
    priority := .high },
  { id := "CYCLE_REFUSAL_AT_CONSTRUCTION_BOUNDARIES"
    statement := "Every production graph-construction boundary rejects cycles because the closed-form IR has no delay/state primitive."
    evidence := #[.theorem, .executableGate, .inspection]
    implementationPaths := #["lean/Tropical/Ir/Cycles.lean", "lean/Tropical/Engine/Compile.lean", "lean/Tropical/Engine/ProgramIO/Export.lean"]
    gateNames := #["patch-bay-refusal", "mcp-protocol"]
    owner := "Compiler front door"
    status := .evidenceBacked
    limitation := "Acyclicity checks cover current constructors; new construction boundaries must register their own gate and ledger update."
    priority := .high }
]

private def evidenceName : EvidenceKind → String
  | .theorem => "theorem"
  | .executableGate => "executable gate"
  | .differential => "differential"
  | .golden => "golden"
  | .inspection => "inspection"
  | .numericTolerance => "numeric tolerance"
  | .unsafeOptimization => "unsafe optimization"
  | .externalDependency => "external dependency"

private def statusName : Status → String
  | .proved => "proved"
  | .evidenceBacked => "evidence-backed"
  | .open => "open"
  | .external => "external"
  | .scoped => "scoped"

private def priorityName : Priority → String
  | .critical => "critical"
  | .high => "high"
  | .medium => "medium"
  | .low => "low"

private def join (separator : String) (items : Array String) : String :=
  String.intercalate separator items.toList

private def renderedPaths (paths : Array String) : String :=
  join ", " (paths.map fun path => s!"`{path}`")

private def renderObligation (entry : Obligation) : String :=
  let formal := match entry.theoremSymbol with
    | some symbol => s!"- Formal symbol: `{symbol}`\n"
    | none => "- Formal symbol: none\n"
  let evidence := join ", " (entry.evidence.map evidenceName)
  let gates := join ", " (entry.gateNames.map fun gate => "`" ++ gate ++ "`")
  s!"## {entry.id}\n\n" ++
  s!"{entry.statement}\n\n" ++
  s!"- Status: {statusName entry.status}\n" ++
  s!"- Priority: {priorityName entry.priority}\n" ++
  s!"- Owner: {entry.owner}\n" ++
  s!"- Evidence: {evidence}\n" ++
  formal ++
  s!"- Implementation: {renderedPaths entry.implementationPaths}\n" ++
  s!"- Gates: {gates}\n" ++
  s!"- Limitation: {entry.limitation}\n"

/-- Deterministic, complete human report generated from `obligations`. -/
def renderMarkdown : String :=
  "# Trusted boundary ledger\n\n" ++
  "_Generated from `lean/Tropical/Trust.lean`; do not edit this file by hand._\n\n" ++
  "Only entries with status `proved` and a checked theorem symbol are formal proofs. " ++
  "Runtime execution, numeric tolerances, external tools, and unsafe refinements remain explicit obligations.\n\n" ++
  "## Adding an obligation\n\n" ++
  "Add one owned entry to `obligations`, use only maintained gate names (or an explicit `manual:` gate), " ++
  "and regenerate this report with `trustreport --write`. A new production `unsafe def` or " ++
  "`implemented_by` marker must also add one typed `productionTrustSites` row; " ++
  "`tools/audit_trust_sites.py` makes an unledgered or stale site fail validation.\n\n" ++
  join "\n" (obligations.map renderObligation)

/-- Stable TSV projection consumed by the filesystem audit. -/
def renderTrustSites : String :=
  join "" (productionTrustSites.map fun site =>
    let target := site.target.getD ""
    s!"{site.kind}\t{site.path}\t{site.symbol}\t{target}\n")

private def isKnownGate (gate : String) : Bool :=
  gate.startsWith "manual:" || knownGates.contains gate

private def obligationExists (id : String) : Bool :=
  obligations.any (·.id == id)

/-- Pure audit used both by the library theorem below and the ordinary test
    runner.  Filesystem discovery of trust sites is a documented companion
    shell check; the checked list here makes omissions and ownership queryable. -/
def auditLedger : Array String := Id.run do
  let mut errors := #[]
  for entry in obligations do
    if (obligations.filter (·.id == entry.id)).size != 1 then
      errors := errors.push s!"duplicate obligation id: {entry.id}"
    if entry.id.isEmpty then errors := errors.push "empty obligation id"
    if entry.owner.isEmpty then errors := errors.push s!"ownerless obligation: {entry.id}"
    if entry.statement.isEmpty then errors := errors.push s!"missing statement: {entry.id}"
    if entry.implementationPaths.isEmpty then
      errors := errors.push s!"missing implementation path: {entry.id}"
    if entry.evidence.isEmpty then errors := errors.push s!"missing evidence: {entry.id}"
    for gate in entry.gateNames do
      if !isKnownGate gate then errors := errors.push s!"unknown gate '{gate}' on {entry.id}"
    if entry.status == .proved then
      if entry.theoremSymbol.isNone then
        errors := errors.push s!"proved without theorem symbol: {entry.id}"
      if !(entry.evidence.contains .theorem) then
        errors := errors.push s!"proved without theorem evidence: {entry.id}"
  for site in productionTrustSites do
    if (productionTrustSites.filter (·.id == site.id)).size != 1 then
      errors := errors.push s!"duplicate trust site id: {site.id}"
    if site.path.isEmpty || site.symbol.isEmpty || site.kind.isEmpty then
      errors := errors.push s!"incomplete trust site: {site.id}"
    if !obligationExists site.obligationId then
      errors := errors.push s!"unowned trust site: {site.id}"
  if productionTrustSites.size != 3 then
    errors := errors.push "production trust-site inventory must contain exactly the current three Lean sites"
  return errors

theorem ledger_audit_passes : auditLedger = #[] := by native_decide

end Tropical.Trust
