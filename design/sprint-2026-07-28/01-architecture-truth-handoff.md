# Architecture truth pass — sprint handoff

- **Sprint:** 2026-07-28 through 2026-08-10
- **Lane:** A — documentation and architectural truth
- **DRI:** Assign at kickoff
- **Supervisor:** Staff engineer
- **Status:** Planned
- **Master:** [Staff engineer sprint handoff](00-staff-engineer-master-handoff.md)
- **Depends on:** Performance lane supplies current measurements by Day 7;
  Trust lane supplies the final obligation names by Day 7.
- **Must not overlap:** This lane does not change compiler or runtime behavior.

## Mission

Make the checked-in explanation of Tropical describe the program that is
actually in the tree after the July 25–27 retirements.

At sprint close, a new engineer must be able to follow one current path from
an arrow builder or patch document to a running JIT, WebAssembly, or Metal
kernel without encountering a deleted type, pass, tool, or behavioral claim.
Historical documents may retain old vocabulary only when they are marked
historical at the point of use.

This is specification work, not prose cleanup. In a proof-oriented compiler,
the architecture documents name the intended refinement boundaries. Stale
boundaries make later proofs and reviews point at the wrong program.

## Current mismatch ledger

The opening audit starts from these confirmed mismatches:

1. `design/architecture.md` still routes the JSON front door through
   `ParsedProgram` and `Elaborator`; both were retired on 2026-07-26.
2. The same document still describes the deleted rich strata passes and the
   former `CoreArena` twin.
3. Root documentation describes two execution backends while Metal is a live,
   tested backend and the audio path on supported Apple builds.
4. `README.md` describes name resolution, monomorphization, sum lowering, and
   sub-millisecond compilation as present-day properties.
5. `lean/Tropical/Engine/Compile.lean` has a module comment saying the session
   is elaborated and downcast even though it now constructs the resolved root
   directly.
6. `engine/tests/CLAUDE.md` presents legacy state/register fixtures as if they
   described the production CF-only compiler.
7. Playground performance notes stop before the banks-as-data landing and
   therefore cannot support a current compile-latency claim.

The DRI must extend this ledger before editing. Do not assume this list is
complete.

## Settled vocabulary for the rewrite

The following terms are current and must be used consistently:

- **Authoring tree:** `Tropical.EmitArrow.Sig`, fourteen constructors.
- **Resolved IR:** `Tropical.Ir.ENode` in an `ExprArena`; this is the trunk
  vocabulary, not one stage in a rich-to-core sequence.
- **Patch front door:** `tropical_program_2` is a patch bay over registered
  program types. It cannot define a new program body.
- **Wire grammar:** `Tropical.WireExpr`, decoded at ingest.
- **Direct lowering:** acyclicity check, optional instance inlining,
  identity elimination, and reachability copy into the same vocabulary.
- **Plan:** `tropical_plan_5`, with typed slots, sources, sinks, stage blocks,
  and optional bank regions.
- **Execution targets:**
  - LLVM IR in ORC JIT for native execution and the CPU reference;
  - the same LLVM route compiled to WebAssembly for the browser player;
  - MSL plus `MetalKernel` for supported Apple live audio.
- **State contract:** production Tropical kernels are pure
  `f(τ, params)`. Legacy plan/runtime state support, if retained, is
  compatibility substrate and must be labeled as such.

## Owned files

Primary ownership:

- `README.md`
- `CLAUDE.md`
- `design/architecture.md`
- `engine/CLAUDE.md`
- `engine/tests/CLAUDE.md`
- `mcp/CLAUDE.md`
- `patches/CLAUDE.md`
- stale module-level comments identified by the audit

This lane may edit comments in source files, but not definitions, signatures,
tests, build flags, or runtime behavior. If a truthful statement requires a
code change, file an issue in the sprint decision log and hand it to the lane
that owns that code.

## Deliverables

### A1. Source-to-sound walkthrough

Rewrite the authoritative architecture path with three front doors and three
execution targets:

```text
Lean arrow builders ─┐
MCP patch mutations ─┼─> ResolvedProgram/ExprArena
program_2 patch JSON ┘        │
                              ├─ direct lowering
                              ├─ partition + stage-0 split
                              └─ tropical_plan_5
                                  ├─ LLVM → ORC JIT
                                  ├─ LLVM → wasm32
                                  └─ MSL → Metal
```

The document must distinguish authoring-time construction, compile-time
lowering, control-plane parameter writes, and per-sample execution.

### A2. Invariant index

Add a compact index that states where each invariant is created, represented,
checked, and consumed:

| Invariant | Created/checked at | Represented by | Downstream consumer |
|---|---|---|---|
| Acyclic graph | patch ingest/export/session construction | topological order / cycle refusal | lowering and emit |
| Closed-form kernel | source vocabulary | absence of state constructors | all backends |
| Typed wire | JSON decoder | `WireExpr` | session lowering |
| Bank order | authoring + emitter theorem | `bankSum` / reduce region | JIT, wasm, Metal |
| Stage separation | `Stage0.hoist` | typed stage blocks | staged load |
| Host write discipline | playground report/plan | `param_disciplines` | Lean and socket hosts |

Use links to exact modules, not approximate line-number prose.

### A3. Live-edit contract

State the difference between:

- parameter edits: slot writes, no relower;
- structural selector changes: relower and hot-swap;
- topology changes: relower and hot-swap;
- kernel publication: clickless because the function is re-evaluated at the
  current coordinate, not because arbitrary state is copied.

Replace “every edit is sub-millisecond” or equivalent claims with measured,
qualified statements supplied by Lane D.

### A4. Backend and test matrix

Document what each backend is for and what proves what:

| Target | Numeric regime | Product role | Correctness evidence |
|---|---|---|---|
| ORC JIT | f64 plus i64 rails | native reference, scopes, portable native audio | goldens and native realization checks |
| WebAssembly | shared LLVM semantics | precompiled browser player | wasm-vs-JIT |
| Metal | f32 value path plus exact integer clock rail | heavy live modal audio on Apple | Metal-vs-JIT tolerance/SNR and runtime tests |

Do not say backend agreement proves source semantics. Cross-reference the trust
ledger produced by Lane C.

### A5. Historical labeling

Every retained document that describes an extinct architecture must begin with
one of:

- `Status: Historical — not a description of current main`
- `Status: Superseded by <link>`
- `Status: Current`

Do not rewrite archived design conversations into current prose. Preserve the
reasoning and make their status unmistakable.

## Work plan

### Days 1–2: audit

- Search tracked prose and module headers for:
  `ParsedProgram`, `Elaborator`, `CoreArena`, `ArrayLower`, `SumLower`,
  `Specialize`, `programDecl`, “two backends,” “state transfer,” and
  “sub-millisecond.”
- Classify each hit as current, historical, compatibility-only, or wrong.
- Commit the mismatch ledger to the lane PR before the rewrite begins.

### Days 3–5: authoritative rewrite

- Rewrite `design/architecture.md` top to bottom.
- Bring root `CLAUDE.md` and `README.md` into agreement with it.
- Update the current module map and build/test commands.
- Submit the first review by the end of Day 5. The reviewer checks against
  source, not against another document.

### Days 6–7: subordinate docs

- Update engine, engine-test, MCP, and patch documentation.
- Consume the compatibility classification from Lane F.
- Consume preliminary performance numbers from Lane D.

### Days 8–9: consistency pass

- Run the stale-token audit again.
- Check every local Markdown link.
- Resolve contradictions between root, architecture, and subsystem docs.
- Integrate the final obligation names from Lane C.

### Day 10: freeze

- No new architectural prose after noon except corrections required by the
  release candidate.
- Hand the audit output and remaining historical exceptions to the staff
  engineer.

## Acceptance gates

The lane is complete only when:

1. `design/architecture.md` contains no live path through a deleted elaborator
   or deleted strata pass.
2. All three execution targets and their numeric regimes are described.
3. The parameter-edit/topology-edit distinction is explicit.
4. Every use of old stateful behavior is marked compatibility-only or
   historical.
5. Performance claims cite Lane D’s dated measurements.
6. The following searches return only reviewed, annotated hits:

   ```bash
   git grep -n -E 'ParsedProgram|Ir/Elaborator|CoreArena|ArrayLower|SumLower|Specialize'
   git grep -n -E -i 'two backends|sub-millisecond|state transfer'
   ```

7. `make validate` remains green; prose-only changes are not allowed to hide a
   pre-existing red gate.

## Non-goals

- No new DSP nodes or modal atoms.
- No compiler pass resurrection.
- No API redesign.
- No automatic documentation generator in this sprint.
- No claim that an empirical gate is a theorem.

## Stop and escalate

Stop the lane and ask the staff engineer for a decision if:

- source and current tests enforce different contracts;
- the compatibility lane cannot classify a stateful path;
- the performance lane cannot reproduce a number currently used in product
  claims;
- making the documentation true would require a behavior change.

## Handoff package

At closure, leave:

- the final mismatch ledger, with every entry resolved or assigned;
- the stale-token audit output;
- a short “current architecture in five minutes” section in
  `design/architecture.md`;
- links to the trust ledger, performance report, Metal report, and
  compatibility matrix;
- no uncommitted generated files.
