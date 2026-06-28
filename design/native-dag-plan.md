# Native-DAG strata refactor (#190) — execution plan

Worktree `tropical-dag`, branch `feat/native-dag-strata` off origin/main.
Goal: the inlining bloat **never forms** — the strata pipeline substitutes
shared `ExprId`s, and the DAG flows to emit without ever being re-materialized
as a tree. (Step 1, already on main, interns *after* the bloat formed; this
makes it never form.)

## Substrate (already on main)
- `Ir/ExprArena.lean` — `ENode`/`ExprArena` (full 27-ctor Expr DAG), `EProgram`,
  `toEProgram`, `ExprArena.toExpr`.
- `Ir/CoreArena.lean` — `CNode`/`CoreArena` (14-ctor post-strata), wired into emit.

## Phase A — id-form strata, materialize-to-tree at exit  (CONTAINED, green-gated)
`Strata.run` keeps its exact signature. Internally: `Arena → EArena` (entry),
run id-passes, `EArena → (Arena, ProgramIdx)` (exit, via `ExprArena.toExpr`).
Downstream (Core.check, compileResolved, emit) and all callers UNCHANGED.
The exit re-materializes the tree (temporary re-bloat — thrown away in B), but
goldens validate every id-pass, especially inlineInstances' shared-id
substitution. No perf win yet; this is the correctness firewall.

Blast radius: `Strata/{Basic,Specialize,SumLower,InlineInstances,ArrayLower,
IdentityElim}.lean` + `Strata.lean` + new `Strata/EArena.lean` + `RecursionId`.

Pieces:
1. `EArena` = { base : Arena (typeParam/typeDef pools), programs : Array EProgram,
   exprs : ExprArena }. PassM = StateT EArena (Except Error). einternP/derefP/
   pushProgramP/typeParam?/typeDef?. `ofArena` (entry), `materialize` (exit).
2. `mapExprId` (id MapHooks: node : ENode → PassM (Option ExprId); binder).
   Serves Specialize + IdentityElim.
3. Convert passes to PassM (custom id-walkers for SumLower/ArrayLower/Inline).
   inlineInstances: hook `inputRef i → wired ExprId` (shared, not re-walked) —
   the bloat never forms in the arena.
4. Strata.run: ofArena → id-passes → materialize.
GATE: tropicaltest 12/12 + web wasm_vs_jit/web_plans_vs_jit bit-identical. Commit.

## Phase B — id-form downstream, drop the exit materialize  (perf realized)
- `Core.check` id-form: walk post-strata `EProgram`+`ExprArena`, assert core
  subset (same error strings), map `ENode→CNode` into a `CoreArena`, resolve
  port types (needs typeDef pool). Produces an id-valued CoreProgram + CoreArena.
- `compileResolved`/`emitResolvedProgram`: consume `ExprId` roots over the shared
  CoreArena (emit's `compileNode` already takes ExprId; drop `internExpr`).
- `Strata.run` returns id-form (new signature) OR a thin seam hands the EArena to
  check directly; update `Engine.runStrataChecked` + the per-program path.
GATE: same. Commit. Bloat now never forms end-to-end; measure flanger compile.

## Phase C — finale (stretch, = old task #15)
Elaborator interns at construction; Codec encode-deref/decode-intern; delete the
tree `Expr` and `mapExpr`. Assess after B; not required for the perf/principle win.

## Gate commands
- `cd /Users/willishoke/tropical-dag && cd lean && PATH="$HOME/.elan/bin:$PATH" lake build diffcli tropicaltest && cd ..`
- `./lean/.lake/build/bin/tropicaltest`  (12/12, from repo root)
- `TROPICAL_ENGINE_CMD="./lean/.lake/build/bin/frontend --rpc" bun test tests/web/wasm_vs_jit.test.ts tests/web/web_plans_vs_jit.test.ts`
- Perf: `time ./lean/.lake/build/bin/diffcli compile patches/flanger_probe.json`
