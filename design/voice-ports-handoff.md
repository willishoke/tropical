Status: Historical — not a description of current main

# Voice-ports sprint — handoff

Branch `feat/voice-ports` (PR #194), off `main` @ `13c51ea` (native-DAG merged).
Worktree `/Users/willishoke/tropical-voice`. This doc lets a fresh session resume
without re-deriving the design.

## Where it stands

**Phase 0 + Phase 1 core are DONE and proven.** The generic voice-bearing flanger
compiles **byte-identical** to the hand-wired `FlangeSin`. Remaining: the session
wire-binding (Phase 1 piece #2) + Phases 2–4.

### Commits on the branch
- `b3f8949` Phase 0: `FlangeSin` — golden reference (arbitrary clock-parametric
  voice at 3 warped clocks; renders peak 0.039).
- `a65b078` representation lock (voice ports = sugar over instances; no new IR node).
- `158f7d8` `rw`/`desugarBody` — the voice-application desugar.
- `2dcf29c` `desugarProgram` — desugar + strip consumed voice ports.
- `a3f22fa` render-proof — `voice-desugar` verb + `flanger.md`; generic ≡ FlangeSin.

### Reproduce the proof
```
cd /Users/willishoke/tropical-voice
./lean/.lake/build/bin/diffcli parse-md tests/fixtures/voice/flanger.md \
  | ./lean/.lake/build/bin/diffcli voice-desugar /dev/stdin > /tmp/d.json
./lean/.lake/build/bin/diffcli emit-file /tmp/d.json > /tmp/a.json
./lean/.lake/build/bin/diffcli emit-file stdlib/parsed/FlangeSin.json > /tmp/b.json
diff /tmp/a.json /tmp/b.json   # byte-identical
```

## The design (settled — see `design/voice-ports-plan.md` for the full pass)

tropical = **cartesian, untraced, multi-port** category of clock-indexed closed
forms ("arrows at home" over a closed/pure/compilable base, not Hask). Effects are
`Voice → Voice` morphisms; the one new primitive is **`warp`** (precompose a voice
with a clock transform); **no `loop`** (no feedback). **There are no values** —
`arr f` (clock-transparent) and `warp` (clock-bending) are both morphisms;
"value consumption" is just apply-at-ambient-clock, surviving only as an emit-layer
sharing optimization (hash-cons on `(closedform, clock)`).

Decisive realizations:
- The **warp core already works** — instantiate a clock-parametric voice at chosen
  clocks (`ClockChord`, `ClockReverseProbe`, `ReversibleComb`). The sprint is the
  **binding surface**.
- **Voice ports are sugar over instances.** `v(clockExpr)` desugars to an instance
  of the bound voice clocked at `clockExpr` + `nestedOut`. No new IR node, no
  strata/emit change. (`VoiceDesugar.lean`.)
- **Pointfree, not pointful.** The voice is the morphism's *domain*, bound by the
  wire — not a `<V>` type param. (User chose this over the explicit-generic surface,
  knowing the wire-binding is more work.)
- The desugar **can't live in `resolveExpr`** (it returns an `Expr`, can't *add*
  instances) — it's a pre-elaboration `ParsedProgram → ParsedProgram` pass.
- `voice` parses as `PortTypeDecl.scalar "voice"` (no parser change); recognition
  is in the desugar/elaborator. `desugarProgram` strips the voice port before
  elaboration, so the elaborator never sees the unknown `"voice"` type.

## REMAINING WORK

### Phase 1 piece #2 — the session wire-binding (the crux, next up)
Derive the `VoiceBinding` from an actual voice-port wire (`osc → flanger.v`)
instead of the verb's hardcoded binding. The hard part: a voice-bearing program
**can't resolve standalone** (its voice is unbound), so per-instance resolution
must be **deferred** until the voice wire lands, then re-resolved on rewiring.

Concrete plan:
1. **Detect voice ports.** A program with an input typed `voice` (or, post-`v(...)`
   usage). The session needs to know `flanger.v` is a voice port — read it off the
   program's parsed/port metadata. (`Session.lean` instance/port info;
   `Wiring.lean:56-82` port types.)
2. **Hold the instance unresolved.** When a `Flanger` instance is added without its
   voice bound, `resolveProgramType` (`Engine.lean` ~600-680) can't resolve it —
   skip/defer rather than error.
3. **Derive the binding from the wire.** When `osc.sine → flanger.v` is wired
   (`handleWire`, `Engine.lean` ~875-955), build a `VoiceBinding` from the *source*:
   `programName` = osc's program, `clkInput`/`output` from osc's signature, `params`
   = osc's non-clock instance config. (3a = single clock-parametric source.)
4. **Desugar + resolve then.** Run `VoiceDesugar.desugarProgram` on the Flanger
   parsed form with the binding → concrete program → elaborate + strata → use as
   the instance's resolved type. Needs the Flanger **parsed form retained** (voice
   programs are resolved per-binding, like generics).
5. **Re-resolve on rewiring.** Changing/removing the voice wire re-derives.
6. **The source instance** (osc) is consumed as a voice — it should be absorbed
   into the flanger's applications, not emitted standalone.

Key files: `Engine.lean` (`syncCompile` ~301, `resolveProgramType` ~600,
`handleWire` ~875), `Session.lean` (Wire ~94, instance/wire state ~153),
`Lowering.lean` (`sessionToParsed` ~359). Gate: a session wiring
`FixedSinOsc → Flanger.v` renders == `FlangeSin`.

### Phase 2 — composition + cartesian goldens (mostly verification)
`Flanger → Flanger` one flat kernel (native-DAG); two flangers on a shared sine,
different δ ≡ two independent (the diagonal law). Same-clock consumers share a pure
let, not a recompute.

### Phase 3 — multi-instance cone-capture + clock abstraction
3b: lift a multi-instance upstream cone (`osc → relu → flanger`) into one
clock-parametric program — richer than `WireProgram.lift` (which only lifts a wire
*expression*, refs→inputs; it does NOT inline the cone or touch the clock). Thread
one `clk` input to all clock leaves. 3c: rewrite global `clock()` leaves → the
supplied `clk` for upstreams that aren't already clock-parametric (β-abstract the
clock). Value-port effects (relu) are clock-transparent — the warp propagates
through them to the generators.

### Phase 4 — cable UX + arrow-law goldens
Wiring into a voice port triggers the capture automatically (MCP `wire`/`fan_*`
tools, `Tools.lean`); the bound's signature validates cables. Arrow-law goldens:
algebraically-equal patches render bit-identical.

## Code map (from the exploration agents)
- Surface port types: `Parse/Surface/Declarations.lean:59-110`; `PortTypeDecl` in
  `Parse/Nodes.lean:171`. (No change needed — `voice` is a scalar type name.)
- ParsedExpr / instance / block: `Parse/Nodes.lean:105` / `:243` / `:267`.
- Elaborator: scope `:98-109`, `resolvePortType` `:230-242` (errors on unknown
  type — moot once voice ports are stripped pre-elaborate), `resolveCall` `:348`,
  `resolveNestedOut` `:336`.
- Type-param/specialize machinery (if a pointful escape hatch is ever wanted):
  `Strata/Specialize.lean`, `TypeArgs.lean`, `Nodes.lean` (`TypeParamDecl`,
  `InstanceTypeArg`, `BodyDecl.inst`).
- Clock-first-class (the warp foundation): `clock() = sampleIndex << 32`
  (`Elaborator.lean:362`); instance clock inputs flow via `Emit.lean:510-550`;
  examples `stdlib/reversible/{ClockChord,ClockReverseProbe,ReversibleComb}.md`.

## Build & test (run binaries directly — never `lake exe`; see CLAUDE.md)
```
cd lean && PATH="$HOME/.elan/bin:$PATH" lake build diffcli tropicaltest
./lean/.lake/build/bin/tropicaltest                     # 12/12, from repo root
TROPICAL_ENGINE_CMD="./lean/.lake/build/bin/frontend --rpc" bun test tests/web/wasm_vs_jit.test.ts tests/web/web_plans_vs_jit.test.ts
```
Note: Bash `cd` resets between tool calls in this environment — use absolute paths
or `cd … &&` chains. Submodules in a fresh worktree need
`git submodule update --init --recursive` (rtaudio clone is slow).
