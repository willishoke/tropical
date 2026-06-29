# Voice ports — pointfree generic effects (sprint plan)

Off origin/main (native-DAG merged, 13c51ea). The forcing example is the flanger;
the goal is **generic effects wired like a DAW insert chain**: `osc → flanger`,
`flanger → flanger`, `(osc, env) → vca → filter` — composition, no feedback.

## Semantics (settled — the law layer)

tropical denotes a **cartesian, untraced** category of **clock-indexed,
multi-port** closed forms ("arrows at home" — the composition skeleton over a
closed, compilable base, *not* Hask; no `arr`-of-arbitrary-function leak).

- A **voice** is `Clock → Signal` (multi-port: `Clock → {named outs}`). Element
  type ranges over the signal typeclass (primitives + arrays).
- A **generator** is a point (`I → Voice`); an **effect** is a morphism
  (`Voice → Voice`). Generator/effect is the closed-term/one-hole-context split —
  UX taxonomy, **not** an IR category (the PM-osc sits on the line). One species
  in the IR.
- Combinators: `>>>` (compose = wire), `&&&`/`***` (fan-out/parallel = the
  cartesian diagonal, compiled to a **pure let** — never a slot; there is no
  per-sample state), and **`warp : (Clock→Clock) → Voice → Voice`** — the one
  "stranger primitive", precompose a voice with a clock transform. `delay δ =
  warp(·−δ)`, `reverse = warp(−·)`, `timescale k = warp(k·)`. **No `loop`** (no
  trace, no feedback — the coherence axiom, not a restriction).
- **There are no values.** `arr f` (clock-transparent) and `warp φ` (clock-bending)
  are both `Voice → Voice`. "Value consumption" is just "apply at the ambient
  clock" — the degenerate single-clock case. The value/voice split survives **only**
  at the emit layer as a sharing optimization (same `(closedform, clock)` →
  shared via hash-cons; different clock → its own eval), never as a type.

**Pointfree, not pointful.** An effect is a morphism with a voice *domain*;
genericity is **free by composition** (`osc >>> flanger`, `synth >>> flanger`
reuse the same morphism). No program-valued type parameters — the binding is the
wire, not a `<V>`.

**The cost model** (one line for the doc): you pay one evaluation per distinct
`(cone, clock)`; clock-transparent consumers pass the clock through (share),
clock-bending consumers (voice ports) multiply it by their tap count; hash-cons
dedups coincidences. A many-tap voice port re-evaluates its upstream cone per tap
— O(taps × cone), embarrassingly parallel, the price of reversal/random-access.
(The stateful effect-bus that amortizes many-tap delays is many sprints away and
out of scope.)

## What already exists (the warp core)

- **Warp on a concrete voice**: instantiate a clock-parametric program at chosen
  clocks. `ClockChord` instantiates `ModalVoice` at three clocks; `ClockReverseProbe`
  at `−clk`. Clock is a first-class value (`clock() = sampleIndex << 32`), flows as
  an expression into a child's clock input (`Emit.lean:510-550`).
- **Depth without bloat**: native-DAG strata — `flanger → flanger → flanger`
  stays one flat kernel, shared-id substitution.
- **Fan-out as a pure let**: CSE / the cartesian diagonal. No slots in the signal
  path (only param slots, which are control-plane inputs).

So the missing piece is the **binding surface**: a `voice` port, an application
form, and wire-composition that fills it.

## The new work

### 1. Surface (`Parse/Surface/`, `Parse/Nodes.lean`)
- A **`voice` input port type** with a port-signature bound: `v: voice(out: float)`
  (the bound is the typed interface the cable UX later reads — the principal type
  the row-inference engine would derive; hand-written until that engine exists).
- An **application form** `v(clockExpr)` in body exprs — evaluate the voice at a
  clock. Closest existing construct: instance-call syntax + `nestedOut`.
- A way to name **the effect's own clock** in the body (the ambient clock it is
  evaluated at, which downstream drives) — likely `clock()` reused.

Example:
```tropical
program Flanger(v: voice(out: float), depth: float = 0.0007) -> (out: float) {
  out = v(clock()) + v(clock() - depth)
}
```

### 2. IR + elaboration  (REPRESENTATION LOCKED — Phase 0 confirmed)
A voice input binds a **clock-parametric program**; each `v(clockExpr)` desugars
to an **instance of that program clocked at clockExpr**. This reuses the instance
+ clock + native-DAG machinery wholesale — `v(clk−δ)` is "an instance of the
bound voice, `clk: clk−δ`, take its `out`". The elaborator, knowing `v`'s bound
program, lowers each application to an instance + `nestedOut`.

**The key simplification (verified against the parser/IR):** there is **no new
IR node, no strata change, no emit change.** The pieces all already exist:
- Application `v(clockExpr)` parses as the existing `ParsedExpr.call (nameRef "v")
  [clockExpr]` — no new surface production.
- Its desugar target is `BodyDecl.inst` + `ParsedExpr.nestedOut` — both existing.
- Lowering is the proven FlangeSin shape (Phase 0): instances of a clock-parametric
  voice at warped clocks, summed.

So the novelty is concentrated in exactly two places:
1. **Parse** — a `voice` port *type* (`PortTypeDecl.voice` carrying the output
   signature/bound). The application reuses `call`.
2. **Elaborate** — when a `call`'s callee resolves to a voice-typed input, desugar
   it to a fresh instance of the bound program (clock = the call's arg, other
   params threaded from the binding) + a `nestedOut` to that instance's output.

Plus the **binding** (§3): how the voice input acquires its bound program. Phase 1
takes the 3a path — the voice is wired to a single clock-parametric program; the
binding records that program + its non-clock params; each application supplies the
clock. `v` is the morphism's *domain* (pointfree), bound by the wire — not a `<V>`.

Net: voice ports are **sugar over the program-at-warped-clocks shape Phase 0 already
renders.** The hard part is the desugar + binding plumbing in the elaborator/session,
not the IR.

### 3. The cone-capture binding (the one genuinely new pass — DE-RISK FIRST)
Wiring `osc → flanger.v` must bind `v` to the upstream as a **clock-parametric
program** the flanger instantiates. Phased by difficulty:
- **3a (MVP): single clock-parametric program.** The wired upstream is one
  program already authored with a `clk` input (the reversible-family convention —
  `FixedSinOsc(freq, clk)`, `ClockPhasor(clk, …)`). Binding = attach that program
  to the voice slot; no lifting, no abstraction. `v(clkE)` instantiates it at `clkE`.
- **3b: multi-instance cone.** The upstream is a subpatch (`osc → relu → v`). Lift
  the whole upstream cone into one clock-parametric program (richer than
  `WireProgram.lift`, which only lifts a wire *expression* with refs→inputs — it
  does **not** inline the cone or touch the clock). Clock-transparency propagates:
  the lifted cone threads one `clk` input to all its clock leaves.
- **3c: clock abstraction.** The upstream uses the global `clock()` (not a `clk`
  input). Capture must rewrite `clock()` leaves → the supplied `clk` (β-abstract
  the clock; the voice becomes `λθ. <cone with clock()→θ>`). General; needed for
  arbitrary upstreams.

### 4. Cable UX (`Engine.handleWire`, `Tools.lean`)
Wiring into a voice-typed port triggers 3a–c automatically; the bound's signature
is what the UI reads to validate cables. The user drags a cable; never writes a type.

## Phases & gates

| phase | deliverable | gate |
|---|---|---|
| 0 | **De-risk 3a**: spike a hand-written program with a `voice` input bound to one clock-parametric program, applied at two clocks; confirm it lowers + renders | renders; bit-exact vs a hand-wired two-tap flanger |
| 1 | Surface + IR + elaboration for `voice` ports + `v(clkE)` application; binding 3a | **`Flanger` over `FixedSinOsc` ≡ hand-wired flanger** (hash) |
| 2 | Composition + cartesian goldens | `Flanger→Flanger` one flat kernel; two flangers / shared sine / different δ ≡ two independent (the diagonal golden) |
| 3 | Cone-capture 3b + clock-abstraction 3c | `(osc→relu)→flanger` ≡ hand-wired; arbitrary-upstream voices |
| 4 | Cable UX (wire into voice port) + arrow-law goldens | dragging a cable binds the voice; algebraically-equal patches hash-identical |

## Goldens = the arrow laws

The equivalence spec is stronger than one hash: **every algebraically-equal patch
must render bit-identical.** `arr id >>> f = f`; associativity of `>>>`;
`(id &&& warp(·−δ)) >>> arr(+)` (the flanger) ≡ the hand-wired flanger; the
diagonal law (shared upstream = independent copies). Plus the existing 12 goldens
+ wasm≡JIT, all bit-exact (relabeling from any of this is invisible to the audio).

## Risks

- **Cone-capture + clock-abstraction (3b/3c)** is the load-bearing new pass. De-risk
  in phase 0 with the single-program case before building the surface around it.
- **The effect's-own-clock surface** (`clock()` in an effect body meaning "my
  ambient clock") needs care — when the effect is composed downstream, its clock is
  driven; confirm the propagation matches the clock-transparency semantics.
- **Bound surface ergonomics**: explicit signature is verbose (the C++-concepts
  tradeoff). Acceptable because it's the metadata the cable reads and the principal
  type inference would later derive; not user-facing.

## Gate commands (per the native-DAG sprint)
- build: `cd lean && PATH="$HOME/.elan/bin:$PATH" lake build diffcli tropicaltest`
- goldens: `./lean/.lake/build/bin/tropicaltest` (from repo root)
- web: `TROPICAL_ENGINE_CMD="./lean/.lake/build/bin/frontend --rpc" bun test tests/web/wasm_vs_jit.test.ts tests/web/web_plans_vs_jit.test.ts`
