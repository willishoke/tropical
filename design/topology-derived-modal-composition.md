# Topology-derived modal composition

Date: 2026-08-11 (America/Los_Angeles)

Status: grounded proposal for the next development phase. This document does
not supersede [`architecture.md`](architecture.md); it proposes a bounded route
from the implementation on `main` to a more deeply composable authoring model.

## 1. Decision requested

Make the authored graph, including graphs nested inside reusable modules, the
primary language of modal construction.

Do not expose `cascade`, `parallel`, or a growing catalog of effect-specific
compiler combinators as boxes the user must understand. In the authoring
surface:

- a connection is sequential composition;
- a fan-out is shared use of one value;
- several wires entering an ordinary typed inlet are ordered parallel
  composition;
- a module boundary names and hides a subgraph; and
- opening a module reveals the same patching interface again.

The compiler still needs a small algebraic representation of these operations.
That representation is derived from topology and remains compiler-private. It
is what gives the graph a denotation, preserves products without expanding
them, and permits exact structural specializations. The choice is therefore not
“combinators or topology.” It is:

> Topology is the user notation; an algebra is the compiler meaning of that
> topology.

The first landing milestone should make the shipped six-section Phaser a
versioned library graph assembled from first-order modal sections and ordinary
typed wiring. Expanding it must expose an editable patch; compiling the
expanded graph must recover the same compact product and the same production
terminal schedule as the current privileged Phaser path. The existing
resonant low-pass filter should pass through the same generic kernel algebra as
a second consumer, so the new representation is not merely a renamed Phaser.

This phase remains feed-forward and closed-form. It does not relax Tropical's
global cycle rule. A later filter-designer feedback graph may be useful, but it
must be a scoped linear graph that is solved to a closed-form modal operator
before entering the existing acyclic patch. Arbitrary signal feedback still
belongs to the future stateful sister runtime.

## 2. Why this is the next useful seam

Tropical already has most of the mathematical machinery needed for a
bottom-up modal construction environment. What it lacks is an authoring-level
representation that retains the user's factorization.

The implementation that just landed demonstrates both sides:

1. A first-order continuous-time all-pass is already decomposed in
   [`Modal.Oriented`](../lean/Tropical/EmitArrow/Modal/Oriented.lean) as an
   identity path plus one causal exponential tail.
2. `Bank.phaser` proves that six such sections composed in sequence have the
   intended rational meaning.
3. That generic construction duplicates the identity-plus-tail expression at
   each section. It is an oracle, not a viable six-section production carrier.
4. `decorateDegreeZeroCausalPhaser` retains one row per source pole and one per
   section pole instead.
5. `FactoredTwoRoomPhaserTerminal` then preserves the two-room routed images
   rather than flattening the whole product into scalar expressions.

The resulting product is not fast because “Phaser” is inherently a primitive.
It is fast because the compiler knows the expression is a factored cascade of
identity-plus-tail kernels and delays expansion until an appropriate terminal.
That fact can come from topology just as well as from a `.phaser` enum case.

The resonant filter is the complementary example. `filterPair` in
[`Playground.Vocabulary`](../lean/Tropical/Playground/Vocabulary.lean) already
constructs an exact complex-conjugate pole pair, then presents the filter as a
two-mode room. Resonance is not evidence that a cascade secretly contains a
loop. It is a property of the poles of the filter kernel. A feedback circuit
is one way to *derive* those poles; an explicit pole pair is another. Once the
kernel is known, both are the same kind of modal action.

The opportunity is to let users move between these levels:

```text
Phaser
  └─ six Allpass1 sections + dry/wet network
       └─ direct path + causal PoleTail1
            └─ modal pole/residue data and scalar coefficient graph
```

No level needs to be the permanent user-facing floor. A normal patch may show
only `Phaser`. A filter-design view may show the six sections and a response
plot. An expert may open one section and edit its pole and residue network.

## 3. Concrete observations on `main`

The proposal builds beside the current paths rather than replacing the trunk.

| Current fact | Concrete location | Consequence for this phase |
|---|---|---|
| The product graph is a downstream-only DAG and lowering uses topological rank as its termination measure. | `Node.inputIds`, `lowerAt`, and `lowerModal` in [`EmitArrow.Patch`](../lean/Tropical/EmitArrow/Patch.lean) | Feed-forward topology can be elaborated directly. Global feedback cannot be admitted by weakening one check. |
| The public modal compiler value is an authored-order `ModalForest`; a stage maps over every branch and modal mix concatenates branches. | [`Modal.Forest`](../lean/Tropical/EmitArrow/Modal/Forest.lean) and `Semantics.ModalUniverse.lowerGraph` | This preserves meaning and ordering but distributes later stages over earlier sums. A factor-preserving expression must sit before or replace that eager distribution for linear regions. |
| Production stages are a closed enum. | `ModalStage = ordinaryRoom | gauge | phaser` in [`Modal.Forest`](../lean/Tropical/EmitArrow/Modal/Forest.lean) | Adding every effect as another stage case will recreate a curated ceiling and another terminal pattern matrix. |
| Terminal selection pattern-matches stage count and exact case order. | `resolvePlainStages` in [`EmitArrow.Patch`](../lean/Tropical/EmitArrow/Patch.lean) | Specialization should move from effect names to normalized kernel structure. |
| The semantic forest model is already generic over `Source` and `Stage`. | [`Semantics.ModalUniverse.Graph`](../lean/Tropical/Semantics/ModalUniverse.lean) | The whole-universe and authored-order laws can be retained while production grows a richer linear-stage meaning. |
| The oriented algebra already supplies convolution, addition, scaling, blending, first-order all-pass tails, a generic section, and a compact Phaser decoration. | [`Modal.Oriented`](../lean/Tropical/EmitArrow/Modal/Oriented.lean) | These are the initial interpreter and oracle for a topology-derived kernel expression. |
| A `Bank` carries future atoms, past atoms, and a point value at exactly zero. | `Oriented.Bank` | The identity/direct path must be represented separately. `Bank.atZero` is a function value at one point, not a Dirac impulse and cannot stand for feedthrough. |
| `ModalMode` already carries live `Sig` expressions and a declared damping interval. | [`Modal.Residue`](../lean/Tropical/EmitArrow/Modal/Residue.lean) | Atomic pole nodes can remain current-universe and participate in admission without baking live controls. |
| The executable trunk has no modal or Phaser opcode. It has ordinary scalar DAG nodes plus bounded `bankSum` and `routedSum` regions. | [`Ir.Nodes`](../lean/Tropical/Ir/Nodes.lean) | The first phase should eliminate the new modal algebra at the Modal-to-Sig seam and keep LLVM, WASM, MSL, and Plan 6 unchanged. |
| The product vocabulary is closed in Lean and mirrored by a closed Swift `NodeKind`. | `portSpecs`/`buildNode` in [`Playground`](../lean/Tropical/Playground/) and [`NodeKind.swift`](../reversible/Sources/Reversible/NodeKind.swift) | Atomic nodes must first become registered kinds. Reusable user modules require dynamic descriptors rather than another growing Swift enum. |
| Reversible's v2 document is one flat `nodes` array plus presentation, monitors, transport, and one output. | [`PatchDocumentV2.swift`](../reversible/Sources/Reversible/PatchDocumentV2.swift) | There is nowhere to persist a module definition or nested graph today. A versioned document change is required. |
| Reversible and the engine now accept multiple ordered wires on ordinary `in` ports, while controls, addresses, and modulation ports remain exclusive. | `PortSpec.multi`, `buildNode` implicit fan-in, `PatchModel.connect`, and their gates | The needed visual composition gesture has landed. It should remain type-directed and implicit. |
| Cycles are rejected in the Reversible model, v2 validation, `Patch.lowerAt`, session compilation, and direct program/export construction. | [`PatchModel.swift`](../reversible/Sources/Reversible/PatchModel.swift), [`PatchDocumentV2.swift`](../reversible/Sources/Reversible/PatchDocumentV2.swift), and [`Ir.Cycles`](../lean/Tropical/Ir/Cycles.lean) | A filter-design loop must be a separate scoped source form with a total elaboration, not a normal patch edge. |
| `tropical_program_2` is a patch bay over registered types and rejects `programDecl`. | [`Engine.ProgramIO.Ingest`](../lean/Tropical/Engine/ProgramIO/Ingest.lean) | Hierarchical Reversible definitions must not silently revive the retired general wire program language. They should expand hygienically to served atomic graph nodes before the current patch lowering. |

### 3.1 The current performance boundary is already narrow

The canonical `Reson(6) -> Room(32) -> Room(32)` baseline publishes 22,320 of
the allowed 24,576 bytes of Metal threadgroup scratch. The Phaser product
publishes 22,688 bytes. The qualifier reports:

| Shape | Non-coefficient arrays | Largest routed image | Total scratch |
|---|---:|---:|---:|
| two-room baseline | 13,824 B | 2,112 records x 4 B | 22,320 B |
| two rooms plus six-section Phaser | 14,164 B | 2,112 records x 4 B | 22,688 B |

Most of the 22 KiB is therefore the existing factored two-room terminal, not
the Phaser. The Phaser adds 368 bytes, but leaves only 1,888 bytes of policy
headroom. A new compositional authoring layer is acceptable only if it
recovers the compact product. Eagerly lowering the visible graph to generic
`Bank.add`/`Bank.convolveKernel` trees would spend orders of magnitude more
than those 368 bytes and can also make compile-time exact folding pathological.

This is the central implementation constraint of the proposal:

> Fine-grained authored topology must not imply fine-grained expanded runtime
> storage.

## 4. User-visible type model

Keep the existing three visible data domains:

- `Modal`: a deferred modal value;
- `Sig`: an ordinary closed-form signal; and
- `Control`: a patchable scalar control source.

Processing modules retain one primary data morphism: `Modal -> Modal`,
`Sig -> Sig`, or `Modal -> Sig`. Sources, controls, monitors, and sinks are the
expected nullary/boundary exceptions. A module may also have named coefficient
ports, but those do not make its primary data inlet polymorphic.

The connection rules remain simple:

1. A modal primary inlet accepts `Modal`, never `Sig`.
2. A signal primary inlet accepts `Sig`; a `Modal` source may realize there at
   the established one-way seam.
3. A signal never converts back into poles.
4. The destination port descriptor determines the implicit fan-in operation.
   Multiple sources into a modal inlet form an ordered modal sum; multiple
   sources into a signal inlet form an ordered signal sum.
5. Named coefficient, address, and modulation ports remain single-source
   unless their own descriptor explicitly says otherwise.

There is no polymorphic `Mix` operation to explain. `Mix` and `Modal union`
may remain decodable compatibility nodes and optional explicit junctions, but
ordinary construction should use multi-connection inlets. The port type tells
the compiler which sum exists. A signal connected to a modal inlet is a type
error rather than a request for a clever conversion.

## 5. Compiler meaning derived from topology

### 5.1 Linear kernel algebra

Introduce a compiler-private, hash-consed `ModalKernelExpr` with an explicit
distributional direct path:

```text
KernelExpr K ::=
    identity
  | proper(OrientedKernelAtom)
  | scale(control, K)
  | parallel([K])          -- authored order
  | cascade([K])           -- authored order
```

Its denotation is a current-universe kernel `d delta + h(t)`, where `delta` is
the convolution identity and `h` is an oriented modal expansion. The direct
coefficient and proper tail may remain factored rather than being collected
into one pole array.

`parallel` denotes kernel addition. `cascade` denotes convolution in connection
order. Neither constructor authorizes reordering or floating-point
reassociation. Arrays retain authored order, singleton constructors normalize
away, and empty cases have explicit typed identities.

The modal value graph is correspondingly small:

```text
ValueExpr V ::=
    source(ModalSource)
  | apply(K, V)
  | parallel([V])          -- authored order
  | gauge(control, V)
```

`gauge` stays outside `KernelExpr` because it measures and rescales the complete
current modal value. It is not linear and may not be distributed through a
sum or cascade. Bloom likewise retains its existing specialized source truth
and named crossing rules until a general carrier handles it.

These are not new backend instructions and not user-visible combinator nodes.
They are the elaborated meaning of graph topology between `lowerModal` and the
existing Modal-to-Sig terminal.

### 5.2 Topology elaboration is not effect recognition

Each registered atomic node kind must declare a local semantic action:

- a linear modal atom contributes one `KernelExpr` factor;
- a modal source contributes one `ValueExpr.source`;
- a nonlinear modal atom contributes an explicit value operation such as
  `gauge`; and
- a typed multi-inlet contributes ordered `parallel`.

The graph elaborator composes those local meanings from edges. It does not
search an arbitrary graph for something that “looks like a Phaser.” The
normalized result may then be matched by optimization rules to choose a
realizer. Such a match changes cost and representation, never denotation.

This distinction prevents two failure modes:

- semantic behavior does not depend on node names, positions, grouping, or a
  fragile pattern heuristic; and
- an optimizer can refuse an expensive shape without declaring the authored
  graph meaningless.

### 5.3 Initial identities

The current implementations supply the first algebra instances:

```text
Room(h, dir)       = proper(oriented(h, dir))
PoleTail1(a)       = proper(-2a * exp(-a t) * 1[t>0])
Allpass1(a)        = parallel([identity, PoleTail1(a)])
AllpassCascade(as) = cascade(as.map(Allpass1))
Phaser(as, mix)    = parallel([
                        scale(1 - mix, identity),
                        scale(mix, AllpassCascade(as))
                      ])
Filter(fc, q)      = proper(filterPair(fc, q))
```

Applying the Phaser expression to a source is precisely the existing rational
meaning. The expression must stay factored: distributing every
identity-plus-tail sum through every later cascade produces the generic
doubling that the current oracle intentionally exhibits.

### 5.4 Realization policy

A terminal interpreter chooses among existing and future carriers:

1. Small expressions may use the generic `Oriented.Bank` interpreter. This is
   the simple reference path.
2. A causal degree-zero cascade of identity-plus-tail factors may use
   `decorateDegreeZeroCausalPhaser`, generalized and renamed around its
   structural contract rather than the effect name.
3. A source, two compatible rooms, and such a factor may use the current
   `FactoredTwoRoomPhaserTerminal`, likewise selected from normalized structure.
4. Unsupported orientation, coincidence, bloom, or repeated-room crossings
   retain named refusals until an admitted carrier exists.
5. A later structured or lifted realization may interpret the same expression
   without changing the authoring graph.

The first phase should not add `KernelExpr` to `Ir.ENode`. Architecture on
`main` admits backend-visible structure only when a backend needs it as data.
The existing terminal can already eliminate this expression to `Sig`,
`bankSum`, and `routedSum`. Keeping that waist unchanged materially reduces the
risk and gives plan equivalence a strong qualification target.

## 6. Atomic floor and reusable definitions

Arbitrary levels of abstraction do not remove the need for primitives. They
make the primitive floor explicit and allow it to move downward over time.

The first served floor should be small enough to build the Phaser graph without
giving every internal residue field equal prominence in the normal patcher:

| Atom | Primary type | Meaning |
|---|---|---|
| `PoleTail1` | `Modal -> Modal` | Convolution with one causal first-order proper tail. Pole and residue controls remain live and carry declared ranges. |
| `ModalScale` | `Modal -> Modal` | Scale a complete modal value by one control. |
| `ControlAdd`, `ControlMul`, `ControlNeg`, `ControlExp2`, `ControlSin` | `Sig -> Sig` or scalar multi-input | The coefficient network needed to relate a module's exposed controls to pole/residue controls. These are ordinary closed-form signal atoms, not a second formula language. |
| module inlet/outlet boundaries | typed boundaries | A modal outlet may accept several ordered connections and therefore supplies implicit modal fan-in. |

`Allpass1` should itself be a library definition over `PoleTail1`, not an
irreducible compiler case. The Phaser definition can then contain six
`Allpass1` instances and a dry/wet network. The default UI does not display the
coefficient subgraph; opening `Allpass1` does.

If this atom floor proves too large for the first vertical slice, a temporary
`SweptAllpass1 : Modal -> Modal` registered atom is an acceptable bootstrap,
provided that:

- its semantic output is still a generic `KernelExpr`, not a new Phaser stage;
- it is stored as a versioned library dependency rather than hidden in the
  Phaser decoder; and
- the next slice replaces it with the lower atom graph before the hierarchy
  schema is declared stable.

This makes the compromise temporary and testable instead of establishing
“all-pass section” as a new permanent abstraction ceiling.

## 7. Hierarchical patch documents

### 7.1 A separate authoring schema

Reversible needs a v3 document with definitions. Conceptually:

```text
PatchDocumentV3
  vocabulary
  definitions : [ModuleDefinition]
  scene       : Graph
  monitors
  transport
  presentation

ModuleDefinition
  stable id + version
  typed inlet/outlet declarations
  exposed parameter declarations
  graph
  nested presentation
```

A node kind in a graph is either a served engine atom or a reference to one of
these module definitions. Definition references must form a DAG. Each
definition's own graph follows the ordinary typed cycle rule in the first
phase.

This is deliberately not a `tropical_program_2 programDecl` revival:

- it cannot carry arbitrary `ENode`, binder, state, or backend instructions;
- its leaves are only engine-served atomic kinds from the negotiated
  vocabulary;
- it is validated and hygienically expanded before `EmitArrow.PatchGraph`;
  and
- the existing program/session schema and refusal remain unchanged.

The hierarchical elaborator should be Lean-owned so Reversible, a future web
editor, and MCP-driven tooling cannot disagree about expansion. Reversible
owns the lossless document and presentation, sends the definition graph to the
frontend, and receives source-path diagnostics and the normal realized-patch
handshake.

### 7.2 Hygienic expansion

Expansion needs these invariants:

1. Every authored node retains a stable source path.
2. Flat engine ids are derived hygienically from instance path plus local id;
   user text is not concatenated without escaping.
3. Parameter exposure creates one outer live slot and routes it to all inner
   consumers; it must not duplicate independently writable slots.
4. Port order and multi-wire source order survive expansion exactly.
5. Definition and instance graphs are cycle-checked separately and errors name
   the authored path, not only the generated id.
6. A collapsed module and its expanded form lower to the same topology and
   therefore the same plan.
7. Library definitions are versioned. Opening and editing a shipped definition
   creates a detached user definition rather than mutating the installed
   version in place.

### 7.3 Reversible migration

The Swift model has already begun the useful migration: `EngineVocabulary`
has dynamic `NodeKindID` and descriptor-derived port/type checks. The canvas
model still stores the closed `NodeKind` enum and derives layout and behavior
from `NodeKind.spec`.

The hierarchy slice should finish that move:

- store `NodeKindID` plus the negotiated descriptor in `PatchNode`;
- move titles, port types, defaults, ranges, arity, and primary-domain truth to
  `EngineVocabulary`;
- keep only presentation overrides client-side;
- introduce a module-instance descriptor synthesized from its definition;
- add a navigation stack for the scene and nested definition graphs; and
- retain v2 loading as an explicit one-way migration to a v3 scene with no
  definitions.

Unknown-field preservation in v2 is useful but insufficient: a v2 client can
round-trip an unfamiliar field, yet its validator and compiler cannot give a
definition meaning. The version bump must be explicit.

## 8. UI contract

The ordinary user should not be shown category-theory vocabulary or a wall of
coefficient arithmetic.

The default patch remains compact:

```text
[Resonator] -> [Phaser] -> [Reverb] -> [Out]
```

Opening `Phaser` moves into a nested canvas with a breadcrumb:

```text
Patch / Phaser

Modal In -> AP1 -> AP2 -> AP3 -> AP4 -> AP5 -> AP6 -> wet
     |                                                   |
     +---------------- dry ------------------------------+-> blend -> Modal Out
```

Opening an all-pass section shows its direct and tail branches. Control
routing may be collapsed by default and revealed with “Show control routing.”
There is no `Cascade` box: the row of connections is the cascade. There is no
required `Modal union` box at the output: two connections into the typed modal
outlet are the ordered parallel sum. An explicit junction can still be placed
when it improves layout or gives the sum a name.

The filter-designer view is another projection of the same nested graph, not a
second compiler language. It may add:

- transfer magnitude and phase;
- pole/zero display;
- section list and reorder gesture;
- current modal-row/factor count;
- estimated compile and Metal scratch cost; and
- a switch between compact module, section graph, and coefficient graph.

Edits in any view update the same stable node ids. Returning to the main patch
does not synthesize a different hidden definition.

The sequence editor can follow the same document/view principle later, but it
is not part of this landing. There is not yet a shared event/lane authoring IR
in the current patch path, so claiming it here would make the modal milestone
depend on a separate language project.

## 9. Feedback is a later scoped elaboration

The visual topology of a resonant filter often contains feedback. The current
Tropical patch cannot represent that loop, and it should not be taught to do so
by slipping one cyclic edge past `Ir.Cycles`.

A credible later design is a nested `LinearCircuit` definition with different
rules:

1. Every atom in the region declares a linear continuous-time transfer or
   generator meaning.
2. The elaborator finds strongly connected components.
3. Each component is solved symbolically to a rational, state-space, or sparse
   generator representation.
4. Stability, invertibility, parameter ranges, and cost are admitted before
   compilation.
5. The solved result becomes an acyclic `KernelExpr` node at the outer patch
   boundary.

This is how a user could draw the familiar integrator/feedback topology of a
resonant filter while Tropical still emits a random-access closed-form kernel.
The cycle exists in the *design notation* and is eliminated at compile time;
it is not a sample-to-sample runtime recurrence.

The difficult part is not graph traversal. Live coefficients make arbitrary
high-order root finding, whole-range stability, and factor selection real
obligations. The first feedback spike should therefore target one bounded
second-order continuous-time circuit, compare its solved transfer with the
existing `filterPair`, and refuse:

- nonlinear atoms inside the component;
- instantaneous algebraic loops without a certified inverse;
- coefficient ranges that can cross instability or a singular solve;
- arbitrary signal excitation that cannot be represented as a modal source;
  and
- general recursive audio feedback.

A sparse state-space or matrix-exponential carrier, such as the candidate in
[`lifted-modal-totality-spike.md`](lifted-modal-totality-spike.md), may
eventually avoid live polynomial root finding. It is not a prerequisite for
the feed-forward topology phase and must earn its own JIT/Metal cost witness.

## 10. Landing plan

The phase should land as a sequence of independently gated PRs. Later PRs must
not be started by copying the current Phaser special case into a new enum.

### PR 1: kernel semantics and factor-preserving prototype

Add a new semantic module, tentatively
`Tropical.Semantics.ModalKernel`, and a production representation,
tentatively `Tropical.EmitArrow.Modal.Kernel`.

Deliverables:

- the `identity`, `proper`, `scale`, `parallel`, and `cascade` denotations;
- explicit separation of Dirac feedthrough from `Bank.atZero`;
- authored-order laws and whole-universe control freezing;
- a generic interpreter into `Oriented.Bank` for bounded oracle cases;
- a factor-preserving six-all-pass prototype whose expression size grows
  linearly with section count; and
- an independent rational-transfer differential.

This PR has no product UI and no trunk/backend change.

### PR 2: structural terminal selection

Replace effect-name selection with normalized-kernel selection for the served
linear shapes.

Deliverables:

- generalize `decorateDegreeZeroCausalPhaser` around “causal degree-zero
  identity-plus-tail cascade” rather than Phaser;
- select the compact one-stage and two-room product terminals from
  `KernelExpr` structure;
- retain generic `Bank` lowering as the f64 reference;
- add a pre-emission cost witness; and
- keep emitted Plan 6, LLVM, WASM, and MSL vocabularies unchanged.

The old `.phaser` stage may coexist as a compatibility producer during this
PR, but it must elaborate to the generic kernel expression.

### PR 3: served atoms and library-graph prototype

Add the minimum registered atom floor and replace the privileged Phaser stage
producer with a programmatically constructed flat `PatchGraph` fixture. This
proves the atom graph before the persistence format depends on it.

Deliverables:

- typed atom descriptors and lowering actions;
- direct, tail, scale, and coefficient routing sufficient to express one
  all-pass section;
- programmatic `Allpass1` and six-section Phaser graph factories;
- the served legacy Phaser kind expands through that graph factory; and
- no production semantic match on the string `"phaser"` or
  `ModalStage.phaser` remains after the compatibility window.

The existing filter must also lower through `KernelExpr.proper` in this PR.

### PR 4: hierarchical Lean elaboration, library definitions, and document v3

Add definitions, typed boundary ports, hygienic expansion, source maps, and v2
migration. Keep `tropical_program_2` unchanged.

Deliverables:

- a closed v3 schema and round-trip fixtures;
- versioned `Allpass1` and six-section `Phaser` library definitions replacing
  the programmatic factories as the product source of truth;
- legacy Phaser documents migrate to an instance of the shipped definition;
- definition-reference and inner-graph cycle refusal;
- stable expansion ids and one-slot parameter forwarding;
- expanded/collapsed plan equivalence; and
- realized diagnostics mapped back to the nested authored path.

### PR 5: Reversible hierarchy and filter view

Move the canvas to dynamic vocabulary descriptors, add expand/collapse and
breadcrumbs, and render the first response/pole view.

Deliverables:

- compact default Phaser UI;
- editable six-section view;
- editable all-pass internal view;
- save/reload without changing ids, order, or definitions;
- detach-and-edit behavior for shipped definitions; and
- visible cost/admission feedback before a compile is attempted.

### Optional PR 6: bounded linear-feedback spike

This PR is not required to declare the feed-forward topology phase landed. It
must remain a spike until one second-order feedback circuit solves to the same
filter response and meets an explicit cost/stability contract.

## 11. Cost and admission contract

`KernelExpr` needs a structural cost result before it can become a served
authoring feature. At minimum record:

- source modal rows;
- proper kernel rows by factor;
- direct-path count;
- generic expansion bound;
- selected terminal realization;
- routed item and output counts;
- non-coefficient array floats;
- coefficient array slots;
- estimated Metal threadgroup scratch;
- emitted expression/plan node bound; and
- whether live parameter ranges preserve each specialization's admission.

The estimator must run on the factor graph. It must not expand a product in
order to discover that expansion is too large. A failure is a named compile
refusal attached to the authored module path. The compiler must not silently
drop sections, approximate a stable shape, switch wet to dry, or realize a
modal intermediate as `Sig` to escape the cost.

The first product gate remains 24,576 bytes of threadgroup scratch on the
current Metal policy. Because the baseline already uses 22,320 bytes, the
milestone target is equality with the landed compact Phaser, not merely “under
the cap.”

## 12. Qualification: what “landed” means

The phase is complete only when all of the following are true.

### Semantic qualification

- One-, two-, and six-section topology-derived all-pass cascades agree with an
  independent rational evaluator over mix endpoints, cancellation
  neighborhoods, and the audible frequency grid.
- Wet all-pass magnitude remains unity within the existing lens.
- The topology-derived Phaser preserves the whole-universe law for live
  center, sweep, rate, and mix.
- Modal fan-in preserves authored source order.
- Gauge and bloom are not crossed by a linear rewrite without a named law and
  admission predicate.
- The resonant low-pass still passes attenuation, ping-frequency, and live
  cutoff gates through the generic kernel path.

### Structural qualification

- Expanding or collapsing a definition does not change the elaborated graph.
- The six-section expression has linear, not exponential, retained size.
- The factory Phaser and an independently authored equivalent graph select the
  same terminal realization.
- The backend plan contains ordinary routed/bank regions and no Phaser or
  hierarchy opcode.
- Untouched v2 documents migrate deterministically and retain connection order.
- Signal-to-modal edges are rejected at the typed boundary in every view.

### Performance qualification

- The canonical `6 -> 32 -> six sections -> 32` plan publishes no more than
  the current 22,688 bytes of Metal scratch, subject to an explicitly reviewed
  baseline change rather than tolerance creep.
- Largest routed image, coefficient columns, array slots, MSL size, first load,
  and warmed block percentiles are reported beside the current Phaser row.
- Topology elaboration and cost analysis remain bounded without invoking the
  exponentially expanded reference.
- JIT, WASM, and Metal retain their current numerical contracts.

### Product qualification

- Reversible can display Phaser as one ordinary module.
- A user can open it, inspect all six sections, change section topology, return
  to the parent, save, reload, and hear the changed graph.
- A user can open one all-pass section to its direct and tail branches.
- Connection menus offer only type-legal sources; modal main inlets never list
  signal nodes.
- Ordinary multi-connections remain legible without requiring a visible Mix or
  Modal-union node.
- Errors and cost refusals point to the nested authored module and node.

The standing `make validate` suite remains mandatory. New gates should live
beside the current Phaser, oriented-algebra, modal-filter, vocabulary,
malformed-document, Reversible surface, compile-handshake, and Metal-vs-JIT
tests rather than in an isolated prototype runner.

## 13. Risks and explicit decisions still needed

### Direct feedthrough

The compiler must decide on one representation of `d delta + h`. Reusing
`Bank.atZero` would be mathematically wrong. The recommendation is an explicit
direct scalar on `KernelExpr`, eliminated only when the kernel is applied.

### Numerical order

Analytic convolution identities do not imply bit-identical floating
evaluation. Normalization may remove identities and singletons but should not
freely commute or reassociate parallel terms. Each optimized terminal needs an
observable differential against an independent oracle and the current generic
route.

### Definition ownership

Shipped definitions need stable ids, versions, and migration policy. Editing a
library instance must choose explicitly between overriding parameters and
detaching a definition. Silent mutation of a shared definition is unacceptable.

### Scalar atom scope

Building the complete Phaser control network from arithmetic atoms will grow
the initial vocabulary. Those atoms should be chosen from concrete definition
needs, not from an attempt to recreate a general source language. Each remains
an ordinary registered closed-form node.

### Compilation latency

The full generic all-pass tree is not merely a runtime problem. Exact constant
folding over duplicated `Sig` trees became a multi-minute test during the
Phaser qualification and had to be replaced with one evaluation of the
interned DAG plus a bounded two-section generic witness. The production
elaborator, cost model, and tests must all operate on shared DAGs and must never
use full expansion as a routine oracle.

### Feedback carrier

A solved linear circuit needs a representation for live high-order systems.
Explicit roots, sparse state space, and structured matrix-exponential action
have different stability and backend costs. No choice is implicit in the
feed-forward phase.

## 14. Non-goals for this phase

- arbitrary cycles in `PatchGraph`, `ResolvedProgram`, `ENode`, or Plan 6;
- a delay/register/state primitive;
- signal-to-modal conversion;
- nonlinear feedback or arbitrary external-signal recursive filtering;
- a new Phaser, modal-kernel, or hierarchy backend opcode;
- automatic approximation when a factor graph exceeds admission;
- fully general formula/binder definitions over the wire;
- a sequence editor implementation; or
- exposing every residue, route, and backend array in the ordinary patch view.

## 15. What this opens afterward

Once topology can elaborate to a factored kernel, several effects stop needing
privileged nodes:

- static and swept notch/EQ networks from direct and proper branches;
- resonant low-pass, band-pass, and high-pass variants from shared pole-pair
  definitions;
- arbitrary all-pass and dispersive cascades;
- room networks composed in series and parallel, subject to current carrier
  admission;
- modal differentiation/integration and affine reclocking from the existing
  residue operations; and
- bounded polynomial modal distortion, whose products remain finite sums of
  exponential-polynomial atoms but carry an explicit multiplicative mode-count
  cost.

Non-polynomial saturation does not in general remain a finite modal value, and
polynomial distortion can grow quadratically or worse. Those facts become cost
and representation questions in the same framework rather than reasons to add
an opaque `Distortion` effect prematurely.

The larger trajectory is a single hierarchical design environment with
different views over one typed graph. The immediate proof is deliberately
smaller: make one shipped effect cease to be semantically privileged, preserve
its exact compact runtime shape, and let a user descend through its construction
without leaving the patching language.
