# Static routed-reduction backend design

Date: 2026-08-08 (America/Los_Angeles)

Status: accepted architecture and implementation record.

The routed-sum denotation, compact Plan 6 delimiters, scalar LLVM reference,
cooperative Metal execution model, and first modal Cauchy migration are part of
the production implementation. The exact fully factored ordinary two-room
image remains the next modal specialization described by section 10; existing
scalar Bank formulas remain its independent oracle until that specialization
is qualified over the complete live pole box.

## 1. Decision

Implement a backend-neutral, static-shaped **routed sum** in Tropical's trunk
IR and Plan layer. Use it as the common representation for:

- ordinary single-output bank sums;
- row-wise and segmented Cauchy transforms;
- one-to-many symmetric pair updates;
- gauge energy sums;
- divided-difference/confluent correction sums; and
- final modal carrier and strike reductions.

The semantic meaning is a fixed, authored-order scalar fold. LLVM and WASM
lower that meaning directly. Metal may realize the independent map phase over
a SIMD group and then perform a deterministic per-output gather in authored
order. The backend schedule may change; the routed-sum denotation may not.

The first production consumer is the modal Cauchy fold, where one routed map
feeds the real and imaginary output folds. The exact factored ordinary-room
terminal is the next specialization, followed by broader `bankSum` migration.
The abstraction is therefore exercised by production modal code while its
larger consumer retains an independent scalar oracle during qualification.

This strategy explicitly rejects these initial implementations:

- an opaque room-specific MSL source splice;
- continuing the nested `DirectTerminal` expression as the production path;
- FMM, HSS/H², butterfly, or NUFFT machinery at 32 room modes;
- polynomial `H = N/P` evaluation near live poles;
- float atomics or an unspecified GPU reduction order; and
- a universal GPU/parallel language including scans, recurrence, FFT stages,
  or arbitrary dynamic dispatch.

The bounded abstraction is:

> Static-capacity pure maps over a fixed item domain, followed by statically
> routed additive reductions into a fixed output image, with an optional live
> effective count and a defined contribution order.

That is the concrete pattern present in the current modal compiler. It is not
a claim that every future DSP algorithm is map/reduce.

## 2. Why sample parallelism is insufficient

The current Metal ABI maps one Metal thread to one sample. At a 512-frame
render quantum on a 32-lane Apple SIMD group, the grid contains only 512
threads, or 16 SIMD groups:

```text
group 0:  sample 0  ... sample 31
group 1:  sample 32 ... sample 63
...
group 15: sample 480 ... sample 511
```

Every lane then executes every nested modal loop serially. The independent
work actually has multiple axes:

```text
sample x interaction-family x target-mode x source-mode
```

The current IR exposes only `sample`. A 32x32 room interaction therefore
appears to Metal as a long private dependency chain inside one lane, not as
1,024 independent pair locations.

SIMD is not an additional device layered on top of current Metal threads; the
threads are already executed as SIMD groups. The change is to remap the lanes.
The cooperative kernel will use one threadgroup per sample and one initial
SIMD group per threadgroup:

```text
threadgroup_position_in_grid.x = sample
thread_index_in_threadgroup    = map lane
```

At 512 frames this produces 512 independently schedulable SIMD groups rather
than 16. Within a routed region, lanes stride over pair items, write mapped
contributions to bounded threadgroup scratch, synchronize, and own output rows
for deterministic gathers.

This does not reduce the operation count. The modal factorization does that.
The routed IR makes the remaining independent operations legally schedulable.

## 3. Starting repository facts

The implementation must respect these current facts rather than inventing a
parallel subsystem beside them:

1. `Sig` is the authoring tree and `ENode` is the hash-consed trunk IR.
2. The trunk is monomorphic, acyclic, closed-form, and state-free.
3. `bankSum` is the existing exception that keeps backend-relevant structure
   as data.
4. `ENode.bankSum` lowers to `ReduceBegin`, a scalar body, an `Add`, and
   `ReduceEnd`.
5. Its denotation is an increasing-index scalar left fold. It requires no
   associativity premise.
6. `REDUCE_REGION_EXECUTES_IN_ARRAY_ORDER` is a named trust obligation.
7. `Stage0` treats a delimiter-matched reduction as a placement unit; live
   sample-dependent regions remain in the audio kernel.
8. `EmitLlvm` wraps the whole graph in a serial sample loop.
9. `EmitMsl` emits one thread per sample and serial `for` loops for both
   elementwise operations and reductions.
10. Metal currently has one pipeline, one dispatch, an output buffer, a slot
    snapshot, and an optional packed coefficient-column buffer.
11. The live Metal worker prepares exact-epoch tiles off the audio callback;
    the callback never submits or waits for GPU work.
12. Metal sessions currently dual-load the LLVM JIT for scopes and f64
    reference rendering. JIT audio throughput is not the product gate, but
    JIT compilation remains on the load path until explicitly decoupled.

Relevant implementation seams:

- `lean/Tropical/EmitArrow/Sig.lean`
- `lean/Tropical/Ir/Nodes.lean`
- `lean/Tropical/Semantics/Sig.lean`
- `lean/Tropical/Semantics/LowerSig.lean`
- `lean/Tropical/Ir/Emit.lean`
- `lean/Tropical/Plan.lean`
- `lean/Tropical/Ir/Stage0.lean`
- `lean/Tropical/Ir/EmitLlvm.lean`
- `lean/Tropical/Ir/EmitMsl.lean`
- `lean/Tropical/Engine/Compile.lean`
- `engine/runtime/NumericProgramParser.hpp`
- `engine/metal/MetalKernel.hpp`
- `engine/metal/MetalKernel.mm`
- `engine/metal/MetalRenderWorker.*`
- `engine/runtime/FlatRuntime.*`
- `lean/Tropical/Trust.lean`

Current branch state at the time of this note:

- `feat/room-universe-law` is 23 commits ahead of its remote.
- `lean/Tropical/EmitArrow/Modal/OrientedRealize.lean` contains an uncommitted
  nested `DirectTerminal` experiment.
- `lean/ffi/shim.o` is a dirty generated build artifact and must not land.
- The DirectTerminal experiment is useful as a semantic/diagnostic oracle,
  not as the intended production lowering.

Before backend edits begin, preserve the DirectTerminal diff outside landing
history or checkpoint only the portions intentionally retained as a reference.
Do not mix the generated object into any commit.

### 3.1 Current diagnostic evidence

The uncommitted DirectTerminal diagnostic is a useful negative baseline:

- production-shape plan footprint: 47,601 by the current footprint metric;
- emitted body: approximately 11,162 IR instructions and 48 serial reduction
  loops;
- 25 array slots, with approximately 2,930 private floats per Metal thread
  (11,720 bytes before compiler allocation/spill effects);
- emitted MSL: approximately 1,010,313 characters and 25,092 lines; and
- logical terminal work: approximately 20,160 complex quotient sites per
  sample because landing, rendering, and strike calculations repeat related
  transforms.

A one-block local Metal invocation took roughly 26 seconds end to end. A
100-block invocation did not complete within 145 seconds and was terminated.
Those walls include generation, dual JIT compilation, MSL compilation/load,
and command processing, so they are not kernel-only performance numbers. They
are sufficient to reject the current candidate as a shipped artifact, not to
estimate the cooperative kernel's eventual block time.

The retained baseline for ordinary banked work on the canonical M1 Pro is much
healthier: at B=512, K=512, synchronous Metal measured 1.3862 ms per block,
while the fixed renderer measured 2.948 ms. The product deadline is 11.610 ms
and the standing half-deadline is 5.805 ms. These measurements demonstrate
that Metal is viable for large resident modal work while leaving the new
factored kernel responsible for proving its own shape.

## 4. Semantic invariants

The new backend machinery may not modify these contracts:

### 4.1 Closed-form time

Every kernel remains a pure `f(tau, params)`. A routed sum owns no persistent
state and observes no previous control value. Hold, seek, reverse, and repeated
random-access rendering remain ordinary evaluation cases.

### 4.2 Whole-universe freezing

All live room and source controls are evaluated at the terminal observation
coordinate and used consistently throughout the complete deferred modal graph:

```text
renderLive(G, p, t) = renderStatic(G, freeze(p, t), t)
```

Neither publication-time tables nor a cooperative Metal schedule may freeze a
live s1 value for a whole tile.

### 4.3 Direction

Direction belongs to each room kernel independently. For room `r`:

```text
K_r(s) = (1 - d_r) H(s + sigma_r) + d_r H(sigma_r - s)
```

The public control may remain a switch or the incumbent linear value. The
backend must not reinterpret direction as complete-output reversal.

### 4.4 Gauge

Gauge is semantically `Modal -> Modal`, but its scale is nonlinear in the
complete direction-weighted bank. It is evaluated after the preceding linear
room run has been assembled. A compiler may not distribute gauge through
direction branches or room factors.

### 4.5 Order and numerical meaning

For a routed output, contributions are added in lexicographic order:

```text
(item index ascending, emit position ascending)
```

That order is part of the initial semantics. A Metal map may be parallel, but
the gather for each result follows this order. There is no float reassociation
or atomic accumulation in the first implementation.

### 4.6 Fixed topology

Capacities, route tables, room frequencies, pair schedules, and singularity
lanes are structural. Live values may select safe values or continuous formula
arms, but may not change item counts, result counts, routes, scratch size, or
dispatch shape.

### 4.7 Runtime safety

The Metal worker remains the sole submission/wait owner. No audio-callback
allocation, lock, log, compilation, slot packing, coefficient rebuild, or GPU
submission is introduced.

## 5. The routed-sum abstraction

### 5.1 Proposed authoring/trunk shape

Add a static routed additive region. Names may be adjusted during review, but
the fields and denotation should remain explicit:

```text
routedSum
  capacity       : Nat
  outputCount    : Nat
  routes         : Array (Option Nat)
  tables         : Array Expr
  values         : Array Expr
  dynCount?      : Option Expr
  idxId          : Nat
  resultType     : ScalarKind
```

`values.size` is the fixed fanout. `routes.size` must equal
`capacity * values.size`. Entry `(item, emit)` either names a destination in
`[0, outputCount)` or is `none`.

The node returns an array of `outputCount` scalar results. `index` selects a
result for ordinary downstream scalar algebra.

The optional effective count follows the current `bankSum` rule:

```text
trips = clamp(toInt(dynCount), 0, capacity)
```

No body value is evaluated for `item >= trips`.

### 5.2 Denotation

For `fanout = values.size`:

```text
acc = [0, ..., 0]                         -- outputCount entries
for item in 0 .. trips-1:
    bind idxId = item
    mapped = evaluate every values[e]
    for emit in 0 .. fanout-1:
        if routes[item * fanout + emit] = some output:
            acc[output] = acc[output] + mapped[emit]
return acc
```

This definition supports:

- `bankSum`: every item routes one value to output zero;
- row reduction: item `(row, col)` routes to `row`;
- symmetric pair reuse: one pair computes one inverse, then multiple values
  route to both endpoints;
- complex images: concatenate real and imaginary output ranges;
- independent families: concatenate family ranges; and
- masks: route structurally possible work and make the mapped value safely
  zero when the current numeric lane is inactive.

### 5.3 Why routes are structural

Static routes avoid float atomics and dynamic scatter. They let the compiler
precompute a compressed gather schedule for every output:

```text
routeOffsets      : outputCount + 1
routeRecordIndices: number of non-none routes
```

Within each output's CSR slice, record indices are already sorted by
`(item, emit)`, so the Metal gather is definitionally the authored fold.

The fixed-topology room contract already provides these routes. A future
operator needing live destination keys is outside this abstraction and must
justify a separate design.

### 5.4 Validation

Construction or lowering fails before publication when:

- `capacity`, `outputCount`, or fanout is zero where a nonempty region is
  required;
- `routes.size != capacity * fanout`;
- a route target is outside `outputCount`;
- `idxId` collides with an open ancestor binder;
- `dynCount` is array-valued or not coercible to integer;
- body values disagree on scalar type;
- the region body has a side effect, slot write, array mutation, or nested
  routed region;
- a table is written inside the region;
- scratch arithmetic overflows the declared capacity type; or
- a backend cannot honor the authored-order contract.

Nested routed regions are not needed for the factored modal schedule and are
rejected initially. Sequential routed phases are supported. Legacy nested
`bankSum` remains legal through its scalar lowering until explicitly migrated.

## 6. Type and storage work

The existing `CompileResult.array` records an element scalar type, but
`Pack`/`Index` and MSL array declarations currently behave as float storage.
Routed results make the latent distinction concrete.

Add `arraySlotTypes : Array ScalarType` parallel to `arraySlotSizes` inside
the typed Plan. Existing slots default to `.float`, and the serialized field
is omitted when every slot is float so existing plan goldens remain stable.

Required behavior:

- `allocArraySlot` accepts a size and element type.
- `Pack` retains its current float behavior unless all operands establish a
  different supported homogeneous type.
- `Index`, `SetElement`, and elementwise operations read the slot's declared
  type rather than assuming float.
- LLVM continues using the existing i64 backing cells, encoding float as f64
  bits, int as i64, and bool as the documented zero/one representation.
- MSL declares `float`, `long`, or `bool` locals/threadgroup arrays as needed.
- coefficient columns remain float-only; Stage0 rejects a non-float slot as a
  coefficient-column candidate until that host ABI is deliberately expanded.
- C++ remains a storage/manifest host and does not regain instruction codegen.

If typed general arrays prove broader than needed in the first commit, routed
results may initially be restricted to float, but this restriction must be
explicit in the constructor and refusal gates. It must not be hidden behind a
nominally generic type field. Complete migration of integer `bankSum` then
waits for typed array support; the room, Cauchy, and gauge clients do not.

## 7. Plan representation

Lower `routedSum` to a structured delimiter region rather than unrolling its
capacity:

```text
RoutedSumBegin
  dst array slot
  capacity / effective-count operand
  output count
  binder id
  structural routes or route-table id

  ... mapped body instructions ...

RoutedSumYield
  args = the fixed fanout values

RoutedSumEnd
```

Add explicit fields to `NInstr` for the region metadata instead of overloading
`strides` with unrelated meanings. Only `RoutedSumBegin` carries the route
table. Omit new wire keys on all other instructions.

The emitter state records region nesting and restores temp memoization at the
end just as `ReduceBegin` does. Body-local temps do not escape. The returned
array slot does escape.

The Plan-level structural theorem should state that compiling a well-formed
`ENode.routedSum` emits exactly:

```text
loop-invariant tables/count
++ RoutedSumBegin
++ body
++ RoutedSumYield
++ RoutedSumEnd
```

with no capacity-sized body duplication.

### 7.1 Wire/schema handling

The engine currently parses the instruction stream even though it no longer
uses it for code generation. Routed regions add new executable vocabulary, so
the production representation bumps to `tropical_plan_6`; do not reinterpret
Plan 5 in place.

The compiler, runtime parser, compatibility matrix, negative schema gates,
precompiled web artifacts, and fixture manifests move together. The runtime
continues to fail closed on Plan 5 at the runnable load boundary after the
cutover, consistent with the repository's current refusal policy for older
runnable schemas. There is no instruction translation layer and no C++
code-generation fallback.

The new tags are recognized by the C++ side only so the manifest reader can
validate current artifacts. Their extra fields are parsed as structural
metadata where the Metal host needs them and otherwise discarded after
validation.

Keeping the typed parallel structure only in an internal emission sidecar is
acceptable for the standalone R3 microbenchmark, but it is not the production
end state and does not satisfy R6.

## 8. Scalar LLVM/WASM lowering

LLVM is the executable definition and f64 oracle, not the performance target.

For each routed region:

1. Zero the fixed output array in output-index order.
2. Resolve and clamp the effective count once before the loop.
3. Iterate items in ascending order.
4. Evaluate the mapped body once for the item.
5. Apply yields in emit-position order to their structural destinations.
6. Close the loop and expose the output array.

This exactly implements the denotation and gives WASM the same behavior
through the existing LLVM path.

The scalar lowering must cache common body subexpressions within an item. In
particular, one complex inverse feeding several route values is emitted once,
not re-spelled per destination.

JIT audio throughput is not an acceptance gate. The gates are:

- compilation terminates within the existing cold-load envelope;
- the reference can render bounded observation windows;
- native JIT and WASM preserve the routed-sum denotation; and
- the emitted code remains compact in capacities.

If LLVM O2 remains a load-time problem on a Metal session, use the existing
unoptimized compile capability for the reference artifact or split audio
Metal publication from lazy observation/reference compilation. Do not distort
the Metal algorithm to satisfy CPU realtime throughput.

## 9. Cooperative Metal execution model

### 9.1 Two ABI modes

Preserve the existing byte-frozen one-thread-per-sample kernel for plans with
no routed regions.

Plans containing routed regions use a cooperative header:

```metal
kernel void tropical_kernel(
    ... existing buffers ...,
    uint sample [[threadgroup_position_in_grid]],
    uint lane   [[thread_index_in_threadgroup]],
    uint3 groupSize [[threads_per_threadgroup]])
```

`groupSize.x` is the lane stride. Every lane in a threadgroup has the same
`sample`, so the bounds return is uniform. The host dispatches `frames`
threadgroups, initially with one pipeline execution width per group.

The source must not hard-code 32 as semantic truth. `MetalKernel::create`
queries `pso.threadExecutionWidth`, verifies it against device limits, records
it in the kernel object, and uses it for dispatch. The M1 Pro qualification is
expected to report 32.

### 9.2 Scalar phases and captures

Simply running the entire scalar graph in all lanes would duplicate every
oscillator, envelope, and control calculation by the group width. Instead,
cooperative MSL partitions the instruction stream at routed-region boundaries:

```text
lane 0: execute scalar phase
lane 0: publish region captures / captured arrays to threadgroup storage
all lanes: barrier
all lanes: execute routed map and gather
all lanes: barrier
lane 0: copy routed outputs to ordinary per-sample values
lane 0: continue scalar phase
```

Before emission, run a use/definition analysis for each routed body:

- scalar temps or slots defined/read outside and consumed inside are captures;
- arrays consumed inside are classified as constant coefficient, immutable
  structural literal, or per-sample capture;
- mapped body locals remain lane-local;
- routed outputs are copied back to the lane-0 scalar image unless a later
  routed phase consumes them directly;
- no module-slot write or effectful instruction may occur inside a region.

Per-sample captured arrays are filled by lane 0 and exposed through
threadgroup storage before the barrier. Constant coefficient columns remain
in `constant` address space and can be read directly by every lane.

This capture analysis is mandatory for production. Replicating the entire
scalar graph in all lanes is permitted only in a standalone microbenchmark.

### 9.3 Routed map and gather

For one region:

```text
for item = lane; item < trips; item += lanes:
    evaluate body(item)
    write each yielded value to record[item, emit]

threadgroup_barrier

for output = lane; output < outputCount; output += lanes:
    acc = 0
    for record in CSRRecords(output):
        acc = acc + mappedRecord[record]
    result[output] = acc

threadgroup_barrier
```

The route CSR arrays are compile-time constants in MSL. No dynamic key buffer,
float atomic, or target-dependent contribution order is allowed.

### 9.4 Scratch allocation

Compute a static scratch requirement for every region:

```text
mapped records
+ routed outputs
+ captured scalars/arrays
+ alignment
```

Allocate one reusable threadgroup scratch arena sized to the maximum live
region, not the sum across sequential regions. Region outputs are copied out
or explicitly retained before the arena is reused.

The emitted artifact records the required bytes. `MetalKernel::create`
compares them with `device.maxThreadgroupMemoryLength` and fails publication
before audio activation if the requirement is unsupported. The product gate
should initially cap usage below 75 percent of the qualified device limit to
leave occupancy and compiler headroom.

The compiler also reports:

- mapped record count;
- active route count;
- capture bytes;
- result bytes;
- barriers;
- static group width policy; and
- largest region identity.

These are diagnostics, not client-selected parameters.

### 9.5 Host/runtime changes

Add an optional Metal execution description to the emitted artifact/manifest:

```text
kind: sample_threads | sample_threadgroups
threadgroup_scratch_bytes
requested_group_policy: pipeline_width
```

`MetalKernel` stores the dispatch kind and actual execution width. In
`render_tile_impl`:

- `sample_threads` retains the exact current `dispatchThreads` call;
- `sample_threadgroups` uses `dispatchThreadgroups(frames, 1, 1)` with the
  validated pipeline width;
- both paths use the same buffers, command queue, completion handling, and
  destination copy;
- one tile remains one command buffer and one kernel dispatch initially.

The epoch queue, activation tags, render worker, callback copying, and failure
latching do not change semantically. Existing lifecycle and provenance gates
must remain green.

## 10. Exact modal factorization

### 10.1 Transfer form

For structural room frequencies `omega_j` and residues `b_j`, define:

```text
H(z) = sum_j b_j / (z - i*omega_j)
```

For room `r` with live damping `s_r` and local direction `d_r`:

```text
K_r(s) = (1 - d_r) H(s + s_r) + d_r H(s_r - s)
```

For source transfer `S(s)`, the two-room response is:

```text
Y(s) = S(s) K_1(s) K_2(s)
```

This representation preserves independent room direction without explicitly
materializing four complete Cartesian banks.

### 10.2 Publication-time data

For the fixed ordinary room, publish:

- ordered `omega_j` and complex `b_j`;
- oscillator increments and fixed phase data;
- every off-diagonal `omega_j - omega_k` and its square;
- every physical `omega_j + omega_k` and its square;
- unordered difference pairs `j < k` (496 at M=32);
- unordered physical pairs `j <= k` (528 at M=32);
- diagonal/coincidence adjacency;
- routed-sum route tables and CSR gathers;
- structural source/room admissible pole boxes; and
- fixed capacities for source modes, room modes, branches, and room stages.

These values belong in emitted constants or immutable coefficient columns.
They are not re-created per sample.

### 10.3 Live image

For source poles `lambda_i` and residues `a_i`, evaluate and cache:

```text
u[r,i,j] = 1 / (lambda_i - p[r,j])
v[r,i,j] = 1 / (q[r,j] - lambda_i)

D[r,i] = sum_j b_j * u[r,i,j]
P[r,i] = sum_j b_j * v[r,i,j]
E[r,j] = -sum_i a_i * u[r,i,j]
M[r,j] =  sum_i a_i * v[r,i,j]
```

One `u` inverse updates both `D` and `E`; one `v` inverse updates both `P`
and `M`. For N=6, M=32, two rooms, this is 768 unique inverses.

For the room-room interaction, with:

```text
delta = s_2 - s_1
sum   = s_1 + s_2
```

evaluate symmetry-reduced vectors:

```text
A[j] = sum_k b_k / ( delta + i*(omega_j - omega_k))
C[j] = sum_k b_k / (-delta + i*(omega_j - omega_k))
B[j] = sum_k b_k / ( sum   - i*(omega_j + omega_k))
```

Off-diagonal difference pairs use 496 inverses and update both endpoints.
Physical pairs use 528 inverses and update both endpoints. Equal-frequency
difference diagonals do not divide; they go to the confluent lane.

Grand total for the live two-room image:

- 1,792 real reciprocal operations when each complex inverse uses
  `(x - i*y)/(x*x + y*y)`;
- 4,544 weighted Cauchy accumulator updates;
- 384 division-free source products for the diagonal confluent correction;
  and
- O(N+M) final residue assembly.

This replaces the DirectTerminal experiment's approximately 20,160 logical
complex quotient sites per sample across landing, rendering, and strike work.
The saving is realized only if the intermediate image is materialized once and
reused; re-spelling the formulas as independent `Sig` trees forfeits it.

### 10.4 Routed phases

Express the evaluator as sequential routed regions:

1. source-room inverse map and `D/P/E/M` routed sums;
2. off-diagonal room-difference inverse map and `A/C` routed sums;
3. room physical-sum inverse map and `B` routed sums;
4. diagonal DD/confluent correction sums;
5. O(N+M) coefficient assembly with direction applied here;
6. shared landing calculation;
7. shared oscillator/envelope carrier evaluation; and
8. final future/past/strike/gauge reductions.

Directions are scalar coefficients in phase 5. They do not duplicate phases
1 through 4.

### 10.5 Image layout

Use one compiler-owned layout allocator rather than hand-authored numeric
offsets. A named slice records element type, component count, logical extent,
and offset. For the N=6, M=32 two-room image, the principal slices are:

| Slice family | Logical extent | Components | Purpose |
|---|---:|---:|---|
| `D[r]`, `P[r]` | N per room | complex | room transfer at source poles |
| `E[r]`, `M[r]` | M per room | complex | source transfer at room poles |
| `A`, `C`, `B` | M each | complex | room-room difference/physical transforms |
| `TF`, `TP` | M each | complex | diagonal confluent source corrections |
| future simple residues | N + 2M | complex | source and both room pole rows |
| past simple residues | 2M | complex | mirrored room pole rows |
| future/past DD rows | M each | complex | equal-frequency degree-one atoms |

The source-room region's output image is 304 floats before padding; the
room-room `A/C/B` image is 192 floats. These are group results, not 304- or
192-element collections of independent `Sig` trees.

Every slice offset is structural, checked for overflow, emitted once, and
available to both scalar and Metal lowering. Layout construction must reject
overlap and out-of-range routes. Tests address slices by name and extent,
never by duplicating magic offsets.

### 10.6 Divided differences and exact strike

The equal-frequency room diagonal is structurally hot and always uses the
confluent formula. It must never form `(S(p1) - S(p2))/delta` near
`delta = 0`; use products of cached inverses instead.

All 32 diagonal pairs share the same real `delta`, so each active orientation
uses one `cexpm1`/envelope scalar feeding 32 carrier rows rather than 32
independent transcendental bodies.

The strike may be obtained from the sum of future degree-zero residues because
degree-one DD atoms vanish at age zero. That identity can suffer cancellation
in f32. Implement both this compact route and the direct factored mixed seam in
the f64 oracle, measure the complete declared box, and retain the direct seam
if the residue sum loses the predeclared error budget. Do not restore a full
Cartesian strike calculation merely for convenience.

### 10.7 Source-room crossings

A live source pole can coincide with a room pole when both damping and
frequency meet. The existing runtime cannot safely discover such an equality
by evaluating an ordinary reciprocal and selecting afterward.

Before product exposure:

1. derive and test the source-room confluent formula;
2. use structural source/room pole intervals to mark every pair that can enter
   its conditioning lens;
3. emit both fixed-topology safe arms for those pairs;
4. replace the inactive denominator with a nonzero safe value before division;
5. use a backend-identical selector with a proved overlap/continuity gate; and
6. refuse publication if the declared live box contains a singular family for
   which no continuous lane exists.

This is an early algebra gate, not a late performance discovery.

### 10.8 Gauges and longer stage chains

Partition the deferred modal stage spine at nonlinear gauges:

```text
linear room run -> materialized factored bank image -> gauge -> next room run
```

Within a linear run, use product-of-transfer-factor residue evaluation rather
than expanding direction combinations. At a gauge, assemble the complete
current bank, compute one scale through routed norm sums, scale all atoms and
the exact strike image, then continue.

The first optimized product shape is the source plus two ordinary 32-mode
rooms already covered by the branch tests. The routed backend is not limited to
two rooms, but repeated coincident rooms can require higher-order confluent
atoms. Existing scalar Bank algebra remains the reference/fallback until a
bounded higher-order factored schedule is separately proven. No public chain
may silently choose a mathematically different shortcut.

### 10.9 Forests, bloom, and terminal binding

The sample axis is not permission to flatten modal branches. Batch branch
work as another structural domain while retaining each branch's authored
identity:

- source representation (`plain` or `bloomed`);
- strike anchor and clock/address path;
- ordered modal stages;
- direction/control bindings;
- admitted source-mode count; and
- terminal destination/order.

Branches with equal room topology may share immutable room tables and route
schedules, but their live images remain separate unless extensional equality
under arbitrary pointwise controls has already been proved. Bloom is crossed
once at the true terminal seam; a room does not lower a bloomed branch to
signal in order to use the routed backend.

The final branch contributions flow through the existing authored-order Term
bundle/multi-input binding seam. Lane 0 performs the final sink/port writes.
Observation projections and multiple sinks do not add another modal
realization, and enabling observation must not change the audio plan's routed
regions or selected sinks.

Qualification includes a mixed plain/bloomed forest with independent anchors
and at least two terminal inputs. Cost and capacity diagnostics identify the
exact branch, room stage, and routed region on refusal.

## 11. Stage0 and binding time

`RoutedSum` receives one stage signature from its tables, body, and effective
count. Its loop index is a binding-time identity exactly like `bankSum`.

Initial placement rules:

- an s1 routed region stays wholly in the audio plan;
- an s0 routed region may stay in the audio plan without changing semantics;
- no individual instruction may be hoisted out of a region;
- a routed region is hoisted only after `Stage0` can move the complete
  delimiter-matched region, its routed output image, and every downstream
  boundary value atomically;
- group-local/threadgroup storage never crosses the coefficient/audio kernel
  boundary; hoisted results use the existing generation-published slot/column
  mechanism; and
- numeric values never select whether a region is s0 or s1.

S1 correctness and Metal performance come first. S0 routed-region hoisting is
a transparent later optimization, not a prerequisite for accepting live
controls.

## 12. Proof and trust obligations

Add or revise explicit obligations:

1. `ROUTED_SUM_DENOTES_AUTHORED_FOLD`
   - the `Sig`/ENode denotation matches the lexicographic routed fold;
   - invalid routes/counts remain explicit refusals.
2. `LOWER_ROUTED_SUM_PRESERVES`
   - authoring lowering into the hash-consed arena preserves denotation.
3. `ROUTED_SUM_REGION_HAS_CANONICAL_SHAPE`
   - Plan emission has the exact begin/body/yield/end shape and remains compact
     in capacity.
4. `LLVM_ROUTED_SUM_EXECUTES_IN_AUTHORED_ORDER`
   - inspection plus executable equivalence against explicit folds.
5. `MSL_ROUTED_SUM_EXECUTES_IN_AUTHORED_ORDER`
   - parallel map plus CSR gather implements the same per-output order.
6. `COOPERATIVE_MSL_PHASES_ARE_UNIFORM`
   - every group barrier is reached by every lane; no data-dependent branch or
     early exit may surround a routed phase.
7. `METAL_DISPATCH_MATCHES_PLAN_EXECUTION_KIND`
   - host grid/group shape and scratch limit match the emitted kernel.
8. `FACTORED_ROOM_EQUALS_DIRECT_PRODUCT`
   - exact algebra before floating-point error; includes direction endpoints,
     mixed directions, DD limits, and strike identity.

Keep `REDUCE_REGION_EXECUTES_IN_ARRAY_ORDER` for legacy `bankSum` until its
production clients migrate. Do not weaken it merely because the new routed
path exists.

## 13. Implementation checkpoints and commits

Every checkpoint ends with focused tests and a coherent commit. This durable
record tracks the accepted architecture and the implementation status above.

### R0 - Preserve the current diagnostic and baseline

- Record branch/base, dirty paths, toolchain, current test counts, plan/MSL
  sizes, and existing Metal baseline.
- Preserve the DirectTerminal experiment as a local oracle patch or a clearly
  marked non-production checkpoint.
- Restore/exclude `lean/ffi/shim.o` from landing work.
- Add no product behavior.

Exit: reproducible pre-backend baseline and no accidental artifact delta.

### R1 - Routed-sum semantics

- Add `Sig` and `ENode` routed-sum representation.
- Add structural hash, children, cycle traversal, staging signature, and
  well-formedness cases.
- Define the reference denotation with authored routing order.
- Prove/law-test the single-output, multi-output, sparse-route, dynamic-count,
  empty, and refusal cases.
- Add a bank-sum equivalence theorem for the one-output route table.

Suggested commit:
`feat(ir): define static routed sums`

Exit: the new structure has a precise backend-independent meaning.

### R2 - Compact scalar Plan and LLVM lowering

- Add routed delimiter instructions and validation.
- Add result array typing needed by the selected scope.
- Emit compact regions from `Ir/Emit`.
- Implement scalar LLVM/WASM authored-order lowering.
- Make `Stage0` treat routed regions atomically and conservatively retain s1.
- Update the metadata-only C++ parser and compatibility matrix/schema.
- Gate plan/IR growth across capacities 4, 16, 32, 64.

Suggested commits:

1. `feat(plan): lower routed sums as compact regions`
2. `feat(llvm): execute routed sums in authored order`
3. `test(ir): prove routed sum fallback equivalence`

Exit: scalar reference behavior is exact and compact; no Metal scheduling yet.

### R3 - Cooperative MSL synthetic proof

- Add Plan region capture/liveness analysis.
- Emit cooperative MSL with lane-0 scalar phases.
- Add route CSR constants and reusable bounded threadgroup scratch.
- Extend the Metal artifact/manifest with execution kind and scratch bytes.
- Extend `MetalKernel` dispatch without changing worker/callback ownership.
- Test device-width discovery and explicit over-cap refusal.
- Compare at least two schedules in a standalone benchmark:
  - lane-per-output with serial item scans;
  - striped map plus routed gather.

Suggested commits:

1. `feat(msl): emit cooperative sample threadgroups`
2. `feat(metal): dispatch routed reduction kernels`
3. `test(metal): gate routed reduction lifecycle and order`

Exit: a synthetic routed Cauchy fixture is correct on real Metal and has a
measured advantage over the serial MSL form.

### R4 - Factor the ordinary-room terminal

- Move the exact factor schedule into a focused module, proposed:
  `lean/Tropical/EmitArrow/Modal/FactoredTerminal.lean`.
- Generate publication-time pair tables and routes.
- Build sequential routed phases for source-room, room-difference,
  room-physical, confluent, assembly, landing, carrier, and strike work.
- Apply direction only during coefficient assembly.
- Share carriers/envelopes across room rows.
- Keep the current scalar Bank/direct formulas as independent references.
- Add the source-room confluence lane before broadening the admitted live box.

Suggested commits:

1. `refactor(modal): describe factored terminal images`
2. `perf(modal): route shared room interactions`
3. `fix(modal): close factored pole coincidences`

Exit: the product fixture no longer emits the repeated DirectTerminal Cauchy
work and uses the generic routed IR.

### R5 - Migrate existing reduction clients

- Express a representative float `bankSum` through routed-sum and prove
  equality to the existing left fold.
- Migrate modal Cauchy sums, gauge norms, and final float carrier sums where
  they benefit.
- Keep fixed-point/int banks on the legacy path until typed routed outputs are
  complete; then migrate them with byte-exact gates.
- Remove any room-only backend branch made redundant by routed lowering.

Suggested commit:
`refactor(ir): realize modal banks through routed sums`

Exit: the abstraction has multiple production consumers and the Metal emitter
does not recognize a room-specific node.

### R6 - Backend and product qualification

- Run the complete semantic, structural, numeric, performance, lifecycle, and
  live-control matrix below.
- Remove or quarantine the rejected DirectTerminal production path.
- Update durable architecture/trust documentation only with settled facts.
- Audit the complete PR diff and every commit; do not land this plan,
  generated reports, captures, object files, or benchmark streams.

Suggested commits:

1. `test(modal): qualify routed room backends`
2. `docs(architecture): define cooperative routed reductions`

Exit: the implementation is reviewable, qualified, and ready for a PR against
`main`; the PR remains unmerged by this sprint.

### R7 - Optional Metal-primary JIT decoupling

Only after R6 shows the Metal kernel is viable:

- compile the Metal-session f64 reference unoptimized or lazily;
- keep observation/scope rendering available;
- do not make CPU audio throughput a Metal publication requirement;
- retain explicit failure if a requested observation reference cannot build;
  and
- measure publication/load latency separately from audio rendering.

This is a runtime/product cleanup, not part of the routed-sum proof. Keep it in
a separate commit or PR so backend correctness is reviewable without a policy
change to JIT lifecycle.

### 13.1 Effort and branch discipline

Expected focused engineering effort, excluding CI and the required hardware
soak:

| Work | Estimate |
|---|---:|
| R0-R1 semantics/proofs | 2-3 working days |
| R2 Plan/LLVM/Stage0/schema | 3-4 working days |
| R3 cooperative MSL/runtime | 4-6 working days |
| R4 exact modal integration | 4-6 working days |
| R5 migration/hardening | 2-3 working days |
| R6 qualification and PR audit | 3-5 working days |

The likely critical path is 3-4 calendar weeks for one lead with parallel
review/qualification help. This supersedes the earlier 1-2 week estimate for a
room-specific terminal because the chosen outcome now includes the reusable
IR, typed storage boundary, Plan schema, generic Metal schedule, and a second
production consumer.

Implementation is staged on `feat/room-universe-law` for review against
`main`. Keep semantic, backend, and modal-specialization changes separable so
the series can be split or restacked without rewriting the denotational core.
Branch publication and merge remain explicit integration decisions.

## 14. Test matrix

### 14.1 Semantic and proof tests

- routed single-output sum equals explicit left fold;
- sparse/multi-destination routes equal explicit per-output folds;
- one mapped expression shared by multiple yields is evaluated once in Plan;
- dynamic count clamps at negative/zero/interior/capacity/over-cap values;
- binder shadow/collision behavior fails exactly as specified;
- route order is stable with duplicate destinations;
- lowering tree and pointer-memoized lowering agree;
- `bankSum` adapter equivalence;
- Stage0 never tears a routed region; and
- plan size is flat in body complexity and linear only in structural routes.

### 14.2 LLVM/WASM tests

- f64 output byte-equals explicit authored folds;
- nested legacy banks around sequential routed phases remain legal;
- far Q32.32 coordinates remain exact;
- WASM equals JIT on synthetic routed fixtures;
- cold compilation remains below the standing six-second ceiling on the exact
  product graph; and
- reference rendering remains available in Metal sessions.

### 14.3 MSL source/ABI tests

- ordinary plans retain the existing header/goldens exactly;
- routed plans emit the cooperative header and no serial routed `for` nest;
- one uniform bounds exit per sample group;
- barriers occur only at verified uniform phase boundaries;
- route CSR records preserve lexicographic contribution order;
- lane 0 alone writes sinks and ordinary module slots;
- coefficient columns retain the current optional `buffer(3)` ABI;
- scratch byte accounting matches emitted declarations;
- unsupported scratch/group geometry refuses at create; and
- no room/profile identifier appears in generic MSL lowering.

### 14.4 Metal runtime tests

- old and cooperative dispatch kinds choose the correct host API;
- actual pipeline width is captured and used;
- frames map to exactly frames threadgroups in cooperative mode;
- terminal command errors latch closed;
- callback provenance guard still rejects direct submission;
- epoch tags, source/device coordinates, activation, and retargeting remain
  exact;
- hot-swap between dispatch kinds is safe; and
- zero-frame, over-cap, missing-buffer, and device-unavailable paths fail
  closed.

### 14.5 Modal correctness matrix

- source alone, one room, and two rooms;
- FF, FP, PF, and PP direction endpoints;
- independently changed room directions;
- interior direction values if retained publicly;
- constant and live RT60, sway, and rate;
- changed-history/restored-image equality;
- forward, reverse, hold, seek, and arbitrary render order;
- fractional/far anchors and address warps;
- equal room frequencies, equal damping, near-equal damping, and exact DD
  limits;
- live source-room coincidence neighborhoods;
- gauges before, between, and after rooms;
- plain, bloomed, and mixed/timed forest branches admitted by the semantic PR;
- `atZero` versus both independent strike formulas; and
- non-finite, capacity, and declared-pole-box refusals.

Use an independent f64 direct-product oracle. Do not use the factorized code as
its own reference.

### 14.6 Numerical gates

- retain the existing repository-wide Metal-vs-JIT gates;
- require greater than 100 dB SNR for the banked/product Metal path unless a
  stricter existing fixture already applies;
- record the ordinary non-cancelling region, DD lens, source-crossing lens,
  strike, and long-tail errors separately;
- choose every tolerance before optimizing against the corpus;
- compile MSL with `MTLMathModeSafe` initially;
- reject any path needing fast-math or unspecified reassociation to pass its
  performance gate; and
- require finite output over the complete admitted live box.

### 14.7 Performance gates on the canonical M1 Pro

Measure kernel-only GPU time and end-to-end tile time separately at render
quanta 128, 256, and 512.

Required product gate at B=512, 44.1 kHz:

- p99 worker tile render below the existing half-deadline, 5.805 ms;
- maximum below the full 11.610 ms deadline during the qualification row;
- zero starvation, underrun, overrun, dispatch, tag, activation, ownership,
  callback-provenance, and non-finite failures;
- no material RSS growth; and
- live slot/control writes cause no relower or graph publication.

Also record:

- median/p99/max command encode, GPU execution, wait, and copy time;
- SIMD width and occupancy-relevant pipeline properties;
- threadgroup scratch and largest live private allocation;
- MSL source bytes/lines and compile time;
- LLVM IR bytes and JIT compile time, reported but not used as an audio
  throughput gate;
- route record count and real reciprocal count; and
- serial sample-thread, cooperative lane-per-output, and cooperative routed
  map/gather variants on the same exact formulas.

The cooperative design advances only if the real product fixture, not merely a
synthetic arithmetic kernel, clears the gate with headroom.

## 15. Expected performance shape

For the two-room N=6, M=32 case, one sample group distributes approximately
1,792 real reciprocals across the available lanes, or roughly 56 reciprocal
sites per lane before imbalance and non-reciprocal work. The previous sample
thread executed approximately 20,160 logical complex quotient sites serially.

The expected wins come from four separate effects:

1. exact algebraic elimination of repeated landing/render/strike transforms;
2. reciprocal reuse across both endpoints of an interaction;
3. 32-lane execution of independent mapped items; and
4. removal of thousands of floats of private per-thread array state.

No single one of these is assumed sufficient. Profiling must attribute the
result. If mapped reciprocal work ceases to dominate, do not introduce an
approximation merely because one exists in the literature.

## 16. Known risks and planned responses

### Threadgroup scratch limits occupancy

Mitigation: CSR routes, structural keys, a reusable maximum-sized arena,
phase-by-phase liveness, and a 75-percent device-limit admission cap. Compare
lane-per-output with map-materialize/gather before fixing the schedule.

### Lane-0 scalar phases dominate

Mitigation: measure phase time; promote pure independent elementwise work to
maps where already expressible; share captured tables; do not replicate the
whole graph across lanes. A later scheduler may combine independent scalar
nodes, but it is not required to prove routed reduction.

### Barriers erase the arithmetic win

Mitigation: use sequential coarse routed phases, fuse adjacent compatible map
and gather phases when liveness proves it, and retain one command dispatch.
Reject a multi-dispatch design unless measurement shows the single-kernel
phase model cannot fit scratch or register constraints.

### Authored-order gathers remain serial per output

Mitigation: outputs are themselves distributed across lanes and each row is
only O(32) contributions in the current room. If profiling proves that gather
dominates, introduce an explicitly different pairwise numerical policy with a
new oracle/error contract; never silently change `bankSum` order.

### Generated MSL remains too large

Mitigation: keep pair topology in compact constant route tables, emit one map
body per region rather than per item, and gate source bytes/compile time. Do
not meta-unroll capacity-sized arithmetic.

### Source-room pole crossings produce `0/0`

Mitigation: make the source-room confluent lane an R4 prerequisite and use
structural interval admission. Safe-select denominators before division on
eager backends.

### Gauge prevents a global linear factorization

Mitigation: treat gauge as an explicit materialization boundary. Factor each
linear room run independently and reuse routed reductions on both sides.

### Longer repeated-room chains raise pole degree

Mitigation: keep a declared maximum stage capacity and the scalar Bank
reference. Add higher-order confluent images only with a proved bounded
schedule. Do not claim an arbitrary-length optimized fast path from the
two-room proof.

### JIT compilation still blocks Metal publication

Mitigation: first reduce the scalar reference expression through the same
factorization. If O2 is still the problem, use unoptimized/lazy reference
compilation in a separately reviewed Metal-primary lifecycle change.

### A general IR change destabilizes unrelated plans

Mitigation: ordinary plans preserve their old MSL header, dispatch, and
goldens. New fields are omitted when unused. Introduce routed regions beside
legacy reductions, prove both, then migrate clients incrementally.

## 17. Stop/replan conditions

Pause implementation for an architectural decision if any of these occurs:

- the routed denotation cannot express one-to-many reciprocal reuse without
  dynamic atomics or capacity-sized code unrolling;
- cooperative barriers cannot be placed uniformly in the fused stateless
  kernel;
- the exact product fixture requires more than the admitted threadgroup memory
  on the canonical device;
- Metal requires a different mathematical branch or capacity from LLVM/WASM;
- authored-order gathering misses the realtime gate and the user does not
  approve a new numerical reduction contract;
- source-room confluence cannot be made continuous over the declared live pole
  box;
- a gauge must be moved through a direction/room factor to meet cost;
- s1 controls would have to be frozen per tile rather than per observation;
- the audio callback would need to wait, allocate, or submit; or
- the production gain appears only after enabling unsafe math or weakening the
  existing semantic/error gates.

These are bounded gates on the chosen architecture, not permission to expose a
partial public room contract.

## 18. Literature-informed boundary

The implementation begins with exact batched direct evaluation because the
problem is many independent small transforms, not one large matrix.

Relevant primary references:

- Michon et al., GPU/CPU/FPGA modal synthesis comparison (2025):
  https://www.frontiersin.org/journals/signal-processing/articles/10.3389/frsip.2025.1522604/full
- Skare and Abel, GPU modal synthesis (DAFx 2019):
  https://www.dafx.de/paper-archive/2019/DAFx2019_paper_48.pdf
- Létourneau, Cecka, and Darve, Cauchy FMM:
  https://doi.org/10.1137/120891617
- Goude and Engblom, adaptive GPU FMM and direct crossover:
  https://link.springer.com/article/10.1007/s11227-012-0836-0
- Kobel and Sagraloff, fast approximate multipoint evaluation:
  https://arxiv.org/abs/1304.8069
- Jhurani and Mullowney, batched small dense GPU kernels:
  https://arxiv.org/abs/1304.7053
- Apple Metal threadgroup sizing:
  https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes
- Apple SIMD-group reductions:
  https://developer-rno.apple.com/videos/play/tech-talks/10858/

At M=32, FMM/HSS setup, hierarchy traversal, and approximation management are
not justified. Revisit hierarchical methods only if a future product uses
hundreds or thousands of modes per transform or many rooms share a resident
hierarchy.

If exact reciprocal throughput remains the measured bottleneck after routed
Metal execution, the only near-term approximation worth a bounded research
spike is a certified fixed-degree approximation in the shared damping scalar
for structurally far pairs, with exact near/confluent lanes. It is not part of
this implementation strategy and may not be introduced without a uniform
error proof over the complete admitted box.

## 19. Definition of done

The backend work is complete when all of the following are true:

1. Tropical has one backend-neutral static routed-sum representation with a
   precise authored-order denotation.
2. LLVM/WASM implement its scalar reference semantics compactly.
3. Metal implements it through cooperative sample threadgroups with bounded
   scratch, explicit captures, no atomics, and no room-specific emitter case.
4. Existing ordinary kernels retain their exact dispatch/header behavior.
5. The two-room ordinary terminal uses the 1,792-reciprocal factored image,
   including DD and source-crossing safety, rather than the repeated nested
   DirectTerminal work.
6. Direction remains independent per room and gauge remains a complete-bank
   operator.
7. The real product fixture clears correctness, p99/deadline, lifecycle,
   capacity, compile-size, and live-control gates on the canonical M1 Pro.
8. The current semantic PR/diff contains no generated object, benchmark
   output, capture, qualification stream, or local diagnostic material.
9. The completed work is pushed to a reviewed PR against `main` and is not
   merged by the implementation agent.
