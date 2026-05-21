# First-Class Array Support

Zero-cost (at the type level) arrays with static shapes, numpy-style semantics, and automatic lowering. One unified array type covers everything from stereo pairs to delay buffers to mixing matrices.

## Design Principles

- **One array type.** `array<T, shape>` with statically-known dimensions. No separate "runtime array" vs "unrolled array" — the compiler chooses a lowering strategy, the user never thinks about it.
- **Runtime arrays first.** All arrays lower to the existing `array_storage` / `array_ptrs` C++ mechanism initially. Unrolling to scalar registers is a future optimization, not a prerequisite.
- **Static shapes, any access pattern.** A `float[4,4]` mixing matrix (all constant indices) and a `float[4096]` delay buffer (dynamic indices) are the same type. The compiler can optimize the former more aggressively, but both work through the same path.
- **Numpy-style semantics.** Element-wise arithmetic, broadcasting, matmul, reductions. Arrays are mutable (register arrays persist across samples, output arrays are computed fresh each sample).
- **Composable with existing types.** `array<MyStruct, [4]>` works — each struct is laid out as N scalars, the array is N * shape_size scalars total.

## Type System

```typescript
// addition to PortType
| { tag: 'array'; element: PortType; shape: number[] }
```

- `shape: [8]` — vector of 8
- `shape: [4, 4]` — 4x4 matrix
- `shape: [2, 4, 4]` — pair of 4x4 matrices
- `element` is recursive: `array<{x: float, y: float}, [4]>` has 8 underlying scalars

### Layout

Row-major, contiguous. For `array<T, [d0, d1, ..., dn]>`:
- `strides = [d1*...*dn, d2*...*dn, ..., dn, 1]` (scaled by element_size(T))
- `total_scalars = product(shape) * scalar_count(element)`
- Access `[i0, i1, ..., in].field_k` maps to scalar offset `sum(ij * strides[j]) + field_offset(k)`

For arrays of structs:
```
array<{x: float, y: float, z: float}, [4]>

element_size = 3, shape = [4], total = 12 scalars
access [2].y  →  2*3 + 1  →  scalar 7
```

For arrays of sum types: each variant payload is padded to max variant size, plus a tag scalar per element.

### Broadcasting Rules (numpy)

Shapes align from trailing dimensions:
1. Scalar promotes to any shape
2. Dimension of size 1 stretches to match the other
3. Mismatched non-1 dimensions → compile error

Full numpy broadcasting is feasible because shapes are static — it's just integer arithmetic at compile time. Initial implementation covers the common cases; edge cases are added as needed.

```
float       + float[4]     → float[4]       (scalar broadcast)
float[1,4]  + float[4,1]   → float[4,4]     (mutual broadcast)
float[4,4]  @ float[4]     → float[4]       (matmul, not broadcasting)
```

## Phases

### Phase 0 — Type Foundation

The type system learns about shaped arrays. Nothing compiles differently yet.

**Changes:**
- `compiler/term.ts` — add `array` case to PortType, update equality, printing, product flattening
- `compiler/type_check.ts` — propagate array types through composition, validate shape compatibility at connection boundaries
- New: `compiler/shape.ts` — shape algebra utilities:
  - `broadcast_shapes(a, b) → result | error`
  - `shape_size(shape) → number`
  - `strides(shape) → number[]`
  - `element_scalar_count(element: PortType) → number`
  - `flatten_index(indices, strides) → offset`

**Deliverable:** `type.array(type.float, [4, 4])` works in port type declarations. Type checker rejects shape mismatches at module boundaries. No runtime effect.

### Phase 1 — DSL Surface & Expression Nodes

Program definitions can include array-typed expressions in their process bodies.

**New/extended ExprNode ops:**

| Op | Semantics |
|---|---|
| `array_literal(shape, elements)` | Construct shaped array from scalars |
| `zeros(shape)` / `ones(shape)` / `fill(shape, val)` | Constructors |
| `reshape(expr, new_shape)` | Zero-cost view change (product of dims must match) |
| `transpose(expr, axes?)` | Axis permutation (index remapping) |
| `slice(expr, ranges)` | Static sub-array extraction |
| `reduce(op, expr, axis?)` | Sum/prod/max/min along axis |
| `broadcast_to(expr, shape)` | Explicit broadcast (usually implicit) |
| `map(fn, ...arrays)` | Element-wise apply |

Existing `add`, `mul`, `sqrt`, etc. become **shape-polymorphic**: when operands are arrays, the compiler wraps them in element-wise map + broadcast.

`matmul` (already exists) gets shape validation: `[m,k] @ [k,n] → [m,n]`, `[m,k] @ [k] → [m]`.

**DSL surface** (in `module.ts` or new `array_dsl.ts`):
```typescript
const stereo = input('audio')        // float[2]
const gain = input('gain')           // float
const out = stereo * gain            // broadcasting: float[2] * float → float[2]
const mono = stereo.sum(0)           // reduce along axis 0: float[2] → float
const matrix = input('mix')          // float[4,4]
const mixed = matrix.matmul(input('ins'))  // float[4,4] @ float[4] → float[4]
```

**Changes:**
- `compiler/expr.ts` — new op types, shape annotations on ExprNodes
- `compiler/module.ts` — SignalExpr gains shape-aware operator overloads
- New: `compiler/shape.ts` (extends Phase 0 utilities)

**Deliverable:** Modules can be defined with array-typed I/O and array expressions. Expressions carry shape metadata. Still lowers to existing mechanisms.

### Phase 2 — Compiler Lowering (runtime arrays)

Array expressions compile to flat plans using the existing `array_storage` mechanism. This is the core integration phase.

**Approach:** All arrays lower to runtime arrays (heap-allocated `array_storage` slots with `array_ptrs` / `array_sizes`). No unrolling. This reuses the existing C++ infrastructure directly.

**New pass: `lower_arrays.ts`** (runs after type check, before flatten):

1. **Shape propagation.** Walk expressions bottom-up, infer shapes everywhere. Attach shape metadata to every array-typed sub-expression.
2. **Broadcasting insertion.** Where operand shapes don't match, insert explicit `broadcast_to` nodes (resolved to index-remapping during codegen).
3. **Operation expansion.** Element-wise ops on arrays become loops over array storage. `a + b` on `float[4]` becomes 4 indexed add operations writing to 4 slots of a result array.
4. **Matmul expansion.** `A @ B` for `[m,k] @ [k,n]` becomes the standard triply-nested index/multiply/accumulate pattern over array slots.
5. **Reduce expansion.** `sum(arr, axis=0)` becomes sequential accumulation over the relevant array indices.

**Layout table** (compile-time bookkeeping):
```typescript
interface ArrayLayout {
  slotId: number       // index into array_storage
  shape: number[]
  strides: number[]
  elementLayout?: StructLayout  // for arrays of structs
}
```

The flattener maintains `Map<string, ArrayLayout>` keyed by `(moduleId, portName)` to resolve array references across module boundaries.

**Flatten changes** (`compiler/flatten.ts`):
- An array output/register maps to one or more array_storage slots
- `ref(module, output)` for array outputs resolves via the layout table
- Register offset logic accounts for array slot allocation alongside scalar registers

**Plan format** — add optional `array_layouts` metadata to `egress_plan_2`:
```json
{
  "array_layouts": {
    "MatrixMixer.out": { "slot": 3, "shape": [4], "strides": [1] }
  }
}
```

The C++ runtime and JIT already support `ArrayPack`, `Index`, `ArraySet`, `array_storage`, `array_ptrs`, and `array_sizes`. Minimal C++ changes expected — primarily ensuring the plan parser handles the new shape metadata.

**Deliverable:** Array modules compile and run end-to-end through the JIT. A `float[4,4]` matrix lives in `array_storage` as 16 contiguous int64 values. Delay buffers, mixing matrices, and stereo pairs all use the same mechanism.

### Phase 3 — Inter-Module Array Wiring

Array-typed outputs connect to array-typed inputs across module boundaries with shape checking and broadcasting at the connection boundary.

**Changes:**
- `flatten.ts` — resolving `ref(A, 'out')` for array ports returns the array slot reference; consuming module reads from that slot via index ops
- Shape validation at connection time: `float[4]` output to `float[8]` input → compile error. `float` (scalar) to `float[4]` input → auto-broadcast
- MCP `connect_modules` / `set_module_input` tools validate shape compatibility
- Patch format gains optional type annotations for validation:
  ```json
  { "module": "VCO1", "output": "saw", "type": { "tag": "array", "shape": [8] } }
  ```

**Deliverable:** Full array signal chains — `PolyVCO[8] → PolyFilter[8] → PolyVCA[8] → Mixdown`.

### Phase 4 — Standard Library

Ship useful array-aware modules:

- **PolyVoice\<N\>** — N-voice unison with per-voice detune, phase, amplitude
- **MatrixMixer\<M,N\>** — M inputs → N outputs via M*N gain matrix
- **StereoField** — stereo width, pan, mid-side encoding/decoding
- **Crossfade\<N\>** — N-way crossfade with coefficient matrix
- **ParallelFilter\<N\>** — N parallel filters (formant banks, EQ)
- **DelayLine\<N\>** — standard delay buffer, same array type as everything else

Also: array-aware parameter support — `smoothed_param` for a `float[4,4]` matrix (16 control params, one per element).

### Phase 5 — Optimizations

**Scalar unrolling.** For small, statically-indexed arrays, unroll to scalar registers instead of runtime arrays. Decision criteria: scan all accesses; if every index is a compile-time constant, unroll. Otherwise, keep as runtime array. This is an optimization pass, not a type-level distinction.

Benefits of unrolling (for small arrays):
- LLVM can constant-fold, DCE, CSE, and auto-vectorize across elements
- No pointer indirection (values live in CPU registers)
- For audio-rate small matrices (4x4), register access beats pointer chase

When NOT to unroll:
- Any dynamic index forces runtime array
- Large arrays (>~256 elements) — unrolling bloats LLVM IR and pressures I-cache
- Arrays passed across module boundaries where the consuming module uses dynamic indexing

**Other optimizations:**
- Tree-structured reductions (balanced tree instead of left fold for better numerical behavior and ILP)
- CSE across broadcast-duplicated expressions
- SIMD vectorization — pattern-match sequences of identical ops on consecutive array elements, emit vector instructions in `NumericProgramBuilder`
- BLAS callout for large control-rate matrix operations (threshold ~32x32)

### Phase 6 — Dynamic Lists (deferred)

Variable-length containers for control-rate applications. Genuinely different from fixed-shape arrays — the shape itself varies at runtime.

```typescript
| { tag: 'list'; element: PortType; max_size: number }
```

- Runtime carries actual length alongside data
- Supports append, remove, variable iteration
- Not unrollable (length unknown at compile time)
- Bounds-checked (off in release, on in debug)
- Use cases: variable polyphony, MIDI note lists, dynamic voice allocation

## Execution Order

```
Phase 0 (types)  →  Phase 1 (DSL)  →  Phase 2 (lowering)  →  Phase 3 (wiring)
                                                                     ↓
                                                               Phase 4 (stdlib)
                                                                     ↓
                                                               Phase 5 (optimize)
                                                                     ↓
                                                               Phase 6 (dynamic lists)
```

Phases 0-2 are the core — after those, arrays work within single modules. Phase 3 unlocks array signals flowing through the graph. Phase 4+ is iterative.

## Key Architectural Decisions

1. **One array type, one initial lowering.** No premature optimization. Runtime arrays (`array_storage`) for everything. Unrolling is Phase 5.
2. **Static shapes only (initially).** All dimensions known at compile time. Dynamic lists are Phase 6.
3. **Numpy broadcasting.** Full rules are feasible (static shape arithmetic). Implement common cases first, expand as needed.
4. **Composable with structs/sums.** Arrays of structs unroll to `N * fields` scalars. Stride tables compose.
5. **Delay buffers are just arrays.** `float[4096]` with a dynamic write index. Same type, same mechanism. No special casing.
