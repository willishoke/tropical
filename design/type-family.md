# Type families: a registered type at a size

Date: 2026-11-25. Code: `lean/Tropical/Family.lean`, the spline families in
`lean/Tropical/EmitArrow/Stdlib.lean`, `Session/Registry.lean`
(`registerFamilyMember`), `Session/Ingest.lean` (`resolveInstanceMeta`).
Gates: `lean/Tropical/Tropicaltest/TypeFamily.lean`
(`tropicaltest --spline-only`), `mcp/type_family.test.ts`, and the
type-family case in `tests/web/wasm_vs_jit.test.ts`.

A type family is a builder from a size to a program. The periodic spline of
`K` control points is one: its ports `c0 … c(K−1)` depend on `K`. The patch
bay instantiates registered types by name, and a registered type has a fixed
port list, so before families the splines were registered at `K = 8` only.

## How it works

- The registry holds families beside concrete types
  (`SessionSt.families`). A family has a name, a size range, and a builder
  `name → size → arena → chain → program` with the `StdBuilder` contract.
- An instance names a member: the family name followed by the size in
  decimal, `PeriodicCatmullRom12`.
- `resolveInstanceMeta` is the one resolution point for `load`, `merge`,
  `add_instance` and `replicate`. A registered name resolves as before.
  Otherwise, if the name is a member, the family's builder assembles it and
  `registerResolved` registers it under the member name: the same strata,
  entry and adoption a boot program gets. Then it resolves like any type.
  A family spelling that names no member is refused before anything is built.
- `registerFamilyMember` passes the builder the registry's *post-strata*
  programs as `chain` (`templateByName`), where the boot chain passes raw
  ones. `strataConcrete` relinks by name either way, so this does not matter
  to the spline families, which instantiate nothing; a future family whose
  member instantiates another type links against the lowered program.
- A member is assembled once per session and stays registered when `load`
  replaces the graph, like an exported type.
- Nothing past ingest sees a family. The IR, the plan, the kernel manifest,
  `save` and `export_program` see a concrete program with a fixed port list
  and its own name. No type parameter reaches the IR.

## Why a name, not a field

Everything past ingest keys a type by name: the catalog, `templateByName`,
`save`, the body links `export_program` writes, the plan's instance types. A
member is a distinct program with its own ports, so it needs its own name in
all of those places anyway. A `size` field on the instance would be a second
spelling of that identity, and every one of those places would have to carry
it or derive the name from it. With the size in the name:

- `save → load` and `export_program` round-trip with no change.
- The MCP tool surface keeps its arguments.
- The existing `PeriodicCatmullRom8` and `PeriodicBSpline8` are the size-8
  members, with no aliases. They are still registered at boot, by the
  family's own builder (`family.build "…8" 8`), so patches, goldens and the
  boot catalog do not move.

`type_args` is not used. It is the spelling of the retired generics and stays
refused, so there is one way to name a member.

The cost: a family name must end in a non-digit, so a member name splits
back into family and size; and a family has one size. A family with two
indices (rows × columns) would need a separator or a field. That waits for a
second consumer.

## Refusals

| Spelling | Example | Tool (`add_instance`) | Ingest (`load`) |
|---|---|---|---|
| not a type or family spelling | `PeriodicHermite12` | `unknown_program`; message lists the families | `internal_error`, same list |
| misspelt member | `PeriodicCatmulRom12` | `unknown_program`, suggests `PeriodicCatmullRom12` | `internal_error` |
| size on a non-family | `VCA3` | `unknown_program`, "'VCA' is not a type family and takes no size", suggests `VCA` | `internal_error`, same reason |
| missing size | `PeriodicCatmullRom` | `invalid_value` | `internal_error` |
| size out of range | `PeriodicCatmullRom1025`, `…00` | `invalid_value`, `valid` = predicate `{min, max}`, `got` | `internal_error` |
| leading zeros | `PeriodicCatmullRom012` | `invalid_value`, names `PeriodicCatmullRom12` | `internal_error` |

The range is checked before the spelling, so `PeriodicCatmullRom00` is out
of range rather than advised to become `PeriodicCatmullRom0`. A size with more
significant digits than the family's largest is out of range without being
converted, and messages cut names past 64 characters, so a 5000-digit name
costs nothing and is not echoed back.

`export_program` refuses a family's name and every `<family><size>` spelling,
because an export there would shadow a member.

The registry refuses to boot a family set that breaks the contract
(`familyConflicts`): a name that ends in a digit, a duplicate, a family that
is also a registered type, a member at `sizeMax` past the port budget, or a
registered name that spells a refused size (`PeriodicBSpline07`).

## Bounds

A family declares its own range, and its largest member must fit the
registry's port budget, `memberPortBudget` = 1056 ports. Everything
downstream of a member scales with its port count: its assembly, lowering
and plan, and the kernel's parameter list (one per live input). The budget
is what keeps one request from making the server assemble a member of
billions of ports; it is checked at boot for every family, so a future
family cannot declare past it.

Both periodic splines take `1 ≤ K ≤ 1024`:

- The minimum is the boundary's (`SplineBoundary.wrap.minControls`), not the
  basis's. One point is a constant curve. Two or three points wrap through
  the four-point window and are still a closed `C¹` (Catmull-Rom) or `C²`
  (B-spline) curve. The gates render `K = 1, 2, 3` against the reference.
- The maximum is the budget's: a member has `K + 2` ports (`phase`,
  `c0 … c(K−1)`, `out`), so 1024 control points fit with room for a few
  control ports. The gates render `K = 1024` against the reference. The index
  arithmetic alone would allow `2³¹ − 1` (`p · n` needs fewer than 2³¹
  segments).
- The browser's limit is lower than the cap. The WebAssembly JS API refuses
  a function of more than 1000 parameters, and the kernel takes one pointer
  per param plus `tau` and `out`, so a member whose controls are all live
  params cannot play in the browser above about 998 of them.
  Checked at the cap: a `PeriodicCatmullRom1024` with all 1024 controls
  wired from params compiles to a `.wasm` (`diffcli compile-wasm` succeeds),
  and the JS engine refuses the module when it loads it ("argument count of
  Type at index 0 is too big 1026 maximum 1000"). At 998 it loads.

## Listing

`list_programs` lists the concrete types, then one entry per family with a
`family` descriptor: `member` (`PeriodicCatmullRom<size>`), `size_min`,
`size_max`, `size_meaning`, `ports`. The `tropical://programs` resource adds
the same lines.

## Not built

- Families other than the periodic splines. Spectral and modal bank sizes
  are candidates.
- Playground nodes. Playground port tables are static per node kind
  (`Playground/VocabularyMetadata.lean`). A node kind with a size could
  resolve to a member the same way, if its port table were computed from
  the member's entry. That is not built.
- Eviction: members stay registered for the session's life.
- A clean refusal on the wasm path for a kernel past 1000 parameters. Today
  `diffcli compile-wasm` writes the module and the failure comes late, when
  a JS engine compiles it. Literal (non-param) controls do not count toward
  the limit.
