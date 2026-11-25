# Type families: a registered type at a size

Date: 2026-11-25. Code: `lean/Tropical/Family.lean`,
`lean/Tropical/EmitArrow/Families.lean`, `Session/Registry.lean`
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
  members, with no aliases. They are still registered at boot, by the same
  `splineProgram` call the family makes, so patches, goldens and the boot
  catalog do not move.

`type_args` is not used. It is the spelling of the retired generics and stays
refused, so there is one way to name a member.

The cost: a family name must end in a non-digit, so a member name splits
back into family and size; and a family has one size. A family with two
indices (rows × columns) would need a separator or a field. That waits for a
second consumer.

## Refusals

| Spelling | Example | Tool (`add_instance`) | Ingest (`load`) |
|---|---|---|---|
| unknown family | `PeriodicHermite12` | `unknown_program`; message lists the families | `internal_error`, same list |
| missing size | `PeriodicCatmullRom` | `invalid_value` | `internal_error` |
| size out of range | `PeriodicCatmullRom0` | `invalid_value`, `valid` = predicate `{min, max}`, `got` | `internal_error` |
| leading zeros | `PeriodicCatmullRom012` | `invalid_value`, names `PeriodicCatmullRom12` | `internal_error` |
| size on a non-family | `VCA3` | `invalid_value` | `internal_error` |

`export_program` refuses a family's name and every `<family><size>` spelling,
because an export there would shadow a member.

The registry refuses to boot a family set that breaks the naming contract
(`familyConflicts`): a name that ends in a digit, a duplicate, a family that
is also a registered type, or a registered name that spells a refused size
(`PeriodicBSpline07`).

## Bounds

A family declares its own range. Both periodic splines take
`1 ≤ K ≤ 2³¹ − 1`:

- The minimum is the boundary's (`SplineBoundary.wrap.minControls`), not the
  basis's. One point is a constant curve. Two or three points wrap through
  the four-point window and are still a closed `C¹` (Catmull-Rom) or `C²`
  (B-spline) curve. The gates render `K = 1, 2, 3` against the reference.
- The maximum is the index arithmetic's: `splineSig` needs fewer than 2³¹
  segments for `p · n` to fit, and a periodic spline has `K` segments.

The maximum is not a memory budget. A member at a very large size assembles
`K` inputs and a `K + 3` table.

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
