# Legacy compatibility quarantine — sprint handoff

> **Outcome (2026-07-28): completed and superseded by staff decision S-09.**
> The audit found no production dependency on the Plan-4 lift. Staff selected
> immediate removal with no deprecation window: serialized plan entry points
> are Plan-5-only, retired carriers fail clearly, and specialized parameter
> method aliases are gone. The material below is the original investigation
> brief, preserved as design history; it does not describe current support.

- **Sprint:** 2026-07-28 through 2026-08-10
- **Lane:** F — legacy plan/runtime boundary
- **DRI:** Runtime/compiler retirement lane
- **Supervisor:** Staff engineer
- **Status:** Historical — completed/superseded
- **Master:** [Staff engineer sprint handoff](00-staff-engineer-master-handoff.md)
- **Depends on:** None. Lane A consumes the classification; Lane C records the
  retired-boundary rejection evidence.
- **Must not overlap:** No deletion of supported behavior without a staff
  decision; no Metal-specific runtime edits.

## Mission

Make it impossible to confuse the CF-only production compiler with legacy
stateful capabilities still accepted or exercised below the waist.

Tropical’s current source language cannot express registers, updates, delays,
or feedback. The C++ runtime/parser and hand-written `tropical_plan_4`
fixtures retain older machinery for compatibility and low-level tests. That
may be useful, dead, or a seed for a sister runtime; what is unacceptable is
leaving its status implicit.

The sprint must classify, label, and gate the boundary. It need not delete it.

## Questions to answer

1. Which `tropical_plan_4` fields and instructions are still accepted?
2. Which are reachable from any production Lean compile path?
3. Which C API entry points can load them?
4. Which engine tests intentionally depend on registers/state?
5. Does any current runtime state transfer occur during production hot-swap?
6. Are plan-4 fixtures compatibility tests, generic-runtime tests, or dead
   artifacts?
7. Would deleting any retained state machinery simplify the production
   runtime measurably?
8. What belongs to a future `supertropical` extraction rather than Tropical?

## Classification

Every legacy item receives one status:

- **production-required** — reachable and supported by a current documented
  product path;
- **compatibility-supported** — accepted intentionally, but not emitted by
  current Tropical;
- **test substrate** — retained only to test low-level typed execution;
- **historical/dead** — no supported caller; safe-removal candidate;
- **sister-runtime candidate** — useful stateful machinery, explicitly outside
  Tropical’s language contract.

Every item also receives an owner and a proposed removal/review date where
appropriate.

## Deliverables

### F1. Compatibility matrix

Create:

```text
design/compatibility-matrix.md
```

with rows for:

- schema versions;
- parser branches;
- plan instructions;
- state/register metadata;
- load APIs;
- runtime state allocation/transfer;
- tests and fixtures;
- documentation claims.

For each row record reachability from:

- Lean stdlib builder;
- playground patch graph;
- MCP/session compiler;
- `diffcli`;
- web build;
- direct C API only.

### F2. Production non-emission gate

Add a focused gate proving that representative production compile paths emit
no legacy state surface:

- no register/update instructions;
- no delay/state-init metadata used by current source semantics;
- no plan-4 schema from a production compiler;
- no back-edge acceptance at source/session boundaries.

Prefer inspecting typed plans before JSON serialization. Use wire text only
for schema-version assertions.

### F3. Test quarantine

Reorganize or relabel C++ tests so their intent is visible:

```text
engine/tests/
  current/ or clearly current test sections
  compatibility/ or clearly compatibility-labeled sections
```

Physical file moves are optional; semantic labeling is mandatory. CTest names
must communicate `legacy_plan4` or `compat` for compatibility fixtures.

Update `engine/tests/CLAUDE.md` so a reader cannot mistake a state-register
test for the current language.

### F4. Removal recommendation

Present one recommendation:

- retain the bounded compatibility surface and document its cost;
- deprecate with a removal date and downstream-owner list;
- extract test substrate from production builds;
- remove dead paths in a separately reviewed follow-up.

Do not perform a broad removal in this sprint unless the staff engineer
explicitly converts it into must-land scope after the Day 4 review.

## Owned files

Primary ownership:

- `design/compatibility-matrix.md`
- `engine/tests/`
- `engine/runtime/NumericProgramParser.hpp`
- legacy-specific comments in `engine/runtime/FlatRuntime.*`
- legacy-specific C API comments and tests
- schema compatibility documentation

Shared runtime changes require coordination with Lane E. Lane F owns generic
legacy classification; Lane E owns Metal pipeline behavior.

## Work plan

### Days 1–2: reachability audit

- Trace every plan-4/state field from parser to runtime.
- Trace all current Lean emit paths in the opposite direction.
- Inventory tests by actual behavior, not filename.
- Produce the first compatibility matrix.

### Day 3: non-emission gate

- Add the typed production-plan check.
- Cover stdlib, playground, session, and export paths.
- Ensure failure output names the legacy construct.

### Day 4: staff decision review

The staff engineer decides the status of every ambiguous row. No ambiguous row
may survive the meeting as “probably dead.”

This review decides whether any deletion is allowed in the sprint.

### Day 5: labeling and first integration

- Rename CTest cases or sections.
- Update test documentation.
- Run the complete C++ and Lean gates.

### Days 6–7: bounded quarantine

- Separate compatibility-only helper code where this is mechanical and
  behavior-preserving.
- Add comments at parser/runtime branches naming their classification.
- Do not redesign the parser.

### Day 8: removal-cost analysis

- Estimate code, build, and trust-surface cost of retained compatibility.
- Identify the exact extraction/removal sequence for a future sprint.

### Day 9: adversarial reachability review

An independent reviewer attempts to create state or a cycle through every
current front door. All must refuse before backend code.

Direct legacy C API loading may succeed only where the matrix says it should.

### Day 10: freeze

- Finalize matrix and recommendation.
- Give Lane A exact wording.
- Give Lane C the retained obligation.

## Acceptance gates

1. Every plan-4/state path has one explicit classification.
2. Production Lean paths are gated against legacy state emission.
3. Current front doors cannot author state or feedback.
4. C++ test names/docs distinguish compatibility tests from production
   semantics.
5. No supported direct C API caller is removed silently.
6. Retained compatibility has an owner and rationale.
7. `make validate` and direct CTest execution are green.

## Non-goals

- No stateful sister runtime implementation.
- No source-level register/delay restoration.
- No feedback semantics.
- No schema-version bump.
- No wholesale C++ runtime rewrite.
- No deletion based only on search results.

## Stop and escalate

Stop if:

- a production path can emit or activate legacy state;
- a compatibility path is used by an undocumented external caller;
- current and compatibility tests require contradictory runtime behavior;
- removing state support would alter Metal/JIT shared allocation;
- a destructive deletion lacks an approved downstream inventory.

The first condition is a sprint-critical correctness defect.

## Handoff package

Leave:

- complete compatibility matrix;
- production non-emission gate;
- clearly named current versus compatibility tests;
- a retain/deprecate/extract/remove recommendation;
- a scoped follow-up suitable for the future stateful sister runtime.
