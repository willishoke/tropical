# ModalForest M0/M1 implementation summary

Date: 2026-08-02 (America/Los_Angeles)

Status: **M0 and bounded M1 complete locally; full validation green before
checkpoint publication**

This record supplements
`03-modal-forest-generalized-room-handoff.md`. It records the baseline
qualification, implementation choices, and scope boundary for the first
ModalForest checkpoint. It does not change the later gong, bloom-seam, or
generalized grouped-room architecture gates.

## 1. Branch and worktree context

| Item | Value |
|---|---|
| Worktree | `/private/tmp/tropical-modal-forest-grouped-room` |
| Branch | `sprint/modal-forest-grouped-room` |
| Starting branch head | `672ea243778a618d3874da8ad5b88d35a61ddc22` |
| Starting migration ancestor | `78c12a955ba42af356e6e43b776339f495743096` |
| Published branch head before this checkpoint | `672ea243778a618d3874da8ad5b88d35a61ddc22` |
| Current canonical migration head observed during work | `0d0d7544be3d5f0963ba2b100c91181d3a1e3312` |

The canonical migration branch advanced after this worktree was initialized.
This checkpoint deliberately does not merge, rebase, retarget, or rewrite
history. Its M0/M1 changes are committed first so the base update can be
integrated as a separate, auditable operation.

No sprint edits were made in the primary checkout. No benchmark evidence,
`benchmarks/demo_release/data`, listening/capture artifacts, or packaged room
assets were changed.

## 2. M0 baseline

### 2.1 Import and dependency state

- `./reversible/scripts/verify-import.sh`: PASS, all 15 replayed commits,
  ending at standalone target
  `6a0e0c3765ed1ce263b8f96e8c096863ddd28710`.
- The fresh worktree's pinned `lib/json` and `lib/rtaudio` submodules were
  initialized at `9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03` and
  `409636b5dcad3054ae5a9e85014bba3861b8edab` respectively.
- The manifest-pinned Turnstile checkout was seeded into the ignored Lake
  package cache at `a6c2a9774a9b1f6d3946293d3a19291e80c0b8b6` because the isolated worktree
  began without that build cache. These are local build prerequisites, not
  tracked source changes.

### 2.2 Validation

- `make validate`: PASS outside the filesystem sandbox.
  - native `tropicaltest`: 124/124;
  - Bun: 155 passed, one intentional capability skip, zero failed, 1,139
    assertions;
  - CTest: 4/4;
  - Lean build, trust audit, import/provenance checks, and native builds: PASS.
- `swift build --package-path reversible`: PASS.
- `swift build --package-path reversible -c release`: PASS.

The first sandboxed validation attempt reached the socket tests but could not
bind their Unix-domain socket (`Operation not permitted`), and the Metal
capability probe was unavailable there. The same suite passed outside the
sandbox; this was an execution-environment restriction, not a product failure.

### 2.3 Toolchain difference

The machine remains macOS 26.3 / Darwin 25.3.0 with Bun 1.3.12, CMake 4.2.3,
Lean 4.29.1, and Apple Swift 6.2.3. The currently selected macOS SDK is 26.2.

The handoff's explicit macOS 15.4 SDK is not compatible with this installed
Swift 6.2.3 compiler because its Swift interfaces were produced for Swift 6.1.
Accordingly, the debug and release Swift builds were qualified with the
compiler-matched current SDK. This is a toolchain-manifest difference to keep
visible; it is not treated as proof that the older SDK combination remains
green.

## 3. Bounded M1 implementation

Modal lowering now carries an authored-order `Array ModalBranch` instead of a
single modal value. Each branch owns its bank, strike anchor, realization
clock, address input, direction metadata, and optional live mode count.

- Existing modal sources lower to singleton forests.
- Reverb/filter and gauge transformations map independently over every branch.
- `modalmix` flattens input forests in authored order.
- Plain branches whose realization metadata are structurally identical retain
  the incumbent pole-union path, preserving existing direct plans.
- Branches with different realization metadata stay separate.
- The Modal-to-Signal seam realizes each branch with its own metadata, then
  forms one stable authored-order signal sum. Singleton forests retain the
  exact incumbent term shape.
- The frozen grouped-room v1 terminal remains deliberately single-branch. It
  refuses a larger forest rather than borrowing metadata or implying that the
  fixed 12-by-12 carrier is already generalized.
- Bloomed branches remain refused by `modalmix` under the existing v1
  contract. Served `gong`, vocabulary truth, live `beta`, the bloom seam, and
  the generalized room profile remain later milestones.

The semantic fixture covers both sides of the compatibility rule:

- two branches with the same anchor lower and render byte-identically to the
  historical union; and
- branches anchored at samples 200 and 700 match the first-only graph through
  sample 700, then differ on every tested sample after the second onset.

Post-change validation:

- focused `lake build Tropical.EmitArrow.Patch`: PASS;
- full `lake build tropicaltest`: PASS;
- native `tropicaltest`: 125/125, including
  `modal-forest-anchors` (`same-anchor bitDiff=0/2048`, pre-second-anchor
  bitDiff `0/701`, post-second-anchor differing `1347/1347`);
- Bun: 155 passed, one intentional Metal-capability skip, zero failed, 1,139
  assertions; and
- CTest: 4/4, including the Metal kernel target.

The post-change `make validate` wrapper completed its build, trust audit, and
native 125/125 run, then stopped because the escalated shell's explicit PATH
omitted `$HOME/.bun/bin`. Running the wrapper's remaining Bun-build, Bun-test,
and CTest commands with the actual Bun path produced the green results above.
One Metal-worker timing assertion failed only while CTest was accidentally run
concurrently with the GPU-heavy Bun suite; the required sequential CTest rerun
passed 4/4.

## 4. Small-scale decision record

| ID | Decision | Reason |
|---|---|---|
| MF-S01 | Forest order is authored order and is semantically stable. | It gives deterministic lowering and fixes the floating-point signal-sum association order. |
| MF-S02 | Preserve the legacy pole union only when every branch has structurally equal realization metadata. | Existing direct graphs retain exact plan/audio behavior without permitting one branch to borrow another's timing. |
| MF-S03 | Compare anchor, clock, address, direction, and count metadata structurally; do not compare banks for merge eligibility. | The bank payload is what the compatible union combines; all realization context must agree first. |
| MF-S04 | Map reverb and gauge branchwise. | Each timed island remains an independent semantic source through downstream modal effects. |
| MF-S05 | Realize each branch before the stable signal sum; keep singleton realization structurally unchanged. | This preserves branch-local timing and current graph/tap behavior. |
| MF-S06 | Keep frozen `groupedroom` v1 single-branch and fail explicitly on forests. | The existing fixed-profile carrier is an oracle/compatibility surface, not the proposed generalized live room. |
| MF-S07 | Keep bloomed `modalmix` refusal and served gong/vocabulary unchanged in M1. | M3 must close the conditioning and live-`beta` contracts before M4 changes the product surface. |
| MF-S08 | Treat Unix-socket/Metal sandbox failures and the SDK 15.4 compiler mismatch as recorded environment qualifications. | Neither should be hidden as a source regression or silently generalized into a compatibility claim. |

## 5. Remaining scope

M2 and later remain open: accepted-scene timed-island serialization and
forward/hold/reverse/seek truth, bloom-seam hardening, served modal gong,
source-independent grouped-room profile v2, evaluator cost selection, product
re-authoring, and release/listening qualification.

In particular, this checkpoint proves the internal forest timing seam. It does
not claim that the served gong is modal or that the current grouped room is a
generalized live-room profile.
