# Current performance baseline — sprint handoff

- **Sprint:** 2026-07-28 through 2026-08-10
- **Lane:** D — compiler and runtime measurement
- **DRI:** Assign at kickoff
- **Supervisor:** Staff engineer
- **Status:** Planned
- **Master:** [Staff engineer sprint handoff](00-staff-engineer-master-handoff.md)
- **Depends on:** None. Lane A consumes the report; Lane G consumes regression
  thresholds.
- **Must not overlap:** Measure first. Do not implement tiered compilation,
  automatic backend selection, or broad optimizer changes in this sprint.

## Outcome — fixed renderer measured, optimization retired

The original flagship/tiering fork below is retained as the lane's historical
brief. The measured graph is the exact fixed `playground/renderer` demo and
reference fixture, not the separately scoped flagship product. Its
approximately 1.4–1.6 s warm and 5–6 s cold generation/load walls missed the
kickoff hypothesis, but no interactive topology editor or structural selector
exists in this renderer; live controls use `set_param` without relowering.
Staff therefore accepts the machine-local baseline and authorizes neither a
tiered preview nor compile-time optimization for this fixed renderer. Reopen
the question only for a product-facing structural editor or an explicit boot
latency contract.

## Mission

Replace pre-bank and anecdotal performance claims with a reproducible baseline
of current `main`.

The report must separate:

- authoring/lowering time;
- plan emission time;
- LLVM code generation time;
- MSL compile/load time;
- cold cache and warm cache;
- topology edit and parameter write;
- JIT execution and Metal execution;
- throughput, callback cost, and user-visible control latency.

The sprint ends with a decision, not necessarily an optimization:

> Is current structural editing interactive enough for the flagship modal
> instrument, or should the next sprint build a tiered preview path?

## Test matrix

### Patch shapes

At minimum:

1. pure fixed sine;
2. through-zero flanger;
3. one modal ring;
4. ring → reverb;
5. four-ring playground circuit;
6. gong;
7. plucked string;
8. uniform modal banks at capacities 16, 64, 128, 256, and 512;
9. one dynamic-count bank measured at several live counts without topology
   changes;
10. one nested-bank fixture.

Use checked-in fixtures or add generated fixtures under the benchmark
directory. Do not rely on an untracked local patch.

### Measurements

For every relevant row capture:

- source/graph size;
- arena node count;
- plan instruction count;
- bank capacity and live trip count;
- generated LLVM line/byte count;
- generated MSL line/byte count;
- wasm artifact size where applicable;
- front-end lower time;
- stage-0 split time;
- LLVM emit time;
- ORC compile time;
- MSL compile/load time;
- total cold load;
- total warm load;
- parameter write latency;
- topology edit to publication latency;
- JIT time per 512-sample block;
- Metal sustained block interval where applicable.

If a measurement cannot be separated without invasive instrumentation, report
the coarser boundary honestly.

## Reproducibility contract

Every result row records:

- commit sha;
- dirty/clean status;
- machine model and chip;
- OS version;
- Lean toolchain;
- LLVM version;
- build type and flags;
- sample rate and block size;
- backend and pipeline depth;
- stage-0 and bank-unroll flags;
- cache state;
- repeat count and statistic.

Use median plus p95/p99 for runtime latency and minimum/median for deterministic
compile stages. Keep raw samples.

Do not clear or mutate the user’s ordinary kernel cache from the harness.
Implement or use a benchmark-specific cache root/disable flag. Cold and warm
runs must be reproducible without destructive setup instructions.

## Owned files

Create a focused benchmark surface:

```text
benchmarks/current_baseline/
  README.md
  run.sh or run.py
  fixtures/
  data/
  findings.md
```

A script is preferred over manual command transcripts. Python is acceptable
for benchmark orchestration; it must not become production code.

This lane may add opt-in timing instrumentation to `diffcli` or the Lean compile
path, guarded so normal output/schema behavior is unchanged. Coordinate
ownership of `Diffcli.lean` with Lane C on Day 1.

## Work plan

### Day 1: protocol and smoke

- Freeze the matrix and metrics with the staff engineer.
- Select the canonical Apple measurement host.
- Run a three-patch smoke to ensure the harness captures raw data.

### Days 2–3: harness

- Add deterministic fixture generation where needed.
- Add benchmark-specific cache control.
- Add stage timing with machine-readable output.
- Check that instrumentation changes no plan or render.

### Days 4–5: compile baseline

- Run the complete patch matrix cold and warm.
- Run bank-capacity/live-count sweeps.
- Verify the expected compile-flatness claim after banks-as-data.

### Days 6–7: edit and execution baseline

- Measure raw/glide/anchor/velocity parameter writes.
- Measure one topology edit and one structural selector edit.
- Measure JIT and Metal execution separately.
- Preserve raw results before interpretation.

### Day 8: analysis

Answer:

1. Which phase dominates each patch class?
2. Is compile time flat in live bank count and acceptably bounded in capacity?
3. What is the cold and warm structural-edit latency for the flagship circuit?
4. Where does Metal cross the JIT?
5. Which product claims are supportable?

### Day 9: decision review

Present one of:

- **No tiering needed:** structural edits meet the agreed interactive budget;
- **Tiering recommended:** a preview mode is justified, with measured target;
- **More isolation needed:** instrumentation cannot yet identify the wall.

Do not implement the chosen architecture during this sprint.

### Day 10: report freeze

- Commit raw data and findings.
- Give Lane A the approved numbers and wording.
- Give Lane G candidate regression budgets.

## Acceptance gates

1. One command reproduces the benchmark suite after required builds.
2. Raw data is committed in a stable machine-readable form.
3. Cold and warm cache states are isolated and repeatable.
4. At least five bank capacities and three live counts are measured.
5. Parameter writes and topology edits are reported separately.
6. The current four-ring circuit has a dated cold and warm result.
7. Instrumentation does not change emitted plans or frozen renders.
8. Findings name noise, limitations, and machine specificity.
9. A signed staff decision records whether tiered preview is next-sprint work.
10. `make validate` is green after instrumentation.

## Candidate product budgets

These are kickoff hypotheses, not pass/fail facts. The staff engineer may
adjust them after Day 3:

- raw parameter write: below one audio block;
- glided/anchored write dispatch: below one audio block, audible effect subject
  to the declared glide/pipeline contract;
- warm structural edit: target below 500 ms;
- cold structural edit: target below 2 s for the flagship circuit;
- audio callback: p99 below 50% of block deadline;
- no render drift caused by instrumentation.

If current performance misses a target, record the miss. Do not tune the target
after seeing the result merely to obtain green.

## Non-goals

- No automatic cost model.
- No microkernel preview implementation.
- No new caching architecture.
- No LLVM pass-pipeline redesign.
- No performance-driven numeric weakening.
- No claim that one M1-class machine predicts every Apple GPU.

## Stop and escalate

Escalate if:

- cold/warm cache state cannot be controlled safely;
- instrumentation changes emitted bytes or behavior;
- the same benchmark has more than 20% unexplained median variance;
- current fixture generation depends on uncommitted files;
- JIT and Metal results disagree beyond existing correctness tolerances.

Correctness disagreement transfers immediately to Lane G as a stop-the-line
issue; performance measurement pauses for the affected fixture.

## Handoff package

Leave:

- reproducible harness and fixtures;
- raw samples and summarized tables;
- approved product wording for Lane A;
- regression-budget proposal for Lane G;
- explicit tiering decision and the evidence behind it.
