.PHONY: build repl run lean mcp-lean clean validate validate-write \
        diff diff-plan diff-audio diff-engine diff-render diff-raise diff-elab \
        diff-strata diff-emit test-lean-engine

ROOT := $(shell pwd)
BUILD_DIR := $(ROOT)/build

PYTHON := $(shell which python3)
LLVM_DIR ?= /opt/homebrew/opt/llvm/lib/cmake/llvm

JOBS ?= 4
EXTRA_CMAKE_ARGS ?=

build:
	cmake -S $(ROOT) -B $(BUILD_DIR) \
		-DTROPICAL_BUILD_PYTHON=ON \
		-DLLVM_DIR=$(LLVM_DIR) \
		$(EXTRA_CMAKE_ARGS)
	cmake --build $(BUILD_DIR) -j$(JOBS)

repl: build
	PYTHONPATH=$(BUILD_DIR) $(PYTHON)

run: repl

# Lean front-door (Turnstile). Builds the lean/ subtree, which pulls Turnstile
# via Lake and links against the IR service (mcp/ir_service.ts) at runtime.
lean:
	cd lean && PATH="$$HOME/.elan/bin:$$PATH" lake env leanc -c ffi/shim.c -o ffi/shim.o -I ../engine/c_api
	cd lean && PATH="$$HOME/.elan/bin:$$PATH" lake build

# Launch the Lean MCP server. As of Phase 6 the whole stack is the one
# binary (session, compiler, runtime FFI) — no bun subprocess.
mcp-lean: build lean
	./lean/.lake/build/bin/frontend

clean:
	rm -rf $(BUILD_DIR)

validate: build
	bun test
	ctest --test-dir $(BUILD_DIR) --output-on-failure
	bun run scripts/validate_stdlib.ts

validate-write: build
	bun run scripts/validate_stdlib.ts --write

# Lean-port differential harness (scripts/diff/). Phase 6 gate: the Lean
# pipeline (diffcli compile — engine boot + ingest + Lean emit/partition)
# against the TS oracle, across all three compilation modes.
diff-plan: build lean
	cd lean && PATH="$$HOME/.elan/bin:$$PATH" lake build diffcli
	bun run scripts/build_parsed_stdlib.ts
	for m in fused microkernel microkernel-deep; do \
		bun run scripts/diff/diff_plan.ts --b="./lean/.lake/build/bin/diffcli compile" --mode=$$m || exit 1; \
	done

diff-audio: build lean
	cd lean && PATH="$$HOME/.elan/bin:$$PATH" lake build diffcli
	bun run scripts/build_parsed_stdlib.ts
	for m in fused microkernel microkernel-deep; do \
		bun run scripts/diff/diff_audio.ts --b="./lean/.lake/build/bin/diffcli compile" --mode=$$m || exit 1; \
	done

diff-engine: build lean
	bun run scripts/diff/diff_engine.ts --b="./lean/.lake/build/bin/frontend --rpc"

# Phase 2 gate: Lean FFI rendering is byte-for-byte the koffi path.
diff-render: build lean
	bash scripts/diff/diff_render.sh

# Phase 4 gate (parsed layer): Lean raise + ParsedProgram codec against
# the TS oracle, over patches + raise fixtures + the pre-parsed stdlib
# bridge (regenerated first so a stale corpus can't pass silently).
diff-raise: build lean
	cd lean && PATH="$$HOME/.elan/bin:$$PATH" lake build diffcli
	bun run scripts/build_parsed_stdlib.ts
	bun run scripts/diff/diff_raise.ts --b="./lean/.lake/build/bin/diffcli"

# Phase 4 gate (resolved layer): Lean elaborator + resolved codec against
# the TS oracle, over the stdlib chain + elaborable raise fixtures + the
# error-message fixtures under tests/fixtures/elab/.
diff-elab: build lean
	cd lean && PATH="$$HOME/.elan/bin:$$PATH" lake build diffcli
	bun run scripts/build_parsed_stdlib.ts
	bun run scripts/diff/diff_elab.ts --b="./lean/.lake/build/bin/diffcli"

# Phase 5 gate (strata): whole-strata hybrid differential — Lean runs
# passes 1..K (STRATA_K, the stage ratchet; tracks Strata.portedPasses),
# ships the prefix through the resolved codec, the TS suffix finishes,
# and only final post-strata output is compared. K=5 is pure Lean-vs-TS.
STRATA_K ?= 5
diff-strata: build lean
	cd lean && PATH="$$HOME/.elan/bin:$$PATH" lake build diffcli
	bun run scripts/build_parsed_stdlib.ts
	bun run scripts/diff/diff_strata.ts --b="./lean/.lake/build/bin/diffcli" --k=$(STRATA_K)

# Phase 6 stage 6b gate (per-program emit): Lean strata + Core downcast +
# compileResolved against the TS oracle; wire PerInstancePlan compared
# structurally over the strata corpus (inline mode).
diff-emit: build lean
	cd lean && PATH="$$HOME/.elan/bin:$$PATH" lake build diffcli
	bun run scripts/build_parsed_stdlib.ts
	bun run scripts/diff/diff_emit.ts --b="./lean/.lake/build/bin/diffcli"

# Phase 1 gate: the protocol test suites against the Lean engine.
test-lean-engine: build lean
	TROPICAL_ENGINE_CMD="./lean/.lake/build/bin/frontend --rpc" bun test mcp/errors.test.ts mcp/wire_dac.test.ts

diff: diff-plan diff-audio diff-engine diff-render diff-raise diff-elab diff-strata diff-emit
