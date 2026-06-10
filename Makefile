.PHONY: build repl run lean mcp-lean clean validate validate-write \
        diff diff-plan diff-audio diff-engine

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
	cd lean && PATH="$$HOME/.elan/bin:$$PATH" lake build

# Launch the Lean front-door MCP server. It spawns mcp/ir_service.ts itself, so
# this is the single command that brings up the whole Lean → IR → C++ stack.
mcp-lean: build lean
	bun install --silent && ./lean/.lake/build/bin/frontend

clean:
	rm -rf $(BUILD_DIR)

validate: build
	bun test
	ctest --test-dir $(BUILD_DIR) --output-on-failure
	bun run scripts/validate_stdlib.ts

validate-write: build
	bun run scripts/validate_stdlib.ts --write

# Lean-port differential harness (scripts/diff/). Each differ compares two
# pipelines — both default to TS, so these are green before any port work;
# as Lean layers land, pass --b="<lean cmd>" (or edit the default here) to
# pit Lean against the TS oracle.
diff-plan: build
	bun run scripts/diff/diff_plan.ts

diff-audio: build
	bun run scripts/diff/diff_audio.ts

diff-engine: build
	bun run scripts/diff/diff_engine.ts

diff: diff-plan diff-audio diff-engine
