.PHONY: build repl run lean mcp-lean clean validate parse-all test-lean-engine

ROOT := $(shell pwd)
BUILD_DIR := $(ROOT)/build

PYTHON := $(shell which python3)
LLVM_DIR ?= /opt/homebrew/opt/llvm/lib/cmake/llvm

JOBS ?= 4
EXTRA_CMAKE_ARGS ?=

build:
	cmake -S $(ROOT) -B $(BUILD_DIR) \
		-DTROPICAL_BUILD_PYTHON=ON \
		-DTROPICAL_WASM_EMIT=ON \
		-DLLVM_DIR=$(LLVM_DIR) \
		$(EXTRA_CMAKE_ARGS)
	cmake --build $(BUILD_DIR) -j$(JOBS)
# TROPICAL_WASM_EMIT=ON links the WebAssembly backend + lld so `diffcli
# compile-wasm` (the web build) works out of the box; requires `brew install
# lld`. The CMake default is OFF, so a slim release build (no lld) is just
# `cmake ... -DTROPICAL_WASM_EMIT=OFF` via EXTRA_CMAKE_ARGS.

repl: build
	PYTHONPATH=$(BUILD_DIR) $(PYTHON)

run: repl

# Lean front-door. Builds the lean/ subtree (the production compiler + MCP
# server, one binary), which pulls Turnstile via Lake.
lean:
	cd lean && PATH="$$HOME/.elan/bin:$$PATH" lake env leanc -c ffi/shim.c -o ffi/shim.o -I ../engine/c_api
	cd lean && PATH="$$HOME/.elan/bin:$$PATH" lake build

# Launch the Lean MCP server. As of Phase 6 the whole stack is the one
# binary (session, compiler, runtime FFI) — no bun subprocess.
mcp-lean: build lean
	./lean/.lake/build/bin/frontend

clean:
	rm -rf $(BUILD_DIR)

# Full gate: the Lean golden/equiv runner (replaces validate_stdlib + the
# native equiv suite), the surviving behavioral bun suites (WASM≡JIT in
# tests/web + the MCP protocol tests against the Lean engine), and ctest.
validate: build lean
	cd lean && PATH="$$HOME/.elan/bin:$$PATH" lake build diffcli tropicaltest
	./lean/.lake/build/bin/tropicaltest
	bun web/build_patches.ts
	TROPICAL_ENGINE_CMD="./lean/.lake/build/bin/frontend --rpc" bun test
	ctest --test-dir $(BUILD_DIR) --output-on-failure

# Regenerate the committed stdlib/parsed bridge from stdlib/*.md (the Lean
# surface parser). Replaces the former scripts/build_parsed_stdlib.ts.
parse-all: lean
	cd lean && PATH="$$HOME/.elan/bin:$$PATH" lake build diffcli
	./lean/.lake/build/bin/diffcli parse-all

# Behavioral MCP protocol suites against the live Lean engine.
test-lean-engine: build lean
	TROPICAL_ENGINE_CMD="./lean/.lake/build/bin/frontend --rpc" bun test mcp/errors.test.ts mcp/wire_dac.test.ts
