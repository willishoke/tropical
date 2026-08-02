.PHONY: build repl run lean mcp-lean clean validate test-lean-engine reversible-build reversible-test reversible-bundle reversible-bundle-smoke

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
		-DTROPICAL_METAL=ON \
		-DLLVM_DIR=$(LLVM_DIR) \
		$(EXTRA_CMAKE_ARGS)
	cmake --build $(BUILD_DIR) -j$(JOBS)
# TROPICAL_WASM_EMIT=ON links the WebAssembly backend + lld so `diffcli
# compile-wasm` (the web build) works out of the box; requires `brew install
# lld`. TROPICAL_METAL=ON builds the GPU backend (macOS only; runtime MSL
# compile, no Xcode needed). The CMake defaults are OFF, so a slim build is
# `cmake ... -DTROPICAL_WASM_EMIT=OFF -DTROPICAL_METAL=OFF` via EXTRA_CMAKE_ARGS.

repl: build
	PYTHONPATH=$(BUILD_DIR) $(PYTHON)

run: repl

# Lean front-door. Builds the lean/ subtree (the production compiler + MCP
# server, one binary), which pulls Turnstile via Lake. Builds ALL THREE
# executables, not just the default `frontend`: the gate runners are meant
# to be invoked as built binaries (never `lake exe`), and a bare
# `lake build` leaves stale tropicaltest/diffcli binaries in place.
lean:
	cd lean && PATH="$$HOME/.elan/bin:$$PATH" lake env leanc -c ffi/shim.c -o ffi/shim.o -I ../engine/c_api
	cd lean && PATH="$$HOME/.elan/bin:$$PATH" lake build frontend diffcli tropicaltest trustreport

# Launch the Lean MCP server. As of Phase 6 the whole stack is the one
# binary (session, compiler, runtime FFI) — no bun subprocess.
mcp-lean: build lean
	./lean/.lake/build/bin/frontend

clean:
	rm -rf $(BUILD_DIR)

# Full gate: the Lean golden/equiv runner (replaces validate_stdlib + the
# native equiv suite), the surviving behavioral bun suites (WASM≡JIT in
# tests/web + the MCP protocol tests against the Lean engine), and ctest.
# NB: run the BUILT binaries directly, never `lake exe` — lake forces the Lean
# lib dir onto DYLD_LIBRARY_PATH (for libleanshared), which shadows Homebrew's
# libLLVM with Lean's (no AMDGPU target) and the lld wasm emitter then dies at
# load (_LLVMInitializeAMDGPUAsmParser). See CLAUDE.md "Test".
validate: build lean
	cd lean && PATH="$$HOME/.elan/bin:$$PATH" lake build diffcli tropicaltest trustreport
	$(PYTHON) tools/audit_trust_sites.py
	./lean/.lake/build/bin/tropicaltest
	bun web/build_patches.ts
	TROPICAL_ENGINE_CMD="./lean/.lake/build/bin/frontend --rpc" bun test
	ctest --test-dir $(BUILD_DIR) --output-on-failure

# Behavioral MCP protocol suites against the live Lean engine.
test-lean-engine: build lean
	TROPICAL_ENGINE_CMD="./lean/.lake/build/bin/frontend --rpc" bun test mcp/errors.test.ts mcp/wire_dac.test.ts

# Native Reversible gates. SwiftUI is a macOS surface; Linux CI must not
# pretend to compile it.
reversible-build:
	swift build --package-path reversible

reversible-test:
	swift test --package-path reversible

# macOS release packaging. The bundle gate compiles a live grouped-room graph
# after relocating the app beneath the system temporary directory.
reversible-bundle: build lean
	reversible/scripts/bundle-app

reversible-bundle-smoke: reversible-bundle
	node reversible/scripts/smoke-bundle.mjs
