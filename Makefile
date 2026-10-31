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
# TROPICAL_WASM_EMIT=ON builds the WebAssembly backend + lld into
# libtropical_wasm (the toolchain library only `diffcli compile-wasm` links;
# libtropical, the runtime, never carries it) so the web build works out of
# the box; requires `brew install lld`. TROPICAL_METAL=ON builds the GPU
# backend (macOS only; runtime MSL compile, no Xcode needed). The CMake
# defaults are OFF, so a slim build is
# `cmake ... -DTROPICAL_WASM_EMIT=OFF -DTROPICAL_METAL=OFF` via EXTRA_CMAKE_ARGS.

repl: build
	PYTHONPATH=$(BUILD_DIR) $(PYTHON)

run: repl

# Lean front-door. Builds the lean/ subtree (the production compiler + MCP
# server, one binary), which pulls Turnstile via Lake. Builds every declared
# executables, not just the default `frontend`: the gate runners are meant
# to be invoked as built binaries (never `lake exe`), and a bare
# `lake build` leaves stale tropicaltest/diffcli binaries in place.
lean:
	cd lean && PATH="$$HOME/.elan/bin:$$PATH" lake env leanc -c ffi/shim.c -o ffi/shim.o -I ../engine/c_api
	cd lean && PATH="$$HOME/.elan/bin:$$PATH" lake env leanc -c ffi/shim_wasm.c -o ffi/shim_wasm.o -I ../engine/toolchain
	cd lean && PATH="$$HOME/.elan/bin:$$PATH" lake build frontend diffcli tropicaltest trustreport tropicalc tropicaltest-core

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
# libLLVM with Lean's, and any binary carrying the lld wasm emitter (diffcli,
# via libtropical_wasm) then dies at load. See CLAUDE.md "Test".
validate: build lean
	cd lean && PATH="$$HOME/.elan/bin:$$PATH" lake build diffcli tropicaltest trustreport tropicalc tropicaltest-core
	$(PYTHON) tools/audit_trust_sites.py
	$(PYTHON) tools/check_layers.py
	$(PYTHON) tools/check_artifact_parity.py
	$(PYTHON) tools/check_runtime_deps.py
	bun run tsc --noEmit
	./lean/.lake/build/bin/tropicaltest-core
	./lean/.lake/build/bin/tropicaltest
	bun web/build_patches.ts
	TROPICAL_ENGINE_CMD="./lean/.lake/build/bin/frontend --rpc" bun test
	ctest --test-dir $(BUILD_DIR) --output-on-failure
	$(PYTHON) benchmarks/metal_live/test_run.py
	$(PYTHON) benchmarks/metal_live/run.py --mode smoke --skip-build

# Behavioral MCP protocol suites against the live Lean engine.
test-lean-engine: build lean
	TROPICAL_ENGINE_CMD="./lean/.lake/build/bin/frontend --rpc" bun test mcp/errors.test.ts mcp/wire_dac.test.ts

# Native Reversible gates. SwiftUI is a macOS surface; Linux CI must not
# pretend to compile it.
reversible-build:
	swift build --package-path reversible

reversible-test:
	swift test --package-path reversible

# macOS release packaging. The bundle gate compiles a canonical source graph
# after relocating the app beneath the system temporary directory.
reversible-bundle: build lean
	reversible/scripts/bundle-app

reversible-bundle-smoke: reversible-bundle
	node reversible/scripts/smoke-bundle.mjs
