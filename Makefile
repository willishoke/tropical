.PHONY: build repl run mcp-ts lean clean validate validate-write

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

mcp-ts: build
	bun install --silent && bun run mcp/server.ts

# Lean front-door (Turnstile). Builds the lean/ subtree, which pulls Turnstile
# via Lake and links against the IR service (mcp/ir_service.ts) at runtime.
lean:
	cd lean && PATH="$$HOME/.elan/bin:$$PATH" lake build

clean:
	rm -rf $(BUILD_DIR)

validate: build
	bun test
	ctest --test-dir $(BUILD_DIR) --output-on-failure
	bun run scripts/validate_stdlib.ts

validate-write: build
	bun run scripts/validate_stdlib.ts --write
