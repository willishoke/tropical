.PHONY: build profile repl run tui-ts clean

ROOT := $(shell pwd)
BUILD_DIR := $(ROOT)/build
PROFILE_BUILD_DIR := $(ROOT)/build-profile

PYTHON := $(shell which python3)
LLVM_DIR ?= /opt/homebrew/opt/llvm/lib/cmake/llvm

JOBS ?= 4
EXTRA_CMAKE_ARGS ?=

define configure_and_build
	cmake -S $(ROOT) -B $(1) \
		-DEGRESS_BUILD_PYTHON=ON \
		-DEGRESS_PROFILE=$(2) \
		-DEGRESS_LLVM_ORC_JIT=ON \
		-DLLVM_DIR=$(LLVM_DIR) \
		-DPython3_EXECUTABLE=$(PYTHON) \
		$(EXTRA_CMAKE_ARGS)
	cmake --build $(1) -j$(JOBS)
endef

build:
	$(call configure_and_build,$(BUILD_DIR),OFF)

profile:
	$(call configure_and_build,$(PROFILE_BUILD_DIR),ON)

repl: build
	PYTHONPATH=$(BUILD_DIR) $(PYTHON)

run: repl

tui-ts: build
	cd tui && bun install --silent && bun run src/index.tsx

mcp-ts: build
	cd tui && bun install --silent && bun run src/server.ts

clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(PROFILE_BUILD_DIR)
