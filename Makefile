.PHONY: build profile repl run mcp-ts clean

ROOT := $(shell pwd)
BUILD_DIR := $(ROOT)/build
PROFILE_BUILD_DIR := $(ROOT)/build-profile

PYTHON := $(shell which python3)
LLVM_DIR ?= /opt/homebrew/opt/llvm/lib/cmake/llvm

JOBS ?= 4
EXTRA_CMAKE_ARGS ?=

define configure_and_build
	cmake -S $(ROOT) -B $(1) \
		-DTROPICAL_BUILD_PYTHON=ON \
		-DTROPICAL_PROFILE=$(2) \
		-DLLVM_DIR=$(LLVM_DIR) \
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

mcp-ts: build
	bun install --silent && bun run mcp/server.ts

clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(PROFILE_BUILD_DIR)
