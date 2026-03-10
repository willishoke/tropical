.PHONY: build profile jit jit-profile repl run clean

ROOT := /Users/willishoke/egress
BUILD_DIR := $(ROOT)/build
PROFILE_BUILD_DIR := $(ROOT)/build-profile
JIT_BUILD_DIR := $(ROOT)/build-jit
JIT_PROFILE_BUILD_DIR := $(ROOT)/build-jit-profile

PYTHON := /opt/homebrew/bin/python3
PYBIND11_DIR := /opt/homebrew/Caskroom/miniforge/base/lib/python3.9/site-packages/pybind11/share/cmake/pybind11
LLVM_DIR ?= /opt/homebrew/opt/llvm/lib/cmake/llvm

JOBS ?= 4
EXTRA_CMAKE_ARGS ?=

define configure_and_build
	cmake -S $(ROOT) -B $(1) \
		-DEGRESS_BUILD_PYTHON=ON \
		-DEGRESS_PROFILE=$(2) \
		-DEGRESS_LLVM_ORC_JIT=$(3) \
		-DPython3_EXECUTABLE=$(PYTHON) \
		-Dpybind11_DIR=$(PYBIND11_DIR) \
		$(if $(filter ON,$(3)),-DLLVM_DIR=$(LLVM_DIR),) \
		$(EXTRA_CMAKE_ARGS)
	cmake --build $(1) -j$(JOBS)
endef

build:
	$(call configure_and_build,$(BUILD_DIR),OFF,OFF)

profile:
	$(call configure_and_build,$(PROFILE_BUILD_DIR),ON,OFF)

jit:
	$(call configure_and_build,$(JIT_BUILD_DIR),OFF,ON)

jit-profile:
	$(call configure_and_build,$(JIT_PROFILE_BUILD_DIR),ON,ON)

repl: build
	PYTHONPATH=$(BUILD_DIR) $(PYTHON)

run: repl

clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(PROFILE_BUILD_DIR)
	rm -rf $(JIT_BUILD_DIR)
	rm -rf $(JIT_PROFILE_BUILD_DIR)
