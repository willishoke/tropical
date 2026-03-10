.PHONY: build profile repl run clean

ROOT := /Users/willishoke/egress
BUILD_DIR := $(ROOT)/build
PROFILE_BUILD_DIR := $(ROOT)/build-profile
PYTHON := /opt/homebrew/bin/python3
PYBIND11_DIR := /opt/homebrew/Caskroom/miniforge/base/lib/python3.9/site-packages/pybind11/share/cmake/pybind11

build:
	cmake -S $(ROOT) -B $(BUILD_DIR) -DEGRESS_BUILD_PYTHON=ON -DEGRESS_PROFILE=OFF -DPython3_EXECUTABLE=$(PYTHON) -Dpybind11_DIR=$(PYBIND11_DIR)
	cmake --build $(BUILD_DIR) -j4

profile:
	cmake -S $(ROOT) -B $(PROFILE_BUILD_DIR) -DEGRESS_BUILD_PYTHON=ON -DEGRESS_PROFILE=ON -DPython3_EXECUTABLE=$(PYTHON) -Dpybind11_DIR=$(PYBIND11_DIR)
	cmake --build $(PROFILE_BUILD_DIR) -j4

repl: build
	PYTHONPATH=$(BUILD_DIR) $(PYTHON)

run: repl

clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(PROFILE_BUILD_DIR)
