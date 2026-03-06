.PHONY: build repl run clean

ROOT := /Users/willishoke/egress
BUILD_DIR := $(ROOT)/build
PYTHON := /opt/homebrew/bin/python3
PYBIND11_DIR := /opt/homebrew/Caskroom/miniforge/base/lib/python3.9/site-packages/pybind11/share/cmake/pybind11

build:
	cmake -S $(ROOT) -B $(BUILD_DIR) -DEGRESS_BUILD_PYTHON=ON -DPython3_EXECUTABLE=$(PYTHON) -Dpybind11_DIR=$(PYBIND11_DIR)
	cmake --build $(BUILD_DIR) -j4

repl: build
	PYTHONPATH=$(BUILD_DIR) $(PYTHON)

run: repl

clean:
	rm -rf $(BUILD_DIR)
