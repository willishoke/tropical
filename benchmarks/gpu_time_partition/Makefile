# GPU time-partition latency benchmark — Apple Silicon / Metal, unified memory.
# Companion to ../simd_time_partition.cpp: the same closed-form-in-τ kernel, but a
# realtime block is dispatched across the GPU's lanes instead of 4 NEON lanes. Metal is
# reached at runtime (newLibraryWithSource), so there is no offline .metal compile step —
# just Objective-C++ linked against the Metal and Foundation frameworks.

CXX        ?= clang++
CXXFLAGS   ?= -O3 -std=c++20 -fobjc-arc -Wall -Wno-deprecated-declarations
FRAMEWORKS := -framework Metal -framework Foundation

bench: gpu_time_partition.mm
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) $< -o $@

run: bench
	./bench

clean:
	rm -f bench

.PHONY: run clean
