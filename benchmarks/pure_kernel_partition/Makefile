CXX        ?= clang++
CXXFLAGS   ?= -O3 -std=c++20 -fobjc-arc -Wall -Wextra -Wno-deprecated-declarations
FRAMEWORKS := -framework Metal -framework Foundation

bench: pure_kernel_partition.mm
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) $< -o $@

run: bench
	./bench

clean:
	rm -f bench

.PHONY: run clean
