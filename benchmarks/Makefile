CXX      ?= clang++
# loop-vectorizer is ON (for the "auto-vec" path C); -fno-slp-vectorize keeps the
# recurrent baseline R from getting within-sample SLP it can't get across time;
# -ffp-contract=off pins every path to pure mul-add so C and B agree bit-for-bit.
# The explicit-SIMD path B uses vector types, unaffected by any of these. Add
# -march=native (x86) / -mcpu=native (arm) for the host's widest SIMD.
CXXFLAGS ?= -O3 -std=c++20 -fno-slp-vectorize -ffp-contract=off

bench: simd_time_partition.cpp            # float, rich kernel (8 partials x 8 offsets)
	$(CXX) $(CXXFLAGS) $< -o $@

bench_d: simd_time_partition.cpp          # double
	$(CXX) $(CXXFLAGS) -DBENCH_T=double $< -o $@

bench_thin: simd_time_partition.cpp       # single partial: time is the ONLY parallel axis
	$(CXX) $(CXXFLAGS) -DBENCH_M=1 -DBENCH_K=1 $< -o $@

run: bench bench_d bench_thin
	./bench
	@echo
	./bench_d
	@echo
	./bench_thin

clean:
	rm -f bench bench_d bench_thin

.PHONY: run clean
