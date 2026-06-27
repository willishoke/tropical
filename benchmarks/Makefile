CXX      ?= clang++
# loop-vectorizer is ON (for the "auto-vec" bar C); -fno-slp-vectorize keeps the
# scalar baseline A genuinely scalar; -ffp-contract=off pins every path to pure
# mul-add so the three agree bit-for-bit. The explicit-SIMD path B uses vector
# types, unaffected by any of these. Add -march=native (x86) / -mcpu=native (arm)
# for the host's widest SIMD (AVX2/AVX-512).
CXXFLAGS ?= -O3 -std=c++20 -fno-slp-vectorize -ffp-contract=off

bench: simd_time_partition.cpp            # float, rich kernel (8 partials x 8 taps)
	$(CXX) $(CXXFLAGS) $< -o $@

bench_d: simd_time_partition.cpp          # double
	$(CXX) $(CXXFLAGS) -DBENCH_T=double $< -o $@

bench_thin: simd_time_partition.cpp       # single oscillator: time is the ONLY parallel axis
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
