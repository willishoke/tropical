# Makefile — build the inlining-across-modules benchmark variants.
#
# Each variant is a separate binary; the variants differ in WHICH
# scheduler implementation gets linked in and at WHAT optimization
# level / linkage mode. The runner is shared.
#
# Available targets:
#   all          — build all variants
#   v1           — monolithic (one TU)
#   v3-nolto     — separate modules, no LTO
#   ir           — dump optimized LLVM IR for each variant
#   run          — run all variants and append to data/results.csv
#   clean        — remove build artifacts

CC      ?= clang
OPT_FLAGS = -O2

# Compile flags for object files. -ffast-math is intentionally OFF —
# we want measurements consistent with tropical's JIT, which does not
# enable fast-math. -fno-omit-frame-pointer makes profiling easier.
CFLAGS = $(OPT_FLAGS) -Wall -Wextra -fno-omit-frame-pointer -Isrc

BUILD = build
SRC   = src

# Variant 1: monolithic. Single object linking the runner + v1
# scheduler. The kernel is inlined at the call site by definition.
# Each .c file becomes its own TU; the linker connects them. WITHOUT
# LTO, the optimizer at runner.c's TU sees scheduler_v1 only as an
# external function — it CAN'T inline the scheduler into the runner's
# outer loop.
$(BUILD)/v1_runner: $(SRC)/v1_monolithic.c $(SRC)/runner.c | $(BUILD)
	$(CC) $(CFLAGS) -DBENCH_VARIANT_V1 -o $@ $^

# Variant 1+LTO: same monolithic source, with -flto. The kernel is
# already inlined into scheduler_v1 by definition; with LTO, the
# linker can additionally inline scheduler_v1 into runner's loop.
# This isolates the "outer-loop inlining" effect from the "kernel
# inlining" effect.
$(BUILD)/v1lto_monolithic.o: $(SRC)/v1_monolithic.c | $(BUILD)
	$(CC) $(CFLAGS) $(LTO_FLAGS) -c -o $@ $<

$(BUILD)/v1lto_runner.o: $(SRC)/runner.c | $(BUILD)
	$(CC) $(CFLAGS) $(LTO_FLAGS) -DBENCH_VARIANT_V1LTO -c -o $@ $<

$(BUILD)/v1lto_runner: $(BUILD)/v1lto_monolithic.o $(BUILD)/v1lto_runner.o
	$(CC) $(CFLAGS) $(LTO_FLAGS) -o $@ $^

# Variant 3: separate modules, no LTO. Each kernel is its own TU,
# compiled WITHOUT -flto. The optimizer at each TU can't see across.
$(BUILD)/v3_kernel_module.o: $(SRC)/v3_kernel_module.c | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/v3_scheduler.o: $(SRC)/v3_scheduler.c | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/v3_runner.o: $(SRC)/runner.c | $(BUILD)
	$(CC) $(CFLAGS) -DBENCH_VARIANT_V3 -c -o $@ $<

$(BUILD)/v3_runner: $(BUILD)/v3_kernel_module.o $(BUILD)/v3_scheduler.o $(BUILD)/v3_runner.o
	$(CC) $(CFLAGS) -o $@ $^

# Variant 4: separate modules, WITH LTO. Same source as v3; compile and
# link both with -flto so the optimizer sees across TU boundaries at
# link time. Reuses v3's source files (we just rename the symbols by
# defining BENCH_VARIANT_V3 — both v3 and v4 use scheduler_v3).
LTO_FLAGS = -flto
$(BUILD)/v4_kernel_module.o: $(SRC)/v3_kernel_module.c | $(BUILD)
	$(CC) $(CFLAGS) $(LTO_FLAGS) -c -o $@ $<

$(BUILD)/v4_scheduler.o: $(SRC)/v3_scheduler.c | $(BUILD)
	$(CC) $(CFLAGS) $(LTO_FLAGS) -c -o $@ $<

$(BUILD)/v4_runner.o: $(SRC)/runner.c | $(BUILD)
	$(CC) $(CFLAGS) $(LTO_FLAGS) -DBENCH_VARIANT_V4 -c -o $@ $<

$(BUILD)/v4_runner: $(BUILD)/v4_kernel_module.o $(BUILD)/v4_scheduler.o $(BUILD)/v4_runner.o
	$(CC) $(CFLAGS) $(LTO_FLAGS) -o $@ $^

# IR dumps for inspection. The .ll files show the optimized LLVM IR
# that's actually being executed; they're the primary evidence for
# whether the alive check / state hoisting / inlining survived.
$(BUILD)/v1.ll: $(SRC)/v1_monolithic.c | $(BUILD)
	$(CC) $(CFLAGS) -S -emit-llvm -DBENCH_VARIANT_V1 -o $@ $<

$(BUILD)/v3_kernel.ll: $(SRC)/v3_kernel_module.c | $(BUILD)
	$(CC) $(CFLAGS) -S -emit-llvm -o $@ $<

$(BUILD)/v3_scheduler.ll: $(SRC)/v3_scheduler.c | $(BUILD)
	$(CC) $(CFLAGS) -S -emit-llvm -o $@ $<

ir: $(BUILD)/v1.ll $(BUILD)/v3_kernel.ll $(BUILD)/v3_scheduler.ll

# Build everything.
all: $(BUILD)/v1_runner $(BUILD)/v1lto_runner $(BUILD)/v3_runner $(BUILD)/v4_runner ir

# Run all variants, append CSV rows to data/results.csv.
# Add the header if the file doesn't yet exist.
run: all
	@if [ ! -f data/results.csv ]; then \
	  echo "variant,buf_len,iters,ns_per_sample,sink_value" > data/results.csv; \
	fi
	@$(BUILD)/v1_runner >> data/results.csv
	@$(BUILD)/v1lto_runner >> data/results.csv
	@$(BUILD)/v3_runner >> data/results.csv
	@$(BUILD)/v4_runner >> data/results.csv
	@echo "appended results to data/results.csv"
	@tail -5 data/results.csv

$(BUILD):
	@mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)

.PHONY: all ir run clean
