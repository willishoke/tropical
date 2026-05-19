# Makefile — build the inlining-across-modules benchmark variants.
#
# Each variant × chain length is a separate binary:
#   v1_chain4   v1_chain16    monolithic, no LTO
#   v1lto_chain4   v1lto_chain16   monolithic, with LTO
#   v3_chain4   v3_chain16    separate modules, no LTO
#   v4_chain4   v4_chain16    separate modules, with LTO
#
# Targets:
#   all          — build all variants × chain lengths
#   ir           — dump optimized LLVM IR for inspection
#   run          — run all binaries; append rows to data/results.csv
#   clean        — remove build artifacts

CC      ?= clang
OPT_FLAGS = -O2
CFLAGS    = $(OPT_FLAGS) -Wall -Wextra -fno-omit-frame-pointer -Isrc
LTO_FLAGS = -flto

BUILD = build
SRC   = src

# ─── Chain-length 4 ──────────────────────────────────────────────────────

# v1 chain4: kernel + scheduler in same TU, kernel inlined, no LTO.
$(BUILD)/v1_chain4: $(SRC)/v1_monolithic.c $(SRC)/runner.c | $(BUILD)
	$(CC) $(CFLAGS) -DBENCH_VARIANT_V1 -DBENCH_CHAIN_LEN=4 -o $@ $^

# v1lto chain4: same source as v1, compiled with -flto.
$(BUILD)/v1lto_chain4_monolithic.o: $(SRC)/v1_monolithic.c | $(BUILD)
	$(CC) $(CFLAGS) $(LTO_FLAGS) -c -o $@ $<
$(BUILD)/v1lto_chain4_runner.o: $(SRC)/runner.c | $(BUILD)
	$(CC) $(CFLAGS) $(LTO_FLAGS) -DBENCH_VARIANT_V1LTO -DBENCH_CHAIN_LEN=4 -c -o $@ $<
$(BUILD)/v1lto_chain4: $(BUILD)/v1lto_chain4_monolithic.o $(BUILD)/v1lto_chain4_runner.o
	$(CC) $(CFLAGS) $(LTO_FLAGS) -o $@ $^

# v3 chain4: kernel as external function, scheduler in separate TU, no LTO.
$(BUILD)/v3_chain4_kernel.o: $(SRC)/v3_kernel_module.c | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<
$(BUILD)/v3_chain4_scheduler.o: $(SRC)/v3_scheduler.c | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<
$(BUILD)/v3_chain4_runner.o: $(SRC)/runner.c | $(BUILD)
	$(CC) $(CFLAGS) -DBENCH_VARIANT_V3 -DBENCH_CHAIN_LEN=4 -c -o $@ $<
$(BUILD)/v3_chain4: $(BUILD)/v3_chain4_kernel.o $(BUILD)/v3_chain4_scheduler.o $(BUILD)/v3_chain4_runner.o
	$(CC) $(CFLAGS) -o $@ $^

# v4 chain4: same sources as v3, compiled with -flto.
$(BUILD)/v4_chain4_kernel.o: $(SRC)/v3_kernel_module.c | $(BUILD)
	$(CC) $(CFLAGS) $(LTO_FLAGS) -c -o $@ $<
$(BUILD)/v4_chain4_scheduler.o: $(SRC)/v3_scheduler.c | $(BUILD)
	$(CC) $(CFLAGS) $(LTO_FLAGS) -c -o $@ $<
$(BUILD)/v4_chain4_runner.o: $(SRC)/runner.c | $(BUILD)
	$(CC) $(CFLAGS) $(LTO_FLAGS) -DBENCH_VARIANT_V4 -DBENCH_CHAIN_LEN=4 -c -o $@ $<
$(BUILD)/v4_chain4: $(BUILD)/v4_chain4_kernel.o $(BUILD)/v4_chain4_scheduler.o $(BUILD)/v4_chain4_runner.o
	$(CC) $(CFLAGS) $(LTO_FLAGS) -o $@ $^

# ─── Chain-length 16 ─────────────────────────────────────────────────────

# v1 chain16: 16-kernel chain in one TU, no LTO.
$(BUILD)/v1_chain16: $(SRC)/v1_monolithic_16.c $(SRC)/runner.c | $(BUILD)
	$(CC) $(CFLAGS) -DBENCH_VARIANT_V1 -DBENCH_CHAIN_LEN=16 -o $@ $^

# v1lto chain16: same source, with -flto.
$(BUILD)/v1lto_chain16_monolithic.o: $(SRC)/v1_monolithic_16.c | $(BUILD)
	$(CC) $(CFLAGS) $(LTO_FLAGS) -c -o $@ $<
$(BUILD)/v1lto_chain16_runner.o: $(SRC)/runner.c | $(BUILD)
	$(CC) $(CFLAGS) $(LTO_FLAGS) -DBENCH_VARIANT_V1LTO -DBENCH_CHAIN_LEN=16 -c -o $@ $<
$(BUILD)/v1lto_chain16: $(BUILD)/v1lto_chain16_monolithic.o $(BUILD)/v1lto_chain16_runner.o
	$(CC) $(CFLAGS) $(LTO_FLAGS) -o $@ $^

# v3 chain16: 16-kernel chain, scheduler in separate TU, no LTO.
$(BUILD)/v3_chain16_kernel.o: $(SRC)/v3_kernel_module.c | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<
$(BUILD)/v3_chain16_scheduler.o: $(SRC)/v3_scheduler_16.c | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<
$(BUILD)/v3_chain16_runner.o: $(SRC)/runner.c | $(BUILD)
	$(CC) $(CFLAGS) -DBENCH_VARIANT_V3 -DBENCH_CHAIN_LEN=16 -c -o $@ $<
$(BUILD)/v3_chain16: $(BUILD)/v3_chain16_kernel.o $(BUILD)/v3_chain16_scheduler.o $(BUILD)/v3_chain16_runner.o
	$(CC) $(CFLAGS) -o $@ $^

# v4 chain16: same sources, with -flto.
$(BUILD)/v4_chain16_kernel.o: $(SRC)/v3_kernel_module.c | $(BUILD)
	$(CC) $(CFLAGS) $(LTO_FLAGS) -c -o $@ $<
$(BUILD)/v4_chain16_scheduler.o: $(SRC)/v3_scheduler_16.c | $(BUILD)
	$(CC) $(CFLAGS) $(LTO_FLAGS) -c -o $@ $<
$(BUILD)/v4_chain16_runner.o: $(SRC)/runner.c | $(BUILD)
	$(CC) $(CFLAGS) $(LTO_FLAGS) -DBENCH_VARIANT_V4 -DBENCH_CHAIN_LEN=16 -c -o $@ $<
$(BUILD)/v4_chain16: $(BUILD)/v4_chain16_kernel.o $(BUILD)/v4_chain16_scheduler.o $(BUILD)/v4_chain16_runner.o
	$(CC) $(CFLAGS) $(LTO_FLAGS) -o $@ $^

# ─── Alive variants (chain4 only for now) ───────────────────────────────

# v1 chain4 alive: kernel + scheduler same TU, alive check inlined,
# default-alive write each sample, no LTO. Expect GVN to fold the
# (alive > 0.5) checks since alive is constant-written 1.0 each sample.
$(BUILD)/v1_chain4_alive: $(SRC)/v1_monolithic_alive.c $(SRC)/runner_alive.c | $(BUILD)
	$(CC) $(CFLAGS) -DBENCH_VARIANT_V1_ALIVE -o $@ $^

# v1lto chain4 alive: same source, with LTO. Expect to match v1_alive
# (LTO has nothing to do beyond what -O2 already did intra-TU).
$(BUILD)/v1lto_chain4_alive_mono.o: $(SRC)/v1_monolithic_alive.c | $(BUILD)
	$(CC) $(CFLAGS) $(LTO_FLAGS) -c -o $@ $<
$(BUILD)/v1lto_chain4_alive_runner.o: $(SRC)/runner_alive.c | $(BUILD)
	$(CC) $(CFLAGS) $(LTO_FLAGS) -DBENCH_VARIANT_V1LTO_ALIVE -c -o $@ $<
$(BUILD)/v1lto_chain4_alive: $(BUILD)/v1lto_chain4_alive_mono.o $(BUILD)/v1lto_chain4_alive_runner.o
	$(CC) $(CFLAGS) $(LTO_FLAGS) -o $@ $^

# v3 chain4 alive: kernel as external function with alive check,
# scheduler writes alive=1.0 each sample, no LTO. Expect the alive
# check to STAY as a real runtime branch — the scheduler TU can't see
# into the kernel function, so it can't propagate the constant.
$(BUILD)/v3_chain4_alive_kernel.o: $(SRC)/v3_kernel_module_alive.c | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<
$(BUILD)/v3_chain4_alive_scheduler.o: $(SRC)/v3_scheduler_alive.c | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<
$(BUILD)/v3_chain4_alive_runner.o: $(SRC)/runner_alive.c | $(BUILD)
	$(CC) $(CFLAGS) -DBENCH_VARIANT_V3_ALIVE -c -o $@ $<
$(BUILD)/v3_chain4_alive: $(BUILD)/v3_chain4_alive_kernel.o $(BUILD)/v3_chain4_alive_scheduler.o $(BUILD)/v3_chain4_alive_runner.o
	$(CC) $(CFLAGS) -o $@ $^

# v4 chain4 alive: same sources as v3_alive, with -flto. THE KEY
# QUESTION: does LTO let GVN fold the alive check across the module
# boundary?
$(BUILD)/v4_chain4_alive_kernel.o: $(SRC)/v3_kernel_module_alive.c | $(BUILD)
	$(CC) $(CFLAGS) $(LTO_FLAGS) -c -o $@ $<
$(BUILD)/v4_chain4_alive_scheduler.o: $(SRC)/v3_scheduler_alive.c | $(BUILD)
	$(CC) $(CFLAGS) $(LTO_FLAGS) -c -o $@ $<
$(BUILD)/v4_chain4_alive_runner.o: $(SRC)/runner_alive.c | $(BUILD)
	$(CC) $(CFLAGS) $(LTO_FLAGS) -DBENCH_VARIANT_V4_ALIVE -c -o $@ $<
$(BUILD)/v4_chain4_alive: $(BUILD)/v4_chain4_alive_kernel.o $(BUILD)/v4_chain4_alive_scheduler.o $(BUILD)/v4_chain4_alive_runner.o
	$(CC) $(CFLAGS) $(LTO_FLAGS) -o $@ $^

# ─── IR dumps ────────────────────────────────────────────────────────────

$(BUILD)/v1_chain4.ll: $(SRC)/v1_monolithic.c | $(BUILD)
	$(CC) $(CFLAGS) -S -emit-llvm -DBENCH_VARIANT_V1 -DBENCH_CHAIN_LEN=4 -o $@ $<
$(BUILD)/v1_chain16.ll: $(SRC)/v1_monolithic_16.c | $(BUILD)
	$(CC) $(CFLAGS) -S -emit-llvm -DBENCH_VARIANT_V1 -DBENCH_CHAIN_LEN=16 -o $@ $<
$(BUILD)/v3_kernel.ll: $(SRC)/v3_kernel_module.c | $(BUILD)
	$(CC) $(CFLAGS) -S -emit-llvm -o $@ $<
$(BUILD)/v3_chain4_scheduler.ll: $(SRC)/v3_scheduler.c | $(BUILD)
	$(CC) $(CFLAGS) -S -emit-llvm -o $@ $<
$(BUILD)/v3_chain16_scheduler.ll: $(SRC)/v3_scheduler_16.c | $(BUILD)
	$(CC) $(CFLAGS) -S -emit-llvm -o $@ $<

$(BUILD)/v1_chain4_alive.ll: $(SRC)/v1_monolithic_alive.c | $(BUILD)
	$(CC) $(CFLAGS) -S -emit-llvm -DBENCH_VARIANT_V1_ALIVE -o $@ $<
$(BUILD)/v3_alive_kernel.ll: $(SRC)/v3_kernel_module_alive.c | $(BUILD)
	$(CC) $(CFLAGS) -S -emit-llvm -o $@ $<
$(BUILD)/v3_chain4_alive_scheduler.ll: $(SRC)/v3_scheduler_alive.c | $(BUILD)
	$(CC) $(CFLAGS) -S -emit-llvm -o $@ $<

ir: $(BUILD)/v1_chain4.ll $(BUILD)/v1_chain16.ll \
    $(BUILD)/v3_kernel.ll $(BUILD)/v3_chain4_scheduler.ll $(BUILD)/v3_chain16_scheduler.ll \
    $(BUILD)/v1_chain4_alive.ll $(BUILD)/v3_alive_kernel.ll $(BUILD)/v3_chain4_alive_scheduler.ll

# ─── Aggregate targets ───────────────────────────────────────────────────

BINARIES = \
    $(BUILD)/v1_chain4    $(BUILD)/v1lto_chain4 \
    $(BUILD)/v3_chain4    $(BUILD)/v4_chain4 \
    $(BUILD)/v1_chain16   $(BUILD)/v1lto_chain16 \
    $(BUILD)/v3_chain16   $(BUILD)/v4_chain16 \
    $(BUILD)/v1_chain4_alive    $(BUILD)/v1lto_chain4_alive \
    $(BUILD)/v3_chain4_alive    $(BUILD)/v4_chain4_alive

all: $(BINARIES) ir

run: all
	@if [ ! -f data/results.csv ]; then \
	  echo "variant,buf_len,iters,ns_per_sample,sink_value" > data/results.csv; \
	fi
	@for b in $(BINARIES); do $$b >> data/results.csv; done
	@echo "appended results to data/results.csv"
	@tail -12 data/results.csv

$(BUILD):
	@mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)

.PHONY: all ir run clean
