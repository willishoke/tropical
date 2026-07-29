#pragma once
// MetalKernel — the GPU execution backend behind FlatRuntime (pure C++
// interface; the implementation is ObjC++ in MetalKernel.mm, gated by
// TROPICAL_METAL).
//
// One MetalKernel = one compiled tropical kernel: MSL library (compiled at
// runtime via newLibraryWithSource with MTLMathModeSafe — the numeric
// contract: IEEE-ordered f32, no fma contraction), one compute pipeline
// state, one shared output buffer, one shared slot-snapshot buffer. It rides
// inside KernelState and swaps with it — the existing double-buffered
// publish/flip IS the Metal hot-swap; compilation happens on the control
// thread like the LLVM JIT.
//
// Live epoch rendering uses render_tile() from a dedicated worker. The legacy
// process_block entry point remains temporarily while the callback cutover is
// staged; both fail closed if invoked from a thread marked as an audio callback.

#include <cstdint>
#include <memory>
#include <string>

namespace tropical_metal
{

struct MetalKernel;  // opaque — ObjC++ members live in the .mm

using MetalKernelPtr = std::shared_ptr<MetalKernel>;

/// Compile MSL source and build the pipeline + buffers. Control-thread only.
/// `buffer_length` sizes the output buffer; `slot_count` sizes the slot
/// snapshot; `column_count` sizes the packed coefficient-column buffer
/// (`buffer(3)`, banks-as-data) — 0 for plans without hoisted columns
/// (the kernel then has the plain 3-binding ABI and no column buffer is
/// allocated or bound). Returns nullptr on failure with the compiler
/// diagnostic in `err`.
MetalKernelPtr create(const std::string & msl_source,
                      uint32_t buffer_length,
                      uint32_t slot_count,
                      uint32_t column_count,
                      std::string & err);

/// Worker-only blocking primitive. `slots` and `columns` are immutable f32
/// snapshots owned by one render epoch request. The destination is stable,
/// preallocated tile storage. A command-buffer failure permanently latches the
/// kernel closed.
bool render_tile(MetalKernel & k,
                 const float * slots, uint32_t n_slots,
                 const float * columns, uint32_t n_columns,
                 double sample_rate, uint64_t source_start,
                 uint32_t frames, double * destination);

/// Render one block synchronously: snapshot slots (f64→f32) and the packed
/// coefficient columns (`columns` — already narrowed to f32 by the caller
/// from ONE whole captured generation), encode one thread per sample,
/// commit, wait, widen the f32 output into `out_f64`. Returns false on a
/// device/command-buffer error before copying output. A failure latches this
/// kernel closed: later calls return false immediately until control publishes
/// a fresh MetalKernel (caller emits silence throughout).
bool process_block(MetalKernel & k,
                   const double * slots, uint32_t n_slots,
                   const float * columns, uint32_t n_columns,
                   double sample_rate, uint64_t start_sample_index,
                   double * out_f64, uint32_t len);

/// Consume the one-shot report for a newly latched dispatch failure. Once
/// latched, process_block remains false until this MetalKernel is replaced;
/// consuming the report does not clear the latch.
bool take_dispatch_failure(MetalKernel & k);

/// Qualification diagnostic: zero for synchronous dispatch, otherwise the
/// configured number of future blocks.  This is deliberately runtime-only;
/// it does not alter the plan or protocol schema.
uint32_t pipeline_depth(const MetalKernel & k);

/// Deterministic provenance seam. Production callback code marks itself for
/// the duration of FlatRuntime::process(); tests may also mark a thread
/// directly. Every Metal submit/wait entry point refuses the call and records
/// one violation while marked.
void set_audio_callback_thread(bool value);
uint64_t callback_thread_violation_count();
void reset_callback_thread_violation_count();

} // namespace tropical_metal
