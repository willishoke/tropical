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
// v1 dispatch is synchronous per block (encode → commit → wait inside
// process_block), keeping process()'s buffer-filled-on-return contract so
// fades, sample_index, and render_window semantics are untouched.

#include <cstdint>
#include <memory>
#include <string>

namespace tropical_metal
{

struct MetalKernel;  // opaque — ObjC++ members live in the .mm

using MetalKernelPtr = std::shared_ptr<MetalKernel>;

/// Compile MSL source and build the pipeline + buffers. Control-thread only.
/// `buffer_length` sizes the output buffer; `slot_count` sizes the slot
/// snapshot. Returns nullptr on failure with the compiler diagnostic in
/// `err`.
MetalKernelPtr create(const std::string & msl_source,
                      uint32_t buffer_length,
                      uint32_t slot_count,
                      std::string & err);

/// Render one block synchronously: snapshot slots (f64→f32), encode one
/// thread per sample, commit, wait, widen the f32 output into `out_f64`.
/// Returns false on a device/command-buffer error (caller emits silence).
bool process_block(MetalKernel & k,
                   const double * slots, uint32_t n_slots,
                   double sample_rate, uint64_t start_sample_index,
                   double * out_f64, uint32_t len);

} // namespace tropical_metal
