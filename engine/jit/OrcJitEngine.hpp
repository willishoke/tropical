#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>

#include <llvm/Config/llvm-config.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/OptimizationLevel.h>

#include <filesystem>

namespace tropical_jit
{
#ifdef TROPICAL_WASM_EMIT
// Lower textual LLVM IR (Lean's EmitLlvm output) to a complete wasm32 module,
// fully in-process: a wasm32 TargetMachine emits an object, lld::wasm::link
// assembles it into a module. No subprocess, no PATH — the same LLVM the JIT
// uses, which makes native≡wasm bit-exactness structural. Returns the .wasm
// bytes; on failure returns empty and sets `err`.
std::vector<uint8_t> compile_ir_to_wasm(const std::string & ir_text, std::string & err);
#endif

enum class JitScalarType : uint8_t { Float, Int, Bool };

// ─────────────────────────────────────────────────────────────────────────────
// New flat instruction format (FlatProgram / FlatInstr)
//
// Terminals (literals, inputs, registers, params) are Operand kinds —
// not instructions. OpTag covers only genuine computations (~30 entries).
// loop_count > 1 triggers an elementwise loop; strides[i] controls whether
// args[i] advances with the loop index (1) or broadcasts (0).
// ─────────────────────────────────────────────────────────────────────────────

enum class OpTag : uint8_t
{
  // arity 2
  Add, Sub, Mul, Div, Mod, FloorDiv,
  Less, LessEq, Greater, GreaterEq, Equal, NotEqual,
  BitAnd, BitOr, BitXor, LShift, RShift,
  Index,    // args[0]=ArrayReg, args[1]=scalar idx → scalar element
  And, Or,  // logical (truthy coercion: float/int → bool, then and/or)
  // arity 1
  Neg, Abs, Sqrt, Floor, Ceil, Round, Not, BitNot,
  FloatExponent,  // extract IEEE-754 unbiased exponent of a float as a float-valued integer
  // Scalar-type cast ops. Truncate-toward-zero semantics (FPToSI), not floor —
  // a stdlib author wanting floor-to-int writes `to_int(floor(x))`.
  ToInt, ToBool, ToFloat,
  // arity 2 (listed here for locality with FloatExponent; Ldexp is binary)
  Ldexp,          // x * 2^n (n is a float-valued integer)
  // arity 3
  Clamp, Select,
  SetElement,  // args[0]=ArrayReg, args[1]=idx, args[2]=val; no dst slot written
  // arity N
  Pack,     // args = scalar values → arrays[dst]
  // M6+: write a computed value to slots[dst]. `dst` is the slot index;
  // `args[0]` is the value to write (scalar). No temp slot consumed.
  WriteSlot,
  // Indexed-reduction region delimiters (banks-as-data): dst = the
  // accumulator temp; instructions between Begin and End run once per
  // iteration (loop_count), with the `loop_idx` operand as the index.
  // Metadata-only here — codegen is Lean's (EmitLlvm/EmitMsl).
  ReduceBegin,
  ReduceEnd,
};

enum class OperandKind : uint8_t
{
  Const,    // floating-point constant (const_val field)
  Input,    // module input port (slot field)
  Reg,      // virtual register — scalar result in temps[slot]
  ArrayReg, // virtual register — array result in arrays[slot]
  Param,    // ControlParam pointer (ptr field)
  Source,   // input source `sources[slot]` — resolved by kind at runtime
            // (slot field holds the source index; sources[i].kind → tick|rate)
  Slot,     // shared inter-module slot (slot field, indexes slots[]) — M6+
};

struct Operand
{
  OperandKind   kind        = OperandKind::Const;
  JitScalarType scalar_type = JitScalarType::Float;
  double        const_val   = 0.0;  // Const
  uint32_t      slot        = 0;    // Input, Reg, ArrayReg, Source, Slot
  uint64_t      ptr         = 0;    // Param

  static Operand make_const(double v, JitScalarType t = JitScalarType::Float)
  { Operand o; o.kind = OperandKind::Const; o.scalar_type = t; o.const_val = v; return o; }
  static Operand make_input(uint32_t s, JitScalarType t = JitScalarType::Float)
  { Operand o; o.kind = OperandKind::Input; o.scalar_type = t; o.slot = s; return o; }
  static Operand make_reg(uint32_t id, JitScalarType t = JitScalarType::Float)
  { Operand o; o.kind = OperandKind::Reg; o.scalar_type = t; o.slot = id; return o; }
  static Operand make_array_reg(uint32_t s)
  { Operand o; o.kind = OperandKind::ArrayReg; o.slot = s; return o; }
  static Operand make_param(uint64_t p)
  { Operand o; o.kind = OperandKind::Param; o.scalar_type = JitScalarType::Float; o.ptr = p; return o; }
  static Operand make_source(uint32_t index, JitScalarType t = JitScalarType::Float)
  { Operand o; o.kind = OperandKind::Source; o.scalar_type = t; o.slot = index; return o; }
  // M6: slot operand reads from the shared slots[] arg passed to the
  // kernel. Stored as double, coerced at use site to match scalar_type.
  static Operand make_slot(uint32_t s, JitScalarType t = JitScalarType::Float)
  { Operand o; o.kind = OperandKind::Slot; o.scalar_type = t; o.slot = s; return o; }
};

/** Writeback-namespace discriminator. Mirrors the TS-side `DstSlot`
 *  union (`'temp' | 'array' | 'moduleSlot'`); the wire format carries
 *  this as a string tag (`dst_kind`) and the parser reconstructs it
 *  into this enum. Direct dispatch on this enum replaces the prior
 *  "`loop_count > 1` implies array dst" proxy — the proxy silently
 *  misclassified `loop_count==1` array writes (degenerate elementwise
 *  ops on size-1 arrays), storing scalar results into temp slots at
 *  array-slot indices. */
enum class DstKind : uint8_t {
  Temp,
  Array,
  ModuleSlot,
};

struct FlatInstr
{
  OpTag                tag         = OpTag::Add;
  JitScalarType        result_type = JitScalarType::Float;
  DstKind              dst_kind    = DstKind::Temp;
  uint32_t             dst         = 0;
  std::vector<Operand> args;
  uint32_t             loop_count  = 1;       // iteration count for elementwise emission
  std::vector<uint8_t> strides;               // per-arg: 1 = iterate, 0 = broadcast
};

// Per-instance kernel slice. Each session instance contributes one of
// these to the multi-function plan. The compiler emits all instances
// inline within one LLVM function — `children` are emitted recursively
// inside the parent's body (the fractal architecture: one kernel per
// InstanceDecl at every nesting depth).
struct InstanceProgram
{
  std::string                  instance_name;     // dotted path (e.g. voice1.env)
  /** Preamble — emitted at the START of this kernel's block, before
   *  any child dispatch. Holds temp-compute instructions for
   *  session-wired input translations whose results are referenced by
   *  child pre_input_instructions. Splitting this out of
   *  `instructions` is essential: child dispatch happens before
   *  `instructions`, so the temps that pre_input writes must already
   *  be computed by the time children start. */
  std::vector<FlatInstr>       preamble_instructions;
  std::vector<FlatInstr>       instructions;      // already shifted into unified offset space
  /** Slot-based parent→child input wiring. Per-child block — runs in
   *  the PARENT's namespace immediately before THIS kernel's body. The
   *  parent's emit_kernel_block dispatches each child as:
   *      emit_instrs(child.pre_input_instructions)
   *      emit_kernel_block(child)
   *  Per-child placement (vs hoisting into a parent-wide pre-children
   *  block) preserves sibling-to-sibling NestedOut dependencies: a
   *  later sibling's wire can read an earlier sibling's output slot
   *  because the earlier sibling's body has already executed. Empty
   *  for top-level kernels. */
  std::vector<FlatInstr>       pre_input_instructions;
  uint32_t                     register_count = 0; // local temp count
  /** Nested child kernels. Emitted recursively inside this kernel's
   *  body. Empty for leaf kernels. */
  std::vector<InstanceProgram> children;
};

// A device-bound output sink (the DAC, neutrally named). Reads its input
// module slots, sums them, scales by `gain`, and writes channel/device
// `target`. A family, not a singleton — multiple output devices are normal.
// v1 realizes target 0 → the single output buffer.
struct Sink
{
  std::vector<uint32_t> inputs;        // output module-slot indices
  double                gain   = 0.05; // was the hardcoded ÷20 (now data)
  uint32_t              target = 0;    // output channel/device index
};

// A runtime-bound input source — the dual of `Sink`. The kernel reads source
// values via `OperandKind::Source` operands carrying an index into the plan's
// `sources` array; the engine switches on the source's `kind` to the
// appropriate kernel argument (Tick → current_sample_idx, Rate →
// sample_rate_arg). v1: sources[0] = Tick, sources[1] = Rate.
enum class SourceKind : uint8_t
{
  Tick,  // current sample index (integer)
  Rate,  // current sample rate (float)
};

struct Source
{
  SourceKind kind = SourceKind::Tick;
};

struct FlatProgram
{
  // ── Plan-wide scratch/storage metadata ──
  uint32_t                   register_count   = 0;
  std::vector<uint32_t>      array_slot_sizes; // element count per array slot

  // ── Multi-function layout (tropical_plan_5) ──
  std::vector<InstanceProgram> instance_functions;
  std::vector<Sink>            sinks;    // device-bound outputs (plan_5 mix path)
  std::vector<Source>          sources;  // runtime-bound inputs (plan_5 source path)
};

// Engine realization strategy.
//
//   Fused           — one monolithic LLVM kernel function that inlines
//                     every instance body inside the outer sample loop.
//                     Canonical default; consumed by FlatRuntime via
//                     NumericKernelFn.
//   Microkernel     — N+3 LLVM functions in one module (preamble, N per-
//                     instance kernels, state_evolution, postamble_mix);
//                     the C++ scheduler dispatches them via function
//                     pointers per sample. Top-level session instances
//                     each get their own function; nested children are
//                     inlined into their parent's function. Consumed by
//                     FlatRuntime via MicrokernelKernels.
//   MicrokernelDeep — like Microkernel, but children are also emitted
//                     as their own LLVM functions instead of being
//                     inlined — one function per `InstanceProgram` at
//                     every nesting depth. Requires plans with non-
//                     empty `children` arrays. Codegen lands in a
//                     follow-up (Phase 2); the parser accepts the value
//                     so plans round-trip, but the compile path throws
//                     "not yet implemented" if asked to realize it.
//
// Distinguished by *return type* from the compiler, not by a runtime
// flag — the cache must be partitioned by mode so a fused-mode cache
// hit cannot satisfy a microkernel-mode query.
enum class CompilationMode : uint8_t { Fused, Microkernel, MicrokernelDeep };

// ── Fused-mode kernel signature ──
// `slots` (M6+) is the shared inter-module slot array passed by
// reference for both reads (slot operands) and writes (WriteSlot
// instructions). Always passed; plans without slots pass an empty vector's
// data ptr (the kernel does not reference it). The argument is
// annotated `nocapture noalias` in the IR so GVN / store-to-load
// forwarding can engage on slot reads (per spike #3).
using NumericKernelFn = void (*)(
  const int64_t * inputs,
  int64_t * registers,
  int64_t * const * arrays,
  const uint64_t * array_sizes,
  int64_t * temps,
  double sample_rate,
  uint64_t start_sample_index,
  const uint64_t * param_ptrs,
  double * output_buffer,
  uint64_t buffer_length,
  double * slots);

// ── Microkernel-mode function-pointer signatures ──
//
// Per-sample, not per-buffer. The C++ scheduler in FlatRuntime supplies
// `sample_index` (= state.sample_index + i within the current buffer)
// and walks the buffer one sample at a time, calling each function in
// order. The slim signature drops `buffer_length` (= 1 implicitly) and
// `output_buffer` (only `postamble_mix` writes audio); both would be
// dead args for the per-sample callers.
//
// `inputs` is omitted for the same reason — fused-mode plans pass a
// null pointer here and the kernels never read it (the parser doesn't
// emit Input operands for session-built plans). If a future use case
// needs per-call inputs, we'll widen the signature.
//
// `preamble`, `instance_i`, and `state_evolution` share an identical
// signature (PerSampleFn). `postamble_mix` widens it with the audio
// buffer and the per-sample destination index, because it is the
// single LLVM-land site that touches `output_buffer`. Keeping the mix
// inside LLVM preserves the optimizer's view of mix-temp coercions.
using PerSampleFn = void (*)(
  int64_t * registers,
  int64_t * const * arrays,
  const uint64_t * array_sizes,
  int64_t * temps,
  double sample_rate,
  uint64_t sample_index,
  const uint64_t * param_ptrs,
  double * slots);

using PostambleMixFn = void (*)(
  int64_t * registers,
  int64_t * const * arrays,
  const uint64_t * array_sizes,
  int64_t * temps,
  double sample_rate,
  uint64_t sample_index,
  const uint64_t * param_ptrs,
  double * slots,
  double * output_buffer,
  uint64_t output_index);

struct MicrokernelKernels
{
  std::vector<PerSampleFn>  instances;   // one per FlatProgram::instance_functions entry
  PostambleMixFn            postamble_mix   = nullptr;  // the output sink mix
};

class KernelObjectCache;

class OrcJitEngine
{
  public:
    static OrcJitEngine & instance();

    bool available() const;
    const std::string & init_error() const;

    llvm::Error add_module(
      std::unique_ptr<llvm::LLVMContext> context,
      std::unique_ptr<llvm::Module> module);

    llvm::Expected<uint64_t> lookup(const std::string & symbol_name);

    /** Receive textual LLVM IR, JIT-compile it, return the kernel.
     *
     *  The engine's "receive IR, JIT, run" core — Lean owns IR generation
     *  (EmitLlvm); C++ is strictly an engine. Parses the text into a fresh
     *  module, renames its single function definition to a content-
     *  addressed symbol (`tropical_k_<md5(ir)>`) so distinct IR never
     *  collides in the JIT and identical IR is served from
     *  `ir_kernel_cache_` without a duplicate-symbol re-add, then
     *  add_module + lookup. The text must define exactly one function with
     *  the fused `NumericKernelFn` signature.
     *
     *  `unoptimized` compiles the module at O0 regardless of the engine
     *  opt level (a module flag steers the transform layer; the cache key
     *  is salted so the two levels never collide). For kernels whose
     *  codegen quality is irrelevant — the stage-0 coefficient kernel runs
     *  once per control write, and O2's vectorizer/scheduler/regalloc on
     *  its thousands-of-flops single block is exactly the compile wall the
     *  split exists to remove. */
    llvm::Expected<NumericKernelFn> compile_ir_text(
      const std::string & ir_text, bool unoptimized = false);

  private:
    OrcJitEngine();

    /** Lazily-built sibling LLJIT at CodeGenOptLevel::None (fast ISel,
     *  linear scheduler, regalloc-fast) for `compile_ir_text(_, true)`.
     *  nullptr + `unopt_init_error_` on init failure. */
    llvm::orc::LLJIT * unoptimized_jit();

    std::unique_ptr<llvm::orc::LLJIT> jit_;
    std::unique_ptr<llvm::orc::LLJIT> unopt_jit_;
    std::string init_error_;
    std::string unopt_init_error_;
    mutable std::mutex jit_mutex_;
    /** IR-text kernels, keyed by md5(ir_text), so a reload of identical IR
     *  (e.g. a hot-swap back to a prior plan) returns the cached kernel
     *  instead of re-adding a duplicate symbol to the JIT. */
    std::unordered_map<std::string, NumericKernelFn>      ir_kernel_cache_;
    std::unique_ptr<KernelObjectCache> object_cache_;
    llvm::OptimizationLevel opt_level_ = llvm::OptimizationLevel::O2;
};
} // namespace tropical_jit
