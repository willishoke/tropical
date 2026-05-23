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
  // Stateful param ops (special handling)
  SmoothParam,   // args[0]=Param(ptr), args[1]=StateReg(slot), args[2]=Const(coeff)
  // M6+: write a computed value to slots[dst]. `dst` is the slot index;
  // `args[0]` is the value to write (scalar). No temp slot consumed.
  WriteSlot,
};

enum class OperandKind : uint8_t
{
  Const,    // floating-point constant (const_val field)
  Input,    // module input port (slot field)
  Reg,      // virtual register — scalar result in temps[slot]
  ArrayReg, // virtual register — array result in arrays[slot]
  StateReg, // persistent module register in registers[slot]
  Param,    // ControlParam pointer (ptr field)
  Rate,     // sample rate (runtime constant)
  Tick,     // sample index (runtime counter)
  Slot,     // shared inter-module slot (slot field, indexes slots[]) — M6+
};

struct Operand
{
  OperandKind   kind        = OperandKind::Const;
  JitScalarType scalar_type = JitScalarType::Float;
  double        const_val   = 0.0;  // Const
  uint32_t      slot        = 0;    // Input, Reg, StateReg
  uint64_t      ptr         = 0;    // Param

  static Operand make_const(double v, JitScalarType t = JitScalarType::Float)
  { Operand o; o.kind = OperandKind::Const; o.scalar_type = t; o.const_val = v; return o; }
  static Operand make_input(uint32_t s, JitScalarType t = JitScalarType::Float)
  { Operand o; o.kind = OperandKind::Input; o.scalar_type = t; o.slot = s; return o; }
  static Operand make_reg(uint32_t id, JitScalarType t = JitScalarType::Float)
  { Operand o; o.kind = OperandKind::Reg; o.scalar_type = t; o.slot = id; return o; }
  static Operand make_array_reg(uint32_t s)
  { Operand o; o.kind = OperandKind::ArrayReg; o.slot = s; return o; }
  static Operand make_state(uint32_t s, JitScalarType t = JitScalarType::Float)
  { Operand o; o.kind = OperandKind::StateReg; o.scalar_type = t; o.slot = s; return o; }
  static Operand make_param(uint64_t p)
  { Operand o; o.kind = OperandKind::Param; o.scalar_type = JitScalarType::Float; o.ptr = p; return o; }
  static Operand make_rate()
  { Operand o; o.kind = OperandKind::Rate; o.scalar_type = JitScalarType::Float; return o; }
  static Operand make_tick()
  { Operand o; o.kind = OperandKind::Tick; o.scalar_type = JitScalarType::Float; return o; }
  // M6: slot operand reads from the shared slots[] arg passed to the
  // kernel. Stored as double, coerced at use site to match scalar_type.
  static Operand make_slot(uint32_t s, JitScalarType t = JitScalarType::Float)
  { Operand o; o.kind = OperandKind::Slot; o.scalar_type = t; o.slot = s; return o; }
};

struct FlatInstr
{
  OpTag                tag         = OpTag::Add;
  JitScalarType        result_type = JitScalarType::Float;
  uint32_t             dst         = 0;
  std::vector<Operand> args;
  uint32_t             loop_count  = 1;       // 1 = scalar; N > 1 = elementwise loop
  std::vector<uint8_t> strides;               // per-arg: 1 = iterate, 0 = broadcast
};

// Per-instance kernel slice. Each session instance contributes one of
// these to the multi-function plan. The compiler emits all instances
// inline within one LLVM function — `children` are emitted recursively
// inside the parent's body (M11 fractal architecture).
struct InstanceProgram
{
  std::string                  instance_name;     // dotted path (e.g. voice1.env)
  std::vector<FlatInstr>       instructions;      // already shifted into unified offset space
  /** M11 fractal slot-based input wiring. Per-child block — runs in
   *  the PARENT's namespace immediately before THIS kernel's body. The
   *  parent's emit_kernel_block dispatches each child as:
   *      emit_instrs(child.pre_input_instructions)
   *      emit_kernel_block(child)
   *  Per-child placement (vs hoisting into a parent-wide pre-children
   *  block) preserves sibling-to-sibling NestedOut dependencies: a
   *  later sibling's wire can read an earlier sibling's output slot
   *  because the earlier sibling's body has already executed. Empty
   *  for top-level kernels and for the legacy flat-IR
   *  (inlineNested:true) path. */
  std::vector<FlatInstr>       pre_input_instructions;
  uint32_t                     register_count = 0; // local temp count
  /** State register writebacks for this instance. Each entry pairs a
   *  state register slot index (absolute, into the unified register
   *  array) with the temp index (also absolute) whose value feeds it. */
  struct Writeback {
    uint32_t state_slot;   // index into the unified state-register array
    int32_t  temp_slot;    // index into the unified temp array (-1 = skip)
  };
  std::vector<Writeback>       writebacks;
  /** Nested child kernels. Emitted recursively inside this kernel's
   *  body. Empty for leaf kernels. */
  std::vector<InstanceProgram> children;
};

// Top-level driver: runs once per sample. Preamble fires before any
// instance dispatch; postamble fires after. state_evolution sits
// between instance dispatch and postamble — holds delay-slot updates
// from MCP wire auto-delays. Empty for legacy plans.
struct SchedulerProgram
{
  std::vector<FlatInstr>       preamble;
  std::vector<FlatInstr>       state_evolution;
  std::vector<FlatInstr>       postamble;
};

struct FlatProgram
{
  // ── Plan-wide unified state (shared across all instance functions) ──
  uint32_t                   register_count   = 0;
  std::vector<uint32_t>      array_slot_sizes; // element count per array slot
  std::vector<uint32_t>      output_targets;
  std::vector<int32_t>       register_targets;
  std::vector<uint32_t>      mix_output_temps;  // temp indices summed to audio
  std::vector<JitScalarType> register_types;

  // ── Multi-function layout (tropical_plan_5) ──
  std::vector<InstanceProgram> instance_functions;
  SchedulerProgram             scheduler;
};

// Engine realization strategy.
//
//   Fused       — one monolithic LLVM kernel function that inlines every
//                 instance body inside the outer sample loop. Legacy
//                 default; consumed by FlatRuntime via NumericKernelFn.
//   Microkernel — N+1 LLVM functions in one module (preamble, N per-
//                 instance kernels, state_evolution, postamble_mix);
//                 the C++ scheduler dispatches them via function
//                 pointers per sample. Consumed by FlatRuntime via
//                 MicrokernelKernels.
//
// Distinguished by *return type* from the compiler, not by a runtime
// flag — the cache must be partitioned by mode so a fused-mode cache
// hit cannot satisfy a microkernel-mode query.
enum class CompilationMode : uint8_t { Fused, Microkernel };

// ── Fused-mode kernel signature ──
// `slots` (M6+) is the shared inter-module slot array passed by
// reference for both reads (slot operands) and writes (WriteSlot
// instructions). Always passed; legacy plans pass an empty vector's
// data ptr (the kernel just doesn't reference it). The argument is
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
  PerSampleFn               preamble        = nullptr;
  std::vector<PerSampleFn>  instances;   // one per FlatProgram::instance_functions entry
  PerSampleFn               state_evolution = nullptr;
  PostambleMixFn            postamble_mix   = nullptr;
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

    /** Fused-mode codegen: one LLVM function per plan. */
    llvm::Expected<NumericKernelFn> compile_flat_program(
      const FlatProgram & program);

    /** Microkernel-mode codegen: N+1 LLVM functions in one module.
     *  Phase 3 supplies the implementation; Phase 2 lands the type
     *  surface and a stub that errors. */
    llvm::Expected<MicrokernelKernels> compile_microkernel(
      const FlatProgram & program);

  private:
    OrcJitEngine();

    std::unique_ptr<llvm::orc::LLJIT> jit_;
    std::string init_error_;
    mutable std::mutex jit_mutex_;
    std::unordered_map<std::string, NumericKernelFn>      kernel_cache_;
    /** Microkernel cache lives alongside the fused-mode cache because
     *  the return types differ. Phase 5 mode-tags the keys so the two
     *  caches never collide on a shared program hash. */
    std::unordered_map<std::string, MicrokernelKernels>   microkernel_cache_;
    std::unique_ptr<KernelObjectCache> object_cache_;
    llvm::OptimizationLevel opt_level_ = llvm::OptimizationLevel::O2;
};
} // namespace tropical_jit
