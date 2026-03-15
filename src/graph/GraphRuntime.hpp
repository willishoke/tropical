#pragma once

#include "graph/GraphTypes.hpp"
#include "graph/ModuleNumericJit.hpp"
#include "jit/OrcJitEngine.hpp"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <mutex>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace egress_graph_detail
{
struct ModuleShape
{
  unsigned int in_count;
  unsigned int out_count;
};

struct ControlModule
{
  std::shared_ptr<Module> module;
  unsigned int in_count = 0;
  unsigned int out_count = 0;
  std::vector<ExprSpecPtr> input_exprs;
};

enum class OpCode
{
  Literal,
  Ref,
  RefIndex,
  ArrayPack,
  Index,
  ArraySet,
  Add,
  AddConst,
  Sub,
  SubConstRhs,
  SubConstLhs,
  Mul,
  MulConst,
  Div,
  MatMul,
  Pow,
  DivConstLhs,
  Mod,
  FloorDiv,
  BitAnd,
  BitOr,
  BitXor,
  LShift,
  RShift,
  Abs,
  Clamp,
  Log,
  Neg,
  Not,
  BitNot,
  Sin,
  Less,
  LessEqual,
  Greater,
  GreaterEqual,
  Equal,
  NotEqual
};

struct ExprInstr
{
  OpCode opcode = OpCode::Literal;
  uint32_t dst = 0;
  uint32_t src_a = 0;
  uint32_t src_b = 0;
  uint32_t src_c = 0;
  std::vector<uint32_t> args;
  Value literal;
  uint32_t ref_module_id = 0;
  unsigned int ref_output_id = 0;
  int64_t ref_index = -1;
};

struct CompiledInputProgram
{
  std::vector<ExprInstr> instructions;
  uint32_t register_count = 0;
  std::vector<uint32_t> result_registers;
};

struct ModuleSlot
{
  std::string name;
  std::shared_ptr<Module> module;
  CompiledInputProgram input_program;
  std::vector<Value> input_registers;
  std::vector<bool> output_materialize_mask;
  std::vector<bool> output_prev_materialize_mask;
  std::vector<std::vector<int64_t>> indexed_output_indices;
  std::vector<std::vector<Value>> indexed_prev_output_values;
};

struct MixTap
{
  uint32_t module_id;
  unsigned int output_id;
};

struct MixExpr
{
  CompiledInputProgram program;
  std::vector<Value> registers;
  uint32_t result_register = 0;
};

struct TapBuffer
{
  std::array<std::vector<double>, 2> buffers;
  std::atomic<uint32_t> readable{0};

  TapBuffer() = default;
  TapBuffer(const TapBuffer &) = delete;
  TapBuffer & operator=(const TapBuffer &) = delete;

  TapBuffer(TapBuffer && other) noexcept
    : buffers(std::move(other.buffers))
  {
    readable.store(other.readable.load(std::memory_order_relaxed), std::memory_order_relaxed);
  }

  TapBuffer & operator=(TapBuffer && other) noexcept
  {
    if (this != &other)
    {
      buffers = std::move(other.buffers);
      readable.store(other.readable.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
    return *this;
  }
};

struct OutputTap
{
  uint32_t module_id = 0;
  unsigned int output_id = 0;
  bool valid = false;
  TapBuffer buffer;

  OutputTap() = default;
  OutputTap(const OutputTap &) = delete;
  OutputTap & operator=(const OutputTap &) = delete;
  OutputTap(OutputTap &&) noexcept = default;
  OutputTap & operator=(OutputTap &&) noexcept = default;
};

struct FusedGraphModuleSpan
{
  uint32_t first_output_slot = 0;
  uint32_t output_count = 0;
};

struct FusedGraphSourceOutput
{
  uint32_t module_id = 0;
  unsigned int output_id = 0;
  bool materialized = true;
};

using FusedGraphValueKind = egress_module_detail::NumericValueKind;
using FusedGraphValueRef = egress_module_detail::NumericValueRef;

struct FusedGraphInputBinding
{
  uint32_t module_id = 0;
  unsigned int input_id = 0;
  uint32_t fused_input_slot = std::numeric_limits<uint32_t>::max();
  uint32_t fused_input_array_slot = std::numeric_limits<uint32_t>::max();
  FusedGraphValueRef value;
};

struct FusedGraphMixBinding
{
  FusedGraphValueRef value;
};

struct FusedPrimitiveBodyModule
{
  uint32_t module_id = 0;
  uint32_t input_base = 0;
  uint32_t register_base = 0;
  uint32_t array_base = 0;
  uint32_t array_slot_count = 0;
  std::vector<uint32_t> output_registers;
  std::vector<egress_module_detail::NumericInputInfo> input_info;
  std::vector<egress_module_detail::NumericOutputInfo> output_info;
  std::vector<bool> register_scalar_mask;
  std::vector<uint32_t> register_array_slot;
  std::vector<int32_t> register_targets;
  std::vector<int32_t> array_register_targets;
  std::vector<bool> array_register_can_swap;
};

struct FusedGraphKernelState
{
  bool available = false;
  std::string status;
  std::vector<double> scalar_inputs;
  std::vector<double> temps;
  std::vector<std::vector<double>> array_storage;
  std::vector<double *> array_ptrs;
  std::vector<uint64_t> array_sizes;
  std::unordered_map<uint32_t, uint32_t> source_array_slots;
  std::unordered_map<uint64_t, uint32_t> indexed_prev_scalar_slots;
  std::vector<FusedGraphInputBinding> input_bindings;
  std::vector<FusedGraphMixBinding> mix_bindings;

#ifdef EGRESS_LLVM_ORC_JIT
  egress_jit::NumericProgram program;
  egress_jit::NumericKernelFn kernel = nullptr;
#endif
};

struct FusedGraphState
{
  bool numeric_candidate = false;
  std::string candidate_reason;
  std::vector<FusedGraphModuleSpan> module_output_spans;
  std::vector<FusedGraphSourceOutput> source_outputs;
  std::unordered_map<uint64_t, uint32_t> source_output_lookup;
  std::vector<Value> current_outputs;
  std::vector<Value> prev_outputs;
  std::vector<std::vector<int64_t>> indexed_prev_indices;
  std::vector<std::vector<Value>> indexed_prev_values;
  std::vector<uint32_t> mix_source_output_slots;
  std::vector<uint32_t> tap_source_output_slots;
  std::vector<FusedPrimitiveBodyModule> primitive_body_modules;
  std::vector<bool> primitive_body_module_mask;
  bool primitive_body_covers_all_modules = false;
  bool primitive_body_available = false;
  std::string primitive_body_status;
  double primitive_body_sample_rate = 44100.0;
  FusedGraphKernelState input_kernel;
  FusedGraphKernelState mix_kernel;

#ifdef EGRESS_LLVM_ORC_JIT
  egress_jit::NumericProgram program;
  egress_jit::NumericKernelFn kernel = nullptr;
  std::vector<double> inputs;
  std::vector<double> registers;
  std::vector<double> temps;
  std::vector<std::vector<double>> array_storage;
  std::vector<double *> array_ptrs;
  std::vector<uint64_t> array_sizes;
#endif
};

struct RuntimeState
{
  std::vector<ModuleSlot> modules;
  std::unordered_map<std::string, uint32_t> name_to_id;
  std::vector<MixTap> mix;
  std::vector<MixExpr> mix_exprs;
  std::vector<OutputTap> taps;
  std::unique_ptr<FusedGraphState> fused_graph;
};

#ifdef EGRESS_PROFILE
struct ModuleTimingCounters
{
  uint64_t call_count = 0;
  uint64_t total_ns = 0;
  uint64_t max_ns = 0;
};

struct ProcessModuleTiming
{
  uint64_t call_count = 0;
  uint64_t total_ns = 0;
  uint64_t max_ns = 0;
};

inline void update_atomic_max(std::atomic<uint64_t> & dst, uint64_t candidate)
{
  uint64_t current = dst.load(std::memory_order_relaxed);
  while (current < candidate &&
         !dst.compare_exchange_weak(current, candidate, std::memory_order_relaxed))
  {
  }
}
#endif
}  // namespace egress_graph_detail
