#pragma once

#include "graph/GraphTypes.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
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

struct RuntimeState
{
  std::vector<ModuleSlot> modules;
  std::unordered_map<std::string, uint32_t> name_to_id;
  std::vector<MixTap> mix;
  std::vector<MixExpr> mix_exprs;
};

#ifdef EGRESS_PROFILE
struct ModuleTimingCounters
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
