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

#include <filesystem>

namespace egress_jit
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
  Add, Sub, Mul, Div, Mod, Pow, FloorDiv,
  Less, LessEq, Greater, GreaterEq, Equal, NotEqual,
  BitAnd, BitOr, BitXor, LShift, RShift,
  Index,    // args[0]=ArrayReg, args[1]=scalar idx → scalar element
  MatMul,
  // arity 1
  Neg, Abs, Sin, Cos, Log, Exp, Sqrt, Floor, Ceil, Round, Not, BitNot,
  // arity 3
  Clamp, Select,
  SetElement,  // args[0]=ArrayReg, args[1]=idx, args[2]=val; no dst slot written
  // arity N
  Pack,     // args = scalar values → arrays[dst]
  // Stateful param ops (special handling)
  SmoothParam,   // args[0]=Param(ptr), args[1]=StateReg(slot), args[2]=Const(coeff)
  TriggerParam,  // args[0]=Param(ptr)
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
};

struct Operand
{
  OperandKind kind = OperandKind::Const;
  double      const_val = 0.0;  // Const
  uint32_t    slot      = 0;    // Input, Reg, StateReg
  uint64_t    ptr       = 0;    // Param

  static Operand make_const(double v)    { Operand o; o.kind = OperandKind::Const;    o.const_val = v; return o; }
  static Operand make_input(uint32_t s)  { Operand o; o.kind = OperandKind::Input;    o.slot = s;      return o; }
  static Operand make_reg(uint32_t id)   { Operand o; o.kind = OperandKind::Reg;      o.slot = id;     return o; }
  static Operand make_array_reg(uint32_t s) { Operand o; o.kind = OperandKind::ArrayReg; o.slot = s;   return o; }
  static Operand make_state(uint32_t s)  { Operand o; o.kind = OperandKind::StateReg; o.slot = s;      return o; }
  static Operand make_param(uint64_t p)  { Operand o; o.kind = OperandKind::Param;    o.ptr = p;       return o; }
  static Operand make_rate()             { Operand o; o.kind = OperandKind::Rate;                      return o; }
  static Operand make_tick()             { Operand o; o.kind = OperandKind::Tick;                      return o; }
};

struct FlatInstr
{
  OpTag                tag        = OpTag::Add;
  uint32_t             dst        = 0;
  std::vector<Operand> args;
  uint32_t             loop_count = 1;       // 1 = scalar; N > 1 = elementwise loop
  std::vector<uint8_t> strides;              // per-arg: 1 = iterate, 0 = broadcast
};

struct FlatProgram
{
  std::vector<FlatInstr> instructions;
  uint32_t               register_count   = 0;
  std::vector<uint32_t>  output_targets;
  std::vector<int32_t>   register_targets;
};

// ─────────────────────────────────────────────────────────────────────────────
// DEPRECATED — kept for NumericProgramBuilder until egress_plan_2 is removed
// ─────────────────────────────────────────────────────────────────────────────

enum class NumericOp : uint8_t
{
  Literal,
  InputValue,
  RegisterValue,
  SampleRate,
  SampleIndex,
  Not,
  Less,
  LessEqual,
  Greater,
  GreaterEqual,
  Equal,
  NotEqual,
  ArrayLessScalar,
  ArrayLessEqualScalar,
  ArrayGreaterScalar,
  ArrayGreaterEqualScalar,
  ArrayEqualScalar,
  ArrayNotEqualScalar,
  ArrayPack,
  Add,
  ArrayAdd,
  ArrayAddScalar,
  Sub,
  ArraySub,
  Mul,
  ArrayMul,
  ArrayMulScalar,
  Div,
  ArrayDiv,
  ArrayDivScalar,
  ArrayModScalar,
  MatMul,
  Pow,
  Mod,
  FloorDiv,
  BitAnd,
  BitOr,
  BitXor,
  LShift,
  RShift,
  Abs,
  Clamp,
  Select,
  Log,
  IndexArray,
  SetArrayElement,
  Sin,
  Neg,
  BitNot,
  SmoothedParam,
  CopySlot
};

struct NumericInstr
{
  NumericOp op = NumericOp::Literal;
  uint32_t dst = 0;
  uint32_t src_a = 0;
  uint32_t src_b = 0;
  uint32_t src_c = 0;
  uint32_t slot_id = 0;
  double literal = 0.0;
  int64_t int_literal = 0;
  uint64_t param_ptr = 0;
  JitScalarType dst_type = JitScalarType::Float;
  JitScalarType src_a_type = JitScalarType::Float;
  JitScalarType src_b_type = JitScalarType::Float;
  JitScalarType src_c_type = JitScalarType::Float;
  std::vector<uint32_t> args;
};

struct NumericProgram
{
  std::vector<NumericInstr> instructions;
  uint32_t register_count = 0;
};

using NumericKernelFn = void (*)(
  const int64_t * inputs,
  int64_t * registers,
  int64_t * const * arrays,
  const uint64_t * array_sizes,
  int64_t * temps,
  double sample_rate,
  uint64_t sample_index,
  const uint64_t * param_ptrs);

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

    llvm::Expected<NumericKernelFn> compile_numeric_program(
      const NumericProgram & program);

    llvm::Expected<NumericKernelFn> compile_flat_program(
      const FlatProgram & program);

  private:
    OrcJitEngine();

    std::unique_ptr<llvm::orc::LLJIT> jit_;
    std::string init_error_;
    mutable std::mutex jit_mutex_;
    std::unordered_map<std::string, NumericKernelFn> kernel_cache_;
    std::unique_ptr<KernelObjectCache> object_cache_;
};
} // namespace egress_jit
