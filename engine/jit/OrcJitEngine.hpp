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

struct FlatProgram
{
  std::vector<FlatInstr> instructions;
  uint32_t               register_count   = 0;
  std::vector<uint32_t>  array_slot_sizes; // element count per array slot
  std::vector<uint32_t>  output_targets;
  std::vector<int32_t>   register_targets;
  std::vector<uint32_t>  mix_output_temps;  // temp indices whose values mix to audio
};

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
  uint64_t buffer_length);

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
} // namespace tropical_jit
