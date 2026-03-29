#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>

#ifdef EGRESS_LLVM_ORC_JIT
#include <llvm/Config/llvm-config.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#endif

#include <filesystem>

namespace egress_jit
{
enum class JitScalarType : uint8_t { Float, Int, Bool };

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
  SmoothedParam
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
  std::vector<uint32_t> args;
};

struct NumericProgram
{
  std::vector<NumericInstr> instructions;
  uint32_t register_count = 0;
};

using NumericKernelFn = void (*)(
  const double * inputs,
  double * registers,
  double * const * arrays,
  const uint64_t * array_sizes,
  double * temps,
  double sample_rate,
  uint64_t sample_index,
  const uint64_t * param_ptrs,
  const int64_t * int_inputs,
  int64_t * int_regs,
  int64_t * const * int_arrays,
  int64_t * int_temps);

#ifdef EGRESS_LLVM_ORC_JIT
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

  private:
    OrcJitEngine();

    std::unique_ptr<llvm::orc::LLJIT> jit_;
    std::string init_error_;
    mutable std::mutex jit_mutex_;
    std::unordered_map<std::string, NumericKernelFn> kernel_cache_;
    std::unique_ptr<KernelObjectCache> object_cache_;
};
#else
class OrcJitEngine
{
  public:
    static OrcJitEngine & instance();

    bool available() const;
    const std::string & init_error() const;

  private:
    OrcJitEngine() = default;
};
#endif
} // namespace egress_jit
