#pragma once

#include <memory>
#include <string>
#include <vector>
#include <cstdint>

#ifdef EGRESS_LLVM_ORC_JIT
#include <llvm/Config/llvm-config.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#endif

namespace egress_jit
{
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
  Add,
  Sub,
  Mul,
  Div,
  Mod,
  FloorDiv,
  BitAnd,
  BitOr,
  BitXor,
  LShift,
  RShift,
  IndexArray,
  Sin,
  Neg,
  BitNot
};

struct NumericInstr
{
  NumericOp op = NumericOp::Literal;
  uint32_t dst = 0;
  uint32_t src_a = 0;
  uint32_t src_b = 0;
  uint32_t slot_id = 0;
  double literal = 0.0;
};

struct NumericProgram
{
  std::vector<NumericInstr> instructions;
  uint32_t register_count = 0;
};

using NumericKernelFn = void (*)(
  const double * inputs,
  const double * registers,
  const double * const * arrays,
  const uint64_t * array_sizes,
  double * temps,
  double sample_rate,
  uint64_t sample_index);

using GraphKernelFn = void (*)(double * output_buffer, uint64_t frame_count);

using SingleUdmGraphKernelFn = void (*) (
  double * output_buffer,
  uint64_t frame_count,
  uint64_t module_kernel_addr,
  const double * module_inputs,
  double * module_registers,
  double * module_next_registers,
  const double * const * module_arrays,
  const uint64_t * module_array_sizes,
  double * module_temps,
  const int32_t * register_targets,
  uint64_t register_target_count,
  const uint32_t * output_targets,
  uint64_t output_target_count,
  uint64_t temp_count,
  const uint32_t * mix_output_ids,
  uint64_t mix_count,
  double sample_rate,
  uint64_t * sample_index_io);

#ifdef EGRESS_LLVM_ORC_JIT
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
      const NumericProgram & program,
      const std::string & symbol_prefix);

    llvm::Expected<GraphKernelFn> compile_graph_stub(
      const std::string & symbol_prefix);

    llvm::Expected<SingleUdmGraphKernelFn> compile_single_udm_graph_kernel(
      const std::string & symbol_prefix);

  private:
    OrcJitEngine();

    std::unique_ptr<llvm::orc::LLJIT> jit_;
    std::string init_error_;
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
