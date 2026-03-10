#include "OrcJitEngine.hpp"

#ifdef EGRESS_LLVM_ORC_JIT
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Verifier.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>

#include <atomic>
#endif

namespace egress_jit
{
#ifdef EGRESS_LLVM_ORC_JIT
OrcJitEngine & OrcJitEngine::instance()
{
  static OrcJitEngine engine;
  return engine;
}

OrcJitEngine::OrcJitEngine()
{
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  auto jit_or_err = llvm::orc::LLJITBuilder().create();
  if (!jit_or_err)
  {
    init_error_ = llvm::toString(jit_or_err.takeError());
    return;
  }

  jit_ = std::move(*jit_or_err);
}

bool OrcJitEngine::available() const
{
  return static_cast<bool>(jit_);
}

const std::string & OrcJitEngine::init_error() const
{
  return init_error_;
}

llvm::Error OrcJitEngine::add_module(
  std::unique_ptr<llvm::LLVMContext> context,
  std::unique_ptr<llvm::Module> module)
{
  if (!jit_)
  {
    return llvm::make_error<llvm::StringError>(
      "ORC JIT is not available: " + init_error_,
      llvm::inconvertibleErrorCode());
  }

  llvm::orc::ThreadSafeModule thread_safe_module(std::move(module), std::move(context));
  if (auto err = jit_->addIRModule(std::move(thread_safe_module)))
  {
    return std::move(err);
  }

  return llvm::Error::success();
}

llvm::Expected<uint64_t> OrcJitEngine::lookup(const std::string & symbol_name)
{
  if (!jit_)
  {
    return llvm::make_error<llvm::StringError>(
      "ORC JIT is not available: " + init_error_,
      llvm::inconvertibleErrorCode());
  }

  auto symbol_or_err = jit_->lookup(symbol_name);
  if (!symbol_or_err)
  {
    return symbol_or_err.takeError();
  }

#if LLVM_VERSION_MAJOR >= 17
  return symbol_or_err->getValue();
#elif LLVM_VERSION_MAJOR >= 15
  return symbol_or_err->getValue();
#else
  return symbol_or_err->getAddress();
#endif
}

llvm::Expected<NumericKernelFn> OrcJitEngine::compile_numeric_program(
  const NumericProgram & program,
  const std::string & symbol_prefix)
{
  if (!jit_)
  {
    return llvm::make_error<llvm::StringError>(
      "ORC JIT is not available: " + init_error_,
      llvm::inconvertibleErrorCode());
  }

  auto context = std::make_unique<llvm::LLVMContext>();
  auto module = std::make_unique<llvm::Module>("egress_udm_numeric", *context);
  module->setDataLayout(jit_->getDataLayout());

  llvm::IRBuilder<> builder(*context);
  llvm::Type * void_ty = builder.getVoidTy();
  llvm::Type * f64_ty = builder.getDoubleTy();
  llvm::Type * i64_ty = builder.getInt64Ty();
  llvm::Type * f64_ptr_ty = builder.getDoubleTy()->getPointerTo();

  llvm::FunctionType * fn_ty = llvm::FunctionType::get(
    void_ty,
    {f64_ptr_ty, f64_ptr_ty, f64_ptr_ty, f64_ty, i64_ty},
    false);

  static std::atomic<uint64_t> function_counter{0};
  const std::string function_name = symbol_prefix + "_" + std::to_string(function_counter.fetch_add(1, std::memory_order_relaxed));

  llvm::Function * fn = llvm::Function::Create(fn_ty, llvm::Function::ExternalLinkage, function_name, module.get());

  auto arg_it = fn->arg_begin();
  llvm::Value * inputs_arg = &*arg_it++;
  inputs_arg->setName("inputs");
  llvm::Value * regs_arg = &*arg_it++;
  regs_arg->setName("registers");
  llvm::Value * temps_arg = &*arg_it++;
  temps_arg->setName("temps");
  llvm::Value * sample_rate_arg = &*arg_it++;
  sample_rate_arg->setName("sample_rate");
  llvm::Value * sample_index_arg = &*arg_it++;
  sample_index_arg->setName("sample_index");

  llvm::BasicBlock * entry = llvm::BasicBlock::Create(*context, "entry", fn);
  builder.SetInsertPoint(entry);

  auto gep_f64 = [&](llvm::Value * base_ptr, uint32_t index) -> llvm::Value * {
    return builder.CreateInBoundsGEP(f64_ty, base_ptr, builder.getInt64(index));
  };

  auto load_temp = [&](uint32_t index) -> llvm::Value * {
    return builder.CreateLoad(f64_ty, gep_f64(temps_arg, index));
  };

  auto store_temp = [&](uint32_t index, llvm::Value * value) {
    builder.CreateStore(value, gep_f64(temps_arg, index));
  };

  llvm::Function * llvm_sin = llvm::Intrinsic::getDeclaration(module.get(), llvm::Intrinsic::sin, {f64_ty});

  for (const auto & instr : program.instructions)
  {
    llvm::Value * result = nullptr;
    switch (instr.op)
    {
      case NumericOp::Literal:
        result = llvm::ConstantFP::get(f64_ty, instr.literal);
        break;
      case NumericOp::InputValue:
        result = builder.CreateLoad(f64_ty, gep_f64(inputs_arg, instr.slot_id));
        break;
      case NumericOp::RegisterValue:
        result = builder.CreateLoad(f64_ty, gep_f64(regs_arg, instr.slot_id));
        break;
      case NumericOp::SampleRate:
        result = sample_rate_arg;
        break;
      case NumericOp::SampleIndex:
        result = builder.CreateSIToFP(sample_index_arg, f64_ty);
        break;
      case NumericOp::Add:
        result = builder.CreateFAdd(load_temp(instr.src_a), load_temp(instr.src_b));
        break;
      case NumericOp::Sub:
        result = builder.CreateFSub(load_temp(instr.src_a), load_temp(instr.src_b));
        break;
      case NumericOp::Mul:
        result = builder.CreateFMul(load_temp(instr.src_a), load_temp(instr.src_b));
        break;
      case NumericOp::Div:
      {
        llvm::Value * lhs = load_temp(instr.src_a);
        llvm::Value * rhs = load_temp(instr.src_b);
        llvm::Value * is_zero = builder.CreateFCmpOEQ(rhs, llvm::ConstantFP::get(f64_ty, 0.0));
        llvm::Value * div_value = builder.CreateFDiv(lhs, rhs);
        result = builder.CreateSelect(is_zero, llvm::ConstantFP::get(f64_ty, 0.0), div_value);
        break;
      }
      case NumericOp::Sin:
        result = builder.CreateCall(llvm_sin, {load_temp(instr.src_a)});
        break;
      case NumericOp::Neg:
        result = builder.CreateFNeg(load_temp(instr.src_a));
        break;
    }

    if (!result)
    {
      return llvm::make_error<llvm::StringError>(
        "Failed to lower numeric instruction to LLVM IR",
        llvm::inconvertibleErrorCode());
    }

    store_temp(instr.dst, result);
  }

  builder.CreateRetVoid();

  if (llvm::verifyFunction(*fn, &llvm::errs()) || llvm::verifyModule(*module, &llvm::errs()))
  {
    return llvm::make_error<llvm::StringError>(
      "Generated invalid LLVM IR for numeric UDM kernel",
      llvm::inconvertibleErrorCode());
  }

  if (auto err = add_module(std::move(context), std::move(module)))
  {
    return std::move(err);
  }

  auto addr_or_err = lookup(function_name);
  if (!addr_or_err)
  {
    return addr_or_err.takeError();
  }

  return reinterpret_cast<NumericKernelFn>(*addr_or_err);
}
#else
OrcJitEngine & OrcJitEngine::instance()
{
  static OrcJitEngine engine;
  return engine;
}

bool OrcJitEngine::available() const
{
  return false;
}

const std::string & OrcJitEngine::init_error() const
{
  static const std::string kDisabled = "Built without EGRESS_LLVM_ORC_JIT";
  return kDisabled;
}
#endif
} // namespace egress_jit
