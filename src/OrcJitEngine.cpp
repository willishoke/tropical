#include "OrcJitEngine.hpp"

#ifdef EGRESS_LLVM_ORC_JIT
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Verifier.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>

#include <atomic>
#endif

extern "C" void egress_graph_apply_register_targets(
  double * registers,
  double * next_registers,
  const double * temps,
  const int32_t * targets,
  uint64_t target_count,
  uint64_t temp_count)
{
  for (uint64_t i = 0; i < target_count; ++i)
  {
    const int32_t target = targets[i];
    if (target >= 0)
    {
      const uint64_t idx = static_cast<uint64_t>(target);
      next_registers[i] = idx < temp_count ? temps[idx] : registers[i];
    }
    else
    {
      next_registers[i] = registers[i];
    }
  }

  for (uint64_t i = 0; i < target_count; ++i)
  {
    registers[i] = next_registers[i];
  }
}

extern "C" double egress_graph_mix_single_udm(
  const double * temps,
  const uint32_t * output_targets,
  uint64_t output_target_count,
  const uint32_t * mix_output_ids,
  uint64_t mix_count,
  uint64_t temp_count)
{
  double mixed = 0.0;
  for (uint64_t i = 0; i < mix_count; ++i)
  {
    const uint32_t output_id = mix_output_ids[i];
    if (output_id >= output_target_count)
    {
      continue;
    }
    const uint32_t target = output_targets[output_id];
    if (target >= temp_count)
    {
      continue;
    }
    double sample = temps[target];
    if (sample > 10.0)
    {
      sample = 10.0;
    }
    else if (sample < -10.0)
    {
      sample = -10.0;
    }
    mixed += sample / 20.0;
  }
  return mixed;
}

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

  llvm::orc::SymbolMap helper_symbols;
  helper_symbols[jit_->mangleAndIntern("egress_graph_apply_register_targets")] = llvm::orc::ExecutorSymbolDef(
    llvm::orc::ExecutorAddr::fromPtr(&egress_graph_apply_register_targets),
    llvm::JITSymbolFlags::Exported);
  helper_symbols[jit_->mangleAndIntern("egress_graph_mix_single_udm")] = llvm::orc::ExecutorSymbolDef(
    llvm::orc::ExecutorAddr::fromPtr(&egress_graph_mix_single_udm),
    llvm::JITSymbolFlags::Exported);

  if (auto err = jit_->getMainJITDylib().define(llvm::orc::absoluteSymbols(std::move(helper_symbols))))
  {
    init_error_ = llvm::toString(std::move(err));
    jit_.reset();
    return;
  }
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
  llvm::Type * f64_ptr_ty = llvm::PointerType::get(*context, 0);
  llvm::Type * f64_ptr_ptr_ty = llvm::PointerType::get(*context, 0);
  llvm::Type * i64_ptr_ty = llvm::PointerType::get(*context, 0);

  llvm::FunctionType * fn_ty = llvm::FunctionType::get(
    void_ty,
    {f64_ptr_ty, f64_ptr_ty, f64_ptr_ptr_ty, i64_ptr_ty, f64_ptr_ty, f64_ty, i64_ty},
    false);

  static std::atomic<uint64_t> function_counter{0};
  const std::string function_name = symbol_prefix + "_" + std::to_string(function_counter.fetch_add(1, std::memory_order_relaxed));

  llvm::Function * fn = llvm::Function::Create(fn_ty, llvm::Function::ExternalLinkage, function_name, module.get());

  auto arg_it = fn->arg_begin();
  llvm::Value * inputs_arg = &*arg_it++;
  inputs_arg->setName("inputs");
  llvm::Value * regs_arg = &*arg_it++;
  regs_arg->setName("registers");
  llvm::Value * arrays_arg = &*arg_it++;
  arrays_arg->setName("arrays");
  llvm::Value * array_sizes_arg = &*arg_it++;
  array_sizes_arg->setName("array_sizes");
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

  llvm::FunctionCallee llvm_sin = llvm::Intrinsic::getOrInsertDeclaration(module.get(), llvm::Intrinsic::sin, {f64_ty});
  llvm::FunctionCallee llvm_fmod = module->getOrInsertFunction(
    "fmod",
    llvm::FunctionType::get(f64_ty, {f64_ty, f64_ty}, false));

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
      case NumericOp::Not:
      {
        llvm::Value * value = load_temp(instr.src_a);
        llvm::Value * truthy = builder.CreateFCmpUNE(value, llvm::ConstantFP::get(f64_ty, 0.0));
        result = builder.CreateSelect(truthy, llvm::ConstantFP::get(f64_ty, 0.0), llvm::ConstantFP::get(f64_ty, 1.0));
        break;
      }
      case NumericOp::Less:
      {
        llvm::Value * cmp = builder.CreateFCmpOLT(load_temp(instr.src_a), load_temp(instr.src_b));
        result = builder.CreateUIToFP(cmp, f64_ty);
        break;
      }
      case NumericOp::LessEqual:
      {
        llvm::Value * cmp = builder.CreateFCmpOLE(load_temp(instr.src_a), load_temp(instr.src_b));
        result = builder.CreateUIToFP(cmp, f64_ty);
        break;
      }
      case NumericOp::Greater:
      {
        llvm::Value * cmp = builder.CreateFCmpOGT(load_temp(instr.src_a), load_temp(instr.src_b));
        result = builder.CreateUIToFP(cmp, f64_ty);
        break;
      }
      case NumericOp::GreaterEqual:
      {
        llvm::Value * cmp = builder.CreateFCmpOGE(load_temp(instr.src_a), load_temp(instr.src_b));
        result = builder.CreateUIToFP(cmp, f64_ty);
        break;
      }
      case NumericOp::Equal:
      {
        llvm::Value * cmp = builder.CreateFCmpOEQ(load_temp(instr.src_a), load_temp(instr.src_b));
        result = builder.CreateUIToFP(cmp, f64_ty);
        break;
      }
      case NumericOp::NotEqual:
      {
        llvm::Value * cmp = builder.CreateFCmpUNE(load_temp(instr.src_a), load_temp(instr.src_b));
        result = builder.CreateUIToFP(cmp, f64_ty);
        break;
      }
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
      case NumericOp::Mod:
      {
        llvm::Value * lhs = load_temp(instr.src_a);
        llvm::Value * rhs = load_temp(instr.src_b);
        llvm::Value * is_zero = builder.CreateFCmpOEQ(rhs, llvm::ConstantFP::get(f64_ty, 0.0));
        llvm::Value * mod_value = builder.CreateCall(llvm_fmod, {lhs, rhs});
        result = builder.CreateSelect(is_zero, llvm::ConstantFP::get(f64_ty, 0.0), mod_value);
        break;
      }
      case NumericOp::FloorDiv:
      {
        llvm::Value * lhs = load_temp(instr.src_a);
        llvm::Value * rhs = load_temp(instr.src_b);
        llvm::Value * is_zero = builder.CreateFCmpOEQ(rhs, llvm::ConstantFP::get(f64_ty, 0.0));
        llvm::Value * div_value = builder.CreateFDiv(lhs, rhs);
        llvm::Value * floor_value = builder.CreateUnaryIntrinsic(llvm::Intrinsic::floor, div_value);
        result = builder.CreateSelect(is_zero, llvm::ConstantFP::get(f64_ty, 0.0), floor_value);
        break;
      }
      case NumericOp::BitAnd:
      {
        llvm::Value * lhs = builder.CreateFPToSI(load_temp(instr.src_a), i64_ty);
        llvm::Value * rhs = builder.CreateFPToSI(load_temp(instr.src_b), i64_ty);
        result = builder.CreateSIToFP(builder.CreateAnd(lhs, rhs), f64_ty);
        break;
      }
      case NumericOp::BitOr:
      {
        llvm::Value * lhs = builder.CreateFPToSI(load_temp(instr.src_a), i64_ty);
        llvm::Value * rhs = builder.CreateFPToSI(load_temp(instr.src_b), i64_ty);
        result = builder.CreateSIToFP(builder.CreateOr(lhs, rhs), f64_ty);
        break;
      }
      case NumericOp::BitXor:
      {
        llvm::Value * lhs = builder.CreateFPToSI(load_temp(instr.src_a), i64_ty);
        llvm::Value * rhs = builder.CreateFPToSI(load_temp(instr.src_b), i64_ty);
        result = builder.CreateSIToFP(builder.CreateXor(lhs, rhs), f64_ty);
        break;
      }
      case NumericOp::LShift:
      case NumericOp::RShift:
      {
        llvm::Value * lhs = builder.CreateFPToSI(load_temp(instr.src_a), i64_ty);
        llvm::Value * shift_raw = builder.CreateFPToSI(load_temp(instr.src_b), i64_ty);
        llvm::Value * shift_non_negative = builder.CreateSelect(
          builder.CreateICmpSLT(shift_raw, builder.getInt64(0)),
          builder.getInt64(0),
          shift_raw);
        llvm::Value * shift_clamped = builder.CreateSelect(
          builder.CreateICmpSGT(shift_non_negative, builder.getInt64(63)),
          builder.getInt64(63),
          shift_non_negative);
        llvm::Value * shift_amount = shift_clamped;
        llvm::Value * shifted = instr.op == NumericOp::LShift
                                  ? builder.CreateShl(lhs, shift_amount)
                                  : builder.CreateAShr(lhs, shift_amount);
        result = builder.CreateSIToFP(shifted, f64_ty);
        break;
      }
      case NumericOp::IndexArray:
      {
        llvm::Value * array_slot_ptr = builder.CreateInBoundsGEP(i64_ty, array_sizes_arg, builder.getInt64(instr.slot_id));
        llvm::Value * array_size = builder.CreateLoad(i64_ty, array_slot_ptr);

        llvm::Value * raw_index = builder.CreateFPToSI(load_temp(instr.src_a), i64_ty);
        llvm::Value * is_negative = builder.CreateICmpSLT(raw_index, builder.getInt64(0));
        llvm::Value * in_range_upper = builder.CreateICmpULT(raw_index, array_size);
        llvm::Value * in_range = builder.CreateAnd(builder.CreateNot(is_negative), in_range_upper);

        llvm::Value * array_ptr_slot = builder.CreateInBoundsGEP(f64_ptr_ty, arrays_arg, builder.getInt64(instr.slot_id));
        llvm::Value * array_ptr = builder.CreateLoad(f64_ptr_ty, array_ptr_slot);
        llvm::Value * elem_ptr = builder.CreateInBoundsGEP(f64_ty, array_ptr, raw_index);
        llvm::Value * elem_value = builder.CreateLoad(f64_ty, elem_ptr);
        result = builder.CreateSelect(in_range, elem_value, llvm::ConstantFP::get(f64_ty, 0.0));
        break;
      }
      case NumericOp::Sin:
        result = builder.CreateCall(llvm_sin, {load_temp(instr.src_a)});
        break;
      case NumericOp::Neg:
        result = builder.CreateFNeg(load_temp(instr.src_a));
        break;
      case NumericOp::BitNot:
      {
        llvm::Value * value = builder.CreateFPToSI(load_temp(instr.src_a), i64_ty);
        result = builder.CreateSIToFP(builder.CreateNot(value), f64_ty);
        break;
      }
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

llvm::Expected<GraphKernelFn> OrcJitEngine::compile_graph_stub(
  const std::string & symbol_prefix)
{
  if (!jit_)
  {
    return llvm::make_error<llvm::StringError>(
      "ORC JIT is not available: " + init_error_,
      llvm::inconvertibleErrorCode());
  }

  auto context = std::make_unique<llvm::LLVMContext>();
  auto module = std::make_unique<llvm::Module>("egress_graph_stub", *context);
  module->setDataLayout(jit_->getDataLayout());

  llvm::IRBuilder<> builder(*context);
  llvm::Type * void_ty = builder.getVoidTy();
  llvm::Type * f64_ty = builder.getDoubleTy();
  llvm::Type * i64_ty = builder.getInt64Ty();
  llvm::Type * f64_ptr_ty = llvm::PointerType::get(*context, 0);

  llvm::FunctionType * fn_ty = llvm::FunctionType::get(void_ty, {f64_ptr_ty, i64_ty}, false);

  static std::atomic<uint64_t> graph_function_counter{0};
  const std::string function_name = symbol_prefix + "_" + std::to_string(graph_function_counter.fetch_add(1, std::memory_order_relaxed));

  llvm::Function * fn = llvm::Function::Create(fn_ty, llvm::Function::ExternalLinkage, function_name, module.get());
  auto arg_it = fn->arg_begin();
  llvm::Value * output_arg = &*arg_it++;
  output_arg->setName("output_buffer");
  llvm::Value * frame_count_arg = &*arg_it++;
  frame_count_arg->setName("frame_count");

  llvm::BasicBlock * entry = llvm::BasicBlock::Create(*context, "entry", fn);
  llvm::BasicBlock * cond_bb = llvm::BasicBlock::Create(*context, "loop.cond", fn);
  llvm::BasicBlock * body_bb = llvm::BasicBlock::Create(*context, "loop.body", fn);
  llvm::BasicBlock * exit_bb = llvm::BasicBlock::Create(*context, "exit", fn);

  builder.SetInsertPoint(entry);
  llvm::AllocaInst * i_ptr = builder.CreateAlloca(i64_ty, nullptr, "i");
  builder.CreateStore(builder.getInt64(0), i_ptr);
  builder.CreateBr(cond_bb);

  builder.SetInsertPoint(cond_bb);
  llvm::Value * i_val = builder.CreateLoad(i64_ty, i_ptr);
  llvm::Value * cond = builder.CreateICmpULT(i_val, frame_count_arg);
  builder.CreateCondBr(cond, body_bb, exit_bb);

  builder.SetInsertPoint(body_bb);
  llvm::Value * elem_ptr = builder.CreateInBoundsGEP(f64_ty, output_arg, i_val);
  builder.CreateStore(llvm::ConstantFP::get(f64_ty, 0.0), elem_ptr);
  llvm::Value * next_i = builder.CreateAdd(i_val, builder.getInt64(1));
  builder.CreateStore(next_i, i_ptr);
  builder.CreateBr(cond_bb);

  builder.SetInsertPoint(exit_bb);
  builder.CreateRetVoid();

  if (llvm::verifyFunction(*fn, &llvm::errs()) || llvm::verifyModule(*module, &llvm::errs()))
  {
    return llvm::make_error<llvm::StringError>(
      "Generated invalid LLVM IR for graph JIT stub",
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

  return reinterpret_cast<GraphKernelFn>(*addr_or_err);
}

llvm::Expected<SingleUdmGraphKernelFn> OrcJitEngine::compile_single_udm_graph_kernel(
  const std::string & symbol_prefix)
{
  if (!jit_)
  {
    return llvm::make_error<llvm::StringError>(
      "ORC JIT is not available: " + init_error_,
      llvm::inconvertibleErrorCode());
  }

  auto context = std::make_unique<llvm::LLVMContext>();
  auto module = std::make_unique<llvm::Module>("egress_graph_single_udm", *context);
  module->setDataLayout(jit_->getDataLayout());

  llvm::IRBuilder<> builder(*context);
  llvm::Type * void_ty = builder.getVoidTy();
  llvm::Type * f64_ty = builder.getDoubleTy();
  llvm::Type * i64_ty = builder.getInt64Ty();
  llvm::Type * i32_ty = builder.getInt32Ty();
  llvm::Type * f64_ptr_ty = llvm::PointerType::get(*context, 0);
  llvm::Type * i64_ptr_ty = llvm::PointerType::get(*context, 0);
  llvm::Type * i32_ptr_ty = llvm::PointerType::get(*context, 0);
  llvm::Type * u32_ptr_ty = llvm::PointerType::get(*context, 0);
  llvm::Type * f64_ptr_ptr_ty = llvm::PointerType::get(*context, 0);

  llvm::FunctionType * fn_ty = llvm::FunctionType::get(
    void_ty,
    {
      f64_ptr_ty,
      i64_ty,
      i64_ty,
      f64_ptr_ty,
      f64_ptr_ty,
      f64_ptr_ty,
      f64_ptr_ptr_ty,
      i64_ptr_ty,
      f64_ptr_ty,
      i32_ptr_ty,
      i64_ty,
      u32_ptr_ty,
      i64_ty,
      i64_ty,
      u32_ptr_ty,
      i64_ty,
      f64_ty,
      i64_ptr_ty,
    },
    false);

  static std::atomic<uint64_t> graph_function_counter{0};
  const std::string function_name = symbol_prefix + "_" + std::to_string(graph_function_counter.fetch_add(1, std::memory_order_relaxed));

  llvm::Function * fn = llvm::Function::Create(fn_ty, llvm::Function::ExternalLinkage, function_name, module.get());
  auto arg_it = fn->arg_begin();
  llvm::Value * output_buffer_arg = &*arg_it++;
  llvm::Value * frame_count_arg = &*arg_it++;
  llvm::Value * module_kernel_addr_arg = &*arg_it++;
  llvm::Value * module_inputs_arg = &*arg_it++;
  llvm::Value * module_registers_arg = &*arg_it++;
  llvm::Value * module_next_registers_arg = &*arg_it++;
  llvm::Value * module_arrays_arg = &*arg_it++;
  llvm::Value * module_array_sizes_arg = &*arg_it++;
  llvm::Value * module_temps_arg = &*arg_it++;
  llvm::Value * register_targets_arg = &*arg_it++;
  llvm::Value * register_target_count_arg = &*arg_it++;
  llvm::Value * output_targets_arg = &*arg_it++;
  llvm::Value * output_target_count_arg = &*arg_it++;
  llvm::Value * temp_count_arg = &*arg_it++;
  llvm::Value * mix_output_ids_arg = &*arg_it++;
  llvm::Value * mix_count_arg = &*arg_it++;
  llvm::Value * sample_rate_arg = &*arg_it++;
  llvm::Value * sample_index_io_arg = &*arg_it++;

  output_buffer_arg->setName("output_buffer");
  frame_count_arg->setName("frame_count");
  module_kernel_addr_arg->setName("module_kernel_addr");
  module_inputs_arg->setName("module_inputs");
  module_registers_arg->setName("module_registers");
  module_next_registers_arg->setName("module_next_registers");
  module_arrays_arg->setName("module_arrays");
  module_array_sizes_arg->setName("module_array_sizes");
  module_temps_arg->setName("module_temps");
  register_targets_arg->setName("register_targets");
  register_target_count_arg->setName("register_target_count");
  output_targets_arg->setName("output_targets");
  output_target_count_arg->setName("output_target_count");
  temp_count_arg->setName("temp_count");
  mix_output_ids_arg->setName("mix_output_ids");
  mix_count_arg->setName("mix_count");
  sample_rate_arg->setName("sample_rate");
  sample_index_io_arg->setName("sample_index_io");

  llvm::FunctionType * module_kernel_ty = llvm::FunctionType::get(
    void_ty,
    {f64_ptr_ty, f64_ptr_ty, f64_ptr_ptr_ty, i64_ptr_ty, f64_ptr_ty, f64_ty, i64_ty},
    false);
  llvm::FunctionType * mix_fn_ty = llvm::FunctionType::get(
    f64_ty,
    {f64_ptr_ty, u32_ptr_ty, i64_ty, u32_ptr_ty, i64_ty, i64_ty},
    false);
  llvm::FunctionType * apply_fn_ty = llvm::FunctionType::get(
    void_ty,
    {f64_ptr_ty, f64_ptr_ty, f64_ptr_ty, i32_ptr_ty, i64_ty, i64_ty},
    false);

  llvm::FunctionCallee mix_fn = module->getOrInsertFunction("egress_graph_mix_single_udm", mix_fn_ty);
  llvm::FunctionCallee apply_fn = module->getOrInsertFunction("egress_graph_apply_register_targets", apply_fn_ty);

  llvm::Value * module_kernel_ptr = builder.CreateIntToPtr(module_kernel_addr_arg, llvm::PointerType::getUnqual(module_kernel_ty));

  llvm::BasicBlock * entry = llvm::BasicBlock::Create(*context, "entry", fn);
  llvm::BasicBlock * cond_bb = llvm::BasicBlock::Create(*context, "loop.cond", fn);
  llvm::BasicBlock * body_bb = llvm::BasicBlock::Create(*context, "loop.body", fn);
  llvm::BasicBlock * done_bb = llvm::BasicBlock::Create(*context, "loop.done", fn);

  builder.SetInsertPoint(entry);
  llvm::AllocaInst * i_ptr = builder.CreateAlloca(i64_ty, nullptr, "i");
  llvm::AllocaInst * sample_index_ptr = builder.CreateAlloca(i64_ty, nullptr, "sample_index");
  builder.CreateStore(builder.getInt64(0), i_ptr);
  builder.CreateStore(builder.CreateLoad(i64_ty, sample_index_io_arg), sample_index_ptr);
  builder.CreateBr(cond_bb);

  builder.SetInsertPoint(cond_bb);
  llvm::Value * i_val = builder.CreateLoad(i64_ty, i_ptr);
  llvm::Value * keep_going = builder.CreateICmpULT(i_val, frame_count_arg);
  builder.CreateCondBr(keep_going, body_bb, done_bb);

  builder.SetInsertPoint(body_bb);
  llvm::Value * cur_sample_index = builder.CreateLoad(i64_ty, sample_index_ptr);
  builder.CreateCall(module_kernel_ty, module_kernel_ptr, {
    module_inputs_arg,
    module_registers_arg,
    module_arrays_arg,
    module_array_sizes_arg,
    module_temps_arg,
    sample_rate_arg,
    cur_sample_index,
  });

  llvm::Value * mixed = builder.CreateCall(
    mix_fn,
    {module_temps_arg, output_targets_arg, output_target_count_arg, mix_output_ids_arg, mix_count_arg, temp_count_arg});
  llvm::Value * out_ptr = builder.CreateInBoundsGEP(f64_ty, output_buffer_arg, i_val);
  builder.CreateStore(mixed, out_ptr);

  builder.CreateCall(apply_fn, {
    module_registers_arg,
    module_next_registers_arg,
    module_temps_arg,
    register_targets_arg,
    register_target_count_arg,
    temp_count_arg,
  });

  llvm::Value * next_sample_index = builder.CreateAdd(cur_sample_index, builder.getInt64(1));
  builder.CreateStore(next_sample_index, sample_index_ptr);
  llvm::Value * i_next = builder.CreateAdd(i_val, builder.getInt64(1));
  builder.CreateStore(i_next, i_ptr);
  builder.CreateBr(cond_bb);

  builder.SetInsertPoint(done_bb);
  builder.CreateStore(builder.CreateLoad(i64_ty, sample_index_ptr), sample_index_io_arg);
  builder.CreateRetVoid();

  if (llvm::verifyFunction(*fn, &llvm::errs()) || llvm::verifyModule(*module, &llvm::errs()))
  {
    return llvm::make_error<llvm::StringError>(
      "Generated invalid LLVM IR for single-UDM graph kernel",
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

  return reinterpret_cast<SingleUdmGraphKernelFn>(*addr_or_err);
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
