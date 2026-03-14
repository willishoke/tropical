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

  auto load_array_ptr = [&](uint32_t slot) -> llvm::Value * {
    llvm::Value * array_ptr_slot = builder.CreateInBoundsGEP(f64_ptr_ty, arrays_arg, builder.getInt64(slot));
    return builder.CreateLoad(f64_ptr_ty, array_ptr_slot);
  };

  auto load_array_size = [&](uint32_t slot) -> llvm::Value * {
    llvm::Value * array_size_slot = builder.CreateInBoundsGEP(i64_ty, array_sizes_arg, builder.getInt64(slot));
    return builder.CreateLoad(i64_ty, array_size_slot);
  };

  llvm::Value * zero_f64 = llvm::ConstantFP::get(f64_ty, 0.0);
  llvm::FunctionCallee llvm_sin = llvm::Intrinsic::getOrInsertDeclaration(module.get(), llvm::Intrinsic::sin, {f64_ty});
  llvm::FunctionCallee llvm_pow = llvm::Intrinsic::getOrInsertDeclaration(module.get(), llvm::Intrinsic::pow, {f64_ty});
  llvm::FunctionCallee llvm_log = llvm::Intrinsic::getOrInsertDeclaration(module.get(), llvm::Intrinsic::log, {f64_ty});
  llvm::FunctionCallee llvm_fmod = module->getOrInsertFunction(
    "fmod",
    llvm::FunctionType::get(f64_ty, {f64_ty, f64_ty}, false));

  enum class ArrayBinaryOp
  {
    Add,
    Sub,
    Mul,
    Div
  };

  enum class ArrayScalarCompareOp
  {
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Equal,
    NotEqual
  };

  auto emit_array_binary_op = [&](llvm::Value * dst_ptr,
                                  llvm::Value * lhs_ptr,
                                  llvm::Value * rhs_ptr,
                                  llvm::Value * size,
                                  ArrayBinaryOp op) {
    llvm::BasicBlock * loop_cond = llvm::BasicBlock::Create(*context, "array_loop_cond", fn);
    llvm::BasicBlock * loop_body = llvm::BasicBlock::Create(*context, "array_loop_body", fn);
    llvm::BasicBlock * loop_end = llvm::BasicBlock::Create(*context, "array_loop_end", fn);

    llvm::Value * index_ptr = builder.CreateAlloca(i64_ty, nullptr, "array_idx");
    builder.CreateStore(builder.getInt64(0), index_ptr);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value * idx = builder.CreateLoad(i64_ty, index_ptr);
    llvm::Value * cond = builder.CreateICmpULT(idx, size);
    builder.CreateCondBr(cond, loop_body, loop_end);

    builder.SetInsertPoint(loop_body);
    llvm::Value * lhs_elem_ptr = builder.CreateInBoundsGEP(f64_ty, lhs_ptr, idx);
    llvm::Value * rhs_elem_ptr = builder.CreateInBoundsGEP(f64_ty, rhs_ptr, idx);
    llvm::Value * lhs_val = builder.CreateLoad(f64_ty, lhs_elem_ptr);
    llvm::Value * rhs_val = builder.CreateLoad(f64_ty, rhs_elem_ptr);
    llvm::Value * result = zero_f64;
    switch (op)
    {
      case ArrayBinaryOp::Add:
        result = builder.CreateFAdd(lhs_val, rhs_val);
        break;
      case ArrayBinaryOp::Sub:
        result = builder.CreateFSub(lhs_val, rhs_val);
        break;
      case ArrayBinaryOp::Mul:
        result = builder.CreateFMul(lhs_val, rhs_val);
        break;
      case ArrayBinaryOp::Div:
      {
        llvm::Value * is_zero = builder.CreateFCmpOEQ(rhs_val, zero_f64);
        llvm::Value * div_value = builder.CreateFDiv(lhs_val, rhs_val);
        result = builder.CreateSelect(is_zero, zero_f64, div_value);
        break;
      }
    }
    llvm::Value * dst_elem_ptr = builder.CreateInBoundsGEP(f64_ty, dst_ptr, idx);
    builder.CreateStore(result, dst_elem_ptr);
    llvm::Value * next_idx = builder.CreateAdd(idx, builder.getInt64(1));
    builder.CreateStore(next_idx, index_ptr);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_end);
  };

  auto emit_array_scalar_op = [&](llvm::Value * dst_ptr,
                                  llvm::Value * lhs_ptr,
                                  llvm::Value * scalar,
                                  llvm::Value * size,
                                  ArrayBinaryOp op) {
    llvm::BasicBlock * loop_cond = llvm::BasicBlock::Create(*context, "array_scalar_cond", fn);
    llvm::BasicBlock * loop_body = llvm::BasicBlock::Create(*context, "array_scalar_body", fn);
    llvm::BasicBlock * loop_end = llvm::BasicBlock::Create(*context, "array_scalar_end", fn);

    llvm::Value * index_ptr = builder.CreateAlloca(i64_ty, nullptr, "array_scalar_idx");
    builder.CreateStore(builder.getInt64(0), index_ptr);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value * idx = builder.CreateLoad(i64_ty, index_ptr);
    llvm::Value * cond = builder.CreateICmpULT(idx, size);
    builder.CreateCondBr(cond, loop_body, loop_end);

    builder.SetInsertPoint(loop_body);
    llvm::Value * lhs_elem_ptr = builder.CreateInBoundsGEP(f64_ty, lhs_ptr, idx);
    llvm::Value * lhs_val = builder.CreateLoad(f64_ty, lhs_elem_ptr);
    llvm::Value * result = zero_f64;
    switch (op)
    {
      case ArrayBinaryOp::Add:
        result = builder.CreateFAdd(lhs_val, scalar);
        break;
      case ArrayBinaryOp::Sub:
        result = builder.CreateFSub(lhs_val, scalar);
        break;
      case ArrayBinaryOp::Mul:
        result = builder.CreateFMul(lhs_val, scalar);
        break;
      case ArrayBinaryOp::Div:
      {
        llvm::Value * is_zero = builder.CreateFCmpOEQ(scalar, zero_f64);
        llvm::Value * div_value = builder.CreateFDiv(lhs_val, scalar);
        result = builder.CreateSelect(is_zero, zero_f64, div_value);
        break;
      }
    }
    llvm::Value * dst_elem_ptr = builder.CreateInBoundsGEP(f64_ty, dst_ptr, idx);
    builder.CreateStore(result, dst_elem_ptr);
    llvm::Value * next_idx = builder.CreateAdd(idx, builder.getInt64(1));
    builder.CreateStore(next_idx, index_ptr);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_end);
  };

  auto emit_array_scalar_compare_op = [&](llvm::Value * dst_ptr,
                                          llvm::Value * lhs_ptr,
                                          llvm::Value * scalar,
                                          llvm::Value * size,
                                          ArrayScalarCompareOp op) {
    llvm::BasicBlock * loop_cond = llvm::BasicBlock::Create(*context, "array_compare_cond", fn);
    llvm::BasicBlock * loop_body = llvm::BasicBlock::Create(*context, "array_compare_body", fn);
    llvm::BasicBlock * loop_end = llvm::BasicBlock::Create(*context, "array_compare_end", fn);

    llvm::Value * index_ptr = builder.CreateAlloca(i64_ty, nullptr, "array_compare_idx");
    builder.CreateStore(builder.getInt64(0), index_ptr);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value * idx = builder.CreateLoad(i64_ty, index_ptr);
    llvm::Value * cond = builder.CreateICmpULT(idx, size);
    builder.CreateCondBr(cond, loop_body, loop_end);

    builder.SetInsertPoint(loop_body);
    llvm::Value * lhs_elem_ptr = builder.CreateInBoundsGEP(f64_ty, lhs_ptr, idx);
    llvm::Value * lhs_val = builder.CreateLoad(f64_ty, lhs_elem_ptr);
    llvm::Value * cmp = nullptr;
    switch (op)
    {
      case ArrayScalarCompareOp::Less:
        cmp = builder.CreateFCmpOLT(lhs_val, scalar);
        break;
      case ArrayScalarCompareOp::LessEqual:
        cmp = builder.CreateFCmpOLE(lhs_val, scalar);
        break;
      case ArrayScalarCompareOp::Greater:
        cmp = builder.CreateFCmpOGT(lhs_val, scalar);
        break;
      case ArrayScalarCompareOp::GreaterEqual:
        cmp = builder.CreateFCmpOGE(lhs_val, scalar);
        break;
      case ArrayScalarCompareOp::Equal:
        cmp = builder.CreateFCmpOEQ(lhs_val, scalar);
        break;
      case ArrayScalarCompareOp::NotEqual:
        cmp = builder.CreateFCmpUNE(lhs_val, scalar);
        break;
    }
    llvm::Value * result = builder.CreateUIToFP(cmp, f64_ty);
    llvm::Value * dst_elem_ptr = builder.CreateInBoundsGEP(f64_ty, dst_ptr, idx);
    builder.CreateStore(result, dst_elem_ptr);
    llvm::Value * next_idx = builder.CreateAdd(idx, builder.getInt64(1));
    builder.CreateStore(next_idx, index_ptr);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_end);
  };

  auto emit_array_scalar_mod_op = [&](llvm::Value * dst_ptr,
                                      llvm::Value * lhs_ptr,
                                      llvm::Value * scalar,
                                      llvm::Value * size) {
    llvm::BasicBlock * loop_cond = llvm::BasicBlock::Create(*context, "array_mod_cond", fn);
    llvm::BasicBlock * loop_body = llvm::BasicBlock::Create(*context, "array_mod_body", fn);
    llvm::BasicBlock * loop_end = llvm::BasicBlock::Create(*context, "array_mod_end", fn);

    llvm::Value * index_ptr = builder.CreateAlloca(i64_ty, nullptr, "array_mod_idx");
    builder.CreateStore(builder.getInt64(0), index_ptr);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value * idx = builder.CreateLoad(i64_ty, index_ptr);
    llvm::Value * cond = builder.CreateICmpULT(idx, size);
    builder.CreateCondBr(cond, loop_body, loop_end);

    builder.SetInsertPoint(loop_body);
    llvm::Value * lhs_elem_ptr = builder.CreateInBoundsGEP(f64_ty, lhs_ptr, idx);
    llvm::Value * lhs_val = builder.CreateLoad(f64_ty, lhs_elem_ptr);
    llvm::Value * is_zero = builder.CreateFCmpOEQ(scalar, zero_f64);
    llvm::Value * mod_value = builder.CreateCall(llvm_fmod, {lhs_val, scalar});
    llvm::Value * result = builder.CreateSelect(is_zero, zero_f64, mod_value);
    llvm::Value * dst_elem_ptr = builder.CreateInBoundsGEP(f64_ty, dst_ptr, idx);
    builder.CreateStore(result, dst_elem_ptr);
    llvm::Value * next_idx = builder.CreateAdd(idx, builder.getInt64(1));
    builder.CreateStore(next_idx, index_ptr);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_end);
  };

  auto emit_matmul = [&](llvm::Value * dst_ptr,
                         llvm::Value * matrix_ptr,
                         llvm::Value * vector_ptr,
                         uint32_t rows,
                         uint32_t cols) {
    llvm::Value * row_ptr = builder.CreateAlloca(i64_ty, nullptr, "matmul_row");
    llvm::Value * col_ptr = builder.CreateAlloca(i64_ty, nullptr, "matmul_col");
    llvm::Value * sum_ptr = builder.CreateAlloca(f64_ty, nullptr, "matmul_sum");

    llvm::Value * rows_val = builder.getInt64(rows);
    llvm::Value * cols_val = builder.getInt64(cols);

    llvm::BasicBlock * outer_cond = llvm::BasicBlock::Create(*context, "matmul_outer_cond", fn);
    llvm::BasicBlock * outer_body = llvm::BasicBlock::Create(*context, "matmul_outer_body", fn);
    llvm::BasicBlock * outer_end = llvm::BasicBlock::Create(*context, "matmul_outer_end", fn);

    llvm::BasicBlock * inner_cond = llvm::BasicBlock::Create(*context, "matmul_inner_cond", fn);
    llvm::BasicBlock * inner_body = llvm::BasicBlock::Create(*context, "matmul_inner_body", fn);
    llvm::BasicBlock * inner_end = llvm::BasicBlock::Create(*context, "matmul_inner_end", fn);

    builder.CreateStore(builder.getInt64(0), row_ptr);
    builder.CreateBr(outer_cond);

    builder.SetInsertPoint(outer_cond);
    llvm::Value * row = builder.CreateLoad(i64_ty, row_ptr);
    llvm::Value * row_cond = builder.CreateICmpULT(row, rows_val);
    builder.CreateCondBr(row_cond, outer_body, outer_end);

    builder.SetInsertPoint(outer_body);
    builder.CreateStore(builder.getInt64(0), col_ptr);
    builder.CreateStore(zero_f64, sum_ptr);
    builder.CreateBr(inner_cond);

    builder.SetInsertPoint(inner_cond);
    llvm::Value * col = builder.CreateLoad(i64_ty, col_ptr);
    llvm::Value * col_cond = builder.CreateICmpULT(col, cols_val);
    builder.CreateCondBr(col_cond, inner_body, inner_end);

    builder.SetInsertPoint(inner_body);
    llvm::Value * row_offset = builder.CreateMul(row, cols_val);
    llvm::Value * index = builder.CreateAdd(row_offset, col);
    llvm::Value * matrix_elem_ptr = builder.CreateInBoundsGEP(f64_ty, matrix_ptr, index);
    llvm::Value * matrix_val = builder.CreateLoad(f64_ty, matrix_elem_ptr);
    llvm::Value * vector_elem_ptr = builder.CreateInBoundsGEP(f64_ty, vector_ptr, col);
    llvm::Value * vector_val = builder.CreateLoad(f64_ty, vector_elem_ptr);
    llvm::Value * sum_val = builder.CreateLoad(f64_ty, sum_ptr);
    llvm::Value * product = builder.CreateFMul(matrix_val, vector_val);
    llvm::Value * next_sum = builder.CreateFAdd(sum_val, product);
    builder.CreateStore(next_sum, sum_ptr);
    llvm::Value * next_col = builder.CreateAdd(col, builder.getInt64(1));
    builder.CreateStore(next_col, col_ptr);
    builder.CreateBr(inner_cond);

    builder.SetInsertPoint(inner_end);
    llvm::Value * final_sum = builder.CreateLoad(f64_ty, sum_ptr);
    llvm::Value * dst_elem_ptr = builder.CreateInBoundsGEP(f64_ty, dst_ptr, row);
    builder.CreateStore(final_sum, dst_elem_ptr);
    llvm::Value * next_row = builder.CreateAdd(row, builder.getInt64(1));
    builder.CreateStore(next_row, row_ptr);
    builder.CreateBr(outer_cond);

    builder.SetInsertPoint(outer_end);
  };

  for (const auto & instr : program.instructions)
  {
    llvm::Value * result = nullptr;
    bool writes_temp = true;
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
      case NumericOp::ArrayLessScalar:
      {
        llvm::Value * dst_ptr = load_array_ptr(instr.dst);
        llvm::Value * lhs_ptr = load_array_ptr(instr.src_a);
        llvm::Value * size = load_array_size(instr.dst);
        llvm::Value * scalar = load_temp(instr.src_b);
        emit_array_scalar_compare_op(dst_ptr, lhs_ptr, scalar, size, ArrayScalarCompareOp::Less);
        writes_temp = false;
        break;
      }
      case NumericOp::ArrayLessEqualScalar:
      {
        llvm::Value * dst_ptr = load_array_ptr(instr.dst);
        llvm::Value * lhs_ptr = load_array_ptr(instr.src_a);
        llvm::Value * size = load_array_size(instr.dst);
        llvm::Value * scalar = load_temp(instr.src_b);
        emit_array_scalar_compare_op(dst_ptr, lhs_ptr, scalar, size, ArrayScalarCompareOp::LessEqual);
        writes_temp = false;
        break;
      }
      case NumericOp::ArrayGreaterScalar:
      {
        llvm::Value * dst_ptr = load_array_ptr(instr.dst);
        llvm::Value * lhs_ptr = load_array_ptr(instr.src_a);
        llvm::Value * size = load_array_size(instr.dst);
        llvm::Value * scalar = load_temp(instr.src_b);
        emit_array_scalar_compare_op(dst_ptr, lhs_ptr, scalar, size, ArrayScalarCompareOp::Greater);
        writes_temp = false;
        break;
      }
      case NumericOp::ArrayGreaterEqualScalar:
      {
        llvm::Value * dst_ptr = load_array_ptr(instr.dst);
        llvm::Value * lhs_ptr = load_array_ptr(instr.src_a);
        llvm::Value * size = load_array_size(instr.dst);
        llvm::Value * scalar = load_temp(instr.src_b);
        emit_array_scalar_compare_op(dst_ptr, lhs_ptr, scalar, size, ArrayScalarCompareOp::GreaterEqual);
        writes_temp = false;
        break;
      }
      case NumericOp::ArrayEqualScalar:
      {
        llvm::Value * dst_ptr = load_array_ptr(instr.dst);
        llvm::Value * lhs_ptr = load_array_ptr(instr.src_a);
        llvm::Value * size = load_array_size(instr.dst);
        llvm::Value * scalar = load_temp(instr.src_b);
        emit_array_scalar_compare_op(dst_ptr, lhs_ptr, scalar, size, ArrayScalarCompareOp::Equal);
        writes_temp = false;
        break;
      }
      case NumericOp::ArrayNotEqualScalar:
      {
        llvm::Value * dst_ptr = load_array_ptr(instr.dst);
        llvm::Value * lhs_ptr = load_array_ptr(instr.src_a);
        llvm::Value * size = load_array_size(instr.dst);
        llvm::Value * scalar = load_temp(instr.src_b);
        emit_array_scalar_compare_op(dst_ptr, lhs_ptr, scalar, size, ArrayScalarCompareOp::NotEqual);
        writes_temp = false;
        break;
      }
      case NumericOp::ArrayPack:
      {
        llvm::Value * dst_ptr = load_array_ptr(instr.dst);
        for (std::size_t i = 0; i < instr.args.size(); ++i)
        {
          llvm::Value * dst_elem_ptr = builder.CreateInBoundsGEP(f64_ty, dst_ptr, builder.getInt64(static_cast<uint64_t>(i)));
          builder.CreateStore(load_temp(instr.args[i]), dst_elem_ptr);
        }
        writes_temp = false;
        break;
      }
      case NumericOp::Add:
        result = builder.CreateFAdd(load_temp(instr.src_a), load_temp(instr.src_b));
        break;
      case NumericOp::ArrayAdd:
      {
        llvm::Value * dst_ptr = load_array_ptr(instr.dst);
        llvm::Value * lhs_ptr = load_array_ptr(instr.src_a);
        llvm::Value * rhs_ptr = load_array_ptr(instr.src_b);
        llvm::Value * size = load_array_size(instr.dst);
        emit_array_binary_op(dst_ptr, lhs_ptr, rhs_ptr, size, ArrayBinaryOp::Add);
        writes_temp = false;
        break;
      }
      case NumericOp::ArrayAddScalar:
      {
        llvm::Value * dst_ptr = load_array_ptr(instr.dst);
        llvm::Value * lhs_ptr = load_array_ptr(instr.src_a);
        llvm::Value * size = load_array_size(instr.dst);
        llvm::Value * scalar = load_temp(instr.src_b);
        emit_array_scalar_op(dst_ptr, lhs_ptr, scalar, size, ArrayBinaryOp::Add);
        writes_temp = false;
        break;
      }
      case NumericOp::Sub:
        result = builder.CreateFSub(load_temp(instr.src_a), load_temp(instr.src_b));
        break;
      case NumericOp::ArraySub:
      {
        llvm::Value * dst_ptr = load_array_ptr(instr.dst);
        llvm::Value * lhs_ptr = load_array_ptr(instr.src_a);
        llvm::Value * rhs_ptr = load_array_ptr(instr.src_b);
        llvm::Value * size = load_array_size(instr.dst);
        emit_array_binary_op(dst_ptr, lhs_ptr, rhs_ptr, size, ArrayBinaryOp::Sub);
        writes_temp = false;
        break;
      }
      case NumericOp::Mul:
        result = builder.CreateFMul(load_temp(instr.src_a), load_temp(instr.src_b));
        break;
      case NumericOp::ArrayMul:
      {
        llvm::Value * dst_ptr = load_array_ptr(instr.dst);
        llvm::Value * lhs_ptr = load_array_ptr(instr.src_a);
        llvm::Value * rhs_ptr = load_array_ptr(instr.src_b);
        llvm::Value * size = load_array_size(instr.dst);
        emit_array_binary_op(dst_ptr, lhs_ptr, rhs_ptr, size, ArrayBinaryOp::Mul);
        writes_temp = false;
        break;
      }
      case NumericOp::ArrayMulScalar:
      {
        llvm::Value * dst_ptr = load_array_ptr(instr.dst);
        llvm::Value * lhs_ptr = load_array_ptr(instr.src_a);
        llvm::Value * size = load_array_size(instr.dst);
        llvm::Value * scalar = load_temp(instr.src_b);
        emit_array_scalar_op(dst_ptr, lhs_ptr, scalar, size, ArrayBinaryOp::Mul);
        writes_temp = false;
        break;
      }
      case NumericOp::Div:
      {
        llvm::Value * lhs = load_temp(instr.src_a);
        llvm::Value * rhs = load_temp(instr.src_b);
        llvm::Value * is_zero = builder.CreateFCmpOEQ(rhs, llvm::ConstantFP::get(f64_ty, 0.0));
        llvm::Value * div_value = builder.CreateFDiv(lhs, rhs);
        result = builder.CreateSelect(is_zero, llvm::ConstantFP::get(f64_ty, 0.0), div_value);
        break;
      }
      case NumericOp::ArrayDiv:
      {
        llvm::Value * dst_ptr = load_array_ptr(instr.dst);
        llvm::Value * lhs_ptr = load_array_ptr(instr.src_a);
        llvm::Value * rhs_ptr = load_array_ptr(instr.src_b);
        llvm::Value * size = load_array_size(instr.dst);
        emit_array_binary_op(dst_ptr, lhs_ptr, rhs_ptr, size, ArrayBinaryOp::Div);
        writes_temp = false;
        break;
      }
      case NumericOp::ArrayDivScalar:
      {
        llvm::Value * dst_ptr = load_array_ptr(instr.dst);
        llvm::Value * lhs_ptr = load_array_ptr(instr.src_a);
        llvm::Value * size = load_array_size(instr.dst);
        llvm::Value * scalar = load_temp(instr.src_b);
        emit_array_scalar_op(dst_ptr, lhs_ptr, scalar, size, ArrayBinaryOp::Div);
        writes_temp = false;
        break;
      }
      case NumericOp::ArrayModScalar:
      {
        llvm::Value * dst_ptr = load_array_ptr(instr.dst);
        llvm::Value * lhs_ptr = load_array_ptr(instr.src_a);
        llvm::Value * size = load_array_size(instr.dst);
        llvm::Value * scalar = load_temp(instr.src_b);
        emit_array_scalar_mod_op(dst_ptr, lhs_ptr, scalar, size);
        writes_temp = false;
        break;
      }
      case NumericOp::MatMul:
      {
        llvm::Value * dst_ptr = load_array_ptr(instr.dst);
        llvm::Value * matrix_ptr = load_array_ptr(instr.src_a);
        llvm::Value * vector_ptr = load_array_ptr(instr.src_b);
        emit_matmul(dst_ptr, matrix_ptr, vector_ptr, instr.src_c, instr.slot_id);
        writes_temp = false;
        break;
      }
      case NumericOp::Pow:
        result = builder.CreateCall(llvm_pow, {load_temp(instr.src_a), load_temp(instr.src_b)});
        break;
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
      case NumericOp::Abs:
        result = builder.CreateUnaryIntrinsic(llvm::Intrinsic::fabs, load_temp(instr.src_a));
        break;
      case NumericOp::Clamp:
      {
        llvm::Value * value = load_temp(instr.src_a);
        llvm::Value * min_value = load_temp(instr.src_b);
        llvm::Value * max_value = load_temp(instr.src_c);
        llvm::Value * lower_bounded = builder.CreateSelect(
          builder.CreateFCmpOLT(value, min_value),
          min_value,
          value);
        result = builder.CreateSelect(
          builder.CreateFCmpOGT(lower_bounded, max_value),
          max_value,
          lower_bounded);
        break;
      }
      case NumericOp::Log:
        result = builder.CreateCall(llvm_log, {load_temp(instr.src_a)});
        break;
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

    if (writes_temp && !result)
    {
      return llvm::make_error<llvm::StringError>(
        "Failed to lower numeric instruction to LLVM IR",
        llvm::inconvertibleErrorCode());
    }

    if (writes_temp)
    {
      store_temp(instr.dst, result);
    }
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
