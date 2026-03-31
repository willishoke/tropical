#include "jit/OrcJitEngine.hpp"

#ifdef EGRESS_LLVM_ORC_JIT
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Verifier.h>
#include <llvm/ExecutionEngine/ObjectCache.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MD5.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <unordered_map>

#if defined(__APPLE__)
#  include <dlfcn.h>
#  include <mach-o/dyld.h>
#  include <mach-o/loader.h>
#elif defined(__linux__)
#  include <dlfcn.h>
#  include <elf.h>
#  include <link.h>
#endif

namespace fs = std::filesystem;
#endif

namespace egress_jit
{
#ifdef EGRESS_LLVM_ORC_JIT

// ---------------------------------------------------------------------------
// KernelObjectCache — persists compiled object code to disk so kernels survive
// process restarts. Files are named <md5-of-canonical-program>.o and live in a
// versioned subdirectory; bump kCacheVersion when IR generation changes.
// ---------------------------------------------------------------------------

class KernelObjectCache : public llvm::ObjectCache
{
public:
  explicit KernelObjectCache(fs::path dir) : dir_(std::move(dir))
  {
    std::error_code ec;
    fs::create_directories(dir_, ec);
    // If directory creation fails the cache simply won't persist — not fatal.
  }

#if LLVM_VERSION_MAJOR >= 14
  void notifyObjectCompiled(const llvm::Module * M, llvm::MemoryBufferRef obj) override
  {
    auto path = dir_ / (M->getModuleIdentifier() + ".o");
    std::error_code ec;
    llvm::raw_fd_ostream out(path.string(), ec);
    if (!ec)
      out.write(obj.getBufferStart(), obj.getBufferSize());
  }
#else
  void notifyObjectCompiled(const llvm::Module * M, std::unique_ptr<llvm::MemoryBuffer> obj) override
  {
    auto path = dir_ / (M->getModuleIdentifier() + ".o");
    std::error_code ec;
    llvm::raw_fd_ostream out(path.string(), ec);
    if (!ec)
      out.write(obj->getBufferStart(), obj->getBufferSize());
  }
#endif

  std::unique_ptr<llvm::MemoryBuffer> getObject(const llvm::Module * M) override
  {
    auto path = dir_ / (M->getModuleIdentifier() + ".o");
    auto buf = llvm::MemoryBuffer::getFile(path.string());
    if (buf)
      return std::move(*buf);
    return nullptr;
  }

private:
  fs::path dir_;
};

static std::string md5_hex(const std::string & data)
{
  llvm::MD5 hasher;
  hasher.update(llvm::StringRef(data.data(), data.size()));
  llvm::MD5::MD5Result result;
  hasher.final(result);
  llvm::SmallString<32> hex;
  llvm::MD5::stringifyResult(result, hex);
  return std::string(hex.str());
}

// ---------------------------------------------------------------------------

OrcJitEngine & OrcJitEngine::instance()
{
  static OrcJitEngine engine;
  return engine;
}

// Returns an 8-hex-char string derived from the build ID of the shared library
// containing this code. Changes on every relink, providing automatic cache
// invalidation without a manually bumped version number.
static std::string binary_build_id()
{
#if defined(__APPLE__)
  Dl_info info;
  if (!dladdr(reinterpret_cast<void *>(&binary_build_id), &info) || !info.dli_fbase)
    return "unknown";

  // Walk Mach-O load commands looking for LC_UUID.
  const auto * hdr = static_cast<const mach_header_64 *>(info.dli_fbase);
  const auto * lc  = reinterpret_cast<const load_command *>(hdr + 1);
  for (uint32_t i = 0; i < hdr->ncmds; ++i)
  {
    if (lc->cmd == LC_UUID)
    {
      const auto * ucmd = reinterpret_cast<const uuid_command *>(lc);
      char buf[9];
      std::snprintf(buf, sizeof(buf), "%02x%02x%02x%02x",
        ucmd->uuid[0], ucmd->uuid[1], ucmd->uuid[2], ucmd->uuid[3]);
      return buf;
    }
    lc = reinterpret_cast<const load_command *>(
      reinterpret_cast<const char *>(lc) + lc->cmdsize);
  }
  return "unknown";

#elif defined(__linux__)
  struct Result { std::string id; const void * target; };
  Result result{"unknown", reinterpret_cast<const void *>(&binary_build_id)};

  dl_iterate_phdr(
    [](dl_phdr_info * info, std::size_t, void * data) -> int
    {
      auto * r = static_cast<Result *>(data);
      // Check if this image contains our target address.
      for (int i = 0; i < info->dlpi_phnum; ++i)
      {
        const auto & ph = info->dlpi_phdr[i];
        if (ph.p_type != PT_LOAD) continue;
        const auto start = info->dlpi_addr + ph.p_vaddr;
        if (reinterpret_cast<uintptr_t>(r->target) < start) continue;
        if (reinterpret_cast<uintptr_t>(r->target) >= start + ph.p_memsz) continue;

        // Found our image — look for PT_NOTE with build-id.
        for (int j = 0; j < info->dlpi_phnum; ++j)
        {
          const auto & note_ph = info->dlpi_phdr[j];
          if (note_ph.p_type != PT_NOTE) continue;
          const auto * p   = reinterpret_cast<const char *>(info->dlpi_addr + note_ph.p_vaddr);
          const auto * end = p + note_ph.p_filesz;
          while (p + sizeof(Elf64_Nhdr) <= end)
          {
            const auto * nhdr = reinterpret_cast<const Elf64_Nhdr *>(p);
            const char * name = p + sizeof(Elf64_Nhdr);
            const char * desc = name + ((nhdr->n_namesz + 3) & ~3u);
            if (nhdr->n_type == NT_GNU_BUILD_ID &&
                nhdr->n_namesz == 4 && std::memcmp(name, "GNU\0", 4) == 0 &&
                nhdr->n_descsz >= 4)
            {
              char buf[9];
              std::snprintf(buf, sizeof(buf), "%02x%02x%02x%02x",
                static_cast<unsigned char>(desc[0]),
                static_cast<unsigned char>(desc[1]),
                static_cast<unsigned char>(desc[2]),
                static_cast<unsigned char>(desc[3]));
              r->id = buf;
              return 1;
            }
            p = desc + ((nhdr->n_descsz + 3) & ~3u);
          }
        }
        return 1;
      }
      return 0;
    },
    &result);

  return result.id;

#else
  return "unknown";
#endif
}

static fs::path kernel_cache_dir()
{
  fs::path base;
  if (const char * xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg)
    base = fs::path(xdg);
  else if (const char * home = std::getenv("HOME"); home && *home)
    base = fs::path(home) / ".cache";
  else
    base = fs::temp_directory_path();
  return base / "egress" / "kernels" / binary_build_id();
}

OrcJitEngine::OrcJitEngine()
{
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  object_cache_ = std::make_unique<KernelObjectCache>(kernel_cache_dir());
  KernelObjectCache * cache_ptr = object_cache_.get();

  auto jit_or_err = llvm::orc::LLJITBuilder()
    .setCompileFunctionCreator(
      [cache_ptr](llvm::orc::JITTargetMachineBuilder jtmb)
        -> llvm::Expected<std::unique_ptr<llvm::orc::IRCompileLayer::IRCompiler>>
      {
        return std::make_unique<llvm::orc::ConcurrentIRCompiler>(std::move(jtmb), cache_ptr);
      })
    .create();

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
  const NumericProgram & program)
{
  if (!jit_)
  {
    return llvm::make_error<llvm::StringError>(
      "ORC JIT is not available: " + init_error_,
      llvm::inconvertibleErrorCode());
  }

  std::lock_guard<std::mutex> lock(jit_mutex_);

  // Build canonical param_ptr → index map (order of first appearance).
  std::unordered_map<uint64_t, uint64_t> param_index;
  for (const auto & instr : program.instructions)
  {
    if (instr.op == NumericOp::SmoothedParam && instr.param_ptr != 0 &&
        param_index.find(instr.param_ptr) == param_index.end())
    {
      param_index.emplace(instr.param_ptr, static_cast<uint64_t>(param_index.size()));
    }
  }

  // Serialize canonical program to a cache key (param_ptrs replaced by their index).
  std::string cache_key;
  {
    auto append = [&](const void * data, std::size_t size) {
      cache_key.append(static_cast<const char *>(data), size);
    };
    append(&program.register_count, sizeof(uint32_t));
    uint32_t instr_count = static_cast<uint32_t>(program.instructions.size());
    append(&instr_count, sizeof(uint32_t));
    for (const auto & instr : program.instructions)
    {
      append(&instr.op,        sizeof(NumericOp));
      append(&instr.dst,       sizeof(uint32_t));
      append(&instr.src_a,     sizeof(uint32_t));
      append(&instr.src_b,     sizeof(uint32_t));
      append(&instr.src_c,     sizeof(uint32_t));
      append(&instr.slot_id,   sizeof(uint32_t));
      append(&instr.literal,   sizeof(double));
      append(&instr.int_literal, sizeof(int64_t));
      append(&instr.dst_type,   sizeof(JitScalarType));
      append(&instr.src_a_type, sizeof(JitScalarType));
      append(&instr.src_b_type, sizeof(JitScalarType));
      uint64_t canonical_ptr = 0;
      if (instr.op == NumericOp::SmoothedParam && instr.param_ptr != 0)
      {
        auto it = param_index.find(instr.param_ptr);
        if (it != param_index.end())
          canonical_ptr = it->second;
      }
      append(&canonical_ptr, sizeof(uint64_t));
      uint32_t args_size = static_cast<uint32_t>(instr.args.size());
      append(&args_size, sizeof(uint32_t));
      if (!instr.args.empty())
        append(instr.args.data(), instr.args.size() * sizeof(uint32_t));
    }
  }

  auto cache_it = kernel_cache_.find(cache_key);
  if (cache_it != kernel_cache_.end())
    return cache_it->second;

  const std::string hash = md5_hex(cache_key);
  const std::string function_name = "egress_k_" + hash;

  auto context = std::make_unique<llvm::LLVMContext>();
  auto module = std::make_unique<llvm::Module>(hash, *context);
  module->setDataLayout(jit_->getDataLayout());

  llvm::IRBuilder<> builder(*context);
  llvm::Type * void_ty = builder.getVoidTy();
  llvm::Type * f64_ty = builder.getDoubleTy();
  llvm::Type * i64_ty = builder.getInt64Ty();
  llvm::Type * ptr_ty = llvm::PointerType::get(*context, 0);

  llvm::FunctionType * fn_ty = llvm::FunctionType::get(
    void_ty,
    {ptr_ty, ptr_ty, ptr_ty, ptr_ty, ptr_ty, f64_ty, i64_ty, ptr_ty},
    false);

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
  llvm::Value * param_ptrs_arg = &*arg_it++;
  param_ptrs_arg->setName("param_ptrs");

  llvm::BasicBlock * entry = llvm::BasicBlock::Create(*context, "entry", fn);
  builder.SetInsertPoint(entry);

  // All slot storage is unified int64_t*. Use bitcast to reinterpret as double.

  auto gep_slot = [&](llvm::Value * base_ptr, uint32_t index) -> llvm::Value * {
    return builder.CreateInBoundsGEP(i64_ty, base_ptr, builder.getInt64(index));
  };

  auto load_slot = [&](uint32_t index) -> llvm::Value * {
    return builder.CreateLoad(i64_ty, gep_slot(temps_arg, index));
  };

  auto store_slot = [&](uint32_t index, llvm::Value * value) {
    builder.CreateStore(value, gep_slot(temps_arg, index));
  };

  auto load_slot_as_float = [&](uint32_t index, JitScalarType type) -> llvm::Value * {
    llvm::Value * raw = builder.CreateLoad(i64_ty, gep_slot(temps_arg, index));
    if (type == JitScalarType::Float)
      return builder.CreateBitCast(raw, f64_ty);
    return builder.CreateSIToFP(raw, f64_ty);
  };

  auto load_slot_as_int = [&](uint32_t index, JitScalarType type) -> llvm::Value * {
    llvm::Value * raw = builder.CreateLoad(i64_ty, gep_slot(temps_arg, index));
    if (type != JitScalarType::Float)
      return raw;
    return builder.CreateFPToSI(builder.CreateBitCast(raw, f64_ty), i64_ty);
  };

  auto store_float_to_slot = [&](uint32_t index, llvm::Value * fval) {
    builder.CreateStore(builder.CreateBitCast(fval, i64_ty), gep_slot(temps_arg, index));
  };

  auto load_input = [&](uint32_t index, JitScalarType type) -> llvm::Value * {
    llvm::Value * raw = builder.CreateLoad(i64_ty, gep_slot(inputs_arg, index));
    if (type == JitScalarType::Float)
      return builder.CreateBitCast(raw, f64_ty);
    return raw;
  };

  auto load_reg = [&](uint32_t index, JitScalarType type) -> llvm::Value * {
    llvm::Value * raw = builder.CreateLoad(i64_ty, gep_slot(regs_arg, index));
    if (type == JitScalarType::Float)
      return builder.CreateBitCast(raw, f64_ty);
    return raw;
  };

  auto load_array_ptr = [&](uint32_t slot) -> llvm::Value * {
    llvm::Value * array_ptr_slot = builder.CreateInBoundsGEP(ptr_ty, arrays_arg, builder.getInt64(slot));
    return builder.CreateLoad(ptr_ty, array_ptr_slot);
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
    llvm::Value * lhs_elem_ptr = builder.CreateInBoundsGEP(i64_ty, lhs_ptr, idx);
    llvm::Value * rhs_elem_ptr = builder.CreateInBoundsGEP(i64_ty, rhs_ptr, idx);
    llvm::Value * lhs_val = builder.CreateBitCast(builder.CreateLoad(i64_ty, lhs_elem_ptr), f64_ty);
    llvm::Value * rhs_val = builder.CreateBitCast(builder.CreateLoad(i64_ty, rhs_elem_ptr), f64_ty);
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
    llvm::Value * dst_elem_ptr = builder.CreateInBoundsGEP(i64_ty, dst_ptr, idx);
    builder.CreateStore(builder.CreateBitCast(result, i64_ty), dst_elem_ptr);
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
    llvm::Value * lhs_elem_ptr = builder.CreateInBoundsGEP(i64_ty, lhs_ptr, idx);
    llvm::Value * lhs_val = builder.CreateBitCast(builder.CreateLoad(i64_ty, lhs_elem_ptr), f64_ty);
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
    llvm::Value * dst_elem_ptr = builder.CreateInBoundsGEP(i64_ty, dst_ptr, idx);
    builder.CreateStore(builder.CreateBitCast(result, i64_ty), dst_elem_ptr);
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
    llvm::Value * lhs_elem_ptr = builder.CreateInBoundsGEP(i64_ty, lhs_ptr, idx);
    llvm::Value * lhs_val = builder.CreateBitCast(builder.CreateLoad(i64_ty, lhs_elem_ptr), f64_ty);
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
    llvm::Value * dst_elem_ptr = builder.CreateInBoundsGEP(i64_ty, dst_ptr, idx);
    builder.CreateStore(builder.CreateBitCast(result, i64_ty), dst_elem_ptr);
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
    llvm::Value * lhs_elem_ptr = builder.CreateInBoundsGEP(i64_ty, lhs_ptr, idx);
    llvm::Value * lhs_val = builder.CreateBitCast(builder.CreateLoad(i64_ty, lhs_elem_ptr), f64_ty);
    llvm::Value * is_zero = builder.CreateFCmpOEQ(scalar, zero_f64);
    llvm::Value * mod_value = builder.CreateCall(llvm_fmod, {lhs_val, scalar});
    llvm::Value * result = builder.CreateSelect(is_zero, zero_f64, mod_value);
    llvm::Value * dst_elem_ptr = builder.CreateInBoundsGEP(i64_ty, dst_ptr, idx);
    builder.CreateStore(builder.CreateBitCast(result, i64_ty), dst_elem_ptr);
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
    llvm::Value * matrix_elem_ptr = builder.CreateInBoundsGEP(i64_ty, matrix_ptr, index);
    llvm::Value * matrix_val = builder.CreateBitCast(builder.CreateLoad(i64_ty, matrix_elem_ptr), f64_ty);
    llvm::Value * vector_elem_ptr = builder.CreateInBoundsGEP(i64_ty, vector_ptr, col);
    llvm::Value * vector_val = builder.CreateBitCast(builder.CreateLoad(i64_ty, vector_elem_ptr), f64_ty);
    llvm::Value * sum_val = builder.CreateLoad(f64_ty, sum_ptr);
    llvm::Value * product = builder.CreateFMul(matrix_val, vector_val);
    llvm::Value * next_sum = builder.CreateFAdd(sum_val, product);
    builder.CreateStore(next_sum, sum_ptr);
    llvm::Value * next_col = builder.CreateAdd(col, builder.getInt64(1));
    builder.CreateStore(next_col, col_ptr);
    builder.CreateBr(inner_cond);

    builder.SetInsertPoint(inner_end);
    llvm::Value * final_sum = builder.CreateLoad(f64_ty, sum_ptr);
    llvm::Value * dst_elem_ptr = builder.CreateInBoundsGEP(i64_ty, dst_ptr, row);
    builder.CreateStore(builder.CreateBitCast(final_sum, i64_ty), dst_elem_ptr);
    llvm::Value * next_row = builder.CreateAdd(row, builder.getInt64(1));
    builder.CreateStore(next_row, row_ptr);
    builder.CreateBr(outer_cond);

    builder.SetInsertPoint(outer_end);
  };

  for (const auto & instr : program.instructions)
  {
    // Unified result: every instruction produces either a float (result_f64)
    // or an int (result_i64). Both are stored to the same i64 slot array;
    // float results are bitcast to i64 before storing.
    llvm::Value * result_f64 = nullptr;
    llvm::Value * result_i64 = nullptr;
    bool writes_slot = true;
    switch (instr.op)
    {
      case NumericOp::Literal:
        if (instr.dst_type == JitScalarType::Float)
          result_f64 = llvm::ConstantFP::get(f64_ty, instr.literal);
        else
          result_i64 = builder.getInt64(instr.int_literal);
        break;
      case NumericOp::InputValue:
      {
        llvm::Value * raw = load_input(instr.slot_id, instr.dst_type);
        if (instr.dst_type == JitScalarType::Float)
          result_f64 = raw;
        else
          result_i64 = raw;
        break;
      }
      case NumericOp::RegisterValue:
      {
        llvm::Value * raw = load_reg(instr.slot_id, instr.dst_type);
        if (instr.dst_type == JitScalarType::Float)
          result_f64 = raw;
        else
          result_i64 = raw;
        break;
      }
      case NumericOp::SampleRate:
        result_f64 = sample_rate_arg;
        break;
      case NumericOp::SampleIndex:
        result_i64 = sample_index_arg;
        break;
      case NumericOp::Not:
      {
        llvm::Value * truthy;
        if (instr.src_a_type == JitScalarType::Float)
          truthy = builder.CreateFCmpUNE(load_slot_as_float(instr.src_a, JitScalarType::Float), llvm::ConstantFP::get(f64_ty, 0.0));
        else
          truthy = builder.CreateICmpNE(load_slot(instr.src_a), builder.getInt64(0));
        result_i64 = builder.CreateZExt(builder.CreateNot(truthy), i64_ty);
        break;
      }
      case NumericOp::Less:
      {
        const bool use_float = (instr.src_a_type == JitScalarType::Float || instr.src_b_type == JitScalarType::Float);
        llvm::Value * cmp = use_float
          ? builder.CreateFCmpOLT(load_slot_as_float(instr.src_a, instr.src_a_type), load_slot_as_float(instr.src_b, instr.src_b_type))
          : builder.CreateICmpSLT(load_slot(instr.src_a), load_slot(instr.src_b));
        result_i64 = builder.CreateZExt(cmp, i64_ty);
        break;
      }
      case NumericOp::LessEqual:
      {
        const bool use_float = (instr.src_a_type == JitScalarType::Float || instr.src_b_type == JitScalarType::Float);
        llvm::Value * cmp = use_float
          ? builder.CreateFCmpOLE(load_slot_as_float(instr.src_a, instr.src_a_type), load_slot_as_float(instr.src_b, instr.src_b_type))
          : builder.CreateICmpSLE(load_slot(instr.src_a), load_slot(instr.src_b));
        result_i64 = builder.CreateZExt(cmp, i64_ty);
        break;
      }
      case NumericOp::Greater:
      {
        const bool use_float = (instr.src_a_type == JitScalarType::Float || instr.src_b_type == JitScalarType::Float);
        llvm::Value * cmp = use_float
          ? builder.CreateFCmpOGT(load_slot_as_float(instr.src_a, instr.src_a_type), load_slot_as_float(instr.src_b, instr.src_b_type))
          : builder.CreateICmpSGT(load_slot(instr.src_a), load_slot(instr.src_b));
        result_i64 = builder.CreateZExt(cmp, i64_ty);
        break;
      }
      case NumericOp::GreaterEqual:
      {
        const bool use_float = (instr.src_a_type == JitScalarType::Float || instr.src_b_type == JitScalarType::Float);
        llvm::Value * cmp = use_float
          ? builder.CreateFCmpOGE(load_slot_as_float(instr.src_a, instr.src_a_type), load_slot_as_float(instr.src_b, instr.src_b_type))
          : builder.CreateICmpSGE(load_slot(instr.src_a), load_slot(instr.src_b));
        result_i64 = builder.CreateZExt(cmp, i64_ty);
        break;
      }
      case NumericOp::Equal:
      {
        const bool use_float = (instr.src_a_type == JitScalarType::Float || instr.src_b_type == JitScalarType::Float);
        llvm::Value * cmp = use_float
          ? builder.CreateFCmpOEQ(load_slot_as_float(instr.src_a, instr.src_a_type), load_slot_as_float(instr.src_b, instr.src_b_type))
          : builder.CreateICmpEQ(load_slot(instr.src_a), load_slot(instr.src_b));
        result_i64 = builder.CreateZExt(cmp, i64_ty);
        break;
      }
      case NumericOp::NotEqual:
      {
        const bool use_float = (instr.src_a_type == JitScalarType::Float || instr.src_b_type == JitScalarType::Float);
        llvm::Value * cmp = use_float
          ? builder.CreateFCmpUNE(load_slot_as_float(instr.src_a, instr.src_a_type), load_slot_as_float(instr.src_b, instr.src_b_type))
          : builder.CreateICmpNE(load_slot(instr.src_a), load_slot(instr.src_b));
        result_i64 = builder.CreateZExt(cmp, i64_ty);
        break;
      }
      case NumericOp::ArrayLessScalar:
      {
        llvm::Value * dst_ptr = load_array_ptr(instr.dst);
        llvm::Value * lhs_ptr = load_array_ptr(instr.src_a);
        llvm::Value * size = load_array_size(instr.dst);
        llvm::Value * scalar = load_slot_as_float(instr.src_b, instr.src_b_type);
        emit_array_scalar_compare_op(dst_ptr, lhs_ptr, scalar, size, ArrayScalarCompareOp::Less);
        writes_slot = false;
        break;
      }
      case NumericOp::ArrayLessEqualScalar:
      {
        llvm::Value * dst_ptr = load_array_ptr(instr.dst);
        llvm::Value * lhs_ptr = load_array_ptr(instr.src_a);
        llvm::Value * size = load_array_size(instr.dst);
        llvm::Value * scalar = load_slot_as_float(instr.src_b, instr.src_b_type);
        emit_array_scalar_compare_op(dst_ptr, lhs_ptr, scalar, size, ArrayScalarCompareOp::LessEqual);
        writes_slot = false;
        break;
      }
      case NumericOp::ArrayGreaterScalar:
      {
        llvm::Value * dst_ptr = load_array_ptr(instr.dst);
        llvm::Value * lhs_ptr = load_array_ptr(instr.src_a);
        llvm::Value * size = load_array_size(instr.dst);
        llvm::Value * scalar = load_slot_as_float(instr.src_b, instr.src_b_type);
        emit_array_scalar_compare_op(dst_ptr, lhs_ptr, scalar, size, ArrayScalarCompareOp::Greater);
        writes_slot = false;
        break;
      }
      case NumericOp::ArrayGreaterEqualScalar:
      {
        llvm::Value * dst_ptr = load_array_ptr(instr.dst);
        llvm::Value * lhs_ptr = load_array_ptr(instr.src_a);
        llvm::Value * size = load_array_size(instr.dst);
        llvm::Value * scalar = load_slot_as_float(instr.src_b, instr.src_b_type);
        emit_array_scalar_compare_op(dst_ptr, lhs_ptr, scalar, size, ArrayScalarCompareOp::GreaterEqual);
        writes_slot = false;
        break;
      }
      case NumericOp::ArrayEqualScalar:
      {
        llvm::Value * dst_ptr = load_array_ptr(instr.dst);
        llvm::Value * lhs_ptr = load_array_ptr(instr.src_a);
        llvm::Value * size = load_array_size(instr.dst);
        llvm::Value * scalar = load_slot_as_float(instr.src_b, instr.src_b_type);
        emit_array_scalar_compare_op(dst_ptr, lhs_ptr, scalar, size, ArrayScalarCompareOp::Equal);
        writes_slot = false;
        break;
      }
      case NumericOp::ArrayNotEqualScalar:
      {
        llvm::Value * dst_ptr = load_array_ptr(instr.dst);
        llvm::Value * lhs_ptr = load_array_ptr(instr.src_a);
        llvm::Value * size = load_array_size(instr.dst);
        llvm::Value * scalar = load_slot_as_float(instr.src_b, instr.src_b_type);
        emit_array_scalar_compare_op(dst_ptr, lhs_ptr, scalar, size, ArrayScalarCompareOp::NotEqual);
        writes_slot = false;
        break;
      }
      case NumericOp::ArrayPack:
      {
        llvm::Value * dst_ptr = load_array_ptr(instr.dst);
        for (std::size_t i = 0; i < instr.args.size(); ++i)
        {
          llvm::Value * dst_elem_ptr = builder.CreateInBoundsGEP(i64_ty, dst_ptr, builder.getInt64(static_cast<uint64_t>(i)));
          llvm::Value * val = load_slot_as_float(instr.args[i], JitScalarType::Float);
          builder.CreateStore(builder.CreateBitCast(val, i64_ty), dst_elem_ptr);
        }
        writes_slot = false;
        break;
      }
      case NumericOp::Add:
        if (instr.dst_type == JitScalarType::Float)
          result_f64 = builder.CreateFAdd(load_slot_as_float(instr.src_a, instr.src_a_type), load_slot_as_float(instr.src_b, instr.src_b_type));
        else
          result_i64 = builder.CreateAdd(load_slot(instr.src_a), load_slot(instr.src_b));
        break;
      case NumericOp::ArrayAdd:
      {
        llvm::Value * dst_ptr = load_array_ptr(instr.dst);
        llvm::Value * lhs_ptr = load_array_ptr(instr.src_a);
        llvm::Value * rhs_ptr = load_array_ptr(instr.src_b);
        llvm::Value * size = load_array_size(instr.dst);
        emit_array_binary_op(dst_ptr, lhs_ptr, rhs_ptr, size, ArrayBinaryOp::Add);
        writes_slot = false;
        break;
      }
      case NumericOp::ArrayAddScalar:
      {
        llvm::Value * dst_ptr = load_array_ptr(instr.dst);
        llvm::Value * lhs_ptr = load_array_ptr(instr.src_a);
        llvm::Value * size = load_array_size(instr.dst);
        llvm::Value * scalar = load_slot_as_float(instr.src_b, instr.src_b_type);
        emit_array_scalar_op(dst_ptr, lhs_ptr, scalar, size, ArrayBinaryOp::Add);
        writes_slot = false;
        break;
      }
      case NumericOp::Sub:
        if (instr.dst_type == JitScalarType::Float)
          result_f64 = builder.CreateFSub(load_slot_as_float(instr.src_a, instr.src_a_type), load_slot_as_float(instr.src_b, instr.src_b_type));
        else
          result_i64 = builder.CreateSub(load_slot(instr.src_a), load_slot(instr.src_b));
        break;
      case NumericOp::ArraySub:
      {
        llvm::Value * dst_ptr = load_array_ptr(instr.dst);
        llvm::Value * lhs_ptr = load_array_ptr(instr.src_a);
        llvm::Value * rhs_ptr = load_array_ptr(instr.src_b);
        llvm::Value * size = load_array_size(instr.dst);
        emit_array_binary_op(dst_ptr, lhs_ptr, rhs_ptr, size, ArrayBinaryOp::Sub);
        writes_slot = false;
        break;
      }
      case NumericOp::Mul:
        if (instr.dst_type == JitScalarType::Float)
          result_f64 = builder.CreateFMul(load_slot_as_float(instr.src_a, instr.src_a_type), load_slot_as_float(instr.src_b, instr.src_b_type));
        else
          result_i64 = builder.CreateMul(load_slot(instr.src_a), load_slot(instr.src_b));
        break;
      case NumericOp::ArrayMul:
      {
        llvm::Value * dst_ptr = load_array_ptr(instr.dst);
        llvm::Value * lhs_ptr = load_array_ptr(instr.src_a);
        llvm::Value * rhs_ptr = load_array_ptr(instr.src_b);
        llvm::Value * size = load_array_size(instr.dst);
        emit_array_binary_op(dst_ptr, lhs_ptr, rhs_ptr, size, ArrayBinaryOp::Mul);
        writes_slot = false;
        break;
      }
      case NumericOp::ArrayMulScalar:
      {
        llvm::Value * dst_ptr = load_array_ptr(instr.dst);
        llvm::Value * lhs_ptr = load_array_ptr(instr.src_a);
        llvm::Value * size = load_array_size(instr.dst);
        llvm::Value * scalar = load_slot_as_float(instr.src_b, instr.src_b_type);
        emit_array_scalar_op(dst_ptr, lhs_ptr, scalar, size, ArrayBinaryOp::Mul);
        writes_slot = false;
        break;
      }
      case NumericOp::Div:
      {
        llvm::Value * lhs = load_slot_as_float(instr.src_a, instr.src_a_type);
        llvm::Value * rhs = load_slot_as_float(instr.src_b, instr.src_b_type);
        llvm::Value * is_zero = builder.CreateFCmpOEQ(rhs, llvm::ConstantFP::get(f64_ty, 0.0));
        llvm::Value * div_value = builder.CreateFDiv(lhs, rhs);
        result_f64 = builder.CreateSelect(is_zero, llvm::ConstantFP::get(f64_ty, 0.0), div_value);
        break;
      }
      case NumericOp::ArrayDiv:
      {
        llvm::Value * dst_ptr = load_array_ptr(instr.dst);
        llvm::Value * lhs_ptr = load_array_ptr(instr.src_a);
        llvm::Value * rhs_ptr = load_array_ptr(instr.src_b);
        llvm::Value * size = load_array_size(instr.dst);
        emit_array_binary_op(dst_ptr, lhs_ptr, rhs_ptr, size, ArrayBinaryOp::Div);
        writes_slot = false;
        break;
      }
      case NumericOp::ArrayDivScalar:
      {
        llvm::Value * dst_ptr = load_array_ptr(instr.dst);
        llvm::Value * lhs_ptr = load_array_ptr(instr.src_a);
        llvm::Value * size = load_array_size(instr.dst);
        llvm::Value * scalar = load_slot_as_float(instr.src_b, instr.src_b_type);
        emit_array_scalar_op(dst_ptr, lhs_ptr, scalar, size, ArrayBinaryOp::Div);
        writes_slot = false;
        break;
      }
      case NumericOp::ArrayModScalar:
      {
        llvm::Value * dst_ptr = load_array_ptr(instr.dst);
        llvm::Value * lhs_ptr = load_array_ptr(instr.src_a);
        llvm::Value * size = load_array_size(instr.dst);
        llvm::Value * scalar = load_slot_as_float(instr.src_b, instr.src_b_type);
        emit_array_scalar_mod_op(dst_ptr, lhs_ptr, scalar, size);
        writes_slot = false;
        break;
      }
      case NumericOp::MatMul:
      {
        llvm::Value * dst_ptr = load_array_ptr(instr.dst);
        llvm::Value * matrix_ptr = load_array_ptr(instr.src_a);
        llvm::Value * vector_ptr = load_array_ptr(instr.src_b);
        emit_matmul(dst_ptr, matrix_ptr, vector_ptr, instr.src_c, instr.slot_id);
        writes_slot = false;
        break;
      }
      case NumericOp::Pow:
        result_f64 = builder.CreateCall(llvm_pow, {load_slot_as_float(instr.src_a, instr.src_a_type), load_slot_as_float(instr.src_b, instr.src_b_type)});
        break;
      case NumericOp::Mod:
      {
        if (instr.dst_type == JitScalarType::Float)
        {
          llvm::Value * lhs = load_slot_as_float(instr.src_a, instr.src_a_type);
          llvm::Value * rhs = load_slot_as_float(instr.src_b, instr.src_b_type);
          llvm::Value * is_zero = builder.CreateFCmpOEQ(rhs, llvm::ConstantFP::get(f64_ty, 0.0));
          llvm::Value * mod_value = builder.CreateCall(llvm_fmod, {lhs, rhs});
          result_f64 = builder.CreateSelect(is_zero, llvm::ConstantFP::get(f64_ty, 0.0), mod_value);
        }
        else
        {
          llvm::Value * lhs = load_slot(instr.src_a);
          llvm::Value * rhs = load_slot(instr.src_b);
          llvm::Value * is_zero = builder.CreateICmpEQ(rhs, builder.getInt64(0));
          llvm::Value * mod_value = builder.CreateSRem(lhs, rhs);
          result_i64 = builder.CreateSelect(is_zero, builder.getInt64(0), mod_value);
        }
        break;
      }
      case NumericOp::FloorDiv:
      {
        llvm::Value * lhs = load_slot_as_float(instr.src_a, instr.src_a_type);
        llvm::Value * rhs = load_slot_as_float(instr.src_b, instr.src_b_type);
        llvm::Value * is_zero = builder.CreateFCmpOEQ(rhs, llvm::ConstantFP::get(f64_ty, 0.0));
        llvm::Value * div_value = builder.CreateFDiv(lhs, rhs);
        llvm::Value * floor_value = builder.CreateUnaryIntrinsic(llvm::Intrinsic::floor, div_value);
        llvm::Value * int_value = builder.CreateFPToSI(floor_value, i64_ty);
        result_i64 = builder.CreateSelect(is_zero, builder.getInt64(0), int_value);
        break;
      }
      case NumericOp::BitAnd:
      {
        result_i64 = builder.CreateAnd(load_slot_as_int(instr.src_a, instr.src_a_type), load_slot_as_int(instr.src_b, instr.src_b_type));
        break;
      }
      case NumericOp::BitOr:
      {
        result_i64 = builder.CreateOr(load_slot_as_int(instr.src_a, instr.src_a_type), load_slot_as_int(instr.src_b, instr.src_b_type));
        break;
      }
      case NumericOp::BitXor:
      {
        result_i64 = builder.CreateXor(load_slot_as_int(instr.src_a, instr.src_a_type), load_slot_as_int(instr.src_b, instr.src_b_type));
        break;
      }
      case NumericOp::LShift:
      case NumericOp::RShift:
      {
        llvm::Value * lhs = load_slot_as_int(instr.src_a, instr.src_a_type);
        llvm::Value * shift_raw = load_slot_as_int(instr.src_b, instr.src_b_type);
        llvm::Value * shift_non_negative = builder.CreateSelect(
          builder.CreateICmpSLT(shift_raw, builder.getInt64(0)),
          builder.getInt64(0),
          shift_raw);
        llvm::Value * shift_clamped = builder.CreateSelect(
          builder.CreateICmpSGT(shift_non_negative, builder.getInt64(63)),
          builder.getInt64(63),
          shift_non_negative);
        result_i64 = instr.op == NumericOp::LShift
          ? builder.CreateShl(lhs, shift_clamped)
          : builder.CreateAShr(lhs, shift_clamped);
        break;
      }
      case NumericOp::Abs:
        if (instr.dst_type == JitScalarType::Float)
          result_f64 = builder.CreateUnaryIntrinsic(llvm::Intrinsic::fabs, load_slot_as_float(instr.src_a, instr.src_a_type));
        else
        {
          llvm::Value * v = load_slot(instr.src_a);
          llvm::Value * neg_v = builder.CreateNeg(v);
          result_i64 = builder.CreateSelect(builder.CreateICmpSLT(v, builder.getInt64(0)), neg_v, v);
        }
        break;
      case NumericOp::Clamp:
      {
        if (instr.dst_type == JitScalarType::Float)
        {
          llvm::Value * value     = load_slot_as_float(instr.src_a, instr.src_a_type);
          llvm::Value * min_value = load_slot_as_float(instr.src_b, instr.src_b_type);
          llvm::Value * max_value = load_slot_as_float(instr.src_c, instr.src_c_type);
          llvm::Value * lower_bounded = builder.CreateSelect(
            builder.CreateFCmpOLT(value, min_value), min_value, value);
          result_f64 = builder.CreateSelect(
            builder.CreateFCmpOGT(lower_bounded, max_value), max_value, lower_bounded);
        }
        else
        {
          llvm::Value * value     = load_slot(instr.src_a);
          llvm::Value * min_value = load_slot(instr.src_b);
          llvm::Value * max_value = load_slot(instr.src_c);
          llvm::Value * lower_bounded = builder.CreateSelect(
            builder.CreateICmpSLT(value, min_value), min_value, value);
          result_i64 = builder.CreateSelect(
            builder.CreateICmpSGT(lower_bounded, max_value), max_value, lower_bounded);
        }
        break;
      }
      case NumericOp::Select:
      {
        llvm::Value * cond_bool;
        if (instr.src_a_type == JitScalarType::Float)
          cond_bool = builder.CreateFCmpUNE(load_slot_as_float(instr.src_a, JitScalarType::Float), llvm::ConstantFP::get(f64_ty, 0.0));
        else
          cond_bool = builder.CreateICmpNE(load_slot(instr.src_a), builder.getInt64(0));
        if (instr.dst_type == JitScalarType::Float)
          result_f64 = builder.CreateSelect(cond_bool, load_slot_as_float(instr.src_b, instr.src_b_type), load_slot_as_float(instr.src_c, instr.src_c_type));
        else
          result_i64 = builder.CreateSelect(cond_bool, load_slot_as_int(instr.src_b, instr.src_b_type), load_slot_as_int(instr.src_c, instr.src_c_type));
        break;
      }
      case NumericOp::Log:
        result_f64 = builder.CreateCall(llvm_log, {load_slot_as_float(instr.src_a, instr.src_a_type)});
        break;
      case NumericOp::IndexArray:
      {
        llvm::Value * array_slot_ptr = builder.CreateInBoundsGEP(i64_ty, array_sizes_arg, builder.getInt64(instr.slot_id));
        llvm::Value * array_size = builder.CreateLoad(i64_ty, array_slot_ptr);

        llvm::Value * raw_index = (instr.src_a_type == JitScalarType::Float)
          ? builder.CreateFPToSI(load_slot_as_float(instr.src_a, JitScalarType::Float), i64_ty)
          : load_slot(instr.src_a);
        llvm::Value * is_negative = builder.CreateICmpSLT(raw_index, builder.getInt64(0));
        llvm::Value * in_range_upper = builder.CreateICmpULT(raw_index, array_size);
        llvm::Value * in_range = builder.CreateAnd(builder.CreateNot(is_negative), in_range_upper);

        llvm::Value * array_ptr_slot = builder.CreateInBoundsGEP(ptr_ty, arrays_arg, builder.getInt64(instr.slot_id));
        llvm::Value * array_ptr = builder.CreateLoad(ptr_ty, array_ptr_slot);
        llvm::Value * elem_ptr = builder.CreateInBoundsGEP(i64_ty, array_ptr, raw_index);
        llvm::Value * elem_bits = builder.CreateLoad(i64_ty, elem_ptr);
        llvm::Value * elem_value = builder.CreateBitCast(elem_bits, f64_ty);
        result_f64 = builder.CreateSelect(in_range, elem_value, llvm::ConstantFP::get(f64_ty, 0.0));
        break;
      }
      case NumericOp::SetArrayElement:
      {
        llvm::Value * array_size = load_array_size(instr.slot_id);
        llvm::Value * raw_index = (instr.src_a_type == JitScalarType::Float)
          ? builder.CreateFPToSI(load_slot_as_float(instr.src_a, JitScalarType::Float), i64_ty)
          : load_slot(instr.src_a);
        llvm::Value * is_negative = builder.CreateICmpSLT(raw_index, builder.getInt64(0));
        llvm::Value * in_range_upper = builder.CreateICmpULT(raw_index, array_size);
        llvm::Value * in_range = builder.CreateAnd(builder.CreateNot(is_negative), in_range_upper);

        llvm::BasicBlock * write_bb = llvm::BasicBlock::Create(*context, "set_arr_write", fn);
        llvm::BasicBlock * merge_bb = llvm::BasicBlock::Create(*context, "set_arr_merge", fn);
        builder.CreateCondBr(in_range, write_bb, merge_bb);

        builder.SetInsertPoint(write_bb);
        llvm::Value * array_ptr = load_array_ptr(instr.slot_id);
        llvm::Value * elem_ptr = builder.CreateInBoundsGEP(i64_ty, array_ptr, raw_index);
        llvm::Value * val_f64 = load_slot_as_float(instr.src_b, instr.src_b_type);
        builder.CreateStore(builder.CreateBitCast(val_f64, i64_ty), elem_ptr);
        builder.CreateBr(merge_bb);

        builder.SetInsertPoint(merge_bb);
        writes_slot = false;
        break;
      }
      case NumericOp::Sin:
        result_f64 = builder.CreateCall(llvm_sin, {load_slot_as_float(instr.src_a, instr.src_a_type)});
        break;
      case NumericOp::Neg:
        if (instr.dst_type == JitScalarType::Float)
          result_f64 = builder.CreateFNeg(load_slot_as_float(instr.src_a, instr.src_a_type));
        else
          result_i64 = builder.CreateNeg(load_slot(instr.src_a));
        break;
      case NumericOp::BitNot:
      {
        result_i64 = builder.CreateNot(load_slot(instr.src_a));
        break;
      }
      case NumericOp::SmoothedParam:
      {
        // Load current smoother state from registers[slot_id] (stored as bitcast float)
        llvm::Value * reg_ptr = gep_slot(regs_arg, instr.slot_id);
        llvm::Value * current_bits = builder.CreateLoad(i64_ty, reg_ptr, "smooth_current_bits");
        llvm::Value * current = builder.CreateBitCast(current_bits, f64_ty, "smooth_current");

        uint64_t canonical_idx = 0;
        {
          auto it = param_index.find(instr.param_ptr);
          if (it != param_index.end())
            canonical_idx = it->second;
        }
        llvm::Value * param_ptr_slot = builder.CreateInBoundsGEP(
          i64_ty, param_ptrs_arg, builder.getInt64(canonical_idx));
        llvm::Value * param_ptr_raw = builder.CreateLoad(i64_ty, param_ptr_slot, "smooth_param_raw");
        llvm::Value * param_addr = builder.CreateIntToPtr(param_ptr_raw, ptr_ty, "smooth_param_ptr");
        llvm::LoadInst * target_load = builder.CreateAlignedLoad(
          f64_ty, param_addr, llvm::Align(sizeof(double)), "smooth_target");
        target_load->setAtomic(llvm::AtomicOrdering::Monotonic);

        // new_val = current + coeff * (target - current)
        llvm::Value * coeff = llvm::ConstantFP::get(f64_ty, instr.literal);
        llvm::Value * diff = builder.CreateFSub(target_load, current, "smooth_diff");
        llvm::Value * step = builder.CreateFMul(coeff, diff, "smooth_step");
        result_f64 = builder.CreateFAdd(current, step, "smooth_new");

        // Write back smoother state as bitcast i64
        builder.CreateStore(builder.CreateBitCast(result_f64, i64_ty), reg_ptr);
        break;
      }
    }

    if (writes_slot)
    {
      if (result_f64)
        store_float_to_slot(instr.dst, result_f64);
      else if (result_i64)
        store_slot(instr.dst, result_i64);
      else
      {
        return llvm::make_error<llvm::StringError>(
          "Failed to lower numeric instruction to LLVM IR",
          llvm::inconvertibleErrorCode());
      }
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

  auto kernel = reinterpret_cast<NumericKernelFn>(*addr_or_err);
  kernel_cache_.emplace(std::move(cache_key), kernel);
  return kernel;
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
