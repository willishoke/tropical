#include "jit/OrcJitEngine.hpp"

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

namespace egress_jit
{

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


// ---------------------------------------------------------------------------
// compile_flat_program — emission path for egress_plan_4 / FlatProgram.
//
// Terminals (Const, Input, Reg, StateReg, Param, Rate, Tick) are Operands
// embedded in instructions rather than separate pseudo-ops.  loop_count > 1
// drives an elementwise loop; strides[i] selects per-iteration GEP (1) or
// broadcast scalar (0) for each argument.
// ---------------------------------------------------------------------------

llvm::Expected<NumericKernelFn> OrcJitEngine::compile_flat_program(
  const FlatProgram & program)
{
  if (!jit_)
  {
    return llvm::make_error<llvm::StringError>(
      "ORC JIT is not available: " + init_error_,
      llvm::inconvertibleErrorCode());
  }

  std::lock_guard<std::mutex> lock(jit_mutex_);

  // Build canonical param_ptr → ordinal map (order of first appearance).
  std::unordered_map<uint64_t, uint64_t> param_index;
  for (const auto & instr : program.instructions)
  {
    for (const auto & arg : instr.args)
    {
      if (arg.kind == OperandKind::Param && arg.ptr != 0 &&
          param_index.find(arg.ptr) == param_index.end())
      {
        param_index.emplace(arg.ptr, static_cast<uint64_t>(param_index.size()));
      }
    }
  }

  // Serialize to a canonical cache key (param ptrs replaced by ordinal).
  std::string cache_key;
  cache_key += "flat:";
  {
    auto append = [&](const void * data, std::size_t size) {
      cache_key.append(static_cast<const char *>(data), size);
    };
    append(&program.register_count, sizeof(uint32_t));
    uint32_t n = static_cast<uint32_t>(program.instructions.size());
    append(&n, sizeof(uint32_t));
    for (const auto & instr : program.instructions)
    {
      append(&instr.tag,        sizeof(OpTag));
      append(&instr.dst,        sizeof(uint32_t));
      append(&instr.loop_count, sizeof(uint32_t));
      uint32_t na = static_cast<uint32_t>(instr.args.size());
      append(&na, sizeof(uint32_t));
      for (const auto & arg : instr.args)
      {
        append(&arg.kind,      sizeof(OperandKind));
        append(&arg.const_val, sizeof(double));
        append(&arg.slot,      sizeof(uint32_t));
        uint64_t canonical_ptr = 0;
        if (arg.kind == OperandKind::Param && arg.ptr != 0)
        {
          auto it = param_index.find(arg.ptr);
          if (it != param_index.end()) canonical_ptr = it->second;
        }
        append(&canonical_ptr, sizeof(uint64_t));
      }
      uint32_t ns = static_cast<uint32_t>(instr.strides.size());
      append(&ns, sizeof(uint32_t));
      if (!instr.strides.empty())
        append(instr.strides.data(), instr.strides.size());
    }
    // Include mix_output_temps in cache key (different mixes → different kernels)
    uint32_t nm = static_cast<uint32_t>(program.mix_output_temps.size());
    append(&nm, sizeof(uint32_t));
    for (uint32_t mt : program.mix_output_temps)
      append(&mt, sizeof(uint32_t));
  }

  auto cache_it = kernel_cache_.find(cache_key);
  if (cache_it != kernel_cache_.end())
    return cache_it->second;

  const std::string hash = md5_hex(cache_key);
  const std::string function_name = "egress_f_" + hash;

  auto context = std::make_unique<llvm::LLVMContext>();
  auto module = std::make_unique<llvm::Module>(hash, *context);
  module->setDataLayout(jit_->getDataLayout());

  llvm::IRBuilder<> builder(*context);
  llvm::Type * void_ty  = builder.getVoidTy();
  llvm::Type * f64_ty   = builder.getDoubleTy();
  llvm::Type * i64_ty   = builder.getInt64Ty();
  llvm::Type * ptr_ty   = llvm::PointerType::get(*context, 0);

  llvm::FunctionType * fn_ty = llvm::FunctionType::get(
    void_ty,
    {ptr_ty, ptr_ty, ptr_ty, ptr_ty, ptr_ty, f64_ty, i64_ty, ptr_ty, ptr_ty, i64_ty},
    false);

  llvm::Function * fn = llvm::Function::Create(
    fn_ty, llvm::Function::ExternalLinkage, function_name, module.get());

  auto arg_it = fn->arg_begin();
  llvm::Value * inputs_arg      = &*arg_it++;  inputs_arg->setName("inputs");
  llvm::Value * regs_arg        = &*arg_it++;  regs_arg->setName("registers");
  llvm::Value * arrays_arg      = &*arg_it++;  arrays_arg->setName("arrays");
  llvm::Value * array_sizes_arg = &*arg_it++;  array_sizes_arg->setName("array_sizes");
  llvm::Value * temps_arg       = &*arg_it++;  temps_arg->setName("temps");
  llvm::Value * sample_rate_arg      = &*arg_it++;  sample_rate_arg->setName("sample_rate");
  llvm::Value * start_sample_idx_arg = &*arg_it++;  start_sample_idx_arg->setName("start_sample_index");
  llvm::Value * param_ptrs_arg       = &*arg_it++;  param_ptrs_arg->setName("param_ptrs");
  llvm::Value * output_buffer_arg    = &*arg_it++;  output_buffer_arg->setName("output_buffer");
  llvm::Value * buffer_length_arg    = &*arg_it++;  buffer_length_arg->setName("buffer_length");

  llvm::BasicBlock * entry = llvm::BasicBlock::Create(*context, "entry", fn);
  builder.SetInsertPoint(entry);

  // ── Slot helpers ──
  auto gep_temp = [&](uint32_t idx) {
    return builder.CreateInBoundsGEP(i64_ty, temps_arg, builder.getInt64(idx));
  };
  auto load_temp_f64 = [&](uint32_t idx) -> llvm::Value * {
    return builder.CreateBitCast(builder.CreateLoad(i64_ty, gep_temp(idx)), f64_ty);
  };
  auto store_temp_f64 = [&](uint32_t idx, llvm::Value * v) {
    builder.CreateStore(builder.CreateBitCast(v, i64_ty), gep_temp(idx));
  };
  auto load_reg_f64 = [&](uint32_t slot) -> llvm::Value * {
    llvm::Value * ptr = builder.CreateInBoundsGEP(i64_ty, regs_arg, builder.getInt64(slot));
    return builder.CreateBitCast(builder.CreateLoad(i64_ty, ptr), f64_ty);
  };
  auto load_input_f64 = [&](uint32_t slot) -> llvm::Value * {
    llvm::Value * ptr = builder.CreateInBoundsGEP(i64_ty, inputs_arg, builder.getInt64(slot));
    return builder.CreateBitCast(builder.CreateLoad(i64_ty, ptr), f64_ty);
  };
  auto load_array_ptr_f = [&](uint32_t slot) -> llvm::Value * {
    llvm::Value * pp = builder.CreateInBoundsGEP(ptr_ty, arrays_arg, builder.getInt64(slot));
    return builder.CreateLoad(ptr_ty, pp);
  };
  auto load_array_size_f = [&](uint32_t slot) -> llvm::Value * {
    llvm::Value * sp = builder.CreateInBoundsGEP(i64_ty, array_sizes_arg, builder.getInt64(slot));
    return builder.CreateLoad(i64_ty, sp);
  };

  // ── Intrinsics ──
  llvm::Value * zero_f64 = llvm::ConstantFP::get(f64_ty, 0.0);
  auto intr1 = [&](llvm::Intrinsic::ID id) {
    return llvm::Intrinsic::getOrInsertDeclaration(module.get(), id, {f64_ty});
  };
  llvm::FunctionCallee llvm_sin   = intr1(llvm::Intrinsic::sin);
  llvm::FunctionCallee llvm_cos   = intr1(llvm::Intrinsic::cos);
  llvm::FunctionCallee llvm_log   = intr1(llvm::Intrinsic::log);
  llvm::FunctionCallee llvm_exp   = intr1(llvm::Intrinsic::exp);
  llvm::FunctionCallee llvm_pow   = intr1(llvm::Intrinsic::pow);
  llvm::FunctionCallee llvm_sqrt  = intr1(llvm::Intrinsic::sqrt);
  llvm::FunctionCallee llvm_floor = intr1(llvm::Intrinsic::floor);
  llvm::FunctionCallee llvm_ceil  = intr1(llvm::Intrinsic::ceil);
  llvm::FunctionCallee llvm_round = intr1(llvm::Intrinsic::round);
  llvm::FunctionCallee llvm_fabs  = intr1(llvm::Intrinsic::fabs);
  llvm::FunctionCallee llvm_fmod  = module->getOrInsertFunction(
    "fmod", llvm::FunctionType::get(f64_ty, {f64_ty, f64_ty}, false));

  // current_sample_idx is set inside the buffer loop (PHI node)
  llvm::Value * current_sample_idx = nullptr;

  // ── resolve_scalar: Operand → llvm::Value* (f64 or nullptr for array/param) ──
  auto resolve_scalar = [&](const Operand & op) -> llvm::Value * {
    switch (op.kind)
    {
      case OperandKind::Const:    return llvm::ConstantFP::get(f64_ty, op.const_val);
      case OperandKind::Input:    return load_input_f64(op.slot);
      case OperandKind::Reg:      return load_temp_f64(op.slot);
      case OperandKind::StateReg: return load_reg_f64(op.slot);
      case OperandKind::Rate:     return sample_rate_arg;
      case OperandKind::Tick:     return builder.CreateSIToFP(current_sample_idx, f64_ty);
      case OperandKind::ArrayReg:
      case OperandKind::Param:    return nullptr;  // handled specially
    }
    return nullptr;
  };

  // ── emit_scalar_op: OpTag × [f64] → f64 ──
  auto emit_scalar_op = [&](OpTag tag, const std::vector<llvm::Value *> & v) -> llvm::Value * {
    switch (tag)
    {
      case OpTag::Add:      return builder.CreateFAdd(v[0], v[1]);
      case OpTag::Sub:      return builder.CreateFSub(v[0], v[1]);
      case OpTag::Mul:      return builder.CreateFMul(v[0], v[1]);
      case OpTag::Div:
      {
        llvm::Value * safe = builder.CreateSelect(builder.CreateFCmpOEQ(v[1], zero_f64), zero_f64, builder.CreateFDiv(v[0], v[1]));
        return safe;
      }
      case OpTag::Mod:
      {
        llvm::Value * safe = builder.CreateSelect(builder.CreateFCmpOEQ(v[1], zero_f64), zero_f64, builder.CreateCall(llvm_fmod, {v[0], v[1]}));
        return safe;
      }
      case OpTag::Pow:      return builder.CreateCall(llvm_pow, {v[0], v[1]});
      case OpTag::FloorDiv:
      {
        llvm::Value * dv = builder.CreateSelect(builder.CreateFCmpOEQ(v[1], zero_f64), zero_f64, builder.CreateFDiv(v[0], v[1]));
        return builder.CreateCall(llvm_floor, {dv});
      }
      case OpTag::Less:      return builder.CreateUIToFP(builder.CreateFCmpOLT(v[0], v[1]), f64_ty);
      case OpTag::LessEq:    return builder.CreateUIToFP(builder.CreateFCmpOLE(v[0], v[1]), f64_ty);
      case OpTag::Greater:   return builder.CreateUIToFP(builder.CreateFCmpOGT(v[0], v[1]), f64_ty);
      case OpTag::GreaterEq: return builder.CreateUIToFP(builder.CreateFCmpOGE(v[0], v[1]), f64_ty);
      case OpTag::Equal:     return builder.CreateUIToFP(builder.CreateFCmpOEQ(v[0], v[1]), f64_ty);
      case OpTag::NotEqual:  return builder.CreateUIToFP(builder.CreateFCmpUNE(v[0], v[1]), f64_ty);
      case OpTag::BitAnd:
      {
        llvm::Value * r = builder.CreateAnd(builder.CreateFPToSI(v[0], i64_ty), builder.CreateFPToSI(v[1], i64_ty));
        return builder.CreateSIToFP(r, f64_ty);
      }
      case OpTag::BitOr:
      {
        llvm::Value * r = builder.CreateOr(builder.CreateFPToSI(v[0], i64_ty), builder.CreateFPToSI(v[1], i64_ty));
        return builder.CreateSIToFP(r, f64_ty);
      }
      case OpTag::BitXor:
      {
        llvm::Value * r = builder.CreateXor(builder.CreateFPToSI(v[0], i64_ty), builder.CreateFPToSI(v[1], i64_ty));
        return builder.CreateSIToFP(r, f64_ty);
      }
      case OpTag::LShift:
      {
        llvm::Value * r = builder.CreateShl(builder.CreateFPToSI(v[0], i64_ty), builder.CreateFPToSI(v[1], i64_ty));
        return builder.CreateSIToFP(r, f64_ty);
      }
      case OpTag::RShift:
      {
        llvm::Value * r = builder.CreateAShr(builder.CreateFPToSI(v[0], i64_ty), builder.CreateFPToSI(v[1], i64_ty));
        return builder.CreateSIToFP(r, f64_ty);
      }
      case OpTag::Neg:    return builder.CreateFNeg(v[0]);
      case OpTag::Abs:    return builder.CreateCall(llvm_fabs,  {v[0]});
      case OpTag::Sin:    return builder.CreateCall(llvm_sin,   {v[0]});
      case OpTag::Cos:    return builder.CreateCall(llvm_cos,   {v[0]});
      case OpTag::Log:    return builder.CreateCall(llvm_log,   {v[0]});
      case OpTag::Exp:    return builder.CreateCall(llvm_exp,   {v[0]});
      case OpTag::Sqrt:   return builder.CreateCall(llvm_sqrt,  {v[0]});
      case OpTag::Floor:  return builder.CreateCall(llvm_floor, {v[0]});
      case OpTag::Ceil:   return builder.CreateCall(llvm_ceil,  {v[0]});
      case OpTag::Round:  return builder.CreateCall(llvm_round, {v[0]});
      case OpTag::Not:
      {
        llvm::Value * is_zero = builder.CreateFCmpOEQ(v[0], zero_f64);
        return builder.CreateUIToFP(is_zero, f64_ty);
      }
      case OpTag::BitNot:
        return builder.CreateSIToFP(builder.CreateNot(builder.CreateFPToSI(v[0], i64_ty)), f64_ty);
      case OpTag::Clamp:
      {
        // clamp(val, lo, hi)
        llvm::Value * lo_clamped = builder.CreateSelect(builder.CreateFCmpOGT(v[0], v[1]), v[0], v[1]);
        return builder.CreateSelect(builder.CreateFCmpOLT(lo_clamped, v[2]), lo_clamped, v[2]);
      }
      case OpTag::Select:
      {
        // select(cond, then, else)
        llvm::Value * cond = builder.CreateFCmpUNE(v[0], zero_f64);
        return builder.CreateSelect(cond, v[1], v[2]);
      }
      default:
        return nullptr;
    }
  };

  // ── Buffer loop: process buffer_length samples ──
  llvm::BasicBlock * loop_cond_bb = llvm::BasicBlock::Create(*context, "loop_cond", fn);
  llvm::BasicBlock * loop_body_bb = llvm::BasicBlock::Create(*context, "loop_body", fn);
  llvm::BasicBlock * loop_end_bb  = llvm::BasicBlock::Create(*context, "loop_end",  fn);

  builder.CreateBr(loop_cond_bb);

  // Loop condition: %s < buffer_length
  builder.SetInsertPoint(loop_cond_bb);
  llvm::PHINode * loop_counter = builder.CreatePHI(i64_ty, 2, "s");
  loop_counter->addIncoming(builder.getInt64(0), entry);
  current_sample_idx = builder.CreateAdd(start_sample_idx_arg, loop_counter, "current_idx");
  llvm::Value * loop_cond_val = builder.CreateICmpULT(loop_counter, buffer_length_arg);
  builder.CreateCondBr(loop_cond_val, loop_body_bb, loop_end_bb);

  // Loop body: all instructions + writeback + output mixing
  builder.SetInsertPoint(loop_body_bb);

  // ── Main instruction loop ──
  for (const auto & instr : program.instructions)
  {
    // ── SmoothParam ──
    if (instr.tag == OpTag::SmoothParam)
    {
      // args: [Param(ptr), StateReg(slot), Const(coeff)]
      const uint64_t param_ptr = instr.args[0].ptr;
      const uint32_t state_slot = instr.args[1].slot;
      const double coeff = instr.args[2].const_val;

      llvm::Value * reg_gep = builder.CreateInBoundsGEP(i64_ty, regs_arg, builder.getInt64(state_slot));
      llvm::Value * current  = builder.CreateBitCast(builder.CreateLoad(i64_ty, reg_gep), f64_ty);

      auto pi = param_index.find(param_ptr);
      uint64_t canonical_idx = (pi != param_index.end()) ? pi->second : 0;
      llvm::Value * pp_slot = builder.CreateInBoundsGEP(i64_ty, param_ptrs_arg, builder.getInt64(canonical_idx));
      llvm::Value * pp_raw  = builder.CreateLoad(i64_ty, pp_slot);
      llvm::Value * pp_addr = builder.CreateIntToPtr(pp_raw, ptr_ty);
      auto * target_ld = builder.CreateAlignedLoad(f64_ty, pp_addr, llvm::Align(sizeof(double)));
      target_ld->setAtomic(llvm::AtomicOrdering::Monotonic);

      llvm::Value * diff    = builder.CreateFSub(target_ld, current);
      llvm::Value * new_val = builder.CreateFAdd(current, builder.CreateFMul(llvm::ConstantFP::get(f64_ty, coeff), diff));
      builder.CreateStore(builder.CreateBitCast(new_val, i64_ty), reg_gep);
      store_temp_f64(instr.dst, new_val);
      continue;
    }

    // ── TriggerParam ── (reads frame_value via ptr; no smoothing)
    if (instr.tag == OpTag::TriggerParam)
    {
      // ControlParam::frame_value is the second double field after value.
      // Offset: sizeof(std::atomic<double>) ≈ 8 bytes for value + 8 bytes for frame_value.
      // We read frame_value via a pointer offset identical to the existing NumericOp path.
      const uint64_t param_ptr = instr.args[0].ptr;
      auto pi = param_index.find(param_ptr);
      uint64_t canonical_idx = (pi != param_index.end()) ? pi->second : 0;
      llvm::Value * pp_slot = builder.CreateInBoundsGEP(i64_ty, param_ptrs_arg, builder.getInt64(canonical_idx));
      llvm::Value * pp_raw  = builder.CreateLoad(i64_ty, pp_slot);
      // frame_value is at offset +16 bytes (two atomic<double> fields: value, frame_value)
      llvm::Value * pp_base = builder.CreateIntToPtr(pp_raw, ptr_ty);
      llvm::Value * fv_ptr  = builder.CreateInBoundsGEP(
        builder.getInt8Ty(), pp_base, builder.getInt64(16));
      auto * fv_ld = builder.CreateAlignedLoad(f64_ty, fv_ptr, llvm::Align(sizeof(double)));
      fv_ld->setAtomic(llvm::AtomicOrdering::Monotonic);
      store_temp_f64(instr.dst, fv_ld);
      continue;
    }

    // ── Pack: N scalar args → arrays[dst] ──
    if (instr.tag == OpTag::Pack)
    {
      llvm::Value * dst_ptr = load_array_ptr_f(instr.dst);
      for (std::size_t i = 0; i < instr.args.size(); ++i)
      {
        llvm::Value * val = resolve_scalar(instr.args[i]);
        llvm::Value * ep  = builder.CreateInBoundsGEP(i64_ty, dst_ptr, builder.getInt64(static_cast<int64_t>(i)));
        builder.CreateStore(builder.CreateBitCast(val, i64_ty), ep);
      }
      continue;
    }

    // ── Index: arrays[args[0].slot][args[1]] → temps[dst] ──
    if (instr.tag == OpTag::Index)
    {
      const uint32_t arr_slot = instr.args[0].slot;
      llvm::Value * array_size = load_array_size_f(arr_slot);
      llvm::Value * array_ptr  = load_array_ptr_f(arr_slot);
      llvm::Value * idx_f      = resolve_scalar(instr.args[1]);
      llvm::Value * raw_idx    = builder.CreateFPToSI(idx_f, i64_ty);
      llvm::Value * in_range   = builder.CreateAnd(
        builder.CreateNot(builder.CreateICmpSLT(raw_idx, builder.getInt64(0))),
        builder.CreateICmpULT(raw_idx, array_size));
      llvm::Value * ep  = builder.CreateInBoundsGEP(i64_ty, array_ptr, raw_idx);
      llvm::Value * val = builder.CreateBitCast(builder.CreateLoad(i64_ty, ep), f64_ty);
      store_temp_f64(instr.dst, builder.CreateSelect(in_range, val, zero_f64));
      continue;
    }

    // ── SetElement: side-effect write to arrays[args[0].slot] ──
    if (instr.tag == OpTag::SetElement)
    {
      const uint32_t arr_slot = instr.args[0].slot;
      llvm::Value * array_size = load_array_size_f(arr_slot);
      llvm::Value * array_ptr  = load_array_ptr_f(arr_slot);
      llvm::Value * raw_idx    = builder.CreateFPToSI(resolve_scalar(instr.args[1]), i64_ty);
      llvm::Value * in_range   = builder.CreateAnd(
        builder.CreateNot(builder.CreateICmpSLT(raw_idx, builder.getInt64(0))),
        builder.CreateICmpULT(raw_idx, array_size));
      llvm::BasicBlock * write_bb = llvm::BasicBlock::Create(*context, "set_elem_write", fn);
      llvm::BasicBlock * merge_bb = llvm::BasicBlock::Create(*context, "set_elem_merge", fn);
      builder.CreateCondBr(in_range, write_bb, merge_bb);
      builder.SetInsertPoint(write_bb);
      llvm::Value * ep = builder.CreateInBoundsGEP(i64_ty, array_ptr, raw_idx);
      builder.CreateStore(builder.CreateBitCast(resolve_scalar(instr.args[2]), i64_ty), ep);
      builder.CreateBr(merge_bb);
      builder.SetInsertPoint(merge_bb);
      continue;
    }

    // ── MatMul: stub (unimplemented) ──
    if (instr.tag == OpTag::MatMul)
    {
      store_temp_f64(instr.dst, zero_f64);  // TODO: implement matmul
      continue;
    }

    // ── Elementwise loop (loop_count > 1) ──
    if (instr.loop_count > 1)
    {
      // Pre-resolve non-iterating args (stride == 0) and collect array ptrs for iterating args.
      const std::size_t nargs = instr.args.size();
      std::vector<llvm::Value *> arr_ptrs(nargs, nullptr);
      std::vector<llvm::Value *> scalar_vals(nargs, nullptr);
      for (std::size_t i = 0; i < nargs; ++i)
      {
        if (i < instr.strides.size() && instr.strides[i] == 1)
          arr_ptrs[i] = load_array_ptr_f(instr.args[i].slot);
        else
          scalar_vals[i] = resolve_scalar(instr.args[i]);
      }

      llvm::Value * loop_n    = builder.getInt64(instr.loop_count);
      llvm::Value * dst_ptr   = load_array_ptr_f(instr.dst);
      llvm::Value * idx_alloc = builder.CreateAlloca(i64_ty, nullptr, "ew_idx");
      builder.CreateStore(builder.getInt64(0), idx_alloc);

      llvm::BasicBlock * cond_bb = llvm::BasicBlock::Create(*context, "ew_cond", fn);
      llvm::BasicBlock * body_bb = llvm::BasicBlock::Create(*context, "ew_body", fn);
      llvm::BasicBlock * end_bb  = llvm::BasicBlock::Create(*context, "ew_end",  fn);
      builder.CreateBr(cond_bb);

      builder.SetInsertPoint(cond_bb);
      llvm::Value * idx = builder.CreateLoad(i64_ty, idx_alloc);
      builder.CreateCondBr(builder.CreateICmpULT(idx, loop_n), body_bb, end_bb);

      builder.SetInsertPoint(body_bb);
      std::vector<llvm::Value *> iter_vals(nargs, nullptr);
      for (std::size_t i = 0; i < nargs; ++i)
      {
        if (arr_ptrs[i])
        {
          llvm::Value * ep = builder.CreateInBoundsGEP(i64_ty, arr_ptrs[i], idx);
          iter_vals[i] = builder.CreateBitCast(builder.CreateLoad(i64_ty, ep), f64_ty);
        }
        else
        {
          iter_vals[i] = scalar_vals[i];
        }
      }
      llvm::Value * elem_result = emit_scalar_op(instr.tag, iter_vals);
      if (!elem_result)
      {
        return llvm::make_error<llvm::StringError>(
          "compile_flat_program: unsupported OpTag in elementwise loop",
          llvm::inconvertibleErrorCode());
      }
      llvm::Value * dep = builder.CreateInBoundsGEP(i64_ty, dst_ptr, idx);
      builder.CreateStore(builder.CreateBitCast(elem_result, i64_ty), dep);
      builder.CreateStore(builder.CreateAdd(idx, builder.getInt64(1)), idx_alloc);
      builder.CreateBr(cond_bb);

      builder.SetInsertPoint(end_bb);
      continue;
    }

    // ── Scalar instruction ──
    std::vector<llvm::Value *> vals;
    vals.reserve(instr.args.size());
    for (const auto & arg : instr.args)
      vals.push_back(resolve_scalar(arg));

    llvm::Value * result = emit_scalar_op(instr.tag, vals);
    if (!result)
    {
      return llvm::make_error<llvm::StringError>(
        "compile_flat_program: unsupported scalar OpTag",
        llvm::inconvertibleErrorCode());
    }
    store_temp_f64(instr.dst, result);
  }

  // ── Register writeback: temps[register_targets[i]] → registers[i] ──
  // Emitted as fixed stores after all computation, so all reads of old
  // state (via StateReg operands) have already completed.
  for (uint32_t ri = 0; ri < program.register_targets.size(); ++ri)
  {
    const int32_t ti = program.register_targets[ri];
    if (ti >= 0)
    {
      llvm::Value * val     = builder.CreateLoad(i64_ty, gep_temp(static_cast<uint32_t>(ti)));
      llvm::Value * reg_ptr = builder.CreateInBoundsGEP(i64_ty, regs_arg, builder.getInt64(ri));
      builder.CreateStore(val, reg_ptr);
    }
  }

  // ── Output mixing: accumulate mix_output_temps, scale, store to output_buffer[s] ──
  {
    llvm::Value * mixed = zero_f64;
    for (uint32_t mt : program.mix_output_temps)
    {
      llvm::Value * val = load_temp_f64(mt);
      mixed = builder.CreateFAdd(mixed, val);
    }
    llvm::Value * scaled = builder.CreateFDiv(mixed, llvm::ConstantFP::get(f64_ty, 20.0));
    llvm::Value * out_ptr = builder.CreateInBoundsGEP(f64_ty, output_buffer_arg, loop_counter);
    builder.CreateStore(scaled, out_ptr);
  }

  // Loop back-edge
  llvm::Value * s_next = builder.CreateAdd(loop_counter, builder.getInt64(1), "s_next");
  loop_counter->addIncoming(s_next, builder.GetInsertBlock());
  builder.CreateBr(loop_cond_bb);

  // Loop end
  builder.SetInsertPoint(loop_end_bb);
  builder.CreateRetVoid();

  if (llvm::verifyFunction(*fn, &llvm::errs()) || llvm::verifyModule(*module, &llvm::errs()))
  {
    return llvm::make_error<llvm::StringError>(
      "compile_flat_program: generated invalid LLVM IR",
      llvm::inconvertibleErrorCode());
  }

  if (auto err = add_module(std::move(context), std::move(module)))
    return std::move(err);

  auto addr_or_err = lookup(function_name);
  if (!addr_or_err)
    return addr_or_err.takeError();

  auto kernel = reinterpret_cast<NumericKernelFn>(*addr_or_err);
  kernel_cache_.emplace(std::move(cache_key), kernel);
  return kernel;
}

} // namespace egress_jit
