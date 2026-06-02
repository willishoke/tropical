#include "jit/OrcJitEngine.hpp"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/ExecutionEngine/ObjectCache.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/IRTransformLayer.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MD5.h>
#if LLVM_VERSION_MAJOR >= 21
#include <llvm/Support/ModRef.h>
#endif
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <functional>
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

namespace tropical_jit
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
  return base / "tropical" / "kernels" / binary_build_id();
}

// Read TROPICAL_JIT_OPT_LEVEL env var. Accepts "O0"/"0", "O1"/"1",
// "O2"/"2", "O3"/"3", "Os", "Oz". Defaults to O2.
static llvm::OptimizationLevel parse_opt_level()
{
  const char * v = std::getenv("TROPICAL_JIT_OPT_LEVEL");
  if (!v || !*v) return llvm::OptimizationLevel::O2;
  std::string s = v;
  if (s == "O0" || s == "0") return llvm::OptimizationLevel::O0;
  if (s == "O1" || s == "1") return llvm::OptimizationLevel::O1;
  if (s == "O2" || s == "2") return llvm::OptimizationLevel::O2;
  if (s == "O3" || s == "3") return llvm::OptimizationLevel::O3;
  if (s == "Os")             return llvm::OptimizationLevel::Os;
  if (s == "Oz")             return llvm::OptimizationLevel::Oz;
  return llvm::OptimizationLevel::O2;
}

static const char * opt_level_tag(llvm::OptimizationLevel lvl)
{
  if (lvl == llvm::OptimizationLevel::O0) return "O0";
  if (lvl == llvm::OptimizationLevel::O1) return "O1";
  if (lvl == llvm::OptimizationLevel::O2) return "O2";
  if (lvl == llvm::OptimizationLevel::O3) return "O3";
  if (lvl == llvm::OptimizationLevel::Os) return "Os";
  if (lvl == llvm::OptimizationLevel::Oz) return "Oz";
  return "O2";
}

OrcJitEngine::OrcJitEngine()
{
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  object_cache_ = std::make_unique<KernelObjectCache>(kernel_cache_dir());
  KernelObjectCache * cache_ptr = object_cache_.get();

  opt_level_ = parse_opt_level();

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

  // Install an IR transform layer that runs LLVM's per-module default
  // optimization pipeline at the configured level. Without this, LLJIT
  // does codegen-only — the IR-level passes (loop-vectorize, SLP, DCE,
  // CSE, const-fold, instcombine) never run, leaving ~2x of kernel
  // performance on the table on real patches.
  llvm::OptimizationLevel level = opt_level_;
  jit_->getIRTransformLayer().setTransform(
    [level](llvm::orc::ThreadSafeModule TSM,
            const llvm::orc::MaterializationResponsibility &)
        -> llvm::Expected<llvm::orc::ThreadSafeModule>
    {
      TSM.withModuleDo([level](llvm::Module & M) {
        llvm::PassBuilder PB;
        llvm::LoopAnalysisManager LAM;
        llvm::FunctionAnalysisManager FAM;
        llvm::CGSCCAnalysisManager CGAM;
        llvm::ModuleAnalysisManager MAM;
        PB.registerModuleAnalyses(MAM);
        PB.registerCGSCCAnalyses(CGAM);
        PB.registerFunctionAnalyses(FAM);
        PB.registerLoopAnalyses(LAM);
        PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
        llvm::ModulePassManager MPM = (level == llvm::OptimizationLevel::O0)
          ? PB.buildO0DefaultPipeline(level)
          : PB.buildPerModuleDefaultPipeline(level);
        MPM.run(M, MAM);
      });
      return std::move(TSM);
    });
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
// EmitCtx — per-compilation emission context for the microkernel-mode path.
//
// Holds the LLVM builder, the in-flight function's argument bindings, the
// per-temp scalar-type table, and the cached intrinsic declarations. Methods
// on EmitCtx are the codegen shared between the N+3 LLVM functions a single
// microkernel-mode compile emits (preamble, per-instance kernels,
// state_evolution, postamble_mix). Each function gets its own EmitCtx with
// its own argument bindings; the temp_types pointer is shared across them
// so a write in one function's body is observable to a later function's
// read of the same unified-namespace temp slot.
//
// Both compile_flat_program (fused mode) and compile_microkernel
// (microkernel mode) build an EmitCtx; codegen is shared via the methods
// on this struct. The microkernel-vs-fused equivalence suite
// (tests/equiv/microkernel_vs_fused.test.ts) covers the shared path
// sample-for-sample at 1e-12.
// ---------------------------------------------------------------------------

namespace
{

using ST = JitScalarType;
using TypedVal = std::pair<llvm::Value *, ST>;

struct EmitCtx
{
  // Required references (set once at construction)
  llvm::LLVMContext * context  = nullptr;
  llvm::IRBuilder<> * builder  = nullptr;
  llvm::Module      * module   = nullptr;
  llvm::Function    * fn       = nullptr;

  // LLVM types
  llvm::Type * void_ty = nullptr;
  llvm::Type * f64_ty  = nullptr;
  llvm::Type * i64_ty  = nullptr;
  llvm::Type * i1_ty   = nullptr;
  llvm::Type * ptr_ty  = nullptr;

  // Common constants
  llvm::Value * zero_f64 = nullptr;
  llvm::Value * one_f64  = nullptr;
  llvm::Value * zero_i64 = nullptr;
  llvm::Value * zero_i1  = nullptr;

  // Cached intrinsic declarations (created per-module)
  llvm::FunctionCallee llvm_sqrt;
  llvm::FunctionCallee llvm_floor;
  llvm::FunctionCallee llvm_ceil;
  llvm::FunctionCallee llvm_round;
  llvm::FunctionCallee llvm_fabs;

  // Per-function argument values. inputs_arg may be null in microkernel
  // mode (session-built plans don't emit Input operands).
  llvm::Value * inputs_arg      = nullptr;
  llvm::Value * regs_arg        = nullptr;
  llvm::Value * arrays_arg      = nullptr;
  llvm::Value * array_sizes_arg = nullptr;
  llvm::Value * temps_arg       = nullptr;
  llvm::Value * sample_rate_arg = nullptr;
  llvm::Value * param_ptrs_arg  = nullptr;
  llvm::Value * slots_arg       = nullptr;

  // Set per-function in microkernel mode (function arg). Tick operands
  // read this.
  llvm::Value * current_sample_idx = nullptr;

  // Program context
  const FlatProgram                                   * program     = nullptr;
  const std::unordered_map<uint64_t, uint64_t>        * param_index = nullptr;

  // Per-temp scalar type table. Pointer so multiple EmitCtxs in a single
  // compile share the same symbol table.
  std::vector<ST> * temp_types = nullptr;

  // ── Deep-mode dispatch ──
  // When set, `emit_kernel_block` emits `CreateCall(child_fn, args)` for
  // each child instead of recursing inline. The map gives each
  // InstanceProgram its pre-declared LLVM function. nullptr → shallow
  // mode (inline recursion).
  const std::unordered_map<const InstanceProgram*, llvm::Function*> * deep_child_fns = nullptr;

  // ── Slot helpers ──
  llvm::Value * gep_temp(uint32_t idx) const {
    return builder->CreateInBoundsGEP(i64_ty, temps_arg, builder->getInt64(idx));
  }
  llvm::Value * load_temp_f64(uint32_t idx) const {
    return builder->CreateBitCast(builder->CreateLoad(i64_ty, gep_temp(idx)), f64_ty);
  }
  void store_temp_f64(uint32_t idx, llvm::Value * v) const {
    builder->CreateStore(builder->CreateBitCast(v, i64_ty), gep_temp(idx));
  }
  llvm::Value * load_reg_f64(uint32_t slot) const {
    llvm::Value * ptr = builder->CreateInBoundsGEP(i64_ty, regs_arg, builder->getInt64(slot));
    return builder->CreateBitCast(builder->CreateLoad(i64_ty, ptr), f64_ty);
  }
  llvm::Value * load_input_f64(uint32_t slot) const {
    llvm::Value * ptr = builder->CreateInBoundsGEP(i64_ty, inputs_arg, builder->getInt64(slot));
    return builder->CreateBitCast(builder->CreateLoad(i64_ty, ptr), f64_ty);
  }
  llvm::Value * load_array_ptr_f(uint32_t slot) const {
    llvm::Value * pp = builder->CreateInBoundsGEP(ptr_ty, arrays_arg, builder->getInt64(slot));
    return builder->CreateLoad(ptr_ty, pp);
  }
  llvm::Value * load_array_size_f(uint32_t slot) const {
    llvm::Value * sp = builder->CreateInBoundsGEP(i64_ty, array_sizes_arg, builder->getInt64(slot));
    return builder->CreateLoad(i64_ty, sp);
  }
  llvm::Value * load_slot_f64(uint32_t idx) const {
    llvm::Value * ptr = builder->CreateInBoundsGEP(f64_ty, slots_arg, builder->getInt64(idx));
    return builder->CreateLoad(f64_ty, ptr);
  }
  void store_slot_f64(uint32_t idx, llvm::Value * v) const {
    llvm::Value * ptr = builder->CreateInBoundsGEP(f64_ty, slots_arg, builder->getInt64(idx));
    builder->CreateStore(v, ptr);
  }

  // ── Typed load/store ──
  llvm::Value * load_temp_typed(uint32_t idx, ST ty) const {
    llvm::Value * raw = builder->CreateLoad(i64_ty, gep_temp(idx));
    switch (ty) {
      case ST::Float: return builder->CreateBitCast(raw, f64_ty);
      case ST::Int:   return raw;
      case ST::Bool:  return builder->CreateTrunc(raw, i1_ty);
    }
    return builder->CreateBitCast(raw, f64_ty);
  }
  void store_temp_typed(uint32_t idx, llvm::Value * v, ST ty) const {
    llvm::Value * as_i64;
    switch (ty) {
      case ST::Float: as_i64 = builder->CreateBitCast(v, i64_ty); break;
      case ST::Int:   as_i64 = v; break;
      case ST::Bool:  as_i64 = builder->CreateZExt(v, i64_ty); break;
    }
    builder->CreateStore(as_i64, gep_temp(idx));
  }
  llvm::Value * load_reg_typed(uint32_t slot, ST ty) const {
    llvm::Value * ptr = builder->CreateInBoundsGEP(i64_ty, regs_arg, builder->getInt64(slot));
    llvm::Value * raw = builder->CreateLoad(i64_ty, ptr);
    switch (ty) {
      case ST::Float: return builder->CreateBitCast(raw, f64_ty);
      case ST::Int:   return raw;
      case ST::Bool:  return builder->CreateTrunc(raw, i1_ty);
    }
    return builder->CreateBitCast(raw, f64_ty);
  }
  llvm::Value * coerce_val(llvm::Value * v, ST from, ST to) const {
    if (from == to) return v;
    if (from == ST::Float && to == ST::Int)   return builder->CreateFPToSI(v, i64_ty);
    if (from == ST::Float && to == ST::Bool)  return builder->CreateFCmpUNE(v, zero_f64);
    if (from == ST::Int   && to == ST::Float) return builder->CreateSIToFP(v, f64_ty);
    if (from == ST::Int   && to == ST::Bool)  return builder->CreateICmpNE(v, zero_i64);
    if (from == ST::Bool  && to == ST::Float) return builder->CreateUIToFP(v, f64_ty);
    if (from == ST::Bool  && to == ST::Int)   return builder->CreateZExt(v, i64_ty);
    return v;
  }

  // ── Operand resolution ──
  TypedVal resolve_typed(const Operand & op) const;
  llvm::Value * resolve_as_f64(const Operand & op) const {
    auto [v, t] = resolve_typed(op);
    return v ? coerce_val(v, t, ST::Float) : nullptr;
  }

  // ── Operation emit (out-of-line; large switch) ──
  TypedVal emit_typed_op(OpTag tag, ST result_type, const std::vector<TypedVal> & tv) const;

  // ── Instruction emit (out-of-line; large switch) ──
  llvm::Error emit_instr(const FlatInstr & instr);
  llvm::Error emit_instrs(const std::vector<FlatInstr> & instrs) {
    for (const auto & instr : instrs)
      if (auto err = emit_instr(instr)) return err;
    return llvm::Error::success();
  }

  // ── Per-instance writebacks ──
  void emit_writebacks(const std::vector<InstanceProgram::Writeback> & wbs);

  // ── Per-instance dispatch (fractal interleaved order) ──
  // For each child:
  //   1. emit child.pre_input_instructions (in THIS kernel's namespace
  //      — wire expressions referencing parent regs/inputs/params resolve,
  //      results land in WriteSlot to the child's input slots)
  //   2. emit child.body recursively (reads input slots, writes output
  //      slots; nested children dispatch the same way)
  // Then:
  //   3. emit THIS kernel's main instructions (may read child outputs via
  //      NestedOut → Slot read, observing the values from step 2 of each
  //      child)
  //   4. emit THIS kernel's writebacks (state-register updates)
  //
  // Interleaving step 1 with step 2 per-child (vs hoisting all step-1
  // blocks ahead of all step-2 blocks) preserves sibling-to-sibling
  // NestedOut dependencies: child[k]'s wires can read child[j].output
  // for j < k because child[j]'s body has already executed.
  //
  // Legacy plans (inlineNested:true) have empty children and empty
  // pre_input_instructions throughout — the loop is a no-op and the
  // behavior reduces to "main body then writebacks", same as before.
  llvm::Error emit_kernel_block(const InstanceProgram & inst) {
    // Preamble runs first: temp-computes for session-wired input
    // translations whose results are referenced by child pre_input
    // WriteSlots. Must precede child dispatch.
    if (auto err = emit_instrs(inst.preamble_instructions)) return err;
    for (const auto & child : inst.children) {
      if (auto err = emit_instrs(child.pre_input_instructions)) return err;
      if (deep_child_fns != nullptr) {
        // Deep mode: each child is its own LLVM function. Call it
        // with the same per-sample args we received; the child's
        // body executes (and may recursively call its own children).
        // The four-phase order is preserved across the function
        // boundary because pre_input ran above in OUR scope, and the
        // child's body runs as a leaf operation here.
        auto it = deep_child_fns->find(&child);
        if (it == deep_child_fns->end()) {
          return llvm::make_error<llvm::StringError>(
            "emit_kernel_block: deep mode missing function for child '"
            + child.instance_name + "'",
            llvm::inconvertibleErrorCode());
        }
        llvm::Value * args[] = {
          regs_arg, arrays_arg, array_sizes_arg, temps_arg,
          sample_rate_arg, current_sample_idx, param_ptrs_arg, slots_arg,
        };
        builder->CreateCall(it->second, args);
      } else {
        // Shallow mode: inline the child's body recursively.
        if (auto err = emit_kernel_block(child)) return err;
      }
    }
    if (auto err = emit_instrs(inst.instructions)) return err;
    emit_writebacks(inst.writebacks);
    return llvm::Error::success();
  }
};

// Bind type / constant / intrinsic members from an LLVM context, builder,
// and module. Argument values (regs_arg etc.) are bound separately by the
// caller after the function is created and its args are pulled.
inline void emit_ctx_init_module(EmitCtx & c,
                                 llvm::LLVMContext & context,
                                 llvm::IRBuilder<> & builder,
                                 llvm::Module & module)
{
  c.context = &context;
  c.builder = &builder;
  c.module  = &module;

  c.void_ty = builder.getVoidTy();
  c.f64_ty  = builder.getDoubleTy();
  c.i64_ty  = builder.getInt64Ty();
  c.i1_ty   = builder.getInt1Ty();
  c.ptr_ty  = llvm::PointerType::get(context, 0);

  c.zero_f64 = llvm::ConstantFP::get(c.f64_ty, 0.0);
  c.one_f64  = llvm::ConstantFP::get(c.f64_ty, 1.0);
  c.zero_i64 = builder.getInt64(0);
  c.zero_i1  = builder.getFalse();

  auto intr1 = [&](llvm::Intrinsic::ID id) {
    return llvm::Intrinsic::getOrInsertDeclaration(&module, id, {c.f64_ty});
  };
  c.llvm_sqrt  = intr1(llvm::Intrinsic::sqrt);
  c.llvm_floor = intr1(llvm::Intrinsic::floor);
  c.llvm_ceil  = intr1(llvm::Intrinsic::ceil);
  c.llvm_round = intr1(llvm::Intrinsic::round);
  c.llvm_fabs  = intr1(llvm::Intrinsic::fabs);
}

TypedVal EmitCtx::resolve_typed(const Operand & op) const
{
  switch (op.kind) {
    case OperandKind::Const:
      switch (op.scalar_type) {
        case ST::Int:   return {builder->getInt64(static_cast<int64_t>(op.const_val)), ST::Int};
        case ST::Bool:  return {builder->getInt1(op.const_val != 0.0), ST::Bool};
        default:        return {llvm::ConstantFP::get(f64_ty, op.const_val), ST::Float};
      }
    case OperandKind::Input:    return {load_input_f64(op.slot), ST::Float};
    case OperandKind::Reg:      return {load_temp_typed(op.slot, (*temp_types)[op.slot]), (*temp_types)[op.slot]};
    case OperandKind::StateReg: return {load_reg_typed(op.slot, op.scalar_type), op.scalar_type};
    case OperandKind::Rate:     return {sample_rate_arg, ST::Float};
    case OperandKind::Tick:     return {current_sample_idx, ST::Int};
    case OperandKind::Slot: {
      llvm::Value * v = load_slot_f64(op.slot);
      switch (op.scalar_type) {
        case ST::Float: return {v, ST::Float};
        case ST::Int:   return {builder->CreateFPToSI(v, i64_ty), ST::Int};
        case ST::Bool:  return {builder->CreateFCmpONE(v, llvm::ConstantFP::get(f64_ty, 0.0)), ST::Bool};
      }
      return {v, ST::Float};
    }
    case OperandKind::ArrayReg:
    case OperandKind::Param:    return {nullptr, ST::Float};
  }
  return {nullptr, ST::Float};
}

TypedVal EmitCtx::emit_typed_op(OpTag tag, ST result_type,
                                 const std::vector<TypedVal> & tv) const
{
  auto arg_as = [&](size_t i, ST target) -> llvm::Value * {
    return coerce_val(tv[i].first, tv[i].second, target);
  };

  switch (tag) {
    case OpTag::Add:
      if (result_type == ST::Int)   return {builder->CreateAdd(arg_as(0, ST::Int), arg_as(1, ST::Int)), ST::Int};
      return {builder->CreateFAdd(arg_as(0, ST::Float), arg_as(1, ST::Float)), ST::Float};
    case OpTag::Sub:
      if (result_type == ST::Int)   return {builder->CreateSub(arg_as(0, ST::Int), arg_as(1, ST::Int)), ST::Int};
      return {builder->CreateFSub(arg_as(0, ST::Float), arg_as(1, ST::Float)), ST::Float};
    case OpTag::Mul:
      if (result_type == ST::Int)   return {builder->CreateMul(arg_as(0, ST::Int), arg_as(1, ST::Int)), ST::Int};
      return {builder->CreateFMul(arg_as(0, ST::Float), arg_as(1, ST::Float)), ST::Float};
    case OpTag::Div: {
      if (result_type == ST::Int) {
        llvm::Value * b = arg_as(1, ST::Int);
        return {builder->CreateSelect(builder->CreateICmpEQ(b, zero_i64), zero_i64, builder->CreateSDiv(arg_as(0, ST::Int), b)), ST::Int};
      }
      llvm::Value * b = arg_as(1, ST::Float);
      return {builder->CreateSelect(builder->CreateFCmpOEQ(b, zero_f64), zero_f64, builder->CreateFDiv(arg_as(0, ST::Float), b)), ST::Float};
    }
    case OpTag::Mod: {
      if (result_type == ST::Int) {
        llvm::Value * b = arg_as(1, ST::Int);
        return {builder->CreateSelect(builder->CreateICmpEQ(b, zero_i64), zero_i64, builder->CreateSRem(arg_as(0, ST::Int), b)), ST::Int};
      }
      llvm::Value * b = arg_as(1, ST::Float);
      llvm::Value * a = arg_as(0, ST::Float);
      llvm::Value * is_zero = builder->CreateFCmpOEQ(b, zero_f64);
      llvm::Value * q = builder->CreateFDiv(a, b);
      llvm::Value * fq = builder->CreateCall(llvm_floor, {q});
      llvm::Value * fmod_result = builder->CreateFSub(a, builder->CreateFMul(fq, b));
      return {builder->CreateSelect(is_zero, zero_f64, fmod_result), ST::Float};
    }
    case OpTag::FloorDiv: {
      if (result_type == ST::Int) {
        llvm::Value * b = arg_as(1, ST::Int);
        return {builder->CreateSelect(builder->CreateICmpEQ(b, zero_i64), zero_i64, builder->CreateSDiv(arg_as(0, ST::Int), b)), ST::Int};
      }
      llvm::Value * b = arg_as(1, ST::Float);
      llvm::Value * dv = builder->CreateSelect(builder->CreateFCmpOEQ(b, zero_f64), zero_f64, builder->CreateFDiv(arg_as(0, ST::Float), b));
      return {builder->CreateCall(llvm_floor, {dv}), ST::Float};
    }
    case OpTag::Less:
      if (tv[0].second == ST::Int || tv[1].second == ST::Int)
        return {builder->CreateICmpSLT(arg_as(0, ST::Int), arg_as(1, ST::Int)), ST::Bool};
      return {builder->CreateFCmpOLT(arg_as(0, ST::Float), arg_as(1, ST::Float)), ST::Bool};
    case OpTag::LessEq:
      if (tv[0].second == ST::Int || tv[1].second == ST::Int)
        return {builder->CreateICmpSLE(arg_as(0, ST::Int), arg_as(1, ST::Int)), ST::Bool};
      return {builder->CreateFCmpOLE(arg_as(0, ST::Float), arg_as(1, ST::Float)), ST::Bool};
    case OpTag::Greater:
      if (tv[0].second == ST::Int || tv[1].second == ST::Int)
        return {builder->CreateICmpSGT(arg_as(0, ST::Int), arg_as(1, ST::Int)), ST::Bool};
      return {builder->CreateFCmpOGT(arg_as(0, ST::Float), arg_as(1, ST::Float)), ST::Bool};
    case OpTag::GreaterEq:
      if (tv[0].second == ST::Int || tv[1].second == ST::Int)
        return {builder->CreateICmpSGE(arg_as(0, ST::Int), arg_as(1, ST::Int)), ST::Bool};
      return {builder->CreateFCmpOGE(arg_as(0, ST::Float), arg_as(1, ST::Float)), ST::Bool};
    case OpTag::Equal:
      if (tv[0].second == ST::Int || tv[1].second == ST::Int)
        return {builder->CreateICmpEQ(arg_as(0, ST::Int), arg_as(1, ST::Int)), ST::Bool};
      return {builder->CreateFCmpOEQ(arg_as(0, ST::Float), arg_as(1, ST::Float)), ST::Bool};
    case OpTag::NotEqual:
      if (tv[0].second == ST::Int || tv[1].second == ST::Int)
        return {builder->CreateICmpNE(arg_as(0, ST::Int), arg_as(1, ST::Int)), ST::Bool};
      return {builder->CreateFCmpUNE(arg_as(0, ST::Float), arg_as(1, ST::Float)), ST::Bool};
    case OpTag::And:  return {builder->CreateAnd(arg_as(0, ST::Bool), arg_as(1, ST::Bool)), ST::Bool};
    case OpTag::Or:   return {builder->CreateOr (arg_as(0, ST::Bool), arg_as(1, ST::Bool)), ST::Bool};
    case OpTag::BitAnd:  return {builder->CreateAnd(arg_as(0, ST::Int), arg_as(1, ST::Int)), ST::Int};
    case OpTag::BitOr:   return {builder->CreateOr(arg_as(0, ST::Int), arg_as(1, ST::Int)), ST::Int};
    case OpTag::BitXor:  return {builder->CreateXor(arg_as(0, ST::Int), arg_as(1, ST::Int)), ST::Int};
    case OpTag::LShift:  return {builder->CreateShl(arg_as(0, ST::Int), arg_as(1, ST::Int)), ST::Int};
    case OpTag::RShift:  return {builder->CreateAShr(arg_as(0, ST::Int), arg_as(1, ST::Int)), ST::Int};
    case OpTag::Neg:
      if (result_type == ST::Int)   return {builder->CreateNeg(arg_as(0, ST::Int)), ST::Int};
      return {builder->CreateFNeg(arg_as(0, ST::Float)), ST::Float};
    case OpTag::Abs: {
      if (result_type == ST::Int) {
        llvm::Value * v = arg_as(0, ST::Int);
        llvm::Value * neg = builder->CreateNeg(v);
        return {builder->CreateSelect(builder->CreateICmpSLT(v, zero_i64), neg, v), ST::Int};
      }
      return {builder->CreateCall(llvm_fabs, {arg_as(0, ST::Float)}), ST::Float};
    }
    case OpTag::Sqrt:   return {builder->CreateCall(llvm_sqrt,  {arg_as(0, ST::Float)}), ST::Float};
    case OpTag::Floor:  return {builder->CreateCall(llvm_floor, {arg_as(0, ST::Float)}), ST::Float};
    case OpTag::Ceil:   return {builder->CreateCall(llvm_ceil,  {arg_as(0, ST::Float)}), ST::Float};
    case OpTag::Round:  return {builder->CreateCall(llvm_round, {arg_as(0, ST::Float)}), ST::Float};
    case OpTag::Ldexp: {
      llvm::Value * x  = arg_as(0, ST::Float);
      llvm::Value * ni = builder->CreateFPToSI(arg_as(1, ST::Float), i64_ty);
      llvm::Value * bias  = builder->CreateShl(builder->CreateAdd(ni, builder->getInt64(1023)), 52);
      llvm::Value * scale = builder->CreateBitCast(bias, f64_ty);
      return {builder->CreateFMul(x, scale), ST::Float};
    }
    case OpTag::FloatExponent: {
      llvm::Value * bits = builder->CreateBitCast(arg_as(0, ST::Float), i64_ty);
      llvm::Value * e    = builder->CreateSub(builder->CreateAShr(bits, 52), builder->getInt64(1023));
      return {builder->CreateSIToFP(e, f64_ty), ST::Float};
    }
    case OpTag::Not: {
      if (tv[0].second == ST::Bool)  return {builder->CreateNot(tv[0].first), ST::Bool};
      if (tv[0].second == ST::Int)   return {builder->CreateICmpEQ(tv[0].first, zero_i64), ST::Bool};
      return {builder->CreateFCmpOEQ(tv[0].first, zero_f64), ST::Bool};
    }
    case OpTag::BitNot:
      return {builder->CreateNot(arg_as(0, ST::Int)), ST::Int};
    case OpTag::ToInt:
      return {coerce_val(tv[0].first, tv[0].second, ST::Int), ST::Int};
    case OpTag::ToBool:
      return {coerce_val(tv[0].first, tv[0].second, ST::Bool), ST::Bool};
    case OpTag::ToFloat:
      return {coerce_val(tv[0].first, tv[0].second, ST::Float), ST::Float};
    case OpTag::Clamp: {
      if (result_type == ST::Int) {
        llvm::Value * val = arg_as(0, ST::Int);
        llvm::Value * lo = arg_as(1, ST::Int);
        llvm::Value * hi = arg_as(2, ST::Int);
        llvm::Value * lo_c = builder->CreateSelect(builder->CreateICmpSGT(val, lo), val, lo);
        return {builder->CreateSelect(builder->CreateICmpSLT(lo_c, hi), lo_c, hi), ST::Int};
      }
      llvm::Value * val = arg_as(0, ST::Float);
      llvm::Value * lo = arg_as(1, ST::Float);
      llvm::Value * hi = arg_as(2, ST::Float);
      llvm::Value * lo_c = builder->CreateSelect(builder->CreateFCmpOGT(val, lo), val, lo);
      return {builder->CreateSelect(builder->CreateFCmpOLT(lo_c, hi), lo_c, hi), ST::Float};
    }
    case OpTag::Select: {
      llvm::Value * cond_val;
      if (tv[0].second == ST::Bool)
        cond_val = tv[0].first;
      else if (tv[0].second == ST::Int)
        cond_val = builder->CreateICmpNE(tv[0].first, zero_i64);
      else
        cond_val = builder->CreateFCmpUNE(tv[0].first, zero_f64);
      return {builder->CreateSelect(cond_val, arg_as(1, result_type), arg_as(2, result_type)), result_type};
    }
    default:
      return {nullptr, ST::Float};
  }
}

void EmitCtx::emit_writebacks(const std::vector<InstanceProgram::Writeback> & wbs)
{
  for (const auto & wb : wbs) {
    if (wb.temp_slot < 0) continue;
    const uint32_t ti = static_cast<uint32_t>(wb.temp_slot);
    const ST src_ty = (*temp_types)[ti];
    const ST dst_ty = (wb.state_slot < program->register_types.size())
      ? program->register_types[wb.state_slot] : ST::Float;
    llvm::Value * typed_val = load_temp_typed(ti, src_ty);
    llvm::Value * coerced   = coerce_val(typed_val, src_ty, dst_ty);
    llvm::Value * as_i64 = nullptr;
    if (dst_ty == ST::Float)
      as_i64 = builder->CreateBitCast(coerced, i64_ty);
    else if (dst_ty == ST::Int)
      as_i64 = builder->CreateSExtOrBitCast(coerced, i64_ty);
    else
      as_i64 = builder->CreateZExt(coerced, i64_ty);
    llvm::Value * reg_ptr = builder->CreateInBoundsGEP(i64_ty, regs_arg, builder->getInt64(wb.state_slot));
    builder->CreateStore(as_i64, reg_ptr);
  }
}

llvm::Error EmitCtx::emit_instr(const FlatInstr & instr)
{
  if (instr.tag == OpTag::SmoothParam) {
    const uint64_t param_ptr = instr.args[0].ptr;
    const uint32_t state_slot = instr.args[1].slot;
    const double coeff = instr.args[2].const_val;

    llvm::Value * reg_gep = builder->CreateInBoundsGEP(i64_ty, regs_arg, builder->getInt64(state_slot));
    llvm::Value * current  = builder->CreateBitCast(builder->CreateLoad(i64_ty, reg_gep), f64_ty);

    auto pi = param_index->find(param_ptr);
    uint64_t canonical_idx = (pi != param_index->end()) ? pi->second : 0;
    llvm::Value * pp_slot = builder->CreateInBoundsGEP(i64_ty, param_ptrs_arg, builder->getInt64(canonical_idx));
    llvm::Value * pp_raw  = builder->CreateLoad(i64_ty, pp_slot);
    llvm::Value * pp_addr = builder->CreateIntToPtr(pp_raw, ptr_ty);
    auto * target_ld = builder->CreateAlignedLoad(f64_ty, pp_addr, llvm::Align(sizeof(double)));
    target_ld->setAtomic(llvm::AtomicOrdering::Monotonic);

    llvm::Value * diff    = builder->CreateFSub(target_ld, current);
    llvm::Value * new_val = builder->CreateFAdd(current, builder->CreateFMul(llvm::ConstantFP::get(f64_ty, coeff), diff));
    builder->CreateStore(builder->CreateBitCast(new_val, i64_ty), reg_gep);
    store_temp_f64(instr.dst, new_val);
    (*temp_types)[instr.dst] = ST::Float;
    return llvm::Error::success();
  }
  if (instr.tag == OpTag::WriteSlot) {
    auto [v, t] = resolve_typed(instr.args[0]);
    llvm::Value * v_f64 = coerce_val(v, t, ST::Float);
    store_slot_f64(instr.dst, v_f64);
    return llvm::Error::success();
  }
  if (instr.tag == OpTag::Pack) {
    llvm::Value * dst_ptr = load_array_ptr_f(instr.dst);
    for (std::size_t i = 0; i < instr.args.size(); ++i) {
      llvm::Value * val = resolve_as_f64(instr.args[i]);
      llvm::Value * ep  = builder->CreateInBoundsGEP(i64_ty, dst_ptr, builder->getInt64(static_cast<int64_t>(i)));
      builder->CreateStore(builder->CreateBitCast(val, i64_ty), ep);
    }
    return llvm::Error::success();
  }
  if (instr.tag == OpTag::Index) {
    const uint32_t arr_slot = instr.args[0].slot;
    llvm::Value * array_size = load_array_size_f(arr_slot);
    llvm::Value * array_ptr  = load_array_ptr_f(arr_slot);
    auto [idx_v, idx_t] = resolve_typed(instr.args[1]);
    llvm::Value * raw_idx = coerce_val(idx_v, idx_t, ST::Int);
    llvm::Value * in_range = builder->CreateAnd(
      builder->CreateNot(builder->CreateICmpSLT(raw_idx, builder->getInt64(0))),
      builder->CreateICmpULT(raw_idx, array_size));
    llvm::Value * ep  = builder->CreateInBoundsGEP(i64_ty, array_ptr, raw_idx);
    llvm::Value * val = builder->CreateBitCast(builder->CreateLoad(i64_ty, ep), f64_ty);
    store_temp_f64(instr.dst, builder->CreateSelect(in_range, val, zero_f64));
    (*temp_types)[instr.dst] = ST::Float;
    return llvm::Error::success();
  }
  if (instr.tag == OpTag::SetElement) {
    const uint32_t arr_slot = instr.args[0].slot;
    llvm::Value * array_size = load_array_size_f(arr_slot);
    llvm::Value * array_ptr  = load_array_ptr_f(arr_slot);
    auto [idx_v, idx_t] = resolve_typed(instr.args[1]);
    llvm::Value * raw_idx = coerce_val(idx_v, idx_t, ST::Int);
    llvm::Value * in_range = builder->CreateAnd(
      builder->CreateNot(builder->CreateICmpSLT(raw_idx, builder->getInt64(0))),
      builder->CreateICmpULT(raw_idx, array_size));
    llvm::BasicBlock * write_bb = llvm::BasicBlock::Create(*context, "set_elem_write", fn);
    llvm::BasicBlock * merge_bb = llvm::BasicBlock::Create(*context, "set_elem_merge", fn);
    builder->CreateCondBr(in_range, write_bb, merge_bb);
    builder->SetInsertPoint(write_bb);
    llvm::Value * ep = builder->CreateInBoundsGEP(i64_ty, array_ptr, raw_idx);
    builder->CreateStore(builder->CreateBitCast(resolve_as_f64(instr.args[2]), i64_ty), ep);
    builder->CreateBr(merge_bb);
    builder->SetInsertPoint(merge_bb);
    return llvm::Error::success();
  }
  // Array dst: emit elementwise. Dispatch is on `dst_kind`, not on
  // `loop_count > 1` — the latter was an unreliable proxy that silently
  // misclassified degenerate `loop_count == 1` writes to array slots
  // as scalar emissions, storing a scalar result via `store_temp_f64`
  // at an array-slot index (out-of-bounds into the temp buffer).
  // `dst_kind` is the honest discriminator carried through the wire
  // format from the TS `DstSlot` union.
  if (instr.dst_kind == DstKind::Array) {
    const std::size_t nargs = instr.args.size();
    std::vector<llvm::Value *> arr_ptrs(nargs, nullptr);
    std::vector<TypedVal> scalar_tvs(nargs, {nullptr, ST::Float});
    for (std::size_t i = 0; i < nargs; ++i) {
      if (i < instr.strides.size() && instr.strides[i] == 1)
        arr_ptrs[i] = load_array_ptr_f(instr.args[i].slot);
      else
        scalar_tvs[i] = resolve_typed(instr.args[i]);
    }

    auto is_simd_float_op = [](OpTag t) {
      return t == OpTag::Add || t == OpTag::Sub
          || t == OpTag::Mul || t == OpTag::Div
          || t == OpTag::Neg || t == OpTag::Abs
          || t == OpTag::Sqrt;
    };
    // SIMD path is an OPTIMIZATION trigger — only worth it for N > 1.
    // For N == 1 (degenerate single-element array write) fall straight
    // to the scalar-loop body below; LLVM emits one iteration's
    // worth of GEP + load + op + store, which the optimizer collapses
    // to the right two-instruction store.
    bool simd_ok = (instr.result_type == ST::Float)
                && is_simd_float_op(instr.tag)
                && instr.loop_count > 1
                && (instr.strides.size() == nargs);
    for (std::size_t i = 0; simd_ok && i < nargs; ++i) {
      const uint8_t s = instr.strides[i];
      if (s != 0 && s != 1) simd_ok = false;
    }
    if (simd_ok) {
      const uint32_t N = instr.loop_count;
      llvm::Type * vec_ty = llvm::FixedVectorType::get(f64_ty, N);
      llvm::Value * dst_ptr_v = load_array_ptr_f(instr.dst);

      std::vector<llvm::Value *> vops(nargs, nullptr);
      for (std::size_t i = 0; i < nargs; ++i) {
        if (instr.strides[i] == 1) {
          vops[i] = builder->CreateAlignedLoad(vec_ty, arr_ptrs[i], llvm::Align(sizeof(double)));
        } else {
          llvm::Value * sv = coerce_val(scalar_tvs[i].first, scalar_tvs[i].second, ST::Float);
          vops[i] = builder->CreateVectorSplat(N, sv);
        }
      }

      llvm::Value * vres = nullptr;
      switch (instr.tag) {
        case OpTag::Add: vres = builder->CreateFAdd(vops[0], vops[1]); break;
        case OpTag::Sub: vres = builder->CreateFSub(vops[0], vops[1]); break;
        case OpTag::Mul: vres = builder->CreateFMul(vops[0], vops[1]); break;
        case OpTag::Div: {
          llvm::Value * zv = llvm::ConstantFP::get(vec_ty, 0.0);
          llvm::Value * is_zero = builder->CreateFCmpOEQ(vops[1], zv);
          llvm::Value * dv = builder->CreateFDiv(vops[0], vops[1]);
          vres = builder->CreateSelect(is_zero, zv, dv);
          break;
        }
        case OpTag::Neg:  vres = builder->CreateFNeg(vops[0]); break;
        case OpTag::Abs: {
          llvm::FunctionCallee fabs_v = llvm::Intrinsic::getOrInsertDeclaration(
            module, llvm::Intrinsic::fabs, {vec_ty});
          vres = builder->CreateCall(fabs_v, {vops[0]});
          break;
        }
        case OpTag::Sqrt: {
          llvm::FunctionCallee sqrt_v = llvm::Intrinsic::getOrInsertDeclaration(
            module, llvm::Intrinsic::sqrt, {vec_ty});
          vres = builder->CreateCall(sqrt_v, {vops[0]});
          break;
        }
        default: break;
      }

      if (vres) {
        builder->CreateAlignedStore(vres, dst_ptr_v, llvm::Align(sizeof(double)));
        return llvm::Error::success();
      }
    }

    llvm::Value * loop_n    = builder->getInt64(instr.loop_count);
    llvm::Value * dst_ptr   = load_array_ptr_f(instr.dst);
    llvm::Value * idx_alloc = builder->CreateAlloca(i64_ty, nullptr, "ew_idx");
    builder->CreateStore(builder->getInt64(0), idx_alloc);

    llvm::BasicBlock * cond_bb = llvm::BasicBlock::Create(*context, "ew_cond", fn);
    llvm::BasicBlock * body_bb = llvm::BasicBlock::Create(*context, "ew_body", fn);
    llvm::BasicBlock * end_bb  = llvm::BasicBlock::Create(*context, "ew_end",  fn);
    builder->CreateBr(cond_bb);

    builder->SetInsertPoint(cond_bb);
    llvm::Value * idx = builder->CreateLoad(i64_ty, idx_alloc);
    builder->CreateCondBr(builder->CreateICmpULT(idx, loop_n), body_bb, end_bb);

    builder->SetInsertPoint(body_bb);
    std::vector<TypedVal> iter_tvs(nargs, {nullptr, ST::Float});
    for (std::size_t i = 0; i < nargs; ++i) {
      if (arr_ptrs[i]) {
        llvm::Value * ep = builder->CreateInBoundsGEP(i64_ty, arr_ptrs[i], idx);
        iter_tvs[i] = {builder->CreateBitCast(builder->CreateLoad(i64_ty, ep), f64_ty), ST::Float};
      } else {
        iter_tvs[i] = scalar_tvs[i];
      }
    }
    auto [elem_result, elem_actual_ty] = emit_typed_op(instr.tag, instr.result_type, iter_tvs);
    if (!elem_result) {
      return llvm::make_error<llvm::StringError>(
        "emit_instr: unsupported OpTag in elementwise loop",
        llvm::inconvertibleErrorCode());
    }
    llvm::Value * as_f64 = coerce_val(elem_result, elem_actual_ty, ST::Float);
    llvm::Value * dep = builder->CreateInBoundsGEP(i64_ty, dst_ptr, idx);
    builder->CreateStore(builder->CreateBitCast(as_f64, i64_ty), dep);
    builder->CreateStore(builder->CreateAdd(idx, builder->getInt64(1)), idx_alloc);
    builder->CreateBr(cond_bb);

    builder->SetInsertPoint(end_bb);
    return llvm::Error::success();
  }

  // Fallthrough: scalar emission writes to a temp slot. Assert the
  // dst_kind matches — the tag-specific handlers above (WriteSlot,
  // Pack, Index, SetElement, SmoothParam) catch all non-Temp tags,
  // and the array-dst branch above catches all elementwise array
  // writes. Reaching here with a non-Temp dst is a contract
  // violation from the producer.
  if (instr.dst_kind != DstKind::Temp) {
    return llvm::make_error<llvm::StringError>(
      "emit_instr: scalar fallthrough reached with non-Temp dst_kind — "
      "producer emitted a scalar op pointing at an Array/ModuleSlot dst",
      llvm::inconvertibleErrorCode());
  }

  std::vector<TypedVal> tvs;
  tvs.reserve(instr.args.size());
  for (const auto & arg : instr.args)
    tvs.push_back(resolve_typed(arg));

  auto [result, actual_ty] = emit_typed_op(instr.tag, instr.result_type, tvs);
  if (!result) {
    return llvm::make_error<llvm::StringError>(
      "emit_instr: unsupported scalar OpTag",
      llvm::inconvertibleErrorCode());
  }
  llvm::Value * coerced = coerce_val(result, actual_ty, instr.result_type);
  store_temp_typed(instr.dst, coerced, instr.result_type);
  (*temp_types)[instr.dst] = instr.result_type;
  return llvm::Error::success();
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// compile_flat_program — emission path for tropical_plan_4 / FlatProgram.
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

  // Helper: visit every FlatInstr in the plan (preamble + each
  // instance + postamble) in a fixed order. Used for param index
  // collection and cache-key serialization so the canonical traversal
  // matches between the two passes.
  auto visit_all_instructions = [&](auto && fn) {
    for (const auto & instr : program.scheduler.preamble) fn(instr);
    for (const auto & inst : program.instance_functions)
      for (const auto & instr : inst.instructions) fn(instr);
    for (const auto & instr : program.scheduler.state_evolution) fn(instr);
    for (const auto & instr : program.scheduler.postamble) fn(instr);
  };

  // Build canonical param_ptr → ordinal map (order of first appearance).
  std::unordered_map<uint64_t, uint64_t> param_index;
  visit_all_instructions([&](const FlatInstr & instr) {
    for (const auto & arg : instr.args)
    {
      if (arg.kind == OperandKind::Param && arg.ptr != 0 &&
          param_index.find(arg.ptr) == param_index.end())
      {
        param_index.emplace(arg.ptr, static_cast<uint64_t>(param_index.size()));
      }
    }
  });

  // Serialize to a canonical cache key. Opt level is part of the key
  // so plans cached at one level aren't served at another.
  std::string cache_key;
  cache_key += "flat5:";
  cache_key += opt_level_tag(opt_level_);
  cache_key += ":";
  {
    auto append = [&](const void * data, std::size_t size) {
      cache_key.append(static_cast<const char *>(data), size);
    };
    auto serialize_instr = [&](const FlatInstr & instr) {
      append(&instr.tag,        sizeof(OpTag));
      append(&instr.result_type, sizeof(JitScalarType));
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
    };

    // Serialize one instance function INCLUDING its preamble,
    // pre_input, writebacks, and nested children. Recursing into
    // children is essential: under the root-program (Option A)
    // lowering every plan's single top-level function has an empty
    // body and the same `instance_name` ("instance___root__"), so the
    // whole plan's identity lives in the child subtree. Hashing only
    // the top-level `instructions` (as before) made all such plans
    // collide and serve each other's cached kernels. Leaf/flat plans
    // are unaffected — their bodies were already distinct.
    std::function<void(const InstanceProgram &)> serialize_fn =
      [&](const InstanceProgram & inst) {
        append(&inst.register_count, sizeof(uint32_t));
        uint32_t nameLen = static_cast<uint32_t>(inst.instance_name.size());
        append(&nameLen, sizeof(uint32_t));
        if (!inst.instance_name.empty())
          append(inst.instance_name.data(), inst.instance_name.size());
        auto serialize_block = [&](const std::vector<FlatInstr> & block) {
          uint32_t n = static_cast<uint32_t>(block.size());
          append(&n, sizeof(uint32_t));
          for (const auto & instr : block) serialize_instr(instr);
        };
        serialize_block(inst.preamble_instructions);
        serialize_block(inst.pre_input_instructions);
        serialize_block(inst.instructions);
        uint32_t nwb = static_cast<uint32_t>(inst.writebacks.size());
        append(&nwb, sizeof(uint32_t));
        for (const auto & wb : inst.writebacks)
        {
          append(&wb.state_slot, sizeof(uint32_t));
          append(&wb.temp_slot,  sizeof(int32_t));
        }
        uint32_t nch = static_cast<uint32_t>(inst.children.size());
        append(&nch, sizeof(uint32_t));
        for (const auto & child : inst.children) serialize_fn(child);
      };

    append(&program.register_count, sizeof(uint32_t));
    // Scheduler: preamble + state_evolution + postamble
    uint32_t npre = static_cast<uint32_t>(program.scheduler.preamble.size());
    append(&npre, sizeof(uint32_t));
    for (const auto & instr : program.scheduler.preamble) serialize_instr(instr);
    uint32_t nstate = static_cast<uint32_t>(program.scheduler.state_evolution.size());
    append(&nstate, sizeof(uint32_t));
    for (const auto & instr : program.scheduler.state_evolution) serialize_instr(instr);
    uint32_t npost = static_cast<uint32_t>(program.scheduler.postamble.size());
    append(&npost, sizeof(uint32_t));
    for (const auto & instr : program.scheduler.postamble) serialize_instr(instr);
    // Instance functions (recursively, including children)
    uint32_t nfns = static_cast<uint32_t>(program.instance_functions.size());
    append(&nfns, sizeof(uint32_t));
    for (const auto & inst : program.instance_functions) serialize_fn(inst);
    // Mix outputs (legacy temp-mix path)
    uint32_t nm = static_cast<uint32_t>(program.mix_output_temps.size());
    append(&nm, sizeof(uint32_t));
    for (uint32_t mt : program.mix_output_temps)
      append(&mt, sizeof(uint32_t));
    // Sinks (slot-mix path): inputs + gain + target all affect codegen.
    uint32_t nsink = static_cast<uint32_t>(program.sinks.size());
    append(&nsink, sizeof(uint32_t));
    for (const auto & sink : program.sinks) {
      uint32_t ni = static_cast<uint32_t>(sink.inputs.size());
      append(&ni, sizeof(uint32_t));
      for (uint32_t s : sink.inputs) append(&s, sizeof(uint32_t));
      append(&sink.gain, sizeof(double));
      append(&sink.target, sizeof(uint32_t));
    }
    // Register writebacks
    uint32_t nrt = static_cast<uint32_t>(program.register_targets.size());
    append(&nrt, sizeof(uint32_t));
    for (int32_t rt : program.register_targets)
      append(&rt, sizeof(int32_t));
    uint32_t nrty = static_cast<uint32_t>(program.register_types.size());
    append(&nrty, sizeof(uint32_t));
    for (JitScalarType t : program.register_types)
      append(&t, sizeof(JitScalarType));
  }

  auto cache_it = kernel_cache_.find(cache_key);
  if (cache_it != kernel_cache_.end())
    return cache_it->second;

  const std::string hash = md5_hex(cache_key);
  const std::string function_name = "tropical_f_" + hash;

  auto context = std::make_unique<llvm::LLVMContext>();
  auto module = std::make_unique<llvm::Module>(hash, *context);
  module->setDataLayout(jit_->getDataLayout());

  llvm::IRBuilder<> builder(*context);
  llvm::Type * void_ty  = builder.getVoidTy();
  llvm::Type * f64_ty   = builder.getDoubleTy();
  llvm::Type * i64_ty   = builder.getInt64Ty();
  llvm::Type * i1_ty    = builder.getInt1Ty();
  llvm::Type * ptr_ty   = llvm::PointerType::get(*context, 0);

  // Kernel signature (M6+): adds `slots` (double *) as the trailing arg.
  // The IRTransformLayer's nocapture-inference handles aliasing
  // analysis on the existing pointer args; we additionally tag
  // `slots` explicitly with `nocapture` and `noalias` below so GVN /
  // store-to-load forwarding engages on slot reads (per spike #3 —
  // verified that the JIT's IRTransformLayer pipeline includes GVN).
  llvm::FunctionType * fn_ty = llvm::FunctionType::get(
    void_ty,
    {ptr_ty, ptr_ty, ptr_ty, ptr_ty, ptr_ty, f64_ty, i64_ty, ptr_ty, ptr_ty, i64_ty, ptr_ty},
    false);

  llvm::Function * fn = llvm::Function::Create(
    fn_ty, llvm::Function::ExternalLinkage, function_name, module.get());

  auto arg_it = fn->arg_begin();
  llvm::Value * inputs_arg      = &*arg_it++;  inputs_arg->setName("inputs");
  llvm::Value * regs_arg        = &*arg_it++;  regs_arg->setName("registers");
  llvm::Value * arrays_arg      = &*arg_it++;  arrays_arg->setName("arrays");
  llvm::Value * array_sizes_arg = &*arg_it++;  array_sizes_arg->setName("array_sizes");
  llvm::Value * temps_arg       = &*arg_it++;  temps_arg->setName("temps");
  llvm::Value * sample_rate_arg      = &*arg_it++;  sample_rate_arg->setName("sampleRate");
  llvm::Value * start_sample_idx_arg = &*arg_it++;  start_sample_idx_arg->setName("start_sample_index");
  llvm::Value * param_ptrs_arg       = &*arg_it++;  param_ptrs_arg->setName("param_ptrs");
  llvm::Value * output_buffer_arg    = &*arg_it++;  output_buffer_arg->setName("output_buffer");
  llvm::Value * buffer_length_arg    = &*arg_it++;  buffer_length_arg->setName("buffer_length");
  llvm::Argument * slots_arg_a       = &*arg_it++;  slots_arg_a->setName("slots");
  llvm::Value * slots_arg            = slots_arg_a;
  // M6: tag slots explicitly so GVN can prove no-alias and forward
  // store-to-load. Without these, even with IRTransformLayer running
  // GVN, slot loads may not be eliminated (spike #3 finding).
  //
  // LLVM 21 replaced the boolean `NoCapture` attribute with the
  // granular `Captures` attribute carrying a `CaptureInfo`. The
  // semantic equivalent of the old NoCapture is
  // `CaptureInfo::none()` — pointer is not captured at all.
#if LLVM_VERSION_MAJOR >= 21
  slots_arg_a->addAttr(llvm::Attribute::getWithCaptureInfo(*context, llvm::CaptureInfo::none()));
#else
  slots_arg_a->addAttr(llvm::Attribute::NoCapture);
#endif
  slots_arg_a->addAttr(llvm::Attribute::NoAlias);

  llvm::BasicBlock * entry = llvm::BasicBlock::Create(*context, "entry", fn);
  builder.SetInsertPoint(entry);

  // ── EmitCtx setup ──
  // The slot/typed/coerce/resolve/emit_* helpers live as methods on EmitCtx
  // so both fused-mode (here) and microkernel-mode (compile_microkernel)
  // share one codegen path. EmitCtx holds the per-compilation builder +
  // types + intrinsics + per-temp type table; argument values are bound
  // just below from this function's signature.
  EmitCtx ctx;
  emit_ctx_init_module(ctx, *context, builder, *module);
  ctx.fn              = fn;
  ctx.inputs_arg      = inputs_arg;
  ctx.regs_arg        = regs_arg;
  ctx.arrays_arg      = arrays_arg;
  ctx.array_sizes_arg = array_sizes_arg;
  ctx.temps_arg       = temps_arg;
  ctx.sample_rate_arg = sample_rate_arg;
  ctx.param_ptrs_arg  = param_ptrs_arg;
  ctx.slots_arg       = slots_arg;
  ctx.program         = &program;
  ctx.param_index     = &param_index;
  // Per-temp register type tracking (updated per instruction). Lives on
  // the stack so its lifetime spans the whole compile; ctx points to it.
  std::vector<ST> temp_types(program.register_count, ST::Float);
  ctx.temp_types      = &temp_types;

  // ── Buffer loop: process buffer_length samples ──
  llvm::BasicBlock * loop_cond_bb = llvm::BasicBlock::Create(*context, "loop_cond", fn);
  llvm::BasicBlock * loop_body_bb = llvm::BasicBlock::Create(*context, "loop_body", fn);
  llvm::BasicBlock * loop_end_bb  = llvm::BasicBlock::Create(*context, "loop_end",  fn);

  builder.CreateBr(loop_cond_bb);

  // Loop condition: %s < buffer_length
  builder.SetInsertPoint(loop_cond_bb);
  llvm::PHINode * loop_counter = builder.CreatePHI(i64_ty, 2, "s");
  loop_counter->addIncoming(builder.getInt64(0), entry);
  ctx.current_sample_idx = builder.CreateAdd(start_sample_idx_arg, loop_counter, "current_idx");
  llvm::Value * loop_cond_val = builder.CreateICmpULT(loop_counter, buffer_length_arg);
  builder.CreateCondBr(loop_cond_val, loop_body_bb, loop_end_bb);

  // Loop body: scheduler.preamble + per-instance dispatch + scheduler.postamble
  //            + writeback (per-instance, inside conditional) + output mixing
  builder.SetInsertPoint(loop_body_bb);

  // ── Scheduler preamble: per-sample setup, runs before any
  //    instance bodies. ──
  if (auto err = ctx.emit_instrs(program.scheduler.preamble)) return std::move(err);

  // ── Per-instance dispatch ──
  // For each top-level instance, EmitCtx::emit_kernel_block recursively
  // emits its nested children inside the parent's body, then the
  // parent's own instructions, then writebacks. Children run BEFORE
  // the parent so it can read their freshly-written slot values.

  for (const auto & inst : program.instance_functions)
  {
    if (auto err = ctx.emit_kernel_block(inst)) return std::move(err);
  }

  // ── Scheduler state-evolution: delay-slot updates from MCP wire
  //    auto-delays. Runs AFTER all instance dispatches and BEFORE
  //    the observation postamble. Writes here become visible to the
  //    NEXT sample's instance kernels (which read these slots at the
  //    start of their bodies and therefore see the previous-sample
  //    WriteSlot — one sample of latency per wire, by construction). ──
  if (auto err = ctx.emit_instrs(program.scheduler.state_evolution)) return std::move(err);

  // ── Scheduler postamble: DAC stitch reads. Runs AFTER state
  //    evolution, so slot reads observe the current sample's
  //    WriteSlot values from instance kernels (asleep instances
  //    retain their previous-sample slot value). ──
  if (auto err = ctx.emit_instrs(program.scheduler.postamble)) return std::move(err);

  // ── Output sinks: each sums its input slots, scales by its gain, and
  //    writes its target channel. v1 realizes target 0 → output_buffer[s].
  //    Falls back to the legacy temp-mix for plan_4 (no sinks). ──
  if (!program.sinks.empty())
  {
    for (const auto & sink : program.sinks)
    {
      llvm::Value * mixed = ctx.zero_f64;
      for (uint32_t slot : sink.inputs)
        mixed = builder.CreateFAdd(mixed, ctx.load_slot_f64(slot));
      llvm::Value * scaled = builder.CreateFMul(mixed, llvm::ConstantFP::get(f64_ty, sink.gain));
      // target 0 → the single output buffer (multi-device deferred).
      llvm::Value * out_ptr = builder.CreateInBoundsGEP(f64_ty, output_buffer_arg, loop_counter);
      builder.CreateStore(scaled, out_ptr);
    }
  }
  else
  {
    llvm::Value * mixed = ctx.zero_f64;
    for (uint32_t mt : program.mix_output_temps)
    {
      llvm::Value * val = ctx.load_temp_typed(mt, temp_types[mt]);
      llvm::Value * f64_val = ctx.coerce_val(val, temp_types[mt], ST::Float);
      mixed = builder.CreateFAdd(mixed, f64_val);
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

  if (std::getenv("TROPICAL_DUMP_IR"))
  {
    module->print(llvm::errs(), nullptr);
  }

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

// ---------------------------------------------------------------------------
// compile_microkernel — emission path for the microkernel execution model.
//
// Emits N+3 LLVM functions in a single module:
//   - mk_<hash>_preamble        : scheduler.preamble  (per-sample)
//   - mk_<hash>_inst_<i>        : one per instance_functions[i]
//   - mk_<hash>_state_evo       : scheduler.state_evolution
//   - mk_<hash>_postamble_mix   : scheduler.postamble + output-mix write
//
// preamble, per-instance kernels, and state_evolution share a slim
// PerSampleFn signature (sample_index in, no buffer length). postamble_mix
// adds output_buffer + output_index so the mix write stays inside the
// optimizer's view.
//
// All functions share the unified register / temp / slot / array buffers
// through pointer args; the C++ scheduler in FlatRuntime walks the buffer
// sample-by-sample and calls each function in topo order per sample.
// ---------------------------------------------------------------------------

llvm::Expected<MicrokernelKernels> OrcJitEngine::compile_microkernel(
  const FlatProgram & program,
  bool deep)
{
  if (!jit_)
  {
    return llvm::make_error<llvm::StringError>(
      "ORC JIT is not available: " + init_error_,
      llvm::inconvertibleErrorCode());
  }

  std::lock_guard<std::mutex> lock(jit_mutex_);

  // ── Param index + cache-key serialization ──
  // Duplicated from compile_flat_program with the only diff being the
  // mode tag prefix on the cache key. A future refactor could share
  // the serialization across both modes.
  auto visit_all_instructions = [&](auto && fn) {
    for (const auto & instr : program.scheduler.preamble) fn(instr);
    for (const auto & inst : program.instance_functions)
      for (const auto & instr : inst.instructions) fn(instr);
    for (const auto & instr : program.scheduler.state_evolution) fn(instr);
    for (const auto & instr : program.scheduler.postamble) fn(instr);
  };

  std::unordered_map<uint64_t, uint64_t> param_index;
  visit_all_instructions([&](const FlatInstr & instr) {
    for (const auto & arg : instr.args) {
      if (arg.kind == OperandKind::Param && arg.ptr != 0 &&
          param_index.find(arg.ptr) == param_index.end()) {
        param_index.emplace(arg.ptr, static_cast<uint64_t>(param_index.size()));
      }
    }
  });

  std::string cache_key;
  // Mode-tagged: keeps microkernel cache disjoint from fused mode's
  // "flat5:fused:" entries, and keeps shallow ("flat5:mk:") and deep
  // ("flat5:mkd:") modes disjoint from each other — both compile_*
  // calls return different LLVM module shapes for the same input
  // FlatProgram, so a wrong-mode hit would be a type-confusion bug.
  cache_key += deep ? "flat5:mkd:" : "flat5:mk:";
  cache_key += opt_level_tag(opt_level_);
  cache_key += ":";
  {
    auto append = [&](const void * data, std::size_t size) {
      cache_key.append(static_cast<const char *>(data), size);
    };
    auto serialize_instr = [&](const FlatInstr & instr) {
      append(&instr.tag,        sizeof(OpTag));
      append(&instr.result_type, sizeof(JitScalarType));
      append(&instr.dst,        sizeof(uint32_t));
      append(&instr.loop_count, sizeof(uint32_t));
      uint32_t na = static_cast<uint32_t>(instr.args.size());
      append(&na, sizeof(uint32_t));
      for (const auto & arg : instr.args) {
        append(&arg.kind,      sizeof(OperandKind));
        append(&arg.const_val, sizeof(double));
        append(&arg.slot,      sizeof(uint32_t));
        uint64_t canonical_ptr = 0;
        if (arg.kind == OperandKind::Param && arg.ptr != 0) {
          auto it = param_index.find(arg.ptr);
          if (it != param_index.end()) canonical_ptr = it->second;
        }
        append(&canonical_ptr, sizeof(uint64_t));
      }
      uint32_t ns = static_cast<uint32_t>(instr.strides.size());
      append(&ns, sizeof(uint32_t));
      if (!instr.strides.empty())
        append(instr.strides.data(), instr.strides.size());
    };

    // Recurse into preamble/pre_input/writebacks/children — same
    // completeness requirement as the fused cache key (a root-program
    // plan's identity lives entirely in its child subtree).
    std::function<void(const InstanceProgram &)> serialize_fn =
      [&](const InstanceProgram & inst) {
        append(&inst.register_count, sizeof(uint32_t));
        uint32_t nameLen = static_cast<uint32_t>(inst.instance_name.size());
        append(&nameLen, sizeof(uint32_t));
        if (!inst.instance_name.empty())
          append(inst.instance_name.data(), inst.instance_name.size());
        auto serialize_block = [&](const std::vector<FlatInstr> & block) {
          uint32_t n = static_cast<uint32_t>(block.size());
          append(&n, sizeof(uint32_t));
          for (const auto & instr : block) serialize_instr(instr);
        };
        serialize_block(inst.preamble_instructions);
        serialize_block(inst.pre_input_instructions);
        serialize_block(inst.instructions);
        uint32_t nwb = static_cast<uint32_t>(inst.writebacks.size());
        append(&nwb, sizeof(uint32_t));
        for (const auto & wb : inst.writebacks) {
          append(&wb.state_slot, sizeof(uint32_t));
          append(&wb.temp_slot,  sizeof(int32_t));
        }
        uint32_t nch = static_cast<uint32_t>(inst.children.size());
        append(&nch, sizeof(uint32_t));
        for (const auto & child : inst.children) serialize_fn(child);
      };

    append(&program.register_count, sizeof(uint32_t));
    uint32_t npre = static_cast<uint32_t>(program.scheduler.preamble.size());
    append(&npre, sizeof(uint32_t));
    for (const auto & instr : program.scheduler.preamble) serialize_instr(instr);
    uint32_t nstate = static_cast<uint32_t>(program.scheduler.state_evolution.size());
    append(&nstate, sizeof(uint32_t));
    for (const auto & instr : program.scheduler.state_evolution) serialize_instr(instr);
    uint32_t npost = static_cast<uint32_t>(program.scheduler.postamble.size());
    append(&npost, sizeof(uint32_t));
    for (const auto & instr : program.scheduler.postamble) serialize_instr(instr);
    uint32_t nfns = static_cast<uint32_t>(program.instance_functions.size());
    append(&nfns, sizeof(uint32_t));
    for (const auto & inst : program.instance_functions) serialize_fn(inst);
    uint32_t nm = static_cast<uint32_t>(program.mix_output_temps.size());
    append(&nm, sizeof(uint32_t));
    for (uint32_t mt : program.mix_output_temps)
      append(&mt, sizeof(uint32_t));
    uint32_t nsink = static_cast<uint32_t>(program.sinks.size());
    append(&nsink, sizeof(uint32_t));
    for (const auto & sink : program.sinks) {
      uint32_t ni = static_cast<uint32_t>(sink.inputs.size());
      append(&ni, sizeof(uint32_t));
      for (uint32_t s : sink.inputs) append(&s, sizeof(uint32_t));
      append(&sink.gain, sizeof(double));
      append(&sink.target, sizeof(uint32_t));
    }
    uint32_t nrt = static_cast<uint32_t>(program.register_targets.size());
    append(&nrt, sizeof(uint32_t));
    for (int32_t rt : program.register_targets)
      append(&rt, sizeof(int32_t));
    uint32_t nrty = static_cast<uint32_t>(program.register_types.size());
    append(&nrty, sizeof(uint32_t));
    for (JitScalarType t : program.register_types)
      append(&t, sizeof(JitScalarType));
  }

  auto cache_it = microkernel_cache_.find(cache_key);
  if (cache_it != microkernel_cache_.end())
    return cache_it->second;

  const std::string hash = md5_hex(cache_key);

  // ── LLVM module setup ──
  auto context = std::make_unique<llvm::LLVMContext>();
  auto module  = std::make_unique<llvm::Module>(hash, *context);
  module->setDataLayout(jit_->getDataLayout());

  llvm::IRBuilder<> builder(*context);
  llvm::Type * void_ty = builder.getVoidTy();
  llvm::Type * f64_ty  = builder.getDoubleTy();
  llvm::Type * i64_ty  = builder.getInt64Ty();
  llvm::Type * ptr_ty  = llvm::PointerType::get(*context, 0);

  // PerSampleFn signature:
  //   (regs, arrays, array_sizes, temps, sample_rate, sample_index,
  //    param_ptrs, slots) → void
  llvm::FunctionType * per_sample_ty = llvm::FunctionType::get(
    void_ty,
    {ptr_ty, ptr_ty, ptr_ty, ptr_ty, f64_ty, i64_ty, ptr_ty, ptr_ty},
    false);
  // PostambleMixFn signature: adds (output_buffer, output_index)
  llvm::FunctionType * postamble_mix_ty = llvm::FunctionType::get(
    void_ty,
    {ptr_ty, ptr_ty, ptr_ty, ptr_ty, f64_ty, i64_ty, ptr_ty, ptr_ty, ptr_ty, i64_ty},
    false);

  // Per-temp scalar type table, shared across the N+3 EmitCtxs. Each
  // function call observes writes from prior calls' bodies because the
  // table outlives any single function emission.
  std::vector<ST> temp_types(program.register_count, ST::Float);

  // ── Helper: bind arg names + slots_arg attributes for a PerSampleFn ──
  auto bind_per_sample_args = [&](llvm::Function * fn, EmitCtx & ctx) {
    auto it = fn->arg_begin();
    ctx.regs_arg        = &*it++;  ctx.regs_arg->setName("registers");
    ctx.arrays_arg      = &*it++;  ctx.arrays_arg->setName("arrays");
    ctx.array_sizes_arg = &*it++;  ctx.array_sizes_arg->setName("array_sizes");
    ctx.temps_arg       = &*it++;  ctx.temps_arg->setName("temps");
    ctx.sample_rate_arg = &*it++;  ctx.sample_rate_arg->setName("sampleRate");
    ctx.current_sample_idx = &*it++; ctx.current_sample_idx->setName("sample_index");
    ctx.param_ptrs_arg  = &*it++;  ctx.param_ptrs_arg->setName("param_ptrs");
    llvm::Argument * slots_a = &*it++; slots_a->setName("slots");
    ctx.slots_arg = slots_a;
    ctx.inputs_arg = nullptr;  // microkernel mode: no Input operands
    // Match fused-mode slot attributes so GVN can fold redundant loads.
#if LLVM_VERSION_MAJOR >= 21
    slots_a->addAttr(llvm::Attribute::getWithCaptureInfo(*context, llvm::CaptureInfo::none()));
#else
    slots_a->addAttr(llvm::Attribute::NoCapture);
#endif
    slots_a->addAttr(llvm::Attribute::NoAlias);
  };

  // ── Emit one PerSampleFn from a list of instructions ──
  auto emit_per_sample_function = [&](
    const std::string & sym,
    const std::vector<FlatInstr> & instrs) -> llvm::Error
  {
    llvm::Function * fn = llvm::Function::Create(
      per_sample_ty, llvm::Function::ExternalLinkage, sym, module.get());
    EmitCtx ctx;
    emit_ctx_init_module(ctx, *context, builder, *module);
    ctx.fn          = fn;
    ctx.program     = &program;
    ctx.param_index = &param_index;
    ctx.temp_types  = &temp_types;
    bind_per_sample_args(fn, ctx);

    llvm::BasicBlock * entry = llvm::BasicBlock::Create(*context, "entry", fn);
    builder.SetInsertPoint(entry);
    if (auto err = ctx.emit_instrs(instrs)) return err;
    builder.CreateRetVoid();

    if (llvm::verifyFunction(*fn, &llvm::errs()))
      return llvm::make_error<llvm::StringError>(
        "compile_microkernel: invalid IR in " + sym,
        llvm::inconvertibleErrorCode());
    return llvm::Error::success();
  };

  // ── Pre-declare LLVM functions for deep mode ──
  // In deep mode, each InstanceProgram (top-level AND every descendant)
  // becomes its own LLVM function. The parent's body emits CreateCall
  // to each child's function instead of inlining. All function symbols
  // must exist BEFORE any function body is emitted, so we pre-declare
  // every node's function first, then emit bodies in a second pass.
  //
  // `nested_to_emit` lists ONLY the descendants (not top-level), since
  // top-level functions are created in the main emission loop below
  // anyway and have the C++-scheduler-visible symbols.
  std::unordered_map<const InstanceProgram *, llvm::Function *> node_to_fn;
  std::vector<std::pair<const InstanceProgram *, llvm::Function *>> nested_to_emit;
  if (deep) {
    std::function<void(const InstanceProgram &, const std::string &)> declare_descendants =
      [&](const InstanceProgram & inst, const std::string & path) {
        for (std::size_t i = 0; i < inst.children.size(); ++i) {
          const auto & child = inst.children[i];
          const std::string sub_path = path + "_c" + std::to_string(i);
          llvm::Function * child_fn = llvm::Function::Create(
            per_sample_ty, llvm::Function::ExternalLinkage,
            "mk_" + hash + sub_path, module.get());
          node_to_fn[&child] = child_fn;
          nested_to_emit.push_back({&child, child_fn});
          declare_descendants(child, sub_path);
        }
      };
    for (std::size_t i = 0; i < program.instance_functions.size(); ++i) {
      declare_descendants(program.instance_functions[i], "_inst_" + std::to_string(i));
    }
  }

  // ── Emit one PerSampleFn from one InstanceProgram (preamble + per-child {pre_input + child} + main + writebacks) ──
  // Shallow mode (`deep_child_fns == nullptr`): emit_kernel_block
  // recurses inline — one LLVM function per top-level session
  // instance, children inlined.
  // Deep mode (`deep_child_fns` populated): emit_kernel_block emits
  // CreateCall to each child's pre-declared function. The C++
  // scheduler still only iterates top-level instances; the recursive
  // tree of calls happens entirely in LLVM IR.
  auto emit_instance_function = [&](
    llvm::Function * fn,
    const InstanceProgram & inst) -> llvm::Error
  {
    EmitCtx ctx;
    emit_ctx_init_module(ctx, *context, builder, *module);
    ctx.fn          = fn;
    ctx.program     = &program;
    ctx.param_index = &param_index;
    ctx.temp_types  = &temp_types;
    ctx.deep_child_fns = deep ? &node_to_fn : nullptr;
    bind_per_sample_args(fn, ctx);

    llvm::BasicBlock * entry = llvm::BasicBlock::Create(*context, "entry", fn);
    builder.SetInsertPoint(entry);
    if (auto err = ctx.emit_kernel_block(inst)) return err;
    builder.CreateRetVoid();

    if (llvm::verifyFunction(*fn, &llvm::errs()))
      return llvm::make_error<llvm::StringError>(
        "compile_microkernel: invalid IR in " + std::string(fn->getName()),
        llvm::inconvertibleErrorCode());
    return llvm::Error::success();
  };

  // ── Emit postamble_mix: scheduler.postamble + output mix store ──
  auto emit_postamble_mix_function = [&](const std::string & sym) -> llvm::Error
  {
    llvm::Function * fn = llvm::Function::Create(
      postamble_mix_ty, llvm::Function::ExternalLinkage, sym, module.get());
    EmitCtx ctx;
    emit_ctx_init_module(ctx, *context, builder, *module);
    ctx.fn          = fn;
    ctx.program     = &program;
    ctx.param_index = &param_index;
    ctx.temp_types  = &temp_types;

    auto it = fn->arg_begin();
    ctx.regs_arg        = &*it++;  ctx.regs_arg->setName("registers");
    ctx.arrays_arg      = &*it++;  ctx.arrays_arg->setName("arrays");
    ctx.array_sizes_arg = &*it++;  ctx.array_sizes_arg->setName("array_sizes");
    ctx.temps_arg       = &*it++;  ctx.temps_arg->setName("temps");
    ctx.sample_rate_arg = &*it++;  ctx.sample_rate_arg->setName("sampleRate");
    ctx.current_sample_idx = &*it++; ctx.current_sample_idx->setName("sample_index");
    ctx.param_ptrs_arg  = &*it++;  ctx.param_ptrs_arg->setName("param_ptrs");
    llvm::Argument * slots_a = &*it++; slots_a->setName("slots");
    ctx.slots_arg = slots_a;
    llvm::Value * output_buffer_arg = &*it++; output_buffer_arg->setName("output_buffer");
    llvm::Value * output_index_arg  = &*it++; output_index_arg->setName("output_index");
    ctx.inputs_arg = nullptr;
#if LLVM_VERSION_MAJOR >= 21
    slots_a->addAttr(llvm::Attribute::getWithCaptureInfo(*context, llvm::CaptureInfo::none()));
#else
    slots_a->addAttr(llvm::Attribute::NoCapture);
#endif
    slots_a->addAttr(llvm::Attribute::NoAlias);

    llvm::BasicBlock * entry = llvm::BasicBlock::Create(*context, "entry", fn);
    builder.SetInsertPoint(entry);

    if (auto err = ctx.emit_instrs(program.scheduler.postamble)) return err;

    // Output sinks: same slot-mix as fused-mode tail (legacy temp-mix
    // fallback for plan_4). v1 realizes target 0 → output_buffer[output_index].
    if (!program.sinks.empty()) {
      for (const auto & sink : program.sinks) {
        llvm::Value * mixed = ctx.zero_f64;
        for (uint32_t slot : sink.inputs)
          mixed = builder.CreateFAdd(mixed, ctx.load_slot_f64(slot));
        llvm::Value * scaled = builder.CreateFMul(mixed, llvm::ConstantFP::get(f64_ty, sink.gain));
        llvm::Value * out_ptr = builder.CreateInBoundsGEP(f64_ty, output_buffer_arg, output_index_arg);
        builder.CreateStore(scaled, out_ptr);
      }
    } else {
      llvm::Value * mixed = ctx.zero_f64;
      for (uint32_t mt : program.mix_output_temps) {
        llvm::Value * val = ctx.load_temp_typed(mt, temp_types[mt]);
        llvm::Value * f64_val = ctx.coerce_val(val, temp_types[mt], ST::Float);
        mixed = builder.CreateFAdd(mixed, f64_val);
      }
      llvm::Value * scaled = builder.CreateFDiv(mixed, llvm::ConstantFP::get(f64_ty, 20.0));
      llvm::Value * out_ptr = builder.CreateInBoundsGEP(f64_ty, output_buffer_arg, output_index_arg);
      builder.CreateStore(scaled, out_ptr);
    }

    builder.CreateRetVoid();

    if (llvm::verifyFunction(*fn, &llvm::errs()))
      return llvm::make_error<llvm::StringError>(
        "compile_microkernel: invalid IR in " + sym,
        llvm::inconvertibleErrorCode());
    return llvm::Error::success();
  };

  // ── Emit all functions ──
  // Both modes emit one LLVM function per top-level session instance;
  // the C++ scheduler iterates only those. The shapes differ inside:
  // shallow inlines nested children; deep emits each nested child as
  // its own LLVM function and the parent calls them via CreateCall.
  // Either way, preamble / state_evolution / postamble_mix are their
  // own functions.
  const std::string sym_preamble        = "mk_" + hash + "_preamble";
  const std::string sym_state_evolution = "mk_" + hash + "_state_evo";
  const std::string sym_postamble_mix   = "mk_" + hash + "_postamble_mix";

  // Top-level instance function names. The C++ scheduler dispatches
  // these in source order per sample.
  std::vector<std::string> sym_instances;
  sym_instances.reserve(program.instance_functions.size());
  std::vector<llvm::Function *> top_fns;
  top_fns.reserve(program.instance_functions.size());
  for (std::size_t i = 0; i < program.instance_functions.size(); ++i) {
    const std::string sym = "mk_" + hash + "_inst_" + std::to_string(i);
    sym_instances.push_back(sym);
    llvm::Function * fn = llvm::Function::Create(
      per_sample_ty, llvm::Function::ExternalLinkage, sym, module.get());
    if (deep) {
      // Register the top-level function in node_to_fn too, in case any
      // sibling references it (shouldn't normally happen — children
      // can't reference top-level instances — but the map is the
      // authoritative lookup for child dispatches).
      node_to_fn[&program.instance_functions[i]] = fn;
    }
    top_fns.push_back(fn);
  }

  // Emit in scheduler-call order so temp_types builds up consistently
  // with the runtime dispatch sequence.
  if (auto err = emit_per_sample_function(sym_preamble, program.scheduler.preamble))
    return std::move(err);
  // In deep mode, emit nested-child function bodies first (any order is
  // safe — they all call into each other via pre-declared symbols),
  // then top-levels. Bodies don't depend on emission order because
  // cross-function data flow uses the unified slot/state-reg arrays,
  // not LLVM SSA across function boundaries.
  if (deep) {
    for (const auto & [inst_ptr, fn] : nested_to_emit) {
      if (auto err = emit_instance_function(fn, *inst_ptr))
        return std::move(err);
    }
  }
  for (std::size_t i = 0; i < program.instance_functions.size(); ++i) {
    if (auto err = emit_instance_function(top_fns[i], program.instance_functions[i]))
      return std::move(err);
  }
  if (auto err = emit_per_sample_function(sym_state_evolution, program.scheduler.state_evolution))
    return std::move(err);
  if (auto err = emit_postamble_mix_function(sym_postamble_mix))
    return std::move(err);

  if (std::getenv("TROPICAL_DUMP_IR"))
    module->print(llvm::errs(), nullptr);

  if (llvm::verifyModule(*module, &llvm::errs()))
    return llvm::make_error<llvm::StringError>(
      "compile_microkernel: generated invalid LLVM module",
      llvm::inconvertibleErrorCode());

  if (auto err = add_module(std::move(context), std::move(module)))
    return std::move(err);

  // ── Look up each symbol and populate the result struct ──
  auto lookup_fn = [&](const std::string & sym) -> llvm::Expected<uint64_t> {
    auto a = lookup(sym);
    if (!a) return a.takeError();
    return *a;
  };

  MicrokernelKernels result;
  {
    auto a = lookup_fn(sym_preamble);
    if (!a) return a.takeError();
    result.preamble = reinterpret_cast<PerSampleFn>(*a);
  }
  result.instances.reserve(sym_instances.size());
  for (const auto & s : sym_instances) {
    auto a = lookup_fn(s);
    if (!a) return a.takeError();
    result.instances.push_back(reinterpret_cast<PerSampleFn>(*a));
  }
  {
    auto a = lookup_fn(sym_state_evolution);
    if (!a) return a.takeError();
    result.state_evolution = reinterpret_cast<PerSampleFn>(*a);
  }
  {
    auto a = lookup_fn(sym_postamble_mix);
    if (!a) return a.takeError();
    result.postamble_mix = reinterpret_cast<PostambleMixFn>(*a);
  }

  microkernel_cache_.emplace(std::move(cache_key), result);
  return result;
}

} // namespace tropical_jit
