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
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>
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

#ifdef TROPICAL_WASM_EMIT
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Triple.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/FileSystem.h>
#include <lld/Common/Driver.h>
LLD_HAS_DRIVER(wasm)
#endif

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

// The unoptimized sibling JIT: CodeGenOptLevel::None end to end — fast
// instruction selection, the linear source-order scheduler, regalloc-fast —
// and no IR pipeline beyond O0's. Compile cost is O(size) in the module,
// which is the whole point: the stage-0 coefficient kernel is a
// thousands-of-flops single block that runs once per control write, and
// the default backend's scheduler/regalloc go superlinear on exactly that
// shape (the compile wall the split exists to remove). Built lazily; shares
// the disk object cache (module ids are salted so the tiers never collide).
llvm::orc::LLJIT * OrcJitEngine::unoptimized_jit()
{
  if (unopt_jit_) return unopt_jit_.get();
  if (!unopt_init_error_.empty()) return nullptr;

  KernelObjectCache * cache_ptr = object_cache_.get();
  auto jit_or_err = llvm::orc::LLJITBuilder()
    .setCompileFunctionCreator(
      [cache_ptr](llvm::orc::JITTargetMachineBuilder jtmb)
        -> llvm::Expected<std::unique_ptr<llvm::orc::IRCompileLayer::IRCompiler>>
      {
        jtmb.setCodeGenOptLevel(llvm::CodeGenOptLevel::None);
        return std::make_unique<llvm::orc::ConcurrentIRCompiler>(std::move(jtmb), cache_ptr);
      })
    .create();
  if (!jit_or_err)
  {
    unopt_init_error_ = llvm::toString(jit_or_err.takeError());
    return nullptr;
  }
  unopt_jit_ = std::move(*jit_or_err);
  unopt_jit_->getIRTransformLayer().setTransform(
    [](llvm::orc::ThreadSafeModule TSM,
       const llvm::orc::MaterializationResponsibility &)
        -> llvm::Expected<llvm::orc::ThreadSafeModule>
    {
      TSM.withModuleDo([](llvm::Module & M) {
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
        llvm::ModulePassManager MPM =
          PB.buildO0DefaultPipeline(llvm::OptimizationLevel::O0);
        MPM.run(M, MAM);
      });
      return std::move(TSM);
    });
  return unopt_jit_.get();
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
// compile_ir_text — the engine's "receive IR, JIT, run" core.
//
// Parses textual LLVM IR into a fresh module, content-addresses its
// single function definition so distinct IR never collides and identical
// IR is served from ir_kernel_cache_, then add_module + lookup. With the
// C++ plan compiler retired (Phase 2), this is the only kernel producer;
// Lean's EmitLlvm emits the IR.
// ---------------------------------------------------------------------------
llvm::Expected<NumericKernelFn> OrcJitEngine::compile_ir_text(
  const std::string & ir_text, bool unoptimized)
{
  if (!jit_)
  {
    return llvm::make_error<llvm::StringError>(
      "ORC JIT is not available: " + init_error_,
      llvm::inconvertibleErrorCode());
  }

  std::lock_guard<std::mutex> lock(jit_mutex_);

  // Salt the key by pipeline choice so identical text compiled at both
  // levels (however unlikely) never serves the wrong object from either
  // cache tier.
  const std::string key = md5_hex(ir_text) + (unoptimized ? "_u" : "");
  if (auto it = ir_kernel_cache_.find(key); it != ir_kernel_cache_.end())
    return it->second;

  llvm::orc::LLJIT * target = jit_.get();
  if (unoptimized)
  {
    target = unoptimized_jit();
    if (!target)
      return llvm::make_error<llvm::StringError>(
        "unoptimized JIT is not available: " + unopt_init_error_,
        llvm::inconvertibleErrorCode());
  }

  const std::string symbol = "tropical_k_" + key;

  auto context = std::make_unique<llvm::LLVMContext>();
  llvm::SMDiagnostic diag;
  auto buffer = llvm::MemoryBuffer::getMemBuffer(ir_text, "tropical_ir");
  auto module = llvm::parseIR(buffer->getMemBufferRef(), diag, *context);
  if (!module)
  {
    std::string err;
    llvm::raw_string_ostream os(err);
    diag.print("tropical_ir", os);
    return llvm::make_error<llvm::StringError>(
      "compile_ir_text: failed to parse IR: " + err,
      llvm::inconvertibleErrorCode());
  }
  module->setDataLayout(target->getDataLayout());

  // Rename the single function definition to the content-addressed
  // symbol. Declarations (e.g. intrinsics) are left alone.
  llvm::Function * kernel_fn = nullptr;
  for (llvm::Function & f : *module)
  {
    if (f.isDeclaration()) continue;
    if (kernel_fn)
      return llvm::make_error<llvm::StringError>(
        "compile_ir_text: IR defines more than one function",
        llvm::inconvertibleErrorCode());
    kernel_fn = &f;
  }
  if (!kernel_fn)
    return llvm::make_error<llvm::StringError>(
      "compile_ir_text: IR defines no function",
      llvm::inconvertibleErrorCode());
  kernel_fn->setName(symbol);
  kernel_fn->setLinkage(llvm::Function::ExternalLinkage);

  // Give the module a unique identity. parseIR preserves the original
  // `; ModuleID` from the printed text, which would make the disk
  // KernelObjectCache serve the *plan-compiled* object (defining
  // tropical_f_<hash>) for this renamed module — whose symbol isn't in
  // it, so materialization fails. Key the module on the IR content hash
  // instead, matching the renamed symbol.
  module->setModuleIdentifier(key);
  module->setSourceFileName(key);

  if (llvm::verifyModule(*module, &llvm::errs()))
    return llvm::make_error<llvm::StringError>(
      "compile_ir_text: parsed IR fails verification",
      llvm::inconvertibleErrorCode());

  llvm::orc::ThreadSafeModule tsm(std::move(module), std::move(context));
  if (auto err = target->addIRModule(std::move(tsm)))
    return std::move(err);

  auto symbol_or_err = target->lookup(symbol);
  if (!symbol_or_err)
    return symbol_or_err.takeError();
#if LLVM_VERSION_MAJOR >= 15
  const uint64_t addr = symbol_or_err->getValue();
#else
  const uint64_t addr = symbol_or_err->getAddress();
#endif

  auto kernel = reinterpret_cast<NumericKernelFn>(addr);
  ir_kernel_cache_.emplace(key, kernel);
  return kernel;
}


#ifdef TROPICAL_WASM_EMIT
// ---------------------------------------------------------------------------
// compile_ir_to_wasm — the IR→wasm sibling of compile_ir_text. Same parse;
// retargets to wasm32 and links with lld in-process instead of JITting.
// ---------------------------------------------------------------------------
std::vector<uint8_t> compile_ir_to_wasm(const std::string & ir_text, std::string & err)
{
  LLVMInitializeWebAssemblyTargetInfo();
  LLVMInitializeWebAssemblyTarget();
  LLVMInitializeWebAssemblyTargetMC();
  LLVMInitializeWebAssemblyAsmPrinter();

  llvm::LLVMContext context;
  llvm::SMDiagnostic diag;
  auto buffer = llvm::MemoryBuffer::getMemBuffer(ir_text, "tropical_ir");
  auto module = llvm::parseIR(buffer->getMemBufferRef(), diag, context);
  if (!module)
  {
    llvm::raw_string_ostream os(err);
    diag.print("tropical_ir", os);
    return {};
  }

  const llvm::Triple triple("wasm32-unknown-unknown");
  std::string lookup_err;
  const llvm::Target * target = llvm::TargetRegistry::lookupTarget(triple, lookup_err);
  if (!target) { err = "no wasm32 target: " + lookup_err; return {}; }

  llvm::TargetOptions opts;
  // Honor FP flags only (the default). Lean's IR carries no `contract`/`fast`
  // flags, so no FMA is ever formed — the kernel computes op-for-op identically
  // to the ORC JIT. This is the load-bearing line for native≡wasm bit-exactness.
  opts.AllowFPOpFusion = llvm::FPOpFusion::Standard;
  std::unique_ptr<llvm::TargetMachine> tm(target->createTargetMachine(
    triple, "generic", "", opts, std::nullopt, std::nullopt,
    llvm::CodeGenOptLevel::Default));
  module->setTargetTriple(triple);
  module->setDataLayout(tm->createDataLayout());

  llvm::SmallString<128> obj_path, wasm_path;
  if (auto ec = llvm::sys::fs::createTemporaryFile("tropical", "o", obj_path))
  { err = "temp obj: " + ec.message(); return {}; }
  if (auto ec = llvm::sys::fs::createTemporaryFile("tropical", "wasm", wasm_path))
  { llvm::sys::fs::remove(obj_path); err = "temp wasm: " + ec.message(); return {}; }

  {
    std::error_code ec;
    llvm::raw_fd_ostream obj_out(obj_path, ec, llvm::sys::fs::OF_None);
    if (ec)
    {
      llvm::sys::fs::remove(obj_path); llvm::sys::fs::remove(wasm_path);
      err = "open obj: " + ec.message(); return {};
    }
    llvm::legacy::PassManager pm;
    if (tm->addPassesToEmitFile(pm, obj_out, nullptr, llvm::CodeGenFileType::ObjectFile))
    {
      llvm::sys::fs::remove(obj_path); llvm::sys::fs::remove(wasm_path);
      err = "wasm32 target cannot emit an object file"; return {};
    }
    pm.run(*module);
  } // obj_out flushes/closes here

  const char * argv[] = {
    "wasm-ld", "--no-entry", "--import-memory",
    "--export=tropical_kernel", "--export=__heap_base", "--allow-undefined",
    obj_path.c_str(), "-o", wasm_path.c_str()
  };
  std::string link_err;
  llvm::raw_string_ostream link_os(link_err);
  bool ok = lld::wasm::link(
    llvm::ArrayRef<const char *>(argv, sizeof(argv) / sizeof(argv[0])),
    llvm::outs(), link_os, /*exitEarly=*/false, /*disableOutput=*/false);
  if (!ok)
  {
    llvm::sys::fs::remove(obj_path); llvm::sys::fs::remove(wasm_path);
    err = "wasm-ld: " + link_err;
    return {};
  }

  auto wasm = llvm::MemoryBuffer::getFile(wasm_path);
  llvm::sys::fs::remove(obj_path);
  llvm::sys::fs::remove(wasm_path);
  if (!wasm) { err = "read wasm: " + wasm.getError().message(); return {}; }

  auto bytes = (*wasm)->getBuffer();
  const auto * p = reinterpret_cast<const uint8_t *>(bytes.data());
  return std::vector<uint8_t>(p, p + bytes.size());
}
#endif


} // namespace tropical_jit
