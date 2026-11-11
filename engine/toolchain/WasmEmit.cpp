#include "toolchain/WasmEmit.hpp"

#include <llvm/IR/Instructions.h>
#include <llvm/MC/MCSubtargetInfo.h>
#include <llvm/Object/ObjectFile.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/TargetParser/Triple.h>
#include <lld/Common/Driver.h>

#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

LLD_HAS_DRIVER(wasm)

namespace tropical_toolchain
{

// ---------------------------------------------------------------------------
// compile_ir_to_wasm — the IR→wasm sibling of OrcJitEngine::compile_ir_text.
// Same parse; retargets to wasm32 and links with lld in-process instead of
// JITting.
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
  opts.NoInfsFPMath = false;
  opts.NoNaNsFPMath = false;
  opts.NoSignedZerosFPMath = false;
  std::unique_ptr<llvm::TargetMachine> tm(target->createTargetMachine(
    triple, "generic", "", opts, std::nullopt, std::nullopt,
    llvm::CodeGenOptLevel::Default));
  module->setTargetTriple(triple);
  module->setDataLayout(tm->createDataLayout());

  // The port kernel keeps its per-call storage (slots, SSA temps, arrays) in
  // allocas, which wasm puts on the shadow stack. wasm-ld's default stack is
  // 64 KiB and sits below the data segment, so a large plan would silently
  // overwrite its own constants. Size the stack from the module's allocas
  // plus headroom for the codegen's own frame, and link it first so an
  // overflow traps (the address wraps out of bounds) instead.
  uint64_t frame_bytes = 0;
  for (const auto & fn : *module)
    for (const auto & bb : fn)
      for (const auto & inst : bb)
        if (const auto * ai = llvm::dyn_cast<llvm::AllocaInst>(&inst))
        {
          const auto size = ai->getAllocationSize(module->getDataLayout());
          if (!size || size->isScalable())
          { err = "wasm: alloca of non-constant size in @" + fn.getName().str(); return {}; }
          frame_bytes += size->getFixedValue();
        }
  constexpr uint64_t kStackHeadroom = 64 * 1024;
  const uint64_t stack_bytes =
    (frame_bytes + kStackHeadroom + kStackHeadroom - 1) / kStackHeadroom * kStackHeadroom;

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

  const std::string stack_size = "stack-size=" + std::to_string(stack_bytes);
  const std::vector<const char *> argv = {
    "wasm-ld", "--no-entry", "--import-memory", "--stack-first",
    "-z", stack_size.c_str(),
    "--export=tropical_kernel", "--export=__heap_base", "--allow-undefined",
    obj_path.c_str(), "-o", wasm_path.c_str()
  };
  std::string link_err;
  llvm::raw_string_ostream link_os(link_err);
  // lldMain, not lld::wasm::link directly: only lldMain destroys the linker's
  // global context afterwards, so a second link in the same process (a test,
  // a batch build) starts from a fresh config instead of the last one's.
  const lld::Result linked = lld::lldMain(
    argv, llvm::outs(), link_os, {{lld::Wasm, &lld::wasm::link}});
  if (linked.retCode != 0)
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

// The triple as LLVM's tools read it: normalized, so `thumbv7em-none-eabihf`
// is arch-vendor-environment (EABIHF, the hard-float procedure call standard)
// rather than arch-vendor-os with an unknown OS `eabihf`. Unnormalized, the
// ARM target falls back to the soft-float ABI and marks the object base
// AAPCS (no Tag_ABI_VFP_args), which a hard-float firmware link refuses.
// "" is the host.
static llvm::Triple target_triple(const std::string & text)
{
  return llvm::Triple(llvm::Triple::normalize(
    text.empty() ? llvm::sys::getDefaultTargetTriple() : text));
}

// ---------------------------------------------------------------------------
// compile_ir_to_object — the same parse and the same TargetOptions as the
// wasm path, retargeted to any triple this LLVM carries, and stopped at the
// relocatable object. Initialized targets: the host, 32-bit ARM/Thumb (the
// Cortex-M class) and wasm32.
// ---------------------------------------------------------------------------
std::vector<uint8_t> compile_ir_to_object(const std::string & ir_text,
                                          const std::string & triple_text,
                                          const std::string & cpu,
                                          const std::string & features,
                                          std::string & err)
{
  LLVMInitializeWebAssemblyTargetInfo();
  LLVMInitializeWebAssemblyTarget();
  LLVMInitializeWebAssemblyTargetMC();
  LLVMInitializeWebAssemblyAsmPrinter();
  LLVMInitializeARMTargetInfo();
  LLVMInitializeARMTarget();
  LLVMInitializeARMTargetMC();
  LLVMInitializeARMAsmPrinter();
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

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

  const llvm::Triple triple = target_triple(triple_text);
  std::string lookup_err;
  const llvm::Target * target = llvm::TargetRegistry::lookupTarget(triple, lookup_err);
  if (!target) { err = "no target for " + triple.str() + ": " + lookup_err; return {}; }

  llvm::TargetOptions opts;
  // The numeric contract the manifest states: IEEE ops as written, no FMA
  // contraction (Standard fuses only an explicit llvm.fmuladd, which the
  // emitter never writes), no fast-math (no flags in the IR, none here).
  // Pinned field by field rather than left to defaults: with these, an
  // fmul feeding an fadd stays two rounded ops on every target (AArch64
  // FMADD, the M4F's VFMA and wasm's relaxed madd are never selected), so
  // the host, wasm32 and thumbv7em objects of one binary32 kernel compute
  // the same bits.
  opts.AllowFPOpFusion = llvm::FPOpFusion::Standard;
  opts.NoInfsFPMath = false;
  opts.NoNaNsFPMath = false;
  opts.NoSignedZerosFPMath = false;
  // ELF objects carry a .stack_sizes section (per-function frame bytes), so
  // the manifest's stack depth is read from the object, not guessed.
  opts.EmitStackSizeSection = triple.isOSBinFormatELF();
  // One section per function on ELF, so a firmware link with --gc-sections
  // keeps tropical_kernel and drops the _argv thunk it never calls.
  opts.FunctionSections = triple.isOSBinFormatELF();
  opts.DataSections = triple.isOSBinFormatELF();
  // The port functions are nounwind and nothing unwinds through a kernel:
  // no EH tables (.ARM.exidx / .eh_frame) for a firmware link to keep.
  opts.ExceptionModel = llvm::ExceptionHandling::None;
  // Static relocation, no PIC: what a bare-metal link expects, and harmless
  // for a host object linked into an executable.
  std::unique_ptr<llvm::TargetMachine> tm(target->createTargetMachine(
    triple, cpu.empty() ? "generic" : cpu, features, opts,
    llvm::Reloc::Static, std::nullopt, llvm::CodeGenOptLevel::Default));
  if (!tm) { err = "cannot create a target machine for " + triple.str(); return {}; }
  module->setTargetTriple(triple);
  module->setDataLayout(tm->createDataLayout());

  // The JIT's IR pipeline (per-module O2, OrcJitEngine's transform layer),
  // here target-aware and without unrolling: the emitted IR is uniform and un-folded by design,
  // and narrowing of masked i64 arithmetic happens in these passes.
  {
    // No loop unrolling or vectorization: one loop iteration is one frame
    // (the static cycle count reads it directly) and flash stays small.
    // Neither changes a value: the IR has no reassociable FP.
    llvm::PipelineTuningOptions pto;
    pto.LoopUnrolling = false;
    pto.LoopInterleaving = false;
    pto.LoopVectorization = false;
    pto.SLPVectorization = false;
    llvm::PassBuilder PB(tm.get(), pto);
    llvm::LoopAnalysisManager LAM;
    llvm::FunctionAnalysisManager FAM;
    llvm::CGSCCAnalysisManager CGAM;
    llvm::ModuleAnalysisManager MAM;
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
    llvm::ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O2);
    MPM.run(*module, MAM);
  }

  llvm::SmallString<0> bytes;
  llvm::raw_svector_ostream obj_out(bytes);
  llvm::legacy::PassManager pm;
  if (tm->addPassesToEmitFile(pm, obj_out, nullptr, llvm::CodeGenFileType::ObjectFile))
  { err = triple.str() + " target cannot emit an object file"; return {}; }
  pm.run(*module);
  const auto * p = reinterpret_cast<const uint8_t *>(bytes.data());
  return std::vector<uint8_t>(p, p + bytes.size());
}

// ---------------------------------------------------------------------------
// target_float_info — what the FPU of `triple`/`cpu`/`features` realizes, as
// LLVM's own subtarget sees it: binary32 and binary64 arithmetic in hardware,
// and the float ABI. A JSON object, so the caller refuses a request the
// target cannot realize by the target's own description, not a table.
// ---------------------------------------------------------------------------
std::string target_float_info(const std::string & triple_text, const std::string & cpu,
                              const std::string & features, std::string & err)
{
  LLVMInitializeWebAssemblyTargetInfo();
  LLVMInitializeWebAssemblyTarget();
  LLVMInitializeWebAssemblyTargetMC();
  LLVMInitializeARMTargetInfo();
  LLVMInitializeARMTarget();
  LLVMInitializeARMTargetMC();
  llvm::InitializeNativeTarget();

  const llvm::Triple triple = target_triple(triple_text);
  std::string lookup_err;
  const llvm::Target * target = llvm::TargetRegistry::lookupTarget(triple, lookup_err);
  if (!target) { err = "no target for " + triple.str() + ": " + lookup_err; return {}; }
  llvm::TargetOptions opts;
  std::unique_ptr<llvm::TargetMachine> tm(target->createTargetMachine(
    triple, cpu.empty() ? "generic" : cpu, features, opts,
    llvm::Reloc::Static, std::nullopt, llvm::CodeGenOptLevel::Default));
  if (!tm) { err = "cannot create a target machine for " + triple.str(); return {}; }
  const llvm::MCSubtargetInfo * sti = tm->getMCSubtargetInfo();
  bool b32 = true, b64 = true;
  std::string abi = "n/a";
  if (triple.isARM() || triple.isThumb())
  {
    // VFPv2 single precision is the base of every ARM FPU LLVM models;
    // `fp64` is the double-precision extension (absent on FPv4-SP-D16).
    b32 = sti->checkFeatures("+vfp2sp");
    b64 = b32 && sti->checkFeatures("+fp64");
    abi = tm->Options.FloatABIType == llvm::FloatABI::Hard ? "hard" : "soft";
  }
  return std::string("{\"triple\":\"") + triple.str() + "\",\"binary32\":"
    + (b32 ? "true" : "false") + ",\"binary64\":" + (b64 ? "true" : "false")
    + ",\"float_abi\":\"" + abi + "\"}";
}

// ---------------------------------------------------------------------------
// object_undefined_symbols — the symbols an object references and does not
// define: what a link must supply. For a port kernel, every such symbol is a
// runtime routine (a libcall) the target could not do inline.
// ---------------------------------------------------------------------------
std::vector<std::string> object_undefined_symbols(const uint8_t * data, size_t len,
                                                  std::string & err)
{
  std::vector<std::string> out;
  llvm::MemoryBufferRef ref(
    llvm::StringRef(reinterpret_cast<const char *>(data), len), "tropical_object");
  auto obj = llvm::object::ObjectFile::createObjectFile(ref);
  if (!obj) { err = llvm::toString(obj.takeError()); return {}; }
  for (const llvm::object::SymbolRef & sym : (*obj)->symbols())
  {
    auto flags = sym.getFlags();
    if (!flags) { err = llvm::toString(flags.takeError()); return {}; }
    if (!(*flags & llvm::object::SymbolRef::SF_Undefined)) continue;
    auto name = sym.getName();
    if (!name) { err = llvm::toString(name.takeError()); return {}; }
    if (name->empty()) continue;
    out.push_back(name->str());
  }
  return out;
}

} // namespace tropical_toolchain
