#include "toolchain/WasmEmit.hpp"

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Triple.h>
#include <lld/Common/Driver.h>

#include <memory>
#include <optional>
#include <system_error>

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

} // namespace tropical_toolchain
