import Tropical.Session.Compile
import Tropical.Artifact

/-!
# tropicalc — the compiler, with no runtime

    tropicalc <patch.json> --out=<dir> [--metal] [--mode=M] [--frames=N]
    tropicalc <patch.json> --emit=plan|kernel [--mode=M] [--frames=N]

Boots the stdlib registry as a value, compiles the patch through the same
`Tropical.Session` path the engine runs (`compilePatch`: normalize, load,
lift, resolved root, partition), and writes the kernel artifact for
`--frames` samples per call (default 512): `kernel.ll` (the port kernel),
`manifest.json` (its `tropical_kernel_1` type, the plan nested), plus
`kernel.metal` and `stage0.ll` with `--metal`. `--emit=plan` prints the
plan's wire JSON instead, byte-identical to `diffcli compile`
(tools/check_artifact_parity.py holds that gate); `--emit=kernel` prints the
manifest.

This binary links nothing from libtropical, and tools/check_layers.py keeps
its import closure inside TropicalCore. Everything a host loads comes out of
here; what a host does with it is the host's.
-/

def parseStrFlag (args : List String) (flag : String) : Option String :=
  args.findSome? fun a =>
    if a.startsWith (flag ++ "=") then some (a.drop (flag.length + 1)).toString else none

def usage : String :=
  "usage: tropicalc <patch.json> (--out=<dir> [--metal] [--frames=N] | --emit=plan|kernel [--frames=N]) " ++
  "[--mode=fused|microkernel|microkernel-deep]"

def main (args : List String) : IO UInt32 := do
  let some patch := args.find? (fun a => !a.startsWith "--")
    | IO.eprintln usage; return 1
  let modeStr := (parseStrFlag args "--mode").getD "fused"
  let some mode := Tropical.Plan.CompilationMode.ofWire? modeStr
    | IO.eprintln s!"tropicalc: unknown compilation mode: {modeStr}"; return 1
  let text ← IO.FS.readFile patch
  let registry ← match Tropical.Session.bootRegistry with
    | .error f => IO.eprintln s!"tropicalc: stdlib boot failed: {f.toJson.compress}"; return 1
    | .ok st => pure st
  let (plan, stageBlocks) ←
    match (Tropical.Session.compilePatch text mode).runOn registry with
    | (.error f, _) => IO.eprintln f.toJson.compress; return 1
    | (.ok r, _) => pure r
  let frames : Nat ← match parseStrFlag args "--frames" with
    | none => pure Tropical.Artifact.defaultFrames
    | some text =>
      match text.toNat? with
      | some n =>
        if n > 0 then pure n
        else
          IO.eprintln "tropicalc: --frames must be positive"
          return 1
      | none =>
        IO.eprintln s!"tropicalc: bad --frames '{text}'"
        return 1
  match parseStrFlag args "--emit" with
  | some "plan" =>
    match plan.toWire with
    | .error e => IO.eprintln s!"tropicalc: toWire: {e}"; return 1
    | .ok j => IO.println j.compress; return 0
  | some "kernel" =>
    match Tropical.Kernel.manifest plan frames with
    | .error e => IO.eprintln s!"tropicalc: kernel: {e}"; return 1
    | .ok m => IO.println m; return 0
  | some other =>
    IO.eprintln s!"tropicalc: unknown --emit target '{other}' (plan, kernel)"; return 1
  | none =>
    let some out := parseStrFlag args "--out"
      | IO.eprintln usage; return 1
    let metal := args.contains "--metal"
    let split ← Tropical.Artifact.splitTyped plan stageBlocks
    let emitted := if metal then Tropical.Artifact.emitMetal split frames
                   else Tropical.Artifact.emitJit split frames
    let artifact ← match emitted with
      | .error e => IO.eprintln s!"tropicalc: emit: {e}"; return 1
      | .ok a => pure a
    let dir := System.FilePath.mk out
    IO.FS.createDirAll dir
    IO.FS.writeFile (dir / "kernel.ll") artifact.ir
    IO.FS.writeFile (dir / "manifest.json") artifact.manifest
    if metal then
      IO.FS.writeFile (dir / "kernel.metal") artifact.msl
      IO.FS.writeFile (dir / "stage0.ll") artifact.stage0Ir
    return 0
