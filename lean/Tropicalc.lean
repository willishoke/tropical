import Tropical.Session.Compile
import Tropical.Artifact

/-!
# tropicalc — the compiler, with no runtime

    tropicalc <patch.json> --out=<dir> [--metal] [--mode=M]
    tropicalc <patch.json> --emit=plan [--mode=M]

Boots the stdlib registry as a value, compiles the patch through the same
`Tropical.Session` path the engine runs (`compilePatch`: normalize, load,
lift, resolved root, partition), and writes the kernel artifact: `audio.ll`,
`coeff.ll`, `manifest.json`, plus `tile.ll` and `audio.metal` with
`--metal`. `--emit=plan` prints the plan's wire JSON instead, byte-identical
to `diffcli compile` (tools/check_artifact_parity.py holds that gate).

This binary links nothing from libtropical, and tools/check_layers.py keeps
its import closure inside TropicalCore. Everything a host loads comes out of
here; what a host does with it is the host's.
-/

def parseStrFlag (args : List String) (flag : String) : Option String :=
  args.findSome? fun a =>
    if a.startsWith (flag ++ "=") then some (a.drop (flag.length + 1)).toString else none

def usage : String :=
  "usage: tropicalc <patch.json> (--out=<dir> [--metal] | --emit=plan) " ++
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
  match parseStrFlag args "--emit" with
  | some "plan" =>
    match plan.toWire with
    | .error e => IO.eprintln s!"tropicalc: toWire: {e}"; return 1
    | .ok j => IO.println j.compress; return 0
  | some other =>
    IO.eprintln s!"tropicalc: unknown --emit target '{other}' (plan)"; return 1
  | none =>
    let some out := parseStrFlag args "--out"
      | IO.eprintln usage; return 1
    let metal := args.contains "--metal"
    let split ← Tropical.Artifact.splitTyped plan stageBlocks
    let emitted := if metal then Tropical.Artifact.emitMetal split
                   else Tropical.Artifact.emitJit split
    let artifact ← match emitted with
      | .error e => IO.eprintln s!"tropicalc: emit: {e}"; return 1
      | .ok a => pure a
    let dir := System.FilePath.mk out
    IO.FS.createDirAll dir
    IO.FS.writeFile (dir / "audio.ll") artifact.ir
    IO.FS.writeFile (dir / "coeff.ll") artifact.coefficientIr
    IO.FS.writeFile (dir / "manifest.json") artifact.manifest
    if metal then
      IO.FS.writeFile (dir / "tile.ll") artifact.tileIr
      IO.FS.writeFile (dir / "audio.metal") artifact.msl
    return 0
