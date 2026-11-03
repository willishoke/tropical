import Tropical.Session.Compile
import Tropical.Playground.Compile
import Tropical.Artifact
import Tropical.CliArgs

/-!
# tropicalc — the compiler, with no runtime

    tropicalc <patch-or-graph.json> --out=<dir> [--metal] [--mode=M] [--frames=N] [--rate=R] [--graph]
    tropicalc <patch-or-graph.json> --emit=plan|kernel [--mode=M] [--frames=N] [--rate=R] [--graph]

Boots the stdlib registry as a value, compiles the patch through the same
`Tropical.Session` path the engine runs (`compilePatch`: normalize, load,
lift, resolved root, partition) — or, for a playground graph (a flat
PatchGraph or a v3 hierarchy document, `Playground.isGraphDocument`; `--graph`
forces it), through `Playground.compilePlan`, the pure twin of the engine's
`load_patch_graph` (fused only) — and writes the kernel artifact for
`--frames` samples per call (default 512): `kernel.ll` (the port kernel),
`manifest.json` (its `tropical_kernel_1` type, the plan nested), plus
`kernel.metal` and `stage0.ll` with `--metal`. `--emit=plan` prints the
plan's wire JSON instead, byte-identical to `diffcli compile`
(tools/check_artifact_parity.py holds that gate); `--emit=kernel` prints the
manifest.

Flags are read as `diffcli` reads them (`Tropical.CliArgs`): a value flag is
`--name=V` or `--name V`, and an unknown flag or a malformed value (a
`--rate` or `--frames` that is not a positive integer) is an error.

This binary links nothing from libtropical, and tools/check_layers.py keeps
its import closure inside TropicalCore. Everything a host loads comes out of
here; what a host does with it is the host's.
-/

open Tropical.CliArgs

def usage : String :=
  "usage: tropicalc <patch-or-graph.json> (--out=<dir> [--metal] [--frames=N] | --emit=plan|kernel [--frames=N]) " ++
  "[--mode=fused|microkernel|microkernel-deep] [--rate=R] [--graph]"

def flags : List Flag :=
  [valued "out", valued "emit", valued "mode", valued "frames", valued "rate",
    switch "metal", switch "graph"]

def main (args : List String) : IO UInt32 := do
  let decoded : Except String _ := do
    let a ← parse flags args
    let patch ← a.single
    let modeStr := (a.value? "mode").getD "fused"
    let some mode := Tropical.Plan.CompilationMode.ofWire? modeStr
      | throw s!"unknown compilation mode: {modeStr}"
    let frames := (← a.positive? "frames").getD Tropical.Artifact.defaultFrames
    pure (a, patch, modeStr, mode, ← a.rate?, frames)
  let (a, patch, modeStr, mode, rate?, frames) ← match decoded with
    | .ok v => pure v
    | .error e => IO.eprintln s!"tropicalc: {e}\n{usage}"; return 1
  let text ← IO.FS.readFile patch
  let graph? := match Lean.Json.parse text with
    | .ok j => if a.has "graph" || Tropical.Playground.isGraphDocument j then some j else none
    | .error _ => none
  let (plan, stageBlocks) ← match graph? with
    | some j =>
      if mode != .fused then
        IO.eprintln s!"tropicalc: a PatchGraph compiles fused only (got --mode={modeStr})"; return 1
      match Tropical.Playground.compilePlan j (rate?.getD Tropical.Plan.defaultSampleRate) with
      | .error e => IO.eprintln s!"tropicalc: graph: {e}"; return 1
      | .ok compiled => pure (compiled.plan, compiled.stageBlocks)
    | none =>
      let registry ← match Tropical.Session.bootRegistry with
        | .error f => IO.eprintln s!"tropicalc: stdlib boot failed: {f.toJson.compress}"; return 1
        | .ok st => pure st
      match (Tropical.Session.compilePatch text mode).runOn registry with
      | (.error f, _) => IO.eprintln f.toJson.compress; return 1
      | (.ok (plan, blocks), _) =>
        -- A patch takes the rate after compile (nothing above the plan bakes one).
        pure (match rate? with
          | some r => { plan with sampleRate := (r : Lean.JsonNumber) }
          | none => plan, blocks)
  match a.value? "emit" with
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
    let some out := a.value? "out"
      | IO.eprintln s!"tropicalc: --out=<dir> or --emit=plan|kernel is required\n{usage}"; return 1
    let metal := a.has "metal"
    -- The one environment read, at the root: `TROPICAL_STAGE0=0` is the
    -- stage-0 kill-switch the engine honours too.
    let opts : Tropical.Artifact.Options :=
      { stage0 := (← IO.getEnv "TROPICAL_STAGE0") != some "0" }
    let split ← match Tropical.Artifact.splitTyped opts plan stageBlocks with
      | .error e => IO.eprintln s!"tropicalc: {e}"; return 1
      | .ok s => pure s
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
