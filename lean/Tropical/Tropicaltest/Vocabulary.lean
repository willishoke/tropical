import Tropical.Tropicaltest.BanksStaging

/-!
# Tropical.Tropicaltest.Vocabulary

The vocabulary contract gates: dead-slot lint (every param slot is read), the served-vocabulary drive (each kind compiles), manifest-discipline consistency, the realized-state report, and the slide cascade.
-/

open Tropical
open Tropical.Plan
open Tropical.Ir (Arena ProgramIdx)

-- ── THE DEAD-SLOT LINT ─────────────────────────────────────────────────────
/-- Module-slot indices READ by an instance function: every `.slot` operand of
    every instruction (preamble, body, pre-input), recursively through children.
    A `WriteSlot` dst is a write, not a read — a slot only written is still dead. -/
private def slotReadsOf (f : Tropical.Plan.InstanceFunction) : Array Nat :=
  let ofInstrs := fun (instrs : Array Tropical.Plan.NInstr) =>
    instrs.flatMap fun ins => ins.args.filterMap fun
      | .slot i _ => some i
      | _ => none
  ofInstrs f.preambleInstructions ++ ofInstrs f.instructions
    ++ ofInstrs f.preInputInstructions
    ++ f.children.attach.flatMap fun c => slotReadsOf c.1
termination_by sizeOf f
decreasing_by exact Tropical.Plan.InstanceFunction.sizeOf_lt_of_mem_children c.2

/-- `param:*` entries of `slotNames` referenced by NO instruction operand in the
    plan. Sink `inputs` are slot reads too (they consume the `__root__.out`-style
    output slots), but a param slot whose only consumer is a sink would itself be
    a wiring bug, so the lint demands an instruction read for `param:*`.

    The one manifest-derived exception is a glide's scalar `#t0` when all four
    exact-t0 limbs are present. That scalar deliberately remains write-only
    compatibility/introspection state; the kernel must read the limbs instead.
    A scalar `#t0` without the complete exact representation is NOT exempt,
    because it is the legacy glide's live clock input. -/
def unreadParamSlots (plan : Tropical.Plan.FlatPlan) : Array String := Id.run do
  let reads := plan.instanceFunctions.flatMap slotReadsOf
  let isCompatibilityT0 := fun (name : String) =>
    plan.paramDisciplines.any fun d =>
      d.discipline == "glide"
        && name == s!"param:{d.name}#t0"
        && (Array.range 4).all fun i =>
          d.companions.contains s!"{d.name}#t0#u{i}"
  let mut dead : Array String := #[]
  for i in [0:plan.slotNames.size] do
    let name := plan.slotNames[i]!
    if "param:".isPrefixOf name && !reads.contains i
        && !isCompatibilityT0 name then
      dead := dead.push name
  return dead

open Tropical.Playground in
/-- THE VOCABULARY-DRIVENNESS gate (successor to the table-coherence gate,
    which retired with the static `nodeSchema` it was pinning). `get_vocabulary`
    is only honest if the served table can DRIVE a client: for every kind,
    generate a minimal patch FROM the vocabulary alone (wire each `in` inlet
    from a domain-matching helper; leave optional normals intact), compile it
    through the real `compilePlanPure`, and assert every knob the table
    declares registers a live slot and nothing registered goes unread. A kind
    added to the table is covered here with no test edit. -/
def runVocabDriven (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let mut ok := true
  let mut covered := 0
  let mut issues : Array String := #[]
  -- Served identifiers are protocol keys, not display labels: every kind must
  -- occur exactly once, and an explicitly withheld implementation must never
  -- leak into the surface vocabulary. Keep these as direct set facts in
  -- addition to the compile-driven checks below, so a duplicate cannot pass by
  -- compiling the same minimal patch twice.
  let mut seen : Array String := #[]
  for kind in vocabularyKinds do
    if seen.contains kind then
      ok := false
      issues := issues.push s!"duplicate served kind: {kind}"
    else
      seen := seen.push kind
  for kind in withheldKinds do
    if vocabularyKinds.contains kind then
      ok := false
      issues := issues.push s!"withheld kind served: {kind}"
  -- Envelope identity and the algorithm-labelled deterministic fingerprint are
  -- part of the same vocabulary contract the generated patches exercise.
  let schemaOk := (vocabularyJson.getObjVal? "schema").toOption
      == some (Lean.Json.str vocabularySchema)
    && (vocabularyJson.getObjVal? "schema_version").toOption
      == some (Lean.toJson vocabularySchemaVersion)
  let fingerprintOk := (vocabularyJson.getObjVal? "fingerprint").toOption
      == some (Lean.Json.str vocabularyFingerprint)
    && vocabularyFingerprint.startsWith "fnv1a64:"
    && vocabularyFingerprint.length == "fnv1a64:".length + 16
    && vocabularyFingerprint
      == s!"fnv1a64:{uint64Hex (fnv1a64 vocabularyPayloadJson.compress.toUTF8)}"
  if !schemaOk then
    ok := false
    issues := issues.push "vocabulary schema identifier/version missing or changed"
  if !fingerprintOk then
    ok := false
    issues := issues.push "vocabulary fingerprint is not deterministic labelled FNV-1a/64"
  -- The served room contract is deliberately signal-capable and glided on every
  -- numeric control. This direct table assertion plus the payload hash above pins
  -- the metadata clients actually receive; a decoder-only implementation cannot
  -- make a wire compile while leaving the public vocabulary knob-only.
  let roomControlNames := #["rt60", "dir", "sway", "rate"]
  let roomSurfaceOk := roomControlNames.all fun name =>
    match portOf "reverb" name with
    | some spec => spec.accepts == sigIn && spec.knob.isSome && spec.discipline == .glide
    | none => false
  let roomInputOk := match portOf "reverb" "in" with
    | some spec => spec.accepts == modalIn && spec.knob.isNone
    | none => false
  if !roomSurfaceOk || !roomInputOk then
    ok := false
    issues := issues.push "reverb: served rt60/dir/sway/rate are not signal-capable glided controls over a modal input"
  for kind in vocabularyKinds do
    if kind == "out" then continue
    let mut nodes : Array Lean.Json := #[]
    let mut inFields : List (String × Lean.Json) := []
    for p in portSpecs kind do
      if !p.accepts.isEmpty && p.name == "in" then
        if p.accepts == #[PortDomain.modal] then
          nodes := nodes.push (Lean.Json.mkObj
            [("id", .str "helper_m"), ("kind", .str "resonator"), ("params", Lean.Json.mkObj [])])
          inFields := inFields ++ [("in", Lean.Json.arr #[.str "helper_m"])]
        else
          nodes := nodes.push (Lean.Json.mkObj
            [("id", .str "helper_s"), ("kind", .str "source"), ("params", Lean.Json.mkObj [])])
          inFields := inFields ++ [("in", Lean.Json.arr #[.str "helper_s"])]
    nodes := nodes.push (Lean.Json.mkObj
      [("id", .str "dut"), ("kind", .str kind), ("params", Lean.Json.mkObj []),
       ("in", Lean.Json.mkObj inFields)])
    -- a control-outlet kind (a bare Knob) drives nothing by itself — its
    -- natural minimal patch consumes it through a source's control inlet, so
    -- the patch has a generator (and the master-clock slots have a reader).
    if outletOf kind == some .control then
      nodes := nodes.push (Lean.Json.mkObj
        [("id", .str "consumer"), ("kind", .str "source"), ("params", Lean.Json.mkObj []),
         ("in", Lean.Json.mkObj [("freq", Lean.Json.arr #[.str "dut"])])])
      nodes := nodes.push (Lean.Json.mkObj
        [("id", .str "outn"), ("kind", .str "out"),
         ("in", Lean.Json.mkObj [("in", Lean.Json.arr #[.str "consumer"])])])
    else
      nodes := nodes.push (Lean.Json.mkObj
        [("id", .str "outn"), ("kind", .str "out"),
         ("in", Lean.Json.mkObj [("in", Lean.Json.arr #[.str "dut"])])])
    let patch := Lean.Json.mkObj [("nodes", Lean.Json.arr nodes), ("out", .str "outn")]
    match Tropical.Playground.compilePlanPure arena resolved patch with
    | .error e => ok := false; issues := issues.push s!"{kind}: compile: {firstLine e}"
    | .ok compiled =>
      let plan := compiled.plan
      covered := covered + 1
      -- every table knob must land as a slot: raw/anchor knobs under the bare
      -- name, glided knobs as their #v0 anchor triple.
      for p in portSpecs kind do
        if p.knob.isSome then
          let base := s!"param:dut.{p.name}"
          if !(plan.slotNames.any (fun s => s == base || s == s!"{base}#v0")) then
            ok := false; issues := issues.push s!"{kind}: {base} not registered"
      let dead := unreadParamSlots plan
      if !dead.isEmpty then
        ok := false; issues := issues.push s!"{kind}: unread {dead}"
  IO.println s!"        schema {vocabularySchema} v{vocabularySchemaVersion} · fingerprint {vocabularyFingerprint}"
  IO.println s!"        {covered} non-sink served kinds compiled from vocabulary-generated minimal patches; room controls signal+glide={roomSurfaceOk}; all served identifiers unique, withheld absent:"
  IO.println s!"        result   {if issues.isEmpty then "every declared knob registers and is read" else toString issues}"
  if ok then
    passGate "vocab-driven" "versioned/fingerprinted vocabulary serves all four reverb controls as signal-capable glided knobs and drives one compiling patch per kind — declared knobs live, nothing unread"
  else
    failGate "vocab-driven" s!"{issues}"

/-- THE MODAL-CLASSIFICATION AGREEMENT gate (Finding 2). The connection-typing
    rule is decided at TWO independent sites: `checkEdgeTypes` colors an outlet
    through `outletOf`, the lowering decides modal-ness through `nodeIsModal` on
    the constructed node. Nothing proves they agree, so a new modal kind missing
    an `outletOf` case would be silently signal-colored (the checker rejecting
    wiring the lowering accepts, or the reverse). Driven off `buildNodeKinds` (every
    kind `buildNode` constructs, not just the served table), so a served-but-unlisted
    kind is SEEN. Asserts two properties: (1) no SERVED kind drifts —
    `nodeIsModal(default node) ⟺ outletOf(kind) == some .modal` — one meaning for
    "modal"; a WITHHELD kind may drift (that drift is precisely why it is withheld,
    and `checkServedKinds` rejects it pre-lowering, so it cannot mis-type an edge);
    (2) the three kind lists are mutually consistent — every served (non-`out`) and
    withheld kind is BUILT, and every built kind is served-or-withheld — so
    `buildNode` cannot grow a kind the vocabulary/withholding machinery misses. -/
def runModalClassAgreement : IO Bool := do
  let vocab    := Tropical.Playground.vocabularyKinds
  let built    := Tropical.Playground.buildNodeKinds
  let withheld := Tropical.Playground.withheldKinds
  let drift    := Tropical.Playground.modalClassificationDrift          -- over `built`
  let servedDrift   := drift.filter (fun k => !withheld.contains k)
  let withheldDrift := drift.filter (fun k => withheld.contains k)
  -- list-consistency: served/withheld ⊆ built, built ⊆ served ∪ withheld
  let servedNotBuilt   := (vocab.filter (· != "out")).filter (fun k => !built.contains k)
  let withheldNotBuilt := withheld.filter (fun k => !built.contains k)
  let builtUnaccounted := built.filter (fun k => !vocab.contains k && !withheld.contains k)
  let setIssues := servedNotBuilt.map (s!"served-not-built {·}")
    ++ withheldNotBuilt.map (s!"withheld-not-built {·}")
    ++ builtUnaccounted.map (s!"built-unaccounted {·}")
  IO.println s!"        {built.size} built kinds: nodeIsModal(node) ⟺ outletOf == modal (served drift asserted empty)"
  IO.println s!"        served drift {servedDrift} · withheld drift (contained, rejected pre-lowering) {withheldDrift} · list-consistency {if setIssues.isEmpty then "ok" else toString setIssues}"
  if servedDrift.isEmpty && setIssues.isEmpty then
    passGate "modal-class-agreement" s!"outletOf and nodeIsModal agree for every SERVED kind, and buildNodeKinds ⊆ served∪withheld (the {withheldDrift.size} withheld kind(s) may drift — rejected pre-lowering, so the drift cannot mis-type an edge)"
  else
    failGate "modal-class-agreement" s!"servedDrift {servedDrift} · setIssues {setIssues}"

open Tropical.Playground in
/-- Ordinary module inputs are variadic at the surface but remain explicitly
    typed in the lowering. Signal ports synthesize an authored-order `.mix`,
    modal ports synthesize an authored-order `.modalMix`, singleton inputs keep
    their historical graph, and non-`in` controls remain exclusive. -/
def runImplicitFanIn : IO Bool := do
  let empty := Lean.Json.mkObj []
  let pair := Lean.Json.mkObj
    [("in", Lean.Json.arr #[.str "a", .str "b"])]
  let single := Lean.Json.mkObj
    [("in", Lean.Json.arr #[.str "a"])]
  let noParams : String → Option Nat := fun _ => none

  let (delayNode, delayExtras) :=
    buildNode noParams "delayN" "delay" empty empty pair
  let signalHelperId := "__fanin_signal_delayN_in"
  let signalNodeOk := match delayNode with
    | .warpFx input _ => input == signalHelperId
    | _ => false
  let signalHelperOk := match delayExtras[0]? with
    | some helper => helper.id == signalHelperId && match helper.node with
      | .mix inputs => inputs == #["a", "b"]
      | _ => false
    | none => false

  let (phaserNode, phaserExtras) :=
    buildNode noParams "phaserN" "phaser" empty empty pair
  let modalHelperId := "__fanin_modal_phaserN_in"
  let modalNodeOk := match phaserNode with
    | .modalPhaser input _ _ _ _ _ => input == modalHelperId
    | _ => false
  let modalHelperOk := match phaserExtras[0]? with
    | some helper => helper.id == modalHelperId && match helper.node with
      | .modalMix inputs => inputs == #["a", "b"]
      | _ => false
    | none => false

  let (singleNode, singleExtras) :=
    buildNode noParams "singleN" "delay" empty empty single
  let singletonOk := singleExtras.isEmpty && match singleNode with
    | .warpFx input _ => input == "a"
    | _ => false

  let implicitKinds :=
    #["comb", "flange", "sflange", "fm", "delay", "reverse",
      "reverb", "filter", "phaser", "gauge"]
  let servedMultiOk := implicitKinds.all fun kind =>
    match portOf kind "in" with
    | some spec => spec.multi
    | none => false
  let exclusivePorts := #[
    ("source", "pm"), ("sflange", "mod"), ("resonator", "addr"),
    ("reverb", "rt60"), ("phaser", "center")]
  let exclusiveOk := exclusivePorts.all fun (kind, port) =>
    match portOf kind port with
    | some spec => !spec.multi
    | none => false

  let duplicateControl := Lean.Json.mkObj [("nodes", Lean.Json.arr #[
    Lean.Json.mkObj [("id", .str "a"), ("kind", .str "source")],
    Lean.Json.mkObj [("id", .str "b"), ("kind", .str "source")],
    Lean.Json.mkObj [("id", .str "res"), ("kind", .str "resonator")],
    Lean.Json.mkObj [("id", .str "room"), ("kind", .str "reverb"),
      ("in", Lean.Json.mkObj [
        ("in", Lean.Json.arr #[.str "res"]),
        ("rt60", Lean.Json.arr #[.str "a", .str "b"])])]])]
  let exclusiveRejected := match checkEdgeTypes (rawsOf duplicateControl) with
    | .error e => (e.splitOn "accepts one source").length > 1
    | .ok _ => false

  let ok := signalNodeOk && signalHelperOk && modalNodeOk && modalHelperOk &&
    singletonOk && servedMultiOk && exclusiveOk && exclusiveRejected
  IO.println s!"        signal helper={signalNodeOk && signalHelperOk} · modal helper={modalNodeOk && modalHelperOk} · singleton identity={singletonOk}"
  IO.println s!"        ordinary inlets multi={servedMultiOk} · control/address/modulation ports exclusive={exclusiveOk}/{exclusiveRejected}"
  if ok then
    passGate "implicit-fan-in" "ordinary signal/modal inlets synthesize ordered typed fan-in; singleton graphs stay identical; controls, addresses, and modulators remain exclusive"
  else
    failGate "implicit-fan-in" s!"signal={signalNodeOk}/{signalHelperOk} modal={modalNodeOk}/{modalHelperOk} singleton={singletonOk} servedMulti={servedMultiOk} exclusive={exclusiveOk}/{exclusiveRejected}"

/-- THE MALFORMED-DOCUMENT REJECTION gate (Findings 1 & 3). A malformed patch is a
    BROKEN document — distinct from a legal-incomplete one (an unwired inlet →
    silence, no error). These malformations must each return a clear `Except.error`
    and, critically for the live MCP server, must NOT crash the process:
      · a color-LEGAL CYCLE — a reverb whose modal outlet feeds its own modal
        inlet, a mix fed by itself — passes `checkEdgeTypes` (the colors match);
        `lowerGraph`'s own topo-rank entry refuses it with the loop named (the
        lowering is total now — no stack to overflow);
      · a WIRE naming no node (a typo'd source) — previously died downstream as
        `lower: node '…' not found`, or vanished silently if unreferenced;
      · a top-level `"out"` naming no node — previously rendered the WHOLE patch as
        silence with no error;
      · a cycle entering a room through a signal-capable numeric control — control
        references are graph dependencies, not side channels around cycle refusal.
    The boundary stays crisp: a valid modal patch still compiles. -/
def runMalformedRejection (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let hasSub := fun (s sub : String) => (s.splitOn sub).length != 1
  let compileErr := fun (src : String) =>
    match Lean.Json.parse src with
    | .error e => s!"json parse: {e}"
    | .ok j => match Tropical.Playground.compilePlanPure arena resolved j with
      | .error e => firstLine e
      | .ok _ => "(compiled — should have been rejected)"
  -- color-legal reverb self-edge: modal outlet → modal inlet, but a cycle.
  let cycleReverb := "{\"nodes\":[" ++
    "{\"id\":\"rev\",\"kind\":\"reverb\",\"params\":{\"rt60\":2},\"in\":{\"in\":[\"rev\"]}}]," ++
    "\"out\":\"rev\"}"
  -- color-legal mix self-loop: signal outlet → signal inlet, but a cycle.
  let cycleMix := "{\"nodes\":[" ++
    "{\"id\":\"m\",\"kind\":\"mix\",\"in\":{\"in\":[\"m\"]}}],\"out\":\"m\"}"
  -- a wire naming no node.
  let danglingWire := "{\"nodes\":[" ++
    "{\"id\":\"osc\",\"kind\":\"source\",\"params\":{\"freq\":220}}," ++
    "{\"id\":\"mx\",\"kind\":\"mix\",\"in\":{\"in\":[\"osc\",\"ghost\"]}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"mx\"]}}],\"out\":\"out\"}"
  -- a top-level out naming no node (every wire otherwise valid).
  let danglingOut := "{\"nodes\":[" ++
    "{\"id\":\"osc\",\"kind\":\"source\",\"params\":{\"freq\":220}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"osc\"]}}],\"out\":\"typo\"}"
  -- a WITHHELD kind (built by buildNode, not surface-served): its `bloomgong`→out
  -- edge is color-legal by luck (modal→signal realizes), so `checkServedKinds`
  -- must reject it FIRST, honestly — not let it reach the unguarded factor site.
  let withheldKind := "{\"nodes\":[" ++
    "{\"id\":\"bg\",\"kind\":\"bloomgong\",\"params\":{\"freq\":110}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"bg\"]}}],\"out\":\"out\"}"
  -- an UNKNOWN kind (a typo): otherwise silent silence under buildNode's fallthrough.
  let unknownKind := "{\"nodes\":[" ++
    "{\"id\":\"x\",\"kind\":\"reverbb\",\"params\":{\"rt60\":2}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"x\"]}}],\"out\":\"out\"}"
  -- A room numeric control is now a real signal inlet with a knob fallback. This
  -- exact graph used to be rejected as a wire into a set-only knob; it must compile.
  let wiredRoomControl := "{\"nodes\":[" ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4}}," ++
    "{\"id\":\"k\",\"kind\":\"knob\",\"params\":{\"value\":3}}," ++
    "{\"id\":\"rev\",\"kind\":\"reverb\",\"in\":{\"in\":[\"res\"],\"rt60\":[\"k\"]}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"rev\"]}}],\"out\":\"out\"}"
  -- The same signal-capable port must participate in the ordinary graph cycle
  -- check. A room cannot source the control that selects its own universe.
  let cycleRoomControl := "{\"nodes\":[" ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4}}," ++
    "{\"id\":\"rev\",\"kind\":\"reverb\",\"in\":{\"in\":[\"res\"],\"rt60\":[\"rev\"]}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"rev\"]}}],\"out\":\"out\"}"
  -- the crisp boundary: a valid modal patch still compiles.
  let valid := "{\"nodes\":[" ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4}}," ++
    "{\"id\":\"rev\",\"kind\":\"reverb\",\"params\":{\"rt60\":2},\"in\":{\"in\":[\"res\"]}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"rev\"]}}],\"out\":\"out\"}"
  let mRev := compileErr cycleReverb
  let mMix := compileErr cycleMix
  let mWire := compileErr danglingWire
  let mOut := compileErr danglingOut
  let mWithheld := compileErr withheldKind
  let mUnknown := compileErr unknownKind
  let mControlCycle := compileErr cycleRoomControl
  let wiredControlOk := match Lean.Json.parse wiredRoomControl with
    | .ok j => (Tropical.Playground.compilePlanPure arena resolved j).toOption.isSome
    | .error _ => false
  let mValid := match Lean.Json.parse valid with
    | .ok j => (Tropical.Playground.compilePlanPure arena resolved j).toOption.isSome
    | .error _ => false
  let revOk := hasSub mRev "connection cycle"
  let mixOk := hasSub mMix "connection cycle"
  let wireOk := hasSub mWire "not a node in the patch"
  let outOk := hasSub mOut "output target error"
  let withheldOk := hasSub mWithheld "unserved kind"
  let unknownOk := hasSub mUnknown "unknown kind"
  let controlCycleOk := hasSub mControlCycle "connection cycle"
  IO.println s!"        malformed patches rejected (no crash); legal-incomplete unaffected:"
  IO.println s!"        cycle(reverb self)   {mRev}"
  IO.println s!"        cycle(mix self)      {mMix}"
  IO.println s!"        dangling wire        {mWire}"
  IO.println s!"        dangling out         {mOut}"
  IO.println s!"        withheld kind        {mWithheld}"
  IO.println s!"        unknown kind         {mUnknown}"
  IO.println s!"        cycle(room.rt60)     {mControlCycle}"
  IO.println s!"        wired room.rt60 compiles={wiredControlOk}"
  IO.println s!"        valid modal patch compiles={mValid}"
  if revOk && mixOk && wireOk && outOk && withheldOk && unknownOk &&
      controlCycleOk && wiredControlOk && mValid then
    passGate "malformed-rejection" "ordinary and room-control cycles, a dangling wire/out, a WITHHELD kind, and an UNKNOWN kind return clear errors; wiring room.rt60 is valid and compiles"
  else
    failGate "malformed-rejection" s!"revOk={revOk} mixOk={mixOk} wireOk={wireOk} outOk={outOk} withheldOk={withheldOk} unknownOk={unknownOk} controlCycle={controlCycleOk} wiredControl={wiredControlOk} valid={mValid}"

open Tropical.Playground in
/-- THE REALIZED-STATE REPORT gate. The `load_patch_graph` reply must state
    FACTS a surface can render — wired vs normalled inputs, live params with
    disciplines, excluded nodes — and never a warning (house contract: legal-
    but-incomplete compiles silently; the report tells, it does not scold).
    This is the protocol-level net for the silence-with-`{ok:true}` class. -/
def runRealizedReport : IO Bool := do
  let mkPatch := fun (withMod : Bool) (dangler : Bool) =>
    let base := "{\"nodes\":[" ++
      "{\"id\":\"osc\",\"kind\":\"source\",\"params\":{\"freq\":220}}," ++
      (if withMod then "{\"id\":\"lfo\",\"kind\":\"source\",\"params\":{\"freq\":0.4}}," else "") ++
      (if dangler then "{\"id\":\"orphan\",\"kind\":\"source\",\"params\":{\"freq\":99}}," else "") ++
      "{\"id\":\"sfw\",\"kind\":\"sflange\",\"params\":{\"depth\":0.002,\"rate\":0.3},\"in\":{\"in\":[\"osc\"]" ++
      (if withMod then ",\"mod\":[\"lfo\"]" else "") ++ "}}," ++
      "{\"id\":\"outn\",\"kind\":\"out\",\"in\":{\"in\":[\"sfw\"]}}],\"out\":\"outn\"}"
    base
  let mkRoomPatch := fun (wiredRt60 : Bool) => "{\"nodes\":[" ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4}}," ++
    (if wiredRt60 then "{\"id\":\"k\",\"kind\":\"knob\",\"params\":{\"value\":3}}," else "") ++
    "{\"id\":\"room\",\"kind\":\"reverb\",\"params\":{\"rt60\":2,\"dir\":0,\"sway\":0,\"rate\":0.3},\"in\":{\"in\":[\"res\"]" ++
    (if wiredRt60 then ",\"rt60\":[\"k\"]" else "") ++ "}}," ++
    "{\"id\":\"outn\",\"kind\":\"out\",\"in\":{\"in\":[\"room\"]}}],\"out\":\"outn\"}"
  let inputState := fun (rep : Lean.Json) (node port : String) =>
    match rep.getObjVal? "inputs" with
    | .ok (.arr a) => (a.find? fun ij =>
        ((ij.getObjVal? "node").toOption.bind (·.getStr?.toOption)) == some node &&
        ((ij.getObjVal? "port").toOption.bind (·.getStr?.toOption)) == some port).bind
        fun ij => (ij.getObjVal? "state").toOption.bind (·.getStr?.toOption)
    | _ => none
  let paramNames := fun (rep : Lean.Json) =>
    match rep.getObjVal? "params" with
    | .ok (.arr a) => a.filterMap fun (pj : Lean.Json) =>
        (pj.getObjVal? "name").toOption.bind (·.getStr?.toOption)
    | _ => #[]
  let nodeStatus := fun (rep : Lean.Json) (id : String) =>
    match rep.getObjVal? "nodes" with
    | .ok (.arr a) => (a.find? fun nj =>
        ((nj.getObjVal? "id").toOption.bind (·.getStr?.toOption)) == some id).bind
        fun nj => (nj.getObjVal? "status").toOption.bind (·.getStr?.toOption)
    | _ => none
  let fingerprintOf := fun (rep : Lean.Json) =>
    (rep.getObjVal? "vocabulary_fingerprint").toOption.bind (·.getStr?.toOption)
  match Lean.Json.parse (mkPatch false false), Lean.Json.parse (mkPatch true false),
        Lean.Json.parse (mkPatch false true), Lean.Json.parse (mkRoomPatch false),
        Lean.Json.parse (mkRoomPatch true) with
  | .ok jUnwired, .ok jWired, .ok jDangler, .ok jRoomU, .ok jRoomW =>
    let repU := realizedReport jUnwired #[]
    let repW := realizedReport jWired #[]
    let repD := realizedReport jDangler #[]
    let repRoomU := realizedReport jRoomU #[]
    let repRoomW := realizedReport jRoomW #[]
    let unwiredOk := inputState repU "sfw" "mod" == some "normalled"
      && (paramNames repU).contains "sfw.rate"
    let wiredOk := inputState repW "sfw" "mod" == some "wired"
      && !(paramNames repW).contains "sfw.rate"
    let danglerOk := nodeStatus repD "orphan" == some "excluded"
      && nodeStatus repD "osc" == some "active"
    let roomControls := #["rt60", "dir", "sway", "rate"]
    let roomUnwiredOk := roomControls.all fun name =>
      inputState repRoomU "room" name == some "normalled" &&
        (paramNames repRoomU).contains s!"room.{name}"
    let roomWiredOk := inputState repRoomW "room" "rt60" == some "wired" &&
      !(paramNames repRoomW).contains "room.rt60" &&
      (paramNames repRoomW).contains "k.value" &&
      #["dir", "sway", "rate"].all fun name =>
        inputState repRoomW "room" name == some "normalled" &&
          (paramNames repRoomW).contains s!"room.{name}"
    let reportFingerprintOk := #[repU, repW, repD, repRoomU, repRoomW].all fun rep =>
      fingerprintOf rep == some vocabularyFingerprint
    -- the no-warnings contract, checked on the wire form itself
    let reports := repU.compress ++ repW.compress ++ repD.compress ++
      repRoomU.compress ++ repRoomW.compress
    let noWarnOk := (reports.toLower.splitOn "warn").length == 1
    IO.println s!"        unwired: mod={inputState repU "sfw" "mod"} rate-param={((paramNames repU).contains "sfw.rate")} · wired: mod={inputState repW "sfw" "mod"} rate-param={((paramNames repW).contains "sfw.rate")} · room={roomUnwiredOk}/{roomWiredOk} · fingerprint={reportFingerprintOk} · orphan={nodeStatus repD "orphan"}"
    if unwiredOk && wiredOk && danglerOk && roomUnwiredOk && roomWiredOk &&
        reportFingerprintOk && noWarnOk then
      passGate "realized-report" "facts, not warnings: room controls report normalled/wired truth, wired fallbacks disappear, every report carries the authoritative vocabulary fingerprint, and excluded nodes are named"
    else
      failGate "realized-report" s!"unwired={unwiredOk} wired={wiredOk} room={roomUnwiredOk}/{roomWiredOk} fingerprint={reportFingerprintOk} dangler={danglerOk} noWarn={noWarnOk}"
  | _, _, _, _, _ => failGate "realized-report" "patch json parse"

open Tropical.Playground in
/-- THE MANIFEST-DISCIPLINE gate. `param_disciplines` is host-contract data:
    every entry must be consistent with the slots the same plan carries (glide
    companions exist and the base slot doesn't; anchor/raw base slots exist;
    the velocity entry names its tau_base companion), and a knob superseded by
    a wired normal must be absent from the table exactly as it is from the
    slots. A host dispatching from this table can then trust it blind. -/
def runManifestDisciplines (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let patch := fun (withMod : Bool) => "{\"nodes\":[" ++
    "{\"id\":\"osc\",\"kind\":\"source\",\"params\":{\"freq\":220}}," ++
    (if withMod then "{\"id\":\"lfo\",\"kind\":\"source\",\"params\":{\"freq\":0.4}}," else "") ++
    "{\"id\":\"sfw\",\"kind\":\"sflange\",\"params\":{\"depth\":0.002,\"rate\":0.3},\"in\":{\"in\":[\"osc\"]" ++
    (if withMod then ",\"mod\":[\"lfo\"]" else "") ++ "}}," ++
    "{\"id\":\"outn\",\"kind\":\"out\",\"in\":{\"in\":[\"sfw\"]}}],\"out\":\"outn\"}"
  let roomPatch := fun (wired : Bool) => "{\"nodes\":[" ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4}}," ++
    (if wired then "{\"id\":\"driver\",\"kind\":\"knob\",\"params\":{\"value\":1}}," else "") ++
    "{\"id\":\"room\",\"kind\":\"reverb\",\"params\":{\"rt60\":2,\"dir\":0,\"sway\":0,\"rate\":0.3},\"in\":{\"in\":[\"res\"]" ++
    (if wired then ",\"rt60\":[\"driver\"],\"dir\":[\"driver\"],\"sway\":[\"driver\"],\"rate\":[\"driver\"]" else "") ++ "}}," ++
    "{\"id\":\"outn\",\"kind\":\"out\",\"in\":{\"in\":[\"room\"]}}],\"out\":\"outn\"}"
  let check := fun (label : String) (plan : Tropical.Plan.FlatPlan) => Id.run do
    let mut issues : Array String := #[]
    let names := plan.slotNames
    for d in plan.paramDisciplines do
      let base := s!"param:{d.name}"
      for c in d.companions do
        if !names.contains s!"param:{c}" then
          issues := issues.push s!"{label}: {d.name} companion {c} has no slot"
      match d.discipline with
      | "glide" =>
        if names.contains base then
          issues := issues.push s!"{label}: glided {d.name} has a base slot (companions are the value)"
        if d.glideDurSec.isNone then
          issues := issues.push s!"{label}: glided {d.name} missing glide_dur_sec"
      | "raw" | "anchor" | "velocity" =>
        if !names.contains base then
          issues := issues.push s!"{label}: {d.discipline} {d.name} has no base slot"
      | other => issues := issues.push s!"{label}: unknown discipline {other}"
    return issues
  match Lean.Json.parse (patch false), Lean.Json.parse (patch true),
        Lean.Json.parse (roomPatch false), Lean.Json.parse (roomPatch true) with
  | .ok ju, .ok jw, .ok jru, .ok jrw =>
    match Tropical.Playground.compilePlanPure arena resolved ju,
          Tropical.Playground.compilePlanPure arena resolved jw,
          Tropical.Playground.compilePlanPure arena resolved jru,
          Tropical.Playground.compilePlanPure arena resolved jrw with
    | .ok compiledU, .ok compiledW, .ok compiledRoomU, .ok compiledRoomW =>
      let pu := compiledU.plan
      let pw := compiledW.plan
      let pru := compiledRoomU.plan
      let prw := compiledRoomW.plan
      let mut issues := check "unwired" pu ++ check "wired" pw ++
        check "room-unwired" pru ++ check "room-wired" prw
      if !(pu.paramDisciplines.any (·.name == "sfw.rate")) then
        issues := issues.push "unwired: sfw.rate missing from disciplines"
      if pw.paramDisciplines.any (·.name == "sfw.rate") then
        issues := issues.push "wired: superseded sfw.rate present in disciplines"
      for name in #["rt60", "dir", "sway", "rate"] do
        if !(pru.paramDisciplines.any fun d =>
            d.name == s!"room.{name}" && d.discipline == "glide") then
          issues := issues.push s!"room-unwired: room.{name} missing glide discipline"
        if prw.paramDisciplines.any (·.name == s!"room.{name}") then
          issues := issues.push s!"room-wired: superseded room.{name} present in disciplines"
      if !(prw.paramDisciplines.any fun d =>
          d.name == "driver.value" && d.discipline == "raw") then
        issues := issues.push "room-wired: driver.value missing raw discipline"
      if !(pu.paramDisciplines.any fun d => d.name == "master.velocity"
            && d.discipline == "velocity" && d.companions.contains "master.tau_base") then
        issues := issues.push "master.velocity entry wrong"
      IO.println s!"        {pu.paramDisciplines.size}+{pw.paramDisciplines.size}+{pru.paramDisciplines.size}+{prw.paramDisciplines.size} manifest entries checked against their plans' slots:"
      IO.println s!"        result   {if issues.isEmpty then "consistent" else toString issues}"
      if issues.isEmpty then
        passGate "manifest-disciplines" "param_disciplines ≡ the plan's slots; all four unwired room controls are glided and wiring them removes their fallback disciplines"
      else
        failGate "manifest-disciplines" s!"{issues}"
    | .error e, _, _, _ | _, .error e, _, _ | _, _, .error e, _ |
        _, _, _, .error e =>
      failGate "manifest-disciplines" s!"compile: {firstLine e}"
  | _, _, _, _ => failGate "manifest-disciplines" "json parse"

/-- THE DEAD-SLOT LINT gate (the systemic net for the dead-knob class). Canonical
    patches covering every playground node kind compile through the real
    `compilePlanPure`, and every kernel-consumed `param:*` slot each plan registers
    must be READ by some instruction operand. Presence-only checks pass a dead
    knob — the slot exists, `setSlot` succeeds, nothing listens — which is exactly
    how the reverb-drops-the-source's-poles regression stayed invisible;
    unreadness in the PLAN is the property that catches the whole class, whatever
    node grows it next. The exact-glide scalar `#t0` exception is derived narrowly
    from the manifest by `unreadParamSlots`, not blanket-allowlisted here. -/
def runDeadSlotLint (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  -- Allowlist for any further slots proven legitimately unread, one justified
  -- entry at a time (`<patch-label>:<slot>`); a blanket param exclusion would
  -- re-open the hole the gate exists to close. Currently empty.
  let allow : Array String := #[]
  -- Every buildNode kind: knob, source, pluck, comb, flange, sflange, fm,
  -- delay, reverse, mix, ring, resonator, reverb, modalmix, out. The top-level
  -- "out" field is mandatory — without it the patch gracefully compiles to
  -- silence and EVERY knob goes dead (which this gate would report).
  let signalChain := "{\"nodes\":[" ++
    "{\"id\":\"k\",\"kind\":\"knob\",\"params\":{\"value\":110}}," ++
    "{\"id\":\"osc\",\"kind\":\"source\",\"params\":{\"morph\":0.2},\"in\":{\"freq\":[\"k\"]}}," ++
    "{\"id\":\"fl\",\"kind\":\"flange\",\"params\":{\"depth\":0.0007},\"in\":{\"in\":[\"osc\"]}}," ++
    "{\"id\":\"sf\",\"kind\":\"sflange\",\"params\":{\"depth\":0.002,\"rate\":0.3},\"in\":{\"in\":[\"fl\"]}}," ++
    "{\"id\":\"fmn\",\"kind\":\"fm\",\"params\":{\"carrier\":330,\"depth\":8},\"in\":{\"in\":[\"sf\"]}}," ++
    "{\"id\":\"dl\",\"kind\":\"delay\",\"params\":{\"amount\":0.004},\"in\":{\"in\":[\"fmn\"]}}," ++
    "{\"id\":\"rv\",\"kind\":\"reverse\",\"in\":{\"in\":[\"dl\"]}}," ++
    "{\"id\":\"osc2\",\"kind\":\"source\",\"params\":{\"freq\":330}}," ++
    "{\"id\":\"rg\",\"kind\":\"ring\",\"in\":{\"in\":[\"rv\",\"osc2\"]}}," ++
    "{\"id\":\"pl\",\"kind\":\"pluck\",\"params\":{\"freq\":110}}," ++
    "{\"id\":\"cb\",\"kind\":\"comb\",\"params\":{\"delay\":0.012,\"decay\":0.7},\"in\":{\"in\":[\"pl\"]}}," ++
    "{\"id\":\"mx\",\"kind\":\"mix\",\"in\":{\"in\":[\"rg\",\"cb\"]}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"mx\"]}}],\"out\":\"out\"}"
  -- The regression's exact shape: the reverb must keep its source's poles live.
  let modalChain := "{\"nodes\":[" ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4}}," ++
    "{\"id\":\"rvb\",\"kind\":\"reverb\",\"params\":{\"rt60\":2},\"in\":{\"in\":[\"res\"]}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"rvb\"]}}],\"out\":\"out\"}"
  let modalMix := "{\"nodes\":[" ++
    "{\"id\":\"res1\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4}}," ++
    "{\"id\":\"res2\",\"kind\":\"resonator\",\"params\":{\"freq\":330,\"decay\":3}}," ++
    "{\"id\":\"mm\",\"kind\":\"modalmix\",\"in\":{\"in\":[\"res1\",\"res2\"]}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"mm\"]}}],\"out\":\"out\"}"
  -- The once-KNOWN-HOLE, now in the canonical set: `sflange` with a `mod` cord
  -- patched. `rate` parameterizes `mod`'s normalled LFO (ownerPort in the
  -- port-spec table), so wiring `mod` removes the LFO, the knob, and the slot
  -- together — the lint additionally asserts the slot's ABSENCE below, so the
  -- old dead-knob encoding (registered-but-unread) can't quietly return.
  let sflangeWired := "{\"nodes\":[" ++
    "{\"id\":\"osc\",\"kind\":\"source\",\"params\":{\"freq\":220}}," ++
    "{\"id\":\"lfo\",\"kind\":\"source\",\"params\":{\"freq\":0.4,\"morph\":1}}," ++
    "{\"id\":\"sfw\",\"kind\":\"sflange\",\"params\":{\"depth\":0.002,\"rate\":0.3},\"in\":{\"in\":[\"osc\"],\"mod\":[\"lfo\"]}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"sfw\"]}}],\"out\":\"out\"}"
  -- Four independent drivers make each terminal room-control binding observable:
  -- if any one is dropped, its source knob becomes an unread param. The room's own
  -- four glided fallbacks must be absent, not merely registered-but-unused.
  let reverbWired := "{\"nodes\":[" ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4}}," ++
    "{\"id\":\"krt\",\"kind\":\"knob\",\"params\":{\"value\":2}}," ++
    "{\"id\":\"kdir\",\"kind\":\"knob\",\"params\":{\"value\":0.5}}," ++
    "{\"id\":\"ksway\",\"kind\":\"knob\",\"params\":{\"value\":0.2}}," ++
    "{\"id\":\"krate\",\"kind\":\"knob\",\"params\":{\"value\":0.3}}," ++
    "{\"id\":\"rvw\",\"kind\":\"reverb\",\"in\":{\"in\":[\"res\"],\"rt60\":[\"krt\"],\"dir\":[\"kdir\"],\"sway\":[\"ksway\"],\"rate\":[\"krate\"]}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"rvw\"]}}],\"out\":\"out\"}"
  let patches : Array (String × String) := #[
    ("signal-chain", signalChain), ("modal-chain", modalChain), ("modal-mix", modalMix),
    ("sflange-wired", sflangeWired), ("reverb-wired", reverbWired)]
  let mut ok := true
  let mut checked := 0
  let mut deadAll : Array String := #[]
  for (label, src) in patches do
    match Lean.Json.parse src with
    | .error e => IO.println s!"  FAIL  dead-slot-lint  {label}: json parse: {e}"; ok := false
    | .ok j =>
      match Tropical.Playground.compilePlanPure arena resolved j with
      | .error e => IO.println s!"  FAIL  dead-slot-lint  {label}: compile: {firstLine e}"; ok := false
      | .ok compiled =>
        let plan := compiled.plan
        -- Byte-identity harness for refactor phases: TROPICAL_DUMP_PLANS=<dir>
        -- writes each canonical plan's wire form for before/after comparison
        -- (a refactor that promises plan identity proves it with `cmp`).
        if let some dir := (← IO.getEnv "TROPICAL_DUMP_PLANS") then
          if let .ok m := plan.toWire then
            IO.FS.writeFile s!"{dir}/{label}.json" m.compress
        let nParams := (plan.slotNames.filter ("param:".isPrefixOf ·)).size
        -- zero registered params means the patch didn't decode as intended —
        -- vacuous passes are the graceful-exclusion failure mode.
        if nParams == 0 then
          IO.println s!"  FAIL  dead-slot-lint  {label}: no param:* slots registered — the patch decoded to nothing"; ok := false
        checked := checked + nParams
        let dead := (unreadParamSlots plan).filter (fun s => !allow.contains s!"{label}:{s}")
        if !dead.isEmpty then
          deadAll := deadAll ++ dead.map (s!"{label}: {·}")
          ok := false
        -- the sflange fix, asserted structurally: a knob owned by a wired
        -- normal must not even REGISTER (absence, not just unreadness).
        if label == "sflange-wired" && plan.slotNames.contains "param:sfw.rate" then
          IO.println s!"  FAIL  dead-slot-lint  {label}: param:sfw.rate registered despite wired mod — the owned-knob rule regressed"
          ok := false
        if label == "modal-chain" then
          for control in #["rt60", "dir", "sway", "rate"] do
            if !plan.slotNames.contains s!"param:rvb.{control}#v0" then
              IO.println s!"  FAIL  dead-slot-lint  {label}: unwired rvb.{control} glide fallback missing"
              ok := false
        if label == "reverb-wired" then
          for control in #["rt60", "dir", "sway", "rate"] do
            let slotPrefix := s!"param:rvw.{control}"
            if plan.slotNames.any (slotPrefix.isPrefixOf ·) then
              IO.println s!"  FAIL  dead-slot-lint  {label}: wired {slotPrefix} fallback still registered"
              ok := false
          for driver in #["krt", "kdir", "ksway", "krate"] do
            if !plan.slotNames.contains s!"param:{driver}.value" then
              IO.println s!"  FAIL  dead-slot-lint  {label}: param:{driver}.value missing"
              ok := false
  IO.println s!"        {patches.size} canonical patches over the full node vocabulary, {checked} param:* slots:"
  IO.println s!"        result   unread param slots: {if deadAll.isEmpty then "none" else toString deadAll}"
  if ok then
    passGate "dead-slot-lint" "every registered param:* slot is read; unwired room glides stay live, while four independently wired room controls suppress their fallbacks and consume all four drivers"
  else
    failGate "dead-slot-lint" s!"dead knobs (setSlot lands, no instruction reads): {deadAll}"

/-- Test 3: `osc ⋙ flange ⋙ flange` — the slide pushes the outer warps through
    the inner flanger's sum and fuses them, producing the oscillator read at the
    nine convolved offsets automatically (the proper multiplicity, derived). We
    assert the generator count (9 vs 3) and that the cascade is a real, non-silent
    filter distinct from a single flanger. -/
def runSlideCascade (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  match Tropical.EmitArrow.buildSlideDoubleFlanger arena resolved,
        Tropical.EmitArrow.buildSlideSingleFlanger arena resolved with
  | .ok (aD, iD), .ok (aS, iS) =>
    let ninstD := ((aD.program? iD).map (·.decls.size)).getD 0
    let ninstS := ((aS.program? iS).map (·.decls.size)).getD 0
    match buildAndFinish (.ok (aD, iD)), buildAndFinish (.ok (aS, iS)) with
    | .ok planD, .ok planS =>
      match ← renderPlanSamples planD 512, ← renderPlanSamples planS 512 with
      | .ok dbl, .ok sgl =>
        let mut energy : Float := 0.0
        let mut diff : Float := 0.0
        for t in [8:512] do
          energy := energy + dbl[t]! * dbl[t]!
          if (dbl[t]! - sgl[t]!).abs > diff then diff := (dbl[t]! - sgl[t]!).abs
        IO.println s!"        cascade osc ⋙ flange ⋙ flange: {ninstD} generator instances (single flange: {ninstS}); the slide convolved the kernels — 9 = 3⊛3 taps, no coincident-offset merge"
        if ninstD == 9 && ninstS == 3 && energy > 1e-6 && diff > 1e-4 then
          passGate "slide-cascade" s!"9-tap multiplicity derived by the slide (energy={energy}, |double−single|max={diff})"
        else
          failGate "slide-cascade" s!"ninstD={ninstD} (want 9) ninstS={ninstS} (want 3) energy={energy} diff={diff}"
      | .error e, _ | _, .error e => failGate "slide-cascade" s!"render: {firstLine e}"
    | .error e, _ | _, .error e => failGate "slide-cascade" s!"finish: {firstLine e}"
  | .error e, _ | _, .error e => failGate "slide-cascade" s!"build: {firstLine e}"
