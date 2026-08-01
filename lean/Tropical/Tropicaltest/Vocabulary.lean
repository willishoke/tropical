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
    a wiring bug, so the lint demands an instruction read for `param:*`. A hit is
    a dead knob: `setSlot` succeeds, no instruction listens — the class the
    reverb-discards-the-voice's-poles regression shipped in. -/
def unreadParamSlots (plan : Tropical.Plan.FlatPlan) : Array String := Id.run do
  let reads := plan.instanceFunctions.flatMap slotReadsOf
  let mut dead : Array String := #[]
  for i in [0:plan.slotNames.size] do
    let name := plan.slotNames[i]!
    if "param:".isPrefixOf name && !reads.contains i then
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
  for kind in vocabularyKinds do
    if kind == "out" then continue
    let mut nodes : Array Lean.Json := #[]
    let mut inFields : List (String × Lean.Json) := []
    let mut dutParams := Lean.Json.mkObj []
    if kind == "groupedroom" then
      let coords : Array (Nat × Int × Nat) := #[
        (211, 75, 1), (211, 955, 1), (433, 865, 2), (433, 10035, 2),
        (887, 98, 1), (887, 1052, 1), (1511, 1095, 2), (1511, 11005, 2),
        (2837, 121, 1), (2837, 1149, 1), (5081, 1325, 2), (5081, 11975, 2)]
      let modes := Lean.Json.arr (coords.map fun (frequency, sigma, exponent) =>
        Lean.Json.arr #[Lean.toJson frequency, .num ⟨sigma, exponent⟩,
          Lean.toJson (1 : Nat), Lean.toJson (0 : Nat)])
      nodes := nodes.push (Lean.Json.mkObj
        [("id", .str "helper_m"), ("kind", .str "string"),
         ("params", Lean.Json.mkObj [("modes", modes)])])
      nodes := nodes.push (Lean.Json.mkObj
        [("id", .str "helper_p"), ("kind", .str "glideknob"),
         ("params", Lean.Json.mkObj [("value", Lean.toJson (1 : Nat))])])
      inFields := [("in", Lean.Json.arr #[.str "helper_m"]),
        ("position", Lean.Json.arr #[.str "helper_p"])]
      dutParams := Lean.Json.mkObj
        [("profile", .str Tropical.EmitArrow.groupedRoomProfile)]
    else if kind == "groupedroomcache" then
      nodes := nodes.push (Lean.Json.mkObj
        [("id", .str "helper_p"), ("kind", .str "glideknob"),
         ("params", Lean.Json.mkObj [("value", Lean.toJson (1 : Nat))])])
      inFields := [("position", Lean.Json.arr #[.str "helper_p"])]
      dutParams := Lean.Json.mkObj
        [("profile", .str Tropical.EmitArrow.groupedRoomCacheProfile)]
    else
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
      [("id", .str "dut"), ("kind", .str kind), ("params", dutParams),
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
  IO.println s!"        {covered} kinds, each compiled from a vocabulary-generated minimal patch:"
  IO.println s!"        result   {if issues.isEmpty then "every declared knob registers and is read" else toString issues}"
  if ok then
    passGate "vocab-driven" "the served vocabulary drives a compiling patch per kind — declared knobs live, nothing unread"
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

/-- THE MALFORMED-DOCUMENT REJECTION gate (Findings 1 & 3). A malformed patch is a
    BROKEN document — distinct from a legal-incomplete one (an unwired inlet →
    silence, no error). Three malformations must each return a clear `Except.error`
    and, critically for the live MCP server, must NOT crash the process:
      · a color-LEGAL CYCLE — a reverb whose modal outlet feeds its own modal
        inlet, a mix fed by itself — passes `checkEdgeTypes` (the colors match);
        `lowerGraph`'s own topo-rank entry refuses it with the loop named (the
        lowering is total now — no stack to overflow);
      · a WIRE naming no node (a typo'd source) — previously died downstream as
        `lower: node '…' not found`, or vanished silently if unreferenced;
      · a top-level `"out"` naming no node — previously rendered the WHOLE patch as
        silence with no error.
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
  -- a non-empty WIRE INTO A KNOB (`dir` is a set value, not an inlet): slips past
  -- the color loop (empty accepts) yet suppresses the slot — a silently dead knob.
  let knobWire := "{\"nodes\":[" ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4}}," ++
    "{\"id\":\"k\",\"kind\":\"knob\",\"params\":{\"value\":3}}," ++
    "{\"id\":\"rev\",\"kind\":\"reverb\",\"in\":{\"in\":[\"res\"],\"dir\":[\"k\"]}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"rev\"]}}],\"out\":\"out\"}"
  -- The positive twin: cutoff IS now a control inlet. Two modal islands may
  -- share one glided control triple, with both private cutoff params absent.
  let sharedFilter := "{\"nodes\":[" ++
    "{\"id\":\"k\",\"kind\":\"glideknob\",\"params\":{\"value\":700}}," ++
    "{\"id\":\"r1\",\"kind\":\"resonator\",\"params\":{\"freq\":220,\"decay\":4}}," ++
    "{\"id\":\"r2\",\"kind\":\"resonator\",\"params\":{\"freq\":330,\"decay\":4}}," ++
    "{\"id\":\"f1\",\"kind\":\"filter\",\"in\":{\"in\":[\"r1\"],\"cutoff\":[\"k\"]}}," ++
    "{\"id\":\"f2\",\"kind\":\"filter\",\"in\":{\"in\":[\"r2\"],\"cutoff\":[\"k\"]}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"f1\",\"f2\"]}}],\"out\":\"out\"}"
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
  let mKnob := compileErr knobWire
  let sharedOk := match Lean.Json.parse sharedFilter with
    | .ok j => match Tropical.Playground.compilePlanPure arena resolved j with
      | .ok compiled =>
        compiled.plan.slotNames.contains "param:k.value#v0"
          && compiled.plan.slotNames.contains "param:k.value#v1"
          && compiled.plan.slotNames.contains "param:k.value#t0"
          && !compiled.plan.slotNames.contains "param:k.value"
          && !compiled.plan.slotNames.contains "param:f1.cutoff#v0"
          && !compiled.plan.slotNames.contains "param:f2.cutoff#v0"
          && compiled.plan.paramDisciplines.any (fun d =>
            d.name == "k.value" && d.discipline == "glide")
          && (unreadParamSlots compiled.plan).isEmpty
      | .error _ => false
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
  let knobOk := hasSub mKnob "is a knob"
  IO.println s!"        malformed patches rejected (no crash); legal-incomplete unaffected:"
  IO.println s!"        cycle(reverb self)   {mRev}"
  IO.println s!"        cycle(mix self)      {mMix}"
  IO.println s!"        dangling wire        {mWire}"
  IO.println s!"        dangling out         {mOut}"
  IO.println s!"        withheld kind        {mWithheld}"
  IO.println s!"        unknown kind         {mUnknown}"
  IO.println s!"        wire into a knob     {mKnob}"
  IO.println s!"        shared glided filter control compiles={sharedOk}"
  IO.println s!"        valid modal patch compiles={mValid}"
  if revOk && mixOk && wireOk && outOk && withheldOk && unknownOk && knobOk && sharedOk && mValid then
    passGate "malformed-rejection" "a color-legal cycle, a dangling wire/out, a WITHHELD kind (bloomgong), an UNKNOWN kind, and a wire-into-a-knob each return a clear error (no stack-overflow); shared glided control inlets and a valid patch compile"
  else
    failGate "malformed-rejection" s!"revOk={revOk} mixOk={mixOk} wireOk={wireOk} outOk={outOk} withheldOk={withheldOk} unknownOk={unknownOk} knobOk={knobOk} sharedOk={sharedOk} valid={mValid}"

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
  match Lean.Json.parse (mkPatch false false), Lean.Json.parse (mkPatch true false),
        Lean.Json.parse (mkPatch false true) with
  | .ok jUnwired, .ok jWired, .ok jDangler =>
    let repU := realizedReport jUnwired #[]
    let repW := realizedReport jWired #[]
    let repD := realizedReport jDangler #[]
    let unwiredOk := inputState repU "sfw" "mod" == some "normalled"
      && (paramNames repU).contains "sfw.rate"
    let wiredOk := inputState repW "sfw" "mod" == some "wired"
      && !(paramNames repW).contains "sfw.rate"
    let danglerOk := nodeStatus repD "orphan" == some "excluded"
      && nodeStatus repD "osc" == some "active"
    -- the no-warnings contract, checked on the wire form itself
    let noWarnOk := ((repU.compress ++ repW.compress ++ repD.compress).toLower.splitOn "warn").length == 1
    IO.println s!"        unwired: mod={inputState repU "sfw" "mod"} rate-param={((paramNames repU).contains "sfw.rate")} · wired: mod={inputState repW "sfw" "mod"} rate-param={((paramNames repW).contains "sfw.rate")} · orphan={nodeStatus repD "orphan"}"
    if unwiredOk && wiredOk && danglerOk && noWarnOk then
      passGate "realized-report" "facts, not warnings: normalled/wired inputs, owned knobs absent when superseded, excluded nodes named"
    else
      failGate "realized-report" s!"unwired={unwiredOk} wired={wiredOk} dangler={danglerOk} noWarn={noWarnOk}"
  | _, _, _ => failGate "realized-report" "patch json parse"

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
  match Lean.Json.parse (patch false), Lean.Json.parse (patch true) with
  | .ok ju, .ok jw =>
    match Tropical.Playground.compilePlanPure arena resolved ju,
          Tropical.Playground.compilePlanPure arena resolved jw with
    | .ok compiledU, .ok compiledW =>
      let pu := compiledU.plan
      let pw := compiledW.plan
      let mut issues := check "unwired" pu ++ check "wired" pw
      if !(pu.paramDisciplines.any (·.name == "sfw.rate")) then
        issues := issues.push "unwired: sfw.rate missing from disciplines"
      if pw.paramDisciplines.any (·.name == "sfw.rate") then
        issues := issues.push "wired: superseded sfw.rate present in disciplines"
      if !(pu.paramDisciplines.any fun d => d.name == "master.velocity"
            && d.discipline == "velocity" && d.companions.contains "master.tau_base") then
        issues := issues.push "master.velocity entry wrong"
      IO.println s!"        {pu.paramDisciplines.size}+{pw.paramDisciplines.size} manifest entries checked against their plans' slots:"
      IO.println s!"        result   {if issues.isEmpty then "consistent" else toString issues}"
      if issues.isEmpty then
        passGate "manifest-disciplines" "param_disciplines ≡ the plan's slots — a host can dispatch from it blind"
      else
        failGate "manifest-disciplines" s!"{issues}"
    | .error e, _ | _, .error e =>
      failGate "manifest-disciplines" s!"compile: {firstLine e}"
  | _, _ => failGate "manifest-disciplines" "json parse"

/-- THE DEAD-SLOT LINT gate (the systemic net for the dead-knob class). Canonical
    patches covering every playground node kind compile through the real
    `compilePlanPure`, and every `param:*` slot each plan registers must be READ
    by some instruction operand. Presence-only checks pass a dead knob — the slot
    exists, `setSlot` succeeds, nothing listens — which is exactly how the
    reverb-drops-the-source's-poles regression stayed invisible; unreadness in the
    PLAN is the property that catches the whole class, whatever node grows it next. -/
def runDeadSlotLint (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  -- Allowlist for slots proven legitimately unread, one justified entry at a
  -- time (`<patch-label>:<slot>`); a blanket param exclusion would re-open the
  -- hole the gate exists to close. Currently empty: every registered knob in
  -- the canonical patches must be live.
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
  let patches : Array (String × String) := #[
    ("signal-chain", signalChain), ("modal-chain", modalChain), ("modal-mix", modalMix),
    ("sflange-wired", sflangeWired)]
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
  IO.println s!"        {patches.size} canonical patches over the full node vocabulary, {checked} param:* slots:"
  IO.println s!"        result   unread param slots: {if deadAll.isEmpty then "none" else toString deadAll}"
  if ok then
    passGate "dead-slot-lint" "every registered param:* slot is read by an instruction — no dead knobs behind graceful exclusion"
  else
    failGate "dead-slot-lint" s!"dead knobs (setSlot lands, no instruction reads): {deadAll}"

open Tropical.Playground in
/-- A structural `room_modes` bank is the whole fixed transfer: its exact row
    count reaches the modal node, NONE of the incumbent coefficient knobs is
    registered, and it compiles/renders as finite non-silence. This is the seam
    the fixed demo room uses; omission remains covered by the canonical legacy
    reverb patch in `dead-slot-lint` and the existing frozen patch goldens. -/
def runExplicitRoomModes (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let src := "{\"nodes\":[" ++
    "{\"id\":\"res\",\"kind\":\"resonator\",\"params\":{\"freq\":211,\"decay\":8,\"partials\":2}}," ++
    "{\"id\":\"rvb\",\"kind\":\"reverb\",\"params\":{" ++
      "\"rt60\":99,\"dir\":1,\"sway\":0.9,\"rate\":8," ++
      "\"room_modes\":[[111,0.812677,3,0],[111,115.812677,-3,0]," ++
      "[887,1.255955,2,1.2],[887,116.255955,-2,1.2]]}," ++
      "\"in\":{\"in\":[\"res\"]}}," ++
    "{\"id\":\"out\",\"kind\":\"out\",\"in\":{\"in\":[\"rvb\"]}}],\"out\":\"out\"}"
  match Lean.Json.parse src with
  | .error e => failGate "explicit-room-modes" s!"json parse: {e}"
  | .ok j =>
    match decodeGraph j with
    | .error e => failGate "explicit-room-modes" s!"decode: {firstLine e}"
    | .ok (graph, params) =>
      let roomShapeOk := match graph.nodes.find? (·.id == "rvb") with
        | some { node := .modalReverb "res" room none, .. } => room.size == 4
        | _ => false
      let frozenOk := params.all fun (name, _) => !"rvb.".isPrefixOf name
      match Tropical.Playground.compilePlanPure arena resolved j with
      | .error e => failGate "explicit-room-modes" s!"compile: {firstLine e}"
      | .ok compiled =>
        match ← renderPlanSamples compiled.plan 4096 with
        | .error e => failGate "explicit-room-modes" s!"render: {firstLine e}"
        | .ok samples =>
          let finite := samples.all (·.isFinite)
          let peak := samples.foldl (fun p x => max p x.abs) 0.0
          let planFrozen := compiled.plan.slotNames.all fun name =>
            !"param:rvb.".isPrefixOf name
          IO.println s!"        explicit rows=4 shape={roomShapeOk} frozen params={frozenOk && planFrozen} finite={finite} peak={peak}"
          if roomShapeOk && frozenOk && planFrozen && finite && peak > 1.0e-8 then
            passGate "explicit-room-modes" "authored rows are exact structural room data: fixed count, no coefficient slots, finite audible render"
          else
            failGate "explicit-room-modes" s!"shape={roomShapeOk} frozen={frozenOk}/{planFrozen} finite={finite} peak={peak}"

private def groupedRoomModesJson (firstFrequency : Nat := 211) : Lean.Json :=
  let coords : Array (Nat × Int × Nat) := #[
    (firstFrequency, 75, 1), (211, 955, 1), (433, 865, 2), (433, 10035, 2),
    (887, 98, 1), (887, 1052, 1), (1511, 1095, 2), (1511, 11005, 2),
    (2837, 121, 1), (2837, 1149, 1), (5081, 1325, 2), (5081, 11975, 2)]
  Lean.Json.arr (coords.map fun (frequency, sigma, exponent) =>
    Lean.Json.arr #[Lean.toJson frequency, .num ⟨sigma, exponent⟩,
      Lean.toJson (1 : Nat), Lean.toJson (0 : Nat)])

private def groupedRoomPatchJson (profile : String := Tropical.EmitArrow.groupedRoomProfile)
    (positionSources : Array String := #["position"])
    (firstFrequency : Nat := 211) (throughDirection : Bool := false) : Lean.Json := Id.run do
  let mut nodes : Array Lean.Json := #[
    Lean.Json.mkObj [
      ("id", .str "hit"), ("kind", .str "string"),
      ("params", Lean.Json.mkObj [("t", Lean.toJson (0 : Nat)),
        ("modes", groupedRoomModesJson firstFrequency)]),
      ("in", Lean.Json.mkObj [])],
    Lean.Json.mkObj [
      ("id", .str "position"), ("kind", .str "glideknob"),
      ("params", Lean.Json.mkObj [("value", Lean.toJson (1 : Nat))]),
      ("in", Lean.Json.mkObj [])]]
  if positionSources.size > 1 then
    nodes := nodes.push (Lean.Json.mkObj [
      ("id", .str "position2"), ("kind", .str "knob"),
      ("params", Lean.Json.mkObj [("value", Lean.toJson (0 : Nat))]),
      ("in", Lean.Json.mkObj [])])
  let roomInput := if throughDirection then "incumbent" else "hit"
  if throughDirection then
    nodes := nodes.push (Lean.Json.mkObj [
      ("id", .str "incumbent"), ("kind", .str "reverb"),
      ("params", Lean.Json.mkObj []),
      ("in", Lean.Json.mkObj [("in", Lean.Json.arr #[.str "hit"])])])
  nodes := nodes.push (Lean.Json.mkObj [
    ("id", .str "room"), ("kind", .str "groupedroom"),
    ("params", Lean.Json.mkObj [("profile", .str profile)]),
    ("in", Lean.Json.mkObj [
      ("in", Lean.Json.arr #[.str roomInput]),
      ("position", Lean.Json.arr (positionSources.map Lean.Json.str))])])
  nodes := nodes.push (Lean.Json.mkObj [
    ("id", .str "outn"), ("kind", .str "out"),
    ("in", Lean.Json.mkObj [("in", Lean.Json.arr #[.str "room"])])])
  return Lean.Json.mkObj [("nodes", Lean.Json.arr nodes), ("out", .str "outn")]

private def groupedRoomInstructions (f : Tropical.Plan.InstanceFunction) :
    Array Tropical.Plan.NInstr :=
  f.preambleInstructions ++ f.preInputInstructions ++ f.instructions
    ++ f.children.attach.flatMap fun c => groupedRoomInstructions c.1
termination_by sizeOf f
decreasing_by exact Tropical.Plan.InstanceFunction.sizeOf_lt_of_mem_children c.2

private def groupedRoomCachePatchJson
    (profile : String := Tropical.EmitArrow.groupedRoomCacheProfile)
    (positionSources : Array String := #["position"]) : Lean.Json := Id.run do
  let mut nodes : Array Lean.Json := #[
    Lean.Json.mkObj [
      ("id", .str "position"), ("kind", .str "glideknob"),
      ("params", Lean.Json.mkObj [("value", Lean.toJson (1 : Nat))]),
      ("in", Lean.Json.mkObj [])]]
  if positionSources.size > 1 then
    nodes := nodes.push (Lean.Json.mkObj [
      ("id", .str "position2"), ("kind", .str "knob"),
      ("params", Lean.Json.mkObj [("value", Lean.toJson (0 : Nat))]),
      ("in", Lean.Json.mkObj [])])
  nodes := nodes.push (Lean.Json.mkObj [
    ("id", .str "room"), ("kind", .str "groupedroomcache"),
    ("params", Lean.Json.mkObj [("profile", .str profile)]),
    ("in", Lean.Json.mkObj [
      ("position", Lean.Json.arr (positionSources.map Lean.Json.str))])])
  nodes := nodes.push (Lean.Json.mkObj [
    ("id", .str "outn"), ("kind", .str "out"),
    ("in", Lean.Json.mkObj [("in", Lean.Json.arr #[.str "room"])])])
  return Lean.Json.mkObj [("nodes", Lean.Json.arr nodes), ("out", .str "outn")]

def runGroupedRoomContract (arena : Arena)
    (resolved : Array (String × ProgramIdx)) : IO Bool := do
  let refusal := fun (j : Lean.Json) (needle : String) =>
    match Tropical.Playground.compilePlanPure arena resolved j with
    | .ok _ => false
    | .error e => e.contains needle
  let unknownProfile := refusal (groupedRoomPatchJson "unknown") "unsupported profile"
  let missingPosition := refusal
    (groupedRoomPatchJson (positionSources := #[])) "requires exactly one control source"
  let ambiguousPosition := refusal
    (groupedRoomPatchJson (positionSources := #["position", "position2"]))
    "requires exactly one control source"
  let coordinateMismatch := refusal
    (groupedRoomPatchJson (firstFrequency := 212)) "does not match the frozen"
  let incumbentDirection := refusal
    (groupedRoomPatchJson (throughDirection := true)) "incumbent modal direction"
  let cacheUnknown := refusal (groupedRoomCachePatchJson "unknown") "unsupported profile"
  let cacheMissing := refusal
    (groupedRoomCachePatchJson (positionSources := #[])) "requires exactly one control source"
  let cacheAmbiguous := refusal
    (groupedRoomCachePatchJson (positionSources := #["position", "position2"]))
    "requires exactly one control source"
  let cacheOk := match Tropical.Playground.compilePlanPure arena resolved groupedRoomCachePatchJson with
    | .error _ => false
    | .ok compiled =>
      let cachePlan := compiled.plan
      let cacheInstrs := cachePlan.instanceFunctions.flatMap groupedRoomInstructions
      let assetOk := match cachePlan.immutableAssets[0]? with
        | some asset =>
          cachePlan.immutableAssets.size == 1
            && asset.path == Tropical.EmitArrow.groupedRoomCacheAssetPath
            && asset.sha256 == Tropical.EmitArrow.groupedRoomCacheAssetSha256
            && asset.sampleRate == 44100 && asset.arrays.size == 2
            && asset.arrays.all (·.elementCount == 705600)
        | none => false
      let noReduce := cacheInstrs.all (·.tag != "ReduceBegin")
      let wireOk := (cachePlan.toWire.toOption.map
        (·.compress.contains "tropical_plan_6")).getD false
      let mslOk := match Tropical.Ir.EmitMsl.emitKernel cachePlan with
        | .ok src => src.contains "immutable_asset [[buffer(4)]]"
        | .error _ => false
      assetOk && noReduce && wireOk && mslOk
  match Tropical.Playground.compilePlanPure arena resolved groupedRoomPatchJson with
  | .error e => failGate "grouped-room-contract" s!"compile: {firstLine e}"
  | .ok compiled =>
    let plan := compiled.plan
    let instrs := plan.instanceFunctions.flatMap groupedRoomInstructions
    let reduces := instrs.filter (·.tag == "ReduceBegin")
    let mut depth := 0
    let mut maxDepth := 0
    let mut balanced := true
    for ins in instrs do
      if ins.tag == "ReduceBegin" then
        depth := depth + 1
        maxDepth := max maxDepth depth
      else if ins.tag == "ReduceEnd" then
        if depth == 0 then balanced := false else depth := depth - 1
    balanced := balanced && depth == 0
    let assetOk := match plan.immutableAssets[0]? with
      | some asset =>
        plan.immutableAssets.size == 1
          && asset.path == Tropical.EmitArrow.groupedRoomAssetPath
          && asset.sha256 == Tropical.EmitArrow.groupedRoomAssetSha256
          && asset.sampleRate == 44100 && asset.arrays.size == 2
          && asset.arrays.all (·.elementCount == 339600)
      | none => false
    let assetSlots := plan.immutableAssets.flatMap fun a => a.arrays.map (·.slot)
    let noAssetPack := instrs.all fun ins =>
      if ins.tag != "Pack" then true else match ins.dst with
      | .array slot => !assetSlots.contains slot && ins.args.size ≤ 12
      | _ => ins.args.size ≤ 12
    let reduceOk := reduces.size == 2 && reduces.all (·.loopCount == 12)
      && balanced && maxDepth == 2
    let wireOk := (plan.toWire.toOption.map
      (·.compress.contains "tropical_plan_6")).getD false
    let mslOk := match Tropical.Ir.EmitMsl.emitKernel plan with
      | .ok src => src.contains "immutable_asset [[buffer(4)]]"
      | .error _ => false
    let refusals := unknownProfile && missingPosition && ambiguousPosition
      && coordinateMismatch && incumbentDirection
    let cacheRefusals := cacheUnknown && cacheMissing && cacheAmbiguous
    IO.println s!"        direct Plan-6={assetOk && wireOk && mslOk} nested 12×12={reduceOk}; cache Plan-6/no-reduce={cacheOk}; no asset Pack={noAssetPack}; refusals={refusals && cacheRefusals}"
    if assetOk && wireOk && mslOk && reduceOk && noAssetPack && refusals
        && cacheOk && cacheRefusals then
      passGate "grouped-room-contract"
        "direct profile keeps one nested 12×12 evaluator; selected scene cache is a loop-free Plan-6 linear read; mismatches refuse"
    else
      failGate "grouped-room-contract"
        s!"asset={assetOk} wire={wireOk} msl={mslOk} reduce={reduceOk} cache={cacheOk} noPack={noAssetPack} refusals={refusals}/{cacheRefusals}"

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
