import Tropical.Ir.Nodes

/-!
# Type families — registered types indexed by a size

A type family is a builder from a size to a program: a periodic spline with
`K` control points, say, whose ports `c0 … c(K−1)` depend on `K`. The patch
bay names a member by the family name followed by the size in decimal
(`PeriodicCatmullRom12`). Ingest resolves that name, assembles the member
once per session, and registers it under that name as an ordinary concrete
type. Nothing past ingest sees a family: the IR, the plan and the kernel
manifest only see the member, a monomorphic program with a fixed port list.

Design and refusals: `design/type-family.md`.
-/

namespace Tropical

/-- Builds the member called `name` at `size`, appending its raw program to
    the arena. `chain` maps registered type names to programs the member may
    instantiate (the `StdBuilder` contract). -/
abbrev FamilyBuilder :=
  (name : String) → (size : Nat) → Ir.Arena → Array (String × Ir.ProgramIdx) →
    Except String (Ir.Arena × Ir.ProgramIdx)

/-- The most ports (inputs and outputs) a family member may have. Everything
    downstream of a member scales with its port count: its assembly, its
    lowering, the plan, and one kernel parameter per live input. A family's
    `sizeMax` is refused at boot unless its member at `sizeMax` fits
    (`familyConflicts`), so no request can make the registry assemble a member
    past this. 1024 data ports and a few control ports; design note:
    `design/type-family.md`. -/
def memberPortBudget : Nat := 1056

structure TypeFamily where
  /-- The family name. A member is this name followed by its size. -/
  name : String
  /-- The admitted sizes, `sizeMin ≤ size ≤ sizeMax`. -/
  sizeMin : Nat
  sizeMax : Nat
  /-- What the size counts, for messages ("control points"). -/
  sizeMeaning : String
  /-- The ports of a member, in words, for the catalog. -/
  ports : String
  /-- The number of ports (inputs and outputs) of the member at a size. -/
  portsAt : Nat → Nat
  build : FamilyBuilder
deriving Inhabited

def TypeFamily.memberName (f : TypeFamily) (size : Nat) : String :=
  s!"{f.name}{size}"

def TypeFamily.rangeText (f : TypeFamily) : String :=
  s!"{f.sizeMin} to {f.sizeMax} {f.sizeMeaning}"

/-- A name as messages show it: names past 64 characters are cut, so a
    refusal never echoes an unbounded digit string. -/
def shownName (name : String) : String :=
  if name.length ≤ 64 then name
  else s!"{String.ofList (name.toList.take 48)}… ({name.length} characters)"

/-- A name split into a stem and its trailing decimal digits, when it has a
    non-empty stem and at least one trailing digit. -/
def splitSizeSuffix (name : String) : Option (String × String) :=
  let chars := name.toList
  let digits := (chars.reverse.takeWhile Char.isDigit).reverse
  if digits.isEmpty || digits.length == chars.length then none
  else some (String.ofList (chars.take (chars.length - digits.length)), String.ofList digits)

/-- The value of a decimal digit string, when it can be at most `sizeMax`:
    leading zeros are skipped, and a string with more significant digits than
    `sizeMax` is `none` without being converted. -/
def TypeFamily.sizeOf? (f : TypeFamily) (digits : String) : Option Nat :=
  let significant := String.ofList (digits.toList.dropWhile (· == '0'))
  if significant.length > (toString f.sizeMax).length then none
  else some significant.toNat!

/-- What a type name means with respect to a set of families. -/
inductive FamilyLookup where
  /-- Not a family name and not a family name followed by digits. -/
  | none
  /-- A member at an admitted size, spelled canonically. -/
  | member (f : TypeFamily) (size : Nat)
  /-- The bare family name. -/
  | missingSize (f : TypeFamily)
  /-- A size outside the family's range; `none` when it has more digits than
      the family's largest size, and was not converted. -/
  | outOfRange (f : TypeFamily) (size : Option Nat)
  /-- An admitted size written with leading zeros (`PeriodicCatmullRom08`);
      the member has one name. -/
  | leadingZero (f : TypeFamily) (size : Nat)

/-- Classify a name. The size is parsed once, in bounded time. The range is
    checked before the spelling, so a refusal never advises an out-of-range
    name. -/
def lookupFamily (families : Array TypeFamily) (name : String) : FamilyLookup :=
  match families.find? (·.name == name) with
  | some f => .missingSize f
  | none =>
    match splitSizeSuffix name with
    | none => .none
    | some (stem, digits) =>
      match families.find? (·.name == stem) with
      | none => .none
      | some f =>
        match f.sizeOf? digits with
        | none => .outOfRange f none
        | some size =>
          if size < f.sizeMin || size > f.sizeMax then .outOfRange f (some size)
          else if digits.length > 1 && digits.front == '0' then .leadingZero f size
          else .member f size

/-- The refusal for a family spelling that names no member (`none` for a
    member or a name that is not a family spelling). -/
def familyRefusal (lookup : FamilyLookup) (name : String) : Option String :=
  let n := shownName name
  match lookup with
  | .member _ _ | .none => none
  | .missingSize f =>
    some s!"'{n}' is a type family and needs a size: name a member as '{n}<size>', e.g. '{f.memberName (max f.sizeMin (min 8 f.sizeMax))}' (sizes {f.rangeText})"
  | .outOfRange f size =>
    let what := match size with
      | some k => s!"size {k}"
      | none => "the size"
    some s!"'{n}': {what} is out of range for the type family '{f.name}' (sizes {f.rangeText})"
  | .leadingZero f size =>
    some s!"'{n}': write the size without leading zeros, '{f.memberName size}'"

/-- For a name that is no type and no family spelling: the members of every
    family at the size the name ends in, when admitted. They are what a
    misspelt member name (`PeriodicCatmulRom12`) is nearest to. -/
def familySpellingsAt (families : Array TypeFamily) (name : String) : Array String :=
  match splitSizeSuffix name with
  | none => #[]
  | some (_, digits) =>
    families.filterMap fun f =>
      match f.sizeOf? digits with
      | some k => if f.sizeMin ≤ k && k ≤ f.sizeMax then some (f.memberName k) else none
      | none => none

/-- The registration-time contract of a family set against the registered
    names: family names are distinct, non-empty, end in a non-digit (so a
    member name splits back into family and size), and are not registered
    types; every member fits the port budget (`memberPortBudget`); no
    registered name is a family spelling other than a canonical member. The
    registry refuses to boot otherwise. -/
def familyConflicts (families : Array TypeFamily) (registered : Array String) :
    Array String := Id.run do
  let mut out := #[]
  for i in [0:families.size] do
    let f := families[i]!
    if f.name.isEmpty || f.name.back.isDigit then
      out := out.push s!"type family '{f.name}': a family name must end in a non-digit"
    if f.sizeMin > f.sizeMax then
      out := out.push s!"type family '{f.name}': empty size range {f.sizeMin} to {f.sizeMax}"
    if f.portsAt f.sizeMax > memberPortBudget then
      out := out.push s!"type family '{f.name}': its member at size {f.sizeMax} has {f.portsAt f.sizeMax} ports, past the budget of {memberPortBudget}"
    if (families.extract 0 i).any (·.name == f.name) then
      out := out.push s!"type family '{f.name}' is declared twice"
    if registered.contains f.name then
      out := out.push s!"type family '{f.name}' is also a registered type"
  for n in registered do
    match lookupFamily families n with
    | .none | .member _ _ | .missingSize _ => pure ()
    | _ => out := out.push s!"registered type '{shownName n}' spells a size its family '{(splitSizeSuffix n).map (·.1) |>.getD n}' refuses"
  return out

end Tropical
