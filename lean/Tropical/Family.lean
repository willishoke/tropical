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
  build : FamilyBuilder
deriving Inhabited

def TypeFamily.memberName (f : TypeFamily) (size : Nat) : String :=
  s!"{f.name}{size}"

def TypeFamily.rangeText (f : TypeFamily) : String :=
  s!"{f.sizeMin} to {f.sizeMax} {f.sizeMeaning}"

/-- A name split into a stem and its trailing decimal digits, when it has a
    non-empty stem and at least one trailing digit. -/
def splitSizeSuffix (name : String) : Option (String × String) :=
  let chars := name.toList
  let digits := (chars.reverse.takeWhile Char.isDigit).reverse
  if digits.isEmpty || digits.length == chars.length then none
  else some (String.ofList (chars.take (chars.length - digits.length)), String.ofList digits)

/-- What a type name means with respect to a set of families. -/
inductive FamilyLookup where
  /-- Not a family name and not a family name followed by digits. -/
  | none
  /-- A member at an admitted size, spelled canonically. -/
  | member (f : TypeFamily) (size : Nat)
  /-- The bare family name. -/
  | missingSize (f : TypeFamily)
  /-- A size outside the family's range. -/
  | outOfRange (f : TypeFamily) (size : Nat)
  /-- A size written with leading zeros (`PeriodicCatmullRom08`); the member
      has one name. -/
  | leadingZero (f : TypeFamily) (size : Nat)

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
        let size := digits.toNat!
        if digits.length > 1 && digits.front == '0' then .leadingZero f size
        else if size < f.sizeMin || size > f.sizeMax then .outOfRange f size
        else .member f size

/-- The family-spelling refusal for `name`, or `none` when `name` is a member
    (or not a family spelling at all). `isRegistered` answers whether a stem is
    a registered concrete type, for the "takes no size" refusal. -/
def familyRefusal (families : Array TypeFamily) (isRegistered : String → Bool)
    (name : String) : Option String :=
  match lookupFamily families name with
  | .member _ _ => none
  | .missingSize f =>
    some s!"'{name}' is a type family and needs a size: name a member as '{name}<size>', e.g. '{f.memberName (max f.sizeMin (min 8 f.sizeMax))}' (sizes {f.rangeText})"
  | .outOfRange f size =>
    some s!"'{name}': size {size} is out of range for the type family '{f.name}' (sizes {f.rangeText})"
  | .leadingZero f size =>
    some s!"'{name}': write the size without leading zeros, '{f.memberName size}'"
  | .none =>
    match splitSizeSuffix name with
    | some (stem, _) =>
      if isRegistered stem then
        some s!"'{name}': '{stem}' is not a type family and takes no size"
      else none
    | none => none

/-- The registration-time contract of a family set against the registered
    names: family names are distinct, non-empty, end in a non-digit (so a
    member name splits back into family and size), and are not registered
    types; no registered name is a family spelling other than a canonical
    member. The registry refuses to boot otherwise. -/
def familyConflicts (families : Array TypeFamily) (registered : Array String) :
    Array String := Id.run do
  let mut out := #[]
  for i in [0:families.size] do
    let f := families[i]!
    if f.name.isEmpty || f.name.back.isDigit then
      out := out.push s!"type family '{f.name}': a family name must end in a non-digit"
    if f.sizeMin > f.sizeMax then
      out := out.push s!"type family '{f.name}': empty size range {f.sizeMin} to {f.sizeMax}"
    if (families.extract 0 i).any (·.name == f.name) then
      out := out.push s!"type family '{f.name}' is declared twice"
    if registered.contains f.name then
      out := out.push s!"type family '{f.name}' is also a registered type"
  for n in registered do
    match lookupFamily families n with
    | .none | .member _ _ | .missingSize _ => pure ()
    | _ => out := out.push s!"registered type '{n}' spells a size its family '{(splitSizeSuffix n).map (·.1) |>.getD n}' refuses"
  return out

end Tropical
