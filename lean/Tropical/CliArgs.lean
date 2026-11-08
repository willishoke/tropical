import Lean.Data.Json

/-!
# Tropical.CliArgs

The command-line flag reader `tropicalc` and `diffcli` share. A verb declares
the flags it takes; anything else is an error, never silently ignored. A
value flag is spelled `--name=V` or `--name V`; a switch is `--name`. Every
other argument is positional. Pure: the binaries print the error.
-/

namespace Tropical.CliArgs

/-- A flag a verb takes: a switch (`--name`) or a value flag. -/
structure Flag where
  name : String
  value : Bool := false

def switch (name : String) : Flag := { name }
def valued (name : String) : Flag := { name, value := true }

structure Args where
  positionals : Array String := #[]
  switches : Array String := #[]
  values : Array (String × String) := #[]

/-- Read `args` against the flags a verb takes. An unknown flag, a value
    flag with no value, a switch given a value, or a flag given twice is an
    error. -/
def parse (flags : List Flag) (args : List String) : Except String Args :=
  go args {}
where
  go : List String → Args → Except String Args
    | [], acc => pure acc
    | arg :: rest, acc => do
      unless arg.startsWith "--" && arg.length > 2 do
        return ← go rest { acc with positionals := acc.positionals.push arg }
      let body := (arg.drop 2).toString
      let (name, inline?) := match body.splitOn "=" with
        | name :: value :: more => (name, some ("=".intercalate (value :: more)))
        | _ => (body, none)
      let some flag := flags.find? (·.name == name)
        | throw s!"unknown flag '--{name}'"
      if acc.switches.contains name || acc.values.any (·.1 == name) then
        throw s!"--{name} given twice"
      if !flag.value then
        if inline?.isSome then throw s!"--{name} takes no value"
        return ← go rest { acc with switches := acc.switches.push name }
      if let some value := inline? then
        return ← go rest { acc with values := acc.values.push (name, value) }
      match rest with
      | value :: rest' =>
        if value.startsWith "--" then throw s!"--{name} needs a value"
        go rest' { acc with values := acc.values.push (name, value) }
      | [] => throw s!"--{name} needs a value"

def Args.has (a : Args) (name : String) : Bool := a.switches.contains name

def Args.value? (a : Args) (name : String) : Option String :=
  (a.values.find? (·.1 == name)).map (·.2)

/-- A natural-number flag; absent is `none`, malformed an error. -/
def Args.nat? (a : Args) (name : String) : Except String (Option Nat) :=
  match a.value? name with
  | none => pure none
  | some text => match text.toNat? with
    | some n => pure (some n)
    | none => throw s!"--{name} must be a natural number (got '{text}')"

def Args.natD (a : Args) (name : String) (dflt : Nat) : Except String Nat :=
  return (← a.nat? name).getD dflt

/-- A number flag (`--center=800`, `--octaves=-1.5`); absent is `dflt`,
    malformed an error. -/
def Args.floatD (a : Args) (name : String) (dflt : Float) : Except String Float :=
  match a.value? name with
  | none => pure dflt
  | some text => match Lean.Json.parse text with
    | .ok (.num n) => pure n.toFloat
    | _ => throw s!"--{name} must be a number (got '{text}')"

/-- A positive natural-number flag (a rate, a frame count). -/
def Args.positive? (a : Args) (name : String) : Except String (Option Nat) :=
  match a.value? name with
  | none => pure none
  | some text => match text.toNat? with
    | some n => if n > 0 then pure (some n)
      else throw s!"--{name} must be a positive integer (got '{text}')"
    | none => throw s!"--{name} must be a positive integer (got '{text}')"

/-- `--rate`: the rate a kernel is compiled for, a positive integer. -/
def Args.rate? (a : Args) : Except String (Option Nat) := a.positive? "rate"

/-- The one positional argument a verb takes (the patch or plan path). -/
def Args.single (a : Args) : Except String String :=
  match a.positionals.toList with
  | [path] => pure path
  | [] => throw "missing the input path"
  | _ => throw s!"expected one input path, got {a.positionals.size}: {a.positionals.toList}"

end Tropical.CliArgs
