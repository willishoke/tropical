import Turnstile

/-!
# Tool argument types

Typed fronts for the tropical tools, validated at the MCP boundary
(`Frontend.tropicalEngineTools` dispatches them in-process). Simple fields
are typed precisely; ExprNodes, "port name or index", port-ref arrays, and
maps stay opaque `Json` (tropical leaves them untyped too).
-/

open Lean Turnstile

tool_args AddInstance where
  /-- Registered program/type name (builtin or user-defined). -/
  program : String
  /-- Unique name for this instance. -/
  instance_name : String
  /-- Compile-time type args for generics, e.g. {"N": 44100}. Omit for non-generic. -/
  type_args : Option Json
deriving instance ToJson for AddInstance

tool_args RemoveInstance where
  /-- Name of the instance to remove (its wiring is cascade-cleaned). -/
  instance_name : String
deriving instance ToJson for RemoveInstance

tool_args Replicate where
  /-- Registered program type name. -/
  program : String
  /-- Number of instances to create. -/
  count : Nat where 1 ≤ count
  /-- Name prefix (default: lowercase program name); instances named prefix1, prefix2, … -/
  name_prefix : Option String
  /-- Compile-time type args applied to every created instance. Omit for non-generic. -/
  type_args : Option Json
deriving instance ToJson for Replicate

tool_args GetInfo where
  /-- Instance to describe (ports, wiring, registers). -/
  instance_name : String
deriving instance ToJson for GetInfo

tool_args Wire where
  /-- Inputs to set: each {instance, input, expr, [combine]}. -/
  set : Option (Array Json)
  /-- Inputs to disconnect: each {instance, input}. -/
  remove : Option (Array Json)
deriving instance ToJson for Wire

tool_args WireChain where
  /-- Ordered instance names to chain in series. -/
  instances : Array String
  /-- Output port name or index to read from each instance. -/
  output : Json
  /-- Input port name or index to feed on each instance. -/
  input : Json
  /-- Optional ExprNode wired into the first instance's input. -/
  initial_expr : Option Json
deriving instance ToJson for WireChain

tool_args WireZip where
  /-- Sources: each {instance, output}. -/
  sources : Array Json
  /-- Targets: each {instance, input}; equal length to sources. -/
  targets : Array Json
deriving instance ToJson for WireZip

tool_args FanOut where
  /-- Source: {instance, output} or any ExprNode. -/
  source : Json
  /-- Targets: each {instance, input}. -/
  targets : Array Json
deriving instance ToJson for FanOut

tool_args FanIn where
  /-- Sources: each {instance, output, [gain]}; summed. -/
  sources : Array Json
  /-- Target: {instance, input}. -/
  target : Json
deriving instance ToJson for FanIn


tool_args ListWiring where
  /-- If given, filter to inputs of this instance. -/
  «instance» : Option String
deriving instance ToJson for ListWiring

tool_args ExportProgram where
  /-- Name for the new program type (PascalCase recommended). -/
  name : String
  /-- Map: new input name → "instance:port". -/
  inputs : Json
  /-- Map: new output name → {instance, output}. -/
  outputs : Json
  /-- If true, remove the exported instances afterward (default false). -/
  remove_exported : Option Bool
deriving instance ToJson for ExportProgram

tool_args Load where
  /-- Path to a .json file on disk (preferred). -/
  path : Option String
  /-- Inline tropical_program_2 object. -/
  program : Option Json
deriving instance ToJson for Load

tool_args Merge where
  /-- tropical_program_2 object to merge into the current session. -/
  program : Json
deriving instance ToJson for Merge

tool_args SetParam where
  /-- Param name. -/
  name : String
  /-- New value (thread-safe, smoothed). -/
  value : Float
deriving instance ToJson for SetParam

tool_args StartAudio where
  /-- Optional partial device name match. -/
  device_name : Option String
  /-- DAC sample rate in Hz (default 44100). First DAC creation only. -/
  sample_rate : Option Nat where 1 ≤ sample_rate
  /-- DAC output channel count (default 2). First DAC creation only. -/
  channels : Option Nat where 1 ≤ channels
deriving instance ToJson for StartAudio
