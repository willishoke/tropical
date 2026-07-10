#pragma once

/**
 * NumericProgramParser.hpp — Parse `tropical_plan_5` JSON → FlatProgram.
 *
 * Thin deserialiser. Since Phase 2 (Lean owns codegen) this is the
 * *manifest reader*: load_ir consumes the plan's metadata — temp pool size
 * (register_count), array / slot names + sizes, sample_rate, slot defaults —
 * to set up runtime scratch and slots. CF-only removed per-sample state, so
 * there are no state registers (state_init / register_names / register_types /
 * register_targets / state_reg_offset / writebacks). The instruction stream
 * still parses into FlatProgram's fields but is no longer used for codegen
 * (the kernel comes from Lean-emitted IR).
 */

#include "jit/OrcJitEngine.hpp"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace tropical_plan5
{

// ─────────────────────────────────────────────────────────────────────────────
// Tag / operand parsing helpers
// ─────────────────────────────────────────────────────────────────────────────

inline tropical_jit::OpTag parse_op_tag(const std::string & s)
{
  using T = tropical_jit::OpTag;
  static const std::unordered_map<std::string, T> MAP = {
    {"Add",         T::Add},
    {"Sub",         T::Sub},
    {"Mul",         T::Mul},
    {"Div",         T::Div},
    {"Mod",         T::Mod},
    {"FloorDiv",    T::FloorDiv},
    {"Less",        T::Less},
    {"LessEq",      T::LessEq},
    {"Greater",     T::Greater},
    {"GreaterEq",   T::GreaterEq},
    {"Equal",       T::Equal},
    {"NotEqual",    T::NotEqual},
    {"BitAnd",      T::BitAnd},
    {"BitOr",       T::BitOr},
    {"BitXor",      T::BitXor},
    {"LShift",      T::LShift},
    {"RShift",      T::RShift},
    {"Index",       T::Index},
    {"And",         T::And},
    {"Or",          T::Or},
    {"Neg",         T::Neg},
    {"Abs",         T::Abs},
    {"Sqrt",        T::Sqrt},
    {"Floor",       T::Floor},
    {"Ceil",        T::Ceil},
    {"Round",       T::Round},
    {"Not",         T::Not},
    {"BitNot",      T::BitNot},
    {"Clamp",       T::Clamp},
    {"Select",      T::Select},
    {"SetElement",  T::SetElement},
    {"Pack",        T::Pack},
    {"Ldexp",         T::Ldexp},
    {"FloatExponent", T::FloatExponent},
    {"ToInt",       T::ToInt},
    {"ToBool",      T::ToBool},
    {"ToFloat",     T::ToFloat},
    {"SmoothParam", T::SmoothParam},
    {"WriteSlot",   T::WriteSlot},
    {"ReduceBegin", T::ReduceBegin},
    {"ReduceEnd",   T::ReduceEnd},
  };
  const auto it = MAP.find(s);
  if (it == MAP.end())
    throw std::runtime_error("NumericProgramParser: unknown tag '" + s + "'");
  return it->second;
}

inline tropical_jit::JitScalarType parse_scalar_type(const std::string & s)
{
  if (s == "int")  return tropical_jit::JitScalarType::Int;
  if (s == "bool") return tropical_jit::JitScalarType::Bool;
  return tropical_jit::JitScalarType::Float;
}

inline tropical_jit::Operand parse_operand(const nlohmann::json & j)
{
  const std::string kind = j.at("kind").get<std::string>();
  const auto st = parse_scalar_type(j.value("scalar_type", "float"));
  if (kind == "const")
    return tropical_jit::Operand::make_const(j.at("val").get<double>(), st);
  if (kind == "input")
    return tropical_jit::Operand::make_input(j.at("slot").get<uint32_t>(), st);
  if (kind == "reg")
    return tropical_jit::Operand::make_reg(j.at("slot").get<uint32_t>(), st);
  if (kind == "array_reg")
    return tropical_jit::Operand::make_array_reg(j.at("slot").get<uint32_t>());
  if (kind == "param")
  {
    const uint64_t ptr = std::stoull(j.at("ptr").get<std::string>());
    return tropical_jit::Operand::make_param(ptr);
  }
  if (kind == "source")
    return tropical_jit::Operand::make_source(j.at("index").get<uint32_t>(), st);
  // Legacy plan_4 / pre-sources fixtures emit `rate`/`tick` directly;
  // upgrade them to indexed source operands so the engine path is
  // fully migrated. Canonical: sources[0] = tick, sources[1] = rate.
  if (kind == "rate") return tropical_jit::Operand::make_source(1u, st);
  if (kind == "tick") return tropical_jit::Operand::make_source(0u, st);
  if (kind == "slot")
    return tropical_jit::Operand::make_slot(j.at("index").get<uint32_t>(), st);
  // Reduce-region iteration index. The instruction stream is metadata
  // only (codegen is Lean's) — parse it as an inert constant so the
  // manifest read never fails closed on a banked plan.
  if (kind == "loop_idx")
    return tropical_jit::Operand::make_const(0.0, st);
  throw std::runtime_error("NumericProgramParser: unknown operand kind '" + kind + "'");
}

inline tropical_jit::DstKind parse_dst_kind(const std::string & s)
{
  using K = tropical_jit::DstKind;
  if (s == "temp")       return K::Temp;
  if (s == "array")      return K::Array;
  if (s == "moduleSlot") return K::ModuleSlot;
  throw std::runtime_error("NumericProgramParser: unknown dst_kind '" + s + "'");
}

/** Legacy fallback: derive the writeback namespace from the op tag +
 *  `loop_count` proxy. Used only when the wire format omits the
 *  explicit `dst_kind` field — newer producers (`compile_session_slotted.ts`)
 *  always emit it. The derivation matches what the JIT used to assume
 *  before `dst_kind` was carried explicitly, so pre-existing fixtures
 *  parse identically. */
inline tropical_jit::DstKind derive_legacy_dst_kind(tropical_jit::OpTag tag, uint32_t loop_count)
{
  using T = tropical_jit::OpTag;
  using K = tropical_jit::DstKind;
  switch (tag) {
    case T::WriteSlot:   return K::ModuleSlot;
    case T::Pack:        return K::Array;
    case T::SetElement:  return K::Array;
    case T::Index:       return K::Temp;
    case T::SmoothParam: return K::Temp;
    default:             return loop_count > 1 ? K::Array : K::Temp;
  }
}

inline tropical_jit::FlatInstr parse_instr(const nlohmann::json & ji)
{
  tropical_jit::FlatInstr instr;
  instr.tag         = parse_op_tag(ji.at("tag").get<std::string>());
  instr.result_type = parse_scalar_type(ji.value("result_type", "float"));
  instr.dst         = ji.at("dst").get<uint32_t>();
  instr.loop_count  = ji.value("loop_count", 1u);
  if (ji.contains("dst_kind")) {
    instr.dst_kind = parse_dst_kind(ji["dst_kind"].get<std::string>());
  } else {
    instr.dst_kind = derive_legacy_dst_kind(instr.tag, instr.loop_count);
  }
  if (ji.contains("args"))
    for (const auto & a : ji["args"])
      instr.args.push_back(parse_operand(a));
  if (ji.contains("strides"))
    for (const auto & s : ji["strides"])
      instr.strides.push_back(s.get<uint8_t>());
  return instr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Parsed plan_5 result
// ─────────────────────────────────────────────────────────────────────────────

struct ParsedPlan5
{
  tropical_jit::FlatProgram                program;
  std::vector<std::string>                 array_slot_names;
  std::vector<uint32_t>                    mix_indices;
  double                                   sample_rate = 44100.0;

  // Engine realization strategy. Defaults to Fused for legacy plans
  // and any plan whose `compilation_mode` field is absent.
  tropical_jit::CompilationMode compilation_mode = tropical_jit::CompilationMode::Fused;

  // Inter-module slot array.
  uint32_t                 slot_count = 0;
  std::vector<std::string> slot_names;
  std::vector<double>      slot_defaults;

  // Host-contract param write dispatch (design/host-param-dispatch.md): the
  // discipline is a fact about the compiled patch, carried by the plan, so a
  // host dispatches writes itself and no client ever chooses a verb. Absent
  // field ⇒ empty table ⇒ every write is raw (old plans unaffected).
  struct ParamDiscipline
  {
    std::string              name;             // base param ("sf.depth"; slot "param:<name>")
    std::string              discipline;       // raw | glide | anchor | velocity
    double                   glide_dur_sec = 0.0;
    std::vector<std::string> companions;       // #v0/#v1/#t0, #phase, tau_base sibling
  };
  std::vector<ParamDiscipline> param_disciplines;

  // Array slots filled by the stage-0 coefficient kernel (banks-as-data
  // coefficient columns). The runtime double-buffers exactly these so the audio
  // kernel reads a whole, consistent generation of columns across a live knob
  // move. Absent ⇒ empty ⇒ no double-buffering (old plans unaffected).
  std::vector<uint32_t> coeff_array_slots;
};

// Parse the optional `compilation_mode` JSON string. Fails closed on
// unknown values — mirrors parseCompilationMode in compiler/flat_plan.ts.
inline tropical_jit::CompilationMode parse_compilation_mode(const nlohmann::json & plan)
{
  if (!plan.contains("compilation_mode")) return tropical_jit::CompilationMode::Fused;
  const std::string s = plan["compilation_mode"].get<std::string>();
  if (s == "fused")             return tropical_jit::CompilationMode::Fused;
  if (s == "microkernel")       return tropical_jit::CompilationMode::Microkernel;
  if (s == "microkernel-deep")  return tropical_jit::CompilationMode::MicrokernelDeep;
  throw std::runtime_error("NumericProgramParser: unknown compilation_mode '" + s + "'");
}

/** Parse a single `InstanceProgram` from JSON, recursing into nested
 *  `children`. CF-only: no state registers, so no `register_targets` /
 *  `state_reg_offset` / writebacks — only the temp pool (`register_count`)
 *  and the instruction stream. */
inline tropical_jit::InstanceProgram parse_instance_program(const nlohmann::json & jf)
{
  tropical_jit::InstanceProgram inst;
  inst.instance_name    = jf.value("instance_name", std::string{});
  inst.register_count   = jf.value("register_count", 0u);
  if (jf.contains("instructions"))
    for (const auto & ji : jf["instructions"])
      inst.instructions.push_back(parse_instr(ji));
  // Slot-based parent→child input wiring. Optional in the wire format
  // (legacy plans don't have it; absent → empty). Same instruction
  // shape as `instructions`; difference is purely WHEN they emit —
  // the parent's emit_kernel_block runs each child's
  // pre_input_instructions immediately before recursing into that
  // child's body.
  if (jf.contains("preamble_instructions"))
    for (const auto & ji : jf["preamble_instructions"])
      inst.preamble_instructions.push_back(parse_instr(ji));
  if (jf.contains("pre_input_instructions"))
    for (const auto & ji : jf["pre_input_instructions"])
      inst.pre_input_instructions.push_back(parse_instr(ji));
  // Recurse into nested children (the fractal architecture: one
  // kernel per InstanceDecl at every nesting depth). Absent or
  // empty `children` array means a leaf kernel.
  if (jf.contains("children"))
    for (const auto & jc : jf["children"])
      inst.children.push_back(parse_instance_program(jc));
  return inst;
}

/** Parse a tropical_plan_5 JSON object. The C++ side supports nested
 *  `instance_functions` (the fractal architecture): each function may
 *  have `children: InstanceFunction[]` that emit recursively inside the
 *  parent's body. Legacy plans without nested children parse as
 *  leaf-only kernels. */
inline ParsedPlan5 parse_plan5(const nlohmann::json & plan)
{
  ParsedPlan5 result;

  result.sample_rate = plan.value("config", nlohmann::json::object())
                           .value("sampleRate", 44100.0);

  result.compilation_mode = parse_compilation_mode(plan);

  if (plan.contains("array_slot_names"))
    for (const auto & n : plan["array_slot_names"])
      result.array_slot_names.push_back(n.get<std::string>());

  auto & prog = result.program;
  prog.register_count = plan.value("register_count", 0u);

  if (plan.contains("array_slot_sizes"))
    for (const auto & s : plan["array_slot_sizes"])
      prog.array_slot_sizes.push_back(s.get<uint32_t>());

  // ── instance_functions[] ──
  if (plan.contains("instance_functions"))
  {
    for (const auto & jf : plan["instance_functions"])
      prog.instance_functions.push_back(parse_instance_program(jf));
  }

  // (The `scheduler_function` field was retired with the per-instance
  // lowering — plan_5 outputs are `sinks`; session-level delays are root
  // RegDecl writebacks inside the kernel. Any legacy `scheduler_function`
  // key in an old hand fixture is simply ignored by nlohmann.)

  // ── sinks: device-bound outputs (slot-mix path) ──
  if (plan.contains("sinks"))
  {
    for (const auto & js : plan["sinks"])
    {
      tropical_jit::Sink sink;
      if (js.contains("inputs"))
        for (const auto & i : js["inputs"])
          sink.inputs.push_back(i.get<uint32_t>());
      sink.gain   = js.value("gain", 0.05);
      sink.target = js.value("target", 0u);
      prog.sinks.push_back(sink);
    }
  }

  // ── sources: runtime-bound inputs (the dual of sinks). When absent,
  //    default to the canonical pair [tick, rate] so plan_4 / pre-sources
  //    fixtures using legacy `tick`/`rate` operands still resolve via the
  //    upgrade in `parse_operand`. ──
  if (plan.contains("sources"))
  {
    for (const auto & js : plan["sources"])
    {
      tropical_jit::Source src;
      const std::string k = js.value("kind", std::string{"tick"});
      src.kind = (k == "rate") ? tropical_jit::SourceKind::Rate
                               : tropical_jit::SourceKind::Tick;
      prog.sources.push_back(src);
    }
  }
  else
  {
    prog.sources.push_back({ tropical_jit::SourceKind::Tick });
    prog.sources.push_back({ tropical_jit::SourceKind::Rate });
  }

  // ── Legacy temp-mix carrier (top-level, plan_4-style) — used when a
  //    plan has no `sinks` (hand fixtures / single-kernel plans). ──
  if (plan.contains("output_targets"))
    for (const auto & t : plan["output_targets"])
      prog.output_targets.push_back(t.get<uint32_t>());
  if (plan.contains("outputs"))
    for (const auto & o : plan["outputs"])
      result.mix_indices.push_back(o.get<uint32_t>());
  for (uint32_t idx : result.mix_indices)
    if (idx < prog.output_targets.size())
      prog.mix_output_temps.push_back(prog.output_targets[idx]);

  // ── Slot array ──
  if (plan.contains("slot_count") && plan["slot_count"].is_number())
    result.slot_count = plan["slot_count"].get<uint32_t>();
  if (plan.contains("slot_names"))
    for (const auto & n : plan["slot_names"])
      result.slot_names.push_back(n.get<std::string>());
  if (plan.contains("slot_defaults"))
    for (const auto & v : plan["slot_defaults"])
      result.slot_defaults.push_back(v.get<double>());

  // ── Host-contract dispatch table (contains-guarded: absent ⇒ all-raw) ──
  if (plan.contains("param_disciplines"))
    for (const auto & d : plan["param_disciplines"])
    {
      ParsedPlan5::ParamDiscipline pd;
      pd.name       = d.value("name", std::string{});
      pd.discipline = d.value("discipline", std::string{"raw"});
      pd.glide_dur_sec = d.value("glide_dur_sec", 0.0);
      if (d.contains("companions"))
        for (const auto & c : d["companions"])
          pd.companions.push_back(c.get<std::string>());
      result.param_disciplines.push_back(std::move(pd));
    }

  if (plan.contains("coeff_array_slots"))
    for (const auto & s : plan["coeff_array_slots"])
      result.coeff_array_slots.push_back(s.get<uint32_t>());

  return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Backward compat: tropical_plan_4 → tropical_plan_5
// Lifts a single-program plan_4 into a one-instance-function plan_5
// so hand-crafted plan_4 unit tests and any disk-saved plans continue
// to load. The legacy plan's whole `instructions[]` becomes the body
// of `instance_functions[0]`; DAC stitching moves to the scheduler
// postamble.
// ─────────────────────────────────────────────────────────────────────────────

inline ParsedPlan5 parse_plan4(const nlohmann::json & plan)
{
  ParsedPlan5 result;
  result.sample_rate = plan.value("config", nlohmann::json::object())
                           .value("sampleRate", 44100.0);

  result.compilation_mode = parse_compilation_mode(plan);

  if (plan.contains("array_slot_names"))
    for (const auto & n : plan["array_slot_names"])
      result.array_slot_names.push_back(n.get<std::string>());

  auto & prog = result.program;
  prog.register_count = plan.value("register_count", 0u);
  // plan_4 predates `sources`; seed the canonical pair so legacy
  // `tick`/`rate` operands (upgraded to `source:0/1`) resolve.
  prog.sources.push_back({ tropical_jit::SourceKind::Tick });
  prog.sources.push_back({ tropical_jit::SourceKind::Rate });
  if (plan.contains("array_slot_sizes"))
    for (const auto & s : plan["array_slot_sizes"])
      prog.array_slot_sizes.push_back(s.get<uint32_t>());
  if (plan.contains("output_targets"))
    for (const auto & t : plan["output_targets"])
      prog.output_targets.push_back(t.get<uint32_t>());
  if (plan.contains("outputs"))
    for (const auto & o : plan["outputs"])
      result.mix_indices.push_back(o.get<uint32_t>());
  for (uint32_t idx : result.mix_indices)
  {
    if (idx < prog.output_targets.size())
      prog.mix_output_temps.push_back(prog.output_targets[idx]);
  }

  // ── Synthesize one InstanceProgram for the entire body ──
  tropical_jit::InstanceProgram inst;
  inst.instance_name = "legacy_kernel";
  inst.register_count = prog.register_count;

  if (plan.contains("instructions"))
    for (const auto & ji : plan["instructions"])
      inst.instructions.push_back(parse_instr(ji));

  // CF-only: no state registers, so no writebacks to lift.

  // ── Slot array ──
  if (plan.contains("slot_count") && plan["slot_count"].is_number())
    result.slot_count = plan["slot_count"].get<uint32_t>();
  if (plan.contains("slot_names"))
    for (const auto & n : plan["slot_names"])
      result.slot_names.push_back(n.get<std::string>());
  if (plan.contains("slot_defaults"))
    for (const auto & v : plan["slot_defaults"])
      result.slot_defaults.push_back(v.get<double>());

  prog.instance_functions.push_back(std::move(inst));

  return result;
}

} // namespace tropical_plan5

// Backward-compat alias: the legacy namespace is still referenced by
// some unit tests; keep the symbol resolving.
namespace tropical_plan4
{
  using ParsedPlan4 = tropical_plan5::ParsedPlan5;
  inline tropical_plan5::ParsedPlan5 parse_plan4(const nlohmann::json & plan)
  {
    return tropical_plan5::parse_plan4(plan);
  }
}
