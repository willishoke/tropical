#pragma once

/**
 * NumericProgramParser.hpp — Parse `tropical_plan_5` JSON → FlatProgram.
 *
 * Thin deserialiser. Reads the instruction stream emitted by
 * compiler/ir/compile_session_slotted.ts and produces the
 * tropical_jit::FlatProgram passed to compile_flat_program(). The
 * plan's multi-function layout (instance_functions[] +
 * scheduler_function) maps onto FlatProgram's parallel fields; the
 * C++ JIT engine emits one LLVM function per instance plus one
 * scheduler that drives them.
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
  if (kind == "state_reg")
    return tropical_jit::Operand::make_state(j.at("slot").get<uint32_t>(), st);
  if (kind == "param")
  {
    const uint64_t ptr = std::stoull(j.at("ptr").get<std::string>());
    return tropical_jit::Operand::make_param(ptr);
  }
  if (kind == "rate") return tropical_jit::Operand::make_rate();
  if (kind == "tick") return tropical_jit::Operand::make_tick();
  if (kind == "slot")
    return tropical_jit::Operand::make_slot(j.at("index").get<uint32_t>(), st);
  throw std::runtime_error("NumericProgramParser: unknown operand kind '" + kind + "'");
}

inline tropical_jit::FlatInstr parse_instr(const nlohmann::json & ji)
{
  tropical_jit::FlatInstr instr;
  instr.tag         = parse_op_tag(ji.at("tag").get<std::string>());
  instr.result_type = parse_scalar_type(ji.value("result_type", "float"));
  instr.dst         = ji.at("dst").get<uint32_t>();
  instr.loop_count  = ji.value("loop_count", 1u);
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
  std::vector<double>                      state_init;
  std::vector<std::string>                 register_names;
  std::vector<tropical_jit::JitScalarType> register_types;
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
};

// Parse the optional `compilation_mode` JSON string. Fails closed on
// unknown values — mirrors parseCompilationMode in compiler/flat_plan.ts.
inline tropical_jit::CompilationMode parse_compilation_mode(const nlohmann::json & plan)
{
  if (!plan.contains("compilation_mode")) return tropical_jit::CompilationMode::Fused;
  const std::string s = plan["compilation_mode"].get<std::string>();
  if (s == "fused")       return tropical_jit::CompilationMode::Fused;
  if (s == "microkernel") return tropical_jit::CompilationMode::Microkernel;
  throw std::runtime_error("NumericProgramParser: unknown compilation_mode '" + s + "'");
}

/** Parse a single `InstanceProgram` from JSON, recursing into nested
 *  `children`. Each child kernel's writebacks resolve against its own
 *  `state_reg_offset` — nested kernels are independent in the unified
 *  state-register space, just like top-level ones. */
inline tropical_jit::InstanceProgram parse_instance_program(const nlohmann::json & jf)
{
  tropical_jit::InstanceProgram inst;
  inst.instance_name    = jf.value("instance_name", std::string{});
  inst.register_count   = jf.value("register_count", 0u);
  if (jf.contains("instructions"))
    for (const auto & ji : jf["instructions"])
      inst.instructions.push_back(parse_instr(ji));
  // M11 fractal slot-based input wiring. Optional in the wire format
  // (legacy plans don't have it; absent → empty). Same instruction
  // shape as `instructions`; difference is purely WHEN they emit —
  // the parent's emit_kernel_block runs each child's
  // pre_input_instructions immediately before recursing into that
  // child's body.
  if (jf.contains("pre_input_instructions"))
    for (const auto & ji : jf["pre_input_instructions"])
      inst.pre_input_instructions.push_back(parse_instr(ji));
  // Writebacks: each entry of register_targets[] is the (absolute,
  // already shifted) temp slot whose value feeds the state register at
  // position j among this instance's slots, which sits at
  // `state_reg_offset + j` in the unified state array.
  const uint32_t state_reg_offset = jf.value("state_reg_offset", 0u);
  if (jf.contains("register_targets"))
  {
    const auto & rts = jf["register_targets"];
    for (uint32_t j = 0; j < rts.size(); ++j)
    {
      inst.writebacks.push_back({
        state_reg_offset + j,
        rts[j].get<int32_t>()
      });
    }
  }
  // Recurse into nested children (M11 fractal architecture). Absent or
  // empty `children` array means a leaf kernel.
  if (jf.contains("children"))
    for (const auto & jc : jf["children"])
      inst.children.push_back(parse_instance_program(jc));
  return inst;
}

/** Parse a tropical_plan_5 JSON object. The C++ side supports nested
 *  `instance_functions` (M11 fractal architecture): each function may
 *  have `children: InstanceFunction[]` that emit recursively inside the
 *  parent's alive-conditional. Legacy plans without nested children
 *  parse as leaf-only kernels. */
inline ParsedPlan5 parse_plan5(const nlohmann::json & plan)
{
  ParsedPlan5 result;

  result.sample_rate = plan.value("config", nlohmann::json::object())
                           .value("sampleRate", 44100.0);

  result.compilation_mode = parse_compilation_mode(plan);

  if (plan.contains("state_init"))
  {
    for (const auto & v : plan["state_init"])
    {
      if (v.is_number())
        result.state_init.push_back(v.get<double>());
      else if (v.is_boolean())
        result.state_init.push_back(v.get<bool>() ? 1.0 : 0.0);
      else
        result.state_init.push_back(0.0);
    }
  }

  if (plan.contains("register_names"))
    for (const auto & n : plan["register_names"])
      result.register_names.push_back(n.get<std::string>());

  if (plan.contains("register_types"))
    for (const auto & t : plan["register_types"])
      result.register_types.push_back(parse_scalar_type(t.get<std::string>()));

  result.program.register_types = result.register_types;

  if (plan.contains("array_slot_names"))
    for (const auto & n : plan["array_slot_names"])
      result.array_slot_names.push_back(n.get<std::string>());

  auto & prog = result.program;
  prog.register_count = plan.value("register_count", 0u);

  if (plan.contains("array_slot_sizes"))
    for (const auto & s : plan["array_slot_sizes"])
      prog.array_slot_sizes.push_back(s.get<uint32_t>());

  if (plan.contains("register_targets"))
    for (const auto & t : plan["register_targets"])
      prog.register_targets.push_back(t.get<int32_t>());

  // ── instance_functions[] ──
  if (plan.contains("instance_functions"))
  {
    for (const auto & jf : plan["instance_functions"])
      prog.instance_functions.push_back(parse_instance_program(jf));
  }

  // ── scheduler_function ──
  if (plan.contains("scheduler_function"))
  {
    const auto & jsched = plan["scheduler_function"];
    if (jsched.contains("preamble"))
      for (const auto & ji : jsched["preamble"])
        prog.scheduler.preamble.push_back(parse_instr(ji));
    // state_evolution: optional in the wire format (legacy plans
    // predating Phase 3 omit it; treated as empty list). Holds
    // delay-slot updates from MCP wire auto-delays.
    if (jsched.contains("state_evolution"))
      for (const auto & ji : jsched["state_evolution"])
        prog.scheduler.state_evolution.push_back(parse_instr(ji));
    if (jsched.contains("postamble"))
      for (const auto & ji : jsched["postamble"])
        prog.scheduler.postamble.push_back(parse_instr(ji));
    if (jsched.contains("output_targets"))
      for (const auto & t : jsched["output_targets"])
        prog.output_targets.push_back(t.get<uint32_t>());
    if (jsched.contains("outputs"))
      for (const auto & o : jsched["outputs"])
        result.mix_indices.push_back(o.get<uint32_t>());
  }

  // Compute mix_output_temps: map mix_indices through output_targets.
  for (uint32_t idx : result.mix_indices)
  {
    if (idx < prog.output_targets.size())
      prog.mix_output_temps.push_back(prog.output_targets[idx]);
  }

  // ── Slot array ──
  if (plan.contains("slot_count") && plan["slot_count"].is_number())
    result.slot_count = plan["slot_count"].get<uint32_t>();
  if (plan.contains("slot_names"))
    for (const auto & n : plan["slot_names"])
      result.slot_names.push_back(n.get<std::string>());
  if (plan.contains("slot_defaults"))
    for (const auto & v : plan["slot_defaults"])
      result.slot_defaults.push_back(v.get<double>());

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

  if (plan.contains("state_init"))
  {
    for (const auto & v : plan["state_init"])
    {
      if (v.is_number())
        result.state_init.push_back(v.get<double>());
      else if (v.is_boolean())
        result.state_init.push_back(v.get<bool>() ? 1.0 : 0.0);
      else
        result.state_init.push_back(0.0);
    }
  }
  if (plan.contains("register_names"))
    for (const auto & n : plan["register_names"])
      result.register_names.push_back(n.get<std::string>());
  if (plan.contains("register_types"))
    for (const auto & t : plan["register_types"])
      result.register_types.push_back(parse_scalar_type(t.get<std::string>()));
  if (plan.contains("array_slot_names"))
    for (const auto & n : plan["array_slot_names"])
      result.array_slot_names.push_back(n.get<std::string>());

  auto & prog = result.program;
  prog.register_types = result.register_types;
  prog.register_count = plan.value("register_count", 0u);
  if (plan.contains("array_slot_sizes"))
    for (const auto & s : plan["array_slot_sizes"])
      prog.array_slot_sizes.push_back(s.get<uint32_t>());
  if (plan.contains("output_targets"))
    for (const auto & t : plan["output_targets"])
      prog.output_targets.push_back(t.get<uint32_t>());
  if (plan.contains("register_targets"))
    for (const auto & t : plan["register_targets"])
      prog.register_targets.push_back(t.get<int32_t>());
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

  // Lift the top-level register_targets[i] into writebacks. State
  // slots are 0..register_targets.size()-1 since the plan has only
  // one instance worth of state.
  for (uint32_t j = 0; j < prog.register_targets.size(); ++j)
  {
    inst.writebacks.push_back({j, prog.register_targets[j]});
  }

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
