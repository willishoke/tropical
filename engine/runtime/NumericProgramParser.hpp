#pragma once

/**
 * NumericProgramParser.hpp — Parse tropical_plan_4 JSON → FlatProgram.
 *
 * Thin deserialiser: no expression tree walking, no second compiler.
 * Reads the instruction stream emitted by compiler/emit_numeric.ts and
 * produces the tropical_jit::FlatProgram passed to compile_flat_program().
 */

#include "jit/OrcJitEngine.hpp"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace tropical_plan4
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
    {"TriggerParam",T::TriggerParam},
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
  throw std::runtime_error("NumericProgramParser: unknown operand kind '" + kind + "'");
}

// ─────────────────────────────────────────────────────────────────────────────
// Parsed plan_4 result
// ─────────────────────────────────────────────────────────────────────────────

struct ParsedPlan4
{
  tropical_jit::FlatProgram  program;
  std::vector<double>      state_init;
  std::vector<std::string> register_names;
  std::vector<tropical_jit::JitScalarType> register_types;
  std::vector<std::string> array_slot_names;
  std::vector<uint32_t>    mix_indices;
  double                   sample_rate = 44100.0;
};

inline ParsedPlan4 parse_plan4(const nlohmann::json & plan)
{
  ParsedPlan4 result;

  result.sample_rate = plan.value("config", nlohmann::json::object())
                           .value("sampleRate", 44100.0);

  // state_init — scalars only; arrays (not used by current modules) are zeroed
  if (plan.contains("state_init"))
  {
    for (const auto & v : plan["state_init"])
    {
      if (v.is_number())
        result.state_init.push_back(v.get<double>());
      else if (v.is_boolean())
        result.state_init.push_back(v.get<bool>() ? 1.0 : 0.0);
      else
        result.state_init.push_back(0.0);  // array or unknown — zeroed
    }
  }

  if (plan.contains("register_names"))
    for (const auto & n : plan["register_names"])
      result.register_names.push_back(n.get<std::string>());

  if (plan.contains("register_types"))
    for (const auto & t : plan["register_types"])
      result.register_types.push_back(parse_scalar_type(t.get<std::string>()));

  // Forward register_types onto FlatProgram so the JIT can coerce at writeback.
  result.program.register_types = result.register_types;

  if (plan.contains("array_slot_names"))
    for (const auto & n : plan["array_slot_names"])
      result.array_slot_names.push_back(n.get<std::string>());

  if (plan.contains("outputs"))
    for (const auto & o : plan["outputs"])
      result.mix_indices.push_back(o.get<uint32_t>());

  // FlatProgram fields
  auto & prog = result.program;
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

  // Compute mix_output_temps: map mix_indices through output_targets
  for (uint32_t idx : result.mix_indices)
  {
    if (idx < prog.output_targets.size())
      prog.mix_output_temps.push_back(prog.output_targets[idx]);
  }

  if (plan.contains("instructions"))
  {
    for (const auto & ji : plan["instructions"])
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

      if (ji.contains("group_id") && ji["group_id"].is_string())
        instr.group_id = ji["group_id"].get<std::string>();

      prog.instructions.push_back(std::move(instr));
    }
  }

  // Gateable-subgraph groups table — present only when source_tag wrappers
  // survived the pipeline. Phase 7 reads the field; Phase 8 consumes it at
  // codegen time.
  if (plan.contains("groups"))
  {
    for (const auto & jg : plan["groups"])
    {
      tropical_jit::GroupInfo g;
      g.id = jg.at("id").get<std::string>();
      g.gate_operand = parse_operand(jg.at("gate_operand"));
      prog.groups.push_back(std::move(g));
    }
  }

  return result;
}

} // namespace tropical_plan4
