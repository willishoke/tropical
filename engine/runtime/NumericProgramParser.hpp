#pragma once

/**
 * NumericProgramParser.hpp — Parse egress_plan_3 JSON → FlatProgram.
 *
 * Thin deserialiser: no expression tree walking, no second compiler.
 * Reads the instruction stream emitted by compiler/emit_numeric.ts and
 * produces the egress_jit::FlatProgram passed to compile_flat_program().
 */

#include "jit/OrcJitEngine.hpp"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace egress_plan3
{

// ─────────────────────────────────────────────────────────────────────────────
// Tag / operand parsing helpers
// ─────────────────────────────────────────────────────────────────────────────

inline egress_jit::OpTag parse_op_tag(const std::string & s)
{
  using T = egress_jit::OpTag;
  static const std::unordered_map<std::string, T> MAP = {
    {"Add",         T::Add},
    {"Sub",         T::Sub},
    {"Mul",         T::Mul},
    {"Div",         T::Div},
    {"Mod",         T::Mod},
    {"Pow",         T::Pow},
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
    {"MatMul",      T::MatMul},
    {"Neg",         T::Neg},
    {"Abs",         T::Abs},
    {"Sin",         T::Sin},
    {"Cos",         T::Cos},
    {"Log",         T::Log},
    {"Exp",         T::Exp},
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
    {"SmoothParam", T::SmoothParam},
    {"TriggerParam",T::TriggerParam},
  };
  const auto it = MAP.find(s);
  if (it == MAP.end())
    throw std::runtime_error("NumericProgramParser: unknown tag '" + s + "'");
  return it->second;
}

inline egress_jit::Operand parse_operand(const nlohmann::json & j)
{
  const std::string kind = j.at("kind").get<std::string>();
  if (kind == "const")
    return egress_jit::Operand::make_const(j.at("val").get<double>());
  if (kind == "input")
    return egress_jit::Operand::make_input(j.at("slot").get<uint32_t>());
  if (kind == "reg")
    return egress_jit::Operand::make_reg(j.at("slot").get<uint32_t>());
  if (kind == "array_reg")
    return egress_jit::Operand::make_array_reg(j.at("slot").get<uint32_t>());
  if (kind == "state_reg")
    return egress_jit::Operand::make_state(j.at("slot").get<uint32_t>());
  if (kind == "param")
  {
    const uint64_t ptr = std::stoull(j.at("ptr").get<std::string>());
    return egress_jit::Operand::make_param(ptr);
  }
  if (kind == "rate") return egress_jit::Operand::make_rate();
  if (kind == "tick") return egress_jit::Operand::make_tick();
  throw std::runtime_error("NumericProgramParser: unknown operand kind '" + kind + "'");
}

// ─────────────────────────────────────────────────────────────────────────────
// Parsed plan_3 result
// ─────────────────────────────────────────────────────────────────────────────

struct ParsedPlan3
{
  egress_jit::FlatProgram  program;
  std::vector<double>      state_init;
  std::vector<std::string> register_names;
  std::vector<uint32_t>    mix_indices;
  double                   sample_rate = 44100.0;
};

inline ParsedPlan3 parse_plan3(const nlohmann::json & plan)
{
  ParsedPlan3 result;

  result.sample_rate = plan.value("config", nlohmann::json::object())
                           .value("sample_rate", 44100.0);

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
      egress_jit::FlatInstr instr;
      instr.tag        = parse_op_tag(ji.at("tag").get<std::string>());
      instr.dst        = ji.at("dst").get<uint32_t>();
      instr.loop_count = ji.value("loop_count", 1u);

      if (ji.contains("args"))
        for (const auto & a : ji["args"])
          instr.args.push_back(parse_operand(a));

      if (ji.contains("strides"))
        for (const auto & s : ji["strides"])
          instr.strides.push_back(s.get<uint8_t>());

      prog.instructions.push_back(std::move(instr));
    }
  }

  return result;
}

} // namespace egress_plan3
