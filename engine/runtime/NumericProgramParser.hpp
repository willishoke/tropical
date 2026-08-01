#pragma once

/**
 * NumericProgramParser.hpp — Parse `tropical_plan_5` / `_6` JSON metadata.
 *
 * Thin deserialiser. Since Phase 2 (Lean owns codegen) this is the
 * *manifest reader*: load_ir consumes the plan's metadata — temp pool size
 * (register_count), array / slot names + sizes, sample_rate, slot defaults —
 * to set up runtime scratch and slots. CF-only has no per-sample state.
 * Retired schemas and state/output compatibility fields are rejected rather
 * than translated or ignored. The instruction stream still parses into
 * FlatProgram's fields but is no longer used for codegen (the kernel comes
 * from Lean-emitted IR).
 */

#include "jit/OrcJitEngine.hpp"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
  if (s == "float") return tropical_jit::JitScalarType::Float;
  if (s == "int")  return tropical_jit::JitScalarType::Int;
  if (s == "bool") return tropical_jit::JitScalarType::Bool;
  throw std::runtime_error(
    "NumericProgramParser: unknown scalar type '" + s + "'");
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

inline tropical_jit::FlatInstr parse_instr(const nlohmann::json & ji)
{
  tropical_jit::FlatInstr instr;
  instr.tag         = parse_op_tag(ji.at("tag").get<std::string>());
  instr.result_type = parse_scalar_type(ji.value("result_type", "float"));
  instr.dst         = ji.at("dst").get<uint32_t>();
  instr.loop_count  = ji.value("loop_count", 1u);
  if (!ji.contains("dst_kind"))
    throw std::runtime_error(
      "NumericProgramParser: plan-5 instruction missing required dst_kind");
  instr.dst_kind = parse_dst_kind(ji["dst_kind"].get<std::string>());
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
  double                                   sample_rate = 44100.0;

  // Engine realization strategy. Canonical fused plans omit the field.
  tropical_jit::CompilationMode compilation_mode = tropical_jit::CompilationMode::Fused;

  // Inter-module slot array.
  uint32_t                 slot_count = 0;
  std::vector<std::string> slot_names;
  std::vector<double>      slot_defaults;

  // Host-contract param write dispatch (design/host-param-dispatch.md): the
  // discipline is a fact about the compiled patch, carried by the plan, so a
  // host dispatches writes itself and no client ever chooses a verb. An absent
  // field is the canonical empty table.
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
  // move. An absent field is the canonical empty table.
  std::vector<uint32_t> coeff_array_slots;

  struct ImmutableAssetArray
  {
    uint32_t    slot = 0;
    uint64_t    byte_offset = 0;
    uint64_t    element_count = 0;
    std::string scalar_format;
  };

  struct ImmutableAsset
  {
    std::string path;
    uint64_t byte_count = 0;
    std::string sha256;
    uint32_t sample_rate = 0;
    std::vector<ImmutableAssetArray> arrays;
  };
  std::vector<ImmutableAsset> immutable_assets;
};

inline void validate_immutable_assets(const ParsedPlan5 & plan)
{
  std::unordered_set<uint32_t> bound_slots;
  for (const auto & asset : plan.immutable_assets)
  {
    const std::filesystem::path path(asset.path);
    if (asset.path.empty())
      throw std::runtime_error(
        "NumericProgramParser: immutable asset path is empty");
    if (path.is_absolute())
      throw std::runtime_error(
        "NumericProgramParser: immutable asset path must be package-relative");
    for (const auto & component : path)
      if (component == "..")
        throw std::runtime_error(
          "NumericProgramParser: immutable asset path may not contain '..'");
    if (asset.byte_count == 0)
      throw std::runtime_error(
        "NumericProgramParser: immutable asset byte_count must be nonzero");
    if (asset.sample_rate != plan.sample_rate)
      throw std::runtime_error(
        "NumericProgramParser: immutable asset sample_rate mismatch");
    if (asset.arrays.empty())
      throw std::runtime_error(
        "NumericProgramParser: immutable asset has no array_slots");
    for (const auto & binding : asset.arrays)
    {
      if (binding.slot >= plan.program.array_slot_sizes.size())
        throw std::runtime_error(
          "NumericProgramParser: immutable asset array slot is out of range");
      if (!bound_slots.insert(binding.slot).second)
        throw std::runtime_error(
          "NumericProgramParser: immutable asset array slot is bound more than once");
      if (std::find(plan.coeff_array_slots.begin(),
                    plan.coeff_array_slots.end(), binding.slot)
          != plan.coeff_array_slots.end())
        throw std::runtime_error(
          "NumericProgramParser: immutable asset slot cannot be a coefficient column");
      if (binding.scalar_format != "float32")
        throw std::runtime_error(
          "NumericProgramParser: immutable asset scalar_format must be 'float32'");
      if (binding.element_count == 0
          || binding.element_count != plan.program.array_slot_sizes[binding.slot])
        throw std::runtime_error(
          "NumericProgramParser: immutable asset element_count does not match array slot size");
      if (binding.byte_offset % sizeof(float) != 0
          || binding.byte_offset > asset.byte_count
          || binding.element_count >
               (asset.byte_count - binding.byte_offset) / sizeof(float))
        throw std::runtime_error(
          "NumericProgramParser: immutable asset array byte range is invalid");
    }
  }
}

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

inline void reject_retired_fields(const nlohmann::json & plan)
{
  static constexpr const char * retired[] = {
    "state_init",
    "register_names",
    "register_types",
    "register_targets",
    "state_reg_offset",
    "output_targets",
    "outputs",
    "instructions",
    "scheduler_function",
  };
  for (const char * field : retired)
    if (plan.contains(field))
      throw std::runtime_error(
        std::string{"NumericProgramParser: retired field '"} + field +
        "' is not valid in tropical_plan_5/6");
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
  // Slot-based parent→child input wiring. Empty arrays are omitted by the
  // canonical Plan-5 encoder. Same instruction
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

/** Parse a tropical_plan_5 or tropical_plan_6 JSON object. The C++ side supports nested
 *  `instance_functions` (the fractal architecture): each function may
 *  have `children: InstanceFunction[]` that emit recursively inside the
 *  parent's body. Canonical leaf kernels omit `children`. */
inline ParsedPlan5 parse_plan5(const nlohmann::json & plan)
{
  ParsedPlan5 result;
  const std::string schema = plan.value("schema", std::string{});
  if (schema != "tropical_plan_5" && schema != "tropical_plan_6")
    throw std::runtime_error(
      "NumericProgramParser: unsupported schema '" + schema
      + "'; expected 'tropical_plan_5' or 'tropical_plan_6'");
  reject_retired_fields(plan);

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

  // ── sources: runtime-bound inputs (the dual of sinks). The Plan-5 encoder
  //    omits the canonical pair [tick, rate], which is restored here. ──
  if (plan.contains("sources"))
  {
    for (const auto & js : plan["sources"])
    {
      tropical_jit::Source src;
      const std::string k = js.get<std::string>();
      if (k == "tick")
        src.kind = tropical_jit::SourceKind::Tick;
      else if (k == "rate")
        src.kind = tropical_jit::SourceKind::Rate;
      else
        throw std::runtime_error(
          "NumericProgramParser: unknown source kind '" + k + "'");
      prog.sources.push_back(src);
    }
  }
  else
  {
    prog.sources.push_back({ tropical_jit::SourceKind::Tick });
    prog.sources.push_back({ tropical_jit::SourceKind::Rate });
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

  if (schema == "tropical_plan_5"
      && plan.contains("immutable_assets")
      && !plan["immutable_assets"].empty())
    throw std::runtime_error(
      "NumericProgramParser: tropical_plan_5 cannot carry immutable_assets");
  if (schema == "tropical_plan_6"
      && (!plan.contains("immutable_assets")
          || !plan["immutable_assets"].is_array()
          || plan["immutable_assets"].empty()))
    throw std::runtime_error(
      "NumericProgramParser: tropical_plan_6 requires nonempty immutable_assets");

  if (plan.contains("immutable_assets"))
    for (const auto & a : plan["immutable_assets"])
    {
      ParsedPlan5::ImmutableAsset asset;
      asset.path = a.at("path").get<std::string>();
      asset.byte_count = a.at("byte_count").get<uint64_t>();
      asset.sha256 = a.at("sha256").get<std::string>();
      asset.sample_rate = a.at("sample_rate").get<uint32_t>();
      if (!a.contains("array_slots") || !a["array_slots"].is_array())
        throw std::runtime_error(
          "NumericProgramParser: immutable asset missing array_slots");
      for (const auto & b : a["array_slots"])
      {
        ParsedPlan5::ImmutableAssetArray binding;
        binding.slot = b.at("slot").get<uint32_t>();
        binding.byte_offset = b.at("byte_offset").get<uint64_t>();
        binding.element_count = b.at("element_count").get<uint64_t>();
        binding.scalar_format = b.at("scalar_format").get<std::string>();
        asset.arrays.push_back(std::move(binding));
      }
      result.immutable_assets.push_back(std::move(asset));
    }

  validate_immutable_assets(result);

  return result;
}

} // namespace tropical_plan5
