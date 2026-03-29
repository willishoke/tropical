#include "c_api/egress_c.h"

#include "expr/Expr.hpp"
#include "graph/Graph.hpp"
#include "graph/Module.hpp"
#include "dac/EgressDAC.hpp"

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace expr = egress_expr;
using ExprSpecPtr = expr::ExprSpecPtr;
using Value = expr::Value;

// ---------- Thread-local error string ----------

static thread_local std::string tls_last_error;

static void set_error(const std::string & msg) { tls_last_error = msg; }

extern "C" const char* egress_last_error(void)
{
  return tls_last_error.c_str();
}

// ---------- Opaque wrapper types ----------

struct EgressExpr
{
  ExprSpecPtr spec;
};

struct EgressParam
{
  expr::ControlParam * param;
  EgressParam(double init, double tc) : param(new expr::ControlParam(init, tc)) {}
  ~EgressParam() { delete param; }
  // Non-copyable
  EgressParam(const EgressParam &) = delete;
  EgressParam & operator=(const EgressParam &) = delete;
};

struct EgressValue
{
  Value value;
};

struct EgressModuleSpecBuilder
{
  unsigned int input_count = 0;
  double sample_rate = 44100.0;
  std::vector<ExprSpecPtr> output_exprs;
  std::vector<ExprSpecPtr> register_exprs;
  std::vector<Value> initial_registers;
  std::vector<Module::DelayStateSpec> delay_state_specs;
  std::vector<Module::NestedModuleSpec> nested_module_specs;
  std::vector<uint32_t> composite_schedule;
  uint32_t output_boundary_id = std::numeric_limits<uint32_t>::max();
  uint32_t next_node_id = 0;
};

struct EgressNestedSpecBuilder
{
  uint32_t node_id = 0;
  std::string label;
  unsigned int input_count = 0;
  double sample_rate = 44100.0;
  std::vector<ExprSpecPtr> input_exprs;
  std::vector<ExprSpecPtr> output_exprs;
  std::vector<ExprSpecPtr> register_exprs;
  std::vector<Value> initial_registers;
  std::vector<Module::DelayStateSpec> delay_state_specs;
  std::vector<Module::NestedModuleSpec> nested_module_specs;
  std::vector<uint32_t> composite_schedule;
  uint32_t output_boundary_id = std::numeric_limits<uint32_t>::max();
  uint32_t next_node_id = 0;
};

// Global counter for nested spec node IDs (avoids collisions across specs)
static std::atomic<uint32_t> g_nested_node_counter{0};

// Fixed sentinel value for the output boundary node ID used in auto-computed schedules.
// Chosen to be far above any realistic nested-module node counter value.
static constexpr uint32_t kAutoOutputBoundaryId = 0x7FFFFFFEu;

// Compute composite_schedule and output_boundary_id when the caller hasn't set them
// but the module has delays or nested sub-modules that require a composite runtime.
static void maybe_fill_composite_schedule(
  std::vector<uint32_t> & composite_schedule,
  uint32_t & output_boundary_id,
  const std::vector<Module::DelayStateSpec> & delay_state_specs,
  const std::vector<Module::NestedModuleSpec> & nested_module_specs)
{
  const bool needs_composite = !delay_state_specs.empty() || !nested_module_specs.empty();
  if (!needs_composite || !composite_schedule.empty())
  {
    return;  // nothing to do
  }
  output_boundary_id = kAutoOutputBoundaryId;
  for (const auto & ns : nested_module_specs)
  {
    composite_schedule.push_back(ns.node_id);
  }
  composite_schedule.push_back(output_boundary_id);
}

// Helper: materialise a NestedModuleSpec from a builder
static Module::NestedModuleSpec nested_spec_from_builder(const EgressNestedSpecBuilder* n)
{
  Module::NestedModuleSpec ns;
  ns.node_id = n->node_id;
  ns.label = n->label;
  ns.input_count = n->input_count;
  ns.input_exprs = n->input_exprs;
  ns.output_exprs = n->output_exprs;
  ns.register_exprs = n->register_exprs;
  ns.initial_registers = n->initial_registers;
  ns.delay_state_specs = n->delay_state_specs;
  ns.nested_module_specs = n->nested_module_specs;
  ns.composite_schedule = n->composite_schedule;
  ns.output_boundary_id = n->output_boundary_id;
  ns.sample_rate = n->sample_rate;
  maybe_fill_composite_schedule(
    ns.composite_schedule, ns.output_boundary_id,
    ns.delay_state_specs, ns.nested_module_specs);
  return ns;
}

// Helper: cast ExprKind int to enum
static expr::ExprKind kind_from_int(int kind)
{
  return static_cast<expr::ExprKind>(kind);
}

// ============================================================
// C API implementation
// ============================================================

extern "C" {

// ---------- Value API ----------

egress_value_t egress_value_float(double v)
{
  try { return new EgressValue{expr::float_value(v)}; }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

egress_value_t egress_value_int(int64_t v)
{
  try { return new EgressValue{expr::int_value(v)}; }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

egress_value_t egress_value_bool(bool v)
{
  try { return new EgressValue{expr::bool_value(v)}; }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

egress_value_t egress_value_array(const egress_value_t* items, size_t n)
{
  try
  {
    std::vector<Value> vals;
    vals.reserve(n);
    for (size_t i = 0; i < n; ++i)
    {
      vals.push_back(static_cast<EgressValue*>(items[i])->value);
    }
    return new EgressValue{expr::array_value(std::move(vals))};
  }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

egress_value_t egress_value_matrix(const egress_value_t* items, size_t rows, size_t cols)
{
  try
  {
    std::vector<Value> vals;
    vals.reserve(rows * cols);
    for (size_t i = 0; i < rows * cols; ++i)
    {
      vals.push_back(static_cast<EgressValue*>(items[i])->value);
    }
    return new EgressValue{expr::matrix_value(rows, cols, std::move(vals))};
  }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

double egress_value_to_float(egress_value_t v)
{
  try { return expr::to_float64(static_cast<EgressValue*>(v)->value); }
  catch (const std::exception & e) { set_error(e.what()); return 0.0; }
}

int64_t egress_value_to_int(egress_value_t v)
{
  try { return expr::to_int64(static_cast<EgressValue*>(v)->value); }
  catch (const std::exception & e) { set_error(e.what()); return 0; }
}

void egress_value_free(egress_value_t v)
{
  delete static_cast<EgressValue*>(v);
}

// ---------- ControlParam API ----------

egress_param_t egress_param_new(double init_value, double time_const)
{
  try { return new EgressParam(init_value, time_const); }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

void egress_param_free(egress_param_t p)
{
  delete static_cast<EgressParam *>(p);
}

void egress_param_set(egress_param_t p, double value)
{
  if (p)
  {
    static_cast<EgressParam *>(p)->param->value.store(value, std::memory_order_relaxed);
  }
}

double egress_param_get(egress_param_t p)
{
  if (!p) return 0.0;
  return static_cast<EgressParam *>(p)->param->value.load(std::memory_order_relaxed);
}

egress_expr_t egress_expr_param(egress_param_t p)
{
  try
  {
    if (!p)
    {
      set_error("egress_expr_param: null param handle");
      return nullptr;
    }
    return new EgressExpr{expr::smoothed_param_expr(static_cast<EgressParam *>(p)->param)};
  }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

egress_param_t egress_param_new_trigger(void)
{
  try { return new EgressParam(0.0, 0.0); }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

egress_expr_t egress_expr_trigger_param(egress_param_t p)
{
  try
  {
    if (!p)
    {
      set_error("egress_expr_trigger_param: null param handle");
      return nullptr;
    }
    return new EgressExpr{expr::trigger_param_expr(static_cast<EgressParam *>(p)->param)};
  }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

// ---------- Expression factory API ----------

egress_expr_t egress_expr_literal_float(double v)
{
  try { return new EgressExpr{expr::literal_expr(v)}; }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

egress_expr_t egress_expr_literal_int(int64_t v)
{
  try { return new EgressExpr{expr::literal_expr(v)}; }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

egress_expr_t egress_expr_literal_bool(bool v)
{
  try { return new EgressExpr{expr::literal_expr(v)}; }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

egress_expr_t egress_expr_literal_value(egress_value_t v)
{
  try { return new EgressExpr{expr::literal_expr(static_cast<EgressValue*>(v)->value)}; }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

egress_expr_t egress_expr_input(unsigned int input_id)
{
  try { return new EgressExpr{expr::input_value_expr(input_id)}; }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

egress_expr_t egress_expr_register(unsigned int reg_id)
{
  try { return new EgressExpr{expr::register_value_expr(reg_id)}; }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

egress_expr_t egress_expr_nested_output(unsigned int node_id, unsigned int output_id)
{
  try { return new EgressExpr{expr::nested_value_expr(node_id, output_id)}; }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

egress_expr_t egress_expr_delay_value(unsigned int node_id)
{
  try { return new EgressExpr{expr::delay_value_expr(node_id)}; }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

egress_expr_t egress_expr_ref(const char* module_name, unsigned int output_id)
{
  try { return new EgressExpr{expr::ref_expr(std::string(module_name), output_id)}; }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

egress_expr_t egress_expr_sample_rate(void)
{
  try { return new EgressExpr{expr::sample_rate_expr()}; }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

egress_expr_t egress_expr_sample_index(void)
{
  try { return new EgressExpr{expr::sample_index_expr()}; }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

egress_expr_t egress_expr_unary(int kind, egress_expr_t operand)
{
  try
  {
    return new EgressExpr{
      expr::unary_expr(kind_from_int(kind), static_cast<EgressExpr*>(operand)->spec)};
  }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

egress_expr_t egress_expr_binary(int kind, egress_expr_t lhs, egress_expr_t rhs)
{
  try
  {
    return new EgressExpr{expr::binary_expr(
      kind_from_int(kind),
      static_cast<EgressExpr*>(lhs)->spec,
      static_cast<EgressExpr*>(rhs)->spec)};
  }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

egress_expr_t egress_expr_clamp(egress_expr_t v, egress_expr_t lo, egress_expr_t hi)
{
  try
  {
    return new EgressExpr{expr::clamp_expr(
      static_cast<EgressExpr*>(v)->spec,
      static_cast<EgressExpr*>(lo)->spec,
      static_cast<EgressExpr*>(hi)->spec)};
  }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

egress_expr_t egress_expr_select(egress_expr_t cond, egress_expr_t then_expr, egress_expr_t else_expr)
{
  try {
    return new EgressExpr{expr::select_expr(
      static_cast<EgressExpr*>(cond)->spec,
      static_cast<EgressExpr*>(then_expr)->spec,
      static_cast<EgressExpr*>(else_expr)->spec)};
  }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

egress_expr_t egress_expr_array_pack(const egress_expr_t* items, size_t n)
{
  try
  {
    std::vector<ExprSpecPtr> specs;
    specs.reserve(n);
    for (size_t i = 0; i < n; ++i)
    {
      specs.push_back(static_cast<EgressExpr*>(items[i])->spec);
    }
    return new EgressExpr{expr::array_pack_expr(std::move(specs))};
  }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

egress_expr_t egress_expr_index(egress_expr_t arr, egress_expr_t idx)
{
  try
  {
    return new EgressExpr{expr::index_expr(
      static_cast<EgressExpr*>(arr)->spec,
      static_cast<EgressExpr*>(idx)->spec)};
  }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

egress_expr_t egress_expr_array_set(egress_expr_t arr, egress_expr_t idx, egress_expr_t val)
{
  try
  {
    return new EgressExpr{expr::array_set_expr(
      static_cast<EgressExpr*>(arr)->spec,
      static_cast<EgressExpr*>(idx)->spec,
      static_cast<EgressExpr*>(val)->spec)};
  }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

egress_expr_t egress_expr_function(unsigned int param_count, egress_expr_t body)
{
  try
  {
    return new EgressExpr{
      expr::function_expr(param_count, static_cast<EgressExpr*>(body)->spec)};
  }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

egress_expr_t egress_expr_call(egress_expr_t callee, const egress_expr_t* args, size_t n)
{
  try
  {
    std::vector<ExprSpecPtr> arg_specs;
    arg_specs.reserve(n);
    for (size_t i = 0; i < n; ++i)
    {
      arg_specs.push_back(static_cast<EgressExpr*>(args[i])->spec);
    }
    return new EgressExpr{expr::call_expr(
      static_cast<EgressExpr*>(callee)->spec,
      std::move(arg_specs))};
  }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

void egress_expr_free(egress_expr_t e)
{
  delete static_cast<EgressExpr*>(e);
}

// ---------- Module spec builder API ----------

egress_module_spec_t egress_module_spec_new(unsigned int input_count, double sample_rate)
{
  try
  {
    auto* b = new EgressModuleSpecBuilder;
    b->input_count = input_count;
    b->sample_rate = sample_rate;
    return b;
  }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

void egress_module_spec_add_output(egress_module_spec_t spec, egress_expr_t expr)
{
  static_cast<EgressModuleSpecBuilder*>(spec)->output_exprs.push_back(
    static_cast<EgressExpr*>(expr)->spec);
}

void egress_module_spec_add_register(
  egress_module_spec_t spec,
  egress_expr_t body,
  egress_value_t init)
{
  auto* b = static_cast<EgressModuleSpecBuilder*>(spec);
  b->register_exprs.push_back(static_cast<EgressExpr*>(body)->spec);
  b->initial_registers.push_back(static_cast<EgressValue*>(init)->value);
}

unsigned int egress_module_spec_add_delay_state(
  egress_module_spec_t spec,
  egress_value_t init,
  egress_expr_t update_expr)
{
  auto* b = static_cast<EgressModuleSpecBuilder*>(spec);
  const uint32_t node_id = b->next_node_id++;
  Module::DelayStateSpec ds;
  ds.node_id = node_id;
  ds.initial_value = static_cast<EgressValue*>(init)->value;
  ds.update_expr = static_cast<EgressExpr*>(update_expr)->spec;
  b->delay_state_specs.push_back(std::move(ds));
  return node_id;
}

void egress_module_spec_add_nested(egress_module_spec_t spec, egress_nested_spec_t nested)
{
  auto* b = static_cast<EgressModuleSpecBuilder*>(spec);
  auto* n = static_cast<EgressNestedSpecBuilder*>(nested);
  b->nested_module_specs.push_back(nested_spec_from_builder(n));
}

void egress_module_spec_set_composite_schedule(
  egress_module_spec_t spec,
  const unsigned int* schedule,
  size_t n)
{
  auto* b = static_cast<EgressModuleSpecBuilder*>(spec);
  b->composite_schedule.assign(schedule, schedule + n);
}

void egress_module_spec_set_output_boundary(egress_module_spec_t spec, unsigned int boundary_id)
{
  static_cast<EgressModuleSpecBuilder*>(spec)->output_boundary_id = boundary_id;
}

void egress_module_spec_free(egress_module_spec_t spec)
{
  delete static_cast<EgressModuleSpecBuilder*>(spec);
}

// ---------- Nested spec builder API ----------

egress_nested_spec_t egress_nested_spec_new(unsigned int input_count, double sample_rate)
{
  try
  {
    auto* b = new EgressNestedSpecBuilder;
    b->node_id = g_nested_node_counter.fetch_add(1, std::memory_order_relaxed);
    b->input_count = input_count;
    b->sample_rate = sample_rate;
    return b;
  }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

unsigned int egress_nested_spec_node_id(egress_nested_spec_t nested)
{
  return static_cast<EgressNestedSpecBuilder*>(nested)->node_id;
}

void egress_nested_spec_add_input_expr(egress_nested_spec_t nested, egress_expr_t expr)
{
  static_cast<EgressNestedSpecBuilder*>(nested)->input_exprs.push_back(
    static_cast<EgressExpr*>(expr)->spec);
}

void egress_nested_spec_add_output(egress_nested_spec_t nested, egress_expr_t expr)
{
  static_cast<EgressNestedSpecBuilder*>(nested)->output_exprs.push_back(
    static_cast<EgressExpr*>(expr)->spec);
}

void egress_nested_spec_add_register(
  egress_nested_spec_t nested,
  egress_expr_t body,
  egress_value_t init)
{
  auto* b = static_cast<EgressNestedSpecBuilder*>(nested);
  b->register_exprs.push_back(static_cast<EgressExpr*>(body)->spec);
  b->initial_registers.push_back(static_cast<EgressValue*>(init)->value);
}

unsigned int egress_nested_spec_add_delay_state(
  egress_nested_spec_t nested,
  egress_value_t init,
  egress_expr_t update_expr)
{
  auto* b = static_cast<EgressNestedSpecBuilder*>(nested);
  const uint32_t node_id = b->next_node_id++;
  Module::DelayStateSpec ds;
  ds.node_id = node_id;
  ds.initial_value = static_cast<EgressValue*>(init)->value;
  ds.update_expr = static_cast<EgressExpr*>(update_expr)->spec;
  b->delay_state_specs.push_back(std::move(ds));
  return node_id;
}

void egress_nested_spec_add_nested(egress_nested_spec_t outer, egress_nested_spec_t inner)
{
  auto* b = static_cast<EgressNestedSpecBuilder*>(outer);
  auto* n = static_cast<EgressNestedSpecBuilder*>(inner);
  b->nested_module_specs.push_back(nested_spec_from_builder(n));
}

void egress_nested_spec_set_composite_schedule(
  egress_nested_spec_t nested,
  const unsigned int* schedule,
  size_t n)
{
  auto* b = static_cast<EgressNestedSpecBuilder*>(nested);
  b->composite_schedule.assign(schedule, schedule + n);
}

void egress_nested_spec_set_output_boundary(
  egress_nested_spec_t nested,
  unsigned int boundary_id)
{
  static_cast<EgressNestedSpecBuilder*>(nested)->output_boundary_id = boundary_id;
}

void egress_nested_spec_free(egress_nested_spec_t nested)
{
  delete static_cast<EgressNestedSpecBuilder*>(nested);
}

// ---------- Graph API ----------

egress_graph_t egress_graph_new(unsigned int buffer_length)
{
  try { return new Graph(buffer_length); }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

void egress_graph_free(egress_graph_t g)
{
  delete static_cast<Graph*>(g);
}

bool egress_graph_add_module(egress_graph_t g, const char* name, egress_module_spec_t spec)
{
  try
  {
    auto* graph = static_cast<Graph*>(g);
    auto* b = static_cast<EgressModuleSpecBuilder*>(spec);
    std::vector<uint32_t> composite_schedule = b->composite_schedule;
    uint32_t output_boundary_id = b->output_boundary_id;
    maybe_fill_composite_schedule(
      composite_schedule, output_boundary_id,
      b->delay_state_specs, b->nested_module_specs);
    return graph->addModule(
      std::string(name),
      std::make_unique<Module>(
        b->input_count,
        b->output_exprs,
        b->register_exprs,
        b->initial_registers,
        b->delay_state_specs,
        b->nested_module_specs,
        composite_schedule,
        output_boundary_id,
        b->sample_rate));
  }
  catch (const std::exception & e) { set_error(e.what()); return false; }
}

bool egress_graph_remove_module(egress_graph_t g, const char* name)
{
  try { return static_cast<Graph*>(g)->remove_module(std::string(name)); }
  catch (const std::exception & e) { set_error(e.what()); return false; }
}

bool egress_graph_connect(
  egress_graph_t g,
  const char* src,
  unsigned int src_out,
  const char* dst,
  unsigned int dst_in)
{
  try
  {
    return static_cast<Graph*>(g)->connect(
      std::string(src), src_out, std::string(dst), dst_in);
  }
  catch (const std::exception & e) { set_error(e.what()); return false; }
}

bool egress_graph_disconnect(
  egress_graph_t g,
  const char* src,
  unsigned int src_out,
  const char* dst,
  unsigned int dst_in)
{
  try
  {
    return static_cast<Graph*>(g)->remove_connection(
      std::string(src), src_out, std::string(dst), dst_in);
  }
  catch (const std::exception & e) { set_error(e.what()); return false; }
}

bool egress_graph_set_input_expr(
  egress_graph_t g,
  const char* module,
  unsigned int input_id,
  egress_expr_t e)
{
  try
  {
    ExprSpecPtr spec = e ? static_cast<EgressExpr*>(e)->spec : nullptr;
    return static_cast<Graph*>(g)->set_input_expr(
      std::string(module), input_id, std::move(spec));
  }
  catch (const std::exception & e2) { set_error(e2.what()); return false; }
}

void egress_graph_begin_update(egress_graph_t g)
{
  try { static_cast<Graph*>(g)->begin_update(); }
  catch (const std::exception & e) { set_error(e.what()); }
}

bool egress_graph_end_update(egress_graph_t g)
{
  try { return static_cast<Graph*>(g)->end_update(); }
  catch (const std::exception & e) { set_error(e.what()); return false; }
}

egress_expr_t egress_graph_get_input_expr(
  egress_graph_t g,
  const char* module,
  unsigned int input_id)
{
  try
  {
    auto spec = static_cast<Graph*>(g)->get_input_expr(std::string(module), input_id);
    if (!spec)
    {
      return nullptr;
    }
    return new EgressExpr{std::move(spec)};
  }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

bool egress_graph_add_output(egress_graph_t g, const char* module, unsigned int output_id)
{
  try
  {
    return static_cast<Graph*>(g)->addOutput(
      std::make_pair(std::string(module), output_id));
  }
  catch (const std::exception & e) { set_error(e.what()); return false; }
}

bool egress_graph_add_output_expr(egress_graph_t g, egress_expr_t e)
{
  try
  {
    return static_cast<Graph*>(g)->addOutputExpr(static_cast<EgressExpr*>(e)->spec);
  }
  catch (const std::exception & e2) { set_error(e2.what()); return false; }
}

size_t egress_graph_add_output_tap(egress_graph_t g, const char* module, unsigned int output_id)
{
  try
  {
    return static_cast<Graph*>(g)->addOutputTap(std::string(module), output_id);
  }
  catch (const std::exception & e) { set_error(e.what()); return static_cast<size_t>(-1); }
}

bool egress_graph_remove_output_tap(egress_graph_t g, size_t tap_id)
{
  try { return static_cast<Graph*>(g)->removeOutputTap(tap_id); }
  catch (const std::exception & e) { set_error(e.what()); return false; }
}

void egress_graph_process(egress_graph_t g)
{
  try { static_cast<Graph*>(g)->process(); }
  catch (const std::exception & e) { set_error(e.what()); }
}

void egress_graph_prime_jit(egress_graph_t g)
{
  try { static_cast<Graph*>(g)->prime_numeric_jit(); }
  catch (const std::exception & e) { set_error(e.what()); }
}

const double* egress_graph_output_buffer(egress_graph_t g)
{
  return static_cast<Graph*>(g)->outputBuffer.data();
}

const double* egress_graph_tap_buffer(egress_graph_t g, size_t tap_id, size_t* out_len)
{
  try
  {
    static thread_local std::vector<double> tap_buffer;
    tap_buffer = static_cast<Graph*>(g)->outputTapBuffer(tap_id);
    if (out_len)
    {
      *out_len = tap_buffer.size();
    }
    return tap_buffer.data();
  }
  catch (const std::exception & e)
  {
    set_error(e.what());
    if (out_len)
    {
      *out_len = 0;
    }
    return nullptr;
  }
}

void egress_graph_set_worker_count(egress_graph_t g, unsigned int n)
{
  try { static_cast<Graph*>(g)->set_worker_count(n); }
  catch (const std::exception & e) { set_error(e.what()); }
}

unsigned int egress_graph_get_worker_count(egress_graph_t g)
{
  return static_cast<Graph*>(g)->worker_count();
}

void egress_graph_set_fusion_enabled(egress_graph_t g, bool enabled)
{
  try { static_cast<Graph*>(g)->set_fusion_enabled(enabled); }
  catch (const std::exception & e) { set_error(e.what()); }
}

bool egress_graph_get_fusion_enabled(egress_graph_t g)
{
  return static_cast<Graph*>(g)->fusion_enabled();
}

unsigned int egress_graph_get_buffer_length(egress_graph_t g)
{
  return static_cast<Graph*>(g)->getBufferLength();
}

// ---------- Profile stats ----------

static thread_local std::string tls_profile_stats_json;

static std::string profile_escape_json(const std::string & s)
{
  std::string out;
  out.reserve(s.size() + 2);
  out += '"';
  for (unsigned char c : s)
  {
    if      (c == '"')  out += "\\\"";
    else if (c == '\\') out += "\\\\";
    else if (c == '\n') out += "\\n";
    else if (c == '\r') out += "\\r";
    else if (c == '\t') out += "\\t";
    else                out += static_cast<char>(c);
  }
  out += '"';
  return out;
}

const char* egress_graph_get_profile_stats_json(egress_graph_t g)
{
  const auto s = static_cast<Graph*>(g)->profile_stats();
  std::string j;
  j.reserve(512);
  j += "{\"enabled\":";         j += s.enabled ? "true" : "false";
  j += ",\"callback_count\":";  j += std::to_string(s.callback_count);
  j += ",\"avg_callback_ms\":"; j += std::to_string(s.avg_callback_ms);
  j += ",\"max_callback_ms\":"; j += std::to_string(s.max_callback_ms);
  j += ",\"primitive_body_available\":";          j += s.primitive_body_available ? "true" : "false";
  j += ",\"primitive_body_covers_all_modules\":"; j += s.primitive_body_covers_all_modules ? "true" : "false";
  j += ",\"input_kernel_available\":";            j += s.input_kernel_available ? "true" : "false";
  j += ",\"fused_input_use_count\":";  j += std::to_string(s.fused_input_use_count);
  j += ",\"fused_body_use_count\":";   j += std::to_string(s.fused_body_use_count);
  j += ",\"fusion_candidate_reason\":"; j += profile_escape_json(s.fusion_candidate_reason);
  j += ",\"primitive_body_status\":";   j += profile_escape_json(s.primitive_body_status);
  j += ",\"input_kernel_status\":";     j += profile_escape_json(s.input_kernel_status);
  j += ",\"modules\":[";
  for (size_t i = 0; i < s.modules.size(); ++i)
  {
    const auto & m = s.modules[i];
    if (i > 0) j += ",";
    j += "{\"name\":";        j += profile_escape_json(m.module_name);
    j += ",\"call_count\":";  j += std::to_string(m.call_count);
    j += ",\"avg_call_ms\":"; j += std::to_string(m.avg_call_ms);
    j += ",\"max_call_ms\":"; j += std::to_string(m.max_call_ms);
    j += "}";
  }
  j += "]";
#ifdef EGRESS_PROFILE
  auto fused_sync_json = [](const Graph::ProfileStats::FusedSyncStats & fs) {
    std::string r;
    r += "{\"call_count\":";       r += std::to_string(fs.call_count);
    r += ",\"total_ms\":";         r += std::to_string(fs.total_ms);
    r += ",\"max_ms\":";           r += std::to_string(fs.max_ms);
    r += ",\"output_copy_count\":";r += std::to_string(fs.output_copy_count);
    r += "}";
    return r;
  };
  j += ",\"fused_current_output_sync\":"; j += fused_sync_json(s.fused_current_output_sync);
  j += ",\"fused_prev_output_sync\":";    j += fused_sync_json(s.fused_prev_output_sync);
#endif
  j += "}";
  tls_profile_stats_json = std::move(j);
  return tls_profile_stats_json.c_str();
}

void egress_graph_reset_profile_stats(egress_graph_t g)
{
  static_cast<Graph*>(g)->reset_profile_stats();
}

// ---------- Device enumeration ----------

unsigned int egress_audio_device_count(void)
{
  RtAudio tmp;
  return tmp.getDeviceCount();
}

void egress_audio_get_device_ids(unsigned int* out, unsigned int count)
{
  if (!out) return;
  RtAudio tmp;
  const auto ids = tmp.getDeviceIds();
  const unsigned int n = std::min(static_cast<unsigned int>(ids.size()), count);
  for (unsigned int i = 0; i < n; ++i)
    out[i] = ids[i];
}

bool egress_audio_get_device_info(unsigned int device_id, egress_device_info_t* out)
{
  if (!out) return false;
  try
  {
    RtAudio tmp;
    const RtAudio::DeviceInfo info = tmp.getDeviceInfo(device_id);
    out->id                  = info.ID;
    std::strncpy(out->name, info.name.c_str(), sizeof(out->name) - 1);
    out->name[sizeof(out->name) - 1] = '\0';
    out->output_channels     = info.outputChannels;
    out->input_channels      = info.inputChannels;
    out->is_default_output   = info.isDefaultOutput;
    out->preferred_sample_rate = info.preferredSampleRate;
    const unsigned int n = std::min(static_cast<unsigned int>(info.sampleRates.size()),
                                    static_cast<unsigned int>(32));
    out->sample_rate_count = n;
    for (unsigned int i = 0; i < n; ++i)
      out->sample_rates[i] = info.sampleRates[i];
    return true;
  }
  catch (...) { return false; }
}

unsigned int egress_audio_default_output_device(void)
{
  RtAudio tmp;
  return tmp.getDefaultOutputDevice();
}

// ---------- DAC API ----------

egress_dac_t egress_dac_new(egress_graph_t g, unsigned int sample_rate, unsigned int channels)
{
  try { return new EgressDAC(static_cast<Graph*>(g), sample_rate, channels); }
  catch (const std::exception & e) { set_error(e.what()); return nullptr; }
}

void egress_dac_free(egress_dac_t d)
{
  delete static_cast<EgressDAC*>(d);
}

void egress_dac_start(egress_dac_t d)
{
  try { static_cast<EgressDAC*>(d)->start(); }
  catch (const std::exception & e) { set_error(e.what()); }
}

void egress_dac_stop(egress_dac_t d)
{
  try { static_cast<EgressDAC*>(d)->stop(); }
  catch (const std::exception & e) { set_error(e.what()); }
}

bool egress_dac_is_running(egress_dac_t d)
{
  return static_cast<EgressDAC*>(d)->running;
}

void egress_dac_get_stats(egress_dac_t d, egress_dac_stats_t* out)
{
  if (!d || !out) return;
  const auto s    = static_cast<EgressDAC*>(d)->stats();
  out->callback_count  = s.callback_count;
  out->avg_callback_ms = s.avg_callback_ms;
  out->max_callback_ms = s.max_callback_ms;
  out->underrun_count  = s.underrun_count;
  out->overrun_count   = s.overrun_count;
}

void egress_dac_reset_stats(egress_dac_t d)
{
  if (d) static_cast<EgressDAC*>(d)->reset_stats();
}

bool egress_dac_is_reconnecting(egress_dac_t d)
{
  return d && static_cast<EgressDAC*>(d)->is_reconnecting();
}

unsigned int egress_dac_get_active_device(egress_dac_t d)
{
  return d ? static_cast<EgressDAC*>(d)->active_device() : 0;
}

bool egress_dac_switch_device(egress_dac_t d, unsigned int device_id)
{
  if (!d) return false;
  try { return static_cast<EgressDAC*>(d)->switch_device(device_id); }
  catch (const std::exception& e) { set_error(e.what()); return false; }
}

} // extern "C"
