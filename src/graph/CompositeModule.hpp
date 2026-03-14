#pragma once

#include <cstdint>
#include <stdexcept>
#include <queue>
#include <string>
#include <utility>
#include <vector>

namespace egress_composition
{
enum class ConnectionTiming : uint8_t
{
  SameTick,
  Delayed
};

enum class NodeKind : uint8_t
{
  InputBoundary,
  OutputBoundary,
  ModuleCall,
  StatefulFunctionCall,
  Delay
};

struct PortRef
{
  uint32_t node_id = 0;
  uint32_t port_id = 0;

  bool operator==(const PortRef & other) const
  {
    return node_id == other.node_id && port_id == other.port_id;
  }
};

struct CompositeNodeSpec
{
  uint32_t id = 0;
  NodeKind kind = NodeKind::ModuleCall;
  std::string label;
  uint32_t input_count = 0;
  uint32_t output_count = 0;
};

struct CompositeEdgeSpec
{
  PortRef src;
  PortRef dst;
  ConnectionTiming timing = ConnectionTiming::SameTick;
};

struct CompositeModuleSpec
{
  std::vector<CompositeNodeSpec> nodes;
  std::vector<CompositeEdgeSpec> edges;
  uint32_t input_boundary_id = 0;
  uint32_t output_boundary_id = 0;

  uint32_t add_node(NodeKind kind, std::string label, uint32_t input_count, uint32_t output_count)
  {
    const uint32_t id = static_cast<uint32_t>(nodes.size());
    nodes.push_back(CompositeNodeSpec{id, kind, std::move(label), input_count, output_count});
    return id;
  }

  void add_edge(PortRef src, PortRef dst, ConnectionTiming timing)
  {
    edges.push_back(CompositeEdgeSpec{src, dst, timing});
  }
};

struct ValidationResult
{
  bool ok = true;
  std::string message;
  std::vector<uint32_t> same_tick_topology;
};

struct LoweredCompositeModule
{
  std::vector<uint32_t> same_tick_schedule;
  uint32_t delayed_node_count = 0;
};

inline ValidationResult validate_same_tick_acyclic(const CompositeModuleSpec & spec)
{
  ValidationResult result;
  result.same_tick_topology.reserve(spec.nodes.size());

  std::vector<uint32_t> indegree(spec.nodes.size(), 0);
  std::vector<std::vector<uint32_t>> outgoing(spec.nodes.size());
  for (const auto & edge : spec.edges)
  {
    if (edge.timing != ConnectionTiming::SameTick)
    {
      continue;
    }
    if (edge.src.node_id >= spec.nodes.size() || edge.dst.node_id >= spec.nodes.size())
    {
      result.ok = false;
      result.message = "Composite module spec contains an out-of-range node reference.";
      return result;
    }
    outgoing[edge.src.node_id].push_back(edge.dst.node_id);
    ++indegree[edge.dst.node_id];
  }

  std::queue<uint32_t> ready;
  for (uint32_t node_id = 0; node_id < indegree.size(); ++node_id)
  {
    if (indegree[node_id] == 0)
    {
      ready.push(node_id);
    }
  }

  while (!ready.empty())
  {
    const uint32_t node_id = ready.front();
    ready.pop();
    result.same_tick_topology.push_back(node_id);
    for (uint32_t dst : outgoing[node_id])
    {
      if (--indegree[dst] == 0)
      {
        ready.push(dst);
      }
    }
  }

  if (result.same_tick_topology.size() != spec.nodes.size())
  {
    result.ok = false;
    result.message = "Same-tick module composition must be acyclic; use eg.delay(...) to break feedback loops.";
  }

  return result;
}

inline LoweredCompositeModule lower_composite_module(const CompositeModuleSpec & spec)
{
  const ValidationResult validation = validate_same_tick_acyclic(spec);
  if (!validation.ok)
  {
    throw std::invalid_argument(validation.message);
  }

  LoweredCompositeModule lowered;
  lowered.same_tick_schedule = validation.same_tick_topology;
  for (const auto & node : spec.nodes)
  {
    if (node.kind == NodeKind::Delay)
    {
      ++lowered.delayed_node_count;
    }
  }
  return lowered;
}
}  // namespace egress_composition
