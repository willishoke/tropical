#pragma once

#include <algorithm>
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
  struct ScheduledNode
  {
    uint32_t node_id = 0;
    NodeKind kind = NodeKind::ModuleCall;
    std::string label;
  };

  struct DelayedEdgeState
  {
    uint32_t edge_id = 0;
    PortRef src;
    PortRef dst;
    uint32_t state_slot = 0;
  };

  std::vector<uint32_t> same_tick_schedule;
  std::vector<ScheduledNode> scheduled_nodes;
  std::vector<DelayedEdgeState> delayed_edges;
  uint32_t delayed_node_count = 0;
};

inline ValidationResult validate_composite_module(const CompositeModuleSpec & spec)
{
  ValidationResult result;
  result.same_tick_topology.reserve(spec.nodes.size());

  std::vector<uint32_t> indegree(spec.nodes.size(), 0);
  std::vector<std::vector<uint32_t>> outgoing(spec.nodes.size());
  std::vector<uint32_t> delayed_incoming(spec.nodes.size(), 0);
  std::vector<uint32_t> same_tick_incoming(spec.nodes.size(), 0);
  for (const auto & edge : spec.edges)
  {
    if (edge.src.node_id >= spec.nodes.size() || edge.dst.node_id >= spec.nodes.size())
    {
      result.ok = false;
      result.message = "Composite module spec contains an out-of-range node reference.";
      return result;
    }
    const auto & src_node = spec.nodes[edge.src.node_id];
    const auto & dst_node = spec.nodes[edge.dst.node_id];
    if (edge.src.port_id >= src_node.output_count)
    {
      result.ok = false;
      result.message = "Composite module spec contains an out-of-range source port reference.";
      return result;
    }
    if (edge.dst.port_id >= dst_node.input_count)
    {
      result.ok = false;
      result.message = "Composite module spec contains an out-of-range destination port reference.";
      return result;
    }
    if (edge.timing == ConnectionTiming::Delayed)
    {
      ++delayed_incoming[edge.dst.node_id];
      if (dst_node.kind != NodeKind::Delay)
      {
        result.ok = false;
        result.message = "Delayed edges must terminate at delay nodes.";
        return result;
      }
      if (edge.dst.port_id != 0)
      {
        result.ok = false;
        result.message = "Delay nodes accept delayed input only on port 0.";
        return result;
      }
      if (src_node.kind == NodeKind::Delay)
      {
        result.ok = false;
        result.message = "Delay nodes cannot feed other delay nodes through delayed edges.";
        return result;
      }
      continue;
    }
    ++same_tick_incoming[edge.dst.node_id];
    if (dst_node.kind == NodeKind::Delay)
    {
      result.ok = false;
      result.message = "Delay nodes cannot receive same-tick inputs; use eg.delay(...) to create delayed edges.";
      return result;
    }
    outgoing[edge.src.node_id].push_back(edge.dst.node_id);
    ++indegree[edge.dst.node_id];
  }

  for (const auto & node : spec.nodes)
  {
    if (node.kind != NodeKind::Delay)
    {
      continue;
    }
    if (same_tick_incoming[node.id] != 0)
    {
      result.ok = false;
      result.message = "Delay nodes may only receive delayed inputs.";
      return result;
    }
    if (delayed_incoming[node.id] != 1)
    {
      result.ok = false;
      result.message = "Each delay node must have exactly one delayed input.";
      return result;
    }
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

inline ValidationResult validate_same_tick_acyclic(const CompositeModuleSpec & spec)
{
  return validate_composite_module(spec);
}

inline LoweredCompositeModule lower_composite_module(const CompositeModuleSpec & spec)
{
  const ValidationResult validation = validate_composite_module(spec);
  if (!validation.ok)
  {
    throw std::invalid_argument(validation.message);
  }

  LoweredCompositeModule lowered;
  lowered.same_tick_schedule = validation.same_tick_topology;
  const auto boundary_it = std::find(
    lowered.same_tick_schedule.begin(),
    lowered.same_tick_schedule.end(),
    spec.output_boundary_id);
  if (boundary_it != lowered.same_tick_schedule.end() &&
      boundary_it + 1 != lowered.same_tick_schedule.end())
  {
    const uint32_t boundary_id = *boundary_it;
    lowered.same_tick_schedule.erase(boundary_it);
    lowered.same_tick_schedule.push_back(boundary_id);
  }
  lowered.scheduled_nodes.reserve(validation.same_tick_topology.size());
  for (uint32_t node_id : validation.same_tick_topology)
  {
    const auto & node = spec.nodes[node_id];
    lowered.scheduled_nodes.push_back(LoweredCompositeModule::ScheduledNode{
      node.id,
      node.kind,
      node.label});
    if (node.kind == NodeKind::Delay)
    {
      ++lowered.delayed_node_count;
    }
  }
  for (uint32_t edge_id = 0; edge_id < spec.edges.size(); ++edge_id)
  {
    const auto & edge = spec.edges[edge_id];
    if (edge.timing != ConnectionTiming::Delayed)
    {
      continue;
    }
    lowered.delayed_edges.push_back(LoweredCompositeModule::DelayedEdgeState{
      edge_id,
      edge.src,
      edge.dst,
      static_cast<uint32_t>(lowered.delayed_edges.size())});
  }
  return lowered;
}
}  // namespace egress_composition
