#pragma once

/**
 * plan_loader.hpp — Apply wiring/outputs from a plan JSON to a Graph.
 *
 * Uses parse_expr from PlanParser.hpp for JSON→ExprSpec conversion.
 * One-shot entrypoint: load_plan_wiring(graph, json_string).
 */

#include "runtime/PlanParser.hpp"
#include "graph/Graph.hpp"

namespace egress_plan
{

// parse_expr and helpers are in runtime/PlanParser.hpp

/**
 * Load wiring and outputs from a plan JSON into an existing Graph.
 *
 * This function:
 * 1. Clears existing input expressions and outputs on the graph
 * 2. Resolves ref output names to numeric indices using kernel output lists
 * 3. Parses wiring ExprNode JSON → ExprSpec trees, applies via set_input_expr
 * 4. Adds output taps from the plan
 * 5. Wraps everything in begin_update/end_update for atomic swap
 *
 * Returns true on success, throws std::runtime_error on failure.
 */
inline bool load_plan_wiring(Graph & graph, const std::string & plan_json)
{
    const json plan = json::parse(plan_json);

    // Validate schema
    if (!plan.contains("schema") || plan["schema"] != "egress_plan_1")
        throw std::runtime_error("plan_loader: unsupported or missing schema (expected 'egress_plan_1')");

    // Build kernel ID → name + output name mapping for ref resolution
    struct KernelInfo
    {
        std::string name;
        std::vector<std::string> outputs;
        std::vector<std::string> inputs;
    };
    std::unordered_map<int, KernelInfo> kernel_map;
    // Also build name → output name→index for resolving string refs
    std::unordered_map<std::string, std::unordered_map<std::string, unsigned int>> output_name_map;

    for (const auto & k : plan["kernels"])
    {
        KernelInfo info;
        info.name = k["name"].get<std::string>();
        for (const auto & o : k["outputs"])
            info.outputs.push_back(o.get<std::string>());
        for (const auto & i : k["inputs"])
            info.inputs.push_back(i.get<std::string>());

        int id = k["id"].get<int>();
        kernel_map[id] = info;

        auto & omap = output_name_map[info.name];
        for (unsigned int i = 0; i < info.outputs.size(); ++i)
            omap[info.outputs[i]] = i;
    }

    // Pre-pass: recursively resolve string output names in ref nodes to numeric indices.
    // This mutates a copy of each wiring expr JSON so that parse_expr (which requires
    // numeric output IDs) can handle everything uniformly.
    std::function<json(const json &)> resolve_refs =
        [&](const json & node) -> json
    {
        if (!node.is_object()) return node;
        if (!node.contains("op")) return node;

        json resolved = node;
        const std::string op = node["op"].get<std::string>();

        // Resolve ref output names → indices
        if (op == "ref" && node.contains("output") && node["output"].is_string())
        {
            const auto & mod_name = node["module"].get<std::string>();
            const auto & out_name = node["output"].get<std::string>();
            auto it = output_name_map.find(mod_name);
            if (it == output_name_map.end())
                throw std::runtime_error(
                    "plan_loader: ref to unknown module '" + mod_name + "'");
            auto it2 = it->second.find(out_name);
            if (it2 == it->second.end())
                throw std::runtime_error(
                    "plan_loader: unknown output '" + out_name +
                    "' on module '" + mod_name + "'");
            resolved["output"] = it2->second;
            return resolved;
        }

        // Recurse into args
        if (resolved.contains("args") && resolved["args"].is_array())
        {
            json new_args = json::array();
            for (const auto & arg : resolved["args"])
                new_args.push_back(resolve_refs(arg));
            resolved["args"] = std::move(new_args);
        }
        // Recurse into ADT-specific fields
        if (resolved.contains("fields") && resolved["fields"].is_array())
        {
            json new_fields = json::array();
            for (const auto & f : resolved["fields"])
                new_fields.push_back(resolve_refs(f));
            resolved["fields"] = std::move(new_fields);
        }
        if (resolved.contains("struct_expr"))
            resolved["struct_expr"] = resolve_refs(resolved["struct_expr"]);
        if (resolved.contains("payload") && resolved["payload"].is_array())
        {
            json new_payload = json::array();
            for (const auto & p : resolved["payload"])
                new_payload.push_back(resolve_refs(p));
            resolved["payload"] = std::move(new_payload);
        }
        if (resolved.contains("scrutinee"))
            resolved["scrutinee"] = resolve_refs(resolved["scrutinee"]);
        if (resolved.contains("branches") && resolved["branches"].is_array())
        {
            json new_branches = json::array();
            for (const auto & br : resolved["branches"])
                new_branches.push_back(resolve_refs(br));
            resolved["branches"] = std::move(new_branches);
        }

        return resolved;
    };

    // If already inside a batch (e.g. caller batching module additions + wiring),
    // join the existing batch instead of starting a new one.
    const bool own_batch = !graph.is_batch_active();
    if (own_batch) graph.begin_update();
    graph.clear_wiring_deferred();

    try
    {
        for (const auto & w : plan["wiring"])
        {
            int kernel_id = w["kernel"].get<int>();
            auto kit = kernel_map.find(kernel_id);
            if (kit == kernel_map.end())
                throw std::runtime_error(
                    "plan_loader: wiring targets unknown kernel ID " + std::to_string(kernel_id));

            unsigned int input_idx = w["input"].get<unsigned int>();
            json resolved_json = resolve_refs(w["expr"]);
            ExprSpecPtr expr = parse_expr(resolved_json);

            if (!graph.set_input_expr(kit->second.name, input_idx, std::move(expr)))
                throw std::runtime_error(
                    "plan_loader: failed to set input expr on '" + kit->second.name +
                    "' input " + std::to_string(input_idx) +
                    " ('" + w.value("input_name", std::string("?")) + "')");
        }

        for (const auto & o : plan["outputs"])
        {
            int kernel_id = o["kernel"].get<int>();
            auto kit = kernel_map.find(kernel_id);
            if (kit == kernel_map.end())
                throw std::runtime_error(
                    "plan_loader: output references unknown kernel ID " + std::to_string(kernel_id));

            if (!graph.addOutput(std::make_pair(kit->second.name, o["output"].get<unsigned int>())))
                throw std::runtime_error(
                    "plan_loader: failed to add output on '" + kit->second.name +
                    "' output " + std::to_string(o["output"].get<unsigned int>()));
        }

        if (own_batch) graph.end_update();
    }
    catch (...)
    {
        try { if (own_batch) graph.end_update(); } catch (...) {}
        throw;
    }

    return true;
}

} // namespace egress_plan
