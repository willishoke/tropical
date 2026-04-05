#pragma once

/**
 * PlanParser.hpp — Parse ExprNode JSON trees into ExprSpec trees.
 *
 * Extracted from plan_loader.hpp. Pure parsing, no Graph dependency.
 */

#include "expr/Expr.hpp"

#include <nlohmann/json.hpp>

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace egress_plan
{

using json = nlohmann::json;
using ExprSpecPtr = egress_expr::ExprSpecPtr;

// ─────────────────────────────────────────────────────────────
// ExprNode JSON → ExprSpec
// ─────────────────────────────────────────────────────────────

/**
 * Map from ExprNode op strings to ExprKind.
 * Binary and unary ops share the same dispatch; multi-arg ops
 * (clamp, select, index, array_set) have dedicated handling.
 */
inline egress_expr::ExprKind op_to_kind(const std::string & op)
{
    using K = egress_expr::ExprKind;

    // Binary ops
    if (op == "add")       return K::Add;
    if (op == "sub")       return K::Sub;
    if (op == "mul")       return K::Mul;
    if (op == "div")       return K::Div;
    if (op == "floor_div") return K::FloorDiv;
    if (op == "mod")       return K::Mod;
    if (op == "pow")       return K::Pow;
    if (op == "matmul")    return K::MatMul;
    if (op == "lt")        return K::Less;
    if (op == "lte")       return K::LessEqual;
    if (op == "gt")        return K::Greater;
    if (op == "gte")       return K::GreaterEqual;
    if (op == "eq")        return K::Equal;
    if (op == "neq")       return K::NotEqual;
    if (op == "bit_and")   return K::BitAnd;
    if (op == "bit_or")    return K::BitOr;
    if (op == "bit_xor")   return K::BitXor;
    if (op == "lshift")    return K::LShift;
    if (op == "rshift")    return K::RShift;

    // Unary ops
    if (op == "neg")       return K::Neg;
    if (op == "abs")       return K::Abs;
    if (op == "sin")       return K::Sin;
    if (op == "log")       return K::Log;
    if (op == "not")       return K::Not;
    if (op == "bit_not")   return K::BitNot;

    // Multi-arg
    if (op == "clamp")     return K::Clamp;
    if (op == "select")    return K::Select;
    if (op == "index")     return K::Index;
    if (op == "array_set") return K::ArraySet;

    throw std::runtime_error("plan_loader: unknown expr op '" + op + "'");
}

inline bool is_binary_op(const std::string & op)
{
    static const std::unordered_map<std::string, bool> ops = {
        {"add",1},{"sub",1},{"mul",1},{"div",1},{"floor_div",1},{"mod",1},{"pow",1},{"matmul",1},
        {"lt",1},{"lte",1},{"gt",1},{"gte",1},{"eq",1},{"neq",1},
        {"bit_and",1},{"bit_or",1},{"bit_xor",1},{"lshift",1},{"rshift",1},
    };
    return ops.count(op) > 0;
}

inline bool is_unary_op(const std::string & op)
{
    static const std::unordered_map<std::string, bool> ops = {
        {"neg",1},{"abs",1},{"sin",1},{"log",1},{"not",1},{"bit_not",1},
    };
    return ops.count(op) > 0;
}

/**
 * Recursively convert an ExprNode JSON value to an ExprSpec tree.
 *
 * Handles: scalars, arrays, ref, input, reg, sample_rate, sample_index,
 * all binary/unary ops, clamp, select, index, array_set, array, matrix,
 * float/int/bool literals, and ADT operations.
 */
inline ExprSpecPtr parse_expr(const json & node)
{
    using K = egress_expr::ExprKind;

    // Scalar literal shorthand
    if (node.is_number_integer())
    {
        auto spec = std::make_shared<egress_expr::ExprSpec>();
        spec->kind = K::Literal;
        spec->literal = egress_expr::float_value(static_cast<double>(node.get<int64_t>()));
        return spec;
    }
    if (node.is_number_float())
    {
        auto spec = std::make_shared<egress_expr::ExprSpec>();
        spec->kind = K::Literal;
        spec->literal = egress_expr::float_value(node.get<double>());
        return spec;
    }
    if (node.is_boolean())
    {
        auto spec = std::make_shared<egress_expr::ExprSpec>();
        spec->kind = K::Literal;
        spec->literal = egress_expr::bool_value(node.get<bool>());
        return spec;
    }

    // Inline array → ArrayPack
    if (node.is_array())
    {
        auto spec = std::make_shared<egress_expr::ExprSpec>();
        spec->kind = K::ArrayPack;
        for (const auto & item : node)
            spec->args.push_back(parse_expr(item));
        return spec;
    }

    // Object node — op-based dispatch
    if (!node.is_object() || !node.contains("op"))
        throw std::runtime_error("plan_loader: invalid expr node (expected object with 'op')");

    const std::string op = node["op"].get<std::string>();

    // ── References ──
    if (op == "ref")
    {
        auto spec = std::make_shared<egress_expr::ExprSpec>();
        spec->kind = K::Ref;
        spec->module_name = node["module"].get<std::string>();
        // output can be string (name) or number (index). The plan format uses
        // string names which have already been resolved to indices by the TS compiler.
        // But the wiring expr in the plan carries the original ExprNode, which uses
        // string output names. We need the kernel's output name list to resolve.
        // For now, support both formats.
        if (node["output"].is_number())
            spec->output_id = node["output"].get<unsigned int>();
        else
            throw std::runtime_error(
                "plan_loader: ref output must be numeric index (got string '" +
                node["output"].get<std::string>() + "' for module '" + spec->module_name + "')");
        return spec;
    }

    if (op == "input")
    {
        auto spec = std::make_shared<egress_expr::ExprSpec>();
        spec->kind = K::InputValue;
        if (node.contains("id"))
            spec->slot_id = node["id"].get<unsigned int>();
        else if (node.contains("name"))
            throw std::runtime_error("plan_loader: 'input' by name not supported — use numeric id");
        return spec;
    }

    if (op == "reg")
    {
        auto spec = std::make_shared<egress_expr::ExprSpec>();
        spec->kind = K::RegisterValue;
        if (node.contains("id"))
            spec->slot_id = node["id"].get<unsigned int>();
        else if (node.contains("name"))
            throw std::runtime_error("plan_loader: 'reg' by name not supported — use numeric id");
        return spec;
    }

    if (op == "sample_rate")
    {
        auto spec = std::make_shared<egress_expr::ExprSpec>();
        spec->kind = K::SampleRate;
        return spec;
    }

    if (op == "sample_index")
    {
        auto spec = std::make_shared<egress_expr::ExprSpec>();
        spec->kind = K::SampleIndex;
        return spec;
    }

    // ── Explicit typed literals ──
    if (op == "float")
    {
        auto spec = std::make_shared<egress_expr::ExprSpec>();
        spec->kind = K::Literal;
        spec->literal = egress_expr::float_value(node["value"].get<double>());
        return spec;
    }
    if (op == "int")
    {
        auto spec = std::make_shared<egress_expr::ExprSpec>();
        spec->kind = K::Literal;
        spec->literal = egress_expr::int_value(node["value"].get<int64_t>());
        return spec;
    }
    if (op == "bool")
    {
        auto spec = std::make_shared<egress_expr::ExprSpec>();
        spec->kind = K::Literal;
        spec->literal = egress_expr::bool_value(node["value"].get<bool>());
        return spec;
    }

    // ── Binary ops ──
    if (is_binary_op(op))
    {
        const auto & args = node["args"];
        if (!args.is_array() || args.size() != 2)
            throw std::runtime_error("plan_loader: binary op '" + op + "' requires exactly 2 args");
        auto spec = std::make_shared<egress_expr::ExprSpec>();
        spec->kind = op_to_kind(op);
        spec->lhs = parse_expr(args[0]);
        spec->rhs = parse_expr(args[1]);
        return spec;
    }

    // ── Unary ops ──
    if (is_unary_op(op))
    {
        const auto & args = node["args"];
        if (!args.is_array() || args.size() != 1)
            throw std::runtime_error("plan_loader: unary op '" + op + "' requires exactly 1 arg");
        auto spec = std::make_shared<egress_expr::ExprSpec>();
        spec->kind = op_to_kind(op);
        spec->lhs = parse_expr(args[0]);
        return spec;
    }

    // ── Multi-arg ops ──
    if (op == "clamp")
    {
        const auto & args = node["args"];
        if (!args.is_array() || args.size() != 3)
            throw std::runtime_error("plan_loader: 'clamp' requires exactly 3 args");
        auto spec = std::make_shared<egress_expr::ExprSpec>();
        spec->kind = K::Clamp;
        spec->lhs = parse_expr(args[0]);
        spec->rhs = parse_expr(args[1]);
        spec->args.push_back(parse_expr(args[2]));
        return spec;
    }

    if (op == "select")
    {
        const auto & args = node["args"];
        if (!args.is_array() || args.size() != 3)
            throw std::runtime_error("plan_loader: 'select' requires exactly 3 args");
        auto spec = std::make_shared<egress_expr::ExprSpec>();
        spec->kind = K::Select;
        spec->lhs = parse_expr(args[0]);   // condition
        spec->rhs = parse_expr(args[1]);   // then
        spec->args.push_back(parse_expr(args[2]));  // else
        return spec;
    }

    if (op == "index")
    {
        const auto & args = node["args"];
        if (!args.is_array() || args.size() != 2)
            throw std::runtime_error("plan_loader: 'index' requires exactly 2 args");
        auto spec = std::make_shared<egress_expr::ExprSpec>();
        spec->kind = K::Index;
        spec->lhs = parse_expr(args[0]);
        spec->rhs = parse_expr(args[1]);
        return spec;
    }

    if (op == "array_set")
    {
        const auto & args = node["args"];
        if (!args.is_array() || args.size() != 3)
            throw std::runtime_error("plan_loader: 'array_set' requires exactly 3 args");
        auto spec = std::make_shared<egress_expr::ExprSpec>();
        spec->kind = K::ArraySet;
        spec->lhs = parse_expr(args[0]);   // array
        spec->rhs = parse_expr(args[1]);   // index
        spec->args.push_back(parse_expr(args[2]));  // value
        return spec;
    }

    if (op == "array")
    {
        auto spec = std::make_shared<egress_expr::ExprSpec>();
        spec->kind = K::ArrayPack;
        const auto & items = node["items"];
        for (const auto & item : items)
            spec->args.push_back(parse_expr(item));
        return spec;
    }

    if (op == "matrix")
    {
        // Flatten rows into a single ArrayPack of ArrayPacks
        const auto & rows = node["rows"];
        auto spec = std::make_shared<egress_expr::ExprSpec>();
        spec->kind = K::ArrayPack;
        for (const auto & row : rows)
        {
            auto row_spec = std::make_shared<egress_expr::ExprSpec>();
            row_spec->kind = K::ArrayPack;
            for (const auto & val : row)
                row_spec->args.push_back(parse_expr(val));
            spec->args.push_back(std::move(row_spec));
        }
        return spec;
    }

    // ── ADT operations ──
    if (op == "construct_struct")
    {
        auto spec = std::make_shared<egress_expr::ExprSpec>();
        spec->kind = K::ConstructStruct;
        spec->module_name = node["type_name"].get<std::string>();
        for (const auto & f : node["fields"])
            spec->args.push_back(parse_expr(f));
        return spec;
    }

    if (op == "field_access")
    {
        auto spec = std::make_shared<egress_expr::ExprSpec>();
        spec->kind = K::FieldAccess;
        spec->module_name = node["type_name"].get<std::string>();
        spec->lhs = parse_expr(node["struct_expr"]);
        spec->slot_id = node["field_index"].get<unsigned int>();
        return spec;
    }

    if (op == "construct_variant")
    {
        auto spec = std::make_shared<egress_expr::ExprSpec>();
        spec->kind = K::ConstructVariant;
        spec->module_name = node["type_name"].get<std::string>();
        spec->slot_id = node["variant_tag"].get<unsigned int>();
        for (const auto & p : node["payload"])
            spec->args.push_back(parse_expr(p));
        return spec;
    }

    if (op == "match_variant")
    {
        auto spec = std::make_shared<egress_expr::ExprSpec>();
        spec->kind = K::MatchVariant;
        spec->module_name = node["type_name"].get<std::string>();
        spec->lhs = parse_expr(node["scrutinee"]);
        for (const auto & b : node["branches"])
            spec->args.push_back(parse_expr(b));
        return spec;
    }

    if (op == "delay_value")
    {
        auto spec = std::make_shared<egress_expr::ExprSpec>();
        spec->kind = K::DelayValue;
        spec->slot_id = node["node_id"].get<unsigned int>();
        return spec;
    }

    throw std::runtime_error("plan_loader: unsupported expr op '" + op + "'");
}

} // namespace egress_plan
