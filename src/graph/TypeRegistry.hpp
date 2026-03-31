#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include "../jit/OrcJitEngine.hpp"  // for JitScalarType

namespace egress {

struct FieldDef {
    std::string name;
    egress_jit::JitScalarType scalar_type;
};

struct VariantDef {
    std::string name;
    std::vector<FieldDef> payload;
};

struct TypeDef {
    enum class Kind { Struct, Sum } kind;
    std::string name;
    std::vector<FieldDef> fields;        // for Struct
    std::vector<VariantDef> variants;    // for Sum

    uint32_t slot_count() const {
        if (kind == Kind::Struct) return static_cast<uint32_t>(fields.size());
        // Sum: discriminant slot + max payload slots
        uint32_t max_payload = 0;
        for (const auto& v : variants)
            max_payload = std::max(max_payload, static_cast<uint32_t>(v.payload.size()));
        return 1 + max_payload;
    }
};

class TypeRegistry {
    std::unordered_map<std::string, TypeDef> types_;
public:
    void define_struct(std::string name, std::vector<FieldDef> fields) {
        TypeDef td;
        td.kind = TypeDef::Kind::Struct;
        td.name = name;
        td.fields = std::move(fields);
        types_[name] = std::move(td);
    }
    void define_sum(std::string name, std::vector<VariantDef> variants) {
        TypeDef td;
        td.kind = TypeDef::Kind::Sum;
        td.name = name;
        td.variants = std::move(variants);
        types_[name] = std::move(td);
    }
    const TypeDef* find(const std::string& name) const {
        auto it = types_.find(name);
        return it == types_.end() ? nullptr : &it->second;
    }
};

} // namespace egress
