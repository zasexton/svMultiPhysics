/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/ParameterRegistry.h"

#include <algorithm>

namespace svmp {
namespace FE {
namespace systems {

namespace {

bool valueTypesCompatible(params::ValueType expected, params::ValueType actual) noexcept
{
    if (expected == params::ValueType::Any || actual == params::ValueType::Any) return true;
    return expected == actual;
}

} // namespace

void ParameterRegistry::clear()
{
    specs_.clear();
    by_key_.clear();
}

const params::Spec* ParameterRegistry::find(std::string_view key) const noexcept
{
    const auto it = by_key_.find(std::string(key));
    if (it == by_key_.end()) return nullptr;
    return &it->second.spec;
}

void ParameterRegistry::validateDefaultType(const params::Spec& spec)
{
    if (!spec.default_value.has_value()) return;
    if (spec.type == params::ValueType::Any) return;
    const auto t = params::typeOf(*spec.default_value);
    FE_THROW_IF(!valueTypesCompatible(spec.type, t), InvalidArgumentException,
                "ParameterRegistry: default for '" + spec.key + "' has type " +
                    std::string(params::typeName(t)) + " but Spec expects " +
                    std::string(params::typeName(spec.type)));
}

void ParameterRegistry::mergeInto(params::Spec& dst, const params::Spec& src)
{
    if (dst.type == params::ValueType::Any) {
        dst.type = src.type;
    } else if (src.type != params::ValueType::Any) {
        FE_THROW_IF(dst.type != src.type, InvalidArgumentException,
                    "ParameterRegistry: conflicting types for '" + dst.key +
                        "' (" + std::string(params::typeName(dst.type)) + " vs " +
                        std::string(params::typeName(src.type)) + ")");
    }

    dst.required = dst.required || src.required;

    if (!dst.default_value.has_value() && src.default_value.has_value()) {
        dst.default_value = src.default_value;
    } else if (dst.default_value.has_value() && src.default_value.has_value()) {
        const auto td = params::typeOf(*dst.default_value);
        const auto ts = params::typeOf(*src.default_value);
        if (td != ts || *dst.default_value != *src.default_value) {
            FE_THROW(InvalidArgumentException,
                     "ParameterRegistry: conflicting defaults for '" + dst.key + "'");
        }
    }

    if (dst.doc.empty() && !src.doc.empty()) {
        dst.doc = src.doc;
    }
}

void ParameterRegistry::add(params::Spec spec, std::string source)
{
    FE_THROW_IF(spec.key.empty(), InvalidArgumentException, "ParameterRegistry::add: empty Spec.key");
    validateDefaultType(spec);

    auto it = by_key_.find(spec.key);
    if (it == by_key_.end()) {
        Entry entry;
        entry.spec = std::move(spec);
        entry.source = std::move(source);
        specs_.push_back(entry.spec);
        by_key_.emplace(specs_.back().key, std::move(entry));
        return;
    }

    // Merge into existing canonical spec (and update list copy).
    auto& entry = it->second;
    mergeInto(entry.spec, spec);
    validateDefaultType(entry.spec);

    for (auto& s : specs_) {
        if (s.key == entry.spec.key) {
            s = entry.spec;
            break;
        }
    }
}

void ParameterRegistry::addAll(const std::vector<params::Spec>& specs, std::string source)
{
    for (const auto& s : specs) {
        add(s, source);
    }
}

void ParameterRegistry::validate(const SystemStateView& state) const
{
    for (const auto& spec : specs_) {
        // Fast accept if optional with no defaults.
        if (!spec.required && !spec.default_value.has_value()) continue;

        std::optional<params::Value> v;
        if (state.getParam) {
            v = state.getParam(spec.key);
        }

        if (!v && spec.type == params::ValueType::Real && state.getRealParam) {
            if (const auto r = state.getRealParam(spec.key)) {
                v = params::Value{*r};
            }
        }

        if (!v) {
            if (spec.default_value.has_value()) continue;
            if (!spec.required) continue;
            FE_THROW(InvalidArgumentException,
                     "ParameterRegistry: missing required parameter '" + spec.key + "'");
        }

        if (spec.type == params::ValueType::Any) continue;
        const auto actual = params::typeOf(*v);
        FE_THROW_IF(!valueTypesCompatible(spec.type, actual), InvalidArgumentException,
                    "ParameterRegistry: parameter '" + spec.key + "' has type " +
                        std::string(params::typeName(actual)) + " but expected " +
                        std::string(params::typeName(spec.type)));
    }
}

std::function<std::optional<params::Value>(std::string_view)>
ParameterRegistry::makeParamGetter(const SystemStateView& state) const
{
    return [&state, this](std::string_view key) -> std::optional<params::Value> {
        if (state.getParam) {
            if (auto v = state.getParam(key)) return v;
        }

        const params::Spec* spec = find(key);
        if (spec == nullptr) return std::nullopt;

        if (spec->type == params::ValueType::Real && state.getRealParam) {
            if (const auto r = state.getRealParam(key)) return params::Value{*r};
        }

        if (spec->default_value.has_value()) return *spec->default_value;
        return std::nullopt;
    };
}

std::function<std::optional<Real>(std::string_view)>
ParameterRegistry::makeRealGetter(const SystemStateView& state) const
{
    return [&state, this](std::string_view key) -> std::optional<Real> {
        if (state.getRealParam) {
            if (auto r = state.getRealParam(key)) return r;
        }

        if (state.getParam) {
            if (auto v = state.getParam(key)) {
                if (auto r = params::get<Real>(*v)) return r;
            }
        }

        const params::Spec* spec = find(key);
        if (spec == nullptr) return std::nullopt;
        if (spec->default_value.has_value()) {
            if (auto r = params::get<Real>(*spec->default_value)) return r;
        }

        return std::nullopt;
    };
}

} // namespace systems
} // namespace FE
} // namespace svmp

