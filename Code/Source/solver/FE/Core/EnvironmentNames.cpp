/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Core/EnvironmentNames.h"

#include "Core/OptionSchema.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>

namespace svmp {
namespace FE {
namespace environment {

namespace {

// The catalogue: every environment name the solver reads, grouped by owning
// service.  A name is read through the registry only; a reader that is
// deleted takes its entry with it.
std::span<const NameDeclaration> solverCatalogue() noexcept
{
    return {};
}

[[nodiscard]] bool validName(std::string_view name) noexcept
{
    return !name.empty() && std::all_of(name.begin(), name.end(), [](char ch) {
        return (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_';
    });
}

[[nodiscard]] bool nonEmpty(const char* text) noexcept
{
    return text != nullptr && text[0] != '\0';
}

[[nodiscard]] bool lessByName(const NameDeclaration* lhs, const NameDeclaration* rhs) noexcept
{
    return std::string_view(lhs->name) < std::string_view(rhs->name);
}

} // namespace

const char* toString(NameKind kind) noexcept
{
    switch (kind) {
    case NameKind::Diagnostic: return "diagnostic";
    case NameKind::Performance: return "performance";
    case NameKind::TestHook: return "test_hook";
    case NameKind::NumericalOverride: return "numerical_override";
    }
    return "unknown";
}

const char* toString(ValueType type) noexcept
{
    switch (type) {
    case ValueType::Flag: return "flag";
    case ValueType::Integer: return "integer";
    case ValueType::Real: return "real";
    case ValueType::Text: return "text";
    case ValueType::List: return "list";
    }
    return "unknown";
}

EnvironmentNameRegistry::EnvironmentNameRegistry(std::span<const NameDeclaration> declarations)
{
    by_name_.reserve(declarations.size());
    for (const auto& declaration : declarations) {
        if (declaration.name == nullptr || !validName(declaration.name)) {
            throw std::logic_error(
                "EnvironmentNameRegistry: a declared name must be nonempty and spelled with "
                "upper-case letters, digits and underscores");
        }
        if (!nonEmpty(declaration.owner) || !nonEmpty(declaration.default_value) ||
            !nonEmpty(declaration.controls)) {
            throw std::logic_error(std::string("EnvironmentNameRegistry: '") + declaration.name +
                                   "' needs an owner, a default value and what it controls");
        }
        by_name_.push_back(&declaration);
    }
    std::sort(by_name_.begin(), by_name_.end(), lessByName);
    const auto duplicate = std::adjacent_find(
        by_name_.begin(), by_name_.end(), [](const NameDeclaration* lhs, const NameDeclaration* rhs) {
            return std::string_view(lhs->name) == std::string_view(rhs->name);
        });
    if (duplicate != by_name_.end()) {
        throw std::logic_error(std::string("EnvironmentNameRegistry: '") + (*duplicate)->name +
                               "' is declared twice");
    }
}

const EnvironmentNameRegistry& EnvironmentNameRegistry::solver()
{
    static const EnvironmentNameRegistry registry(solverCatalogue());
    return registry;
}

const NameDeclaration* EnvironmentNameRegistry::find(std::string_view name) const noexcept
{
    const auto it = std::lower_bound(
        by_name_.begin(), by_name_.end(), name,
        [](const NameDeclaration* declaration, std::string_view key) {
            return std::string_view(declaration->name) < key;
        });
    if (it == by_name_.end() || std::string_view((*it)->name) != name) {
        return nullptr;
    }
    return *it;
}

const NameDeclaration& EnvironmentNameRegistry::declaration(std::string_view name) const
{
    if (const auto* declaration = find(name)) {
        return *declaration;
    }
    throw std::logic_error("EnvironmentNameRegistry: the environment name '" + std::string(name) +
                           "' is not declared");
}

const char* EnvironmentNameRegistry::value(std::string_view name) const
{
    return std::getenv(declaration(name).name);
}

bool EnvironmentNameRegistry::isSet(std::string_view name) const
{
    return value(name) != nullptr;
}

bool EnvironmentNameRegistry::enabled(std::string_view name) const
{
    const auto& declared = declaration(name);
    if (declared.type != ValueType::Flag) {
        throw std::logic_error(std::string("EnvironmentNameRegistry: '") + declared.name +
                               "' is not declared as a flag");
    }
    const char* text = std::getenv(declared.name);
    if (text == nullptr || text[0] == '\0') {
        return false;
    }
    std::string lowered(text);
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return !(lowered == "0" || lowered == "false" || lowered == "off" || lowered == "no");
}

std::vector<NameUse> EnvironmentNameRegistry::setNames() const
{
    std::vector<NameUse> uses;
    for (const auto* declaration : by_name_) {
        if (const char* text = std::getenv(declaration->name)) {
            uses.push_back(NameUse{.declaration = declaration, .value = text});
        }
    }
    return uses;
}

std::string EnvironmentNameRegistry::configurationRecord() const
{
    namespace json = options::json;
    std::string names;
    bool reference_run = true;
    for (const auto& use : setNames()) {
        const auto& declared = *use.declaration;
        if (declared.kind == NameKind::Diagnostic) {
            continue;
        }
        reference_run = reference_run && !declared.marksNonReference();
        if (!names.empty()) {
            names += ',';
        }
        names += "{\"name\":" + json::quote(declared.name) +
                 ",\"value\":" + json::quote(use.value) +
                 ",\"kind\":" + json::quote(toString(declared.kind)) +
                 ",\"value_type\":" + json::quote(toString(declared.type)) +
                 ",\"owner\":" + json::quote(declared.owner) +
                 ",\"marks_non_reference\":" + json::boolean(declared.marksNonReference()) + "}";
    }
    if (names.empty()) {
        return {};
    }
    return std::string("\"environment_names\":{\"reference_run\":") + json::boolean(reference_run) +
           ",\"set\":[" + names + "]}";
}

const char* value(std::string_view name)
{
    return EnvironmentNameRegistry::solver().value(name);
}

bool isSet(std::string_view name)
{
    return EnvironmentNameRegistry::solver().isSet(name);
}

bool enabled(std::string_view name)
{
    return EnvironmentNameRegistry::solver().enabled(name);
}

std::string configurationRecord()
{
    return EnvironmentNameRegistry::solver().configurationRecord();
}

} // namespace environment
} // namespace FE
} // namespace svmp
