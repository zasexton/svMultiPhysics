/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/FieldBinding.h"

#include "Forms/Vocabulary.h"
#include "Spaces/FunctionSpace.h"
#include "Spaces/GlobalSpace.h"
#include "Spaces/MortarSpace.h"
#include "Systems/FESystem.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

bool spacesCompatible(const spaces::FunctionSpace& lhs,
                      const spaces::FunctionSpace& rhs) noexcept
{
    return lhs.space_type() == rhs.space_type() &&
           lhs.field_type() == rhs.field_type() &&
           lhs.value_dimension() == rhs.value_dimension() &&
           lhs.topological_dimension() == rhs.topological_dimension() &&
           lhs.polynomial_order() == rhs.polynomial_order() &&
           lhs.element_type() == rhs.element_type();
}

namespace {

/// The shadow fields of a system (FieldBinding::lagged and previous).
struct ShadowFieldRecords {
    std::vector<ShadowField> fields;
};

/// The scope and marker FieldRegistry::add gives @p spec.
[[nodiscard]] std::pair<FieldScope, int> registeredScope(const FieldSpec& spec)
{
    if (spec.space->space_type() == spaces::SpaceType::Global) {
        return {FieldScope::Global, -1};
    }
    if (spec.space->space_type() == spaces::SpaceType::Mortar) {
        const auto* mortar = dynamic_cast<const spaces::MortarSpace*>(spec.space.get());
        return {FieldScope::InterfaceFace, mortar != nullptr ? mortar->interface_marker() : spec.interface_marker};
    }
    return {spec.scope, spec.interface_marker};
}

[[nodiscard]] std::string scopeText(FieldScope scope, int marker)
{
    switch (scope) {
    case FieldScope::VolumeCell:
        return "a volume field";
    case FieldScope::InterfaceFace:
        return "an interface field on marker " + std::to_string(marker);
    case FieldScope::Global:
        return "a global field";
    }
    return "a field of unknown scope";
}

void requireSourceKind(const FieldSpec& spec, FieldSourceKind expected, std::string_view context)
{
    if (spec.source_kind != expected) {
        throw std::invalid_argument(
            std::string(context) + ": field '" + spec.name +
            (expected == FieldSourceKind::Unknown
                 ? "' must be specified as an unknown field"
                 : "' must be specified as a prescribed data field"));
    }
}

} // namespace

forms::FormExpr derivative(const forms::FormExpr& form, const BoundUnknown& unknown)
{
    return forms::derivative(form, unknown.state, unknown.test);
}

forms::FormExpr derivative(const forms::FormExpr& form, const BoundUnknown& unknown,
                           const forms::FormExpr& direction)
{
    return forms::derivative(form, unknown.state, direction);
}

forms::FormExpr derivative(const forms::FormExpr& form, std::initializer_list<BoundUnknown> unknowns)
{
    std::vector<std::pair<forms::FormExpr, forms::FormExpr>> variations;
    variations.reserve(unknowns.size());
    for (const auto& unknown : unknowns) {
        variations.emplace_back(unknown.state, unknown.test);
    }
    return forms::derivative(form, variations);
}

FieldBinding::FieldBinding(FESystem& system, std::string owner)
    : system_(&system)
    , owner_(std::move(owner))
{
}

void FieldBinding::check(const FESystem& system,
                         const FieldSpec& spec,
                         FieldSourceKind expected_source,
                         bool allow_missing,
                         std::string_view context)
{
    if (spec.name.empty()) {
        throw std::invalid_argument(
            std::string(context) + ": field name must be non-empty");
    }
    if (!spec.space) {
        throw std::invalid_argument(
            std::string(context) + ": field '" + spec.name +
            "' requires a function space");
    }
    const auto existing = system.findFieldByName(spec.name);
    if (existing == INVALID_FIELD_ID) {
        if (!allow_missing) {
            throw std::invalid_argument(
                std::string(context) + ": prescribed field '" + spec.name +
                "' was requested but is not registered");
        }
        return;
    }

    const auto& rec = system.fieldRecord(existing);
    if (rec.source_kind != expected_source) {
        throw std::invalid_argument(
            std::string(context) + ": existing field '" + rec.name +
            (expected_source == FieldSourceKind::Unknown
                 ? "' must be an unknown field"
                 : "' must be a prescribed data field"));
    }
    // The scope before the components and the space: a global or an interface
    // field differs from a volume field in its space too, and the scope is
    // what the caller got wrong.
    const auto [scope, marker] = registeredScope(spec);
    if (rec.scope != scope || (scope == FieldScope::InterfaceFace && rec.interface_marker != marker)) {
        throw std::invalid_argument(
            std::string(context) + ": existing field '" + rec.name + "' is " +
            scopeText(rec.scope, rec.interface_marker) + ", expected " + scopeText(scope, marker));
    }
    if (rec.components != spec.components) {
        throw std::invalid_argument(
            std::string(context) + ": existing field '" + rec.name +
            "' has component count " + std::to_string(rec.components) +
            ", expected " + std::to_string(spec.components));
    }
    if (!rec.space || !spacesCompatible(*rec.space, *spec.space)) {
        throw std::invalid_argument(
            std::string(context) + ": existing field '" + rec.name +
            "' uses an incompatible function space");
    }
}

FieldId FieldBinding::unknown(FieldSpec spec, std::string_view role)
{
    check(*system_, spec, FieldSourceKind::Unknown, /*allow_missing=*/true, roleContext(role));
    const auto existing = system_->findFieldByName(spec.name);
    if (existing == INVALID_FIELD_ID) {
        return system_->addField(std::move(spec));
    }
    return existing;
}

BoundUnknown FieldBinding::unknownSymbols(FieldSpec spec, std::string_view role,
                                          std::string state_symbol, std::string test_symbol)
{
    // Preserve check()'s diagnostics before checking the caller's source kind.
    check(*system_, spec, FieldSourceKind::Unknown, /*allow_missing=*/true, roleContext(role));
    requireSourceKind(spec, FieldSourceKind::Unknown, roleContext(role));
    const auto space = spec.space;
    const auto id = unknown(std::move(spec), role);
    return {id, forms::FormExpr::stateField(id, *space, std::move(state_symbol)),
            forms::FormExpr::testFunction(id, *space, std::move(test_symbol))};
}

BoundUnknown FieldBinding::addUnknownSymbols(FieldSpec spec,
                                             std::string state_symbol, std::string test_symbol)
{
    requireSourceKind(spec, FieldSourceKind::Unknown, owner_);
    const auto space = spec.space;
    const auto id = system_->addField(std::move(spec));
    return {id, forms::FormExpr::stateField(id, *space, std::move(state_symbol)),
            forms::FormExpr::testFunction(id, *space, std::move(test_symbol))};
}

FieldId FieldBinding::global(std::string name, std::string_view role)
{
    FieldSpec spec;
    spec.name = std::move(name);
    spec.space = std::make_shared<spaces::GlobalSpace>(1);
    spec.components = 1;
    spec.scope = FieldScope::Global;
    return unknown(std::move(spec), role);
}

FieldId FieldBinding::prescribed(FieldSpec spec, bool auto_register, std::string_view role)
{
    check(*system_, spec, FieldSourceKind::PrescribedData, /*allow_missing=*/auto_register, roleContext(role));
    const auto existing = system_->findFieldByName(spec.name);
    if (existing == INVALID_FIELD_ID) {
        return system_->addField(std::move(spec));
    }
    return existing;
}

BoundPrescribed FieldBinding::prescribedSymbol(FieldSpec spec, bool auto_register,
                                               std::string_view role, std::string symbol)
{
    check(*system_, spec, FieldSourceKind::PrescribedData, /*allow_missing=*/auto_register, roleContext(role));
    requireSourceKind(spec, FieldSourceKind::PrescribedData, roleContext(role));
    const auto space = spec.space;
    const auto id = prescribed(std::move(spec), auto_register, role);
    return {id, forms::FormExpr::stateField(id, *space, std::move(symbol))};
}

std::vector<ShadowField> shadowFields(const FESystem& system)
{
    const auto* records = system.findAttachment<ShadowFieldRecords>();
    return records != nullptr ? records->fields : std::vector<ShadowField>{};
}

forms::FormExpr FieldBinding::lagged(const BoundUnknown& unknown, Refresh refresh, std::string shadow_name)
{
    if (shadow_name.empty() && unknown.id != INVALID_FIELD_ID) {
        shadow_name = system_->fieldRecord(unknown.id).name + "_lagged";
    }
    return shadow(unknown, ShadowField{.kind = ShadowField::Kind::Lagged, .refresh = refresh}, std::move(shadow_name),
                  "lagged copy");
}

forms::FormExpr FieldBinding::previous(const BoundUnknown& unknown, int k, std::string shadow_name)
{
    if (k < 1) {
        throw std::invalid_argument(roleContext("previous value") + ": the value k = " + std::to_string(k) +
                                    " steps back is not k >= 1");
    }
    if (shadow_name.empty() && unknown.id != INVALID_FIELD_ID) {
        shadow_name = system_->fieldRecord(unknown.id).name + "_previous" + std::to_string(k);
    }
    return shadow(unknown, ShadowField{.kind = ShadowField::Kind::Previous, .steps_back = k}, std::move(shadow_name),
                  "previous value");
}

forms::FormExpr FieldBinding::shadow(const BoundUnknown& unknown, ShadowField record, std::string shadow_name,
                                     std::string_view role)
{
    const auto context = roleContext(role);
    if (unknown.id == INVALID_FIELD_ID || !system_->hasField(unknown.id) ||
        !system_->fieldParticipatesInUnknownVector(unknown.id)) {
        throw std::invalid_argument(context + ": the source is not a registered unknown");
    }
    const auto& source = system_->fieldRecord(unknown.id);
    const auto space = source.space;
    const auto id = prescribed(FieldSpec{.name = shadow_name,
                                         .space = space,
                                         .components = source.components,
                                         .scope = source.scope,
                                         .interface_marker = source.interface_marker,
                                         .source_kind = FieldSourceKind::PrescribedData},
                               /*auto_register=*/true, role);
    record.source = unknown.id;
    record.shadow = id;
    auto& records = system_->attachment<ShadowFieldRecords>().fields;
    const auto existing = std::find_if(records.begin(), records.end(),
                                       [&](const ShadowField& r) { return r.shadow == id; });
    if (existing == records.end()) {
        records.push_back(record);
    } else if (existing->source != record.source || existing->kind != record.kind ||
               existing->refresh != record.refresh || existing->steps_back != record.steps_back) {
        throw std::invalid_argument(context + ": field '" + shadow_name + "' is already another shadow field");
    }
    return forms::FormExpr::discreteField(id, *space, std::move(shadow_name));
}

void FieldBinding::requireDistinctNames(
    std::string_view owner,
    std::initializer_list<std::pair<std::string_view, std::string_view>> role_names)
{
    std::unordered_map<std::string_view, std::string_view> roles;
    for (const auto& [role, name] : role_names) {
        if (name.empty()) {
            continue;
        }
        const auto [existing, inserted] = roles.emplace(name, role);
        if (!inserted) {
            throw std::invalid_argument(
                std::string(owner) + ": field name '" + std::string(name) +
                "' is assigned to both " + std::string(existing->second) +
                " and " + std::string(role));
        }
    }
}

std::string FieldBinding::roleContext(std::string_view role) const
{
    return owner_ + " " + std::string(role);
}

} // namespace systems
} // namespace FE
} // namespace svmp
