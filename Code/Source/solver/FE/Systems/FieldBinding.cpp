/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/FieldBinding.h"

#include "Spaces/FunctionSpace.h"
#include "Systems/FESystem.h"

#include <stdexcept>
#include <unordered_map>

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

FieldId FieldBinding::prescribed(FieldSpec spec, bool auto_register, std::string_view role)
{
    check(*system_, spec, FieldSourceKind::PrescribedData, /*allow_missing=*/auto_register, roleContext(role));
    const auto existing = system_->findFieldByName(spec.name);
    if (existing == INVALID_FIELD_ID) {
        return system_->addField(std::move(spec));
    }
    return existing;
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
