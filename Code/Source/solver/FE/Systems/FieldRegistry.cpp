/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/FieldRegistry.h"

#include "Spaces/FunctionSpace.h"
#include "Spaces/MortarSpace.h"

#include <algorithm>

namespace svmp {
namespace FE {
namespace systems {

FieldId FieldRegistry::add(FieldSpec spec)
{
    FE_THROW_IF(spec.name.empty(), InvalidArgumentException, "FieldRegistry::add: field name is empty");
    FE_CHECK_NOT_NULL(spec.space.get(), "FieldRegistry::add: spec.space");

    FE_THROW_IF(name_to_id_.count(spec.name) > 0, InvalidArgumentException,
                "FieldRegistry::add: field '" + spec.name + "' already exists");

    FE_THROW_IF(next_id_ == INVALID_FIELD_ID, InvalidArgumentException,
                "FieldRegistry::add: out of FieldId values");

    FieldRecord rec;
    rec.id = next_id_++;
    rec.name = std::move(spec.name);
    rec.space = std::move(spec.space);
    rec.components = spec.components > 0 ? spec.components : rec.space->value_dimension();
    rec.scope = spec.scope;
    rec.interface_marker = spec.interface_marker;
    rec.participant_name = std::move(spec.participant_name);
    rec.participant_domain_id = spec.participant_domain_id;
    rec.source_kind = spec.source_kind;
    rec.derived = spec.derived;

    if (rec.space->space_type() == spaces::SpaceType::Mortar) {
        const auto* mortar_space = dynamic_cast<const spaces::MortarSpace*>(rec.space.get());
        FE_CHECK_NOT_NULL(mortar_space, "FieldRegistry::add: mortar field space");
        rec.scope = FieldScope::InterfaceFace;
        if (rec.interface_marker >= 0) {
            FE_THROW_IF(rec.interface_marker != mortar_space->interface_marker(),
                        InvalidArgumentException,
                        "FieldRegistry::add: mortar field '" + rec.name +
                            "' was given interface_marker=" + std::to_string(rec.interface_marker) +
                            ", but MortarSpace is scoped to marker " +
                            std::to_string(mortar_space->interface_marker()));
        }
        rec.interface_marker = mortar_space->interface_marker();
    }

    if (rec.scope == FieldScope::InterfaceFace) {
        FE_THROW_IF(rec.interface_marker < 0, InvalidArgumentException,
                    "FieldRegistry::add: interface-scoped field '" + rec.name +
                        "' must specify interface_marker >= 0");
    } else {
        FE_THROW_IF(rec.interface_marker >= 0, InvalidArgumentException,
                    "FieldRegistry::add: volume field '" + rec.name +
                        "' cannot specify interface_marker");
    }
    if (rec.participant_name.has_value()) {
        FE_THROW_IF(rec.participant_name->empty(), InvalidArgumentException,
                    "FieldRegistry::add: participant-scoped field '" + rec.name +
                        "' cannot use an empty participant name");
    }
    FE_THROW_IF(rec.participant_name.has_value() && rec.participant_domain_id.has_value(),
                InvalidArgumentException,
                "FieldRegistry::add: field '" + rec.name +
                    "' cannot specify both participant_name and participant_domain_id");

    if (rec.source_kind == FieldSourceKind::DerivedFromUnknown) {
        FE_THROW_IF(rec.derived.source_field == INVALID_FIELD_ID,
                    InvalidArgumentException,
                    "FieldRegistry::add: derived field '" + rec.name +
                        "' must name a source field");
        FE_THROW_IF(rec.derived.role == DerivedFieldRole::None ||
                        rec.derived.derivative_order <= 0,
                    InvalidArgumentException,
                    "FieldRegistry::add: derived field '" + rec.name +
                        "' must specify a valid derived role and derivative order");
    } else {
        FE_THROW_IF(rec.derived.source_field != INVALID_FIELD_ID ||
                        rec.derived.role != DerivedFieldRole::None ||
                        rec.derived.derivative_order != 0,
                    InvalidArgumentException,
                    "FieldRegistry::add: non-derived field '" + rec.name +
                        "' cannot carry derived-field metadata");
    }

    fields_.push_back(rec);
    name_to_id_[fields_.back().name] = fields_.back().id;
    return fields_.back().id;
}

const FieldRecord& FieldRegistry::get(FieldId id) const
{
    FE_THROW_IF(id == INVALID_FIELD_ID, InvalidArgumentException, "FieldRegistry::get: invalid field id");
    FE_THROW_IF(static_cast<std::size_t>(id) >= fields_.size(), InvalidArgumentException, "FieldRegistry::get: unknown FieldId");
    return fields_[static_cast<std::size_t>(id)];
}

void FieldRegistry::markTimeDependent(FieldId id, int max_order)
{
    if (max_order <= 0) {
        return;
    }

    FE_THROW_IF(id == INVALID_FIELD_ID, InvalidArgumentException, "FieldRegistry::markTimeDependent: invalid field id");
    FE_THROW_IF(static_cast<std::size_t>(id) >= fields_.size(), InvalidArgumentException, "FieldRegistry::markTimeDependent: unknown FieldId");
    
    auto& f = fields_[static_cast<std::size_t>(id)];
    f.time_dependent = true;
    f.max_time_derivative_order = std::max(f.max_time_derivative_order, max_order);
}

FieldId FieldRegistry::findByName(std::string_view name) const noexcept
{
    auto it = name_to_id_.find(std::string(name));
    if (it == name_to_id_.end()) {
        return INVALID_FIELD_ID;
    }
    return it->second;
}

bool FieldRegistry::has(FieldId id) const noexcept
{
    if (id == INVALID_FIELD_ID) {
        return false;
    }
    return static_cast<std::size_t>(id) < fields_.size();
}

StateGroupId FieldRegistry::addStateGroup(StateGroupSpec spec)
{
    FE_THROW_IF(spec.name.empty(), InvalidArgumentException,
                "FieldRegistry::addStateGroup: state group name is empty");
    FE_THROW_IF(state_group_name_to_id_.count(spec.name) > 0, InvalidArgumentException,
                "FieldRegistry::addStateGroup: state group '" + spec.name + "' already exists");
    FE_THROW_IF(next_state_group_id_ == INVALID_STATE_GROUP_ID, InvalidArgumentException,
                "FieldRegistry::addStateGroup: out of StateGroupId values");

    std::vector<FieldId> resolved_fields = spec.fields;
    if (!spec.field_names.empty()) {
        for (const auto& field_name : spec.field_names) {
            FE_THROW_IF(field_name.empty(), InvalidArgumentException,
                        "FieldRegistry::addStateGroup: state group '" + spec.name +
                            "' references an empty field name");
            const FieldId field_id = findByName(field_name);
            FE_THROW_IF(field_id == INVALID_FIELD_ID, InvalidArgumentException,
                        "FieldRegistry::addStateGroup: state group '" + spec.name +
                            "' references unknown field '" + field_name + "'");
            if (std::find(resolved_fields.begin(), resolved_fields.end(), field_id) ==
                resolved_fields.end()) {
                resolved_fields.push_back(field_id);
            }
        }
    }

    FE_THROW_IF(resolved_fields.empty(), InvalidArgumentException,
                "FieldRegistry::addStateGroup: state group '" + spec.name +
                    "' must contain at least one field");

    std::vector<std::string> resolved_names;
    resolved_names.reserve(resolved_fields.size());
    int inferred_components = 0;
    for (const FieldId field_id : resolved_fields) {
        FE_THROW_IF(!has(field_id), InvalidArgumentException,
                    "FieldRegistry::addStateGroup: state group '" + spec.name +
                        "' references an unknown FieldId");
        const auto& field = get(field_id);
        resolved_names.push_back(field.name);
        inferred_components += std::max(field.components, 0);
    }

    FE_THROW_IF(spec.component_count < 0, InvalidArgumentException,
                "FieldRegistry::addStateGroup: state group '" + spec.name +
                    "' cannot specify a negative component count");
    FE_THROW_IF(spec.bounds.lower.has_value() && spec.bounds.upper.has_value() &&
                    *spec.bounds.lower > *spec.bounds.upper,
                InvalidArgumentException,
                "FieldRegistry::addStateGroup: state group '" + spec.name +
                    "' has lower bound greater than upper bound");
    if (spec.sum_constraint.has_value()) {
        FE_THROW_IF(spec.sum_constraint->tolerance < 0.0, InvalidArgumentException,
                    "FieldRegistry::addStateGroup: state group '" + spec.name +
                        "' cannot specify a negative sum-constraint tolerance");
    }
    if (spec.conserved_quantity_name.has_value()) {
        FE_THROW_IF(spec.conserved_quantity_name->empty(), InvalidArgumentException,
                    "FieldRegistry::addStateGroup: state group '" + spec.name +
                        "' cannot use an empty conserved quantity name");
    }
    for (const auto& tag : spec.analysis_tags) {
        FE_THROW_IF(tag.empty(), InvalidArgumentException,
                    "FieldRegistry::addStateGroup: state group '" + spec.name +
                        "' cannot use an empty analysis tag");
    }

    StateGroupRecord rec;
    rec.id = next_state_group_id_++;
    rec.name = std::move(spec.name);
    rec.kind = spec.kind;
    rec.shape = spec.shape;
    rec.fields = std::move(resolved_fields);
    rec.field_names = std::move(resolved_names);
    rec.component_count = spec.component_count > 0 ? spec.component_count : inferred_components;
    rec.conserved_quantity_name = std::move(spec.conserved_quantity_name);
    rec.sum_constraint = spec.sum_constraint;
    rec.bounds = spec.bounds;
    rec.analysis_tags = std::move(spec.analysis_tags);

    state_groups_.push_back(rec);
    state_group_name_to_id_[state_groups_.back().name] = state_groups_.back().id;
    return state_groups_.back().id;
}

const StateGroupRecord& FieldRegistry::getStateGroup(StateGroupId id) const
{
    FE_THROW_IF(id == INVALID_STATE_GROUP_ID, InvalidArgumentException,
                "FieldRegistry::getStateGroup: invalid state group id");
    FE_THROW_IF(static_cast<std::size_t>(id) >= state_groups_.size(), InvalidArgumentException,
                "FieldRegistry::getStateGroup: unknown StateGroupId");
    return state_groups_[static_cast<std::size_t>(id)];
}

StateGroupId FieldRegistry::findStateGroupByName(std::string_view name) const noexcept
{
    auto it = state_group_name_to_id_.find(std::string(name));
    if (it == state_group_name_to_id_.end()) {
        return INVALID_STATE_GROUP_ID;
    }
    return it->second;
}

bool FieldRegistry::hasStateGroup(StateGroupId id) const noexcept
{
    if (id == INVALID_STATE_GROUP_ID) {
        return false;
    }
    return static_cast<std::size_t>(id) < state_groups_.size();
}

} // namespace systems
} // namespace FE
} // namespace svmp
