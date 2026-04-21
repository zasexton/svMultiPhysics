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

} // namespace systems
} // namespace FE
} // namespace svmp
