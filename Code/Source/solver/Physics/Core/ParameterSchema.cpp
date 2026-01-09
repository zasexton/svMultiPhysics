/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Core/ParameterSchema.h"

namespace svmp {
namespace Physics {

void ParameterSchema::clear()
{
    registry_.clear();
}

void ParameterSchema::add(FE::params::Spec spec, std::string source)
{
    registry_.add(std::move(spec), std::move(source));
}

void ParameterSchema::addAll(const std::vector<FE::params::Spec>& specs, std::string source)
{
    registry_.addAll(specs, std::move(source));
}

const std::vector<FE::params::Spec>& ParameterSchema::specs() const noexcept
{
    return registry_.specs();
}

const FE::params::Spec* ParameterSchema::find(std::string_view key) const noexcept
{
    return registry_.find(key);
}

void ParameterSchema::validate(const FE::systems::SystemStateView& state) const
{
    registry_.validate(state);
}

} // namespace Physics
} // namespace svmp

