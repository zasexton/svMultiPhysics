/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/OperatorRegistry.h"

#include <algorithm>

namespace svmp {
namespace FE {
namespace systems {

void OperatorRegistry::addOperator(OperatorTag tag)
{
    FE_THROW_IF(tag.empty(), InvalidArgumentException, "OperatorRegistry::addOperator: empty tag");
    FE_THROW_IF(ops_.count(tag) > 0, InvalidArgumentException,
                "OperatorRegistry::addOperator: operator '" + tag + "' already exists");
    OperatorDefinition def;
    def.tag = std::move(tag);
    ops_.emplace(def.tag, std::move(def));
}

bool OperatorRegistry::has(const OperatorTag& tag) const noexcept
{
    return ops_.count(tag) > 0;
}

OperatorDefinition& OperatorRegistry::get(const OperatorTag& tag)
{
    auto it = ops_.find(tag);
    FE_THROW_IF(it == ops_.end(), InvalidArgumentException,
                "OperatorRegistry::get: unknown operator '" + tag + "'");
    return it->second;
}

const OperatorDefinition& OperatorRegistry::get(const OperatorTag& tag) const
{
    auto it = ops_.find(tag);
    FE_THROW_IF(it == ops_.end(), InvalidArgumentException,
                "OperatorRegistry::get: unknown operator '" + tag + "'");
    return it->second;
}

std::vector<OperatorTag> OperatorRegistry::list() const
{
    std::vector<OperatorTag> tags;
    tags.reserve(ops_.size());
    for (const auto& kv : ops_) {
        tags.push_back(kv.first);
    }
    std::sort(tags.begin(), tags.end());
    return tags;
}

} // namespace systems
} // namespace FE
} // namespace svmp
