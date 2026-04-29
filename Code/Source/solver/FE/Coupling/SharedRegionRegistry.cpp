/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/SharedRegionRegistry.h"

#include <algorithm>
#include <utility>

namespace svmp {
namespace FE {
namespace coupling {

void SharedRegionRegistry::add(SharedRegionRef region)
{
    records_.push_back(std::move(region));
}

const SharedRegionRef* SharedRegionRegistry::find(std::string_view name) const noexcept
{
    const auto it = std::find_if(records_.begin(), records_.end(),
                                 [name](const SharedRegionRef& record) {
                                     return record.name == name;
                                 });
    return it == records_.end() ? nullptr : &*it;
}

const CouplingRegionRef* SharedRegionRegistry::findParticipantRegion(
    std::string_view name,
    std::string_view participant) const noexcept
{
    const SharedRegionRef* group = find(name);
    if (group == nullptr) {
        return nullptr;
    }
    const auto it = std::find_if(group->participant_regions.begin(),
                                 group->participant_regions.end(),
                                 [participant](const CouplingRegionRef& region) {
                                     return region.participant_name == participant;
                                 });
    return it == group->participant_regions.end() ? nullptr : &*it;
}

CouplingValidationResult SharedRegionRegistry::validate() const
{
    CouplingValidationResult result;
    for (std::size_t i = 0; i < records_.size(); ++i) {
        const auto& record = records_[i];
        if (record.name.empty()) {
            result.addError("shared region requires a name");
        }
        for (std::size_t j = i + 1u; j < records_.size(); ++j) {
            if (!record.name.empty() && record.name == records_[j].name) {
                result.addError("duplicate shared region name");
            }
        }
        for (const auto& region : record.participant_regions) {
            if (!region.valid()) {
                result.addError("shared region contains an invalid participant region");
            }
            if (record.required_region_kind.has_value() &&
                region.kind != *record.required_region_kind) {
                result.addError("shared region participant mapping does not match the required region kind");
            }
        }
    }
    return result;
}

const std::vector<SharedRegionRef>& SharedRegionRegistry::records() const noexcept
{
    return records_;
}

} // namespace coupling
} // namespace FE
} // namespace svmp
