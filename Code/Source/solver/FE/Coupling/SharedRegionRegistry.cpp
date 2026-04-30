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

namespace {

void validateInterfaceSideOwnership(const SharedRegionRef& record,
                                    CouplingValidationResult& result)
{
    const CouplingRegionRef* minus_owner = nullptr;
    const CouplingRegionRef* plus_owner = nullptr;

    for (const auto& region : record.participant_regions) {
        if (region.kind != CouplingRegionKind::InterfaceFace) {
            continue;
        }

        if (region.side == CouplingInterfaceSide::None) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = region.participant_name,
                .region_name = record.name,
                .message = "interface shared-region participant mapping requires a minus or plus side",
            });
            continue;
        }

        const CouplingRegionRef*& owner =
            region.side == CouplingInterfaceSide::Minus ? minus_owner : plus_owner;
        if (owner != nullptr) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = region.participant_name,
                .region_name = record.name,
                .message = "interface shared-region side ownership is ambiguous",
            });
            continue;
        }
        owner = &region;
    }
}

} // namespace

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
        for (const auto& participant : record.required_participant_names) {
            const auto found = std::any_of(
                record.participant_regions.begin(),
                record.participant_regions.end(),
                [&participant](const CouplingRegionRef& region) {
                    return region.participant_name == participant;
                });
            if (!found) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .participant_name = participant,
                    .region_name = record.name,
                    .message = "shared region is missing required participant mapping",
                });
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
        validateInterfaceSideOwnership(record, result);
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
