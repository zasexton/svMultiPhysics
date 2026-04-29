/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/CouplingContext.h"

#include "Coupling/SharedRegionRegistry.h"
#include "Core/FEException.h"

#include <algorithm>
#include <utility>

namespace svmp {
namespace FE {
namespace coupling {

bool CouplingParticipantRef::valid() const noexcept
{
    return !participant_name.empty() && !system_name.empty() && system != nullptr;
}

bool CouplingFieldRef::valid() const noexcept
{
    return !participant_name.empty() && !system_name.empty() && system != nullptr &&
           !field_name.empty() && field_id != INVALID_FIELD_ID && space != nullptr &&
           components > 0;
}

bool CouplingRegionRef::valid() const noexcept
{
    return !participant_name.empty() && !system_name.empty() && system != nullptr &&
           !region_name.empty();
}

CouplingParticipantRef CouplingContext::participant(std::string_view participant_name) const
{
    const auto it = std::find_if(participants_.begin(), participants_.end(),
                                 [participant_name](const CouplingParticipantRef& participant) {
                                     return participant.participant_name == participant_name;
                                 });
    FE_THROW_IF(it == participants_.end(), InvalidArgumentException,
                "unknown coupling participant");
    return *it;
}

CouplingFieldRef CouplingContext::field(std::string_view participant_name,
                                        std::string_view field_name) const
{
    const auto it = std::find_if(fields_.begin(), fields_.end(),
                                 [participant_name, field_name](const CouplingFieldRef& field) {
                                     return field.participant_name == participant_name &&
                                            field.field_name == field_name;
                                 });
    FE_THROW_IF(it == fields_.end(), InvalidArgumentException,
                "unknown coupling field");
    return *it;
}

CouplingRegionRef CouplingContext::region(std::string_view participant_name,
                                          std::string_view region_name) const
{
    const auto it = std::find_if(regions_.begin(), regions_.end(),
                                 [participant_name, region_name](const CouplingRegionRef& region) {
                                     return region.participant_name == participant_name &&
                                            region.region_name == region_name;
                                 });
    FE_THROW_IF(it == regions_.end(), InvalidArgumentException,
                "unknown coupling region");
    return *it;
}

CouplingRegionRef CouplingContext::sharedRegion(std::string_view name,
                                                std::string_view participant_name) const
{
    const auto group = sharedRegionGroup(name);
    const auto it = std::find_if(group.participant_regions.begin(),
                                 group.participant_regions.end(),
                                 [participant_name](const CouplingRegionRef& region) {
                                     return region.participant_name == participant_name;
                                 });
    FE_THROW_IF(it == group.participant_regions.end(), InvalidArgumentException,
                "unknown coupling shared-region participant mapping");
    return *it;
}

SharedRegionRef CouplingContext::sharedRegionGroup(std::string_view name) const
{
    const auto it = std::find_if(shared_regions_.begin(), shared_regions_.end(),
                                 [name](const SharedRegionRef& region) {
                                     return region.name == name;
                                 });
    FE_THROW_IF(it == shared_regions_.end(), InvalidArgumentException,
                "unknown coupling shared region");
    return *it;
}

bool CouplingContext::hasParticipant(std::string_view participant_name) const noexcept
{
    return std::any_of(participants_.begin(), participants_.end(),
                       [participant_name](const CouplingParticipantRef& participant) {
                           return participant.participant_name == participant_name;
                       });
}

bool CouplingContext::hasField(std::string_view participant_name,
                               std::string_view field_name) const noexcept
{
    return std::any_of(fields_.begin(), fields_.end(),
                       [participant_name, field_name](const CouplingFieldRef& field) {
                           return field.participant_name == participant_name &&
                                  field.field_name == field_name;
                       });
}

bool CouplingContext::hasRegion(std::string_view participant_name,
                                std::string_view region_name) const noexcept
{
    return std::any_of(regions_.begin(), regions_.end(),
                       [participant_name, region_name](const CouplingRegionRef& region) {
                           return region.participant_name == participant_name &&
                                  region.region_name == region_name;
                       });
}

bool CouplingContext::hasSharedRegion(std::string_view name) const noexcept
{
    return std::any_of(shared_regions_.begin(), shared_regions_.end(),
                       [name](const SharedRegionRef& region) {
                           return region.name == name;
                       });
}

const std::vector<CouplingParticipantRef>& CouplingContext::participants() const noexcept
{
    return participants_;
}

const std::vector<CouplingFieldRef>& CouplingContext::fields() const noexcept
{
    return fields_;
}

const std::vector<CouplingRegionRef>& CouplingContext::regions() const noexcept
{
    return regions_;
}

const std::vector<SharedRegionRef>& CouplingContext::sharedRegions() const noexcept
{
    return shared_regions_;
}

CouplingContextBuilder& CouplingContextBuilder::addParticipant(CouplingParticipantRef participant)
{
    context_.participants_.push_back(std::move(participant));
    return *this;
}

CouplingContextBuilder& CouplingContextBuilder::addField(CouplingFieldRef field)
{
    context_.fields_.push_back(std::move(field));
    return *this;
}

CouplingContextBuilder& CouplingContextBuilder::addRegion(CouplingRegionRef region)
{
    context_.regions_.push_back(std::move(region));
    return *this;
}

CouplingContextBuilder& CouplingContextBuilder::addSharedRegion(SharedRegionRef region)
{
    context_.shared_regions_.push_back(std::move(region));
    return *this;
}

CouplingValidationResult CouplingContextBuilder::validate() const
{
    CouplingValidationResult result;

    for (std::size_t i = 0; i < context_.participants_.size(); ++i) {
        const auto& participant = context_.participants_[i];
        if (!participant.valid()) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = participant.participant_name,
                .message = "coupling participant requires participant name, system name, and owning system",
            });
        }
        for (std::size_t j = i + 1u; j < context_.participants_.size(); ++j) {
            if (!participant.participant_name.empty() &&
                participant.participant_name == context_.participants_[j].participant_name) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .participant_name = participant.participant_name,
                    .message = "duplicate coupling participant",
                });
            }
        }
    }

    for (std::size_t i = 0; i < context_.fields_.size(); ++i) {
        const auto& field = context_.fields_[i];
        if (!field.valid()) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = field.participant_name,
                .field_name = field.field_name,
                .message = "coupling field requires participant, system ownership, field id, space, and positive component count",
            });
        }
        if (!field.participant_name.empty() &&
            !context_.hasParticipant(field.participant_name)) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = field.participant_name,
                .field_name = field.field_name,
                .message = "coupling field references an unknown participant",
            });
        }
        for (std::size_t j = i + 1u; j < context_.fields_.size(); ++j) {
            if (field.participant_name == context_.fields_[j].participant_name &&
                field.field_name == context_.fields_[j].field_name) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .participant_name = field.participant_name,
                    .field_name = field.field_name,
                    .message = "duplicate coupling field mapping",
                });
            }
        }
    }

    for (std::size_t i = 0; i < context_.regions_.size(); ++i) {
        const auto& region = context_.regions_[i];
        if (!region.valid()) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = region.participant_name,
                .region_name = region.region_name,
                .message = "coupling region requires participant, system ownership, and region name",
            });
        }
        if (!region.participant_name.empty() &&
            !context_.hasParticipant(region.participant_name)) {
            result.add(CouplingDiagnostic{
                .severity = CouplingDiagnosticSeverity::Error,
                .participant_name = region.participant_name,
                .region_name = region.region_name,
                .message = "coupling region references an unknown participant",
            });
        }
        for (std::size_t j = i + 1u; j < context_.regions_.size(); ++j) {
            if (region.participant_name == context_.regions_[j].participant_name &&
                region.region_name == context_.regions_[j].region_name) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .participant_name = region.participant_name,
                    .region_name = region.region_name,
                    .message = "duplicate coupling region mapping",
                });
            }
        }
    }

    SharedRegionRegistry registry;
    for (const auto& region : context_.shared_regions_) {
        registry.add(region);
    }
    result.append(registry.validate());

    for (const auto& shared_region : context_.shared_regions_) {
        for (const auto& region : shared_region.participant_regions) {
            if (!context_.hasRegion(region.participant_name, region.region_name)) {
                result.add(CouplingDiagnostic{
                    .severity = CouplingDiagnosticSeverity::Error,
                    .participant_name = region.participant_name,
                    .region_name = region.region_name,
                    .message = "shared region references a participant region missing from the context",
                });
            }
        }
    }

    return result;
}

CouplingContext CouplingContextBuilder::build() const
{
    const auto validation = validate();
    throwIfInvalid(validation);
    return context_;
}

} // namespace coupling
} // namespace FE
} // namespace svmp
