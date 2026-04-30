/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/CouplingGeometryRequirements.h"

#include <algorithm>

namespace svmp {
namespace FE {
namespace coupling {

namespace {

template <typename T>
bool contains(const std::vector<T>& values, T value)
{
    return std::find(values.begin(), values.end(), value) != values.end();
}

void appendUnique(std::vector<CouplingGeometryTerminalQuantity>& values,
                  CouplingGeometryTerminalQuantity value)
{
    if (!contains(values, value)) {
        values.push_back(value);
    }
}

void appendUnique(std::vector<analysis::DomainKind>& values,
                  analysis::DomainKind value)
{
    if (!contains(values, value)) {
        values.push_back(value);
    }
}

std::optional<analysis::DomainKind> resolveRequirementDomain(
    const CouplingContext& context,
    const CouplingGeometryTerminalRequirement& requirement,
    CouplingValidationResult* result)
{
    if (requirement.scope.location.has_value()) {
        return toAnalysisDomainKind(requirement.scope.location->region_kind);
    }

    if (requirement.scope.region.has_value()) {
        const auto& region = *requirement.scope.region;
        if (region.shared_region_name.has_value()) {
            if (region.participant_name.empty()) {
                if (result != nullptr &&
                    requirement.requirement == CouplingRequirement::Required) {
                    result->addError("shared-region geometry terminal requirements need a participant owner");
                }
                return std::nullopt;
            }
            if (!context.hasSharedRegion(*region.shared_region_name)) {
                if (result != nullptr &&
                    requirement.requirement == CouplingRequirement::Required) {
                    result->addError("geometry terminal shared region is missing from the context");
                }
                return std::nullopt;
            }
            const auto resolved =
                context.sharedRegion(*region.shared_region_name, region.participant_name);
            return toAnalysisDomainKind(resolved.kind);
        }

        if (!context.hasRegion(region.participant_name, region.region_name)) {
            if (result != nullptr &&
                requirement.requirement == CouplingRequirement::Required) {
                result->addError("geometry terminal region is missing from the context");
            }
            return std::nullopt;
        }
        return toAnalysisDomainKind(
            context.region(region.participant_name, region.region_name).kind);
    }

    return analysis::DomainKind::Cell;
}

void updateConfigurationSummary(CouplingGeometryTerminalRequirementSummary& summary,
                                const CouplingGeometryTerminalRequirement& requirement)
{
    const auto configuration =
        requirement.scope.location.has_value()
            ? requirement.scope.location->coordinate_configuration
            : forms::GeometryConfiguration::Reference;
    if (configuration == forms::GeometryConfiguration::Current) {
        summary.requires_current_configuration = true;
    } else {
        summary.requires_reference_configuration = true;
    }
}

void validateOwnerScope(const CouplingGeometryTerminalRequirement& requirement,
                        CouplingValidationResult& result)
{
    const bool required = requirement.requirement == CouplingRequirement::Required;
    const bool has_owner =
        requirement.scope.participant_name.has_value() ||
        requirement.scope.region.has_value();
    if (required && !has_owner) {
        result.addError("geometry terminal requirement requires owner scope");
    }

    if (requirement.scope.participant_name.has_value() &&
        requirement.scope.region.has_value() &&
        !requirement.scope.region->participant_name.empty() &&
        *requirement.scope.participant_name !=
            requirement.scope.region->participant_name) {
        result.addError("geometry terminal owner participant conflicts with region participant");
    }
}

} // namespace

const char* toString(CouplingGeometryTerminalQuantity quantity) noexcept
{
    switch (quantity) {
    case CouplingGeometryTerminalQuantity::MeshDisplacement:
        return "mesh_displacement";
    case CouplingGeometryTerminalQuantity::Coordinate:
        return "coordinate";
    case CouplingGeometryTerminalQuantity::ReferenceCoordinate:
        return "reference_coordinate";
    case CouplingGeometryTerminalQuantity::CurrentCoordinate:
        return "current_coordinate";
    case CouplingGeometryTerminalQuantity::PreviousCoordinate:
        return "previous_coordinate";
    case CouplingGeometryTerminalQuantity::ReferencePhysicalCoordinate:
        return "reference_physical_coordinate";
    case CouplingGeometryTerminalQuantity::Jacobian:
        return "jacobian";
    case CouplingGeometryTerminalQuantity::JacobianInverse:
        return "jacobian_inverse";
    case CouplingGeometryTerminalQuantity::JacobianDeterminant:
        return "jacobian_determinant";
    case CouplingGeometryTerminalQuantity::CurrentJacobian:
        return "current_jacobian";
    case CouplingGeometryTerminalQuantity::ReferenceJacobian:
        return "reference_jacobian";
    case CouplingGeometryTerminalQuantity::CurrentJacobianDeterminant:
        return "current_jacobian_determinant";
    case CouplingGeometryTerminalQuantity::ReferenceJacobianDeterminant:
        return "reference_jacobian_determinant";
    case CouplingGeometryTerminalQuantity::Normal:
        return "normal";
    case CouplingGeometryTerminalQuantity::CurrentNormal:
        return "current_normal";
    case CouplingGeometryTerminalQuantity::ReferenceNormal:
        return "reference_normal";
    case CouplingGeometryTerminalQuantity::CurrentMeasure:
        return "current_measure";
    case CouplingGeometryTerminalQuantity::ReferenceMeasure:
        return "reference_measure";
    case CouplingGeometryTerminalQuantity::SurfaceJacobian:
        return "surface_jacobian";
    case CouplingGeometryTerminalQuantity::CellDiameter:
        return "cell_diameter";
    case CouplingGeometryTerminalQuantity::CellVolume:
        return "cell_volume";
    case CouplingGeometryTerminalQuantity::FacetArea:
        return "facet_area";
    case CouplingGeometryTerminalQuantity::CellDomainId:
        return "cell_domain_id";
    }
    return "unknown";
}

std::optional<analysis::DomainKind> toAnalysisDomainKind(CouplingRegionKind kind) noexcept
{
    switch (kind) {
    case CouplingRegionKind::Domain:
        return analysis::DomainKind::Cell;
    case CouplingRegionKind::Boundary:
        return analysis::DomainKind::Boundary;
    case CouplingRegionKind::InteriorFace:
        return analysis::DomainKind::InteriorFace;
    case CouplingRegionKind::InterfaceFace:
        return analysis::DomainKind::InterfaceFace;
    case CouplingRegionKind::UserDefined:
        return std::nullopt;
    }
    return std::nullopt;
}

CouplingGeometryTerminalRequirementSummary summarizeGeometryTerminalRequirements(
    const CouplingContext& context,
    std::span<const CouplingGeometryTerminalRequirement> requirements)
{
    CouplingGeometryTerminalRequirementSummary summary;
    for (const auto& requirement : requirements) {
        appendUnique(summary.quantities, requirement.quantity);
        if (const auto domain = resolveRequirementDomain(context, requirement, nullptr)) {
            appendUnique(summary.domains, *domain);
        }
        updateConfigurationSummary(summary, requirement);
    }
    return summary;
}

CouplingValidationResult validateGeometryTerminalRequirements(
    const CouplingContext& context,
    std::span<const CouplingGeometryTerminalRequirement> requirements,
    const CouplingGeometryTerminalAvailability& availability)
{
    CouplingValidationResult result;
    for (const auto& requirement : requirements) {
        const bool required = requirement.requirement == CouplingRequirement::Required;
        validateOwnerScope(requirement, result);

        const auto domain = resolveRequirementDomain(context, requirement, &result);
        if (!domain.has_value()) {
            if (required) {
                result.addError("geometry terminal requirement uses an unsupported region kind");
            }
            continue;
        }

        if (required &&
            !contains(availability.supported_quantities, requirement.quantity)) {
            result.addError("geometry terminal quantity is not available from installed metadata");
        }
        if (required && !contains(availability.supported_domains, *domain)) {
            result.addError("geometry terminal domain is not available from installed metadata");
        }

        const auto configuration =
            requirement.scope.location.has_value()
                ? requirement.scope.location->coordinate_configuration
                : forms::GeometryConfiguration::Reference;
        if (required && configuration == forms::GeometryConfiguration::Current &&
            !availability.supports_current_configuration) {
            result.addError("current-configuration geometry terminal is unavailable");
        }
        if (required && configuration == forms::GeometryConfiguration::Reference &&
            !availability.supports_reference_configuration) {
            result.addError("reference-configuration geometry terminal is unavailable");
        }
    }
    return result;
}

} // namespace coupling
} // namespace FE
} // namespace svmp
