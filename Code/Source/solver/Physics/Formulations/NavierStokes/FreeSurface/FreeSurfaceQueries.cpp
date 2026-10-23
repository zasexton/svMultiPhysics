/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceQueries.h"

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceSemiImplicitSurfaceTension.h"
#include "Physics/Formulations/NavierStokes/NavierStokesBCFactories.h"

#include "FE/Assembly/Assembler.h"
#include "FE/Assembly/CutIntegrationContext.h"
#include "FE/Assembly/GlobalSystemView.h"
#include "FE/Constraints/ConstraintPlan.h"
#include "FE/Constraints/LevelSetActiveSideVertexDirichletConstraint.h"
#include "FE/Constraints/SmallCutAggregationConstraint.h"
#include "FE/Constraints/ContactPointVelocityConstraint.h"
#include "FE/Constraints/VertexDirichletConstraint.h"
#include "FE/Constitutive/MetadataTaggedModel.h"
#include "FE/Core/Logger.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Elements/ReferenceElement.h"
#include "FE/Forms/CutCellForms.h"
#include "FE/Forms/IntegrationDomain.h"
#include "FE/Forms/SymbolicDifferentiation.h"
#include "FE/Forms/Vocabulary.h"
#include "FE/Geometry/FrameGeometry.h"
#include "FE/Geometry/MappingFactory.h"
#include "FE/LevelSet/LevelSetCellEvaluator.h"
#include "FE/LevelSet/LevelSetGeometry.h"
#include "FE/Backends/Interfaces/GenericVector.h"
#include "FE/Systems/BoundaryConditionManager.h"
#include "FE/Systems/ALEBinding.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"
#include "Interfaces/GeneratedActiveBoundaryDomain.h"
#include "Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"
#include "Interfaces/LevelSetInterfaceDomain.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Fields/MeshFields.h"
#  include "Mesh/Mesh.h"
#endif

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <array>
#include <cmath>
#include <exception>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

namespace {

[[nodiscard]] const std::array<IncompressibleNavierStokesVMSOptions::ScalarValue, 3>&
contactLineWallNormal(const FreeSurfaceContactLine& contact_line)
{
    return std::visit(
        [](const auto& configuration)
            -> const std::array<
                IncompressibleNavierStokesVMSOptions::ScalarValue, 3>& {
            using Configuration = std::decay_t<decltype(configuration)>;
            if constexpr (std::is_same_v<
                              Configuration,
                              FreeSurfaceContactLine::PrescribedAngle> ||
                          std::is_same_v<
                              Configuration,
                              FreeSurfaceContactLine::DynamicRenE>) {
                return configuration.wall_normal;
            } else {
                throw std::logic_error(
                    "contactLineWallNormal requires an angle-bearing contact configuration");
            }
        },
        contact_line.configuration);
}

[[nodiscard]] std::optional<FE::Real> constantScalarValue(
    const IncompressibleNavierStokesVMSOptions::ScalarValue& value)
{
    const auto* real = std::get_if<FE::Real>(&value);
    if (real == nullptr) {
        return std::nullopt;
    }
    return *real;
}

[[nodiscard]] bool isHighOrderImplicitGeneratedInterface(
    const FreeSurfaceBoundary& bc)
{
    const auto token =
        normalizedFreeSurfaceOptionToken(bc.generated_interface_geometry);
    return token == "highorderimplicit" || token == "implicit" ||
           token == "saye" || token == "sayeimplicit";
}

[[nodiscard]] bool pressureStabilizationDisabledByPolicy(
    const FreeSurfaceBoundary& bc)
{
    return std::string_view(pressureStabilizationDisabledReason(bc)) != "none";
}

[[nodiscard]] FE::forms::CutVolumeSide oppositeCutVolumeSide(
    FE::forms::CutVolumeSide side) noexcept
{
    return side == FE::forms::CutVolumeSide::Negative
               ? FE::forms::CutVolumeSide::Positive
               : FE::forms::CutVolumeSide::Negative;
}

} // namespace

[[nodiscard]] const char* pressureActiveDomainName(
    FreeSurfaceActiveDomain domain) noexcept
{
    switch (domain) {
    case FreeSurfaceActiveDomain::None:
        return "None";
    case FreeSurfaceActiveDomain::LevelSetNegative:
        return "LevelSetNegative";
    case FreeSurfaceActiveDomain::LevelSetPositive:
        return "LevelSetPositive";
    }
    return "Unknown";
}

[[nodiscard]] bool isUnfittedLevelSet(const FreeSurfaceBoundary& bc) noexcept
{
    return bc.implementation == FreeSurfaceImplementation::UnfittedLevelSet;
}

[[nodiscard]] bool isExteriorOnePhaseBoundary(
    const FreeSurfaceBoundary& bc) noexcept
{
    return bc.role == FreeSurfaceBoundaryRole::ExteriorOnePhaseBoundary;
}

[[nodiscard]] bool isInternalMaterialInterfaceVolume(
    const FreeSurfaceBoundary& bc) noexcept
{
    return bc.role ==
        FreeSurfaceBoundaryRole::InternalMaterialInterfaceVolume;
}

[[nodiscard]] bool usesSurfaceStress(
    const FreeSurfaceBoundary& bc) noexcept
{
    switch (bc.surface_tension_form) {
    case FreeSurfaceSurfaceTensionForm::Automatic:
        // The generated-interface rule supplies both dI and its geometric
        // normal, so the unfitted default can be the variation of one
        // discrete polygonal surface energy.  Preserve the historical fitted
        // supplied-curvature contract unless SurfaceStress is requested.
        return isUnfittedLevelSet(bc);
    case FreeSurfaceSurfaceTensionForm::CurvatureTraction:
        return false;
    case FreeSurfaceSurfaceTensionForm::KinematicAreaGradientTraction:
        return false;
    case FreeSurfaceSurfaceTensionForm::SurfaceStress:
        return true;
    }
    return false;
}

[[nodiscard]] bool usesKinematicAreaGradientTraction(
    const FreeSurfaceBoundary& bc) noexcept
{
    return bc.surface_tension_form ==
           FreeSurfaceSurfaceTensionForm::KinematicAreaGradientTraction;
}

[[nodiscard]] bool usesGeneratedInterfaceNormal(
    const FreeSurfaceBoundary& bc) noexcept
{
    return usesSurfaceStress(bc) || usesKinematicAreaGradientTraction(bc);
}

[[nodiscard]] const char* surfaceTensionFormName(
    const FreeSurfaceBoundary& bc) noexcept
{
    switch (bc.surface_tension_form) {
    case FreeSurfaceSurfaceTensionForm::Automatic:
        return isUnfittedLevelSet(bc) ? "SurfaceStress"
                                      : "CurvatureTraction";
    case FreeSurfaceSurfaceTensionForm::CurvatureTraction:
        return "CurvatureTraction";
    case FreeSurfaceSurfaceTensionForm::KinematicAreaGradientTraction:
        return "KinematicAreaGradientTraction";
    case FreeSurfaceSurfaceTensionForm::SurfaceStress:
        return "SurfaceStress";
    }
    return "Unsupported";
}

[[nodiscard]] bool surfaceStressActive(
    const FreeSurfaceBoundary& bc) noexcept
{
    return usesSurfaceStress(bc) &&
           !FE::forms::bc::isZeroConstantScalarValue(bc.surface_tension);
}

[[nodiscard]] int freeSurfaceMarker(const FreeSurfaceBoundary& bc)
{
    return isUnfittedLevelSet(bc) ? bc.interface_marker : bc.boundary_marker;
}

[[nodiscard]] bool useFittedCurrentGeometry(const FreeSurfaceBoundary& bc,
                                            bool ale_enabled) noexcept
{
    return ale_enabled && !isUnfittedLevelSet(bc);
}

[[nodiscard]] std::string freeSurfaceValueName(std::string_view prefix,
                                               const FreeSurfaceBoundary& bc)
{
    const char* kind = isUnfittedLevelSet(bc) ? "_i" : "_b";
    return std::string(prefix) + kind + std::to_string(freeSurfaceMarker(bc));
}

[[nodiscard]] ContactLineKind contactLineKind(
    const FreeSurfaceContactLine& contact_line) noexcept
{
    if (std::holds_alternative<FreeSurfaceContactLine::Pinned>(
            contact_line.configuration)) {
        return ContactLineKind::Pinned;
    }
    if (std::holds_alternative<FreeSurfaceContactLine::PrescribedAngle>(
            contact_line.configuration)) {
        return ContactLineKind::PrescribedAngle;
    }
    if (std::holds_alternative<FreeSurfaceContactLine::DynamicRenE>(
            contact_line.configuration)) {
        return ContactLineKind::DynamicRenE;
    }
    return ContactLineKind::None;
}

// Input-facing name of a wall-owning contact law, used in validation messages
// shared by PrescribedContactAngle and DynamicContactAngle walls.
[[nodiscard]] std::string contactWallLawName(
    const FreeSurfaceContactLine& contact_line)
{
    return contactLineKind(contact_line) == ContactLineKind::PrescribedAngle
               ? "PrescribedContactAngle"
               : "DynamicContactAngle";
}

[[nodiscard]] int contactLineWallBoundaryMarker(
    const FreeSurfaceContactLine& contact_line) noexcept
{
    return std::visit(
        [](const auto& configuration) noexcept -> int {
            using Configuration = std::decay_t<decltype(configuration)>;
            if constexpr (std::is_same_v<Configuration,
                                         FreeSurfaceContactLine::None>) {
                return -1;
            } else {
                return configuration.wall_boundary_marker;
            }
        },
        contact_line.configuration);
}

[[nodiscard]] int contactLineMarker(
    const FreeSurfaceContactLine& contact_line) noexcept
{
    return std::visit(
        [](const auto& configuration) noexcept -> int {
            using Configuration = std::decay_t<decltype(configuration)>;
            if constexpr (std::is_same_v<Configuration,
                                         FreeSurfaceContactLine::None>) {
                return -1;
            } else {
                return configuration.contact_line_marker;
            }
        },
        contact_line.configuration);
}

[[nodiscard]] const IncompressibleNavierStokesVMSOptions::ScalarValue&
contactLineAngleRadians(const FreeSurfaceContactLine& contact_line)
{
    return std::visit(
        [](const auto& configuration)
            -> const IncompressibleNavierStokesVMSOptions::ScalarValue& {
            using Configuration = std::decay_t<decltype(configuration)>;
            if constexpr (std::is_same_v<
                              Configuration,
                              FreeSurfaceContactLine::PrescribedAngle>) {
                return configuration.contact_angle_radians;
            } else if constexpr (std::is_same_v<
                                     Configuration,
                                     FreeSurfaceContactLine::DynamicRenE>) {
                return configuration.equilibrium_contact_angle_radians;
            } else {
                throw std::logic_error(
                    "contactLineAngleRadians requires an angle-bearing contact configuration");
            }
        },
        contact_line.configuration);
}

[[nodiscard]] const IncompressibleNavierStokesVMSOptions::ScalarValue&
contactLineMobility(const FreeSurfaceContactLine& contact_line)
{
    const auto* configuration =
        std::get_if<FreeSurfaceContactLine::DynamicRenE>(
            &contact_line.configuration);
    if (configuration == nullptr) {
        throw std::logic_error("contactLineMobility requires DynamicRenE");
    }
    return configuration->mobility;
}

[[nodiscard]] const IncompressibleNavierStokesVMSOptions::ScalarValue&
contactLineSlipLength(const FreeSurfaceContactLine& contact_line)
{
    return std::visit(
        [](const auto& configuration)
            -> const IncompressibleNavierStokesVMSOptions::ScalarValue& {
            using Configuration = std::decay_t<decltype(configuration)>;
            if constexpr (std::is_same_v<
                              Configuration,
                              FreeSurfaceContactLine::PrescribedAngle>) {
                if (!configuration.slip_length.has_value()) {
                    throw std::logic_error(
                        "contactLineSlipLength requires configured Navier slip");
                }
                return *configuration.slip_length;
            } else if constexpr (std::is_same_v<
                                     Configuration,
                                     FreeSurfaceContactLine::DynamicRenE>) {
                return configuration.slip_length;
            } else {
                throw std::logic_error(
                    "contactLineSlipLength requires configured Navier slip");
            }
        },
        contact_line.configuration);
}

[[nodiscard]] bool contactLineHasNavierSlip(
    const FreeSurfaceContactLine& contact_line) noexcept
{
    return std::visit(
        [](const auto& configuration) noexcept {
            using Configuration = std::decay_t<decltype(configuration)>;
            if constexpr (std::is_same_v<
                              Configuration,
                              FreeSurfaceContactLine::PrescribedAngle>) {
                return configuration.slip_length.has_value();
            } else {
                return std::is_same_v<Configuration,
                                      FreeSurfaceContactLine::DynamicRenE>;
            }
        },
        contact_line.configuration);
}

[[nodiscard]] int contactLineConstraintMarker(
    const IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine& contact_line)
{
    return contactLineMarker(contact_line) >= 0
               ? contactLineMarker(contact_line)
               : contactLineWallBoundaryMarker(contact_line);
}

[[nodiscard]] FE::Real constantScalarValueOrThrow(
    const IncompressibleNavierStokesVMSOptions::ScalarValue& value,
    std::string_view context)
{
    const auto* real = std::get_if<FE::Real>(&value);
    if (real == nullptr) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " currently requires a literal scalar value");
    }
    return *real;
}

[[nodiscard]] bool hasWallAwareContactLaw(
    const FreeSurfaceBoundary& boundary)
{
    return std::any_of(
        boundary.contact_lines.begin(),
        boundary.contact_lines.end(),
        [](const auto& contact_line) {
            const auto kind = contactLineKind(contact_line);
            return kind == ContactLineKind::PrescribedAngle ||
                   kind == ContactLineKind::DynamicRenE;
        });
}

[[nodiscard]] std::array<FE::Real, 3> normalizedWallNormal(
    const IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine& contact_line)
{
    std::array<FE::Real, 3> normal{
        constantScalarValueOrThrow(contactLineWallNormal(contact_line)[0], "contact-line wall_normal"),
        constantScalarValueOrThrow(contactLineWallNormal(contact_line)[1], "contact-line wall_normal"),
        constantScalarValueOrThrow(contactLineWallNormal(contact_line)[2], "contact-line wall_normal")};
    if (!std::all_of(normal.begin(), normal.end(), [](FE::Real component) {
            return std::isfinite(component);
        })) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: prescribed contact angle requires finite wall_normal components");
    }
    const auto norm = std::sqrt(normal[0] * normal[0] +
                                normal[1] * normal[1] +
                                normal[2] * normal[2]);
    if (!(norm > FE::Real{0.0}) || !std::isfinite(norm)) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: prescribed contact angle requires a finite, nonzero wall_normal");
    }
    for (auto& component : normal) {
        component /= norm;
    }
    return normal;
}

void validateWallNormalDimension(
    const IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine& contact_line,
    int dim,
    std::string_view context)
{
    const auto normal = normalizedWallNormal(contact_line);
    constexpr FE::Real tolerance = FE::Real{1.0e-12};
    for (int component = dim; component < 3; ++component) {
        if (std::abs(normal[static_cast<std::size_t>(component)]) > tolerance) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: " +
                std::string(context) +
                " wall_normal must lie in the velocity space dimension");
        }
    }
}

void validateFiniteConstantScalar(
    const IncompressibleNavierStokesVMSOptions::ScalarValue& value,
    std::string_view context)
{
    const auto literal = constantScalarValue(value);
    if (literal.has_value() && !std::isfinite(*literal)) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " must be finite");
    }
}

void validateNonnegativeConstantScalar(
    const IncompressibleNavierStokesVMSOptions::ScalarValue& value,
    std::string_view context)
{
    validateFiniteConstantScalar(value, context);
    const auto literal = constantScalarValue(value);
    if (literal.has_value() && *literal < FE::Real{0.0}) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " must be nonnegative");
    }
}

void validatePositiveConstantScalar(
    const IncompressibleNavierStokesVMSOptions::ScalarValue& value,
    std::string_view context)
{
    validateFiniteConstantScalar(value, context);
    const auto literal = constantScalarValue(value);
    if (literal.has_value() && !(*literal > FE::Real{0.0})) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " must be positive");
    }
}

[[nodiscard]] FE::forms::FormExpr wallNormalExpression(
    const IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine& contact_line,
    int dim)
{
    const auto wall_normal = normalizedWallNormal(contact_line);
    std::vector<FE::forms::FormExpr> wall_components;
    wall_components.reserve(static_cast<std::size_t>(dim));
    for (int d = 0; d < dim; ++d) {
        wall_components.push_back(FE::forms::FormExpr::constant(
            wall_normal[static_cast<std::size_t>(d)]));
    }
    return FE::forms::FormExpr::asVector(std::move(wall_components));
}

[[nodiscard]] std::string normalizedFreeSurfaceOptionToken(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    value.erase(std::remove_if(value.begin(), value.end(),
                               [](unsigned char c) {
                                   return c == '_' || c == '-' || std::isspace(c);
                               }),
                value.end());
    return value;
}

[[nodiscard]] bool isRefreshedFrozenGeometryTangent(
    const FreeSurfaceBoundary& bc)
{
    const auto token =
        normalizedFreeSurfaceOptionToken(bc.geometry_tangent_policy);
    return token.empty() || token == "refreshedfrozenquadrature" ||
           token == "refreshedfrozen" || token == "frozenquadrature";
}

[[nodiscard]] const char* pressureStabilizationPolicyName(
    FreeSurfacePressureStabilizationPolicy policy) noexcept
{
    switch (policy) {
    case FreeSurfacePressureStabilizationPolicy::Enabled:
        return "Enabled";
    case FreeSurfacePressureStabilizationPolicy::Incremental:
        return "Incremental";
    case FreeSurfacePressureStabilizationPolicy::Disabled:
        return "Disabled";
    case FreeSurfacePressureStabilizationPolicy::DisabledForRefreshedFrozenHighOrder:
        return "DisabledForRefreshedFrozenHighOrder";
    }
    return "Unknown";
}

[[nodiscard]] const char* pressureStabilizationDisabledReason(
    const FreeSurfaceBoundary& bc)
{
    switch (bc.cut_cell_stabilization.pressure_policy) {
    case FreeSurfacePressureStabilizationPolicy::Enabled:
    case FreeSurfacePressureStabilizationPolicy::Incremental:
        return "none";
    case FreeSurfacePressureStabilizationPolicy::Disabled:
        return "explicit_policy_disabled";
    case FreeSurfacePressureStabilizationPolicy::DisabledForRefreshedFrozenHighOrder:
        break;
    }
    return isHighOrderImplicitGeneratedInterface(bc) &&
                   isRefreshedFrozenGeometryTangent(bc)
               ? "refreshed_frozen_high_order_policy"
               : "none";
}

[[nodiscard]] bool pressureStabilizationActive(const FreeSurfaceBoundary& bc)
{
    return !pressureStabilizationDisabledByPolicy(bc) &&
           !FE::forms::bc::isZeroConstantScalarValue(
               bc.cut_cell_stabilization.pressure_gradient_penalty);
}

[[nodiscard]] std::string freeSurfaceCurvaturePolicyName(
    const FreeSurfaceBoundary& bc)
{
    if (FE::forms::bc::isZeroConstantScalarValue(bc.surface_tension)) {
        return "inactive_zero_surface_tension";
    }
    if (usesSurfaceStress(bc)) {
        return "variational_surface_stress_generated_geometry";
    }
    if (usesKinematicAreaGradientTraction(bc)) {
        return "kinematic_area_gradient_total_surface_young_energy_field";
    }
    if (isUnfittedLevelSet(bc)) {
        if (!bc.curvature_field_name.empty()) {
            return "curvature_field_level_set_normal_signed";
        }
        if (bc.use_level_set_curvature) {
            return "raw_level_set_curvature_guarded";
        }
        return "supplied_scalar_level_set_normal_signed";
    }
    if (!bc.curvature_field_name.empty()) {
        return "curvature_field";
    }
    return "supplied_scalar";
}

[[nodiscard]] const char* activeDomainName(FreeSurfaceActiveDomain domain) noexcept
{
    switch (domain) {
    case FreeSurfaceActiveDomain::None:
        return "None";
    case FreeSurfaceActiveDomain::LevelSetNegative:
        return "LevelSetNegative";
    case FreeSurfaceActiveDomain::LevelSetPositive:
        return "LevelSetPositive";
    }
    return "Unknown";
}

[[nodiscard]] const char* activeDomainMethodName(
    FreeSurfaceActiveDomainMethod method) noexcept
{
    switch (method) {
    case FreeSurfaceActiveDomainMethod::CutVolume:
        return "CutVolume";
    }
    return "Unknown";
}

[[nodiscard]] const char* kinematicEnforcementName(
    FreeSurfaceKinematicEnforcement enforcement) noexcept
{
    switch (enforcement) {
    case FreeSurfaceKinematicEnforcement::None:
        return "None";
    case FreeSurfaceKinematicEnforcement::MeshNitsche:
        return "MeshNitsche";
    }
    return "Unknown";
}

[[nodiscard]] const char* fieldSourceKindName(
    FE::systems::FieldSourceKind source_kind) noexcept
{
    switch (source_kind) {
    case FE::systems::FieldSourceKind::Unknown:
        return "Unknown";
    case FE::systems::FieldSourceKind::PrescribedData:
        return "PrescribedData";
    case FE::systems::FieldSourceKind::DerivedFromUnknown:
        return "DerivedFromUnknown";
    }
    return "Unknown";
}

[[nodiscard]] const char* cutVolumeSideName(FE::forms::CutVolumeSide side) noexcept
{
    return side == FE::forms::CutVolumeSide::Negative ? "Negative" : "Positive";
}

[[nodiscard]] FE::forms::CutVolumeSide activeDomainSide(
    FreeSurfaceActiveDomain domain) noexcept
{
    switch (domain) {
    case FreeSurfaceActiveDomain::None:
    case FreeSurfaceActiveDomain::LevelSetNegative:
        return FE::forms::CutVolumeSide::Negative;
    case FreeSurfaceActiveDomain::LevelSetPositive:
        return FE::forms::CutVolumeSide::Positive;
    }
    return FE::forms::CutVolumeSide::Negative;
}

[[nodiscard]] FE::systems::MeshTangentialBoundaryPolicy
systemTangentialMeshPolicy(FreeSurfaceTangentialMeshPolicy policy)
{
    switch (policy) {
    case FreeSurfaceTangentialMeshPolicy::Free:
        return FE::systems::MeshTangentialBoundaryPolicy::Free;
    case FreeSurfaceTangentialMeshPolicy::SmoothingOnly:
        return FE::systems::MeshTangentialBoundaryPolicy::SmoothingOnly;
    case FreeSurfaceTangentialMeshPolicy::Prescribed:
        return FE::systems::MeshTangentialBoundaryPolicy::Prescribed;
    }
    throw std::invalid_argument(
        "IncompressibleNavierStokesVMSModule: unknown fitted "
        "free-surface tangential mesh policy");
}

[[nodiscard]] std::string fittedTangentialOperatorSource(
    int boundary_marker,
    FreeSurfaceTangentialMeshPolicy policy)
{
    switch (policy) {
    case FreeSurfaceTangentialMeshPolicy::Free:
        return "Fitted free-surface natural tangential state on marker " +
               std::to_string(boundary_marker);
    case FreeSurfaceTangentialMeshPolicy::SmoothingOnly:
        return "Fitted free-surface tangential surface smoothing on marker " +
               std::to_string(boundary_marker);
    case FreeSurfaceTangentialMeshPolicy::Prescribed:
        return "Fitted free-surface prescribed tangential mesh velocity on "
               "marker " +
               std::to_string(boundary_marker);
    }
    throw std::invalid_argument(
        "IncompressibleNavierStokesVMSModule: unknown fitted tangential "
        "mesh policy source");
}

[[nodiscard]] std::string fittedFluidNormalDescriptorSource(
    const FreeSurfaceBoundary& bc)
{
    // MeshNitsche, the only normal relation, adds no fluid kinematic row.
    return "Fitted free-surface fluid natural dynamic condition on marker " +
           std::to_string(bc.boundary_marker);
}

[[nodiscard]] const char* freeSurfaceImplementationName(
    FreeSurfaceImplementation implementation) noexcept
{
    return implementation == FreeSurfaceImplementation::FittedALE
               ? "FittedALE"
               : "UnfittedLevelSet";
}

[[nodiscard]] const char* freeSurfaceBoundaryRoleName(
    FreeSurfaceBoundaryRole role) noexcept
{
    switch (role) {
    case FreeSurfaceBoundaryRole::ExteriorOnePhaseBoundary:
        return "ExteriorOnePhaseBoundary";
    case FreeSurfaceBoundaryRole::InternalMaterialInterfaceVolume:
        return "InternalMaterialInterfaceVolume";
    }
    return "Unknown";
}

[[nodiscard]] const char* freeSurfacePhysicalModelName(
    FreeSurfacePhysicalModel model) noexcept
{
    switch (model) {
    case FreeSurfacePhysicalModel::
        OnePhaseLiquidPrescribedExteriorPressure:
        return "one_phase_liquid_prescribed_exterior_pressure";
    }
    return nullptr;
}

[[nodiscard]] const char* normalKinematicPolicyName(
    FreeSurfaceNormalKinematicPolicy policy) noexcept
{
    switch (policy) {
    case FreeSurfaceNormalKinematicPolicy::None:
        return "None";
    case FreeSurfaceNormalKinematicPolicy::MatchFluidNormalVelocity:
        return "MatchFluidNormalVelocity";
    }
    return "Unknown";
}

[[nodiscard]] const char* tangentialMeshPolicyName(
    FreeSurfaceTangentialMeshPolicy policy) noexcept
{
    switch (policy) {
    case FreeSurfaceTangentialMeshPolicy::Free:
        return "Free";
    case FreeSurfaceTangentialMeshPolicy::SmoothingOnly:
        return "SmoothingOnly";
    case FreeSurfaceTangentialMeshPolicy::Prescribed:
        return "Prescribed";
    }
    return "Unknown";
}

[[nodiscard]] const char* contactLineKindName(ContactLineKind kind) noexcept
{
    switch (kind) {
    case ContactLineKind::None:
        return "None";
    case ContactLineKind::Pinned:
        return "Pinned";
    case ContactLineKind::PrescribedAngle:
        return "PrescribedAngle";
    case ContactLineKind::DynamicRenE:
        return "DynamicRenE";
    }
    return "Unknown";
}

[[nodiscard]] const char* geometryTangentPathName(
    FE::forms::GeometryTangentPath path) noexcept
{
    switch (path) {
    case FE::forms::GeometryTangentPath::Auto:
        return "Auto";
    case FE::forms::GeometryTangentPath::ADReference:
        return "ADReference";
    case FE::forms::GeometryTangentPath::SymbolicRequired:
        return "SymbolicRequired";
    case FE::forms::GeometryTangentPath::SymbolicWithADCheck:
        return "SymbolicWithADCheck";
    }
    return "Unknown";
}

[[nodiscard]] std::string activePhaseSignName(
    FreeSurfaceActiveDomain domain)
{
    switch (domain) {
    case FreeSurfaceActiveDomain::None:
        return "full_domain";
    case FreeSurfaceActiveDomain::LevelSetNegative:
        return "negative";
    case FreeSurfaceActiveDomain::LevelSetPositive:
        return "positive";
    }
    return "unknown";
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
