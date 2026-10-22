/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceFittedALE.h"

#include "Physics/Formulations/NavierStokes/FreeSurface/FittedALEStageRecords.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceGeometry.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceQueries.h"

#include "FE/Constraints/ConstraintPlan.h"
#include "FE/Core/Logger.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Forms/CutCellForms.h"
#include "FE/Forms/Vocabulary.h"
#include "FE/Geometry/FrameGeometry.h"
#include "FE/Systems/ALEBinding.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Fields/MeshFields.h"
#  include "Mesh/Mesh.h"
#endif

#include <algorithm>
#include <cstddef>
#include <array>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

void applyFreeSurfaceContactLineConstraints(
    FE::constraints::ConstraintPlan& constraint_plan,
    const FreeSurfaceBoundary& bc,
    const FE::systems::ALEBinding& ale_binding,
    int dim)
{
    for (const auto& contact_line : bc.contact_lines) {
        switch (contactLineKind(contact_line)) {
        case ContactLineKind::None:
        case ContactLineKind::PrescribedAngle:
        case ContactLineKind::DynamicRenE:
            continue;
        case ContactLineKind::Pinned:
            break;
        }

        if (ale_binding.mesh_displacement_field == FE::INVALID_FIELD_ID) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: pinned fitted contact lines require a coupled mesh displacement unknown");
        }

        const int marker = contactLineConstraintMarker(contact_line);
        std::vector<FE::forms::bc::StrongDirichlet> constraints;
        constraints.reserve(static_cast<std::size_t>(dim));
        for (int component = 0; component < dim; ++component) {
            constraints.push_back(FE::forms::bc::StrongDirichlet{
                .field = ale_binding.mesh_displacement_field,
                .boundary_marker = marker,
                .component = component,
                .value = FE::forms::FormExpr::constant(0.0),
                .symbol = "mesh_displacement",
            });
        }
        constraint_plan.essentials([&constraints](FE::systems::FESystem& system) {
            FE::systems::installStrongDirichlet(system, constraints);
        });
    }
}

std::vector<FE::systems::MeshNormalBoundaryConstraintDeclaration>
fittedNormalKinematicDeclarations(
    const FE::systems::ALEBinding& ale_binding,
    const std::vector<FreeSurfaceBoundary>& effective_free_surfaces,
    const FE::forms::FormExpr& u,
    FE::FieldId u_id)
{
    using namespace svmp::FE::forms;
    std::vector<FE::systems::MeshNormalBoundaryConstraintDeclaration>
        normal_declarations;
    if (ale_binding.coupled()) {
        normal_declarations.reserve(effective_free_surfaces.size());
        for (const auto& boundary : effective_free_surfaces) {
            const bool installs_normal_relation =
                boundary.implementation ==
                    FreeSurfaceImplementation::FittedALE &&
                boundary.kinematic_enforcement !=
                    FreeSurfaceKinematicEnforcement::None &&
                boundary.normal_kinematic_policy ==
                    FreeSurfaceNormalKinematicPolicy::
                        MatchFluidNormalVelocity;
            if (!installs_normal_relation) {
                continue;
            }
            normal_declarations.push_back(
                FE::systems::MeshNormalBoundaryConstraintDeclaration{
                    .mesh_displacement_field =
                        ale_binding.mesh_displacement_field,
                    .boundary_marker = boundary.boundary_marker,
                    .quantity = FE::systems::MeshNormalBoundaryQuantity::
                        MeshVelocityTrace,
                    .target_kind =
                        FE::systems::MeshNormalBoundaryTargetKind::
                            FluidNormalVelocity,
                    .target_expression =
                        normalTrace(u, currentNormal()),
                    .enforcement_kind =
                        FE::analysis::EnforcementKind::WeakPenalty,
                    .related_velocity_field = u_id,
                    .owner_component =
                        "IncompressibleNavierStokesVMSModule."
                        "FreeSurfaceBoundary",
                    .requires_mesh_flux_consistency =
                        boundary.kinematic_enforcement ==
                        FreeSurfaceKinematicEnforcement::MeshNitsche,
                    .mesh_flux_consistency_nitsche_gamma =
                        boundary.kinematic_enforcement ==
                                FreeSurfaceKinematicEnforcement::MeshNitsche
                            ? boundary.kinematic_nitsche_gamma
                            : FE::Real{0.0},
                });
        }
    }
    return normal_declarations;
}

void declareFittedNormalKinematics(
    FE::systems::FESystem& system,
    const std::string& operator_tag,
    std::vector<FE::systems::MeshNormalBoundaryConstraintDeclaration>
        normal_declarations,
    const std::vector<FreeSurfaceBoundary>& effective_free_surfaces)
{
    for (auto& declaration : normal_declarations) {
        const auto marker = declaration.boundary_marker;
        const auto displacement_field =
            declaration.mesh_displacement_field;
        const auto fitted_boundary = std::find_if(
            effective_free_surfaces.begin(),
            effective_free_surfaces.end(),
            [&](const auto& candidate) {
                return candidate.implementation ==
                           FreeSurfaceImplementation::FittedALE &&
                       candidate.boundary_marker == marker;
            });
        if (fitted_boundary == effective_free_surfaces.end()) {
            throw std::logic_error(
                "IncompressibleNavierStokesVMSModule: fitted normal "
                "measurement lost its owning free-surface boundary");
        }
        std::optional<std::array<FE::Real, 3>>
            prescribed_tangential_mesh_velocity;
        if (fitted_boundary->tangential_mesh_policy ==
            FreeSurfaceTangentialMeshPolicy::Prescribed) {
            std::array<FE::Real, 3> target{};
            for (std::size_t component = 0u;
                 component < target.size();
                 ++component) {
                target[component] = constantScalarValueOrThrow(
                    fitted_boundary
                        ->prescribed_tangential_mesh_velocity[component],
                    "fitted prescribed tangential mesh velocity");
            }
            prescribed_tangential_mesh_velocity = target;
        }
        system.declareMeshNormalBoundaryConstraint(
            std::move(declaration));
        system.bindMeshNormalBoundaryConstraintConsumer(
            displacement_field,
            marker,
            operator_tag,
            "Fitted free-surface mesh normal kinematic row on marker " +
                std::to_string(marker),
            fittedFluidNormalDescriptorSource(*fitted_boundary));
        free_surface::fittedALERecords(system)
            .registerFittedALENormalOperatorStageMeasurement(
                displacement_field,
                marker,
                prescribed_tangential_mesh_velocity);
    }
}

const FE::systems::FieldRecord& fittedMeshDisplacementRecord(
    const FE::systems::FESystem& system,
    const FE::systems::ALEBinding& ale_binding)
{
    if (ale_binding.mesh_displacement_field == FE::INVALID_FIELD_ID) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: fitted free-surface mesh kinematics require a coupled mesh displacement unknown");
    }

    const auto& rec = system.fieldRecord(ale_binding.mesh_displacement_field);
    if (!rec.space) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: fitted free-surface mesh displacement field has no function space");
    }
    return rec;
}

void declareFittedTangentialMeshPolicy(
    FE::systems::FESystem& system,
    const FE::systems::ALEBinding& ale_binding,
    const FreeSurfaceBoundary& bc)
{
    const bool install_tangential_smoothing =
        bc.tangential_mesh_policy ==
        FreeSurfaceTangentialMeshPolicy::SmoothingOnly;
    const bool install_tangential_prescription =
        bc.tangential_mesh_policy ==
        FreeSurfaceTangentialMeshPolicy::Prescribed;

    system.declareMeshTangentialBoundaryPolicy(
        FE::systems::MeshTangentialBoundaryPolicyDeclaration{
            .mesh_displacement_field = ale_binding.mesh_displacement_field,
            .boundary_marker = bc.boundary_marker,
            .policy =
                systemTangentialMeshPolicy(bc.tangential_mesh_policy),
            .owner_component =
                "IncompressibleNavierStokesVMSModule.FreeSurfaceBoundary",
        });

    FE_LOG_INFO(
        std::string("IncompressibleNavierStokesVMSModule: fitted "
                    "free-surface mesh policy") +
        " marker=" + std::to_string(bc.boundary_marker) +
        " tangential_policy=" +
        tangentialMeshPolicyName(bc.tangential_mesh_policy) +
        " tangential_owner=free_surface_boundary" +
        " tangential_enforcement=" +
        (install_tangential_prescription
             ? "weak_velocity_penalty"
             : install_tangential_smoothing
                   ? "surface_smoothing_functional"
                   : "natural_no_tangential_constraint") +
        " diagnostic=fitted_free_surface_mesh_policy");
}

void declareFittedMeshRowDescriptors(
    FE::systems::FESystem& system,
    const std::string& operator_tag,
    const FE::systems::ALEBinding& ale_binding,
    const FreeSurfaceBoundary& bc,
    FE::FieldId velocity_field,
    bool rows_installed)
{
    const bool install_normal_relation =
        bc.kinematic_enforcement != FreeSurfaceKinematicEnforcement::None &&
        bc.normal_kinematic_policy ==
            FreeSurfaceNormalKinematicPolicy::MatchFluidNormalVelocity;
    const bool install_tangential_prescription =
        bc.tangential_mesh_policy ==
        FreeSurfaceTangentialMeshPolicy::Prescribed;

    if (!rows_installed) {
        FE::analysis::BoundaryConditionDescriptor descriptor;
        descriptor.primary_variable =
            FE::analysis::VariableKey::field(
                ale_binding.mesh_displacement_field);
        descriptor.boundary_marker = bc.boundary_marker;
        descriptor.trace_kind =
            FE::analysis::TraceKind::TangentialComponent;
        descriptor.enforcement_kind =
            FE::analysis::EnforcementKind::WeakConsistent;
        const auto consumer_source = fittedTangentialOperatorSource(
            bc.boundary_marker, bc.tangential_mesh_policy);
        descriptor.source = consumer_source;
        system.addBoundaryConditionDescriptor(
            std::move(descriptor), operator_tag);
        system.bindMeshTangentialBoundaryPolicyConsumer(
            ale_binding.mesh_displacement_field,
            bc.boundary_marker,
            systemTangentialMeshPolicy(bc.tangential_mesh_policy),
            operator_tag,
            consumer_source);
        return;
    }

    if (install_normal_relation) {
        if (velocity_field == FE::INVALID_FIELD_ID) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: fitted normal "
                "kinematics require a related fluid velocity field");
        }
        const auto mesh_source =
            "Fitted free-surface mesh normal kinematic row on marker " +
            std::to_string(bc.boundary_marker);
        FE::analysis::BoundaryConditionDescriptor mesh_descriptor;
        mesh_descriptor.primary_variable =
            FE::analysis::VariableKey::field(
                ale_binding.mesh_displacement_field);
        mesh_descriptor.related_variables.push_back(
            FE::analysis::VariableKey::field(velocity_field));
        mesh_descriptor.boundary_marker = bc.boundary_marker;
        mesh_descriptor.trace_kind =
            FE::analysis::TraceKind::NormalComponent;
        mesh_descriptor.enforcement_kind =
            FE::analysis::EnforcementKind::WeakPenalty;
        mesh_descriptor.source = mesh_source;
        system.addBoundaryConditionDescriptor(
            std::move(mesh_descriptor), operator_tag);

        const auto fluid_source =
            fittedFluidNormalDescriptorSource(bc);
        FE::analysis::BoundaryConditionDescriptor fluid_descriptor;
        fluid_descriptor.primary_variable =
            FE::analysis::VariableKey::field(velocity_field);
        fluid_descriptor.related_variables.push_back(
            FE::analysis::VariableKey::field(
                ale_binding.mesh_displacement_field));
        fluid_descriptor.boundary_marker = bc.boundary_marker;
        fluid_descriptor.trace_kind =
            FE::analysis::TraceKind::NormalComponent;
        fluid_descriptor.enforcement_kind =
            bc.kinematic_enforcement ==
                    FreeSurfaceKinematicEnforcement::Nitsche
                ? FE::analysis::EnforcementKind::WeakNitsche
                : bc.kinematic_enforcement ==
                          FreeSurfaceKinematicEnforcement::MeshNitsche
                      ? FE::analysis::EnforcementKind::WeakConsistent
                      : FE::analysis::EnforcementKind::WeakPenalty;
        fluid_descriptor.source = fluid_source;
        system.addBoundaryConditionDescriptor(
            std::move(fluid_descriptor), operator_tag);
    }

    FE::analysis::BoundaryConditionDescriptor descriptor;
    descriptor.primary_variable =
        FE::analysis::VariableKey::field(
            ale_binding.mesh_displacement_field);
    descriptor.boundary_marker = bc.boundary_marker;
    descriptor.trace_kind =
        FE::analysis::TraceKind::TangentialComponent;
    descriptor.enforcement_kind =
        install_tangential_prescription
            ? FE::analysis::EnforcementKind::WeakPenalty
            : FE::analysis::EnforcementKind::WeakConsistent;
    const auto consumer_source = fittedTangentialOperatorSource(
        bc.boundary_marker, bc.tangential_mesh_policy);
    descriptor.source = consumer_source;
    system.addBoundaryConditionDescriptor(
        std::move(descriptor), operator_tag);
    system.bindMeshTangentialBoundaryPolicyConsumer(
        ale_binding.mesh_displacement_field,
        bc.boundary_marker,
        systemTangentialMeshPolicy(bc.tangential_mesh_policy),
        operator_tag,
        consumer_source);
}

FE::forms::FormExpr tangentialMeshPolicyResidual(
    const FreeSurfaceBoundary& b,
    const FE::forms::FormExpr& d_mesh,
    const FE::forms::FormExpr& psi,
    const FE::forms::FormExpr& n,
    int components,
    const FE::forms::BoundaryDomain& gamma)
{
    using namespace FE::forms;
    if (b.tangential_mesh_policy == FreeSurfaceTangentialMeshPolicy::SmoothingOnly) {
        const auto P = tangentialProjector(n, components);
        const auto weight =
            freeSurfaceScalar(b.tangential_mesh_penalty, "ns_free_surface_tangential_mesh_smoothing_weight", b);
        return (weight * inner(P * grad(d_mesh) * P, P * grad(psi) * P)).ds(gamma);
    }
    if (b.tangential_mesh_policy == FreeSurfaceTangentialMeshPolicy::Prescribed) {
        const auto target = FormExpr::asVector(bc::toVectorExpr(
            b.prescribed_tangential_mesh_velocity, components, "ns_free_surface_tangential_mesh_velocity",
            b.boundary_marker, bc::ComponentValueNameStyle::Component));
        const auto gap = dt(d_mesh) - target;
        const auto penalty =
            freeSurfaceScalar(b.tangential_mesh_penalty, "ns_free_surface_tangential_mesh_penalty", b);
        return (penalty * (inner(gap, psi) - normalTrace(gap, n) * normalTrace(psi, n))).ds(gamma);
    }
    return {};
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
