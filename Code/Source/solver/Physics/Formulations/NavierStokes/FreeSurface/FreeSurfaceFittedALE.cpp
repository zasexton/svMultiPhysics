/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceFittedALE.h"

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceCollectives.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceContactCertificate.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceEnergy.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceGeometry.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfacePressureGauge.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceQueries.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceSemiImplicitSurfaceTension.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceValidation.h"
#include "Physics/Formulations/NavierStokes/NavierStokesBCFactories.h"
#include "Physics/Formulations/NavierStokes/NavierStokesFields.h"
#include "Physics/Formulations/NavierStokes/NavierStokesGeneratedNitsche.h"
#include "Physics/Formulations/NavierStokes/NavierStokesInitialConditions.h"
#include "Physics/Formulations/NavierStokes/NavierStokesPreflight.h"

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
        system.registerFittedALENormalOperatorStageMeasurement(
            displacement_field,
            marker,
            prescribed_tangential_mesh_velocity);
    }
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
