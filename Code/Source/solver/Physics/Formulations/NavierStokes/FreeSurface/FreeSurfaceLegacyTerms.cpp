/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceLegacyTerms.h"

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceConfigurationArtifact.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceContactCertificate.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceDiagnosticOperators.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceEnergy.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceFittedALE.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceGeometry.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfacePressureGauge.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceQueries.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceResidualWork.h"
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

FE::forms::FormExpr unfittedCurvatureTraction(
    const FreeSurfaceBoundary& bc,
    const FE::systems::FESystem& system,
    const FE::forms::FormExpr& phi,
    const FE::forms::FormExpr& p_ext,
    const FE::forms::FormExpr& gamma,
    const FE::forms::FormExpr& n,
    const FE::forms::FormExpr& v)
{
    using namespace FE::forms;
    const auto curvature = bc::isZeroConstantScalarValue(bc.surface_tension)
        ? FormExpr::constant(0.0)
        : unfittedTractionCurvature(bc, system, phi);
    const auto traction_scalar = -p_ext - gamma * curvature;
    const auto traction = traction_scalar * n;
    return FE::forms::FormExpr::constant(FE::Real{-1.0}) *
           inner(traction, v);
}

void addFittedFluidKinematicRow(
    FE::forms::FormExpr& momentum_form,
    FE::forms::FormExpr& continuity_form,
    const FreeSurfaceBoundary& bc,
    const FE::forms::FormExpr& u,
    const FE::forms::FormExpr& p,
    const FE::forms::FormExpr& v,
    const FE::forms::FormExpr& q,
    const FE::forms::FormExpr& mesh_velocity,
    const FE::forms::FormExpr& mu,
    const FE::forms::FormExpr& n,
    bool ale_enabled,
    FE::forms::FormExpr* weak_boundary_momentum_work_form,
    FE::forms::FormExpr* weak_boundary_continuity_work_form)
{
    using namespace FE::forms;

    const auto append_weak_work = [](
        FormExpr* destination,
        const FormExpr& contribution) {
        if (destination == nullptr) {
            return;
        }
        *destination = destination->isValid()
                           ? (*destination + contribution)
                           : contribution;
    };

    switch (bc.kinematic_enforcement) {
    case FreeSurfaceKinematicEnforcement::None:
        return;
    case FreeSurfaceKinematicEnforcement::MeshNitsche:
        // The fluid keeps the natural dynamic condition; the normal
        // kinematics are owned by the mesh-displacement row and the
        // mesh-motion consistency term.  A fluid kinematic row here would
        // replace the normal dynamic condition by u.n = w.n.
        if (isUnfittedLevelSet(bc)) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: MeshNitsche free-surface kinematics are only supported on fitted ALE boundaries");
        }
        return;
    case FreeSurfaceKinematicEnforcement::Penalty: {
        if (bc.normal_kinematic_policy !=
            FreeSurfaceNormalKinematicPolicy::MatchFluidNormalVelocity) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: unsupported fitted free-surface normal kinematic policy");
        }
        const auto penalty = bc::toScalarExpr(
            bc.kinematic_penalty,
            freeSurfaceValueName("ns_free_surface_kinematic_penalty", bc));
        const auto normal_mismatch = normalTrace(u - mesh_velocity, n);
        const auto residual_integrand =
            penalty * normal_mismatch * normalTrace(v, n);
        const auto residual_form = integrateOnFreeSurface(
            residual_integrand, bc, ale_enabled);
        momentum_form = momentum_form + residual_form;
        append_weak_work(
            weak_boundary_momentum_work_form, residual_form);
        return;
    }
    case FreeSurfaceKinematicEnforcement::Nitsche: {
        if (isUnfittedLevelSet(bc)) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: Nitsche free-surface kinematics are only supported on fitted ALE boundaries");
        }
        if (bc.normal_kinematic_policy !=
            FreeSurfaceNormalKinematicPolicy::MatchFluidNormalVelocity) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: unsupported fitted free-surface normal kinematic policy");
        }
        if (!(bc.kinematic_nitsche_gamma > 0.0)) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: Nitsche free-surface kinematics require kinematic_nitsche_gamma > 0");
        }

        const auto normal_mismatch = normalTrace(u - mesh_velocity, n);
        const auto v_normal = normalTrace(v, n);
        const auto stress_u = FormExpr::constant(2.0) * mu * sym(grad(u));
        const auto stress_v = FormExpr::constant(2.0) * mu * sym(grad(v));
        const auto normal_stress_u = normalTrace(stress_u * n, n);
        const auto normal_stress_v = normalTrace(stress_v * n, n);
        const auto penalty = bc::buildTraceNitschePenalty(
            mu / hNormal(),
            u,
            bc::TraceNitscheOptions{
                .gamma = bc.kinematic_nitsche_gamma,
                .variant = bc.kinematic_nitsche_symmetric
                    ? bc::NitscheVariant::Symmetric
                    : bc::NitscheVariant::Unsymmetric,
                .scale_with_p = bc.kinematic_nitsche_scale_with_p});

        const auto consistency_penalty_form = integrateOnFreeSurface(
            (p - normal_stress_u) * v_normal +
            penalty * normal_mismatch * v_normal,
            bc,
            ale_enabled);
        momentum_form = momentum_form + consistency_penalty_form;
        append_weak_work(
            weak_boundary_momentum_work_form,
            consistency_penalty_form);
        if (bc.kinematic_nitsche_symmetric) {
            const auto adjoint_form = -integrateOnFreeSurface(
                normal_stress_v * normal_mismatch,
                bc,
                ale_enabled);
            const auto kinematic_continuity_form =
                integrateOnFreeSurface(
                    q * normal_mismatch, bc, ale_enabled);
            momentum_form = momentum_form + adjoint_form;
            continuity_form =
                continuity_form + kinematic_continuity_form;
            append_weak_work(
                weak_boundary_momentum_work_form, adjoint_form);
            append_weak_work(
                weak_boundary_continuity_work_form,
                kinematic_continuity_form);
        } else {
            const auto adjoint_form = integrateOnFreeSurface(
                normal_stress_v * normal_mismatch,
                bc,
                ale_enabled);
            const auto kinematic_continuity_form =
                -integrateOnFreeSurface(
                    q * normal_mismatch, bc, ale_enabled);
            momentum_form = momentum_form + adjoint_form;
            continuity_form =
                continuity_form + kinematic_continuity_form;
            append_weak_work(
                weak_boundary_momentum_work_form, adjoint_form);
            append_weak_work(
                weak_boundary_continuity_work_form,
                kinematic_continuity_form);
        }
        return;
    }
    }

    throw std::invalid_argument(
        "IncompressibleNavierStokesVMSModule: unsupported free-surface kinematic enforcement");
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
