/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceResidualWork.h"

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceContactCertificate.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceEnergy.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceFittedALE.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceGeometry.h"
#include "Physics/Options/NavierStokesOptionRules.h"
#include "Physics/Options/NavierStokesOptionSchema.h"
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
#include "FE/Systems/SystemCollectives.h"
#include "Interfaces/GeneratedActiveBoundaryDomain.h"
#include "Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"
#include "Interfaces/LevelSetInterfaceDomain.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceRecords.h"

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

void installFreeSurfaceResidualWork(
    FE::systems::FESystem& system,
    const IncompressibleNavierStokesVMSOptions& options,
    FE::FieldId u_id,
    FE::FieldId p_id,
    const FE::systems::FormInstallOptions& install,
    const FreeSurfaceResidualWorkForms& work,
    const FreeSurfaceResidualWorkApplicability& applicability)
{
    using FE::forms::FormExpr;
    const auto& convection_work_form = work.convection;
    const auto& pressure_work_form = work.pressure;
    const auto& continuity_work_form = work.continuity;
    const auto& nonconservative_body_force_work_form =
        work.nonconservative_body_force;
    const auto& weak_boundary_momentum_work_form = work.weak_boundary_momentum;
    const auto& weak_boundary_continuity_work_form =
        work.weak_boundary_continuity;
    const auto& vms_pspg_momentum_work_form = work.vms_momentum;
    const auto& vms_pspg_continuity_work_form = work.vms_continuity;
    const auto& vms_pspg_pressure_gradient_form =
        work.vms_pspg_pressure_gradient;
    const auto& pressure_ghost_penalty_form = work.ghost_penalty;
    const bool include_mcv = applicability.include_mcv;
    const bool has_nonconservative_body_force =
        applicability.has_nonconservative_body_force;
    const bool enable_vms = applicability.enable_vms;
    const auto work_operator_tag =
        [&](std::string_view suffix) {
            return options.operator_tag + "_" +
                std::string(suffix);
        };
    const auto install_work_operator =
        [&](FE::systems::FreeSurfaceResidualWorkChannel channel,
            std::string_view suffix,
            const FormExpr& momentum_work_form,
            const FormExpr& continuity_work_form,
            const FormExpr& supplemental_continuity_work_form,
            bool retain_generated_nitsche_trace_requests) {
            const auto operator_tag = work_operator_tag(suffix);
            bool installed = false;
            if (momentum_work_form.isValid()) {
                auto momentum_install = install;
                momentum_install.contributes_to_problem_analysis = false;
                if (!retain_generated_nitsche_trace_requests) {
                    momentum_install
                        .generated_boundary_nitsche_trace_requests
                        .clear();
                }
                momentum_install.source_component_tag = operator_tag;
                (void)FE::systems::installFormulation(
                    system,
                    operator_tag,
                    {u_id},
                    momentum_work_form,
                    momentum_install);
                installed = true;
            }
            const auto install_continuity_component =
                [&](const FormExpr& work_form) {
                    if (!work_form.isValid()) {
                        return;
                    }
                    auto continuity_install = install;
                    continuity_install.contributes_to_problem_analysis =
                        false;
                    continuity_install
                        .generated_boundary_nitsche_trace_requests
                        .clear();
                    continuity_install.source_component_tag =
                        operator_tag;
                    if (std::find(
                            continuity_install.extra_trial_fields.begin(),
                            continuity_install.extra_trial_fields.end(),
                            u_id) ==
                        continuity_install.extra_trial_fields.end()) {
                        continuity_install.extra_trial_fields.push_back(
                            u_id);
                    }
                    (void)FE::systems::installFormulation(
                        system,
                        operator_tag,
                        {p_id},
                        work_form,
                        continuity_install);
                    installed = true;
                };
            install_continuity_component(continuity_work_form);
            install_continuity_component(
                supplemental_continuity_work_form);
            FE_THROW_IF(
                !installed,
                FE::InvalidArgumentException,
                "IncompressibleNavierStokesVMSModule: produced free-surface residual-work channel has no form component");
            free_surface::records(system).declareFreeSurfaceResidualWork(
                FE::systems::FreeSurfaceResidualWorkDeclaration{
                    .channel = channel,
                    .applicability = FE::systems::
                        FreeSurfaceResidualWorkApplicability::Produced,
                    .operator_tag = operator_tag,
                    .owner_component =
                        "IncompressibleNavierStokesVMSModule." +
                        std::string(
                            FE::systems::
                                freeSurfaceResidualWorkChannelName(
                                    channel)),
                });
        };
    const auto declare_inapplicable =
        [&](FE::systems::FreeSurfaceResidualWorkChannel channel) {
            free_surface::records(system).declareFreeSurfaceResidualWork(
                FE::systems::FreeSurfaceResidualWorkDeclaration{
                    .channel = channel,
                    .applicability = FE::systems::
                        FreeSurfaceResidualWorkApplicability::
                            NotApplicable,
                    .operator_tag = {},
                    .owner_component =
                        "IncompressibleNavierStokesVMSModule." +
                        std::string(
                            FE::systems::
                                freeSurfaceResidualWorkChannelName(
                                    channel)),
                });
        };

    if (options.enable_convection || include_mcv) {
        install_work_operator(
            FE::systems::FreeSurfaceResidualWorkChannel::Convection,
            FreeSurfaceResidualWorkOperatorSuffixes::convection,
            convection_work_form,
            FormExpr{},
            FormExpr{},
            false);
    } else {
        declare_inapplicable(
            FE::systems::FreeSurfaceResidualWorkChannel::Convection);
    }
    install_work_operator(
        FE::systems::FreeSurfaceResidualWorkChannel::
            PressureContinuity,
        FreeSurfaceResidualWorkOperatorSuffixes::
            pressure_continuity,
        pressure_work_form,
        continuity_work_form,
        FormExpr{},
        false);
    if (has_nonconservative_body_force) {
        install_work_operator(
            FE::systems::FreeSurfaceResidualWorkChannel::
                NonconservativeBodyForce,
            FreeSurfaceResidualWorkOperatorSuffixes::
                nonconservative_body_force,
            nonconservative_body_force_work_form,
            FormExpr{},
            FormExpr{},
            false);
    } else {
        declare_inapplicable(
            FE::systems::FreeSurfaceResidualWorkChannel::
                NonconservativeBodyForce);
    }
    if (weak_boundary_momentum_work_form.isValid() ||
        weak_boundary_continuity_work_form.isValid()) {
        install_work_operator(
            FE::systems::FreeSurfaceResidualWorkChannel::WeakBoundary,
            FreeSurfaceResidualWorkOperatorSuffixes::weak_boundary,
            weak_boundary_momentum_work_form,
            weak_boundary_continuity_work_form,
            FormExpr{},
            true);
    } else {
        declare_inapplicable(
            FE::systems::FreeSurfaceResidualWorkChannel::WeakBoundary);
    }
    if (enable_vms) {
        install_work_operator(
            FE::systems::FreeSurfaceResidualWorkChannel::VmsPspg,
            FreeSurfaceResidualWorkOperatorSuffixes::vms_pspg,
            vms_pspg_momentum_work_form,
            vms_pspg_continuity_work_form,
            vms_pspg_pressure_gradient_form,
            false);
    } else {
        declare_inapplicable(
            FE::systems::FreeSurfaceResidualWorkChannel::VmsPspg);
    }
    declare_inapplicable(
        FE::systems::FreeSurfaceResidualWorkChannel::CutStabilization);
    if (pressure_ghost_penalty_form.isValid()) {
        install_work_operator(
            FE::systems::FreeSurfaceResidualWorkChannel::GhostPenalty,
            FreeSurfaceResidualWorkOperatorSuffixes::ghost_penalty,
            FormExpr{},
            pressure_ghost_penalty_form,
            FormExpr{},
            false);
    } else {
        declare_inapplicable(
            FE::systems::FreeSurfaceResidualWorkChannel::GhostPenalty);
    }
    FE_LOG_INFO(
        "IncompressibleNavierStokesVMSModule: diagnostic=free_surface_residual_work_operators status=declared channels=convection|pressure_continuity|nonconservative_body_force|weak_boundary|vms_pspg|cut_stabilization|ghost_penalty contract=negative_dt_times_constrained_operator_stage_pairing");
}

FreeSurfaceResidualWorkForms bulkResidualWork(
    const NavierStokesVMSIntegrands& integrands,
    const FE::forms::FormExpr& momentum_form,
    const FE::forms::FormExpr& continuity_form,
    const FE::forms::FormExpr& vms_pspg_pressure_gradient_form,
    const FE::forms::FormExpr& conservative_body_force_residual,
    bool has_nonconservative_body_force,
    bool enable_vms,
    const FE::forms::VolumeDomain& liquid)
{
    FreeSurfaceResidualWorkForms work;
    work.convection = integrands.convection.dx(liquid);
    work.pressure = integrands.pressure.dx(liquid);
    work.continuity = integrands.galerkin_continuity.dx(liquid);
    if (has_nonconservative_body_force) {
        work.nonconservative_body_force =
            (integrands.forcing - conservative_body_force_residual).dx(liquid);
    }
    if (enable_vms) {
        work.vms_momentum = momentum_form - integrands.galerkin_momentum.dx(liquid);
        work.vms_continuity = continuity_form - work.continuity;
        work.vms_pspg_pressure_gradient = vms_pspg_pressure_gradient_form;
    }
    return work;
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
