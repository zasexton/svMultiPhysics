#ifndef SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_BC_FACTORIES_H
#define SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_BC_FACTORIES_H

/**
 * @file NavierStokesBCFactories.h
 * @brief The boundary conditions of the Navier–Stokes formulation, from its options
 *
 * Header-only: the standard builders (traction, Robin, outflow, coupled RCR
 * and RCRCR outlets, essential conditions, the velocity Nitsche terms) and
 * the free-surface boundary condition, unfitted and fitted: its contact-line
 * laws, traction, semi-implicit capillary increment, cut-facet pressure
 * penalty, mesh rows, contact pin and pressure anchor; and the two-fluid
 * material-interface coupling of the two phases.  Keeps
 * `IncompressibleNavierStokesVMSModule.cpp` the statement of the weak form.
 */

#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"
#include "Physics/Formulations/NavierStokes/NavierStokesPostProcessing.h"
#include "Physics/Options/NavierStokesOptionRules.h"
#include "Physics/Options/NavierStokesOptionSchema.h"

#include "FE/Constraints/ConstraintPlan.h"
#include "FE/Constraints/InterfaceMeasureGaugeAnchor.h"
#include "FE/Core/Logger.h"
#include "FE/Forms/BoundaryConditions.h"
#include "FE/Forms/BoundaryFunctional.h"
#include "FE/Forms/CutCellForms.h"
#include "FE/Forms/IntegrationDomain.h"
#include "FE/Forms/ResidualContributions.h"
#include "FE/Forms/StandardBCs.h"
#include "FE/Forms/Vocabulary.h"
#include "FE/Interfaces/ContactLineGeometryCertificate.h"
#include "FE/Auxiliary/AuxiliaryBindings.h"
#include "FE/Auxiliary/AuxiliaryModelBuilder.h"
#include "FE/Auxiliary/AuxiliaryModelDSL.h"
#include "FE/LevelSet/LevelSetGeometry.h"
#include "FE/Systems/ALEBinding.h"
#include "FE/Systems/BoundaryConditionManager.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FieldBinding.h"
#include "FE/Systems/FormsInstaller.h"
#include "FE/Systems/FormulationBuilder.h"
#include "FE/Systems/MeshBoundaryKinematics.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

/// The configured wall normal of a contact line, normalized, as a constant
/// vector form (formerly FreeSurface/FreeSurfaceQueries).
[[nodiscard]] inline FE::forms::FormExpr wallNormalExpression(
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

// The fitted-ALE mesh rows and the contact-line certificates of a free
// surface (formerly FreeSurface/FreeSurfaceFittedALE and
// FreeSurface/FreeSurfaceContactCertificate), on the FE moving-boundary and
// contact-line services.

[[nodiscard]] inline const FE::systems::FieldRecord& fittedMeshDisplacementRecord(
    const FE::systems::FESystem& system,
    const FE::systems::ALEBinding& ale_binding)
{
    return FE::systems::requireMeshDisplacementRecord(
        system,
        ale_binding,
        "IncompressibleNavierStokesVMSModule: fitted free-surface mesh kinematics require a coupled mesh displacement unknown",
        "IncompressibleNavierStokesVMSModule: fitted free-surface mesh displacement field has no function space");
}

inline void declareFittedTangentialMeshPolicy(
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

    FE::logRecord(post::navierStokesLog("fitted free-surface mesh policy")
                      .diagnostic("fitted_free_surface_mesh_policy")
                      .integer("marker", bc.boundary_marker)
                      .text("tangential_policy", tangentialMeshPolicyName(bc.tangential_mesh_policy))
                      .text("tangential_owner", "free_surface_boundary")
                      .text("tangential_enforcement", install_tangential_prescription ? "weak_velocity_penalty"
                                                      : install_tangential_smoothing
                                                          ? "surface_smoothing_functional"
                                                          : "natural_no_tangential_constraint"));
}

inline void declareFittedMeshRowDescriptors(
    FE::systems::FESystem& system,
    const std::string& operator_tag,
    const FE::systems::ALEBinding& ale_binding,
    const FreeSurfaceBoundary& bc,
    FE::FieldId velocity_field,
    bool rows_installed)
{
    FE::systems::declareMeshBoundaryRowDescriptors(
        system,
        operator_tag,
        FE::systems::MeshBoundaryRows{
            .boundary_marker = bc.boundary_marker,
            .mesh_displacement_field = ale_binding.mesh_displacement_field,
            .velocity_field = velocity_field,
            .normal_relation =
                bc.kinematic_enforcement != FreeSurfaceKinematicEnforcement::None &&
                bc.normal_kinematic_policy ==
                    FreeSurfaceNormalKinematicPolicy::MatchFluidNormalVelocity,
            .tangential_prescribed =
                bc.tangential_mesh_policy ==
                FreeSurfaceTangentialMeshPolicy::Prescribed,
            .tangential_policy =
                systemTangentialMeshPolicy(bc.tangential_mesh_policy),
            .mesh_normal_source =
                "Fitted free-surface mesh normal kinematic row on marker " +
                std::to_string(bc.boundary_marker),
            .fluid_normal_source = fittedFluidNormalDescriptorSource(bc),
            .tangential_source = fittedTangentialOperatorSource(
                bc.boundary_marker, bc.tangential_mesh_policy),
            .missing_velocity_message =
                "IncompressibleNavierStokesVMSModule: fitted normal "
                "kinematics require a related fluid velocity field",
        },
        rows_installed);
}

inline void registerFreeSurfacePrescribedAngleGeometry(
    FE::systems::FESystem& system,
    const FreeSurfaceBoundary& bc,
    const PreparedSurfaceIdentity& surface)
{
    for (const auto& identity : surface.contacts) {
        const auto& contact_line = bc.contact_lines[identity.contact_index];
        if (contactLineKind(contact_line) != ContactLineKind::PrescribedAngle) {
            continue;
        }
        if (isUnfittedLevelSet(bc)) {
            const auto phi_id = surface.geometry.value().levelSetField();
            if (!system.fieldParticipatesInUnknownVector(phi_id)) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: prescribed unfitted contact angles require the level-set field to be an unknown");
            }

            const int contact_marker = identity.domain.marker();
            system.registerGeneratedEmbeddedInterfaceMarker(contact_marker);
            const auto check = contactLineCheck(contact_line, contact_marker);
            FE::interfaces::registerContactLineCertificate(
                system,
                check,
                "navier_stokes_prescribed_contact_geometry:" +
                    std::to_string(contact_marker),
                "prescribed_contact_geometry",
                "prescribed_contact_geometry_initial",
                FE::interfaces::ContactLineCertificateRefresh::EveryContext);
            FE::logRecord(
                post::navierStokesLog("registered prescribed-angle contact geometry without a level-set residual")
                    .diagnostic("navier_stokes_prescribed_contact_geometry")
                    .integer("interface_marker", bc.interface_marker)
                    .integer("contact_marker", contact_marker)
                    .text("level_set_geometry_owner", "none_angle_preserving_wall_maintenance")
                    .text("momentum_owner", "young_wall_energy")
                    .text("literal_codimension_two_level_set_residual", "retired"));
            continue;
        }

        throw std::logic_error(
            "IncompressibleNavierStokesVMSModule: fitted prescribed contact-angle validation did not fail closed");
    }
}

inline void registerDynamicContactWallNormalCheck(
    FE::systems::FESystem& system,
    const FreeSurfaceContactLine& contact_line,
    int contact_marker)
{
    const auto check = contactLineCheck(contact_line, contact_marker);
    FE::interfaces::registerContactLineCertificate(
        system,
        check,
        "navier_stokes_dynamic_contact_wall_normal:" +
            std::to_string(contact_marker),
        "dynamic_contact_operator_angle",
        "dynamic_contact_operator_angle_initial",
        FE::interfaces::ContactLineCertificateRefresh::SetUpSystemContext);
}

namespace Factories {

namespace detail {

    /**
     * @brief Create an RCR outflow BC model.
     *
     * This is code defines the RCR windkessel model for an AuxiliaryState
     * system defining the ODE that will be coupled to the boundary of a fluid
     * problem.
     *
     * @note See also: toCoupledOutflowBC() for the registration of this
     * AuxiliaryState model to the FESystem.
     */
[[nodiscard]] inline std::shared_ptr<FE::systems::BuiltAuxiliaryModel> rcrOutflowModel()
{
    static const auto model = FE::systems::aux::model("rcr_windkessel", [](FE::systems::ModelFacade& m) {
        auto Q = m.input("Q");
        auto X = m.state("X");
        auto [Rp, C, Rd, Pd] = m.params("Rp", "C", "Rd", "Pd");

        m << FE::systems::ddt(X) == (Q - (X - Pd) / Rd) / C;
        m << FE::systems::out("P_out") == X + Rp * Q;
    });
    return model;
}

[[nodiscard]] inline std::shared_ptr<FE::systems::BuiltAuxiliaryModel> resistiveOutflowModel()
{
    static const auto model = FE::systems::aux::model("resistive_outflow", [](FE::systems::ModelFacade& m) {
        auto Q = m.input("Q");
        auto P = m.state("P", FE::systems::AuxiliaryVariableKind::Algebraic);
        auto [Rsum, Pd] = m.params("Rsum", "Pd");

        m.initialGuess("P", 0.0);
        m << FE::systems::alg(P) == P - (Pd + Rsum * Q);
        m << FE::systems::out("P_out") == P;
    });
    return model;
}

[[nodiscard]] inline std::shared_ptr<FE::systems::BuiltAuxiliaryModel> rcrcrOutflowModel()
{
    static const auto model = FE::systems::aux::model("rcrcr_windkessel", [](FE::systems::ModelFacade& m) {
        auto Q = m.input("Q");
        auto P1 = m.state("P1");
        auto P2 = m.state("P2");
        auto Rp = m.param("Rp");
        auto C1 = m.param("C1");
        auto Rm = m.param("Rm");
        auto C2 = m.param("C2");
        auto Rd = m.param("Rd");
        auto Pd = m.param("Pd");

        m << FE::systems::ddt(P1) == (Q - (P1 - P2) / Rm) / C1;
        m << FE::systems::ddt(P2) == ((P1 - P2) / Rm - (P2 - Pd) / Rd) / C2;
        m << FE::systems::out("P_out") == P1 + Rp * Q;
    });
    return model;
}

} // namespace detail

[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition> reserveMarker(
    const IncompressibleNavierStokesVMSOptions::VelocityDirichletBC& bc)
{
    const int marker = FE::forms::bc::detail::boundaryMarkerOrThrow(bc, "navier_stokes::Factories::reserveMarker");
    return std::make_unique<FE::forms::bc::ReservedBC>(marker);
}

[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition> toTractionBC(
    const IncompressibleNavierStokesVMSOptions::TractionNeumannBC& bc,
    int dim,
    std::optional<int> generated_active_boundary_marker = std::nullopt)
{
    const int marker = FE::forms::bc::detail::boundaryMarkerOrThrow(bc, "navier_stokes::Factories::toTractionBC");

    auto t_comp = FE::forms::bc::toVectorExpr(
        bc.traction, dim, "ns_traction_neumann", marker, FE::forms::bc::ComponentValueNameStyle::Component);
    return std::make_unique<FE::forms::bc::NaturalBC>(
        marker,
        FE::forms::FormExpr::asVector(std::move(t_comp)),
        generated_active_boundary_marker);
}

[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition> toTractionRobinBC(
    const IncompressibleNavierStokesVMSOptions::TractionRobinBC& bc,
    int dim,
    std::optional<int> generated_active_boundary_marker = std::nullopt)
{
    const int marker =
        FE::forms::bc::detail::boundaryMarkerOrThrow(bc, "navier_stokes::Factories::toTractionRobinBC");

    const auto alpha = FE::forms::bc::toScalarExpr(
        bc.alpha, FE::forms::bc::markerValueName("ns_traction_robin_alpha", marker));

    auto r_comp = FE::forms::bc::toVectorExpr(
        bc.rhs, dim, "ns_traction_robin_rhs", marker, FE::forms::bc::ComponentValueNameStyle::Component);

    return std::make_unique<FE::forms::bc::RobinBC>(
        marker,
        alpha,
        FE::forms::FormExpr::asVector(std::move(r_comp)),
        generated_active_boundary_marker);
}

[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition> toVelocityEssentialBC(
    const IncompressibleNavierStokesVMSOptions::VelocityDirichletBC& bc,
    int dim,
    std::string_view symbol)
{
    const int marker =
        FE::forms::bc::detail::boundaryMarkerOrThrow(bc, "navier_stokes::Factories::toVelocityEssentialBC");

    auto uD_comp = FE::forms::bc::toVectorExpr(
        bc.value, dim, "ns_u_dirichlet", marker, FE::forms::bc::ComponentValueNameStyle::Component);

    bool all_active = true;
    for (int d = 0; d < dim; ++d) {
        all_active = all_active && bc.active_components[static_cast<std::size_t>(d)];
    }
    if (all_active) {
        return std::make_unique<FE::forms::bc::EssentialBC>(marker, std::move(uD_comp), std::string(symbol));
    }

    return FE::forms::bc::vectorComponentEssentialBC(
        bc.value,
        bc.active_components,
        dim,
        marker,
        "ns_u_dirichlet",
        symbol,
        "Navier-Stokes velocity Dirichlet",
        FE::forms::bc::ComponentValueNameStyle::Component);
}

[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition> toPressureEssentialBC(
    const IncompressibleNavierStokesVMSOptions::PressureDirichletBC& bc,
    std::string_view symbol)
{
    const int marker =
        FE::forms::bc::detail::boundaryMarkerOrThrow(bc, "navier_stokes::Factories::toPressureEssentialBC");
    const auto value =
        FE::forms::bc::toScalarExpr(bc.value, FE::forms::bc::markerValueName("ns_p_dirichlet", marker));
    return std::make_unique<FE::forms::bc::EssentialBC>(marker, value, std::string(symbol));
}

[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition> toOutflowBC(
    const IncompressibleNavierStokesVMSOptions::PressureOutflowBC& bc,
    const FE::forms::FormExpr& u,
    const FE::forms::FormExpr& rho,
    std::optional<int> generated_active_boundary_marker = std::nullopt)
{
    const int marker = FE::forms::bc::detail::boundaryMarkerOrThrow(bc, "navier_stokes::Factories::toOutflowBC");

    const auto n = FE::forms::FormExpr::normal();
    const auto un = FE::forms::inner(u, n);
    const auto max_backflow =
        FE::forms::FormExpr::constant(0.5) * (FE::forms::abs(un) - un); // max(0, -u·n)

    const auto p_out =
        FE::forms::bc::toScalarExpr(bc.pressure, FE::forms::bc::markerValueName("ns_p_out", marker));
    const auto beta =
        FE::forms::bc::toScalarExpr(bc.backflow_beta, FE::forms::bc::markerValueName("ns_backflow_beta", marker));

    const auto flux = -p_out * n - beta * rho * max_backflow * u;
    return std::make_unique<FE::forms::bc::NaturalBC>(
        marker, flux, generated_active_boundary_marker);
}

/**
 * @brief Create an RCR outflow BC using the math-first auxiliary DSL.
 *
 * This is the preferred implementation path.  It registers the boundary
 * flow integral as an auxiliary input, builds the RCR model via the
 * math-first DSL, deploys it with typed handles, and returns a standard
 * NaturalBC that reads the outlet pressure from the deployment handle.
 *
 * @param bc                  RCR outflow boundary condition options.
 * @param system              The FESystem to register inputs and deploy models on.
 * @param u                   FormExpr for velocity (test/trial context).
 * @param rho                 FormExpr for density.
 */
[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition> toCoupledOutflowBC(
    const IncompressibleNavierStokesVMSOptions::CoupledRCROutflowBC& bc,
    FE::systems::FESystem& system,
    const FE::forms::FormExpr& u,
    const FE::forms::FormExpr& rho,
    std::optional<int> generated_active_boundary_marker = std::nullopt)
{
    using namespace FE::forms;
    using namespace FE::systems;

    const int marker =
        FE::forms::bc::detail::boundaryMarkerOrThrow(bc, "navier_stokes::Factories::toCoupledOutflowBC");

    const FE::Real Rp = bc.Rp;
    const FE::Real C  = bc.C;
    const FE::Real Rd = bc.Rd;
    const FE::Real Pd = bc.Pd;

    if (Rd == 0.0) {
        throw std::invalid_argument("CoupledRCROutflowBC: Rd must be nonzero");
    }

    const auto n  = FormExpr::normal();
    const auto un = inner(u, n);
    const auto max_backflow = FormExpr::constant(0.5) * (abs(un) - un);
    const auto beta =
        FE::forms::bc::toScalarExpr(
            bc.backflow_beta, FE::forms::bc::markerValueName("ns_rcr_backflow_beta", marker));

    if (C == 0.0) {
        FE::forms::BoundaryFunctional flow_rate;
        flow_rate.integrand = inner(u, n);
        flow_rate.boundary_marker = marker;
        flow_rate.generated_active_boundary_marker =
            generated_active_boundary_marker;
        flow_rate.reduction = FE::forms::BoundaryFunctional::Reduction::Sum;
        flow_rate.name = bc.functional_name;
        auto Q = system.boundaryIntegral(
            std::move(flow_rate),
            FE::systems::AuxiliaryInputUpdateSchedule::EachNonlinearIteration);

        auto resistive = system.deploy(
            use(detail::resistiveOutflowModel())
                .boundary(marker)
                .monolithic()
                .params({{"Rsum", Rp + Rd}, {"Pd", Pd}})
                .bind("Q", Q)
                .initialState({{"P", Pd}})
        );
        const auto p_out = resistive.output("P_out");
        const auto flux = -p_out * n - beta * rho * max_backflow * u;
        return std::make_unique<FE::forms::bc::NaturalBC>(
            marker, flux, generated_active_boundary_marker);
    } else {
        // Step 1: Register boundary-integral input for Q via handle-returning API.
        FE::forms::BoundaryFunctional flow_rate;
        flow_rate.integrand = inner(u, n);
        flow_rate.boundary_marker = marker;
        flow_rate.generated_active_boundary_marker =
            generated_active_boundary_marker;
        flow_rate.reduction = FE::forms::BoundaryFunctional::Reduction::Sum;
        flow_rate.name = bc.functional_name;
        auto Q = system.boundaryIntegral(
            std::move(flow_rate),
            FE::systems::AuxiliaryInputUpdateSchedule::EachNonlinearIteration);

        // Step 2: Deploy the standard RCR model into the monolithic Newton
        // block so the outlet pressure and flow Jacobian are solved together.
        auto rcr = system.deploy(
            use(detail::rcrOutflowModel())
                .boundary(marker)
                .monolithic()
                .params({{"Rp", Rp}, {"C", C}, {"Rd", Rd}, {"Pd", Pd}})
                .bind("Q", Q)
                .initialState({{"X", bc.X0}})
        );
        // Step 3: Return a standard NaturalBC.
        const auto p_out = rcr.output("P_out");
        const auto flux = -p_out * n - beta * rho * max_backflow * u;
        return std::make_unique<FE::forms::bc::NaturalBC>(
            marker, flux, generated_active_boundary_marker);
    }
}

[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition> toCoupledOutflowBC(
    const IncompressibleNavierStokesVMSOptions::CoupledRCRCROutflowBC& bc,
    FE::systems::FESystem& system,
    const FE::forms::FormExpr& u,
    const FE::forms::FormExpr& rho,
    std::optional<int> generated_active_boundary_marker = std::nullopt)
{
    using namespace FE::forms;
    using namespace FE::systems;

    const int marker =
        FE::forms::bc::detail::boundaryMarkerOrThrow(bc, "navier_stokes::Factories::toCoupledOutflowBC");

    if (bc.C1 == 0.0 || bc.C2 == 0.0) {
        throw std::invalid_argument("CoupledRCRCROutflowBC: C1 and C2 must be nonzero");
    }
    if (bc.Rm == 0.0 || bc.Rd == 0.0) {
        throw std::invalid_argument("CoupledRCRCROutflowBC: Rm and Rd must be nonzero");
    }

    const auto n = FormExpr::normal();
    const auto un = inner(u, n);
    const auto max_backflow = FormExpr::constant(0.5) * (abs(un) - un);
    const auto beta =
        FE::forms::bc::toScalarExpr(
            bc.backflow_beta, FE::forms::bc::markerValueName("ns_rcrcr_backflow_beta", marker));

    FE::forms::BoundaryFunctional flow_rate;
    flow_rate.integrand = inner(u, n);
    flow_rate.boundary_marker = marker;
    flow_rate.generated_active_boundary_marker =
        generated_active_boundary_marker;
    flow_rate.reduction = FE::forms::BoundaryFunctional::Reduction::Sum;
    flow_rate.name = bc.functional_name;
    auto Q = system.boundaryIntegral(
        std::move(flow_rate),
        FE::systems::AuxiliaryInputUpdateSchedule::EachNonlinearIteration);

    auto rcrcr = system.deploy(
        use(detail::rcrcrOutflowModel())
            .boundary(marker)
            .monolithic()
            .params({
                {"Rp", bc.Rp},
                {"C1", bc.C1},
                {"Rm", bc.Rm},
                {"C2", bc.C2},
                {"Rd", bc.Rd},
                {"Pd", bc.Pd},
            })
            .bind("Q", Q)
            .initialState({{"P1", bc.P10}, {"P2", bc.P20}})
    );

    const auto p_out = rcrcr.output("P_out");
    const auto flux = -p_out * n - beta * rho * max_backflow * u;
    return std::make_unique<FE::forms::bc::NaturalBC>(
        marker, flux, generated_active_boundary_marker);
}

namespace detail {

// Resolve one boundary only when its turn is reached, after the preceding
// route has been constructed. This preserves raw-input rejection precedence.
template <class Boundaries, class ResolveBoundary, class AppendMomentum, class AppendContinuity>
inline void appendVelocityNitscheBoundaries(
    const IncompressibleNavierStokesVMSOptions& options,
    const Boundaries& boundaries,
    ResolveBoundary&& resolve_boundary,
    const FE::forms::FormExpr& u,
    const FE::forms::FormExpr& p,
    const FE::forms::FormExpr& v,
    const FE::forms::FormExpr& q,
    const FE::forms::FormExpr& mu,
    std::vector<
        FE::forms::bc::
            GeneratedBoundaryNitscheTraceFormBinding>*
        generated_trace_bindings,
    AppendMomentum&& append_momentum,
    AppendContinuity&& append_continuity,
    std::vector<PreparedWeakVelocityBoundary>* prepared_boundaries)
{
    if (boundaries.empty()) {
        return;
    }
    if (!FE::numeric::isPositive(options.nitsche_gamma) ||
        !FE::numeric::isFinite(options.nitsche_gamma)) {
        throw std::invalid_argument(
            "applyVelocityNitscheBCs: nitsche_gamma must be finite and > 0");
    }

    const auto n = FE::forms::FormExpr::normal();
    const auto penalty = FE::forms::bc::buildTraceNitschePenalty(
        mu / FE::forms::hNormal(),
        u,
        FE::forms::bc::TraceNitscheOptions{
            .gamma = options.nitsche_gamma,
            .variant = options.nitsche_symmetric
                ? FE::forms::bc::NitscheVariant::Symmetric
                : FE::forms::bc::NitscheVariant::Unsymmetric,
            .scale_with_p = options.nitsche_scale_with_p});

    const auto stress_u = FE::forms::FormExpr::constant(2.0) * mu * FE::forms::symgrad(u);
    const auto stress_v = FE::forms::FormExpr::constant(2.0) * mu * FE::forms::symgrad(v);
    const auto integrate =
        [](const FE::forms::FormExpr& integrand,
           int marker,
           std::optional<int> generated_marker) {
            const auto measure =
                generated_marker.has_value()
                    ? FE::forms::ExteriorBoundaryMeasure::
                          generatedActiveSubset(
                              marker, *generated_marker)
                    : FE::forms::ExteriorBoundaryMeasure::
                          fullPhysical(marker);
            return integrand.dExteriorBoundary(measure);
        };
    for (const auto& boundary : boundaries) {
        const auto& prepared = resolve_boundary(boundary);
        const auto& [marker, generated_marker, uD] = prepared;
        const auto diff = u - uD;

        // Pressure consistency remains outside the form-bound viscous route.
        append_momentum(
            integrate(
                p * FE::forms::inner(n, v),
                marker,
                generated_marker));

        if (generated_marker.has_value() &&
            generated_trace_bindings != nullptr) {
            auto bound_terms =
                FE::forms::bc::
                    buildGeneratedBoundaryNitscheTraceTerms(
                        u,
                        v,
                        uD,
                        FE::forms::bc::symmetricGradientForm(mu),   // the viscous form 2 mu eps(u) : eps(v)
                        marker,
                        *generated_marker,
                        FE::forms::bc::TraceNitscheOptions{
                            .gamma = options.nitsche_gamma,
                            .variant = options.nitsche_symmetric
                                ? FE::forms::bc::
                                      NitscheVariant::Symmetric
                                : FE::forms::bc::
                                      NitscheVariant::Unsymmetric,
                            .scale_with_p =
                                options.nitsche_scale_with_p});
            append_momentum(bound_terms.route_contribution);
            if (options.nitsche_symmetric) {
                append_continuity(
                    integrate(
                        q * FE::forms::inner(n, diff),
                        marker,
                        generated_marker));
            } else {
                append_continuity(
                    -integrate(
                        q * FE::forms::inner(n, diff),
                        marker,
                        generated_marker));
            }
            generated_trace_bindings->push_back(
                std::move(bound_terms.binding));
            if (prepared_boundaries != nullptr) {
                prepared_boundaries->push_back(prepared);
            }
            continue;
        }

        // Ordinary fitted or uncertified routes retain the generic lowering.
        append_momentum(
            -integrate(
                FE::forms::inner(stress_u * n, v),
                marker,
                generated_marker));
        if (options.nitsche_symmetric) {
            append_momentum(
                -integrate(
                    FE::forms::inner(stress_v * n, diff),
                    marker,
                    generated_marker));
            append_continuity(
                integrate(
                    q * FE::forms::inner(n, diff),
                    marker,
                    generated_marker));
        } else {
            append_momentum(
                integrate(
                    FE::forms::inner(stress_v * n, diff),
                    marker,
                    generated_marker));
            append_continuity(
                -integrate(
                    q * FE::forms::inner(n, diff),
                    marker,
                    generated_marker));
        }
        append_momentum(
            integrate(
                penalty * FE::forms::inner(diff, v),
                marker,
                generated_marker));
        if (prepared_boundaries != nullptr) {
            prepared_boundaries->push_back(prepared);
        }
    }
}

// Compatibility destinations may alias one another. Preserve their sequential
// updates rather than passing them through the stricter production-row sink.
template <class Boundaries, class ResolveBoundary>
inline void applyVelocityNitscheBoundaries(
    FE::forms::FormExpr& momentum_form,
    FE::forms::FormExpr& continuity_form,
    const IncompressibleNavierStokesVMSOptions& options,
    const Boundaries& boundaries,
    ResolveBoundary&& resolve_boundary,
    const FE::forms::FormExpr& u,
    const FE::forms::FormExpr& p,
    const FE::forms::FormExpr& v,
    const FE::forms::FormExpr& q,
    const FE::forms::FormExpr& mu,
    std::vector<FE::forms::bc::GeneratedBoundaryNitscheTraceFormBinding>* generated_trace_bindings,
    FE::forms::FormExpr* momentum_residual_work_form,
    FE::forms::FormExpr* continuity_residual_work_form,
    std::vector<PreparedWeakVelocityBoundary>* prepared_boundaries)
{
    const auto append_momentum = [&](const FE::forms::FormExpr& contribution) {
        momentum_form = FE::forms::sumOf(momentum_form, contribution);
        if (momentum_residual_work_form != nullptr) {
            *momentum_residual_work_form = FE::forms::sumOf(
                *momentum_residual_work_form, contribution);
        }
    };
    const auto append_continuity = [&](const FE::forms::FormExpr& contribution) {
        continuity_form = FE::forms::sumOf(continuity_form, contribution);
        if (continuity_residual_work_form != nullptr) {
            *continuity_residual_work_form = FE::forms::sumOf(
                *continuity_residual_work_form, contribution);
        }
    };
    appendVelocityNitscheBoundaries(options, boundaries, std::forward<ResolveBoundary>(resolve_boundary),
        u, p, v, q, mu, generated_trace_bindings, append_momentum, append_continuity, prepared_boundaries);
}

} // namespace detail

inline void applyVelocityNitscheBCs(
    FE::forms::FormExpr& momentum_form,
    FE::forms::FormExpr& continuity_form,
    const IncompressibleNavierStokesVMSOptions& options,
    int dim,
    const FE::forms::FormExpr& u,
    const FE::forms::FormExpr& p,
    const FE::forms::FormExpr& v,
    const FE::forms::FormExpr& q,
    const FE::forms::FormExpr& mu,
    const std::function<std::optional<int>(int)>& generated_active_boundary_marker = {},
    std::vector<FE::forms::bc::GeneratedBoundaryNitscheTraceFormBinding>* generated_trace_bindings = nullptr,
    FE::forms::FormExpr* momentum_residual_work_form = nullptr,
    FE::forms::FormExpr* continuity_residual_work_form = nullptr,
    std::vector<PreparedWeakVelocityBoundary>* prepared_boundaries = nullptr)
{
    const auto resolve_boundary = [&](const auto& bc) {
        const int marker = FE::forms::bc::detail::boundaryMarkerOrThrow(
            bc, "navier_stokes::Factories::applyVelocityNitscheBCs");
        const auto generated_marker = generated_active_boundary_marker
            ? generated_active_boundary_marker(marker) : std::nullopt;
        for (int d = 0; d < dim; ++d) {
            if (!bc.active_components[static_cast<std::size_t>(d)]) {
                throw std::invalid_argument(
                    "applyVelocityNitscheBCs: partial-component weak velocity Dirichlet is not supported");
            }
        }
        auto uD_comp = FE::forms::bc::toVectorExpr(
            bc.value, dim, "ns_uD", marker, FE::forms::bc::ComponentValueNameStyle::Component);
        return PreparedWeakVelocityBoundary{
            .physical_marker = marker,
            .generated_marker = generated_marker,
            .prescribed_velocity = FE::forms::FormExpr::asVector(std::move(uD_comp))};
    };
    detail::applyVelocityNitscheBoundaries(
        momentum_form, continuity_form, options, options.velocity_dirichlet_weak, resolve_boundary,
        u, p, v, q, mu, generated_trace_bindings, momentum_residual_work_form,
        continuity_residual_work_form, prepared_boundaries);
}

/// Consume inputs prepared in this registration's preflight. The generated
/// viscous route and its immutable binding are rebuilt with the actual fields.
inline void applyVelocityNitscheBCs(
    FE::forms::FormExpr& momentum_form,
    FE::forms::FormExpr& continuity_form,
    const IncompressibleNavierStokesVMSOptions& options,
    std::span<const PreparedWeakVelocityBoundary> prepared_boundaries,
    const FE::forms::FormExpr& u,
    const FE::forms::FormExpr& p,
    const FE::forms::FormExpr& v,
    const FE::forms::FormExpr& q,
    const FE::forms::FormExpr& mu,
    std::vector<FE::forms::bc::GeneratedBoundaryNitscheTraceFormBinding>* generated_trace_bindings = nullptr,
    FE::forms::FormExpr* momentum_residual_work_form = nullptr,
    FE::forms::FormExpr* continuity_residual_work_form = nullptr)
{
    detail::applyVelocityNitscheBoundaries(
        momentum_form, continuity_form, options, prepared_boundaries,
        [](const PreparedWeakVelocityBoundary& boundary) -> const PreparedWeakVelocityBoundary& { return boundary; },
        u, p, v, q, mu, generated_trace_bindings, momentum_residual_work_form,
        continuity_residual_work_form, nullptr);
}

enum class GeneratedTraceCertification { None, Required };

/// Append prepared terms through actual-field rows and return owned trace
/// bindings. Each term is consumed immediately; a late failure retains earlier
/// production and observation appends. Compatibility overloads remain above.
[[nodiscard]] inline std::vector<FE::forms::bc::GeneratedBoundaryNitscheTraceFormBinding>
appendVelocityNitscheBCs(
    FE::forms::ResidualContributions& rows,
    const IncompressibleNavierStokesVMSOptions& options,
    std::span<const PreparedWeakVelocityBoundary> prepared_boundaries,
    const FE::systems::BoundUnknown& velocity,
    const FE::systems::BoundUnknown& pressure,
    const FE::forms::FormExpr& mu,
    GeneratedTraceCertification certification)
{
    std::vector<FE::forms::bc::GeneratedBoundaryNitscheTraceFormBinding> bindings;
    detail::appendVelocityNitscheBoundaries(options, prepared_boundaries,
        [](const PreparedWeakVelocityBoundary& boundary) -> const PreparedWeakVelocityBoundary& { return boundary; },
        velocity.state, pressure.state, velocity.test, pressure.test, mu,
        certification == GeneratedTraceCertification::Required ? &bindings : nullptr,
        [&](const FE::forms::FormExpr& term) { rows.append(velocity.id, term); },
        [&](const FE::forms::FormExpr& term) { rows.append(pressure.id, term); }, nullptr);
    return bindings;
}

// ---------------------------------------------------------------------------
// Free surfaces: the contact-line laws, the pressure anchor, the semi-implicit
// capillary increment, the tangential mesh rows and the contact pin
// ---------------------------------------------------------------------------

/// One PrescribedAngle or DynamicRenE contact line of a free surface.
struct ContactLaw {
    FE::forms::ContactLineDomain domain;
    const FreeSurfaceContactLine* line{nullptr};
    int wall{-1};             ///< wall boundary marker
    bool young{false};        ///< PrescribedAngle: Young wall energy
    bool dynamic{false};      ///< DynamicRenE
    bool strong{false};       ///< DynamicRenE as a contact-point law
    FE::forms::FormExpr n_w{};           ///< wall normal, out of the liquid
    FE::forms::FormExpr cos_e{};         ///< cos(theta_e)
    FE::forms::FormExpr mobility{};      ///< M of the weak Ren-E law
    FE::forms::FormExpr slip_length{};   ///< l_s of the Navier slip; empty for a no-slip wall
    std::array<FE::Real, 3> wall_normal{};           ///< strong law: unit wall normal
    std::function<FE::Real(FE::Real)> contact_speed{};   ///< strong law: cos(theta_d) -> M gamma (cos(theta_e) - cos(theta_d))
};

/// The PrescribedAngle and DynamicRenE contact lines of @p bc, in order.  The
/// options are validated by the preflight.
[[nodiscard]] inline std::vector<ContactLaw> contactLaws(
    const FreeSurfaceBoundary& bc, std::span<const PreparedContactIdentity> contacts, int dim)
{
    std::vector<ContactLaw> laws;
    for (const auto& identity : contacts) {
        const auto& c = bc.contact_lines[identity.contact_index];
        const auto kind = contactLineKind(c);
        if (kind != ContactLineKind::PrescribedAngle && kind != ContactLineKind::DynamicRenE) {
            continue;
        }
        ContactLaw law{.domain = identity.domain};
        law.line = &c;
        law.wall = contactLineWallBoundaryMarker(c);
        law.young = kind == ContactLineKind::PrescribedAngle;
        law.dynamic = kind == ContactLineKind::DynamicRenE;
        law.strong = law.dynamic && std::get<FreeSurfaceContactLine::DynamicRenE>(c.configuration).strong_law;
        law.n_w = wallNormalExpression(c, dim);
        law.cos_e = FE::forms::FormExpr::constant(
            std::cos(constantScalarValueOrThrow(contactLineAngleRadians(c), "contact_angle_radians")));
        if (law.dynamic && !law.strong) {
            law.mobility = FE::forms::bc::toScalarExpr(contactLineMobility(c), freeSurfaceValueName("ns_dynamic_contact_angle_mobility", bc));
        }
        if (contactLineHasNavierSlip(c)) {
            law.slip_length = FE::forms::bc::toScalarExpr(contactLineSlipLength(c), freeSurfaceValueName("ns_contact_wall_slip_length", bc));
        }
        if (law.strong) {
            law.wall_normal = normalizedWallNormal(c);
            const FE::Real mobility =
                constantScalarValueOrThrow(contactLineMobility(c), "Strong DynamicContactAngle mobility");
            const FE::Real gamma =
                constantScalarValueOrThrow(bc.surface_tension, "Strong DynamicContactAngle surface tension");
            const FE::Real cos_e = std::cos(constantScalarValueOrThrow(
                contactLineAngleRadians(c), "Strong DynamicContactAngle equilibrium angle"));
            law.contact_speed = [mobility, gamma, cos_e](FE::Real cos_d) {
                return mobility * gamma * (cos_e - cos_d);
            };
        }
        laws.push_back(std::move(law));
    }
    return laws;
}

/// The anchor of the pressure constant by the natural traction on the
/// unfitted CutVolume free surface @p boundary (ConstraintPlan::anchorGauge): with
/// a natural traction on Gamma_h the pressure constant is fixed by the exterior
/// pressure, provided the generated interface has a positive measure.  The texts
/// are the module's (gauge source, kernel and callback names, messages, and the
/// kernel's manifest label).
[[nodiscard]] inline FE::constraints::InterfaceMeasureGaugeAnchor freeSurfacePressureAnchor(
    FE::FieldId p_id,
    const FreeSurfaceBoundary& boundary)
{
    FE::constraints::InterfaceMeasureGaugeAnchor anchor;
    anchor.field = p_id;
    anchor.interface_marker = boundary.interface_marker;
    anchor.level_set_field_name = boundary.level_set_field_name;
    anchor.domain_id = boundary.generated_interface_domain_id;
    auto& texts = anchor.texts;
    texts.owner = "IncompressibleNavierStokesVMSModule";
    texts.anchor_source =
        "Unfitted CutVolume embedded free-surface natural traction anchors absolute pressure";
    texts.kernel_name = "CutVolumePressureAnchorMeasureGuard";
    texts.callback_name = "navier_stokes_cut_volume_pressure_anchor_measure";
    texts.requirement =
        "CutVolume embedded free-surface pressure anchor requires a positive generated "
        "interface measure before a pressure solve";
    texts.preflight_phase = "embedded_free_surface_measure_preflight";
    texts.preflight_rejection =
        "another communicator rank rejected embedded free-surface measure preflight";
    texts.context_phase = "cut_volume_pressure_anchor_context";
    texts.null_context_accepted = "CutVolume pressure-anchor null context was unexpectedly accepted";
    texts.gauge_policy = "reject_zero_interface_no_dynamic_gauge_insertion";
    // The type label the kernel manifest has always shown for this anchor.
    texts.manifest_type_name =
        "svmp::Physics::formulations::navier_stokes::(anonymous namespace)::CutVolumePressureAnchorMeasureGuard";
    return anchor;
}

// The semi-implicit capillary increment (Surface_tension_semi_implicit=NormalIncrement).
// The term (design note free_surface_semi_implicit_surface_tension_design.md,
// section 3.3) is
//
//     R_SI(u; v) = gamma dt_eff int_{Gamma_h}
//                  grad_Gamma((u - u_ref).n_h) . grad_Gamma(v.n_h) dGamma,
//     grad_Gamma(w.n_h) = P_h (grad w)^T n_h,   P_h = I - n_h (x) n_h,
//
// where dt_eff = 1/a0 is the effective step of the time integrator and
// u_ref is the velocity of the iterate that generated Gamma_h.  The
// application overwrites u_ref at every generated-state refresh, so R_SI is
// zero in every freshly refreshed residual: the accepted state and the
// acceptance test of the outer fixed point are unchanged, and only the
// Jacobian of the frozen inner solves gains the constant, symmetric,
// positive-semidefinite velocity block
//
//     gamma dt_eff int_{Gamma_h} grad_Gamma(du.n_h) . grad_Gamma(v.n_h).
//
// n_h is constant on each LinearCorner facet, so the block is the
// Laplace--Beltrami part of the omitted geometry Jacobian.  The coefficient
// is a physical input times an integrator constant; nothing is tuned.

/// P_h (grad w)^T n_h for a vector field w: the surface gradient of the
/// normal component w.n_h on a facet with constant normal n_h.
[[nodiscard]] inline FE::forms::FormExpr normalComponentSurfaceGradient(
    const FE::forms::FormExpr& w,
    const FE::forms::FormExpr& n)
{
    using namespace FE::forms;
    const auto normal_trace_gradient = transpose(grad(w)) * n;
    return normal_trace_gradient - inner(normal_trace_gradient, n) * n;
}

/// gamma dt_eff P_h (grad u - grad u_ref)^T n_h . P_h (grad v)^T n_h.
/// The velocity difference is formed before the projection so that the
/// integrand is exactly zero where u and u_ref carry identical coefficients.
[[nodiscard]] inline FE::forms::FormExpr semiImplicitNormalIncrementIntegrand(
    const FE::forms::FormExpr& gamma,
    const FE::forms::FormExpr& u,
    const FE::forms::FormExpr& u_ref,
    const FE::forms::FormExpr& v,
    const FE::forms::FormExpr& n)
{
    using namespace FE::forms;
    const auto increment_trace_gradient =
        transpose(grad(u) - grad(u_ref)) * n;
    const auto increment_surface_gradient =
        increment_trace_gradient -
        inner(increment_trace_gradient, n) * n;
    return gamma * FormExpr::effectiveTimeStep() *
           inner(increment_surface_gradient,
                 normalComponentSurfaceGradient(v, n));
}

/**
 * @brief The tangential mesh-policy rows of a fitted free surface on @p psi
 * (empty for Free): SmoothingOnly
 *   weight int_Gamma (P grad d P) : (P grad psi P),   P = I - n x n,
 * or Prescribed
 *   penalty int_Gamma (dt d - w_t) . psi - ((dt d - w_t).n)(psi.n),
 * with the coefficients named for the surface.
 */
[[nodiscard]] inline FE::forms::FormExpr tangentialMeshPolicyResidual(
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
            FE::forms::bc::toScalarExpr(b.tangential_mesh_penalty, freeSurfaceValueName("ns_free_surface_tangential_mesh_smoothing_weight", b));
        return (weight * inner(P * grad(d_mesh) * P, P * grad(psi) * P)).ds(gamma);
    }
    if (b.tangential_mesh_policy == FreeSurfaceTangentialMeshPolicy::Prescribed) {
        const auto target = FormExpr::asVector(bc::toVectorExpr(
            b.prescribed_tangential_mesh_velocity, components, "ns_free_surface_tangential_mesh_velocity",
            b.boundary_marker, bc::ComponentValueNameStyle::Component));
        const auto gap = dt(d_mesh) - target;
        const auto penalty =
            FE::forms::bc::toScalarExpr(b.tangential_mesh_penalty, freeSurfaceValueName("ns_free_surface_tangential_mesh_penalty", b));
        return (penalty * (inner(gap, psi) - normalTrace(gap, n) * normalTrace(psi, n))).ds(gamma);
    }
    return {};
}

/// The pinned contact lines of a fitted free surface: d = 0 on their markers
/// (strong, in the essential phase of the plan).
inline void applyFreeSurfaceContactLineConstraints(
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
        constraint_plan.essentials(constraints);
    }
}

// ---------------------------------------------------------------------------
// The log records of the free-surface terms (FE LogRecord), printed through
// the formulation builder at the points where the terms are stated.
// ---------------------------------------------------------------------------

/// The mode of an unfitted free surface.
[[nodiscard]] inline FE::LogRecord unfittedModeLog(const FreeSurfaceBoundary& bc)
{
    const bool has_dynamic_stress =
        !FE::forms::bc::isZeroConstantScalarValue(bc.external_pressure) ||
        !FE::forms::bc::isZeroConstantScalarValue(bc.surface_tension);
    return post::navierStokesLog("unfitted free-surface boundary mode")
        .diagnostic("unfitted_free_surface_boundary_mode")
        .integer("marker", bc.interface_marker)
        .quoted("level_set_field", bc.level_set_field_name)
        .quoted("generated_interface_domain_id", bc.generated_interface_domain_id)
        .text("generated_interface_geometry",
              freeSurfaceGeneratedInterfaceGeometryName(bc.generated_interface_geometry))
        .text("geometry_tangent_policy", freeSurfaceGeometryTangentPolicyName(bc.geometry_tangent_policy))
        .text("active_domain", activeDomainName(bc.active_domain))
        .text("active_domain_method", activeDomainMethodName(bc.active_domain_method))
        .text("dynamic_stress", has_dynamic_stress ? "enabled" : "natural_zero")
        .text("surface_tension_form", surfaceTensionFormName(bc))
        .text("curvature_policy", freeSurfaceCurvaturePolicyName(bc))
        .text("curvature_tangent_policy", freeSurfaceCurvatureTangentPolicyName(bc))
        .text("kinematic_enforcement", kinematicEnforcementName(bc.kinematic_enforcement))
        .text("cut_cell_stabilization", bc.cut_cell_stabilization.enabled ? "enabled" : "disabled")
        .text("pressure_stabilization_policy",
              pressureStabilizationPolicyName(bc.cut_cell_stabilization.pressure_policy))
        .text("pressure_stabilization", pressureStabilizationActive(bc) ? "active" : "inactive")
        .text("pressure_stabilization_disabled_reason", pressureStabilizationDisabledReason(bc));
}

/// Natural mode: no interface term (sigma n = 0).
[[nodiscard]] inline FE::LogRecord naturalUnfittedSurfaceLog(const FreeSurfaceBoundary& bc)
{
    return post::navierStokesLog("unfitted free surface installs no explicit dI boundary residual",
                                 FE::LogLevel::WARNING)
        .integer("marker", bc.interface_marker)
        .quoted("level_set_field", bc.level_set_field_name)
        .text("dynamic_stress", "natural_zero")
        .text("kinematic_enforcement", "None")
        .diagnostic("unfitted_free_surface_natural_mode");
}

/// The installed traction (SurfaceStress) and the semi-implicit capillary term.
[[nodiscard]] inline std::vector<FE::LogRecord> tractionLogs(const FreeSurfaceBoundary& bc)
{
    std::vector<FE::LogRecord> logs;
    if (!surfaceStressActive(bc)) {
        logs.push_back(post::navierStokesLog("installed pressure-only free-surface stress")
                           .integer("marker", freeSurfaceMarker(bc))
                           .text("surface_tension_form", "SurfaceStress")
                           .text("normal_source", "integration_rule_geometry")
                           .text("form", "p_external_n_dot_v")
                           .text("surface_energy_form", "omitted_literal_zero_gamma")
                           .diagnostic("free_surface_pressure_only_surface_stress"));
    } else {
        logs.push_back(post::navierStokesLog("installed variational free-surface stress")
                           .integer("marker", freeSurfaceMarker(bc))
                           .text("surface_tension_form", "SurfaceStress")
                           .text("normal_source", "integration_rule_geometry")
                           .text("form", "gamma_I_minus_n_outer_n_colon_grad_v")
                           .text("capillary_balance_method", "discrete_energy_volume_stationarity")
                           .text("capillary_balance_qualification", "prerequisite_only")
                           .text("force_projection_applied", "0")
                           .text("static_geometry_stationarity_required", "1")
                           .diagnostic("free_surface_variational_surface_stress"));
    }
    if (usesSemiImplicitNormalIncrement(bc)) {
        logs.push_back(post::navierStokesLog("installed semi-implicit capillary term")
                           .integer("marker", freeSurfaceMarker(bc))
                           .text("surface_tension_semi_implicit",
                                 freeSurfaceSurfaceTensionSemiImplicitName(bc.surface_tension_semi_implicit))
                           .text("surface_tension_form", surfaceTensionFormName(bc))
                           .text("form", "gamma_dt_eff_grad_gamma_du_dot_n_dot_grad_gamma_v_dot_n")
                           .text("reference_velocity_field", kFreeSurfaceSemiImplicitReferenceVelocityFieldName)
                           .text("reference_refresh", "every_generated_state_refresh")
                           .text("fresh_residual_contribution", "0")
                           .text("coefficient", "surface_tension_times_effective_time_step")
                           .text("qualification", "Candidate")
                           .diagnostic("free_surface_semi_implicit_normal_increment"));
    }
    return logs;
}

/// The terms of one contact line (@p active_wall_marker: the wetted wall, or -1).
[[nodiscard]] inline std::vector<FE::LogRecord> contactLineLogs(const FreeSurfaceBoundary& bc,
                                                                const FreeSurfaceContactLine& contact_line,
                                                                int contact_marker, int active_wall_marker)
{
    constexpr std::string_view kContactForce = "surface_conormal_plus_equilibrium_wall_energy";
    std::vector<FE::LogRecord> logs;
    if (contactLineKind(contact_line) == ContactLineKind::PrescribedAngle) {
        if (FE::forms::bc::isZeroConstantScalarValue(bc.surface_tension)) {
            return logs;
        }
        logs.push_back(post::navierStokesLog("installed prescribed-angle wall-energy momentum term")
                           .integer("interface_marker", bc.interface_marker)
                           .integer("contact_marker", contact_marker)
                           .text("surface_tension_form", surfaceTensionFormName(bc))
                           .text("contact_force_discretization", kContactForce)
                           .diagnostic("navier_stokes_prescribed_contact_wall_energy"));
        if (contactLineHasNavierSlip(contact_line)) {
            logs.push_back(post::navierStokesLog("installed prescribed-angle sharp Navier wall slip")
                               .integer("interface_marker", bc.interface_marker)
                               .integer("wall_marker", contactLineWallBoundaryMarker(contact_line))
                               .integer("active_wall_marker", active_wall_marker)
                               .text("wetted_wall_domain", "sharp_generated_active_boundary")
                               .text("line_friction", "absent")
                               .diagnostic("navier_stokes_prescribed_contact_navier_slip"));
        }
        return logs;
    }
    if (std::get<FreeSurfaceContactLine::DynamicRenE>(contact_line.configuration).strong_law) {
        logs.push_back(post::navierStokesLog("DynamicContactAngle uses the strong contact-point law")
                           .integer("contact_marker", contact_marker)
                           .text("contact_line_law_enforcement", "Strong")
                           .text("line_friction_and_young_term", "replaced_by_constraint_reaction")
                           .diagnostic("navier_stokes_dynamic_contact_strong_law"));
    }
    logs.push_back(post::navierStokesLog("installed coupled dissipative Ren--E DynamicContactAngle")
                       .integer("interface_marker", bc.interface_marker)
                       .integer("contact_marker", contact_marker)
                       .integer("wall_marker", contactLineWallBoundaryMarker(contact_line))
                       .text("convention", "theta_through_liquid_n_outward_liquid_nw_fluid_to_solid")
                       .text("law", "xi_Vcl_equals_gamma_cos_thetae_minus_cos_thetad")
                       .text("surface_tension_form", surfaceTensionFormName(bc))
                       .text("contact_force_discretization", kContactForce)
                       .integer("active_wall_marker", active_wall_marker)
                       .text("wetted_wall_domain", "sharp_generated_active_boundary")
                       .diagnostic("navier_stokes_dynamic_contact_angle"));
    return logs;
}

/// The strong Ren-E contact-point constraint.
[[nodiscard]] inline FE::LogRecord strongContactPointLog(const FreeSurfaceContactLine& contact_line)
{
    const FE::Real mobility =
        constantScalarValueOrThrow(contactLineMobility(contact_line), "Strong DynamicContactAngle mobility");
    return post::navierStokesLog("registered the strong Ren-E contact-point constraint")
        .integer("wall_marker", contactLineWallBoundaryMarker(contact_line))
        .real("mobility", mobility)
        .text("registration_order", "after_small_cut_aggregation")
        .text("value_lag", "frozen_per_outer_geometry_pass")
        .diagnostic("navier_stokes_strong_contact_point_law");
}

/// The cut-cell stabilization of an unfitted free surface (the velocity and
/// pressure orders), and the warning for pressure orders above two.
[[nodiscard]] inline std::vector<FE::LogRecord> cutCellStabilizationLogs(const FreeSurfaceBoundary& bc,
                                                                         int velocity_polynomial_order,
                                                                         int pressure_polynomial_order)
{
    const auto& cut = bc.cut_cell_stabilization;
    constexpr int supported_derivative_order = 2;
    const auto derivative_order_label = [](int max_order) -> const char* {
        if (max_order <= 0) {
            return "disabled";
        }
        return max_order > 1 ? "1,2" : "1";
    };
    const bool pressure_stabilization_enabled = pressureStabilizationActive(bc);
    const bool pressure_stabilization_incremental =
        pressure_stabilization_enabled && cut.pressure_policy == FreeSurfacePressureStabilizationPolicy::Incremental;
    const int pressure_derivative_order =
        pressure_stabilization_enabled
            ? (pressure_stabilization_incremental ? 1 : (pressure_polynomial_order > 1 ? supported_derivative_order : 1))
            : 0;
    const int max_derivative_order = pressure_derivative_order;
    const bool has_unsupported_derivative_order = pressure_stabilization_enabled &&
                                                  !pressure_stabilization_incremental &&
                                                  pressure_polynomial_order > supported_derivative_order;
    constexpr FE::Real kCutPenaltyTransientCalibration{0.01};
    const auto interface_side =
        bc.active_domain == FreeSurfaceActiveDomain::LevelSetPositive
            ? "Plus"
            : (bc.active_domain == FreeSurfaceActiveDomain::LevelSetNegative ? "Minus" : "All");
    const auto active_domain_side =
        bc.active_domain == FreeSurfaceActiveDomain::None ? "FullDomain"
                                                          : cutVolumeSideName(activeDomainSide(bc.active_domain));

    std::vector<FE::LogRecord> logs;
    auto log = post::navierStokesLog("cut-cell stabilization");
    log.diagnostic("cut_cell_stabilization")
        .integer("marker", bc.interface_marker)
        .quoted("level_set_field", bc.level_set_field_name)
        .text("interface_side", interface_side)
        .text("active_domain", activeDomainName(bc.active_domain))
        .text("active_domain_side", active_domain_side)
        .text("active_domain_method", activeDomainMethodName(bc.active_domain_method))
        .text("generated_interface_geometry",
              freeSurfaceGeneratedInterfaceGeometryName(bc.generated_interface_geometry))
        .text("geometry_tangent_policy", freeSurfaceGeometryTangentPolicyName(bc.geometry_tangent_policy))
        .text("use_cut_metadata_scale", cut.use_cut_metadata_scale ? "true" : "false");
    if (cut.cut_metadata_scale_cap.has_value()) {
        log.real("cut_metadata_scale_cap", *cut.cut_metadata_scale_cap);
    } else {
        log.text("cut_metadata_scale_cap", "unbounded");
    }
    log.text("pressure_stabilization_policy", pressureStabilizationPolicyName(cut.pressure_policy))
        .text("pressure_stabilization", pressure_stabilization_enabled ? "enabled" : "disabled")
        .text("pressure_stabilization_form",
              pressure_stabilization_enabled ? (pressure_stabilization_incremental ? "incremental" : "absolute")
                                             : "disabled")
        .text("pressure_disabled_reason", pressureStabilizationDisabledReason(bc))
        .text("facet_scope", "cut-adjacent")
        .integer("velocity_polynomial_order", velocity_polynomial_order)
        .integer("pressure_polynomial_order", pressure_polynomial_order)
        .text("derivative_orders", derivative_order_label(max_derivative_order))
        .text("pressure_derivative_orders", derivative_order_label(pressure_derivative_order))
        .real("cut_penalty_transient_calibration", kCutPenaltyTransientCalibration)
        .text("small_cut_aggregation", bc.small_cut_aggregation ? "true" : "false")
        .text("velocity_ghost_penalty_mode", "retired_replaced_by_aggregation")
        .text("pressure_scaling", pressure_derivative_order <= 0
                                      ? "disabled"
                                      : (pressure_derivative_order > 1
                                             ? "0.01*h^3/(mu+rho*h^2/dt),0.01*h^5/(mu+rho*h^2/dt)"
                                             : "0.01*h^3/(mu+rho*h^2/dt)"));
    logs.push_back(std::move(log));
    if (has_unsupported_derivative_order) {
        logs.push_back(post::navierStokesLog("high-order cut-cell stabilization supports derivative orders 1 and 2 only",
                                             FE::LogLevel::WARNING)
                           .diagnostic("cut_cell_stabilization_order_unsupported")
                           .integer("supported_derivative_order", supported_derivative_order));
    }
    return logs;
}

/// Registration order of the inactive-side and aggregation constraints
/// (@p liquid_boundary null: none are registered).
[[nodiscard]] inline std::vector<FE::LogRecord> activeDomainConstraintLogs(const FreeSurfaceBoundary* liquid_boundary,
                                                                           bool small_cut_aggregation)
{
    std::vector<FE::LogRecord> logs;
    if (liquid_boundary == nullptr) {
        return logs;
    }
    // Physical essential data owns the exterior trace.  The artificial
    // inactive-side extension is registered afterward so its existing
    // per-DOF precedence rule fills only otherwise unconstrained support.
    // Aggregation remains last and therefore sees the complete physical and
    // inactive constraint set.
    logs.push_back(post::navierStokesLog("active-domain inactive constraints enabled")
                       .diagnostic("active_domain_inactive_constraints")
                       .integer("marker", liquid_boundary->interface_marker)
                       .text("registration_order", "after_physical_essential_constraints")
                       .text("physical_trace_precedence", "per_dof"));
    if (!small_cut_aggregation) {
        return logs;
    }
    // Mixed problems must aggregate both spaces: leaving the small-cut
    // pressures free while their conjugate velocities are slaved breaks the
    // local saddle point.  Registration after all essential constraints is
    // deliberate. SmallCutAggregationConstraint gathers preconstrained
    // component facts communicator-wide before installing canonical lines,
    // so even owner/non-owner boundary visibility cannot turn a partial
    // strong condition into an owner-wins aggregation line.
    const auto& guards = liquid_boundary->small_cut_aggregation_guards;
    logs.push_back(post::navierStokesLog("small-cut aggregation enabled for unfitted free surface")
                       .integer("marker", liquid_boundary->interface_marker)
                       .text("registration_order", "after_strong_constraints")
                       .text("component_precedence", "per_dof")
                       .integer("maximum_root_path_length", guards.maximum_root_path_length)
                       .real("maximum_reference_extrapolation_distance",
                             guards.maximum_reference_extrapolation_distance)
                       .real("maximum_absolute_coefficient", guards.maximum_absolute_coefficient)
                       .real("maximum_row_l1_norm", guards.maximum_row_l1_norm)
                       .text("rootless_fallback", guards.rootless_fallback ? "1" : "0")
                       .diagnostic("small_cut_aggregation_registration")
                       .text("velocity_ghost_penalty", "skipped_by_aggregation"));
    return logs;
}

// ---------------------------------------------------------------------------
// Unfitted free surface Gamma_h = {phi_h = c}, bounding the liquid side L
// ---------------------------------------------------------------------------
//   - gamma cos(theta_e) int_CL v.m                    Young wall energy on the contact line CL = Gamma_h ^ wall,
//   + int_CL (xi u.m - gamma cos(theta_e)) v.m         or the weak Ren-E law, friction xi = 1/M,
//     or u.t_w = M gamma (cos(theta_e) - cos(theta_d))  the strong Ren-E law at the contact point (constraint);
//   + int_W (mu/l_s) u_t . v_t                         Navier slip on the wetted wall W_h;
//   + int_Gamma p_ext n.v + gamma (I - n x n) : grad v  the traction (SurfaceStress, the only form, D48);
//   + gamma dt int_Gamma grad_G((u - u_ref).n) . grad_G(v.n)   semi-implicit capillary term (D13).
// n points out of the liquid, n_w out of the liquid into the wall, m = conormal(n, n_w) is wall-tangential
// and points out of the wetted footprint.  u.n_w = 0 holds strongly on the walls, the dry side is pinned and
// small cuts are aggregated (registerOn), and phi is transported by the level-set equation (the kinematic
// condition).  gamma is a literal (validated); SurfaceStress already contains gamma cos(theta_d) in its
// conormal, so its wall term keeps only cos(theta_e).
/// The SurfaceStress integrands, shared with the diagnostic channels. The
/// caller supplies the integration-frame normal and applies the measure.
[[nodiscard]] inline FE::forms::FormExpr exteriorPressureIntegrand(
    const FE::forms::FormExpr& external_pressure,
    const FE::forms::FormExpr& normal,
    const FE::forms::FormExpr& velocity_test)
{
    return external_pressure * FE::forms::inner(normal, velocity_test);
}

[[nodiscard]] inline FE::forms::FormExpr surfaceEnergyIntegrand(
    const FE::forms::FormExpr& surface_tension,
    const FE::forms::FormExpr& normal,
    const FE::forms::FormExpr& velocity_test,
    int dimension)
{
    return surface_tension * FE::forms::inner(
        FE::forms::tangentialProjector(normal, dimension), FE::forms::grad(velocity_test));
}

inline void addUnfittedFreeSurface(FE::forms::FormExpr& M, FE::systems::FESystem& system,
                                   FE::constraints::ConstraintPlan& plan, FE::systems::FormulationBuilder& F,
                                   const IncompressibleNavierStokesVMSOptions& o, const FreeSurfaceBoundary& b,
                                   PreparedSurfaceIdentity& identity, const NavierStokesTerms& t, FE::FieldId u_id,
                                   FE::FieldId reference_velocity_id, post::ConservativeBalanceForms* balance)
{
    using namespace FE::forms;
    const auto& [u, p, v, q, rho, mu, f, dim] = t;
    const auto L = identity.geometry.value().side(activeDomainSide(b.active_domain));
    const auto gamma = FE::forms::bc::toScalarExpr(b.surface_tension, freeSurfaceValueName("ns_free_surface_surface_tension", b));

    registerFreeSurfacePrescribedAngleGeometry(system, b, identity);   // contact-geometry certificate at each refresh
    for (const auto& c : Factories::contactLaws(b, identity.contacts, dim)) {
        const auto CL = F.registered(c.domain);
        if (c.dynamic) {
            registerDynamicContactWallNormalCheck(system, *c.line, CL.marker());
        }
        const auto n = L.outwardNormal();   // the normal of the generated rule
        const auto m = conormal(n, c.n_w);
        if (c.young && !bc::isZeroConstantScalarValue(b.surface_tension)) {
            const auto young = (-gamma * c.cos_e * dot(v, m)).dL(CL);
            M = M + young;
            post::addSurfaceEnergy(balance, young);
        }
        if (c.dynamic && !c.strong) {
            const auto friction = FormExpr::constant(1.0) / c.mobility * dot(u, m);
            M = M + ((friction - gamma * c.cos_e) * dot(v, m)).dL(CL);
            post::addSurfaceEnergy(balance, (-gamma * c.cos_e * dot(v, m)).dL(CL));
        }
        int wetted_marker = -1;
        if (c.slip_length.isValid()) {
            const auto W = F.registered(L.wetted(c.wall));
            M = M + (mu / c.slip_length * dot(tangentialProjection(u, c.n_w), tangentialProjection(v, c.n_w))).ds(W);
            wetted_marker = W.generatedActiveBoundaryMarker();
        }
        F.log(contactLineLogs(b, *c.line, CL.marker(), wetted_marker));
        if (c.strong) {   // PointLaw phase: registered after the aggregation, so it can expand aggregated DOFs
            plan.contactPointVelocity(u_id, L.geometry(), c.wall, c.wall_normal, c.contact_speed,
                                      "ren_e_contact_point:" + std::to_string(c.wall));
            F.log(strongContactPointLog(*c.line));
        }
    }

    validateFreeSurfaceBoundary(b, o, o.enable_ale, system, dim, identity);
    F.log(unfittedModeLog(b));
    if (bc::isZeroConstantScalarValue(b.external_pressure) && bc::isZeroConstantScalarValue(b.surface_tension)) {
        F.log(naturalUnfittedSurfaceLog(b));   // sigma n = 0: no interface term
        return;
    }
    const auto Gamma = L.interface();
    const auto n = L.outwardNormal();
    const auto p_ext = FE::forms::bc::toScalarExpr(b.external_pressure, freeSurfaceValueName("ns_free_surface_external_pressure", b));
    const auto pressure = exteriorPressureIntegrand(p_ext, n, v);   // SurfaceStress, the only form (D48)
    post::addExteriorPressure(balance, pressure.dI(Gamma));
    if (!surfaceStressActive(b)) {   // gamma = 0
        M = M + pressure.dI(Gamma);
    } else {
        const auto surface_energy = surfaceEnergyIntegrand(gamma, n, v, dim);
        post::addSurfaceEnergy(balance, surface_energy.dI(Gamma));
        M = M + (pressure + surface_energy).dI(Gamma);
    }
    if (usesSemiImplicitNormalIncrement(b)) {
        const auto& reference = system.fieldRecord(reference_velocity_id);
        const auto u_ref = FormExpr::discreteField(reference_velocity_id, *reference.space,
            std::string(kFreeSurfaceSemiImplicitReferenceVelocityFieldName));   // u at the last geometry refresh
        M = M + Factories::semiImplicitNormalIncrementIntegrand(gamma, u, u_ref, v, n).dI(Gamma);
    }
    F.log(tractionLogs(b));
}

// ---------------------------------------------------------------------------
// Fitted ALE free surface Gamma(t) = a boundary marker of the moving mesh
// ---------------------------------------------------------------------------
//   fluid rows: int_Gamma p_ext n.v + gamma (I - n x n) : grad v      (SurfaceStress);
//   mesh rows:  int_Gamma (gamma_N/h_n) ((dt d - u).n)(psi.n)            (MeshNitsche, D35),
//               + the tangential mesh policy (smoothing, or a prescribed tangential rate);
//   a pinned contact line: d = 0 (strong).
inline void addFittedFreeSurface(FE::forms::FormExpr& M, FE::systems::FESystem& system,
                                 FE::constraints::ConstraintPlan& plan, FE::systems::BoundaryConditionManager& bcs,
                                 const IncompressibleNavierStokesVMSOptions& o, const FreeSurfaceBoundary& b,
                                 PreparedSurfaceIdentity& identity, const FE::systems::ALEBinding& ale,
                                 const NavierStokesTerms& t, FE::FieldId u_id,
                                 FE::systems::FormulationBuilder& F, post::ConservativeBalanceForms* balance)
{
    using namespace FE::forms;
    namespace fs = FE::systems;
    const auto& [u, p, v, q, rho, mu, f, dim] = t;
    Factories::applyFreeSurfaceContactLineConstraints(plan, b, ale, dim);   // pinned contact lines: d = 0
    bcs.add(std::make_unique<bc::ReservedBC>(b.boundary_marker));

    validateFreeSurfaceBoundary(b, o, o.enable_ale, system, dim, identity);   // the qualified fitted-ALE contract
    const auto Gamma = BoundaryDomain::fullPhysical(b.boundary_marker);
    const auto n = currentNormal();   // normal of the integration frame
    const auto gamma = FE::forms::bc::toScalarExpr(b.surface_tension, freeSurfaceValueName("ns_free_surface_surface_tension", b));
    const auto p_ext = FE::forms::bc::toScalarExpr(b.external_pressure, freeSurfaceValueName("ns_free_surface_external_pressure", b));
    const bool dynamic_stress =
        !bc::isZeroConstantScalarValue(b.external_pressure) || !bc::isZeroConstantScalarValue(b.surface_tension);
    if (dynamic_stress) {   // SurfaceStress, the only fitted form (validated)
        const auto pressure = exteriorPressureIntegrand(p_ext, n, v);
        post::addExteriorPressure(balance, pressure.ds(Gamma));
        if (!surfaceStressActive(b)) {   // gamma = 0
            M = M + pressure.ds(Gamma);
        } else {
            const auto surface_energy = surfaceEnergyIntegrand(gamma, n, v, dim);
            post::addSurfaceEnergy(balance, surface_energy.ds(Gamma));
            M = M + (pressure + surface_energy).ds(Gamma);
        }
        F.log(tractionLogs(b));
    }

    // Coupled ALE, MeshNitsche and MatchFluidNormalVelocity (validated).
    const auto& rec = fittedMeshDisplacementRecord(system, ale);   // throws without a displacement unknown
    declareFittedTangentialMeshPolicy(system, ale, b);
    const auto psi = TestField(ale.mesh_displacement_field, *rec.space, "psi_free_surface_mesh");
    const auto d = StateField(ale.mesh_displacement_field, *rec.space, "d_mesh_free_surface");
    const auto nc = currentNormal();
    // (dt d - u).n = 0 on the mesh rows
    const auto penalty = FormExpr::constant(b.kinematic_nitsche_gamma) / hNormal();
    FormExpr rows = (penalty * normalTrace(dt(d) - u, nc) * normalTrace(psi, nc)).ds(Gamma);
    if (const auto tangential = Factories::tangentialMeshPolicyResidual(b, d, psi, nc, rec.components, Gamma);
        tangential.isValid()) {
        rows = sumOf(rows, tangential);
    }
    fs::FormulationGroup group{.ale = &ale};   // installed before the fluid residual
    group.extra_trial_fields.push_back(u_id);
    (void)F.installGroup({ale.mesh_displacement_field}, rows, group);
    declareFittedMeshRowDescriptors(system, o.operator_tag, ale, b, u_id, /*rows_installed=*/true);
}

// Pressure gradient-jump penalty on the facets next to the cut (Burman; Schott & Wall):
//   sum_F s gamma_p 0.01 h^3/(mu + rho h^2/dt) [grad p].[grad q]   (+ h^5 [d_nn p][d_nn q] for P2 pressure)
[[nodiscard]] inline FE::forms::FormExpr cutFacetPressurePenalty(const FreeSurfaceBoundary& b,
                                                                 const NavierStokesTerms& t,
                                                                 const IncompressibleNavierStokesVMSOptions& o,
                                                                 int velocity_order, int pressure_order,
                                                                 FE::systems::FormulationBuilder& formulation)
{
    using namespace FE::forms;
    const auto& cut = b.cut_cell_stabilization;
    if (!isUnfittedLevelSet(b) || !cut.enabled) {
        return {};
    }
    formulation.log(cutCellStabilizationLogs(b, velocity_order, pressure_order));
    if (!pressureStabilizationActive(b)) {
        return {};
    }
    const bool incremental = cut.pressure_policy == FreeSurfacePressureStabilizationPolicy::Incremental;
    const auto pp = incremental ? FormExpr::effectiveTimeStep() * dt(t.p) : t.p;
    auto s = cut.use_cut_metadata_scale ? cutStabilizationScale() : FormExpr::constant(1.0);
    if (cut.use_cut_metadata_scale && cut.cut_metadata_scale_cap.has_value()) {
        s = min(s, FormExpr::constant(*cut.cut_metadata_scale_cap));
    }
    const auto h = avg(hNormal());
    const auto h3 = h * h * h;
    const auto mu_gp = t.mu + FormExpr::constant(o.density) * h * h / FormExpr::effectiveTimeStep();
    const auto gamma_p = FE::forms::bc::toScalarExpr(cut.pressure_gradient_penalty, freeSurfaceValueName("ns_free_surface_cut_pressure_penalty", b));
    const auto c = FormExpr::constant(0.01);
    const auto F = b.active_domain == FreeSurfaceActiveDomain::None
                       ? FacetDomain::cutAdjacent(b.interface_marker)
                       : FacetDomain::cutAdjacent(b.interface_marker, activeDomainSide(b.active_domain));
    auto J = (s * gamma_p * c * h3 / (mu_gp + FormExpr::constant(o.stabilization_epsilon)) *
              inner(cutAdjacentFacetGradientJump(pp), cutAdjacentFacetGradientJump(t.q))).dS(F);
    if (!incremental && pressure_order > 1) {
        J = J + (s * gamma_p * c * (h3 * h * h) / (mu_gp + FormExpr::constant(o.stabilization_epsilon)) *
                 cutAdjacentFacetSecondNormalDerivativeJump(pp) *
                 cutAdjacentFacetSecondNormalDerivativeJump(t.q)).dS(F);
    }
    return J;
}

// ---------------------------------------------------------------------------
// Two-fluid material interface: weighted symmetric Nitsche coupling of the
// two phases on the generated interface (was IncompressibleTwoFluidInterface)
// ---------------------------------------------------------------------------

/// The unknowns and test rows of one incompressible phase.
struct IncompressiblePhaseSymbols {
  FE::forms::FormExpr velocity;
  FE::forms::FormExpr pressure;
  FE::forms::FormExpr velocity_test;
  FE::forms::FormExpr pressure_test;
};

struct IncompressibleTwoFluidInterfaceParameters {
  int dimension{0};
  int interface_marker{-1};
  FE::Real negative_density{1.0};
  FE::Real positive_density{1.0};
  FE::Real negative_viscosity{1.0};
  FE::Real positive_viscosity{1.0};
  FE::Real nitsche_gamma{20.0};
  FE::Real surface_tension{0.0};
  bool include_transient_penalty{true};
  std::optional<FE::Real> prescribed_pressure_jump{};
  /** Expected (tau_minus-tau_plus)n in global physical components. */
  std::optional<std::array<FE::Real, 3>>
      prescribed_viscous_traction_jump{};
};

struct IncompressibleTwoFluidInterfaceWeights {
  FE::Real negative_traction{0.5};
  FE::Real positive_traction{0.5};
  FE::Real negative_complement{0.5};
  FE::Real positive_complement{0.5};
  FE::Real harmonic_viscosity{1.0};
  FE::Real harmonic_density{1.0};
};

/**
 * Each returned channel is already integrated on the generated interface.
 * Keeping the channels separate permits accepted-stage work and consistency
 * diagnostics to use the identical production expressions.
 */
struct IncompressibleTwoFluidInterfaceForms {
  IncompressibleTwoFluidInterfaceWeights weights{};
  FE::forms::FormExpr consistency{};
  FE::forms::FormExpr adjoint{};
  FE::forms::FormExpr penalty{};
  FE::forms::FormExpr surface_energy{};
  FE::forms::FormExpr prescribed_pressure_jump{};
  FE::forms::FormExpr prescribed_viscous_traction_jump{};
  FE::forms::FormExpr residual{};
};

namespace detail {

inline void validateTwoFluidInterfaceParameters(
    const IncompressibleTwoFluidInterfaceParameters &parameters) {
  if (parameters.dimension != 2 && parameters.dimension != 3) {
    throw std::invalid_argument(
        "incompressible two-fluid interface requires dimension 2 or 3");
  }
  if (parameters.interface_marker < 0) {
    throw std::invalid_argument(
        "incompressible two-fluid interface marker must be nonnegative");
  }
  const auto finite_positive = [](FE::Real value) {
    return FE::numeric::isFinite(value) && FE::numeric::isPositive(value);
  };
  if (!finite_positive(parameters.negative_density) ||
      !finite_positive(parameters.positive_density)) {
    throw std::invalid_argument("incompressible two-fluid interface densities "
                                "must be finite and positive");
  }
  if (!finite_positive(parameters.negative_viscosity) ||
      !finite_positive(parameters.positive_viscosity)) {
    throw std::invalid_argument("incompressible two-fluid interface "
                                "viscosities must be finite and positive");
  }
  if (!finite_positive(parameters.nitsche_gamma)) {
    throw std::invalid_argument("incompressible two-fluid interface Nitsche "
                                "gamma must be finite and positive");
  }
  if (!FE::numeric::isFinite(parameters.surface_tension) ||
      parameters.surface_tension < FE::Real{0.0}) {
    throw std::invalid_argument("incompressible two-fluid interface surface "
                                "tension must be finite and nonnegative");
  }
  if (parameters.prescribed_pressure_jump.has_value() &&
      !FE::numeric::isFinite(*parameters.prescribed_pressure_jump)) {
    throw std::invalid_argument(
        "incompressible two-fluid prescribed pressure jump must be finite");
  }
  if (parameters.prescribed_viscous_traction_jump.has_value()) {
    const auto& target = *parameters.prescribed_viscous_traction_jump;
    if (!std::all_of(target.begin(), target.end(), [](FE::Real value) {
          return FE::numeric::isFinite(value);
        })) {
      throw std::invalid_argument(
          "incompressible two-fluid prescribed viscous traction jump must be finite");
    }
    for (int component = parameters.dimension; component < 3; ++component) {
      if (target[static_cast<std::size_t>(component)] != FE::Real{0.0}) {
        throw std::invalid_argument(
            "incompressible two-fluid prescribed viscous traction jump has an out-of-plane component");
      }
    }
  }
}

inline void validateTwoFluidInterfaceExpression(const FE::forms::FormExpr &expression,
                        const char *name) {
  if (!expression.isValid()) {
    throw std::invalid_argument(
        std::string("incompressible two-fluid interface requires ") + name);
  }
}

struct StablePositivePair {
  FE::Real first_fraction{0.5};
  FE::Real second_fraction{0.5};
  FE::Real harmonic_mean{1.0};
};

[[nodiscard]] inline StablePositivePair stablePositivePair(FE::Real first,
                                                    FE::Real second) {
  const auto maximum = std::max(first, second);
  const auto first_scaled = first / maximum;
  const auto second_scaled = second / maximum;
  const auto scaled_sum = first_scaled + second_scaled;
  const auto minimum = std::min(first, second);
  return StablePositivePair{
      .first_fraction = first_scaled / scaled_sum,
      .second_fraction = second_scaled / scaled_sum,
      .harmonic_mean =
          minimum * (FE::Real{2.0} / (FE::Real{1.0} + minimum / maximum)),
  };
}

} // namespace detail

[[nodiscard]] inline IncompressibleTwoFluidInterfaceWeights incompressibleTwoFluidInterfaceWeights(
    const IncompressibleTwoFluidInterfaceParameters &parameters) {
  detail::validateTwoFluidInterfaceParameters(parameters);

  const auto viscosity_pair = detail::stablePositivePair(parameters.negative_viscosity,
                                                 parameters.positive_viscosity);
  const auto density_pair = detail::stablePositivePair(parameters.negative_density,
                                               parameters.positive_density);
  return IncompressibleTwoFluidInterfaceWeights{
      .negative_traction = viscosity_pair.second_fraction,
      .positive_traction = viscosity_pair.first_fraction,
      .negative_complement = viscosity_pair.first_fraction,
      .positive_complement = viscosity_pair.second_fraction,
      .harmonic_viscosity = viscosity_pair.harmonic_mean,
      .harmonic_density = density_pair.harmonic_mean,
  };
}

[[nodiscard]] inline IncompressibleTwoFluidInterfaceForms buildIncompressibleTwoFluidInterfaceForms(
    const IncompressiblePhaseSymbols& negative,
    const IncompressiblePhaseSymbols& positive,
    const IncompressibleTwoFluidInterfaceParameters &parameters) {
  const auto& [negative_velocity, negative_pressure, negative_velocity_test, negative_pressure_test] = negative;
  const auto& [positive_velocity, positive_pressure, positive_velocity_test, positive_pressure_test] = positive;
  detail::validateTwoFluidInterfaceParameters(parameters);
  detail::validateTwoFluidInterfaceExpression(negative_velocity, "negative velocity");
  detail::validateTwoFluidInterfaceExpression(negative_pressure, "negative pressure");
  detail::validateTwoFluidInterfaceExpression(negative_velocity_test, "negative velocity test");
  detail::validateTwoFluidInterfaceExpression(negative_pressure_test, "negative pressure test");
  detail::validateTwoFluidInterfaceExpression(positive_velocity, "positive velocity");
  detail::validateTwoFluidInterfaceExpression(positive_pressure, "positive pressure");
  detail::validateTwoFluidInterfaceExpression(positive_velocity_test, "positive velocity test");
  detail::validateTwoFluidInterfaceExpression(positive_pressure_test, "positive pressure test");

  using namespace FE::forms;

  const auto weights = incompressibleTwoFluidInterfaceWeights(parameters);
  const auto n = FormExpr::normal();
  const auto identity = FormExpr::identity(parameters.dimension);
  const auto two = FormExpr::constant(FE::Real{2.0});
  const auto mu_negative = FormExpr::constant(parameters.negative_viscosity);
  const auto mu_positive = FormExpr::constant(parameters.positive_viscosity);
  const auto negative_viscous_traction =
      (two * mu_negative * symgrad(negative_velocity)) * n;
  const auto negative_pressure_traction = (-negative_pressure * identity) * n;
  const auto positive_viscous_traction =
      (two * mu_positive * symgrad(positive_velocity)) * n;
  const auto positive_pressure_traction = (-positive_pressure * identity) * n;
  const auto negative_viscous_test_traction =
      (two * mu_negative * symgrad(negative_velocity_test)) * n;
  const auto negative_pressure_test_traction =
      (-negative_pressure_test * identity) * n;
  const auto positive_viscous_test_traction =
      (two * mu_positive * symgrad(positive_velocity_test)) * n;
  const auto positive_pressure_test_traction =
      (-positive_pressure_test * identity) * n;

  const auto negative_weight = FormExpr::constant(weights.negative_traction);
  const auto positive_weight = FormExpr::constant(weights.positive_traction);

  auto penalty_scale =
      FormExpr::constant(weights.harmonic_viscosity) / hNormal();
  if (parameters.include_transient_penalty) {
    penalty_scale = penalty_scale +
                    FormExpr::constant(weights.harmonic_density) * hNormal() /
                        FormExpr::effectiveTimeStep();
  }
  penalty_scale = FormExpr::constant(parameters.nitsche_gamma) * penalty_scale;

  IncompressibleTwoFluidInterfaceForms forms;
  forms.weights = weights;
  // Keep every generated-interface integral block-separable: each term has
  // exactly one test and at most one trial function. This is algebraically
  // identical to -<{sigma(u,p)n}_w, [v]> and lets the mixed compiler retain
  // all four velocity/pressure fields without an opaque multi-field term.
  forms.consistency = (-negative_weight *
                       inner(negative_viscous_traction, negative_velocity_test))
                          .dI(parameters.interface_marker) +
                      (-negative_weight * inner(negative_pressure_traction,
                                                negative_velocity_test))
                          .dI(parameters.interface_marker) +
                      (negative_weight *
                       inner(negative_viscous_traction, positive_velocity_test))
                          .dI(parameters.interface_marker) +
                      (negative_weight * inner(negative_pressure_traction,
                                               positive_velocity_test))
                          .dI(parameters.interface_marker) +
                      (-positive_weight *
                       inner(positive_viscous_traction, negative_velocity_test))
                          .dI(parameters.interface_marker) +
                      (-positive_weight * inner(positive_pressure_traction,
                                                negative_velocity_test))
                          .dI(parameters.interface_marker) +
                      (positive_weight *
                       inner(positive_viscous_traction, positive_velocity_test))
                          .dI(parameters.interface_marker) +
                      (positive_weight * inner(positive_pressure_traction,
                                               positive_velocity_test))
                          .dI(parameters.interface_marker);

  // The adjoint channel is the expanded -<{sigma(v,q)n}_w, [u]> term.
  forms.adjoint = (-negative_weight *
                   inner(negative_viscous_test_traction, negative_velocity))
                      .dI(parameters.interface_marker) +
                  (-negative_weight *
                   inner(negative_pressure_test_traction, negative_velocity))
                      .dI(parameters.interface_marker) +
                  (negative_weight *
                   inner(negative_viscous_test_traction, positive_velocity))
                      .dI(parameters.interface_marker) +
                  (negative_weight *
                   inner(negative_pressure_test_traction, positive_velocity))
                      .dI(parameters.interface_marker) +
                  (-positive_weight *
                   inner(positive_viscous_test_traction, negative_velocity))
                      .dI(parameters.interface_marker) +
                  (-positive_weight *
                   inner(positive_pressure_test_traction, negative_velocity))
                      .dI(parameters.interface_marker) +
                  (positive_weight *
                   inner(positive_viscous_test_traction, positive_velocity))
                      .dI(parameters.interface_marker) +
                  (positive_weight *
                   inner(positive_pressure_test_traction, positive_velocity))
                      .dI(parameters.interface_marker);

  forms.penalty =
      (penalty_scale * inner(negative_velocity, negative_velocity_test))
          .dI(parameters.interface_marker) +
      (-penalty_scale * inner(negative_velocity, positive_velocity_test))
          .dI(parameters.interface_marker) +
      (-penalty_scale * inner(positive_velocity, negative_velocity_test))
          .dI(parameters.interface_marker) +
      (penalty_scale * inner(positive_velocity, positive_velocity_test))
          .dI(parameters.interface_marker);
  forms.residual = forms.consistency + forms.adjoint + forms.penalty;

  if (parameters.prescribed_pressure_jump.has_value()) {
    const auto target =
        FormExpr::constant(*parameters.prescribed_pressure_jump);
    forms.prescribed_pressure_jump =
        (target * FormExpr::constant(weights.negative_complement) *
         inner(n, negative_velocity_test))
            .dI(parameters.interface_marker) +
        (target * FormExpr::constant(weights.positive_complement) *
         inner(n, positive_velocity_test))
            .dI(parameters.interface_marker);
    forms.residual = forms.residual + forms.prescribed_pressure_jump;
  }

  if (parameters.prescribed_viscous_traction_jump.has_value()) {
    const auto target = constantVector(*parameters.prescribed_viscous_traction_jump, parameters.dimension);
    forms.prescribed_viscous_traction_jump =
        (-FormExpr::constant(weights.negative_complement) *
         inner(target, negative_velocity_test))
            .dI(parameters.interface_marker) +
        (-FormExpr::constant(weights.positive_complement) *
         inner(target, positive_velocity_test))
            .dI(parameters.interface_marker);
    forms.residual =
        forms.residual + forms.prescribed_viscous_traction_jump;
  }

  if (parameters.surface_tension > FE::Real{0.0}) {
    const auto projector = tangentialProjector(n, parameters.dimension);
    const auto surface_tension = FormExpr::constant(parameters.surface_tension);
    forms.surface_energy =
        (surface_tension * FormExpr::constant(weights.negative_complement) *
         inner(projector, grad(negative_velocity_test)))
            .dI(parameters.interface_marker) +
        (surface_tension * FormExpr::constant(weights.positive_complement) *
         inner(projector, grad(positive_velocity_test)))
            .dI(parameters.interface_marker);
    forms.residual = forms.residual + forms.surface_energy;
  }
  return forms;
}

/// Compatibility overload for callers that supply the eight expressions.
[[nodiscard]] inline IncompressibleTwoFluidInterfaceForms buildIncompressibleTwoFluidInterfaceForms(
    const FE::forms::FormExpr& negative_velocity,
    const FE::forms::FormExpr& negative_pressure,
    const FE::forms::FormExpr& negative_velocity_test,
    const FE::forms::FormExpr& negative_pressure_test,
    const FE::forms::FormExpr& positive_velocity,
    const FE::forms::FormExpr& positive_pressure,
    const FE::forms::FormExpr& positive_velocity_test,
    const FE::forms::FormExpr& positive_pressure_test,
    const IncompressibleTwoFluidInterfaceParameters& parameters)
{
  return buildIncompressibleTwoFluidInterfaceForms(
      {negative_velocity, negative_pressure, negative_velocity_test, negative_pressure_test},
      {positive_velocity, positive_pressure, positive_velocity_test, positive_pressure_test},
      parameters);
}

} // namespace Factories
} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_BC_FACTORIES_H
