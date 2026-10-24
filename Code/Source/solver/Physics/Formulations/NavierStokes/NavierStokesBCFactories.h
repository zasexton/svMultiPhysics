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
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceContactCertificate.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceFittedALE.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceGeometry.h"
#include "Physics/Formulations/NavierStokes/NavierStokesPostProcessing.h"
#include "Physics/Options/NavierStokesOptionRules.h"
#include "Physics/Options/NavierStokesOptionSchema.h"

#include "FE/Constraints/ConstraintPlan.h"
#include "FE/Constraints/InterfaceMeasureGaugeAnchor.h"
#include "FE/Forms/BoundaryConditions.h"
#include "FE/Forms/BoundaryFunctional.h"
#include "FE/Forms/CutCellForms.h"
#include "FE/Forms/IntegrationDomain.h"
#include "FE/Forms/StandardBCs.h"
#include "FE/Forms/Vocabulary.h"
#include "FE/Auxiliary/AuxiliaryBindings.h"
#include "FE/Auxiliary/AuxiliaryModelBuilder.h"
#include "FE/Auxiliary/AuxiliaryModelDSL.h"
#include "FE/LevelSet/LevelSetGeometry.h"
#include "FE/Systems/ALEBinding.h"
#include "FE/Systems/BoundaryConditionManager.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"
#include "FE/Systems/FormulationBuilder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
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

namespace Factories {

namespace detail {

[[nodiscard]] inline FE::forms::FormExpr discreteFieldLike(const FE::forms::FormExpr& expr)
{
    if (!expr.isValid() || !expr.node()) {
        return expr;
    }

    const auto* node = expr.node();
    const auto fid = node->fieldId();
    const auto* sig = node->spaceSignature();
    if (!fid.has_value() || !sig) {
        return expr;
    }

    const auto name = node->symbolName().has_value()
        ? std::string(*node->symbolName())
        : node->toString();
    return FE::forms::FormExpr::discreteField(*fid, *sig, name);
}

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

inline void appendDiagnosticForm(FE::forms::FormExpr& destination,
                                 const FE::forms::FormExpr& contribution)
{
    destination = destination.isValid()
                      ? destination + contribution
                      : contribution;
}

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
    const std::function<std::optional<int>(int)>&
        generated_active_boundary_marker = {},
    std::vector<
        FE::forms::bc::
            GeneratedBoundaryNitscheTraceFormBinding>*
        generated_trace_bindings = nullptr,
    FE::forms::FormExpr* momentum_residual_work_form = nullptr,
    FE::forms::FormExpr* continuity_residual_work_form = nullptr)
{
    if (options.velocity_dirichlet_weak.empty()) {
        return;
    }
    if (!(options.nitsche_gamma > 0.0) ||
        !std::isfinite(options.nitsche_gamma)) {
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

    const auto stress_u = FE::forms::FormExpr::constant(2.0) * mu * FE::forms::sym(FE::forms::grad(u));
    const auto stress_v = FE::forms::FormExpr::constant(2.0) * mu * FE::forms::sym(FE::forms::grad(v));
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
    // The option preflight passes empty forms and keeps only the bindings, so
    // the rows are folded with appendDiagnosticForm (the same tree as
    // form + contribution when the form is not empty).
    const auto append_momentum = [&](const FE::forms::FormExpr& contribution) {
        appendDiagnosticForm(momentum_form, contribution);
        if (momentum_residual_work_form != nullptr) {
            appendDiagnosticForm(
                *momentum_residual_work_form, contribution);
        }
    };
    const auto append_continuity = [&](const FE::forms::FormExpr& contribution) {
        appendDiagnosticForm(continuity_form, contribution);
        if (continuity_residual_work_form != nullptr) {
            appendDiagnosticForm(
                *continuity_residual_work_form, contribution);
        }
    };

    for (const auto& bc : options.velocity_dirichlet_weak) {
        const int marker =
            FE::forms::bc::detail::boundaryMarkerOrThrow(bc, "navier_stokes::Factories::applyVelocityNitscheBCs");
        const auto generated_marker =
            generated_active_boundary_marker
                ? generated_active_boundary_marker(marker)
                : std::nullopt;
        for (int d = 0; d < dim; ++d) {
            if (!bc.active_components[static_cast<std::size_t>(d)]) {
                throw std::invalid_argument(
                    "applyVelocityNitscheBCs: partial-component weak velocity Dirichlet is not supported");
            }
        }

        auto uD_comp = FE::forms::bc::toVectorExpr(
            bc.value, dim, "ns_uD", marker, FE::forms::bc::ComponentValueNameStyle::Component);

        const auto uD = FE::forms::FormExpr::asVector(std::move(uD_comp));
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
    }
}

// ---------------------------------------------------------------------------
// Free surfaces: the contact-line laws, the pressure anchor, the semi-implicit
// capillary increment, the tangential mesh rows and the contact pin
// ---------------------------------------------------------------------------

/// One PrescribedAngle or DynamicRenE contact line of a free surface.
struct ContactLaw {
    const FreeSurfaceContactLine* line{nullptr};
    int wall{-1};             ///< wall boundary marker
    int marker{-1};           ///< requested contact-line marker (-1: generated)
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
[[nodiscard]] inline std::vector<ContactLaw> contactLaws(const FreeSurfaceBoundary& bc, int dim)
{
    std::vector<ContactLaw> laws;
    for (const auto& c : bc.contact_lines) {
        const auto kind = contactLineKind(c);
        if (kind != ContactLineKind::PrescribedAngle && kind != ContactLineKind::DynamicRenE) {
            continue;
        }
        ContactLaw law;
        law.line = &c;
        law.wall = contactLineWallBoundaryMarker(c);
        law.marker = contactLineMarker(c);
        law.young = kind == ContactLineKind::PrescribedAngle;
        law.dynamic = kind == ContactLineKind::DynamicRenE;
        law.strong = law.dynamic && std::get<FreeSurfaceContactLine::DynamicRenE>(c.configuration).strong_law;
        law.n_w = wallNormalExpression(c, dim);
        law.cos_e = FE::forms::FormExpr::constant(
            std::cos(constantScalarValueOrThrow(contactLineAngleRadians(c), "contact_angle_radians")));
        if (law.dynamic && !law.strong) {
            law.mobility = freeSurfaceScalar(contactLineMobility(c), "ns_dynamic_contact_angle_mobility", bc);
        }
        if (contactLineHasNavierSlip(c)) {
            law.slip_length = freeSurfaceScalar(contactLineSlipLength(c), "ns_contact_wall_slip_length", bc);
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
        constraint_plan.essentials([&constraints](FE::systems::FESystem& system) {
            FE::systems::installStrongDirichlet(system, constraints);
        });
    }
}

/**
 * @brief The log lines of the free-surface terms
 *
 * The factories below state the free-surface terms; the observer only logs
 * them, at the points where they are stated (D49 decision 5).  It goes when
 * the log lines become FE data.  The defaults log nothing.
 */
class FreeSurfaceObserver {
public:
    virtual ~FreeSurfaceObserver() = default;
    /// The terms of one contact line (the wetted wall marker, or -1).
    virtual void contactLineTerms(const FreeSurfaceBoundary&, const FreeSurfaceContactLine&, int, int) {}
    virtual void strongContactPointLaw(const FreeSurfaceContactLine&) {}
    virtual void unfittedMode(const FreeSurfaceBoundary&) {}
    /// sigma n = 0: no interface term.
    virtual void naturalUnfittedSurface(const FreeSurfaceBoundary&) {}
    /// The installed traction and the semi-implicit capillary term.
    virtual void traction(const FreeSurfaceBoundary&) {}
    /// The cut-cell stabilization (velocity and pressure orders).
    virtual void cutCellStabilization(const FreeSurfaceBoundary&, int, int) {}
};

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
inline void addUnfittedFreeSurface(FE::forms::FormExpr& M, FE::systems::FESystem& system,
                                   FE::constraints::ConstraintPlan& plan, FE::systems::FormulationBuilder& F,
                                   const IncompressibleNavierStokesVMSOptions& o, const FreeSurfaceBoundary& b,
                                   const NavierStokesTerms& t, FE::FieldId u_id, post::ConservativeBalanceForms* balance,
                                   FreeSurfaceObserver& obs)
{
    using namespace FE::forms;
    const auto& [u, p, v, q, rho, mu, f, dim] = t;
    const auto L = freeSurfaceLevelSetGeometry(b, system).side(activeDomainSide(b.active_domain));
    const auto gamma = freeSurfaceScalar(b.surface_tension, "ns_free_surface_surface_tension", b);

    registerFreeSurfacePrescribedAngleGeometry(system, b);   // contact-geometry certificate at each refresh
    for (const auto& c : Factories::contactLaws(b, dim)) {
        const auto CL = F.registered(L.contactLine(c.wall, c.marker));
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
        obs.contactLineTerms(b, *c.line, CL.marker(), wetted_marker);
        if (c.strong) {   // PointLaw phase: registered after the aggregation, so it can expand aggregated DOFs
            plan.contactPointVelocity(u_id, L.geometry(), c.wall, c.wall_normal, c.contact_speed,
                                      "ren_e_contact_point:" + std::to_string(c.wall));
            obs.strongContactPointLaw(*c.line);
        }
    }

    validateFreeSurfaceBoundary(b, o, o.enable_ale, system, dim);
    obs.unfittedMode(b);
    if (bc::isZeroConstantScalarValue(b.external_pressure) && bc::isZeroConstantScalarValue(b.surface_tension)) {
        obs.naturalUnfittedSurface(b);   // sigma n = 0: no interface term
        return;
    }
    const auto Gamma = L.interface();
    const auto n = L.outwardNormal();
    const auto p_ext = freeSurfaceScalar(b.external_pressure, "ns_free_surface_external_pressure", b);
    const auto pressure = p_ext * inner(n, v);   // SurfaceStress, the only form (D48)
    post::addExteriorPressure(balance, pressure.dI(Gamma));
    if (!surfaceStressActive(b)) {   // gamma = 0
        M = M + pressure.dI(Gamma);
    } else {
        const auto surface_energy = gamma * inner(tangentialProjector(n, dim), grad(v));
        post::addSurfaceEnergy(balance, surface_energy.dI(Gamma));
        M = M + (pressure + surface_energy).dI(Gamma);
    }
    if (usesSemiImplicitNormalIncrement(b)) {
        const auto u_ref = freeSurfaceSemiImplicitReferenceVelocity(system);   // u at the last geometry refresh
        M = M + Factories::semiImplicitNormalIncrementIntegrand(gamma, u, u_ref, v, n).dI(Gamma);
    }
    obs.traction(b);
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
                                 const FE::systems::ALEBinding& ale, const NavierStokesTerms& t, FE::FieldId u_id,
                                 FE::systems::FormulationBuilder& F, post::ConservativeBalanceForms* balance,
                                 FreeSurfaceObserver& obs)
{
    using namespace FE::forms;
    namespace fs = FE::systems;
    const auto& [u, p, v, q, rho, mu, f, dim] = t;
    Factories::applyFreeSurfaceContactLineConstraints(plan, b, ale, dim);   // pinned contact lines: d = 0
    bcs.add(std::make_unique<bc::ReservedBC>(b.boundary_marker));

    validateFreeSurfaceBoundary(b, o, o.enable_ale, system, dim);
    const auto Gamma = BoundaryDomain::fullPhysical(b.boundary_marker);
    const auto n = o.enable_ale ? currentNormal() : FormExpr::normal();   // normal of the integration frame
    const auto gamma = freeSurfaceScalar(b.surface_tension, "ns_free_surface_surface_tension", b);
    const auto p_ext = freeSurfaceScalar(b.external_pressure, "ns_free_surface_external_pressure", b);
    const bool dynamic_stress =
        !bc::isZeroConstantScalarValue(b.external_pressure) || !bc::isZeroConstantScalarValue(b.surface_tension);
    if (dynamic_stress) {   // SurfaceStress, the only fitted form (validated)
        const auto pressure = p_ext * inner(n, v);
        post::addExteriorPressure(balance, pressure.ds(Gamma));
        if (!surfaceStressActive(b)) {   // gamma = 0
            M = M + pressure.ds(Gamma);
        } else {
            const auto surface_energy = gamma * inner(tangentialProjector(n, dim), grad(v));
            post::addSurfaceEnergy(balance, surface_energy.ds(Gamma));
            M = M + (pressure + surface_energy).ds(Gamma);
        }
        obs.traction(b);
    }

    if (!ale.coupled()) {
        return;
    }
    const auto& rec = fittedMeshDisplacementRecord(system, ale);   // throws without a displacement unknown
    declareFittedTangentialMeshPolicy(system, ale, b);
    const bool normal_relation =
        b.kinematic_enforcement != FreeSurfaceKinematicEnforcement::None &&
        b.normal_kinematic_policy == FreeSurfaceNormalKinematicPolicy::MatchFluidNormalVelocity;
    const auto psi = TestField(ale.mesh_displacement_field, *rec.space, "psi_free_surface_mesh");
    const auto d = StateField(ale.mesh_displacement_field, *rec.space, "d_mesh_free_surface");
    const auto nc = currentNormal();
    FormExpr rows;
    if (normal_relation) {   // (dt d - u).n = 0 on the mesh rows
        const auto penalty = FormExpr::constant(b.kinematic_nitsche_gamma) / hNormal();
        rows = (penalty * normalTrace(dt(d) - u, nc) * normalTrace(psi, nc)).ds(Gamma);
    }
    if (const auto tangential = Factories::tangentialMeshPolicyResidual(b, d, psi, nc, rec.components, Gamma);
        tangential.isValid()) {
        rows = sumOf(rows, tangential);
    }
    if (rows.isValid()) {   // installed before the fluid residual
        fs::FormulationGroup group{.ale = &ale};
        if (normal_relation) {
            group.extra_trial_fields.push_back(u_id);
        }
        (void)F.installGroup({ale.mesh_displacement_field}, rows, group);
    }
    declareFittedMeshRowDescriptors(system, o.operator_tag, ale, b, u_id, rows.isValid());
}

// Pressure gradient-jump penalty on the facets next to the cut (Burman; Schott & Wall):
//   sum_F s gamma_p 0.01 h^3/(mu + rho h^2/dt) [grad p].[grad q]   (+ h^5 [d_nn p][d_nn q] for P2 pressure)
[[nodiscard]] inline FE::forms::FormExpr cutFacetPressurePenalty(const FreeSurfaceBoundary& b,
                                                                 const NavierStokesTerms& t,
                                                                 const IncompressibleNavierStokesVMSOptions& o,
                                                                 int velocity_order, int pressure_order,
                                                                 FreeSurfaceObserver& obs)
{
    using namespace FE::forms;
    const auto& cut = b.cut_cell_stabilization;
    if (!isUnfittedLevelSet(b) || !cut.enabled) {
        return {};
    }
    obs.cutCellStabilization(b, velocity_order, pressure_order);
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
    const auto gamma_p = freeSurfaceScalar(cut.pressure_gradient_penalty, "ns_free_surface_cut_pressure_penalty", b);
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
    return std::isfinite(value) && value > FE::Real{0.0};
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
  if (!std::isfinite(parameters.surface_tension) ||
      parameters.surface_tension < FE::Real{0.0}) {
    throw std::invalid_argument("incompressible two-fluid interface surface "
                                "tension must be finite and nonnegative");
  }
  if (parameters.prescribed_pressure_jump.has_value() &&
      !std::isfinite(*parameters.prescribed_pressure_jump)) {
    throw std::invalid_argument(
        "incompressible two-fluid prescribed pressure jump must be finite");
  }
  if (parameters.prescribed_viscous_traction_jump.has_value()) {
    const auto& target = *parameters.prescribed_viscous_traction_jump;
    if (!std::all_of(target.begin(), target.end(), [](FE::Real value) {
          return std::isfinite(value);
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
    const FE::forms::FormExpr &negative_velocity,
    const FE::forms::FormExpr &negative_pressure,
    const FE::forms::FormExpr &negative_velocity_test,
    const FE::forms::FormExpr &negative_pressure_test,
    const FE::forms::FormExpr &positive_velocity,
    const FE::forms::FormExpr &positive_pressure,
    const FE::forms::FormExpr &positive_velocity_test,
    const FE::forms::FormExpr &positive_pressure_test,
    const IncompressibleTwoFluidInterfaceParameters &parameters) {
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
      (two * mu_negative * sym(grad(negative_velocity))) * n;
  const auto negative_pressure_traction = (-negative_pressure * identity) * n;
  const auto positive_viscous_traction =
      (two * mu_positive * sym(grad(positive_velocity))) * n;
  const auto positive_pressure_traction = (-positive_pressure * identity) * n;
  const auto negative_viscous_test_traction =
      (two * mu_negative * sym(grad(negative_velocity_test))) * n;
  const auto negative_pressure_test_traction =
      (-negative_pressure_test * identity) * n;
  const auto positive_viscous_test_traction =
      (two * mu_positive * sym(grad(positive_velocity_test))) * n;
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
    std::vector<FormExpr> target_components;
    target_components.reserve(static_cast<std::size_t>(parameters.dimension));
    for (int component = 0; component < parameters.dimension; ++component) {
      target_components.push_back(FormExpr::constant(
          (*parameters.prescribed_viscous_traction_jump)
              [static_cast<std::size_t>(component)]));
    }
    const auto target = FormExpr::asVector(std::move(target_components));
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
    const auto projector = identity - outer(n, n);
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

} // namespace Factories
} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_BC_FACTORIES_H
