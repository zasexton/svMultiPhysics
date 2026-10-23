#ifndef SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_BC_FACTORIES_H
#define SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_BC_FACTORIES_H

/**
 * @file NavierStokesBCFactories.h
 * @brief Translation helpers from Navier–Stokes option structs to FE boundary-condition objects
 *
 * Keeps `IncompressibleNavierStokesVMSModule.cpp` declarative by moving option
 * parsing and coupled-BC construction into reusable helpers.
 */

#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceGeometry.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceQueries.h"

#include "FE/Constraints/ConstraintPlan.h"
#include "FE/Constraints/InterfaceMeasureGaugeAnchor.h"
#include "FE/Forms/BoundaryConditions.h"
#include "FE/Forms/BoundaryFunctional.h"
#include "FE/Forms/IntegrationDomain.h"
#include "FE/Forms/StandardBCs.h"
#include "FE/Forms/Vocabulary.h"
#include "FE/Auxiliary/AuxiliaryBindings.h"
#include "FE/Auxiliary/AuxiliaryModelBuilder.h"
#include "FE/Auxiliary/AuxiliaryModelDSL.h"
#include "FE/Systems/ALEBinding.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"

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
    const auto append_momentum = [&](const FE::forms::FormExpr& contribution) {
        momentum_form = momentum_form + contribution;
        if (momentum_residual_work_form != nullptr) {
            appendDiagnosticForm(
                *momentum_residual_work_form, contribution);
        }
    };
    const auto append_continuity = [&](const FE::forms::FormExpr& contribution) {
        continuity_form = continuity_form + contribution;
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
                    buildGeneratedBoundarySymmetricGradientNitscheTraceTerms(
                        u,
                        v,
                        uD,
                        mu,
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

} // namespace Factories
} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_BC_FACTORIES_H
