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

#include "FE/Forms/BoundaryFunctional.h"
#include "FE/Forms/CoupledBCs.h"
#include "FE/Forms/StandardBCs.h"
#include "FE/Systems/AuxiliaryStateBuilder.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {
namespace Factories {

namespace detail {

[[nodiscard]] inline std::string markerName(std::string_view prefix, int boundary_marker)
{
    std::string out;
    out.reserve(prefix.size() + 1 + 20);
    out.append(prefix.data(), prefix.size());
    out.push_back('_');
    out.append(std::to_string(boundary_marker));
    return out;
}

[[nodiscard]] inline std::string componentName(std::string_view prefix, int boundary_marker, int component)
{
    std::string out;
    out.reserve(prefix.size() + 1 + 20 + 2 + 8);
    out.append(prefix.data(), prefix.size());
    out.push_back('_');
    out.append(std::to_string(boundary_marker));
    out.append("_c");
    out.append(std::to_string(component));
    return out;
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
    int dim)
{
    const int marker = FE::forms::bc::detail::boundaryMarkerOrThrow(bc, "navier_stokes::Factories::toTractionBC");

    std::vector<FE::forms::FormExpr> t_comp;
    t_comp.reserve(static_cast<std::size_t>(dim));
    for (int d = 0; d < dim; ++d) {
        t_comp.push_back(FE::forms::bc::toScalarExpr(
            bc.traction[static_cast<std::size_t>(d)],
            detail::componentName("ns_traction_neumann", marker, d)));
    }
    return std::make_unique<FE::forms::bc::NaturalBC>(marker, FE::forms::FormExpr::asVector(std::move(t_comp)));
}

[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition> toTractionRobinBC(
    const IncompressibleNavierStokesVMSOptions::TractionRobinBC& bc,
    int dim)
{
    const int marker =
        FE::forms::bc::detail::boundaryMarkerOrThrow(bc, "navier_stokes::Factories::toTractionRobinBC");

    const auto alpha = FE::forms::bc::toScalarExpr(bc.alpha, detail::markerName("ns_traction_robin_alpha", marker));

    std::vector<FE::forms::FormExpr> r_comp;
    r_comp.reserve(static_cast<std::size_t>(dim));
    for (int d = 0; d < dim; ++d) {
        r_comp.push_back(FE::forms::bc::toScalarExpr(
            bc.rhs[static_cast<std::size_t>(d)],
            detail::componentName("ns_traction_robin_rhs", marker, d)));
    }

    return std::make_unique<FE::forms::bc::RobinBC>(
        marker, alpha, FE::forms::FormExpr::asVector(std::move(r_comp)));
}

[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition> toVelocityEssentialBC(
    const IncompressibleNavierStokesVMSOptions::VelocityDirichletBC& bc,
    int dim,
    std::string_view symbol)
{
    const int marker =
        FE::forms::bc::detail::boundaryMarkerOrThrow(bc, "navier_stokes::Factories::toVelocityEssentialBC");

    std::vector<FE::forms::FormExpr> uD_comp;
    uD_comp.reserve(static_cast<std::size_t>(dim));
    for (int d = 0; d < dim; ++d) {
        uD_comp.push_back(FE::forms::bc::toScalarExpr(
            bc.value[static_cast<std::size_t>(d)],
            detail::componentName("ns_u_dirichlet", marker, d)));
    }

    return std::make_unique<FE::forms::bc::EssentialBC>(marker, std::move(uD_comp), std::string(symbol));
}

[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition> toPressureEssentialBC(
    const IncompressibleNavierStokesVMSOptions::PressureDirichletBC& bc,
    std::string_view symbol)
{
    const int marker =
        FE::forms::bc::detail::boundaryMarkerOrThrow(bc, "navier_stokes::Factories::toPressureEssentialBC");
    const auto value = FE::forms::bc::toScalarExpr(bc.value, detail::markerName("ns_p_dirichlet", marker));
    return std::make_unique<FE::forms::bc::EssentialBC>(marker, value, std::string(symbol));
}

[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition> toOutflowBC(
    const IncompressibleNavierStokesVMSOptions::PressureOutflowBC& bc,
    const FE::forms::FormExpr& u,
    const FE::forms::FormExpr& rho)
{
    const int marker = FE::forms::bc::detail::boundaryMarkerOrThrow(bc, "navier_stokes::Factories::toOutflowBC");

    const auto n = FE::forms::FormExpr::normal();
    const auto un = FE::forms::inner(u, n);
    const auto max_backflow =
        FE::forms::FormExpr::constant(0.5) * (FE::forms::abs(un) - un); // max(0, -u·n)

    const auto p_out = FE::forms::bc::toScalarExpr(bc.pressure, detail::markerName("ns_p_out", marker));
    const auto beta = FE::forms::bc::toScalarExpr(bc.backflow_beta, detail::markerName("ns_backflow_beta", marker));

    const auto flux = -p_out * n - beta * rho * max_backflow * u;
    return std::make_unique<FE::forms::bc::NaturalBC>(marker, flux);
}

[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition> toCoupledOutflowBC(
    const IncompressibleNavierStokesVMSOptions::CoupledRCROutflowBC& bc,
    FE::FieldId u_id,
    const FE::spaces::FunctionSpace& velocity_space,
    std::string_view velocity_field_name,
    const FE::forms::FormExpr& u,
    const FE::forms::FormExpr& rho)
{
    const int marker =
        FE::forms::bc::detail::boundaryMarkerOrThrow(bc, "navier_stokes::Factories::toCoupledOutflowBC");

    const std::string q_name = bc.functional_name.empty() ? ("ns_Q_" + std::to_string(marker)) : bc.functional_name;
    const std::string x_name = bc.state_name.empty() ? ("ns_X_" + std::to_string(marker)) : bc.state_name;

    const FE::Real Rp = bc.Rp;
    const FE::Real C = bc.C;
    const FE::Real Rd = bc.Rd;
    const FE::Real Pd = bc.Pd;

    if (Rd == 0.0) {
        throw std::invalid_argument("CoupledRCROutflowBC: Rd must be nonzero");
    }

    const auto n = FE::forms::FormExpr::normal();
    const auto un = FE::forms::inner(u, n);
    const auto max_backflow =
        FE::forms::FormExpr::constant(0.5) * (FE::forms::abs(un) - un); // max(0, -u·n)
    const auto beta =
        FE::forms::bc::toScalarExpr(bc.backflow_beta, detail::markerName("ns_rcr_backflow_beta", marker));

    const auto u_disc =
        FE::forms::FormExpr::discreteField(u_id, velocity_space, std::string(velocity_field_name));
    const auto Q_integrand = FE::forms::inner(u_disc, n);
    const auto Qsym = FE::forms::FormExpr::boundaryIntegral(Q_integrand, marker, q_name);

    std::vector<FE::systems::AuxiliaryStateRegistration> regs;

    FE::forms::FormExpr p_out;
    if (C == 0.0) {
        // Purely resistive limit: X = Pd + Rd*Q, so p_out = X + Rp*Q = Pd + (Rd+Rp)*Q.
        const FE::Real Rsum = Rd + Rp;
        p_out = FE::forms::FormExpr::constant(Pd) + FE::forms::FormExpr::constant(Rsum) * Qsym;
    } else {
        FE::forms::BoundaryFunctional Q;
        Q.integrand = Q_integrand;
        Q.boundary_marker = marker;
        Q.name = q_name;
        Q.reduction = FE::forms::BoundaryFunctional::Reduction::Sum;

        // ODE: dX/dt = (Q - (X - Pd)/Rd) / C
        const auto Xsym = FE::forms::FormExpr::auxiliaryState(x_name);
        const auto rhs =
            (Qsym - (Xsym - FE::forms::FormExpr::constant(Pd)) / FE::forms::FormExpr::constant(Rd)) /
            FE::forms::FormExpr::constant(C);
        const auto d_rhs_dX = FE::forms::FormExpr::constant(-1.0 / (Rd * C));

        auto reg = FE::systems::auxiliaryODE(x_name, bc.X0)
                       .requiresIntegral(Q)
                       .withRHS(rhs)
                       .withJacobian(d_rhs_dX)
                       .withIntegrator(FE::systems::ODEMethod::BackwardEuler)
                       .build();
        regs.push_back(std::move(reg));

        p_out = Xsym + FE::forms::FormExpr::constant(Rp) * Qsym;
    }

    const auto flux = -p_out * n - beta * rho * max_backflow * u;
    return std::make_unique<FE::forms::bc::CoupledNaturalBC>(marker, flux, std::move(regs));
}

inline void applyVelocityNitscheBCs(
    FE::forms::FormExpr& momentum_form,
    FE::forms::FormExpr& continuity_form,
    const IncompressibleNavierStokesVMSOptions& options,
    const FE::spaces::FunctionSpace& velocity_space,
    int dim,
    const FE::forms::FormExpr& u,
    const FE::forms::FormExpr& p,
    const FE::forms::FormExpr& v,
    const FE::forms::FormExpr& q,
    const FE::forms::FormExpr& mu)
{
    if (options.velocity_dirichlet_weak.empty()) {
        return;
    }
    if (!(options.nitsche_gamma > 0.0)) {
        throw std::invalid_argument("applyVelocityNitscheBCs: nitsche_gamma must be > 0");
    }

    const auto n = FE::forms::FormExpr::normal();
    const auto h = FE::forms::FormExpr::cellDiameter();
    const auto gamma = FE::forms::FormExpr::constant(options.nitsche_gamma);

    int p_order = 1;
    if (options.nitsche_scale_with_p) {
        p_order = velocity_space.polynomial_order();
        if (p_order < 1) {
            p_order = 1;
        }
    }
    const auto p2 = FE::forms::FormExpr::constant(static_cast<FE::Real>(p_order * p_order));
    const auto penalty = gamma * mu * p2 / h;

    const auto stress_u = FE::forms::FormExpr::constant(2.0) * mu * FE::forms::sym(FE::forms::grad(u));
    const auto stress_v = FE::forms::FormExpr::constant(2.0) * mu * FE::forms::sym(FE::forms::grad(v));

    for (const auto& bc : options.velocity_dirichlet_weak) {
        const int marker =
            FE::forms::bc::detail::boundaryMarkerOrThrow(bc, "navier_stokes::Factories::applyVelocityNitscheBCs");

        std::vector<FE::forms::FormExpr> uD_comp;
        uD_comp.reserve(static_cast<std::size_t>(dim));
        for (int d = 0; d < dim; ++d) {
            uD_comp.push_back(FE::forms::bc::toScalarExpr(
                bc.value[static_cast<std::size_t>(d)],
                detail::componentName("ns_uD", marker, d)));
        }

        const auto uD = FE::forms::FormExpr::asVector(std::move(uD_comp));
        const auto diff = u - uD;

        // Consistency: add the missing stress boundary term -<σ(u,p)n, v>.
        momentum_form = momentum_form + (p * FE::forms::inner(n, v)).ds(marker) -
                        FE::forms::inner(stress_u * n, v).ds(marker);

        // Adjoint consistency (variant-dependent) + penalty.
        if (options.nitsche_symmetric) {
            momentum_form = momentum_form - FE::forms::inner(stress_v * n, diff).ds(marker);
            continuity_form = continuity_form + (q * FE::forms::inner(n, diff)).ds(marker);
        } else {
            momentum_form = momentum_form + FE::forms::inner(stress_v * n, diff).ds(marker);
            continuity_form = continuity_form - (q * FE::forms::inner(n, diff)).ds(marker);
        }
        momentum_form = momentum_form + (penalty * FE::forms::inner(diff, v)).ds(marker);
    }
}

} // namespace Factories
} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_BC_FACTORIES_H

