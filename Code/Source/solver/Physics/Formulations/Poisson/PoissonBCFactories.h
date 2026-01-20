#ifndef SVMP_PHYSICS_FORMULATIONS_POISSON_BC_FACTORIES_H
#define SVMP_PHYSICS_FORMULATIONS_POISSON_BC_FACTORIES_H

/**
 * @file PoissonBCFactories.h
 * @brief Translation helpers from PoissonOptions boundary-condition structs to FE boundary-condition objects
 *
 * This file keeps Physics option parsing/translation out of PoissonModule.cpp,
 * enabling a declarative "installer + factory" workflow.
 */

#include "Physics/Formulations/Poisson/PoissonModule.h"

#include "FE/Forms/BoundaryFunctional.h"
#include "FE/Forms/CoupledBCs.h"
#include "FE/Forms/NitscheBC.h"
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
namespace poisson {
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

} // namespace detail

[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition> toNaturalBC(
    const PoissonOptions::NeumannBC& bc)
{
    const int marker =
        FE::forms::bc::detail::boundaryMarkerOrThrow(bc, "poisson::Factories::toNaturalBC");
    const auto flux = FE::forms::bc::toScalarExpr(bc.flux, detail::markerName("poisson_neumann", marker));
    return std::make_unique<FE::forms::bc::NaturalBC>(marker, flux);
}

[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition> toRobinBC(
    const PoissonOptions::RobinBC& bc)
{
    const int marker = FE::forms::bc::detail::boundaryMarkerOrThrow(bc, "poisson::Factories::toRobinBC");
    const auto alpha = FE::forms::bc::toScalarExpr(bc.alpha, detail::markerName("poisson_robin_alpha", marker));
    const auto rhs = FE::forms::bc::toScalarExpr(bc.rhs, detail::markerName("poisson_robin_rhs", marker));
    return std::make_unique<FE::forms::bc::RobinBC>(marker, alpha, rhs);
}

[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition> toEssentialBC(
    const PoissonOptions::DirichletBC& bc,
    std::string_view symbol)
{
    const int marker =
        FE::forms::bc::detail::boundaryMarkerOrThrow(bc, "poisson::Factories::toEssentialBC");
    const auto value = FE::forms::bc::toScalarExpr(bc.value, detail::markerName("poisson_dirichlet", marker));
    return std::make_unique<FE::forms::bc::EssentialBC>(marker, value, std::string(symbol));
}

[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition> toNitscheBC(
    const PoissonOptions::DirichletBC& bc,
    const FE::forms::FormExpr& diffusion_coeff,
    FE::Real penalty_gamma,
    bool symmetric,
    bool scale_with_p)
{
    const int marker = FE::forms::bc::detail::boundaryMarkerOrThrow(bc, "poisson::Factories::toNitscheBC");
    const auto value =
        FE::forms::bc::toScalarExpr(bc.value, detail::markerName("poisson_nitsche_dirichlet", marker));
    return std::make_unique<FE::forms::bc::ScalarNitscheBC>(
        marker, value, diffusion_coeff, penalty_gamma, symmetric, scale_with_p);
}

/**
 * @brief Coupled Neumann RCR (Windkessel) demo BC factory
 *
 * Registers an auxiliary scalar state X and a boundary functional Q, then applies:
 *   flux = X + Rp * Q  (or the resistive limit when C=0).
 */
[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition> toWindkesselBC(
    const PoissonOptions::CoupledRCRNeumannBC& bc,
    FE::FieldId u_id,
    const FE::spaces::FunctionSpace& space,
    std::string_view field_name)
{
    const int marker =
        FE::forms::bc::detail::boundaryMarkerOrThrow(bc, "poisson::Factories::toWindkesselBC");

    const std::string q_name =
        bc.functional_name.empty() ? ("poisson_Q_" + std::to_string(marker)) : bc.functional_name;
    const std::string x_name = bc.state_name.empty() ? ("poisson_X_" + std::to_string(marker)) : bc.state_name;

    const FE::Real Rp = bc.Rp;
    const FE::Real C = bc.C;
    const FE::Real Rd = bc.Rd;
    const FE::Real Pd = bc.Pd;

    if (Rd == 0.0) {
        throw std::invalid_argument("CoupledRCRNeumannBC: Rd must be nonzero");
    }

    const auto u_state = FE::forms::FormExpr::discreteField(u_id, space, std::string(field_name));
    const auto Qsym = FE::forms::FormExpr::boundaryIntegral(u_state, marker, q_name);

    if (C == 0.0) {
        // Purely resistive limit: X = Pd + Rd*Q, so flux = X + Rp*Q = Pd + (Rd+Rp)*Q.
        const FE::Real Rsum = Rd + Rp;
        const auto flux = FE::forms::FormExpr::constant(Pd) + FE::forms::FormExpr::constant(Rsum) * Qsym;
        return std::make_unique<FE::forms::bc::CoupledNaturalBC>(marker, flux);
    }

    FE::forms::BoundaryFunctional Q;
    Q.integrand = u_state;
    Q.boundary_marker = marker;
    Q.name = q_name;
    Q.reduction = FE::forms::BoundaryFunctional::Reduction::Sum;

    // ODE: dX/dt = (Q - (X - Pd)/Rd) / C
    const auto Xsym = FE::forms::FormExpr::auxiliaryState(x_name);
    const auto rhs = (Qsym - (Xsym - FE::forms::FormExpr::constant(Pd)) / FE::forms::FormExpr::constant(Rd)) /
                     FE::forms::FormExpr::constant(C);
    const auto d_rhs_dX = FE::forms::FormExpr::constant(-1.0 / (Rd * C));

    auto reg = FE::systems::auxiliaryODE(x_name, bc.X0)
                   .requiresIntegral(Q)
                   .withRHS(rhs)
                   .withJacobian(d_rhs_dX)
                   .withIntegrator(FE::systems::ODEMethod::BackwardEuler)
                   .build();

    const auto flux = Xsym + FE::forms::FormExpr::constant(Rp) * Qsym;

    std::vector<FE::systems::AuxiliaryStateRegistration> regs;
    regs.push_back(std::move(reg));
    return std::make_unique<FE::forms::bc::CoupledNaturalBC>(marker, flux, std::move(regs));
}

} // namespace Factories
} // namespace poisson
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_POISSON_BC_FACTORIES_H

