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
#include "FE/Forms/NitscheBC.h"
#include "FE/Forms/StandardBCs.h"
#include "FE/Auxiliary/AuxiliaryBindings.h"
#include "FE/Auxiliary/AuxiliaryModelDSL.h"
#include "FE/Systems/FESystem.h"

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

[[nodiscard]] inline std::shared_ptr<FE::systems::BuiltAuxiliaryModel> rcrNeumannModel()
{
    static const auto model = FE::systems::aux::model("poisson_rcr_neumann", [](FE::systems::ModelFacade& m) {
        auto Q = m.input("Q");
        auto X = m.state("X");
        auto [Rp, C, Rd, Pd] = m.params("Rp", "C", "Rd", "Pd");

        m << FE::systems::ddt(X) == (Q - (X - Pd) / Rd) / C;
        m << FE::systems::out("flux") == X + Rp * Q;
    });
    return model;
}

[[nodiscard]] inline std::shared_ptr<FE::systems::BuiltAuxiliaryModel> resistiveNeumannModel()
{
    static const auto model = FE::systems::aux::model("poisson_resistive_neumann", [](FE::systems::ModelFacade& m) {
        auto Q = m.input("Q");
        auto P = m.state("P", FE::systems::AuxiliaryVariableKind::Algebraic);
        auto [Rsum, Pd] = m.params("Rsum", "Pd");

        m.initialGuess("P", 0.0);
        m << FE::systems::alg(P) == P - (Pd + Rsum * Q);
        m << FE::systems::out("flux") == P;
    });
    return model;
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
 * @brief Coupled Neumann RCR (Windkessel) BC via deployed auxiliary models
 */
[[nodiscard]] inline std::unique_ptr<FE::forms::bc::BoundaryCondition> toWindkesselBC(
    const PoissonOptions::CoupledRCRNeumannBC& bc,
    FE::systems::FESystem& system,
    const FE::forms::FormExpr& u)
{
    using namespace FE::forms;
    using namespace FE::systems;

    const int marker =
        FE::forms::bc::detail::boundaryMarkerOrThrow(bc, "poisson::Factories::toWindkesselBC");

    const FE::Real Rp = bc.Rp;
    const FE::Real C = bc.C;
    const FE::Real Rd = bc.Rd;
    const FE::Real Pd = bc.Pd;

    if (Rd == 0.0) {
        throw std::invalid_argument("CoupledRCRNeumannBC: Rd must be nonzero");
    }

    FE::forms::BoundaryFunctional flow_rate;
    flow_rate.integrand = u;
    flow_rate.boundary_marker = marker;
    flow_rate.reduction = FE::forms::BoundaryFunctional::Reduction::Sum;
    flow_rate.name = bc.functional_name;
    auto Q = system.boundaryIntegral(
        std::move(flow_rate),
        FE::systems::AuxiliaryInputUpdateSchedule::EachNonlinearIteration);

    const auto instance_name =
        bc.state_name.empty() ? ("poisson_rcr_" + std::to_string(marker)) : bc.state_name;

    if (C == 0.0) {
        auto resistive = system.deploy(
            use(detail::resistiveNeumannModel())
                .name(instance_name)
                .boundary(marker)
                .monolithic()
                .params({{"Rsum", Rp + Rd}, {"Pd", Pd}})
                .bind("Q", Q)
                .initialState({{"P", Pd}})
        );
        return std::make_unique<FE::forms::bc::NaturalBC>(marker, resistive.output("flux"));
    }

    auto rcr = system.deploy(
        use(detail::rcrNeumannModel())
            .name(instance_name)
            .boundary(marker)
            .monolithic()
            .params({{"Rp", Rp}, {"C", C}, {"Rd", Rd}, {"Pd", Pd}})
            .bind("Q", Q)
            .initialState({{"X", bc.X0}})
    );
    return std::make_unique<FE::forms::bc::NaturalBC>(marker, rcr.output("flux"));
}

} // namespace Factories
} // namespace poisson
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_POISSON_BC_FACTORIES_H
