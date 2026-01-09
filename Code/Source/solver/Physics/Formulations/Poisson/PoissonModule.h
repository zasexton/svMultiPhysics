#ifndef SVMP_PHYSICS_FORMULATIONS_POISSON_MODULE_H
#define SVMP_PHYSICS_FORMULATIONS_POISSON_MODULE_H

/**
 * @file PoissonModule.h
 * @brief Minimal reference PhysicsModule (hello world) using FE/Forms
 *
 * Solves the scalar Poisson problem:
 *   -div(k grad(u)) = f  in Ω
 *
 * Weak form (residual):
 *   R(u; v) = ∫ k * grad(u)·grad(v) dx - ∫ f * v dx
 *
 * Optional boundary conditions (by boundary marker id):
 * - Neumann: k∇u·n = g          adds  -∫ g v ds
 * - Robin:   k∇u·n + α u = r    adds   ∫ α u v ds - ∫ r v ds
 * - Dirichlet (strong): u = uD  declared via FE/Forms and lowered by FE/Systems to constraints
 * - Dirichlet (weak): u = uD    imposed weakly via Nitsche boundary terms
 */

#include "Physics/Core/PhysicsModule.h"

#include "FE/Spaces/FunctionSpace.h"

#include <functional>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace poisson {

struct PoissonOptions {
    using SpatialFunction = std::function<FE::Real(FE::Real x, FE::Real y, FE::Real z)>;
    using TimeFunction = std::function<FE::Real(FE::Real x, FE::Real y, FE::Real z, FE::Real t)>;
    using ScalarValue = std::variant<FE::Real, SpatialFunction, TimeFunction>;

    struct DirichletBC {
        int boundary_marker{-1};
        ScalarValue value{};
    };

    struct NeumannBC {
        int boundary_marker{-1};
        ScalarValue flux{};
    };

    struct RobinBC {
        int boundary_marker{-1};
        ScalarValue alpha{};
        ScalarValue rhs{};
    };

    /**
     * @brief Example coupled Neumann BC (RCR-style) for demonstration
     *
     * This is a physics-agnostic example of the coupled-BC infrastructure:
     * - Defines a boundary functional `Q = ∫_Γ u ds` on `boundary_marker`.
     * - Evolves an auxiliary scalar `X` via a 0D ODE: C dX/dt = Q - (X - Pd)/Rd.
     * - Applies a Neumann flux g = X + Rp * Q via -∫ g v ds.
     *
     * Notes:
     * - This introduces non-local dependence on the solution through Q. The
     *   assembled Jacobian currently does not include the rank-1 coupling
     *   derivative dQ/du; treat this as a lagged/explicit coupling example.
     */
    struct CoupledRCRNeumannBC {
        int boundary_marker{-1};
        FE::Real Rp{0.0};
        FE::Real C{0.0};
        FE::Real Rd{1.0};
        FE::Real Pd{0.0};
        FE::Real X0{0.0};

        std::string functional_name{};
        std::string state_name{};
    };

    std::string field_name{"u"};
    FE::Real diffusion{1.0};
    FE::Real source{0.0};

    // Boundary conditions.
    //
    // Notes:
    // - Neumann/Robin are imposed weakly through boundary integrals (`.ds(marker)`).
    // - Dirichlet is imposed strongly through algebraic constraints.
    std::vector<DirichletBC> dirichlet{};
    std::vector<DirichletBC> dirichlet_weak{};
    std::vector<NeumannBC> neumann{};
    std::vector<RobinBC> robin{};
    std::vector<CoupledRCRNeumannBC> coupled_neumann_rcr{};

    // Weak Dirichlet (Nitsche) options.
    FE::Real nitsche_gamma{10.0};
    bool nitsche_symmetric{true};
    bool nitsche_scale_with_p{true};
};

class PoissonModule final : public PhysicsModule {
public:
    explicit PoissonModule(std::shared_ptr<const FE::spaces::FunctionSpace> space,
                           PoissonOptions options = {});

    void registerOn(FE::systems::FESystem& system) const override;

private:
    std::shared_ptr<const FE::spaces::FunctionSpace> space_{};
    PoissonOptions options_{};
};

} // namespace poisson
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_POISSON_MODULE_H
