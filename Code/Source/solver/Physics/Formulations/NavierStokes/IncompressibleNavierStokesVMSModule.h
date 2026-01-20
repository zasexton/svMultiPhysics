#ifndef SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_INCOMPRESSIBLE_NAVIER_STOKES_VMS_MODULE_H
#define SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_INCOMPRESSIBLE_NAVIER_STOKES_VMS_MODULE_H

/**
 * @file IncompressibleNavierStokesVMSModule.h
 * @brief Unsteady incompressible Navier–Stokes formulation using FE/Forms (residual-based VMS + backflow stabilization)
 *
 * This module installs a coupled velocity/pressure system:
 *   ρ (∂u/∂t + (u·∇)u - f) + ∇p - ∇·(2 μ(γ) ε(u)) = 0  in Ω
 *   ∇·u = 0                                                in Ω
 *
 * with residual-based VMS static subscales:
 *   u' = -τ_M R_m,
 *   p' = -τ_C (∇·u),
 *
 * assembled in a subscale-consistent way that yields SUPG/PSPG/LSIC-like
 * contributions without separate "add-on" terms.
 *
 * where R_m is the strong momentum residual:
 *   R_m = ρ (∂u/∂t + (u·∇)u - f) + ∇p - ∇·(2 μ(γ) ε(u)).
 *
 * Optional outflow pressure + backflow stabilization on boundary marker Γ:
 * - traction:    σn = -p_out n     -> adds +∫ p_out (n·v) ds(Γ)
 * - backflow:    +∫ β ρ max(0,-u·n) (u·v) ds(Γ)
 *
 * Notes:
 * - This formulation is written entirely in FE/Forms expressions and lowered
 *   to FE/Systems kernels via `installCoupledResidual(...)`.
 * - Time dependence is expressed symbolically using `dt(u)`; the application
 *   must assemble through a transient time-integration context (see
 *   `FE::systems::TransientSystem` or set `SystemStateView::time_integration`).
 */

#include "Physics/Core/PhysicsModule.h"

#include "FE/Forms/BoundaryConditions.h"
#include "FE/Forms/ConstitutiveModel.h"
#include "FE/Spaces/FunctionSpace.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

struct IncompressibleNavierStokesVMSOptions {
    using ScalarValue = FE::forms::bc::ScalarValue;

    struct VelocityDirichletBC {
        int boundary_marker{-1};
        std::array<ScalarValue, 3> value{ScalarValue{0.0}, ScalarValue{0.0}, ScalarValue{0.0}};
    };

    struct PressureDirichletBC {
        int boundary_marker{-1};
        ScalarValue value{0.0};
    };

    /**
     * @brief Traction Neumann BC for momentum equation
     *
     * Represents: σn = t̄ on boundary marker Γ(marker).
     */
    struct TractionNeumannBC {
        int boundary_marker{-1};
        std::array<ScalarValue, 3> traction{ScalarValue{0.0}, ScalarValue{0.0}, ScalarValue{0.0}};
    };

    /**
     * @brief Traction Robin BC for momentum equation
     *
     * Represents: σn + α u = r on Γ(marker), where α is scalar and r is vector-valued.
     */
    struct TractionRobinBC {
        int boundary_marker{-1};
        ScalarValue alpha{0.0};
        std::array<ScalarValue, 3> rhs{ScalarValue{0.0}, ScalarValue{0.0}, ScalarValue{0.0}};
    };

    struct PressureOutflowBC {
        int boundary_marker{-1};
        ScalarValue pressure{0.0};

        // Dimensionless backflow coefficient (0 disables the term).
        ScalarValue backflow_beta{0.0};
    };

    /**
     * @brief Example coupled pressure outflow (RCR-style demo) for Navier–Stokes
     *
     * Defines a boundary functional Q = ∫_Γ (u·n) ds on `boundary_marker` and
     * evolves an auxiliary scalar X via:
     *   C dX/dt = Q - (X - Pd)/Rd
     *
     * The applied traction is σn = -p_out n with p_out = X + Rp*Q (or the
     * purely resistive limit when C=0).
     *
     * Notes:
     * - As in the Poisson demo, the assembled Jacobian currently does not
     *   include the rank-1 coupling derivative dQ/du; treat this as a
     *   lagged/explicit coupling example.
     */
    struct CoupledRCROutflowBC {
        int boundary_marker{-1};
        FE::Real Rp{0.0};
        FE::Real C{0.0};
        FE::Real Rd{1.0};
        FE::Real Pd{0.0};
        FE::Real X0{0.0};

        // Dimensionless backflow coefficient (0 disables the term).
        ScalarValue backflow_beta{0.0};

        std::string functional_name{};
        std::string state_name{};
    };

    std::string velocity_field_name{"u"};
    std::string pressure_field_name{"p"};

    FE::Real density{1.0};

    // If viscosity_model is set, mu is computed as mu = mu(gamma) where:
    //   gamma = sqrt(2 * ε(u):ε(u)),   ε(u) = sym(grad(u)).
    //
    // If viscosity_model is null, viscosity is treated as a constant mu.
    FE::Real viscosity{0.01};
    std::shared_ptr<const FE::forms::ConstitutiveModel> viscosity_model{};

    // Constant body force (length-3; only first dim components are used).
    std::array<FE::Real, 3> body_force{0.0, 0.0, 0.0};

    // Enable residual-based VMS stabilization (static subscales u', p').
    bool enable_vms{true};

    // Legacy-inspired tuning constants used in tau_M (metric-based):
    //   tau_M/rho = 1 / (rho * sqrt( 4*(ct_m/dt)^2 + u^T Kxi u + ct_c * ||Kxi||_F^2 * nu^2 ))
    // with Kxi = J^{-T} J^{-1} and nu = mu/rho.
    FE::Real ct_m{1.0};
    FE::Real ct_c{36.0};

    // Small positive value added inside sqrt/divisions to avoid singular parameters.
    FE::Real stabilization_epsilon{1e-12};

    // Optional boundary conditions.
    std::vector<VelocityDirichletBC> velocity_dirichlet{};
    std::vector<VelocityDirichletBC> velocity_dirichlet_weak{};
    std::vector<PressureDirichletBC> pressure_dirichlet{};
    std::vector<TractionNeumannBC> traction_neumann{};
    std::vector<TractionRobinBC> traction_robin{};
    std::vector<PressureOutflowBC> pressure_outflow{};
    std::vector<CoupledRCROutflowBC> coupled_outflow_rcr{};

    // Weak Dirichlet (Nitsche) options for velocity.
    FE::Real nitsche_gamma{10.0};
    bool nitsche_symmetric{true};
    bool nitsche_scale_with_p{true};
};

class IncompressibleNavierStokesVMSModule final : public PhysicsModule {
public:
    IncompressibleNavierStokesVMSModule(std::shared_ptr<const FE::spaces::FunctionSpace> velocity_space,
                                        std::shared_ptr<const FE::spaces::FunctionSpace> pressure_space,
                                        IncompressibleNavierStokesVMSOptions options = {});

    void registerOn(FE::systems::FESystem& system) const override;

private:
    std::shared_ptr<const FE::spaces::FunctionSpace> velocity_space_{};
    std::shared_ptr<const FE::spaces::FunctionSpace> pressure_space_{};
    IncompressibleNavierStokesVMSOptions options_{};
};

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_INCOMPRESSIBLE_NAVIER_STOKES_VMS_MODULE_H
