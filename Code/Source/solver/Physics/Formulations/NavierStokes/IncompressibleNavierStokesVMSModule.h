#ifndef SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_INCOMPRESSIBLE_NAVIER_STOKES_VMS_MODULE_H
#define SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_INCOMPRESSIBLE_NAVIER_STOKES_VMS_MODULE_H

/**
 * @file IncompressibleNavierStokesVMSModule.h
 * @brief Unsteady incompressible Navier–Stokes formulation using FE/Forms (residual-based VMS + backflow stabilization)
 *
 * This module installs a coupled velocity/pressure system:
 *   ρ (∂u/∂t + (u·∇)u - f) + ∇p - ∇·(2 μ(γ) ε(u)) = 0      in Ω
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

#include "Physics/Core/PhysicsJITPolicy.h"
#include "Physics/Core/PhysicsModule.h"

#include "FE/Forms/BoundaryConditions.h"
#include "FE/Forms/ConstitutiveModel.h"
#include "FE/Forms/FormExpr.h"
#include "FE/Spaces/FunctionSpace.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

enum class ALEMeshVelocitySource {
    PrescribedData,
    CoupledDisplacement
};

enum class FreeSurfaceImplementation : std::uint8_t {
    FittedALE,
    UnfittedLevelSet
};

enum class FreeSurfaceKinematicEnforcement : std::uint8_t {
    None,
    Penalty,
    Nitsche
};

enum class FreeSurfaceNormalKinematicPolicy : std::uint8_t {
    None,
    MatchFluidNormalVelocity
};

enum class FreeSurfaceTangentialMeshPolicy : std::uint8_t {
    Free,
    SmoothingOnly,
    Prescribed
};

enum class FreeSurfaceContactLineModel : std::uint8_t {
    None,
    Pinned,
    PrescribedContactAngle
};

enum class FreeSurfaceWallSlipModel : std::uint8_t {
    None,
    Navier
};

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
     * @brief Physics-side model for the wall intersection of one free surface
     *
     * The free-surface boundary owns these declarations.  FE geometry remains
     * generic: later fitted and unfitted implementations can use the wall marker,
     * optional contact-line marker, and scalar model parameters to install the
     * appropriate Physics residuals or constraints.
     */
    struct FreeSurfaceContactLine {
        FreeSurfaceContactLineModel model{FreeSurfaceContactLineModel::None};
        int wall_boundary_marker{-1};
        int contact_line_marker{-1};

        // Internal contact-angle value is in radians, measured through the
        // fluid phase.  The parser may translate user-facing units later.
        ScalarValue contact_angle_radians{1.57079632679489661923};
        std::array<ScalarValue, 3> wall_normal{
            ScalarValue{0.0}, ScalarValue{0.0}, ScalarValue{0.0}};
        ScalarValue contact_angle_penalty{1.0};
        ScalarValue mobility{0.0};

        FreeSurfaceWallSlipModel wall_slip_model{FreeSurfaceWallSlipModel::None};
        ScalarValue slip_length{0.0};
    };

    /**
     * @brief Physics-side free-surface relation on a fitted or embedded interface
     *
     * This option only declares Navier-Stokes free-surface equations.  The FE
     * layer remains physics-agnostic and supplies the generic form vocabulary,
     * moving-geometry terminals, and integration domains.
     */
    struct FreeSurfaceBoundary {
        FreeSurfaceImplementation implementation{FreeSurfaceImplementation::FittedALE};

        // Fitted ALE free surfaces integrate on ds(boundary_marker).
        int boundary_marker{-1};

        // Unfitted/level-set free surfaces integrate on dI(interface_marker).
        int interface_marker{-1};
        std::string level_set_field_name{"level_set"};

        // Dynamic stress balance: sigma(u,p)n = (-p_ext + gamma*kappa)n.
        ScalarValue external_pressure{0.0};
        ScalarValue surface_tension{0.0};

        // Fitted ALE can use current mesh geometry or a supplied curvature
        // expression/value.  Unfitted level-set surfaces use
        // meanCurvatureFromLevelSet(phi) by default.
        ScalarValue curvature{0.0};
        bool use_current_geometry_curvature{false};
        bool use_level_set_curvature{true};

        // Fitted ALE kinematic relation: (u - meshVelocity()) · n = 0.
        FreeSurfaceNormalKinematicPolicy normal_kinematic_policy{
            FreeSurfaceNormalKinematicPolicy::MatchFluidNormalVelocity};
        FreeSurfaceTangentialMeshPolicy tangential_mesh_policy{
            FreeSurfaceTangentialMeshPolicy::SmoothingOnly};
        std::array<ScalarValue, 3> prescribed_tangential_mesh_velocity{
            ScalarValue{0.0}, ScalarValue{0.0}, ScalarValue{0.0}};
        FreeSurfaceKinematicEnforcement kinematic_enforcement{
            FreeSurfaceKinematicEnforcement::None};
        ScalarValue kinematic_penalty{0.0};

        std::vector<FreeSurfaceContactLine> contact_lines{};
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
     * - When deployed through the generalized AuxiliaryState path in
     *   `NavierStokesBCFactories.h`, the monolithic outlet Jacobian includes
     *   the exact FE-backed input coupling for `Q`.
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

    /**
     * @brief Two-capacitor RCRCR outlet model for Navier-Stokes
     *
     * Defines a boundary flow functional Q = ∫_Γ (u·n) ds and evolves two
     * capacitive node pressures:
     *
     *   C1 dP1/dt = Q - (P1 - P2) / Rm
     *   C2 dP2/dt = (P1 - P2) / Rm - (P2 - Pd) / Rd
     *
     * The applied traction is σn = -p_out n with
     *
     *   p_out = P1 + Rp * Q
     *
     * This extends the standard RCR outlet with an additional capacitive
     * storage node and intermediate resistance.
     */
    struct CoupledRCRCROutflowBC {
        int boundary_marker{-1};
        FE::Real Rp{0.0};
        FE::Real C1{0.0};
        FE::Real Rm{1.0};
        FE::Real C2{0.0};
        FE::Real Rd{1.0};
        FE::Real Pd{0.0};
        FE::Real P10{0.0};
        FE::Real P20{0.0};

        ScalarValue backflow_beta{0.0};

        std::string functional_name{};
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

    // Enable the convective term rho * (u · ∇) u.
    //
    // This can be disabled to recover a (possibly unsteady) Stokes formulation.
    bool enable_convection{true};

    // ALE/moving-domain options. Disabled by default to preserve static-mesh behavior.
    //
    // When enabled, the module binds a generic FE mesh-velocity field and uses
    // FE Forms moving-domain terminals. Physics remains responsible for the
    // ALE-specific residual choices.
    bool enable_ale{false};
    bool include_moving_control_volume_transient{true};
    ALEMeshVelocitySource mesh_velocity_source{ALEMeshVelocitySource::PrescribedData};
    std::string mesh_velocity_field_name{"mesh_velocity"};
    std::string mesh_displacement_field_name{"mesh_displacement"};
    bool auto_register_mesh_velocity_field{true};
    bool auto_register_mesh_displacement_field{false};
    std::shared_ptr<const FE::spaces::FunctionSpace> mesh_velocity_space{};
    FE::forms::GeometryTangentPath moving_mesh_tangent_path{
        FE::forms::GeometryTangentPath::SymbolicRequired};

    // Enable residual-based VMS stabilization (static subscales u', p').
    bool enable_vms{true};
    core::PhysicsJITPolicy jit_policy{};

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
    std::vector<FreeSurfaceBoundary> free_surface{};
    std::vector<CoupledRCROutflowBC> coupled_outflow_rcr{};
    std::vector<CoupledRCRCROutflowBC> coupled_outflow_rcrcr{};

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
