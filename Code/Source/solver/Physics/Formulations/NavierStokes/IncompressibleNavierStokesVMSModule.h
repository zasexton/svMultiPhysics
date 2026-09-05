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
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceOptions.h"

#include "FE/Forms/BoundaryConditions.h"
#include "FE/Forms/ConstitutiveModel.h"
#include "FE/Forms/FormExpr.h"
#include "FE/Spaces/FunctionSpace.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

enum class ALEMeshVelocitySource {
    PrescribedData,
    CoupledDisplacement
};

/**
 * @brief Named velocity-row operators for the conservative free-surface split
 *
 * These diagnostic operators are installed only when
 * `SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC` is enabled and the
 * active free-surface configuration uses constant-gamma `SurfaceStress` or
 * `KinematicAreaGradientTraction`.
 * Their residual vectors satisfy `physical_potential_virtual_work =
 * surface_energy_virtual_work + gravitational_potential_virtual_work`,
 * `pressure_representability_load_virtual_work = prescribed exterior-
 * pressure virtual work + physical_potential_virtual_work`, and
 * `conservative_balance = pressure_virtual_work +
 * physical_potential_virtual_work` on the same generated cut/contact and
 * active-volume rules as the production momentum residual.  Here
 * `pressure_virtual_work` contains both the unknown liquid-pressure term and
 * the prescribed exterior-pressure term, while the representability load
 * deliberately excludes the unknown pressure.
 */
struct FreeSurfaceConservativeBalanceDiagnosticOperators {
    inline static constexpr std::string_view pressure_virtual_work{
        "equations_diagnostic_ns_free_surface_pressure_virtual_work"};
    inline static constexpr std::string_view surface_energy_virtual_work{
        "equations_diagnostic_ns_free_surface_surface_energy_virtual_work"};
    inline static constexpr std::string_view
        gravitational_potential_virtual_work{
            "equations_diagnostic_ns_free_surface_gravitational_potential_virtual_work"};
    inline static constexpr std::string_view physical_potential_virtual_work{
        "equations_diagnostic_ns_free_surface_physical_potential_virtual_work"};
    inline static constexpr std::string_view
        pressure_representability_load_virtual_work{
            "equations_diagnostic_ns_free_surface_pressure_representability_load_virtual_work"};
    inline static constexpr std::string_view conservative_balance{
        "equations_diagnostic_ns_free_surface_conservative_balance"};
    // Symmetric mixed diagnostic operator
    //
    //     K = [ 0   G  ] ,
    //         [ G^T 0  ]
    //
    // where G maps pressure coefficients to velocity-test virtual work.  The
    // Newton diagnostic uses this one matrix for both G and G^T actions, so it
    // remains backend independent without requiring a GenericMatrix transpose
    // API or forming normal equations.
    inline static constexpr std::string_view pressure_representability_pair{
        "equations_diagnostic_ns_free_surface_pressure_representability_pair"};
};

/**
 * System-registered residual operators used by the accepted-stage
 * free-surface work account.
 *
 * Each installed operator is an exact additive group from the production
 * residual.  Pairing its constrained residual with the converged state and
 * multiplying by minus the step duration gives signed work added to modeled
 * stored energy.  The system declaration records the effective operator tag,
 * so application code does not depend on these suffixes.
 */
struct FreeSurfaceResidualWorkOperatorSuffixes {
    inline static constexpr std::string_view convection{
        "free_surface_work_convection"};
    inline static constexpr std::string_view pressure_continuity{
        "free_surface_work_pressure_continuity"};
    inline static constexpr std::string_view nonconservative_body_force{
        "free_surface_work_nonconservative_body_force"};
    inline static constexpr std::string_view weak_boundary{
        "free_surface_work_weak_boundary"};
    inline static constexpr std::string_view vms_pspg{
        "free_surface_work_vms_pspg"};
    inline static constexpr std::string_view ghost_penalty{
        "free_surface_work_ghost_penalty"};
};

/**
 * @brief Velocity-block operators for a symmetric Nitsche energy certificate
 *
 * These operators are installed only for constant viscosity when
 * `SVMP_NS_SYMMETRIC_NITSCHE_ENERGY_DIAGNOSTIC` is enabled, the C++ caller
 * explicitly selects `JointLowLevelPrerequisite`, and at least one symmetric
 * weak velocity boundary is present. They use the production active-volume
 * and generated-active-boundary forms. The energy norm is the bulk viscous
 * form plus the Nitsche penalty; the symmetric operator adds the two
 * consistency terms. The bulk-plus-consistency operator provides an
 * independently assembled component-parity check while retaining a volume
 * anchor for boundary-kernel dispatch. Pressure and transient terms are
 * deliberately excluded so the generalized spectrum measures the actual
 * viscous Nitsche form rather than a mass-regularized surrogate.
 */
struct SymmetricNitscheEnergyDiagnosticOperators {
    inline static constexpr std::string_view bulk_viscous{
        "equations_diagnostic_ns_symmetric_nitsche_bulk_viscous"};
    inline static constexpr std::string_view bulk_plus_consistency{
        "equations_diagnostic_ns_symmetric_nitsche_bulk_plus_consistency"};
    inline static constexpr std::string_view symmetric_operator{
        "equations_diagnostic_ns_symmetric_nitsche_operator"};
    inline static constexpr std::string_view energy_norm{
        "equations_diagnostic_ns_symmetric_nitsche_energy_norm"};
};

enum class SymmetricNitscheEnergyQualificationScope {
    RejectUnqualified = 0,
    JointLowLevelPrerequisite
};

struct IncompressibleNavierStokesVMSOptions {
    using ScalarValue = FE::forms::bc::ScalarValue;
    using FreeSurfaceContactLine = navier_stokes::FreeSurfaceContactLine;
    using FreeSurfaceCutCellStabilization =
        navier_stokes::FreeSurfaceCutCellStabilization;
    using FreeSurfaceSmallCutAggregationGuards =
        FE::constraints::SmallCutAggregationGuardOptions;
    using FreeSurfaceVelocityExtension =
        navier_stokes::FreeSurfaceVelocityExtension;
    using FreeSurfaceBoundary = navier_stokes::FreeSurfaceBoundary;

    inline static constexpr int current_configuration_schema_version{2};
    int input_configuration_schema_version{
        current_configuration_schema_version};
    bool explicit_legacy_configuration{false};
    // C++-only diagnostic authorization. It is intentionally absent from the
    // input schema and defaults to fail-closed nonqualification.
    SymmetricNitscheEnergyQualificationScope
        symmetric_nitsche_energy_qualification_scope{
            SymmetricNitscheEnergyQualificationScope::
                RejectUnqualified};
    FreeSurfacePhysicalModel free_surface_physical_model{
        FreeSurfacePhysicalModel::OnePhaseLiquidPrescribedExteriorPressure};

    struct VelocityDirichletBC {
        int boundary_marker{-1};
        std::array<ScalarValue, 3> value{ScalarValue{0.0}, ScalarValue{0.0}, ScalarValue{0.0}};
        std::array<bool, 3> active_components{true, true, true};
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
    // Owning coupled operator for the Navier--Stokes residual. A level-set
    // contact condition is installed only on that field's independently
    // resolved owner formulation.
    std::string operator_tag{"equations"};

    FE::Real density{1.0};

    // If viscosity_model is set, mu is computed as mu = mu(gamma) where:
    //   gamma = sqrt(2 * ε(u):ε(u)),   ε(u) = sym(grad(u)).
    //
    // If viscosity_model is null, viscosity is treated as a constant mu.
    FE::Real viscosity{0.01};
    std::shared_ptr<const FE::forms::ConstitutiveModel> viscosity_model{};

    // Constant body force (length-3; only first dim components are used).
    std::array<FE::Real, 3> body_force{0.0, 0.0, 0.0};
    std::array<ScalarValue, 3> body_force_spacetime{
        ScalarValue{0.0}, ScalarValue{0.0}, ScalarValue{0.0}};
    bool has_body_force_spacetime{false};
    std::string body_force_field_name{};
    bool auto_register_body_force_field{true};
    bool rotating_frame_coriolis_enabled{false};
    std::array<ScalarValue, 3> rotating_frame_angular_velocity{
        ScalarValue{0.0}, ScalarValue{0.0}, ScalarValue{0.0}};

    struct HydrostaticPressureInitialization {
        bool enabled{false};
        std::array<FE::Real, 3> reference_point{0.0, 0.0, 0.0};
        FE::Real reference_pressure{0.0};
        std::string field_name{};
    };
    HydrostaticPressureInitialization hydrostatic_pressure_initialization{};

    enum class NodePressureConstraintIdType : std::uint8_t {
        GlobalVertexGid,
        LocalVertexId
    };

    struct NodePressureConstraint {
        FE::GlobalIndex node_id{FE::INVALID_GLOBAL_INDEX};
        FE::Real pressure{0.0};
    };

    struct NodePressureConstraintOptions {
        NodePressureConstraintIdType id_type{NodePressureConstraintIdType::GlobalVertexGid};
        std::vector<NodePressureConstraint> values{};
    };
    NodePressureConstraintOptions node_pressure_constraints{};

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
    // Generated active-boundary symmetric routes are accepted only when the
    // exact current-state trace certificate proves at least this fraction of
    // the bulk-plus-penalty energy norm.  This is not applied to fitted or
    // unsymmetric routes, but it remains serialized configuration and must
    // always be finite and strictly between zero and one.
    FE::Real generated_boundary_nitsche_minimum_energy_ratio{0.25};
};

class IncompressibleNavierStokesVMSModule final : public PhysicsModule {
public:
    IncompressibleNavierStokesVMSModule(std::shared_ptr<const FE::spaces::FunctionSpace> velocity_space,
                                        std::shared_ptr<const FE::spaces::FunctionSpace> pressure_space,
                                        IncompressibleNavierStokesVMSOptions options = {});

    void registerOn(FE::systems::FESystem& system) const override;
    void applyInitialConditions(const FE::systems::FESystem& system,
                                FE::backends::GenericVector& u0) const override;
    [[nodiscard]] std::optional<EffectiveConfigurationArtifact>
    effectiveConfigurationArtifact() const override;

private:
    std::shared_ptr<const FE::spaces::FunctionSpace> velocity_space_{};
    std::shared_ptr<const FE::spaces::FunctionSpace> pressure_space_{};
    IncompressibleNavierStokesVMSOptions options_{};
    mutable std::optional<EffectiveConfigurationArtifact>
        effective_configuration_artifact_{};
};

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_INCOMPRESSIBLE_NAVIER_STOKES_VMS_MODULE_H
