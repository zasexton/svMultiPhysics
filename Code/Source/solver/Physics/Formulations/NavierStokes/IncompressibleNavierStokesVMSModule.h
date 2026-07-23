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

enum class FreeSurfaceImplementation : std::uint8_t {
    FittedALE,
    UnfittedLevelSet
};

enum class FreeSurfaceActiveDomain : std::uint8_t {
    None,
    LevelSetNegative,
    LevelSetPositive
};

enum class FreeSurfaceActiveDomainMethod : std::uint8_t {
    CutVolume,
    SmoothedIndicator
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

enum class FreeSurfacePressureStabilizationPolicy : std::uint8_t {
    Enabled,
    Incremental,
    Disabled,
    DisabledForRefreshedFrozenHighOrder
};

/**
 * @brief Discrete representation of constant surface-tension forces
 *
 * Automatic selects SurfaceStress for unfitted level-set interfaces, where
 * the generated-interface measure and normal define one discrete surface
 * energy, and retains CurvatureTraction for fitted ALE boundaries for
 * backwards compatibility.  CurvatureTraction remains available explicitly
 * for verification and legacy supplied-curvature data.
 */
enum class FreeSurfaceSurfaceTensionForm : std::uint8_t {
    Automatic,
    CurvatureTraction,
    SurfaceStress
};

/**
 * @brief Named velocity-row operators for the conservative free-surface split
 *
 * These diagnostic operators are installed only when
 * `SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC` is enabled and the
 * active free-surface configuration uses constant-gamma `SurfaceStress`.
 * Their residual vectors satisfy `conservative_balance = pressure_virtual_work
 * + surface_energy_virtual_work` on the same generated cut/contact rules as
 * the production momentum residual.
 */
struct FreeSurfaceConservativeBalanceDiagnosticOperators {
    inline static constexpr std::string_view pressure_virtual_work{
        "equations_diagnostic_ns_free_surface_pressure_virtual_work"};
    inline static constexpr std::string_view surface_energy_virtual_work{
        "equations_diagnostic_ns_free_surface_surface_energy_virtual_work"};
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

struct IncompressibleNavierStokesVMSOptions {
    using ScalarValue = FE::forms::bc::ScalarValue;

    inline static constexpr int current_configuration_schema_version{2};
    int input_configuration_schema_version{
        current_configuration_schema_version};
    bool explicit_legacy_configuration{false};

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
     * @brief Physics-side model for the wall intersection of one free surface
     *
     * The free-surface boundary owns these declarations.  FE geometry remains
     * generic: later fitted and unfitted implementations can use the wall marker,
     * optional contact-line marker, and scalar model parameters to install the
     * appropriate Physics residuals or constraints.
     */
    struct FreeSurfaceContactLine {
        struct None {};

        struct Pinned {
            int wall_boundary_marker{-1};
            int contact_line_marker{-1};
        };

        struct PrescribedAngle {
            int wall_boundary_marker{-1};
            int contact_line_marker{-1};

            // Radians, measured through the liquid phase. wall_normal points
            // outward from the liquid and into the solid. With n the outward
            // liquid-interface normal, Young geometry is
            //     n . wall_normal = -cos(contact_angle_radians).
            ScalarValue contact_angle_radians{1.57079632679489661923};
            std::array<ScalarValue, 3> wall_normal{
                ScalarValue{0.0}, ScalarValue{0.0}, ScalarValue{0.0}};
        };

        struct DynamicRenE {
            int wall_boundary_marker{-1};
            int contact_line_marker{-1};

            // Ren--E sharp-interface law:
            //     xi V_CL = gamma (cos(theta_e) - cos(theta_d)),
            //     xi = 1 / mobility.
            // Navier wall slip is intrinsic to this alternative, so an
            // invalid dynamic/no-slip combination is not representable.
            ScalarValue equilibrium_contact_angle_radians{
                1.57079632679489661923};
            std::array<ScalarValue, 3> wall_normal{
                ScalarValue{0.0}, ScalarValue{0.0}, ScalarValue{0.0}};
            ScalarValue mobility{0.0};
            ScalarValue slip_length{0.0};
        };

        using Configuration =
            std::variant<None, Pinned, PrescribedAngle, DynamicRenE>;

        Configuration configuration{None{}};
    };

    struct FreeSurfaceCutCellStabilization {
        bool enabled{false};
        // The velocity gradient-jump ghost penalty was retired: small-cut
        // aggregation (SmallCutAggregationConstraint) replaces it for
        // conditioning, parameter-free. Pressure jump stabilization remains.
        ScalarValue pressure_gradient_penalty{1.0};
        FreeSurfacePressureStabilizationPolicy pressure_policy{
            FreeSurfacePressureStabilizationPolicy::Enabled};
        bool use_cut_metadata_scale{false};
        std::optional<FE::Real> cut_metadata_scale_cap{};
    };

    struct FreeSurfaceSmallCutAggregationGuards {
        // Bound the distance and algebraic amplification of every aggregate
        // extension row.  These defaults match the FE constraint contract;
        // keeping them boundary-local makes the effective production state
        // explicit and allows different interfaces to be qualified
        // independently.
        std::size_t maximum_root_path_length{8u};
        FE::Real maximum_reference_extrapolation_distance{4.0};
        FE::Real maximum_absolute_coefficient{16.0};
        FE::Real maximum_row_l1_norm{32.0};
    };

    struct FreeSurfaceVelocityExtension {
        bool enabled{false};
        ScalarValue diffusivity{1.0};
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
        std::string generated_interface_domain_id{"free_surface"};
        std::string generated_interface_geometry{"LinearCorner"};
        std::string geometry_tangent_policy{"RefreshedFrozenQuadrature"};
        FE::Real level_set_isovalue{0.0};

        // Optional active-domain restriction for unfitted level-set volume terms.
        FreeSurfaceActiveDomain active_domain{FreeSurfaceActiveDomain::None};
        FreeSurfaceActiveDomainMethod active_domain_method{
            FreeSurfaceActiveDomainMethod::CutVolume};
        // Width of the supported algebraic smooth Heaviside, whose tails are
        // noncompact.  The diagnostic bulk SmoothedIndicator path retains its
        // legacy assumption that phi is already a signed-distance field.  The
        // DynamicContactAngle wetted-wall law instead constructs a homogeneous
        // physical signed-distance proxy and applies a compact C1 cubic: zero
        // on the dry side beyond one width, one on the wet side beyond one
        // width.  Positive phi rescaling is therefore immaterial.  This is a
        // smooth wall weight, not a sharp cut-wall integration operator.  Zero
        // selects the local cell diameter.
        FE::Real active_domain_smoothing_width{0.0};
        bool allow_full_domain_unfitted_free_surface{false};

        // Dynamic stress balance: sigma(u,p)n = (-p_ext - gamma*kappa)n.
        // Curvature is positive for a convex liquid surface with n pointing
        // outward from the active liquid, so static equilibrium gives the
        // Young--Laplace jump p_liquid - p_ext = gamma*kappa.
        // For unfitted level-set boundaries, supplied curvature values and
        // curvature fields are signed with grad(phi)/|grad(phi)|; the
        // Navier-Stokes module converts them to the active-domain outward
        // normal convention used by n before forming the traction.
        ScalarValue external_pressure{0.0};
        // Literal constant only.  Variable surface tension requires the
        // tangential Marangoni term grad_Gamma(gamma), which is not currently
        // implemented and therefore fails closed during module validation.
        ScalarValue surface_tension{0.0};

        // SurfaceStress uses the variational surface-energy force
        //     gamma (I - n_g tensor n_g) : grad(v)
        // with n_g supplied by the same generated-interface rule as dI.  At a
        // moving contact line it must be paired with wall energy
        // -gamma*cos(theta_e), not with a second cos(theta_d) force.
        FreeSurfaceSurfaceTensionForm surface_tension_form{
            FreeSurfaceSurfaceTensionForm::Automatic};

        // Fitted ALE can use current mesh geometry or a supplied curvature
        // expression/value.  Unfitted level-set surface tension requires a
        // supplied curvature until projected/smoothed curvature is validated.
        ScalarValue curvature{0.0};
        std::string curvature_field_name{};
        bool use_current_geometry_curvature{false};
        bool use_level_set_curvature{true};

        // Fitted ALE kinematic relation: (u - meshVelocity()) · n = 0.
        FreeSurfaceNormalKinematicPolicy normal_kinematic_policy{
            FreeSurfaceNormalKinematicPolicy::MatchFluidNormalVelocity};
        FreeSurfaceTangentialMeshPolicy tangential_mesh_policy{
            FreeSurfaceTangentialMeshPolicy::SmoothingOnly};
        std::array<ScalarValue, 3> prescribed_tangential_mesh_velocity{
            ScalarValue{0.0}, ScalarValue{0.0}, ScalarValue{0.0}};
        // Weak penalty applied to the tangential mesh-velocity trace when
        // tangential_mesh_policy is Prescribed.  Free and SmoothingOnly add
        // no tangential boundary row.
        ScalarValue tangential_mesh_penalty{1.0};
        FreeSurfaceKinematicEnforcement kinematic_enforcement{
            FreeSurfaceKinematicEnforcement::None};
        ScalarValue kinematic_penalty{0.0};
        // Local to this fitted free-surface boundary. Generic weak velocity
        // conditions retain the module-level Nitsche policy below.
        FE::Real kinematic_nitsche_gamma{10.0};
        bool kinematic_nitsche_symmetric{true};
        bool kinematic_nitsche_scale_with_p{true};

        FreeSurfaceCutCellStabilization cut_cell_stabilization{};
        FreeSurfaceVelocityExtension velocity_extension{};
        std::vector<FreeSurfaceContactLine> contact_lines{};

        // AgFEM-style small-cut aggregation: slave ill-posed cut-cell
        // velocity DOFs to the polynomial extension of nearby full-active
        // cells (parameter-free conditioning control). When enabled the
        // velocity ghost-penalty terms are skipped — aggregation replaces
        // their job. Requires an order-1 (vertex-DOF) velocity space.
        // Default ON: with the velocity ghost penalty retired, small-cut
        // aggregation is the conditioning mechanism for unfitted cut bands.
        bool small_cut_aggregation{true};
        FreeSurfaceSmallCutAggregationGuards
            small_cut_aggregation_guards{};
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
