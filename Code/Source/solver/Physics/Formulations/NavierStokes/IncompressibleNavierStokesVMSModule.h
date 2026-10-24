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
 * The header declares every option type of the formulation: the
 * Navier–Stokes options, the free-surface boundary options and the two-fluid
 * options (D50), with the bulk-integrand types and the public names.
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

#include "FE/Constraints/SmallCutAggregationConstraint.h"
#include "FE/Core/OptionSchema.h"
#include "FE/Forms/BoundaryConditions.h"
#include "FE/Forms/ConstitutiveModel.h"
#include "FE/Forms/FormExpr.h"
#include "FE/Spaces/FunctionSpace.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

// ---------------------------------------------------------------------------
// Free-surface boundary options (physical option types of the incompressible
// free-surface formulations)
// ---------------------------------------------------------------------------

enum class FreeSurfaceImplementation : std::uint8_t {
    FittedALE,
    UnfittedLevelSet
};

/**
 * @brief Ownership role of a free-surface-shaped interface declaration
 *
 * ExteriorOnePhaseBoundary retains the liquid/exterior-pressure boundary
 * contract. InternalMaterialInterfaceVolume is a phase-local implementation
 * detail for a coupled two-fluid owner: it restricts volume and stabilization
 * terms to one cut side but installs no interface traction, kinematic law,
 * pressure anchor, contact law, or boundary-energy declaration.
 */
enum class FreeSurfaceBoundaryRole : std::uint8_t {
    ExteriorOnePhaseBoundary,
    InternalMaterialInterfaceVolume
};

enum class FreeSurfacePhysicalModel : std::uint8_t {
    OnePhaseLiquidPrescribedExteriorPressure
};

enum class FreeSurfaceActiveDomain : std::uint8_t {
    None,
    LevelSetNegative,
    LevelSetPositive
};

// SmoothedIndicator (a smooth-Heaviside weighted bulk integral, explicit
// schema-1 legacy diagnostic) was retired by D42.
enum class FreeSurfaceActiveDomainMethod : std::uint8_t {
    CutVolume
};

// Geometry of a generated (unfitted) interface: the piecewise-linear corner
// geometry or the high-order implicit surface (the FE level-set
// GeneratedInterfaceGeometryMode of the cut request).
enum class FreeSurfaceGeneratedInterfaceGeometry : std::uint8_t {
    LinearCorner,
    HighOrderImplicit
};

// Newton tangent of the generated geometry: quadrature refreshed and frozen
// within a step, or differentiated with respect to the level set (the FE
// level-set GeometryTangentPolicy of the cut request).
enum class FreeSurfaceGeometryTangentPolicy : std::uint8_t {
    RefreshedFrozenQuadrature,
    DifferentiatedQuadrature
};

// Penalty and Nitsche (schema-1 fluid kinematic rows on fitted boundaries)
// were retired by decision D47 Q9.
enum class FreeSurfaceKinematicEnforcement : std::uint8_t {
    None,
    // Fitted ALE only.  The fluid keeps its natural dynamic traction
    // condition on the free surface (no fluid kinematic row).  The normal
    // kinematics (w - u).n = 0 are enforced on the mesh displacement by a
    // consistent Nitsche row: the penalty
    //     gamma_N * deltat_eff / h_n * (w - u).n * (psi.n)
    // installed by the Navier-Stokes module, plus the consistency term that
    // removes the mesh-motion operator's own normal flux, installed by the
    // mesh-motion module on the boundaries this relation declares.
    MeshNitsche
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
 * Automatic selects SurfaceStress, whose measure and normal (the
 * generated-interface rule of an unfitted level-set interface, or the
 * current fitted ALE boundary) define one discrete surface energy (decision
 * D47 Q2(b)).  SurfaceStress is the only capillary route: CurvatureTraction,
 * with its supplied or raw level-set curvature, was retired by the same
 * decision, and KinematicAreaGradientTraction, with its projected curvature
 * field, by decision D48.
 */
enum class FreeSurfaceSurfaceTensionForm : std::uint8_t {
    Automatic,
    SurfaceStress
};

/**
 * @brief Semi-implicit capillary term for the outer generated-state loop
 *
 * NormalIncrement adds the lagged normal-increment term
 *     gamma*dt_eff * int_{Gamma_h} grad_Gamma((u - u_ref).n_h)
 *                                  . grad_Gamma(v.n_h),
 * with grad_Gamma(w.n_h) = P_h (grad w)^T n_h and dt_eff the integrator's
 * 1/a0.  u_ref is the velocity of the iterate that generated Gamma_h; the
 * application overwrites it at every generated-state refresh, so the term
 * vanishes in every freshly refreshed residual and changes only the
 * Jacobian of the frozen inner solves (design note
 * free_surface_semi_implicit_surface_tension_design.md, section 3.3).
 */
enum class FreeSurfaceSurfaceTensionSemiImplicit : std::uint8_t {
    None,
    NormalIncrement
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
    using ScalarValue = FE::forms::bc::ScalarValue;

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

        // Optional sharp Navier slip on the wetted portion of this wall.
        // Omitting the value retains the no-slip prescribed-angle model.
        std::optional<ScalarValue> slip_length{};
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
        // Contact_line_law_enforcement=Strong: the Ren-E law is an essential
        // condition on the contact-point velocity (FE
        // ContactPointVelocityConstraint) instead of the weak line friction
        // and Young term.  Default Weak.
        bool strong_law{false};
    };

    using Configuration =
        std::variant<None, Pinned, PrescribedAngle, DynamicRenE>;

    Configuration configuration{None{}};
};

struct FreeSurfaceCutCellStabilization {
    using ScalarValue = FE::forms::bc::ScalarValue;

    bool enabled{false};
    // The velocity gradient-jump ghost penalty was retired: small-cut
    // aggregation now constrains both velocity and pressure cut-space
    // rows. Pressure jump stabilization and VMS/PSPG remain separate
    // assembled terms; aggregation is not a standalone stability proof
    // for that combined mixed method.
    ScalarValue pressure_gradient_penalty{1.0};
    FreeSurfacePressureStabilizationPolicy pressure_policy{
        FreeSurfacePressureStabilizationPolicy::Enabled};
    bool use_cut_metadata_scale{false};
    std::optional<FE::Real> cut_metadata_scale_cap{};
};

/**
 * @brief Physics-side free-surface relation on a fitted or embedded interface
 *
 * This option only declares Navier-Stokes free-surface equations.  The FE
 * layer remains physics-agnostic and supplies the generic form vocabulary,
 * moving-geometry terminals, and integration domains.
 */
struct FreeSurfaceBoundary {
    using ScalarValue = FE::forms::bc::ScalarValue;

    FreeSurfaceBoundaryRole role{
        FreeSurfaceBoundaryRole::ExteriorOnePhaseBoundary};
    FreeSurfaceImplementation implementation{FreeSurfaceImplementation::FittedALE};

    // Fitted ALE free surfaces integrate on ds(boundary_marker).
    int boundary_marker{-1};

    // Unfitted/level-set free surfaces integrate on dI(interface_marker).
    int interface_marker{-1};
    std::string level_set_field_name{"level_set"};
    std::string generated_interface_domain_id{"free_surface"};
    FreeSurfaceGeneratedInterfaceGeometry generated_interface_geometry{
        FreeSurfaceGeneratedInterfaceGeometry::LinearCorner};
    FreeSurfaceGeometryTangentPolicy geometry_tangent_policy{
        FreeSurfaceGeometryTangentPolicy::RefreshedFrozenQuadrature};
    FE::Real level_set_isovalue{0.0};

    // Optional active-domain restriction for unfitted level-set volume terms.
    FreeSurfaceActiveDomain active_domain{FreeSurfaceActiveDomain::None};
    FreeSurfaceActiveDomainMethod active_domain_method{
        FreeSurfaceActiveDomainMethod::CutVolume};
    // Smoothing width of the DynamicContactAngle wetted-wall weight.  The
    // law constructs a homogeneous
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
    // Optional semi-implicit capillary term (see the enum above).  It is
    // admitted only for unfitted LinearCorner CutVolume interfaces with a
    // literal positive gamma; everything else fails closed.
    FreeSurfaceSurfaceTensionSemiImplicit surface_tension_semi_implicit{
        FreeSurfaceSurfaceTensionSemiImplicit::None};

    // No curvature input: the supplied curvature and the raw level-set
    // curvature of CurvatureTraction (decision D47 Q2) and the projected
    // curvature field of KinematicAreaGradientTraction (decision D48) were
    // retired, and so was the fitted SurfaceStress opt-in: the fitted form
    // integrates gamma (I - n n) : grad(v) over the current boundary.

    // Fitted ALE kinematic relation: (u - meshVelocity()) · n = 0.
    FreeSurfaceNormalKinematicPolicy normal_kinematic_policy{
        FreeSurfaceNormalKinematicPolicy::MatchFluidNormalVelocity};
    FreeSurfaceTangentialMeshPolicy tangential_mesh_policy{
        FreeSurfaceTangentialMeshPolicy::SmoothingOnly};
    std::array<ScalarValue, 3> prescribed_tangential_mesh_velocity{
        ScalarValue{0.0}, ScalarValue{0.0}, ScalarValue{0.0}};
    // Boundary-local tangential weight. Prescribed uses it for the
    // projected mesh-velocity penalty; SmoothingOnly uses it for the
    // tangential surface-gradient functional. Free intentionally adds no
    // tangential boundary row.
    ScalarValue tangential_mesh_penalty{1.0};
    FreeSurfaceKinematicEnforcement kinematic_enforcement{
        FreeSurfaceKinematicEnforcement::None};
    // Local to this fitted free-surface boundary. Generic weak velocity
    // conditions retain the module-level Nitsche policy below.
    FE::Real kinematic_nitsche_gamma{10.0};

    FreeSurfaceCutCellStabilization cut_cell_stabilization{};
    std::vector<FreeSurfaceContactLine> contact_lines{};

    // AgFEM-style small-cut aggregation: slave ill-posed cut-cell
    // velocity and pressure DOFs to polynomial extensions from nearby
    // full-active cells. The row-construction guards bound path length,
    // extrapolation distance, coefficient magnitude, and row norm; no
    // penalty coefficient is tuned by this constraint. Pressure
    // stabilization and VMS/PSPG remain part of the effective mixed
    // method and require joint qualification. Requires order-1
    // vertex-DOF velocity and pressure spaces.
    // Default ON: with the velocity ghost penalty retired, small-cut
    // aggregation is the conditioning mechanism for unfitted cut bands.
    bool small_cut_aggregation{true};
    // Bound the distance and algebraic amplification of every aggregate
    // extension row.  These defaults match the FE constraint contract;
    // keeping them boundary-local makes the effective production state
    // explicit and allows different interfaces to be qualified
    // independently.
    FE::constraints::SmallCutAggregationGuardOptions
        small_cut_aggregation_guards{};

    // The keys of the boundary condition this boundary was read from (empty
    // when it is built in code).  The effective-configuration artifact
    // records from it where each resolved value came from (decision D47).
    FE::options::OptionInput input{};
};

enum class ALEMeshVelocitySource {
    PrescribedData,
    CoupledDisplacement
};

/**
 * @brief Named velocity-row operators for the conservative free-surface split
 *
 * These diagnostic operators are installed only when they are requested
 * (`IncompressibleNavierStokesVMSOptions::conservative_balance_diagnostics`,
 * set for the static-capillary initialization, or the environment switch
 * `SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC`) and the
 * active free-surface configuration uses constant-gamma `SurfaceStress`.
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

/// Prescribed vector field (velocity space) holding u_ref.  The
/// Navier--Stokes module registers it when any free surface requests
/// NormalIncrement; the application refreshes it from the velocity unknown.
inline constexpr std::string_view
    kFreeSurfaceSemiImplicitReferenceVelocityFieldName =
        "ns_free_surface_semi_implicit_reference_velocity";

struct IncompressibleNavierStokesVMSOptions {
    using ScalarValue = FE::forms::bc::ScalarValue;
    using FreeSurfaceContactLine = navier_stokes::FreeSurfaceContactLine;
    using FreeSurfaceCutCellStabilization =
        navier_stokes::FreeSurfaceCutCellStabilization;
    using FreeSurfaceSmallCutAggregationGuards =
        FE::constraints::SmallCutAggregationGuardOptions;
    using FreeSurfaceBoundary = navier_stokes::FreeSurfaceBoundary;

    inline static constexpr int current_configuration_schema_version{2};
    int input_configuration_schema_version{
        current_configuration_schema_version};
    bool explicit_legacy_configuration{false};
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

    // The equation and domain keys of the input this module was translated
    // from, with their resolved values and sources, as members of the
    // artifact's "resolved" section (written by the input translation,
    // decision D47; empty when the options are built in code).
    std::string resolved_input_json{};

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

    // Install the conservative-balance diagnostic operators (see
    // FreeSurfaceConservativeBalanceDiagnosticOperators).  The Application
    // sets it when the static-capillary initialization is configured, since
    // its startup certificate reads them; the environment switch
    // SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC requests them for
    // observation as well.  Observation only: the residual and Jacobian of
    // the production operator do not change.
    bool conservative_balance_diagnostics{false};

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

// ---------------------------------------------------------------------------
// Option predicates the forms read (formerly FreeSurface/FreeSurfaceQueries;
// decisions D50 and D52).  They read only option fields.
// ---------------------------------------------------------------------------

enum class ContactLineKind : std::uint8_t {
    None,
    Pinned,
    PrescribedAngle,
    DynamicRenE
};

[[nodiscard]] inline bool isUnfittedLevelSet(const FreeSurfaceBoundary& bc) noexcept
{
    return bc.implementation == FreeSurfaceImplementation::UnfittedLevelSet;
}

[[nodiscard]] inline bool isExteriorOnePhaseBoundary(
    const FreeSurfaceBoundary& bc) noexcept
{
    return bc.role == FreeSurfaceBoundaryRole::ExteriorOnePhaseBoundary;
}

[[nodiscard]] inline bool isInternalMaterialInterfaceVolume(
    const FreeSurfaceBoundary& bc) noexcept
{
    return bc.role ==
        FreeSurfaceBoundaryRole::InternalMaterialInterfaceVolume;
}

[[nodiscard]] inline bool usesSurfaceStress(
    const FreeSurfaceBoundary& bc) noexcept
{
    switch (bc.surface_tension_form) {
    case FreeSurfaceSurfaceTensionForm::Automatic:
        // The generated-interface rule (unfitted) or the current boundary
        // (fitted) supplies both the measure and its normal, so the default
        // is the variation of one discrete surface energy on both kinds of
        // boundary (decision D47 Q2(b)).
        return true;
    case FreeSurfaceSurfaceTensionForm::SurfaceStress:
        return true;
    }
    return false;
}

[[nodiscard]] inline bool surfaceStressActive(
    const FreeSurfaceBoundary& bc) noexcept
{
    return usesSurfaceStress(bc) &&
           !FE::forms::bc::isZeroConstantScalarValue(bc.surface_tension);
}

inline bool usesSemiImplicitNormalIncrement(const FreeSurfaceBoundary& bc) noexcept
{
    return bc.surface_tension_semi_implicit ==
           FreeSurfaceSurfaceTensionSemiImplicit::NormalIncrement;
}

[[nodiscard]] inline int freeSurfaceMarker(const FreeSurfaceBoundary& bc)
{
    return isUnfittedLevelSet(bc) ? bc.interface_marker : bc.boundary_marker;
}

[[nodiscard]] inline ContactLineKind contactLineKind(
    const FreeSurfaceContactLine& contact_line) noexcept
{
    if (std::holds_alternative<FreeSurfaceContactLine::Pinned>(
            contact_line.configuration)) {
        return ContactLineKind::Pinned;
    }
    if (std::holds_alternative<FreeSurfaceContactLine::PrescribedAngle>(
            contact_line.configuration)) {
        return ContactLineKind::PrescribedAngle;
    }
    if (std::holds_alternative<FreeSurfaceContactLine::DynamicRenE>(
            contact_line.configuration)) {
        return ContactLineKind::DynamicRenE;
    }
    return ContactLineKind::None;
}

[[nodiscard]] inline int contactLineWallBoundaryMarker(
    const FreeSurfaceContactLine& contact_line) noexcept
{
    return std::visit(
        [](const auto& configuration) noexcept -> int {
            using Configuration = std::decay_t<decltype(configuration)>;
            if constexpr (std::is_same_v<Configuration,
                                         FreeSurfaceContactLine::None>) {
                return -1;
            } else {
                return configuration.wall_boundary_marker;
            }
        },
        contact_line.configuration);
}

[[nodiscard]] inline int contactLineMarker(
    const FreeSurfaceContactLine& contact_line) noexcept
{
    return std::visit(
        [](const auto& configuration) noexcept -> int {
            using Configuration = std::decay_t<decltype(configuration)>;
            if constexpr (std::is_same_v<Configuration,
                                         FreeSurfaceContactLine::None>) {
                return -1;
            } else {
                return configuration.contact_line_marker;
            }
        },
        contact_line.configuration);
}

[[nodiscard]] inline const IncompressibleNavierStokesVMSOptions::ScalarValue&
contactLineAngleRadians(const FreeSurfaceContactLine& contact_line)
{
    return std::visit(
        [](const auto& configuration)
            -> const IncompressibleNavierStokesVMSOptions::ScalarValue& {
            using Configuration = std::decay_t<decltype(configuration)>;
            if constexpr (std::is_same_v<
                              Configuration,
                              FreeSurfaceContactLine::PrescribedAngle>) {
                return configuration.contact_angle_radians;
            } else if constexpr (std::is_same_v<
                                     Configuration,
                                     FreeSurfaceContactLine::DynamicRenE>) {
                return configuration.equilibrium_contact_angle_radians;
            } else {
                throw std::logic_error(
                    "contactLineAngleRadians requires an angle-bearing contact configuration");
            }
        },
        contact_line.configuration);
}

[[nodiscard]] inline const IncompressibleNavierStokesVMSOptions::ScalarValue&
contactLineMobility(const FreeSurfaceContactLine& contact_line)
{
    const auto* configuration =
        std::get_if<FreeSurfaceContactLine::DynamicRenE>(
            &contact_line.configuration);
    if (configuration == nullptr) {
        throw std::logic_error("contactLineMobility requires DynamicRenE");
    }
    return configuration->mobility;
}

[[nodiscard]] inline const IncompressibleNavierStokesVMSOptions::ScalarValue&
contactLineSlipLength(const FreeSurfaceContactLine& contact_line)
{
    return std::visit(
        [](const auto& configuration)
            -> const IncompressibleNavierStokesVMSOptions::ScalarValue& {
            using Configuration = std::decay_t<decltype(configuration)>;
            if constexpr (std::is_same_v<
                              Configuration,
                              FreeSurfaceContactLine::PrescribedAngle>) {
                if (!configuration.slip_length.has_value()) {
                    throw std::logic_error(
                        "contactLineSlipLength requires configured Navier slip");
                }
                return *configuration.slip_length;
            } else if constexpr (std::is_same_v<
                                     Configuration,
                                     FreeSurfaceContactLine::DynamicRenE>) {
                return configuration.slip_length;
            } else {
                throw std::logic_error(
                    "contactLineSlipLength requires configured Navier slip");
            }
        },
        contact_line.configuration);
}

[[nodiscard]] inline bool contactLineHasNavierSlip(
    const FreeSurfaceContactLine& contact_line) noexcept
{
    return std::visit(
        [](const auto& configuration) noexcept {
            using Configuration = std::decay_t<decltype(configuration)>;
            if constexpr (std::is_same_v<
                              Configuration,
                              FreeSurfaceContactLine::PrescribedAngle>) {
                return configuration.slip_length.has_value();
            } else {
                return std::is_same_v<Configuration,
                                      FreeSurfaceContactLine::DynamicRenE>;
            }
        },
        contact_line.configuration);
}

[[nodiscard]] inline int contactLineConstraintMarker(
    const IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine& contact_line)
{
    return contactLineMarker(contact_line) >= 0
               ? contactLineMarker(contact_line)
               : contactLineWallBoundaryMarker(contact_line);
}

[[nodiscard]] inline FE::Real constantScalarValueOrThrow(
    const IncompressibleNavierStokesVMSOptions::ScalarValue& value,
    std::string_view context)
{
    const auto* real = std::get_if<FE::Real>(&value);
    if (real == nullptr) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: " + std::string(context) +
            " currently requires a literal scalar value");
    }
    return *real;
}

[[nodiscard]] inline bool hasWallAwareContactLaw(
    const FreeSurfaceBoundary& boundary)
{
    return std::any_of(
        boundary.contact_lines.begin(),
        boundary.contact_lines.end(),
        [](const auto& contact_line) {
            const auto kind = contactLineKind(contact_line);
            return kind == ContactLineKind::PrescribedAngle ||
                   kind == ContactLineKind::DynamicRenE;
        });
}

[[nodiscard]] inline const std::array<IncompressibleNavierStokesVMSOptions::ScalarValue, 3>&
contactLineWallNormal(const FreeSurfaceContactLine& contact_line)
{
    return std::visit(
        [](const auto& configuration)
            -> const std::array<
                IncompressibleNavierStokesVMSOptions::ScalarValue, 3>& {
            using Configuration = std::decay_t<decltype(configuration)>;
            if constexpr (std::is_same_v<
                              Configuration,
                              FreeSurfaceContactLine::PrescribedAngle> ||
                          std::is_same_v<
                              Configuration,
                              FreeSurfaceContactLine::DynamicRenE>) {
                return configuration.wall_normal;
            } else {
                throw std::logic_error(
                    "contactLineWallNormal requires an angle-bearing contact configuration");
            }
        },
        contact_line.configuration);
}

[[nodiscard]] inline std::array<FE::Real, 3> normalizedWallNormal(
    const IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine& contact_line)
{
    std::array<FE::Real, 3> normal{
        constantScalarValueOrThrow(contactLineWallNormal(contact_line)[0], "contact-line wall_normal"),
        constantScalarValueOrThrow(contactLineWallNormal(contact_line)[1], "contact-line wall_normal"),
        constantScalarValueOrThrow(contactLineWallNormal(contact_line)[2], "contact-line wall_normal")};
    if (!std::all_of(normal.begin(), normal.end(), [](FE::Real component) {
            return std::isfinite(component);
        })) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: prescribed contact angle requires finite wall_normal components");
    }
    const auto norm = std::sqrt(normal[0] * normal[0] +
                                normal[1] * normal[1] +
                                normal[2] * normal[2]);
    if (!(norm > FE::Real{0.0}) || !std::isfinite(norm)) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: prescribed contact angle requires a finite, nonzero wall_normal");
    }
    for (auto& component : normal) {
        component /= norm;
    }
    return normal;
}

[[nodiscard]] inline bool isRefreshedFrozenGeometryTangent(
    const FreeSurfaceBoundary& bc)
{
    return bc.geometry_tangent_policy ==
           FreeSurfaceGeometryTangentPolicy::RefreshedFrozenQuadrature;
}

[[nodiscard]] inline bool isHighOrderImplicitGeneratedInterface(
    const FreeSurfaceBoundary& bc)
{
    return bc.generated_interface_geometry ==
           FreeSurfaceGeneratedInterfaceGeometry::HighOrderImplicit;
}

[[nodiscard]] inline const char* pressureStabilizationDisabledReason(
    const FreeSurfaceBoundary& bc)
{
    switch (bc.cut_cell_stabilization.pressure_policy) {
    case FreeSurfacePressureStabilizationPolicy::Enabled:
    case FreeSurfacePressureStabilizationPolicy::Incremental:
        return "none";
    case FreeSurfacePressureStabilizationPolicy::Disabled:
        return "explicit_policy_disabled";
    case FreeSurfacePressureStabilizationPolicy::DisabledForRefreshedFrozenHighOrder:
        break;
    }
    return isHighOrderImplicitGeneratedInterface(bc) &&
                   isRefreshedFrozenGeometryTangent(bc)
               ? "refreshed_frozen_high_order_policy"
               : "none";
}

[[nodiscard]] inline bool pressureStabilizationDisabledByPolicy(
    const FreeSurfaceBoundary& bc)
{
    return std::string_view(pressureStabilizationDisabledReason(bc)) != "none";
}

[[nodiscard]] inline bool pressureStabilizationActive(const FreeSurfaceBoundary& bc)
{
    return !pressureStabilizationDisabledByPolicy(bc) &&
           !FE::forms::bc::isZeroConstantScalarValue(
               bc.cut_cell_stabilization.pressure_gradient_penalty);
}

[[nodiscard]] inline FE::forms::CutVolumeSide activeDomainSide(
    FreeSurfaceActiveDomain domain) noexcept
{
    switch (domain) {
    case FreeSurfaceActiveDomain::None:
    case FreeSurfaceActiveDomain::LevelSetNegative:
        return FE::forms::CutVolumeSide::Negative;
    case FreeSurfaceActiveDomain::LevelSetPositive:
        return FE::forms::CutVolumeSide::Positive;
    }
    return FE::forms::CutVolumeSide::Negative;
}

// ---------------------------------------------------------------------------
// Two-fluid options (the two-fluid mode of IncompressibleNavierStokesVMSModule)
// ---------------------------------------------------------------------------

struct IncompressibleTwoFluidPhaseOptions {
    std::string velocity_field_name{};
    std::string pressure_field_name{};
    FE::Real density{1.0};
    FE::Real viscosity{1.0};
    std::vector<IncompressibleNavierStokesVMSOptions::VelocityDirichletBC>
        velocity_dirichlet{};
};

struct IncompressibleTwoFluidSharedPressureGauge {
    FE::GlobalIndex vertex_gid{-1};
    FE::Real pressure{0.0};
};

/**
 * @brief Initial supported envelope for a sharp incompressible two-fluid pair
 *
 * The current owner intentionally admits only fixed-Eulerian affine C0 P1
 * Triangle3/Tetra4 spaces, constant phase materials, one shared generated
 * interface, CutVolume phase restriction, phasewise VMS/PSPG and ghost
 * stabilization, and phasewise small-cut aggregation. Compressible gas,
 * contact, phase change, turbulence, variable material laws, and moving-mesh
 * combinations remain outside this contract.
 */
struct IncompressibleTwoFluidOptions {
    IncompressibleTwoFluidPhaseOptions negative_phase{
        .velocity_field_name = "u_negative",
        .pressure_field_name = "p_negative",
        .density = 1000.0,
        .viscosity = 0.001,
    };
    IncompressibleTwoFluidPhaseOptions positive_phase{
        .velocity_field_name = "u_positive",
        .pressure_field_name = "p_positive",
        .density = 1.0,
        .viscosity = 0.00001,
    };

    /**
     * Physical exterior velocity data applied identically to both phase
     * restrictions. Phase-local velocity data remain homogeneous-only.
     */
    std::vector<IncompressibleNavierStokesVMSOptions::VelocityDirichletBC>
        shared_velocity_dirichlet{};

    /** Optional partition-invariant anchor for the shared pressure mode. */
    std::optional<IncompressibleTwoFluidSharedPressureGauge>
        shared_pressure_gauge{};

    std::string operator_tag{"equations"};
    std::string level_set_field_name{"level_set"};
    std::string generated_interface_domain_id{"two_fluid_interface"};
    FreeSurfaceGeneratedInterfaceGeometry generated_interface_geometry{
        FreeSurfaceGeneratedInterfaceGeometry::LinearCorner};
    FreeSurfaceGeometryTangentPolicy geometry_tangent_policy{
        FreeSurfaceGeometryTangentPolicy::RefreshedFrozenQuadrature};
    int interface_marker{-1};
    FE::Real level_set_isovalue{0.0};

    FE::Real surface_tension{0.0};
    /** Optional manufactured p_minus-p_plus load and diagnostic target. */
    std::optional<FE::Real> prescribed_pressure_jump{};
    /** Optional manufactured (tau_minus-tau_plus)n target in global axes. */
    std::optional<std::array<FE::Real, 3>>
        prescribed_viscous_traction_jump{};
    FE::Real interface_nitsche_gamma{20.0};
    bool include_transient_interface_penalty{true};
    bool require_conservative_phase_momentum_reconciliation{false};

    std::array<FE::Real, 3> body_force{0.0, 0.0, 0.0};
    bool enable_convection{true};
    bool enable_vms{true};
    FE::Real ct_m{1.0};
    FE::Real ct_c{36.0};
    FE::Real stabilization_epsilon{1.0e-12};

    FE::Real pressure_gradient_penalty{1.0};
    bool use_cut_metadata_scale{false};
    std::optional<FE::Real> cut_metadata_scale_cap{};
    IncompressibleNavierStokesVMSOptions::FreeSurfaceSmallCutAggregationGuards
        small_cut_aggregation_guards{};
    core::PhysicsJITPolicy jit_policy{};

    // The two-fluid equation and domain keys of the input, with their
    // resolved values and sources, as members of the artifact's "resolved"
    // section (written by the input translation, decision D47; empty when the
    // options are built in code).
    std::string resolved_input_json{};
};

// ---------------------------------------------------------------------------
// The bulk residual of the incompressible Navier-Stokes VMS formulation
// ---------------------------------------------------------------------------
//   R_m = rho (dt u + (grad u) a + chi - f) - div(2 mu eps(u)) + grad p,   a = u - w,
//   u' = -tau_M R_m,   p' = -tau_C div u          (static subscales)
// The module integrates these integrands over the liquid (the mesh, or the
// cut side of a level-set geometry); the two-fluid module can integrate them
// over each phase.  The integrands are returned unintegrated so that the
// caller chooses the volume domain.

/// The unknowns, test functions and material data of one fluid.
struct NavierStokesTerms {
    FE::forms::FormExpr u, p;   // velocity, pressure
    FE::forms::FormExpr v, q;   // their test functions
    FE::forms::FormExpr rho;    // density
    FE::forms::FormExpr mu;     // dynamic viscosity, mu or mu(shear rate)
    FE::forms::FormExpr f;      // body acceleration
    int dim{3};
};

struct NavierStokesVMSSettings {
    bool convection{true};
    bool ale{false};
    bool moving_control_volume{false};      // chi = div(w) u (ALE only)
    bool vms{true};
    FE::Real ct_m{1.0};
    FE::Real ct_c{36.0};
    FE::Real epsilon{1.0e-12};
};

/// Integrands of the bulk residual.
struct NavierStokesVMSIntegrands {
    FE::forms::FormExpr momentum;                // Galerkin + SUPG + LSIC + cross stress
    FE::forms::FormExpr continuity;              // q div u + PSPG without the pressure gradient
    FE::forms::FormExpr pspg_pressure_gradient;  // tau_M grad q . grad p (its own kernel group)
    FE::forms::FormExpr pspg_without_pressure_gradient;   // tau_M grad q . (R_m - grad p)
    FE::forms::FormExpr galerkin_momentum;
    FE::forms::FormExpr galerkin_continuity;
    // Galerkin parts named for the free-surface residual-work ledger.
    FE::forms::FormExpr convection;              // moving control volume + convection
    FE::forms::FormExpr pressure;                // -p div v
    FE::forms::FormExpr forcing;                 // -rho f . v
};

[[nodiscard]] NavierStokesVMSIntegrands navierStokesVMSIntegrands(
    const NavierStokesTerms& terms,
    const NavierStokesVMSSettings& settings);

class IncompressibleNavierStokesVMSModule final : public PhysicsModule {
public:
    IncompressibleNavierStokesVMSModule(std::shared_ptr<const FE::spaces::FunctionSpace> velocity_space,
                                        std::shared_ptr<const FE::spaces::FunctionSpace> pressure_space,
                                        IncompressibleNavierStokesVMSOptions options = {});

    /// The two-fluid mode: one incompressible fluid on each side of a
    /// generated material interface (the negative and the positive phase),
    /// coupled on the interface and through one shared pressure gauge.
    IncompressibleNavierStokesVMSModule(std::shared_ptr<const FE::spaces::FunctionSpace> negative_velocity_space,
                                        std::shared_ptr<const FE::spaces::FunctionSpace> negative_pressure_space,
                                        std::shared_ptr<const FE::spaces::FunctionSpace> positive_velocity_space,
                                        std::shared_ptr<const FE::spaces::FunctionSpace> positive_pressure_space,
                                        IncompressibleTwoFluidOptions options = {});

    void registerOn(FE::systems::FESystem& system) const override;
    void applyInitialConditions(const FE::systems::FESystem& system,
                                FE::backends::GenericVector& u0) const override;
    [[nodiscard]] std::optional<EffectiveConfigurationArtifact>
    effectiveConfigurationArtifact() const override;

private:
    void registerTwoFluidOn(FE::systems::FESystem& system) const;
    void applyTwoFluidInitialConditions(const FE::systems::FESystem& system,
                                        FE::backends::GenericVector& u0) const;

    std::shared_ptr<const FE::spaces::FunctionSpace> velocity_space_{};   ///< the negative phase's in the two-fluid mode
    std::shared_ptr<const FE::spaces::FunctionSpace> pressure_space_{};
    std::shared_ptr<const FE::spaces::FunctionSpace> positive_velocity_space_{};   ///< two-fluid mode only
    std::shared_ptr<const FE::spaces::FunctionSpace> positive_pressure_space_{};
    IncompressibleNavierStokesVMSOptions options_{};
    std::optional<IncompressibleTwoFluidOptions> two_fluid_options_{};   ///< set in the two-fluid mode
    mutable std::optional<EffectiveConfigurationArtifact>
        effective_configuration_artifact_{};
};

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_INCOMPRESSIBLE_NAVIER_STOKES_VMS_MODULE_H
