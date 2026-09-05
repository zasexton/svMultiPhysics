#ifndef SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_OPTIONS_H
#define SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_OPTIONS_H

/**
 * @file FreeSurfaceOptions.h
 * @brief Physical option types for incompressible free-surface formulations.
 */

#include "FE/Constraints/SmallCutAggregationConstraint.h"
#include "FE/Forms/BoundaryConditions.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

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
 * GeneratedCurvatureTraction is an unfitted-only candidate that evaluates a
 * supplied or projected curvature against the normal carried by the same
 * generated-interface rule as the surface measure.
 * KinematicAreaGradientTraction is an unfitted-only total-energy route.  Its
 * prescribed curvature represents the variation of liquid--gas area and all
 * declared Young wetted-wall energies, so no separate equilibrium line force
 * is assembled.
 */
enum class FreeSurfaceSurfaceTensionForm : std::uint8_t {
    Automatic,
    CurvatureTraction,
    SurfaceStress,
    GeneratedCurvatureTraction,
    KinematicAreaGradientTraction
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

struct FreeSurfaceVelocityExtension {
    using ScalarValue = FE::forms::bc::ScalarValue;

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
    // Boundary-local tangential weight. Prescribed uses it for the
    // projected mesh-velocity penalty; SmoothingOnly uses it for the
    // tangential surface-gradient functional. Free intentionally adds no
    // tangential boundary row.
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
};

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_OPTIONS_H
