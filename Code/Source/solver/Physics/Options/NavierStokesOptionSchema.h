/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */


#ifndef SVMP_PHYSICS_OPTIONS_NAVIER_STOKES_OPTION_SCHEMA_H
#define SVMP_PHYSICS_OPTIONS_NAVIER_STOKES_OPTION_SCHEMA_H

/**
 * @file NavierStokesOptionSchema.h
 * @brief The Navier-Stokes option schema, the names of option values and the
 *        effective-configuration artifact
 *
 * Moved from the formulation folder (decisions D50 and D52):
 *  - the input keys of a free-surface boundary condition and of its contact
 *    lines as option schemas (formerly FreeSurface/FreeSurfaceBoundarySchema;
 *    decision D47, step P1.2a).  The register resolves a Free_surface
 *    boundary condition with them (step P2), and the "resolved" section of
 *    the artifact records where each value came from;
 *  - the input, artifact and log names of the free-surface option values,
 *    the mapping of the fitted tangential mesh policy to the FE system
 *    policy and the descriptor texts of the fitted rows (formerly
 *    FreeSurface/FreeSurfaceQueries);
 *  - the effective-configuration artifact (version 4) of a registration,
 *    written by the ApplicationDriver next to the results (formerly
 *    FreeSurface/FreeSurfaceConfigurationArtifact).
 */

#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"
#include "Physics/Core/EquationModuleInput.h"

#include "FE/Core/OptionSchema.h"
#include "FE/Core/Types.h"
#include "FE/Forms/BoundaryConditions.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace svmp::FE::systems {
class FESystem;
enum class MeshTangentialBoundaryPolicy : std::uint8_t;
} // namespace svmp::FE::systems

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

namespace Factories {
struct IncompressibleTwoFluidInterfaceWeights;
} // namespace Factories

/// A scalar boundary value: the input is a real number (the free-surface
/// input reads only steady constants); other kinds are printed by name.
struct ScalarValueCodec {
    using value_type = FE::forms::bc::ScalarValue;
    static constexpr FE::options::EmptyPolicy empty_policy = FE::options::EmptyPolicy::Parse;
    FE::options::RealCodec real{};

    [[nodiscard]] value_type parse(std::string_view text, std::string& error) const;
    [[nodiscard]] std::string format(const value_type& value) const;
    [[nodiscard]] std::string describe() const { return real.describe(); }
};

/// Three scalar boundary values (commas and/or white space between them).
struct ScalarVector3Codec {
    using value_type = std::array<FE::forms::bc::ScalarValue, 3>;
    static constexpr FE::options::EmptyPolicy empty_policy = FE::options::EmptyPolicy::Absent;

    [[nodiscard]] value_type parse(std::string_view text, std::string& error) const;
    [[nodiscard]] std::string format(const value_type& value) const;
    [[nodiscard]] std::string describe() const { return "3 reals"; }
};

/// The keys that fill a FreeSurfaceBoundary, in the register's order.
[[nodiscard]] const FE::options::OptionSchema<FreeSurfaceBoundary>& freeSurfaceBoundarySchema();

enum class FreeSurfaceContactLineModelInput : std::uint8_t {
    None,
    Pinned,
    PrescribedAngle,
    DynamicRenE
};

enum class FreeSurfaceWallSlipModelInput : std::uint8_t {
    None,
    Navier
};

enum class FreeSurfaceContactLawEnforcementInput : std::uint8_t {
    Weak,
    Strong
};

/// The contact-line keys of one free-surface boundary, as one value: the
/// register expands them into FreeSurfaceBoundary::contact_lines (one line
/// per wall marker), and contactLineInputView() folds the lines back.
struct FreeSurfaceContactLineInputView {
    FreeSurfaceContactLineModelInput model{FreeSurfaceContactLineModelInput::None};
    FE::Real contact_angle_degrees{90.0};
    std::vector<int> wall_markers{};
    int contact_line_marker{-1};
    std::vector<std::array<FE::Real, 3>> wall_normals{};
    FE::forms::bc::ScalarValue mobility{FE::Real{0.0}};
    FreeSurfaceWallSlipModelInput wall_slip_model{FreeSurfaceWallSlipModelInput::None};
    FE::forms::bc::ScalarValue wall_slip_length{FE::Real{0.0}};
    FreeSurfaceContactLawEnforcementInput law_enforcement{
        FreeSurfaceContactLawEnforcementInput::Weak};
};

[[nodiscard]] FreeSurfaceContactLineInputView contactLineInputView(const FreeSurfaceBoundary& boundary);

/// The contact-line keys, in the register's order.
[[nodiscard]] const FE::options::OptionSchema<FreeSurfaceContactLineInputView>&
freeSurfaceContactLineInputSchema();

/// The defined entries of a parameter map, as schema input.
[[nodiscard]] FE::options::OptionInput optionInputFromParameters(const ParameterMap& params);

/// The Navier-Stokes keys of an equation element and of its domain element,
/// as one value (decision D47, P2): navierStokesEquationSchema() reads the
/// keys of the equation, navierStokesDomainSchema() those of the domain.
/// Keys of the moving domain, the momentum source and the rotating frame
/// belong to both elements; the register merges the two values (the domain
/// value of the moving domain and the momentum source replaces the equation
/// value; the equation value of the rotating frame and of the momentum
/// source file replaces the domain value).
struct NavierStokesInputView {
    // Equation element.
    std::string physical_model{"OnePhaseLiquidPrescribedExteriorPressure"};
    bool use_taylor_hood_type_basis{false};
    // Equation and domain elements.
    bool enable_ale{false};
    std::string mesh_velocity_field{"mesh_velocity"};
    ALEMeshVelocitySource mesh_velocity_source{ALEMeshVelocitySource::PrescribedData};
    std::string mesh_displacement_field{"mesh_displacement"};
    bool auto_register_mesh_displacement_field{false};
    std::string momentum_source_field_name{};
    bool auto_register_momentum_source_field{true};
    std::string momentum_source_file{};
    std::array<FE::Real, 3> rotating_frame_angular_velocity{0.0, 0.0, 0.0};
    std::string rotating_frame_angular_velocity_file{};
    // Domain element.
    FE::Real density{1.0};
    std::array<FE::Real, 3> body_force{0.0, 0.0, 0.0};
    bool hydrostatic_pressure_initialization{false};
    FE::Real hydrostatic_pressure_reference{0.0};
    std::array<FE::Real, 3> hydrostatic_pressure_reference_point{0.0, 0.0, 0.0};
    std::string viscosity_model{};
    FE::Real viscosity_value{0.0};
    FE::Real limiting_high_shear_rate_viscosity{0.0};
    FE::Real limiting_low_shear_rate_viscosity{0.0};
    FE::Real shear_rate_tensor_multiplier{0.0};
    FE::Real power_law_index{0.0};
    FE::Real shear_rate_tensor_exponent{0.0};
    FE::Real backflow_stabilization_coefficient{0.0};
};

/// The keys of the equation element of a one-phase Navier-Stokes input.
[[nodiscard]] const FE::options::OptionSchema<NavierStokesInputView>& navierStokesEquationSchema();

/// The keys of the domain element of a one-phase Navier-Stokes input.
[[nodiscard]] const FE::options::OptionSchema<NavierStokesInputView>& navierStokesDomainSchema();

/// The keys of the equation element of the incompressible two-fluid mode
/// (Free_surface_physical_model=IncompressibleTwoFluid).
struct IncompressibleTwoFluidInputView {
    std::string physical_model{"IncompressibleTwoFluid"};
    std::string level_set_field_name{"level_set"};
    std::string generated_interface_domain_id{"two_fluid_interface"};
    int material_interface_marker{-1};
    FE::Real negative_phase_density{1000.0};
    FE::Real negative_phase_dynamic_viscosity{0.001};
    FE::Real positive_phase_density{1.0};
    FE::Real positive_phase_dynamic_viscosity{0.00001};
    FE::Real two_fluid_surface_tension{0.0};
    FE::Real two_fluid_interface_nitsche_gamma{20.0};
    std::optional<FE::Real> prescribed_pressure_jump{};
    std::optional<FE::Real> prescribed_viscous_traction_jump_x{};
    std::optional<FE::Real> prescribed_viscous_traction_jump_y{};
    std::optional<FE::Real> prescribed_viscous_traction_jump_z{};
    bool use_taylor_hood_type_basis{false};
};

[[nodiscard]] const FE::options::OptionSchema<IncompressibleTwoFluidInputView>&
incompressibleTwoFluidSchema();

/// The keys of the default domain of the two-fluid mode (the body force).
[[nodiscard]] const FE::options::OptionSchema<NavierStokesInputView>&
incompressibleTwoFluidDomainSchema();

/// The "resolved" section of the Navier-Stokes artifact, for the boundaries
/// in the given order:
///   {"version":1,"free_surfaces":[{"boundary":{...},"contact_line":{...}},...]
///    [,"equation":{...},"domain":{...}]}
/// Each key carries its value and its source (OptionSchema::describe against
/// the boundary's stored input).  The equation and domain members are the
/// options' resolved_input_json (written by the input translation; absent
/// for options built in code).
[[nodiscard]] std::string resolvedFreeSurfacesJson(const std::vector<FreeSurfaceBoundary>& free_surfaces,
                                                   const std::string& resolved_input_json = {});

[[nodiscard]] const char* freeSurfaceGeneratedInterfaceGeometryName(
    FreeSurfaceGeneratedInterfaceGeometry geometry) noexcept;

[[nodiscard]] const char* freeSurfaceGeometryTangentPolicyName(
    FreeSurfaceGeometryTangentPolicy policy) noexcept;

[[nodiscard]] const char* freeSurfaceSurfaceTensionSemiImplicitName(
    FreeSurfaceSurfaceTensionSemiImplicit value) noexcept;

[[nodiscard]] std::string freeSurfaceValueName(std::string_view prefix,
                                               const FreeSurfaceBoundary& bc);

[[nodiscard]] std::string contactWallLawName(
    const FreeSurfaceContactLine& contact_line);

[[nodiscard]] std::string freeSurfaceCurvaturePolicyName(
    const FreeSurfaceBoundary& bc);

[[nodiscard]] std::string freeSurfaceCurvatureTangentPolicyName(
    const FreeSurfaceBoundary& bc);

[[nodiscard]] const char* activeDomainName(FreeSurfaceActiveDomain domain) noexcept;

[[nodiscard]] const char* kinematicEnforcementName(
    FreeSurfaceKinematicEnforcement enforcement) noexcept;

[[nodiscard]] const char* cutVolumeSideName(FE::forms::CutVolumeSide side) noexcept;

[[nodiscard]] FE::systems::MeshTangentialBoundaryPolicy
systemTangentialMeshPolicy(FreeSurfaceTangentialMeshPolicy policy);

[[nodiscard]] std::string fittedTangentialOperatorSource(
    int boundary_marker,
    FreeSurfaceTangentialMeshPolicy policy);

[[nodiscard]] std::string fittedFluidNormalDescriptorSource(
    const FreeSurfaceBoundary& bc);

[[nodiscard]] const char* freeSurfaceImplementationName(
    FreeSurfaceImplementation implementation) noexcept;

[[nodiscard]] const char* freeSurfaceBoundaryRoleName(
    FreeSurfaceBoundaryRole role) noexcept;

[[nodiscard]] const char* freeSurfacePhysicalModelName(
    FreeSurfacePhysicalModel model) noexcept;

[[nodiscard]] const char* normalKinematicPolicyName(
    FreeSurfaceNormalKinematicPolicy policy) noexcept;

[[nodiscard]] const char* tangentialMeshPolicyName(
    FreeSurfaceTangentialMeshPolicy policy) noexcept;

[[nodiscard]] const char* contactLineKindName(ContactLineKind kind) noexcept;

[[nodiscard]] const char* geometryTangentPathName(
    FE::forms::GeometryTangentPath path) noexcept;

[[nodiscard]] std::string activePhaseSignName(
    FreeSurfaceActiveDomain domain);

[[nodiscard]] EffectiveConfigurationArtifact makeEffectiveConfigurationArtifact(
    const IncompressibleNavierStokesVMSOptions& options,
    std::vector<FreeSurfaceBoundary> free_surfaces,
    const FE::systems::FESystem& system,
    int dimension,
    bool effective_enable_vms);

/// The effective configuration of the two-fluid mode: artifact_schema_version
/// 3, component "incompressible_two_fluid", with the interface weights.
[[nodiscard]] EffectiveConfigurationArtifact makeIncompressibleTwoFluidConfigurationArtifact(
    const IncompressibleTwoFluidOptions& options,
    int dimension,
    int interface_marker,
    const Factories::IncompressibleTwoFluidInterfaceWeights& weights);

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

namespace svmp::Physics::options {

/// The parser of a Free_surface boundary condition: its keys resolve through
/// freeSurfaceBoundarySchema() and its contact lines through
/// freeSurfaceContactLineInputSchema() (decision D47, step P2).
bool is_free_surface_type(std::string_view raw);
svmp::Physics::formulations::navier_stokes::FreeSurfaceImplementation
free_surface_implementation_from_params(const svmp::Physics::ParameterMap& params);
svmp::Physics::formulations::navier_stokes::FreeSurfaceBoundary
resolve_free_surface_bc(
    const svmp::Physics::BoundaryConditionInput& bc,
    bool is_steady,
    bool has_temp_spat,
    bool has_other_files);

/// The keys of a Free_surface boundary condition that the application reads
/// before the register runs (generated-cut construction, face-name
/// translation, solver settings), resolved through freeSurfaceBoundarySchema()
/// so that the application and the register agree on them.
struct FreeSurfaceApplicationKeys {
  /// The stated implementation; std::nullopt when the input gives none, an
  /// empty value or a value that does not parse (the register reports it).
  std::optional<svmp::Physics::formulations::navier_stokes::FreeSurfaceImplementation>
      implementation{};
  /// The active domain of a boundary that does not state Active_domain:
  /// LevelSetNegative for an unfitted surface (decision D47 item 3), None
  /// otherwise.
  svmp::Physics::formulations::navier_stokes::FreeSurfaceActiveDomain
      default_active_domain{
          svmp::Physics::formulations::navier_stokes::FreeSurfaceActiveDomain::None};
  /// Small_cut_aggregation (a relaxed boolean; true when not given).
  bool small_cut_aggregation{true};
};

/// Reads Implementation and Small_cut_aggregation from the given entries;
/// other keys are not read.
FreeSurfaceApplicationKeys
free_surface_application_keys(const svmp::FE::options::OptionInput& input);

svmp::Physics::formulations::navier_stokes::
    IncompressibleNavierStokesVMSOptions
translate_fitted_surface_contact_capability(
    const svmp::Physics::EquationModuleInput& input);

} // namespace svmp::Physics::options

#endif // SVMP_PHYSICS_OPTIONS_NAVIER_STOKES_OPTION_SCHEMA_H
