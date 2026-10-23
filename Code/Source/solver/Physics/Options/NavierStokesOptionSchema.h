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
    FE::forms::bc::ScalarValue contact_angle_radians{FE::Real{1.57079632679489661923}};
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

/// The "resolved" section of the Navier-Stokes artifact, for the boundaries
/// in the given order:
///   {"version":1,"free_surfaces":[{"boundary":{...},"contact_line":{...}},...]}
/// Each key carries its value and its source (OptionSchema::describe against
/// the boundary's stored input).
[[nodiscard]] std::string resolvedFreeSurfacesJson(const std::vector<FreeSurfaceBoundary>& free_surfaces);

[[nodiscard]] const char* pressureActiveDomainName(
    FreeSurfaceActiveDomain domain) noexcept;

[[nodiscard]] const char* surfaceTensionFormName(
    const FreeSurfaceBoundary& bc) noexcept;

[[nodiscard]] const char* freeSurfaceSurfaceTensionSemiImplicitName(
    FreeSurfaceSurfaceTensionSemiImplicit value) noexcept;

[[nodiscard]] std::string freeSurfaceValueName(std::string_view prefix,
                                               const FreeSurfaceBoundary& bc);

[[nodiscard]] std::string contactWallLawName(
    const FreeSurfaceContactLine& contact_line);

[[nodiscard]] const char* pressureStabilizationPolicyName(
    FreeSurfacePressureStabilizationPolicy policy) noexcept;

[[nodiscard]] std::string freeSurfaceCurvaturePolicyName(
    const FreeSurfaceBoundary& bc);

[[nodiscard]] const char* activeDomainName(FreeSurfaceActiveDomain domain) noexcept;

[[nodiscard]] const char* activeDomainMethodName(
    FreeSurfaceActiveDomainMethod method) noexcept;

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
