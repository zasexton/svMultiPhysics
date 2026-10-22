/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_BOUNDARY_SCHEMA_H
#define SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_BOUNDARY_SCHEMA_H

/**
 * @file FreeSurfaceBoundarySchema.h
 * @brief The input keys of a free-surface boundary condition as option schemas
 *
 * One schema declares the keys that the Navier-Stokes register reads into a
 * FreeSurfaceBoundary (append_free_surface_bc), with their spellings, codecs
 * and the members they fill; a second one declares the contact-line keys,
 * read into a view of FreeSurfaceBoundary::contact_lines.  The register
 * still parses the input by hand.  The schemas record where each resolved
 * value came from (input, default, derived) in the "resolved" section of the
 * effective-configuration artifact (decision D47, step P1.2a); the parser
 * rewrite of step P2 resolves with the same declarations.
 */

#include "FE/Core/OptionSchema.h"
#include "FE/Core/Types.h"
#include "FE/Forms/BoundaryConditions.h"
#include "Physics/Core/EquationModuleInput.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceOptions.h"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

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

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_FREE_SURFACE_FREE_SURFACE_BOUNDARY_SCHEMA_H
