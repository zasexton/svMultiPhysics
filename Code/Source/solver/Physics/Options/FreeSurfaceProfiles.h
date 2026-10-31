/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_OPTIONS_FREE_SURFACE_PROFILES_H
#define SVMP_PHYSICS_OPTIONS_FREE_SURFACE_PROFILES_H

/**
 * @file FreeSurfaceProfiles.h
 * @brief Versioned method profiles of a free surface (decision D47, step P4)
 *
 * A profile names the method values that a free-surface protocol shares
 * (stabilization, cut geometry, the level-set advection velocity and the
 * accepted-step maintenance), so that a deck states its physics and one
 * profile:
 *
 *   <Add_BC name="free_surface">
 *     <Type>Free_surface</Type>
 *     <Implementation>UnfittedLevelSet</Implementation>
 *     <Free_surface_profile>Wetting@1</Free_surface_profile>
 *     ...
 *
 * The profile states each of its values as an input entry of the boundary
 * or of the level-set equation of the boundary's field, unless that element
 * states the key itself (every explicit key overrides the profile) or a key
 * the value yields to.  The entries resolve through the same schemas as
 * explicit ones, so a profile deck resolves to exactly the options of the
 * deck that writes the values out; the effective-configuration artifact
 * records them with the source "profile" and the profile's name.
 *
 * A profile is versioned: a later revision of a value is a new version, and
 * a deck that names an earlier one keeps its values.  "None" states nothing
 * (the code defaults).
 */

#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"
#include "Physics/Options/NavierStokesOptionSchema.h"

#include "FE/Core/OptionSchema.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace svmp::Physics::options {

/// The input element of a profile value.
enum class FreeSurfaceProfileElement : std::uint8_t {
  /// The Free_surface boundary condition that names the profile.
  Boundary,
  /// The level-set equation whose Level_set_field_name is the boundary's.
  LevelSetEquation,
};

/// One value of a profile, in the input spelling of its element.
struct FreeSurfaceProfileValue {
  FreeSurfaceProfileElement element{FreeSurfaceProfileElement::Boundary};
  std::string key{};
  std::string value{};
  /// Keys of the same element that replace the value: it applies only when
  /// the element states none of them.  A stated velocity source takes the
  /// whole advection velocity of a profile, a stated stabilization switch
  /// its penalty, and a stated prescribed tangential mesh velocity or
  /// tangential penalty its tangential policy.
  std::vector<std::string> yields_to{};
  /// The value applies only to a boundary with this contact-line model.
  std::optional<svmp::Physics::formulations::navier_stokes::
                    FreeSurfaceContactLineModelInput>
      contact_line_model{};
};

struct FreeSurfaceProfile {
  std::string name{};
  /// 0 for None, which has no versions.
  int version{0};
  /// The implementation the profile is written for; std::nullopt for None.
  std::optional<svmp::Physics::formulations::navier_stokes::
                    FreeSurfaceImplementation>
      implementation{};
  std::string summary{};
  std::vector<FreeSurfaceProfileValue> values{};

  /// "Capillary@1", or "None".
  [[nodiscard]] std::string label() const;
};

/// Every profile version, in declaration order (None last).
[[nodiscard]] const std::vector<FreeSurfaceProfile>& freeSurfaceProfiles();

/// The profile a Free_surface_profile value names: "<name>@<version>", or
/// "<name>" for its latest version (case, '_', '-' and white space are
/// ignored).  Throws FE::options::OptionError naming the accepted values.
[[nodiscard]] const FreeSurfaceProfile& freeSurfaceProfile(std::string_view text);

/// The canonical spelling of the boundary key that names a profile.
inline constexpr std::string_view kFreeSurfaceProfileKey = "Free_surface_profile";

/// The profile a Free_surface boundary names, or nullptr when it names none.
/// Throws when the name is unknown or the profile is written for another
/// Implementation than the boundary states.
[[nodiscard]] const FreeSurfaceProfile* namedFreeSurfaceProfile(
    const svmp::FE::options::OptionInput& boundary_input);

/// The values of `element` that apply: those whose key and yields_to keys
/// `element_input` (the entries the element states) does not give and whose
/// contact-line model the boundary states.
[[nodiscard]] std::vector<FreeSurfaceProfileValue> applicableFreeSurfaceProfileValues(
    const FreeSurfaceProfile& profile,
    FreeSurfaceProfileElement element,
    const svmp::FE::options::OptionInput& element_input,
    const svmp::FE::options::OptionInput& boundary_input);

} // namespace svmp::Physics::options

#endif // SVMP_PHYSICS_OPTIONS_FREE_SURFACE_PROFILES_H
