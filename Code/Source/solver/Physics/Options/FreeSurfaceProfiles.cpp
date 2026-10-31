/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Options/FreeSurfaceProfiles.h"

#include <algorithm>
#include <charconv>
#include <system_error>
#include <utility>

namespace svmp::Physics::options {

namespace {

namespace opt = svmp::FE::options;
namespace navier_stokes = svmp::Physics::formulations::navier_stokes;
using Element = FreeSurfaceProfileElement;
using Value = FreeSurfaceProfileValue;

constexpr const char* kElement = "Free_surface";

Value boundary(std::string key, std::string value, std::vector<std::string> yields_to = {})
{
  return Value{.element = Element::Boundary,
               .key = std::move(key),
               .value = std::move(value),
               .yields_to = std::move(yields_to)};
}

Value levelSet(std::string key, std::string value, std::vector<std::string> yields_to = {})
{
  return Value{.element = Element::LevelSetEquation,
               .key = std::move(key),
               .value = std::move(value),
               .yields_to = std::move(yields_to)};
}

void append(std::vector<Value>& values, std::vector<Value> more)
{
  values.insert(values.end(), std::make_move_iterator(more.begin()),
                std::make_move_iterator(more.end()));
}

/// The values every unfitted protocol shares: the level-set field is data
/// initialized from the mesh, advected by the harmonic PDE extension of the
/// fluid velocity with monolithic coupling (D9), SUPG with c_tau = 0.5 and
/// c_t = 2, accepted-step kinematic reconciliation (D14), LinearCorner cut
/// geometry at isovalue 0, the cut-facet pressure-gradient penalty with
/// gamma = 1 and small-cut aggregation.
std::vector<Value> unfittedProtocolValues()
{
  // A stated velocity source (or a constant velocity) replaces the whole
  // advection velocity of the profile.
  const std::vector<std::string> velocity{"Velocity_source", "Constant_velocity"};
  return {
      levelSet("Level_set_source", "prescribed_data"),
      levelSet("Velocity_source", "prescribed_data", velocity),
      levelSet("Velocity_field_name", "LevelSetAdvectionVelocity", velocity),
      levelSet("Auto_register_velocity_field", "true", velocity),
      levelSet("Source_velocity_field_name", "Velocity", velocity),
      levelSet("Advection_velocity_extension_method", "pde_harmonic", velocity),
      levelSet("Advection_velocity_extension_coupling", "monolithic", velocity),
      levelSet("Enable_SUPG", "true"),
      levelSet("SUPG_tau_scale", "0.5"),
      levelSet("SUPG_transient_scale", "2.0"),
      levelSet("Enable_kinematic_reconciliation", "true"),
      boundary("Level_set_isovalue", "0.0"),
      boundary("Generated_interface_geometry", "LinearCorner"),
      boundary("Enable_cut_cell_stabilization", "true"),
      // A stated switch keeps the penalty to the deck: a deck that turns the
      // stabilization off states no suboption.
      boundary("Cut_cell_pressure_gradient_penalty", "1.0", {"Enable_cut_cell_stabilization"}),
      boundary("Small_cut_aggregation", "true"),
  };
}

/// The geometry that surface tension needs: refreshed-frozen geometry
/// tangents and interface quadrature order 2 (D47 item 7).
std::vector<Value> capillaryGeometryValues()
{
  return {
      boundary("Geometry_tangent_policy", "RefreshedFrozenQuadrature"),
      boundary("Interface_quadrature_order", "2"),
  };
}

std::vector<FreeSurfaceProfile> makeProfiles()
{
  using navier_stokes::FreeSurfaceContactLineModelInput;
  using navier_stokes::FreeSurfaceImplementation;

  FreeSurfaceProfile capillary{
      .name = "Capillary",
      .version = 1,
      .implementation = FreeSurfaceImplementation::UnfittedLevelSet,
      .summary = "unfitted surface tension without a contact line: static "
                 "drop and sphere, oscillating drop, capillary wave",
  };
  capillary.values = unfittedProtocolValues();
  append(capillary.values, capillaryGeometryValues());
  // The lagged normal-increment capillary term (D13).
  capillary.values.push_back(
      boundary("Surface_tension_semi_implicit", "NormalIncrement"));

  FreeSurfaceProfile wetting{
      .name = "Wetting",
      .version = 1,
      .implementation = FreeSurfaceImplementation::UnfittedLevelSet,
      .summary = "unfitted surface tension with contact lines: sessile "
                 "drops, capillary rise, Ren-E drop",
  };
  wetting.values = unfittedProtocolValues();
  // Sign-definite patch bounds (D21) and triggered interface-preserving
  // redistancing (D35, D39) after each accepted step.
  append(wetting.values,
         {levelSet("Enable_sign_definite_patch_bounds", "true"),
          levelSet("Enable_interface_preserving_redistancing", "true")});
  append(wetting.values, capillaryGeometryValues());
  wetting.values.push_back(boundary("Surface_tension_semi_implicit", "None"));
  // The strong Ren-E law (D47 item 6); the code default stays weak.
  auto strong = boundary("Contact_line_law_enforcement", "Strong");
  strong.contact_line_model = FreeSurfaceContactLineModelInput::DynamicRenE;
  wetting.values.push_back(std::move(strong));

  FreeSurfaceProfile gravity_wave{
      .name = "Gravity_wave",
      .version = 1,
      .implementation = FreeSurfaceImplementation::UnfittedLevelSet,
      .summary = "unfitted open vessel without surface tension: linear "
                 "sloshing, the tank at rest",
  };
  gravity_wave.values = unfittedProtocolValues();
  gravity_wave.values.push_back(boundary("Surface_tension_semi_implicit", "None"));

  FreeSurfaceProfile fitted_ale{
      .name = "Fitted_ALE",
      .version = 1,
      .implementation = FreeSurfaceImplementation::FittedALE,
      .summary = "fitted ALE free surface: MeshNitsche normal kinematics "
                 "with gamma_N = 10 and a free tangential mesh",
  };
  fitted_ale.values = {
      // A stated prescribed tangential mesh velocity implies Prescribed, and
      // a stated tangential penalty needs a policy with a tangential row.
      boundary("Tangential_mesh_policy", "Free",
               {"Prescribed_tangential_mesh_velocity", "Tangential_mesh_penalty"}),
      boundary("Kinematic_nitsche_gamma", "10"),
      boundary("Surface_tension_semi_implicit", "None"),
  };

  FreeSurfaceProfile none{
      .name = "None",
      .version = 0,
      .implementation = std::nullopt,
      .summary = "no profile: every value is the code default unless stated",
  };

  return {std::move(capillary), std::move(wetting), std::move(gravity_wave),
          std::move(fitted_ale), std::move(none)};
}

std::string acceptedNames()
{
  std::string out;
  for (const auto& profile : freeSurfaceProfiles()) {
    if (!out.empty()) {
      out += " | ";
    }
    out += profile.label();
  }
  return out;
}

bool states(const opt::OptionInput& input, std::string_view key)
{
  return input.find(key) != nullptr;
}

} // namespace

std::string FreeSurfaceProfile::label() const
{
  return version > 0 ? name + "@" + std::to_string(version) : name;
}

const std::vector<FreeSurfaceProfile>& freeSurfaceProfiles()
{
  static const auto profiles = makeProfiles();
  return profiles;
}

const FreeSurfaceProfile& freeSurfaceProfile(std::string_view text)
{
  const auto trimmed = opt::detail::trimmed(text);
  const auto at = trimmed.find('@');
  const auto name = opt::detail::normalizedToken(std::string_view(trimmed).substr(0, at));
  std::optional<int> version;
  if (at != std::string::npos) {
    const auto digits = opt::detail::trimmed(std::string_view(trimmed).substr(at + 1u));
    int parsed = 0;
    const auto result = std::from_chars(digits.data(), digits.data() + digits.size(), parsed);
    if (!digits.empty() && result.ec == std::errc{} &&
        result.ptr == digits.data() + digits.size()) {
      version = parsed;
    } else {
      version = -1;
    }
  }
  const FreeSurfaceProfile* match = nullptr;
  for (const auto& profile : freeSurfaceProfiles()) {
    if (opt::detail::normalizedToken(profile.name) != name) {
      continue;
    }
    if (version.has_value()) {
      if (profile.version > 0 && profile.version == *version) {
        return profile;
      }
      continue;
    }
    if (match == nullptr || profile.version > match->version) {
      match = &profile;
    }
  }
  if (match == nullptr) {
    throw opt::OptionError(kElement, std::string(kFreeSurfaceProfileKey),
                           "expected one of " + acceptedNames() + " (a name alone is its "
                           "latest version), got '" + std::string(text) + "'");
  }
  return *match;
}

const FreeSurfaceProfile* namedFreeSurfaceProfile(const opt::OptionInput& boundary_input)
{
  const auto* entry = boundary_input.find(kFreeSurfaceProfileKey);
  if (entry == nullptr) {
    return nullptr;
  }
  const auto& profile = freeSurfaceProfile(entry->text);
  if (!profile.implementation.has_value()) {
    return &profile;
  }
  const auto& schema = navier_stokes::freeSurfaceBoundarySchema();
  std::optional<navier_stokes::FreeSurfaceImplementation> implementation;
  try {
    const auto resolved = schema.resolve(boundary_input.selectSpellings({"Implementation"}));
    if (resolved.isExplicit("Implementation")) {
      implementation = resolved.value.implementation;
    }
  } catch (const opt::OptionError&) {
    // The register reports a value that does not parse.
  }
  if (implementation != profile.implementation) {
    throw opt::OptionError(
        kElement, std::string(kFreeSurfaceProfileKey),
        profile.label() + " is written for Implementation=" +
            navier_stokes::freeSurfaceImplementationName(*profile.implementation));
  }
  return &profile;
}

std::vector<FreeSurfaceProfileValue> applicableFreeSurfaceProfileValues(
    const FreeSurfaceProfile& profile,
    FreeSurfaceProfileElement element,
    const opt::OptionInput& element_input,
    const opt::OptionInput& boundary_input)
{
  std::optional<navier_stokes::FreeSurfaceContactLineModelInput> contact_line_model;
  try {
    const auto resolved = navier_stokes::freeSurfaceContactLineInputSchema().resolve(
        boundary_input.selectSpellings({"Contact_line_model"}));
    if (resolved.isExplicit("Contact_line_model")) {
      contact_line_model = resolved.value.model;
    }
  } catch (const opt::OptionError&) {
    // The register reports a value that does not parse.
  }

  std::vector<FreeSurfaceProfileValue> out;
  for (const auto& value : profile.values) {
    if (value.element != element || states(element_input, value.key) ||
        std::any_of(value.yields_to.begin(), value.yields_to.end(),
                    [&](const std::string& key) { return states(element_input, key); })) {
      continue;
    }
    if (value.contact_line_model.has_value() &&
        contact_line_model != value.contact_line_model) {
      continue;
    }
    out.push_back(value);
  }
  return out;
}

} // namespace svmp::Physics::options
