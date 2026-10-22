#pragma once

#include "FE/LevelSet/LevelSetOptions.h"
#include "Physics/Core/EquationModuleInput.h"

#include <array>
#include <initializer_list>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace application::translators::level_set::configuration {

enum class LevelSetInputPolicy {
  Installation,
  LegacyMaintenance,
};

struct LevelSetSelectedParameter {
  std::string canonical_key{};
  std::string selected_spelling{};
  std::string text{};
  bool supplied{false};
  bool compatibility_fallback{false};
};

class LevelSetConfigurationReader {
public:
  LevelSetConfigurationReader(const svmp::Physics::ParameterMap &parameters,
                              LevelSetInputPolicy policy) noexcept;

  [[nodiscard]] std::optional<LevelSetSelectedParameter>
  selected(std::span<const std::string_view> aliases,
           std::string_view canonical_key = {}) const;

  [[nodiscard]] std::optional<LevelSetSelectedParameter>
  selected(std::initializer_list<std::string_view> aliases,
           std::string_view canonical_key = {}) const;

  [[nodiscard]] std::optional<LevelSetSelectedParameter>
  string(std::span<const std::string_view> aliases,
         std::string_view canonical_key = {});

  [[nodiscard]] std::optional<LevelSetSelectedParameter>
  string(std::initializer_list<std::string_view> aliases,
         std::string_view canonical_key = {});

  [[nodiscard]] std::optional<bool>
  boolean(std::span<const std::string_view> aliases,
          std::string_view canonical_key = {});

  [[nodiscard]] std::optional<bool>
  boolean(std::initializer_list<std::string_view> aliases,
          std::string_view canonical_key = {});

  [[nodiscard]] std::optional<svmp::FE::Real>
  real(std::span<const std::string_view> aliases, std::string_view context);

  [[nodiscard]] std::optional<svmp::FE::Real>
  real(std::initializer_list<std::string_view> aliases,
       std::string_view context);

  [[nodiscard]] std::optional<int>
  integer(std::span<const std::string_view> aliases, std::string_view context,
          bool positive_for_installation = false);

  [[nodiscard]] std::optional<int>
  integer(std::initializer_list<std::string_view> aliases,
          std::string_view context, bool positive_for_installation = false);

  [[nodiscard]] std::optional<std::array<svmp::FE::Real, 3>>
  vector3(std::span<const std::string_view> aliases, std::string_view context,
          std::string_view canonical_key = {});

  [[nodiscard]] std::optional<std::array<svmp::FE::Real, 3>>
  vector3(std::initializer_list<std::string_view> aliases,
          std::string_view context, std::string_view canonical_key = {});

  [[nodiscard]] const std::vector<LevelSetSelectedParameter> &selections() const
      noexcept;

private:
  void observe(const LevelSetSelectedParameter &selection);

  const svmp::Physics::ParameterMap *parameters_{nullptr};
  LevelSetInputPolicy policy_{LevelSetInputPolicy::Installation};
  std::vector<LevelSetSelectedParameter> selections_{};
};

[[nodiscard]] std::string trimCopy(std::string value);
[[nodiscard]] std::string normalizedToken(std::string value);

[[nodiscard]] int parseStrictPositiveInteger(std::string_view raw,
                                             std::string_view context);

[[nodiscard]] svmp::FE::level_set::LevelSetTransportForm
parseTransportForm(std::string_view raw);

[[nodiscard]] svmp::FE::level_set::LevelSetVelocitySource
parseVelocitySource(std::string_view raw, LevelSetInputPolicy policy);

[[nodiscard]] svmp::FE::level_set::LevelSetPhaseSide
parsePhaseSide(std::string_view raw);

[[nodiscard]] svmp::FE::level_set::LevelSetConservativePhaseBoundaryFluxPolicy
parseConservativePhaseBoundaryFluxPolicy(std::string_view raw);

[[nodiscard]] svmp::FE::level_set::LevelSetReinitializationMethod
parseReinitializationMethod(std::string_view raw, LevelSetInputPolicy policy);

namespace aliases {

inline constexpr std::array<std::string_view, 1> transport_form{
    "Transport_form"};
inline constexpr std::array<std::string_view, 2> level_set_field_name{
    "Level_set_field_name",
    "Field_name"};
inline constexpr std::array<std::string_view, 1> level_set_isovalue{
    "Level_set_isovalue"};
inline constexpr std::array<std::string_view, 1> velocity_field_name{
    "Velocity_field_name"};
inline constexpr std::array<std::string_view, 1> velocity_auto_register{
    "Auto_register_velocity_field"};
inline constexpr std::array<std::string_view, 1> velocity_source{
    "Velocity_source"};
inline constexpr std::array<std::string_view, 1> material_interface_marker{
    "Material_interface_marker"};
inline constexpr std::array<std::string_view, 1> constant_velocity{
    "Constant_velocity"};
inline constexpr std::array<std::string_view, 1> wet_extension_enable{
    "Use_wet_extension_advection_velocity"};
inline constexpr std::array<std::string_view, 2>
    advection_velocity_extension_method{
        "Advection_velocity_extension_method",
        "Wet_extension_advection_velocity_method"};
inline constexpr std::array<std::string_view, 1>
    advection_velocity_extension_coupling{
        "Advection_velocity_extension_coupling"};
inline constexpr std::array<std::string_view, 2> wet_extension_source{
    "Advection_velocity_from_field",
    "Source_velocity_field_name"};
inline constexpr std::array<std::string_view, 1> conservative_phase_enable{
    "Enable_conservative_phase_transport"};
inline constexpr std::array<std::string_view, 1> conservative_phase_field{
    "Conservative_phase_field_name"};
inline constexpr std::array<std::string_view, 1>
    conservative_phase_auto_register{"Auto_register_conservative_phase_field"};
inline constexpr std::array<std::string_view, 1> conservative_phase_side{
    "Conservative_phase_liquid_side"};
inline constexpr std::array<std::string_view, 1> reinitialization_enable{
    "Enable_reinitialization"};
inline constexpr std::array<std::string_view, 1> reinitialization_method{
    "Reinitialization_method"};
inline constexpr std::array<std::string_view, 1> reinitialization_cadence{
    "Reinitialization_cadence_steps"};
inline constexpr std::array<std::string_view, 1> reinitialization_iterations{
    "Reinitialization_max_iterations"};
inline constexpr std::array<std::string_view, 1>
    reinitialization_pseudo_time_step{"Reinitialization_pseudo_time_step_scale"};
inline constexpr std::array<std::string_view, 1>
    reinitialization_interface_band{"Reinitialization_interface_band_width"};
inline constexpr std::array<std::string_view, 1>
    reinitialization_signed_distance_tolerance{
        "Reinitialization_signed_distance_tolerance"};
inline constexpr std::array<std::string_view, 1>
    reinitialization_max_zero_set_displacement{
        "Reinitialization_max_zero_set_displacement"};
inline constexpr std::array<std::string_view, 8> curvature_field{
    "Curvature_field_name",
    "CurvatureFieldName",
    "Curvature_field",
    "CurvatureField",
    "Projected_curvature_field",
    "ProjectedCurvatureField",
    "Free_surface_curvature_field",
    "FreeSurfaceCurvatureField"};

} // namespace aliases

} // namespace application::translators::level_set::configuration
