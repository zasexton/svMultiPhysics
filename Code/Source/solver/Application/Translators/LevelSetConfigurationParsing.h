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

inline constexpr std::array<std::string_view, 6> transport_form{
    "Transport_form",           "TransportForm",
    "Advection_form",           "AdvectionForm",
    "Level_set_transport_form", "LevelSetTransportForm"};
inline constexpr std::array<std::string_view, 5> level_set_field_name{
    "Level_set_field_name", "LevelSetFieldName", "Level_set_field",
    "LevelSetField", "Field_name"};
inline constexpr std::array<std::string_view, 4> level_set_isovalue{
    "Level_set_isovalue", "LevelSetIsovalue", "Interface_isovalue",
    "InterfaceIsovalue"};
inline constexpr std::array<std::string_view, 4> velocity_field_name{
    "Velocity_field_name", "VelocityFieldName", "Advection_velocity_field",
    "AdvectionVelocityField"};
inline constexpr std::array<std::string_view, 2> velocity_auto_register{
    "Auto_register_velocity_field", "AutoRegisterVelocityField"};
inline constexpr std::array<std::string_view, 2> velocity_source{
    "Velocity_source", "VelocitySource"};
inline constexpr std::array<std::string_view, 2> material_interface_marker{
    "Material_interface_marker", "MaterialInterfaceMarker"};
inline constexpr std::array<std::string_view, 4> constant_velocity{
    "Constant_velocity", "ConstantVelocity", "Velocity_value", "VelocityValue"};
inline constexpr std::array<std::string_view, 4> wet_extension_enable{
    "Use_wet_extension_advection_velocity", "UseWetExtensionAdvectionVelocity",
    "Update_advection_velocity_from_wet_region",
    "UpdateAdvectionVelocityFromWetRegion"};
inline constexpr std::array<std::string_view, 6> wet_extension_source{
    "Advection_velocity_from_field", "AdvectionVelocityFromField",
    "Source_velocity_field_name",    "SourceVelocityFieldName",
    "Physical_velocity_field_name",  "PhysicalVelocityFieldName"};
inline constexpr std::array<std::string_view, 4> conservative_phase_enable{
    "Enable_conservative_phase_transport", "EnableConservativePhaseTransport",
    "Conservative_phase_transport", "ConservativePhaseTransport"};
inline constexpr std::array<std::string_view, 4> conservative_phase_field{
    "Conservative_phase_field_name", "ConservativePhaseFieldName",
    "Liquid_indicator_field_name", "LiquidIndicatorFieldName"};
inline constexpr std::array<std::string_view, 2>
    conservative_phase_auto_register{"Auto_register_conservative_phase_field",
                                     "AutoRegisterConservativePhaseField"};
inline constexpr std::array<std::string_view, 2> conservative_phase_side{
    "Conservative_phase_liquid_side", "ConservativePhaseLiquidSide"};
inline constexpr std::array<std::string_view, 5> reinitialization_enable{
    "Enable_reinitialization", "Enable_level_set_reinitialization",
    "Reinitialization", "Reinitialization_enabled", "Reinitialize_level_set"};
inline constexpr std::array<std::string_view, 3> reinitialization_method{
    "Reinitialization_method", "Level_set_reinitialization_method",
    "ReinitializationMethod"};
inline constexpr std::array<std::string_view, 4> reinitialization_cadence{
    "Reinitialization_cadence_steps", "Reinitialization_cadence",
    "Level_set_reinitialization_cadence_steps", "ReinitializationCadenceSteps"};
inline constexpr std::array<std::string_view, 3> reinitialization_iterations{
    "Reinitialization_max_iterations", "Reinitialization_iterations",
    "ReinitializationMaxIterations"};
inline constexpr std::array<std::string_view, 2>
    reinitialization_pseudo_time_step{"Reinitialization_pseudo_time_step_scale",
                                      "ReinitializationPseudoTimeStepScale"};
inline constexpr std::array<std::string_view, 2>
    reinitialization_interface_band{"Reinitialization_interface_band_width",
                                    "ReinitializationInterfaceBandWidth"};
inline constexpr std::array<std::string_view, 2>
    reinitialization_signed_distance_tolerance{
        "Reinitialization_signed_distance_tolerance",
        "ReinitializationSignedDistanceTolerance"};
inline constexpr std::array<std::string_view, 2>
    reinitialization_max_zero_set_displacement{
        "Reinitialization_max_zero_set_displacement",
        "ReinitializationMaxZeroSetDisplacement"};
inline constexpr std::array<std::string_view, 5> volume_correction_enable{
    "Enable_volume_correction", "Enable_level_set_volume_correction",
    "Volume_correction", "VolumeCorrection", "Correct_level_set_volume"};
inline constexpr std::array<std::string_view, 4> volume_correction_cadence{
    "Volume_correction_cadence_steps", "Volume_correction_cadence",
    "Level_set_volume_correction_cadence_steps",
    "VolumeCorrectionCadenceSteps"};
inline constexpr std::array<std::string_view, 3> volume_correction_use_initial{
    "Volume_correction_use_initial_volume",
    "Use_initial_level_set_volume_as_target",
    "VolumeCorrectionUseInitialVolume"};
inline constexpr std::array<std::string_view, 3> volume_correction_target{
    "Volume_correction_target_negative_volume",
    "Level_set_volume_correction_target_negative_volume",
    "VolumeCorrectionTargetNegativeVolume"};
inline constexpr std::array<std::string_view, 4> volume_correction_tolerance{
    "Volume_correction_tolerance", "Volume_correction_volume_tolerance",
    "Level_set_volume_correction_tolerance", "VolumeCorrectionTolerance"};
inline constexpr std::array<std::string_view, 2> volume_correction_iterations{
    "Volume_correction_max_iterations", "VolumeCorrectionMaxIterations"};
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
