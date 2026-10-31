#pragma once

#include "Application/Core/ResolvedMovingDomainConfiguration.h"
#include "FE/Core/OptionSchema.h"
#include "FE/LevelSet/LevelSetOptions.h"
#include "Physics/Core/EquationModuleInput.h"

#include <array>
#include <initializer_list>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

class EquationParameters;

namespace application::translators::level_set::configuration {

/// The keys of a level-set equation (decision D47, step P3), declared once:
/// spellings, codec, default (the FE option default), the values one key
/// implies for another and documentation.  The installation resolves the
/// equation element and then its domain elements, each over the values
/// before it; describe() gives the "resolved" section of the transport
/// artifact.
[[nodiscard]] const svmp::FE::options::OptionSchema<
    application::core::LevelSetEquationKeys> &
levelSetEquationSchema();

/// The defined entries of a parameter map, as schema input.
[[nodiscard]] svmp::FE::options::OptionInput
levelSetEquationInput(const svmp::Physics::ParameterMap &parameters);

/// The defined parameters of an equation element of the input file, as
/// schema input (for readers of the element that run without the
/// installation, such as the driver's extension requests).
[[nodiscard]] svmp::FE::options::OptionInput
levelSetEquationInput(const EquationParameters &equation);

/// The input of a domain element: its defined entries without the keys that
/// only the equation element gives (Level_set_isovalue and the keys of the
/// advection-velocity extension).
[[nodiscard]] svmp::FE::options::OptionInput
levelSetDomainInput(const svmp::Physics::ParameterMap &parameters);

/// Resolves the keys of one element over the values of the elements before
/// it, then sets the values that a key selects beyond its own field: Coupled
/// routes the transport to the monolithic operator, Constant_velocity
/// selects a constant velocity without a registered field, and
/// Interface_kinematic_marker enables the interface kinematic term.
[[nodiscard]] svmp::FE::options::ResolvedOptions<
    application::core::LevelSetEquationKeys>
resolveLevelSetEquationKeys(const svmp::FE::options::OptionInput &input,
                            application::core::LevelSetEquationKeys previous = {});

/// The "resolved" section of the transport artifact:
///   {"version":1,"equation":{...}}
/// Each key carries its value and its source (OptionSchema::describe of the
/// resolved keys against the input as read, the later element first).
[[nodiscard]] std::string
resolvedLevelSetEquationJson(const svmp::FE::options::OptionInput &input,
                             const application::core::LevelSetEquationKeys &keys);

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

[[nodiscard]] svmp::FE::level_set::LevelSetPhaseSide
parsePhaseSide(std::string_view raw);

[[nodiscard]] svmp::FE::level_set::LevelSetConservativePhaseBoundaryFluxPolicy
parseConservativePhaseBoundaryFluxPolicy(std::string_view raw);

namespace aliases {

inline constexpr std::array<std::string_view, 1> conservative_phase_enable{
    "Enable_conservative_phase_transport"};
inline constexpr std::array<std::string_view, 1> conservative_phase_field{
    "Conservative_phase_field_name"};
inline constexpr std::array<std::string_view, 1>
    conservative_phase_auto_register{"Auto_register_conservative_phase_field"};
inline constexpr std::array<std::string_view, 1> conservative_phase_side{
    "Conservative_phase_liquid_side"};
} // namespace aliases

} // namespace application::translators::level_set::configuration
