#include "Application/Translators/LevelSetConfigurationParsing.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace application::translators::level_set::configuration {
namespace {

std::string lowerCopy(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

bool relaxedBoolean(std::string_view raw) {
  const auto value = lowerCopy(trimCopy(std::string(raw)));
  return value == "true" || value == "1" || value == "yes" || value == "on";
}

} // namespace

LevelSetConfigurationReader::LevelSetConfigurationReader(
    const svmp::Physics::ParameterMap &parameters,
    LevelSetInputPolicy policy) noexcept
    : parameters_(&parameters), policy_(policy) {}

std::optional<LevelSetSelectedParameter>
LevelSetConfigurationReader::selected(std::span<const std::string_view> aliases,
                                      std::string_view canonical_key) const {
  for (const auto alias : aliases) {
    const auto found = parameters_->find(std::string(alias));
    if (found == parameters_->end()) {
      continue;
    }
    const auto trimmed = trimCopy(found->second.value);
    if (trimmed.empty() || (policy_ == LevelSetInputPolicy::Installation &&
                            !found->second.defined)) {
      continue;
    }
    return LevelSetSelectedParameter{
        .canonical_key = canonical_key.empty() ? std::string(alias)
                                               : std::string(canonical_key),
        .selected_spelling = std::string(alias),
        .text = policy_ == LevelSetInputPolicy::Installation
                    ? trimmed
                    : found->second.value,
        .supplied = found->second.defined,
        .compatibility_fallback =
            policy_ == LevelSetInputPolicy::LegacyMaintenance &&
            !found->second.defined,
    };
  }
  return std::nullopt;
}

std::optional<LevelSetSelectedParameter> LevelSetConfigurationReader::selected(
    std::initializer_list<std::string_view> aliases,
    std::string_view canonical_key) const {
  return selected(
      std::span<const std::string_view>(aliases.begin(), aliases.size()),
      canonical_key);
}

std::optional<LevelSetSelectedParameter>
LevelSetConfigurationReader::string(std::span<const std::string_view> aliases,
                                    std::string_view canonical_key) {
  const auto selection = selected(aliases, canonical_key);
  if (selection) {
    observe(*selection);
  }
  return selection;
}

std::optional<LevelSetSelectedParameter> LevelSetConfigurationReader::string(
    std::initializer_list<std::string_view> aliases,
    std::string_view canonical_key) {
  return string(
      std::span<const std::string_view>(aliases.begin(), aliases.size()),
      canonical_key);
}

void LevelSetConfigurationReader::observe(
    const LevelSetSelectedParameter &selection) {
  selections_.push_back(selection);
}

std::optional<bool>
LevelSetConfigurationReader::boolean(std::span<const std::string_view> aliases,
                                     std::string_view canonical_key) {
  const auto selection = selected(aliases, canonical_key);
  if (!selection) {
    return std::nullopt;
  }
  observe(*selection);
  return relaxedBoolean(selection->text);
}

std::optional<bool> LevelSetConfigurationReader::boolean(
    std::initializer_list<std::string_view> aliases,
    std::string_view canonical_key) {
  return boolean(
      std::span<const std::string_view>(aliases.begin(), aliases.size()),
      canonical_key);
}

std::optional<svmp::FE::Real>
LevelSetConfigurationReader::real(std::span<const std::string_view> aliases,
                                  std::string_view context) {
  const auto selection = selected(aliases, context);
  if (!selection) {
    return std::nullopt;
  }
  observe(*selection);
  if (policy_ == LevelSetInputPolicy::LegacyMaintenance) {
    return static_cast<svmp::FE::Real>(std::stod(selection->text));
  }
  try {
    std::size_t consumed = 0u;
    const auto value = std::stod(selection->text, &consumed);
    if (consumed != selection->text.size() || !std::isfinite(value)) {
      throw std::runtime_error("");
    }
    return static_cast<svmp::FE::Real>(value);
  } catch (...) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Failed to parse numeric value '" +
        selection->text + "' for " + std::string(context) + ".");
  }
}

std::optional<svmp::FE::Real> LevelSetConfigurationReader::real(
    std::initializer_list<std::string_view> aliases, std::string_view context) {
  return real(
      std::span<const std::string_view>(aliases.begin(), aliases.size()),
      context);
}

std::optional<int>
LevelSetConfigurationReader::integer(std::span<const std::string_view> aliases,
                                     std::string_view context,
                                     bool positive_for_installation) {
  const auto selection = selected(aliases, context);
  if (!selection) {
    return std::nullopt;
  }
  observe(*selection);
  if (policy_ == LevelSetInputPolicy::LegacyMaintenance) {
    return std::stoi(selection->text);
  }
  if (positive_for_installation) {
    return parseStrictPositiveInteger(selection->text, context);
  }
  try {
    std::size_t consumed = 0u;
    const auto value = std::stoi(selection->text, &consumed);
    if (consumed != selection->text.size()) {
      throw std::runtime_error("");
    }
    return value;
  } catch (...) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Failed to parse integer value '" +
        selection->text + "' for " + std::string(context) + ".");
  }
}

std::optional<int> LevelSetConfigurationReader::integer(
    std::initializer_list<std::string_view> aliases, std::string_view context,
    bool positive_for_installation) {
  return integer(
      std::span<const std::string_view>(aliases.begin(), aliases.size()),
      context, positive_for_installation);
}

std::optional<std::array<svmp::FE::Real, 3>>
LevelSetConfigurationReader::vector3(std::span<const std::string_view> aliases,
                                     std::string_view context,
                                     std::string_view canonical_key) {
  const auto selection =
      selected(aliases, canonical_key.empty() ? context : canonical_key);
  if (!selection) {
    return std::nullopt;
  }
  observe(*selection);
  auto text = selection->text;
  if (policy_ == LevelSetInputPolicy::LegacyMaintenance) {
    std::replace(text.begin(), text.end(), ',', ' ');
  }
  std::istringstream input(text);
  std::array<svmp::FE::Real, 3> result{};
  for (auto &value : result) {
    double parsed = 0.0;
    if (!(input >> parsed) || !std::isfinite(parsed)) {
      if (policy_ == LevelSetInputPolicy::LegacyMaintenance) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] " + std::string(context) +
            " must contain exactly three finite numeric components.");
      }
      throw std::runtime_error(
          "[svMultiPhysics::Application] Failed to parse three numeric "
          "components for " +
          std::string(context) + ".");
    }
    value = static_cast<svmp::FE::Real>(parsed);
  }
  std::string trailing;
  if (input >> trailing) {
    if (policy_ == LevelSetInputPolicy::LegacyMaintenance) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] " + std::string(context) +
          " must contain exactly three finite numeric components.");
    }
    throw std::runtime_error(
        "[svMultiPhysics::Application] Failed to parse three numeric "
        "components for " +
        std::string(context) + ".");
  }
  return result;
}

std::optional<std::array<svmp::FE::Real, 3>>
LevelSetConfigurationReader::vector3(
    std::initializer_list<std::string_view> aliases, std::string_view context,
    std::string_view canonical_key) {
  return vector3(
      std::span<const std::string_view>(aliases.begin(), aliases.size()),
      context, canonical_key);
}

const std::vector<LevelSetSelectedParameter> &
LevelSetConfigurationReader::selections() const noexcept {
  return selections_;
}

std::string trimCopy(std::string value) {
  const auto not_space = [](unsigned char c) { return !std::isspace(c); };
  value.erase(value.begin(),
              std::find_if(value.begin(), value.end(), not_space));
  value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(),
              value.end());
  return value;
}

std::string normalizedToken(std::string value) {
  value = lowerCopy(trimCopy(std::move(value)));
  value.erase(std::remove_if(value.begin(), value.end(),
                             [](unsigned char c) {
                               return c == '_' || c == '-' || std::isspace(c);
                             }),
              value.end());
  return value;
}

int parseStrictPositiveInteger(std::string_view raw,
                               std::string_view context) {
  const auto text = trimCopy(std::string(raw));
  try {
    std::size_t consumed = 0u;
    const auto value = std::stoi(text, &consumed);
    if (consumed != text.size() || value < 1) {
      throw std::runtime_error("");
    }
    return value;
  } catch (...) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Failed to parse positive integer "
        "value '" +
        std::string(raw) + "' for " + std::string(context) + ".");
  }
}

svmp::FE::level_set::LevelSetTransportForm
parseTransportForm(std::string_view raw) {
  const auto value = normalizedToken(std::string(raw));
  using Form = svmp::FE::level_set::LevelSetTransportForm;
  if (value == "advective" || value == "classical" || value == "standard") {
    return Form::Advective;
  }
  if (value == "conservative" || value == "conservativedivergence" ||
      value == "divergence" || value == "divergenceform") {
    return Form::ConservativeDivergence;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Application] Level-set Transport_form must be one of "
      "'advective' or 'conservative_divergence'.");
}

svmp::FE::level_set::LevelSetVelocitySource
parseVelocitySource(std::string_view raw, LevelSetInputPolicy policy) {
  const auto value = normalizedToken(std::string(raw));
  using Source = svmp::FE::level_set::LevelSetVelocitySource;
  if ((policy == LevelSetInputPolicy::LegacyMaintenance && value.empty()) ||
      value == "coupled" || value == "coupledfield" ||
      value == "unknown" || (policy == LevelSetInputPolicy::Installation &&
                             value == "navierstokes")) {
    return Source::CoupledField;
  }
  if (value == "prescribed" || value == "prescribeddata" || value == "data") {
    return Source::PrescribedData;
  }
  if (value == "constant" || value == "constantvector") {
    return Source::ConstantVector;
  }
  if (value == "materialinterfacephasepair") {
    return Source::MaterialInterfacePhasePair;
  }
  if (policy == LevelSetInputPolicy::LegacyMaintenance) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Level-set Velocity_source must be one "
        "of 'coupled_field', 'prescribed_data', 'constant_vector', or "
        "'material_interface_phase_pair'.");
  }
  throw std::runtime_error(
      "[svMultiPhysics::Application] Velocity_source must be one of "
      "'coupled_field', 'prescribed_data', 'constant', or "
      "'material_interface_phase_pair'.");
}

svmp::FE::level_set::LevelSetPhaseSide parsePhaseSide(std::string_view raw) {
  const auto value = normalizedToken(std::string(raw));
  using Side = svmp::FE::level_set::LevelSetPhaseSide;
  if (value == "negative" || value == "minus") {
    return Side::Negative;
  }
  if (value == "positive" || value == "plus") {
    return Side::Positive;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Application] Conservative_phase_liquid_side must be "
      "'negative' or 'positive'.");
}

svmp::FE::level_set::LevelSetConservativePhaseBoundaryFluxPolicy
parseConservativePhaseBoundaryFluxPolicy(std::string_view raw) {
  const auto value = normalizedToken(std::string(raw));
  using Policy =
      svmp::FE::level_set::LevelSetConservativePhaseBoundaryFluxPolicy;
  if (value == "closeddomaindiscreteqfluxonly") {
    return Policy::ClosedDomainDiscreteQFluxOnly;
  }
  if (value == "globallybalanceddiscreteqflux") {
    return Policy::GloballyBalancedDiscreteQFlux;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Application] Conservative_phase_boundary_flux_policy "
      "must be 'closed_domain_discrete_q_flux_only' or "
      "'globally_balanced_discrete_q_flux'.");
}

svmp::FE::level_set::LevelSetReinitializationMethod
parseReinitializationMethod(std::string_view raw, LevelSetInputPolicy) {
  const auto value = normalizedToken(std::string(raw));
  using Method = svmp::FE::level_set::LevelSetReinitializationMethod;
  if (value == "projection" || value == "signeddistanceprojection" ||
      value == "repairprojection") {
    return Method::Projection;
  }
  if (value == "hamiltonjacobi" || value == "hamiltonjacobipde" ||
      value == "pde") {
    throw std::runtime_error(
        "[svMultiPhysics::Application] "
        "Reinitialization_method=HamiltonJacobiPDE is reserved until runtime "
        "Hamilton-Jacobi reinitialization is implemented; use 'Projection'.");
  }
  if (value == "fastmarching" || value == "fastmarchingmethod" ||
      value == "fmm") {
    throw std::runtime_error(
        "[svMultiPhysics::Application] "
        "Reinitialization_method=FastMarching is reserved until runtime "
        "fast-marching reinitialization is implemented; use 'Projection'.");
  }
  throw std::runtime_error(
      "[svMultiPhysics::Application] Reinitialization_method currently "
      "supports 'Projection' only.");
}

} // namespace application::translators::level_set::configuration
