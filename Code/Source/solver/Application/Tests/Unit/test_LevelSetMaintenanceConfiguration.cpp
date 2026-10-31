#include <gtest/gtest.h>

#include "Application/Core/LevelSetCutConfiguration.h"
#include "Application/Core/LevelSetMaintenanceConfiguration.h"
#include "Application/Core/ResolvedMovingDomainConfiguration.h"
#include "Application/Translators/LevelSetConfigurationParsing.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace parsing = application::translators::level_set::configuration;

using application::core::ActiveCutVolumeRequest;
using application::core::LegacyLevelSetBoundaryInput;
using application::core::LegacyLevelSetMaintenanceInput;
using application::core::LevelSetEquationInputSnapshot;
using application::core::ResolvedLevelSetEquationConfiguration;
using application::core::ResolvedLevelSetEquationHandle;
using svmp::Physics::ParameterMap;
using svmp::Physics::ParameterValue;

void put(ParameterMap &parameters, std::string key, std::string value,
         bool defined = true) {
  parameters[std::move(key)] = ParameterValue{defined, std::move(value)};
}

// The maintenance input of a level-set equation (its type and boundaries)
// with the parameters of its equation element, which the installation
// resolves.
struct MaintenanceTestInput : LegacyLevelSetMaintenanceInput {
  ParameterMap equation_parameters{};
};

MaintenanceTestInput makeLegacyInput(std::string type = "level_set") {
  MaintenanceTestInput input{};
  input.equation_type_defined = true;
  input.equation_type = std::move(type);
  return input;
}

// An installation of the input's equation element: its keys as the
// translator resolves them (levelSetEquationSchema()), and installed options
// that the maintenance must not read.
ResolvedLevelSetEquationHandle
makeInstallation(const MaintenanceTestInput &legacy,
                 std::string field_name = "phi_installed") {
  auto snapshot = std::make_shared<LevelSetEquationInputSnapshot>();
  snapshot->legacy_maintenance_input = legacy;
  snapshot->installation_input.equation_type = "level_set";
  snapshot->installation_input.equation_params = legacy.equation_parameters;

  auto installation = std::make_shared<ResolvedLevelSetEquationConfiguration>();
  installation->equation_keys =
      parsing::resolveLevelSetEquationKeys(
          parsing::levelSetEquationInput(legacy.equation_parameters))
          .value;
  installation->options.level_set.field_name = std::move(field_name);
  installation->input_snapshot = std::move(snapshot);
  return installation;
}

// The maintenance isovalue is a C++ option of the equation keys (decision
// 18A): `isovalue` sets it on the installation.
std::optional<application::core::LevelSetMaintenanceConfigurationHandle>
resolveInstalled(const MaintenanceTestInput &input,
                 std::span<const ActiveCutVolumeRequest> active_requests = {},
                 std::optional<double> isovalue = std::nullopt) {
  auto installation = makeInstallation(input);
  if (isovalue.has_value()) {
    std::const_pointer_cast<ResolvedLevelSetEquationConfiguration>(installation)
        ->equation_keys.level_set_isovalue = *isovalue;
  }
  return application::core::resolveLegacyLevelSetMaintenanceConfiguration(
      input, active_requests, installation);
}

template <typename ExpectedException, typename Action>
std::string exceptionMessage(Action &&action) {
  try {
    std::forward<Action>(action)();
  } catch (const ExpectedException &error) {
    return error.what();
  } catch (const std::exception &error) {
    ADD_FAILURE() << "Unexpected exception category: " << error.what();
    return {};
  } catch (...) {
    ADD_FAILURE() << "Unexpected non-standard exception category.";
    return {};
  }
  ADD_FAILURE() << "Expected an exception.";
  return {};
}

} // namespace

TEST(LevelSetMaintenanceConfiguration,
     ProducesEquationOnlyTypedViewAndRetainsLayeredInstallation) {
  auto legacy = makeLegacyInput(" Level_Set-Transport ");
  put(legacy.equation_parameters, "Level_set_field_name", " phi_equation ");
  put(legacy.equation_parameters, "Transport_form", "advective");
  put(legacy.equation_parameters, "Enable_kinematic_reconciliation", "true");
  put(legacy.equation_parameters,
      "Conservative_phase_flux_artifact_cadence_steps", "2");
  // A parameter that is not defined does not count.
  put(legacy.equation_parameters, "Enable_sign_definite_patch_bounds", "true",
      false);

  auto installation = makeInstallation(legacy, "phi_domain");
  auto mutable_installation =
      std::const_pointer_cast<ResolvedLevelSetEquationConfiguration>(
          installation);
  // The maintenance isovalue and the geometry tolerance are C++ options of
  // the equation keys (decision 18A).
  mutable_installation->equation_keys.level_set_isovalue = 0.25;
  mutable_installation->equation_keys.transport.conservative_phase
      .geometry_measure_tolerance = 1.23457;
  mutable_installation->options.operator_tag = "domain_transport";
  mutable_installation->options.transport_form =
      svmp::FE::level_set::LevelSetTransportForm::ConservativeDivergence;
  mutable_installation->options.kinematic_reconciliation.enabled = true;
  mutable_installation->options.conservative_phase
      .flux_artifact_cadence_steps = 7;
  mutable_installation->options.conservative_phase
      .geometry_measure_tolerance = 1.2345678901234567;

  const auto resolved =
      application::core::resolveLegacyLevelSetMaintenanceConfiguration(
          legacy, std::span<const ActiveCutVolumeRequest>{}, installation);

  ASSERT_TRUE(resolved.has_value());
  ASSERT_TRUE(*resolved);
  const auto &configuration = **resolved;
  EXPECT_EQ(configuration.installation, installation);
  EXPECT_EQ(configuration.transport.level_set.field_name, "phi_equation");
  EXPECT_EQ(configuration.transport.operator_tag, "level_set");
  EXPECT_EQ(configuration.transport.transport_form,
            svmp::FE::level_set::LevelSetTransportForm::Advective);
  EXPECT_TRUE(configuration.transport.kinematic_reconciliation.enabled);
  EXPECT_FALSE(configuration.transport.sign_definite_patch_bounds.enabled);
  EXPECT_EQ(
      configuration.transport.conservative_phase.flux_artifact_cadence_steps,
      2);
  EXPECT_DOUBLE_EQ(
      configuration.transport.conservative_phase.geometry_measure_tolerance,
      1.23457);
  EXPECT_FALSE(configuration.transport.supg.enabled);
  EXPECT_FALSE(configuration.transport.interface_kinematic.enabled);
  EXPECT_DOUBLE_EQ(configuration.isovalue, 0.25);

  EXPECT_EQ(installation->options.level_set.field_name, "phi_domain");
  EXPECT_EQ(installation->options.transport_form,
            svmp::FE::level_set::LevelSetTransportForm::ConservativeDivergence);
  EXPECT_EQ(
      installation->options.conservative_phase.flux_artifact_cadence_steps, 7);
  EXPECT_DOUBLE_EQ(
      installation->options.conservative_phase.geometry_measure_tolerance,
      1.2345678901234567);
}

TEST(LevelSetMaintenanceConfiguration,
     OmitsUndefinedUnrecognizedAndFullyDisabledEquations) {
  auto undefined = makeLegacyInput();
  undefined.equation_type_defined = false;
  put(undefined.equation_parameters, "Enable_kinematic_reconciliation", "true");
  EXPECT_FALSE(resolveInstalled(undefined).has_value());

  auto unrecognized = makeLegacyInput("fluid");
  put(unrecognized.equation_parameters, "Enable_kinematic_reconciliation", "true");
  EXPECT_FALSE(resolveInstalled(unrecognized).has_value());

  auto disabled = makeLegacyInput();
  put(disabled.equation_parameters, "Enable_kinematic_reconciliation", "false");
  EXPECT_FALSE(resolveInstalled(disabled).has_value());
}

TEST(LevelSetMaintenanceConfiguration, RequiresTheResolvedEquation) {
  auto input = makeLegacyInput();
  put(input.equation_parameters, "Enable_kinematic_reconciliation", "true");
  EXPECT_EQ((exceptionMessage<std::invalid_argument>([&] {
              (void)application::core::
                  resolveLegacyLevelSetMaintenanceConfiguration(
                      input, std::span<const ActiveCutVolumeRequest>{}, {});
            })),
            "[svMultiPhysics::Application] Level-set maintenance requires the "
            "resolved level-set equation.");
}

TEST(LevelSetMaintenanceConfiguration,
     ReadsTheEquationKeysAsTheInstallationResolvesThem) {
  // A whitespace-only value counts as absent (the default cadence 1).
  struct AcceptedCase {
    std::string vector;
    std::string cadence;
    int expected_cadence;
    std::array<svmp::FE::Real, 3> expected_vector;
  };
  const std::array<AcceptedCase, 3> cases{{
      {"1.25, -0.5, 3.0", "-1", -1, {1.25, -0.5, 3.0}},
      {"4.0 5.0 6.0", "   ", 1, {4.0, 5.0, 6.0}},
      {"7.0,8.0,9.0", "3", 3, {7.0, 8.0, 9.0}},
  }};

  for (const auto &test_case : cases) {
    auto input = makeLegacyInput();
    put(input.equation_parameters, "Velocity_source", "constant");
    put(input.equation_parameters, "Constant_velocity", test_case.vector);
    put(input.equation_parameters, "Enable_kinematic_reconciliation", "true");
    put(input.equation_parameters,
        "Conservative_phase_flux_artifact_cadence_steps", test_case.cadence);
    put(input.equation_parameters, "Static_capillary_max_iterations", "2");
    put(input.equation_parameters,
        "Conservative_phase_momentum_relative_tolerance", "1.25");
    put(input.equation_parameters, "Enable_sign_definite_patch_bounds",
        "unrecognized");
    put(input.equation_parameters, "Static_capillary_volume_tolerance", "nan");

    const auto resolved = resolveInstalled(input);

    ASSERT_TRUE(resolved.has_value());
    ASSERT_TRUE(*resolved);
    const auto &transport = (*resolved)->transport;
    EXPECT_EQ(transport.velocity.source,
              svmp::FE::level_set::LevelSetVelocitySource::ConstantVector);
    EXPECT_EQ(transport.velocity.constant_value, test_case.expected_vector);
    EXPECT_EQ(transport.conservative_phase.flux_artifact_cadence_steps,
              test_case.expected_cadence);
    EXPECT_EQ((*resolved)->static_capillary_equilibrium.max_iterations, 2);
    EXPECT_DOUBLE_EQ(transport.conservative_phase.momentum_relative_tolerance,
                     1.25);
    EXPECT_FALSE(transport.sign_definite_patch_bounds.enabled);
    EXPECT_TRUE(std::isnan(
        (*resolved)->static_capillary_equilibrium.volume_tolerance));
  }
}

TEST(LevelSetMaintenanceConfiguration, TakesTheVelocityFromTheEquationKeys) {
  auto unpromoted = makeLegacyInput();
  put(unpromoted.equation_parameters, "Enable_kinematic_reconciliation", "true");
  auto resolved = resolveInstalled(unpromoted);
  ASSERT_TRUE(resolved.has_value());
  EXPECT_EQ((*resolved)->transport.velocity.source,
            svmp::FE::level_set::LevelSetVelocitySource::CoupledField);

  // A source velocity field selects the extension, whose field is a coupled
  // unknown; the installation's registration and extension source stay out
  // of the maintained transport.
  auto input = makeLegacyInput();
  put(input.equation_parameters, "Level_set_field_name", "phi");
  put(input.equation_parameters, "Velocity_field_name", "extended_velocity");
  put(input.equation_parameters, "Velocity_source", "prescribed_data");
  put(input.equation_parameters, "Source_velocity_field_name",
      "physical_velocity");
  put(input.equation_parameters, "Advection_velocity_extension_method",
      "pde_harmonic");
  put(input.equation_parameters, "Advection_velocity_extension_coupling",
      "monolithic");
  put(input.equation_parameters, "Enable_kinematic_reconciliation", "true");
  resolved = resolveInstalled(input);
  ASSERT_TRUE(resolved.has_value());
  EXPECT_EQ((*resolved)->transport.level_set.field_name, "phi");
  EXPECT_EQ((*resolved)->transport.velocity.field_name, "extended_velocity");
  EXPECT_EQ((*resolved)->transport.velocity.source,
            svmp::FE::level_set::LevelSetVelocitySource::CoupledField);
  EXPECT_FALSE((*resolved)->transport.velocity.auto_register_field);
  EXPECT_TRUE(
      (*resolved)
          ->transport.velocity.algebraic_extension_source_field_name.empty());
  EXPECT_EQ((*resolved)->transport.velocity.space, nullptr);

  // A constant velocity stays constant.
  put(input.equation_parameters, "Constant_velocity", "7.0, 8.0, 9.0");
  resolved = resolveInstalled(input);
  ASSERT_TRUE(resolved.has_value());
  EXPECT_EQ((*resolved)->transport.velocity.source,
            svmp::FE::level_set::LevelSetVelocitySource::ConstantVector);
  EXPECT_EQ((*resolved)->transport.velocity.constant_value,
            (std::array<svmp::FE::Real, 3>{7.0, 8.0, 9.0}));
}

TEST(LevelSetMaintenanceConfiguration, ReadsTheInflowLiteralWithStod) {
  auto input = makeLegacyInput();
  put(input.equation_parameters, "Enable_kinematic_reconciliation", "true");
  LegacyLevelSetBoundaryInput boundary{};
  boundary.type_defined = true;
  boundary.name_defined = true;
  boundary.type = "LevelSetInflow";
  boundary.name = "inlet";
  put(boundary.parameters, "Value", "invalid");
  input.boundaries.push_back(boundary);

  EXPECT_EQ((exceptionMessage<std::invalid_argument>(
                [&] { (void)resolveInstalled(input); })),
            "stod");

  input.boundaries.clear();
  EXPECT_TRUE(resolveInstalled(input).has_value());

  put(input.equation_parameters, "Enable_kinematic_reconciliation", "false");
  EXPECT_FALSE(resolveInstalled(input).has_value());
}

TEST(LevelSetMaintenanceConfiguration,
     PreservesBoundaryFlagsAliasesNamesAndLiteralSelection) {
  auto input = makeLegacyInput();
  put(input.equation_parameters, "Enable_kinematic_reconciliation", "true");

  LegacyLevelSetBoundaryInput undefined_type{};
  undefined_type.name_defined = true;
  undefined_type.type = "inflow";
  undefined_type.name = "omitted_type";
  input.boundaries.push_back(undefined_type);

  LegacyLevelSetBoundaryInput undefined_name{};
  undefined_name.type_defined = true;
  undefined_name.type = "inflow";
  undefined_name.name = "omitted_name";
  input.boundaries.push_back(undefined_name);

  LegacyLevelSetBoundaryInput inflow{};
  inflow.type_defined = true;
  inflow.name_defined = true;
  inflow.type = " Level_Set-Dirichlet ";
  inflow.name = "  inlet  ";
  put(inflow.parameters, "Value", "1.25tail", false);
  put(inflow.parameters, "Level_set_value", "9.0");
  input.boundaries.push_back(inflow);

  LegacyLevelSetBoundaryInput outflow{};
  outflow.type_defined = true;
  outflow.name_defined = true;
  outflow.type = "outflow";
  outflow.name = " outlet ";
  put(outflow.parameters, "Value", "invalid");
  input.boundaries.push_back(outflow);

  LegacyLevelSetBoundaryInput unrelated{};
  unrelated.type_defined = true;
  unrelated.name_defined = true;
  unrelated.type = "wall";
  unrelated.name = "wall";
  input.boundaries.push_back(unrelated);

  const auto resolved = resolveInstalled(input);

  ASSERT_TRUE(resolved.has_value());
  ASSERT_EQ((*resolved)->open_boundaries.size(), 2u);
  const auto &boundaries = (*resolved)->open_boundaries;
  EXPECT_EQ(boundaries[0].face_name, "inlet");
  EXPECT_TRUE(boundaries[0].inflow);
  ASSERT_TRUE(boundaries[0].literal_inflow_value.has_value());
  EXPECT_DOUBLE_EQ(*boundaries[0].literal_inflow_value, 1.25);
  EXPECT_EQ(boundaries[1].face_name, "outlet");
  EXPECT_FALSE(boundaries[1].inflow);
  EXPECT_FALSE(boundaries[1].literal_inflow_value.has_value());
}

TEST(LevelSetMaintenanceConfiguration,
     AssociatesExactActiveCutBeforeFieldFallbackAndPreservesOptionals) {
  auto input = makeLegacyInput();
  put(input.equation_parameters, "Level_set_field_name", "phi");
  put(input.equation_parameters, "Enable_kinematic_reconciliation", "true");

  ActiveCutVolumeRequest other{};
  other.level_set_field_name = "other";
  other.domain_id = "other_domain";

  ActiveCutVolumeRequest fallback{};
  fallback.level_set_field_name = "phi";
  fallback.domain_id = "fallback_domain";
  fallback.isovalue = -2.0;

  ActiveCutVolumeRequest exact{};
  exact.level_set_field_name = "phi";
  exact.domain_id = "exact_domain";
  exact.isovalue = 0.5 + 5.0e-13;
  exact.requested_interface_marker = 17;

  const std::array<ActiveCutVolumeRequest, 3> requests{other, fallback, exact};
  auto resolved = resolveInstalled(input, requests, 0.5);
  ASSERT_TRUE(resolved.has_value());
  ASSERT_TRUE((*resolved)->volume_cut_request.has_value());
  EXPECT_EQ((*resolved)->volume_cut_request->domain_id, "exact_domain");
  EXPECT_EQ((*resolved)->volume_cut_request->requested_interface_marker, 17);
  EXPECT_FALSE((*resolved)->volume_cut_request->quadrature_order.has_value());
  EXPECT_FALSE(
      (*resolved)->volume_cut_request->interface_quadrature_order.has_value());
  EXPECT_FALSE(
      (*resolved)->volume_cut_request->volume_quadrature_order.has_value());

  const std::array<ActiveCutVolumeRequest, 2> fallback_requests{other,
                                                                fallback};
  resolved = resolveInstalled(input, fallback_requests, 0.5);
  ASSERT_TRUE(resolved.has_value());
  ASSERT_TRUE((*resolved)->volume_cut_request.has_value());
  EXPECT_EQ((*resolved)->volume_cut_request->domain_id, "fallback_domain");
}

TEST(LevelSetMaintenanceConfiguration,
     ResolvesRetainedHandlesInInputOrderAcrossNullSlots) {
  auto first_input = makeLegacyInput();
  put(first_input.equation_parameters, "Level_set_field_name", "phi_first");
  put(first_input.equation_parameters, "Enable_sign_definite_patch_bounds",
      "true");
  auto second_input = makeLegacyInput();
  put(second_input.equation_parameters, "Level_set_field_name", "phi_second");
  put(second_input.equation_parameters, "Enable_kinematic_reconciliation",
      "true");

  auto first = makeInstallation(first_input, "installed_first");
  auto second = makeInstallation(second_input, "installed_second");
  std::weak_ptr<const LevelSetEquationInputSnapshot> retained_snapshot =
      first->input_snapshot;
  std::vector<ResolvedLevelSetEquationHandle> equations{
      first, ResolvedLevelSetEquationHandle{}, second};

  const auto resolved =
      application::core::resolveLevelSetMaintenanceConfigurations(
          equations, std::span<const ActiveCutVolumeRequest>{});

  ASSERT_EQ(resolved.size(), 2u);
  EXPECT_EQ(resolved[0]->transport.level_set.field_name, "phi_first");
  EXPECT_EQ(resolved[0]->installation, first);
  EXPECT_EQ(resolved[1]->transport.level_set.field_name, "phi_second");
  EXPECT_EQ(resolved[1]->installation, second);
  equations.clear();
  first.reset();
  second.reset();
  EXPECT_FALSE(retained_snapshot.expired());
  ASSERT_TRUE(resolved[0]->installation->input_snapshot);
  EXPECT_EQ(resolved[0]
                ->installation->input_snapshot->installation_input
                .equation_params.at("Level_set_field_name")
                .value,
            "phi_first");
}

TEST(LevelSetMaintenanceConfiguration,
     UsesInputEquationOrderForFailuresAndRejectsMissingLegacyAttachment) {
  const auto make_inflow = [](std::string value) {
    auto input = makeLegacyInput();
    put(input.equation_parameters, "Enable_kinematic_reconciliation", "true");
    LegacyLevelSetBoundaryInput boundary{};
    boundary.type_defined = true;
    boundary.name_defined = true;
    boundary.type = "LevelSetInflow";
    boundary.name = "inlet";
    put(boundary.parameters, "Value", std::move(value));
    input.boundaries.push_back(boundary);
    return makeInstallation(input);
  };
  const auto invalid = make_inflow("invalid");
  const auto overflow = make_inflow("1e9999");

  std::array<ResolvedLevelSetEquationHandle, 3> equations{
      invalid, ResolvedLevelSetEquationHandle{}, overflow};
  EXPECT_EQ((exceptionMessage<std::invalid_argument>([&] {
              (void)application::core::resolveLevelSetMaintenanceConfigurations(
                  equations, std::span<const ActiveCutVolumeRequest>{});
            })),
            "stod");

  equations = {overflow, ResolvedLevelSetEquationHandle{}, invalid};
  EXPECT_EQ((exceptionMessage<std::out_of_range>([&] {
              (void)application::core::resolveLevelSetMaintenanceConfigurations(
                  equations, std::span<const ActiveCutVolumeRequest>{});
            })),
            "stod");

  auto snapshot_without_legacy =
      std::make_shared<LevelSetEquationInputSnapshot>();
  snapshot_without_legacy->installation_input.equation_type = "level_set";
  auto missing = std::make_shared<ResolvedLevelSetEquationConfiguration>();
  missing->input_snapshot = std::move(snapshot_without_legacy);
  const std::array<ResolvedLevelSetEquationHandle, 1> missing_equations{
      missing};
  EXPECT_EQ((exceptionMessage<std::invalid_argument>([&] {
              (void)application::core::resolveLevelSetMaintenanceConfigurations(
                  missing_equations, std::span<const ActiveCutVolumeRequest>{});
            })),
            "[svMultiPhysics::Application] Cannot resolve level-set "
            "maintenance from a resolved equation without a legacy "
            "maintenance input attachment.");

  const std::array<ResolvedLevelSetEquationHandle, 1> null_equations{};
  EXPECT_TRUE(application::core::resolveLevelSetMaintenanceConfigurations(
                  null_equations, std::span<const ActiveCutVolumeRequest>{})
                  .empty());
}

TEST(LevelSetMaintenanceConfiguration, AcceptsANonfiniteStaticTolerance) {
  auto input = makeLegacyInput();
  put(input.equation_parameters, "Enable_sign_definite_patch_bounds", "true");
  put(input.equation_parameters, "Static_capillary_volume_tolerance", "nan");

  const auto resolved = resolveInstalled(input);
  ASSERT_TRUE(resolved.has_value());
  EXPECT_TRUE((*resolved)->transport.sign_definite_patch_bounds.enabled);
  EXPECT_TRUE(
      std::isnan((*resolved)->static_capillary_equilibrium.volume_tolerance));
}

TEST(LevelSetMaintenanceConfiguration,
     RetainsTheInstallationVelocityMetadata) {
  auto legacy = makeLegacyInput();
  put(legacy.equation_parameters, "Enable_kinematic_reconciliation", "true");
  auto installation = makeInstallation(legacy);
  auto mutable_installation =
      std::const_pointer_cast<ResolvedLevelSetEquationConfiguration>(
          installation);
  mutable_installation->options.velocity.auto_register_field = true;
  mutable_installation->options.velocity.algebraic_extension_source_field_name =
      "physical_velocity";

  const auto resolved =
      application::core::resolveLegacyLevelSetMaintenanceConfiguration(
          legacy, std::span<const ActiveCutVolumeRequest>{}, installation);
  ASSERT_TRUE(resolved.has_value());
  ASSERT_TRUE(*resolved);
  EXPECT_TRUE((*resolved)->installation->options.velocity.auto_register_field);
  EXPECT_EQ((*resolved)
                ->installation->options.velocity
                .algebraic_extension_source_field_name,
            "physical_velocity");
  EXPECT_EQ((*resolved)->installation->options.velocity.space, nullptr);
}
