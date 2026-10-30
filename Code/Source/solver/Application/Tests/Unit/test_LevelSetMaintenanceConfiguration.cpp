#include <gtest/gtest.h>

#include "Application/Core/LevelSetCutConfiguration.h"
#include "Application/Core/LevelSetMaintenanceConfiguration.h"
#include "Application/Core/ResolvedMovingDomainConfiguration.h"

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

using application::core::ActiveCutVolumeRequest;
using application::core::LegacyLevelSetBoundaryInput;
using application::core::LegacyLevelSetMaintenanceInput;
using application::core::LevelSetEquationInputSnapshot;
using application::core::LevelSetInputObservation;
using application::core::ResolvedLevelSetEquationConfiguration;
using application::core::ResolvedLevelSetEquationHandle;
using svmp::Physics::ParameterMap;
using svmp::Physics::ParameterValue;

void put(ParameterMap &parameters, std::string key, std::string value,
         bool defined = true) {
  parameters[std::move(key)] = ParameterValue{defined, std::move(value)};
}

LegacyLevelSetMaintenanceInput makeLegacyInput(std::string type = "level_set") {
  LegacyLevelSetMaintenanceInput input{};
  input.equation_type_defined = true;
  input.equation_type = std::move(type);
  return input;
}

ResolvedLevelSetEquationHandle
makeInstallation(const LegacyLevelSetMaintenanceInput &legacy,
                 std::string field_name = "phi_installed") {
  auto snapshot = std::make_shared<LevelSetEquationInputSnapshot>();
  snapshot->legacy_maintenance_input = legacy;
  snapshot->installation_input.equation_type = "level_set";
  snapshot->installation_input.equation_params["Level_set_field_name"] =
      ParameterValue{true, field_name};

  auto installation = std::make_shared<ResolvedLevelSetEquationConfiguration>();
  installation->options.level_set.field_name = std::move(field_name);
  installation->input_snapshot = std::move(snapshot);
  return installation;
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

const std::string kVelocityDiagnostic =
    "[svMultiPhysics::Application] Level-set Velocity_source must be one of "
    "'coupled_field', 'prescribed_data', 'constant_vector', or "
    "'material_interface_phase_pair'.";

const std::string kPhaseDiagnostic =
    "[svMultiPhysics::Application] Conservative_phase_liquid_side must be "
    "'negative' or 'positive'.";

const LevelSetInputObservation *
findObservation(const std::vector<LevelSetInputObservation> &observations,
                std::string_view canonical_key,
                std::string_view representation) {
  const auto found = std::find_if(
      observations.begin(), observations.end(), [&](const auto &observation) {
        return observation.canonical_key == canonical_key &&
               observation.representation == representation;
      });
  return found == observations.end() ? nullptr : &*found;
}

std::vector<std::string>
observationLayers(const std::vector<LevelSetInputObservation> &observations,
                  std::string_view canonical_key,
                  std::string_view representation) {
  std::vector<std::string> result;
  for (const auto &observation : observations) {
    if (observation.canonical_key == canonical_key &&
        observation.representation == representation) {
      result.push_back(observation.source_layer);
    }
  }
  return result;
}

} // namespace

TEST(LevelSetMaintenanceConfiguration,
     ProducesEquationOnlyTypedViewAndRetainsLayeredInstallation) {
  auto legacy = makeLegacyInput(" Level_Set-Transport ");
  put(legacy.equation_parameters, "Level_set_field_name", " phi_equation ");
  put(legacy.equation_parameters, "Level_set_isovalue", "0.25");
  put(legacy.equation_parameters, "Transport_form", "advective");
  put(legacy.equation_parameters, "Enable_kinematic_reconciliation", "true");
  put(legacy.equation_parameters,
      "Conservative_phase_flux_artifact_cadence_steps", "2", false);
  put(legacy.equation_parameters,
      "Conservative_phase_geometry_measure_tolerance", "1.23457", false);

  auto installation = makeInstallation(legacy, "phi_domain");
  auto mutable_installation =
      std::const_pointer_cast<ResolvedLevelSetEquationConfiguration>(
          installation);
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
  EXPECT_FALSE(application::core::resolveLegacyLevelSetMaintenanceConfiguration(
                   undefined, std::span<const ActiveCutVolumeRequest>{})
                   .has_value());

  auto unrecognized = makeLegacyInput("fluid");
  put(unrecognized.equation_parameters, "Enable_kinematic_reconciliation", "true");
  EXPECT_FALSE(application::core::resolveLegacyLevelSetMaintenanceConfiguration(
                   unrecognized, std::span<const ActiveCutVolumeRequest>{})
                   .has_value());

  auto disabled = makeLegacyInput();
  put(disabled.equation_parameters, "Enable_kinematic_reconciliation", "false");
  EXPECT_FALSE(application::core::resolveLegacyLevelSetMaintenanceConfiguration(
                   disabled, std::span<const ActiveCutVolumeRequest>{})
                   .has_value());
}

TEST(LevelSetMaintenanceConfiguration,
     RetainsLegacyNumericVectorBooleanAndAliasPolicies) {
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
    put(input.equation_parameters,
        "Conservative_phase_geometry_correction_max_iterations", "2tail");
    put(input.equation_parameters,
        "Conservative_phase_geometry_measure_tolerance", "1.25tail");
    put(input.equation_parameters, "Enable_sign_definite_patch_bounds",
        "unrecognized");
    put(input.equation_parameters, "Static_capillary_volume_tolerance", "nan");

    const auto resolved =
        application::core::resolveLegacyLevelSetMaintenanceConfiguration(
            input, std::span<const ActiveCutVolumeRequest>{});

    ASSERT_TRUE(resolved.has_value());
    ASSERT_TRUE(*resolved);
    const auto &transport = (*resolved)->transport;
    EXPECT_EQ(transport.velocity.source,
              svmp::FE::level_set::LevelSetVelocitySource::ConstantVector);
    EXPECT_EQ(transport.velocity.constant_value, test_case.expected_vector);
    EXPECT_EQ(transport.conservative_phase.flux_artifact_cadence_steps,
              test_case.expected_cadence);
    EXPECT_EQ(transport.conservative_phase.geometry_correction_max_iterations,
              2);
    EXPECT_DOUBLE_EQ(transport.conservative_phase.geometry_measure_tolerance,
                     1.25);
    EXPECT_FALSE(transport.sign_definite_patch_bounds.enabled);
    EXPECT_TRUE(std::isnan(
        (*resolved)->static_capillary_equilibrium.volume_tolerance));
  }

  auto zero = makeLegacyInput();
  put(zero.equation_parameters, "Enable_kinematic_reconciliation", "true");
  put(zero.equation_parameters,
      "Conservative_phase_geometry_correction_max_iterations", "0");
  const auto zero_resolved =
      application::core::resolveLegacyLevelSetMaintenanceConfiguration(
          zero, std::span<const ActiveCutVolumeRequest>{});
  ASSERT_TRUE(zero_resolved.has_value());
  EXPECT_EQ((*zero_resolved)
                ->transport.conservative_phase.geometry_correction_max_iterations,
            0);
}

TEST(LevelSetMaintenanceConfiguration,
     ReportsLegacyLexicalFailureCategoriesAndText) {
  const auto make_constant = [] {
    auto input = makeLegacyInput();
    put(input.equation_parameters, "Velocity_source", "constant");
    put(input.equation_parameters, "Constant_velocity", "0.0 0.0 0.0");
    put(input.equation_parameters, "Enable_kinematic_reconciliation", "true");
    return input;
  };

  for (const std::string value :
       {"1.0 2.0", "1.0 2.0 3.0 4.0", "1.0 nan 3.0"}) {
    auto invalid = make_constant();
    put(invalid.equation_parameters, "Constant_velocity", value);
    EXPECT_EQ(
        (exceptionMessage<std::runtime_error>([&] {
          (void)
              application::core::resolveLegacyLevelSetMaintenanceConfiguration(
                  invalid, std::span<const ActiveCutVolumeRequest>{});
        })),
        "[svMultiPhysics::Application] Constant_velocity must contain "
        "exactly three finite numeric components.");
  }

  auto invalid_real = make_constant();
  put(invalid_real.equation_parameters,
      "Conservative_phase_geometry_measure_tolerance", "invalid");
  EXPECT_EQ(
      (exceptionMessage<std::invalid_argument>([&] {
        (void)application::core::resolveLegacyLevelSetMaintenanceConfiguration(
            invalid_real, std::span<const ActiveCutVolumeRequest>{});
      })),
      "stod");

  auto overflow_real = make_constant();
  put(overflow_real.equation_parameters,
      "Conservative_phase_geometry_measure_tolerance", "1e9999");
  EXPECT_EQ(
      (exceptionMessage<std::out_of_range>([&] {
        (void)application::core::resolveLegacyLevelSetMaintenanceConfiguration(
            overflow_real, std::span<const ActiveCutVolumeRequest>{});
      })),
      "stod");

  auto overflow_count = make_constant();
  put(overflow_count.equation_parameters,
      "Conservative_phase_flux_artifact_cadence_steps",
      "999999999999999999999");
  EXPECT_EQ(
      (exceptionMessage<std::out_of_range>([&] {
        (void)application::core::resolveLegacyLevelSetMaintenanceConfiguration(
            overflow_count, std::span<const ActiveCutVolumeRequest>{});
      })),
      "stoi");
}

TEST(LevelSetMaintenanceConfiguration,
     PromotesWetExtensionBeforeConstantAndIgnoresInstallationOnlyKeys) {
  auto input = makeLegacyInput();
  put(input.equation_parameters, "Level_set_field_name", "phi");
  put(input.equation_parameters, "Velocity_field_name", "extended_velocity");
  put(input.equation_parameters, "Velocity_source", "prescribed_data");
  put(input.equation_parameters, "Source_velocity_field_name",
      "physical_velocity");
  put(input.equation_parameters, "Enable_kinematic_reconciliation", "true");
  put(input.equation_parameters, "Operator_tag", "ignored_operator");
  put(input.equation_parameters, "Level_set_source", "invalid_source");
  put(input.equation_parameters, "SUPG_tau_scale", "invalid_number");
  put(input.equation_parameters, "Interface_kinematic_marker",
      "invalid_marker");

  auto resolved =
      application::core::resolveLegacyLevelSetMaintenanceConfiguration(
          input, std::span<const ActiveCutVolumeRequest>{});
  ASSERT_TRUE(resolved.has_value());
  EXPECT_EQ((*resolved)->transport.velocity.source,
            svmp::FE::level_set::LevelSetVelocitySource::CoupledField);
  EXPECT_FALSE((*resolved)->transport.velocity.auto_register_field);
  EXPECT_TRUE(
      (*resolved)
          ->transport.velocity.algebraic_extension_source_field_name.empty());
  EXPECT_EQ((*resolved)->transport.velocity.space, nullptr);

  put(input.equation_parameters, "Constant_velocity", "7.0, 8.0, 9.0");
  resolved = application::core::resolveLegacyLevelSetMaintenanceConfiguration(
      input, std::span<const ActiveCutVolumeRequest>{});
  ASSERT_TRUE(resolved.has_value());
  EXPECT_EQ((*resolved)->transport.velocity.source,
            svmp::FE::level_set::LevelSetVelocitySource::ConstantVector);
  EXPECT_EQ((*resolved)->transport.velocity.constant_value,
            (std::array<svmp::FE::Real, 3>{7.0, 8.0, 9.0}));
}

TEST(LevelSetMaintenanceConfiguration,
     RejectsTheInstallationOnlyNavierStokesVelocityToken) {
  auto input = makeLegacyInput();
  put(input.equation_parameters, "Velocity_source", "navier_stokes");
  put(input.equation_parameters, "Enable_kinematic_reconciliation", "true");
  auto installation = makeInstallation(input);
  EXPECT_EQ(installation->options.velocity.source,
            svmp::FE::level_set::LevelSetVelocitySource::CoupledField);

  EXPECT_EQ(
      (exceptionMessage<std::runtime_error>([&] {
        (void)application::core::resolveLegacyLevelSetMaintenanceConfiguration(
            input, std::span<const ActiveCutVolumeRequest>{}, installation);
      })),
      kVelocityDiagnostic);
}

TEST(LevelSetMaintenanceConfiguration,
     PreservesVelocityPhaseAndBoundaryOrder) {
  auto input = makeLegacyInput();
  put(input.equation_parameters, "Velocity_source", "invalid_velocity");
  put(input.equation_parameters, "Conservative_phase_liquid_side",
      "invalid_phase");
  put(input.equation_parameters, "Enable_kinematic_reconciliation", "true");
  LegacyLevelSetBoundaryInput boundary{};
  boundary.type_defined = true;
  boundary.name_defined = true;
  boundary.type = "LevelSetInflow";
  boundary.name = "inlet";
  put(boundary.parameters, "Value", "invalid");
  input.boundaries.push_back(boundary);

  const auto resolve = [&] {
    return application::core::resolveLegacyLevelSetMaintenanceConfiguration(
        input, std::span<const ActiveCutVolumeRequest>{});
  };
  EXPECT_EQ((exceptionMessage<std::runtime_error>([&] { (void)resolve(); })),
            kVelocityDiagnostic);

  input.equation_parameters.erase("Velocity_source");
  EXPECT_EQ((exceptionMessage<std::runtime_error>([&] { (void)resolve(); })),
            kPhaseDiagnostic);

  input.equation_parameters.erase("Conservative_phase_liquid_side");
  EXPECT_EQ((exceptionMessage<std::invalid_argument>([&] { (void)resolve(); })),
            "stod");

  input.boundaries.clear();
  EXPECT_TRUE(resolve().has_value());

  put(input.equation_parameters, "Enable_kinematic_reconciliation", "false");
  EXPECT_FALSE(resolve().has_value());
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

  const auto resolved =
      application::core::resolveLegacyLevelSetMaintenanceConfiguration(
          input, std::span<const ActiveCutVolumeRequest>{});

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
  put(input.equation_parameters, "Level_set_isovalue", "0.5");
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
  auto resolved =
      application::core::resolveLegacyLevelSetMaintenanceConfiguration(
          input, requests);
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
  resolved = application::core::resolveLegacyLevelSetMaintenanceConfiguration(
      input, fallback_requests);
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
                ->installation->input_snapshot->legacy_maintenance_input
                ->equation_parameters.at("Level_set_field_name")
                .value,
            "phi_first");
}

TEST(LevelSetMaintenanceConfiguration,
     UsesInputEquationOrderForFailuresAndRejectsMissingLegacyAttachment) {
  auto velocity = makeLegacyInput();
  put(velocity.equation_parameters, "Velocity_source", "invalid_velocity");
  put(velocity.equation_parameters, "Enable_kinematic_reconciliation", "true");
  auto phase = makeLegacyInput();
  put(phase.equation_parameters, "Conservative_phase_liquid_side",
      "invalid_phase");
  put(phase.equation_parameters, "Enable_kinematic_reconciliation", "true");
  const auto velocity_installation = makeInstallation(velocity);
  const auto phase_installation = makeInstallation(phase);

  std::array<ResolvedLevelSetEquationHandle, 3> equations{
      velocity_installation, ResolvedLevelSetEquationHandle{},
      phase_installation};
  EXPECT_EQ((exceptionMessage<std::runtime_error>([&] {
              (void)application::core::resolveLevelSetMaintenanceConfigurations(
                  equations, std::span<const ActiveCutVolumeRequest>{});
            })),
            kVelocityDiagnostic);

  equations = {phase_installation, ResolvedLevelSetEquationHandle{},
               velocity_installation};
  EXPECT_EQ((exceptionMessage<std::runtime_error>([&] {
              (void)application::core::resolveLevelSetMaintenanceConfigurations(
                  equations, std::span<const ActiveCutVolumeRequest>{});
            })),
            kPhaseDiagnostic);

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

TEST(LevelSetMaintenanceConfiguration,
     SerializesPairedViewsProvenanceAndUnobservedFields) {
  auto legacy = makeLegacyInput();
  put(legacy.equation_parameters, "Level_set_field_name", "phi_equation");
  put(legacy.equation_parameters, "Transport_form", "advective");
  put(legacy.equation_parameters, "Velocity_source", "constant");
  put(legacy.equation_parameters, "Constant_velocity", "1.0, 2.0, 3.0");
  put(legacy.equation_parameters, "Enable_kinematic_reconciliation", "true");
  put(legacy.equation_parameters,
      "Conservative_phase_flux_artifact_cadence_steps", "2", false);

  auto installation = makeInstallation(legacy, "phi_domain");
  auto mutable_installation =
      std::const_pointer_cast<ResolvedLevelSetEquationConfiguration>(
          installation);
  mutable_installation->options.transport_form =
      svmp::FE::level_set::LevelSetTransportForm::ConservativeDivergence;
  mutable_installation->options.velocity.source =
      svmp::FE::level_set::LevelSetVelocitySource::CoupledField;
  mutable_installation->options.velocity.auto_register_field = true;
  mutable_installation->options.velocity.algebraic_extension_source_field_name =
      "physical_velocity";
  mutable_installation->options.conservative_phase
      .flux_artifact_cadence_steps = 7;
  auto mutable_snapshot =
      std::const_pointer_cast<LevelSetEquationInputSnapshot>(
          mutable_installation->input_snapshot);
  mutable_snapshot->installation_input.equation_params["Level_set_field_name"] =
      ParameterValue{true, "phi_equation"};
  mutable_snapshot->installation_input.default_domain
      .params["Level_set_field_name"] = ParameterValue{true, "phi_default"};
  svmp::Physics::DomainInput domain{};
  domain.id = "liquid";
  domain.params["Level_set_field_name"] = ParameterValue{true, "phi_domain"};
  mutable_snapshot->installation_input.domains.push_back(std::move(domain));

  const auto resolved =
      application::core::resolveLegacyLevelSetMaintenanceConfiguration(
          legacy, std::span<const ActiveCutVolumeRequest>{}, installation);
  ASSERT_TRUE(resolved.has_value());

  const auto *installation_field =
      findObservation((*resolved)->input_observations, "level_set_field_name",
                      "installation_snapshot");
  ASSERT_NE(installation_field, nullptr);
  EXPECT_EQ(installation_field->selected_spelling, "Level_set_field_name");
  EXPECT_EQ(installation_field->source_layer, "domain:liquid");
  EXPECT_TRUE(installation_field->supplied);
  EXPECT_FALSE(installation_field->compatibility_fallback);
  EXPECT_EQ(installation_field->ordered_overrides,
            (std::vector<std::string>{"equation", "default_domain",
                                      "domain:liquid"}));

  const auto *legacy_cadence = findObservation(
      (*resolved)->input_observations,
      "conservative_phase_flux_artifact_cadence_steps", "legacy_getter");
  ASSERT_NE(legacy_cadence, nullptr);
  EXPECT_EQ(legacy_cadence->selected_spelling,
            "Conservative_phase_flux_artifact_cadence_steps");
  EXPECT_EQ(legacy_cadence->source_layer, "equation");
  EXPECT_FALSE(legacy_cadence->supplied);
  EXPECT_TRUE(legacy_cadence->compatibility_fallback);
  EXPECT_TRUE(legacy_cadence->ordered_overrides.empty());

  for (const auto canonical :
       {"level_set_field_name", "transport_form", "velocity_source",
        "constant_velocity", "enable_kinematic_reconciliation",
        "conservative_phase_flux_artifact_cadence_steps"}) {
    const auto *selected = findObservation((*resolved)->input_observations,
                                           canonical, "legacy_getter");
    ASSERT_NE(selected, nullptr) << canonical;
    EXPECT_EQ(selected->source_layer, "equation") << canonical;
    EXPECT_FALSE(selected->selected_spelling.empty()) << canonical;
  }
  const auto *unknown_installation_form = findObservation(
      (*resolved)->input_observations, "transport_form", "unavailable");
  ASSERT_NE(unknown_installation_form, nullptr);
  EXPECT_TRUE(unknown_installation_form->selected_spelling.empty());
  EXPECT_EQ(unknown_installation_form->source_layer, "unavailable");

  EXPECT_EQ((*resolved)->transport.velocity.source,
            svmp::FE::level_set::LevelSetVelocitySource::ConstantVector);
  EXPECT_EQ((*resolved)->transport.velocity.constant_value,
            (std::array<svmp::FE::Real, 3>{1.0, 2.0, 3.0}));
}

TEST(LevelSetMaintenanceConfiguration,
     ResolvesUnavailableInstallationAndNonfiniteLegacyTolerance) {
  auto input = makeLegacyInput();
  put(input.equation_parameters, "Enable_sign_definite_patch_bounds", "true");
  put(input.equation_parameters, "Static_capillary_volume_tolerance", "nan");

  const auto resolved =
      application::core::resolveLegacyLevelSetMaintenanceConfiguration(
          input, std::span<const ActiveCutVolumeRequest>{});
  ASSERT_TRUE(resolved.has_value());
  EXPECT_EQ((*resolved)->installation, nullptr);
  ASSERT_TRUE(
      std::isnan((*resolved)->static_capillary_equilibrium.volume_tolerance));
  const auto *enabled =
      findObservation((*resolved)->input_observations,
                      "enable_sign_definite_patch_bounds", "legacy_getter");
  ASSERT_NE(enabled, nullptr);
  EXPECT_EQ(enabled->selected_spelling, "Enable_sign_definite_patch_bounds");
  EXPECT_TRUE(enabled->supplied);
  EXPECT_FALSE(enabled->compatibility_fallback);
  const auto *tolerance =
      findObservation((*resolved)->input_observations,
                      "static_capillary_volume_tolerance", "legacy_getter");
  ASSERT_NE(tolerance, nullptr);
  EXPECT_EQ(tolerance->selected_spelling, "Static_capillary_volume_tolerance");
  EXPECT_TRUE(tolerance->supplied);
  EXPECT_FALSE(tolerance->compatibility_fallback);
  const auto *default_field = findObservation(
      (*resolved)->input_observations, "level_set_field_name", "fe_default");
  ASSERT_NE(default_field, nullptr);
  EXPECT_TRUE(default_field->selected_spelling.empty());
  EXPECT_EQ(default_field->source_layer, "fe_default");
}

TEST(LevelSetMaintenanceConfiguration,
     RecordsOnlySuccessfulLegacyVelocityPromotionsInApplicatorOrder) {
  auto unpromoted = makeLegacyInput();
  put(unpromoted.equation_parameters, "Enable_kinematic_reconciliation", "true");
  const auto unpromoted_resolved =
      application::core::resolveLegacyLevelSetMaintenanceConfiguration(
          unpromoted, std::span<const ActiveCutVolumeRequest>{});
  ASSERT_TRUE(unpromoted_resolved.has_value());
  ASSERT_TRUE(*unpromoted_resolved);
  EXPECT_EQ((*unpromoted_resolved)->transport.velocity.source,
            svmp::FE::level_set::LevelSetVelocitySource::CoupledField);
  EXPECT_TRUE(observationLayers((*unpromoted_resolved)->input_observations,
                                "velocity_source", "derived")
                  .empty());
  const auto *default_source =
      findObservation((*unpromoted_resolved)->input_observations,
                      "velocity_source", "fe_default");
  EXPECT_NE(default_source, nullptr);
  if (default_source != nullptr) {
    EXPECT_EQ(default_source->source_layer, "fe_default");
  }

  auto ordered = makeLegacyInput();
  put(ordered.equation_parameters, "Velocity_source", "prescribed_data");
  put(ordered.equation_parameters, "Source_velocity_field_name", "Velocity");
  put(ordered.equation_parameters, "Constant_velocity", "1.0 2.0 3.0");
  put(ordered.equation_parameters, "Enable_kinematic_reconciliation", "true");
  const auto ordered_resolved =
      application::core::resolveLegacyLevelSetMaintenanceConfiguration(
          ordered, std::span<const ActiveCutVolumeRequest>{});
  ASSERT_TRUE(ordered_resolved.has_value());
  ASSERT_TRUE(*ordered_resolved);
  EXPECT_EQ((*ordered_resolved)->transport.velocity.source,
            svmp::FE::level_set::LevelSetVelocitySource::ConstantVector);
  EXPECT_EQ((*ordered_resolved)->transport.velocity.constant_value,
            (std::array<svmp::FE::Real, 3>{1.0, 2.0, 3.0}));
  EXPECT_EQ(observationLayers((*ordered_resolved)->input_observations,
                              "velocity_source", "derived"),
            (std::vector<std::string>{"derived:wet_extension",
                                      "derived:constant_velocity"}));
}

TEST(LevelSetMaintenanceConfiguration,
     MarksInstallationVelocityMetadataUnavailableWithoutSnapshotEvidence) {
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
  for (const std::string_view canonical :
       {"velocity_auto_register_field", "velocity_source_field_name",
        "velocity_space_present"}) {
    const auto *unavailable = findObservation(
        (*resolved)->input_observations, canonical, "unavailable");
    EXPECT_NE(unavailable, nullptr) << canonical;
    if (unavailable != nullptr) {
      EXPECT_TRUE(unavailable->selected_spelling.empty()) << canonical;
      EXPECT_EQ(unavailable->source_layer, "unavailable") << canonical;
    }
  }
}
