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

const std::string kFastMarchingDiagnostic =
    "[svMultiPhysics::Application] "
    "Reinitialization_method=FastMarching is reserved until runtime "
    "fast-marching reinitialization is implemented; use 'Projection'.";

const std::string kCurvatureDiagnostic =
    "level-set curvature recovery mode 'invalid_recovery' must be "
    "level_set_quadratic, generated_interface_patch, or "
    "kinematic_area_gradient";

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
  put(legacy.equation_parameters, "Enable_reinitialization", "true");
  put(legacy.equation_parameters, "Reinitialization_cadence_steps", "2", false);
  put(legacy.equation_parameters, "ReinitializationCadenceSteps", "9");
  put(legacy.equation_parameters, "Reinitialization_signed_distance_tolerance",
      "1.23457", false);
  put(legacy.equation_parameters, "Projected_curvature_field", " k_equation ");

  auto installation = makeInstallation(legacy, "phi_domain");
  auto mutable_installation =
      std::const_pointer_cast<ResolvedLevelSetEquationConfiguration>(
          installation);
  mutable_installation->options.operator_tag = "domain_transport";
  mutable_installation->options.transport_form =
      svmp::FE::level_set::LevelSetTransportForm::ConservativeDivergence;
  mutable_installation->options.reinitialization.enabled = true;
  mutable_installation->options.reinitialization.cadence_steps = 7;
  mutable_installation->options.reinitialization.signed_distance_tolerance =
      1.2345678901234567;
  mutable_installation->projected_curvature_fields = {"k_equation", "k_default",
                                                      "k_domain"};

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
  EXPECT_TRUE(configuration.transport.reinitialization.enabled);
  EXPECT_EQ(configuration.transport.reinitialization.cadence_steps, 2);
  EXPECT_DOUBLE_EQ(
      configuration.transport.reinitialization.signed_distance_tolerance,
      1.23457);
  EXPECT_FALSE(configuration.transport.supg.enabled);
  EXPECT_FALSE(configuration.transport.interface_kinematic.enabled);
  EXPECT_DOUBLE_EQ(configuration.isovalue, 0.25);
  EXPECT_TRUE(configuration.curvature_projection_enabled);
  EXPECT_EQ(configuration.curvature_field_name, "k_equation");
  EXPECT_DOUBLE_EQ(configuration.curvature_projection.isovalue, 0.25);

  EXPECT_EQ(installation->options.level_set.field_name, "phi_domain");
  EXPECT_EQ(installation->options.transport_form,
            svmp::FE::level_set::LevelSetTransportForm::ConservativeDivergence);
  EXPECT_EQ(installation->options.reinitialization.cadence_steps, 7);
  EXPECT_DOUBLE_EQ(
      installation->options.reinitialization.signed_distance_tolerance,
      1.2345678901234567);
}

TEST(LevelSetMaintenanceConfiguration,
     OmitsUndefinedUnrecognizedAndFullyDisabledEquations) {
  auto undefined = makeLegacyInput();
  undefined.equation_type_defined = false;
  put(undefined.equation_parameters, "Enable_reinitialization", "true");
  EXPECT_FALSE(application::core::resolveLegacyLevelSetMaintenanceConfiguration(
                   undefined, std::span<const ActiveCutVolumeRequest>{})
                   .has_value());

  auto unrecognized = makeLegacyInput("fluid");
  put(unrecognized.equation_parameters, "Enable_reinitialization", "true");
  EXPECT_FALSE(application::core::resolveLegacyLevelSetMaintenanceConfiguration(
                   unrecognized, std::span<const ActiveCutVolumeRequest>{})
                   .has_value());

  auto disabled = makeLegacyInput();
  put(disabled.equation_parameters, "Enable_reinitialization", "false");
  EXPECT_FALSE(application::core::resolveLegacyLevelSetMaintenanceConfiguration(
                   disabled, std::span<const ActiveCutVolumeRequest>{})
                   .has_value());
}

TEST(LevelSetMaintenanceConfiguration,
     RetainsLegacyNumericVectorBooleanAndAliasPolicies) {
  struct AcceptedCase {
    std::string vector;
    std::string first_cadence;
    std::string second_cadence;
    int expected_cadence;
  };
  const std::array<AcceptedCase, 3> cases{{
      {"1.25, -0.5, 3.0", "-1", "8", -1},
      {"4.0 5.0 6.0", "   ", "8", 8},
      {"7.0,8.0,9.0", "3", "8", 3},
  }};

  for (const auto &test_case : cases) {
    auto input = makeLegacyInput();
    put(input.equation_parameters, "Velocity_source", "constant");
    put(input.equation_parameters, "Constant_velocity", test_case.vector);
    put(input.equation_parameters, "Enable_reinitialization", "true");
    put(input.equation_parameters, "Reinitialization_cadence_steps",
        test_case.first_cadence);
    put(input.equation_parameters, "ReinitializationCadenceSteps",
        test_case.second_cadence);
    put(input.equation_parameters, "Reinitialization_max_iterations", "2tail");
    put(input.equation_parameters, "Reinitialization_signed_distance_tolerance",
        "1.25tail");
    put(input.equation_parameters, "Enable_volume_correction", "unrecognized");
    put(input.equation_parameters, "Volume_correction_tolerance", "nan");

    const auto resolved =
        application::core::resolveLegacyLevelSetMaintenanceConfiguration(
            input, std::span<const ActiveCutVolumeRequest>{});

    ASSERT_TRUE(resolved.has_value());
    ASSERT_TRUE(*resolved);
    const auto &transport = (*resolved)->transport;
    EXPECT_EQ(transport.velocity.source,
              svmp::FE::level_set::LevelSetVelocitySource::ConstantVector);
    if (test_case.expected_cadence == -1) {
      EXPECT_EQ(transport.velocity.constant_value,
                (std::array<svmp::FE::Real, 3>{1.25, -0.5, 3.0}));
    } else if (test_case.expected_cadence == 8) {
      EXPECT_EQ(transport.velocity.constant_value,
                (std::array<svmp::FE::Real, 3>{4.0, 5.0, 6.0}));
    } else {
      EXPECT_EQ(transport.velocity.constant_value,
                (std::array<svmp::FE::Real, 3>{7.0, 8.0, 9.0}));
    }
    EXPECT_EQ(transport.reinitialization.cadence_steps,
              test_case.expected_cadence);
    EXPECT_EQ(transport.reinitialization.max_iterations, 2);
    EXPECT_DOUBLE_EQ(transport.reinitialization.signed_distance_tolerance,
                     1.25);
    EXPECT_FALSE(transport.volume_correction.enabled);
    EXPECT_TRUE(std::isnan(transport.volume_correction.volume_tolerance));
  }

  auto zero = makeLegacyInput();
  put(zero.equation_parameters, "Enable_reinitialization", "true");
  put(zero.equation_parameters, "Reinitialization_max_iterations", "0");
  const auto zero_resolved =
      application::core::resolveLegacyLevelSetMaintenanceConfiguration(
          zero, std::span<const ActiveCutVolumeRequest>{});
  ASSERT_TRUE(zero_resolved.has_value());
  EXPECT_EQ((*zero_resolved)->transport.reinitialization.max_iterations, 0);
}

TEST(LevelSetMaintenanceConfiguration,
     ReportsLegacyLexicalFailureCategoriesAndText) {
  const auto make_constant = [] {
    auto input = makeLegacyInput();
    put(input.equation_parameters, "Velocity_source", "constant");
    put(input.equation_parameters, "Constant_velocity", "0.0 0.0 0.0");
    put(input.equation_parameters, "Enable_reinitialization", "true");
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
      "Reinitialization_signed_distance_tolerance", "invalid");
  EXPECT_EQ(
      (exceptionMessage<std::invalid_argument>([&] {
        (void)application::core::resolveLegacyLevelSetMaintenanceConfiguration(
            invalid_real, std::span<const ActiveCutVolumeRequest>{});
      })),
      "stod");

  auto overflow_real = make_constant();
  put(overflow_real.equation_parameters,
      "Reinitialization_signed_distance_tolerance", "1e9999");
  EXPECT_EQ(
      (exceptionMessage<std::out_of_range>([&] {
        (void)application::core::resolveLegacyLevelSetMaintenanceConfiguration(
            overflow_real, std::span<const ActiveCutVolumeRequest>{});
      })),
      "stod");

  auto overflow_count = make_constant();
  put(overflow_count.equation_parameters, "Reinitialization_cadence_steps",
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
  put(input.equation_parameters, "Advection_velocity_from_field",
      "physical_velocity");
  put(input.equation_parameters, "Enable_reinitialization", "true");
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
  put(input.equation_parameters, "Enable_reinitialization", "true");
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
     PreservesVelocityPhaseBoundaryReinitializationAndCurvatureOrder) {
  auto input = makeLegacyInput();
  put(input.equation_parameters, "Velocity_source", "invalid_velocity");
  put(input.equation_parameters, "Conservative_phase_liquid_side",
      "invalid_phase");
  put(input.equation_parameters, "Enable_reinitialization", "true");
  put(input.equation_parameters, "Reinitialization_method", "FastMarching");
  put(input.equation_parameters, "Curvature_projection_recovery_mode",
      "invalid_recovery");
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
  EXPECT_EQ((exceptionMessage<std::runtime_error>([&] { (void)resolve(); })),
            kFastMarchingDiagnostic);

  input.equation_parameters.erase("Reinitialization_method");
  put(input.equation_parameters, "Enable_reinitialization", "false");
  EXPECT_EQ((exceptionMessage<std::invalid_argument>([&] { (void)resolve(); })),
            kCurvatureDiagnostic);

  input.equation_parameters.erase("Curvature_projection_recovery_mode");
  EXPECT_FALSE(resolve().has_value());
}

TEST(LevelSetMaintenanceConfiguration,
     PreservesBoundaryFlagsAliasesNamesAndLiteralSelection) {
  auto input = makeLegacyInput();
  put(input.equation_parameters, "Enable_reinitialization", "true");

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
  put(input.equation_parameters, "Enable_reinitialization", "true");

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
  put(first_input.equation_parameters, "Enable_reinitialization", "true");
  auto second_input = makeLegacyInput();
  put(second_input.equation_parameters, "Level_set_field_name", "phi_second");
  put(second_input.equation_parameters, "Enable_volume_correction", "true");

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
  put(velocity.equation_parameters, "Enable_reinitialization", "true");
  auto phase = makeLegacyInput();
  put(phase.equation_parameters, "Conservative_phase_liquid_side",
      "invalid_phase");
  put(phase.equation_parameters, "Enable_reinitialization", "true");
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
  put(legacy.equation_parameters, "Enable_reinitialization", "true");
  put(legacy.equation_parameters, "Reinitialization_cadence_steps", "2", false);
  put(legacy.equation_parameters, "ReinitializationCadenceSteps", "9", true);
  put(legacy.equation_parameters, "Projected_curvature_field", "k_equation");

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
  mutable_installation->options.reinitialization.enabled = true;
  mutable_installation->options.reinitialization.cadence_steps = 7;
  mutable_installation->projected_curvature_fields = {"k_equation", "k_domain"};
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

  const auto *legacy_cadence =
      findObservation((*resolved)->input_observations,
                      "reinitialization_cadence_steps", "legacy_getter");
  ASSERT_NE(legacy_cadence, nullptr);
  EXPECT_EQ(legacy_cadence->selected_spelling,
            "Reinitialization_cadence_steps");
  EXPECT_EQ(legacy_cadence->source_layer, "equation");
  EXPECT_FALSE(legacy_cadence->supplied);
  EXPECT_TRUE(legacy_cadence->compatibility_fallback);
  EXPECT_TRUE(legacy_cadence->ordered_overrides.empty());

  for (const auto canonical :
       {"level_set_field_name", "transport_form", "velocity_source",
        "constant_velocity", "enable_reinitialization",
        "reinitialization_cadence_steps", "curvature_field_name"}) {
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
  const auto *derived_curvature_enable =
      findObservation((*resolved)->input_observations,
                      "enable_curvature_projection", "derived");
  ASSERT_NE(derived_curvature_enable, nullptr);
  EXPECT_TRUE(derived_curvature_enable->selected_spelling.empty());
  EXPECT_EQ(derived_curvature_enable->source_layer,
            "derived:curvature_field_name");

  const auto serialized =
      application::core::serializeLevelSetMaintenanceCompatibility(**resolved);
  RecordProperty("level_set_maintenance_compatibility", serialized);
  EXPECT_NE(serialized.find(
                R"json("constant_value":[1,2,3])json"),
            std::string::npos);
  EXPECT_NE(
      serialized.find(
          R"json("ordered_overrides":["equation","default_domain","domain:liquid"])json"),
      std::string::npos);
  EXPECT_NE(serialized.find(
                R"json("representation":"unavailable")json"),
            std::string::npos);
}

TEST(LevelSetMaintenanceConfiguration,
     SerializesUnavailableInstallationAndNonfiniteLegacyRealAsString) {
  auto input = makeLegacyInput();
  put(input.equation_parameters, "Enable_volume_correction", "true");
  put(input.equation_parameters, "Volume_correction_tolerance", "nan");

  const auto resolved =
      application::core::resolveLegacyLevelSetMaintenanceConfiguration(
          input, std::span<const ActiveCutVolumeRequest>{});
  ASSERT_TRUE(resolved.has_value());
  ASSERT_TRUE(
      std::isnan((*resolved)->transport.volume_correction.volume_tolerance));
  const auto *enabled =
      findObservation((*resolved)->input_observations,
                      "enable_volume_correction", "legacy_getter");
  ASSERT_NE(enabled, nullptr);
  EXPECT_EQ(enabled->selected_spelling, "Enable_volume_correction");
  EXPECT_TRUE(enabled->supplied);
  EXPECT_FALSE(enabled->compatibility_fallback);
  const auto *tolerance =
      findObservation((*resolved)->input_observations,
                      "volume_correction_tolerance", "legacy_getter");
  ASSERT_NE(tolerance, nullptr);
  EXPECT_EQ(tolerance->selected_spelling, "Volume_correction_tolerance");
  EXPECT_TRUE(tolerance->supplied);
  EXPECT_FALSE(tolerance->compatibility_fallback);
  const auto *default_field = findObservation(
      (*resolved)->input_observations, "level_set_field_name", "fe_default");
  ASSERT_NE(default_field, nullptr);
  EXPECT_TRUE(default_field->selected_spelling.empty());
  EXPECT_EQ(default_field->source_layer, "fe_default");

  const auto serialized =
      application::core::serializeLevelSetMaintenanceCompatibility(**resolved);
  RecordProperty("level_set_maintenance_compatibility", serialized);
  EXPECT_NE(serialized.find(
                R"json("installation":{"availability":"unavailable")json"),
            std::string::npos);
  EXPECT_NE(serialized.find(
                R"json("volume_tolerance":"nan")json"),
            std::string::npos);
  EXPECT_NE(serialized.find(
                R"json("selected_spelling":"Enable_volume_correction")json"),
            std::string::npos);
}

TEST(LevelSetMaintenanceConfiguration,
     RecordsOnlySuccessfulLegacyVelocityPromotionsInApplicatorOrder) {
  auto false_wet = makeLegacyInput();
  put(false_wet.equation_parameters,
      "Use_wet_extension_advection_velocity", "false");
  put(false_wet.equation_parameters, "Enable_reinitialization", "true");
  const auto false_wet_resolved =
      application::core::resolveLegacyLevelSetMaintenanceConfiguration(
          false_wet, std::span<const ActiveCutVolumeRequest>{});
  ASSERT_TRUE(false_wet_resolved.has_value());
  ASSERT_TRUE(*false_wet_resolved);
  EXPECT_EQ((*false_wet_resolved)->transport.velocity.source,
            svmp::FE::level_set::LevelSetVelocitySource::CoupledField);
  EXPECT_TRUE(observationLayers((*false_wet_resolved)->input_observations,
                                "velocity_source", "derived")
                  .empty());
  const auto *default_source =
      findObservation((*false_wet_resolved)->input_observations,
                      "velocity_source", "fe_default");
  EXPECT_NE(default_source, nullptr);
  if (default_source != nullptr) {
    EXPECT_EQ(default_source->source_layer, "fe_default");
  }

  auto ordered = makeLegacyInput();
  put(ordered.equation_parameters, "Velocity_source", "prescribed_data");
  put(ordered.equation_parameters,
      "Use_wet_extension_advection_velocity", "true");
  put(ordered.equation_parameters, "Constant_velocity", "1.0 2.0 3.0");
  put(ordered.equation_parameters, "Enable_reinitialization", "true");
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
     RecordsCurvatureTargetPromotionOverExplicitFalseSelection) {
  auto input = makeLegacyInput();
  put(input.equation_parameters, "Enable_curvature_projection", "false");
  put(input.equation_parameters, "Projected_curvature_field", "kappa");

  const auto resolved =
      application::core::resolveLegacyLevelSetMaintenanceConfiguration(
          input, std::span<const ActiveCutVolumeRequest>{});
  ASSERT_TRUE(resolved.has_value());
  ASSERT_TRUE(*resolved);
  EXPECT_TRUE((*resolved)->curvature_projection_enabled);
  EXPECT_EQ((*resolved)->curvature_field_name, "kappa");
  const auto *explicit_false =
      findObservation((*resolved)->input_observations,
                      "enable_curvature_projection", "legacy_getter");
  ASSERT_NE(explicit_false, nullptr);
  EXPECT_EQ(explicit_false->selected_spelling,
            "Enable_curvature_projection");
  EXPECT_EQ(observationLayers((*resolved)->input_observations,
                              "enable_curvature_projection", "derived"),
            (std::vector<std::string>{"derived:curvature_field_name"}));
}

TEST(LevelSetMaintenanceConfiguration,
     RecordsEnabledVolumeCorrectionDefaultToleranceProvenance) {
  auto input = makeLegacyInput();
  put(input.equation_parameters, "Enable_volume_correction", "true");

  const auto resolved =
      application::core::resolveLegacyLevelSetMaintenanceConfiguration(
          input, std::span<const ActiveCutVolumeRequest>{});
  ASSERT_TRUE(resolved.has_value());
  ASSERT_TRUE(*resolved);
  EXPECT_DOUBLE_EQ(
      (*resolved)->transport.volume_correction.volume_tolerance, 1.0e-10);
  const auto *tolerance =
      findObservation((*resolved)->input_observations,
                      "volume_correction_tolerance", "fe_default");
  EXPECT_NE(tolerance, nullptr);
  if (tolerance != nullptr) {
    EXPECT_TRUE(tolerance->selected_spelling.empty());
    EXPECT_EQ(tolerance->source_layer, "fe_default");
  }

  const auto serialized =
      application::core::serializeLevelSetMaintenanceCompatibility(**resolved);
  EXPECT_NE(serialized.find(R"json("volume_tolerance":1e-10)json"),
            std::string::npos);
  EXPECT_NE(serialized.find(
                R"json("canonical_key":"volume_correction_tolerance")json"),
            std::string::npos);
}

TEST(LevelSetMaintenanceConfiguration,
     MarksInstallationVelocityMetadataUnavailableWithoutSnapshotEvidence) {
  auto legacy = makeLegacyInput();
  put(legacy.equation_parameters, "Enable_reinitialization", "true");
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
  const auto serialized =
      application::core::serializeLevelSetMaintenanceCompatibility(**resolved);
  EXPECT_NE(serialized.find(R"json("auto_register_field":true)json"),
            std::string::npos);
  EXPECT_NE(serialized.find(
                R"json("source_field_name":"physical_velocity")json"),
            std::string::npos);
  EXPECT_NE(serialized.find(R"json("space_present":false)json"),
            std::string::npos);
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
    EXPECT_NE(serialized.find("\"canonical_key\":\"" +
                              std::string(canonical) + "\""),
              std::string::npos)
        << canonical;
  }
}
