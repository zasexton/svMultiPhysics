#include <gtest/gtest.h>

#include "Application/Translators/LevelSetEquationTranslator.h"
#include "FE/Backends/Interfaces/BackendFactory.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Forms/FormExpr.h"
#include "FE/Forms/JIT/JITKernelWrapper.h"
#include "FE/Forms/JIT/LLVMJITBuildInfo.h"
#include "FE/Systems/FESystem.h"
#include "Interfaces/GeneratedActiveBoundaryDomain.h"
#include "Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Fields/MeshFields.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"
#include "Physics/Core/EquationModuleInput.h"
#include "Physics/Core/EquationModuleRegistry.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace svmp::Physics::formulations::navier_stokes {
void forceLink_NavierStokesRegister();
}

namespace {

using svmp::FE::forms::FormExpr;
using svmp::FE::forms::FormExprNode;
using svmp::FE::forms::FormExprType;

class ScopedUnsetEnvironmentVariable {
public:
  explicit ScopedUnsetEnvironmentVariable(const char* key) : key_(key)
  {
    if (const char* value = std::getenv(key_); value != nullptr) {
      original_ = std::string(value);
    }
    unsetenv(key_);
  }

  ~ScopedUnsetEnvironmentVariable()
  {
    if (original_.has_value()) {
      setenv(key_, original_->c_str(), 1);
    } else {
      unsetenv(key_);
    }
  }

  ScopedUnsetEnvironmentVariable(const ScopedUnsetEnvironmentVariable&) =
      delete;
  ScopedUnsetEnvironmentVariable& operator=(
      const ScopedUnsetEnvironmentVariable&) = delete;

private:
  const char* key_;
  std::optional<std::string> original_{};
};

bool containsExprType(const FormExprNode* node, FormExprType target)
{
  if (node == nullptr) {
    return false;
  }
  if (node->type() == target) {
    return true;
  }
  for (const auto* child : node->children()) {
    if (containsExprType(child, target)) {
      return true;
    }
  }
  return false;
}

bool containsExprType(const FormExpr& expr, FormExprType target)
{
  return expr.isValid() && containsExprType(expr.node(), target);
}

bool formulationRecordsContain(const svmp::FE::systems::FESystem& system,
                               FormExprType target)
{
  for (const auto& record : system.formulationRecords()) {
    if (containsExprType(record.residual_expr.get(), target)) {
      return true;
    }
    for (const auto& [block, expr] : record.block_residual_exprs) {
      (void)block;
      if (containsExprType(expr.get(), target)) {
        return true;
      }
    }
  }
  return false;
}

bool containsInterfaceMarker(const FormExprNode* node, int marker)
{
  if (node == nullptr) {
    return false;
  }
  const auto found = node->interfaceMarker();
  if (found.has_value() && *found == marker) {
    return true;
  }
  for (const auto* child : node->children()) {
    if (containsInterfaceMarker(child, marker)) {
      return true;
    }
  }
  return false;
}

const svmp::FE::forms::jit::JITKernelWrapper* asJitKernel(
    const std::shared_ptr<svmp::FE::assembly::AssemblyKernel>& kernel)
{
  return dynamic_cast<const svmp::FE::forms::jit::JITKernelWrapper*>(kernel.get());
}

const svmp::FE::forms::jit::JITKernelWrapper* firstJitKernelInOperator(
    const svmp::FE::systems::FESystem& system,
    const svmp::FE::systems::OperatorTag& op)
{
  if (!system.hasOperator(op)) {
    return nullptr;
  }

  const auto& def = system.operatorDefinition(op);
  for (const auto& term : def.cells) {
    if (const auto* jit = asJitKernel(term.kernel)) {
      return jit;
    }
  }
  for (const auto& term : def.boundary) {
    if (const auto* jit = asJitKernel(term.kernel)) {
      return jit;
    }
  }
  for (const auto& term : def.interior) {
    if (const auto* jit = asJitKernel(term.kernel)) {
      return jit;
    }
  }
  for (const auto& term : def.interface_faces) {
    if (const auto* jit = asJitKernel(term.kernel)) {
      return jit;
    }
  }
  return nullptr;
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
std::shared_ptr<svmp::Mesh> makeRegistryQuadMesh()
{
  auto base = std::make_shared<svmp::MeshBase>();

  const std::vector<svmp::real_t> x_ref = {
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,
  };
  const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 4};
  const std::vector<svmp::index_t> cell2vertex = {0, 1, 2, 3};

  svmp::CellShape shape{};
  shape.family = svmp::CellFamily::Quad;
  shape.num_corners = 4;
  shape.order = 1;
  base->build_from_arrays(
      /*spatial_dim=*/2,
      x_ref,
      cell2vertex_offsets,
      cell2vertex,
      {shape});
  base->finalize();

  return svmp::create_mesh(std::move(base));
}

std::shared_ptr<svmp::Mesh> makeRegistryBiquadraticQuadMesh()
{
  auto base = std::make_shared<svmp::MeshBase>();

  const std::vector<svmp::real_t> x_ref = {
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,
      0.5, 0.0,
      1.0, 0.5,
      0.5, 1.0,
      0.0, 0.5,
      0.5, 0.5,
  };
  const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 9};
  const std::vector<svmp::index_t> cell2vertex = {
      0, 1, 2, 3, 4, 5, 6, 7, 8};

  svmp::CellShape shape{};
  shape.family = svmp::CellFamily::Quad;
  shape.num_corners = 4;
  shape.order = 2;
  base->build_from_arrays(
      /*spatial_dim=*/2,
      x_ref,
      cell2vertex_offsets,
      cell2vertex,
      {shape});
  base->finalize();

  return svmp::create_mesh(std::move(base));
}
#endif

} // namespace

TEST(LevelSetEquationTranslator, RecognizesLegacyEquationTypes)
{
  EXPECT_TRUE(application::translators::level_set::isEquationType("level_set"));
  EXPECT_TRUE(application::translators::level_set::isEquationType("levelSet"));
  EXPECT_TRUE(application::translators::level_set::isEquationType("level_set_transport"));
  EXPECT_FALSE(application::translators::level_set::isEquationType("fluid"));
}

TEST(LevelSetEquationTranslator, RejectsUnsupportedRuntimeReinitializationMethods)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  for (const std::string method : {"HamiltonJacobiPDE", "FastMarching"}) {
    auto mesh = makeRegistryQuadMesh();
    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "level_set";
    input.mesh_name = "quad";
    input.mesh = mesh->local_mesh_ptr();
    input.equation_params["Level_set_field_name"] =
        svmp::Physics::ParameterValue{true, "phi"};
    input.equation_params["Velocity_field_name"] =
        svmp::Physics::ParameterValue{true, "advecting_velocity"};
    input.equation_params["Velocity_source"] =
        svmp::Physics::ParameterValue{true, "prescribed_data"};
    input.equation_params["Enable_reinitialization"] =
        svmp::Physics::ParameterValue{true, "true"};
    input.equation_params["Reinitialization_method"] =
        svmp::Physics::ParameterValue{true, method};

    svmp::FE::systems::FESystem system(mesh);
    EXPECT_THROW(
        (void)application::translators::level_set::createModule(input, system),
        std::runtime_error)
        << "method=" << method;
  }
#endif
}

TEST(LevelSetEquationTranslator, TranslatesFieldsAndBoundaries)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeRegistryQuadMesh();

  svmp::Physics::EquationModuleInput input{};
  input.equation_type = "level_set";
  input.mesh_name = "quad";
  input.mesh = mesh->local_mesh_ptr();
  input.equation_params["Level_set_field_name"] =
      svmp::Physics::ParameterValue{true, "phi"};
  input.equation_params["Velocity_field_name"] =
      svmp::Physics::ParameterValue{true, "advecting_velocity"};
  input.equation_params["Velocity_source"] =
      svmp::Physics::ParameterValue{true, "prescribed_data"};
  input.equation_params["Transport_form"] =
      svmp::Physics::ParameterValue{true, "conservative_divergence"};
  input.equation_params["Operator_tag"] =
      svmp::Physics::ParameterValue{true, "transport"};
  input.equation_params["Enable_SUPG"] =
      svmp::Physics::ParameterValue{true, "true"};
  input.equation_params["SUPG_tau_scale"] =
      svmp::Physics::ParameterValue{true, "0.25"};
  input.equation_params["SUPG_transient_scale"] =
      svmp::Physics::ParameterValue{true, "1.5"};
  input.equation_params["Enable_discontinuity_capturing"] =
      svmp::Physics::ParameterValue{true, "true"};
  input.equation_params["Discontinuity_capturing_scale"] =
      svmp::Physics::ParameterValue{true, "0.2"};
  input.equation_params["Discontinuity_capturing_gradient_epsilon"] =
      svmp::Physics::ParameterValue{true, "1.0e-9"};
  input.equation_params["Discontinuity_capturing_max_courant"] =
      svmp::Physics::ParameterValue{true, "0.3"};
  input.equation_params["Interface_kinematic_marker"] =
      svmp::Physics::ParameterValue{true, "77"};
  input.equation_params["Interface_kinematic_weight_scale"] =
      svmp::Physics::ParameterValue{true, "1.5"};
  input.equation_params["Enable_reinitialization"] =
      svmp::Physics::ParameterValue{true, "true"};
  input.equation_params["Reinitialization_method"] =
      svmp::Physics::ParameterValue{true, "projection"};
  input.equation_params["Reinitialization_cadence_steps"] =
      svmp::Physics::ParameterValue{true, "4"};
  input.equation_params["Reinitialization_max_iterations"] =
      svmp::Physics::ParameterValue{true, "8"};
  input.equation_params["Reinitialization_pseudo_time_step_scale"] =
      svmp::Physics::ParameterValue{true, "0.125"};
  input.equation_params["Reinitialization_interface_band_width"] =
      svmp::Physics::ParameterValue{true, "2.75"};
  input.equation_params["Reinitialization_signed_distance_tolerance"] =
      svmp::Physics::ParameterValue{true, "1.0e-4"};
  input.equation_params["Reinitialization_max_zero_set_displacement"] =
      svmp::Physics::ParameterValue{true, "2.0e-8"};
  input.equation_params["Enable_volume_correction"] =
      svmp::Physics::ParameterValue{true, "true"};
  input.equation_params["Volume_correction_cadence_steps"] =
      svmp::Physics::ParameterValue{true, "5"};
  input.equation_params["Volume_correction_use_initial_volume"] =
      svmp::Physics::ParameterValue{true, "false"};
  input.equation_params["Volume_correction_target_negative_volume"] =
      svmp::Physics::ParameterValue{true, "0.375"};
  input.equation_params["Volume_correction_tolerance"] =
      svmp::Physics::ParameterValue{true, "1.0e-7"};
  input.equation_params["Volume_correction_max_iterations"] =
      svmp::Physics::ParameterValue{true, "24"};
  input.equation_params["Volume_correction_minimum_relative_error"] =
      svmp::Physics::ParameterValue{true, "2.0e-5"};
  input.equation_params["Volume_correction_maximum_interface_displacement_fraction"] =
      svmp::Physics::ParameterValue{true, "0.025"};
  input.equation_params[
      "Volume_correction_maximum_cumulative_interface_displacement_fraction"] =
      svmp::Physics::ParameterValue{true, "0.25"};

  svmp::Physics::BoundaryConditionInput inflow{};
  inflow.name = "inlet";
  inflow.boundary_marker = 4;
  inflow.params["Type"] = svmp::Physics::ParameterValue{true, "LevelSetInflow"};
  inflow.params["Value"] = svmp::Physics::ParameterValue{true, "0.5"};
  inflow.params["Penalty_scale"] = svmp::Physics::ParameterValue{true, "2.0"};
  input.boundary_conditions.push_back(std::move(inflow));

  svmp::Physics::BoundaryConditionInput outflow{};
  outflow.name = "outlet";
  outflow.boundary_marker = 5;
  outflow.params["Type"] = svmp::Physics::ParameterValue{true, "LevelSetOutflow"};
  input.boundary_conditions.push_back(std::move(outflow));

  svmp::FE::systems::FESystem system(mesh);
  auto module = application::translators::level_set::createModule(input, system);

  ASSERT_TRUE(module);
  const auto phi = system.findFieldByName("phi");
  const auto velocity = system.findFieldByName("advecting_velocity");
  ASSERT_NE(phi, svmp::FE::INVALID_FIELD_ID);
  ASSERT_NE(velocity, svmp::FE::INVALID_FIELD_ID);
  EXPECT_EQ(system.fieldRecord(phi).source_kind,
            svmp::FE::systems::FieldSourceKind::Unknown);
  EXPECT_EQ(system.fieldRecord(velocity).source_kind,
            svmp::FE::systems::FieldSourceKind::PrescribedData);
  EXPECT_TRUE(system.hasOperator("transport"));
  EXPECT_TRUE(formulationRecordsContain(system, FormExprType::BoundaryIntegral));
  EXPECT_TRUE(formulationRecordsContain(system, FormExprType::InterfaceIntegral));
  EXPECT_TRUE(formulationRecordsContain(system, FormExprType::CellDiameter));
  EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Divergence));
  const auto artifact = module->effectiveConfigurationArtifact();
  ASSERT_TRUE(artifact.has_value());
  EXPECT_EQ(artifact->component, "level_set_transport");
  constexpr std::string_view expected_without_phase =
      R"json({"artifact_schema_version":1,"component":"level_set_transport","capability_label":"one_phase_interface_transport_nonlocal_conservation","units":{"system":"consistent_solver_units","length":"solver_length","time":"solver_time","volume":"solver_volume"},"operator":"transport","transport_form":"ConservativeDivergence","conservation_diagnostic":"volume_corrected_level_set_advection_not_locally_conservative","level_set":{"field":"phi","source":"Unknown","auto_register":true},"advection_velocity":{"field":"advecting_velocity","source":"PrescribedData","auto_register":true,"constant_value":[0,0,0],"algebraic_extension_source_field":"","dependency_direction":"physical_velocity_to_extension_to_level_set","physical_momentum_coupling_allowed":false,"map_guard_policy":"fixed_bounded_application_policy"},"supg":{"enabled":true,"tau_scale":0.25,"velocity_epsilon":9.9999999999999998e-13,"transient_scale":1.5,"discontinuity_capturing_enabled":true,"discontinuity_capturing_scale":0.20000000000000001,"gradient_epsilon":1.0000000000000001e-09,"residual_epsilon":9.9999999999999998e-13,"maximum_courant":0.29999999999999999},"bound_preserving":{"enabled":false,"method":"nodal_rejection_projection_nonconservative","bound_tolerance":9.9999999999999998e-13,"sign_tolerance":9.9999999999999998e-13,"maximum_courant":1,"courant_tolerance":9.9999999999999998e-13,"enforce_courant_limit":true,"enforce_impermeable_boundaries":true,"impermeable_normal_velocity_tolerance":1e-10},"interface_kinematic":{"enabled":true,"interface_marker":77,"weight_scale":1.5},"maintenance_transaction":{"ordering":"transport_then_reinitialization_then_volume_correction_then_geometry_refresh","reinitialization":{"enabled":true,"method":"Projection","cadence_steps":4,"max_iterations":8,"pseudo_time_step_scale":0.125,"interface_band_width":2.75,"signed_distance_tolerance":0.0001,"preserve_band_width":0,"maximum_zero_set_displacement":2e-08},"volume_correction":{"enabled":true,"cadence_steps":5,"use_initial_negative_volume_as_target":false,"target_negative_volume":0.375,"volume_tolerance":9.9999999999999995e-08,"max_iterations":24,"minimum_relative_volume_error":2.0000000000000002e-05,"maximum_interface_displacement_fraction":0.025000000000000001,"maximum_cumulative_interface_displacement_fraction":0.25}},"boundaries":{"inflow":[{"marker":4,"value":0.5,"penalty_scale":2}],"outflow":[{"marker":5}]}})json";
  std::string expected{expected_without_phase};
  const auto phase_insertion = expected.find(",\"advection_velocity\"");
  ASSERT_NE(phase_insertion, std::string::npos);
  expected.insert(
      phase_insertion,
      R"json(,"conservative_phase":{"enabled":false,"field":"liquid_indicator","source":"Unknown","auto_register":true,"liquid_side":"Negative","invariant_tolerance":9.9999999999999998e-13,"maximum_courant":1,"enforce_courant_limit":true,"require_constant_preservation":true,"impermeable_normal_velocity_tolerance":1e-10,"reconcile_geometry":true,"geometry_measure_tolerance":1e-10,"geometry_correction_max_iterations":50,"maximum_geometry_displacement_fraction":0.10000000000000001,"boundary_flux_policy":"closed_boundary_only","newton_policy":"held_at_previous_accepted_endpoint"})json");
  EXPECT_EQ(artifact->json, expected);
#endif
}

TEST(LevelSetEquationTranslator, TranslatesTemporalSpatialInflowBoundary)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  const std::string file_path = "level_set_inflow_temporal_spatial_test.dat";
  {
    std::ofstream out(file_path);
    out << "1 2 4\n";
    out << "0.0\n";
    out << "1.0\n";
    for (int node = 1; node <= 4; ++node) {
      out << node << "\n";
      out << static_cast<double>(node) << "\n";
      out << static_cast<double>(node) + 0.5 << "\n";
    }
  }

  auto mesh = makeRegistryQuadMesh();
  svmp::Physics::EquationModuleInput input{};
  input.equation_type = "level_set";
  input.mesh_name = "quad";
  input.mesh = mesh->local_mesh_ptr();
  input.equation_params["Level_set_field_name"] =
      svmp::Physics::ParameterValue{true, "phi"};
  input.equation_params["Velocity_source"] =
      svmp::Physics::ParameterValue{true, "constant"};
  input.equation_params["Constant_velocity"] =
      svmp::Physics::ParameterValue{true, "1.0 0.0 0.0"};

  svmp::Physics::BoundaryConditionInput inflow{};
  inflow.name = "inlet";
  inflow.boundary_marker = 4;
  inflow.params["Type"] = svmp::Physics::ParameterValue{true, "LevelSetInflow"};
  inflow.params["Temporal_and_spatial_values_file_path"] =
      svmp::Physics::ParameterValue{true, file_path};
  input.boundary_conditions.push_back(std::move(inflow));

  svmp::FE::systems::FESystem system(mesh);
  auto module = application::translators::level_set::createModule(input, system);
  std::remove(file_path.c_str());

  ASSERT_TRUE(module);
  EXPECT_TRUE(system.hasOperator("level_set"));
  EXPECT_TRUE(formulationRecordsContain(system, FormExprType::BoundaryIntegral));
  EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Coefficient));
#endif
}

TEST(LevelSetEquationTranslator, TranslatesConservativePhaseControls)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeRegistryQuadMesh();

  svmp::Physics::EquationModuleInput input{};
  input.equation_type = "level_set";
  input.mesh_name = "quad";
  input.mesh = mesh->local_mesh_ptr();
  input.equation_params["Level_set_field_name"] =
      svmp::Physics::ParameterValue{true, "phi"};
  input.equation_params["Velocity_source"] =
      svmp::Physics::ParameterValue{true, "constant"};
  input.equation_params["Constant_velocity"] =
      svmp::Physics::ParameterValue{true, "0.0 0.0 0.0"};
  input.equation_params["Enable_conservative_phase_transport"] =
      svmp::Physics::ParameterValue{true, "true"};
  input.equation_params["Conservative_phase_field_name"] =
      svmp::Physics::ParameterValue{true, "phase_fraction"};
  input.equation_params["Conservative_phase_liquid_side"] =
      svmp::Physics::ParameterValue{true, "positive"};
  input.equation_params["Conservative_phase_invariant_tolerance"] =
      svmp::Physics::ParameterValue{true, "3.0e-11"};
  input.equation_params["Conservative_phase_maximum_courant"] =
      svmp::Physics::ParameterValue{true, "0.45"};
  input.equation_params["Conservative_phase_enforce_courant_limit"] =
      svmp::Physics::ParameterValue{true, "false"};
  input.equation_params[
      "Conservative_phase_require_constant_preservation"] =
      svmp::Physics::ParameterValue{true, "false"};
  input.equation_params[
      "Conservative_phase_impermeable_normal_velocity_tolerance"] =
      svmp::Physics::ParameterValue{true, "7.0e-9"};
  input.equation_params["Conservative_phase_reconcile_geometry"] =
      svmp::Physics::ParameterValue{true, "false"};
  input.equation_params[
      "Conservative_phase_geometry_measure_tolerance"] =
      svmp::Physics::ParameterValue{true, "2.0e-9"};
  input.equation_params[
      "Conservative_phase_geometry_correction_max_iterations"] =
      svmp::Physics::ParameterValue{true, "19"};
  input.equation_params[
      "Conservative_phase_maximum_geometry_displacement_fraction"] =
      svmp::Physics::ParameterValue{true, "0.075"};

  svmp::FE::systems::FESystem system(mesh);
  auto module = application::translators::level_set::createModule(input, system);

  ASSERT_TRUE(module);
  const auto phase = system.findFieldByName("phase_fraction");
  ASSERT_NE(phase, svmp::FE::INVALID_FIELD_ID);
  EXPECT_EQ(system.fieldRecord(phase).source_kind,
            svmp::FE::systems::FieldSourceKind::Unknown);
  EXPECT_TRUE(system.fieldParticipatesInUnknownVector(phase));
  EXPECT_TRUE(formulationRecordsContain(
      system, FormExprType::PreviousSolutionRef));

  const auto artifact = module->effectiveConfigurationArtifact();
  ASSERT_TRUE(artifact.has_value());
  EXPECT_NE(artifact->json.find(
                "\"capability_label\":\"one_phase_locally_conservative_p1_indicator_transport\""),
            std::string::npos);
  EXPECT_NE(artifact->json.find(
                "\"conservative_phase\":{\"enabled\":true,\"field\":\"phase_fraction\""),
            std::string::npos);
  EXPECT_NE(artifact->json.find("\"liquid_side\":\"Positive\""),
            std::string::npos);
  EXPECT_NE(artifact->json.find("\"maximum_courant\":0.45000000000000001"),
            std::string::npos);
  EXPECT_NE(artifact->json.find("\"enforce_courant_limit\":false"),
            std::string::npos);
  EXPECT_NE(artifact->json.find("\"reconcile_geometry\":false"),
            std::string::npos);
  EXPECT_NE(artifact->json.find(
                "\"geometry_correction_max_iterations\":19"),
            std::string::npos);
  EXPECT_NE(artifact->json.find(
                "\"ordering\":\"provisional_level_set_then_conservative_phase_then_geometry_reconciliation_then_wall_aware_maintenance\""),
            std::string::npos);
#endif
}

TEST(LevelSetEquationTranslator, InitializesPrescribedLevelSetFromMeshVertexField)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeRegistryQuadMesh();
  auto& local_mesh = mesh->local_mesh();
  const auto field = svmp::MeshFields::attach_field(
      local_mesh,
      svmp::EntityKind::Vertex,
      "phi",
      svmp::FieldScalarType::Float64,
      1);
  auto* phi_values = svmp::MeshFields::field_data_as<svmp::real_t>(local_mesh, field);
  ASSERT_NE(phi_values, nullptr);
  for (svmp::index_t vertex = 0; vertex < local_mesh.n_vertices(); ++vertex) {
    const auto x = local_mesh.get_vertex_coords(vertex);
    phi_values[vertex] = x[0] + x[1] - 0.25;
  }

  svmp::Physics::EquationModuleInput input{};
  input.equation_type = "level_set";
  input.mesh_name = "quad";
  input.mesh = mesh->local_mesh_ptr();
  input.equation_params["Level_set_field_name"] =
      svmp::Physics::ParameterValue{true, "phi"};
  input.equation_params["Level_set_source"] =
      svmp::Physics::ParameterValue{true, "prescribed_data"};
  input.equation_params["Velocity_source"] =
      svmp::Physics::ParameterValue{true, "constant"};
  input.equation_params["Constant_velocity"] =
      svmp::Physics::ParameterValue{true, "0.0 0.0 0.0"};

  svmp::FE::systems::FESystem system(mesh);
  auto module = application::translators::level_set::createModule(input, system);
  ASSERT_TRUE(module);
  ASSERT_NO_THROW(system.setup({}));

  auto factory = svmp::FE::backends::BackendFactory::create(
      svmp::FE::backends::BackendKind::FSILS);
  auto state = factory->createVector(system.dofHandler().getNumDofs());
  state->zero();

  module->applyInitialConditions(system, *state);
  const auto values = state->localSpan();

  const auto phi = system.findFieldByName("phi");
  ASSERT_NE(phi, svmp::FE::INVALID_FIELD_ID);
  const auto* entity_map = system.fieldDofHandler(phi).getEntityDofMap();
  ASSERT_NE(entity_map, nullptr);
  const auto offset = system.fieldDofOffset(phi);

  for (svmp::FE::GlobalIndex vertex = 0;
       vertex < static_cast<svmp::FE::GlobalIndex>(local_mesh.n_vertices());
       ++vertex) {
    const auto vertex_dofs = entity_map->getVertexDofs(vertex);
    ASSERT_EQ(vertex_dofs.size(), 1u);
    const auto dof = offset + vertex_dofs.front();
    ASSERT_GE(dof, 0);
    ASSERT_LT(static_cast<std::size_t>(dof), values.size());
    EXPECT_DOUBLE_EQ(values[static_cast<std::size_t>(dof)],
                     phi_values[static_cast<std::size_t>(vertex)]);
  }
#endif
}

TEST(LevelSetEquationTranslator, InitializesPrescribedHighOrderLevelSetFromMeshPointField)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeRegistryBiquadraticQuadMesh();
  auto& local_mesh = mesh->local_mesh();
  const auto field = svmp::MeshFields::attach_field(
      local_mesh,
      svmp::EntityKind::Vertex,
      "phi",
      svmp::FieldScalarType::Float64,
      1);
  auto* phi_values = svmp::MeshFields::field_data_as<svmp::real_t>(local_mesh, field);
  ASSERT_NE(phi_values, nullptr);
  for (svmp::index_t vertex = 0; vertex < local_mesh.n_vertices(); ++vertex) {
    const auto x = local_mesh.get_vertex_coords(vertex);
    phi_values[vertex] = 10.0 * x[0] + x[1] + 0.125 * static_cast<double>(vertex);
  }

  svmp::Physics::EquationModuleInput input{};
  input.equation_type = "level_set";
  input.mesh_name = "quad9";
  input.mesh = mesh->local_mesh_ptr();
  input.equation_params["Level_set_field_name"] =
      svmp::Physics::ParameterValue{true, "phi"};
  input.equation_params["Level_set_source"] =
      svmp::Physics::ParameterValue{true, "prescribed_data"};
  input.equation_params["Velocity_source"] =
      svmp::Physics::ParameterValue{true, "constant"};
  input.equation_params["Constant_velocity"] =
      svmp::Physics::ParameterValue{true, "0.0 0.0 0.0"};

  svmp::FE::systems::FESystem system(mesh);
  auto module = application::translators::level_set::createModule(input, system);
  ASSERT_TRUE(module);
  ASSERT_NO_THROW(system.setup({}));

  auto factory = svmp::FE::backends::BackendFactory::create(
      svmp::FE::backends::BackendKind::FSILS);
  auto state = factory->createVector(system.dofHandler().getNumDofs());
  state->zero();

  module->applyInitialConditions(system, *state);
  const auto values = state->localSpan();

  const auto phi = system.findFieldByName("phi");
  ASSERT_NE(phi, svmp::FE::INVALID_FIELD_ID);
  const auto* entity_map = system.fieldDofHandler(phi).getEntityDofMap();
  ASSERT_NE(entity_map, nullptr);
  for (svmp::FE::GlobalIndex vertex = 4; vertex < 9; ++vertex) {
    EXPECT_TRUE(entity_map->getVertexDofs(vertex).empty());
  }

  const auto n_field_dofs =
      static_cast<std::size_t>(system.fieldDofHandler(phi).getNumDofs());
  std::vector<svmp::FE::Real> expected_coefficients(n_field_dofs, 0.0);
  std::vector<std::uint8_t> assigned(n_field_dofs, 0u);
  const auto projection =
      system.projectMeshVertexValuesToFieldCoefficients(
          phi,
          std::span<const svmp::FE::Real>(
              phi_values,
              static_cast<std::size_t>(local_mesh.n_vertices())),
          1u,
          std::span<svmp::FE::Real>(expected_coefficients.data(),
                                    expected_coefficients.size()),
          std::span<std::uint8_t>(assigned.data(), assigned.size()),
          "LevelSetEquationTranslator test");
  ASSERT_EQ(projection.unassigned_dofs, 0u);
  ASSERT_EQ(projection.values_written, n_field_dofs);
  const auto offset = system.fieldDofOffset(phi);

  for (std::size_t local_dof = 0; local_dof < n_field_dofs; ++local_dof) {
    ASSERT_NE(assigned[local_dof], 0u);
    const auto dof =
        offset + static_cast<svmp::FE::GlobalIndex>(local_dof);
    ASSERT_GE(dof, 0);
    ASSERT_LT(static_cast<std::size_t>(dof), values.size());
    EXPECT_DOUBLE_EQ(values[static_cast<std::size_t>(dof)],
                     expected_coefficients[local_dof]);
  }

  svmp::FE::systems::SystemStateView state_view{};
  state_view.u = values;
  state_view.u_vector = state.get();
  std::vector<double> sampled(local_mesh.n_vertices(), 0.0);
  EXPECT_TRUE(system.evaluateFieldAtVertices(
      phi,
      state_view,
      static_cast<svmp::FE::GlobalIndex>(local_mesh.n_vertices()),
      sampled));
  for (svmp::index_t vertex = 0; vertex < local_mesh.n_vertices(); ++vertex) {
    EXPECT_DOUBLE_EQ(sampled[static_cast<std::size_t>(vertex)],
                     phi_values[static_cast<std::size_t>(vertex)]);
  }
#endif
}

TEST(LevelSetEquationTranslator,
     InitializesPrescribedHierarchicalLevelSetFromMeshPointField)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeRegistryBiquadraticQuadMesh();
  auto& local_mesh = mesh->local_mesh();
  const auto field = svmp::MeshFields::attach_field(
      local_mesh,
      svmp::EntityKind::Vertex,
      "phi",
      svmp::FieldScalarType::Float64,
      1);
  auto* phi_values =
      svmp::MeshFields::field_data_as<svmp::real_t>(local_mesh, field);
  ASSERT_NE(phi_values, nullptr);
  for (svmp::index_t vertex = 0; vertex < local_mesh.n_vertices(); ++vertex) {
    const auto x = local_mesh.get_vertex_coords(vertex);
    phi_values[vertex] =
        1.5 + 0.25 * x[0] - 0.75 * x[1] + 0.5 * x[0] * x[1];
  }

  svmp::Physics::EquationModuleInput input{};
  input.equation_type = "level_set";
  input.mesh_name = "quad9";
  input.mesh = mesh->local_mesh_ptr();
  input.equation_params["Level_set_field_name"] =
      svmp::Physics::ParameterValue{true, "phi"};
  input.equation_params["Level_set_source"] =
      svmp::Physics::ParameterValue{true, "prescribed_data"};
  input.equation_params["Velocity_source"] =
      svmp::Physics::ParameterValue{true, "constant"};
  input.equation_params["Constant_velocity"] =
      svmp::Physics::ParameterValue{true, "0.0 0.0 0.0"};
  input.equation_params["Basis_type"] =
      svmp::Physics::ParameterValue{true, "hierarchical"};

  svmp::FE::systems::FESystem system(mesh);
  auto module = application::translators::level_set::createModule(input, system);
  ASSERT_TRUE(module);
  ASSERT_NO_THROW(system.setup({}));

  auto factory = svmp::FE::backends::BackendFactory::create(
      svmp::FE::backends::BackendKind::FSILS);
  auto state = factory->createVector(system.dofHandler().getNumDofs());
  state->zero();

  module->applyInitialConditions(system, *state);
  const auto values = state->localSpan();

  const auto phi = system.findFieldByName("phi");
  ASSERT_NE(phi, svmp::FE::INVALID_FIELD_ID);
  ASSERT_EQ(system.fieldRecord(phi).space->element().basis().basis_type(),
            svmp::FE::BasisType::Hierarchical);

  svmp::FE::systems::SystemStateView state_view{};
  state_view.u = values;
  state_view.u_vector = state.get();
  std::vector<double> fast_values(local_mesh.n_vertices(), -1.0);
  EXPECT_FALSE(system.evaluateFieldAtVertices(
      phi,
      state_view,
      static_cast<svmp::FE::GlobalIndex>(local_mesh.n_vertices()),
      fast_values));

  for (svmp::index_t vertex = 0; vertex < local_mesh.n_vertices(); ++vertex) {
    const auto x = local_mesh.get_vertex_coords(vertex);
    const std::array<svmp::FE::Real, 3> point{
        static_cast<svmp::FE::Real>(x[0]),
        static_cast<svmp::FE::Real>(x[1]),
        0.0};
    const auto value = system.evaluateFieldAtPoint(phi, state_view, point);
    ASSERT_TRUE(value.has_value()) << "vertex " << vertex;
    EXPECT_NEAR((*value)[0],
                phi_values[static_cast<std::size_t>(vertex)],
                1.0e-10);
  }
#endif
}

TEST(LevelSetEquationTranslator, TranslatesConstantVelocity)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeRegistryQuadMesh();

  svmp::Physics::EquationModuleInput input{};
  input.equation_type = "level_set";
  input.mesh_name = "quad";
  input.mesh = mesh->local_mesh_ptr();
  input.equation_params["Level_set_field_name"] =
      svmp::Physics::ParameterValue{true, "phi"};
  input.equation_params["Velocity_field_name"] =
      svmp::Physics::ParameterValue{true, "unused_velocity"};
  input.equation_params["Velocity_source"] =
      svmp::Physics::ParameterValue{true, "constant"};
  input.equation_params["Constant_velocity"] =
      svmp::Physics::ParameterValue{true, "1.5 -0.25 0.0"};

  svmp::FE::systems::FESystem system(mesh);
  auto module = application::translators::level_set::createModule(input, system);

  ASSERT_TRUE(module);
  EXPECT_NE(system.findFieldByName("phi"), svmp::FE::INVALID_FIELD_ID);
  EXPECT_EQ(system.findFieldByName("unused_velocity"), svmp::FE::INVALID_FIELD_ID);
  EXPECT_TRUE(system.hasOperator("level_set"));
  EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Constant));
  EXPECT_TRUE(formulationRecordsContain(system, FormExprType::Gradient));
  EXPECT_FALSE(formulationRecordsContain(system, FormExprType::DiscreteField));
#endif
}

TEST(LevelSetEquationTranslator, RoutesCoupledTransportToEquationsOperator)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeRegistryQuadMesh();

  svmp::Physics::EquationModuleInput input{};
  input.equation_type = "level_set";
  input.mesh_name = "quad";
  input.mesh = mesh->local_mesh_ptr();
  input.equation_params["Coupled"] =
      svmp::Physics::ParameterValue{true, "true"};
  input.equation_params["Level_set_field_name"] =
      svmp::Physics::ParameterValue{true, "phi"};
  input.equation_params["Velocity_field_name"] =
      svmp::Physics::ParameterValue{true, "Velocity"};
  input.equation_params["Velocity_source"] =
      svmp::Physics::ParameterValue{true, "coupled_field"};
  input.equation_params["Auto_register_velocity_field"] =
      svmp::Physics::ParameterValue{true, "true"};

  svmp::FE::systems::FESystem system(mesh);
  auto module = application::translators::level_set::createModule(input, system);

  ASSERT_TRUE(module);
  EXPECT_TRUE(system.hasOperator("equations"));
  EXPECT_FALSE(system.hasOperator("level_set"));
#endif
}

TEST(LevelSetEquationTranslator,
     CoupledFreeSurfaceContactAngleUsesTranslatedEquationsOperator)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  constexpr int interface_marker = 242;
  constexpr int wall_marker = 17;
  svmp::Physics::formulations::navier_stokes::
      forceLink_NavierStokesRegister();
  auto mesh = makeRegistryQuadMesh();

  svmp::Physics::EquationModuleInput level_set_input{};
  level_set_input.equation_type = "level_set";
  level_set_input.mesh_name = "quad";
  level_set_input.mesh = mesh->local_mesh_ptr();
  level_set_input.equation_params["Coupled"] =
      svmp::Physics::ParameterValue{true, "true"};
  level_set_input.equation_params["Level_set_field_name"] =
      svmp::Physics::ParameterValue{true, "phi"};
  level_set_input.equation_params["Velocity_field_name"] =
      svmp::Physics::ParameterValue{true, "Velocity"};
  level_set_input.equation_params["Velocity_source"] =
      svmp::Physics::ParameterValue{true, "coupled_field"};
  level_set_input.equation_params["Auto_register_velocity_field"] =
      svmp::Physics::ParameterValue{true, "true"};

  svmp::FE::systems::FESystem system(mesh);
  auto level_set_module =
      application::translators::level_set::createModule(level_set_input, system);
  ASSERT_TRUE(level_set_module);
  const auto phi = system.findFieldByName("phi");
  ASSERT_NE(phi, svmp::FE::INVALID_FIELD_ID);

  svmp::Physics::EquationModuleInput fluid_input{};
  fluid_input.equation_type = "fluid";
  fluid_input.mesh_name = "quad";
  fluid_input.mesh = mesh->local_mesh_ptr();
  fluid_input.equation_params["Operator_tag"] =
      svmp::Physics::ParameterValue{true, "equations"};
  fluid_input.default_domain.params["Density"] =
      svmp::Physics::ParameterValue{true, "1.0"};
  fluid_input.default_domain.params["Viscosity.model"] =
      svmp::Physics::ParameterValue{true, "Constant"};
  fluid_input.default_domain.params["Viscosity.Value"] =
      svmp::Physics::ParameterValue{true, "0.01"};

  svmp::Physics::BoundaryConditionInput free_surface{};
  free_surface.name = "free_surface";
  free_surface.boundary_marker = svmp::INVALID_LABEL;
  free_surface.params["Type"] =
      svmp::Physics::ParameterValue{true, "Free_surface"};
  free_surface.params["Implementation"] =
      svmp::Physics::ParameterValue{true, "UnfittedLevelSet"};
  free_surface.params["Interface_marker"] =
      svmp::Physics::ParameterValue{true, std::to_string(interface_marker)};
  free_surface.params["Level_set_field_name"] =
      svmp::Physics::ParameterValue{true, "phi"};
  free_surface.params["Active_domain"] =
      svmp::Physics::ParameterValue{true, "LevelSetNegative"};
  free_surface.params["Contact_line_model"] =
      svmp::Physics::ParameterValue{true, "PrescribedContactAngle"};
  free_surface.params["Contact_line_wall_marker"] =
      svmp::Physics::ParameterValue{true, std::to_string(wall_marker)};
  free_surface.params["Contact_line_wall_normal"] =
      svmp::Physics::ParameterValue{true, "1.0 0.0 0.0"};
  free_surface.params["Contact_angle_degrees"] =
      svmp::Physics::ParameterValue{true, "60.0"};
  free_surface.params["Contact_angle_penalty"] =
      svmp::Physics::ParameterValue{true, "4.0"};
  fluid_input.boundary_conditions.push_back(std::move(free_surface));

  auto fluid_module =
      svmp::Physics::EquationModuleRegistry::instance().create(
          "fluid", fluid_input, system);
  ASSERT_TRUE(fluid_module);

  svmp::FE::interfaces::GeneratedInterfaceBoundaryIntersectionMarkerKey key{};
  key.source =
      svmp::FE::interfaces::LevelSetInterfaceSource::fromField(phi);
  key.domain_id = "free_surface";
  key.isovalue = 0.0;
  key.interface_marker = interface_marker;
  key.boundary_marker = wall_marker;
  const int contact_marker = svmp::FE::interfaces::
      stableGeneratedInterfaceBoundaryIntersectionMarker(key);

  bool found_contact_residual = false;
  for (const auto& record : system.formulationRecords()) {
    if (!containsInterfaceMarker(record.residual_expr.get(), contact_marker)) {
      continue;
    }
    found_contact_residual = true;
    EXPECT_EQ(record.operator_tag, "equations");
  }
  EXPECT_TRUE(found_contact_residual);
  EXPECT_TRUE(system.hasOperator("equations"));
  EXPECT_FALSE(system.hasOperator("level_set"));
#endif
}

TEST(LevelSetEquationTranslator,
     CoupledDynamicContactAngleRoutesContactAndSharpWallGeometry)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  constexpr int interface_marker = 243;
  constexpr int wall_marker = 18;
  svmp::Physics::formulations::navier_stokes::
      forceLink_NavierStokesRegister();
  auto mesh = makeRegistryQuadMesh();
  // The production DynamicContactAngle validator audits the complete wall
  // marker, not merely the BC declaration.  Give this registry fixture the
  // right-wall topology whose physical outward normal is +x.
  auto& local_mesh = mesh->local_mesh();
  local_mesh.register_label("dynamic_contact_wall", wall_marker);
  for (svmp::index_t face = 0;
       face < static_cast<svmp::index_t>(local_mesh.n_faces()); ++face) {
    const auto vertices = local_mesh.face_vertices(face);
    if (vertices.size() == 2u &&
        std::all_of(vertices.begin(), vertices.end(), [&](svmp::index_t vertex) {
          return local_mesh.X_ref().at(
                     static_cast<std::size_t>(2 * vertex)) == 1.0;
        })) {
      local_mesh.set_boundary_label(face, wall_marker);
    }
  }

  svmp::Physics::EquationModuleInput level_set_input{};
  level_set_input.equation_type = "level_set";
  level_set_input.mesh_name = "quad";
  level_set_input.mesh = mesh->local_mesh_ptr();
  level_set_input.equation_params["Coupled"] =
      svmp::Physics::ParameterValue{true, "true"};
  level_set_input.equation_params["Level_set_field_name"] =
      svmp::Physics::ParameterValue{true, "phi_dynamic"};
  level_set_input.equation_params["Velocity_field_name"] =
      svmp::Physics::ParameterValue{true, "Velocity"};
  level_set_input.equation_params["Velocity_source"] =
      svmp::Physics::ParameterValue{true, "coupled_field"};
  level_set_input.equation_params["Auto_register_velocity_field"] =
      svmp::Physics::ParameterValue{true, "true"};

  svmp::FE::systems::FESystem system(mesh);
  auto level_set_module =
      application::translators::level_set::createModule(level_set_input, system);
  ASSERT_TRUE(level_set_module);
  const auto phi = system.findFieldByName("phi_dynamic");
  ASSERT_NE(phi, svmp::FE::INVALID_FIELD_ID);

  svmp::Physics::EquationModuleInput fluid_input{};
  fluid_input.equation_type = "fluid";
  fluid_input.mesh_name = "quad";
  fluid_input.mesh = mesh->local_mesh_ptr();
  fluid_input.equation_params["Operator_tag"] =
      svmp::Physics::ParameterValue{true, "equations"};
  fluid_input.default_domain.params["Density"] =
      svmp::Physics::ParameterValue{true, "1.0"};
  fluid_input.default_domain.params["Viscosity.model"] =
      svmp::Physics::ParameterValue{true, "Constant"};
  fluid_input.default_domain.params["Viscosity.Value"] =
      svmp::Physics::ParameterValue{true, "0.01"};

  svmp::Physics::BoundaryConditionInput free_surface{};
  free_surface.name = "free_surface";
  free_surface.boundary_marker = svmp::INVALID_LABEL;
  free_surface.params["Type"] =
      svmp::Physics::ParameterValue{true, "Free_surface"};
  free_surface.params["Implementation"] =
      svmp::Physics::ParameterValue{true, "UnfittedLevelSet"};
  free_surface.params["Interface_marker"] =
      svmp::Physics::ParameterValue{true, std::to_string(interface_marker)};
  free_surface.params["Level_set_field_name"] =
      svmp::Physics::ParameterValue{true, "phi_dynamic"};
  free_surface.params["Active_domain"] =
      svmp::Physics::ParameterValue{true, "LevelSetNegative"};
  free_surface.params["Active_domain_method"] =
      svmp::Physics::ParameterValue{true, "CutVolume"};
  free_surface.params["Generated_interface_geometry"] =
      svmp::Physics::ParameterValue{true, "LinearCorner"};
  free_surface.params["Small_cut_aggregation"] =
      svmp::Physics::ParameterValue{true, "false"};
  free_surface.params["Surface_tension"] =
      svmp::Physics::ParameterValue{true, "0.8"};
  free_surface.params["Curvature"] =
      svmp::Physics::ParameterValue{true, "0.0"};
  free_surface.params["Use_level_set_curvature"] =
      svmp::Physics::ParameterValue{true, "false"};
  free_surface.params["Contact_line_model"] =
      svmp::Physics::ParameterValue{true, "DynamicContactAngle"};
  free_surface.params["Contact_line_wall_marker"] =
      svmp::Physics::ParameterValue{true, std::to_string(wall_marker)};
  free_surface.params["Contact_line_wall_normal"] =
      svmp::Physics::ParameterValue{true, "1.0 0.0 0.0"};
  free_surface.params["Contact_angle_degrees"] =
      svmp::Physics::ParameterValue{true, "60.0"};
  free_surface.params["Contact_line_mobility"] =
      svmp::Physics::ParameterValue{true, "0.5"};
  free_surface.params["Wall_slip_model"] =
      svmp::Physics::ParameterValue{true, "Navier"};
  free_surface.params["Wall_slip_length"] =
      svmp::Physics::ParameterValue{true, "0.2"};
  fluid_input.boundary_conditions.push_back(std::move(free_surface));

  svmp::Physics::BoundaryConditionInput wall{};
  wall.name = "dynamic_contact_wall";
  wall.boundary_marker = wall_marker;
  wall.params["Type"] =
      svmp::Physics::ParameterValue{true, "Dirichlet"};
  wall.params["Value"] = svmp::Physics::ParameterValue{true, "0.0"};
  wall.params["Effective_direction"] =
      svmp::Physics::ParameterValue{true, "1 0"};
  fluid_input.boundary_conditions.push_back(std::move(wall));

  auto fluid_module =
      svmp::Physics::EquationModuleRegistry::instance().create(
          "fluid", fluid_input, system);
  ASSERT_TRUE(fluid_module);
  const auto velocity = system.findFieldByName("Velocity");
  ASSERT_NE(velocity, svmp::FE::INVALID_FIELD_ID);

  svmp::FE::interfaces::GeneratedInterfaceBoundaryIntersectionMarkerKey key{};
  key.source = svmp::FE::interfaces::LevelSetInterfaceSource::fromField(phi);
  key.domain_id = "free_surface";
  key.isovalue = 0.0;
  key.interface_marker = interface_marker;
  key.boundary_marker = wall_marker;
  const int contact_marker = svmp::FE::interfaces::
      stableGeneratedInterfaceBoundaryIntersectionMarker(key);
  svmp::FE::interfaces::GeneratedActiveBoundaryMarkerKey active_key{};
  active_key.source =
      svmp::FE::interfaces::LevelSetInterfaceSource::fromField(phi);
  active_key.domain_id = "free_surface";
  active_key.interface_marker = interface_marker;
  active_key.boundary_marker = wall_marker;
  active_key.side = svmp::FE::geometry::CutIntegrationSide::Negative;
  const int active_wall_marker =
      svmp::FE::interfaces::stableGeneratedActiveBoundaryMarker(active_key);

  bool found_contact_residual = false;
  bool found_sharp_wall_residual = false;
  bool found_velocity_phi_coupling = false;
  for (const auto& record : system.formulationRecords()) {
    if (record.operator_tag != "equations") {
      continue;
    }
    found_contact_residual = found_contact_residual ||
        containsInterfaceMarker(record.residual_expr.get(), contact_marker);
    found_sharp_wall_residual = found_sharp_wall_residual ||
        containsInterfaceMarker(record.residual_expr.get(), active_wall_marker);
    found_velocity_phi_coupling = found_velocity_phi_coupling ||
        std::find(record.block_couplings.begin(),
                  record.block_couplings.end(),
                  std::pair<svmp::FE::FieldId, svmp::FE::FieldId>{
                      velocity, phi}) != record.block_couplings.end();
  }
  EXPECT_TRUE(found_contact_residual);
  EXPECT_TRUE(found_sharp_wall_residual);
  EXPECT_FALSE(found_velocity_phi_coupling);
  EXPECT_TRUE(system.hasOperator("equations"));
  EXPECT_FALSE(system.hasOperator("level_set"));
#endif
}

TEST(LevelSetEquationTranslator, AutoRegistersProjectedCurvatureField)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeRegistryQuadMesh();

  svmp::Physics::EquationModuleInput input{};
  input.equation_type = "level_set";
  input.mesh_name = "quad";
  input.mesh = mesh->local_mesh_ptr();
  input.equation_params["Level_set_field_name"] =
      svmp::Physics::ParameterValue{true, "phi"};
  input.equation_params["Velocity_source"] =
      svmp::Physics::ParameterValue{true, "constant"};
  input.equation_params["Constant_velocity"] =
      svmp::Physics::ParameterValue{true, "0.0 0.0 0.0"};
  input.equation_params["Projected_curvature_field"] =
      svmp::Physics::ParameterValue{true, "kappa_projected"};

  svmp::FE::systems::FESystem system(mesh);
  auto module = application::translators::level_set::createModule(input, system);

  ASSERT_TRUE(module);
  const auto kappa = system.findFieldByName("kappa_projected");
  ASSERT_NE(kappa, svmp::FE::INVALID_FIELD_ID);
  const auto& rec = system.fieldRecord(kappa);
  EXPECT_EQ(rec.components, 1);
  ASSERT_TRUE(rec.space);
  EXPECT_EQ(rec.space->value_dimension(), 1);
  EXPECT_EQ(rec.source_kind,
            svmp::FE::systems::FieldSourceKind::PrescribedData);
#endif
}

TEST(LevelSetEquationTranslator,
     InvalidWetExtensionDoesNotRegisterProjectedCurvatureOrTransportFields)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeRegistryQuadMesh();

  svmp::Physics::EquationModuleInput input{};
  input.equation_type = "level_set";
  input.mesh_name = "quad";
  input.mesh = mesh->local_mesh_ptr();
  input.equation_params["Level_set_field_name"] =
      svmp::Physics::ParameterValue{true, "phi_pending"};
  input.equation_params["Velocity_field_name"] =
      svmp::Physics::ParameterValue{true, "extension_pending"};
  input.equation_params["Velocity_source"] =
      svmp::Physics::ParameterValue{true, "coupled_field"};
  input.equation_params["Use_wet_extension_advection_velocity"] =
      svmp::Physics::ParameterValue{true, "true"};
  input.equation_params["Projected_curvature_field"] =
      svmp::Physics::ParameterValue{true, "kappa_pending"};

  svmp::FE::systems::FESystem system(mesh);
  EXPECT_THROW(
      (void)application::translators::level_set::createModule(input, system),
      std::runtime_error);
  EXPECT_EQ(system.findFieldByName("phi_pending"),
            svmp::FE::INVALID_FIELD_ID);
  EXPECT_EQ(system.findFieldByName("extension_pending"),
            svmp::FE::INVALID_FIELD_ID);
  EXPECT_EQ(system.findFieldByName("kappa_pending"),
            svmp::FE::INVALID_FIELD_ID);
  EXPECT_FALSE(system.hasOperator("level_set"));
  EXPECT_TRUE(system.formulationRecords().empty());
#endif
}

TEST(LevelSetEquationTranslator, TranslatesJITPolicy)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  const auto make_input = [](const std::shared_ptr<svmp::Mesh>& mesh,
                             std::string module_options) {
    svmp::Physics::EquationModuleInput input{};
    input.equation_type = "level_set";
    input.mesh_name = "quad";
    input.mesh = mesh->local_mesh_ptr();
    input.module_options = std::move(module_options);
    input.equation_params["Level_set_field_name"] =
        svmp::Physics::ParameterValue{true, "phi"};
    input.equation_params["Velocity_field_name"] =
        svmp::Physics::ParameterValue{true, "unused_velocity"};
    input.equation_params["Velocity_source"] =
        svmp::Physics::ParameterValue{true, "constant"};
    input.equation_params["Constant_velocity"] =
        svmp::Physics::ParameterValue{true, "1.0 0.0 0.0"};
    return input;
  };

  {
    auto mesh = makeRegistryQuadMesh();
    auto input = make_input(mesh, "jit = true; jit_specialization = false");

    svmp::FE::systems::FESystem system(mesh);
    if (svmp::FE::forms::jit::llvmJITEnabled()) {
      auto module =
          application::translators::level_set::createModule(input, system);
      ASSERT_TRUE(module);
      const auto* jit = firstJitKernelInOperator(system, "level_set");
      ASSERT_NE(jit, nullptr);
      const auto& options = jit->jitOptions();
      EXPECT_TRUE(options.enable);
      EXPECT_EQ(options.optimization_level, 3);
      EXPECT_FALSE(options.specialization.enable);
      EXPECT_TRUE(options.specialization.specialize_n_qpts);
      EXPECT_TRUE(options.specialization.specialize_dofs);
    } else {
      try {
        (void)application::translators::level_set::createModule(input, system);
        FAIL() << "explicit level-set jit=true unexpectedly fell back";
      } catch (const std::runtime_error& error) {
        const std::string diagnostic = error.what();
        EXPECT_NE(diagnostic.find("jit=true was explicitly requested"),
                  std::string::npos);
        EXPECT_NE(diagnostic.find("FE_ENABLE_LLVM_JIT"),
                  std::string::npos);
      }
    }
  }

  {
    auto mesh = makeRegistryQuadMesh();
    auto input = make_input(mesh, "jit = false; jit_specialization = true");

    svmp::FE::systems::FESystem system(mesh);
    auto module = application::translators::level_set::createModule(input, system);

    ASSERT_TRUE(module);
    EXPECT_EQ(firstJitKernelInOperator(system, "level_set"), nullptr);
  }

  {
    ScopedUnsetEnvironmentVariable oop_jit_env("SVMP_OOP_JIT_ENABLE");
    ScopedUnsetEnvironmentVariable fe_jit_env("SVMP_FE_JIT_ENABLE");
    auto mesh = makeRegistryQuadMesh();
    auto input = make_input(mesh, "");

    svmp::FE::systems::FESystem system(mesh);
    auto module =
        application::translators::level_set::createModule(input, system);

    ASSERT_TRUE(module);
    const auto* jit = firstJitKernelInOperator(system, "level_set");
    EXPECT_EQ(jit != nullptr, svmp::FE::forms::jit::llvmJITEnabled());
  }
#endif
}
