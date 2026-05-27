#include <gtest/gtest.h>

#include "Application/Translators/LevelSetEquationTranslator.h"
#include "FE/Backends/Interfaces/BackendFactory.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Forms/FormExpr.h"
#include "FE/Forms/JIT/JITKernelWrapper.h"
#include "FE/Forms/JIT/LLVMJITBuildInfo.h"
#include "FE/Systems/FESystem.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Fields/MeshFields.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"
#include "Physics/Core/EquationModuleInput.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using svmp::FE::forms::FormExpr;
using svmp::FE::forms::FormExprNode;
using svmp::FE::forms::FormExprType;

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
    auto module = application::translators::level_set::createModule(input, system);

    ASSERT_TRUE(module);
    const auto* jit = firstJitKernelInOperator(system, "level_set");
    if (svmp::FE::forms::jit::llvmJITEnabled()) {
      ASSERT_NE(jit, nullptr);
      const auto& options = jit->jitOptions();
      EXPECT_TRUE(options.enable);
      EXPECT_EQ(options.optimization_level, 3);
      EXPECT_FALSE(options.specialization.enable);
      EXPECT_TRUE(options.specialization.specialize_n_qpts);
      EXPECT_TRUE(options.specialization.specialize_dofs);
    } else {
      EXPECT_EQ(jit, nullptr);
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
#endif
}
