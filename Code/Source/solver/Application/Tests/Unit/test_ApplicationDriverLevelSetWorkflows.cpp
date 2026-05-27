#include <gtest/gtest.h>

// The workflow helpers exercised here currently live in ApplicationDriver.cpp's
// anonymous namespace; include the implementation to test them without
// widening the production API.
#include "../../Core/ApplicationDriver.cpp"

#include "FE/Assembly/AssemblyKernel.h"
#include "FE/Spaces/H1Space.h"
#include "FE/Spaces/ProductSpace.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Fields/MeshFields.h"
#include "Mesh/Mesh.h"
#include "Parameters.h"
#include "tinyxml2.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::shared_ptr<svmp::Mesh> makeWorkflowBiquadraticQuadMesh()
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

std::array<svmp::FE::Real, 3> workflowVertexPoint(const svmp::Mesh& mesh,
                                                  std::size_t vertex)
{
  const auto& coords = mesh.X_ref();
  const int dim = mesh.dim();
  std::array<svmp::FE::Real, 3> point{0.0, 0.0, 0.0};
  for (int d = 0; d < dim; ++d) {
    point[static_cast<std::size_t>(d)] =
        static_cast<svmp::FE::Real>(
            coords[vertex * static_cast<std::size_t>(dim) +
                   static_cast<std::size_t>(d)]);
  }
  return point;
}

svmp::FE::Real workflowPhi(const svmp::Mesh& mesh, std::size_t vertex)
{
  const auto point = workflowVertexPoint(mesh, vertex);
  return point[0] - svmp::FE::Real{0.25};
}

svmp::FE::Real workflowVerticalPhi(const svmp::Mesh& mesh, std::size_t vertex)
{
  const auto point = workflowVertexPoint(mesh, vertex);
  return point[1] - svmp::FE::Real{0.75};
}

std::array<svmp::FE::Real, 2> workflowVelocity(const svmp::Mesh& mesh,
                                               std::size_t vertex)
{
  const auto point = workflowVertexPoint(mesh, vertex);
  const auto x = point[0];
  const auto y = point[1];
  return {svmp::FE::Real{2.0} + svmp::FE::Real{3.0} * x - y +
              svmp::FE::Real{0.25} * x * y,
          svmp::FE::Real{-1.0} + svmp::FE::Real{0.5} * x +
              svmp::FE::Real{2.0} * y};
}

std::vector<svmp::FE::Real> projectWorkflowVertexValues(
    const svmp::FE::systems::FESystem& system,
    svmp::FE::FieldId field,
    std::span<const svmp::FE::Real> vertex_values,
    std::size_t components,
    std::string_view context)
{
  const auto n_dofs =
      static_cast<std::size_t>(system.fieldDofHandler(field).getNumDofs());
  std::vector<svmp::FE::Real> coefficients(n_dofs, 0.0);
  std::vector<std::uint8_t> assigned(n_dofs, 0u);
  const auto projection = system.projectMeshVertexValuesToFieldCoefficients(
      field,
      vertex_values,
      components,
      std::span<svmp::FE::Real>(coefficients.data(), coefficients.size()),
      std::span<std::uint8_t>(assigned.data(), assigned.size()),
      context);
  if (projection.unassigned_dofs != 0u ||
      projection.values_written != n_dofs) {
    throw std::runtime_error(
        std::string(context) + ": incomplete workflow projection");
  }
  return coefficients;
}

void writeWorkflowFieldSlice(
    const svmp::FE::systems::FESystem& system,
    svmp::FE::FieldId field,
    std::span<const svmp::FE::Real> coefficients,
    std::vector<svmp::FE::Real>& solution)
{
  const auto offset = system.fieldDofOffset(field);
  if (offset < 0 ||
      static_cast<std::size_t>(offset) + coefficients.size() >
          solution.size()) {
    throw std::runtime_error("workflow test field slice is outside solution");
  }
  for (std::size_t i = 0; i < coefficients.size(); ++i) {
    solution[static_cast<std::size_t>(offset) + i] = coefficients[i];
  }
}

std::unique_ptr<Parameters> parseWorkflowParametersXml(const char* xml)
{
  tinyxml2::XMLDocument doc;
  const auto status = doc.Parse(xml);
  if (status != tinyxml2::XML_SUCCESS) {
    throw std::runtime_error(doc.ErrorStr());
  }
  auto* root = doc.FirstChildElement(Parameters::FSI_FILE.c_str());
  if (root == nullptr) {
    throw std::runtime_error("missing root solver element");
  }
  auto params = std::make_unique<Parameters>();
  params->set_equation_values(root);
  return params;
}

class WorkflowScopedEnvVar {
public:
  WorkflowScopedEnvVar(const char* key, std::optional<std::string> value)
      : key_(key)
  {
    if (const char* old = std::getenv(key)) {
      original_ = std::string(old);
    }
    set(std::move(value));
  }

  ~WorkflowScopedEnvVar() { set(original_); }

private:
  void set(const std::optional<std::string>& value) const
  {
    if (value.has_value()) {
      ::setenv(key_, value->c_str(), 1);
    } else {
      ::unsetenv(key_);
    }
  }

  const char* key_;
  std::optional<std::string> original_{};
};

class WorkflowNoOpCellKernel final : public svmp::FE::assembly::AssemblyKernel {
public:
  [[nodiscard]] svmp::FE::assembly::RequiredData getRequiredData()
      const override
  {
    return svmp::FE::assembly::RequiredData::None;
  }

  void computeCell(const svmp::FE::assembly::AssemblyContext&,
                   svmp::FE::assembly::KernelOutput&) override
  {
  }

  [[nodiscard]] std::string name() const override
  {
    return "WorkflowNoOpCellKernel";
  }
};

} // namespace

TEST(ApplicationDriverLevelSetWorkflows,
     CutTopologyChangeTraceIdentifiesNonsmoothNewtonEvent)
{
  WorkflowScopedEnvVar trace("SVMP_OOP_SOLVER_TRACE", std::string("1"));

  ActiveCutContextRefreshReport report{};
  report.refreshed = true;
  report.topology_key = 0x2222u;
  report.request_policy_key = 0x3333u;
  report.value_revision = 7u;
  report.cell_count = 2u;
  report.interface_fragments = 1u;
  report.active_volume_regions = 2u;
  report.active_cut_cells = 1u;
  report.active_quadrature_points = 4u;
  report.domain_total_quadrature_point_count = 6u;
  report.backend_volume_quadrature_point_count = 4u;
  report.backend_interface_quadrature_point_count = 2u;

  std::optional<std::uint64_t> previous_topology_key{0x1111u};

  testing::internal::CaptureStdout();
  logCutTopologyChange(
      report,
      svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint::
          LineSearchTrialResidual,
      previous_topology_key,
      "steady");
  const auto output = testing::internal::GetCapturedStdout();

  ASSERT_TRUE(previous_topology_key.has_value());
  EXPECT_EQ(*previous_topology_key, report.topology_key);
  EXPECT_NE(output.find("diagnostic=cut_topology_change_nonsmooth_event"),
            std::string::npos);
  EXPECT_NE(output.find("event_class=nonsmooth_cut_topology_change"),
            std::string::npos);
  EXPECT_NE(output.find("newton_consistency=not_expected"),
            std::string::npos);
  EXPECT_NE(output.find("jacobian_validity=piecewise_smooth_topology_only"),
            std::string::npos);
  EXPECT_NE(output.find("sync_point=line_search_trial"), std::string::npos);
}

TEST(ApplicationDriverLevelSetWorkflows,
     ActiveSupportRefreshEvaluatesHierarchicalLevelSet)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeWorkflowBiquadraticQuadMesh();
  const auto mesh_field = svmp::MeshFields::attach_field(
      mesh->local_mesh(),
      svmp::EntityKind::Vertex,
      "phi",
      svmp::FieldScalarType::Float64,
      1);
  auto* mesh_phi = svmp::MeshFields::field_data_as<svmp::real_t>(
      mesh->local_mesh(), mesh_field);
  ASSERT_NE(mesh_phi, nullptr);
  std::fill(mesh_phi, mesh_phi + mesh->n_vertices(), 99.0);

  auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Quad4,
      /*order=*/2,
      svmp::FE::BasisType::Hierarchical);

  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto phi = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi",
      .space = scalar_space,
      .components = 1});
  ASSERT_NO_THROW(system->setup({}));

  std::vector<svmp::FE::Real> phi_vertex_values(mesh->n_vertices(), 0.0);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    phi_vertex_values[vertex] = workflowPhi(*mesh, vertex);
  }
  const auto phi_coefficients = projectWorkflowVertexValues(
      *system,
      phi,
      std::span<const svmp::FE::Real>(phi_vertex_values.data(),
                                      phi_vertex_values.size()),
      1u,
      "ApplicationDriver active refresh hierarchical phi");

  std::vector<svmp::FE::Real> solution(
      static_cast<std::size_t>(system->dofHandler().getNumDofs()), 0.0);
  writeWorkflowFieldSlice(*system, phi, phi_coefficients, solution);

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);

  application::core::ActiveCutVolumeRequest request{};
  request.level_set_field_name = "phi";
  request.domain_id = "workflow-active-refresh";
  request.active_side = application::core::LevelSetActiveSide::Negative;

  const auto changed = syncActiveLevelSetVertexFieldsFromSolution(
      sim,
      std::vector<application::core::ActiveCutVolumeRequest>{request},
      std::span<const svmp::FE::Real>(solution.data(), solution.size()));
  EXPECT_EQ(changed, 1u);

  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    EXPECT_NEAR(mesh_phi[vertex], phi_vertex_values[vertex], 1.0e-10)
        << "vertex " << vertex;
  }
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     RefreshesMultipleGeneratedCutDomainsIntoOneContext)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeWorkflowBiquadraticQuadMesh();
  const auto phi_a_mesh_field = svmp::MeshFields::attach_field(
      mesh->local_mesh(),
      svmp::EntityKind::Vertex,
      "phi_a",
      svmp::FieldScalarType::Float64,
      1);
  const auto phi_b_mesh_field = svmp::MeshFields::attach_field(
      mesh->local_mesh(),
      svmp::EntityKind::Vertex,
      "phi_b",
      svmp::FieldScalarType::Float64,
      1);
  ASSERT_NE(svmp::MeshFields::field_data_as<svmp::real_t>(
                mesh->local_mesh(), phi_a_mesh_field),
            nullptr);
  ASSERT_NE(svmp::MeshFields::field_data_as<svmp::real_t>(
                mesh->local_mesh(), phi_b_mesh_field),
            nullptr);

  auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Quad4,
      /*order=*/2);

  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto phi_a = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi_a",
      .space = scalar_space,
      .components = 1});
  const auto phi_b = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi_b",
      .space = scalar_space,
      .components = 1});
  ASSERT_NO_THROW(system->setup({}));

  std::vector<svmp::FE::Real> phi_a_vertex_values(mesh->n_vertices(), 0.0);
  std::vector<svmp::FE::Real> phi_b_vertex_values(mesh->n_vertices(), 0.0);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    phi_a_vertex_values[vertex] = workflowPhi(*mesh, vertex);
    phi_b_vertex_values[vertex] = workflowVerticalPhi(*mesh, vertex);
  }
  const auto phi_a_coefficients = projectWorkflowVertexValues(
      *system,
      phi_a,
      std::span<const svmp::FE::Real>(phi_a_vertex_values.data(),
                                      phi_a_vertex_values.size()),
      1u,
      "ApplicationDriver multiple cut-domain phi_a");
  const auto phi_b_coefficients = projectWorkflowVertexValues(
      *system,
      phi_b,
      std::span<const svmp::FE::Real>(phi_b_vertex_values.data(),
                                      phi_b_vertex_values.size()),
      1u,
      "ApplicationDriver multiple cut-domain phi_b");

  std::vector<svmp::FE::Real> solution(
      static_cast<std::size_t>(system->dofHandler().getNumDofs()), 0.0);
  writeWorkflowFieldSlice(*system, phi_a, phi_a_coefficients, solution);
  writeWorkflowFieldSlice(*system, phi_b, phi_b_coefficients, solution);

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);

  auto params = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Add_BC name="left_free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>phi_a</Level_set_field_name>
      <Generated_interface_domain_id>left_interface</Generated_interface_domain_id>
      <Interface_marker>701</Interface_marker>
      <Allow_corner_linearized_cut_geometry>true</Allow_corner_linearized_cut_geometry>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
    </Add_BC>
    <Add_BC name="top_free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>phi_b</Level_set_field_name>
      <Generated_interface_domain_id>top_interface</Generated_interface_domain_id>
      <Interface_marker>702</Interface_marker>
      <Allow_corner_linearized_cut_geometry>true</Allow_corner_linearized_cut_geometry>
      <Active_domain>LevelSetPositive</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");
  const auto requests = application::core::activeCutVolumeRequests(*params);
  ASSERT_EQ(requests.size(), 2u);

  svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
  const auto report = refreshActiveCutIntegrationContextFromSolution(
      sim,
      *params,
      std::span<const svmp::FE::Real>(solution.data(), solution.size()),
      lifecycle,
      "application-driver-multiple-cut-domain-test");
  EXPECT_TRUE(report.refreshed);
  EXPECT_GE(report.interface_fragments, 2u);
  EXPECT_GT(report.active_volume_regions, 0u);

  const auto* context = sim.fe_system->cutIntegrationContext();
  ASSERT_NE(context, nullptr);
  EXPECT_TRUE(context->hasGeneratedInterfaceMarker(701));
  EXPECT_TRUE(context->hasGeneratedInterfaceMarker(702));
  EXPECT_TRUE(context->hasGeneratedVolumeMarker(701));
  EXPECT_TRUE(context->hasGeneratedVolumeMarker(702));
  EXPECT_FALSE(context->interfaceRulesForMarker(701).empty());
  EXPECT_FALSE(context->interfaceRulesForMarker(702).empty());
  EXPECT_FALSE(context
                   ->generatedVolumeRulesForMarkerAndSide(
                       701,
                       svmp::FE::geometry::CutIntegrationSide::Negative)
                   .empty());
  EXPECT_FALSE(context
                   ->generatedVolumeRulesForMarkerAndSide(
                       702,
                       svmp::FE::geometry::CutIntegrationSide::Positive)
                   .empty());
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     VelocityExtensionRetainsInactiveCutVolumeRules)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeWorkflowBiquadraticQuadMesh();
  const auto mesh_field = svmp::MeshFields::attach_field(
      mesh->local_mesh(),
      svmp::EntityKind::Vertex,
      "phi",
      svmp::FieldScalarType::Float64,
      1);
  ASSERT_NE(svmp::MeshFields::field_data_as<svmp::real_t>(
                mesh->local_mesh(), mesh_field),
            nullptr);

  auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Quad4,
      /*order=*/2);

  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto phi = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi",
      .space = scalar_space,
      .components = 1});
  ASSERT_NO_THROW(system->setup({}));

  std::vector<svmp::FE::Real> phi_vertex_values(mesh->n_vertices(), 0.0);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    phi_vertex_values[vertex] = workflowPhi(*mesh, vertex);
  }
  const auto phi_coefficients = projectWorkflowVertexValues(
      *system,
      phi,
      std::span<const svmp::FE::Real>(phi_vertex_values.data(),
                                      phi_vertex_values.size()),
      1u,
      "ApplicationDriver velocity-extension cut-retention phi");

  std::vector<svmp::FE::Real> solution(
      static_cast<std::size_t>(system->dofHandler().getNumDofs()), 0.0);
  writeWorkflowFieldSlice(*system, phi, phi_coefficients, solution);

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);

  auto params = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>phi</Level_set_field_name>
      <Generated_interface_domain_id>extension_interface</Generated_interface_domain_id>
      <Interface_marker>703</Interface_marker>
      <Allow_corner_linearized_cut_geometry>true</Allow_corner_linearized_cut_geometry>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
      <Enable_velocity_extension>true</Enable_velocity_extension>
      <Velocity_extension_diffusivity>1.0</Velocity_extension_diffusivity>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");
  const auto requests = application::core::activeCutVolumeRequests(*params);
  ASSERT_EQ(requests.size(), 1u);
  EXPECT_EQ(requests.front().volume_retention,
            application::core::ActiveCutVolumeRetention::ActiveAndInactive);

  svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
  const auto report = refreshActiveCutIntegrationContextFromSolution(
      sim,
      *params,
      std::span<const svmp::FE::Real>(solution.data(), solution.size()),
      lifecycle,
      "application-driver-velocity-extension-retention-test");
  EXPECT_TRUE(report.refreshed);

  const auto* context = sim.fe_system->cutIntegrationContext();
  ASSERT_NE(context, nullptr);
  EXPECT_TRUE(context->hasGeneratedInterfaceMarker(703));
  EXPECT_TRUE(context->hasGeneratedVolumeMarker(703));
  EXPECT_FALSE(context
                   ->generatedVolumeRulesForMarkerAndSide(
                       703,
                       svmp::FE::geometry::CutIntegrationSide::Negative)
                   .empty());
  EXPECT_FALSE(context
                   ->generatedVolumeRulesForMarkerAndSide(
                       703,
                       svmp::FE::geometry::CutIntegrationSide::Positive)
                   .empty());
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     ActiveOnlyRetentionRejectsInactiveCutVolumeConsumerWithoutRules)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeWorkflowBiquadraticQuadMesh();
  const auto mesh_field = svmp::MeshFields::attach_field(
      mesh->local_mesh(),
      svmp::EntityKind::Vertex,
      "phi",
      svmp::FieldScalarType::Float64,
      1);
  ASSERT_NE(svmp::MeshFields::field_data_as<svmp::real_t>(
                mesh->local_mesh(), mesh_field),
            nullptr);

  auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Quad4,
      /*order=*/2);

  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto phi = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi",
      .space = scalar_space,
      .components = 1});
  system->addCutVolumeKernel(
      "equations",
      704,
      svmp::FE::geometry::CutIntegrationSide::Positive,
      phi,
      std::make_shared<WorkflowNoOpCellKernel>());
  ASSERT_NO_THROW(system->setup({}));
  EXPECT_EQ(system->cutVolumeKernelCount(
                704, svmp::FE::geometry::CutIntegrationSide::Positive),
            1u);

  std::vector<svmp::FE::Real> phi_vertex_values(mesh->n_vertices(), 0.0);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    phi_vertex_values[vertex] = workflowPhi(*mesh, vertex);
  }
  const auto phi_coefficients = projectWorkflowVertexValues(
      *system,
      phi,
      std::span<const svmp::FE::Real>(phi_vertex_values.data(),
                                      phi_vertex_values.size()),
      1u,
      "ApplicationDriver active-only cut-retention audit phi");

  std::vector<svmp::FE::Real> solution(
      static_cast<std::size_t>(system->dofHandler().getNumDofs()), 0.0);
  writeWorkflowFieldSlice(*system, phi, phi_coefficients, solution);

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);

  auto params = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>phi</Level_set_field_name>
      <Generated_interface_domain_id>active_only_interface</Generated_interface_domain_id>
      <Interface_marker>704</Interface_marker>
      <Allow_corner_linearized_cut_geometry>true</Allow_corner_linearized_cut_geometry>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");
  const auto requests = application::core::activeCutVolumeRequests(*params);
  ASSERT_EQ(requests.size(), 1u);
  EXPECT_EQ(requests.front().volume_retention,
            application::core::ActiveCutVolumeRetention::ActiveOnly);

  svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
  try {
    (void)refreshActiveCutIntegrationContextFromSolution(
        sim,
        *params,
        std::span<const svmp::FE::Real>(solution.data(), solution.size()),
        lifecycle,
        "application-driver-active-only-cut-retention-audit-test");
    FAIL() << "Expected inactive-side cut-volume consumer diagnostic";
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("Generated cut-volume consumer has no retained "
                           "quadrature rules"),
              std::string::npos);
    EXPECT_NE(message.find("marker=704"), std::string::npos);
    EXPECT_NE(message.find("logical_side=inactive"), std::string::npos);
    EXPECT_NE(message.find("cut_volume_side=Positive"), std::string::npos);
    EXPECT_NE(message.find("retained_volume_sides=active_only"),
              std::string::npos);
  }
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     VelocityExtensionProjectsHierarchicalTargetCoefficients)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeWorkflowBiquadraticQuadMesh();
  auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Quad4,
      /*order=*/2,
      svmp::FE::BasisType::Hierarchical);
  auto vector_space =
      std::make_shared<svmp::FE::spaces::ProductSpace>(scalar_space, 2);

  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto phi = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi",
      .space = scalar_space,
      .components = 1});
  const auto source_velocity = system->addField(svmp::FE::systems::FieldSpec{
      .name = "Velocity",
      .space = vector_space,
      .components = 2});
  const auto target_velocity = system->addField(
      svmp::FE::systems::FieldSpec{
          .name = "LevelSetAdvectionVelocity",
          .space = vector_space,
          .components = 2,
          .source_kind = svmp::FE::systems::FieldSourceKind::PrescribedData});
  ASSERT_NO_THROW(system->setup({}));

  std::vector<svmp::FE::Real> phi_vertex_values(mesh->n_vertices(), 0.0);
  std::vector<svmp::FE::Real> source_vertex_values(mesh->n_vertices() * 2u, 0.0);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    phi_vertex_values[vertex] = workflowPhi(*mesh, vertex);
    const auto velocity = workflowVelocity(*mesh, vertex);
    source_vertex_values[2u * vertex] = velocity[0];
    source_vertex_values[2u * vertex + 1u] = velocity[1];
  }

  const auto phi_coefficients = projectWorkflowVertexValues(
      *system,
      phi,
      std::span<const svmp::FE::Real>(phi_vertex_values.data(),
                                      phi_vertex_values.size()),
      1u,
      "ApplicationDriver velocity extension hierarchical phi");
  const auto source_coefficients = projectWorkflowVertexValues(
      *system,
      source_velocity,
      std::span<const svmp::FE::Real>(source_vertex_values.data(),
                                      source_vertex_values.size()),
      2u,
      "ApplicationDriver velocity extension hierarchical source velocity");

  std::vector<svmp::FE::Real> solution(
      static_cast<std::size_t>(system->dofHandler().getNumDofs()), 0.0);
  writeWorkflowFieldSlice(*system, phi, phi_coefficients, solution);
  writeWorkflowFieldSlice(*system,
                          source_velocity,
                          source_coefficients,
                          solution);

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);

  svmp::FE::systems::SystemStateView state{};
  state.u = std::span<const svmp::FE::Real>(solution.data(), solution.size());

  LevelSetAdvectionVelocityRequest request{};
  request.level_set_field_name = "phi";
  request.source_velocity_field_name = "Velocity";
  request.target_velocity_field_name = "LevelSetAdvectionVelocity";
  request.extension_method = "nearest_active_vertex";
  request.active_side = application::core::LevelSetActiveSide::Negative;
  request.isovalue = 0.0;

  EXPECT_TRUE(updateLevelSetAdvectionVelocitiesFromState(
      sim,
      state,
      std::vector<LevelSetAdvectionVelocityRequest>{request}));

  const auto prescribed =
      sim.fe_system->prescribedFieldCoefficients(target_velocity);
  ASSERT_FALSE(prescribed.empty());

  std::vector<std::size_t> active_vertices;
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    if (phi_vertex_values[vertex] <= 0.0) {
      active_vertices.push_back(vertex);
    }
  }
  ASSERT_FALSE(active_vertices.empty());

  auto nearest_active_vertex = [&](std::size_t vertex) {
    if (phi_vertex_values[vertex] <= 0.0) {
      return vertex;
    }
    const auto point = workflowVertexPoint(*mesh, vertex);
    std::size_t best = active_vertices.front();
    svmp::FE::Real best_distance2 =
        std::numeric_limits<svmp::FE::Real>::infinity();
    for (const auto candidate : active_vertices) {
      const auto candidate_point = workflowVertexPoint(*mesh, candidate);
      svmp::FE::Real distance2 = 0.0;
      for (std::size_t d = 0; d < 2u; ++d) {
        const auto delta = point[d] - candidate_point[d];
        distance2 += delta * delta;
      }
      if (distance2 < best_distance2) {
        best_distance2 = distance2;
        best = candidate;
      }
    }
    return best;
  };

  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    const auto expected_source = nearest_active_vertex(vertex);
    const auto expected = workflowVelocity(*mesh, expected_source);
    const auto point = workflowVertexPoint(*mesh, vertex);
    const auto value = sim.fe_system->evaluateFieldAtPoint(
        target_velocity,
        svmp::FE::systems::SystemStateView{},
        point);
    ASSERT_TRUE(value.has_value()) << "vertex " << vertex;
    EXPECT_NEAR((*value)[0], expected[0], 1.0e-10) << "vertex " << vertex;
    EXPECT_NEAR((*value)[1], expected[1], 1.0e-10) << "vertex " << vertex;
  }
#endif
}
