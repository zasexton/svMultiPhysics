#include <gtest/gtest.h>

// The workflow helpers exercised here currently live in ApplicationDriver.cpp's
// anonymous namespace; include the implementation to test them without
// widening the production API.
#include "../../Core/ApplicationDriver.cpp"

#include "FE/Assembly/AssemblyKernel.h"
#include "FE/Backends/Interfaces/BackendFactory.h"
#include "FE/Backends/Interfaces/BackendKind.h"
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

std::shared_ptr<svmp::Mesh> makeWorkflowTriangleMesh()
{
  auto base = std::make_shared<svmp::MeshBase>();

  const std::vector<svmp::real_t> x_ref = {
      0.0, 0.0,
      1.0, 0.0,
      0.0, 1.0,
  };
  const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 3};
  const std::vector<svmp::index_t> cell2vertex = {0, 1, 2};

  svmp::CellShape shape{};
  shape.family = svmp::CellFamily::Triangle;
  shape.num_corners = 3;
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

std::shared_ptr<svmp::Mesh> makeDisconnectedWorkflowQuadPairMesh()
{
  auto base = std::make_shared<svmp::MeshBase>();

  // Keep the two components geometrically close while retaining distinct
  // topology.  A nearest-point extension can then accidentally copy data
  // between components, whereas a cell-graph normal-band extension cannot.
  const std::vector<svmp::real_t> x_ref = {
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,
      0.0, 1.05,
      1.0, 1.05,
      1.0, 2.05,
      0.0, 2.05,
  };
  const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 4, 8};
  const std::vector<svmp::index_t> cell2vertex = {
      0, 1, 2, 3,
      4, 5, 6, 7,
  };

  svmp::CellShape shape{};
  shape.family = svmp::CellFamily::Quad;
  shape.num_corners = 4;
  shape.order = 1;
  base->build_from_arrays(
      /*spatial_dim=*/2,
      x_ref,
      cell2vertex_offsets,
      cell2vertex,
      {shape, shape});
  base->finalize();

  return svmp::create_mesh(std::move(base));
}

std::shared_ptr<svmp::Mesh> makeWorkflowThreeQuadStripMesh()
{
  auto base = std::make_shared<svmp::MeshBase>();

  const std::vector<svmp::real_t> x_ref = {
      0.0, 0.0,
      1.0, 0.0,
      2.0, 0.0,
      3.0, 0.0,
      0.0, 1.0,
      1.0, 1.0,
      2.0, 1.0,
      3.0, 1.0,
  };
  const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 4, 8, 12};
  const std::vector<svmp::index_t> cell2vertex = {
      0, 1, 5, 4,
      1, 2, 6, 5,
      2, 3, 7, 6,
  };

  svmp::CellShape shape{};
  shape.family = svmp::CellFamily::Quad;
  shape.num_corners = 4;
  shape.order = 1;
  base->build_from_arrays(
      /*spatial_dim=*/2,
      x_ref,
      cell2vertex_offsets,
      cell2vertex,
      {shape, shape, shape});
  base->finalize();

  return svmp::create_mesh(std::move(base));
}

std::shared_ptr<svmp::Mesh> makeWorkflowFourQuadStripMesh()
{
  auto base = std::make_shared<svmp::MeshBase>();

  const std::vector<svmp::real_t> x_ref = {
      0.0, 0.0,
      1.0, 0.0,
      2.0, 0.0,
      3.0, 0.0,
      4.0, 0.0,
      0.0, 1.0,
      1.0, 1.0,
      2.0, 1.0,
      3.0, 1.0,
      4.0, 1.0,
  };
  const std::vector<svmp::offset_t> cell2vertex_offsets = {
      0, 4, 8, 12, 16};
  const std::vector<svmp::index_t> cell2vertex = {
      0, 1, 6, 5,
      1, 2, 7, 6,
      2, 3, 8, 7,
      3, 4, 9, 8,
  };

  svmp::CellShape shape{};
  shape.family = svmp::CellFamily::Quad;
  shape.num_corners = 4;
  shape.order = 1;
  base->build_from_arrays(
      /*spatial_dim=*/2,
      x_ref,
      cell2vertex_offsets,
      cell2vertex,
      {shape, shape, shape, shape});
  base->finalize();

  return svmp::create_mesh(std::move(base));
}

std::shared_ptr<svmp::Mesh> makeWorkflowStructuredQuadMesh(int subdivisions)
{
  if (subdivisions <= 0) {
    throw std::invalid_argument(
        "structured workflow mesh requires positive subdivisions");
  }

  auto base = std::make_shared<svmp::MeshBase>();
  const auto vertex_extent = static_cast<std::size_t>(subdivisions + 1);
  std::vector<svmp::real_t> x_ref;
  x_ref.reserve(2u * vertex_extent * vertex_extent);
  for (int row = 0; row <= subdivisions; ++row) {
    for (int column = 0; column <= subdivisions; ++column) {
      x_ref.push_back(static_cast<svmp::real_t>(column) / subdivisions);
      x_ref.push_back(static_cast<svmp::real_t>(row) / subdivisions);
    }
  }

  std::vector<svmp::offset_t> cell2vertex_offsets;
  std::vector<svmp::index_t> cell2vertex;
  cell2vertex_offsets.reserve(
      static_cast<std::size_t>(subdivisions * subdivisions) + 1u);
  cell2vertex.reserve(
      4u * static_cast<std::size_t>(subdivisions * subdivisions));
  cell2vertex_offsets.push_back(0);
  for (int row = 0; row < subdivisions; ++row) {
    for (int column = 0; column < subdivisions; ++column) {
      const auto lower_left = static_cast<svmp::index_t>(
          static_cast<std::size_t>(row) * vertex_extent +
          static_cast<std::size_t>(column));
      const auto lower_right = lower_left + 1;
      const auto upper_left =
          lower_left + static_cast<svmp::index_t>(vertex_extent);
      const auto upper_right = upper_left + 1;
      cell2vertex.insert(cell2vertex.end(),
                         {lower_left,
                          lower_right,
                          upper_right,
                          upper_left});
      cell2vertex_offsets.push_back(
          static_cast<svmp::offset_t>(cell2vertex.size()));
    }
  }

  svmp::CellShape shape{};
  shape.family = svmp::CellFamily::Quad;
  shape.num_corners = 4;
  shape.order = 1;
  std::vector<svmp::CellShape> shapes(
      static_cast<std::size_t>(subdivisions * subdivisions), shape);
  base->build_from_arrays(
      /*spatial_dim=*/2,
      x_ref,
      cell2vertex_offsets,
      cell2vertex,
      shapes);
  base->finalize();

  return svmp::create_mesh(std::move(base));
}

std::shared_ptr<svmp::Mesh> makeWorkflowQuadPatch2x2Mesh()
{
  auto base = std::make_shared<svmp::MeshBase>();

  const std::vector<svmp::real_t> x_ref = {
      0.0, 0.0,
      1.0, 0.0,
      2.0, 0.0,
      0.0, 1.0,
      1.0, 1.0,
      2.0, 1.0,
      0.0, 2.0,
      1.0, 2.0,
      2.0, 2.0,
  };
  const std::vector<svmp::offset_t> cell2vertex_offsets = {
      0, 4, 8, 12, 16};
  const std::vector<svmp::index_t> cell2vertex = {
      0, 1, 4, 3,
      1, 2, 5, 4,
      3, 4, 7, 6,
      4, 5, 8, 7,
  };

  svmp::CellShape shape{};
  shape.family = svmp::CellFamily::Quad;
  shape.num_corners = 4;
  shape.order = 1;
  base->build_from_arrays(
      /*spatial_dim=*/2,
      x_ref,
      cell2vertex_offsets,
      cell2vertex,
      {shape, shape, shape, shape});
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

class WorkflowEffectiveConfigurationModule final
    : public svmp::Physics::PhysicsModule {
public:
  WorkflowEffectiveConfigurationModule(std::string component,
                                       std::string json)
      : artifact_{.component = std::move(component),
                  .json = std::move(json)}
  {
  }

  void registerOn(svmp::FE::systems::FESystem&) const override {}

  [[nodiscard]] std::optional<svmp::Physics::EffectiveConfigurationArtifact>
  effectiveConfigurationArtifact() const override
  {
    return artifact_;
  }

private:
  svmp::Physics::EffectiveConfigurationArtifact artifact_{};
};

} // namespace

TEST(ApplicationDriverLevelSetWorkflows,
     WritesOneDeterministicallyOrderedEffectiveConfigurationArtifact)
{
  const auto unique = std::chrono::steady_clock::now()
                          .time_since_epoch()
                          .count();
  const auto output_directory =
      std::filesystem::temp_directory_path() /
      ("svmp-effective-configuration-" + std::to_string(unique));
  std::filesystem::create_directories(output_directory);

  Parameters params;
  params.general_simulation_parameters.save_results_in_folder.set(
      output_directory.string());
  application::core::SimulationComponents sim;
  sim.physics_modules.push_back(
      std::make_unique<WorkflowEffectiveConfigurationModule>(
          "z_component",
          R"({"artifact_schema_version":1,"component":"z_component"})"));
  sim.physics_modules.push_back(
      std::make_unique<WorkflowEffectiveConfigurationModule>(
          "a_component",
          R"({"artifact_schema_version":1,"component":"a_component"})"));

  writeEffectiveConfigurationArtifact(
      sim, params, svmp::MeshComm::world());

  const auto artifact_path =
      output_directory / "effective_configuration.json";
  std::ifstream input(artifact_path);
  ASSERT_TRUE(input.is_open());
  const std::string contents{
      std::istreambuf_iterator<char>{input},
      std::istreambuf_iterator<char>{}};
  EXPECT_EQ(
      contents,
      "{\"artifact_schema_version\":1,\"modules\":["
      "{\"artifact_schema_version\":1,\"component\":\"a_component\"},"
      "{\"artifact_schema_version\":1,\"component\":\"z_component\"}]}\n");
  EXPECT_FALSE(std::filesystem::exists(
      output_directory / "effective_configuration.json.tmp"));

  std::error_code cleanup_error;
  std::filesystem::remove_all(output_directory, cleanup_error);
  EXPECT_FALSE(cleanup_error);
}

TEST(ApplicationDriverLevelSetWorkflows,
     MonolithicNewtonControlsHonorEveryCoupledEquation)
{
  auto params = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="level_set">
    <Coupled>true</Coupled>
    <Min_iterations>2</Min_iterations>
    <Max_iterations>4</Max_iterations>
    <Tolerance>1.0e-4</Tolerance>
  </Add_equation>
  <Add_equation type="fluid">
    <Coupled>true</Coupled>
    <Min_iterations>1</Min_iterations>
    <Max_iterations>12</Max_iterations>
    <Tolerance>2.0e-2</Tolerance>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  svmp::FE::timestepping::NewtonOptions options{};
  applyMonolithicEquationNewtonControls(*params, options);

  EXPECT_EQ(options.min_iterations, 2);
  EXPECT_EQ(options.max_iterations, 12);
  EXPECT_DOUBLE_EQ(options.rel_tolerance, 2.0e-2);
  EXPECT_DOUBLE_EQ(options.abs_tolerance, 1.0e-10);
}

TEST(ApplicationDriverLevelSetWorkflows,
     UncoupledNewtonControlsRetainPrimaryEquationCompatibility)
{
  auto params = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="level_set">
    <Min_iterations>2</Min_iterations>
    <Max_iterations>4</Max_iterations>
    <Tolerance>1.0e-4</Tolerance>
  </Add_equation>
  <Add_equation type="fluid">
    <Min_iterations>1</Min_iterations>
    <Max_iterations>12</Max_iterations>
    <Tolerance>2.0e-2</Tolerance>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  svmp::FE::timestepping::NewtonOptions options{};
  applyMonolithicEquationNewtonControls(*params, options);

  EXPECT_EQ(options.min_iterations, 1);
  EXPECT_EQ(options.max_iterations, 12);
  EXPECT_DOUBLE_EQ(options.rel_tolerance, 2.0e-2);
  EXPECT_DOUBLE_EQ(options.abs_tolerance, 1.0e-10);
}

TEST(ApplicationDriverLevelSetWorkflows,
     CoupledLevelSetAddsNamedFieldAbsoluteAndRelativeResidualCriterion)
{
  auto params = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="level_set">
    <Coupled>true</Coupled>
    <Tolerance>1.0e-4</Tolerance>
    <Level_set_field_name>phi</Level_set_field_name>
    <LS type="GMRES">
      <Absolute_tolerance>2.0e-12</Absolute_tolerance>
    </LS>
  </Add_equation>
  <Add_equation type="fluid">
    <Coupled>true</Coupled>
    <Tolerance>2.0e-2</Tolerance>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  auto mesh = makeWorkflowTriangleMesh();
  auto space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Triangle3, 1);
  svmp::FE::systems::FESystem system(mesh);
  const auto phi = system.addField(svmp::FE::systems::FieldSpec{
      .name = "phi", .space = space, .components = 1});
  ASSERT_NO_THROW(system.setup({}));

  svmp::FE::timestepping::NewtonOptions options{};
  ASSERT_NO_THROW(applyCoupledLevelSetFieldResidualCriteria(
      system, *params, options));

  ASSERT_EQ(options.field_residual_criteria.size(), 1u);
  EXPECT_EQ(options.field_residual_criteria.front().field, phi);
  EXPECT_DOUBLE_EQ(options.field_residual_criteria.front().abs_tolerance,
                   2.0e-12);
  EXPECT_DOUBLE_EQ(options.field_residual_criteria.front().rel_tolerance,
                   1.0e-4);
}

TEST(ApplicationDriverLevelSetWorkflows,
     CoupledLevelSetFieldResidualUsesLinearAbsoluteToleranceDefault)
{
  auto params = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="level_set">
    <Coupled>true</Coupled>
    <Tolerance>1.0e-4</Tolerance>
    <Level_set_field_name>phi</Level_set_field_name>
  </Add_equation>
  <Add_equation type="fluid">
    <Coupled>true</Coupled>
    <Tolerance>2.0e-2</Tolerance>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  auto mesh = makeWorkflowTriangleMesh();
  auto space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Triangle3, 1);
  svmp::FE::systems::FESystem system(mesh);
  (void)system.addField(svmp::FE::systems::FieldSpec{
      .name = "phi", .space = space, .components = 1});
  ASSERT_NO_THROW(system.setup({}));

  svmp::FE::timestepping::NewtonOptions options{};
  ASSERT_NO_THROW(applyCoupledLevelSetFieldResidualCriteria(
      system, *params, options));

  ASSERT_EQ(options.field_residual_criteria.size(), 1u);
  EXPECT_DOUBLE_EQ(options.field_residual_criteria.front().abs_tolerance,
                   1.0e-10);
  EXPECT_DOUBLE_EQ(options.field_residual_criteria.front().rel_tolerance,
                   1.0e-4);
}

TEST(ApplicationDriverLevelSetWorkflows,
     CoupledLevelSetFieldResidualMergesStrictestEffectiveTolerances)
{
  auto params = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="level_set">
    <Coupled>true</Coupled>
    <Tolerance>1.0e-3</Tolerance>
    <Level_set_field_name>phi</Level_set_field_name>
    <LS type="GMRES">
      <Absolute_tolerance>1.0e-8</Absolute_tolerance>
    </LS>
  </Add_equation>
  <Add_equation type="level_set_transport">
    <Coupled>true</Coupled>
    <Tolerance>1.0e-5</Tolerance>
    <Level_set_field_name>phi</Level_set_field_name>
    <LS type="GMRES">
      <Absolute_tolerance>1.0e-9</Absolute_tolerance>
    </LS>
  </Add_equation>
  <Add_equation type="fluid">
    <Coupled>true</Coupled>
    <Tolerance>2.0e-2</Tolerance>
    <LS type="GMRES">
      <Absolute_tolerance>2.0e-12</Absolute_tolerance>
    </LS>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  auto mesh = makeWorkflowTriangleMesh();
  auto space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Triangle3, 1);
  svmp::FE::systems::FESystem system(mesh);
  (void)system.addField(svmp::FE::systems::FieldSpec{
      .name = "phi", .space = space, .components = 1});
  ASSERT_NO_THROW(system.setup({}));

  svmp::FE::timestepping::NewtonOptions options{};
  ASSERT_NO_THROW(applyCoupledLevelSetFieldResidualCriteria(
      system, *params, options));

  ASSERT_EQ(options.field_residual_criteria.size(), 1u);
  EXPECT_DOUBLE_EQ(options.field_residual_criteria.front().abs_tolerance,
                   1.0e-9);
  EXPECT_DOUBLE_EQ(options.field_residual_criteria.front().rel_tolerance,
                   1.0e-5);
}

TEST(ApplicationDriverLevelSetWorkflows,
     UncoupledLevelSetDoesNotAddFieldResidualCriterion)
{
  auto params = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="level_set">
    <Tolerance>1.0e-4</Tolerance>
    <Level_set_field_name>phi</Level_set_field_name>
  </Add_equation>
  <Add_equation type="fluid">
    <Tolerance>2.0e-2</Tolerance>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  auto mesh = makeWorkflowTriangleMesh();
  auto space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Triangle3, 1);
  svmp::FE::systems::FESystem system(mesh);
  (void)system.addField(svmp::FE::systems::FieldSpec{
      .name = "phi", .space = space, .components = 1});
  ASSERT_NO_THROW(system.setup({}));

  svmp::FE::timestepping::NewtonOptions options{};
  ASSERT_NO_THROW(applyCoupledLevelSetFieldResidualCriteria(
      system, *params, options));
  EXPECT_TRUE(options.field_residual_criteria.empty());
}

TEST(ApplicationDriverLevelSetWorkflows,
     TransientLineSearchTrialSynchronizationFollowsResidualContract)
{
  {
    WorkflowScopedEnvVar unset("SVMP_SYNC_LINE_SEARCH_TRIALS", std::nullopt);
    EXPECT_FALSE(synchronizeTransientLineSearchTrials(
        /*residual_defining_state_changes=*/false));
    EXPECT_TRUE(synchronizeTransientLineSearchTrials(
        /*residual_defining_state_changes=*/true));
  }
  {
    WorkflowScopedEnvVar disabled("SVMP_SYNC_LINE_SEARCH_TRIALS",
                                  std::string("0"));
    EXPECT_FALSE(synchronizeTransientLineSearchTrials(
        /*residual_defining_state_changes=*/true));
  }
  {
    WorkflowScopedEnvVar enabled("SVMP_SYNC_LINE_SEARCH_TRIALS",
                                 std::string("1"));
    EXPECT_TRUE(synchronizeTransientLineSearchTrials(
        /*residual_defining_state_changes=*/false));
  }
}

TEST(ApplicationDriverLevelSetWorkflows,
     ActiveCutGeneralizedAlphaPdeRateInitializationDefaultsOnAndAllowsOptOut)
{
  {
    WorkflowScopedEnvVar unset(
        "SVMP_GENERALIZED_ALPHA_PDE_UDOT_INIT", std::nullopt);
    EXPECT_TRUE(generalizedAlphaPdeRateInitializationRequested(
        /*active_cut_domain_present=*/true));
  }
  {
    WorkflowScopedEnvVar disabled(
        "SVMP_GENERALIZED_ALPHA_PDE_UDOT_INIT", std::string("0"));
    EXPECT_FALSE(generalizedAlphaPdeRateInitializationRequested(
        /*active_cut_domain_present=*/true));
  }
  {
    WorkflowScopedEnvVar enabled(
        "SVMP_GENERALIZED_ALPHA_PDE_UDOT_INIT", std::string("1"));
    EXPECT_TRUE(generalizedAlphaPdeRateInitializationRequested(
        /*active_cut_domain_present=*/true));
  }
}

TEST(ApplicationDriverLevelSetWorkflows,
     LinearCornerRefreshReportsRefreshedJacobianCheckGeometry)
{
  application::core::ActiveCutVolumeRequest request{};
  request.geometry_mode =
      svmp::FE::level_set::GeneratedInterfaceGeometryMode::LinearCorner;
  request.geometry_tangent_policy =
      svmp::FE::level_set::GeometryTangentPolicy::RefreshedFrozenQuadrature;

  svmp::FE::timestepping::NewtonOptions options{};
  applyJacobianCheckGeometryProvenance(
      options,
      std::vector<application::core::ActiveCutVolumeRequest>{request},
      /*refresh_generated_geometry_within_solve=*/true,
      /*has_frozen_algebraic_level_set_extension=*/true);

  EXPECT_EQ(options.jacobian_check_geometry_mode,
            svmp::FE::timestepping::JacobianCheckGeometryMode::
                RefreshedGeometry);
  EXPECT_EQ(options.jacobian_check_geometry_tangent_policy,
            "RefreshedFrozenQuadrature");
}

TEST(ApplicationDriverLevelSetWorkflows,
     FrozenExtensionWithoutCutRefreshReportsFixedJacobianCheckGeometry)
{
  application::core::ActiveCutVolumeRequest request{};
  svmp::FE::timestepping::NewtonOptions options{};
  applyJacobianCheckGeometryProvenance(
      options,
      std::vector<application::core::ActiveCutVolumeRequest>{request},
      /*refresh_generated_geometry_within_solve=*/false,
      /*has_frozen_algebraic_level_set_extension=*/true);

  EXPECT_EQ(options.jacobian_check_geometry_mode,
            svmp::FE::timestepping::JacobianCheckGeometryMode::FixedGeometry);
  EXPECT_EQ(options.jacobian_check_geometry_tangent_policy,
            "fixed-topology algebraic wet-extension solve");
}

TEST(ApplicationDriverLevelSetWorkflows,
     OuterFixedPointReportsFrozenInnerJacobianGeometry)
{
  application::core::ActiveCutVolumeRequest request{};
  request.geometry_mode =
      svmp::FE::level_set::GeneratedInterfaceGeometryMode::LinearCorner;
  request.geometry_tangent_policy =
      svmp::FE::level_set::GeometryTangentPolicy::RefreshedFrozenQuadrature;

  svmp::FE::timestepping::NewtonOptions options{};
  applyJacobianCheckGeometryProvenance(
      options,
      std::vector<application::core::ActiveCutVolumeRequest>{request},
      /*refresh_generated_geometry_within_solve=*/false,
      /*has_frozen_algebraic_level_set_extension=*/false,
      /*use_external_state_fixed_point=*/true);

  EXPECT_EQ(options.jacobian_check_geometry_mode,
            svmp::FE::timestepping::JacobianCheckGeometryMode::FixedGeometry);
  EXPECT_EQ(options.jacobian_check_geometry_tangent_policy,
            "outer-fixed-point frozen geometry (RefreshedFrozenQuadrature)");
}

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
     DefaultSmallCutAggregationRetainsInactiveCutVolumeRules)
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
      "ApplicationDriver aggregation cut-retention phi");

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
      "application-driver-aggregation-retention-test");
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
     AcceptedFunctionalUsesAuthoritativeSnapshotAndRecordsGlobalState)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  constexpr int interface_marker = 707;
  constexpr svmp::FE::Real gamma = svmp::FE::Real{0.65};
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
  svmp::FE::interfaces::FreeSurfaceDiscreteFunctionalParameters parameters;
  parameters.liquid_side =
      svmp::FE::geometry::CutIntegrationSide::Negative;
  parameters.surface_tension = gamma;
  system->declareFreeSurfaceDiscreteFunctional(
      svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration{
          .interface_marker = interface_marker,
          .level_set_field = phi,
          .geometry_domain_id = "functional_interface",
          .parameters = parameters,
          .owner_component =
              "ApplicationDriverLevelSetWorkflows.FunctionalFixture",
      });
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
      "ApplicationDriver accepted functional phi");
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
      <Generated_interface_domain_id>functional_interface</Generated_interface_domain_id>
      <Interface_marker>707</Interface_marker>
      <Allow_corner_linearized_cut_geometry>true</Allow_corner_linearized_cut_geometry>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");
  svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
  const auto report = refreshActiveCutIntegrationContextFromSolution(
      sim,
      *params,
      std::span<const svmp::FE::Real>(solution.data(), solution.size()),
      lifecycle,
      "application-driver-accepted-functional-test");
  ASSERT_TRUE(report.refreshed);
  ASSERT_NO_THROW(recordAcceptedFreeSurfaceDiscreteFunctionals(
      sim,
      /*accepted_step=*/3u,
      svmp::FE::Real{0.15},
      svmp::FE::Real{0.05},
      /*state_revision=*/23u));

  const auto history = sim.fe_system->freeSurfaceDiscreteFunctionalHistory();
  ASSERT_EQ(history.size(), 1u);
  const auto& record = history.front();
  EXPECT_EQ(record.accepted_step, 3u);
  EXPECT_EQ(record.state_revision, 23u);
  EXPECT_TRUE(record.geometry_revision.complete());
  EXPECT_EQ(record.geometry_revision.interface_marker, interface_marker);
  EXPECT_EQ(record.geometry_revision.domain_id, "functional_interface");
  EXPECT_GT(record.state.owned_liquid_volume, 0.0);
  EXPECT_GT(record.state.owned_liquid_gas_area, 0.0);
  EXPECT_NEAR(record.state.liquid_gas_surface_energy,
              gamma * record.state.owned_liquid_gas_area,
              1.0e-13);
  EXPECT_NEAR(record.state.total_potential,
              record.state.liquid_gas_surface_energy,
              1.0e-13);
  ASSERT_NO_THROW(recordAcceptedFreeSurfaceDiscreteFunctionals(
      sim,
      /*accepted_step=*/3u,
      svmp::FE::Real{0.15},
      svmp::FE::Real{0.05},
      /*state_revision=*/23u));
  EXPECT_EQ(sim.fe_system->freeSurfaceDiscreteFunctionalHistory().size(), 1u);
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     ConsumedContactMarkerAcceptsCurvedHighOrderFragments)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  constexpr int wall_marker = 17;
  constexpr int interface_marker = 705;
  auto mesh = makeWorkflowBiquadraticQuadMesh();
  for (const auto face : mesh->local_mesh().boundary_faces()) {
    mesh->local_mesh().set_boundary_label(face, wall_marker);
  }
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
  auto velocity_space =
      std::make_shared<svmp::FE::spaces::ProductSpace>(scalar_space, 2);
  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto phi = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi",
      .space = scalar_space,
      .components = 1});
  const auto velocity = system->addField(svmp::FE::systems::FieldSpec{
      .name = "velocity",
      .space = velocity_space,
      .components = 2});

  svmp::FE::interfaces::FreeSurfaceDiscreteFunctionalParameters
      functional_parameters;
  functional_parameters.liquid_side =
      svmp::FE::geometry::CutIntegrationSide::Negative;
  functional_parameters.surface_tension = svmp::FE::Real{0.8};
  functional_parameters.young_wall_coefficients.push_back(
      svmp::FE::interfaces::FreeSurfaceYoungWallCoefficient{
          .boundary_marker = wall_marker,
          .equilibrium_contact_angle_radians =
              svmp::FE::Real{1.04719755119659774615421446109316763},
      });
  functional_parameters.dynamic_contact_coefficients.push_back(
      svmp::FE::interfaces::FreeSurfaceDynamicContactCoefficient{
          .boundary_marker = wall_marker,
          .equilibrium_contact_angle_radians =
              svmp::FE::Real{1.04719755119659774615421446109316763},
          .mobility = svmp::FE::Real{0.5},
          .slip_length = svmp::FE::Real{0.2},
          .dynamic_viscosity = svmp::FE::Real{0.4},
      });
  system->declareFreeSurfaceDiscreteFunctional(
      svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration{
          .interface_marker = interface_marker,
          .level_set_field = phi,
          .velocity_field = velocity,
          .geometry_domain_id = "degenerate_contact",
          .parameters = functional_parameters,
          .owner_component =
              "ApplicationDriverLevelSetWorkflows.ContactStageFixture",
      });

  svmp::FE::interfaces::GeneratedInterfaceBoundaryIntersectionMarkerKey key{};
  key.source =
      svmp::FE::interfaces::LevelSetInterfaceSource::fromField(phi);
  key.domain_id = "degenerate_contact";
  key.isovalue = 0.0;
  key.interface_marker = interface_marker;
  key.boundary_marker = wall_marker;
  const int contact_marker = svmp::FE::interfaces::
      stableGeneratedInterfaceBoundaryIntersectionMarker(key);
  system->registerGeneratedEmbeddedInterfaceMarker(contact_marker);
  ASSERT_TRUE(
      system->isGeneratedEmbeddedInterfaceMarkerRegistered(contact_marker));
  ASSERT_NO_THROW(system->setup({}));

  std::vector<svmp::FE::Real> phi_vertex_values(mesh->n_vertices(), 0.0);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    const auto point = workflowVertexPoint(*mesh, vertex);
    // The interface crosses curved parent geometry.  The generated contact
    // trace must remain available to a registered consumer without falling
    // back to a skipped fragment.
    phi_vertex_values[vertex] =
        svmp::FE::Real{2.0} * (point[0] - svmp::FE::Real{0.45});
  }
  const auto coefficients = projectWorkflowVertexValues(
      *system,
      phi,
      std::span<const svmp::FE::Real>(phi_vertex_values.data(),
                                      phi_vertex_values.size()),
      1u,
      "ApplicationDriver degenerate-contact phi");
  std::vector<svmp::FE::Real> solution(
      static_cast<std::size_t>(system->dofHandler().getNumDofs()), 0.0);
  writeWorkflowFieldSlice(*system, phi, coefficients, solution);
  const auto velocity_dofs = static_cast<std::size_t>(
      system->fieldDofHandler(velocity).getNumDofs());
  std::vector<svmp::FE::Real> velocity_coefficients(
      velocity_dofs, svmp::FE::Real{0.2});
  writeWorkflowFieldSlice(
      *system, velocity, velocity_coefficients, solution);

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
      <Generated_interface_domain_id>degenerate_contact</Generated_interface_domain_id>
      <Interface_marker>705</Interface_marker>
      <Allow_corner_linearized_cut_geometry>true</Allow_corner_linearized_cut_geometry>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
  ASSERT_NO_THROW((void)refreshActiveCutIntegrationContextFromSolution(
      sim,
      *params,
      std::span<const svmp::FE::Real>(solution.data(), solution.size()),
      lifecycle,
      "application-driver-curved-contact-test"));

  const auto* context = sim.fe_system->cutIntegrationContext();
  ASSERT_NE(context, nullptr);
  ASSERT_TRUE(context->hasFreeSurfaceGeometrySnapshotForMarker(contact_marker));
  ASSERT_EQ(context->freeSurfaceGeometrySnapshots().size(), 1u);
  const auto snapshot = context->freeSurfaceGeometrySnapshots().front();
  ASSERT_NE(snapshot, nullptr);
  ASSERT_EQ(snapshot->contactDomains().size(), 1u);
  EXPECT_EQ(snapshot->contactDomains().front().marker(), contact_marker);
  const auto summary = snapshot->contactDomains().front().summary();
  EXPECT_EQ(summary.fragment_count, 2u);
  EXPECT_EQ(summary.active_fragment_count, 2u);
  EXPECT_EQ(summary.skipped_fragment_count, 0u);
  EXPECT_EQ(snapshot->ledger().orphan_contact_fragment_count, 0u);
  EXPECT_EQ(snapshot->ledger().stale_revision_count, 0u);

  auto endpoint_solution = solution;
  auto previous_solution = solution;
  const auto phi_offset =
      static_cast<std::size_t>(sim.fe_system->fieldDofOffset(phi));
  const auto phi_dof_count = static_cast<std::size_t>(
      sim.fe_system->fieldDofHandler(phi).getNumDofs());
  ASSERT_LE(phi_offset + phi_dof_count, solution.size());
  for (std::size_t i = 0; i < phi_dof_count; ++i) {
    endpoint_solution[phi_offset + i] *= svmp::FE::Real{1.1};
    previous_solution[phi_offset + i] *= svmp::FE::Real{0.9};
  }
  auto factory = svmp::FE::backends::BackendFactory::create(
      svmp::FE::backends::BackendKind::FSILS);
  ASSERT_NE(factory, nullptr);
  auto time_history = svmp::FE::timestepping::TimeHistory::allocate(
      *factory,
      sim.fe_system->dofHandler().getNumDofs(),
      /*history_depth=*/2,
      /*allocate_second_order_state=*/false);
  time_history.setTime(0.10);
  time_history.setDt(0.05);
  time_history.setPrevDt(0.05);
  time_history.setStepIndex(2);
  scatterFeOrderedSolution(time_history.u(), endpoint_solution);
  scatterFeOrderedSolution(time_history.uPrev(), previous_solution);
  scatterFeOrderedSolution(time_history.uPrev2(), previous_solution);

  const auto contact_stages = evaluateAcceptedFreeSurfaceContactStages(
      sim,
      svmp::FE::Real{0.075},
      svmp::FE::Real{0.5},
      time_history.uPrev().valueRevision(),
      time_history.u().valueRevision(),
      std::span<const svmp::FE::Real>(solution.data(), solution.size()));
  ASSERT_EQ(contact_stages.size(), 1u);
  ASSERT_EQ(contact_stages.front().state.walls.size(), 1u);
  EXPECT_GT(contact_stages.front().state.owned_contact_measure, 0.0);
  EXPECT_GT(contact_stages.front().state.line_friction_dissipation, 0.0);
  const auto contact_stage_constraints =
      captureAcceptedContactStageWallConstraints(sim, contact_stages);
  time_history.updateGhosts();
  EXPECT_NE(contact_stages.front().endpoint_state_revision,
            time_history.u().valueRevision());
  ASSERT_NO_THROW((void)refreshActiveCutIntegrationContextFromSolution(
      sim,
      *params,
      std::span<const svmp::FE::Real>(endpoint_solution.data(),
                                      endpoint_solution.size()),
      lifecycle,
      "application-driver-contact-endpoint-finalization-test"));
  const auto endpoint_snapshot_revision =
      sim.fe_system->cutIntegrationContext()
          ->freeSurfaceGeometrySnapshotRevisionForMarker(interface_marker);
  EXPECT_NE(endpoint_snapshot_revision,
            snapshot->revision().snapshot_revision_key);

  LevelSetMaintenanceRequest maintenance_request{};
  maintenance_request.level_set_field_name = "phi";
  maintenance_request.reinitialization.enabled = true;
  maintenance_request.reinitialization.cadence_steps = 1;
  maintenance_request.reinitialization.max_iterations = 100;
  maintenance_request.reinitialization.signed_distance_tolerance = 1.0e-10;
  std::vector<LevelSetMaintenanceRequest> maintenance_requests{
      maintenance_request};
  auto missing_stage_requests = maintenance_requests;
  EXPECT_THROW(
      (void)applyLevelSetMaintenance(
          sim, time_history, missing_stage_requests),
      std::runtime_error);
  EXPECT_EQ(gatherFeOrderedSolution(time_history.u()), endpoint_solution);
  EXPECT_EQ(gatherFeOrderedSolution(time_history.uPrev()), previous_solution);
  testing::internal::CaptureStdout();
  const bool maintenance_changed = applyLevelSetMaintenance(
      sim,
      time_history,
      maintenance_requests,
      contact_stages,
      contact_stage_constraints,
      std::span<const svmp::FE::Real>(solution.data(), solution.size()));
  const auto maintenance_output = testing::internal::GetCapturedStdout();
  ASSERT_TRUE(maintenance_changed);
  EXPECT_NE(maintenance_output.find(
                "wall_contact_model=accepted_dynamic_stage"),
            std::string::npos);
  EXPECT_NE(maintenance_output.find("wall_contact_constraints=1"),
            std::string::npos);
  EXPECT_NE(maintenance_output.find("max_contact_line_displacement=0"),
            std::string::npos);
  EXPECT_NE(maintenance_output.find("max_contact_angle_change_radians=0"),
            std::string::npos);

  const auto endpoint_after = gatherFeOrderedSolution(time_history.u());
  const auto previous_after = gatherFeOrderedSolution(time_history.uPrev());
  ASSERT_EQ(endpoint_after.size(), endpoint_solution.size());
  for (std::size_t i = 0; i < phi_dof_count; ++i) {
    const auto index = phi_offset + i;
    EXPECT_NEAR(endpoint_after[index] - endpoint_solution[index],
                previous_after[index] - previous_solution[index],
                1.0e-12);
    const auto accepted_stage_after =
        svmp::FE::Real{0.5} *
        (endpoint_after[index] + previous_after[index]);
    EXPECT_NEAR(accepted_stage_after,
                svmp::FE::Real{0.5} * solution[index],
                1.0e-10);
  }
  ASSERT_NO_THROW((void)refreshActiveCutIntegrationContextFromSolution(
      sim,
      *params,
      std::span<const svmp::FE::Real>(endpoint_after.data(),
                                      endpoint_after.size()),
      lifecycle,
      "application-driver-wall-aware-maintenance-test"));
  const auto* maintained_context = sim.fe_system->cutIntegrationContext();
  ASSERT_NE(maintained_context, nullptr);
  const auto maintained_snapshot_revision =
      maintained_context->freeSurfaceGeometrySnapshotRevisionForMarker(
          interface_marker);
  EXPECT_NE(maintained_snapshot_revision,
            snapshot->revision().snapshot_revision_key);
  EXPECT_NE(maintained_snapshot_revision, endpoint_snapshot_revision);

  ASSERT_NO_THROW(recordAcceptedFreeSurfaceDiscreteFunctionals(
      sim,
      /*accepted_step=*/2u,
      svmp::FE::Real{0.10},
      svmp::FE::Real{0.05},
      time_history.u().valueRevision(),
      contact_stages));
  const auto history =
      sim.fe_system->freeSurfaceDiscreteFunctionalHistory();
  ASSERT_EQ(history.size(), 1u);
  ASSERT_TRUE(history.front().contact_stage.has_value());
  EXPECT_DOUBLE_EQ(history.front().contact_stage->stage_time,
                   svmp::FE::Real{0.075});
  EXPECT_DOUBLE_EQ(history.front().contact_stage->stage_alpha_f,
                   svmp::FE::Real{0.5});
  EXPECT_EQ(history.front().contact_stage->geometry_revision
                .snapshot_revision_key,
            snapshot->revision().snapshot_revision_key);
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     PrescribedWallSnapshotDrivesEndpointReinitialization)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  constexpr int wall_marker = 29;
  constexpr int interface_marker = 706;
  auto mesh = makeWorkflowQuadPatch2x2Mesh();
  for (const auto face : mesh->local_mesh().boundary_faces()) {
    mesh->local_mesh().set_boundary_label(face, wall_marker);
  }
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
      /*order=*/1);
  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto phi = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi",
      .space = scalar_space,
      .components = 1});
  svmp::FE::interfaces::FreeSurfaceDiscreteFunctionalParameters parameters;
  parameters.liquid_side =
      svmp::FE::geometry::CutIntegrationSide::Negative;
  parameters.surface_tension = svmp::FE::Real{1.0};
  parameters.young_wall_coefficients.push_back(
      svmp::FE::interfaces::FreeSurfaceYoungWallCoefficient{
          .boundary_marker = wall_marker,
          .equilibrium_contact_angle_radians =
              svmp::FE::Real{1.57079632679489661923132169163975144},
      });
  system->declareFreeSurfaceDiscreteFunctional(
      svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration{
          .interface_marker = interface_marker,
          .level_set_field = phi,
          .geometry_domain_id = "prescribed_wall_maintenance",
          .parameters = parameters,
          .owner_component =
              "ApplicationDriverLevelSetWorkflows.PrescribedWallFixture",
      });
  svmp::FE::interfaces::GeneratedInterfaceBoundaryIntersectionMarkerKey key{};
  key.source =
      svmp::FE::interfaces::LevelSetInterfaceSource::fromField(phi);
  key.domain_id = "prescribed_wall_maintenance";
  key.isovalue = 0.0;
  key.interface_marker = interface_marker;
  key.boundary_marker = wall_marker;
  system->registerGeneratedEmbeddedInterfaceMarker(
      svmp::FE::interfaces::
          stableGeneratedInterfaceBoundaryIntersectionMarker(key));
  ASSERT_NO_THROW(system->setup({}));

  std::vector<svmp::FE::Real> vertex_values(mesh->n_vertices(), 0.0);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    vertex_values[vertex] =
        svmp::FE::Real{2.0} *
        (workflowVertexPoint(*mesh, vertex)[0] - svmp::FE::Real{0.8});
  }
  const auto coefficients = projectWorkflowVertexValues(
      *system,
      phi,
      vertex_values,
      /*components=*/1u,
      "ApplicationDriver prescribed-wall maintenance phi");
  std::vector<svmp::FE::Real> solution(
      static_cast<std::size_t>(system->dofHandler().getNumDofs()), 0.0);
  writeWorkflowFieldSlice(*system, phi, coefficients, solution);

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
      <Generated_interface_domain_id>prescribed_wall_maintenance</Generated_interface_domain_id>
      <Interface_marker>706</Interface_marker>
      <Allow_corner_linearized_cut_geometry>true</Allow_corner_linearized_cut_geometry>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");
  svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
  ASSERT_NO_THROW((void)refreshActiveCutIntegrationContextFromSolution(
      sim,
      *params,
      std::span<const svmp::FE::Real>(solution.data(), solution.size()),
      lifecycle,
      "application-driver-prescribed-wall-maintenance-test"));

  auto factory = svmp::FE::backends::BackendFactory::create(
      svmp::FE::backends::BackendKind::FSILS);
  ASSERT_NE(factory, nullptr);
  auto history = svmp::FE::timestepping::TimeHistory::allocate(
      *factory, sim.fe_system->dofHandler().getNumDofs());
  history.setTime(0.1);
  history.setDt(0.05);
  history.setPrevDt(0.05);
  history.setStepIndex(1);
  scatterFeOrderedSolution(history.u(), solution);
  scatterFeOrderedSolution(history.uPrev(), solution);
  scatterFeOrderedSolution(history.uPrev2(), solution);

  LevelSetMaintenanceRequest request{};
  request.level_set_field_name = "phi";
  request.reinitialization.enabled = true;
  request.reinitialization.cadence_steps = 1;
  request.reinitialization.max_iterations = 100;
  request.reinitialization.signed_distance_tolerance = 1.0e-10;
  std::vector<LevelSetMaintenanceRequest> requests{request};
  testing::internal::CaptureStdout();
  const bool changed = applyLevelSetMaintenance(sim, history, requests);
  const auto output = testing::internal::GetCapturedStdout();
  ASSERT_TRUE(changed);
  EXPECT_NE(output.find("wall_contact_model=prescribed_angle"),
            std::string::npos);
  EXPECT_NE(output.find("prescribed_contact_rules=2"),
            std::string::npos);
  EXPECT_NE(output.find("dynamic_contact_rules=0"), std::string::npos);
  EXPECT_NE(output.find("max_contact_line_displacement=0"),
            std::string::npos);
  EXPECT_NE(output.find("max_contact_angle_change_radians=0"),
            std::string::npos);

  const auto repaired = gatherFeOrderedSolution(history.u());
  const auto field_offset =
      static_cast<std::size_t>(sim.fe_system->fieldDofOffset(phi));
  const auto field_dofs = static_cast<std::size_t>(
      sim.fe_system->fieldDofHandler(phi).getNumDofs());
  ASSERT_LE(field_offset + field_dofs, repaired.size());
  for (std::size_t i = 0; i < field_dofs; ++i) {
    EXPECT_NEAR(repaired[field_offset + i],
                svmp::FE::Real{0.5} * solution[field_offset + i],
                1.0e-10);
  }
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
      <Small_cut_aggregation>false</Small_cut_aggregation>
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

TEST(ApplicationDriverLevelSetWorkflows,
     WallCompatibleNormalVelocityExtensionMatchesGeneratedInterfaceTraceAndProjectsOuterDryWalls)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeWorkflowThreeQuadStripMesh();
  constexpr svmp::label_t kWallLabel = 4343;
  constexpr int kInterfaceMarker = 706;
  mesh->register_label("wall", kWallLabel);
  for (const auto face : mesh->local_mesh().boundary_faces()) {
    mesh->set_boundary_label(face, kWallLabel);
  }
  const auto mesh_phi_field = svmp::MeshFields::attach_field(
      mesh->local_mesh(),
      svmp::EntityKind::Vertex,
      "phi",
      svmp::FieldScalarType::Float64,
      1);
  ASSERT_NE(svmp::MeshFields::field_data_as<svmp::real_t>(
                mesh->local_mesh(), mesh_phi_field),
            nullptr);
  auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Quad4,
      /*order=*/1);
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
      "ApplicationDriver nearest-interface extension phi");
  const auto source_coefficients = projectWorkflowVertexValues(
      *system,
      source_velocity,
      std::span<const svmp::FE::Real>(source_vertex_values.data(),
                                      source_vertex_values.size()),
      2u,
      "ApplicationDriver nearest-interface extension source velocity");

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

  auto params = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="fluid">
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>phi</Level_set_field_name>
      <Generated_interface_domain_id>trace_support_interface</Generated_interface_domain_id>
      <Interface_marker>706</Interface_marker>
      <Allow_corner_linearized_cut_geometry>true</Allow_corner_linearized_cut_geometry>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");
  svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
  const auto cut_report = refreshActiveCutIntegrationContextFromSolution(
      sim,
      *params,
      std::span<const svmp::FE::Real>(solution.data(), solution.size()),
      lifecycle,
      "application-driver-p1-trace-support-test");
  ASSERT_TRUE(cut_report.refreshed);
  ASSERT_GT(cut_report.interface_fragments, 0u);

  svmp::FE::systems::SystemStateView state{};
  state.u = std::span<const svmp::FE::Real>(solution.data(), solution.size());

  LevelSetAdvectionVelocityRequest request{};
  request.level_set_field_name = "phi";
  request.source_velocity_field_name = "Velocity";
  request.target_velocity_field_name = "LevelSetAdvectionVelocity";
  request.extension_method = "wall_compatible_normal";
  request.wall_face_names = {"wall"};
  request.wall_constraints = {{.face_name = "wall"}};
  request.active_side = application::core::LevelSetActiveSide::Negative;
  request.isovalue = 0.0;
  request.requested_interface_marker = kInterfaceMarker;
  request.active_cut_request_index = 0u;

  EXPECT_TRUE(updateLevelSetAdvectionVelocitiesFromState(
      sim,
      state,
      std::vector<LevelSetAdvectionVelocityRequest>{request}));

  const auto active_vertex = std::size_t{0};
  ASSERT_LE(phi_vertex_values[active_vertex], 0.0);
  const auto active_point = workflowVertexPoint(*mesh, active_vertex);
  const auto active_value = sim.fe_system->evaluateFieldAtPoint(
      target_velocity,
      svmp::FE::systems::SystemStateView{},
      active_point);
  ASSERT_TRUE(active_value.has_value());
  const auto active_expected = workflowVelocity(*mesh, active_vertex);
  EXPECT_NEAR((*active_value)[0], active_expected[0], 1.0e-10);
  EXPECT_NEAR((*active_value)[1], active_expected[1], 1.0e-10);

  // Vertices 1 and 5 are dry by nodal sign but support the retained cut cell.
  // They must be exact physical-velocity constraints, even though both lie on
  // the labelled wall.  Projecting either one would change the Q1 trace on
  // the free surface inside cell 0.
  for (const auto dry_trace_vertex : {std::size_t{1}, std::size_t{5}}) {
    ASSERT_GT(phi_vertex_values[dry_trace_vertex], 0.0);
    const auto point = workflowVertexPoint(*mesh, dry_trace_vertex);
    const auto value = sim.fe_system->evaluateFieldAtPoint(
        target_velocity,
        svmp::FE::systems::SystemStateView{},
        point);
    ASSERT_TRUE(value.has_value());
    const auto expected = workflowVelocity(*mesh, dry_trace_vertex);
    EXPECT_NEAR((*value)[0], expected[0], 1.0e-12)
        << "dry trace-support vertex " << dry_trace_vertex;
    EXPECT_NEAR((*value)[1], expected[1], 1.0e-12)
        << "dry trace-support vertex " << dry_trace_vertex;
  }

  const auto* context = sim.fe_system->cutIntegrationContext();
  ASSERT_NE(context, nullptr);
  std::size_t checked_interface_points = 0u;
  for (const auto* rule :
       context->interfaceRulesForMarker(kInterfaceMarker)) {
    ASSERT_NE(rule, nullptr);
    ASSERT_GE(rule->provenance.parent_entity, 0);
    const auto mapping = createCellGeometryMapping(
        sim.fe_system->meshAccess(), rule->provenance.parent_entity);
    ASSERT_NE(mapping, nullptr);
    for (const auto& qp : rule->points) {
      std::array<svmp::FE::Real, 3> point{
          qp.point[0], qp.point[1], qp.point[2]};
      if (rule->frame ==
          svmp::FE::geometry::CutGeometryFrame::Reference) {
        const auto physical =
            physicalCellPointAtReference(*mapping, qp.point);
        ASSERT_TRUE(physical.has_value());
        point = *physical;
      }
      const auto source_value = sim.fe_system->evaluateFieldAtPoint(
          source_velocity, state, point);
      const auto extension_value = sim.fe_system->evaluateFieldAtPoint(
          target_velocity,
          svmp::FE::systems::SystemStateView{},
          point);
      ASSERT_TRUE(source_value.has_value());
      ASSERT_TRUE(extension_value.has_value());
      EXPECT_NEAR((*extension_value)[0], (*source_value)[0], 1.0e-12)
          << "generated interface quadrature point "
          << checked_interface_points;
      EXPECT_NEAR((*extension_value)[1], (*source_value)[1], 1.0e-12)
          << "generated interface quadrature point "
          << checked_interface_points;
      ++checked_interface_points;
    }
  }
  EXPECT_GT(checked_interface_points, 0u);

  // Outside the cut-cell trace support, the existing graph extension and
  // wall projection remain in force.  Vertex 3 is a dry outer corner with two
  // independent wall normals, so its projected velocity is zero.
  const auto outer_dry_wall_vertex = std::size_t{3};
  ASSERT_GT(phi_vertex_values[outer_dry_wall_vertex], 0.0);
  const auto outer_point =
      workflowVertexPoint(*mesh, outer_dry_wall_vertex);
  const auto outer_value = sim.fe_system->evaluateFieldAtPoint(
      target_velocity,
      svmp::FE::systems::SystemStateView{},
      outer_point);
  ASSERT_TRUE(outer_value.has_value());
  EXPECT_NEAR((*outer_value)[0], 0.0, 1.0e-12);
  EXPECT_NEAR((*outer_value)[1], 0.0, 1.0e-12);
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     WallCompatibleNormalVelocityExtensionRejectsHigherOrderFields)
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
  system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi", .space = scalar_space, .components = 1});
  system->addField(svmp::FE::systems::FieldSpec{
      .name = "Velocity", .space = vector_space, .components = 2});
  system->addField(svmp::FE::systems::FieldSpec{
      .name = "LevelSetAdvectionVelocity",
      .space = vector_space,
      .components = 2,
      .source_kind = svmp::FE::systems::FieldSourceKind::PrescribedData});
  ASSERT_NO_THROW(system->setup({}));

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);
  LevelSetAdvectionVelocityRequest request{};
  request.level_set_field_name = "phi";
  request.source_velocity_field_name = "Velocity";
  request.target_velocity_field_name = "LevelSetAdvectionVelocity";
  request.extension_method = "wall_compatible_normal";

  EXPECT_THROW(
      {
        try {
          (void)updateLevelSetAdvectionVelocitiesFromState(
              sim,
              svmp::FE::systems::SystemStateView{},
              std::vector<LevelSetAdvectionVelocityRequest>{request});
        } catch (const std::runtime_error& error) {
          EXPECT_NE(std::string(error.what()).find("fixed P1"),
                    std::string::npos);
          throw;
        }
      },
      std::runtime_error);
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     WallCompatibleNormalVelocityExtensionRejectsMismatchedP1Layouts)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeWorkflowThreeQuadStripMesh();
  auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Quad4,
      /*order=*/1);
  auto source_space =
      std::make_shared<svmp::FE::spaces::ProductSpace>(scalar_space, 2);
  auto target_space =
      std::make_shared<svmp::FE::spaces::ProductSpace>(scalar_space, 3);
  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi", .space = scalar_space, .components = 1});
  system->addField(svmp::FE::systems::FieldSpec{
      .name = "Velocity", .space = source_space, .components = 2});
  system->addField(svmp::FE::systems::FieldSpec{
      .name = "LevelSetAdvectionVelocity",
      .space = target_space,
      .components = 3,
      .source_kind = svmp::FE::systems::FieldSourceKind::PrescribedData});
  ASSERT_NO_THROW(system->setup({}));

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);
  LevelSetAdvectionVelocityRequest request{};
  request.level_set_field_name = "phi";
  request.source_velocity_field_name = "Velocity";
  request.target_velocity_field_name = "LevelSetAdvectionVelocity";
  request.extension_method = "wall_compatible_normal";

  EXPECT_THROW(
      {
        try {
          (void)updateLevelSetAdvectionVelocitiesFromState(
              sim,
              svmp::FE::systems::SystemStateView{},
              std::vector<LevelSetAdvectionVelocityRequest>{request});
        } catch (const std::runtime_error& error) {
          EXPECT_NE(std::string(error.what()).find(
                        "identical component layouts"),
                    std::string::npos);
          throw;
        }
      },
      std::runtime_error);
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     AlgebraicWallCompatibleExtensionInvalidatesStaleMapWhenInterfaceDisappears)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeWorkflowThreeQuadStripMesh();
  auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Quad4,
      /*order=*/1);
  auto vector_space =
      std::make_shared<svmp::FE::spaces::ProductSpace>(scalar_space, 2);
  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto physical_velocity = system->addField(
      svmp::FE::systems::FieldSpec{
          .name = "Velocity", .space = vector_space, .components = 2});
  const auto phi = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi", .space = scalar_space, .components = 1});
  system->addOperator("level_set");

  svmp::FE::level_set::LevelSetTransportOptions transport{};
  transport.operator_tag = "level_set";
  transport.level_set.field_name = "phi";
  transport.level_set.auto_register_field = false;
  transport.velocity.field_name = "LevelSetAdvectionVelocity";
  transport.velocity.source =
      svmp::FE::level_set::LevelSetVelocitySource::CoupledField;
  transport.velocity.auto_register_field = true;
  transport.velocity.space = vector_space;
  transport.velocity.algebraic_extension_source_field_name = "Velocity";
  transport.supg.enabled = false;
  (void)svmp::FE::level_set::installLevelSetTransport(
      *system, scalar_space, transport);

  const auto extension_velocity =
      system->findFieldByName("LevelSetAdvectionVelocity");
  ASSERT_NE(extension_velocity, svmp::FE::INVALID_FIELD_ID);
  const auto extension_kernel =
      svmp::FE::level_set::findLevelSetVelocityExtensionConstraintKernel(
          *system, "level_set", extension_velocity);
  ASSERT_TRUE(extension_kernel);
  std::vector<svmp::FE::level_set::VelocityExtensionConstraintRow>
      stale_rows;
  for (svmp::FE::GlobalIndex vertex = 0;
       vertex < static_cast<svmp::FE::GlobalIndex>(mesh->n_vertices());
       ++vertex) {
    for (int component = 0; component < 2; ++component) {
      stale_rows.push_back(
          svmp::FE::level_set::VelocityExtensionConstraintRow{
              .vertex = vertex,
              .component = component,
              .dependencies = {
                  svmp::FE::level_set::VelocityExtensionDependency{
                      .field = svmp::FE::level_set::
                          VelocityExtensionDependencyField::SourceVelocity,
                      .vertex = vertex,
                      .component = component,
                      .coefficient = 1.0}}});
    }
  }
  extension_kernel->setFrozenRows(std::move(stale_rows), 1u);
  ASSERT_TRUE(extension_kernel->hasFrozenMap());
  ASSERT_NO_THROW(system->setup({}));

  std::vector<svmp::FE::Real> phi_values(mesh->n_vertices(), -1.0);
  std::vector<svmp::FE::Real> velocity_values(mesh->n_vertices() * 2u, 0.0);
  const auto phi_coefficients = projectWorkflowVertexValues(
      *system,
      phi,
      phi_values,
      1u,
      "ApplicationDriver disappeared-interface phi");
  const auto velocity_coefficients = projectWorkflowVertexValues(
      *system,
      physical_velocity,
      velocity_values,
      2u,
      "ApplicationDriver disappeared-interface velocity");
  std::vector<svmp::FE::Real> solution(
      static_cast<std::size_t>(system->dofHandler().getNumDofs()), 0.0);
  writeWorkflowFieldSlice(*system, phi, phi_coefficients, solution);
  writeWorkflowFieldSlice(
      *system, physical_velocity, velocity_coefficients, solution);

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);
  svmp::FE::systems::SystemStateView state{};
  state.u = solution;
  LevelSetAdvectionVelocityRequest request{};
  request.level_set_field_name = "phi";
  request.source_velocity_field_name = "Velocity";
  request.target_velocity_field_name = "LevelSetAdvectionVelocity";
  request.operator_tag = "level_set";
  request.extension_method = "wall_compatible_normal";
  request.enforce_wall_impermeability = false;

  EXPECT_THROW(
      {
        try {
          (void)updateLevelSetAdvectionVelocitiesFromState(
              sim,
              state,
              std::vector<LevelSetAdvectionVelocityRequest>{request});
        } catch (const std::runtime_error& error) {
          EXPECT_NE(std::string(error.what()).find(
                        "no resolved interface geometry samples"),
                    std::string::npos);
          throw;
        }
      },
      std::runtime_error);
  EXPECT_FALSE(extension_kernel->hasFrozenMap())
      << "A failed rebuild must not leave the previous interface map valid.";
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     AuthoritativeEmptyCutContextDoesNotFallBackToNodalCrossings)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  constexpr std::string_view kDomainId = "empty_authoritative_context";
  auto mesh = makeWorkflowThreeQuadStripMesh();
  auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Quad4,
      /*order=*/1);
  auto vector_space =
      std::make_shared<svmp::FE::spaces::ProductSpace>(scalar_space, 2);
  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto phi = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi", .space = scalar_space, .components = 1});
  const auto physical_velocity = system->addField(
      svmp::FE::systems::FieldSpec{
          .name = "Velocity", .space = vector_space, .components = 2});
  system->addField(svmp::FE::systems::FieldSpec{
      .name = "LevelSetAdvectionVelocity",
      .space = vector_space,
      .components = 2,
      .source_kind = svmp::FE::systems::FieldSourceKind::PrescribedData});
  ASSERT_NO_THROW(system->setup({}));

  svmp::FE::interfaces::GeneratedInterfaceMarkerKey marker_key{};
  marker_key.source =
      svmp::FE::interfaces::LevelSetInterfaceSource::fromField(phi);
  marker_key.domain_id = std::string(kDomainId);
  marker_key.isovalue = 0.0;
  marker_key.requested_marker = -1;
  const int interface_marker =
      svmp::FE::interfaces::stableGeneratedInterfaceMarker(marker_key);

  std::vector<svmp::FE::Real> phi_values(mesh->n_vertices(), 0.0);
  std::vector<svmp::FE::Real> velocity_values(mesh->n_vertices() * 2u, 0.0);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    phi_values[vertex] = workflowPhi(*mesh, vertex);
    const auto velocity = workflowVelocity(*mesh, vertex);
    velocity_values[2u * vertex] = velocity[0];
    velocity_values[2u * vertex + 1u] = velocity[1];
  }
  const auto phi_coefficients = projectWorkflowVertexValues(
      *system,
      phi,
      phi_values,
      1u,
      "ApplicationDriver empty-authoritative-context phi");
  const auto velocity_coefficients = projectWorkflowVertexValues(
      *system,
      physical_velocity,
      velocity_values,
      2u,
      "ApplicationDriver empty-authoritative-context velocity");
  std::vector<svmp::FE::Real> solution(
      static_cast<std::size_t>(system->dofHandler().getNumDofs()), 0.0);
  writeWorkflowFieldSlice(*system, phi, phi_coefficients, solution);
  writeWorkflowFieldSlice(
      *system, physical_velocity, velocity_coefficients, solution);

  auto empty_context =
      std::make_shared<svmp::FE::assembly::CutIntegrationContext>();
  empty_context->setExpectedGeneratedSourceValueRevision(
      interface_marker, 1u);
  ASSERT_TRUE(empty_context->hasExpectedGeneratedSourceValueRevision(
      interface_marker));
  ASSERT_FALSE(empty_context->hasGeneratedInterfaceMarker(interface_marker));
  ASSERT_FALSE(empty_context->hasGeneratedVolumeMarker(interface_marker));
  system->registerGeneratedEmbeddedInterfaceMarker(interface_marker);
  system->setCutIntegrationContext(empty_context);

  LevelSetAdvectionVelocityRequest request{};
  request.level_set_field_name = "phi";
  request.source_velocity_field_name = "Velocity";
  request.target_velocity_field_name = "LevelSetAdvectionVelocity";
  request.domain_id = std::string(kDomainId);
  request.extension_method = "wall_compatible_normal";
  request.enforce_wall_impermeability = false;
  ASSERT_FALSE(request.active_cut_request_index.has_value());
  EXPECT_EQ(configuredInterfaceVelocityMarker(*system, request), std::nullopt)
      << "Integer-only marker registration must not authorize an unkeyed "
         "nodal request.";
  EXPECT_FALSE(hasAuthoritativeInterfaceVelocityContext(*system, request));
  request.active_cut_request_index = 0u;
  ASSERT_EQ(configuredInterfaceVelocityMarker(*system, request),
            std::optional<int>{interface_marker});
  ASSERT_TRUE(hasAuthoritativeInterfaceVelocityContext(*system, request));
  EXPECT_TRUE(interfaceVelocitySampleCandidateCells(*system, request).empty());
  EXPECT_FALSE(nodalVelocityExtensionInterfaceCells(
                   *mesh, phi_values, request.isovalue)
                   .empty())
      << "The fixture must contain a nodal crossing that the authoritative "
         "empty context suppresses.";

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);
  svmp::FE::systems::SystemStateView state{};
  state.u = solution;
  EXPECT_THROW(
      {
        try {
          (void)updateLevelSetAdvectionVelocitiesFromState(
              sim,
              state,
              std::vector<LevelSetAdvectionVelocityRequest>{request});
        } catch (const std::runtime_error& error) {
          EXPECT_NE(std::string(error.what()).find(
                        "no resolved interface geometry samples"),
                    std::string::npos);
          throw;
        }
      },
      std::runtime_error);
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     VelocityExtensionGraphUsesCellEdgesWithoutQuadDiagonals)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  const auto mesh = makeDisconnectedWorkflowQuadPairMesh();
  const auto adjacency = velocityExtensionEdgeAdjacency(*mesh);
  ASSERT_EQ(adjacency.size(), 8u);

  EXPECT_EQ(adjacency[0], (std::vector<std::size_t>{1u, 3u}));
  EXPECT_EQ(adjacency[1], (std::vector<std::size_t>{0u, 2u}));
  EXPECT_EQ(adjacency[2], (std::vector<std::size_t>{1u, 3u}));
  EXPECT_EQ(adjacency[3], (std::vector<std::size_t>{0u, 2u}));
  EXPECT_EQ(adjacency[4], (std::vector<std::size_t>{5u, 7u}));
  EXPECT_EQ(adjacency[5], (std::vector<std::size_t>{4u, 6u}));
  EXPECT_EQ(adjacency[6], (std::vector<std::size_t>{5u, 7u}));
  EXPECT_EQ(adjacency[7], (std::vector<std::size_t>{4u, 6u}));
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     AlgebraicExtensionRefreshReprojectsStateAndChangesRevision)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeWorkflowThreeQuadStripMesh();
  auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Quad4,
      /*order=*/1);
  auto vector_space =
      std::make_shared<svmp::FE::spaces::ProductSpace>(scalar_space, 2);
  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto source_velocity = system->addField(
      svmp::FE::systems::FieldSpec{
          .name = "Velocity", .space = vector_space, .components = 2});
  const auto phi = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi", .space = scalar_space, .components = 1});
  system->addOperator("level_set");

  svmp::FE::level_set::LevelSetTransportOptions transport{};
  transport.operator_tag = "level_set";
  transport.level_set.field_name = "phi";
  transport.level_set.auto_register_field = false;
  transport.velocity.field_name = "LevelSetAdvectionVelocity";
  transport.velocity.source =
      svmp::FE::level_set::LevelSetVelocitySource::CoupledField;
  transport.velocity.auto_register_field = true;
  transport.velocity.space = vector_space;
  transport.velocity.algebraic_extension_source_field_name = "Velocity";
  transport.supg.enabled = false;
  (void)svmp::FE::level_set::installLevelSetTransport(
      *system, scalar_space, transport);
  const auto extension_velocity =
      system->findFieldByName("LevelSetAdvectionVelocity");
  ASSERT_NE(extension_velocity, svmp::FE::INVALID_FIELD_ID);
  const auto extension_kernel =
      svmp::FE::level_set::findLevelSetVelocityExtensionConstraintKernel(
          *system, "level_set", extension_velocity);
  ASSERT_TRUE(extension_kernel);
  ASSERT_NO_THROW(system->setup({}));

  std::vector<svmp::FE::Real> phi_values(mesh->n_vertices(), 0.0);
  std::vector<svmp::FE::Real> source_values(mesh->n_vertices() * 2u, 0.0);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    const auto point = workflowVertexPoint(*mesh, vertex);
    phi_values[vertex] = point[0] - 0.25;
    source_values[2u * vertex] = 2.0 + 3.0 * point[1];
    source_values[2u * vertex + 1u] = -1.0 + 0.5 * point[1];
  }
  const auto phi_coefficients = projectWorkflowVertexValues(
      *system, phi, phi_values, 1u, "algebraic extension refresh phi");
  const auto source_coefficients = projectWorkflowVertexValues(
      *system,
      source_velocity,
      source_values,
      2u,
      "algebraic extension refresh source");
  std::vector<svmp::FE::Real> initial(
      static_cast<std::size_t>(system->dofHandler().getNumDofs()), 777.0);
  writeWorkflowFieldSlice(*system, phi, phi_coefficients, initial);
  writeWorkflowFieldSlice(
      *system, source_velocity, source_coefficients, initial);

  std::unique_ptr<svmp::FE::backends::BackendFactory> factory;
  try {
    factory = svmp::FE::backends::BackendFactory::create(
        svmp::FE::backends::BackendKind::FSILS);
  } catch (const std::exception&) {
    GTEST_SKIP() << "Requires an available FE vector backend.";
  }
  auto state_vector = factory->createVector(
      system->dofHandler().getNumDofs());
  ASSERT_TRUE(state_vector);
  scatterFeOrderedSolution(*state_vector, initial);

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);
  svmp::FE::systems::SystemStateView state{};
  state.u_vector = state_vector.get();
  LevelSetAdvectionVelocityRequest request{};
  request.level_set_field_name = "phi";
  request.source_velocity_field_name = "Velocity";
  request.target_velocity_field_name = "LevelSetAdvectionVelocity";
  request.operator_tag = "level_set";
  request.extension_method = "wall_compatible_normal";
  request.extension_band_layers = 3;
  request.enforce_wall_impermeability = false;

  ASSERT_TRUE(updateLevelSetAdvectionVelocitiesFromState(
      sim, state, {request}));
  ASSERT_TRUE(extension_kernel->hasFrozenMap());
  const auto first_revision = extension_kernel->frozenMapRevision();
  EXPECT_NE(first_revision, 0u);
  const auto first_solution = gatherFeOrderedSolution(*state_vector);
  const auto extension_offset =
      sim.fe_system->fieldDofOffset(extension_velocity);
  const auto* extension_entity_map =
      sim.fe_system->fieldDofHandler(extension_velocity).getEntityDofMap();
  ASSERT_NE(extension_entity_map, nullptr);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    const auto dofs = extension_entity_map->getVertexDofs(
        static_cast<svmp::FE::GlobalIndex>(vertex));
    ASSERT_EQ(dofs.size(), 2u);
    for (std::size_t component = 0; component < 2u; ++component) {
      EXPECT_NEAR(first_solution[static_cast<std::size_t>(
                      extension_offset + dofs[component])],
                  source_values[2u * vertex + component],
                  1.0e-12);
    }
  }

  phi_values.back() += 1.0e-5;
  const auto changed_phi_coefficients = projectWorkflowVertexValues(
      *sim.fe_system,
      phi,
      phi_values,
      1u,
      "algebraic extension changed phi");
  auto changed_solution = first_solution;
  writeWorkflowFieldSlice(
      *sim.fe_system, phi, changed_phi_coefficients, changed_solution);
  for (const auto local_dof :
       sim.fe_system->fieldDofHandler(extension_velocity)
           .getPartition()
           .locallyOwned()) {
    changed_solution[static_cast<std::size_t>(extension_offset + local_dof)] =
        -999.0;
  }
  scatterFeOrderedSolution(*state_vector, changed_solution);
  ASSERT_TRUE(updateLevelSetAdvectionVelocitiesFromState(
      sim, state, {request}));
  EXPECT_NE(extension_kernel->frozenMapRevision(), first_revision);
  const auto refreshed_solution = gatherFeOrderedSolution(*state_vector);
  for (const auto local_dof :
       sim.fe_system->fieldDofHandler(extension_velocity)
           .getPartition()
           .locallyOwned()) {
    EXPECT_NE(refreshed_solution[static_cast<std::size_t>(
                  extension_offset + local_dof)],
              -999.0);
  }
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     VelocityExtensionRegressionConditionEstimateDetectsNearSingularity)
{
  std::array<std::array<double, 4>, 4> matrix{};
  matrix[0][0] = 1.0;
  matrix[0][1] = 1.0 - 1.0e-12;
  matrix[1][0] = matrix[0][1];
  matrix[1][1] = 1.0;

  const double condition = estimateSymmetricConditionNumber(matrix, 2);
  EXPECT_TRUE(std::isfinite(condition));
  EXPECT_GT(condition, kVelocityExtensionMaxRegressionCondition);
}

TEST(ApplicationDriverLevelSetWorkflows,
     WetExtensionRequestInfersOnlyZeroDirichletWalls)
{
  auto params = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="level_set">
    <Level_set_field_name>phi</Level_set_field_name>
    <Velocity_source>prescribed_data</Velocity_source>
    <Velocity_field_name>LevelSetAdvectionVelocity</Velocity_field_name>
    <Use_wet_extension_advection_velocity>true</Use_wet_extension_advection_velocity>
    <Source_velocity_field_name>Velocity</Source_velocity_field_name>
    <Wet_extension_advection_velocity_method>nearest_interface_point</Wet_extension_advection_velocity_method>
  </Add_equation>
  <Add_equation type="fluid">
    <Add_BC name="wall">
      <Type>Dir</Type>
      <Value>0.0</Value>
    </Add_BC>
    <Add_BC name="normal_wall">
      <Type>Dir</Type>
      <Value>0.0</Value>
      <Effective_direction>0 1</Effective_direction>
    </Add_BC>
    <Add_BC name="moving_lid">
      <Type>Dir</Type>
      <Value>1.0</Value>
    </Add_BC>
    <Add_BC name="outlet">
      <Type>Neu</Type>
      <Value>0.0</Value>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  const auto requests = levelSetAdvectionVelocityRequests(*params);
  ASSERT_EQ(requests.size(), 1u);
  EXPECT_EQ(requests.front().extension_method, "wall_compatible_normal");
  EXPECT_EQ(requests.front().wall_face_names,
            (std::vector<std::string>{"wall", "normal_wall"}));
  ASSERT_EQ(requests.front().wall_constraints.size(), 2u);
  EXPECT_EQ(requests.front().wall_constraints[0].face_name, "wall");
  EXPECT_TRUE(
      requests.front().wall_constraints[0].effective_direction.empty());
  EXPECT_EQ(requests.front().wall_constraints[1].face_name, "normal_wall");
  EXPECT_EQ(requests.front().wall_constraints[1].effective_direction,
            (std::vector<int>{0, 1}));
}

TEST(ApplicationDriverLevelSetWorkflows,
     WetExtensionExplicitWallFailsClosedForNonzeroDirichletData)
{
  auto params = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="level_set">
    <Level_set_field_name>phi</Level_set_field_name>
    <Velocity_source>prescribed_data</Velocity_source>
    <Velocity_field_name>LevelSetAdvectionVelocity</Velocity_field_name>
    <Use_wet_extension_advection_velocity>true</Use_wet_extension_advection_velocity>
    <Source_velocity_field_name>Velocity</Source_velocity_field_name>
    <Wet_extension_wall_faces>moving_lid</Wet_extension_wall_faces>
  </Add_equation>
  <Add_equation type="fluid">
    <Add_BC name="moving_lid">
      <Type>Dir</Type>
      <Value>1.0</Value>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  EXPECT_THROW(
      {
        try {
          (void)levelSetAdvectionVelocityRequests(*params);
        } catch (const std::runtime_error& error) {
          EXPECT_NE(std::string(error.what()).find("nonzero"),
                    std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}

TEST(ApplicationDriverLevelSetWorkflows,
     WetExtensionWallMasksComeOnlyFromOwningFluidEquation)
{
  auto params = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="level_set">
    <Level_set_field_name>phi</Level_set_field_name>
    <Velocity_source>prescribed_data</Velocity_source>
    <Velocity_field_name>LevelSetAdvectionVelocity</Velocity_field_name>
    <Use_wet_extension_advection_velocity>true</Use_wet_extension_advection_velocity>
    <Source_velocity_field_name>Velocity</Source_velocity_field_name>
  </Add_equation>
  <Add_equation type="heatS">
    <Add_BC name="wall">
      <Type>Dir</Type>
      <Value>0.0</Value>
      <Effective_direction>1 0</Effective_direction>
    </Add_BC>
    <Add_BC name="scalar_only">
      <Type>Dir</Type>
      <Value>0.0</Value>
    </Add_BC>
  </Add_equation>
  <Add_equation type="fluid">
    <Add_BC name="wall">
      <Type>Dir</Type>
      <Value>0.0</Value>
      <Effective_direction>0 1</Effective_direction>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  const auto requests = levelSetAdvectionVelocityRequests(*params);
  ASSERT_EQ(requests.size(), 1u);
  EXPECT_EQ(requests.front().wall_face_names,
            (std::vector<std::string>{"wall"}));
  ASSERT_EQ(requests.front().wall_constraints.size(), 1u);
  EXPECT_EQ(requests.front().wall_constraints.front().effective_direction,
            (std::vector<int>{0, 1}));
}

TEST(ApplicationDriverLevelSetWorkflows,
     WetExtensionWallDiscoveryFailsWithoutOwningFluidEquation)
{
  auto params = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="level_set">
    <Level_set_field_name>phi</Level_set_field_name>
    <Velocity_source>prescribed_data</Velocity_source>
    <Velocity_field_name>LevelSetAdvectionVelocity</Velocity_field_name>
    <Use_wet_extension_advection_velocity>true</Use_wet_extension_advection_velocity>
    <Source_velocity_field_name>Velocity</Source_velocity_field_name>
  </Add_equation>
  <Add_equation type="heatS">
    <Add_BC name="wall">
      <Type>Dir</Type>
      <Value>0.0</Value>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");

  EXPECT_THROW(
      {
        try {
          (void)levelSetAdvectionVelocityRequests(*params);
        } catch (const std::runtime_error& error) {
          EXPECT_NE(std::string(error.what()).find("exactly one fluid"),
                    std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}

TEST(ApplicationDriverLevelSetWorkflows,
     NormalBandVelocityExtensionHonorsStrongNoSlipAndNormalOnlyMasks)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  constexpr svmp::label_t kHorizontalWall = 4242;
  const auto mesh = makeWorkflowThreeQuadStripMesh();
  auto& local_mesh = mesh->local_mesh();
  for (const auto face : local_mesh.boundary_faces()) {
    const auto normal = local_mesh.face_normal(face);
    if (std::abs(normal[1]) > 0.9 * std::abs(normal[0])) {
      mesh->set_boundary_label(face, kHorizontalWall);
    }
  }

  std::vector<double> phi(mesh->n_vertices(), 0.0);
  std::vector<double> source(mesh->n_vertices() * 2u, 0.0);
  std::vector<std::uint8_t> active(mesh->n_vertices(), 0u);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    const auto point = workflowVertexPoint(*mesh, vertex);
    phi[vertex] = point[0] - 0.25;
    active[vertex] = phi[vertex] <= 0.0 ? 1u : 0u;
    source[2u * vertex] = 2.0 + point[1];
    source[2u * vertex + 1u] = 5.0 - 0.5 * point[1];
  }

  const std::vector<WallVelocityExtensionConstraint> normal_only{{
      .boundary_label = kHorizontalWall,
      .constrained_components = {false, true, false}}};
  std::vector<double> slip_extension;
  const auto slip_report = extendVelocityInLevelSetNormalBand(
      *mesh,
      svmp::MeshComm::self(),
      phi,
      source,
      /*source_components=*/2u,
      active,
      /*target_components=*/2u,
      /*copy_components=*/2u,
      /*band_layers=*/3,
      /*enforce_wall_impermeability=*/true,
      std::span<const WallVelocityExtensionConstraint>(normal_only),
      slip_extension);

  const std::vector<WallVelocityExtensionConstraint> no_slip{{
      .boundary_label = kHorizontalWall,
      .constrained_components = {true, true, true}}};
  std::vector<double> no_slip_extension;
  const auto no_slip_report = extendVelocityInLevelSetNormalBand(
      *mesh,
      svmp::MeshComm::self(),
      phi,
      source,
      /*source_components=*/2u,
      active,
      /*target_components=*/2u,
      /*copy_components=*/2u,
      /*band_layers=*/3,
      /*enforce_wall_impermeability=*/true,
      std::span<const WallVelocityExtensionConstraint>(no_slip),
      no_slip_extension);

  EXPECT_EQ(slip_report.vertices_outside_band, 0u);
  EXPECT_EQ(no_slip_report.vertices_outside_band, 0u);
  EXPECT_NEAR(slip_report.max_wall_normal_velocity, 0.0, 1.0e-12);
  EXPECT_NEAR(no_slip_report.max_wall_normal_velocity, 0.0, 1.0e-12);
  std::size_t checked_dry_wall_vertices = 0u;
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    if (active[vertex] != 0u) {
      continue;
    }
    const auto point = workflowVertexPoint(*mesh, vertex);
    EXPECT_NEAR(slip_extension[2u * vertex],
                2.0 + point[1],
                1.0e-11)
        << "normal-only wall must retain tangential extension at vertex "
        << vertex;
    EXPECT_NEAR(slip_extension[2u * vertex + 1u], 0.0, 1.0e-12)
        << "normal-only wall must remove its constrained component at vertex "
        << vertex;
    EXPECT_NEAR(no_slip_extension[2u * vertex], 0.0, 1.0e-12)
        << "no-slip wall must remove tangential extension at vertex " << vertex;
    EXPECT_NEAR(no_slip_extension[2u * vertex + 1u], 0.0, 1.0e-12)
        << "no-slip wall must remove normal extension at vertex " << vertex;
    ++checked_dry_wall_vertices;
  }
  EXPECT_EQ(checked_dry_wall_vertices, 6u);
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     NormalBandVelocityExtensionUsesSourceRowsOnDryCutCellSupport)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  const auto mesh = makeWorkflowThreeQuadStripMesh();
  std::vector<double> phi(mesh->n_vertices(), 0.0);
  std::vector<double> source(mesh->n_vertices() * 2u, 0.0);
  std::vector<std::uint8_t> seed(mesh->n_vertices(), 0u);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    const auto point = workflowVertexPoint(*mesh, vertex);
    phi[vertex] = point[0] - 0.25;
    seed[vertex] = phi[vertex] <= 0.0 ? 1u : 0u;
    source[2u * vertex] = 1.25 + 0.75 * point[0] - 0.5 * point[1];
    source[2u * vertex + 1u] = -0.4 + 0.2 * point[0] + point[1];
  }

  // Cell 0 contains the retained Q1 interface.  Its x=1 vertices (1 and 5)
  // are dry by sign but are necessary basis support for the physical trace.
  const std::array<svmp::FE::MeshIndex, 1> cut_cells{{0}};
  ASSERT_EQ(markVelocityExtensionTraceSupportCells(
                *mesh,
                std::span<const svmp::FE::MeshIndex>(cut_cells),
                seed),
            2u);
  ASSERT_EQ(synchronizeVelocityExtensionTraceSupportMask(
                *mesh, svmp::MeshComm::self(), seed),
            4u);

  std::vector<double> extension;
  std::vector<svmp::FE::level_set::VelocityExtensionConstraintRow> rows;
  const auto report = extendVelocityInLevelSetNormalBand(
      *mesh,
      svmp::MeshComm::self(),
      phi,
      source,
      /*source_components=*/2u,
      seed,
      /*target_components=*/2u,
      /*copy_components=*/2u,
      /*band_layers=*/3,
      /*enforce_wall_impermeability=*/false,
      std::span<const WallVelocityExtensionConstraint>{},
      extension,
      &rows);
  EXPECT_EQ(report.vertices_outside_band, 0u);

  auto find_row = [&](std::size_t vertex, int component)
      -> const svmp::FE::level_set::VelocityExtensionConstraintRow* {
    const auto found = std::find_if(
        rows.begin(), rows.end(), [&](const auto& row) {
          return row.vertex == static_cast<svmp::FE::GlobalIndex>(vertex) &&
                 row.component == component;
        });
    return found == rows.end() ? nullptr : &*found;
  };

  for (const auto dry_trace_vertex : {std::size_t{1}, std::size_t{5}}) {
    ASSERT_GT(phi[dry_trace_vertex], 0.0);
    for (int component = 0; component < 2; ++component) {
      const auto c = static_cast<std::size_t>(component);
      EXPECT_DOUBLE_EQ(extension[2u * dry_trace_vertex + c],
                       source[2u * dry_trace_vertex + c]);
      const auto* row = find_row(dry_trace_vertex, component);
      ASSERT_NE(row, nullptr);
      ASSERT_EQ(row->dependencies.size(), 1u);
      EXPECT_EQ(row->dependencies.front().field,
                svmp::FE::level_set::
                    VelocityExtensionDependencyField::SourceVelocity);
      EXPECT_EQ(row->dependencies.front().vertex,
                static_cast<svmp::FE::GlobalIndex>(dry_trace_vertex));
      EXPECT_EQ(row->dependencies.front().component, component);
      EXPECT_DOUBLE_EQ(row->dependencies.front().coefficient, 1.0);
    }
  }

  // Vertex 2 is dry and lies one graph layer beyond the cut-cell support, so
  // it must remain an extension dependency rather than a physical trace row.
  for (int component = 0; component < 2; ++component) {
    const auto* row = find_row(/*vertex=*/2u, component);
    ASSERT_NE(row, nullptr);
    ASSERT_FALSE(row->dependencies.empty());
    EXPECT_TRUE(std::all_of(
        row->dependencies.begin(), row->dependencies.end(),
        [](const auto& dependency) {
          return dependency.field ==
                 svmp::FE::level_set::
                     VelocityExtensionDependencyField::ExtensionVelocity;
        }));
  }
  EXPECT_LE(report.max_abs_graph_coefficient,
            1.0 + kVelocityExtensionRowTolerance);
  EXPECT_LE(report.max_graph_row_l1,
            1.0 + kVelocityExtensionRowTolerance);
  EXPECT_LE(report.max_graph_row_sum_error,
            kVelocityExtensionRowTolerance);
  EXPECT_LE(report.max_negative_graph_coefficient,
            kVelocityExtensionCoefficientTolerance);
  EXPECT_LE(report.max_extended_speed,
            report.max_seed_speed + 1.0e-12);
  for (const auto& row : rows) {
    if (row.dependencies.empty() ||
        row.dependencies.front().field !=
            svmp::FE::level_set::
                VelocityExtensionDependencyField::ExtensionVelocity) {
      continue;
    }
    double coefficient_sum = 0.0;
    double coefficient_l1 = 0.0;
    for (const auto& dependency : row.dependencies) {
      EXPECT_GE(dependency.coefficient,
                -kVelocityExtensionCoefficientTolerance);
      coefficient_sum += dependency.coefficient;
      coefficient_l1 += std::abs(dependency.coefficient);
    }
    EXPECT_NEAR(coefficient_sum, 1.0, kVelocityExtensionRowTolerance);
    EXPECT_LE(coefficient_l1, 1.0 + kVelocityExtensionRowTolerance);
  }
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     NormalBandVelocityExtensionReproducesTangentialAffineField)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  constexpr int subdivisions = 4;
  const auto mesh = makeWorkflowStructuredQuadMesh(subdivisions);
  std::vector<double> phi(mesh->n_vertices(), 0.0);
  std::vector<double> source(mesh->n_vertices() * 2u, 0.0);
  std::vector<std::uint8_t> active(mesh->n_vertices(), 0u);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    const auto point = workflowVertexPoint(*mesh, vertex);
    phi[vertex] = point[0] - 0.25;
    active[vertex] = phi[vertex] <= 0.0 ? 1u : 0u;
    // The exact normal extension for phi=x-0.25 is any field independent of
    // x.  Use two affine tangential components to exercise the local fit.
    source[2u * vertex] = 2.0 + 3.0 * point[1];
    source[2u * vertex + 1u] = -1.0 + 0.5 * point[1];
  }

  std::vector<double> extended;
  const auto report = extendVelocityInLevelSetNormalBand(
      *mesh,
      phi,
      source,
      /*source_components=*/2u,
      active,
      /*target_components=*/2u,
      /*copy_components=*/2u,
      /*band_layers=*/subdivisions,
      /*enforce_wall_impermeability=*/false,
      /*wall_boundary_labels=*/{},
      extended);

  EXPECT_EQ(report.vertices_outside_band, 0u);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    const auto point = workflowVertexPoint(*mesh, vertex);
    EXPECT_NEAR(extended[2u * vertex], 2.0 + 3.0 * point[1], 1.0e-11)
        << "vertex " << vertex;
    EXPECT_NEAR(extended[2u * vertex + 1u],
                -1.0 + 0.5 * point[1],
                1.0e-11)
        << "vertex " << vertex;
  }
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     VelocityExtensionMapSnapshotTracksEveryRevisionDomain)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  const auto mesh = makeWorkflowThreeQuadStripMesh();
  std::vector<double> phi(mesh->n_vertices(), 0.0);
  std::vector<double> source(mesh->n_vertices() * 2u, 0.0);
  std::vector<std::uint8_t> seed(mesh->n_vertices(), 0u);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    const auto point = workflowVertexPoint(*mesh, vertex);
    phi[vertex] = point[0] - 0.25;
    seed[vertex] = phi[vertex] <= 0.0 ? 1u : 0u;
    source[2u * vertex] = 2.0 + 3.0 * point[1];
    source[2u * vertex + 1u] = -1.0 + 0.5 * point[1];
  }
  const std::array<svmp::FE::MeshIndex, 1> cut_cells{{0}};
  ASSERT_EQ(markVelocityExtensionTraceSupportCells(
                *mesh,
                std::span<const svmp::FE::MeshIndex>(cut_cells),
                seed),
            2u);

  const auto revision = application::core::velocityExtensionMapRevision(
      /*mesh_geometry=*/11u,
      /*mesh_topology=*/12u,
      /*mesh_ownership=*/13u,
      /*mesh_numbering=*/14u,
      /*free_surface_geometry=*/15u,
      phi,
      seed);
  const auto snapshot =
      application::core::buildVelocityExtensionMapSnapshot(
          *mesh,
          svmp::MeshComm::self(),
          revision,
          phi,
          source,
          /*source_components=*/2u,
          seed,
          /*target_components=*/2u,
          /*copy_components=*/2u,
          /*band_layers=*/3,
          /*enforce_wall_impermeability=*/false,
          std::span<const WallVelocityExtensionConstraint>{});
  ASSERT_TRUE(snapshot);
  EXPECT_EQ(snapshot->revision(), revision);
  EXPECT_EQ(snapshot->preview().size(), source.size());
  EXPECT_EQ(snapshot->componentAssignment().size(), mesh->n_vertices());
  EXPECT_GT(snapshot->report().max_extrapolation_distance, 0.0);
  EXPECT_LE(snapshot->report().max_constant_reproduction_error,
            kVelocityExtensionRowTolerance);
  EXPECT_LE(snapshot->report().max_linear_reproduction_error,
            kVelocityExtensionRowTolerance);
  EXPECT_LE(snapshot->wetToDryAmplification(), 1.0 + 1.0e-12);

  auto detached_rows = snapshot->copyRows();
  ASSERT_FALSE(detached_rows.empty());
  detached_rows.clear();
  EXPECT_FALSE(snapshot->rows().empty());

  auto changed_phi = phi;
  changed_phi.back() += 1.0e-6;
  const auto phi_revision = application::core::velocityExtensionMapRevision(
      11u, 12u, 13u, 14u, 15u, changed_phi, seed);
  EXPECT_NE(phi_revision.key(), revision.key());

  auto changed_seed = seed;
  changed_seed.back() = changed_seed.back() == 0u ? 1u : 0u;
  const auto active_revision =
      application::core::velocityExtensionMapRevision(
          11u, 12u, 13u, 14u, 15u, phi, changed_seed);
  EXPECT_NE(active_revision.key(), revision.key());

  const auto geometry_revision =
      application::core::velocityExtensionMapRevision(
          16u, 12u, 13u, 14u, 15u, phi, seed);
  const auto topology_revision =
      application::core::velocityExtensionMapRevision(
          11u, 17u, 13u, 14u, 15u, phi, seed);
  const auto ownership_revision =
      application::core::velocityExtensionMapRevision(
          11u, 12u, 18u, 14u, 15u, phi, seed);
  const auto surface_revision =
      application::core::velocityExtensionMapRevision(
          11u, 12u, 13u, 14u, 19u, phi, seed);
  EXPECT_NE(geometry_revision.key(), revision.key());
  EXPECT_NE(topology_revision.key(), revision.key());
  EXPECT_NE(ownership_revision.key(), revision.key());
  EXPECT_NE(surface_revision.key(), revision.key());
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     NormalBandVelocityExtensionManufacturedRefinementConvergesAndProjectsWalls)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  constexpr svmp::label_t kHorizontalWall = 4351;
  constexpr double kInterfaceX = 0.30;
  const double pi = std::acos(-1.0);
  const auto exact_tangential_velocity = [pi](double y) {
    return std::sin(pi * y) + 0.2 * std::cos(2.0 * pi * y);
  };
  const auto exact_normal_velocity = [pi](double y) {
    return 0.15 * std::sin(pi * y);
  };

  struct RefinementError {
    double l2{0.0};
    double linf{0.0};
  };
  std::vector<RefinementError> errors;
  for (const int subdivisions : {8, 16, 32}) {
    SCOPED_TRACE("subdivisions=" + std::to_string(subdivisions));
    const auto mesh = makeWorkflowStructuredQuadMesh(subdivisions);
    for (const auto face : mesh->local_mesh().boundary_faces()) {
      const auto normal = mesh->local_mesh().face_normal(face);
      if (std::abs(normal[1]) > 0.9 * std::abs(normal[0])) {
        mesh->set_boundary_label(face, kHorizontalWall);
      }
    }

    std::vector<double> phi(mesh->n_vertices(), 0.0);
    std::vector<double> source(mesh->n_vertices() * 2u, 0.0);
    std::vector<std::uint8_t> active(mesh->n_vertices(), 0u);
    for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
      const auto point = workflowVertexPoint(*mesh, vertex);
      phi[vertex] = point[0] - kInterfaceX;
      active[vertex] = phi[vertex] <= 0.0 ? 1u : 0u;
      // For the planar interface, n=ex and this smooth field is the exact
      // solution of n.grad(u_ext)=0.  Its second component vanishes at the
      // horizontal walls, so it also manufactures the no-penetration trace
      // without making that component identically zero in the interior.
      source[2u * vertex] = exact_tangential_velocity(point[1]);
      source[2u * vertex + 1u] = exact_normal_velocity(point[1]);
    }

    const std::vector<WallVelocityExtensionConstraint> constraints{{
        .boundary_label = kHorizontalWall,
        .constrained_components = {false, true, false}}};
    std::vector<double> extended;
    const auto report = extendVelocityInLevelSetNormalBand(
        *mesh,
        svmp::MeshComm::self(),
        phi,
        source,
        /*source_components=*/2u,
        active,
        /*target_components=*/2u,
        /*copy_components=*/2u,
        /*band_layers=*/subdivisions,
        /*enforce_wall_impermeability=*/true,
        std::span<const WallVelocityExtensionConstraint>(constraints),
        extended);

    EXPECT_EQ(report.vertices_outside_band, 0u);
    EXPECT_GT(report.wall_projected_vertices, 0u);
    EXPECT_NEAR(report.max_wall_normal_velocity, 0.0, 1.0e-13);
    double squared_error = 0.0;
    double max_error = 0.0;
    std::size_t dry_vertices = 0u;
    for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
      const auto point = workflowVertexPoint(*mesh, vertex);
      const std::array<double, 2> exact{{
          exact_tangential_velocity(point[1]),
          exact_normal_velocity(point[1])}};
      if (active[vertex] != 0u) {
        EXPECT_NEAR(extended[2u * vertex], exact[0], 1.0e-14);
        EXPECT_NEAR(extended[2u * vertex + 1u], exact[1], 1.0e-14);
        continue;
      }
      for (std::size_t component = 0; component < 2u; ++component) {
        const double error =
            std::abs(extended[2u * vertex + component] - exact[component]);
        squared_error += error * error;
        max_error = std::max(max_error, error);
      }
      if (point[1] <= 1.0e-14 || point[1] >= 1.0 - 1.0e-14) {
        EXPECT_NEAR(extended[2u * vertex + 1u], 0.0, 1.0e-13);
      }
      ++dry_vertices;
    }
    ASSERT_GT(dry_vertices, 0u);
    errors.push_back({
        .l2 = std::sqrt(
            squared_error / static_cast<double>(2u * dry_vertices)),
        .linf = max_error});
    RecordProperty("extension_l2_N" + std::to_string(subdivisions),
                   std::to_string(errors.back().l2));
    RecordProperty("extension_linf_N" + std::to_string(subdivisions),
                   std::to_string(errors.back().linf));
  }

  ASSERT_EQ(errors.size(), 3u);
  for (std::size_t level = 1; level < errors.size(); ++level) {
    constexpr double exact_reproduction_tolerance = 5.0e-13;
    if (errors[level - 1u].linf <= exact_reproduction_tolerance &&
        errors[level].linf <= exact_reproduction_tolerance) {
      RecordProperty("extension_exact_reproduction_level" +
                         std::to_string(level),
                     "true");
      EXPECT_LE(errors[level - 1u].l2, exact_reproduction_tolerance);
      EXPECT_LE(errors[level].l2, exact_reproduction_tolerance);
      continue;
    }
    ASSERT_GT(errors[level].l2, 0.0);
    ASSERT_GT(errors[level].linf, 0.0);
    const double l2_rate =
        std::log(errors[level - 1u].l2 / errors[level].l2) /
        std::log(2.0);
    const double linf_rate =
        std::log(errors[level - 1u].linf / errors[level].linf) /
        std::log(2.0);
    RecordProperty("extension_l2_rate_level" + std::to_string(level),
                   std::to_string(l2_rate));
    RecordProperty("extension_linf_rate_level" + std::to_string(level),
                   std::to_string(linf_rate));
    EXPECT_GT(l2_rate, 0.75)
        << "coarse=" << errors[level - 1u].l2
        << " fine=" << errors[level].l2;
    EXPECT_GT(linf_rate, 0.60)
        << "coarse=" << errors[level - 1u].linf
        << " fine=" << errors[level].linf;
  }
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     NormalBandVelocityExtensionDoesNotSwitchDisconnectedComponents)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  const auto mesh = makeDisconnectedWorkflowQuadPairMesh();
  std::vector<double> phi(mesh->n_vertices(), 0.0);
  std::vector<double> source(mesh->n_vertices() * 2u, 0.0);
  std::vector<std::uint8_t> active(mesh->n_vertices(), 0u);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    const auto point = workflowVertexPoint(*mesh, vertex);
    phi[vertex] = point[0] - 0.25;
    active[vertex] = phi[vertex] <= 0.0 ? 1u : 0u;
    const bool first_component = vertex < 4u;
    source[2u * vertex] = first_component ? 2.0 : -7.0;
    source[2u * vertex + 1u] = first_component ? 3.0 : 11.0;
  }

  std::vector<double> extended;
  const auto report = extendVelocityInLevelSetNormalBand(
      *mesh,
      phi,
      source,
      /*source_components=*/2u,
      active,
      /*target_components=*/2u,
      /*copy_components=*/2u,
      /*band_layers=*/1,
      /*enforce_wall_impermeability=*/false,
      /*wall_boundary_labels=*/{},
      extended);

  EXPECT_EQ(report.vertices_outside_band, 0u);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    const bool first_component = vertex < 4u;
    EXPECT_NEAR(extended[2u * vertex], first_component ? 2.0 : -7.0,
                1.0e-12)
        << "vertex " << vertex;
    EXPECT_NEAR(extended[2u * vertex + 1u], first_component ? 3.0 : 11.0,
                1.0e-12)
        << "vertex " << vertex;
  }
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     NormalBandVelocityExtensionFailsClosedOnEquidistantComponentBands)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  const auto mesh = makeWorkflowFourQuadStripMesh();
  std::vector<double> phi(mesh->n_vertices(), 0.0);
  std::vector<double> source(mesh->n_vertices() * 2u, 0.0);
  std::vector<std::uint8_t> active(mesh->n_vertices(), 0u);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    const auto point = workflowVertexPoint(*mesh, vertex);
    // Two disconnected active components live at opposite ends of one
    // connected background mesh.  Their two-layer graph bands collide at
    // x=2, where the old unlabeled propagation blended both source fields.
    // Place both interfaces one quarter cell inside their end cells so that
    // x=2 is also equidistant from the two geometric interfaces.
    phi[vertex] = std::min(point[0] - 0.25,
                           4.0 - point[0] - 0.25);
    active[vertex] = phi[vertex] <= 0.0 ? 1u : 0u;
    const bool left_branch = point[0] <= 2.0;
    source[2u * vertex] = left_branch
                              ? 2.0 + 3.0 * point[1]
                              : -7.0 + 2.0 * point[1];
    source[2u * vertex + 1u] = left_branch
                                   ? -1.0 + 0.5 * point[1]
                                   : 11.0 - 4.0 * point[1];
  }

  std::vector<double> extended;
  EXPECT_THROW(
      {
        try {
          (void)extendVelocityInLevelSetNormalBand(
              *mesh,
              phi,
              source,
              /*source_components=*/2u,
              active,
              /*target_components=*/2u,
              /*copy_components=*/2u,
              /*band_layers=*/2,
              /*enforce_wall_impermeability=*/false,
              /*wall_boundary_labels=*/{},
              extended);
        } catch (const std::runtime_error& error) {
          EXPECT_NE(std::string(error.what()).find(
                        "unresolved equidistant active-component collision"),
                    std::string::npos);
          throw;
        }
      },
      std::runtime_error);
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     NormalBandVelocityExtensionHonorsGraphLayerCutoff)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  const auto mesh = makeWorkflowThreeQuadStripMesh();
  std::vector<double> phi(mesh->n_vertices(), 0.0);
  std::vector<double> source(mesh->n_vertices() * 2u, 0.0);
  std::vector<std::uint8_t> active(mesh->n_vertices(), 0u);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    const auto point = workflowVertexPoint(*mesh, vertex);
    phi[vertex] = point[0] - 0.25;
    active[vertex] = phi[vertex] <= 0.0 ? 1u : 0u;
    source[2u * vertex] = 4.0;
    source[2u * vertex + 1u] = -2.0;
  }

  std::vector<double> extended;
  const auto report = extendVelocityInLevelSetNormalBand(
      *mesh,
      phi,
      source,
      /*source_components=*/2u,
      active,
      /*target_components=*/2u,
      /*copy_components=*/2u,
      /*band_layers=*/1,
      /*enforce_wall_impermeability=*/false,
      /*wall_boundary_labels=*/{},
      extended);

  EXPECT_EQ(report.vertices_outside_band, 4u);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    const auto point = workflowVertexPoint(*mesh, vertex);
    if (point[0] <= 1.0) {
      EXPECT_NEAR(extended[2u * vertex], 4.0, 1.0e-12)
          << "vertex " << vertex;
      EXPECT_NEAR(extended[2u * vertex + 1u], -2.0, 1.0e-12)
          << "vertex " << vertex;
    } else {
      EXPECT_NEAR(extended[2u * vertex], 0.0, 1.0e-12)
          << "vertex " << vertex;
      EXPECT_NEAR(extended[2u * vertex + 1u], 0.0, 1.0e-12)
          << "vertex " << vertex;
    }
  }
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     NonconvergedReinitializationDoesNotModifyAcceptedHistory)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeWorkflowQuadPatch2x2Mesh();
  auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Quad4,
      /*order=*/1);
  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto phi = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi",
      .space = scalar_space,
      .components = 1,
  });
  ASSERT_NO_THROW(system->setup({}));

  std::vector<svmp::FE::Real> phi_vertex_values(mesh->n_vertices(), 0.0);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    const auto point = workflowVertexPoint(*mesh, vertex);
    // Nodal interpolation of this off-center circle gives four connected cut
    // cells with different local gradient magnitudes.  In a fixed continuous
    // P1 space, independently normalizing those gradients would require
    // different positive cell multipliers, but continuity forces their shared
    // vertex values to agree.  Exact zero-set preservation and exact
    // redistancing are therefore not simultaneously representable here.
    const auto dx = point[0] - svmp::FE::Real{0.8};
    const auto dy = point[1] - svmp::FE::Real{0.9};
    phi_vertex_values[vertex] =
        dx * dx + dy * dy - svmp::FE::Real{0.49};
  }
  const auto coefficients = projectWorkflowVertexValues(
      *system,
      phi,
      phi_vertex_values,
      /*components=*/1u,
      "ApplicationDriver nonconverged reinitialization phi");
  std::vector<svmp::FE::Real> initial(
      static_cast<std::size_t>(system->dofHandler().getNumDofs()), 0.0);
  writeWorkflowFieldSlice(*system, phi, coefficients, initial);

  // Verify the low-level operation deterministically produces a geometrically
  // bounded but nonconverged candidate with the production defaults.
  svmp::FE::level_set::LevelSetReinitializationOptions defaults{};
  std::vector<svmp::FE::Real> candidate;
  const auto repair =
      svmp::FE::level_set::repairLevelSetSignedDistanceByProjection(
          *system,
          phi,
          defaults,
          initial,
          candidate);
  ASSERT_TRUE(repair.success) << repair.diagnostic;
  EXPECT_FALSE(repair.converged);
  EXPECT_TRUE(repair.zero_set_bound_satisfied);
  EXPECT_LE(repair.max_interface_displacement,
            defaults.max_zero_set_displacement + 1.0e-12);
  EXPECT_GT(repair.max_signed_distance_error,
            defaults.signed_distance_tolerance);

  std::unique_ptr<svmp::FE::backends::BackendFactory> factory;
  try {
    factory = svmp::FE::backends::BackendFactory::create(
        svmp::FE::backends::BackendKind::Eigen);
  } catch (const std::exception&) {
    GTEST_SKIP() << "Requires the Eigen FE backend.";
  }
  ASSERT_NE(factory, nullptr);
  auto history = svmp::FE::timestepping::TimeHistory::allocate(
      *factory,
      system->dofHandler().getNumDofs());
  history.setStepIndex(1);
  scatterFeOrderedSolution(history.u(), initial);
  scatterFeOrderedSolution(history.uPrev(), initial);
  scatterFeOrderedSolution(history.uPrev2(), initial);

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);

  LevelSetMaintenanceRequest request{};
  request.level_set_field_name = "phi";
  request.reinitialization.enabled = true;
  request.reinitialization.cadence_steps = 1;
  std::vector<LevelSetMaintenanceRequest> requests{request};

  testing::internal::CaptureStdout();
  const bool changed = applyLevelSetMaintenance(sim, history, requests);
  const auto output = testing::internal::GetCapturedStdout();

  EXPECT_FALSE(changed);
  EXPECT_NE(output.find("reason=nonconverged"), std::string::npos);
  EXPECT_EQ(gatherFeOrderedSolution(history.u()), initial);
  EXPECT_EQ(gatherFeOrderedSolution(history.uPrev()), initial);
  EXPECT_EQ(gatherFeOrderedSolution(history.uPrev2()), initial);
#endif
}

TEST(ApplicationDriverLevelSetWorkflows,
     ConvergedMaintenanceAppliesOneRepresentationDeltaToEveryHistoryLevel)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeWorkflowQuadPatch2x2Mesh();
  auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Quad4,
      /*order=*/1);
  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto phi = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi",
      .space = scalar_space,
      .components = 1,
  });
  ASSERT_NO_THROW(system->setup({}));

  const auto make_plane = [&](svmp::FE::Real offset,
                              svmp::FE::Real gradient_scale) {
    std::vector<svmp::FE::Real> values(mesh->n_vertices(), 0.0);
    for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
      values[vertex] = gradient_scale *
          (workflowVertexPoint(*mesh, vertex)[0] - offset);
    }
    const auto coefficients = projectWorkflowVertexValues(
        *system,
        phi,
        values,
        /*components=*/1u,
        "ApplicationDriver maintenance-history plane");
    std::vector<svmp::FE::Real> solution(
        static_cast<std::size_t>(system->dofHandler().getNumDofs()), 0.0);
    writeWorkflowFieldSlice(*system, phi, coefficients, solution);
    return solution;
  };
  const auto current_before = make_plane(0.80, 2.0);
  const auto previous_before = make_plane(0.72, 2.0);
  const auto previous2_before = make_plane(0.63, 2.0);

  std::unique_ptr<svmp::FE::backends::BackendFactory> factory;
  try {
    factory = svmp::FE::backends::BackendFactory::create(
        svmp::FE::backends::BackendKind::FSILS);
  } catch (const std::exception&) {
    GTEST_SKIP() << "Requires an available FE vector backend.";
  }
  auto history = svmp::FE::timestepping::TimeHistory::allocate(
      *factory,
      system->dofHandler().getNumDofs(),
      /*history_depth=*/2,
      /*allocate_second_order_state=*/true);
  history.setStepIndex(1);
  history.setDt(0.1);
  history.setPrevDt(0.1);
  scatterFeOrderedSolution(history.u(), current_before);
  scatterFeOrderedSolution(history.uPrev(), previous_before);
  scatterFeOrderedSolution(history.uPrev2(), previous2_before);
  std::vector<svmp::FE::Real> rate_before(current_before.size(), 0.375);
  scatterFeOrderedSolution(history.uDot(), rate_before);

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);

  LevelSetMaintenanceRequest request{};
  request.level_set_field_name = "phi";
  request.reinitialization.enabled = true;
  request.reinitialization.cadence_steps = 1;
  // With the production relaxation factor (0.3), twenty iterations leave an
  // O(10^-3) residual for this factor-of-two planar distortion.  Give the
  // projection enough iterations to meet the deliberately strict tolerance
  // so this test exercises the converged maintenance/history path.
  request.reinitialization.max_iterations = 100;
  request.reinitialization.signed_distance_tolerance = 1.0e-10;
  std::vector<LevelSetMaintenanceRequest> requests{request};

  testing::internal::CaptureStdout();
  const bool changed = applyLevelSetMaintenance(sim, history, requests);
  const auto output = testing::internal::GetCapturedStdout();
  ASSERT_TRUE(changed);
  EXPECT_NE(output.find("temporal_increments=preserved"), std::string::npos);

  const auto current_after = gatherFeOrderedSolution(history.u());
  const auto previous_after = gatherFeOrderedSolution(history.uPrev());
  const auto previous2_after = gatherFeOrderedSolution(history.uPrev2());
  ASSERT_EQ(current_after.size(), current_before.size());
  bool saw_nonzero_delta = false;
  for (std::size_t i = 0; i < current_after.size(); ++i) {
    const auto current_delta = current_after[i] - current_before[i];
    saw_nonzero_delta = saw_nonzero_delta ||
                        std::abs(current_delta) > 1.0e-12;
    EXPECT_NEAR(previous_after[i] - previous_before[i],
                current_delta,
                1.0e-12)
        << "global DOF " << i;
    EXPECT_NEAR(previous2_after[i] - previous2_before[i],
                current_delta,
                1.0e-12)
        << "global DOF " << i;
    EXPECT_NEAR((current_after[i] - previous_after[i]),
                (current_before[i] - previous_before[i]),
                1.0e-12)
        << "current/previous increment at global DOF " << i;
    EXPECT_NEAR((previous_after[i] - previous2_after[i]),
                (previous_before[i] - previous2_before[i]),
                1.0e-12)
        << "previous/older increment at global DOF " << i;
  }
  EXPECT_TRUE(saw_nonzero_delta);
  EXPECT_EQ(gatherFeOrderedSolution(history.uDot()), rate_before);
#endif
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
namespace {

class ApplicationDriverConservativePhaseCandidatesTest
    : public ::testing::Test {
protected:
  void SetUp() override
  {
    mesh_ = makeWorkflowTriangleMesh();
    (void)svmp::MeshFields::attach_field(
        mesh_->local_mesh(),
        svmp::EntityKind::Vertex,
        "phi",
        svmp::FieldScalarType::Float64,
        1);
    auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
        svmp::FE::ElementType::Triangle3,
        /*order=*/1);
    auto system =
        std::make_unique<svmp::FE::systems::FESystem>(mesh_);
    phi_ = system->addField(svmp::FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    phase_ = system->addField(svmp::FE::systems::FieldSpec{
        .name = "phase",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system->setup({}));

    sim_.primary_mesh = mesh_;
    sim_.fe_system = std::move(system);
    sim_.backend = svmp::FE::backends::BackendFactory::create(
        svmp::FE::backends::BackendKind::FSILS);
    ASSERT_NE(sim_.backend, nullptr);
    auto allocated_history =
        svmp::FE::timestepping::TimeHistory::allocate(
            *sim_.backend,
            sim_.fe_system->dofHandler().getNumDofs(),
            /*history_depth=*/2,
            /*allocate_second_order_state=*/true);
    sim_.time_history =
        std::make_unique<svmp::FE::timestepping::TimeHistory>(
            std::move(allocated_history));
    history().setDt(0.05);
    history().setPrevDt(0.05);

    std::vector<svmp::FE::Real> phi_vertex_values(
        mesh_->n_vertices(), svmp::FE::Real{0.0});
    for (std::size_t vertex = 0u; vertex < mesh_->n_vertices(); ++vertex) {
      phi_vertex_values[vertex] =
          workflowVertexPoint(*mesh_, vertex)[0] - svmp::FE::Real{0.75};
    }
    const auto phi_coefficients = projectWorkflowVertexValues(
        *sim_.fe_system,
        phi_,
        phi_vertex_values,
        /*components=*/1u,
        "ApplicationDriver conservative phase phi");
    std::vector<svmp::FE::Real> initial(solutionSize(),
                                         svmp::FE::Real{0.0});
    writeWorkflowFieldSlice(
        *sim_.fe_system, phi_, phi_coefficients, initial);
    scatterFeOrderedSolution(history().u(), initial);
    scatterFeOrderedSolution(history().uPrev(), initial);
    scatterFeOrderedSolution(history().uPrev2(), initial);
    std::vector<svmp::FE::Real> initial_rates(
        solutionSize(), svmp::FE::Real{0.5});
    scatterFeOrderedSolution(history().uDot(), initial_rates);
    scatterFeOrderedSolution(history().uDDot(), initial_rates);

    params_ = parseWorkflowParametersXml(R"xml(
<svMultiPhysicsFile>
  <Add_equation type="level_set">
    <Level_set_field_name>phi</Level_set_field_name>
    <Velocity_source>constant</Velocity_source>
    <Constant_velocity>0.0 0.0 0.0</Constant_velocity>
    <Enable_conservative_phase_transport>true</Enable_conservative_phase_transport>
    <Conservative_phase_field_name>phase</Conservative_phase_field_name>
  </Add_equation>
  <Add_equation type="fluid">
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>phi</Level_set_field_name>
      <Generated_interface_domain_id>conservative_phase_interface</Generated_interface_domain_id>
      <Interface_marker>911</Interface_marker>
      <Allow_corner_linearized_cut_geometry>true</Allow_corner_linearized_cut_geometry>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");
    active_requests_ = activeCutVolumeRequests(*params_);
    ASSERT_EQ(active_requests_.size(), 1u);
    requests_ = levelSetMaintenanceRequests(*params_);
    ASSERT_EQ(requests_.size(), 1u);
    ASSERT_TRUE(requests_.front().conservative_phase.enabled);
    ASSERT_TRUE(requests_.front().volume_cut_request.has_value());

    const auto initial_refresh = refreshActiveCutIntegrationContextCached(
        sim_,
        *params_,
        history().u(),
        lifecycle_,
        refresh_cache_,
        "application-driver-conservative-phase-initial");
    ASSERT_TRUE(initial_refresh.refreshed);
    ASSERT_NO_THROW(initializeConservativePhaseStates(sim_, requests_));
    ASSERT_TRUE(requests_.front().conservative_phase_initialized);
    initialized_solution_ = gatherFeOrderedSolution(history().u());
  }

  [[nodiscard]] svmp::FE::timestepping::TimeHistory& history()
  {
    return *sim_.time_history;
  }

  [[nodiscard]] std::size_t solutionSize() const
  {
    return static_cast<std::size_t>(
        sim_.fe_system->dofHandler().getNumDofs());
  }

  [[nodiscard]] std::size_t fieldOffset(svmp::FE::FieldId field) const
  {
    const auto offset = sim_.fe_system->fieldDofOffset(field);
    if (offset < 0) {
      throw std::runtime_error(
          "ApplicationDriver conservative phase test has no field offset");
    }
    return static_cast<std::size_t>(offset);
  }

  [[nodiscard]] std::size_t fieldCount(svmp::FE::FieldId field) const
  {
    return static_cast<std::size_t>(
        sim_.fe_system->fieldDofHandler(field).getNumDofs());
  }

  [[nodiscard]] std::vector<svmp::FE::Real> fieldSlice(
      std::span<const svmp::FE::Real> solution,
      svmp::FE::FieldId field) const
  {
    const auto offset = fieldOffset(field);
    const auto count = fieldCount(field);
    if (offset + count > solution.size()) {
      throw std::runtime_error(
          "ApplicationDriver conservative phase test slice is out of range");
    }
    return std::vector<svmp::FE::Real>(
        solution.begin() + static_cast<std::ptrdiff_t>(offset),
        solution.begin() + static_cast<std::ptrdiff_t>(offset + count));
  }

  void refreshCurrentCandidate(const char* provenance)
  {
    (void)refreshActiveCutIntegrationContextCached(
        sim_,
        *params_,
        history().u(),
        lifecycle_,
        refresh_cache_,
        provenance);
  }

  std::shared_ptr<svmp::Mesh> mesh_{};
  svmp::FE::FieldId phi_{svmp::FE::INVALID_FIELD_ID};
  svmp::FE::FieldId phase_{svmp::FE::INVALID_FIELD_ID};
  application::core::SimulationComponents sim_{};
  std::unique_ptr<Parameters> params_{};
  std::vector<ActiveCutVolumeRequest> active_requests_{};
  std::vector<LevelSetMaintenanceRequest> requests_{};
  svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle_{};
  ActiveCutContextRefreshCache refresh_cache_{};
  std::vector<svmp::FE::Real> initialized_solution_{};
};

TEST_F(ApplicationDriverConservativePhaseCandidatesTest,
       InitializesEveryHistoryLevelAndOnlyItsRateSlices)
{
  const auto current = gatherFeOrderedSolution(history().u());
  const auto previous = gatherFeOrderedSolution(history().uPrev());
  const auto older = gatherFeOrderedSolution(history().uPrev2());
  const auto current_phase = fieldSlice(current, phase_);
  EXPECT_EQ(current_phase, fieldSlice(previous, phase_));
  EXPECT_EQ(current_phase, fieldSlice(older, phase_));
  for (const auto value : current_phase) {
    EXPECT_GE(value, svmp::FE::Real{0.0});
    EXPECT_LE(value, svmp::FE::Real{1.0});
  }

  const auto rate = gatherFeOrderedSolution(history().uDot());
  const auto acceleration = gatherFeOrderedSolution(history().uDDot());
  for (const auto value : fieldSlice(rate, phase_)) {
    EXPECT_DOUBLE_EQ(value, svmp::FE::Real{0.0});
  }
  for (const auto value : fieldSlice(acceleration, phase_)) {
    EXPECT_DOUBLE_EQ(value, svmp::FE::Real{0.0});
  }
  for (const auto value : fieldSlice(rate, phi_)) {
    EXPECT_DOUBLE_EQ(value, svmp::FE::Real{0.5});
  }
  for (const auto value : fieldSlice(acceleration, phi_)) {
    EXPECT_DOUBLE_EQ(value, svmp::FE::Real{0.5});
  }
}

TEST_F(ApplicationDriverConservativePhaseCandidatesTest,
       StagesAndCommitsTheTransportedPhaseAgainstAuthoritativeGeometry)
{
  auto raw_candidate = initialized_solution_;
  const auto phase_offset = fieldOffset(phase_);
  for (std::size_t i = 0u; i < fieldCount(phase_); ++i) {
    raw_candidate[phase_offset + i] =
        svmp::FE::Real{0.9} -
        svmp::FE::Real{0.05} * static_cast<svmp::FE::Real>(i);
  }
  scatterFeOrderedSolution(history().u(), raw_candidate);
  refreshCurrentCandidate(
      "application-driver-conservative-phase-commit-raw");
  const auto previous_phase = fieldSlice(
      gatherFeOrderedSolution(history().uPrev()), phase_);

  auto result = applyConservativePhaseCandidates(
      sim_,
      history(),
      requests_,
      *params_,
      lifecycle_,
      refresh_cache_,
      active_requests_);
  EXPECT_TRUE(result.accept_step);
  EXPECT_TRUE(result.changed);
  ASSERT_NE(result.geometry_transaction, nullptr);
  EXPECT_EQ(fieldSlice(gatherFeOrderedSolution(history().u()), phase_),
            previous_phase);
  EXPECT_EQ(fieldSlice(gatherFeOrderedSolution(history().uPrev()), phase_),
            previous_phase);
  ASSERT_NO_THROW(result.geometry_transaction->commit());
  result.geometry_transaction.reset();
  EXPECT_FALSE(sim_.fe_system->cutIntegrationContextTransactionActive());
  EXPECT_FALSE(lifecycle_.transactionActive());

  const auto projection = projectCurrentConservativePhaseGeometry(
      *sim_.fe_system, requests_.front());
  ASSERT_TRUE(projection.success) << projection.diagnostic;
  auto& graph = requireCurrentConservativePhaseGraph(
      *sim_.fe_system, requests_.front());
  svmp::FE::Real accepted_measure = svmp::FE::Real{0.0};
  const auto accepted_phase = fieldSlice(
      gatherFeOrderedSolution(history().u()), phase_);
  for (std::size_t i = 0u; i < graph.nodes; ++i) {
    accepted_measure += graph.lumped_control_volume[i] * accepted_phase[i];
  }
  EXPECT_NEAR(projection.retained_liquid_measure,
              accepted_measure,
              1.0e-10);
}

TEST_F(ApplicationDriverConservativePhaseCandidatesTest,
       LaterRejectionRestoresTheRawCandidateAndEveryGeometryRevision)
{
  auto raw_candidate = initialized_solution_;
  raw_candidate[fieldOffset(phi_)] += svmp::FE::Real{0.01};
  const auto phase_offset = fieldOffset(phase_);
  for (std::size_t i = 0u; i < fieldCount(phase_); ++i) {
    raw_candidate[phase_offset + i] = svmp::FE::Real{0.8};
  }
  scatterFeOrderedSolution(history().u(), raw_candidate);
  refreshCurrentCandidate(
      "application-driver-conservative-phase-rollback-raw");
  const auto* raw_context = sim_.fe_system->cutIntegrationContext();
  ASSERT_NE(raw_context, nullptr);
  const auto lifecycle_revision = lifecycle_.valueRevision();
  const auto constraint_revision =
      sim_.fe_system->constraintLayoutRevision();
  const auto sparsity_revision = sim_.fe_system->sparsityPatternRevision();
  const auto cache_before = refresh_cache_;

  auto result = applyConservativePhaseCandidates(
      sim_,
      history(),
      requests_,
      *params_,
      lifecycle_,
      refresh_cache_,
      active_requests_);
  EXPECT_TRUE(result.accept_step);
  EXPECT_TRUE(result.changed);
  ASSERT_NE(result.geometry_transaction, nullptr);
  EXPECT_TRUE(sim_.fe_system->cutIntegrationContextTransactionActive());
  EXPECT_TRUE(lifecycle_.transactionActive());
  EXPECT_NE(sim_.fe_system->cutIntegrationContext(), raw_context);

  ASSERT_NO_THROW(rollbackConservativePhaseCandidate(history(), result));
  EXPECT_EQ(gatherFeOrderedSolution(history().u()), raw_candidate);
  EXPECT_EQ(sim_.fe_system->cutIntegrationContext(), raw_context);
  EXPECT_EQ(lifecycle_.valueRevision(), lifecycle_revision);
  EXPECT_EQ(sim_.fe_system->constraintLayoutRevision(),
            constraint_revision);
  EXPECT_EQ(sim_.fe_system->sparsityPatternRevision(),
            sparsity_revision);
  EXPECT_FALSE(sim_.fe_system->cutIntegrationContextTransactionActive());
  EXPECT_FALSE(lifecycle_.transactionActive());
  ASSERT_EQ(refresh_cache_.last_signature.has_value(),
            cache_before.last_signature.has_value());
  if (refresh_cache_.last_signature.has_value()) {
    EXPECT_TRUE(*refresh_cache_.last_signature ==
                *cache_before.last_signature);
  }
  ASSERT_EQ(refresh_cache_.last_vector_signature.has_value(),
            cache_before.last_vector_signature.has_value());
  if (refresh_cache_.last_vector_signature.has_value()) {
    EXPECT_TRUE(*refresh_cache_.last_vector_signature ==
                *cache_before.last_vector_signature);
  }
}

TEST_F(ApplicationDriverConservativePhaseCandidatesTest,
       CourantRejectionLeavesTheRawCandidateAndGeometryUntouched)
{
  history().setDt(1.0);
  auto raw_candidate = initialized_solution_;
  const auto phase_offset = fieldOffset(phase_);
  for (std::size_t i = 0u; i < fieldCount(phase_); ++i) {
    raw_candidate[phase_offset + i] = svmp::FE::Real{0.7};
  }
  scatterFeOrderedSolution(history().u(), raw_candidate);
  refreshCurrentCandidate(
      "application-driver-conservative-phase-courant-raw");
  const auto* raw_context = sim_.fe_system->cutIntegrationContext();
  requests_.front().velocity.constant_value = {
      svmp::FE::Real{10.0}, svmp::FE::Real{0.0}, svmp::FE::Real{0.0}};
  requests_.front().conservative_phase
      .impermeable_normal_velocity_tolerance = svmp::FE::Real{2.0};

  const auto result = applyConservativePhaseCandidates(
      sim_,
      history(),
      requests_,
      *params_,
      lifecycle_,
      refresh_cache_,
      active_requests_);
  EXPECT_FALSE(result.accept_step);
  EXPECT_FALSE(result.changed);
  EXPECT_EQ(result.geometry_transaction, nullptr);
  EXPECT_EQ(gatherFeOrderedSolution(history().u()), raw_candidate);
  EXPECT_EQ(sim_.fe_system->cutIntegrationContext(), raw_context);
  EXPECT_FALSE(sim_.fe_system->cutIntegrationContextTransactionActive());
  EXPECT_FALSE(lifecycle_.transactionActive());
}

class ApplicationDriverBoundPreservingCandidatesTest
    : public ::testing::Test {
protected:
  void SetUp() override
  {
    mesh_ = makeWorkflowQuadPatch2x2Mesh();
    auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
        svmp::FE::ElementType::Quad4,
        /*order=*/1);
    auto system =
        std::make_unique<svmp::FE::systems::FESystem>(mesh_);
    phi_ = system->addField(svmp::FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(system->setup({}));

    factory_ = svmp::FE::backends::BackendFactory::create(
        svmp::FE::backends::BackendKind::FSILS);
    ASSERT_NE(factory_, nullptr);
    history_ = svmp::FE::timestepping::TimeHistory::allocate(
        *factory_,
        system->dofHandler().getNumDofs(),
        /*history_depth=*/2,
        /*allocate_second_order_state=*/true);
    history_.setDt(0.2);
    history_.setPrevDt(0.2);

    sim_.primary_mesh = mesh_;
    sim_.fe_system = std::move(system);
  }

  [[nodiscard]] std::size_t solutionSize() const
  {
    return static_cast<std::size_t>(
        sim_.fe_system->dofHandler().getNumDofs());
  }

  [[nodiscard]] std::size_t phiOffset() const
  {
    const auto offset = sim_.fe_system->fieldDofOffset(phi_);
    if (offset < 0) {
      throw std::runtime_error(
          "ApplicationDriver bound-preserving test has no phi offset");
    }
    return static_cast<std::size_t>(offset);
  }

  [[nodiscard]] LevelSetMaintenanceRequest requestWithVelocity(
      std::array<svmp::FE::Real, 3> velocity) const
  {
    LevelSetMaintenanceRequest request{};
    request.level_set_field_name = "phi";
    request.velocity.source =
        svmp::FE::level_set::LevelSetVelocitySource::ConstantVector;
    request.velocity.constant_value = velocity;
    request.bound_preserving.enabled = true;
    return request;
  }

  void setCandidateState(std::span<const svmp::FE::Real> previous,
                         std::span<const svmp::FE::Real> candidate,
                         std::span<const svmp::FE::Real> rates)
  {
    ASSERT_EQ(previous.size(), solutionSize());
    ASSERT_EQ(candidate.size(), solutionSize());
    ASSERT_EQ(rates.size(), solutionSize());
    scatterFeOrderedSolution(history_.uPrev(), previous);
    scatterFeOrderedSolution(history_.uPrev2(), previous);
    scatterFeOrderedSolution(history_.u(), candidate);
    scatterFeOrderedSolution(history_.uDot(), rates);
  }

  std::shared_ptr<svmp::Mesh> mesh_{};
  svmp::FE::FieldId phi_{svmp::FE::INVALID_FIELD_ID};
  std::unique_ptr<svmp::FE::backends::BackendFactory> factory_{};
  svmp::FE::timestepping::TimeHistory history_{};
  application::core::SimulationComponents sim_{};
};

TEST_F(ApplicationDriverBoundPreservingCandidatesTest,
       LimitedCandidateRequestsNonlinearRetryWithoutMutatingHistory)
{
  std::vector<svmp::FE::Real> previous(solutionSize(), 1.0);
  auto raw_candidate = previous;
  const auto limited_dof = phiOffset();
  raw_candidate[limited_dof] = -2.0;
  std::vector<svmp::FE::Real> rates(solutionSize(), 0.0);
  for (std::size_t i = 0; i < rates.size(); ++i) {
    rates[i] = 0.25 + 0.1 * static_cast<svmp::FE::Real>(i);
  }
  setCandidateState(previous, raw_candidate, rates);

  testing::internal::CaptureStdout();
  const auto result = applyLevelSetBoundPreservingCandidates(
      sim_,
      history_,
      std::vector<LevelSetMaintenanceRequest>{
          requestWithVelocity({0.0, 0.0, 0.0})},
      /*generalized_alpha_gamma=*/0.5);
  const auto output = testing::internal::GetCapturedStdout();

  EXPECT_FALSE(result.accept_step);
  EXPECT_TRUE(result.changed);
  EXPECT_NE(output.find(
                "reason=bound_preserving_limiter_requires_nonlinear_retry"),
            std::string::npos);
  EXPECT_EQ(gatherFeOrderedSolution(history_.u()), raw_candidate);
  EXPECT_EQ(gatherFeOrderedSolution(history_.uDot()), rates);
  EXPECT_EQ(gatherFeOrderedSolution(history_.uPrev()), previous);
}

TEST_F(ApplicationDriverBoundPreservingCandidatesTest,
       CourantViolationReturnsRetryableRejectionWithoutMutatingCandidate)
{
  history_.setDt(0.75);
  std::vector<svmp::FE::Real> previous(solutionSize(), 1.0);
  const auto raw_candidate = previous;
  std::vector<svmp::FE::Real> rates(solutionSize(), -0.25);
  setCandidateState(previous, raw_candidate, rates);

  auto request = requestWithVelocity({2.0, 0.0, 0.0});
  request.bound_preserving.bound_tolerance = 10.0;
  request.bound_preserving.courant_tolerance = 1.0e-12;
  request.bound_preserving.enforce_impermeable_boundaries = false;
  testing::internal::CaptureStdout();
  const auto result = applyLevelSetBoundPreservingCandidates(
      sim_,
      history_,
      std::vector<LevelSetMaintenanceRequest>{request},
      /*generalized_alpha_gamma=*/0.5);
  const auto output = testing::internal::GetCapturedStdout();

  EXPECT_FALSE(result.accept_step);
  EXPECT_FALSE(result.changed);
  EXPECT_NE(output.find("reason=bound_preserving_courant_contract"),
            std::string::npos);
  EXPECT_EQ(gatherFeOrderedSolution(history_.u()), raw_candidate);
  EXPECT_EQ(gatherFeOrderedSolution(history_.uDot()), rates);
  EXPECT_EQ(gatherFeOrderedSolution(history_.uPrev()), previous);
}

TEST_F(ApplicationDriverBoundPreservingCandidatesTest,
       ImpermeableWallNormalVelocityFailsClosedWithoutMutatingCandidate)
{
  history_.setDt(0.1);
  std::vector<svmp::FE::Real> previous(solutionSize(), 1.0);
  const auto raw_candidate = previous;
  std::vector<svmp::FE::Real> rates(solutionSize(), 0.125);
  setCandidateState(previous, raw_candidate, rates);

  try {
    (void)applyLevelSetBoundPreservingCandidates(
        sim_,
        history_,
        std::vector<LevelSetMaintenanceRequest>{
            requestWithVelocity({0.0, 0.5, 0.0})},
        /*generalized_alpha_gamma=*/0.5);
    FAIL() << "A nonzero normal wall velocity must fail closed";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find(
                  "incompatible impermeable-wall velocity"),
              std::string::npos);
  }

  EXPECT_EQ(gatherFeOrderedSolution(history_.u()), raw_candidate);
  EXPECT_EQ(gatherFeOrderedSolution(history_.uDot()), rates);
  EXPECT_EQ(gatherFeOrderedSolution(history_.uPrev()), previous);
}

TEST_F(ApplicationDriverBoundPreservingCandidatesTest,
       InBoundsNontrivialCandidatePassesWithoutLimiterOrRateChanges)
{
  std::vector<svmp::FE::Real> vertex_values(mesh_->n_vertices(), 0.0);
  for (std::size_t vertex = 0; vertex < mesh_->n_vertices(); ++vertex) {
    vertex_values[vertex] = workflowVertexPoint(*mesh_, vertex)[0];
  }
  const auto phi_coefficients = projectWorkflowVertexValues(
      *sim_.fe_system,
      phi_,
      vertex_values,
      /*components=*/1u,
      "ApplicationDriver bound-preserving pass-through phi");
  std::vector<svmp::FE::Real> previous(solutionSize(), 0.0);
  writeWorkflowFieldSlice(
      *sim_.fe_system, phi_, phi_coefficients, previous);
  auto raw_candidate = previous;
  for (std::size_t i = 0; i < phi_coefficients.size(); ++i) {
    raw_candidate[phiOffset() + i] *= 0.5;
  }
  ASSERT_NE(raw_candidate, previous);
  std::vector<svmp::FE::Real> rates(solutionSize(), 0.375);
  setCandidateState(previous, raw_candidate, rates);

  const auto result = applyLevelSetBoundPreservingCandidates(
      sim_,
      history_,
      std::vector<LevelSetMaintenanceRequest>{
          requestWithVelocity({0.0, 0.0, 0.0})},
      /*generalized_alpha_gamma=*/0.5);

  EXPECT_TRUE(result.accept_step);
  EXPECT_FALSE(result.changed);
  EXPECT_EQ(gatherFeOrderedSolution(history_.u()), raw_candidate);
  EXPECT_EQ(gatherFeOrderedSolution(history_.uDot()), rates);
  EXPECT_EQ(gatherFeOrderedSolution(history_.uPrev()), previous);
}

void addValidComponentTransferLedger(
    svmp::FE::level_set::LevelSetGlobalShiftCorrectionResult& result)
{
  result.negative_component_topology_preserved = true;
  result.negative_component_volume_transfers.push_back(
      svmp::FE::level_set::LevelSetComponentVolumeTransfer{
          .component_global_vertex_id = 0,
      });
}

TEST(ApplicationDriverLevelSetVolumeCorrection,
     ReportsAuthoritativeFreeSurfacePotentialChange)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeWorkflowTriangleMesh();
  auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Triangle3,
      /*order=*/1);
  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto phi = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi",
      .space = scalar_space,
      .components = 1,
  });
  svmp::FE::interfaces::FreeSurfaceDiscreteFunctionalParameters parameters;
  parameters.surface_tension = 1.0;
  parameters.volume_multiplier = 0.5;
  system->declareFreeSurfaceDiscreteFunctional(
      svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration{
          .interface_marker = 808,
          .level_set_field = phi,
          .geometry_domain_id = "volume_work_fixture",
          .parameters = parameters,
          .owner_component =
              "ApplicationDriverLevelSetVolumeCorrection.WorkFixture",
      });
  ASSERT_NO_THROW(system->setup({}));
  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);

  const auto make_state = [](std::uint64_t source_revision,
                             std::uint64_t snapshot_revision,
                             svmp::FE::Real liquid_volume,
                             svmp::FE::Real liquid_gas_area,
                             svmp::FE::Real wetted_wall_area,
                             svmp::FE::Real contact_measure,
                             svmp::FE::Real surface_energy,
                             svmp::FE::Real wall_energy,
                             svmp::FE::Real volume_potential,
                             svmp::FE::Real total_potential) {
    svmp::FE::systems::AcceptedFreeSurfaceDiscreteFunctionalState state;
    state.interface_marker = 808;
    state.geometry_revision.source_id = "field:0";
    state.geometry_revision.domain_id = "volume_work_fixture";
    state.geometry_revision.interface_marker = 808;
    state.geometry_revision.source_value_revision = source_revision;
    state.geometry_revision.snapshot_revision_key = snapshot_revision;
    state.state.snapshot_revision_key = snapshot_revision;
    state.state.surface_tension = 1.0;
    state.state.volume_multiplier = 0.5;
    state.state.owned_liquid_volume = liquid_volume;
    state.state.owned_liquid_gas_area = liquid_gas_area;
    state.state.owned_wetted_wall_area = wetted_wall_area;
    state.state.owned_contact_measure = contact_measure;
    state.state.liquid_gas_surface_energy = surface_energy;
    state.state.young_wall_energy = wall_energy;
    state.state.volume_constraint_potential = volume_potential;
    state.state.total_potential = total_potential;
    return state;
  };
  const std::vector<
      svmp::FE::systems::AcceptedFreeSurfaceDiscreteFunctionalState>
      before{make_state(
          1u, 101u, 0.5, 2.0, 0.5, 0.25, 2.0, -0.5, 0.25, 1.75)};
  const std::vector<
      svmp::FE::systems::AcceptedFreeSurfaceDiscreteFunctionalState>
      after{make_state(
          2u, 102u, 0.375, 2.25, 0.75, 0.125, 2.25, -0.375,
          0.125, 2.0)};
  LevelSetVolumeCorrectionMaintenanceEvent event;
  event.level_set_field = phi;
  event.level_set_field_name = "phi";
  event.completed_step = 4;
  event.correction.correction_applied = true;
  event.correction.applied_shift = 0.125;
  event.correction.total_component_volume_transfer = -0.125;
  event.correction.negative_component_volume_transfers.push_back(
      svmp::FE::level_set::LevelSetComponentVolumeTransfer{
          .component_global_vertex_id = 0,
          .initial_negative_volume = 0.5,
          .corrected_negative_volume = 0.375,
          .volume_transfer = -0.125,
      });
  const std::vector<LevelSetVolumeCorrectionMaintenanceEvent> events{event};

  testing::internal::CaptureStdout();
  ASSERT_NO_THROW(logLevelSetVolumeCorrectionFreeSurfaceWork(
      sim, events, before, after));
  const auto output = testing::internal::GetCapturedStdout();
  EXPECT_NE(output.find("diagnostic=level_set_volume_correction_work"),
            std::string::npos);
  EXPECT_NE(output.find("scope=global_shift_only"), std::string::npos);
  EXPECT_NE(output.find("numerical_work_sign=energy_after_minus_before"),
            std::string::npos);
  EXPECT_NE(output.find("free_surface_functional_count=1"),
            std::string::npos);
  EXPECT_NE(output.find("initial_snapshot_revision=101"),
            std::string::npos);
  EXPECT_NE(output.find("corrected_snapshot_revision=102"),
            std::string::npos);
  EXPECT_NE(output.find("surface_energy_change="), std::string::npos);
  EXPECT_NE(output.find("young_wall_energy_change="), std::string::npos);
  EXPECT_NE(output.find("volume_constraint_potential_change="),
            std::string::npos);
  EXPECT_NE(output.find("liquid_volume_change=-0.125"),
            std::string::npos);
  EXPECT_NE(output.find("surface_energy_change=0.25"),
            std::string::npos);
  EXPECT_NE(output.find("young_wall_energy_change=0.125"),
            std::string::npos);
  EXPECT_NE(output.find("volume_constraint_potential_change=-0.125"),
            std::string::npos);
  EXPECT_NE(output.find("numerical_free_surface_work=0.25"),
            std::string::npos);
#endif
}

TEST(ApplicationDriverLevelSetVolumeCorrection,
     CandidateGeometryFailureRestoresCompleteMaintenanceTransaction)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  constexpr int interface_marker = 809;
  auto mesh = makeWorkflowTriangleMesh();
  (void)svmp::MeshFields::attach_field(
      mesh->local_mesh(),
      svmp::EntityKind::Vertex,
      "phi",
      svmp::FieldScalarType::Float64,
      1);
  auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Triangle3,
      /*order=*/1);
  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto phi = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi",
      .space = scalar_space,
      .components = 1,
  });
  ASSERT_NO_THROW(system->setup({}));

  std::vector<svmp::FE::Real> phi_vertex_values(mesh->n_vertices(), 0.0);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    phi_vertex_values[vertex] =
        workflowVertexPoint(*mesh, vertex)[0] - svmp::FE::Real{0.5};
  }
  const auto coefficients = projectWorkflowVertexValues(
      *system,
      phi,
      phi_vertex_values,
      /*components=*/1u,
      "ApplicationDriver maintenance transaction phi");
  std::vector<svmp::FE::Real> initial(
      static_cast<std::size_t>(system->dofHandler().getNumDofs()), 0.0);
  writeWorkflowFieldSlice(*system, phi, coefficients, initial);

  std::unique_ptr<svmp::FE::backends::BackendFactory> factory;
  try {
    factory = svmp::FE::backends::BackendFactory::create(
        svmp::FE::backends::BackendKind::FSILS);
  } catch (const std::exception&) {
    GTEST_SKIP() << "Requires an available FE vector backend.";
  }
  auto history = svmp::FE::timestepping::TimeHistory::allocate(
      *factory,
      system->dofHandler().getNumDofs(),
      /*history_depth=*/2,
      /*allocate_second_order_state=*/true);
  history.setStepIndex(1);
  history.setDt(0.1);
  history.setPrevDt(0.1);
  scatterFeOrderedSolution(history.u(), initial);
  scatterFeOrderedSolution(history.uPrev(), initial);
  scatterFeOrderedSolution(history.uPrev2(), initial);
  std::vector<svmp::FE::Real> rates(initial.size(), 0.375);
  scatterFeOrderedSolution(history.uDot(), rates);

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
      <Generated_interface_domain_id>transaction_interface</Generated_interface_domain_id>
      <Interface_marker>809</Interface_marker>
      <Allow_corner_linearized_cut_geometry>true</Allow_corner_linearized_cut_geometry>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
)xml");
  const auto active_requests = activeCutVolumeRequests(*params);
  ASSERT_EQ(active_requests.size(), 1u);
  svmp::FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
  ActiveCutContextRefreshCache refresh_cache;
  ASSERT_TRUE(refreshActiveCutIntegrationContextFromSolutionCached(
                  sim,
                  *params,
                  initial,
                  lifecycle,
                  refresh_cache,
                  "application-driver-maintenance-transaction-initial")
                  .refreshed);

  const auto* original_context = sim.fe_system->cutIntegrationContext();
  ASSERT_NE(original_context, nullptr);
  ASSERT_TRUE(original_context->hasGeneratedInterfaceMarker(interface_marker));
  const auto lifecycle_revision_before = lifecycle.valueRevision();
  const auto constraint_revision_before =
      sim.fe_system->constraintLayoutRevision();
  const auto sparsity_revision_before =
      sim.fe_system->sparsityPatternRevision();
  const auto constraint_count_before =
      sim.fe_system->constraints().numConstraints();
  const auto mesh_revisions_before = mesh->event_bus().revision_state();
  const auto refresh_cache_before = refresh_cache;
  const auto mesh_phi_handle = mesh->field_handle(
      svmp::EntityKind::Vertex, "phi");
  const auto mesh_phi_count =
      mesh->field_components(mesh_phi_handle) *
      mesh->field_entity_count(mesh_phi_handle);
  const auto* mesh_phi_data_before =
      static_cast<const double*>(mesh->field_data(mesh_phi_handle));
  ASSERT_NE(mesh_phi_data_before, nullptr);
  const std::vector<double> mesh_phi_before(
      mesh_phi_data_before, mesh_phi_data_before + mesh_phi_count);

  const auto current_revision_before = history.u().valueRevision();
  const auto previous_revision_before = history.uPrev().valueRevision();
  const auto previous2_revision_before = history.uPrev2().valueRevision();
  const auto rate_revision_before = history.uDot().valueRevision();
  const auto current_before = gatherFeOrderedSolution(history.u());
  const auto previous_before = gatherFeOrderedSolution(history.uPrev());
  const auto previous2_before = gatherFeOrderedSolution(history.uPrev2());
  const auto rates_before = gatherFeOrderedSolution(history.uDot());

  LevelSetMaintenanceRequest request{};
  request.level_set_field_name = "phi";
  request.volume_correction.enabled = true;
  request.volume_correction.cadence_steps = 1;
  request.volume_correction.use_initial_negative_volume_as_target = false;
  request.volume_correction.target_negative_volume = 0.36;
  request.volume_correction.minimum_relative_volume_error = 0.0;
  request.volume_correction.maximum_interface_displacement_fraction = 0.10;
  request.volume_correction
      .maximum_cumulative_interface_displacement_fraction = 1.0;
  std::vector<LevelSetMaintenanceRequest> requests{request};
  std::vector<LevelSetVolumeCorrectionMaintenanceEvent> published_events;
  std::unique_ptr<LevelSetMaintenanceGeometryTransaction>
      geometry_transaction;
  bool candidate_context_replaced = false;
  const LevelSetMaintenanceCandidateValidator reject_candidate =
      [&](std::span<const svmp::FE::Real> candidate,
          std::span<const LevelSetVolumeCorrectionMaintenanceEvent> events) {
        if (events.size() != 1u) {
          throw std::runtime_error(
              "injected maintenance candidate had incomplete event coverage");
        }
        geometry_transaction =
            std::make_unique<LevelSetMaintenanceGeometryTransaction>(
                sim, lifecycle, refresh_cache, active_requests);
        const auto report = geometry_transaction->refresh(*params, candidate);
        candidate_context_replaced =
            report.refreshed &&
            sim.fe_system->cutIntegrationContext() != original_context;
        throw std::runtime_error(
            "injected post-refresh maintenance validation failure");
      };

  testing::internal::CaptureStdout();
  EXPECT_THROW(
      (void)applyLevelSetMaintenance(
          sim,
          history,
          requests,
          {},
          {},
          {},
          &published_events,
          reject_candidate),
      std::runtime_error);
  const auto output = testing::internal::GetCapturedStdout();
  ASSERT_NE(geometry_transaction, nullptr);
  ASSERT_NO_THROW(geometry_transaction->rollback());
  EXPECT_TRUE(candidate_context_replaced);
  EXPECT_EQ(output.find("Level-set volume corrected"), std::string::npos);
  EXPECT_TRUE(published_events.empty());
  ASSERT_EQ(requests.size(), 1u);
  EXPECT_FALSE(requests.front().volume_target_initialized);
  EXPECT_DOUBLE_EQ(
      requests.front().cumulative_volume_correction_interface_displacement,
      0.0);
  EXPECT_DOUBLE_EQ(
      requests.front().cumulative_volume_correction_contact_line_displacement,
      0.0);

  EXPECT_EQ(gatherFeOrderedSolution(history.u()), current_before);
  EXPECT_EQ(gatherFeOrderedSolution(history.uPrev()), previous_before);
  EXPECT_EQ(gatherFeOrderedSolution(history.uPrev2()), previous2_before);
  EXPECT_EQ(gatherFeOrderedSolution(history.uDot()), rates_before);
  EXPECT_EQ(history.u().valueRevision(), current_revision_before);
  EXPECT_EQ(history.uPrev().valueRevision(), previous_revision_before);
  EXPECT_EQ(history.uPrev2().valueRevision(), previous2_revision_before);
  EXPECT_EQ(history.uDot().valueRevision(), rate_revision_before);

  EXPECT_EQ(sim.fe_system->cutIntegrationContext(), original_context);
  EXPECT_FALSE(sim.fe_system->cutIntegrationContextTransactionActive());
  EXPECT_FALSE(lifecycle.transactionActive());
  EXPECT_EQ(lifecycle.valueRevision(), lifecycle_revision_before);
  EXPECT_EQ(sim.fe_system->constraintLayoutRevision(),
            constraint_revision_before);
  EXPECT_EQ(sim.fe_system->sparsityPatternRevision(),
            sparsity_revision_before);
  EXPECT_EQ(sim.fe_system->constraints().numConstraints(),
            constraint_count_before);
  ASSERT_EQ(refresh_cache.last_signature.has_value(),
            refresh_cache_before.last_signature.has_value());
  if (refresh_cache.last_signature.has_value()) {
    EXPECT_TRUE(*refresh_cache.last_signature ==
                *refresh_cache_before.last_signature);
  }
  ASSERT_EQ(refresh_cache.last_vector_signature.has_value(),
            refresh_cache_before.last_vector_signature.has_value());
  if (refresh_cache.last_vector_signature.has_value()) {
    EXPECT_TRUE(*refresh_cache.last_vector_signature ==
                *refresh_cache_before.last_vector_signature);
  }

  const auto mesh_revisions_after = mesh->event_bus().revision_state();
  EXPECT_EQ(mesh_revisions_after.geometry, mesh_revisions_before.geometry);
  EXPECT_EQ(mesh_revisions_after.reference_geometry,
            mesh_revisions_before.reference_geometry);
  EXPECT_EQ(mesh_revisions_after.current_geometry,
            mesh_revisions_before.current_geometry);
  EXPECT_EQ(mesh_revisions_after.reference_rebase,
            mesh_revisions_before.reference_rebase);
  EXPECT_EQ(mesh_revisions_after.topology, mesh_revisions_before.topology);
  EXPECT_EQ(mesh_revisions_after.ownership, mesh_revisions_before.ownership);
  EXPECT_EQ(mesh_revisions_after.numbering, mesh_revisions_before.numbering);
  EXPECT_EQ(mesh_revisions_after.field_layout,
            mesh_revisions_before.field_layout);
  EXPECT_EQ(mesh_revisions_after.labels, mesh_revisions_before.labels);
  EXPECT_EQ(mesh_revisions_after.active_configuration,
            mesh_revisions_before.active_configuration);
  const auto* mesh_phi_data_after =
      static_cast<const double*>(mesh->field_data(mesh_phi_handle));
  ASSERT_NE(mesh_phi_data_after, nullptr);
  EXPECT_EQ(std::vector<double>(
                mesh_phi_data_after, mesh_phi_data_after + mesh_phi_count),
            mesh_phi_before);

  const auto cached_report =
      refreshActiveCutIntegrationContextFromSolutionCached(
          sim,
          *params,
          initial,
          lifecycle,
          refresh_cache,
          "application-driver-maintenance-transaction-restored");
  EXPECT_FALSE(cached_report.refreshed);
  EXPECT_EQ(sim.fe_system->cutIntegrationContext(), original_context);
  EXPECT_EQ(lifecycle.valueRevision(), lifecycle_revision_before);

  std::vector<svmp::FE::Real> committed_candidate;
  bool committed_candidate_refreshed = false;
  const LevelSetMaintenanceCandidateValidator accept_candidate =
      [&](std::span<const svmp::FE::Real> candidate,
          std::span<const LevelSetVolumeCorrectionMaintenanceEvent> events) {
        if (events.size() != 1u) {
          throw std::runtime_error(
              "accepted maintenance candidate had incomplete event coverage");
        }
        committed_candidate.assign(candidate.begin(), candidate.end());
        geometry_transaction =
            std::make_unique<LevelSetMaintenanceGeometryTransaction>(
                sim, lifecycle, refresh_cache, active_requests);
        const auto report = geometry_transaction->refresh(*params, candidate);
        committed_candidate_refreshed =
            report.refreshed &&
            sim.fe_system->cutIntegrationContext() != original_context;
      };
  ASSERT_TRUE(applyLevelSetMaintenance(
      sim,
      history,
      requests,
      {},
      {},
      {},
      &published_events,
      accept_candidate));
  ASSERT_NE(geometry_transaction, nullptr);
  ASSERT_NO_THROW(geometry_transaction->commit());
  EXPECT_TRUE(committed_candidate_refreshed);
  EXPECT_FALSE(sim.fe_system->cutIntegrationContextTransactionActive());
  EXPECT_FALSE(lifecycle.transactionActive());
  EXPECT_NE(sim.fe_system->cutIntegrationContext(), original_context);
  ASSERT_EQ(published_events.size(), 1u);
  ASSERT_EQ(requests.size(), 1u);
  EXPECT_TRUE(requests.front().volume_target_initialized);
  EXPECT_EQ(gatherFeOrderedSolution(history.u()), committed_candidate);
  EXPECT_EQ(gatherFeOrderedSolution(history.uPrev()), committed_candidate);
  EXPECT_EQ(gatherFeOrderedSolution(history.uPrev2()), committed_candidate);
  EXPECT_EQ(gatherFeOrderedSolution(history.uDot()), rates_before);
  EXPECT_GT(history.uDot().valueRevision(), rate_revision_before);
  const auto committed_cached_report =
      refreshActiveCutIntegrationContextFromSolutionCached(
          sim,
          *params,
          committed_candidate,
          lifecycle,
          refresh_cache,
          "application-driver-maintenance-transaction-committed");
  EXPECT_FALSE(committed_cached_report.refreshed);
#endif
}

TEST(ApplicationDriverLevelSetVolumeCorrection,
     CumulativeDisplacementBudgetRejectsBeforeAccountingExcessEvent)
{
  LevelSetMaintenanceRequest request{};
  request.level_set_field_name = "phi";
  request.volume_correction
      .maximum_cumulative_interface_displacement_fraction = 0.10;

  svmp::FE::level_set::LevelSetGlobalShiftCorrectionResult result{};
  result.correction_applied = true;
  result.minimum_edge_length = 1.0;
  result.max_interface_displacement = 0.04;
  result.max_contact_line_displacement = 0.03;
  addValidComponentTransferLedger(result);

  ASSERT_NO_THROW(accountAppliedLevelSetVolumeCorrection(request, result));
  ASSERT_NO_THROW(accountAppliedLevelSetVolumeCorrection(request, result));
  EXPECT_DOUBLE_EQ(
      request.cumulative_volume_correction_interface_displacement, 0.08);
  EXPECT_DOUBLE_EQ(
      request.cumulative_volume_correction_contact_line_displacement, 0.06);
  EXPECT_DOUBLE_EQ(
      request.volume_correction_reference_minimum_edge_length, 1.0);

  result.max_interface_displacement = 0.03;
  result.max_contact_line_displacement = 0.02;
  EXPECT_THROW(
      accountAppliedLevelSetVolumeCorrection(request, result),
      std::runtime_error);
  EXPECT_DOUBLE_EQ(
      request.cumulative_volume_correction_interface_displacement, 0.08);
  EXPECT_DOUBLE_EQ(
      request.cumulative_volume_correction_contact_line_displacement, 0.06);
}

TEST(ApplicationDriverLevelSetVolumeCorrection,
     CumulativeBudgetUsesSmallestObservedEdgeAndIgnoresSkippedEvents)
{
  LevelSetMaintenanceRequest request{};
  request.level_set_field_name = "phi";
  request.volume_correction
      .maximum_cumulative_interface_displacement_fraction = 0.10;

  svmp::FE::level_set::LevelSetGlobalShiftCorrectionResult skipped{};
  skipped.correction_applied = false;
  accountAppliedLevelSetVolumeCorrection(request, skipped);
  EXPECT_DOUBLE_EQ(
      request.cumulative_volume_correction_interface_displacement, 0.0);

  auto applied = skipped;
  applied.correction_applied = true;
  applied.minimum_edge_length = 1.0;
  applied.max_interface_displacement = 0.04;
  applied.max_contact_line_displacement = 0.02;
  addValidComponentTransferLedger(applied);
  ASSERT_NO_THROW(accountAppliedLevelSetVolumeCorrection(request, applied));

  applied.minimum_edge_length = 0.5;
  applied.max_interface_displacement = 0.02;
  applied.max_contact_line_displacement = 0.01;
  EXPECT_THROW(
      accountAppliedLevelSetVolumeCorrection(request, applied),
      std::runtime_error);
  EXPECT_DOUBLE_EQ(
      request.cumulative_volume_correction_interface_displacement, 0.04);
  EXPECT_DOUBLE_EQ(
      request.volume_correction_reference_minimum_edge_length, 1.0);
}

TEST(ApplicationDriverLevelSetVolumeCorrection,
     CumulativeContactLineBudgetRejectsBeforeMutatingAccountingState)
{
  LevelSetMaintenanceRequest request{};
  request.level_set_field_name = "phi";
  request.volume_correction
      .maximum_cumulative_interface_displacement_fraction = 0.10;
  request.volume_correction_reference_minimum_edge_length = 1.0;
  request.cumulative_volume_correction_interface_displacement = 0.02;
  request.cumulative_volume_correction_contact_line_displacement = 0.08;
  request.volume_target_initialized = true;
  request.volume_target = 0.375;

  svmp::FE::level_set::LevelSetGlobalShiftCorrectionResult result{};
  result.correction_applied = true;
  result.minimum_edge_length = 1.0;
  result.max_interface_displacement = 0.01;
  result.max_contact_line_displacement = 0.03;
  addValidComponentTransferLedger(result);

  const auto reference_edge_before =
      request.volume_correction_reference_minimum_edge_length;
  const auto interface_history_before =
      request.cumulative_volume_correction_interface_displacement;
  const auto contact_line_history_before =
      request.cumulative_volume_correction_contact_line_displacement;
  const auto target_initialized_before = request.volume_target_initialized;
  const auto target_before = request.volume_target;

  try {
    accountAppliedLevelSetVolumeCorrection(request, result);
    FAIL() << "Expected the contact-line cumulative path to exceed the budget";
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("limiting_path=contact_line"), std::string::npos);
    EXPECT_NE(message.find("prospective_interface="), std::string::npos);
    EXPECT_NE(message.find("prospective_contact_line="), std::string::npos);
  }

  EXPECT_DOUBLE_EQ(
      request.volume_correction_reference_minimum_edge_length,
      reference_edge_before);
  EXPECT_DOUBLE_EQ(
      request.cumulative_volume_correction_interface_displacement,
      interface_history_before);
  EXPECT_DOUBLE_EQ(
      request.cumulative_volume_correction_contact_line_displacement,
      contact_line_history_before);
  EXPECT_EQ(request.volume_target_initialized, target_initialized_before);
  EXPECT_DOUBLE_EQ(request.volume_target, target_before);
}

TEST(ApplicationDriverLevelSetVolumeCorrection,
     LaterContactLineOnlyBudgetRejectionRollsBackEarlierRequestAndHistory)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeWorkflowTriangleMesh();
  auto scalar_space = std::make_shared<svmp::FE::spaces::H1Space>(
      svmp::FE::ElementType::Triangle3,
      /*order=*/1);
  auto system = std::make_unique<svmp::FE::systems::FESystem>(mesh);
  const auto phi = system->addField(svmp::FE::systems::FieldSpec{
      .name = "phi",
      .space = scalar_space,
      .components = 1,
  });
  ASSERT_NO_THROW(system->setup({}));

  std::vector<svmp::FE::Real> phi_vertex_values(mesh->n_vertices(), 0.0);
  for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
    phi_vertex_values[vertex] =
        workflowVertexPoint(*mesh, vertex)[0] - svmp::FE::Real{0.5};
  }
  const auto coefficients = projectWorkflowVertexValues(
      *system,
      phi,
      phi_vertex_values,
      /*components=*/1u,
      "ApplicationDriver cumulative contact-line budget phi");
  std::vector<svmp::FE::Real> initial(
      static_cast<std::size_t>(system->dofHandler().getNumDofs()), 0.0);
  writeWorkflowFieldSlice(*system, phi, coefficients, initial);

  std::unique_ptr<svmp::FE::backends::BackendFactory> factory;
  try {
    factory = svmp::FE::backends::BackendFactory::create(
        svmp::FE::backends::BackendKind::FSILS);
  } catch (const std::exception&) {
    GTEST_SKIP() << "Requires an available FE vector backend.";
  }
  auto history = svmp::FE::timestepping::TimeHistory::allocate(
      *factory,
      system->dofHandler().getNumDofs(),
      /*history_depth=*/2,
      /*allocate_second_order_state=*/true);
  history.setStepIndex(1);
  history.setDt(0.1);
  history.setPrevDt(0.1);
  scatterFeOrderedSolution(history.u(), initial);
  scatterFeOrderedSolution(history.uPrev(), initial);
  scatterFeOrderedSolution(history.uPrev2(), initial);
  std::vector<svmp::FE::Real> rates(initial.size(), 0.375);
  scatterFeOrderedSolution(history.uDot(), rates);

  application::core::SimulationComponents sim;
  sim.primary_mesh = mesh;
  sim.fe_system = std::move(system);

  LevelSetMaintenanceRequest first_request{};
  first_request.level_set_field_name = "phi";
  first_request.volume_correction.enabled = true;
  first_request.volume_correction.cadence_steps = 1;
  first_request.volume_correction.use_initial_negative_volume_as_target = false;
  first_request.volume_correction.target_negative_volume = 0.36;
  first_request.volume_correction.minimum_relative_volume_error = 0.0;
  first_request.volume_correction.maximum_interface_displacement_fraction =
      0.10;
  first_request.volume_correction
      .maximum_cumulative_interface_displacement_fraction = 1.0;
  first_request.volume_target_initialized = true;
  first_request.volume_target = 0.36;

  std::vector<LevelSetMaintenanceRequest> successful_requests{
      first_request};
  std::vector<LevelSetVolumeCorrectionMaintenanceEvent> successful_events;
  testing::internal::CaptureStdout();
  const bool successful_change = applyLevelSetMaintenance(
      sim,
      history,
      successful_requests,
      {},
      {},
      {},
      &successful_events);
  const auto successful_output = testing::internal::GetCapturedStdout();
  ASSERT_TRUE(successful_change);
  ASSERT_EQ(successful_events.size(), 1u);
  EXPECT_EQ(successful_events.front().level_set_field, phi);
  EXPECT_EQ(successful_events.front().level_set_field_name, "phi");
  EXPECT_TRUE(successful_events.front().correction.correction_applied);
  EXPECT_NE(successful_output.find(
                "max_contact_angle_change_radians=0"),
            std::string::npos);
  EXPECT_NE(successful_output.find(
                "negative_component_topology_preserved=true"),
            std::string::npos);
  EXPECT_NE(successful_output.find("negative_component_count=1"),
            std::string::npos);
  EXPECT_NE(successful_output.find("component_global_vertex_id="),
            std::string::npos);
  EXPECT_NE(successful_output.find("component_volume_transfer="),
            std::string::npos);
  scatterFeOrderedSolution(history.u(), initial);
  scatterFeOrderedSolution(history.uPrev(), initial);
  scatterFeOrderedSolution(history.uPrev2(), initial);
  scatterFeOrderedSolution(history.uDot(), rates);

  auto rejecting_request = first_request;
  rejecting_request.volume_correction.target_negative_volume = 0.35;
  rejecting_request.volume_correction
      .maximum_cumulative_interface_displacement_fraction = 0.10;
  rejecting_request.volume_target = 0.35;
  rejecting_request.volume_correction_reference_minimum_edge_length = 1.0;
  rejecting_request.cumulative_volume_correction_interface_displacement = 0.0;
  rejecting_request.cumulative_volume_correction_contact_line_displacement =
      0.095;
  std::vector<LevelSetMaintenanceRequest> requests{
      first_request,
      rejecting_request};

  const auto current_before = gatherFeOrderedSolution(history.u());
  const auto previous_before = gatherFeOrderedSolution(history.uPrev());
  const auto previous2_before = gatherFeOrderedSolution(history.uPrev2());
  const auto rates_before = gatherFeOrderedSolution(history.uDot());
  auto rejected_events = successful_events;

  testing::internal::CaptureStdout();
  std::string rejection_message;
  try {
    (void)applyLevelSetMaintenance(
        sim,
        history,
        requests,
        {},
        {},
        {},
        &rejected_events);
  } catch (const std::runtime_error& error) {
    rejection_message = error.what();
  }
  const auto output = testing::internal::GetCapturedStdout();
  EXPECT_NE(rejection_message.find("limiting_path=contact_line"),
            std::string::npos);
  EXPECT_EQ(output.find("Level-set volume corrected"), std::string::npos);
  EXPECT_EQ(output.find("Level-set maintenance synchronized"),
            std::string::npos);
  EXPECT_TRUE(rejected_events.empty());

  EXPECT_EQ(gatherFeOrderedSolution(history.u()), current_before);
  EXPECT_EQ(gatherFeOrderedSolution(history.uPrev()), previous_before);
  EXPECT_EQ(gatherFeOrderedSolution(history.uPrev2()), previous2_before);
  EXPECT_EQ(gatherFeOrderedSolution(history.uDot()), rates_before);
  ASSERT_EQ(requests.size(), 2u);
  EXPECT_DOUBLE_EQ(
      requests[0].volume_correction_reference_minimum_edge_length,
      first_request.volume_correction_reference_minimum_edge_length);
  EXPECT_DOUBLE_EQ(
      requests[0].cumulative_volume_correction_interface_displacement,
      first_request.cumulative_volume_correction_interface_displacement);
  EXPECT_DOUBLE_EQ(
      requests[0].cumulative_volume_correction_contact_line_displacement,
      first_request.cumulative_volume_correction_contact_line_displacement);
  EXPECT_DOUBLE_EQ(
      requests[1].volume_correction_reference_minimum_edge_length,
      rejecting_request.volume_correction_reference_minimum_edge_length);
  EXPECT_DOUBLE_EQ(
      requests[1].cumulative_volume_correction_interface_displacement,
      rejecting_request.cumulative_volume_correction_interface_displacement);
  EXPECT_DOUBLE_EQ(
      requests[1].cumulative_volume_correction_contact_line_displacement,
      rejecting_request.cumulative_volume_correction_contact_line_displacement);
#endif
}

} // namespace
#else
TEST(ApplicationDriverBoundPreservingCandidates,
     RequiresMeshIntegration)
{
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
}
#endif
