#include <gtest/gtest.h>

#include "Application/Translators/EquationTranslator.h"
#include "FE/Systems/FESystem.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"
#include "Parameters.h"
#include "tinyxml2.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace fs = std::filesystem;

std::string trimCopy(std::string value)
{
  const auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
  value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
  value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
  return value;
}

fs::path repositoryRoot()
{
  fs::path path = fs::current_path();
  for (int depth = 0; depth < 12; ++depth) {
    if (fs::exists(path / "tests/cases/fluid/open_vessel_free_surface")) {
      return path;
    }
    if (!path.has_parent_path() || path == path.parent_path()) {
      break;
    }
    path = path.parent_path();
  }
  throw std::runtime_error("could not locate repository root from current path");
}

fs::path openVesselCaseDir(std::string_view case_name)
{
  return repositoryRoot() / "tests/cases/fluid/open_vessel_free_surface" /
         std::string(case_name);
}

std::shared_ptr<svmp::Mesh> makeTranslatorQuadMesh()
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
  base->register_label("wall_left", 1);
  base->register_label("wall_right", 2);
  base->register_label("wall_bottom", 3);

  return svmp::create_mesh(std::move(base));
}

void loadXml(const fs::path& path, tinyxml2::XMLDocument& doc)
{
  const auto status = doc.LoadFile(path.string().c_str());
  if (status != tinyxml2::XML_SUCCESS) {
    throw std::runtime_error("failed to load " + path.string() + ": " + doc.ErrorStr());
  }
}

const tinyxml2::XMLElement& child(const tinyxml2::XMLElement& parent,
                                  const char* name)
{
  const auto* element = parent.FirstChildElement(name);
  if (element == nullptr) {
    throw std::runtime_error(std::string("missing child element ") + name);
  }
  return *element;
}

const tinyxml2::XMLElement& childWithAttribute(const tinyxml2::XMLElement& parent,
                                               const char* child_name,
                                               const char* attribute_name,
                                               std::string_view attribute_value)
{
  for (const auto* element = parent.FirstChildElement(child_name);
       element != nullptr;
       element = element->NextSiblingElement(child_name)) {
    const char* value = element->Attribute(attribute_name);
    if (value != nullptr && attribute_value == value) {
      return *element;
    }
  }
  throw std::runtime_error(
      std::string("missing ") + child_name + " with " + attribute_name + "='" +
      std::string(attribute_value) + "'");
}

tinyxml2::XMLElement& mutableChildWithAttribute(tinyxml2::XMLElement& parent,
                                                const char* child_name,
                                                const char* attribute_name,
                                                std::string_view attribute_value)
{
  for (auto* element = parent.FirstChildElement(child_name);
       element != nullptr;
       element = element->NextSiblingElement(child_name)) {
    const char* value = element->Attribute(attribute_name);
    if (value != nullptr && attribute_value == value) {
      return *element;
    }
  }
  throw std::runtime_error(
      std::string("missing ") + child_name + " with " + attribute_name + "='" +
      std::string(attribute_value) + "'");
}

std::unique_ptr<EquationParameters> equationParametersFromElement(
    tinyxml2::XMLElement& element)
{
  const char* type = nullptr;
  element.QueryStringAttribute("type", &type);
  if (type == nullptr) {
    throw std::runtime_error("missing Add_equation type");
  }

  auto params = std::make_unique<EquationParameters>();
  params->type.set(std::string(type));
  params->set_values(&element);
  return params;
}

std::string text(const tinyxml2::XMLElement& parent, const char* name)
{
  const auto& element = child(parent, name);
  return trimCopy(element.GetText() == nullptr ? std::string{} : element.GetText());
}

void expectText(const tinyxml2::XMLElement& parent,
                const char* name,
                std::string_view expected)
{
  EXPECT_EQ(text(parent, name), expected) << name;
}

void expectReferencedFileExists(const fs::path& base_dir,
                                const tinyxml2::XMLElement& parent,
                                const char* path_element)
{
  const auto relative = text(parent, path_element);
  ASSERT_FALSE(relative.empty()) << path_element;
  EXPECT_TRUE(fs::exists(base_dir / relative))
      << "missing referenced file " << (base_dir / relative).string();
}

const tinyxml2::XMLElement& expectFace(const fs::path& base_dir,
                                       const tinyxml2::XMLElement& mesh,
                                       std::string_view name)
{
  const auto& face = childWithAttribute(mesh, "Add_face", "name", name);
  expectReferencedFileExists(base_dir, face, "Face_file_path");
  return face;
}

const tinyxml2::XMLElement& expectBoundaryCondition(const tinyxml2::XMLElement& equation,
                                                    std::string_view name,
                                                    std::string_view type)
{
  const auto& bc = childWithAttribute(equation, "Add_BC", "name", name);
  expectText(bc, "Type", type);
  return bc;
}

void expectOutputFields(const tinyxml2::XMLElement& equation,
                        std::string_view output_type,
                        std::vector<std::string_view> fields)
{
  const auto& output = childWithAttribute(equation, "Output", "type", output_type);
  for (const auto field : fields) {
    expectText(output, std::string(field).c_str(), "true");
  }
}

void expectGmresSolver(const tinyxml2::XMLElement& equation)
{
  const auto& solver = childWithAttribute(equation, "LS", "type", "GMRES");
  EXPECT_FALSE(text(solver, "Max_iterations").empty());
  EXPECT_FALSE(text(solver, "Krylov_space_dimension").empty());
  EXPECT_FALSE(text(solver, "Tolerance").empty());
}

void expectBlockSchurSolver(const tinyxml2::XMLElement& equation)
{
  const auto& solver = childWithAttribute(equation, "LS", "type", "NS");
  EXPECT_FALSE(text(solver, "Max_iterations").empty());
  EXPECT_FALSE(text(solver, "Krylov_space_dimension").empty());
  EXPECT_FALSE(text(solver, "Tolerance").empty());
  expectText(solver, "NS_GM_max_iterations", "1000");
  expectText(solver, "NS_GM_tolerance", "1.0e-6");
  expectText(solver, "NS_CG_max_iterations", "1000");
  expectText(solver, "NS_CG_tolerance", "1.0e-6");
  expectText(solver, "NS_Schur_preconditioner", "algebraic-shat");
  expectText(solver, "NS_Momentum_approximation", "ilu-k");
  expectText(solver, "NS_Use_coupled_outer_FGMRES", "true");
}

bool fileContains(const fs::path& path, std::string_view needle)
{
  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::runtime_error("failed to open " + path.string());
  }
  const std::string contents((std::istreambuf_iterator<char>(input)),
                            std::istreambuf_iterator<char>());
  return contents.find(needle) != std::string::npos;
}

bool startsWith(std::string_view value, std::string_view prefix)
{
  return value.size() >= prefix.size() &&
         value.substr(0, prefix.size()) == prefix;
}

} // namespace

TEST(OpenVesselExamples, FittedAleCaseDeclaresRequiredControls)
{
  const auto case_dir = openVesselCaseDir("fitted_ale");
  tinyxml2::XMLDocument doc;
  ASSERT_NO_THROW(loadXml(case_dir / "solver.xml", doc));
  const auto* root = doc.FirstChildElement("svMultiPhysicsFile");
  ASSERT_NE(root, nullptr);

  const auto& general = child(*root, "GeneralSimulationParameters");
  expectText(general, "Use_new_OOP_solver", "true");
  expectText(general, "Number_of_time_steps", "20");
  expectText(general, "Time_step_size", "0.0025");

  const auto& mesh = childWithAttribute(*root, "Add_mesh", "name", "tank");
  expectReferencedFileExists(case_dir, mesh, "Mesh_file_path");
  expectFace(case_dir, mesh, "wall_left");
  expectFace(case_dir, mesh, "wall_right");
  expectFace(case_dir, mesh, "wall_bottom");
  expectFace(case_dir, mesh, "free_surface");

  const auto mesh_path = case_dir / text(mesh, "Mesh_file_path");
  EXPECT_TRUE(fileContains(mesh_path, "Name=\"Pressure\""));
  EXPECT_TRUE(fileContains(mesh_path, "Name=\"Velocity\""));
  EXPECT_TRUE(fileContains(mesh_path, "Name=\"mesh_displacement\""));
  EXPECT_TRUE(fileContains(mesh_path, "Name=\"mesh_velocity\""));

  const auto& fluid = childWithAttribute(*root, "Add_equation", "type", "fluid");
  expectText(fluid, "Enable_ALE", "true");
  expectText(fluid, "Mesh_velocity_source", "coupled_displacement");
  expectText(fluid, "Mesh_displacement_field", "mesh_displacement");
  expectText(fluid, "Force_x", "0.0");
  expectText(fluid, "Force_y", "-9.81");
  expectText(fluid, "Force_z", "0.0");
  expectText(fluid, "Hydrostatic_pressure_initialization", "true");
  expectText(fluid, "Hydrostatic_pressure_reference_point", "0.0 0.5 0.0");
  expectReferencedFileExists(case_dir, child(fluid, "Node_pressure_constraints"),
                             "Values_file_path");
  expectGmresSolver(fluid);
  expectOutputFields(fluid, "Spatial",
                     {"Velocity", "Pressure", "Mesh_displacement", "Mesh_velocity",
                      "Surface_position"});
  expectOutputFields(fluid, "Volume_integral", {"Volume"});

  expectBoundaryCondition(fluid, "wall_left", "Dir");
  expectBoundaryCondition(fluid, "wall_right", "Dir");
  expectBoundaryCondition(fluid, "wall_bottom", "Dir");
  const auto& free_surface = expectBoundaryCondition(fluid, "free_surface", "Free_surface");
  expectText(free_surface, "Implementation", "FittedALE");
  expectText(free_surface, "External_pressure", "0.0");
  expectText(free_surface, "Surface_tension", "0.0728");
  expectText(free_surface, "Kinematic_enforcement", "Nitsche");
  expectText(free_surface, "Normal_kinematic_policy", "MatchFluidNormalVelocity");

  const auto& mesh_motion = childWithAttribute(*root, "Add_equation", "type", "mesh_motion");
  expectText(mesh_motion, "Model", "Harmonic");
  expectText(mesh_motion, "Field_name", "mesh_displacement");
  expectText(mesh_motion, "Kappa", "1.0");
  expectBoundaryCondition(mesh_motion, "wall_left", "Dir");
  expectBoundaryCondition(mesh_motion, "wall_right", "Dir");
  expectBoundaryCondition(mesh_motion, "wall_bottom", "Dir");
}

TEST(OpenVesselExamples, UnfittedLevelSetCaseDeclaresRequiredControls)
{
  const auto case_dir = openVesselCaseDir("unfitted_level_set");
  tinyxml2::XMLDocument doc;
  ASSERT_NO_THROW(loadXml(case_dir / "solver.xml", doc));
  const auto* root = doc.FirstChildElement("svMultiPhysicsFile");
  ASSERT_NE(root, nullptr);

  const auto& general = child(*root, "GeneralSimulationParameters");
  expectText(general, "Use_new_OOP_solver", "true");
  expectText(general, "Number_of_time_steps", "20");
  expectText(general, "Time_step_size", "0.0025");

  const auto& mesh = childWithAttribute(*root, "Add_mesh", "name", "tank");
  expectReferencedFileExists(case_dir, mesh, "Mesh_file_path");
  expectFace(case_dir, mesh, "wall_left");
  expectFace(case_dir, mesh, "wall_right");
  expectFace(case_dir, mesh, "wall_bottom");
  expectFace(case_dir, mesh, "wall_top");

  const auto mesh_path = case_dir / text(mesh, "Mesh_file_path");
  EXPECT_TRUE(fileContains(mesh_path, "Name=\"phi\""));
  EXPECT_TRUE(fileContains(mesh_path, "Name=\"Pressure\""));
  EXPECT_TRUE(fileContains(mesh_path, "Name=\"Velocity\""));

  const auto& level_set = childWithAttribute(*root, "Add_equation", "type", "level_set");
  expectText(level_set, "Level_set_field_name", "phi");
  expectText(level_set, "Level_set_source", "prescribed_data");
  expectText(level_set, "Velocity_source", "constant");
  expectText(level_set, "Constant_velocity", "0.0 0.0 0.0");
  expectText(level_set, "Enable_SUPG", "true");
  expectText(level_set, "Enable_reinitialization", "true");
  expectText(level_set, "Reinitialization_method", "projection");
  expectText(level_set, "Enable_volume_correction", "true");
  expectText(level_set, "Volume_correction_use_initial_volume", "true");
  expectText(level_set, "Volume_correction_cadence_steps", "5");
  expectGmresSolver(level_set);
  expectOutputFields(level_set, "Spatial",
                     {"Level_set", "Generated_interface", "Surface_position"});
  expectOutputFields(level_set, "Volume_integral", {"Volume"});

  const auto& fluid = childWithAttribute(*root, "Add_equation", "type", "fluid");
  expectText(fluid, "Force_x", "0.0");
  expectText(fluid, "Force_y", "-9.81");
  expectText(fluid, "Force_z", "0.0");
  expectText(fluid, "Hydrostatic_pressure_initialization", "true");
  expectText(fluid, "Hydrostatic_pressure_reference_point", "0.0 0.5 0.0");
  expectReferencedFileExists(case_dir, child(fluid, "Node_pressure_constraints"),
                             "Values_file_path");
  expectGmresSolver(fluid);
  expectOutputFields(fluid, "Spatial", {"Velocity", "Pressure"});
  expectOutputFields(fluid, "Volume_integral", {"Volume"});

  expectBoundaryCondition(fluid, "wall_left", "Dir");
  expectBoundaryCondition(fluid, "wall_right", "Dir");
  expectBoundaryCondition(fluid, "wall_bottom", "Dir");
  const auto& free_surface = expectBoundaryCondition(fluid, "free_surface", "Free_surface");
  expectText(free_surface, "Implementation", "UnfittedLevelSet");
  expectText(free_surface, "Level_set_field_name", "phi");
  expectText(free_surface, "Generated_interface_domain_id", "open_vessel_surface");
  expectText(free_surface, "Level_set_isovalue", "0.0");
  expectText(free_surface, "External_pressure", "0.0");
  expectText(free_surface, "Surface_tension", "0.0");
  expectText(free_surface, "Enable_cut_cell_stabilization", "true");
  expectText(free_surface, "Use_cut_metadata_scale", "false");
  expectText(free_surface, "Cut_cell_velocity_gradient_penalty", "1.0");
  expectText(free_surface, "Cut_cell_pressure_gradient_penalty", "1.0");
}

TEST(OpenVesselExamples, UnfittedLevelSetCaseBuildsOopInputs)
{
  const auto case_dir = openVesselCaseDir("unfitted_level_set");
  tinyxml2::XMLDocument doc;
  ASSERT_NO_THROW(loadXml(case_dir / "solver.xml", doc));
  auto* root = doc.FirstChildElement("svMultiPhysicsFile");
  ASSERT_NE(root, nullptr);

  auto mesh = makeTranslatorQuadMesh();
  const std::map<std::string, std::shared_ptr<svmp::Mesh>> meshes{{"tank", mesh}};

  auto level_set_params = equationParametersFromElement(
      mutableChildWithAttribute(*root, "Add_equation", "type", "level_set"));
  const auto level_set_input =
      application::translators::EquationTranslator::buildInput(
          *level_set_params,
          meshes);

  EXPECT_EQ(level_set_input.equation_type, "level_set");
  EXPECT_EQ(level_set_input.mesh_name, "tank");
  EXPECT_EQ(level_set_input.equation_params.at("Level_set_field_name").value, "phi");
  EXPECT_EQ(level_set_input.equation_params.at("Level_set_source").value,
            "prescribed_data");

  svmp::FE::systems::FESystem system(mesh);
  auto module = application::translators::EquationTranslator::createModule(
      *level_set_params,
      system,
      meshes);
  ASSERT_TRUE(module);
  const auto phi = system.findFieldByName("phi");
  ASSERT_NE(phi, svmp::FE::INVALID_FIELD_ID);
  EXPECT_TRUE(system.fieldParticipatesInUnknownVector(phi));
  EXPECT_TRUE(system.hasOperator("equations"));
  EXPECT_FALSE(system.hasOperator("level_set"));

  auto fluid_params = equationParametersFromElement(
      mutableChildWithAttribute(*root, "Add_equation", "type", "fluid"));
  const auto fluid_input =
      application::translators::EquationTranslator::buildInput(
          *fluid_params,
          meshes);

  const auto free_surface = std::find_if(
      fluid_input.boundary_conditions.begin(),
      fluid_input.boundary_conditions.end(),
      [](const svmp::Physics::BoundaryConditionInput& bc) {
        return bc.name == "free_surface";
      });
  ASSERT_NE(free_surface, fluid_input.boundary_conditions.end());
  EXPECT_EQ(free_surface->boundary_marker, svmp::INVALID_LABEL);
  EXPECT_EQ(free_surface->params.at("Implementation").value, "UnfittedLevelSet");
  EXPECT_EQ(free_surface->params.at("Level_set_field_name").value, "phi");
  EXPECT_EQ(free_surface->params.at("Generated_interface_domain_id").value,
            "open_vessel_surface");
}

TEST(OpenVesselExamples, FittedAleCaseBuildsMeshMotionOopInputs)
{
  const auto case_dir = openVesselCaseDir("fitted_ale");
  tinyxml2::XMLDocument doc;
  ASSERT_NO_THROW(loadXml(case_dir / "solver.xml", doc));
  auto* root = doc.FirstChildElement("svMultiPhysicsFile");
  ASSERT_NE(root, nullptr);

  auto mesh = makeTranslatorQuadMesh();
  const std::map<std::string, std::shared_ptr<svmp::Mesh>> meshes{{"tank", mesh}};

  auto mesh_motion_params = equationParametersFromElement(
      mutableChildWithAttribute(*root, "Add_equation", "type", "mesh_motion"));
  const auto mesh_motion_input =
      application::translators::EquationTranslator::buildInput(
          *mesh_motion_params,
          meshes);
  EXPECT_EQ(mesh_motion_input.equation_type, "mesh_motion");
  EXPECT_EQ(mesh_motion_input.equation_params.at("Model").value, "Harmonic");
  EXPECT_EQ(mesh_motion_input.equation_params.at("Field_name").value,
            "mesh_displacement");
  EXPECT_EQ(mesh_motion_input.equation_params.at("Operator_tag").value,
            "equations");
  EXPECT_EQ(mesh_motion_input.equation_params.at("Kappa").value, "1.0");
  ASSERT_EQ(mesh_motion_input.boundary_conditions.size(), 3u);
}

TEST(OpenVesselExamples, LiteratureValidationCasesDeclareGeneratedMeshes)
{
  struct CaseExpectation {
    std::string representation_dir;
    std::string case_name;
    bool fitted;
    bool top_wall_bc;
    bool obstacle_bc;
    std::vector<std::string> faces;
  };

  const std::vector<CaseExpectation> cases = {
      {"fitted_ale",
       "spheric_test10_lateral_water_1x",
       true,
       false,
       false,
       {"wall_left", "wall_right", "wall_bottom", "wall_front", "wall_back",
        "free_surface"}},
      {"unfitted_level_set",
       "spheric_test10_lateral_water_1x",
       false,
       true,
       false,
       {"wall_left", "wall_right", "wall_bottom", "wall_front", "wall_back",
        "wall_top"}},
      {"unfitted_level_set",
       "spheric_test05_wet_bed_d18",
       false,
       false,
       false,
       {"wall_left", "wall_right", "wall_bottom", "wall_front", "wall_back",
        "wall_top"}},
      {"unfitted_level_set",
       "spheric_test05_wet_bed_d38",
       false,
       false,
       false,
       {"wall_left", "wall_right", "wall_bottom", "wall_front", "wall_back",
        "wall_top"}},
      {"unfitted_level_set",
       "spheric_test02_dambreak_obstacle",
       false,
       false,
       true,
       {"wall_left", "wall_right", "wall_bottom", "wall_front", "wall_back",
        "wall_top", "obstacle"}},
  };

  for (const auto& expected : cases) {
    SCOPED_TRACE(expected.representation_dir + "/" + expected.case_name);
    const auto case_dir =
        openVesselCaseDir(expected.representation_dir) / expected.case_name;
    tinyxml2::XMLDocument doc;
    ASSERT_NO_THROW(loadXml(case_dir / "solver.xml", doc));
    const auto* root = doc.FirstChildElement("svMultiPhysicsFile");
    ASSERT_NE(root, nullptr);
    EXPECT_TRUE(fs::exists(case_dir / "benchmark.json"));

    const auto& general = child(*root, "GeneralSimulationParameters");
    expectText(general, "Use_new_OOP_solver", "true");
    expectText(general, "Number_of_spatial_dimensions", "3");

    const auto& mesh = childWithAttribute(*root, "Add_mesh", "name", "tank");
    expectReferencedFileExists(case_dir, mesh, "Mesh_file_path");
    for (const auto& face : expected.faces) {
      expectFace(case_dir, mesh, face);
    }

    const auto mesh_path = case_dir / text(mesh, "Mesh_file_path");
    EXPECT_TRUE(fileContains(mesh_path, "Name=\"GlobalNodeID\""));
    EXPECT_TRUE(fileContains(mesh_path, "Name=\"Pressure\""));
    EXPECT_TRUE(fileContains(mesh_path, "Name=\"Velocity\""));
    if (expected.fitted) {
      EXPECT_TRUE(fileContains(mesh_path, "Name=\"mesh_displacement\""));
      EXPECT_TRUE(fileContains(mesh_path, "Name=\"mesh_velocity\""));
    } else {
      EXPECT_TRUE(fileContains(mesh_path, "Name=\"phi\""));
    }

    const auto& fluid = childWithAttribute(*root, "Add_equation", "type", "fluid");
    expectReferencedFileExists(case_dir, child(fluid, "Node_pressure_constraints"),
                               "Values_file_path");
    for (const auto& face : expected.faces) {
      if (startsWith(face, "wall_") &&
          (expected.top_wall_bc || face != "wall_top")) {
        expectBoundaryCondition(fluid, face, "Dir");
      }
    }
    if (expected.obstacle_bc) {
      expectBoundaryCondition(fluid, "obstacle", "Dir");
    }

    const auto& free_surface =
        expectBoundaryCondition(fluid, "free_surface", "Free_surface");
    expectText(free_surface,
               "Implementation",
               expected.fitted ? "FittedALE" : "UnfittedLevelSet");
    const bool is_test05 =
        startsWith(expected.case_name, "spheric_test05_wet_bed_");
    if (is_test05) {
      expectText(free_surface, "Active_domain", "LevelSetNegative");
      expectText(free_surface, "Active_domain_method", "CutVolume");
      expectBlockSchurSolver(fluid);
      EXPECT_EQ(free_surface.FirstChildElement("Kinematic_enforcement"),
                nullptr);
      EXPECT_TRUE(fileContains(case_dir / "pressure_gauge.csv",
                               "256,643.659423052"));
      EXPECT_TRUE(fileContains(case_dir / "benchmark.json",
                               "\"node_id\": 256"));
      EXPECT_TRUE(fileContains(case_dir / "benchmark.json",
                               "\"initial_phi\": -0.0142363529744199"));
    }

    if (expected.fitted) {
      expectText(fluid, "Enable_ALE", "true");
      const auto& mesh_motion =
          childWithAttribute(*root, "Add_equation", "type", "mesh_motion");
      expectText(mesh_motion, "Model", "Harmonic");
    } else {
      const auto& level_set =
          childWithAttribute(*root, "Add_equation", "type", "level_set");
      expectText(level_set, "Level_set_field_name", "phi");
      expectText(level_set, "Velocity_source", "coupled_field");
      expectText(free_surface, "Generated_interface_domain_id",
                 "open_vessel_surface");
    }
  }
}
