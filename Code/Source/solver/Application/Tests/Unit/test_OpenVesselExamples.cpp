#include <gtest/gtest.h>

#include "tinyxml2.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iterator>
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
  expectText(free_surface, "Surface_tension", "0.0728");
  expectText(free_surface, "Enable_cut_cell_stabilization", "true");
  expectText(free_surface, "Cut_cell_velocity_gradient_penalty", "1.0");
  expectText(free_surface, "Cut_cell_pressure_gradient_penalty", "1.0");
}
