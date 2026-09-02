#include <gtest/gtest.h>

#include "Application/Core/LevelSetCutConfiguration.h"
#include "Application/Core/SimulationBuilder.h"
#include "Application/Translators/EquationTranslator.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Interfaces/LevelSetInterfaceDomain.h"
#include "FE/Spaces/SpaceFactory.h"
#include "FE/Systems/FESystem.h"
#include "FE/TimeStepping/TimeHistory.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Fields/MeshFields.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"
#include "Parameters.h"
#include "tinyxml2.h"

#if FE_HAS_MPI || defined(MESH_HAS_MPI)
#include <mpi.h>
#include <cstdlib>
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

#if FE_HAS_MPI || defined(MESH_HAS_MPI)
void finalize_mpi_if_needed()
{
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (!finalized) {
    MPI_Finalize();
  }
}
#endif

void ensure_mpi_initialized_for_open_vessel_builder()
{
#if FE_HAS_MPI || defined(MESH_HAS_MPI)
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    int argc = 0;
    char** argv = nullptr;
    MPI_Init(&argc, &argv);
    std::atexit(finalize_mpi_if_needed);
  }
#endif
}

class ScopedCurrentPath {
public:
  explicit ScopedCurrentPath(const fs::path& path)
      : previous_(fs::current_path())
  {
    fs::current_path(path);
  }

  ~ScopedCurrentPath()
  {
    fs::current_path(previous_);
  }

  ScopedCurrentPath(const ScopedCurrentPath&) = delete;
  ScopedCurrentPath& operator=(const ScopedCurrentPath&) = delete;

private:
  fs::path previous_;
};

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

std::shared_ptr<svmp::Mesh> makeTranslatorQuadMesh(
    bool register_free_surface_label = true)
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
  const auto phi_handle = svmp::MeshFields::attach_field(
      *base,
      svmp::EntityKind::Vertex,
      "phi",
      svmp::FieldScalarType::Float64,
      1);
  auto* phi = svmp::MeshFields::field_data_as<svmp::real_t>(*base, phi_handle);
  phi[0] = -1.0;
  phi[1] = -1.0;
  phi[2] = 1.0;
  phi[3] = 1.0;
  base->register_label("wall_left", 1);
  base->register_label("wall_right", 2);
  base->register_label("wall_bottom", 3);
  base->register_label(
      register_free_surface_label ? "free_surface" : "wall_top", 4);

  return svmp::create_mesh(std::move(base));
}

std::shared_ptr<svmp::Mesh> makeTranslatorTriangleMesh()
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

bool hasChildWithAttribute(const tinyxml2::XMLElement& parent,
                           const char* child_name,
                           const char* attribute_name,
                           std::string_view attribute_value)
{
  for (const auto* element = parent.FirstChildElement(child_name);
       element != nullptr;
       element = element->NextSiblingElement(child_name)) {
    const char* value = element->Attribute(attribute_name);
    if (value != nullptr && attribute_value == value) {
      return true;
    }
  }
  return false;
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

std::unique_ptr<EquationParameters> equationParametersFromText(
    const char* xml)
{
  tinyxml2::XMLDocument document;
  if (document.Parse(xml) != tinyxml2::XML_SUCCESS) {
    throw std::runtime_error(
        std::string("failed to parse equation fixture: ") +
        document.ErrorStr());
  }
  auto* equation = document.FirstChildElement("Add_equation");
  if (equation == nullptr) {
    throw std::runtime_error("equation fixture has no Add_equation root");
  }
  return equationParametersFromElement(*equation);
}

std::unique_ptr<EquationParameters> materialInterfaceLevelSetEquation()
{
  return equationParametersFromText(R"xml(
<Add_equation type="level_set">
  <Coupled>true</Coupled>
  <Level_set_field_name>level_set</Level_set_field_name>
  <Level_set_source>unknown</Level_set_source>
  <Auto_register_level_set_field>true</Auto_register_level_set_field>
  <Velocity_source>material_interface_phase_pair</Velocity_source>
  <Material_interface_marker>71</Material_interface_marker>
  <Transport_form>advective</Transport_form>
  <Enable_conservative_phase_transport>true</Enable_conservative_phase_transport>
  <Conservative_phase_field_name>phase</Conservative_phase_field_name>
  <Auto_register_conservative_phase_field>true</Auto_register_conservative_phase_field>
  <Conservative_phase_reconcile_geometry>true</Conservative_phase_reconcile_geometry>
  <Conservative_phase_momentum_relative_tolerance>1e-10</Conservative_phase_momentum_relative_tolerance>
  <Enable_interface_kinematic>true</Enable_interface_kinematic>
  <Interface_kinematic_marker>71</Interface_kinematic_marker>
  <Operator_tag>equations</Operator_tag>
</Add_equation>
)xml");
}

std::unique_ptr<EquationParameters> incompressibleTwoFluidEquation()
{
  return equationParametersFromText(R"xml(
<Add_equation type="fluid">
  <Coupled>true</Coupled>
  <Free_surface_physical_model>IncompressibleTwoFluid</Free_surface_physical_model>
  <Level_set_field_name>level_set</Level_set_field_name>
  <Generated_interface_domain_id>material_interface</Generated_interface_domain_id>
  <Material_interface_marker>71</Material_interface_marker>
  <Negative_phase_density>1000.0</Negative_phase_density>
  <Negative_phase_dynamic_viscosity>0.01</Negative_phase_dynamic_viscosity>
  <Positive_phase_density>1.0</Positive_phase_density>
  <Positive_phase_dynamic_viscosity>0.001</Positive_phase_dynamic_viscosity>
  <Two_fluid_surface_tension>0.072</Two_fluid_surface_tension>
  <Two_fluid_interface_nitsche_gamma>24.0</Two_fluid_interface_nitsche_gamma>
  <Operator_tag>equations</Operator_tag>
  <LS type="NS">
    <Linear_algebra type="fsils">
      <Preconditioner>none</Preconditioner>
    </Linear_algebra>
  </LS>
</Add_equation>
)xml");
}

void appendTwoFluidMaterialInterfaceEquations(
    Parameters& parameters,
    bool level_set_first)
{
  auto level_set = materialInterfaceLevelSetEquation();
  auto fluid = incompressibleTwoFluidEquation();
  if (level_set_first) {
    parameters.equation_parameters.push_back(level_set.release());
    parameters.equation_parameters.push_back(fluid.release());
  } else {
    parameters.equation_parameters.push_back(fluid.release());
    parameters.equation_parameters.push_back(level_set.release());
  }
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

void expectReviewedP1LevelSetTransportControls(
    const tinyxml2::XMLElement& level_set)
{
  expectText(level_set, "Enable_SUPG", "true");
  expectText(level_set, "SUPG_tau_scale", "0.5");
  expectText(level_set, "SUPG_transient_scale", "2.0");
  expectText(level_set, "Enable_discontinuity_capturing", "true");
  expectText(level_set, "Discontinuity_capturing_scale", "0.1");
  expectText(level_set, "Discontinuity_capturing_gradient_epsilon", "1.0e-12");
  expectText(level_set, "Discontinuity_capturing_max_courant", "0.5");
  expectText(level_set, "Enable_bound_preserving_limiter", "true");
  expectText(level_set, "Bound_preserving_maximum_courant", "1.0");
  expectText(level_set, "Bound_preserving_enforce_courant_limit", "true");
  expectText(level_set,
             "Bound_preserving_enforce_impermeable_boundaries",
             "true");
  expectText(level_set,
             "Bound_preserving_impermeable_normal_velocity_tolerance",
             "1.0e-10");
}

void setOrAppendText(tinyxml2::XMLDocument& doc,
                     tinyxml2::XMLElement& parent,
                     const char* name,
                     const char* value)
{
  auto* element = parent.FirstChildElement(name);
  if (element == nullptr) {
    element = doc.NewElement(name);
    parent.InsertEndChild(element);
  }
  element->SetText(value);
}

fs::path writeBuilderRegressionXml(const fs::path& case_dir)
{
  tinyxml2::XMLDocument doc;
  loadXml(case_dir / "solver.xml", doc);
  auto* root = doc.FirstChildElement("svMultiPhysicsFile");
  if (root == nullptr) {
    throw std::runtime_error("solver.xml has no svMultiPhysicsFile root");
  }
  auto& fluid =
      mutableChildWithAttribute(*root, "Add_equation", "type", "fluid");
  auto& level_set =
      mutableChildWithAttribute(*root, "Add_equation", "type", "level_set");
  auto* level_set_solver = level_set.FirstChildElement("LS");
  auto* fluid_solver = fluid.FirstChildElement("LS");
  if (level_set_solver == nullptr || fluid_solver == nullptr) {
    throw std::runtime_error("builder regression requires both LS blocks");
  }
  setOrAppendText(doc, *level_set_solver, "Tolerance", "1.0e-6");
  setOrAppendText(doc, *level_set_solver, "Absolute_tolerance", "1.0e-10");
  setOrAppendText(doc, *fluid_solver, "Tolerance", "1.0e-4");
  setOrAppendText(doc, *fluid_solver, "Absolute_tolerance", "1.0e-4");
  setOrAppendText(doc, fluid, "Enable_level_set_cut_domain", "true");
  setOrAppendText(doc, fluid, "Level_set_field_name", "phi");
  setOrAppendText(doc, fluid, "Generated_interface_domain_id",
                  "open_vessel_surface");
  setOrAppendText(doc, fluid, "Interface_marker", "101");
  setOrAppendText(doc, fluid, "Active_domain", "LevelSetNegative");
  setOrAppendText(doc, fluid, "Active_domain_method", "CutVolume");

  const auto xml_path = fs::temp_directory_path() /
                        "svmp_open_vessel_unfitted_builder_cut_domain.xml";
  const auto status = doc.SaveFile(xml_path.string().c_str());
  if (status != tinyxml2::XML_SUCCESS) {
    throw std::runtime_error("failed to write " + xml_path.string() + ": " +
                             doc.ErrorStr());
  }
  return xml_path;
}

fs::path writeWetExtensionOrderRegressionXml(const fs::path& case_dir,
                                              bool include_fluid_owner,
                                              std::string_view fitted_exclusion =
                                                  {})
{
  tinyxml2::XMLDocument doc;
  loadXml(case_dir / "solver.xml", doc);
  auto* root = doc.FirstChildElement("svMultiPhysicsFile");
  if (root == nullptr) {
    throw std::runtime_error("solver.xml has no svMultiPhysicsFile root");
  }
  auto& level_set =
      mutableChildWithAttribute(*root, "Add_equation", "type", "level_set");
  auto& fluid =
      mutableChildWithAttribute(*root, "Add_equation", "type", "fluid");
  auto* fluid_solver = fluid.FirstChildElement("LS");
  if (fluid_solver == nullptr) {
    throw std::runtime_error(
        "wet-extension order regression requires a fluid LS block");
  }
  fluid_solver->SetAttribute("type", "ns");

  setOrAppendText(doc, level_set, "Velocity_source", "prescribed_data");
  if (auto* constant = level_set.FirstChildElement("Constant_velocity")) {
    level_set.DeleteChild(constant);
  }
  setOrAppendText(doc,
                  level_set,
                  "Velocity_field_name",
                  "LevelSetAdvectionVelocity");
  setOrAppendText(doc, level_set, "Auto_register_velocity_field", "true");
  setOrAppendText(doc,
                  level_set,
                  "Use_wet_extension_advection_velocity",
                  "true");
  setOrAppendText(doc, level_set, "Source_velocity_field_name", "Velocity");
  setOrAppendText(doc,
                  level_set,
                  "Wet_extension_advection_velocity_method",
                  "wall_compatible_normal");
  setOrAppendText(doc, level_set, "Enable_curvature_projection", "true");
  setOrAppendText(doc,
                  level_set,
                  "Projected_curvature_field",
                  "kappa_projected");

  if (!fitted_exclusion.empty()) {
    auto& free_surface =
        mutableChildWithAttribute(fluid, "Add_BC", "name", "free_surface");
    setOrAppendText(doc, free_surface, "Implementation", "FittedALE");
    if (fitted_exclusion == "SurfaceStress" ||
        fitted_exclusion == "GeneratedCurvatureTraction" ||
        fitted_exclusion == "KinematicAreaGradientTraction") {
      setOrAppendText(doc,
                      free_surface,
                      "Surface_tension_form",
                      std::string(fitted_exclusion).c_str());
    } else if (fitted_exclusion == "PrescribedAngle" ||
               fitted_exclusion == "DynamicRenE") {
      setOrAppendText(doc,
                      free_surface,
                      "Contact_line_model",
                      std::string(fitted_exclusion).c_str());
      setOrAppendText(
          doc, free_surface, "Contact_angle_degrees", "60.0");
      setOrAppendText(
          doc, free_surface, "Contact_line_wall_marker", "1");
      setOrAppendText(
          doc, free_surface, "Contact_line_wall_normal", "1.0 0.0 0.0");
      if (fitted_exclusion == "DynamicRenE") {
        setOrAppendText(
            doc, free_surface, "Contact_line_mobility", "0.5");
        setOrAppendText(doc, free_surface, "Wall_slip_model", "Navier");
        setOrAppendText(doc, free_surface, "Wall_slip_length", "0.2");
      }
    } else {
      throw std::invalid_argument(
          "unknown fitted capability exclusion requested by test");
    }
  }

  if (!include_fluid_owner) {
    root->DeleteChild(&fluid);
  }

  const auto suffix = !fitted_exclusion.empty()
                          ? ("rejected_fitted_" +
                             std::string(fitted_exclusion))
                          : (include_fluid_owner ? "future_fluid"
                                                 : "missing_fluid");
  const auto xml_path =
      fs::temp_directory_path() /
      (std::string{"svmp_wet_extension_equation_order_"} + suffix + ".xml");
  const auto status = doc.SaveFile(xml_path.string().c_str());
  if (status != tinyxml2::XML_SUCCESS) {
    throw std::runtime_error("failed to write " + xml_path.string() + ": " +
                             doc.ErrorStr());
  }
  return xml_path;
}

fs::path writeUnfittedGuardRegressionXml(
    const fs::path& case_dir,
    std::string_view suffix,
    const std::vector<std::pair<std::string, std::string>>& free_surface_overrides,
    bool append_second_active_free_surface = false)
{
  tinyxml2::XMLDocument doc;
  loadXml(case_dir / "solver.xml", doc);
  auto* root = doc.FirstChildElement("svMultiPhysicsFile");
  if (root == nullptr) {
    throw std::runtime_error("solver.xml has no svMultiPhysicsFile root");
  }
  auto& fluid =
      mutableChildWithAttribute(*root, "Add_equation", "type", "fluid");
  auto& free_surface =
      mutableChildWithAttribute(fluid, "Add_BC", "name", "free_surface");
  for (const auto& [name, value] : free_surface_overrides) {
    setOrAppendText(doc, free_surface, name.c_str(), value.c_str());
  }

  if (append_second_active_free_surface) {
    auto* clone_node = free_surface.DeepClone(&doc);
    auto* clone = clone_node == nullptr ? nullptr : clone_node->ToElement();
    if (clone == nullptr) {
      throw std::runtime_error("failed to clone free_surface BC");
    }
    clone->SetAttribute("name", "free_surface_duplicate");
    setOrAppendText(doc, *clone, "Generated_interface_domain_id",
                    "duplicate_open_vessel_surface");
    setOrAppendText(doc, *clone, "Interface_marker", "909");
    fluid.InsertEndChild(clone);
  }

  const auto xml_path = fs::temp_directory_path() /
                        ("svmp_open_vessel_unfitted_guard_" +
                         std::string(suffix) + ".xml");
  const auto status = doc.SaveFile(xml_path.string().c_str());
  if (status != tinyxml2::XML_SUCCESS) {
    throw std::runtime_error("failed to write " + xml_path.string() + ": " +
                             doc.ErrorStr());
  }
  return xml_path;
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

void expectEigenDirectSolver(const tinyxml2::XMLElement& equation)
{
  const auto& solver = childWithAttribute(equation, "LS", "type", "Direct");
  const auto& linear_algebra =
      childWithAttribute(solver, "Linear_algebra", "type", "eigen");
  expectText(linear_algebra, "Preconditioner", "none");
  EXPECT_FALSE(text(solver, "Max_iterations").empty());
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

bool startsWith(std::string_view value, std::string_view prefix)
{
  return value.size() >= prefix.size() &&
         value.substr(0, prefix.size()) == prefix;
}

} // namespace

TEST(EquationParameters,
     ParsesConservativePhaseOperationalControlsThroughAliases)
{
  struct Fixture {
    const char* naming;
    const char* xml;
    std::map<std::string, std::string> expected;
  };
  const std::array fixtures{
      Fixture{
          "underscore",
          R"xml(
<Add_equation type="level_set">
  <Conservative_phase_component_activity_tolerance>4e-7</Conservative_phase_component_activity_tolerance>
  <Conservative_phase_write_flux_artifacts>true</Conservative_phase_write_flux_artifacts>
  <Conservative_phase_flux_artifact_cadence_steps>3</Conservative_phase_flux_artifact_cadence_steps>
  <Conservative_phase_classify_nonprimary_components_as_satellites>true</Conservative_phase_classify_nonprimary_components_as_satellites>
  <Conservative_phase_fixed_flux_regions>film|wall_film|0|1|0|0.1|*|*</Conservative_phase_fixed_flux_regions>
</Add_equation>
)xml",
          {{"Conservative_phase_component_activity_tolerance", "4e-7"},
           {"Conservative_phase_write_flux_artifacts", "true"},
           {"Conservative_phase_flux_artifact_cadence_steps", "3"},
           {"Conservative_phase_classify_nonprimary_components_as_satellites",
            "true"},
           {"Conservative_phase_fixed_flux_regions",
            "film|wall_film|0|1|0|0.1|*|*"}}},
      Fixture{
          "compact",
          R"xml(
<Add_equation type="level_set">
  <ConservativePhaseComponentActivityTolerance>8e-8</ConservativePhaseComponentActivityTolerance>
  <ConservativePhaseWriteFluxArtifacts>false</ConservativePhaseWriteFluxArtifacts>
  <ConservativePhaseFluxArtifactCadenceSteps>4</ConservativePhaseFluxArtifactCadenceSteps>
  <ConservativePhaseClassifyNonprimaryComponentsAsSatellites>false</ConservativePhaseClassifyNonprimaryComponentsAsSatellites>
  <ConservativePhaseFixedFluxRegions>rim|edge|0.8|1|0.5|1|*|*</ConservativePhaseFixedFluxRegions>
</Add_equation>
)xml",
          {{"ConservativePhaseComponentActivityTolerance", "8e-8"},
           {"ConservativePhaseWriteFluxArtifacts", "false"},
           {"ConservativePhaseFluxArtifactCadenceSteps", "4"},
           {"ConservativePhaseClassifyNonprimaryComponentsAsSatellites",
            "false"},
           {"ConservativePhaseFixedFluxRegions",
            "rim|edge|0.8|1|0.5|1|*|*"}}},
  };

  for (const auto& fixture : fixtures) {
    SCOPED_TRACE(fixture.naming);
    std::unique_ptr<EquationParameters> parameters;
    ASSERT_NO_THROW(parameters = equationParametersFromText(fixture.xml));
    ASSERT_NE(parameters, nullptr);
    const auto parsed = parameters->get_parameter_list();
    for (const auto& [name, value] : fixture.expected) {
      const auto entry = parsed.find(name);
      ASSERT_NE(entry, parsed.end()) << name;
      EXPECT_EQ(entry->second, value) << name;
    }
  }
}

TEST(GeneralSimulationParameters, ParsesOptionalStartTime)
{
  tinyxml2::XMLDocument doc;
  const auto status = doc.Parse(R"xml(
<svMultiPhysicsFile>
  <GeneralSimulationParameters>
    <Continue_previous_simulation>false</Continue_previous_simulation>
    <Number_of_time_steps>2</Number_of_time_steps>
    <Save_results_to_VTK_format>true</Save_results_to_VTK_format>
    <Spectral_radius_of_infinite_time_step>0.5</Spectral_radius_of_infinite_time_step>
    <Start_saving_after_time_step>1</Start_saving_after_time_step>
    <Start_time>0.9</Start_time>
    <Time_step_size>0.000625</Time_step_size>
  </GeneralSimulationParameters>
</svMultiPhysicsFile>
)xml");
  ASSERT_EQ(status, tinyxml2::XML_SUCCESS) << doc.ErrorStr();

  auto* root = doc.FirstChildElement("svMultiPhysicsFile");
  ASSERT_NE(root, nullptr);

  GeneralSimulationParameters general;
  general.set_values(root);

  ASSERT_TRUE(general.start_time.defined());
  EXPECT_DOUBLE_EQ(general.start_time.value(), 0.9);
}

TEST(GeneralSimulationParameters, ParsesOptionalTransientTimeIntegrationScheme)
{
  tinyxml2::XMLDocument explicit_doc;
  const auto explicit_status = explicit_doc.Parse(R"xml(
<svMultiPhysicsFile>
  <GeneralSimulationParameters>
    <Continue_previous_simulation>false</Continue_previous_simulation>
    <Number_of_time_steps>2</Number_of_time_steps>
    <Save_results_to_VTK_format>true</Save_results_to_VTK_format>
    <Start_saving_after_time_step>1</Start_saving_after_time_step>
    <Time_step_size>0.000625</Time_step_size>
    <Transient_time_integration_scheme>BackwardEuler</Transient_time_integration_scheme>
  </GeneralSimulationParameters>
</svMultiPhysicsFile>
)xml");
  ASSERT_EQ(explicit_status, tinyxml2::XML_SUCCESS)
      << explicit_doc.ErrorStr();

  auto* explicit_root =
      explicit_doc.FirstChildElement("svMultiPhysicsFile");
  ASSERT_NE(explicit_root, nullptr);
  GeneralSimulationParameters explicit_general;
  ASSERT_NO_THROW(explicit_general.set_values(explicit_root));
  ASSERT_TRUE(
      explicit_general.transient_time_integration_scheme.defined());
  EXPECT_EQ(
      explicit_general.transient_time_integration_scheme.value(),
      "BackwardEuler");
  EXPECT_FALSE(
      explicit_general.spectral_radius_of_infinite_time_step.defined());
  EXPECT_DOUBLE_EQ(
      explicit_general.spectral_radius_of_infinite_time_step.value(),
      0.5);

  tinyxml2::XMLDocument omitted_doc;
  const auto omitted_status = omitted_doc.Parse(R"xml(
<svMultiPhysicsFile>
  <GeneralSimulationParameters>
    <Continue_previous_simulation>false</Continue_previous_simulation>
    <Number_of_time_steps>2</Number_of_time_steps>
    <Save_results_to_VTK_format>true</Save_results_to_VTK_format>
    <Start_saving_after_time_step>1</Start_saving_after_time_step>
    <Time_step_size>0.000625</Time_step_size>
  </GeneralSimulationParameters>
</svMultiPhysicsFile>
)xml");
  ASSERT_EQ(omitted_status, tinyxml2::XML_SUCCESS)
      << omitted_doc.ErrorStr();

  auto* omitted_root =
      omitted_doc.FirstChildElement("svMultiPhysicsFile");
  ASSERT_NE(omitted_root, nullptr);
  GeneralSimulationParameters omitted_general;
  ASSERT_NO_THROW(omitted_general.set_values(omitted_root));
  EXPECT_FALSE(
      omitted_general.transient_time_integration_scheme.defined());
  EXPECT_EQ(
      omitted_general.transient_time_integration_scheme.value(),
      "GeneralizedAlpha");
  EXPECT_FALSE(
      omitted_general.spectral_radius_of_infinite_time_step.defined());
  EXPECT_DOUBLE_EQ(
      omitted_general.spectral_radius_of_infinite_time_step.value(),
      0.5);
}

TEST(GeneralSimulationParameters, ParsesOptionalNewtonControls)
{
  tinyxml2::XMLDocument doc;
  const auto status = doc.Parse(R"xml(
<svMultiPhysicsFile>
  <GeneralSimulationParameters>
    <Continue_previous_simulation>false</Continue_previous_simulation>
    <Number_of_time_steps>1</Number_of_time_steps>
    <Save_results_to_VTK_format>false</Save_results_to_VTK_format>
    <Start_saving_after_time_step>1</Start_saving_after_time_step>
    <Time_step_size>0.0001</Time_step_size>
    <Newton_absolute_tolerance>5e-10</Newton_absolute_tolerance>
    <Newton_relative_tolerance>0</Newton_relative_tolerance>
    <Newton_max_iterations>3</Newton_max_iterations>
  </GeneralSimulationParameters>
</svMultiPhysicsFile>
)xml");
  ASSERT_EQ(status, tinyxml2::XML_SUCCESS) << doc.ErrorStr();

  auto* root = doc.FirstChildElement("svMultiPhysicsFile");
  ASSERT_NE(root, nullptr);
  GeneralSimulationParameters general;
  ASSERT_NO_THROW(general.set_values(root));
  ASSERT_TRUE(general.newton_absolute_tolerance.defined());
  ASSERT_TRUE(general.newton_relative_tolerance.defined());
  ASSERT_TRUE(general.newton_max_iterations.defined());
  EXPECT_DOUBLE_EQ(general.newton_absolute_tolerance.value(), 5.0e-10);
  EXPECT_DOUBLE_EQ(general.newton_relative_tolerance.value(), 0.0);
  EXPECT_EQ(general.newton_max_iterations.value(), 3);
}

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
  expectReviewedP1LevelSetTransportControls(level_set);
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
  expectText(fluid, "Hydrostatic_pressure_field_name", "Pressure");
  expectText(fluid, "Hydrostatic_pressure_reference_point", "0.0 0.45 0.0");
  EXPECT_EQ(fluid.FirstChildElement("Node_pressure_constraints"), nullptr);
  expectGmresSolver(fluid);
  expectOutputFields(fluid, "Spatial", {"Velocity", "Pressure"});
  expectOutputFields(fluid, "Volume_integral", {"Volume"});

  expectBoundaryCondition(fluid, "wall_left", "Dir");
  expectBoundaryCondition(fluid, "wall_right", "Dir");
  expectBoundaryCondition(fluid, "wall_bottom", "Dir");
  const auto& free_surface = expectBoundaryCondition(fluid, "free_surface", "Free_surface");
  expectText(free_surface, "Implementation", "UnfittedLevelSet");
  expectText(free_surface, "Small_cut_aggregation", "true");
  expectText(free_surface, "Level_set_field_name", "phi");
  expectText(free_surface, "Generated_interface_domain_id", "open_vessel_surface");
  expectText(free_surface, "Level_set_isovalue", "0.0");
  expectText(free_surface, "External_pressure", "0.0");
  expectText(free_surface, "Surface_tension", "0.0");
  expectText(free_surface, "Enable_cut_cell_stabilization", "true");
  expectText(free_surface, "Use_cut_metadata_scale", "false");
  EXPECT_EQ(free_surface.FirstChildElement("Cut_cell_velocity_gradient_penalty"),
            nullptr);
  expectText(free_surface, "Cut_cell_pressure_gradient_penalty", "1.0");
}

TEST(OpenVesselExamples, UnfittedLevelSetCaseBuildsOopInputs)
{
  const auto case_dir = openVesselCaseDir("unfitted_level_set");
  tinyxml2::XMLDocument doc;
  ASSERT_NO_THROW(loadXml(case_dir / "solver.xml", doc));
  auto* root = doc.FirstChildElement("svMultiPhysicsFile");
  ASSERT_NE(root, nullptr);

  auto mesh = makeTranslatorQuadMesh(
      /*register_free_surface_label=*/false);
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

TEST(OpenVesselExamples, ScopedEquationCutDomainRegistersWithUnfittedFreeSurface)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  const auto case_dir = openVesselCaseDir("unfitted_level_set");
  tinyxml2::XMLDocument doc;
  ASSERT_NO_THROW(loadXml(case_dir / "solver.xml", doc));
  auto* root = doc.FirstChildElement("svMultiPhysicsFile");
  ASSERT_NE(root, nullptr);
  auto& fluid =
      mutableChildWithAttribute(*root, "Add_equation", "type", "fluid");

  setOrAppendText(doc, fluid, "Enable_level_set_cut_domain", "true");
  setOrAppendText(doc, fluid, "Level_set_field_name", "phi");
  setOrAppendText(doc, fluid, "Generated_interface_domain_id",
                  "open_vessel_surface");
  setOrAppendText(doc, fluid, "Interface_marker", "101");
  setOrAppendText(doc, fluid, "Active_domain", "LevelSetNegative");
  setOrAppendText(doc, fluid, "Active_domain_method", "CutVolume");

  auto mesh = makeTranslatorQuadMesh();
  const std::map<std::string, std::shared_ptr<svmp::Mesh>> meshes{{"tank", mesh}};

  auto level_set_params = equationParametersFromElement(
      mutableChildWithAttribute(*root, "Add_equation", "type", "level_set"));
  svmp::FE::systems::FESystem system(mesh);
  auto level_set_module =
      application::translators::EquationTranslator::createModule(
          *level_set_params,
          system,
          meshes);
  ASSERT_TRUE(level_set_module);

  auto fluid_params = equationParametersFromElement(fluid);
  const auto requests = application::core::activeCutVolumeRequests(*fluid_params);
  const auto equation_request = std::find_if(
      requests.begin(),
      requests.end(),
      [](const application::core::ActiveCutVolumeRequest& request) {
        return request.origin ==
               application::core::ActiveCutVolumeRequestOrigin::Equation;
      });
  ASSERT_NE(equation_request, requests.end());
  const auto phi = system.findFieldByName("phi");
  ASSERT_NE(phi, svmp::FE::INVALID_FIELD_ID);
  std::vector<svmp::FE::systems::FESystem::FormCellDomainRestriction>
      restrictions{
          svmp::FE::systems::FESystem::FormCellDomainRestriction{
              .interface_marker = equation_request->requested_interface_marker,
              .side = svmp::FE::geometry::CutIntegrationSide::Negative,
              .level_set_field = phi,
              .isovalue = static_cast<svmp::FE::Real>(equation_request->isovalue),
              .enable_level_set_shape_tangent = true,
              .diagnostic =
                  "test_equation_level_level_set_cut_domain_unfitted_free_surface"}};
  ASSERT_FALSE(restrictions.empty());

  const auto previous = system.formInstallCellDomainRestrictions();
  system.setFormInstallCellDomainRestrictions(restrictions);
  std::unique_ptr<svmp::Physics::PhysicsModule> fluid_module;
  {
    const ScopedCurrentPath cwd(case_dir);
    ASSERT_NO_THROW(
        fluid_module =
            application::translators::EquationTranslator::createModule(
                *fluid_params,
                system,
                meshes));
  }
  system.setFormInstallCellDomainRestrictions(previous);

  ASSERT_TRUE(fluid_module);
  EXPECT_GT(system.cutVolumeKernelCount(
                equation_request->requested_interface_marker,
                svmp::FE::geometry::CutIntegrationSide::Negative),
            0u);
  EXPECT_NO_THROW(application::core::validateEquationLevelCutVolumeConsumer(
      system,
      *equation_request,
      equation_request->requested_interface_marker));
  EXPECT_TRUE(system.formInstallCellDomainRestrictions().empty());
#endif
}

TEST(OpenVesselExamples, SimulationBuilderRegistersEquationCutDomainWithUnfittedFreeSurface)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  ensure_mpi_initialized_for_open_vessel_builder();

  const auto case_dir = openVesselCaseDir("unfitted_level_set");
  const auto xml_path = writeBuilderRegressionXml(case_dir);

  Parameters params;
  {
    const ScopedCurrentPath cwd(case_dir);
    ASSERT_NO_THROW(params.read_xml(xml_path.string()));
    application::core::SimulationBuilder builder(params);
    auto components = builder.build();

    ASSERT_TRUE(components.fe_system);
    EXPECT_EQ(components.primary_mesh_name, "tank");
    ASSERT_EQ(components.physics_modules.size(), 2u);
    EXPECT_TRUE(components.fe_system->formInstallCellDomainRestrictions().empty());

    const auto phi = components.fe_system->findFieldByName("phi");
    ASSERT_NE(phi, svmp::FE::INVALID_FIELD_ID);
    EXPECT_TRUE(components.fe_system->fieldParticipatesInUnknownVector(phi));
    EXPECT_TRUE(components.time_history);
    ASSERT_TRUE(components.linear_solver);
    EXPECT_DOUBLE_EQ(components.linear_solver->getOptions().rel_tol, 1.0e-6);
    EXPECT_DOUBLE_EQ(components.linear_solver->getOptions().abs_tol, 1.0e-10);
  }

  std::error_code ec;
  fs::remove(xml_path, ec);
#endif
}

TEST(OpenVesselExamples,
     SimulationBuilderResolvesWetExtensionVelocityOwnedByLaterFluidEquation)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  ensure_mpi_initialized_for_open_vessel_builder();

  const auto case_dir = openVesselCaseDir("unfitted_level_set");
  const auto xml_path =
      writeWetExtensionOrderRegressionXml(case_dir, /*include_fluid_owner=*/true);

  Parameters params;
  {
    const ScopedCurrentPath cwd(case_dir);
    ASSERT_NO_THROW(params.read_xml(xml_path.string()));
    application::core::SimulationBuilder builder(params);
    auto components = builder.build();

    ASSERT_TRUE(components.fe_system);
    ASSERT_EQ(components.physics_modules.size(), 2u);
    const auto physical_velocity =
        components.fe_system->findFieldByName("Velocity");
    const auto extension_velocity =
        components.fe_system->findFieldByName("LevelSetAdvectionVelocity");
    ASSERT_NE(physical_velocity, svmp::FE::INVALID_FIELD_ID);
    ASSERT_NE(extension_velocity, svmp::FE::INVALID_FIELD_ID);
    EXPECT_TRUE(components.fe_system->fieldParticipatesInUnknownVector(
        physical_velocity));
    EXPECT_TRUE(components.fe_system->fieldParticipatesInUnknownVector(
        extension_velocity));
    EXPECT_EQ(components.fe_system->fieldRecord(physical_velocity).components,
              components.fe_system->fieldRecord(extension_velocity).components);
    EXPECT_TRUE(components.fe_system->hasOperator("equations"));

    ASSERT_TRUE(components.linear_solver);
    const auto& solver_options = components.linear_solver->getOptions();
    ASSERT_TRUE(solver_options.block_layout.has_value());
    const auto& layout = *solver_options.block_layout;
    ASSERT_EQ(layout.blocks.size(), 2u);
    EXPECT_EQ(layout.blocks[0].name,
              "VelocityAuxiliaryComputationalPrimary");
    EXPECT_EQ(layout.blocks[0].start_component, 0);
    const int expected_primary_components =
        static_cast<int>(
            components.fe_system->fieldRecord(physical_velocity).components +
            components.fe_system->fieldRecord(extension_velocity).components +
            1u);
    EXPECT_EQ(layout.blocks[0].n_components, expected_primary_components);
    EXPECT_EQ(layout.blocks[0].role,
              svmp::FE::backends::BlockRole::PrimaryField);
    EXPECT_EQ(layout.blocks[1].name, "Pressure");
    EXPECT_EQ(layout.blocks[1].start_component,
              expected_primary_components);
    EXPECT_EQ(layout.blocks[1].n_components, 1);
    EXPECT_EQ(layout.blocks[1].role,
              svmp::FE::backends::BlockRole::ConstraintField);
    ASSERT_TRUE(layout.momentum_block.has_value());
    ASSERT_TRUE(layout.constraint_block.has_value());
    EXPECT_EQ(*layout.momentum_block, 0);
    EXPECT_EQ(*layout.constraint_block, 1);
  }

  std::error_code ec;
  fs::remove(xml_path, ec);
#endif
}

TEST(OpenVesselExamples,
     SimulationDependencyPreflightResolvesTwoFluidMaterialInterfaceInEitherOrder)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  ensure_mpi_initialized_for_open_vessel_builder();

  for (const bool level_set_first : {true, false}) {
    SCOPED_TRACE(level_set_first ? "level_set_then_fluid"
                                 : "fluid_then_level_set");
    Parameters parameters;
    appendTwoFluidMaterialInterfaceEquations(
        parameters, level_set_first);

    auto mesh = makeTranslatorTriangleMesh();
    application::core::SimulationComponents components;
    components.primary_mesh_name = "triangle";
    components.primary_mesh = mesh;
    components.meshes.emplace("triangle", mesh);
    components.fe_system =
        std::make_unique<svmp::FE::systems::FESystem>(mesh);

    ASSERT_NO_THROW(
        application::core::detail::
            preflightAndPreRegisterPhysicsModuleDependencies(
                parameters, components));
    ASSERT_EQ(
        components.fe_system
            ->materialInterfaceTransportVelocityDeclarations()
            .size(),
        1u);
    ASSERT_TRUE(
        components.fe_system
            ->twoFluidAcceptedStageDiagnosticDeclarations()
            .empty());
    svmp::FE::backends::BlockLayout incomplete_owner_layout{};
    incomplete_owner_layout.blocks.push_back({"sentinel", 0, 1});
    EXPECT_THROW(
        application::core::detail::
            requireTwoFluidMaterialInterfaceSolverLayout(
                *components.fe_system,
                svmp::FE::backends::BackendKind::FSILS,
                svmp::FE::backends::SolverMethod::BlockSchur,
                incomplete_owner_layout),
        std::invalid_argument);
    ASSERT_EQ(incomplete_owner_layout.blocks.size(), 1u);
    EXPECT_EQ(incomplete_owner_layout.blocks.front().name, "sentinel");
    for (const auto* equation : parameters.equation_parameters) {
      ASSERT_NE(equation, nullptr);
      ASSERT_NO_THROW(components.physics_modules.push_back(
          application::translators::EquationTranslator::createModule(
              *equation,
              *components.fe_system,
              components.meshes)));
    }

    ASSERT_EQ(components.physics_modules.size(), 2u);
    EXPECT_TRUE(components.fe_system->hasOperator("equations"));
    const std::array<std::string_view, 6> expected_field_order{
        "level_set",
        "phase",
        "u_negative",
        "u_positive",
        "p_negative",
        "p_positive",
    };
    for (std::size_t index = 0u;
         index < expected_field_order.size(); ++index) {
      const auto name = expected_field_order[index];
      const auto field = components.fe_system->findFieldByName(name);
      ASSERT_NE(field, svmp::FE::INVALID_FIELD_ID) << name;
      EXPECT_EQ(field, static_cast<svmp::FE::FieldId>(index)) << name;
      EXPECT_EQ(components.fe_system->fieldRecord(field).name, name);
      EXPECT_TRUE(components.fe_system->fieldParticipatesInUnknownVector(
          field));
    }
    ASSERT_EQ(
        components.fe_system
            ->materialInterfaceTransportVelocityDeclarations()
            .size(),
        1u);
    EXPECT_EQ(
        components.fe_system
            ->materialInterfaceTransportVelocityDeclarations()
            .front()
            .interface_marker,
        71);
    ASSERT_EQ(
        components.fe_system
            ->twoFluidAcceptedStageDiagnosticDeclarations()
            .size(),
        1u);

    svmp::FE::backends::BlockLayout layout{};
    int component_offset = 0;
    for (const auto name : expected_field_order) {
      const auto field = components.fe_system->findFieldByName(name);
      ASSERT_NE(field, svmp::FE::INVALID_FIELD_ID) << name;
      const int components_count =
          components.fe_system->fieldRecord(field).components;
      layout.blocks.push_back(
          {std::string(name), component_offset, components_count});
      component_offset += components_count;
    }
    ASSERT_TRUE(
        application::core::detail::
            groupTwoFluidMaterialInterfaceFsilsLayout(
                *components.fe_system, layout));
    ASSERT_EQ(layout.blocks.size(), 2u);
    EXPECT_EQ(layout.blocks[0].name,
              "TwoFluidMaterialInterfaceComputationalPrimary");
    EXPECT_EQ(layout.blocks[0].start_component, 0);
    EXPECT_EQ(layout.blocks[0].n_components, 6);
    EXPECT_EQ(layout.blocks[0].role,
              svmp::FE::backends::BlockRole::PrimaryField);
    EXPECT_EQ(layout.blocks[1].name, "TwoFluidPressureConstraints");
    EXPECT_EQ(layout.blocks[1].start_component, 6);
    EXPECT_EQ(layout.blocks[1].n_components, 2);
    EXPECT_EQ(layout.blocks[1].role,
              svmp::FE::backends::BlockRole::ConstraintField);
    ASSERT_TRUE(layout.momentum_block.has_value());
    ASSERT_TRUE(layout.constraint_block.has_value());
    EXPECT_EQ(*layout.momentum_block, 0);
    EXPECT_EQ(*layout.constraint_block, 1);

    svmp::FE::backends::BlockLayout required_layout{};
    component_offset = 0;
    for (const auto name : expected_field_order) {
      const auto field = components.fe_system->findFieldByName(name);
      const int components_count =
          components.fe_system->fieldRecord(field).components;
      required_layout.blocks.push_back(
          {std::string(name), component_offset, components_count});
      component_offset += components_count;
    }
    ASSERT_NO_THROW(
        application::core::detail::
            requireTwoFluidMaterialInterfaceSolverLayout(
                *components.fe_system,
                svmp::FE::backends::BackendKind::FSILS,
                svmp::FE::backends::SolverMethod::BlockSchur,
                required_layout));
    ASSERT_EQ(required_layout.blocks.size(), layout.blocks.size());
    for (std::size_t index = 0u; index < layout.blocks.size(); ++index) {
      EXPECT_EQ(required_layout.blocks[index].name,
                layout.blocks[index].name);
      EXPECT_EQ(required_layout.blocks[index].start_component,
                layout.blocks[index].start_component);
      EXPECT_EQ(required_layout.blocks[index].n_components,
                layout.blocks[index].n_components);
      EXPECT_EQ(required_layout.blocks[index].role,
                layout.blocks[index].role);
    }
    EXPECT_EQ(required_layout.momentum_block, layout.momentum_block);
    EXPECT_EQ(required_layout.constraint_block, layout.constraint_block);

    auto unsupported_backend_layout = required_layout;
    EXPECT_THROW(
        application::core::detail::
            requireTwoFluidMaterialInterfaceSolverLayout(
                *components.fe_system,
                svmp::FE::backends::BackendKind::Eigen,
                svmp::FE::backends::SolverMethod::BlockSchur,
                unsupported_backend_layout),
        std::invalid_argument);
    ASSERT_EQ(unsupported_backend_layout.blocks.size(), 2u);
    EXPECT_EQ(unsupported_backend_layout.blocks[0].name,
              "TwoFluidMaterialInterfaceComputationalPrimary");
    EXPECT_EQ(unsupported_backend_layout.blocks[1].name,
              "TwoFluidPressureConstraints");
    EXPECT_EQ(unsupported_backend_layout.momentum_block, 0);
    EXPECT_EQ(unsupported_backend_layout.constraint_block, 1);

    auto unsupported_method_layout = required_layout;
    EXPECT_THROW(
        application::core::detail::
            requireTwoFluidMaterialInterfaceSolverLayout(
                *components.fe_system,
                svmp::FE::backends::BackendKind::FSILS,
                svmp::FE::backends::SolverMethod::GMRES,
                unsupported_method_layout),
        std::invalid_argument);
    ASSERT_EQ(unsupported_method_layout.blocks.size(), 2u);
    EXPECT_EQ(unsupported_method_layout.blocks[0].name,
              "TwoFluidMaterialInterfaceComputationalPrimary");
    EXPECT_EQ(unsupported_method_layout.blocks[1].name,
              "TwoFluidPressureConstraints");
    EXPECT_EQ(unsupported_method_layout.momentum_block, 0);
    EXPECT_EQ(unsupported_method_layout.constraint_block, 1);

    svmp::FE::backends::BlockLayout malformed_layout{};
    malformed_layout.blocks = required_layout.blocks;
    malformed_layout.blocks.push_back({"unexpected", 8, 1});
    EXPECT_THROW(
        application::core::detail::
            requireTwoFluidMaterialInterfaceSolverLayout(
                *components.fe_system,
                svmp::FE::backends::BackendKind::FSILS,
                svmp::FE::backends::SolverMethod::BlockSchur,
                malformed_layout),
        std::invalid_argument);
    ASSERT_EQ(malformed_layout.blocks.size(), 3u);
    EXPECT_EQ(malformed_layout.blocks.back().name, "unexpected");
    EXPECT_EQ(malformed_layout.blocks.back().start_component, 8);
    EXPECT_EQ(malformed_layout.blocks.back().n_components, 1);
  }
#endif
}

TEST(OpenVesselExamples,
     SimulationBuilderBuildsCompleteTwoFluidMaterialInterfaceSystem)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  ensure_mpi_initialized_for_open_vessel_builder();

  const auto case_dir =
      openVesselCaseDir("unfitted_level_set/two_fluid_builder");
  Parameters parameters;
  const ScopedCurrentPath cwd(case_dir);
  ASSERT_NO_THROW(parameters.read_xml("solver.xml"));

  const auto cut_requests =
      application::core::activeCutVolumeRequests(parameters);
  ASSERT_EQ(cut_requests.size(), 1u);
  EXPECT_EQ(
      cut_requests.front().origin,
      application::core::ActiveCutVolumeRequestOrigin::MaterialInterface);
  EXPECT_EQ(cut_requests.front().requested_interface_marker, 71);
  EXPECT_EQ(
      cut_requests.front().volume_retention,
      application::core::ActiveCutVolumeRetention::ActiveAndInactive);

  application::core::SimulationBuilder builder(parameters);
  auto components = builder.build();

  ASSERT_TRUE(components.fe_system);
  EXPECT_EQ(components.primary_mesh_name, "triangle");
  EXPECT_EQ(components.physics_modules.size(), 2u);
  EXPECT_EQ(components.fe_system->registeredFieldCount(), 6u);
  const std::array<std::string_view, 6> expected_field_order{
      "level_set",
      "phase",
      "u_negative",
      "u_positive",
      "p_negative",
      "p_positive",
  };
  for (std::size_t index = 0u;
       index < expected_field_order.size(); ++index) {
    const auto field = components.fe_system->findFieldByName(
        expected_field_order[index]);
    ASSERT_NE(field, svmp::FE::INVALID_FIELD_ID)
        << expected_field_order[index];
    EXPECT_EQ(field, static_cast<svmp::FE::FieldId>(index));
    EXPECT_TRUE(components.fe_system->fieldParticipatesInUnknownVector(
        field));
  }
  const std::vector<svmp::FE::FieldId> expected_derivative_fields{
      components.fe_system->findFieldByName("level_set"),
      components.fe_system->findFieldByName("u_negative"),
      components.fe_system->findFieldByName("u_positive"),
  };
  EXPECT_EQ(
      components.fe_system->timeDerivativeFields("equations"),
      expected_derivative_fields);
  EXPECT_TRUE(components.fe_system->hasOperator("equations"));
  EXPECT_EQ(
      components.fe_system
          ->materialInterfaceTransportVelocityDeclarations()
          .size(),
      1u);
  EXPECT_EQ(
      components.fe_system
          ->twoFluidAcceptedStageDiagnosticDeclarations()
          .size(),
      1u);

  ASSERT_TRUE(components.linear_solver);
  const auto& solver_options = components.linear_solver->getOptions();
  EXPECT_EQ(solver_options.method,
            svmp::FE::backends::SolverMethod::BlockSchur);
  EXPECT_EQ(solver_options.preconditioner,
            svmp::FE::backends::PreconditionerType::Diagonal);
  ASSERT_TRUE(solver_options.block_layout.has_value());
  const auto& layout = *solver_options.block_layout;
  ASSERT_EQ(layout.blocks.size(), 2u);
  EXPECT_EQ(layout.blocks[0].name,
            "TwoFluidMaterialInterfaceComputationalPrimary");
  EXPECT_EQ(layout.blocks[0].start_component, 0);
  EXPECT_EQ(layout.blocks[0].n_components, 6);
  EXPECT_EQ(layout.blocks[0].role,
            svmp::FE::backends::BlockRole::PrimaryField);
  EXPECT_EQ(layout.blocks[1].name, "TwoFluidPressureConstraints");
  EXPECT_EQ(layout.blocks[1].start_component, 6);
  EXPECT_EQ(layout.blocks[1].n_components, 2);
  EXPECT_EQ(layout.blocks[1].role,
            svmp::FE::backends::BlockRole::ConstraintField);
  ASSERT_TRUE(layout.momentum_block.has_value());
  ASSERT_TRUE(layout.constraint_block.has_value());
  EXPECT_EQ(*layout.momentum_block, 0);
  EXPECT_EQ(*layout.constraint_block, 1);
  EXPECT_TRUE(components.time_history);
  ASSERT_TRUE(components.primary_mesh);
  const auto& local_mesh = components.primary_mesh->local_mesh();
  const auto mesh_level_set = svmp::MeshFields::get_field_handle(
      local_mesh, svmp::EntityKind::Vertex, "level_set");
  ASSERT_NE(mesh_level_set.id, 0u);
  const auto* mesh_values =
      svmp::MeshFields::field_data_as<svmp::real_t>(
          local_mesh, mesh_level_set);
  ASSERT_NE(mesh_values, nullptr);
  bool has_full_negative_cell = false;
  bool has_full_positive_cell = false;
  for (svmp::index_t cell = 0;
       cell < static_cast<svmp::index_t>(local_mesh.n_cells()); ++cell) {
    const auto [vertices, vertex_count] =
        local_mesh.cell_vertices_span(cell);
    ASSERT_NE(vertices, nullptr);
    bool all_negative = true;
    bool all_positive = true;
    for (std::size_t local_vertex = 0;
         local_vertex < vertex_count; ++local_vertex) {
      const auto value =
          mesh_values[static_cast<std::size_t>(vertices[local_vertex])];
      all_negative = all_negative && value < 0.0;
      all_positive = all_positive && value > 0.0;
    }
    has_full_negative_cell = has_full_negative_cell || all_negative;
    has_full_positive_cell = has_full_positive_cell || all_positive;
  }
  EXPECT_TRUE(has_full_negative_cell);
  EXPECT_TRUE(has_full_positive_cell);
  const auto level_set =
      components.fe_system->findFieldByName("level_set");
  ASSERT_NE(level_set, svmp::FE::INVALID_FIELD_ID);
  const auto* entity_map =
      components.fe_system->fieldDofHandler(level_set).getEntityDofMap();
  ASSERT_NE(entity_map, nullptr);
  const auto field_offset =
      components.fe_system->fieldDofOffset(level_set);
  auto state_view =
      components.time_history->u().createGhostedReadView();
  ASSERT_TRUE(state_view);
  for (svmp::FE::GlobalIndex vertex = 0;
       vertex < components.primary_mesh->n_vertices(); ++vertex) {
    const auto vertex_dofs = entity_map->getVertexDofs(vertex);
    ASSERT_EQ(vertex_dofs.size(), 1u);
    EXPECT_DOUBLE_EQ(
        state_view->getVectorEntry(field_offset + vertex_dofs.front()),
        mesh_values[static_cast<std::size_t>(vertex)]);
  }
#endif
}

TEST(OpenVesselExamples,
     SimulationDependencyPreflightSelectsStokesSolverAfterLevelSetAlias)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  ensure_mpi_initialized_for_open_vessel_builder();

  for (const std::string_view alias :
       {std::string_view{"levelSet"},
        std::string_view{"level_set_transport"}}) {
    SCOPED_TRACE(alias);
    Parameters parameters;
    appendTwoFluidMaterialInterfaceEquations(
        parameters, /*level_set_first=*/true);
    ASSERT_EQ(parameters.equation_parameters.size(), 2u);
    parameters.equation_parameters[0]->type.set(std::string(alias));
    parameters.equation_parameters[1]->type.set("stokes");

    auto mesh = makeTranslatorTriangleMesh();
    application::core::SimulationComponents components;
    components.primary_mesh_name = "triangle";
    components.primary_mesh = mesh;
    components.meshes.emplace("triangle", mesh);
    components.fe_system =
        std::make_unique<svmp::FE::systems::FESystem>(mesh);

    ASSERT_NO_THROW(
        application::core::detail::
            preflightAndPreRegisterPhysicsModuleDependencies(
                parameters, components));
    EXPECT_EQ(components.fe_system->registeredFieldCount(), 6u);
    EXPECT_EQ(
        components.fe_system
            ->materialInterfaceTransportVelocityDeclarations()
            .size(),
        1u);
  }
#endif
}

TEST(OpenVesselExamples,
     TwoFluidSolverEnvelopeLeavesUnrelatedLayoutsUnchanged)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  auto mesh = makeTranslatorTriangleMesh();
  svmp::FE::systems::FESystem system(mesh);
  svmp::FE::backends::BlockLayout layout{};
  layout.blocks.push_back({"unrelated", 0, 1});

  ASSERT_NO_THROW(
      application::core::detail::
          requireTwoFluidMaterialInterfaceSolverLayout(
              system,
              svmp::FE::backends::BackendKind::Eigen,
              svmp::FE::backends::SolverMethod::GMRES,
              layout));
  ASSERT_EQ(layout.blocks.size(), 1u);
  EXPECT_EQ(layout.blocks.front().name, "unrelated");
  EXPECT_EQ(layout.blocks.front().start_component, 0);
  EXPECT_EQ(layout.blocks.front().n_components, 1);
  EXPECT_FALSE(layout.momentum_block.has_value());
  EXPECT_FALSE(layout.constraint_block.has_value());
#endif
}

void expectMaterialInterfaceDependencyPreflightRejectedWithoutMutation(
    Parameters& parameters,
    std::string_view expected_diagnostic)
{
  auto mesh = makeTranslatorTriangleMesh();
  application::core::SimulationComponents components;
  components.primary_mesh_name = "triangle";
  components.primary_mesh = mesh;
  components.meshes.emplace("triangle", mesh);
  components.fe_system =
      std::make_unique<svmp::FE::systems::FESystem>(mesh);

  try {
    application::core::detail::
        preflightAndPreRegisterPhysicsModuleDependencies(
            parameters, components);
    FAIL() << "Expected material-interface dependency preflight rejection: "
           << expected_diagnostic;
  } catch (const std::exception& error) {
    EXPECT_NE(
        std::string_view(error.what()).find(expected_diagnostic),
        std::string_view::npos)
        << error.what();
  }
  ASSERT_TRUE(components.fe_system);
  EXPECT_EQ(components.fe_system->fieldMap().numFields(), 0u);
  EXPECT_TRUE(
      components.fe_system
          ->materialInterfaceTransportVelocityDeclarations()
          .empty());
  EXPECT_TRUE(
      components.fe_system
          ->twoFluidAcceptedStageDiagnosticDeclarations()
          .empty());
  EXPECT_FALSE(components.fe_system->hasOperator("equations"));
  EXPECT_TRUE(components.physics_modules.empty());
}

TEST(OpenVesselExamples,
     MaterialInterfaceDependencyPreflightRejectsSolverEnvelopeBeforeMutation)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  ensure_mpi_initialized_for_open_vessel_builder();

  const auto fluid_equation = [](Parameters& parameters) {
    for (auto* equation : parameters.equation_parameters) {
      if (equation != nullptr && equation->type.defined() &&
          equation->type.value() == "fluid") {
        return equation;
      }
    }
    return static_cast<EquationParameters*>(nullptr);
  };

  Parameters unsupported_backend;
  appendTwoFluidMaterialInterfaceEquations(
      unsupported_backend, /*level_set_first=*/true);
  auto* backend_owner = fluid_equation(unsupported_backend);
  ASSERT_NE(backend_owner, nullptr);
  backend_owner->linear_solver.linear_algebra.type.set("eigen");
  expectMaterialInterfaceDependencyPreflightRejectedWithoutMutation(
      unsupported_backend,
      "two_fluid_material_interface_solver_requires_fsils");

  Parameters unsupported_method;
  appendTwoFluidMaterialInterfaceEquations(
      unsupported_method, /*level_set_first=*/false);
  auto* method_owner = fluid_equation(unsupported_method);
  ASSERT_NE(method_owner, nullptr);
  method_owner->linear_solver.type.set("GMRES");
  expectMaterialInterfaceDependencyPreflightRejectedWithoutMutation(
      unsupported_method,
      "two_fluid_material_interface_solver_requires_block_schur");
#endif
}

TEST(OpenVesselExamples,
     MaterialInterfaceDependencyPreflightRejectsAdditionalEquationBeforeMutation)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  ensure_mpi_initialized_for_open_vessel_builder();

  Parameters parameters;
  appendTwoFluidMaterialInterfaceEquations(
      parameters, /*level_set_first=*/true);
  auto additional_equation = std::make_unique<EquationParameters>();
  additional_equation->type.set("poisson");
  parameters.equation_parameters.push_back(
      additional_equation.release());

  expectMaterialInterfaceDependencyPreflightRejectedWithoutMutation(
      parameters,
      "two_fluid_material_interface_solver_requires_exact_equation_pair");
#endif
}

TEST(OpenVesselExamples,
     MaterialInterfaceDependencyPreflightRejectsMissingOwnersBeforeMutation)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  ensure_mpi_initialized_for_open_vessel_builder();

  Parameters missing_fluid;
  missing_fluid.equation_parameters.push_back(
      materialInterfaceLevelSetEquation().release());
  expectMaterialInterfaceDependencyPreflightRejectedWithoutMutation(
      missing_fluid,
      "material_interface_dependency_missing_two_fluid_owner");

  Parameters missing_transport;
  missing_transport.equation_parameters.push_back(
      incompressibleTwoFluidEquation().release());
  expectMaterialInterfaceDependencyPreflightRejectedWithoutMutation(
      missing_transport,
      "two_fluid_dependency_missing_material_interface_transport");
#endif
}

TEST(OpenVesselExamples,
     MaterialInterfaceDependencyPreflightRejectsMismatchAndAmbiguityBeforeMutation)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  ensure_mpi_initialized_for_open_vessel_builder();

  Parameters marker_mismatch;
  appendTwoFluidMaterialInterfaceEquations(
      marker_mismatch, /*level_set_first=*/true);
  marker_mismatch.equation_parameters.front()->set_extra_parameter_value(
      "Material_interface_marker", "72");
  marker_mismatch.equation_parameters.front()->set_extra_parameter_value(
      "Interface_kinematic_marker", "72");
  expectMaterialInterfaceDependencyPreflightRejectedWithoutMutation(
      marker_mismatch,
      "material_interface_dependency_marker_mismatch");

  Parameters ambiguous_owner;
  ambiguous_owner.equation_parameters.push_back(
      materialInterfaceLevelSetEquation().release());
  ambiguous_owner.equation_parameters.push_back(
      incompressibleTwoFluidEquation().release());
  ambiguous_owner.equation_parameters.push_back(
      incompressibleTwoFluidEquation().release());
  expectMaterialInterfaceDependencyPreflightRejectedWithoutMutation(
      ambiguous_owner,
      "ambiguous_material_interface_dependency_pair");
#endif
}

TEST(OpenVesselExamples,
     MaterialInterfaceDependencyRegistersSharedNonhomogeneousVelocityBoundary)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  ensure_mpi_initialized_for_open_vessel_builder();

  Parameters parameters;
  appendTwoFluidMaterialInterfaceEquations(
      parameters, /*level_set_first=*/true);
  auto fluid = std::find_if(
      parameters.equation_parameters.begin(),
      parameters.equation_parameters.end(),
      [](const auto* equation) {
        return equation != nullptr && equation->type.defined() &&
               equation->type.value() == "fluid";
      });
  ASSERT_NE(fluid, parameters.equation_parameters.end());
  auto boundary = std::make_unique<BoundaryConditionParameters>();
  boundary->name.set("wall");
  boundary->type.set("Dir");
  boundary->value.set_raw_value(1.0);
  (*fluid)->boundary_conditions.push_back(boundary.release());

  auto mesh = makeTranslatorTriangleMesh();
  mesh->base().register_label("wall", 9);
  mesh->base().set_boundary_label(0, 9);
  application::core::SimulationComponents components;
  components.primary_mesh_name = "triangle";
  components.primary_mesh = mesh;
  components.meshes.emplace("triangle", mesh);
  components.fe_system =
      std::make_unique<svmp::FE::systems::FESystem>(mesh);

  ASSERT_NO_THROW(
      application::core::detail::
          preflightAndPreRegisterPhysicsModuleDependencies(
              parameters, components));
  for (const auto* equation : parameters.equation_parameters) {
    ASSERT_NE(equation, nullptr);
    ASSERT_NO_THROW(components.physics_modules.push_back(
        application::translators::EquationTranslator::createModule(
            *equation,
            *components.fe_system,
            components.meshes)));
  }
  ASSERT_TRUE(components.fe_system);
  EXPECT_EQ(components.physics_modules.size(), 2u);
  EXPECT_TRUE(components.fe_system->hasOperator("equations"));
  const auto u_negative =
      components.fe_system->findFieldByName("u_negative");
  const auto u_positive =
      components.fe_system->findFieldByName("u_positive");
  ASSERT_NE(u_negative, svmp::FE::INVALID_FIELD_ID);
  ASSERT_NE(u_positive, svmp::FE::INVALID_FIELD_ID);
  const auto shared_descriptors = static_cast<std::size_t>(std::count_if(
      components.fe_system->boundaryConditionDescriptors().begin(),
      components.fe_system->boundaryConditionDescriptors().end(),
      [&](const auto& descriptor) {
        return descriptor.boundary_marker == 9 &&
               (descriptor.primary_variable.field_id == u_negative ||
                descriptor.primary_variable.field_id == u_positive);
      }));
  EXPECT_EQ(shared_descriptors, 4u);
#endif
}

TEST(OpenVesselExamples,
     MaterialInterfaceDependencyPreflightRejectsDuplicateVelocityBoundaryBeforeMutation)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  ensure_mpi_initialized_for_open_vessel_builder();

  Parameters parameters;
  appendTwoFluidMaterialInterfaceEquations(
      parameters, /*level_set_first=*/false);
  auto fluid = std::find_if(
      parameters.equation_parameters.begin(),
      parameters.equation_parameters.end(),
      [](const auto* equation) {
        return equation != nullptr && equation->type.defined() &&
               equation->type.value() == "fluid";
      });
  ASSERT_NE(fluid, parameters.equation_parameters.end());
  for (int copy = 0; copy < 2; ++copy) {
    auto boundary = std::make_unique<BoundaryConditionParameters>();
    boundary->name.set("wall");
    boundary->type.set("Dir");
    boundary->value.set_raw_value(0.0);
    (*fluid)->boundary_conditions.push_back(boundary.release());
  }

  auto mesh = makeTranslatorTriangleMesh();
  mesh->base().register_label("wall", 9);
  mesh->base().set_boundary_label(0, 9);
  application::core::SimulationComponents components;
  components.primary_mesh_name = "triangle";
  components.primary_mesh = mesh;
  components.meshes.emplace("triangle", mesh);
  components.fe_system =
      std::make_unique<svmp::FE::systems::FESystem>(mesh);

  try {
    application::core::detail::
        preflightAndPreRegisterPhysicsModuleDependencies(
            parameters, components);
    FAIL() << "Expected duplicate phase boundary to fail preflight";
  } catch (const std::exception& error) {
    EXPECT_NE(
        std::string_view(error.what()).find(
            "unsupported_two_fluid_duplicate_velocity_boundary_marker"),
        std::string_view::npos)
        << error.what();
  }
  ASSERT_TRUE(components.fe_system);
  EXPECT_EQ(components.fe_system->registeredFieldCount(), 0u);
  EXPECT_TRUE(
      components.fe_system
          ->materialInterfaceTransportVelocityDeclarations()
          .empty());
  EXPECT_TRUE(
      components.fe_system
          ->twoFluidAcceptedStageDiagnosticDeclarations()
          .empty());
  EXPECT_FALSE(components.fe_system->hasOperator("equations"));
  EXPECT_TRUE(components.physics_modules.empty());
#endif
}

TEST(OpenVesselExamples,
     MaterialInterfaceDependencyPreflightRejectsNonfiniteBodyForceBeforeMutation)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  ensure_mpi_initialized_for_open_vessel_builder();

  Parameters parameters;
  appendTwoFluidMaterialInterfaceEquations(
      parameters, /*level_set_first=*/true);
  auto fluid = std::find_if(
      parameters.equation_parameters.begin(),
      parameters.equation_parameters.end(),
      [](const auto* equation) {
        return equation != nullptr && equation->type.defined() &&
               equation->type.value() == "fluid";
      });
  ASSERT_NE(fluid, parameters.equation_parameters.end());
  ASSERT_NE((*fluid)->default_domain, nullptr);
  (*fluid)->default_domain->force_x.set_raw_value(
      std::numeric_limits<double>::infinity());

  expectMaterialInterfaceDependencyPreflightRejectedWithoutMutation(
      parameters,
      "unsupported_two_fluid_nonfinite_body_force");
#endif
}

TEST(OpenVesselExamples,
     MaterialInterfaceDependencyPreflightRejectsNonfinitePressureJumpBeforeMutation)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  ensure_mpi_initialized_for_open_vessel_builder();

  Parameters parameters;
  appendTwoFluidMaterialInterfaceEquations(
      parameters, /*level_set_first=*/false);
  auto fluid = std::find_if(
      parameters.equation_parameters.begin(),
      parameters.equation_parameters.end(),
      [](const auto* equation) {
        return equation != nullptr && equation->type.defined() &&
               equation->type.value() == "fluid";
      });
  ASSERT_NE(fluid, parameters.equation_parameters.end());
  (*fluid)->set_extra_parameter_value(
      "Prescribed_pressure_jump", "inf");

  expectMaterialInterfaceDependencyPreflightRejectedWithoutMutation(
      parameters,
      "unsupported_two_fluid_nonfinite_prescribed_pressure_jump");
#endif
}

TEST(OpenVesselExamples,
     MaterialInterfaceDependencyPreflightRejectsIncompatibleExistingFieldWithoutMutation)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  ensure_mpi_initialized_for_open_vessel_builder();

  Parameters parameters;
  appendTwoFluidMaterialInterfaceEquations(
      parameters, /*level_set_first=*/true);
  auto mesh = makeTranslatorTriangleMesh();
  application::core::SimulationComponents components;
  components.primary_mesh_name = "triangle";
  components.primary_mesh = mesh;
  components.meshes.emplace("triangle", mesh);
  components.fe_system =
      std::make_unique<svmp::FE::systems::FESystem>(mesh);
  auto scalar_space = svmp::FE::spaces::SpaceFactory::create_h1(
      svmp::FE::ElementType::Triangle3, /*order=*/1);
  const auto existing = components.fe_system->addField(
      svmp::FE::systems::FieldSpec{
          .name = "u_negative",
          .space = scalar_space,
          .components = 1,
          .source_kind =
              svmp::FE::systems::FieldSourceKind::PrescribedData,
      });
  ASSERT_EQ(existing, 0);

  EXPECT_THROW(
      application::core::detail::
          preflightAndPreRegisterPhysicsModuleDependencies(
              parameters, components),
      std::invalid_argument);
  EXPECT_EQ(components.fe_system->registeredFieldCount(), 1u);
  EXPECT_EQ(components.fe_system->findFieldByName("u_negative"), existing);
  EXPECT_EQ(
      components.fe_system->findFieldByName("level_set"),
      svmp::FE::INVALID_FIELD_ID);
  EXPECT_EQ(
      components.fe_system->findFieldByName("phase"),
      svmp::FE::INVALID_FIELD_ID);
  EXPECT_TRUE(
      components.fe_system
          ->materialInterfaceTransportVelocityDeclarations()
          .empty());
#endif
}

TEST(OpenVesselExamples,
     SimulationBuilderPreflightsFittedCapabilityBeforeWetExtensionMutation)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  ensure_mpi_initialized_for_open_vessel_builder();

  const auto case_dir = openVesselCaseDir("unfitted_level_set");
  const std::vector<std::pair<std::string_view, std::string_view>> exclusions{
      {"SurfaceStress", "fitted-ALE SurfaceStress is not yet qualified"},
      {"GeneratedCurvatureTraction",
       "GeneratedCurvatureTraction is available only for unfitted"},
      {"KinematicAreaGradientTraction",
       "KinematicAreaGradientTraction is available only for unfitted"},
      {"PrescribedAngle", "prescribed fitted contact angles are unsupported"},
      {"DynamicRenE", "supported only for sharp unfitted level-set"},
  };
  for (const auto& [request, diagnostic] : exclusions) {
    SCOPED_TRACE(request);
    const auto xml_path = writeWetExtensionOrderRegressionXml(
        case_dir,
        /*include_fluid_owner=*/true,
        request);

    Parameters params;
    {
      const ScopedCurrentPath cwd(case_dir);
      ASSERT_NO_THROW(params.read_xml(xml_path.string()));
    }

    auto mesh = makeTranslatorQuadMesh();
    application::core::SimulationComponents components;
    components.meshes.emplace("tank", mesh);
    components.fe_system =
        std::make_unique<svmp::FE::systems::FESystem>(mesh);

    try {
      application::core::detail::
          preflightAndPreRegisterPhysicsModuleDependencies(params, components);
      FAIL() << "Expected excluded fitted request to fail preflight";
    } catch (const std::invalid_argument& error) {
      EXPECT_NE(std::string(error.what()).find(diagnostic),
                std::string::npos)
          << error.what();
    }

    ASSERT_TRUE(components.fe_system);
    EXPECT_EQ(components.fe_system->fieldMap().numFields(), 0u);
    EXPECT_TRUE(components.fe_system->formulationRecords().empty());
    EXPECT_TRUE(
        components.fe_system->boundaryConditionDescriptors().empty());
    EXPECT_TRUE(
        components.fe_system->meshNormalBoundaryConstraints().empty());
    EXPECT_TRUE(
        components.fe_system->meshTangentialBoundaryPolicies().empty());
    EXPECT_TRUE(
        components.fe_system->meshNormalBoundaryConstraintHistory().empty());
    EXPECT_TRUE(
        components.fe_system->meshTangentialBoundaryPolicyHistory().empty());
    EXPECT_FALSE(components.fe_system->hasOperator("equations"));
    EXPECT_FALSE(components.fe_system->hasOperator("level_set"));
    EXPECT_TRUE(components.physics_modules.empty());

    std::error_code ec;
    fs::remove(xml_path, ec);
  }
#endif
}

TEST(OpenVesselExamples,
     SimulationBuilderWetExtensionStillRejectsGenuinelyMissingVelocityOwner)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  ensure_mpi_initialized_for_open_vessel_builder();

  const auto case_dir = openVesselCaseDir("unfitted_level_set");
  const auto xml_path = writeWetExtensionOrderRegressionXml(
      case_dir, /*include_fluid_owner=*/false);

  Parameters params;
  const ScopedCurrentPath cwd(case_dir);
  ASSERT_NO_THROW(params.read_xml(xml_path.string()));
  application::core::SimulationBuilder builder(params);
  try {
    (void)builder.build();
    FAIL() << "Expected missing physical velocity owner to be rejected";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find(
                  "references unknown physical velocity field 'Velocity'"),
              std::string::npos)
        << error.what();
  }

  std::error_code ec;
  fs::remove(xml_path, ec);
#endif
}

TEST(OpenVesselExamples, UnfittedFreeSurfaceBuilderAcceptsSuppliedCurvatureSurfaceTension)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  ensure_mpi_initialized_for_open_vessel_builder();

  const auto case_dir = openVesselCaseDir("unfitted_level_set");
  const auto xml_path = writeUnfittedGuardRegressionXml(
      case_dir,
      "supplied_curvature_surface_tension",
      {{"Surface_tension", "0.0728"},
       {"Use_level_set_curvature", "false"},
       {"Curvature", "2.0"}});

  Parameters params;
  {
    const ScopedCurrentPath cwd(case_dir);
    ASSERT_NO_THROW(params.read_xml(xml_path.string()));
    application::core::SimulationBuilder builder(params);
    auto components = builder.build();

    ASSERT_TRUE(components.fe_system);
    const auto phi = components.fe_system->findFieldByName("phi");
    ASSERT_NE(phi, svmp::FE::INVALID_FIELD_ID);

    svmp::FE::interfaces::GeneratedInterfaceMarkerKey key{};
    key.source = svmp::FE::interfaces::LevelSetInterfaceSource::fromField(phi);
    key.domain_id = "open_vessel_surface";
    const int interface_marker =
        svmp::FE::interfaces::stableGeneratedInterfaceMarker(key);

    const auto& equations =
        components.fe_system->operatorDefinition("equations");
    const auto has_interface_traction =
        std::any_of(equations.interface_faces.begin(),
                    equations.interface_faces.end(),
                    [interface_marker](const auto& term) {
                      return term.marker == interface_marker;
                    });

    EXPECT_TRUE(has_interface_traction);
    EXPECT_TRUE(components.fe_system->formInstallCellDomainRestrictions().empty());
  }

  std::error_code ec;
  fs::remove(xml_path, ec);
#endif
}

TEST(OpenVesselExamples, UnfittedFreeSurfaceBuilderRejectsNitscheKinematics)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  ensure_mpi_initialized_for_open_vessel_builder();

  const auto case_dir = openVesselCaseDir("unfitted_level_set");
  const auto xml_path = writeUnfittedGuardRegressionXml(
      case_dir,
      "nitsche_kinematics",
      {{"Kinematic_enforcement", "Nitsche"},
       {"Normal_kinematic_policy", "MatchFluidNormalVelocity"}});

  Parameters params;
  const ScopedCurrentPath cwd(case_dir);
  ASSERT_NO_THROW(params.read_xml(xml_path.string()));
  application::core::SimulationBuilder builder(params);
  try {
    (void)builder.build();
    FAIL() << "Expected unfitted Nitsche kinematics to be rejected";
  } catch (const std::invalid_argument& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("Eulerian level-set transport"), std::string::npos)
        << message;
    EXPECT_NE(message.find("Penalty/Nitsche enforcement"),
              std::string::npos)
        << message;
  }

  std::error_code ec;
  fs::remove(xml_path, ec);
#endif
}

TEST(OpenVesselExamples, UnfittedFreeSurfaceBuilderRejectsRawCurvatureSurfaceTension)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  ensure_mpi_initialized_for_open_vessel_builder();

  const auto case_dir = openVesselCaseDir("unfitted_level_set");
  const auto xml_path = writeUnfittedGuardRegressionXml(
      case_dir,
      "raw_curvature_surface_tension",
      {{"Surface_tension", "0.0728"},
       {"Surface_tension_form", "CurvatureTraction"},
       {"Use_level_set_curvature", "true"}});

  Parameters params;
  const ScopedCurrentPath cwd(case_dir);
  ASSERT_NO_THROW(params.read_xml(xml_path.string()));
  application::core::SimulationBuilder builder(params);
  try {
    (void)builder.build();
    FAIL() << "Expected raw level-set curvature surface tension to be rejected";
  } catch (const std::invalid_argument& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("raw level-set curvature"), std::string::npos);
    EXPECT_NE(message.find("not validated"), std::string::npos);
  }

  std::error_code ec;
  fs::remove(xml_path, ec);
#endif
}

TEST(OpenVesselExamples, UnfittedFreeSurfaceBuilderRejectsMultipleActiveDomains)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
  GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
  ensure_mpi_initialized_for_open_vessel_builder();

  const auto case_dir = openVesselCaseDir("unfitted_level_set");
  const auto xml_path = writeUnfittedGuardRegressionXml(
      case_dir,
      "multiple_active_domains",
      {},
      /*append_second_active_free_surface=*/true);

  Parameters params;
  const ScopedCurrentPath cwd(case_dir);
  ASSERT_NO_THROW(params.read_xml(xml_path.string()));
  application::core::SimulationBuilder builder(params);
  try {
    (void)builder.build();
    FAIL() << "Expected multiple active-domain free surfaces to be rejected";
  } catch (const std::invalid_argument& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("at most one active-domain free surface"),
              std::string::npos);
  }

  std::error_code ec;
  fs::remove(xml_path, ec);
#endif
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
    bool pressure_constraints;
    std::string level_set_velocity_source;
    std::vector<std::string> faces;
  };

  const std::vector<CaseExpectation> cases = {
      {"fitted_ale",
       "spheric_test10_lateral_water_1x",
       true,
       false,
       false,
       true,
       "",
       {"wall_left", "wall_right", "wall_bottom", "wall_front", "wall_back",
        "free_surface"}},
      {"unfitted_level_set",
       "spheric_test10_lateral_water_1x",
       false,
       true,
       false,
       false,
       "prescribed_data",
       {"wall_left", "wall_right", "wall_bottom", "wall_front", "wall_back",
        "wall_top"}},
      {"unfitted_level_set",
       "spheric_test05_wet_bed_d18",
       false,
       false,
       false,
       false,
       "prescribed_data",
       {"wall_left", "wall_right", "wall_bottom", "wall_front", "wall_back",
        "wall_top"}},
      {"unfitted_level_set",
       "spheric_test05_wet_bed_d38",
       false,
       false,
       false,
       false,
       "prescribed_data",
       {"wall_left", "wall_right", "wall_bottom", "wall_front", "wall_back",
        "wall_top"}},
      {"unfitted_level_set",
       "spheric_test02_dambreak_obstacle",
       false,
       false,
       true,
       false,
       "prescribed_data",
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
    if (expected.case_name == "spheric_test05_wet_bed_d18" ||
        expected.case_name == "spheric_test05_wet_bed_d38") {
      expectText(mesh, "Ghost_layers", "3");
    }
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
    if (expected.pressure_constraints) {
      expectReferencedFileExists(case_dir, child(fluid, "Node_pressure_constraints"),
                                 "Values_file_path");
    } else {
      EXPECT_EQ(fluid.FirstChildElement("Node_pressure_constraints"), nullptr);
    }
    for (const auto& face : expected.faces) {
      if (startsWith(face, "wall_") &&
          (expected.top_wall_bc || face != "wall_top")) {
        expectBoundaryCondition(fluid, face, "Dir");
      }
    }
    if (!expected.top_wall_bc &&
        std::find(expected.faces.begin(), expected.faces.end(), "wall_top") !=
            expected.faces.end()) {
      EXPECT_FALSE(hasChildWithAttribute(fluid, "Add_BC", "name", "wall_top"))
          << "wall_top must remain open (natural traction)";
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
      const auto expected_jit_options =
          std::string("jit=true; jit_specialization=true");
      // SPHERIC Test 05 is a published two-dimensional benchmark.  The thin
      // 3-D extrusion must therefore use symmetry/free-slip on its artificial
      // front/back planes instead of adding spanwise-wall drag.  Retain the
      // physical end-wall and bottom no-slip assumptions separately.
      const auto& wall_front =
          expectBoundaryCondition(fluid, "wall_front", "Dir");
      const auto& wall_back =
          expectBoundaryCondition(fluid, "wall_back", "Dir");
      expectText(wall_front, "Effective_direction", "0 0 1");
      expectText(wall_back, "Effective_direction", "0 0 1");
      EXPECT_EQ(expectBoundaryCondition(fluid, "wall_left", "Dir")
                    .FirstChildElement("Effective_direction"),
                nullptr);
      EXPECT_EQ(expectBoundaryCondition(fluid, "wall_right", "Dir")
                    .FirstChildElement("Effective_direction"),
                nullptr);
      EXPECT_EQ(expectBoundaryCondition(fluid, "wall_bottom", "Dir")
                    .FirstChildElement("Effective_direction"),
                nullptr);
      const auto& level_set =
          childWithAttribute(*root, "Add_equation", "type", "level_set");
      expectBoundaryCondition(level_set, "wall_top", "LevelSetOutflow");
      expectText(level_set, "Module_options", expected_jit_options);
      expectText(level_set,
                 "Wet_extension_advection_velocity_method",
                 "wall_compatible_normal");
      // Curved continuous-P1 redistancing currently fails closed rather than
      // converging generally, so the production deck leaves it disabled.
      // Bounded volume repair remains an explicitly monitored fallback.
      expectText(level_set, "Enable_reinitialization", "false");
      expectText(level_set, "Enable_volume_correction", "true");
      expectText(level_set, "Volume_correction_minimum_relative_error", "1.0e-6");
      expectText(level_set,
                 "Volume_correction_maximum_interface_displacement_fraction",
                 "0.05");
      expectText(fluid, "Module_options", expected_jit_options);
      expectText(free_surface, "Small_cut_aggregation", "true");
      expectText(free_surface, "Active_domain", "LevelSetNegative");
      expectText(free_surface, "Active_domain_method", "CutVolume");
      EXPECT_EQ(free_surface.FirstChildElement("Enable_velocity_extension"),
                nullptr);
      EXPECT_EQ(
          free_surface.FirstChildElement("Velocity_extension_diffusivity"),
          nullptr);
      expectEigenDirectSolver(fluid);
      EXPECT_EQ(free_surface.FirstChildElement("Kinematic_enforcement"),
                nullptr);
      expectText(fluid, "Hydrostatic_pressure_field_name", "Pressure");
      EXPECT_EQ(fluid.FirstChildElement("Node_pressure_constraints"), nullptr);
      EXPECT_TRUE(fileContains(case_dir / "pressure_gauge.csv",
                               "node_id,pressure"));
      EXPECT_TRUE(fileContains(case_dir / "benchmark.json",
                               "\"pressure_gauge\""));
      EXPECT_TRUE(fileContains(case_dir / "benchmark.json",
                               "\"pressure_constraint_enabled\": false"));
    }

    if (expected.fitted) {
      expectText(fluid, "Enable_ALE", "true");
      const auto& mesh_motion =
          childWithAttribute(*root, "Add_equation", "type", "mesh_motion");
      expectText(mesh_motion, "Model", "Harmonic");
    } else {
      const auto& level_set =
          childWithAttribute(*root, "Add_equation", "type", "level_set");
      expectReviewedP1LevelSetTransportControls(level_set);
      expectText(level_set, "Level_set_field_name", "phi");
      expectText(level_set, "Velocity_source",
                 expected.level_set_velocity_source);
      expectText(level_set, "Volume_correction_minimum_relative_error", "1.0e-6");
      expectText(level_set,
                 "Volume_correction_maximum_interface_displacement_fraction",
                 "0.05");
      expectText(fluid, "Hydrostatic_pressure_field_name", "Pressure");
      EXPECT_EQ(fluid.FirstChildElement("Node_pressure_constraints"), nullptr);
      expectText(free_surface, "Generated_interface_domain_id",
                 "open_vessel_surface");
      expectText(level_set,
                 "Wet_extension_advection_velocity_method",
                 "wall_compatible_normal");
    }
  }
}
