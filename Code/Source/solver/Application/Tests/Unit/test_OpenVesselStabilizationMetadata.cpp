#include <gtest/gtest.h>

#include "tinyxml2.h"

#include <array>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

namespace fs = std::filesystem;

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
  const auto* element = parent.FirstChildElement(name);
  if (element == nullptr || element->GetText() == nullptr) {
    return {};
  }
  return element->GetText();
}

void expectText(const tinyxml2::XMLElement& parent,
                const char* name,
                std::string_view expected)
{
  EXPECT_EQ(text(parent, name), expected) << "element " << name;
}

} // namespace

TEST(OpenVesselStabilizationMetadata, D18D38UseActiveCutMetadataScale)
{
  const auto root_dir =
      repositoryRoot() / "tests/cases/fluid/open_vessel_free_surface" /
      "unfitted_level_set";
  constexpr std::array<std::string_view, 2> case_names{{
      "spheric_test05_wet_bed_d18",
      "spheric_test05_wet_bed_d38",
  }};

  for (const auto case_name : case_names) {
    SCOPED_TRACE(case_name);

    tinyxml2::XMLDocument doc;
    const auto solver_xml = root_dir / std::string(case_name) / "solver.xml";
    ASSERT_EQ(doc.LoadFile(solver_xml.string().c_str()), tinyxml2::XML_SUCCESS)
        << doc.ErrorStr();

    const auto* root = doc.FirstChildElement("svMultiPhysicsFile");
    ASSERT_NE(root, nullptr);
    const auto& fluid = childWithAttribute(*root, "Add_equation", "type", "fluid");
    const auto& free_surface =
        childWithAttribute(fluid, "Add_BC", "name", "free_surface");

    expectText(free_surface, "Implementation", "UnfittedLevelSet");
    expectText(free_surface, "Generated_interface_domain_id", "open_vessel_surface");
    expectText(free_surface, "Active_domain", "LevelSetNegative");
    expectText(free_surface, "Active_domain_method", "CutVolume");
    expectText(free_surface, "Enable_cut_cell_stabilization", "true");
    expectText(free_surface, "Use_cut_metadata_scale", "true");
    expectText(free_surface, "Cut_cell_velocity_gradient_penalty", "1.0");
    expectText(free_surface, "Cut_cell_pressure_gradient_penalty", "1.0");
  }
}
