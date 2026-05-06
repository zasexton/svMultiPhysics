#include <gtest/gtest.h>

#include "Application/Translators/EquationTranslator.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"
#include "Parameters.h"
#include "Physics/Core/EquationModuleInput.h"

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::shared_ptr<svmp::Mesh> buildTranslatorMesh()
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
  base->build_from_arrays(/*spatial_dim=*/2, x_ref, cell2vertex_offsets, cell2vertex, {shape});
  base->set_vertex_gids({10, 20, 30, 40});
  base->finalize();

  return svmp::create_mesh(std::move(base));
}

std::unique_ptr<EquationParameters> parseEquationXml(const char* xml)
{
  tinyxml2::XMLDocument doc;
  const auto parse_status = doc.Parse(xml);
  if (parse_status != tinyxml2::XML_SUCCESS) {
    throw std::runtime_error(doc.ErrorStr());
  }

  auto* elem = doc.FirstChildElement(EquationParameters::xml_element_name_.c_str());
  if (elem == nullptr) {
    throw std::runtime_error("missing Add_equation element");
  }

  const char* type = nullptr;
  elem->QueryStringAttribute("type", &type);
  if (type == nullptr) {
    throw std::runtime_error("missing Add_equation type");
  }

  auto params = std::make_unique<EquationParameters>();
  params->type.set(std::string(type));
  params->set_values(elem);
  return params;
}

std::map<std::string, std::shared_ptr<svmp::Mesh>> singleMeshMap(const std::shared_ptr<svmp::Mesh>& mesh)
{
  return {{"mesh", mesh}};
}

} // namespace

TEST(EquationTranslatorNodePressureConstraints, ParsesXmlBlock)
{
  auto params = parseEquationXml(R"xml(
<Add_equation type="darcy">
  <Node_pressure_constraints>
    <Id_type>Global_vertex_gid</Id_type>
    <Values_file_path>known_pressure_nodes.csv</Values_file_path>
  </Node_pressure_constraints>
</Add_equation>
)xml");

  ASSERT_TRUE(params->node_pressure_constraints.value_set);
  EXPECT_EQ(params->node_pressure_constraints.id_type.value(), "Global_vertex_gid");
  EXPECT_EQ(params->node_pressure_constraints.values_file_path.value(), "known_pressure_nodes.csv");
}

TEST(EquationTranslatorNodePressureConstraints, ParsesXmlBlockWithDefaultIdType)
{
  auto params = parseEquationXml(R"xml(
<Add_equation type="darcy">
  <Node_pressure_constraints>
    <Values_file_path>known_pressure_nodes.csv</Values_file_path>
  </Node_pressure_constraints>
</Add_equation>
)xml");

  ASSERT_TRUE(params->node_pressure_constraints.value_set);
  EXPECT_EQ(params->node_pressure_constraints.id_type.value(), "Global_vertex_gid");
  EXPECT_EQ(params->node_pressure_constraints.values_file_path.value(), "known_pressure_nodes.csv");
}

TEST(EquationTranslatorNodePressureConstraints, ParserRejectsMissingValuesFilePath)
{
  EXPECT_THROW((void)parseEquationXml(R"xml(
<Add_equation type="darcy">
  <Node_pressure_constraints>
    <Id_type>Global_vertex_gid</Id_type>
  </Node_pressure_constraints>
</Add_equation>
)xml"),
               std::runtime_error);
}

TEST(EquationTranslatorNodePressureConstraints, ParserRejectsUnknownNestedElement)
{
  EXPECT_THROW((void)parseEquationXml(R"xml(
<Add_equation type="darcy">
  <Node_pressure_constraints>
    <Values_file_path>known_pressure_nodes.csv</Values_file_path>
    <Unexpected>1</Unexpected>
  </Node_pressure_constraints>
</Add_equation>
)xml"),
               std::runtime_error);
}

TEST(EquationTranslatorNodePressureConstraints, BuildInputLeavesAbsentBlockUnset)
{
  auto mesh = buildTranslatorMesh();
  auto params = parseEquationXml(R"xml(
<Add_equation type="darcy">
  <Tolerance>1e-8</Tolerance>
</Add_equation>
)xml");

  const auto input = application::translators::EquationTranslator::buildInput(*params, singleMeshMap(mesh));

  EXPECT_EQ(input.equation_type, "darcy");
  EXPECT_EQ(input.mesh_name, "mesh");
  EXPECT_FALSE(input.node_pressure_constraints.has_value());
}

TEST(EquationTranslatorNodePressureConstraints, BuildInputPopulatesNodePressureConstraints)
{
  auto mesh = buildTranslatorMesh();
  auto params = parseEquationXml(R"xml(
<Add_equation type="darcy">
  <Node_pressure_constraints>
    <Id_type>Global_vertex_gid</Id_type>
    <Values_file_path>known_pressure_nodes.csv</Values_file_path>
  </Node_pressure_constraints>
</Add_equation>
)xml");

  const auto input = application::translators::EquationTranslator::buildInput(*params, singleMeshMap(mesh));

  ASSERT_TRUE(input.node_pressure_constraints.has_value());
  EXPECT_EQ(input.node_pressure_constraints->id_type, "Global_vertex_gid");
  EXPECT_EQ(input.node_pressure_constraints->values_file_path, "known_pressure_nodes.csv");
}

TEST(EquationTranslatorPrecision, BuildInputPreservesHighPrecisionBoundaryValues)
{
  auto mesh = buildTranslatorMesh();
  mesh->base().register_label("loaded", 42);

  auto params = parseEquationXml(R"xml(
<Add_equation type="ustruct">
  <Add_BC name="loaded">
    <Type>Neumann</Type>
    <Value>4811974.1220499845</Value>
    <Follower_pressure_load>true</Follower_pressure_load>
  </Add_BC>
</Add_equation>
)xml");

  const auto input = application::translators::EquationTranslator::buildInput(*params, singleMeshMap(mesh));

  ASSERT_EQ(input.boundary_conditions.size(), 1u);
  const auto value = input.boundary_conditions.front().params.at("Value").value;
  EXPECT_EQ(value, "4811974.1220499845");
}

TEST(EquationTranslatorOutputs, BuildInputCopiesOutputBlocks)
{
  auto mesh = buildTranslatorMesh();
  auto params = parseEquationXml(R"xml(
<Add_equation type="ustruct">
  <Output type="Spatial">
    <Jacobian>true</Jacobian>
    <Stress>false</Stress>
    <Cauchy_stress>true</Cauchy_stress>
  </Output>
</Add_equation>
)xml");

  const auto input = application::translators::EquationTranslator::buildInput(*params, singleMeshMap(mesh));

  ASSERT_EQ(input.outputs.size(), 1u);
  EXPECT_EQ(input.outputs.front().type, "Spatial");
  EXPECT_EQ(input.outputs.front().params.at("Jacobian").value, "1");
  EXPECT_TRUE(input.outputs.front().params.at("Jacobian").defined);
  EXPECT_EQ(input.outputs.front().params.at("Stress").value, "0");
  EXPECT_TRUE(input.outputs.front().params.at("Stress").defined);
  EXPECT_EQ(input.outputs.front().params.at("Cauchy_stress").value, "1");
  EXPECT_TRUE(input.outputs.front().params.at("Cauchy_stress").defined);
}

TEST(EquationTranslatorNodePressureConstraints, BuildInputRejectsUnsupportedIdType)
{
  auto mesh = buildTranslatorMesh();
  auto params = parseEquationXml(R"xml(
<Add_equation type="darcy">
  <Node_pressure_constraints>
    <Id_type>Local_vertex_id</Id_type>
    <Values_file_path>known_pressure_nodes.csv</Values_file_path>
  </Node_pressure_constraints>
</Add_equation>
)xml");

  EXPECT_THROW((void)application::translators::EquationTranslator::buildInput(*params, singleMeshMap(mesh)),
               std::runtime_error);
}
