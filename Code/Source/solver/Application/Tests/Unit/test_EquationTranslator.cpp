#include <gtest/gtest.h>

#include "Application/Translators/EquationTranslator.h"
#include "FE/Spaces/SpaceFactory.h"
#include "FE/Systems/FESystem.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"
#include "Parameters.h"
#include "Physics/Core/EquationModuleInput.h"
#include "Physics/Formulations/NavierStokes/NavierStokesRegister.h"

#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
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

std::shared_ptr<svmp::Mesh> buildTranslatorTriangleMesh()
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

TEST(EquationTranslatorLevelSet, BuildInputKeepsTransportControlsFromXml)
{
  auto mesh = buildTranslatorMesh();
  auto params = parseEquationXml(R"xml(
<Add_equation type="level_set">
  <Level_set_field_name>phi</Level_set_field_name>
  <Transport_form>conservative_divergence</Transport_form>
  <Enable_SUPG>false</Enable_SUPG>
  <SUPG_tau_scale>0.25</SUPG_tau_scale>
</Add_equation>
)xml");

  const auto input =
      application::translators::EquationTranslator::buildInput(*params, singleMeshMap(mesh));

  EXPECT_EQ(input.equation_type, "level_set");
  EXPECT_EQ(input.equation_params.at("Level_set_field_name").value, "phi");
  EXPECT_EQ(input.equation_params.at("Transport_form").value,
            "conservative_divergence");
  EXPECT_EQ(input.equation_params.at("Enable_SUPG").value, "false");
  EXPECT_EQ(input.equation_params.at("SUPG_tau_scale").value, "0.25");
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

TEST(EquationTranslatorDomainDefaults, BuildInputCopiesGravityForceComponents)
{
  auto mesh = buildTranslatorMesh();
  auto params = parseEquationXml(R"xml(
<Add_equation type="fluid">
  <Density>998.2</Density>
  <Force_x>0.0</Force_x>
  <Force_y>-9.81</Force_y>
  <Force_z>0.0</Force_z>
  <Hydrostatic_pressure_initialization>true</Hydrostatic_pressure_initialization>
  <Hydrostatic_pressure_reference>0.0</Hydrostatic_pressure_reference>
  <Hydrostatic_pressure_reference_point>0.0 0.5 0.0</Hydrostatic_pressure_reference_point>
  <Viscosity model="Constant">
    <Value>1.003e-3</Value>
  </Viscosity>
</Add_equation>
)xml");

  const auto input = application::translators::EquationTranslator::buildInput(*params, singleMeshMap(mesh));

  ASSERT_TRUE(input.default_domain.params.at("Force_x").defined);
  ASSERT_TRUE(input.default_domain.params.at("Force_y").defined);
  ASSERT_TRUE(input.default_domain.params.at("Force_z").defined);
  EXPECT_DOUBLE_EQ(std::stod(input.default_domain.params.at("Force_x").value), 0.0);
  EXPECT_DOUBLE_EQ(std::stod(input.default_domain.params.at("Force_y").value), -9.81);
  EXPECT_DOUBLE_EQ(std::stod(input.default_domain.params.at("Force_z").value), 0.0);
  EXPECT_EQ(input.default_domain.params.at("Hydrostatic_pressure_initialization").value, "1");
  EXPECT_TRUE(input.default_domain.params.at("Hydrostatic_pressure_initialization").defined);
  EXPECT_EQ(input.default_domain.params.at("Hydrostatic_pressure_reference_point").value, " 0 0.5 0");
  EXPECT_TRUE(input.default_domain.params.at("Hydrostatic_pressure_reference_point").defined);
}

TEST(EquationTranslatorDomainDefaults, BuildInputKeepsMomentumSourceFieldOptions)
{
  auto mesh = buildTranslatorMesh();
  auto params = parseEquationXml(R"xml(
<Add_equation type="fluid">
  <Density>998.2</Density>
  <Momentum_source_field_name>ManufacturedSource</Momentum_source_field_name>
  <Auto_register_momentum_source_field>false</Auto_register_momentum_source_field>
  <Viscosity model="Constant">
    <Value>1.003e-3</Value>
  </Viscosity>
</Add_equation>
)xml");

  const auto input = application::translators::EquationTranslator::buildInput(*params, singleMeshMap(mesh));

  ASSERT_TRUE(input.equation_params.at("Momentum_source_field_name").defined);
  EXPECT_EQ(input.equation_params.at("Momentum_source_field_name").value, "ManufacturedSource");
  ASSERT_TRUE(input.equation_params.at("Auto_register_momentum_source_field").defined);
  EXPECT_EQ(input.equation_params.at("Auto_register_momentum_source_field").value, "false");
}

TEST(EquationTranslatorDomainDefaults, BuildInputKeepsRotatingFrameCoriolisOptions)
{
  auto mesh = buildTranslatorMesh();
  auto params = parseEquationXml(R"xml(
<Add_equation type="fluid">
  <Rotating_frame_coriolis>true</Rotating_frame_coriolis>
  <Rotating_frame_angular_velocity_temporal_values_file_path>bc/omega.dat</Rotating_frame_angular_velocity_temporal_values_file_path>
  <Viscosity model="Constant">
    <Value>1.003e-3</Value>
  </Viscosity>
</Add_equation>
)xml");

  const auto input = application::translators::EquationTranslator::buildInput(*params, singleMeshMap(mesh));

  ASSERT_TRUE(input.equation_params.at("Rotating_frame_coriolis").defined);
  EXPECT_EQ(input.equation_params.at("Rotating_frame_coriolis").value, "true");
  ASSERT_TRUE(input.equation_params.at("Rotating_frame_angular_velocity_temporal_values_file_path").defined);
  EXPECT_EQ(input.equation_params.at("Rotating_frame_angular_velocity_temporal_values_file_path").value,
            "bc/omega.dat");
}

TEST(EquationTranslatorFreeSurface, BuildInputKeepsOopFreeSurfaceParameters)
{
  auto mesh = buildTranslatorMesh();
  auto params = parseEquationXml(R"xml(
<Add_equation type="fluid">
  <Enable_ALE>true</Enable_ALE>
  <Mesh_velocity_source>coupled_displacement</Mesh_velocity_source>
  <Constant_velocity>0.0 0.0 0.0</Constant_velocity>
  <Add_BC name="free_surface">
    <Type>Free_surface</Type>
    <Implementation>UnfittedLevelSet</Implementation>
    <Level_set_field_name>phi</Level_set_field_name>
    <Generated_interface_domain_id>water_air</Generated_interface_domain_id>
    <Generated_interface_geometry>LinearCorner</Generated_interface_geometry>
    <Implicit_cut_quadrature_backend>LinearCorner</Implicit_cut_quadrature_backend>
    <Implicit_cut_fallback_policy>Fail</Implicit_cut_fallback_policy>
    <Geometry_tangent_policy>RefreshedFrozenQuadrature</Geometry_tangent_policy>
    <Implicit_cut_root_tolerance>1.0e-10</Implicit_cut_root_tolerance>
    <Implicit_cut_max_subdivision_depth>16</Implicit_cut_max_subdivision_depth>
    <Generated_interface_quadrature_order>2</Generated_interface_quadrature_order>
    <Interface_quadrature_order>1</Interface_quadrature_order>
    <Volume_quadrature_order>2</Volume_quadrature_order>
    <Active_domain>LevelSetNegative</Active_domain>
    <Active_domain_method>CutVolume</Active_domain_method>
    <Surface_tension_form>SurfaceStress</Surface_tension_form>
    <Active_domain_smoothing_width>0.02</Active_domain_smoothing_width>
    <Enable_cut_cell_stabilization>true</Enable_cut_cell_stabilization>
  </Add_BC>
</Add_equation>
)xml");

  const auto input = application::translators::EquationTranslator::buildInput(*params, singleMeshMap(mesh));

  EXPECT_EQ(input.equation_params.at("Enable_ALE").value, "true");
  EXPECT_EQ(input.equation_params.at("Mesh_velocity_source").value, "coupled_displacement");
  EXPECT_EQ(input.equation_params.at("Constant_velocity").value, "0.0 0.0 0.0");
  ASSERT_EQ(input.boundary_conditions.size(), 1u);
  const auto& bc = input.boundary_conditions.front();
  EXPECT_EQ(bc.boundary_marker, svmp::INVALID_LABEL);
  EXPECT_EQ(bc.params.at("Implementation").value, "UnfittedLevelSet");
  EXPECT_EQ(bc.params.at("Level_set_field_name").value, "phi");
  EXPECT_EQ(bc.params.at("Generated_interface_domain_id").value, "water_air");
  EXPECT_EQ(bc.params.at("Generated_interface_geometry").value, "LinearCorner");
  EXPECT_EQ(bc.params.at("Implicit_cut_quadrature_backend").value, "LinearCorner");
  EXPECT_EQ(bc.params.at("Implicit_cut_fallback_policy").value, "Fail");
  EXPECT_EQ(bc.params.at("Geometry_tangent_policy").value,
            "RefreshedFrozenQuadrature");
  EXPECT_EQ(bc.params.at("Implicit_cut_root_tolerance").value, "1.0e-10");
  EXPECT_EQ(bc.params.at("Implicit_cut_max_subdivision_depth").value, "16");
  EXPECT_EQ(bc.params.at("Generated_interface_quadrature_order").value, "2");
  EXPECT_EQ(bc.params.at("Interface_quadrature_order").value, "1");
  EXPECT_EQ(bc.params.at("Volume_quadrature_order").value, "2");
  EXPECT_EQ(bc.params.at("Active_domain").value, "LevelSetNegative");
  EXPECT_EQ(bc.params.at("Active_domain_method").value, "CutVolume");
  EXPECT_EQ(bc.params.at("Surface_tension_form").value, "SurfaceStress");
  EXPECT_EQ(bc.params.at("Active_domain_smoothing_width").value, "0.02");
  EXPECT_EQ(bc.params.at("Enable_cut_cell_stabilization").value, "true");
}

TEST(EquationTranslatorMeshMotion,
     XmlAliasesReachTangentialPolicyModuleRegistration)
{
  auto mesh = buildTranslatorMesh();
  mesh->base().register_label("primary_wall", 51);
  mesh->base().register_label("alias_wall", 52);

  auto params = parseEquationXml(R"xml(
<Add_equation type="mesh_motion">
  <Model>Harmonic</Model>
  <Field_name>mesh_displacement</Field_name>
  <Add_BC name="primary_wall">
    <Type>TangentialPolicy</Type>
    <Policy>Prescribed</Policy>
    <Quantity>Velocity</Quantity>
    <Target>0.25 -0.50</Target>
    <Penalty>4.0</Penalty>
    <Velocity_time_scale>2.0</Velocity_time_scale>
  </Add_BC>
  <Add_BC name="alias_wall">
    <Type>TangentialPolicy</Type>
    <Tangential_policy>Prescribed</Tangential_policy>
    <Constraint_quantity>Displacement</Constraint_quantity>
    <Target>-0.125 0.375</Target>
    <Penalty_scale>5.0</Penalty_scale>
    <Time_scale>3.0</Time_scale>
  </Add_BC>
</Add_equation>
)xml");

  const auto meshes = singleMeshMap(mesh);
  const auto input =
      application::translators::EquationTranslator::buildInput(
          *params, meshes);
  ASSERT_EQ(input.boundary_conditions.size(), 2u);
  EXPECT_EQ(input.boundary_conditions[0].params.at("Policy").value,
            "Prescribed");
  EXPECT_EQ(input.boundary_conditions[0].params.at("Quantity").value,
            "Velocity");
  EXPECT_EQ(
      input.boundary_conditions[0].params.at("Velocity_time_scale").value,
      "2.0");
  EXPECT_EQ(
      input.boundary_conditions[1].params.at("Tangential_policy").value,
      "Prescribed");
  EXPECT_EQ(
      input.boundary_conditions[1].params.at("Constraint_quantity").value,
      "Displacement");
  EXPECT_EQ(input.boundary_conditions[1].params.at("Time_scale").value,
            "3.0");

  svmp::FE::systems::FESystem system(mesh);
  auto module =
      application::translators::EquationTranslator::createModule(
          *params, system, meshes);
  ASSERT_TRUE(module);
  const auto policies = system.meshTangentialBoundaryPolicies();
  ASSERT_EQ(policies.size(), 2u);
  EXPECT_EQ(
      policies[0].policy,
      svmp::FE::systems::MeshTangentialBoundaryPolicy::Prescribed);
  EXPECT_EQ(
      policies[1].policy,
      svmp::FE::systems::MeshTangentialBoundaryPolicy::Prescribed);
  EXPECT_EQ(
      std::count_if(
          system.boundaryConditionDescriptors().begin(),
          system.boundaryConditionDescriptors().end(),
          [](const auto& descriptor) {
            return descriptor.trace_kind ==
                   svmp::FE::analysis::TraceKind::TangentialComponent;
          }),
      2);
}

TEST(EquationTranslatorFreeSurface,
     XmlTangentialPenaltyAliasesReachTruthfulFittedModule)
{
  svmp::Physics::formulations::navier_stokes::
      forceLink_NavierStokesRegister();
  auto mesh = buildTranslatorMesh();
  mesh->base().register_label("free_surface", 81);
  const auto meshes = singleMeshMap(mesh);
  constexpr std::array<std::string_view, 4> penalty_aliases{
      "Tangential_mesh_penalty",
      "TangentialMeshPenalty",
      "Prescribed_tangential_mesh_penalty",
      "PrescribedTangentialMeshPenalty",
  };

  for (const auto alias : penalty_aliases) {
    const std::string xml =
        std::string(R"xml(
<Add_equation type="fluid">
  <Enable_ALE>true</Enable_ALE>
  <Mesh_velocity_source>coupled_displacement</Mesh_velocity_source>
  <Auto_register_mesh_displacement_field>true</Auto_register_mesh_displacement_field>
  <Density>1.0</Density>
  <Viscosity model="Constant">
    <Value>0.01</Value>
  </Viscosity>
  <Add_BC name="free_surface">
    <Type>Free_surface</Type>
    <Implementation>FittedALE</Implementation>
    <Normal_kinematic_policy>MatchFluidNormalVelocity</Normal_kinematic_policy>
    <Kinematic_enforcement>Penalty</Kinematic_enforcement>
    <Kinematic_penalty>11.0</Kinematic_penalty>
    <Tangential_mesh_policy>Prescribed</Tangential_mesh_policy>
    <Prescribed_tangential_mesh_velocity>0.25 -0.5 0.0</Prescribed_tangential_mesh_velocity>
    <)xml") +
        std::string(alias) + R"xml(>7.0</)xml" +
        std::string(alias) + R"xml(>
  </Add_BC>
</Add_equation>
)xml";
    auto params = parseEquationXml(xml.c_str());
    svmp::FE::systems::FESystem system(mesh);
    auto module =
        application::translators::EquationTranslator::createModule(
            *params, system, meshes);
    ASSERT_TRUE(module) << alias;

    const auto artifact = module->effectiveConfigurationArtifact();
    ASSERT_TRUE(artifact.has_value()) << alias;
    EXPECT_NE(
        artifact->json.find("\"tangential_mesh_penalty\":7"),
        std::string::npos)
        << alias;
    EXPECT_NE(
        artifact->json.find(
            "\"tangential_mesh_owner\":"
            "\"IncompressibleNavierStokesVMSModule.FreeSurfaceBoundary\""),
        std::string::npos)
        << alias;
    EXPECT_NE(
        artifact->json.find("\"policy_consumed\":true"),
        std::string::npos)
        << alias;
    EXPECT_NE(
        artifact->json.find("\"operator_tag\":\"equations\""),
        std::string::npos)
        << alias;
    EXPECT_NE(
        artifact->json.find(
            "\"operator_source\":"
            "\"Fitted free-surface prescribed tangential mesh velocity on "
            "marker 81\""),
        std::string::npos)
        << alias;
    EXPECT_NE(
        artifact->json.find(
            "\"policy_qualification\":"
            "\"supported_configuration_envelope\""),
        std::string::npos)
        << alias;
  }

  auto smoothing_params = parseEquationXml(R"xml(
<Add_equation type="fluid">
  <Enable_ALE>true</Enable_ALE>
  <Mesh_velocity_source>coupled_displacement</Mesh_velocity_source>
  <Auto_register_mesh_displacement_field>true</Auto_register_mesh_displacement_field>
  <Density>1.0</Density>
  <Viscosity model="Constant">
    <Value>0.01</Value>
  </Viscosity>
  <Add_BC name="free_surface">
    <Type>Free_surface</Type>
    <Implementation>FittedALE</Implementation>
    <Normal_kinematic_policy>MatchFluidNormalVelocity</Normal_kinematic_policy>
    <Kinematic_enforcement>Penalty</Kinematic_enforcement>
    <Kinematic_penalty>11.0</Kinematic_penalty>
    <Tangential_mesh_policy>SmoothingOnly</Tangential_mesh_policy>
    <Tangential_mesh_penalty>5.0</Tangential_mesh_penalty>
  </Add_BC>
</Add_equation>
)xml");
  svmp::FE::systems::FESystem smoothing_system(mesh);
  auto smoothing_module =
      application::translators::EquationTranslator::createModule(
          *smoothing_params, smoothing_system, meshes);
  ASSERT_TRUE(smoothing_module);
  const auto smoothing_artifact =
      smoothing_module->effectiveConfigurationArtifact();
  ASSERT_TRUE(smoothing_artifact.has_value());
  EXPECT_NE(
      smoothing_artifact->json.find(
          "\"tangential_mesh_penalty\":5"),
      std::string::npos);
  EXPECT_NE(
      smoothing_artifact->json.find(
          "\"operator_source\":"
          "\"Fitted free-surface tangential surface smoothing on "
          "marker 81\""),
      std::string::npos);

  auto free_with_weight_params = parseEquationXml(R"xml(
<Add_equation type="fluid">
  <Enable_ALE>true</Enable_ALE>
  <Mesh_velocity_source>coupled_displacement</Mesh_velocity_source>
  <Auto_register_mesh_displacement_field>true</Auto_register_mesh_displacement_field>
  <Density>1.0</Density>
  <Viscosity model="Constant">
    <Value>0.01</Value>
  </Viscosity>
  <Add_BC name="free_surface">
    <Type>Free_surface</Type>
    <Implementation>FittedALE</Implementation>
    <Normal_kinematic_policy>MatchFluidNormalVelocity</Normal_kinematic_policy>
    <Kinematic_enforcement>Penalty</Kinematic_enforcement>
    <Kinematic_penalty>11.0</Kinematic_penalty>
    <Tangential_mesh_policy>Free</Tangential_mesh_policy>
    <Tangential_mesh_penalty>5.0</Tangential_mesh_penalty>
  </Add_BC>
</Add_equation>
)xml");
  svmp::FE::systems::FESystem free_with_weight_system(mesh);
  EXPECT_THROW(
      application::translators::EquationTranslator::createModule(
          *free_with_weight_params, free_with_weight_system, meshes),
      std::runtime_error);
  EXPECT_EQ(free_with_weight_system.fieldMap().numFields(), 0u);
  EXPECT_TRUE(
      free_with_weight_system.meshTangentialBoundaryPolicies().empty());
}

TEST(EquationTranslatorFreeSurface,
     XmlExplicitNoneCannotBePromotedByKinematicPenalty)
{
  svmp::Physics::formulations::navier_stokes::
      forceLink_NavierStokesRegister();
  auto mesh = buildTranslatorMesh();
  mesh->base().register_label("free_surface", 82);
  const auto meshes = singleMeshMap(mesh);
  auto params = parseEquationXml(R"xml(
<Add_equation type="fluid">
  <Enable_ALE>true</Enable_ALE>
  <Mesh_velocity_source>coupled_displacement</Mesh_velocity_source>
  <Auto_register_mesh_displacement_field>true</Auto_register_mesh_displacement_field>
  <Density>1.0</Density>
  <Viscosity model="Constant">
    <Value>0.01</Value>
  </Viscosity>
  <Add_BC name="free_surface">
    <Type>Free_surface</Type>
    <Implementation>FittedALE</Implementation>
    <Normal_kinematic_policy>MatchFluidNormalVelocity</Normal_kinematic_policy>
    <Kinematic_enforcement>None</Kinematic_enforcement>
    <Kinematic_penalty>11.0</Kinematic_penalty>
    <Tangential_mesh_policy>Prescribed</Tangential_mesh_policy>
  </Add_BC>
</Add_equation>
)xml");

  svmp::FE::systems::FESystem system(mesh);
  EXPECT_THROW(
      (void)application::translators::EquationTranslator::createModule(
          *params, system, meshes),
      std::runtime_error);
  EXPECT_EQ(system.fieldMap().numFields(), 0u);
  EXPECT_TRUE(system.formulationRecords().empty());
  EXPECT_TRUE(system.meshTangentialBoundaryPolicies().empty());
}

TEST(EquationTranslatorFreeSurface,
     XmlFittedDynamicContactFailsClosedBeforeSystemMutation)
{
  svmp::Physics::formulations::navier_stokes::
      forceLink_NavierStokesRegister();
  auto mesh = buildTranslatorMesh();
  mesh->base().register_label("free_surface", 83);
  const auto meshes = singleMeshMap(mesh);
  auto params = parseEquationXml(R"xml(
<Add_equation type="fluid">
  <Enable_ALE>true</Enable_ALE>
  <Mesh_velocity_source>coupled_displacement</Mesh_velocity_source>
  <Auto_register_mesh_displacement_field>true</Auto_register_mesh_displacement_field>
  <Density>1.0</Density>
  <Viscosity model="Constant">
    <Value>0.01</Value>
  </Viscosity>
  <Add_BC name="free_surface">
    <Type>Free_surface</Type>
    <Implementation>FittedALE</Implementation>
    <Surface_tension>0.8</Surface_tension>
    <Normal_kinematic_policy>MatchFluidNormalVelocity</Normal_kinematic_policy>
    <Kinematic_enforcement>Penalty</Kinematic_enforcement>
    <Kinematic_penalty>8.0</Kinematic_penalty>
    <Tangential_mesh_policy>Prescribed</Tangential_mesh_policy>
    <Tangential_mesh_penalty>6.0</Tangential_mesh_penalty>
    <Contact_line_model>DynamicContactAngle</Contact_line_model>
    <Contact_angle_degrees>60.0</Contact_angle_degrees>
    <Contact_line_wall_marker>84</Contact_line_wall_marker>
    <Contact_line_wall_normal>1.0 0.0 0.0</Contact_line_wall_normal>
    <Contact_line_mobility>0.5</Contact_line_mobility>
    <Wall_slip_model>Navier</Wall_slip_model>
    <Wall_slip_length>0.2</Wall_slip_length>
  </Add_BC>
</Add_equation>
)xml");

  svmp::FE::systems::FESystem system(mesh);
  const auto layout_revision_before = system.feLayoutRevisionState();
  const auto constraint_revision_before =
      system.constraintRevisionSnapshot();
  const auto constraint_revision_key = [](const auto& revision) {
    return std::array{
        revision.geometry,
        revision.reference_rebase,
        revision.topology,
        revision.ownership,
        revision.numbering,
        revision.mesh_field_layout,
        revision.mesh_field_values,
        revision.labels,
        revision.active_configuration,
        revision.fe_space,
        revision.fe_dof_layout,
        revision.fe_constraint_layout,
        revision.fe_block_layout,
        revision.time_epoch,
    };
  };
  const auto constraint_revision_key_before =
      constraint_revision_key(constraint_revision_before);
  const auto operator_revision_before =
      system.operatorRevisionSnapshot();
  const auto system_layout_revision_before =
      system.systemLayoutRevision();
  const auto sparsity_revision_before =
      system.sparsityPatternRevision();
  const auto definition_plan_before =
      system.computeSetupStoragePlan();
  const auto definition_plan_key = [](const auto& plan) {
    const auto& requirements = plan.requirements;
    return std::array{
        requirements.vertex_topology,
        requirements.cell_topology,
        requirements.edge_topology,
        requirements.boundary_face_topology,
        requirements.interior_face_topology,
        requirements.interface_face_topology,
        requirements.vertex_gids,
        requirements.cell_gids,
        requirements.face_gids,
        requirements.edge_gids,
        requirements.global_vertex_lookup,
        requirements.global_cell_lookup,
        requirements.global_face_lookup,
        requirements.global_edge_lookup,
        requirements.entity_dof_map,
        plan.can_alias_single_field_dof_map,
        plan.uses_single_field_alias,
    };
  };
  const auto definition_plan_key_before =
      definition_plan_key(definition_plan_before);
  const auto constraint_stale_before =
      system.constraintStateStaleForCurrentRevisions();
  try {
    (void)application::translators::EquationTranslator::createModule(
        *params, system, meshes);
    FAIL() << "fitted DynamicContactAngle must fail closed";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(
        std::string(error.what()).find(
            "DynamicContactAngle is currently supported only for sharp "
            "unfitted level-set free surfaces"),
        std::string::npos);
  }

  const auto layout_revision_after = system.feLayoutRevisionState();
  EXPECT_EQ(layout_revision_after.space, layout_revision_before.space);
  EXPECT_EQ(
      layout_revision_after.dof_layout,
      layout_revision_before.dof_layout);
  EXPECT_EQ(
      layout_revision_after.constraint_layout,
      layout_revision_before.constraint_layout);
  EXPECT_EQ(
      layout_revision_after.block_layout,
      layout_revision_before.block_layout);
  EXPECT_EQ(
      system.systemLayoutRevision(),
      system_layout_revision_before);
  EXPECT_EQ(
      system.sparsityPatternRevision(),
      sparsity_revision_before);
  const auto operator_invalidation =
      system.operatorInvalidationDecision(operator_revision_before);
  EXPECT_FALSE(operator_invalidation.any_change())
      << operator_invalidation.reason;
  const auto constraint_revision_after =
      system.constraintRevisionSnapshot();
  EXPECT_EQ(
      constraint_revision_after.valid,
      constraint_revision_before.valid);
  EXPECT_EQ(
      constraint_revision_key(constraint_revision_after),
      constraint_revision_key_before);
  EXPECT_EQ(
      system.constraintStateStaleForCurrentRevisions(),
      constraint_stale_before);

  EXPECT_EQ(system.fieldMap().numFields(), 0u);
  EXPECT_EQ(system.blockMap(), nullptr);
  EXPECT_FALSE(system.isSetup());
  EXPECT_TRUE(system.formulationRecords().empty());
  EXPECT_TRUE(system.boundaryConditionDescriptors().empty());
  EXPECT_TRUE(system.variableDescriptors().empty());
  EXPECT_TRUE(system.contributionDescriptors().empty());
  EXPECT_TRUE(system.constraints().empty());
  EXPECT_EQ(system.constraints().numConstraints(), 0u);
  const auto constraint_dependencies =
      system.constraintDependencyDeclaration();
  EXPECT_FALSE(constraint_dependencies.any());
  EXPECT_EQ(
      constraint_dependencies.tangent_policy,
      svmp::FE::constraints::ConstraintTangentPolicy::None);
  EXPECT_TRUE(constraint_dependencies.tangent_hook_name.empty());

  constexpr std::array mesh_motion_roles{
      svmp::FE::systems::MeshMotionFieldRole::Displacement,
      svmp::FE::systems::MeshMotionFieldRole::Velocity,
      svmp::FE::systems::MeshMotionFieldRole::Acceleration,
      svmp::FE::systems::MeshMotionFieldRole::PreviousCoordinates,
      svmp::FE::systems::MeshMotionFieldRole::PreviousDisplacement,
      svmp::FE::systems::MeshMotionFieldRole::PreviousVelocity,
      svmp::FE::systems::MeshMotionFieldRole::PredictedVelocity,
  };
  for (const auto role : mesh_motion_roles) {
    EXPECT_FALSE(system.meshMotionField(role).has_value());
  }
  EXPECT_TRUE(system.meshTangentialBoundaryPolicies().empty());
  EXPECT_TRUE(system.meshTangentialBoundaryPolicyHistory().empty());
  EXPECT_TRUE(
      system.freeSurfaceDiscreteFunctionalDeclarations().empty());
  EXPECT_TRUE(system.freeSurfaceDiscreteFunctionalHistory().empty());

  EXPECT_FALSE(system.hasGaugeRegistry());
  EXPECT_EQ(system.gaugeRegistryIfPresent(), nullptr);
  EXPECT_EQ(system.auxiliaryStateManagerIfPresent(), nullptr);
  EXPECT_EQ(system.auxiliaryOperatorRegistryIfPresent(), nullptr);
  EXPECT_EQ(system.auxiliaryInputRegistryIfPresent(), nullptr);
  EXPECT_EQ(system.feQuantityRegistryIfPresent(), nullptr);
  EXPECT_TRUE(system.parameterRegistry().specs().empty());

  EXPECT_TRUE(system.formInstallCellDomainRestrictions().empty());
  EXPECT_EQ(system.cutIntegrationContext(), nullptr);
  EXPECT_FALSE(system.cutIntegrationContextTransactionActive());
  EXPECT_FALSE(
      system.isGeneratedEmbeddedInterfaceMarkerRegistered(83));
  EXPECT_FALSE(
      system.isGeneratedEmbeddedInterfaceMarkerRegistered(84));
  EXPECT_EQ(system.latestAnalysisSummaries(), nullptr);
  EXPECT_EQ(system.constraintSummary(), nullptr);
  EXPECT_EQ(system.topologyContext(), nullptr);
  EXPECT_EQ(system.interfaceTopologyContext(), nullptr);
  EXPECT_FALSE(system.geometricNonlinearityEnabled());

  const auto definition_plan_after =
      system.computeSetupStoragePlan();
  EXPECT_EQ(
      definition_plan_key(definition_plan_after),
      definition_plan_key_before);
  EXPECT_EQ(
      definition_plan_after.reasons,
      definition_plan_before.reasons);
  EXPECT_EQ(
      definition_plan_after.summary(),
      definition_plan_before.summary());
}

TEST(EquationTranslatorFreeSurface, BuildInputResolvesUnfittedContactLineWallFaces)
{
  auto mesh = buildTranslatorMesh();
  mesh->base().register_label("wall_left", 11);
  mesh->base().register_label("wall_right", 12);

  auto params = parseEquationXml(R"xml(
<Add_equation type="fluid">
  <Add_BC name="free_surface">
    <Type>Free_surface</Type>
    <Implementation>UnfittedLevelSet</Implementation>
    <Level_set_field_name>phi</Level_set_field_name>
    <Active_domain>LevelSetNegative</Active_domain>
    <Active_domain_method>CutVolume</Active_domain_method>
    <Contact_line_model>PrescribedContactAngle</Contact_line_model>
    <Contact_angle_degrees>90.0</Contact_angle_degrees>
    <Contact_line_wall_faces>wall_left; wall_right</Contact_line_wall_faces>
    <Contact_line_wall_normals>-1.0 0.0 0.0; 1.0 0.0 0.0</Contact_line_wall_normals>
  </Add_BC>
</Add_equation>
)xml");

  const auto input = application::translators::EquationTranslator::buildInput(*params, singleMeshMap(mesh));

  ASSERT_EQ(input.boundary_conditions.size(), 1u);
  const auto& bc = input.boundary_conditions.front();
  ASSERT_TRUE(bc.params.at("Contact_line_wall_markers").defined);
  EXPECT_EQ(bc.params.at("Contact_line_wall_markers").value, "11;12");
  EXPECT_EQ(bc.params.at("Contact_line_wall_normals").value,
            "-1.0 0.0 0.0; 1.0 0.0 0.0");
  for (const std::string_view key : {
           "Contact_line_wall_faces", "ContactLineWallFaces",
           "Contact_line_wall_face", "ContactLineWallFace",
           "Wall_boundary_faces", "WallBoundaryFaces",
           "Wall_boundary_face", "WallBoundaryFace"}) {
    EXPECT_EQ(bc.params.count(std::string(key)), 0u);
  }
}

TEST(EquationTranslatorFreeSurface, BuildInputRejectsContactLineFacesAndMarkersTogether)
{
  auto mesh = buildTranslatorMesh();
  mesh->base().register_label("wall_left", 11);

  auto params = parseEquationXml(R"xml(
<Add_equation type="fluid">
  <Add_BC name="free_surface">
    <Type>Free_surface</Type>
    <Implementation>UnfittedLevelSet</Implementation>
    <Level_set_field_name>phi</Level_set_field_name>
    <Active_domain>LevelSetNegative</Active_domain>
    <Active_domain_method>CutVolume</Active_domain_method>
    <Contact_line_model>PrescribedContactAngle</Contact_line_model>
    <Contact_angle_degrees>90.0</Contact_angle_degrees>
    <Contact_line_wall_faces>wall_left</Contact_line_wall_faces>
    <Contact_line_wall_markers>11</Contact_line_wall_markers>
    <Contact_line_wall_normals>-1.0 0.0 0.0</Contact_line_wall_normals>
  </Add_BC>
</Add_equation>
)xml");

  EXPECT_THROW((void)application::translators::EquationTranslator::buildInput(
                   *params, singleMeshMap(mesh)),
               std::runtime_error);
}

TEST(EquationTranslatorFreeSurface,
     XmlPhysicalModelIsExplicitAndUnsupportedScopeFailsClosed)
{
  svmp::Physics::formulations::navier_stokes::
      forceLink_NavierStokesRegister();
  auto mesh = buildTranslatorMesh();
  const auto meshes = singleMeshMap(mesh);

  for (const std::string alias : {
           "Free_surface_physical_model",
           "FreeSurfacePhysicalModel",
       }) {
    SCOPED_TRACE(alias);
    const std::string xml =
        std::string(R"xml(
<Add_equation type="fluid">
  <)xml") +
        alias +
        R"xml(>OnePhaseLiquidPrescribedExteriorPressure</)xml" +
        alias + R"xml(>
  <Density>1.0</Density>
  <Viscosity model="Constant">
    <Value>0.01</Value>
  </Viscosity>
</Add_equation>
)xml";
    auto params = parseEquationXml(xml.c_str());
    const auto input =
        application::translators::EquationTranslator::buildInput(
            *params, meshes);
    ASSERT_TRUE(input.equation_params.at(alias).defined);
    EXPECT_EQ(
        input.equation_params.at(alias).value,
        "OnePhaseLiquidPrescribedExteriorPressure");

    svmp::FE::systems::FESystem system(mesh);
    auto module =
        application::translators::EquationTranslator::createModule(
            *params, system, meshes);
    ASSERT_TRUE(module);
    const auto artifact = module->effectiveConfigurationArtifact();
    ASSERT_TRUE(artifact.has_value());
    EXPECT_NE(
        artifact->json.find(
            "\"artifact_schema_version\":4"),
        std::string::npos);
    EXPECT_NE(
        artifact->json.find(
            "\"name\":"
            "\"one_phase_liquid_prescribed_exterior_pressure\""),
        std::string::npos);
  }

  const auto expect_factory_rejected =
      [&](const char* xml, std::string_view diagnostic) {
        auto params = parseEquationXml(xml);
        svmp::FE::systems::FESystem system(mesh);
        try {
          (void)application::translators::EquationTranslator::createModule(
              *params, system, meshes);
          FAIL() << "unsupported XML physical input must fail closed";
        } catch (const std::runtime_error& error) {
          EXPECT_NE(
              std::string(error.what()).find(diagnostic),
              std::string::npos);
        }
        EXPECT_EQ(system.fieldMap().numFields(), 0u);
        EXPECT_TRUE(system.formulationRecords().empty());
        EXPECT_EQ(
            system.findFieldByName("Velocity"),
            svmp::FE::INVALID_FIELD_ID);
        EXPECT_EQ(
            system.findFieldByName("Pressure"),
            svmp::FE::INVALID_FIELD_ID);
      };

  expect_factory_rejected(
      R"xml(
<Add_equation type="fluid">
  <Model>two_fluid</Model>
  <Density>1.0</Density>
  <Viscosity model="Constant"><Value>0.01</Value></Viscosity>
</Add_equation>
)xml",
      "unsupported_two_phase_or_jump_free_surface_scope");

  expect_factory_rejected(
      R"xml(
<Add_equation type="fluid">
  <Free_surface_physical_model>OnePhaseLiquidPrescribedExteriorPressure</Free_surface_physical_model>
  <Prescribed_pressure_jump>3.0</Prescribed_pressure_jump>
  <Density>1.0</Density>
  <Viscosity model="Constant"><Value>0.01</Value></Viscosity>
</Add_equation>
)xml",
      "unsupported_two_phase_or_jump_free_surface_scope");

  const auto expect_parser_rejected =
      [](const char* xml, std::string_view diagnostic) {
        try {
          (void)parseEquationXml(xml);
          FAIL() << "invalid XML physical input must fail closed";
        } catch (const std::runtime_error& error) {
          EXPECT_NE(
              std::string(error.what()).find(diagnostic),
              std::string::npos);
        }
      };

  expect_parser_rejected(
      R"xml(
<Add_equation type="fluid">
  <Free_surface_physical_model>two_fluid</Free_surface_physical_model>
</Add_equation>
)xml",
      "unsupported_two_phase_or_jump_free_surface_scope");
  expect_parser_rejected(
      R"xml(
<Add_equation type="fluid">
  <FreeSurfacePhysicalModel>one_phase_liquid_sharp_interface</FreeSurfacePhysicalModel>
</Add_equation>
)xml",
      "unsupported_free_surface_physical_model");
  expect_parser_rejected(
      R"xml(
<Add_equation type="fluid">
  <Free_surface_physical_model>OnePhaseLiquidPrescribedExteriorPressure</Free_surface_physical_model>
  <Free_surface_physical_model>OnePhaseLiquidPrescribedExteriorPressure</Free_surface_physical_model>
</Add_equation>
)xml",
      "ambiguous_free_surface_physical_model");
  expect_parser_rejected(
      R"xml(
<Add_equation type="fluid">
  <Free_surface_physical_model>OnePhaseLiquidPrescribedExteriorPressure</Free_surface_physical_model>
  <FreeSurfacePhysicalModel>OnePhaseLiquidPrescribedExteriorPressure</FreeSurfacePhysicalModel>
</Add_equation>
)xml",
      "ambiguous_free_surface_physical_model");
  expect_parser_rejected(
      R"xml(
<Add_equation type="level_set">
  <Free_surface_physical_model>OnePhaseLiquidPrescribedExteriorPressure</Free_surface_physical_model>
</Add_equation>
)xml",
      "misplaced_free_surface_physical_model");
  expect_parser_rejected(
      R"xml(
<Add_equation type="fluid">
  <Domain id="liquid">
    <FreeSurfacePhysicalModel>OnePhaseLiquidPrescribedExteriorPressure</FreeSurfacePhysicalModel>
  </Domain>
</Add_equation>
)xml",
      "misplaced_free_surface_physical_model");
  expect_parser_rejected(
      R"xml(
<Add_equation type="fluid">
  <Add_BC name="free_surface">
    <Free_surface_physical_model>OnePhaseLiquidPrescribedExteriorPressure</Free_surface_physical_model>
  </Add_BC>
</Add_equation>
)xml",
      "misplaced_free_surface_physical_model");
  expect_parser_rejected(
      R"xml(
<Add_equation type="fluid">
  <Gas_density>1.2</Gas_density>
</Add_equation>
)xml",
      "Unknown Add_equation XML element");
}

TEST(EquationTranslatorFreeSurface,
     XmlIncompressibleTwoFluidSelectsDedicatedModule)
{
  svmp::Physics::formulations::navier_stokes::
      forceLink_NavierStokesRegister();
  auto mesh = buildTranslatorTriangleMesh();
  const auto meshes = singleMeshMap(mesh);
  auto params = parseEquationXml(R"xml(
<Add_equation type="fluid">
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
  <Prescribed_pressure_jump>3.0</Prescribed_pressure_jump>
  <Prescribed_viscous_traction_jump_x>6.0</Prescribed_viscous_traction_jump_x>
  <Prescribed_viscous_traction_jump_y>7.0</Prescribed_viscous_traction_jump_y>
</Add_equation>
)xml");

  const auto input =
      application::translators::EquationTranslator::buildInput(
          *params, meshes);
  EXPECT_EQ(
      input.equation_params.at("Free_surface_physical_model").value,
      "IncompressibleTwoFluid");
  EXPECT_EQ(
      input.equation_params.at("Material_interface_marker").value,
      "71");

  svmp::FE::systems::FESystem system(mesh);
  const auto scalar_space =
      svmp::FE::spaces::SpaceFactory::create_h1(
          svmp::FE::ElementType::Triangle3, 1);
  (void)system.addField(svmp::FE::systems::FieldSpec{
      .name = "level_set",
      .space = scalar_space,
      .components = 1,
  });
  auto module =
      application::translators::EquationTranslator::createModule(
          *params, system, meshes);

  ASSERT_TRUE(module);
  const auto artifact = module->effectiveConfigurationArtifact();
  ASSERT_TRUE(artifact.has_value());
  EXPECT_EQ(artifact->component, "incompressible_two_fluid");
  EXPECT_NE(
      artifact->json.find("\"negative_density\":1000"),
      std::string::npos);
  EXPECT_NE(
      artifact->json.find("\"positive_density\":1"),
      std::string::npos);
  EXPECT_NE(
      artifact->json.find("\"surface_tension\":0.071999999999999995"),
      std::string::npos);
  EXPECT_NE(
      artifact->json.find("\"prescribed_pressure_jump\":3"),
      std::string::npos);
  EXPECT_NE(
      artifact->json.find(
          "\"prescribed_viscous_traction_jump\":[6,7]"),
      std::string::npos);
  EXPECT_NE(
      artifact->json.find(
          "\"momentum_reconciliation_required\":true"),
      std::string::npos);
  EXPECT_EQ(
      artifact->json.find("momentum_flux_reconciliation_qualified"),
      std::string::npos);
  EXPECT_NE(
      system.findFieldByName("u_negative"),
      svmp::FE::INVALID_FIELD_ID);
  EXPECT_NE(
      system.findFieldByName("p_negative"),
      svmp::FE::INVALID_FIELD_ID);
  EXPECT_NE(
      system.findFieldByName("u_positive"),
      svmp::FE::INVALID_FIELD_ID);
  EXPECT_NE(
      system.findFieldByName("p_positive"),
      svmp::FE::INVALID_FIELD_ID);
  ASSERT_EQ(
      system.materialInterfaceTransportVelocityDeclarations().size(),
      1u);
  EXPECT_EQ(
      system.materialInterfaceTransportVelocityDeclarations()
          .front()
          .interface_marker,
      71);
  ASSERT_EQ(system.twoFluidAcceptedStageDiagnosticDeclarations().size(),
            1u);
  EXPECT_TRUE(system.twoFluidAcceptedStageDiagnosticDeclarations()
                  .front()
                  .require_conservative_phase_momentum_reconciliation);
  ASSERT_TRUE(system.twoFluidAcceptedStageDiagnosticDeclarations()
                  .front()
                  .parameters.prescribed_viscous_traction_jump.has_value());
  EXPECT_EQ(*system.twoFluidAcceptedStageDiagnosticDeclarations()
                 .front()
                 .parameters.prescribed_viscous_traction_jump,
            (std::array<svmp::FE::Real, 3>{{6.0, 7.0, 0.0}}));
}

TEST(EquationTranslatorFreeSurface,
     XmlIncompressibleTwoFluidRequiresEveryMaterialCoefficientBeforeMutation)
{
  svmp::Physics::formulations::navier_stokes::
      forceLink_NavierStokesRegister();
  auto mesh = buildTranslatorTriangleMesh();
  const auto meshes = singleMeshMap(mesh);
  auto params = parseEquationXml(R"xml(
<Add_equation type="fluid">
  <Free_surface_physical_model>IncompressibleTwoFluid</Free_surface_physical_model>
  <Level_set_field_name>level_set</Level_set_field_name>
  <Generated_interface_domain_id>material_interface</Generated_interface_domain_id>
  <Material_interface_marker>71</Material_interface_marker>
  <Negative_phase_density>1000.0</Negative_phase_density>
  <Negative_phase_dynamic_viscosity>0.01</Negative_phase_dynamic_viscosity>
  <Positive_phase_density>1.0</Positive_phase_density>
  <Two_fluid_surface_tension>0.072</Two_fluid_surface_tension>
  <Two_fluid_interface_nitsche_gamma>24.0</Two_fluid_interface_nitsche_gamma>
</Add_equation>
)xml");

  svmp::FE::systems::FESystem system(mesh);
  const auto scalar_space =
      svmp::FE::spaces::SpaceFactory::create_h1(
          svmp::FE::ElementType::Triangle3, 1);
  (void)system.addField(svmp::FE::systems::FieldSpec{
      .name = "level_set",
      .space = scalar_space,
      .components = 1,
  });
  try {
    (void)application::translators::EquationTranslator::createModule(
        *params, system, meshes);
    FAIL() << "incomplete two-fluid material input was accepted";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(
        std::string_view(error.what()).find(
            "missing_two_fluid_parameter:Positive_phase_dynamic_viscosity"),
        std::string_view::npos);
  }
  EXPECT_NE(
      system.findFieldByName("level_set"),
      svmp::FE::INVALID_FIELD_ID);
  EXPECT_EQ(
      system.findFieldByName("u_negative"),
      svmp::FE::INVALID_FIELD_ID);
  EXPECT_EQ(
      system.findFieldByName("p_negative"),
      svmp::FE::INVALID_FIELD_ID);
  EXPECT_EQ(
      system.findFieldByName("u_positive"),
      svmp::FE::INVALID_FIELD_ID);
  EXPECT_EQ(
      system.findFieldByName("p_positive"),
      svmp::FE::INVALID_FIELD_ID);
  EXPECT_TRUE(system.formulationRecords().empty());
  EXPECT_TRUE(
      system.materialInterfaceTransportVelocityDeclarations().empty());
}

TEST(EquationTranslatorFreeSurface,
     XmlIncompressibleTwoFluidRejectsDuplicatePressureJumpAliasesBeforeMutation)
{
  svmp::Physics::formulations::navier_stokes::
      forceLink_NavierStokesRegister();
  auto mesh = buildTranslatorTriangleMesh();
  const auto meshes = singleMeshMap(mesh);
  auto params = parseEquationXml(R"xml(
<Add_equation type="fluid">
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
  <Prescribed_pressure_jump>3.0</Prescribed_pressure_jump>
  <PrescribedPressureJump>4.0</PrescribedPressureJump>
</Add_equation>
)xml");

  svmp::FE::systems::FESystem system(mesh);
  const auto scalar_space =
      svmp::FE::spaces::SpaceFactory::create_h1(
          svmp::FE::ElementType::Triangle3, 1);
  (void)system.addField(svmp::FE::systems::FieldSpec{
      .name = "level_set",
      .space = scalar_space,
      .components = 1,
  });
  try {
    (void)application::translators::EquationTranslator::createModule(
        *params, system, meshes);
    FAIL() << "duplicate pressure-jump aliases were accepted";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(
        std::string_view(error.what()).find(
            "ambiguous_two_fluid_parameter:Prescribed_pressure_jump"),
        std::string_view::npos)
        << error.what();
  }
  EXPECT_EQ(system.findFieldByName("u_negative"),
            svmp::FE::INVALID_FIELD_ID);
  EXPECT_EQ(system.findFieldByName("p_negative"),
            svmp::FE::INVALID_FIELD_ID);
  EXPECT_EQ(system.findFieldByName("u_positive"),
            svmp::FE::INVALID_FIELD_ID);
  EXPECT_EQ(system.findFieldByName("p_positive"),
            svmp::FE::INVALID_FIELD_ID);
  EXPECT_TRUE(system.formulationRecords().empty());
  EXPECT_TRUE(
      system.materialInterfaceTransportVelocityDeclarations().empty());
  EXPECT_TRUE(
      system.twoFluidAcceptedStageDiagnosticDeclarations().empty());
}

TEST(EquationTranslatorFreeSurface,
     XmlIncompressibleTwoFluidViscousTractionVectorFailsClosedBeforeMutation)
{
  svmp::Physics::formulations::navier_stokes::
      forceLink_NavierStokesRegister();
  auto mesh = buildTranslatorTriangleMesh();
  const auto meshes = singleMeshMap(mesh);
  struct Case {
    std::string_view target_elements;
    std::string_view expected_diagnostic;
  };
  const std::array cases{
      Case{
          .target_elements =
              "<Prescribed_viscous_traction_jump_x>1.0</Prescribed_viscous_traction_jump_x>",
          .expected_diagnostic =
              "missing_two_fluid_parameter:Prescribed_viscous_traction_jump_y",
      },
      Case{
          .target_elements =
              "<Prescribed_viscous_traction_jump_x>1.0</Prescribed_viscous_traction_jump_x>"
              "<PrescribedViscousTractionJumpX>2.0</PrescribedViscousTractionJumpX>"
              "<Prescribed_viscous_traction_jump_y>3.0</Prescribed_viscous_traction_jump_y>",
          .expected_diagnostic =
              "ambiguous_two_fluid_parameter:Prescribed_viscous_traction_jump_x",
      },
      Case{
          .target_elements =
              "<Prescribed_viscous_traction_jump_x>1.0</Prescribed_viscous_traction_jump_x>"
              "<Prescribed_viscous_traction_jump_y>2.0</Prescribed_viscous_traction_jump_y>"
              "<Prescribed_viscous_traction_jump_z>0.0</Prescribed_viscous_traction_jump_z>",
          .expected_diagnostic =
              "unsupported_two_fluid_out_of_plane_prescribed_viscous_traction_jump",
      },
  };

  for (const auto& item : cases) {
    SCOPED_TRACE(item.target_elements);
    const std::string xml =
        R"xml(<Add_equation type="fluid">
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
)xml" +
        std::string(item.target_elements) + R"xml(
</Add_equation>)xml";
    auto params = parseEquationXml(xml.c_str());
    svmp::FE::systems::FESystem system(mesh);
    const auto scalar_space =
        svmp::FE::spaces::SpaceFactory::create_h1(
            svmp::FE::ElementType::Triangle3, 1);
    (void)system.addField(svmp::FE::systems::FieldSpec{
        .name = "level_set",
        .space = scalar_space,
        .components = 1,
    });
    try {
      (void)application::translators::EquationTranslator::createModule(
          *params, system, meshes);
      FAIL() << "invalid viscous-traction target was accepted";
    } catch (const std::runtime_error& error) {
      EXPECT_NE(std::string_view(error.what()).find(item.expected_diagnostic),
                std::string_view::npos)
          << error.what();
    }
    EXPECT_EQ(system.findFieldByName("u_negative"),
              svmp::FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.findFieldByName("p_negative"),
              svmp::FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.findFieldByName("u_positive"),
              svmp::FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.findFieldByName("p_positive"),
              svmp::FE::INVALID_FIELD_ID);
    EXPECT_TRUE(system.formulationRecords().empty());
    EXPECT_TRUE(
        system.materialInterfaceTransportVelocityDeclarations().empty());
    EXPECT_TRUE(
        system.twoFluidAcceptedStageDiagnosticDeclarations().empty());
  }
}

TEST(EquationTranslatorFreeSurface,
     XmlIncompressibleTwoFluidRejectsDuplicateOperatorTagAliasesBeforeMutation)
{
  svmp::Physics::formulations::navier_stokes::
      forceLink_NavierStokesRegister();
  auto mesh = buildTranslatorTriangleMesh();
  const auto meshes = singleMeshMap(mesh);
  auto params = parseEquationXml(R"xml(
<Add_equation type="fluid">
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
  <OperatorTag>other_equations</OperatorTag>
</Add_equation>
)xml");

  svmp::FE::systems::FESystem system(mesh);
  const auto scalar_space =
      svmp::FE::spaces::SpaceFactory::create_h1(
          svmp::FE::ElementType::Triangle3, 1);
  (void)system.addField(svmp::FE::systems::FieldSpec{
      .name = "level_set",
      .space = scalar_space,
      .components = 1,
  });
  try {
    (void)application::translators::EquationTranslator::createModule(
        *params, system, meshes);
    FAIL() << "duplicate operator-tag aliases were accepted";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(
        std::string_view(error.what()).find(
            "ambiguous_two_fluid_parameter:Operator_tag"),
        std::string_view::npos)
        << error.what();
  }
  EXPECT_EQ(system.findFieldByName("u_negative"),
            svmp::FE::INVALID_FIELD_ID);
  EXPECT_EQ(system.findFieldByName("p_negative"),
            svmp::FE::INVALID_FIELD_ID);
  EXPECT_EQ(system.findFieldByName("u_positive"),
            svmp::FE::INVALID_FIELD_ID);
  EXPECT_EQ(system.findFieldByName("p_positive"),
            svmp::FE::INVALID_FIELD_ID);
  EXPECT_TRUE(system.formulationRecords().empty());
  EXPECT_TRUE(
      system.materialInterfaceTransportVelocityDeclarations().empty());
  EXPECT_TRUE(
      system.twoFluidAcceptedStageDiagnosticDeclarations().empty());
}

TEST(EquationTranslatorFreeSurface,
     XmlIncompressibleTwoFluidRejectsUnusedEquationControlsBeforeMutation)
{
  svmp::Physics::formulations::navier_stokes::
      forceLink_NavierStokesRegister();
  auto mesh = buildTranslatorTriangleMesh();
  const auto meshes = singleMeshMap(mesh);
  struct UnsupportedControl {
    std::string_view label;
    std::string_view xml;
    std::string_view expected_parameter;
  };
  const std::array<UnsupportedControl, 3> unsupported_controls{{
      {"initialize", "<Initialize>zero</Initialize>", "Initialize"},
      {"initialize_rcr",
       "<Initialize_RCR_from_flow>true</Initialize_RCR_from_flow>",
       "Initialize_RCR_from_flow"},
      {"prestress", "<Prestress>true</Prestress>", "Prestress"},
  }};

  for (const auto& control : unsupported_controls) {
    SCOPED_TRACE(control.label);
    const std::string xml = std::string(R"xml(
<Add_equation type="fluid">
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
)xml") + std::string(control.xml) + R"xml(
</Add_equation>
)xml";
    auto params = parseEquationXml(xml.c_str());

    svmp::FE::systems::FESystem system(mesh);
    const auto scalar_space =
        svmp::FE::spaces::SpaceFactory::create_h1(
            svmp::FE::ElementType::Triangle3, 1);
    (void)system.addField(svmp::FE::systems::FieldSpec{
        .name = "level_set",
        .space = scalar_space,
        .components = 1,
    });
    try {
      (void)application::translators::EquationTranslator::createModule(
          *params, system, meshes);
      ADD_FAILURE()
          << "unused equation control was accepted by two-fluid input";
    } catch (const std::runtime_error& error) {
      EXPECT_NE(
          std::string_view(error.what()).find(
              std::string("unsupported_two_fluid_equation_parameter:") +
              std::string(control.expected_parameter)),
          std::string_view::npos)
          << error.what();
    }
    EXPECT_EQ(system.findFieldByName("u_negative"),
              svmp::FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.findFieldByName("p_negative"),
              svmp::FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.findFieldByName("u_positive"),
              svmp::FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.findFieldByName("p_positive"),
              svmp::FE::INVALID_FIELD_ID);
    EXPECT_TRUE(system.formulationRecords().empty());
    EXPECT_TRUE(
        system.materialInterfaceTransportVelocityDeclarations().empty());
    EXPECT_TRUE(
        system.twoFluidAcceptedStageDiagnosticDeclarations().empty());
  }
}

TEST(EquationTranslatorFreeSurface,
     XmlIncompressibleTwoFluidRejectsOutputBlocksBeforeMutation)
{
  svmp::Physics::formulations::navier_stokes::
      forceLink_NavierStokesRegister();
  auto mesh = buildTranslatorTriangleMesh();
  const auto meshes = singleMeshMap(mesh);
  auto params = parseEquationXml(R"xml(
<Add_equation type="fluid">
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
  <Output type="Spatial"><Velocity>true</Velocity></Output>
</Add_equation>
)xml");
  ASSERT_EQ(params->outputs.size(), 1u);

  svmp::FE::systems::FESystem system(mesh);
  const auto scalar_space =
      svmp::FE::spaces::SpaceFactory::create_h1(
          svmp::FE::ElementType::Triangle3, 1);
  (void)system.addField(svmp::FE::systems::FieldSpec{
      .name = "level_set",
      .space = scalar_space,
      .components = 1,
  });
  try {
    (void)application::translators::EquationTranslator::createModule(
        *params, system, meshes);
    FAIL() << "unsupported output block was accepted";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(
        std::string_view(error.what()).find(
            "unsupported_two_fluid_output_configuration"),
        std::string_view::npos)
        << error.what();
  }
  EXPECT_EQ(system.findFieldByName("u_negative"),
            svmp::FE::INVALID_FIELD_ID);
  EXPECT_EQ(system.findFieldByName("p_negative"),
            svmp::FE::INVALID_FIELD_ID);
  EXPECT_EQ(system.findFieldByName("u_positive"),
            svmp::FE::INVALID_FIELD_ID);
  EXPECT_EQ(system.findFieldByName("p_positive"),
            svmp::FE::INVALID_FIELD_ID);
  EXPECT_TRUE(system.formulationRecords().empty());
  EXPECT_TRUE(
      system.materialInterfaceTransportVelocityDeclarations().empty());
  EXPECT_TRUE(
      system.twoFluidAcceptedStageDiagnosticDeclarations().empty());
}

TEST(EquationTranslatorFreeSurface,
     XmlIncompressibleTwoFluidRejectsUnusedNestedBlocksBeforeMutation)
{
  svmp::Physics::formulations::navier_stokes::
      forceLink_NavierStokesRegister();
  auto mesh = buildTranslatorTriangleMesh();
  const auto meshes = singleMeshMap(mesh);
  struct UnsupportedBlock {
    std::string_view label;
    std::string_view xml;
    std::string_view diagnostic;
  };
  const std::array<UnsupportedBlock, 8> unsupported_blocks{{
      {"cplbc",
       R"xml(<Couple_to_cplBC type="Neumann">
  <File_name_for_0D_3D_communication>exchange.dat</File_name_for_0D_3D_communication>
  <File_name_for_saving_unknowns>unknowns.dat</File_name_for_saving_unknowns>
  <Number_of_unknowns>1</Number_of_unknowns>
  <Number_of_user_defined_outputs>0</Number_of_user_defined_outputs>
  <ZeroD_code_file_path>model.py</ZeroD_code_file_path>
</Couple_to_cplBC>)xml",
       "unsupported_two_fluid_equation_block:Couple_to_cplBC"},
      {"genbc",
       R"xml(<Couple_to_genBC type="Neumann">
  <ZeroD_code_file_path>model.py</ZeroD_code_file_path>
</Couple_to_genBC>)xml",
       "unsupported_two_fluid_equation_block:Couple_to_genBC"},
      {"zero_d",
       R"xml(<svZeroDSolver_interface>
  <Coupling_type>implicit</Coupling_type>
  <Configuration_file>model.json</Configuration_file>
  <Shared_library>libmodel.so</Shared_library>
</svZeroDSolver_interface>)xml",
       "unsupported_two_fluid_equation_block:svZeroDSolver_interface"},
      {"remesher", R"xml(<Remesher type="Tetgen"/>)xml",
       "unsupported_two_fluid_equation_block:Remesher"},
      {"variable_wall",
       R"xml(<Variable_wall_properties mesh_name="mesh">
  <Wall_properties_file_path>wall.dat</Wall_properties_file_path>
</Variable_wall_properties>)xml",
       "unsupported_two_fluid_equation_block:Variable_wall_properties"},
      {"ecg",
       R"xml(<ECGLeads><X_coords_file_path>x.dat</X_coords_file_path></ECGLeads>)xml",
       "unsupported_two_fluid_equation_block:ECGLeads"},
      {"fiber",
       R"xml(<Fiber_reinforcement_stress type="Unsteady"><Value>1.0</Value></Fiber_reinforcement_stress>)xml",
       "unsupported_two_fluid_default_domain_block:Fiber_reinforcement_stress"},
      {"stimulus",
       R"xml(<Stimulus type="Istim"><Amplitude>1.0</Amplitude></Stimulus>)xml",
       "unsupported_two_fluid_default_domain_block:Stimulus"},
  }};

  for (const auto& block : unsupported_blocks) {
    SCOPED_TRACE(block.label);
    const std::string xml = std::string(R"xml(
<Add_equation type="fluid">
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
)xml") + std::string(block.xml) + R"xml(
</Add_equation>
)xml";
    auto params = parseEquationXml(xml.c_str());

    svmp::FE::systems::FESystem system(mesh);
    const auto scalar_space =
        svmp::FE::spaces::SpaceFactory::create_h1(
            svmp::FE::ElementType::Triangle3, 1);
    (void)system.addField(svmp::FE::systems::FieldSpec{
        .name = "level_set",
        .space = scalar_space,
        .components = 1,
    });
    try {
      (void)application::translators::EquationTranslator::createModule(
          *params, system, meshes);
      ADD_FAILURE() << "unused nested block was accepted by two-fluid input";
    } catch (const std::runtime_error& error) {
      EXPECT_NE(
          std::string_view(error.what()).find(block.diagnostic),
          std::string_view::npos)
          << error.what();
    }
    EXPECT_EQ(system.findFieldByName("u_negative"),
              svmp::FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.findFieldByName("p_negative"),
              svmp::FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.findFieldByName("u_positive"),
              svmp::FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.findFieldByName("p_positive"),
              svmp::FE::INVALID_FIELD_ID);
    EXPECT_TRUE(system.formulationRecords().empty());
    EXPECT_TRUE(
        system.materialInterfaceTransportVelocityDeclarations().empty());
    EXPECT_TRUE(
        system.twoFluidAcceptedStageDiagnosticDeclarations().empty());
  }
}

TEST(EquationTranslatorFreeSurface,
     XmlIncompressibleTwoFluidRejectsUnusedBoundaryControlsBeforeMutation)
{
  svmp::Physics::formulations::navier_stokes::
      forceLink_NavierStokesRegister();
  auto mesh = buildTranslatorTriangleMesh();
  mesh->base().register_label("wall", 9);
  const auto meshes = singleMeshMap(mesh);
  struct UnsupportedBoundaryControl {
    std::string_view label;
    std::string_view xml;
    std::string_view diagnostic;
  };
  const std::array<UnsupportedBoundaryControl, 3> unsupported_controls{{
      {"unused_parameter",
       R"xml(<Add_BC name="wall">
  <Type>Dir</Type>
  <Value>0.0</Value>
  <Damping>1.0</Damping>
</Add_BC>)xml",
       "unsupported_two_fluid_boundary_parameter:Damping"},
      {"unused_nested_block",
       R"xml(<Add_BC name="wall">
  <Type>Dir</Type>
  <Value>0.0</Value>
  <RCR_values>
    <Capacitance>1.0</Capacitance>
    <Distal_resistance>2.0</Distal_resistance>
    <Proximal_resistance>3.0</Proximal_resistance>
  </RCR_values>
</Add_BC>)xml",
       "unsupported_two_fluid_boundary_block:RCR_values"},
      {"unused_nested_two_capacitor_block",
       R"xml(<Add_BC name="wall">
  <Type>Dir</Type>
  <Value>0.0</Value>
  <RCRCR_values>
    <Distal_capacitance>1.0</Distal_capacitance>
    <Distal_resistance>2.0</Distal_resistance>
    <Intermediate_resistance>3.0</Intermediate_resistance>
    <Proximal_capacitance>4.0</Proximal_capacitance>
    <Proximal_resistance>5.0</Proximal_resistance>
  </RCRCR_values>
</Add_BC>)xml",
       "unsupported_two_fluid_boundary_block:RCRCR_values"},
  }};

  for (const auto& control : unsupported_controls) {
    SCOPED_TRACE(control.label);
    const std::string xml = std::string(R"xml(
<Add_equation type="fluid">
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
)xml") + std::string(control.xml) + R"xml(
</Add_equation>
)xml";
    auto params = parseEquationXml(xml.c_str());

    svmp::FE::systems::FESystem system(mesh);
    const auto scalar_space =
        svmp::FE::spaces::SpaceFactory::create_h1(
            svmp::FE::ElementType::Triangle3, 1);
    (void)system.addField(svmp::FE::systems::FieldSpec{
        .name = "level_set",
        .space = scalar_space,
        .components = 1,
    });
    try {
      (void)application::translators::EquationTranslator::createModule(
          *params, system, meshes);
      ADD_FAILURE()
          << "unused boundary control was accepted by two-fluid input";
    } catch (const std::runtime_error& error) {
      EXPECT_NE(
          std::string_view(error.what()).find(control.diagnostic),
          std::string_view::npos)
          << error.what();
    }
    EXPECT_EQ(system.findFieldByName("u_negative"),
              svmp::FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.findFieldByName("p_negative"),
              svmp::FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.findFieldByName("u_positive"),
              svmp::FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.findFieldByName("p_positive"),
              svmp::FE::INVALID_FIELD_ID);
    EXPECT_TRUE(system.formulationRecords().empty());
    EXPECT_TRUE(
        system.materialInterfaceTransportVelocityDeclarations().empty());
    EXPECT_TRUE(
        system.twoFluidAcceptedStageDiagnosticDeclarations().empty());
  }
}

TEST(EquationTranslatorFreeSurface,
     XmlIncompressibleTwoFluidRejectsOnePhaseDomainControlsBeforeMutation)
{
  svmp::Physics::formulations::navier_stokes::
      forceLink_NavierStokesRegister();
  auto mesh = buildTranslatorTriangleMesh();
  const auto meshes = singleMeshMap(mesh);
  struct UnsupportedControl {
    std::string_view label;
    std::string_view xml;
    std::string_view expected_parameter;
  };
  const std::array<UnsupportedControl, 10> unsupported_controls{{
      {"density", "<Density>997.0</Density>", "Density"},
      {"fluid_density", "<Fluid_density>997.0</Fluid_density>",
       "Fluid_density"},
      {"viscosity",
       "<Viscosity model=\"Constant\"><Value>0.001</Value></Viscosity>",
       "Viscosity.Value"},
      {"backflow",
       "<Backflow_stabilization_coefficient>0.2</Backflow_stabilization_coefficient>",
       "Backflow_stabilization_coefficient"},
      {"continuity_stabilization",
       "<Continuity_stabilization_coefficient>0.3</Continuity_stabilization_coefficient>",
       "Continuity_stabilization_coefficient"},
      {"momentum_stabilization",
       "<Momentum_stabilization_coefficient>0.4</Momentum_stabilization_coefficient>",
       "Momentum_stabilization_coefficient"},
      {"brinkman",
       "<Inverse_darcy_permeability>1.0</Inverse_darcy_permeability>",
       "Inverse_darcy_permeability"},
      {"hydrostatic_enable",
       "<Hydrostatic_pressure_initialization>true</Hydrostatic_pressure_initialization>",
       "Hydrostatic_pressure_initialization"},
      {"hydrostatic_reference",
       "<Hydrostatic_pressure_reference>12.0</Hydrostatic_pressure_reference>",
       "Hydrostatic_pressure_reference"},
      {"hydrostatic_field",
       "<Hydrostatic_pressure_field_name>pressure</Hydrostatic_pressure_field_name>",
       "Hydrostatic_pressure_field_name"},
  }};

  for (const auto& control : unsupported_controls) {
    SCOPED_TRACE(control.label);
    const std::string xml = std::string(R"xml(
<Add_equation type="fluid">
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
)xml") + std::string(control.xml) + R"xml(
</Add_equation>
)xml";
    auto params = parseEquationXml(xml.c_str());

    svmp::FE::systems::FESystem system(mesh);
    const auto scalar_space =
        svmp::FE::spaces::SpaceFactory::create_h1(
            svmp::FE::ElementType::Triangle3, 1);
    (void)system.addField(svmp::FE::systems::FieldSpec{
        .name = "level_set",
        .space = scalar_space,
        .components = 1,
    });
    try {
      (void)application::translators::EquationTranslator::createModule(
          *params, system, meshes);
      ADD_FAILURE()
          << "one-phase domain control was accepted by two-fluid input";
    } catch (const std::runtime_error& error) {
      EXPECT_NE(
          std::string_view(error.what()).find(
              std::string("unsupported_two_fluid_default_domain_parameter:") +
                  std::string(control.expected_parameter)),
          std::string_view::npos)
          << error.what();
    }
    EXPECT_EQ(system.findFieldByName("u_negative"),
              svmp::FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.findFieldByName("p_negative"),
              svmp::FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.findFieldByName("u_positive"),
              svmp::FE::INVALID_FIELD_ID);
    EXPECT_EQ(system.findFieldByName("p_positive"),
              svmp::FE::INVALID_FIELD_ID);
    EXPECT_TRUE(system.formulationRecords().empty());
    EXPECT_TRUE(
        system.materialInterfaceTransportVelocityDeclarations().empty());
    EXPECT_TRUE(
        system.twoFluidAcceptedStageDiagnosticDeclarations().empty());
  }
}

TEST(EquationTranslatorFreeSurface,
     XmlIncompressibleTwoFluidRejectsLegacyBodyForceBlockBeforeMutation)
{
  svmp::Physics::formulations::navier_stokes::
      forceLink_NavierStokesRegister();
  auto mesh = buildTranslatorTriangleMesh();
  const auto meshes = singleMeshMap(mesh);
  auto params = parseEquationXml(R"xml(
<Add_equation type="fluid">
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
  <Add_BF mesh="mesh">
    <Type>volumetric</Type>
    <Value>1.0</Value>
  </Add_BF>
</Add_equation>
)xml");
  ASSERT_EQ(params->body_forces.size(), 1u);

  svmp::FE::systems::FESystem system(mesh);
  const auto scalar_space =
      svmp::FE::spaces::SpaceFactory::create_h1(
          svmp::FE::ElementType::Triangle3, 1);
  (void)system.addField(svmp::FE::systems::FieldSpec{
      .name = "level_set",
      .space = scalar_space,
      .components = 1,
  });
  try {
    (void)application::translators::EquationTranslator::createModule(
        *params, system, meshes);
    FAIL() << "legacy body-force block was silently ignored";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(
        std::string_view(error.what()).find(
            "unsupported_two_fluid_body_force_configuration"),
        std::string_view::npos)
        << error.what();
  }
  EXPECT_EQ(system.findFieldByName("u_negative"),
            svmp::FE::INVALID_FIELD_ID);
  EXPECT_EQ(system.findFieldByName("p_negative"),
            svmp::FE::INVALID_FIELD_ID);
  EXPECT_EQ(system.findFieldByName("u_positive"),
            svmp::FE::INVALID_FIELD_ID);
  EXPECT_EQ(system.findFieldByName("p_positive"),
            svmp::FE::INVALID_FIELD_ID);
  EXPECT_TRUE(system.formulationRecords().empty());
  EXPECT_TRUE(
      system.materialInterfaceTransportVelocityDeclarations().empty());
  EXPECT_TRUE(
      system.twoFluidAcceptedStageDiagnosticDeclarations().empty());
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
