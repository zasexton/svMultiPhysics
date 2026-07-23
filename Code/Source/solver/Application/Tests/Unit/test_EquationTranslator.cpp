#include <gtest/gtest.h>

#include "Application/Translators/EquationTranslator.h"
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
