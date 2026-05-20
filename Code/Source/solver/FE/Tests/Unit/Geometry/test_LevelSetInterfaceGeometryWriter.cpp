#include "Interfaces/LevelSetInterfaceBuilder.h"
#include "Interfaces/LevelSetInterfaceGeometryWriter.h"

#include <gtest/gtest.h>

#include <array>
#include <string>
#include <utility>
#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::interfaces;

namespace {

CutInterfaceDomainRequest writer_request()
{
    CutInterfaceDomainRequest request;
    request.source = LevelSetInterfaceSource::fromField(/*field_id=*/12,
                                                        /*layout_revision=*/1,
                                                        /*value_revision=*/1);
    request.interface_marker = 91;
    return request;
}

bool contains_common_diagnostic_arrays(const std::string& xml)
{
    return xml.find("Name=\"level_set_value\"") != std::string::npos &&
           xml.find("Name=\"parent_coordinate\" NumberOfComponents=\"3\"") != std::string::npos &&
           xml.find("Name=\"interface_normal\" NumberOfComponents=\"3\"") != std::string::npos &&
           xml.find("Name=\"curvature_estimate\"") != std::string::npos &&
           xml.find("Name=\"negative_volume_fraction\"") != std::string::npos &&
           xml.find("Name=\"positive_volume_fraction\"") != std::string::npos &&
           xml.find("Name=\"interface_marker\"") != std::string::npos &&
           xml.find("Name=\"parent_cell\"") != std::string::npos &&
           xml.find("Name=\"fragment_stable_id\"") != std::string::npos &&
           xml.find("Name=\"fragment_kind\"") != std::string::npos &&
           xml.find("Name=\"curved_patch\"") != std::string::npos &&
           xml.find("Name=\"local_fragment_index\"") != std::string::npos &&
           xml.find("Name=\"conditioning_status\"") != std::string::npos &&
           xml.find("Name=\"requested_quadrature_order\"") != std::string::npos &&
           xml.find("Name=\"achieved_quadrature_order\"") != std::string::npos &&
           xml.find("Name=\"fallback_status\"") != std::string::npos &&
           xml.find("Name=\"root_polished\"") != std::string::npos &&
           xml.find("Name=\"root_finder_iterations\"") != std::string::npos &&
           xml.find("Name=\"max_root_residual\"") != std::string::npos &&
           xml.find("Name=\"min_gradient_norm\"") != std::string::npos;
}

Real circle_level_set(const std::array<Real, 3>& point)
{
    constexpr Real radius = Real{0.75};
    return point[0] * point[0] + point[1] * point[1] - radius * radius;
}

Real sphere_level_set(const std::array<Real, 3>& point)
{
    constexpr Real radius = Real{0.75};
    return point[0] * point[0] + point[1] * point[1] + point[2] * point[2] -
           radius * radius;
}

} // namespace

TEST(LevelSetInterfaceGeometryWriter, WritesGeneratedSegmentAsVtpLine)
{
    LevelSetInterfaceDomain domain(writer_request());
    appendLinearLevelSetCellCut2D(
        domain,
        LevelSetCellCutInput{.parent_cell = 1,
                             .element_type = ElementType::Quad4,
                             .node_coordinates = {{{0.0, 0.0, 0.0}},
                                                  {{1.0, 0.0, 0.0}},
                                                  {{1.0, 1.0, 0.0}},
                                                  {{0.0, 1.0, 0.0}}},
                             .level_set_values = {-0.5, 0.5, 0.5, -0.5}});

    const std::string xml = levelSetInterfaceGeometryVtpString(domain);

    EXPECT_NE(xml.find("<VTKFile type=\"PolyData\""), std::string::npos);
    EXPECT_NE(xml.find("NumberOfPoints=\"3\""), std::string::npos);
    EXPECT_NE(xml.find("NumberOfVerts=\"1\""), std::string::npos);
    EXPECT_NE(xml.find("NumberOfLines=\"1\""), std::string::npos);
    EXPECT_NE(xml.find("NumberOfPolys=\"0\""), std::string::npos);
    EXPECT_NE(xml.find("0.5 0 0"), std::string::npos);
    EXPECT_NE(xml.find("0.5 1 0"), std::string::npos);
    EXPECT_NE(xml.find("<PointData>"), std::string::npos);
    EXPECT_NE(xml.find("Name=\"level_set_value\""), std::string::npos);
    EXPECT_NE(xml.find("<CellData>"), std::string::npos);
    EXPECT_NE(xml.find("Name=\"interface_normal\" NumberOfComponents=\"3\""), std::string::npos);
    EXPECT_NE(xml.find("Name=\"curvature_estimate\""), std::string::npos);
    EXPECT_NE(xml.find("Name=\"negative_volume_fraction\""), std::string::npos);
    EXPECT_NE(xml.find("Name=\"positive_volume_fraction\""), std::string::npos);
    EXPECT_NE(xml.find("Name=\"interface_marker\""), std::string::npos);
    EXPECT_NE(xml.find("> 91 91 </DataArray>"), std::string::npos);
    EXPECT_NE(xml.find("Name=\"parent_cell\""), std::string::npos);
    EXPECT_NE(xml.find("> 1 </DataArray>"), std::string::npos);
    EXPECT_NE(xml.find("Name=\"root_polished\""), std::string::npos);
    EXPECT_NE(xml.find("Name=\"root_finder_iterations\""), std::string::npos);
    EXPECT_NE(xml.find("Name=\"max_root_residual\""), std::string::npos);
    EXPECT_NE(xml.find("Name=\"min_gradient_norm\""), std::string::npos);
}

TEST(LevelSetInterfaceGeometryWriter, WritesPlaneCutDiagnosticOutput)
{
    LevelSetInterfaceDomain domain(writer_request());
    appendLinearLevelSetCellCut3D(
        domain,
        LevelSetCellCutInput{.parent_cell = 2,
                             .element_type = ElementType::Tetra4,
                             .node_coordinates = {{{0.0, 0.0, 0.0}},
                                                  {{1.0, 0.0, 0.0}},
                                                  {{0.0, 1.0, 0.0}},
                                                  {{0.0, 0.0, 1.0}}},
                             .level_set_values = {-0.25, 0.75, -0.25, -0.25}});

    const std::string xml = levelSetInterfaceGeometryVtpString(domain);

    EXPECT_NE(xml.find("NumberOfPoints=\"4\""), std::string::npos);
    EXPECT_NE(xml.find("NumberOfVerts=\"1\""), std::string::npos);
    EXPECT_NE(xml.find("NumberOfPolys=\"1\""), std::string::npos);
    EXPECT_TRUE(contains_common_diagnostic_arrays(xml));
}

TEST(LevelSetInterfaceGeometryWriter, WritesCircleCutDiagnosticOutput)
{
    LevelSetInterfaceDomain domain(writer_request());
    const std::vector<std::array<Real, 3>> coordinates = {
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{1.0, 1.0, 0.0}},
        {{0.0, 1.0, 0.0}}};
    appendLinearLevelSetCellCut2D(
        domain,
        LevelSetCellCutInput{.parent_cell = 3,
                             .element_type = ElementType::Quad4,
                             .node_coordinates = coordinates,
                             .level_set_values = {circle_level_set(coordinates[0]),
                                                  circle_level_set(coordinates[1]),
                                                  circle_level_set(coordinates[2]),
                                                  circle_level_set(coordinates[3])}});

    const std::string xml = levelSetInterfaceGeometryVtpString(domain);

    EXPECT_NE(xml.find("NumberOfPoints=\"3\""), std::string::npos);
    EXPECT_NE(xml.find("NumberOfVerts=\"1\""), std::string::npos);
    EXPECT_NE(xml.find("NumberOfLines=\"1\""), std::string::npos);
    EXPECT_TRUE(contains_common_diagnostic_arrays(xml));
}

TEST(LevelSetInterfaceGeometryWriter, WritesSphereCutDiagnosticOutput)
{
    LevelSetInterfaceDomain domain(writer_request());
    const std::vector<std::array<Real, 3>> coordinates = {
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}},
        {{0.0, 0.0, 1.0}}};
    appendLinearLevelSetCellCut3D(
        domain,
        LevelSetCellCutInput{.parent_cell = 4,
                             .element_type = ElementType::Tetra4,
                             .node_coordinates = coordinates,
                             .level_set_values = {sphere_level_set(coordinates[0]),
                                                  sphere_level_set(coordinates[1]),
                                                  sphere_level_set(coordinates[2]),
                                                  sphere_level_set(coordinates[3])}});

    const std::string xml = levelSetInterfaceGeometryVtpString(domain);

    EXPECT_NE(xml.find("NumberOfPoints=\"4\""), std::string::npos);
    EXPECT_NE(xml.find("NumberOfVerts=\"1\""), std::string::npos);
    EXPECT_NE(xml.find("NumberOfPolys=\"1\""), std::string::npos);
    EXPECT_TRUE(contains_common_diagnostic_arrays(xml));
}

TEST(LevelSetInterfaceGeometryWriter,
     WritesCurvedPatchQuadraturePointsAsVtpVerts)
{
    auto request = writer_request();
    request.interface_quadrature_order = 3;
    request.implicit_fallback_status = "None";
    LevelSetInterfaceDomain domain(request);
    CutInterfaceFragment patch;
    patch.interface_marker = 91;
    patch.parent_cell = 7;
    patch.local_fragment_index = 3;
    patch.stable_id = 7007;
    patch.kind = CutInterfaceFragmentKind::CurvedPatch;
    patch.normal = {{0.0, 1.0, 0.0}};
    patch.measure = 1.0;
    patch.curvature_estimate = 2.0;
    patch.negative_volume_fraction = 0.4;
    patch.positive_volume_fraction = 0.6;
    patch.root_polished = true;
    patch.root_finder_iterations = 9;
    patch.max_root_residual = 1.0e-12;
    patch.min_gradient_norm = 3.0;
    patch.conditioning_diagnostic = "ok";
    patch.implicit_fallback_status = "Used";
    patch.vertices = {
        CutInterfaceVertex{.point = {{0.0, 0.0, 0.0}},
                           .parent_coordinate = {{0.0, 0.0, 0.0}},
                           .level_set_value = 0.0},
        CutInterfaceVertex{.point = {{1.0, 0.0, 0.0}},
                           .parent_coordinate = {{1.0, 0.0, 0.0}},
                           .level_set_value = 0.0}};
    patch.quadrature_points = {
        CutInterfaceQuadraturePoint{.point = {{0.25, 0.125, 0.0}},
                                    .parent_coordinate = {{0.25, 0.5, 0.0}},
                                    .normal = {{0.0, 1.0, 0.0}},
                                    .weight = 0.5,
                                    .reference_measure_factor = 1.0,
                                    .level_set_residual = 1.0e-12,
                                    .gradient_norm = 3.0},
        CutInterfaceQuadraturePoint{.point = {{0.75, 0.125, 0.0}},
                                    .parent_coordinate = {{0.75, 0.5, 0.0}},
                                    .normal = {{0.0, 1.0, 0.0}},
                                    .weight = 0.5,
                                    .reference_measure_factor = 1.0,
                                    .level_set_residual = 2.0e-12,
                                    .gradient_norm = 3.5}};
    domain.addFragment(std::move(patch));

    const auto rules = domain.interfaceQuadratureRules();
    ASSERT_EQ(rules.size(), 1u);
    EXPECT_EQ(rules.front().provenance.implicit_fallback_status, "Used");

    const std::string xml = levelSetInterfaceGeometryVtpString(domain);

    EXPECT_NE(xml.find("NumberOfPoints=\"4\""), std::string::npos);
    EXPECT_NE(xml.find("NumberOfVerts=\"2\""), std::string::npos);
    EXPECT_NE(xml.find("NumberOfLines=\"1\""), std::string::npos);
    EXPECT_NE(xml.find("0.25 0.125 0"), std::string::npos);
    EXPECT_NE(xml.find("0.75 0.125 0"), std::string::npos);
    EXPECT_NE(xml.find("<Verts>"), std::string::npos);
    EXPECT_NE(xml.find("Name=\"parent_coordinate\""), std::string::npos);
    EXPECT_NE(xml.find("0.25 0.5 0"), std::string::npos);
    EXPECT_NE(xml.find("0.75 0.5 0"), std::string::npos);
    EXPECT_NE(xml.find("Name=\"quadrature_point\""), std::string::npos);
    EXPECT_NE(xml.find("Name=\"fragment_stable_id\""), std::string::npos);
    EXPECT_NE(xml.find("Name=\"fragment_kind\""), std::string::npos);
    EXPECT_NE(xml.find("Name=\"curved_patch\""), std::string::npos);
    EXPECT_NE(xml.find("Name=\"local_fragment_index\""), std::string::npos);
    EXPECT_NE(xml.find("Name=\"conditioning_status\""), std::string::npos);
    EXPECT_NE(xml.find("Name=\"requested_quadrature_order\" format=\"ascii\"> 3 3 3 </DataArray>"),
              std::string::npos);
    EXPECT_NE(xml.find("Name=\"achieved_quadrature_order\" format=\"ascii\"> 3 3 3 </DataArray>"),
              std::string::npos);
    EXPECT_NE(xml.find("Name=\"fallback_status\" format=\"ascii\"> 2 2 2 </DataArray>"),
              std::string::npos);
    EXPECT_NE(xml.find("Name=\"quadrature_weight\""), std::string::npos);
    EXPECT_NE(xml.find("Name=\"reference_measure_factor\""), std::string::npos);
    EXPECT_NE(xml.find("Name=\"quadrature_level_set_residual\""), std::string::npos);
    EXPECT_NE(xml.find("Name=\"quadrature_gradient_norm\""), std::string::npos);
    EXPECT_NE(xml.find("> 1 1 0 </DataArray>"), std::string::npos);
    EXPECT_NE(xml.find("> 7007 7007 7007 </DataArray>"), std::string::npos);
    EXPECT_NE(xml.find("> 2 2 2 </DataArray>"), std::string::npos);
    EXPECT_NE(xml.find("> 1 1 1 </DataArray>"), std::string::npos);
    EXPECT_NE(xml.find("> 3 3 3 </DataArray>"), std::string::npos);
    EXPECT_NE(xml.find("Name=\"conditioning_status\" format=\"ascii\"> 1 1 1 </DataArray>"),
              std::string::npos);
    EXPECT_NE(xml.find("> 0.5 0.5 0 </DataArray>"), std::string::npos);
}

TEST(LevelSetInterfaceGeometryWriter, DumpsSingleCellQuadratureDebugJson)
{
    auto request = writer_request();
    request.implicit_geometry_mode = "high-order-level-set";
    request.implicit_quadrature_backend = "saye-hyperrectangle";
    request.implicit_fallback_policy = "linear-topology";
    request.implicit_fallback_status = "Used";
    request.geometry_tangent_policy = "refreshed-frozen-quadrature";
    request.implicit_cut_root_tolerance = 1.0e-9;
    request.implicit_cut_root_coordinate_tolerance = 2.0e-11;
    request.implicit_cut_root_max_iterations = 37;
    request.quadrature_policy_key = 77;
    request.interface_quadrature_order = 1;
    request.volume_quadrature_order = 1;

    LevelSetInterfaceDomain domain(request);
    const std::vector<std::array<Real, 3>> coordinates = {
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{1.0, 1.0, 0.0}},
        {{0.0, 1.0, 0.0}}};

    appendLinearLevelSetCellCut2D(
        domain,
        LevelSetCellCutInput{.parent_cell = 11,
                             .element_type = ElementType::Quad4,
                             .node_coordinates = coordinates,
                             .level_set_values = {-0.5, 0.5, 0.5, -0.5}});
    appendLinearLevelSetCellCut2D(
        domain,
        LevelSetCellCutInput{.parent_cell = 12,
                             .element_type = ElementType::Quad4,
                             .node_coordinates = coordinates,
                             .level_set_values = {-0.25, 0.75, 0.75, -0.25}});

    const std::string json =
        levelSetInterfaceQuadratureDebugJsonString(domain, /*parent_cell=*/11);

    EXPECT_NE(json.find("\"parent_cell\": 11"), std::string::npos);
    EXPECT_NE(json.find("\"implicit_geometry_mode\": \"high-order-level-set\""),
              std::string::npos);
    EXPECT_NE(json.find("\"implicit_quadrature_backend\": \"saye-hyperrectangle\""),
              std::string::npos);
    EXPECT_NE(json.find("\"selected_implicit_quadrature_backend\": \"saye-hyperrectangle\""),
              std::string::npos);
    EXPECT_NE(json.find("\"implicit_fallback_policy\": \"linear-topology\""),
              std::string::npos);
    EXPECT_NE(json.find("\"implicit_fallback_status\": \"Used\""),
              std::string::npos);
    EXPECT_NE(json.find("\"geometry_tangent_policy\": \"refreshed-frozen-quadrature\""),
              std::string::npos);
    EXPECT_NE(json.find(
                  "\"implicit_cut_root_tolerance\": 1.0000000000000001e-09"),
              std::string::npos);
    EXPECT_NE(json.find("\"implicit_cut_root_coordinate_tolerance\""),
              std::string::npos);
    EXPECT_NE(json.find("\"implicit_cut_root_max_iterations\": 37"),
              std::string::npos);
    EXPECT_NE(json.find("\"interface_rules\": ["), std::string::npos);
    EXPECT_NE(json.find("\"volume_rules\": ["), std::string::npos);
    EXPECT_NE(json.find("\"quadrature_points\""), std::string::npos);
    EXPECT_NE(json.find("\"parent_coordinate\""), std::string::npos);
    EXPECT_NE(json.find("\"reference_measure_factor\""), std::string::npos);
    EXPECT_NE(json.find("\"level_set_residual\""), std::string::npos);
    EXPECT_NE(json.find("\"gradient_norm\""), std::string::npos);
    EXPECT_NE(json.find("\"parent_entity\": 11"), std::string::npos);
    EXPECT_NE(json.find("\"side\": \"negative\""), std::string::npos);
    EXPECT_NE(json.find("\"side\": \"positive\""), std::string::npos);
    EXPECT_EQ(json.find("\"parent_entity\": 12"), std::string::npos);
}
