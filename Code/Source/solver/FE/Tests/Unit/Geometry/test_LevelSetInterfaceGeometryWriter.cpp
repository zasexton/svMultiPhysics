#include "Interfaces/LevelSetInterfaceBuilder.h"
#include "Interfaces/LevelSetInterfaceGeometryWriter.h"

#include <gtest/gtest.h>

#include <array>
#include <string>
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
           xml.find("Name=\"interface_normal\" NumberOfComponents=\"3\"") != std::string::npos &&
           xml.find("Name=\"curvature_estimate\"") != std::string::npos &&
           xml.find("Name=\"negative_volume_fraction\"") != std::string::npos &&
           xml.find("Name=\"positive_volume_fraction\"") != std::string::npos &&
           xml.find("Name=\"interface_marker\"") != std::string::npos &&
           xml.find("Name=\"parent_cell\"") != std::string::npos;
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
    EXPECT_NE(xml.find("NumberOfPoints=\"2\""), std::string::npos);
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
    EXPECT_NE(xml.find("> 91 </DataArray>"), std::string::npos);
    EXPECT_NE(xml.find("Name=\"parent_cell\""), std::string::npos);
    EXPECT_NE(xml.find("> 1 </DataArray>"), std::string::npos);
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

    EXPECT_NE(xml.find("NumberOfPoints=\"3\""), std::string::npos);
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

    EXPECT_NE(xml.find("NumberOfPoints=\"2\""), std::string::npos);
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

    EXPECT_NE(xml.find("NumberOfPoints=\"3\""), std::string::npos);
    EXPECT_NE(xml.find("NumberOfPolys=\"1\""), std::string::npos);
    EXPECT_TRUE(contains_common_diagnostic_arrays(xml));
}

TEST(LevelSetInterfaceGeometryWriter, DumpsSingleCellQuadratureDebugJson)
{
    auto request = writer_request();
    request.implicit_geometry_mode = "high-order-level-set";
    request.implicit_quadrature_backend = "saye-hyperrectangle";
    request.implicit_fallback_policy = "linear-topology";
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
    EXPECT_NE(json.find("\"implicit_fallback_policy\": \"linear-topology\""),
              std::string::npos);
    EXPECT_NE(json.find("\"interface_rules\": ["), std::string::npos);
    EXPECT_NE(json.find("\"volume_rules\": ["), std::string::npos);
    EXPECT_NE(json.find("\"quadrature_points\""), std::string::npos);
    EXPECT_NE(json.find("\"parent_entity\": 11"), std::string::npos);
    EXPECT_NE(json.find("\"side\": \"negative\""), std::string::npos);
    EXPECT_NE(json.find("\"side\": \"positive\""), std::string::npos);
    EXPECT_EQ(json.find("\"parent_entity\": 12"), std::string::npos);
}
