#include "Interfaces/LevelSetInterfaceBuilder.h"
#include "Interfaces/LevelSetInterfaceGeometryWriter.h"

#include <gtest/gtest.h>

#include <string>

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
}
