#include "Interfaces/LevelSetInterfaceBuilder.h"
#include "Interfaces/LevelSetInterfaceDiagnostics.h"

#include <gtest/gtest.h>

using namespace svmp::FE;
using namespace svmp::FE::interfaces;

namespace {

CutInterfaceDomainRequest diagnostics_request()
{
    CutInterfaceDomainRequest request;
    request.source = LevelSetInterfaceSource::fromField(/*field_id=*/13,
                                                        /*layout_revision=*/1,
                                                        /*value_revision=*/1);
    request.interface_marker = 92;
    return request;
}

} // namespace

TEST(LevelSetInterfaceDiagnostics, SummarizesGeneratedInterfaceDomain)
{
    LevelSetInterfaceDomain domain(diagnostics_request());
    appendLinearLevelSetCellCut2D(
        domain,
        LevelSetCellCutInput{.parent_cell = 4,
                             .element_type = ElementType::Quad4,
                             .node_coordinates = {{{0.0, 0.0, 0.0}},
                                                  {{1.0, 0.0, 0.0}},
                                                  {{1.0, 1.0, 0.0}},
                                                  {{0.0, 1.0, 0.0}}},
                             .level_set_values = {-0.5, 0.5, 0.5, -0.5}});
    appendLinearLevelSetCellCut2D(
        domain,
        LevelSetCellCutInput{.parent_cell = 9,
                             .element_type = ElementType::Quad4,
                             .node_coordinates = {{{1.0, 0.0, 0.0}},
                                                  {{2.0, 0.0, 0.0}},
                                                  {{2.0, 1.0, 0.0}},
                                                  {{1.0, 1.0, 0.0}}},
                             .level_set_values = {-0.5, 0.5, 0.5, -0.5}});

    const auto statistics = summarizeLevelSetInterface(domain);

    EXPECT_EQ(statistics.interface_marker, 92);
    EXPECT_EQ(statistics.fragment_count, 2u);
    EXPECT_EQ(statistics.active_fragment_count, 2u);
    EXPECT_EQ(statistics.degenerate_fragment_count, 0u);
    EXPECT_EQ(statistics.quadrature_point_count, 2u);
    EXPECT_EQ(statistics.cut_cell_count, 2u);
    EXPECT_DOUBLE_EQ(statistics.total_interface_measure, 2.0);
    EXPECT_FALSE(statistics.enclosed_measure_available);
    EXPECT_DOUBLE_EQ(statistics.enclosed_measure, 0.0);
}
