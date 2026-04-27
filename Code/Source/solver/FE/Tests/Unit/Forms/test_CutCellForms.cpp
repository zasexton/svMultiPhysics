#include "Forms/CutCellForms.h"

#include <gtest/gtest.h>

using namespace svmp::FE::forms;

TEST(CutCellForms, BuildsParameterBackedCutMetadataTerminals)
{
    CutCellParameterSlots slots;
    slots.volume_fraction = 10;
    slots.side_indicator = 11;
    slots.embedded_normal = {{12, 13, 14}};
    slots.stabilization_scale = 15;

    const auto terminals = cutCellTerminals(slots);
    EXPECT_NE(terminals.volume_fraction.toString().find("param[10]"), std::string::npos);
    EXPECT_NE(terminals.side_indicator.toString().find("param[11]"), std::string::npos);
    EXPECT_NE(terminals.embedded_normal.toString().find("param[12]"), std::string::npos);
    EXPECT_NE(terminals.embedded_normal.toString().find("param[14]"), std::string::npos);
    EXPECT_NE(terminals.stabilization_scale.toString().find("param[15]"), std::string::npos);
}
