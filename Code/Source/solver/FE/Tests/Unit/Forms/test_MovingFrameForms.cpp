#include "Forms/MovingFrameForms.h"

#include <gtest/gtest.h>

#include <string>

using namespace svmp::FE::forms;

TEST(MovingFrameForms, BuildsFrameTerminalsFromParameterSlots)
{
    MovingFrameParameterSlots slots;
    slots.origin = {{20u, 21u, 22u}};
    slots.angular_velocity = {{30u, 31u, 32u}};

    const auto terminals = movingFrameTerminals(slots);
    EXPECT_TRUE(terminals.current_coordinate.isValid());
    EXPECT_TRUE(terminals.mesh_velocity.isValid());
    EXPECT_TRUE(terminals.relative_velocity.isValid());

    const auto origin = terminals.frame_origin.toString();
    EXPECT_NE(origin.find("param[20]"), std::string::npos);
    EXPECT_NE(origin.find("param[21]"), std::string::npos);
    EXPECT_NE(origin.find("param[22]"), std::string::npos);

    const auto rel = terminals.relative_velocity.toString();
    EXPECT_NE(rel.find("meshVelocity"), std::string::npos);
    EXPECT_NE(rel.find("param[30]"), std::string::npos);
    EXPECT_NE(rel.find("x_current"), std::string::npos);
}
