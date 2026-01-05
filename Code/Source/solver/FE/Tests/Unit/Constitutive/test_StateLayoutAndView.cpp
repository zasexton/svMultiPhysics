/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Constitutive/StateLayout.h"
#include "Constitutive/StateView.h"

#include <array>
#include <cstddef>
#include <cstring>
#include <new>

namespace svmp {
namespace FE {
namespace constitutive {
namespace test {

TEST(StateLayoutTest, BuilderComputesOffsetsAndStride)
{
    StateLayoutBuilder b("Example");
    b.add<int>("counter");
    b.add<double>("value");

    const auto layout = b.build();
    EXPECT_EQ(layout.bytesPerPoint(), 16u);
    EXPECT_EQ(layout.alignment(), 8u);
    EXPECT_EQ(layout.strideBytes(), 16u);
    ASSERT_EQ(layout.fields().size(), 2u);
    EXPECT_EQ(layout.fields()[0].name, "counter");
    EXPECT_EQ(layout.fields()[0].offset_bytes, 0u);
    EXPECT_EQ(layout.fields()[0].size_bytes, sizeof(int));
    EXPECT_EQ(layout.fields()[1].name, "value");
    EXPECT_EQ(layout.fields()[1].offset_bytes, 8u);
    EXPECT_EQ(layout.fields()[1].size_bytes, sizeof(double));
}

TEST(StateViewTest, ReadsAndWritesTypedFields)
{
    StateLayoutBuilder b("Example");
    b.add<int>("counter");
    b.add<double>("value");
    const auto layout = b.build();

    auto* mem = static_cast<std::byte*>(::operator new(layout.strideBytes(), std::align_val_t(layout.alignment())));
    std::memset(mem, 0, layout.strideBytes());

    StateView view({mem, layout.strideBytes()});
    view.get<int>(layout.fields()[0].offset_bytes) = 7;
    view.get<double>(layout.fields()[1].offset_bytes) = 2.5;

    EXPECT_EQ(view.get<int>(layout.fields()[0].offset_bytes), 7);
    EXPECT_DOUBLE_EQ(view.get<double>(layout.fields()[1].offset_bytes), 2.5);

    ::operator delete(mem, std::align_val_t(layout.alignment()));
}

TEST(StateViewTest, ThrowsOnMisalignedAccess)
{
    std::array<std::byte, 17> bytes{};
    StateView view({bytes.data() + 1, bytes.size() - 1});
    EXPECT_THROW((void)view.get<double>(0), InvalidArgumentException);
}

} // namespace test
} // namespace constitutive
} // namespace FE
} // namespace svmp
