/**
 * @file test_GlobalKernelStateProvider.cpp
 * @brief Unit tests for Systems GlobalKernelStateProvider
 */

#include <gtest/gtest.h>

#include "Systems/GlobalKernelStateProvider.h"

#include "Assembly/Assembler.h"
#include "Systems/GlobalKernel.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

using svmp::FE::GlobalIndex;
using svmp::FE::LocalIndex;
using svmp::FE::Real;
using svmp::FE::assembly::MaterialStateView;
using svmp::FE::systems::AssemblyRequest;
using svmp::FE::systems::FESystem;
using svmp::FE::systems::GlobalKernel;
using svmp::FE::systems::GlobalKernelStateProvider;
using svmp::FE::systems::GlobalStateSpec;
using svmp::FE::systems::SystemStateView;

namespace {

class DummyGlobalKernel final : public GlobalKernel {
public:
    [[nodiscard]] std::string name() const override { return "DummyGlobalKernel"; }

    svmp::FE::assembly::AssemblyResult assemble(const FESystem&,
                                                const AssemblyRequest&,
                                                const SystemStateView&,
                                                svmp::FE::assembly::GlobalSystemView*,
                                                svmp::FE::assembly::GlobalSystemView*) override
    {
        return {};
    }
};

[[nodiscard]] std::size_t alignUp(std::size_t value, std::size_t alignment)
{
    if (alignment == 0u) return value;
    const std::size_t mask = alignment - 1u;
    return (value + mask) & ~mask;
}

void fillBytes(std::byte* data, std::size_t n, std::byte value)
{
    ASSERT_NE(data, nullptr);
    std::memset(data, static_cast<int>(value), n);
}

void expectBytesEqual(const std::byte* data, std::size_t n, std::byte value)
{
    ASSERT_NE(data, nullptr);
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(data[i], value);
    }
}

} // namespace

TEST(GlobalKernelStateProvider, GlobalKernelStateProvider_Constructor_InitializesWithCellCount)
{
    EXPECT_NO_THROW(GlobalKernelStateProvider(/*num_cells=*/3));
}

TEST(GlobalKernelStateProvider, GlobalKernelStateProvider_Constructor_AcceptsBoundaryAndInteriorFaceIds)
{
    EXPECT_NO_THROW(GlobalKernelStateProvider(/*num_cells=*/2,
                                              /*boundary_face_ids=*/{10, 10, 20},
                                              /*interior_face_ids=*/{7, 8, 8}));
}

TEST(GlobalKernelStateProvider, GlobalKernelStateProvider_AddKernel_AllocatesStateBuffers)
{
    DummyGlobalKernel kernel;
    GlobalKernelStateProvider provider(/*num_cells=*/2);

    GlobalStateSpec spec;
    spec.domain = GlobalStateSpec::Domain::Cell;
    spec.bytes_per_qpt = 6;
    spec.alignment = 8;
    spec.max_qpts = 3;

    provider.addKernel(kernel, spec);

    const auto view = provider.getCellState(kernel, /*cell_id=*/0, /*num_qpts=*/3);
    ASSERT_TRUE(static_cast<bool>(view));
    EXPECT_EQ(view.bytes_per_qpt, spec.bytes_per_qpt);
    EXPECT_EQ(view.stride_bytes, alignUp(spec.bytes_per_qpt, spec.alignment));
    EXPECT_NE(view.data_old, nullptr);
    EXPECT_NE(view.data_work, nullptr);
    EXPECT_NE(view.data_old, view.data_work);
}

TEST(GlobalKernelStateProvider, GlobalKernelStateProvider_GetCellState_ReturnsValidView)
{
    DummyGlobalKernel kernel;
    GlobalKernelStateProvider provider(/*num_cells=*/1);

    GlobalStateSpec spec;
    spec.domain = GlobalStateSpec::Domain::Cell;
    spec.bytes_per_qpt = sizeof(std::uint32_t);
    spec.alignment = 16;
    spec.max_qpts = 2;

    provider.addKernel(kernel, spec);

    const auto view = provider.getCellState(kernel, /*cell_id=*/0, /*num_qpts=*/2);
    ASSERT_TRUE(view);

    EXPECT_EQ(view.bytes_per_qpt, spec.bytes_per_qpt);
    EXPECT_EQ(view.stride_bytes, alignUp(spec.bytes_per_qpt, spec.alignment));

    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(view.data_old) % spec.alignment, 0u);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(view.data_work) % spec.alignment, 0u);
}

TEST(GlobalKernelStateProvider, GlobalKernelStateProvider_GetBoundaryFaceState_ReturnsValidView)
{
    DummyGlobalKernel kernel;
    GlobalKernelStateProvider provider(/*num_cells=*/0, /*boundary_face_ids=*/{10, 10, 20});

    GlobalStateSpec spec;
    spec.domain = GlobalStateSpec::Domain::BoundaryFace;
    spec.bytes_per_qpt = 5;
    spec.alignment = 8;
    spec.max_qpts = 4;

    provider.addKernel(kernel, spec);

    const auto view10 = provider.getBoundaryFaceState(kernel, /*face_id=*/10, /*num_qpts=*/4);
    const auto view20 = provider.getBoundaryFaceState(kernel, /*face_id=*/20, /*num_qpts=*/4);
    ASSERT_TRUE(view10);
    ASSERT_TRUE(view20);
    EXPECT_EQ(view10.bytes_per_qpt, spec.bytes_per_qpt);
    EXPECT_EQ(view10.stride_bytes, alignUp(spec.bytes_per_qpt, spec.alignment));
}

TEST(GlobalKernelStateProvider, GlobalKernelStateProvider_GetInteriorFaceState_ReturnsValidView)
{
    DummyGlobalKernel kernel;
    GlobalKernelStateProvider provider(/*num_cells=*/0, /*boundary_face_ids=*/{}, /*interior_face_ids=*/{7, 8});

    GlobalStateSpec spec;
    spec.domain = GlobalStateSpec::Domain::InteriorFace;
    spec.bytes_per_qpt = 8;
    spec.alignment = 8;
    spec.max_qpts = 3;

    provider.addKernel(kernel, spec);

    const auto view7 = provider.getInteriorFaceState(kernel, /*face_id=*/7, /*num_qpts=*/3);
    ASSERT_TRUE(view7);
    EXPECT_EQ(view7.bytes_per_qpt, spec.bytes_per_qpt);
    EXPECT_EQ(view7.stride_bytes, alignUp(spec.bytes_per_qpt, spec.alignment));
}

TEST(GlobalKernelStateProvider, GlobalKernelStateProvider_BeginTimeStep_SwapsBuffers)
{
    DummyGlobalKernel kernel;
    GlobalKernelStateProvider provider(/*num_cells=*/1);

    GlobalStateSpec spec;
    spec.domain = GlobalStateSpec::Domain::Cell;
    spec.bytes_per_qpt = 4;
    spec.alignment = 4;
    spec.max_qpts = 2;
    provider.addKernel(kernel, spec);

    auto view = provider.getCellState(kernel, /*cell_id=*/0, /*num_qpts=*/2);
    ASSERT_TRUE(view);

    // Put recognizable bytes in old, clear work.
    fillBytes(view.data_old, spec.bytes_per_qpt * static_cast<std::size_t>(spec.max_qpts), std::byte{0x11});
    fillBytes(view.data_work, spec.bytes_per_qpt * static_cast<std::size_t>(spec.max_qpts), std::byte{0x00});

    provider.beginTimeStep();

    auto view2 = provider.getCellState(kernel, /*cell_id=*/0, /*num_qpts=*/2);
    expectBytesEqual(view2.data_work, spec.bytes_per_qpt * static_cast<std::size_t>(spec.max_qpts), std::byte{0x11});
}

TEST(GlobalKernelStateProvider, GlobalKernelStateProvider_CommitTimeStep_PreservesWorkData)
{
    DummyGlobalKernel kernel;
    GlobalKernelStateProvider provider(/*num_cells=*/1);

    GlobalStateSpec spec;
    spec.domain = GlobalStateSpec::Domain::Cell;
    spec.bytes_per_qpt = 4;
    spec.alignment = 4;
    spec.max_qpts = 2;
    provider.addKernel(kernel, spec);

    auto view = provider.getCellState(kernel, /*cell_id=*/0, /*num_qpts=*/2);
    ASSERT_TRUE(view);

    fillBytes(view.data_work, spec.bytes_per_qpt * static_cast<std::size_t>(spec.max_qpts), std::byte{0xAB});
    provider.commitTimeStep();

    auto view2 = provider.getCellState(kernel, /*cell_id=*/0, /*num_qpts=*/2);
    expectBytesEqual(view2.data_old, spec.bytes_per_qpt * static_cast<std::size_t>(spec.max_qpts), std::byte{0xAB});
}

TEST(GlobalKernelStateProvider, GlobalKernelStateProvider_MultipleTimeSteps_MaintainsCorrectHistory)
{
    DummyGlobalKernel kernel;
    GlobalKernelStateProvider provider(/*num_cells=*/1);

    GlobalStateSpec spec;
    spec.domain = GlobalStateSpec::Domain::Cell;
    spec.bytes_per_qpt = 1;
    spec.alignment = 1;
    spec.max_qpts = 4;
    provider.addKernel(kernel, spec);

    // Step 1: write 0x01 to work and commit -> old == 0x01.
    auto view = provider.getCellState(kernel, 0, spec.max_qpts);
    fillBytes(view.data_work, spec.max_qpts, std::byte{0x01});
    provider.commitTimeStep();

    // Step 2: begin copies old to work.
    provider.beginTimeStep();
    auto view2 = provider.getCellState(kernel, 0, spec.max_qpts);
    expectBytesEqual(view2.data_work, spec.max_qpts, std::byte{0x01});

    // Step 2: update work to 0x02 and commit -> old == 0x02.
    fillBytes(view2.data_work, spec.max_qpts, std::byte{0x02});
    provider.commitTimeStep();
    auto view3 = provider.getCellState(kernel, 0, spec.max_qpts);
    expectBytesEqual(view3.data_old, spec.max_qpts, std::byte{0x02});
}

TEST(GlobalKernelStateProvider, GlobalKernelStateProvider_GetCellState_InvalidCellId_Behavior)
{
    DummyGlobalKernel kernel;
    GlobalKernelStateProvider provider(/*num_cells=*/1);

    GlobalStateSpec spec;
    spec.domain = GlobalStateSpec::Domain::Cell;
    spec.bytes_per_qpt = 4;
    spec.alignment = 4;
    spec.max_qpts = 1;
    provider.addKernel(kernel, spec);

    EXPECT_THROW((void)provider.getCellState(kernel, /*cell_id=*/-1, /*num_qpts=*/1), svmp::FE::InvalidArgumentException);
    EXPECT_THROW((void)provider.getCellState(kernel, /*cell_id=*/1, /*num_qpts=*/1), svmp::FE::InvalidArgumentException);
}

TEST(GlobalKernelStateProvider, GlobalKernelStateProvider_GetBoundaryFaceState_UnregisteredFace_Behavior)
{
    DummyGlobalKernel kernel;
    GlobalKernelStateProvider provider(/*num_cells=*/0, /*boundary_face_ids=*/{10});

    GlobalStateSpec spec;
    spec.domain = GlobalStateSpec::Domain::BoundaryFace;
    spec.bytes_per_qpt = 4;
    spec.alignment = 4;
    spec.max_qpts = 1;
    provider.addKernel(kernel, spec);

    const auto view = provider.getBoundaryFaceState(kernel, /*face_id=*/99, /*num_qpts=*/1);
    EXPECT_FALSE(static_cast<bool>(view));
    EXPECT_EQ(view.data_old, nullptr);
    EXPECT_EQ(view.data_work, nullptr);
}

TEST(GlobalKernelStateProvider, GlobalKernelStateProvider_AddKernel_ZeroBytesPerQpt_Behavior)
{
    DummyGlobalKernel kernel;
    GlobalKernelStateProvider provider(/*num_cells=*/1);

    GlobalStateSpec spec;
    spec.domain = GlobalStateSpec::Domain::Cell;
    spec.bytes_per_qpt = 0;
    spec.alignment = 8;
    spec.max_qpts = 1;
    EXPECT_THROW(provider.addKernel(kernel, spec), svmp::FE::InvalidArgumentException);
}

TEST(GlobalKernelStateProvider, GlobalKernelStateProvider_AddKernel_NonPowerOfTwoAlignment)
{
    DummyGlobalKernel kernel;
    GlobalKernelStateProvider provider(/*num_cells=*/1);

    GlobalStateSpec spec;
    spec.domain = GlobalStateSpec::Domain::Cell;
    spec.bytes_per_qpt = 4;
    spec.alignment = 24; // not a power of two
    spec.max_qpts = 1;
    EXPECT_THROW(provider.addKernel(kernel, spec), svmp::FE::InvalidArgumentException);
}

TEST(GlobalKernelStateProvider, GlobalKernelStateProvider_GetState_RequestedQptsExceedsMax_Behavior)
{
    DummyGlobalKernel kernel;
    GlobalKernelStateProvider provider(/*num_cells=*/1, /*boundary_face_ids=*/{10});

    GlobalStateSpec spec;
    spec.domain = GlobalStateSpec::Domain::Cell;
    spec.bytes_per_qpt = 4;
    spec.alignment = 4;
    spec.max_qpts = 2;
    provider.addKernel(kernel, spec);

    EXPECT_THROW((void)provider.getCellState(kernel, /*cell_id=*/0, /*num_qpts=*/3), svmp::FE::InvalidArgumentException);
}

TEST(GlobalKernelStateProvider, GlobalKernelStateProvider_MoveSemantics_PreservesState)
{
    DummyGlobalKernel kernel;
    GlobalKernelStateProvider provider(/*num_cells=*/1);

    GlobalStateSpec spec;
    spec.domain = GlobalStateSpec::Domain::Cell;
    spec.bytes_per_qpt = 1;
    spec.alignment = 1;
    spec.max_qpts = 4;
    provider.addKernel(kernel, spec);

    auto view = provider.getCellState(kernel, 0, spec.max_qpts);
    fillBytes(view.data_work, spec.max_qpts, std::byte{0x5A});
    provider.commitTimeStep();

    GlobalKernelStateProvider moved = std::move(provider);
    auto view2 = moved.getCellState(kernel, 0, spec.max_qpts);
    ASSERT_TRUE(view2);
    expectBytesEqual(view2.data_old, spec.max_qpts, std::byte{0x5A});
}

