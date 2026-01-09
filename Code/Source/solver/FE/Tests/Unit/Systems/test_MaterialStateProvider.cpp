/**
 * @file test_MaterialStateProvider.cpp
 * @brief Unit tests for Systems MaterialStateProvider
 */

#include <gtest/gtest.h>

#include "Systems/MaterialStateProvider.h"

#include "Assembly/AssemblyKernel.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

using svmp::FE::GlobalIndex;
using svmp::FE::LocalIndex;
using svmp::FE::assembly::IMaterialStateProvider;
using svmp::FE::assembly::MaterialStateSpec;
using svmp::FE::assembly::MaterialStateView;
using svmp::FE::assembly::RequiredData;
using svmp::FE::systems::MaterialStateProvider;

namespace {

class DummyKernel final : public svmp::FE::assembly::LinearFormKernel {
public:
    [[nodiscard]] RequiredData getRequiredData() const override { return RequiredData::None; }
    void computeCell(const svmp::FE::assembly::AssemblyContext&, svmp::FE::assembly::KernelOutput&) override {}
    [[nodiscard]] std::string name() const override { return "DummyKernel"; }
};

[[nodiscard]] std::size_t alignUp(std::size_t value, std::size_t alignment)
{
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

TEST(MaterialStateProvider, MaterialStateProvider_Constructor_InitializesCorrectly)
{
    EXPECT_NO_THROW(MaterialStateProvider(/*num_cells=*/2, /*boundary_face_ids=*/{10, 20}, /*interior_face_ids=*/{7}));
}

TEST(MaterialStateProvider, MaterialStateProvider_AddKernel_AllocatesPerCellStorage)
{
    DummyKernel kernel;
    MaterialStateProvider provider(/*num_cells=*/2);

    MaterialStateSpec spec;
    spec.bytes_per_qpt = 8;
    spec.alignment = 8;

    provider.addKernel(kernel, spec, /*max_cell_qpts=*/3);

    const auto view0 = provider.getCellState(kernel, /*cell_id=*/0, /*num_qpts=*/3);
    const auto view1 = provider.getCellState(kernel, /*cell_id=*/1, /*num_qpts=*/3);
    EXPECT_TRUE(view0);
    EXPECT_TRUE(view1);
}

TEST(MaterialStateProvider, MaterialStateProvider_AddKernel_AllocatesBoundaryFaceStorage)
{
    DummyKernel kernel;
    MaterialStateProvider provider(/*num_cells=*/1, /*boundary_face_ids=*/{10});

    MaterialStateSpec spec;
    spec.bytes_per_qpt = 4;
    spec.alignment = 4;

    provider.addKernel(kernel, spec, /*max_cell_qpts=*/0, /*max_boundary_face_qpts=*/2, /*max_interior_face_qpts=*/0);

    const auto view = provider.getBoundaryFaceState(kernel, /*face_id=*/10, /*num_qpts=*/2);
    ASSERT_TRUE(view);
    EXPECT_EQ(view.bytes_per_qpt, spec.bytes_per_qpt);
    EXPECT_EQ(view.stride_bytes, alignUp(spec.bytes_per_qpt, spec.alignment));
}

TEST(MaterialStateProvider, MaterialStateProvider_AddKernel_AllocatesInteriorFaceStorage)
{
    DummyKernel kernel;
    MaterialStateProvider provider(/*num_cells=*/1, /*boundary_face_ids=*/{}, /*interior_face_ids=*/{7});

    MaterialStateSpec spec;
    spec.bytes_per_qpt = 4;
    spec.alignment = 4;

    provider.addKernel(kernel, spec, /*max_cell_qpts=*/0, /*max_boundary_face_qpts=*/0, /*max_interior_face_qpts=*/2);

    const auto view = provider.getInteriorFaceState(kernel, /*face_id=*/7, /*num_qpts=*/2);
    ASSERT_TRUE(view);
    EXPECT_EQ(view.bytes_per_qpt, spec.bytes_per_qpt);
    EXPECT_EQ(view.stride_bytes, alignUp(spec.bytes_per_qpt, spec.alignment));
}

TEST(MaterialStateProvider, MaterialStateProvider_GetCellState_ReturnsCorrectView)
{
    DummyKernel kernel;
    MaterialStateProvider provider(/*num_cells=*/1);

    MaterialStateSpec spec;
    spec.bytes_per_qpt = 6;
    spec.alignment = 8;

    provider.addKernel(kernel, spec, /*max_cell_qpts=*/2);

    const auto view = provider.getCellState(kernel, /*cell_id=*/0, /*num_qpts=*/2);
    ASSERT_TRUE(view);
    EXPECT_EQ(view.bytes_per_qpt, spec.bytes_per_qpt);
    EXPECT_EQ(view.stride_bytes, alignUp(spec.bytes_per_qpt, spec.alignment));
    EXPECT_NE(view.data_old, nullptr);
    EXPECT_NE(view.data_work, nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(view.data_old) % spec.alignment, 0u);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(view.data_work) % spec.alignment, 0u);
}

TEST(MaterialStateProvider, MaterialStateProvider_ImplementsIMaterialStateProvider)
{
    DummyKernel kernel;
    MaterialStateProvider provider(/*num_cells=*/1);

    MaterialStateSpec spec;
    spec.bytes_per_qpt = 4;
    spec.alignment = 4;

    provider.addKernel(kernel, spec, /*max_cell_qpts=*/1);

    IMaterialStateProvider* iface = &provider;
    const auto view = iface->getCellState(kernel, /*cell_id=*/0, /*num_qpts=*/1);
    EXPECT_TRUE(view);
}

TEST(MaterialStateProvider, MaterialStateProvider_BeginTimeStep_Interface)
{
    DummyKernel kernel;
    MaterialStateProvider provider(/*num_cells=*/1);

    MaterialStateSpec spec;
    spec.bytes_per_qpt = 1;
    spec.alignment = 1;

    provider.addKernel(kernel, spec, /*max_cell_qpts=*/4);

    auto view = provider.getCellState(kernel, /*cell_id=*/0, /*num_qpts=*/4);
    fillBytes(view.data_old, /*n=*/4, std::byte{0x11});
    fillBytes(view.data_work, /*n=*/4, std::byte{0x00});

    IMaterialStateProvider* iface = &provider;
    iface->beginTimeStep();

    auto view2 = provider.getCellState(kernel, /*cell_id=*/0, /*num_qpts=*/4);
    expectBytesEqual(view2.data_work, /*n=*/4, std::byte{0x11});
}

TEST(MaterialStateProvider, MaterialStateProvider_CommitTimeStep_Interface)
{
    DummyKernel kernel;
    MaterialStateProvider provider(/*num_cells=*/1);

    MaterialStateSpec spec;
    spec.bytes_per_qpt = 1;
    spec.alignment = 1;

    provider.addKernel(kernel, spec, /*max_cell_qpts=*/4);

    auto view = provider.getCellState(kernel, /*cell_id=*/0, /*num_qpts=*/4);
    fillBytes(view.data_work, /*n=*/4, std::byte{0xAB});

    IMaterialStateProvider* iface = &provider;
    iface->commitTimeStep();

    auto view2 = provider.getCellState(kernel, /*cell_id=*/0, /*num_qpts=*/4);
    expectBytesEqual(view2.data_old, /*n=*/4, std::byte{0xAB});
}

TEST(MaterialStateProvider, MaterialStateProvider_GetCellState_UnregisteredKernel_Behavior)
{
    DummyKernel kernel;
    MaterialStateProvider provider(/*num_cells=*/1);

    const auto view = provider.getCellState(kernel, /*cell_id=*/0, /*num_qpts=*/1);
    EXPECT_FALSE(static_cast<bool>(view));
}

TEST(MaterialStateProvider, MaterialStateProvider_MultipleKernels_IndependentState)
{
    DummyKernel kernel0;
    DummyKernel kernel1;
    MaterialStateProvider provider(/*num_cells=*/1);

    MaterialStateSpec spec;
    spec.bytes_per_qpt = 1;
    spec.alignment = 1;

    provider.addKernel(kernel0, spec, /*max_cell_qpts=*/4);
    provider.addKernel(kernel1, spec, /*max_cell_qpts=*/4);

    auto v0 = provider.getCellState(kernel0, 0, 4);
    auto v1 = provider.getCellState(kernel1, 0, 4);
    ASSERT_TRUE(v0);
    ASSERT_TRUE(v1);
    EXPECT_NE(v0.data_old, v1.data_old);
    EXPECT_NE(v0.data_work, v1.data_work);

    fillBytes(v0.data_work, /*n=*/4, std::byte{0x5A});
    provider.commitTimeStep();

    auto v0_after = provider.getCellState(kernel0, 0, 4);
    auto v1_after = provider.getCellState(kernel1, 0, 4);
    expectBytesEqual(v0_after.data_old, /*n=*/4, std::byte{0x5A});
    expectBytesEqual(v1_after.data_old, /*n=*/4, std::byte{0x00});
}

TEST(MaterialStateProvider, MaterialStateProvider_LargeStateSize_HandlesCorrectly)
{
    DummyKernel kernel;
    MaterialStateProvider provider(/*num_cells=*/1);

    MaterialStateSpec spec;
    spec.bytes_per_qpt = 1024;
    spec.alignment = 16;

    EXPECT_NO_THROW(provider.addKernel(kernel, spec, /*max_cell_qpts=*/1));

    const auto view = provider.getCellState(kernel, 0, 1);
    ASSERT_TRUE(view);
    EXPECT_EQ(view.bytes_per_qpt, spec.bytes_per_qpt);
    EXPECT_EQ(view.stride_bytes, alignUp(spec.bytes_per_qpt, spec.alignment));
}

TEST(MaterialStateProvider, MaterialStateProvider_MoveSemantics_PreservesState)
{
    DummyKernel kernel;
    MaterialStateProvider provider(/*num_cells=*/1);

    MaterialStateSpec spec;
    spec.bytes_per_qpt = 1;
    spec.alignment = 1;

    provider.addKernel(kernel, spec, /*max_cell_qpts=*/4);
    auto view = provider.getCellState(kernel, 0, 4);
    fillBytes(view.data_work, /*n=*/4, std::byte{0x42});
    provider.commitTimeStep();

    MaterialStateProvider moved = std::move(provider);
    auto view2 = moved.getCellState(kernel, 0, 4);
    ASSERT_TRUE(view2);
    expectBytesEqual(view2.data_old, /*n=*/4, std::byte{0x42});
}

