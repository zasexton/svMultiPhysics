/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_DeviceAssembler.cpp
 * @brief Unit tests for DeviceAssembler and GPU/device assembly strategies
 */

#include <gtest/gtest.h>

#include "Assembly/DeviceAssembler.h"
#include "Assembly/AssemblyContext.h"
#include "Dofs/DofMap.h"
#include "Spaces/H1Space.h"

#include <cmath>
#include <array>
#include <vector>
#include <stdexcept>

namespace svmp {
namespace FE {
namespace assembly {
namespace test {

namespace {

class TrivialMeshAccess final : public IMeshAccess {
public:
    explicit TrivialMeshAccess(GlobalIndex num_cells) : num_cells_(num_cells) {}

    [[nodiscard]] GlobalIndex numCells() const override { return num_cells_; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return num_cells_; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override { return ElementType::Tetra4; }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override {
        nodes.resize(4);
        const GlobalIndex base = cell_id * 4;
        nodes[0] = base + 0;
        nodes[1] = base + 1;
        nodes[2] = base + 2;
        nodes[3] = base + 3;
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex /*node_id*/) const override {
        return {0.0, 0.0, 0.0};
    }

    void getCellCoordinates(GlobalIndex cell_id, std::vector<std::array<Real, 3>>& coords) const override {
        std::vector<GlobalIndex> nodes;
        getCellNodes(cell_id, nodes);
        coords.resize(nodes.size());
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            coords[i] = getNodeCoordinates(nodes[i]);
        }
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex /*face_id*/, GlobalIndex /*cell_id*/) const override {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override { return -1; }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex /*face_id*/) const override {
        return {-1, -1};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override {
        for (GlobalIndex c = 0; c < num_cells_; ++c) {
            callback(c);
        }
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override {
        forEachCell(std::move(callback));
    }

    void forEachBoundaryFace(int /*marker*/, std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

private:
    GlobalIndex num_cells_{0};
};

static dofs::DofMap buildDisjointTetra4DofMap(GlobalIndex num_cells)
{
    dofs::DofMap dof_map(num_cells, /*n_dofs_total=*/num_cells * 4, /*dofs_per_cell=*/4);
    for (GlobalIndex cell = 0; cell < num_cells; ++cell) {
        const std::array<GlobalIndex, 4> dofs = {
            cell * 4 + 0,
            cell * 4 + 1,
            cell * 4 + 2,
            cell * 4 + 3,
        };
        dof_map.setCellDofs(cell, dofs);
    }
    dof_map.setNumDofs(num_cells * 4);
    dof_map.setNumLocalDofs(num_cells * 4);
    dof_map.finalize();
    return dof_map;
}

class ScalingDeviceKernel final : public DeviceKernel {
public:
    explicit ScalingDeviceKernel(Real scale, std::size_t num_dofs)
        : scale_(scale), num_dofs_(num_dofs)
    {
    }

    [[nodiscard]] RequiredData getRequiredData() const noexcept override { return RequiredData::None; }

    void setup(std::size_t num_elements, const void* /*element_data*/, void* qpt_data) override
    {
        ++setup_calls_;
        last_setup_elements_ = num_elements;
        auto* q = static_cast<Real*>(qpt_data);
        if (q != nullptr) {
            q[0] = scale_;
        }
    }

    void apply(std::size_t num_elements, const void* qpt_data, const Real* x, Real* y) override
    {
        ++apply_calls_;
        last_apply_elements_ = num_elements;
        const auto* q = static_cast<const Real*>(qpt_data);
        const Real scale = (q != nullptr) ? q[0] : scale_;
        for (std::size_t i = 0; i < num_dofs_; ++i) {
            y[i] = scale * x[i];
        }
    }

    [[nodiscard]] std::size_t qptDataSize() const noexcept override { return sizeof(Real); }

    [[nodiscard]] int setupCalls() const noexcept { return setup_calls_; }
    [[nodiscard]] int applyCalls() const noexcept { return apply_calls_; }
    [[nodiscard]] std::size_t lastSetupElements() const noexcept { return last_setup_elements_; }
    [[nodiscard]] std::size_t lastApplyElements() const noexcept { return last_apply_elements_; }

private:
    Real scale_{1.0};
    std::size_t num_dofs_{0};
    int setup_calls_{0};
    int apply_calls_{0};
    std::size_t last_setup_elements_{0};
    std::size_t last_apply_elements_{0};
};

} // namespace

// ============================================================================
// DeviceBackend Tests
// ============================================================================

TEST(DeviceBackendTest, EnumValues) {
    EXPECT_NE(DeviceBackend::CPU, DeviceBackend::CUDA);
    EXPECT_NE(DeviceBackend::CUDA, DeviceBackend::HIP);
    EXPECT_NE(DeviceBackend::HIP, DeviceBackend::SYCL);
    EXPECT_NE(DeviceBackend::SYCL, DeviceBackend::Auto);
}

// ============================================================================
// DeviceAssemblyLevel Tests
// ============================================================================

TEST(DeviceAssemblyLevelTest, EnumValues) {
    EXPECT_NE(DeviceAssemblyLevel::Full, DeviceAssemblyLevel::Element);
    EXPECT_NE(DeviceAssemblyLevel::Element, DeviceAssemblyLevel::Partial);
    EXPECT_NE(DeviceAssemblyLevel::Partial, DeviceAssemblyLevel::None);
}

// ============================================================================
// DeviceMemoryMode Tests
// ============================================================================

TEST(DeviceMemoryModeTest, EnumValues) {
    EXPECT_NE(DeviceMemoryMode::Explicit, DeviceMemoryMode::Unified);
    EXPECT_NE(DeviceMemoryMode::Unified, DeviceMemoryMode::Pinned);
}

// ============================================================================
// DeviceOptions Tests
// ============================================================================

TEST(DeviceOptionsTest, Defaults) {
    DeviceOptions options;

    EXPECT_EQ(options.backend, DeviceBackend::Auto);
    EXPECT_EQ(options.assembly_level, DeviceAssemblyLevel::Partial);
    EXPECT_EQ(options.memory_mode, DeviceMemoryMode::Explicit);
    EXPECT_EQ(options.device_id, 0);
    EXPECT_EQ(options.num_streams, 2);
    EXPECT_EQ(options.batch_size, 1024u);
    EXPECT_TRUE(options.use_memory_pool);
    EXPECT_EQ(options.pool_initial_size, 64u * 1024u * 1024u);  // 64 MB
    EXPECT_TRUE(options.async_transfers);
    EXPECT_TRUE(options.prefetch);
    EXPECT_FALSE(options.verbose);
}

TEST(DeviceOptionsTest, CustomValues) {
    DeviceOptions options;
    options.backend = DeviceBackend::CUDA;
    options.assembly_level = DeviceAssemblyLevel::Full;
    options.device_id = 1;
    options.num_streams = 4;
    options.batch_size = 2048;

    EXPECT_EQ(options.backend, DeviceBackend::CUDA);
    EXPECT_EQ(options.assembly_level, DeviceAssemblyLevel::Full);
    EXPECT_EQ(options.device_id, 1);
    EXPECT_EQ(options.num_streams, 4);
    EXPECT_EQ(options.batch_size, 2048u);
}

// ============================================================================
// DeviceInfo Tests
// ============================================================================

TEST(DeviceInfoTest, DefaultValues) {
    DeviceInfo info;

    EXPECT_TRUE(info.name.empty());
    EXPECT_TRUE(info.driver_version.empty());
    EXPECT_EQ(info.total_memory, 0u);
    EXPECT_EQ(info.free_memory, 0u);
    EXPECT_EQ(info.compute_capability_major, 0);
    EXPECT_EQ(info.compute_capability_minor, 0);
    EXPECT_EQ(info.num_multiprocessors, 0);
    EXPECT_EQ(info.max_threads_per_block, 0);
    EXPECT_EQ(info.warp_size, 0);
    EXPECT_FALSE(info.supports_unified_memory);
    EXPECT_FALSE(info.supports_cooperative_groups);
}

TEST(DeviceInfoTest, PopulatedValues) {
    DeviceInfo info;
    info.name = "Test GPU";
    info.driver_version = "12.0";
    info.total_memory = 8u * 1024u * 1024u * 1024u;  // 8 GB
    info.compute_capability_major = 8;
    info.compute_capability_minor = 6;
    info.num_multiprocessors = 82;
    info.max_threads_per_block = 1024;
    info.warp_size = 32;
    info.supports_unified_memory = true;

    EXPECT_EQ(info.name, "Test GPU");
    EXPECT_EQ(info.compute_capability_major, 8);
    EXPECT_TRUE(info.supports_unified_memory);
}

// ============================================================================
// DeviceAssemblyStats Tests
// ============================================================================

TEST(DeviceAssemblyStatsTest, DefaultValues) {
    DeviceAssemblyStats stats;

    EXPECT_EQ(stats.elements_processed, 0);
    EXPECT_DOUBLE_EQ(stats.total_seconds, 0.0);
    EXPECT_DOUBLE_EQ(stats.setup_seconds, 0.0);
    EXPECT_DOUBLE_EQ(stats.transfer_seconds, 0.0);
    EXPECT_DOUBLE_EQ(stats.kernel_seconds, 0.0);
    EXPECT_EQ(stats.host_to_device_bytes, 0u);
    EXPECT_EQ(stats.device_to_host_bytes, 0u);
    EXPECT_EQ(stats.device_memory_used, 0u);
}

TEST(DeviceAssemblyStatsTest, PopulatedValues) {
    DeviceAssemblyStats stats;
    stats.elements_processed = 10000;
    stats.total_seconds = 0.5;
    stats.setup_seconds = 0.1;
    stats.transfer_seconds = 0.05;
    stats.kernel_seconds = 0.35;
    stats.host_to_device_bytes = 1024 * 1024;
    stats.device_to_host_bytes = 512 * 1024;
    stats.device_memory_used = 100 * 1024 * 1024;

    EXPECT_EQ(stats.elements_processed, 10000);
    EXPECT_DOUBLE_EQ(stats.total_seconds, 0.5);
    EXPECT_EQ(stats.host_to_device_bytes, 1024u * 1024u);
}

// ============================================================================
// DeviceMemory Tests (CPU fallback)
// ============================================================================

TEST(DeviceMemoryTest, DefaultConstruction) {
    DeviceMemory<Real> mem;

    EXPECT_FALSE(mem.isAllocated());
    EXPECT_EQ(mem.data(), nullptr);
    EXPECT_EQ(mem.size(), 0u);
}

TEST(DeviceMemoryTest, Allocate) {
    DeviceMemory<Real> mem;

    mem.allocate(100);

    EXPECT_TRUE(mem.isAllocated());
    EXPECT_NE(mem.data(), nullptr);
    EXPECT_EQ(mem.size(), 100u);
}

TEST(DeviceMemoryTest, Free) {
    DeviceMemory<Real> mem;
    mem.allocate(100);

    mem.free();

    EXPECT_FALSE(mem.isAllocated());
    EXPECT_EQ(mem.data(), nullptr);
}

TEST(DeviceMemoryTest, CopyFromHost) {
    DeviceMemory<Real> mem;
    mem.allocate(4);

    std::vector<Real> host_data = {1.0, 2.0, 3.0, 4.0};
    mem.copyFromHost(host_data.data(), 4);

    // For CPU fallback, data should be directly accessible
    if (mem.data()) {
        EXPECT_DOUBLE_EQ(mem.data()[0], 1.0);
        EXPECT_DOUBLE_EQ(mem.data()[3], 4.0);
    }
}

TEST(DeviceMemoryTest, CopyToHost) {
    DeviceMemory<Real> mem;
    mem.allocate(4);

    std::vector<Real> host_in = {5.0, 6.0, 7.0, 8.0};
    mem.copyFromHost(host_in.data(), 4);

    std::vector<Real> host_out(4, 0.0);
    mem.copyToHost(host_out.data(), 4);

    EXPECT_DOUBLE_EQ(host_out[0], 5.0);
    EXPECT_DOUBLE_EQ(host_out[1], 6.0);
    EXPECT_DOUBLE_EQ(host_out[2], 7.0);
    EXPECT_DOUBLE_EQ(host_out[3], 8.0);
}

TEST(DeviceMemoryTest, MoveConstruction) {
    DeviceMemory<Real> mem1;
    mem1.allocate(50);
    Real* ptr = mem1.data();

    DeviceMemory<Real> mem2(std::move(mem1));

    EXPECT_TRUE(mem2.isAllocated());
    EXPECT_EQ(mem2.data(), ptr);
    EXPECT_EQ(mem2.size(), 50u);
    EXPECT_FALSE(mem1.isAllocated());
}

TEST(DeviceMemoryTest, MoveAssignment) {
    DeviceMemory<Real> mem1;
    mem1.allocate(30);

    DeviceMemory<Real> mem2;
    mem2 = std::move(mem1);

    EXPECT_TRUE(mem2.isAllocated());
    EXPECT_EQ(mem2.size(), 30u);
    EXPECT_FALSE(mem1.isAllocated());
}

// ============================================================================
// PartialAssemblyData Tests
// ============================================================================

TEST(PartialAssemblyDataTest, DefaultConstruction) {
    PartialAssemblyData data;

    EXPECT_EQ(data.numElements(), 0u);
    EXPECT_EQ(data.numQpts(), 0u);
    EXPECT_EQ(data.totalBytes(), 0u);
    EXPECT_FALSE(data.isPopulated());
}

TEST(PartialAssemblyDataTest, Allocate) {
    PartialAssemblyData data;

    data.allocate(100, 8, 64);  // 100 elements, 8 qpts, 64 bytes per qpt

    EXPECT_EQ(data.numElements(), 100u);
    EXPECT_EQ(data.numQpts(), 8u);
    EXPECT_EQ(data.totalBytes(), 100u * 8u * 64u);
}

TEST(PartialAssemblyDataTest, Free) {
    PartialAssemblyData data;
    data.allocate(100, 8, 64);

    data.free();

    EXPECT_EQ(data.numElements(), 0u);
    EXPECT_EQ(data.totalBytes(), 0u);
}

TEST(PartialAssemblyDataTest, PopulatedFlag) {
    PartialAssemblyData data;
    data.allocate(10, 4, 32);

    EXPECT_FALSE(data.isPopulated());

    data.setPopulated(true);
    EXPECT_TRUE(data.isPopulated());

    data.setPopulated(false);
    EXPECT_FALSE(data.isPopulated());
}

TEST(PartialAssemblyDataTest, DeviceDataAccess) {
    PartialAssemblyData data;
    data.allocate(10, 4, 32);

    void* ptr = data.deviceData();
    // For CPU fallback, should get valid pointer
    (void)ptr;  // May be nullptr if not allocated

    const PartialAssemblyData& const_data = data;
    [[maybe_unused]] const void* const_ptr = const_data.deviceData();
}

TEST(PartialAssemblyDataTest, MoveConstruction) {
    PartialAssemblyData data1;
    data1.allocate(50, 8, 64);
    data1.setPopulated(true);

    PartialAssemblyData data2(std::move(data1));

    EXPECT_EQ(data2.numElements(), 50u);
    EXPECT_TRUE(data2.isPopulated());
}

TEST(PartialAssemblyDataTest, MoveAssignment) {
    PartialAssemblyData data1;
    data1.allocate(25, 4, 32);

    PartialAssemblyData data2;
    data2 = std::move(data1);

    EXPECT_EQ(data2.numElements(), 25u);
}

// ============================================================================
// DeviceAssembler Tests
// ============================================================================

TEST(DeviceAssemblerTest, DefaultConstruction) {
    DeviceAssembler assembler;

    EXPECT_FALSE(assembler.isConfigured());
    EXPECT_FALSE(assembler.isSetup());
}

TEST(DeviceAssemblerTest, ConstructionWithOptions) {
    DeviceOptions options;
    options.backend = DeviceBackend::CPU;
    options.assembly_level = DeviceAssemblyLevel::Element;

    DeviceAssembler assembler(options);

    EXPECT_EQ(assembler.getDeviceOptions().backend, DeviceBackend::CPU);
    EXPECT_EQ(assembler.getDeviceOptions().assembly_level, DeviceAssemblyLevel::Element);
}

TEST(DeviceAssemblerTest, SetDeviceOptions) {
    DeviceAssembler assembler;

    DeviceOptions options;
    options.batch_size = 512;
    options.num_streams = 8;

    assembler.setDeviceOptions(options);

    EXPECT_EQ(assembler.getDeviceOptions().batch_size, 512u);
    EXPECT_EQ(assembler.getDeviceOptions().num_streams, 8);
}

TEST(DeviceAssemblerTest, SetAssemblyOptions) {
    DeviceAssembler assembler;

    AssemblyOptions options;
    options.verbose = true;

    assembler.setOptions(options);

    EXPECT_TRUE(assembler.getOptions().verbose);
}

TEST(DeviceAssemblerTest, GetDeviceInfo) {
    DeviceAssembler assembler;

    DeviceInfo info = assembler.getDeviceInfo();

    // For CPU fallback, info may be minimal
    // Just verify it returns without error
    SUCCEED();
}

TEST(DeviceAssemblerTest, GetLastStats) {
    DeviceAssembler assembler;

    const DeviceAssemblyStats& stats = assembler.getLastStats();

    // Initial stats should be zeroed
    EXPECT_EQ(stats.elements_processed, 0);
}

TEST(DeviceAssemblerTest, Initialize) {
    DeviceAssembler assembler;

    EXPECT_THROW(assembler.initialize(), std::runtime_error);
}

TEST(DeviceAssemblerTest, Reset) {
    DeviceAssembler assembler;
    EXPECT_NO_THROW(assembler.reset());
    EXPECT_FALSE(assembler.isSetup());
}

TEST(DeviceAssemblerTest, AllocateVector) {
    DeviceAssembler assembler;

    auto vec = assembler.allocateVector(100);

    EXPECT_TRUE(vec.isAllocated());
    EXPECT_EQ(vec.size(), 100u);
}

TEST(DeviceAssemblerTest, CopyToDevice) {
    DeviceAssembler assembler;

    std::vector<Real> host = {1.0, 2.0, 3.0, 4.0};
    DeviceMemory<Real> device;
    device.allocate(4);

    assembler.copyToDevice(host, device);

    // Should not throw
    SUCCEED();
}

TEST(DeviceAssemblerTest, CopyFromDevice) {
    DeviceAssembler assembler;

    DeviceMemory<Real> device;
    device.allocate(4);
    std::vector<Real> host_in = {1.0, 2.0, 3.0, 4.0};
    device.copyFromHost(host_in.data(), 4);

    std::vector<Real> host_out(4, 0.0);
    assembler.copyFromDevice(device, host_out);

    EXPECT_DOUBLE_EQ(host_out[0], 1.0);
    EXPECT_DOUBLE_EQ(host_out[3], 4.0);
}

TEST(DeviceAssemblerTest, Synchronize) {
    DeviceAssembler assembler;
    assembler.synchronize();

    // Should not throw
    SUCCEED();
}

TEST(DeviceAssemblerTest, MoveConstruction) {
    DeviceOptions options;
    options.batch_size = 256;

    DeviceAssembler assembler1(options);
    DeviceAssembler assembler2(std::move(assembler1));

    EXPECT_EQ(assembler2.getDeviceOptions().batch_size, 256u);
}

TEST(DeviceAssemblerTest, MoveAssignment) {
    DeviceOptions options;
    options.assembly_level = DeviceAssemblyLevel::None;

    DeviceAssembler assembler1(options);
    DeviceAssembler assembler2;

    assembler2 = std::move(assembler1);

    EXPECT_EQ(assembler2.getDeviceOptions().assembly_level, DeviceAssemblyLevel::None);
}

// ============================================================================
// Static Query Tests
// ============================================================================

TEST(DeviceAssemblerTest, IsGPUAvailable) {
    bool available = DeviceAssembler::isGPUAvailable();

    // Just verify it returns without error
    (void)available;
    SUCCEED();
}

TEST(DeviceAssemblerTest, GetGPUCount) {
    int count = DeviceAssembler::getGPUCount();

    // Count should be non-negative (0 if no GPU)
    EXPECT_GE(count, 0);
}

TEST(DeviceAssemblerTest, GetBackendInfo) {
    std::string info = DeviceAssembler::getBackendInfo();

    // Should return some string
    EXPECT_FALSE(info.empty());
}

// ============================================================================
// Factory Tests
// ============================================================================

TEST(DeviceAssemblerFactoryTest, CreateDefault) {
    auto assembler = createDeviceAssembler();

    EXPECT_NE(assembler, nullptr);
}

TEST(DeviceAssemblerFactoryTest, CreateWithOptions) {
    DeviceOptions options;
    options.backend = DeviceBackend::CPU;

    auto assembler = createDeviceAssembler(options);

    EXPECT_NE(assembler, nullptr);

    auto* device = dynamic_cast<DeviceAssembler*>(assembler.get());
    if (device) {
        EXPECT_EQ(device->getDeviceOptions().backend, DeviceBackend::CPU);
    }
}

TEST(DeviceAssemblerFactoryTest, CreateAuto) {
    auto assembler = createDeviceAssemblerAuto();

    EXPECT_NE(assembler, nullptr);
}

// ============================================================================
// Integration Tests (CPU Fallback)
// ============================================================================

TEST(DeviceAssemblerIntegrationTest, FullWorkflow) {
    // Create assembler with CPU backend
    DeviceOptions options;
    options.backend = DeviceBackend::CPU;
    options.assembly_level = DeviceAssemblyLevel::Partial;

    DeviceAssembler assembler(options);

    // Allocate vectors
    auto x = assembler.allocateVector(100);
    auto y = assembler.allocateVector(100);

    // Copy data to "device"
    std::vector<Real> host_x(100, 1.0);
    assembler.copyToDevice(host_x, x);

    // Synchronize
    assembler.synchronize();

    // Copy result back
    std::vector<Real> host_y(100, 0.0);
    assembler.copyFromDevice(x, host_y);

    // Verify
    for (const auto& val : host_y) {
        EXPECT_DOUBLE_EQ(val, 1.0);
    }

    // Check stats
    const auto& stats = assembler.getLastStats();
    (void)stats;  // Stats may or may not be populated

    // Reset
    assembler.reset();
    EXPECT_FALSE(assembler.isSetup());
}

TEST(DeviceAssemblerIntegrationTest, MultipleAllocations) {
    DeviceAssembler assembler;

    std::vector<DeviceMemory<Real>> vectors;

    // Allocate multiple vectors
    for (int i = 0; i < 10; ++i) {
        vectors.push_back(assembler.allocateVector(100));
    }

    // All should be allocated
    for (const auto& vec : vectors) {
        EXPECT_TRUE(vec.isAllocated());
        EXPECT_EQ(vec.size(), 100u);
    }
}

// ============================================================================
// Partial Assembly Interface Tests (CPU Fallback)
// ============================================================================

TEST(DeviceAssemblerPartialAssemblyTest, ApplyThrowsWithoutSetup) {
    DeviceAssembler assembler;
    std::vector<Real> x(1, 1.0);
    std::vector<Real> y(1, 0.0);
    EXPECT_THROW(assembler.apply(x, y), std::runtime_error);
}

TEST(DeviceAssemblerPartialAssemblyTest, SetupAndApplyCallsDeviceKernel) {
    const GlobalIndex n_cells = 2;
    TrivialMeshAccess mesh(n_cells);
    auto dof_map = buildDisjointTetra4DofMap(n_cells);
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    DeviceOptions options;
    options.backend = DeviceBackend::CPU;
    options.assembly_level = DeviceAssemblyLevel::Partial;

    DeviceAssembler assembler(options);
    assembler.setDofMap(dof_map);

    const std::size_t num_dofs = static_cast<std::size_t>(dof_map.getNumLocalDofs());
    ScalingDeviceKernel kernel(/*scale=*/2.0, num_dofs);

    assembler.setup(mesh, space, kernel);
    EXPECT_TRUE(assembler.isSetup());
    EXPECT_EQ(kernel.setupCalls(), 1);
    EXPECT_EQ(kernel.lastSetupElements(), static_cast<std::size_t>(n_cells));

    std::vector<Real> x(num_dofs, 0.0);
    for (std::size_t i = 0; i < num_dofs; ++i) {
        x[i] = static_cast<Real>(i + 1);
    }
    std::vector<Real> y(num_dofs, 0.0);

    assembler.apply(x, y);

    EXPECT_EQ(kernel.applyCalls(), 1);
    EXPECT_EQ(kernel.lastApplyElements(), static_cast<std::size_t>(n_cells));
    for (std::size_t i = 0; i < num_dofs; ++i) {
        EXPECT_DOUBLE_EQ(y[i], 2.0 * x[i]);
    }

    const auto& stats = assembler.getLastStats();
    EXPECT_EQ(stats.host_to_device_bytes, num_dofs * sizeof(Real));
    EXPECT_EQ(stats.device_to_host_bytes, num_dofs * sizeof(Real));
}

TEST(DeviceAssemblerPartialAssemblyTest, ApplySizeMismatchThrows) {
    const GlobalIndex n_cells = 1;
    TrivialMeshAccess mesh(n_cells);
    auto dof_map = buildDisjointTetra4DofMap(n_cells);
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    DeviceOptions options;
    options.backend = DeviceBackend::CPU;
    options.assembly_level = DeviceAssemblyLevel::Partial;

    DeviceAssembler assembler(options);
    assembler.setDofMap(dof_map);

    const std::size_t num_dofs = static_cast<std::size_t>(dof_map.getNumLocalDofs());
    ScalingDeviceKernel kernel(/*scale=*/1.0, num_dofs);
    assembler.setup(mesh, space, kernel);

    std::vector<Real> x(num_dofs + 1, 1.0);
    std::vector<Real> y(num_dofs, 0.0);
    EXPECT_THROW(assembler.apply(x, y), std::invalid_argument);
}

// ============================================================================
// Conditional GPU Backend Tests (skipped unless supported + available)
// ============================================================================

TEST(DeviceAssemblerGPUTest, CUDABackendKernelLaunchIfAvailable) {
#if defined(SVMP_HAS_CUDA)
    if (!DeviceAssembler::isGPUAvailable()) {
        GTEST_SKIP() << "No GPU available at runtime";
    }
    GTEST_SKIP() << "CUDA backend kernel launch not implemented in this build";
#else
    GTEST_SKIP() << "CUDA backend not compiled";
#endif
}

TEST(DeviceAssemblerGPUTest, HIPBackendKernelLaunchIfAvailable) {
#if defined(SVMP_HAS_HIP)
    if (!DeviceAssembler::isGPUAvailable()) {
        GTEST_SKIP() << "No GPU available at runtime";
    }
    GTEST_SKIP() << "HIP backend kernel launch not implemented in this build";
#else
    GTEST_SKIP() << "HIP backend not compiled";
#endif
}

TEST(DeviceAssemblerGPUTest, SYCLBackendKernelLaunchIfAvailable) {
#if defined(SVMP_HAS_SYCL)
    if (!DeviceAssembler::isGPUAvailable()) {
        GTEST_SKIP() << "No GPU available at runtime";
    }
    GTEST_SKIP() << "SYCL backend kernel launch not implemented in this build";
#else
    GTEST_SKIP() << "SYCL backend not compiled";
#endif
}

} // namespace test
} // namespace assembly
} // namespace FE
} // namespace svmp
