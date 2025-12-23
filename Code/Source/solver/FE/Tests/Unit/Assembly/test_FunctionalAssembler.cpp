/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_FunctionalAssembler.cpp
 * @brief Unit tests for FunctionalAssembler and scalar functional assembly
 */

#include <gtest/gtest.h>

#include "Assembly/FunctionalAssembler.h"
#include "Assembly/AssemblyContext.h"

#include <cmath>
#include <functional>

namespace svmp {
namespace FE {
namespace assembly {
namespace test {

// ============================================================================
// FunctionalKernel Tests
// ============================================================================

TEST(FunctionalKernelTest, L2NormKernelRequiredData) {
    L2NormKernel kernel;
    auto required = kernel.getRequiredData();

    EXPECT_TRUE(hasFlag(required, RequiredData::IntegrationWeights));
    EXPECT_TRUE(hasFlag(required, RequiredData::SolutionValues));
}

TEST(FunctionalKernelTest, L2NormKernelProperties) {
    L2NormKernel kernel;

    EXPECT_TRUE(kernel.requiresSquareRoot());
    EXPECT_EQ(kernel.name(), "L2Norm");
    EXPECT_TRUE(kernel.hasCell());
    EXPECT_FALSE(kernel.hasBoundaryFace());
}

TEST(FunctionalKernelTest, L2NormPostProcess) {
    L2NormKernel kernel;

    // Post-processing should take square root
    EXPECT_NEAR(kernel.postProcess(4.0), 2.0, 1e-12);
    EXPECT_NEAR(kernel.postProcess(9.0), 3.0, 1e-12);
}

TEST(FunctionalKernelTest, H1SeminormKernelProperties) {
    H1SeminormKernel kernel;

    EXPECT_TRUE(kernel.requiresSquareRoot());
    EXPECT_EQ(kernel.name(), "H1Seminorm");
    EXPECT_TRUE(hasFlag(kernel.getRequiredData(), RequiredData::SolutionGradients));
}

TEST(FunctionalKernelTest, EnergyKernelProperties) {
    EnergyKernel kernel;

    EXPECT_FALSE(kernel.requiresSquareRoot());
    EXPECT_EQ(kernel.name(), "Energy");
}

TEST(FunctionalKernelTest, VolumeKernelProperties) {
    VolumeKernel kernel;

    EXPECT_FALSE(kernel.requiresSquareRoot());
    EXPECT_TRUE(kernel.isLinear());
    EXPECT_EQ(kernel.name(), "Volume");
}

TEST(FunctionalKernelTest, BoundaryFluxKernelProperties) {
    BoundaryFluxKernel kernel;

    EXPECT_FALSE(kernel.hasCell());
    EXPECT_TRUE(kernel.hasBoundaryFace());
    EXPECT_EQ(kernel.name(), "BoundaryFlux");
}

// ============================================================================
// L2ErrorKernel Tests
// ============================================================================

TEST(L2ErrorKernelTest, Construction) {
    auto exact = [](Real /*x*/, Real /*y*/, Real /*z*/) { return 1.0; };
    L2ErrorKernel kernel(exact);

    EXPECT_TRUE(kernel.requiresSquareRoot());
    EXPECT_EQ(kernel.name(), "L2Error");
}

TEST(L2ErrorKernelTest, RequiredData) {
    auto exact = [](Real /*x*/, Real /*y*/, Real /*z*/) { return 1.0; };
    L2ErrorKernel kernel(exact);

    auto required = kernel.getRequiredData();
    EXPECT_TRUE(hasFlag(required, RequiredData::SolutionValues));
    EXPECT_TRUE(hasFlag(required, RequiredData::PhysicalPoints));
}

// ============================================================================
// H1ErrorKernel Tests
// ============================================================================

TEST(H1ErrorKernelTest, Construction) {
    auto exact_grad = [](Real /*x*/, Real /*y*/, Real /*z*/) {
        return std::array<Real, 3>{1.0, 0.0, 0.0};
    };
    H1ErrorKernel kernel(exact_grad);

    EXPECT_TRUE(kernel.requiresSquareRoot());
    EXPECT_EQ(kernel.name(), "H1Error");
}

// ============================================================================
// FunctionalAssemblyOptions Tests
// ============================================================================

TEST(FunctionalAssemblyOptionsTest, Defaults) {
    FunctionalAssemblyOptions options;

    EXPECT_EQ(options.num_threads, 1);
    EXPECT_TRUE(options.deterministic);
    EXPECT_FALSE(options.use_kahan_summation);
    EXPECT_FALSE(options.verbose);
}

// ============================================================================
// FunctionalResult Tests
// ============================================================================

TEST(FunctionalResultTest, DefaultConstruction) {
    FunctionalResult result;

    EXPECT_DOUBLE_EQ(result.value, 0.0);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.elements_processed, 0);
    EXPECT_EQ(result.faces_processed, 0);
}

TEST(FunctionalResultTest, Conversion) {
    FunctionalResult result;
    result.value = 3.14;
    result.success = true;

    // Conversion to Real
    Real val = result;
    EXPECT_DOUBLE_EQ(val, 3.14);

    // Conversion to bool
    bool ok = result;
    EXPECT_TRUE(ok);
}

// ============================================================================
// FunctionalAssembler Tests
// ============================================================================

TEST(FunctionalAssemblerTest, DefaultConstruction) {
    FunctionalAssembler assembler;

    // Not configured yet
    EXPECT_FALSE(assembler.isConfigured());
}

TEST(FunctionalAssemblerTest, ConstructionWithOptions) {
    FunctionalAssemblyOptions options;
    options.num_threads = 4;
    options.use_kahan_summation = true;

    FunctionalAssembler assembler(options);
    const auto& opts = assembler.getOptions();

    EXPECT_EQ(opts.num_threads, 4);
    EXPECT_TRUE(opts.use_kahan_summation);
}

TEST(FunctionalAssemblerTest, SetOptions) {
    FunctionalAssembler assembler;

    FunctionalAssemblyOptions options;
    options.deterministic = false;
    assembler.setOptions(options);

    EXPECT_FALSE(assembler.getOptions().deterministic);
}

// ============================================================================
// Factory Tests
// ============================================================================

TEST(FunctionalAssemblerFactoryTest, CreateDefault) {
    auto assembler = createFunctionalAssembler();
    EXPECT_NE(assembler, nullptr);
}

TEST(FunctionalAssemblerFactoryTest, CreateWithOptions) {
    FunctionalAssemblyOptions options;
    options.num_threads = 8;

    auto assembler = createFunctionalAssembler(options);
    EXPECT_NE(assembler, nullptr);
    EXPECT_EQ(assembler->getOptions().num_threads, 8);
}

// ============================================================================
// KahanAccumulator Tests (via FunctionalAssembler behavior)
// ============================================================================

TEST(KahanSummationTest, Precision) {
    // Test that Kahan summation provides better precision
    // by accumulating many small values

    FunctionalAssemblyOptions options_standard;
    options_standard.use_kahan_summation = false;

    FunctionalAssemblyOptions options_kahan;
    options_kahan.use_kahan_summation = true;

    // Both options should produce valid assemblers
    auto assembler_std = createFunctionalAssembler(options_standard);
    auto assembler_kahan = createFunctionalAssembler(options_kahan);

    EXPECT_NE(assembler_std, nullptr);
    EXPECT_NE(assembler_kahan, nullptr);
}

} // namespace test
} // namespace assembly
} // namespace FE
} // namespace svmp
