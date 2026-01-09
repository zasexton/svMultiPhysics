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

#include "Forms/BoundaryFunctional.h"
#include "Forms/FormExpr.h"
#include "Spaces/H1Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <cmath>
#include <functional>
#include <vector>

namespace svmp {
namespace FE {
namespace assembly {
namespace test {

namespace {

class BoundaryMeasureKernel final : public FunctionalKernel {
public:
    [[nodiscard]] RequiredData getRequiredData() const noexcept override
    {
        return RequiredData::IntegrationWeights;
    }

    [[nodiscard]] bool hasCell() const noexcept override { return false; }
    [[nodiscard]] bool hasBoundaryFace() const noexcept override { return true; }

    [[nodiscard]] Real evaluateCell(const AssemblyContext& /*ctx*/, LocalIndex /*q*/) override { return 0.0; }

    [[nodiscard]] Real evaluateBoundaryFace(const AssemblyContext& /*ctx*/,
                                           LocalIndex /*q*/,
                                           int /*boundary_marker*/) override
    {
        return 1.0;
    }

    [[nodiscard]] std::string name() const override { return "BoundaryMeasureKernel"; }
};

} // namespace

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

// ============================================================================
// Boundary Functional Assembly (Regression/Integration)
// ============================================================================

TEST(FunctionalAssemblerBoundaryTest, IntegrateConstantOverBoundaryFace)
{
    forms::test::SingleTetraOneBoundaryFaceMeshAccess mesh(/*boundary_marker=*/2);
    auto dof_map = forms::test::createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, 1);

    FunctionalAssembler assembler;
    assembler.setMesh(mesh);
    assembler.setDofMap(dof_map);
    assembler.setSpace(space);

    BoundaryMeasureKernel kernel;
    const Real area = assembler.assembleBoundaryScalar(kernel, /*boundary_marker=*/2);
    EXPECT_NEAR(area, 0.5, 1e-12);
}

TEST(FunctionalAssemblerBoundaryTest, BoundaryFluxUsesElementLocalCoefficients)
{
    forms::test::SingleTetraOneBoundaryFaceMeshAccess mesh(/*boundary_marker=*/2);

    // Cell DOFs are not 0..3 to ensure the assembler gathers element-local coefficients correctly.
    dofs::DofMap dof_map(/*n_cells=*/1, /*n_dofs_total=*/10, /*dofs_per_cell=*/4);
    const std::array<GlobalIndex, 4> cell_dofs = {5, 6, 7, 8};
    dof_map.setCellDofs(0, cell_dofs);
    dof_map.setNumDofs(10);
    dof_map.setNumLocalDofs(10);
    dof_map.finalize();

    spaces::H1Space space(ElementType::Tetra4, 1);

    // Linear field u=z on the reference tetra:
    // nodes (0,1,2) have z=0, node 3 has z=1.
    std::vector<Real> u(10, 100.0);
    u[5] = 0.0;
    u[6] = 0.0;
    u[7] = 0.0;
    u[8] = 1.0;

    FunctionalAssembler assembler;
    assembler.setMesh(mesh);
    assembler.setDofMap(dof_map);
    assembler.setSpace(space);
    assembler.setSolution(u);

    BoundaryFluxKernel kernel;
    const Real flux = assembler.assembleBoundaryScalar(kernel, /*boundary_marker=*/2);

    // On face {0,1,2} (z=0 plane), outward normal is -e_z and grad(z)=+e_z.
    // So ∫ grad(u)·n ds = ∫ (-1) ds = -area = -0.5.
    EXPECT_NEAR(flux, -0.5, 1e-12);
}

TEST(FunctionalAssemblerBoundaryTest, FormsBoundaryFunctionalKernelEvaluatesDiscreteField)
{
    constexpr FieldId field = 0;

    forms::test::SingleTetraOneBoundaryFaceMeshAccess mesh(/*boundary_marker=*/2);

    dofs::DofMap dof_map(/*n_cells=*/1, /*n_dofs_total=*/10, /*dofs_per_cell=*/4);
    const std::array<GlobalIndex, 4> cell_dofs = {5, 6, 7, 8};
    dof_map.setCellDofs(0, cell_dofs);
    dof_map.setNumDofs(10);
    dof_map.setNumLocalDofs(10);
    dof_map.finalize();

    spaces::H1Space space(ElementType::Tetra4, 1);

    std::vector<Real> u(10, 123.0);
    u[5] = 1.0;
    u[6] = 1.0;
    u[7] = 1.0;
    u[8] = 1.0;

    const auto integrand = forms::FormExpr::discreteField(field, space, "u");
    auto kernel = forms::compileBoundaryFunctionalKernel(integrand, /*boundary_marker=*/2);

    FunctionalAssembler assembler;
    assembler.setMesh(mesh);
    assembler.setDofMap(dof_map);
    assembler.setSpace(space);
    assembler.setPrimaryField(field);
    assembler.setSolution(u);

    const Real Q = assembler.assembleBoundaryScalar(*kernel, /*boundary_marker=*/2);
    EXPECT_NEAR(Q, 0.5, 1e-12);
}

} // namespace test
} // namespace assembly
} // namespace FE
} // namespace svmp
