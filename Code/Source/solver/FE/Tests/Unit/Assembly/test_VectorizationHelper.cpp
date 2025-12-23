/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_VectorizationHelper.cpp
 * @brief Unit tests for VectorizationHelper and SIMD assembly utilities
 */

#include <gtest/gtest.h>

#include "Assembly/VectorizationHelper.h"

#include <cmath>
#include <numeric>

namespace svmp {
namespace FE {
namespace assembly {
namespace test {

// ============================================================================
// VectorWidth Tests
// ============================================================================

TEST(VectorWidthTest, DoubleWidth) {
    // Double precision width depends on SIMD capabilities
    std::size_t width = VectorWidth::double_width;
    EXPECT_GT(width, 0u);
    // Should be a power of 2 or 1
    EXPECT_TRUE(width == 1 || width == 2 || width == 4 || width == 8);
}

TEST(VectorWidthTest, FloatWidth) {
    std::size_t width = VectorWidth::float_width;
    EXPECT_GT(width, 0u);
    // Float width should be >= double width
    EXPECT_GE(width, VectorWidth::double_width);
}

TEST(VectorWidthTest, DefaultWidth) {
    EXPECT_EQ(VectorWidth::default_width, VectorWidth::double_width);
}

// ============================================================================
// VectorizationOptions Tests
// ============================================================================

TEST(VectorizationOptionsTest, Defaults) {
    VectorizationOptions options;

    EXPECT_TRUE(options.enable_simd);
    EXPECT_GT(options.batch_size, 0u);
    EXPECT_TRUE(options.use_aligned_memory);
    EXPECT_EQ(options.alignment, 64u);  // AVX512 alignment
    EXPECT_FALSE(options.runtime_dispatch);
}

TEST(VectorizationOptionsTest, CustomValues) {
    VectorizationOptions options;
    options.enable_simd = false;
    options.batch_size = 128;
    options.alignment = 32;

    EXPECT_FALSE(options.enable_simd);
    EXPECT_EQ(options.batch_size, 128u);
    EXPECT_EQ(options.alignment, 32u);
}

// ============================================================================
// Aligned Memory Utilities Tests
// ============================================================================

TEST(AlignedMemoryTest, IsAligned) {
    // Stack allocated might not be aligned
    alignas(64) char buffer[128];

    EXPECT_TRUE(isAligned(buffer, 64));
    EXPECT_TRUE(isAligned(buffer, 32));
    EXPECT_TRUE(isAligned(buffer, 16));
}

TEST(AlignedMemoryTest, AlignUp) {
    EXPECT_EQ(alignUp(0, 64), 0u);
    EXPECT_EQ(alignUp(1, 64), 64u);
    EXPECT_EQ(alignUp(63, 64), 64u);
    EXPECT_EQ(alignUp(64, 64), 64u);
    EXPECT_EQ(alignUp(65, 64), 128u);
    EXPECT_EQ(alignUp(100, 16), 112u);
}

TEST(AlignedVectorTest, Allocation) {
    AlignedVector<Real> vec(100, 0.0);

    EXPECT_EQ(vec.size(), 100u);
    // Check alignment of data
    EXPECT_TRUE(isAligned(vec.data(), 64));
}

TEST(AlignedVectorTest, Operations) {
    AlignedVector<Real> vec = {1.0, 2.0, 3.0, 4.0};

    EXPECT_EQ(vec.size(), 4u);
    EXPECT_DOUBLE_EQ(vec[0], 1.0);
    EXPECT_DOUBLE_EQ(vec[3], 4.0);

    vec.push_back(5.0);
    EXPECT_EQ(vec.size(), 5u);
}

// ============================================================================
// ElementBatch Tests
// ============================================================================

TEST(ElementBatchTest, DefaultConstruction) {
    ElementBatch<> batch;

    EXPECT_TRUE(batch.isEmpty());
    EXPECT_FALSE(batch.isFull());
    EXPECT_EQ(batch.numActive(), 0u);
}

TEST(ElementBatchTest, ParameterizedConstruction) {
    ElementBatch<> batch(8, 4, 3);  // max_dofs=8, max_qpts=4, dim=3

    EXPECT_EQ(batch.maxDofs(), 8);
    EXPECT_EQ(batch.maxQpts(), 4);
    EXPECT_EQ(batch.dimension(), 3);
    EXPECT_TRUE(batch.isEmpty());
}

TEST(ElementBatchTest, AddElement) {
    ElementBatch<4> batch(4, 2, 2);

    int slot0 = batch.addElement(100);
    EXPECT_EQ(slot0, 0);
    EXPECT_EQ(batch.numActive(), 1u);
    EXPECT_FALSE(batch.isEmpty());

    int slot1 = batch.addElement(101);
    EXPECT_EQ(slot1, 1);
    EXPECT_EQ(batch.numActive(), 2u);
}

TEST(ElementBatchTest, IsFull) {
    ElementBatch<4> batch(4, 2, 2);

    EXPECT_FALSE(batch.isFull());

    batch.addElement(0);
    batch.addElement(1);
    batch.addElement(2);
    EXPECT_FALSE(batch.isFull());

    batch.addElement(3);
    EXPECT_TRUE(batch.isFull());

    // Adding to full batch should return -1
    int slot = batch.addElement(4);
    EXPECT_EQ(slot, -1);
}

TEST(ElementBatchTest, CellId) {
    ElementBatch<4> batch(4, 2, 2);

    batch.addElement(42);
    batch.addElement(99);

    EXPECT_EQ(batch.cellId(0), 42);
    EXPECT_EQ(batch.cellId(1), 99);
}

TEST(ElementBatchTest, Clear) {
    ElementBatch<4> batch(4, 2, 2);

    batch.addElement(0);
    batch.addElement(1);

    batch.clear();

    EXPECT_TRUE(batch.isEmpty());
    EXPECT_EQ(batch.numActive(), 0u);
}

TEST(ElementBatchTest, QuadWeightsAccess) {
    ElementBatch<> batch(4, 2, 3);

    Real* weights = batch.quadWeights();
    EXPECT_NE(weights, nullptr);

    const ElementBatch<>& const_batch = batch;
    [[maybe_unused]] const Real* const_weights = const_batch.quadWeights();
}

TEST(ElementBatchTest, JacobianDetsAccess) {
    ElementBatch<> batch(4, 2, 3);

    Real* dets = batch.jacobianDets();
    EXPECT_NE(dets, nullptr);
}

TEST(ElementBatchTest, BasisValuesAccess) {
    ElementBatch<> batch(4, 2, 3);

    Real* values = batch.basisValues();
    EXPECT_NE(values, nullptr);
}

TEST(ElementBatchTest, BasisGradientsAccess) {
    ElementBatch<> batch(4, 2, 3);

    Real* grads = batch.basisGradients();
    EXPECT_NE(grads, nullptr);
}

TEST(ElementBatchTest, QuadOffset) {
    ElementBatch<4> batch(8, 4, 3);  // max_dofs=8, max_qpts=4, dim=3

    // offset = slot * max_qpts + q
    EXPECT_EQ(batch.quadOffset(0, 0), 0u);
    EXPECT_EQ(batch.quadOffset(0, 2), 2u);
    EXPECT_EQ(batch.quadOffset(1, 0), 4u);  // 1 * 4 + 0
    EXPECT_EQ(batch.quadOffset(2, 3), 11u); // 2 * 4 + 3
}

TEST(ElementBatchTest, BasisOffset) {
    ElementBatch<4> batch(4, 2, 3);  // max_dofs=4, max_qpts=2

    // offset = slot * max_dofs * max_qpts + dof * max_qpts + q
    EXPECT_EQ(batch.basisOffset(0, 0, 0), 0u);
    EXPECT_EQ(batch.basisOffset(0, 1, 0), 2u);   // 0 * 8 + 1 * 2 + 0
    EXPECT_EQ(batch.basisOffset(1, 0, 0), 8u);   // 1 * 4 * 2 + 0 * 2 + 0
}

TEST(ElementBatchTest, GradOffset) {
    ElementBatch<4> batch(4, 2, 3);  // max_dofs=4, max_qpts=2, dim=3

    // offset = slot * max_dofs * max_qpts * dim + dof * max_qpts * dim + q * dim + d
    EXPECT_EQ(batch.gradOffset(0, 0, 0, 0), 0u);
    EXPECT_EQ(batch.gradOffset(0, 0, 0, 1), 1u);
    EXPECT_EQ(batch.gradOffset(0, 0, 0, 2), 2u);
    EXPECT_EQ(batch.gradOffset(0, 0, 1, 0), 3u);   // q=1
    EXPECT_EQ(batch.gradOffset(0, 1, 0, 0), 6u);   // dof=1
}

// ============================================================================
// VectorizedQuadrature Tests
// ============================================================================

TEST(VectorizedQuadratureTest, WeightedSum) {
    AlignedVector<Real> weights = {0.5, 0.5, 0.5, 0.5};
    AlignedVector<Real> values = {1.0, 2.0, 3.0, 4.0};

    Real sum = VectorizedQuadrature::weightedSum(
        weights.data(), values.data(), 4);

    EXPECT_NEAR(sum, 5.0, 1e-12);  // 0.5*(1+2+3+4) = 5
}

TEST(VectorizedQuadratureTest, ComputeJxW) {
    AlignedVector<Real> jacobian_dets = {2.0, 2.0, 2.0, 2.0};
    AlignedVector<Real> ref_weights = {0.25, 0.25, 0.25, 0.25};
    AlignedVector<Real> output(4);

    VectorizedQuadrature::computeJxW(
        jacobian_dets.data(), ref_weights.data(), output.data(), 4);

    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(output[i], 0.5, 1e-12);  // 2.0 * 0.25 = 0.5
    }
}

TEST(VectorizedQuadratureTest, GradientDotProduct) {
    // 2D case: dim=2, 4 quadrature points
    std::size_t num_qpts = 4;
    int dim = 2;

    // grad_phi = [1, 0, 1, 0] (x-component) + [0, 1, 0, 1] (y-component)
    // grad_psi = [1, 1, 1, 1] (x-component) + [0, 0, 0, 0] (y-component)
    AlignedVector<Real> grad_phi(static_cast<std::size_t>(dim) * num_qpts);
    AlignedVector<Real> grad_psi(static_cast<std::size_t>(dim) * num_qpts);
    AlignedVector<Real> JxW(num_qpts, 0.25);

    // grad_phi_x
    grad_phi[0] = 1.0; grad_phi[1] = 0.0; grad_phi[2] = 1.0; grad_phi[3] = 0.0;
    // grad_phi_y
    grad_phi[4] = 0.0; grad_phi[5] = 1.0; grad_phi[6] = 0.0; grad_phi[7] = 1.0;

    // grad_psi_x
    grad_psi[0] = 1.0; grad_psi[1] = 1.0; grad_psi[2] = 1.0; grad_psi[3] = 1.0;
    // grad_psi_y
    grad_psi[4] = 0.0; grad_psi[5] = 0.0; grad_psi[6] = 0.0; grad_psi[7] = 0.0;

    Real result = VectorizedQuadrature::gradientDotProduct(
        grad_phi.data(), grad_psi.data(), JxW.data(), num_qpts, dim);

    // Expected: sum over q of (grad_phi_x * grad_psi_x + grad_phi_y * grad_psi_y) * JxW
    // = (1*1 + 0*0)*0.25 + (0*1 + 1*0)*0.25 + (1*1 + 0*0)*0.25 + (0*1 + 1*0)*0.25
    // = 0.25 + 0 + 0.25 + 0 = 0.5
    EXPECT_NEAR(result, 0.5, 1e-12);
}

// ============================================================================
// BatchedMatrixAssembly Tests
// ============================================================================

TEST(BatchedMatrixAssemblyTest, AssembleMassMatrices) {
    constexpr std::size_t batch_size = 2;
    ElementBatch<batch_size> batch(2, 2, 2);  // 2 DOFs, 2 qpts, 2D

    // Add 2 elements
    batch.addElement(0);
    batch.addElement(1);

    // Set up quadrature data
    for (std::size_t e = 0; e < 2; ++e) {
        for (LocalIndex q = 0; q < 2; ++q) {
            std::size_t idx = batch.quadOffset(e, q);
            batch.jacobianDets()[idx] = 1.0;
            batch.quadWeights()[idx] = 0.5;
        }
    }

    // Set up basis values (phi_i = 1 for simplicity)
    for (std::size_t e = 0; e < 2; ++e) {
        for (LocalIndex dof = 0; dof < 2; ++dof) {
            for (LocalIndex q = 0; q < 2; ++q) {
                std::size_t idx = batch.basisOffset(e, dof, q);
                batch.basisValues()[idx] = 1.0;
            }
        }
    }

    std::array<std::vector<Real>, batch_size> local_matrices;

    BatchedMatrixAssembly<batch_size>::assembleMassMatrices(batch, local_matrices);

    // Check matrices are populated
    EXPECT_EQ(local_matrices[0].size(), 4u);  // 2x2
    EXPECT_EQ(local_matrices[1].size(), 4u);

    // With phi_i = 1 and JxW = 0.5 * 2 qpts = 1.0, M[i,j] = 1.0
    EXPECT_NEAR(local_matrices[0][0], 1.0, 1e-12);
}

TEST(BatchedMatrixAssemblyTest, AssembleStiffnessMatrices) {
    constexpr std::size_t batch_size = 2;
    ElementBatch<batch_size> batch(2, 2, 2);  // 2 DOFs, 2 qpts, 2D

    batch.addElement(0);
    batch.addElement(1);

    // Set up quadrature data
    for (std::size_t e = 0; e < 2; ++e) {
        for (LocalIndex q = 0; q < 2; ++q) {
            std::size_t idx = batch.quadOffset(e, q);
            batch.jacobianDets()[idx] = 1.0;
            batch.quadWeights()[idx] = 0.5;
        }
    }

    // Set up basis gradients (unit vectors)
    for (std::size_t e = 0; e < 2; ++e) {
        for (LocalIndex dof = 0; dof < 2; ++dof) {
            for (LocalIndex q = 0; q < 2; ++q) {
                for (int d = 0; d < 2; ++d) {
                    std::size_t idx = batch.gradOffset(e, dof, q, d);
                    batch.basisGradients()[idx] = (d == 0) ? 1.0 : 0.0;
                }
            }
        }
    }

    std::array<std::vector<Real>, batch_size> local_matrices;

    BatchedMatrixAssembly<batch_size>::assembleStiffnessMatrices(batch, local_matrices);

    EXPECT_EQ(local_matrices[0].size(), 4u);
    EXPECT_EQ(local_matrices[1].size(), 4u);
}

// ============================================================================
// VectorizationHelper Tests
// ============================================================================

TEST(VectorizationHelperTest, DefaultConstruction) {
    VectorizationHelper helper;

    // Should construct successfully
    SUCCEED();
}

TEST(VectorizationHelperTest, ConstructionWithOptions) {
    VectorizationOptions options;
    options.enable_simd = false;

    VectorizationHelper helper(options);

    EXPECT_FALSE(helper.isSIMDEnabled());
}

TEST(VectorizationHelperTest, SetOptions) {
    VectorizationHelper helper;

    VectorizationOptions options;
    options.batch_size = 64;

    helper.setOptions(options);

    EXPECT_EQ(helper.getOptions().batch_size, 64u);
}

TEST(VectorizationHelperTest, IsSIMDEnabled) {
    VectorizationHelper helper;

    // With default options, SIMD should be enabled (if available)
    bool simd_enabled = helper.isSIMDEnabled();

    // Just verify the function returns without error
    (void)simd_enabled;
    SUCCEED();
}

TEST(VectorizationHelperTest, EffectiveVectorWidth) {
    VectorizationHelper helper;

    std::size_t width = helper.effectiveVectorWidth();
    EXPECT_GT(width, 0u);
}

TEST(VectorizationHelperTest, OptimalBatchSize) {
    VectorizationHelper helper;

    std::size_t batch = helper.optimalBatchSize();
    EXPECT_GT(batch, 0u);
}

TEST(VectorizationHelperTest, CreateBatch) {
    VectorizationHelper helper;

    auto batch = helper.createBatch(8, 4, 3);

    EXPECT_NE(batch, nullptr);
    EXPECT_EQ(batch->maxDofs(), 8);
    EXPECT_EQ(batch->maxQpts(), 4);
    EXPECT_EQ(batch->dimension(), 3);
}

TEST(VectorizationHelperTest, Dot) {
    VectorizationHelper helper;

    std::vector<Real> a = {1.0, 2.0, 3.0, 4.0};
    std::vector<Real> b = {1.0, 1.0, 1.0, 1.0};

    Real result = helper.dot(a, b);

    EXPECT_NEAR(result, 10.0, 1e-12);
}

TEST(VectorizationHelperTest, Norm) {
    VectorizationHelper helper;

    std::vector<Real> a = {3.0, 4.0};

    Real result = helper.norm(a);

    EXPECT_NEAR(result, 5.0, 1e-12);  // sqrt(9 + 16) = 5
}

TEST(VectorizationHelperTest, Axpy) {
    VectorizationHelper helper;

    std::vector<Real> x = {1.0, 2.0, 3.0, 4.0};
    std::vector<Real> y = {1.0, 1.0, 1.0, 1.0};

    helper.axpy(2.0, x, y);  // y = 2*x + y

    EXPECT_NEAR(y[0], 3.0, 1e-12);
    EXPECT_NEAR(y[1], 5.0, 1e-12);
    EXPECT_NEAR(y[2], 7.0, 1e-12);
    EXPECT_NEAR(y[3], 9.0, 1e-12);
}

TEST(VectorizationHelperTest, Scale) {
    VectorizationHelper helper;

    std::vector<Real> x = {1.0, 2.0, 3.0, 4.0};
    std::vector<Real> y(4);

    helper.scale(0.5, x, y);

    EXPECT_NEAR(y[0], 0.5, 1e-12);
    EXPECT_NEAR(y[1], 1.0, 1e-12);
    EXPECT_NEAR(y[2], 1.5, 1e-12);
    EXPECT_NEAR(y[3], 2.0, 1e-12);
}

TEST(VectorizationHelperTest, Gemv) {
    VectorizationHelper helper;

    // 2x2 matrix [[1, 2], [3, 4]]
    std::vector<Real> A = {1.0, 2.0, 3.0, 4.0};
    std::vector<Real> x = {1.0, 1.0};
    std::vector<Real> y(2, 0.0);

    helper.gemv(A, x, y, 2, 2);

    EXPECT_NEAR(y[0], 3.0, 1e-12);  // 1*1 + 2*1
    EXPECT_NEAR(y[1], 7.0, 1e-12);  // 3*1 + 4*1
}

TEST(VectorizationHelperTest, GetSIMDInfo) {
    std::string info = VectorizationHelper::getSIMDInfo();

    EXPECT_FALSE(info.empty());
}

TEST(VectorizationHelperTest, MoveConstruction) {
    VectorizationOptions options;
    options.batch_size = 256;

    VectorizationHelper helper1(options);
    VectorizationHelper helper2(std::move(helper1));

    EXPECT_EQ(helper2.getOptions().batch_size, 256u);
}

// ============================================================================
// Factory Tests
// ============================================================================

TEST(VectorizationHelperFactoryTest, CreateDefault) {
    auto helper = createVectorizationHelper();

    EXPECT_NE(helper, nullptr);
}

TEST(VectorizationHelperFactoryTest, CreateWithOptions) {
    VectorizationOptions options;
    options.alignment = 32;

    auto helper = createVectorizationHelper(options);

    EXPECT_NE(helper, nullptr);
    EXPECT_EQ(helper->getOptions().alignment, 32u);
}

} // namespace test
} // namespace assembly
} // namespace FE
} // namespace svmp
