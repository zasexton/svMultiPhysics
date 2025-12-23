/**
 * @file test_BasisCacheFactory.cpp
 * @brief Tests for basis cache and factory
 */

#include <gtest/gtest.h>
#include "FE/Basis/BasisCache.h"
#include "FE/Basis/BasisFactory.h"
#include "FE/Basis/LagrangeBasis.h"
#include "FE/Quadrature/GaussQuadrature.h"
#include "FE/Quadrature/QuadrilateralQuadrature.h"
#include <array>
#include <thread>
#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::basis;
using namespace svmp::FE::quadrature;

TEST(BasisCache, ReusesEntries) {
    LagrangeBasis basis(ElementType::Line2, 2);
    GaussQuadrature1D quad(4);

    auto& cache = BasisCache::instance();
    cache.clear();
    const auto& entry1 = cache.get_or_compute(basis, quad, true, false);
    const auto& entry2 = cache.get_or_compute(basis, quad, true, false);

    EXPECT_EQ(&entry1, &entry2);
    ASSERT_EQ(entry1.values.size(), quad.num_points());
    ASSERT_EQ(entry1.values.front().size(), basis.size());
}

TEST(BasisCache, VectorBasisPopulatesVectorValues) {
    RaviartThomasBasis basis(ElementType::Quad4, 0);
    QuadrilateralQuadrature quad(2);

    auto& cache = BasisCache::instance();
    cache.clear();
    const auto& entry = cache.get_or_compute(basis, quad, false, false);

    EXPECT_EQ(entry.values.size(), quad.num_points());
    for (const auto& v : entry.values) {
        EXPECT_TRUE(v.empty());
    }
    ASSERT_EQ(entry.vector_values.size(), quad.num_points());
    ASSERT_EQ(entry.vector_values.front().size(), basis.size());
}

TEST(BasisCache, DistinguishesDifferentVectorBases) {
    RaviartThomasBasis rt(ElementType::Quad4, 0);
    NedelecBasis ned(ElementType::Quad4, 0);
    QuadrilateralQuadrature quad(2);

    auto& cache = BasisCache::instance();
    cache.clear();

    const auto& entry_rt = cache.get_or_compute(rt, quad, false, false);
    const auto& entry_ned = cache.get_or_compute(ned, quad, false, false);

    EXPECT_NE(&entry_rt, &entry_ned);
    EXPECT_EQ(cache.size(), 2u);
}

TEST(BasisCache, GradientAndHessianFlagsRespected) {
    LagrangeBasis basis(ElementType::Line2, 2);
    GaussQuadrature1D quad(3);
    auto& cache = BasisCache::instance();
    cache.clear();

    const auto& entry_hess = cache.get_or_compute(basis, quad, false, true);
    EXPECT_TRUE(entry_hess.gradients.empty());
    ASSERT_EQ(entry_hess.hessians.size(), quad.num_points());
    ASSERT_EQ(entry_hess.hessians.front().size(), basis.size());

    const auto& entry_grad = cache.get_or_compute(basis, quad, true, false);
    ASSERT_EQ(entry_grad.gradients.size(), quad.num_points());
    EXPECT_TRUE(entry_grad.hessians.empty());
}

TEST(BasisCache, DifferentQuadratureYieldsDifferentEntries) {
    LagrangeBasis basis(ElementType::Line2, 1);
    GaussQuadrature1D quad2(2);
    GaussQuadrature1D quad3(3);

    auto& cache = BasisCache::instance();
    cache.clear();
    const auto& entry2 = cache.get_or_compute(basis, quad2, true, false);
    const auto& entry3 = cache.get_or_compute(basis, quad3, true, false);

    EXPECT_NE(&entry2, &entry3);
    EXPECT_EQ(entry2.values.size(), quad2.num_points());
    EXPECT_EQ(entry3.values.size(), quad3.num_points());
}

TEST(BasisCache, ThreadSafetySingleEntryUnderConcurrency) {
    LagrangeBasis basis(ElementType::Line2, 2);
    GaussQuadrature1D quad(3);
    auto& cache = BasisCache::instance();
    cache.clear();

    constexpr int nthreads = 8;
    std::array<const BasisCacheEntry*, nthreads> entries{};

    std::vector<std::thread> threads;
    threads.reserve(nthreads);
    for (int t = 0; t < nthreads; ++t) {
        threads.emplace_back([&, t]() {
            const auto& entry = cache.get_or_compute(basis, quad, true, false);
            entries[static_cast<std::size_t>(t)] = &entry;
        });
    }
    for (auto& th : threads) {
        th.join();
    }

    // All threads should see the same cached entry pointer
    const auto* ref = entries[0];
    for (int t = 1; t < nthreads; ++t) {
        EXPECT_EQ(entries[static_cast<std::size_t>(t)], ref);
    }
    EXPECT_EQ(cache.size(), 1u);
    ASSERT_EQ(ref->values.size(), quad.num_points());
}
TEST(BasisFactory, CreatesVectorConformingBasis) {
    BasisRequest req{ElementType::Quad4, BasisType::Lagrange, 0, Continuity::H_div, FieldType::Vector};
    auto basis = BasisFactory::create(req);
    EXPECT_TRUE(basis->is_vector_valued());
    EXPECT_EQ(basis->element_type(), ElementType::Quad4);
}

TEST(BasisFactory, CreatesHDivHigherOrderOnQuad) {
    BasisRequest req{ElementType::Quad4, BasisType::Lagrange, 2, Continuity::H_div, FieldType::Vector};
    auto basis = BasisFactory::create(req);
    EXPECT_TRUE(basis->is_vector_valued());
    EXPECT_EQ(basis->basis_type(), BasisType::RaviartThomas);
    EXPECT_EQ(basis->element_type(), ElementType::Quad4);
    EXPECT_EQ(basis->order(), 2);
    EXPECT_EQ(basis->size(), 24u); // 2*(k+1)*(k+2) with k=2
}

TEST(BasisFactory, CreatesHDivHigherOrderOnTriangle) {
    BasisRequest req{ElementType::Triangle3, BasisType::Lagrange, 2, Continuity::H_div, FieldType::Vector};
    auto basis = BasisFactory::create(req);
    EXPECT_TRUE(basis->is_vector_valued());
    EXPECT_EQ(basis->basis_type(), BasisType::RaviartThomas);
    EXPECT_EQ(basis->element_type(), ElementType::Triangle3);
    EXPECT_EQ(basis->order(), 2);
    EXPECT_EQ(basis->size(), 15u); // (k+1)*(k+3) with k=2
}

TEST(BasisFactory, CreatesHCurlHigherOrderOnQuad) {
    BasisRequest req{ElementType::Quad4, BasisType::Lagrange, 1, Continuity::H_curl, FieldType::Vector};
    auto basis = BasisFactory::create(req);
    EXPECT_TRUE(basis->is_vector_valued());
    EXPECT_EQ(basis->basis_type(), BasisType::Nedelec);
    EXPECT_EQ(basis->element_type(), ElementType::Quad4);
    EXPECT_EQ(basis->order(), 1);
    EXPECT_EQ(basis->size(), 12u); // 2*(k+1)*(k+2) with k=1
}

TEST(BasisFactory, CreatesHCurlHigherOrderOnTriangle) {
    BasisRequest req{ElementType::Triangle3, BasisType::Lagrange, 1, Continuity::H_curl, FieldType::Vector};
    auto basis = BasisFactory::create(req);
    EXPECT_TRUE(basis->is_vector_valued());
    EXPECT_EQ(basis->basis_type(), BasisType::Nedelec);
    EXPECT_EQ(basis->element_type(), ElementType::Triangle3);
    EXPECT_EQ(basis->order(), 1);
    EXPECT_EQ(basis->size(), 8u); // (k+1)*(k+3) with k=1
}

TEST(BasisFactory, CreatesHCurlHigherOrderOnTetra) {
    BasisRequest req{ElementType::Tetra4, BasisType::Lagrange, 1, Continuity::H_curl, FieldType::Vector};
    auto basis = BasisFactory::create(req);
    EXPECT_TRUE(basis->is_vector_valued());
    EXPECT_EQ(basis->basis_type(), BasisType::Nedelec);
    EXPECT_EQ(basis->element_type(), ElementType::Tetra4);
    EXPECT_EQ(basis->order(), 1);
    EXPECT_EQ(basis->size(), 20u); // (k+1)*(k+3)*(k+4)/2 with k=1
}

TEST(BasisFactory, CreatesScalarBasesByType) {
    {
        BasisRequest req{ElementType::Line2, BasisType::Lagrange, 2, Continuity::C0, FieldType::Scalar};
        auto basis = BasisFactory::create(req);
        EXPECT_EQ(basis->basis_type(), BasisType::Lagrange);
        EXPECT_EQ(basis->size(), 3u);
    }
    {
        BasisRequest req{ElementType::Line2, BasisType::Hierarchical, 3, Continuity::C0, FieldType::Scalar};
        auto basis = BasisFactory::create(req);
        EXPECT_EQ(basis->basis_type(), BasisType::Hierarchical);
        EXPECT_EQ(basis->size(), 4u);
    }
    {
        BasisRequest req{ElementType::Line2, BasisType::Spectral, 3, Continuity::C0, FieldType::Scalar};
        auto basis = BasisFactory::create(req);
        EXPECT_EQ(basis->basis_type(), BasisType::Spectral);
        EXPECT_EQ(basis->size(), 4u);
    }
    {
        BasisRequest req{ElementType::Triangle3, BasisType::Bernstein, 2, Continuity::C0, FieldType::Scalar};
        auto basis = BasisFactory::create(req);
        EXPECT_EQ(basis->basis_type(), BasisType::Bernstein);
        EXPECT_EQ(basis->element_type(), ElementType::Triangle3);
    }
    {
        BasisRequest req{ElementType::Quad4, BasisType::Serendipity, 2, Continuity::C0, FieldType::Scalar};
        auto basis = BasisFactory::create(req);
        EXPECT_EQ(basis->basis_type(), BasisType::Serendipity);
        EXPECT_EQ(basis->size(), 8u);
    }
}

TEST(BasisFactory, UnsupportedCombinationThrows) {
    BasisRequest req{ElementType::Tetra4, BasisType::Serendipity, 1, Continuity::C0, FieldType::Scalar};
    EXPECT_THROW(auto b = BasisFactory::create(req), svmp::FE::FEException);
}
