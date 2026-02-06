/**
 * @file test_JacobianCacheGeometryQuadrature.cpp
 * @brief Tests for Jacobian caching and geometry-aware quadrature
 */

#include <gtest/gtest.h>
#include "FE/Geometry/JacobianCache.h"
#include "FE/Geometry/GeometryQuadrature.h"
#include "FE/Geometry/LinearMapping.h"
#include "FE/Geometry/IsoparametricMapping.h"
#include "FE/Basis/LagrangeBasis.h"
#include "FE/Basis/SerendipityBasis.h"
#include "FE/Quadrature/GaussQuadrature.h"
#include "FE/Quadrature/QuadrilateralQuadrature.h"
#include "FE/Quadrature/HexahedronQuadrature.h"
#include "FE/Quadrature/TriangleQuadrature.h"
#include "FE/Quadrature/TetrahedronQuadrature.h"
#include "FE/Quadrature/WedgeQuadrature.h"
#include "FE/Quadrature/PyramidQuadrature.h"
#include "FE/Quadrature/QuadratureRule.h"
#include <array>
#include <cmath>
#include <thread>

using namespace svmp::FE;
using namespace svmp::FE::geometry;

TEST(JacobianCache, ReusesEntryAcrossThreads) {
    std::vector<math::Vector<Real, 3>> nodes = {
        {Real(0), Real(0), Real(0)},
        {Real(2), Real(0), Real(0)}
    };
    LinearMapping map(ElementType::Line2, nodes);
    quadrature::GaussQuadrature1D quad(2);
    auto& cache = JacobianCache::instance();
    cache.clear();

    std::array<const JacobianCacheEntry*, 4> entries{};
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t]() {
            const auto& e = cache.get_or_compute(map, quad);
            entries[static_cast<std::size_t>(t)] = &e;
        });
    }
    for (auto& th : threads) th.join();

    const auto* ref = entries[0];
    for (int t = 1; t < 4; ++t) {
        EXPECT_EQ(entries[static_cast<std::size_t>(t)], ref);
    }
    EXPECT_EQ(cache.size(), 1u);
    ASSERT_EQ(ref->J.size(), quad.num_points());
    EXPECT_NEAR(ref->detJ.front(), 1.0, 1e-12); // length 2 -> J=1
}

namespace {

class CustomLineQuadrature final : public quadrature::QuadratureRule {
public:
    CustomLineQuadrature(std::vector<quadrature::QuadPoint> pts,
                         std::vector<Real> wts,
                         int order)
        : quadrature::QuadratureRule(svmp::CellFamily::Line, 1, order) {
        set_data(std::move(pts), std::move(wts));
    }
};

class CountingAffineLineMapping final : public GeometryMapping {
public:
    ElementType element_type() const noexcept override { return ElementType::Line2; }
    int dimension() const noexcept override { return 1; }
    std::size_t num_nodes() const noexcept override { return nodes_.size(); }
    const std::vector<math::Vector<Real, 3>>& nodes() const noexcept override { return nodes_; }
    bool isAffine() const noexcept override { return true; }

    math::Vector<Real, 3> map_to_physical(const math::Vector<Real, 3>& xi) const override {
        return math::Vector<Real, 3>{xi[0], Real(0), Real(0)};
    }

    math::Vector<Real, 3> map_to_reference(const math::Vector<Real, 3>& x_phys,
                                           const math::Vector<Real, 3>&) const override {
        return math::Vector<Real, 3>{x_phys[0], Real(0), Real(0)};
    }

    math::Matrix<Real, 3, 3> jacobian(const math::Vector<Real, 3>&) const override {
        ++jacobian_calls;
        math::Matrix<Real, 3, 3> J{};
        J(0, 0) = Real(1);
        J(1, 1) = Real(1);
        J(2, 2) = Real(1);
        return J;
    }

    mutable int jacobian_calls{0};

private:
    std::vector<math::Vector<Real, 3>> nodes_{
        {Real(0), Real(0), Real(0)},
        {Real(1), Real(0), Real(0)},
    };
};

class CountingNonAffineLineMapping final : public GeometryMapping {
public:
    ElementType element_type() const noexcept override { return ElementType::Line2; }
    int dimension() const noexcept override { return 1; }
    std::size_t num_nodes() const noexcept override { return nodes_.size(); }
    const std::vector<math::Vector<Real, 3>>& nodes() const noexcept override { return nodes_; }

    math::Vector<Real, 3> map_to_physical(const math::Vector<Real, 3>& xi) const override {
        const Real x = xi[0];
        return math::Vector<Real, 3>{x + Real(0.5) * x * x, Real(0), Real(0)};
    }

    math::Vector<Real, 3> map_to_reference(const math::Vector<Real, 3>& x_phys,
                                           const math::Vector<Real, 3>&) const override {
        return math::Vector<Real, 3>{x_phys[0], Real(0), Real(0)};
    }

    math::Matrix<Real, 3, 3> jacobian(const math::Vector<Real, 3>& xi) const override {
        ++jacobian_calls;
        math::Matrix<Real, 3, 3> J{};
        J(0, 0) = Real(1) + xi[0];
        J(1, 1) = Real(1);
        J(2, 2) = Real(1);
        return J;
    }

    mutable int jacobian_calls{0};

private:
    std::vector<math::Vector<Real, 3>> nodes_{
        {Real(0), Real(0), Real(0)},
        {Real(1), Real(0), Real(0)},
    };
};

} // namespace

TEST(JacobianCache, DistinctQuadratureRulesDoNotCollide) {
    std::vector<math::Vector<Real, 3>> nodes = {
        {Real(0), Real(0), Real(0)},
        {Real(2), Real(0), Real(0)}
    };
    LinearMapping map(ElementType::Line2, nodes);

    CustomLineQuadrature quad_a(
        {quadrature::QuadPoint{Real(-0.5), Real(0), Real(0)},
         quadrature::QuadPoint{Real(0.5), Real(0), Real(0)}},
        {Real(1), Real(1)},
        2);

    CustomLineQuadrature quad_b(
        {quadrature::QuadPoint{Real(-0.25), Real(0), Real(0)},
         quadrature::QuadPoint{Real(0.75), Real(0), Real(0)}},
        {Real(1), Real(1)},
        2);

    auto& cache = JacobianCache::instance();
    cache.clear();
    const auto& a = cache.get_or_compute(map, quad_a);
    const auto& b = cache.get_or_compute(map, quad_b);

    EXPECT_NE(&a, &b);
    EXPECT_EQ(cache.size(), 2u);
}

TEST(JacobianCache, AffineMappingComputesJacobianOnce) {
    CountingAffineLineMapping map;
    quadrature::GaussQuadrature1D quad(4);
    auto& cache = JacobianCache::instance();
    cache.clear();

    const auto& entry = cache.get_or_compute(map, quad);
    ASSERT_EQ(entry.J.size(), quad.num_points());
    EXPECT_EQ(map.jacobian_calls, 1);
}

TEST(JacobianCache, NonAffineMappingComputesJacobianPerQuadraturePoint) {
    CountingNonAffineLineMapping map;
    quadrature::GaussQuadrature1D quad(4);
    auto& cache = JacobianCache::instance();
    cache.clear();

    const auto& entry = cache.get_or_compute(map, quad);
    ASSERT_EQ(entry.J.size(), quad.num_points());
    EXPECT_EQ(map.jacobian_calls, static_cast<int>(quad.num_points()));
}

TEST(JacobianCache, QuadIdentityEntriesAndInverse) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 1);
    auto nodes = basis->nodes();
    IsoparametricMapping map(basis, nodes);
    quadrature::QuadrilateralQuadrature quad(2);

    auto& cache = JacobianCache::instance();
    cache.clear();
    const auto& entry = cache.get_or_compute(map, quad);

    ASSERT_EQ(entry.J.size(), quad.num_points());
    for (std::size_t i = 0; i < quad.num_points(); ++i) {
        auto J = entry.J[i];
        auto Jinv = entry.J_inv[i];
        auto I = J * Jinv;
        EXPECT_NEAR(entry.detJ[i], 1.0, 1e-12);
        EXPECT_NEAR(I(0,0), 1.0, 1e-12);
        EXPECT_NEAR(I(1,1), 1.0, 1e-12);
    }

    cache.clear();
    EXPECT_EQ(cache.size(), 0u);
}

TEST(JacobianCache, HexIdentityEntriesAndInverse) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Hex8, 1);
    auto nodes = basis->nodes();
    IsoparametricMapping map(basis, nodes);
    quadrature::HexahedronQuadrature quad(2);

    auto& cache = JacobianCache::instance();
    cache.clear();
    const auto& entry = cache.get_or_compute(map, quad);

    ASSERT_EQ(entry.J.size(), quad.num_points());
    for (std::size_t i = 0; i < quad.num_points(); ++i) {
        auto J = entry.J[i];
        auto Jinv = entry.J_inv[i];
        auto I = J * Jinv;
        EXPECT_NEAR(entry.detJ[i], 1.0, 1e-12);
        EXPECT_NEAR(I(0,0), 1.0, 1e-12);
        EXPECT_NEAR(I(1,1), 1.0, 1e-12);
        EXPECT_NEAR(I(2,2), 1.0, 1e-12);
    }
}

TEST(JacobianCache, HexReuseAcrossThreads) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Hex8, 1);
    auto nodes = basis->nodes();
    IsoparametricMapping map(basis, nodes);
    quadrature::HexahedronQuadrature quad(2);

    auto& cache = JacobianCache::instance();
    cache.clear();

    std::array<const JacobianCacheEntry*, 4> entries{};
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t]() {
            const auto& e = cache.get_or_compute(map, quad);
            entries[static_cast<std::size_t>(t)] = &e;
        });
    }
    for (auto& th : threads) th.join();

    const auto* ref = entries[0];
    for (int t = 1; t < 4; ++t) {
        EXPECT_EQ(entries[static_cast<std::size_t>(t)], ref);
    }
    EXPECT_EQ(cache.size(), 1u);
    ASSERT_EQ(ref->J.size(), quad.num_points());
    EXPECT_NEAR(ref->detJ.front(), 1.0, 1e-12);
}

TEST(JacobianCache, Quad8SerendipityEntries) {
    auto basis = std::make_shared<basis::SerendipityBasis>(ElementType::Quad8, 2);
    std::vector<math::Vector<Real,3>> nodes = {
        {Real(-1), Real(-1), Real(0)},
        {Real(1),  Real(-1), Real(0)},
        {Real(1),  Real(1),  Real(0)},
        {Real(-1), Real(1),  Real(0)},
        {Real(0),  Real(-1), Real(0)},
        {Real(1),  Real(0),  Real(0)},
        {Real(0),  Real(1),  Real(0)},
        {Real(-1), Real(0),  Real(0)},
    };
    IsoparametricMapping map(basis, nodes);
    quadrature::QuadrilateralQuadrature quad(3);
    auto& cache = JacobianCache::instance();
    cache.clear();
    const auto& entry = cache.get_or_compute(map, quad);
    ASSERT_EQ(entry.detJ.size(), quad.num_points());
}

TEST(JacobianCache, Hex20SerendipityEntries) {
    // Geometry-mode Hex20 basis for robust mapping
    auto basis = std::make_shared<basis::SerendipityBasis>(ElementType::Hex20, 2, true);
    std::vector<math::Vector<Real,3>> nodes = {
        {Real(-1), Real(-1), Real(-1)},
        {Real(1),  Real(-1), Real(-1)},
        {Real(1),  Real(1),  Real(-1)},
        {Real(-1), Real(1),  Real(-1)},
        {Real(-1), Real(-1), Real(1)},
        {Real(1),  Real(-1), Real(1)},
        {Real(1),  Real(1),  Real(1)},
        {Real(-1), Real(1),  Real(1)},
        {Real(0), Real(-1), Real(-1)},
        {Real(0), Real(-1), Real(1)},
        {Real(0), Real(1),  Real(-1)},
        {Real(0), Real(1),  Real(1)},
        {Real(-1), Real(0), Real(-1)},
        {Real(1),  Real(0), Real(-1)},
        {Real(-1), Real(0), Real(1)},
        {Real(1),  Real(0), Real(1)},
        {Real(-1), Real(-1), Real(0)},
        {Real(1),  Real(-1), Real(0)},
        {Real(-1), Real(1),  Real(0)},
        {Real(1),  Real(1),  Real(0)},
    };
    IsoparametricMapping map(basis, nodes);
    quadrature::HexahedronQuadrature quad(2);
    auto& cache = JacobianCache::instance();
    cache.clear();
    const auto& entry = cache.get_or_compute(map, quad);
    ASSERT_EQ(entry.detJ.size(), quad.num_points());
}

TEST(JacobianCache, WedgeIdentityEntries) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Wedge6, 1);
    auto nodes = basis->nodes();
    IsoparametricMapping map(basis, nodes);
    quadrature::WedgeQuadrature quad(2);

    auto& cache = JacobianCache::instance();
    cache.clear();
    const auto& entry = cache.get_or_compute(map, quad);
    ASSERT_EQ(entry.detJ.size(), quad.num_points());
    for (Real d : entry.detJ) {
        EXPECT_GT(d, 0.0);
    }
}

TEST(JacobianCache, PyramidIdentityEntries) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Pyramid5, 1);
    auto nodes = basis->nodes();
    IsoparametricMapping map(basis, nodes);
    quadrature::PyramidQuadrature quad(2);

    auto& cache = JacobianCache::instance();
    cache.clear();
    const auto& entry = cache.get_or_compute(map, quad);
    ASSERT_EQ(entry.detJ.size(), quad.num_points());
    for (Real d : entry.detJ) {
        EXPECT_GT(d, 0.0);
    }
}

TEST(GeometryQuadrature, ScalesWeightsByDeterminant) {
    std::vector<math::Vector<Real, 3>> nodes = {
        {Real(0), Real(0), Real(0)},
        {Real(3), Real(0), Real(0)}
    };
    LinearMapping map(ElementType::Line2, nodes);
    quadrature::GaussQuadrature1D quad(3);

    auto data = GeometryQuadrature::evaluate(map, quad);
    ASSERT_EQ(data.scaled_weights.size(), quad.num_points());
    for (std::size_t i = 0; i < quad.num_points(); ++i) {
        EXPECT_NEAR(data.scaled_weights[i], quad.weight(i) * 1.5, 1e-12); // J=length/2 = 1.5
    }
}

TEST(GeometryQuadrature, TriangleAreaFromScaledWeights) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Triangle3, 1);
    // Physical triangle with nodes matching basis ordering: area 0.5
    std::vector<math::Vector<Real,3>> nodes = {
        {Real(0), Real(0), Real(0)},
        {Real(1), Real(0), Real(0)},
        {Real(0), Real(1), Real(0)}
    };
    IsoparametricMapping map(basis, nodes);
    quadrature::TriangleQuadrature quad(2);

    auto data = GeometryQuadrature::evaluate(map, quad);
    double sum = 0.0;
    for (Real w : data.scaled_weights) sum += w;
    EXPECT_NEAR(sum, 0.5, 1e-12);
}

TEST(GeometryQuadrature, HexVolumeFromScaledWeights) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Hex8, 1);
    auto nodes = basis->nodes();
    IsoparametricMapping map(basis, nodes);
    quadrature::HexahedronQuadrature quad(2);

    auto data = GeometryQuadrature::evaluate(map, quad);
    double sum = 0.0;
    for (Real w : data.scaled_weights) sum += w;
    EXPECT_NEAR(sum, 8.0, 1e-10); // volume of reference hex [-1,1]^3
}

TEST(GeometryQuadrature, TetraVolumeFromScaledWeights) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Tetra4, 1);
    std::vector<math::Vector<Real,3>> nodes = {
        {Real(0), Real(0), Real(0)},
        {Real(1), Real(0), Real(0)},
        {Real(0), Real(1), Real(0)},
        {Real(0), Real(0), Real(1)}
    };
    IsoparametricMapping map(basis, nodes);
    quadrature::TetrahedronQuadrature quad(2);

    auto data = GeometryQuadrature::evaluate(map, quad);
    double sum = 0.0;
    for (Real w : data.scaled_weights) sum += w;
    EXPECT_NEAR(sum, 1.0/6.0, 1e-12);
}

TEST(GeometryQuadrature, TetraVolumePositiveWithReversedOrientation) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Tetra4, 1);
    // Same tetrahedron geometry as the reference, but with swapped node ordering.
    // The signed determinant is negative, but integration weights should remain positive.
    std::vector<math::Vector<Real,3>> nodes = {
        {Real(0), Real(0), Real(0)},
        {Real(0), Real(1), Real(0)}, // swapped
        {Real(1), Real(0), Real(0)}, // swapped
        {Real(0), Real(0), Real(1)}
    };
    IsoparametricMapping map(basis, nodes);
    quadrature::TetrahedronQuadrature quad(2);

    auto data = GeometryQuadrature::evaluate(map, quad);
    EXPECT_LT(data.detJ.front(), 0.0);
    double sum = 0.0;
    for (Real w : data.scaled_weights) sum += w;
    EXPECT_NEAR(sum, 1.0/6.0, 1e-12);
}

TEST(GeometryQuadrature, WedgeVolumeFromScaledWeights) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Wedge6, 1);
    auto nodes = basis->nodes();
    IsoparametricMapping map(basis, nodes);
    quadrature::WedgeQuadrature quad(2);

    auto data = GeometryQuadrature::evaluate(map, quad);
    double sum = 0.0;
    for (Real w : data.scaled_weights) sum += w;
    EXPECT_NEAR(sum, 1.0, 1e-10); // reference wedge volume 1 (triangle area 0.5 * length 2)
}

TEST(GeometryQuadrature, PyramidVolumeFromScaledWeights) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Pyramid5, 1);
    auto nodes = basis->nodes();
    IsoparametricMapping map(basis, nodes);
    quadrature::PyramidQuadrature quad(2);

    auto data = GeometryQuadrature::evaluate(map, quad);
    double sum = 0.0;
    for (Real w : data.scaled_weights) sum += w;
    EXPECT_NEAR(sum, 4.0/3.0, 1e-8);
}

TEST(GeometryQuadrature, Quad8SerendipityArea) {
    auto basis = std::make_shared<basis::SerendipityBasis>(ElementType::Quad8, 2);
    std::vector<math::Vector<Real,3>> nodes = {
        {Real(-1), Real(-1), Real(0)},
        {Real(1),  Real(-1), Real(0)},
        {Real(1),  Real(1),  Real(0)},
        {Real(-1), Real(1),  Real(0)},
        {Real(0),  Real(-1), Real(0)},
        {Real(1),  Real(0),  Real(0)},
        {Real(0),  Real(1),  Real(0)},
        {Real(-1), Real(0),  Real(0)},
    };
    IsoparametricMapping map(basis, nodes);
    quadrature::QuadrilateralQuadrature quad(3);
    auto data = GeometryQuadrature::evaluate(map, quad);
    double sum = 0.0;
    for (Real w : data.scaled_weights) sum += w;
    EXPECT_NEAR(sum, 4.0, 1e-8);
}

TEST(GeometryQuadrature, TiltedQuadAreaFromScaledWeights) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 1);
    // Reference square mapped to the plane z = x + y.
    std::vector<math::Vector<Real, 3>> nodes = {
        {Real(-1), Real(-1), Real(-2)},
        {Real(1),  Real(-1), Real(0)},
        {Real(1),  Real(1),  Real(2)},
        {Real(-1), Real(1),  Real(0)}
    };
    IsoparametricMapping map(basis, nodes);
    quadrature::QuadrilateralQuadrature quad(2);

    auto data = GeometryQuadrature::evaluate(map, quad);
    double sum = 0.0;
    for (Real w : data.scaled_weights) sum += w;
    EXPECT_NEAR(sum, 4.0 * std::sqrt(3.0), 1e-10);
}

TEST(GeometryQuadrature, Hex20SerendipityVolume) {
    // Geometry-mode basis: Hex8-style mapping with 20-node interface
    auto basis = std::make_shared<basis::SerendipityBasis>(ElementType::Hex20, 2, true);
    std::vector<math::Vector<Real,3>> nodes = {
        {Real(-1), Real(-1), Real(-1)},
        {Real(1),  Real(-1), Real(-1)},
        {Real(1),  Real(1),  Real(-1)},
        {Real(-1), Real(1),  Real(-1)},
        {Real(-1), Real(-1), Real(1)},
        {Real(1),  Real(-1), Real(1)},
        {Real(1),  Real(1),  Real(1)},
        {Real(-1), Real(1),  Real(1)},
        {Real(0), Real(-1), Real(-1)},
        {Real(0), Real(1),  Real(-1)},
        {Real(0), Real(-1), Real(1)},
        {Real(0), Real(1),  Real(1)},
        {Real(-1), Real(0), Real(-1)},
        {Real(1),  Real(0), Real(-1)},
        {Real(-1), Real(0), Real(1)},
        {Real(1),  Real(0), Real(1)},
        {Real(-1), Real(-1), Real(0)},
        {Real(1),  Real(-1), Real(0)},
        {Real(1),  Real(1),  Real(0)},
        {Real(-1), Real(1),  Real(0)},
    };
    IsoparametricMapping map(basis, nodes);
    quadrature::HexahedronQuadrature quad(2);
    auto data = GeometryQuadrature::evaluate(map, quad);
    double sum = 0.0;
    for (Real w : data.scaled_weights) sum += w;
    EXPECT_NEAR(sum, 8.0, 1e-8);
}
