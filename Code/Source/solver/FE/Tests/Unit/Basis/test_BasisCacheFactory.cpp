/**
 * @file test_BasisCacheFactory.cpp
 * @brief Tests for basis cache and factory
 */

#include <gtest/gtest.h>
#include "FE/Basis/BasisCache.h"
#include "FE/Basis/BasisFactory.h"
#include "FE/Basis/BSplineBasis.h"
#include "FE/Basis/NURBSTensorBasis.h"
#include "FE/Basis/LagrangeBasis.h"
#include "FE/Basis/TensorBasis.h"
#include "FE/Quadrature/GaussQuadrature.h"
#include "FE/Quadrature/QuadrilateralQuadrature.h"
#include "FE/Quadrature/QuadratureRule.h"
#include <array>
#include <numeric>
#include <thread>
#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::basis;
using namespace svmp::FE::quadrature;

namespace {

std::vector<Real> make_open_uniform_knots(int degree,
                                          int num_basis,
                                          Real u0 = Real(0),
                                          Real u1 = Real(1)) {
    std::vector<Real> knots;
    knots.reserve(static_cast<std::size_t>(num_basis + degree + 1));

    for (int i = 0; i < degree + 1; ++i) {
        knots.push_back(u0);
    }

    const int interior = num_basis - degree - 1;
    for (int j = 1; j <= interior; ++j) {
        knots.push_back(u0 + (u1 - u0) * Real(j) / Real(num_basis - degree));
    }

    for (int i = 0; i < degree + 1; ++i) {
        knots.push_back(u1);
    }

    return knots;
}

class CustomQuadratureRule final : public QuadratureRule {
public:
    CustomQuadratureRule(svmp::CellFamily family,
                         int dimension,
                         int order,
                         std::vector<QuadPoint> points,
                         std::vector<Real> weights)
        : QuadratureRule(family, dimension, order) {
        set_data(std::move(points), std::move(weights));
    }
};

class TestCustomScalarBasis final : public BasisFunction {
public:
    explicit TestCustomScalarBasis(int order)
        : order_(order) {}

    BasisType basis_type() const noexcept override { return BasisType::Custom; }
    ElementType element_type() const noexcept override { return ElementType::Line2; }
    int dimension() const noexcept override { return 1; }
    int order() const noexcept override { return order_; }
    std::size_t size() const noexcept override { return 2u; }

    std::string cache_identity() const override {
        return BasisFunction::cache_identity() + "|test-custom-order=" + std::to_string(order_);
    }

    void evaluate_values(const math::Vector<Real, 3>& xi,
                         std::vector<Real>& values) const override {
        values.resize(2u);
        values[0] = Real(0.5) * (Real(1) - xi[0]);
        values[1] = Real(0.5) * (Real(1) + xi[0]);
    }

    void evaluate_gradients(const math::Vector<Real, 3>&,
                            std::vector<Gradient>& gradients) const override {
        gradients.assign(2u, Gradient{});
        gradients[0][0] = Real(-0.5);
        gradients[1][0] = Real(0.5);
    }

private:
    int order_{1};
};

std::size_t expected_lagrange_size(ElementType type, int order) {
    switch (type) {
        case ElementType::Line2:
        case ElementType::Line3:
            return static_cast<std::size_t>(order + 1);
        case ElementType::Triangle3:
        case ElementType::Triangle6:
            return static_cast<std::size_t>(order + 1) * static_cast<std::size_t>(order + 2) / 2;
        case ElementType::Quad4:
        case ElementType::Quad9:
            return static_cast<std::size_t>(order + 1) * static_cast<std::size_t>(order + 1);
        case ElementType::Tetra4:
        case ElementType::Tetra10:
            return static_cast<std::size_t>(order + 1) *
                   static_cast<std::size_t>(order + 2) *
                   static_cast<std::size_t>(order + 3) / 6;
        case ElementType::Hex8:
        case ElementType::Hex27:
            return static_cast<std::size_t>(order + 1) *
                   static_cast<std::size_t>(order + 1) *
                   static_cast<std::size_t>(order + 1);
        case ElementType::Wedge6:
        case ElementType::Wedge18:
            return static_cast<std::size_t>(order + 1) *
                   static_cast<std::size_t>(order + 1) *
                   static_cast<std::size_t>(order + 2) / 2;
        case ElementType::Pyramid5:
        case ElementType::Pyramid14:
            return static_cast<std::size_t>(order + 1) *
                   static_cast<std::size_t>(order + 2) *
                   static_cast<std::size_t>(2 * order + 3) / 6;
        default:
            return 0u;
    }
}

} // namespace

TEST(BasisCache, ReusesEntries) {
    LagrangeBasis basis(ElementType::Line2, 2);
    GaussQuadrature1D quad(4);

    auto& cache = BasisCache::instance();
    cache.clear();
    const auto& entry1 = cache.get_or_compute(basis, quad, true, false);
    const auto& entry2 = cache.get_or_compute(basis, quad, true, false);

    EXPECT_EQ(&entry1, &entry2);
    EXPECT_EQ(entry1.num_qpts, quad.num_points());
    EXPECT_EQ(entry1.num_dofs, basis.size());
    ASSERT_EQ(entry1.scalar_values.size(), basis.size() * quad.num_points());
}

TEST(BasisCache, VectorBasisPopulatesVectorValues) {
    RaviartThomasBasis basis(ElementType::Quad4, 0);
    QuadrilateralQuadrature quad(2);

    auto& cache = BasisCache::instance();
    cache.clear();
    const auto& entry = cache.get_or_compute(basis, quad, false, false);

    EXPECT_TRUE(entry.scalar_values.empty());
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
    EXPECT_EQ(entry2.num_qpts, quad2.num_points());
    EXPECT_EQ(entry3.num_qpts, quad3.num_points());
}

TEST(BasisCache, EquivalentSplineInstancesReuseEntries) {
    BSplineBasis basis_a(2, make_open_uniform_knots(2, 6));
    BSplineBasis basis_b(2, make_open_uniform_knots(2, 6));
    GaussQuadrature1D quad(4);

    auto& cache = BasisCache::instance();
    cache.clear();
    const auto& entry_a = cache.get_or_compute(basis_a, quad, true, false);
    const auto& entry_b = cache.get_or_compute(basis_b, quad, true, false);

    EXPECT_EQ(&entry_a, &entry_b);
    EXPECT_EQ(cache.size(), 1u);
}

TEST(BasisCache, BSplineDifferentKnotsProduceDifferentEntries) {
    BSplineBasis basis_a(2, {Real(0), Real(0), Real(0), Real(0.25), Real(1), Real(1), Real(1)});
    BSplineBasis basis_b(2, {Real(0), Real(0), Real(0), Real(0.75), Real(1), Real(1), Real(1)});
    GaussQuadrature1D quad(4);

    auto& cache = BasisCache::instance();
    cache.clear();
    const auto& entry_a = cache.get_or_compute(basis_a, quad, true, false);
    const auto& entry_b = cache.get_or_compute(basis_b, quad, true, false);

    EXPECT_NE(&entry_a, &entry_b);
    EXPECT_EQ(cache.size(), 2u);
}

TEST(BasisCache, BSplineDifferentWeightsProduceDifferentEntries) {
    const auto knots = make_open_uniform_knots(2, 5);
    BSplineBasis basis_a(2, knots, {Real(1), Real(0.5), Real(2), Real(0.7), Real(1)});
    BSplineBasis basis_b(2, knots, {Real(1), Real(0.8), Real(2), Real(0.7), Real(1)});
    GaussQuadrature1D quad(4);

    auto& cache = BasisCache::instance();
    cache.clear();
    const auto& entry_a = cache.get_or_compute(basis_a, quad, true, false);
    const auto& entry_b = cache.get_or_compute(basis_b, quad, true, false);

    EXPECT_NE(&entry_a, &entry_b);
    EXPECT_EQ(cache.size(), 2u);
}

TEST(BasisCache, RationalAndNonRationalSplineProduceDifferentEntries) {
    const auto knots = make_open_uniform_knots(2, 5);
    BSplineBasis bspline(2, knots);
    BSplineBasis nurbs(2, knots, {Real(1), Real(0.5), Real(2), Real(0.7), Real(1)});
    GaussQuadrature1D quad(4);

    auto& cache = BasisCache::instance();
    cache.clear();
    const auto& entry_bspline = cache.get_or_compute(bspline, quad, true, false);
    const auto& entry_nurbs = cache.get_or_compute(nurbs, quad, true, false);

    EXPECT_NE(&entry_bspline, &entry_nurbs);
    EXPECT_EQ(cache.size(), 2u);
}

TEST(BasisCache, AnisotropicTensorSplineBasesDoNotAlias) {
    BSplineBasis bx_a(1, make_open_uniform_knots(1, 4));
    BSplineBasis bx_b(1, {Real(0), Real(0), Real(0.2), Real(0.8), Real(1), Real(1)});
    BSplineBasis by(2, make_open_uniform_knots(2, 5));
    TensorProductBasis<BSplineBasis> basis_a(bx_a, by);
    TensorProductBasis<BSplineBasis> basis_b(bx_b, by);
    QuadrilateralQuadrature quad(3);

    auto& cache = BasisCache::instance();
    cache.clear();
    const auto& entry_a = cache.get_or_compute(basis_a, quad, true, false);
    const auto& entry_b = cache.get_or_compute(basis_b, quad, true, false);

    EXPECT_NE(&entry_a, &entry_b);
    EXPECT_EQ(cache.size(), 2u);
}

TEST(BasisCache, CachedSplineValuesMatchDirectEvaluation) {
    BSplineBasis basis(2, make_open_uniform_knots(2, 6));
    GaussQuadrature1D quad(5);

    auto& cache = BasisCache::instance();
    cache.clear();
    const auto& entry = cache.get_or_compute(basis, quad, true, false);

    for (std::size_t qp = 0; qp < quad.num_points(); ++qp) {
        std::vector<Real> values;
        std::vector<Gradient> gradients;
        basis.evaluate_values(quad.point(qp), values);
        basis.evaluate_gradients(quad.point(qp), gradients);

        ASSERT_EQ(values.size(), basis.size());
        ASSERT_EQ(gradients.size(), basis.size());
        for (std::size_t dof = 0; dof < basis.size(); ++dof) {
            EXPECT_NEAR(entry.scalarValue(dof, qp), values[dof], 1e-14);
            EXPECT_NEAR(entry.gradients[qp][dof][0], gradients[dof][0], 1e-12);
        }
    }
}

TEST(BasisCache, CachedTensorSplineValuesMatchDirectEvaluation) {
    BSplineBasis bx(1, make_open_uniform_knots(1, 4));
    BSplineBasis by(2, make_open_uniform_knots(2, 5));
    TensorProductBasis<BSplineBasis> basis(bx, by);
    QuadrilateralQuadrature quad(3);

    auto& cache = BasisCache::instance();
    cache.clear();
    const auto& entry = cache.get_or_compute(basis, quad, true, false);

    for (std::size_t qp = 0; qp < quad.num_points(); ++qp) {
        std::vector<Real> values;
        basis.evaluate_values(quad.point(qp), values);

        ASSERT_EQ(values.size(), basis.size());
        for (std::size_t dof = 0; dof < basis.size(); ++dof) {
            EXPECT_NEAR(entry.scalarValue(dof, qp), values[dof], 1e-14);
        }
    }
}

TEST(BasisCache, EquivalentQuadraturePointSetsReuseEntriesEvenIfWeightsDiffer) {
    LagrangeBasis basis(ElementType::Line2, 2);
    const std::vector<QuadPoint> points = {
        QuadPoint{Real(-0.5), Real(0), Real(0)},
        QuadPoint{Real(0.5), Real(0), Real(0)}
    };
    CustomQuadratureRule quad_a(svmp::CellFamily::Line, 1, 1, points, {Real(1), Real(1)});
    CustomQuadratureRule quad_b(svmp::CellFamily::Line, 1, 99, points, {Real(0.25), Real(1.75)});

    auto& cache = BasisCache::instance();
    cache.clear();
    const auto& entry_a = cache.get_or_compute(basis, quad_a, true, false);
    const auto& entry_b = cache.get_or_compute(basis, quad_b, true, false);

    EXPECT_EQ(&entry_a, &entry_b);
    EXPECT_EQ(cache.size(), 1u);
}

TEST(BasisCache, QuadratureDifferentZCoordinatesProduceDifferentEntries) {
    LagrangeBasis basis(ElementType::Hex8, 1);
    CustomQuadratureRule quad_a(
        svmp::CellFamily::Hex, 3, 1,
        {QuadPoint{Real(0), Real(0), Real(0)}},
        {Real(8)});
    CustomQuadratureRule quad_b(
        svmp::CellFamily::Hex, 3, 1,
        {QuadPoint{Real(0), Real(0), Real(0.25)}},
        {Real(8)});

    auto& cache = BasisCache::instance();
    cache.clear();
    const auto& entry_a = cache.get_or_compute(basis, quad_a, true, false);
    const auto& entry_b = cache.get_or_compute(basis, quad_b, true, false);

    EXPECT_NE(&entry_a, &entry_b);
    EXPECT_EQ(cache.size(), 2u);
}

TEST(BasisCache, QuadratureDifferentTailPointsProduceDifferentEntries) {
    LagrangeBasis basis(ElementType::Line2, 2);
    const std::vector<Real> weights(5, Real(0.4));
    CustomQuadratureRule quad_a(
        svmp::CellFamily::Line, 1, 5,
        {
            QuadPoint{Real(-0.8), Real(0), Real(0)},
            QuadPoint{Real(-0.4), Real(0), Real(0)},
            QuadPoint{Real(0.0), Real(0), Real(0)},
            QuadPoint{Real(0.4), Real(0), Real(0)},
            QuadPoint{Real(0.8), Real(0), Real(0)}
        },
        weights);
    CustomQuadratureRule quad_b(
        svmp::CellFamily::Line, 1, 5,
        {
            QuadPoint{Real(-0.8), Real(0), Real(0)},
            QuadPoint{Real(-0.4), Real(0), Real(0)},
            QuadPoint{Real(0.0), Real(0), Real(0)},
            QuadPoint{Real(0.4), Real(0), Real(0)},
            QuadPoint{Real(0.9), Real(0), Real(0)}
        },
        weights);

    auto& cache = BasisCache::instance();
    cache.clear();
    const auto& entry_a = cache.get_or_compute(basis, quad_a, true, false);
    const auto& entry_b = cache.get_or_compute(basis, quad_b, true, false);

    EXPECT_NE(&entry_a, &entry_b);
    EXPECT_EQ(cache.size(), 2u);
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
    ASSERT_EQ(ref->num_qpts, quad.num_points());
}

TEST(BasisCache, CanonicalAndAliasLagrangePathsReuseEntries) {
    LagrangeBasis alias(ElementType::Pyramid14, 2);
    LagrangeBasis canonical(ElementType::Pyramid5, 2);
    CustomQuadratureRule quad(
        svmp::CellFamily::Pyramid, 3, 4,
        {
            QuadPoint{Real(0.0), Real(0.0), Real(0.2)},
            QuadPoint{Real(0.1), Real(-0.08), Real(0.35)},
            QuadPoint{Real(-0.12), Real(0.1), Real(0.5)}
        },
        {Real(0.5), Real(0.4), Real(0.4333333333333333)});

    auto& cache = BasisCache::instance();
    cache.clear();
    const auto& entry_alias = cache.get_or_compute(alias, quad, true, false);
    const auto& entry_canonical = cache.get_or_compute(canonical, quad, true, false);

    EXPECT_EQ(&entry_alias, &entry_canonical);
    EXPECT_EQ(cache.size(), 1u);
}

TEST(BasisCache, CachedHighOrderLagrangeEvaluationsMatchDirectEvaluation) {
    LagrangeBasis basis(ElementType::Pyramid5, 4);
    CustomQuadratureRule quad(
        svmp::CellFamily::Pyramid, 3, 6,
        {
            QuadPoint{Real(0.0), Real(0.0), Real(0.15)},
            QuadPoint{Real(0.18), Real(-0.12), Real(0.3)},
            QuadPoint{Real(-0.2), Real(0.1), Real(0.42)},
            QuadPoint{Real(0.04), Real(-0.03), Real(0.78)}
        },
        {Real(0.2), Real(0.3), Real(0.4), Real(0.4333333333333333)});

    auto& cache = BasisCache::instance();
    cache.clear();
    const auto& entry_grad = cache.get_or_compute(basis, quad, true, false);
    const auto& entry_hess = cache.get_or_compute(basis, quad, false, true);

    for (std::size_t qp = 0; qp < quad.num_points(); ++qp) {
        std::vector<Real> values;
        std::vector<Gradient> gradients;
        std::vector<Hessian> hessians;
        basis.evaluate_values(quad.point(qp), values);
        basis.evaluate_gradients(quad.point(qp), gradients);
        basis.evaluate_hessians(quad.point(qp), hessians);

        ASSERT_EQ(values.size(), basis.size());
        ASSERT_EQ(gradients.size(), basis.size());
        ASSERT_EQ(hessians.size(), basis.size());
        for (std::size_t dof = 0; dof < basis.size(); ++dof) {
            EXPECT_NEAR(entry_grad.scalarValue(dof, qp), values[dof], 1e-12);
            for (int d = 0; d < basis.dimension(); ++d) {
                const std::size_t sd = static_cast<std::size_t>(d);
                EXPECT_NEAR(entry_grad.gradients[qp][dof][sd], gradients[dof][sd], 1e-10);
                for (int e = 0; e < basis.dimension(); ++e) {
                    const std::size_t se = static_cast<std::size_t>(e);
                    EXPECT_NEAR(entry_hess.hessians[qp][dof](sd, se), hessians[dof](sd, se), 1e-8);
                }
            }
        }
    }
}

TEST(BasisFactory, CreatesVectorConformingBasis) {
    BasisRequest req{ElementType::Quad4, BasisType::Lagrange, 0, Continuity::H_div, FieldType::Vector};
    auto basis = BasisFactory::create(req);
    EXPECT_TRUE(basis->is_vector_valued());
    EXPECT_EQ(basis->element_type(), ElementType::Quad4);
}

TEST(BasisFactory, DefaultHDivOrderOneOnTwoDimensionalCellsUsesBDM) {
    BasisRequest quad_req{ElementType::Quad4, BasisType::Lagrange, 1, Continuity::H_div, FieldType::Vector};
    auto quad_basis = BasisFactory::create(quad_req);
    EXPECT_EQ(quad_basis->basis_type(), BasisType::BDM);

    BasisRequest tri_req{ElementType::Triangle3, BasisType::Lagrange, 1, Continuity::H_div, FieldType::Vector};
    auto tri_basis = BasisFactory::create(tri_req);
    EXPECT_EQ(tri_basis->basis_type(), BasisType::BDM);
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

TEST(BasisFactory, NegativeHDivOrderThrowsInvalidArgumentBeforeConstruction) {
    BasisRequest req{ElementType::Quad4, BasisType::Lagrange, -1, Continuity::H_div, FieldType::Vector};

    try {
        (void)BasisFactory::create(req);
        FAIL() << "Expected FEException";
    } catch (const FEException& e) {
        EXPECT_EQ(e.status(), FEStatus::InvalidArgument);
    }
}

TEST(BasisFactory, NegativeHCurlOrderThrowsInvalidArgumentBeforeConstruction) {
    BasisRequest req{ElementType::Quad4, BasisType::Lagrange, -1, Continuity::H_curl, FieldType::Vector};

    try {
        (void)BasisFactory::create(req);
        FAIL() << "Expected FEException";
    } catch (const FEException& e) {
        EXPECT_EQ(e.status(), FEStatus::InvalidArgument);
    }
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
    {
        BasisRequest req{ElementType::Line2, BasisType::BSpline, 2, Continuity::C0, FieldType::Scalar};
        req.knot_vector = make_open_uniform_knots(2, 6);
        auto basis = BasisFactory::create(req);
        EXPECT_EQ(basis->basis_type(), BasisType::BSpline);
        EXPECT_EQ(basis->size(), 6u);
    }
    {
        BasisRequest req{ElementType::Line2, BasisType::NURBS, 2, Continuity::C0, FieldType::Scalar};
        req.knot_vector = make_open_uniform_knots(2, 5);
        req.weights = {Real(1), Real(0.5), Real(2), Real(0.7), Real(1)};
        auto basis = BasisFactory::create(req);
        EXPECT_EQ(basis->basis_type(), BasisType::NURBS);
        auto* spline = dynamic_cast<BSplineBasis*>(basis.get());
        ASSERT_NE(spline, nullptr);
        EXPECT_TRUE(spline->is_rational());
    }
}

TEST(BasisFactory, CreatesHighOrderCanonicalLagrangeBases) {
    const struct Case {
        ElementType type;
        int order;
    } cases[] = {
        {ElementType::Line2, 5},
        {ElementType::Triangle3, 4},
        {ElementType::Quad4, 5},
        {ElementType::Tetra4, 3},
        {ElementType::Hex8, 4},
        {ElementType::Wedge6, 4},
        {ElementType::Pyramid5, 4},
    };

    for (const auto& c : cases) {
        BasisRequest req{c.type, BasisType::Lagrange, c.order, Continuity::C0, FieldType::Scalar};
        auto basis = BasisFactory::create(req);
        ASSERT_NE(basis, nullptr);
        EXPECT_EQ(basis->basis_type(), BasisType::Lagrange);
        EXPECT_EQ(basis->element_type(), c.type);
        EXPECT_EQ(basis->order(), c.order);
        EXPECT_EQ(basis->size(), expected_lagrange_size(c.type, c.order));
    }
}

TEST(BasisFactory, LowOrderAliasLagrangeDescriptorsMatchCanonicalPaths) {
    const struct Case {
        ElementType alias_type;
        ElementType canonical_type;
        int order;
        math::Vector<Real, 3> xi;
    } cases[] = {
        {ElementType::Line3, ElementType::Line2, 2, {Real(0.2), Real(0), Real(0)}},
        {ElementType::Triangle6, ElementType::Triangle3, 2, {Real(0.2), Real(0.25), Real(0)}},
        {ElementType::Quad9, ElementType::Quad4, 2, {Real(0.15), Real(-0.2), Real(0)}},
        {ElementType::Tetra10, ElementType::Tetra4, 2, {Real(0.15), Real(0.2), Real(0.1)}},
        {ElementType::Hex27, ElementType::Hex8, 2, {Real(0.2), Real(-0.2), Real(0.25)}},
        {ElementType::Wedge18, ElementType::Wedge6, 2, {Real(0.2), Real(0.15), Real(0.1)}},
        {ElementType::Pyramid14, ElementType::Pyramid5, 2, {Real(0.08), Real(-0.06), Real(0.35)}},
    };

    for (const auto& c : cases) {
        BasisRequest alias_req{c.alias_type, BasisType::Lagrange, c.order, Continuity::C0, FieldType::Scalar};
        BasisRequest canonical_req{c.canonical_type, BasisType::Lagrange, c.order, Continuity::C0, FieldType::Scalar};
        auto alias_basis = BasisFactory::create(alias_req);
        auto canonical_basis = BasisFactory::create(canonical_req);

        ASSERT_NE(alias_basis, nullptr);
        ASSERT_NE(canonical_basis, nullptr);
        EXPECT_EQ(alias_basis->element_type(), canonical_basis->element_type());
        EXPECT_EQ(alias_basis->order(), canonical_basis->order());
        EXPECT_EQ(alias_basis->size(), canonical_basis->size());

        std::vector<Real> alias_values;
        std::vector<Real> canonical_values;
        alias_basis->evaluate_values(c.xi, alias_values);
        canonical_basis->evaluate_values(c.xi, canonical_values);

        ASSERT_EQ(alias_values.size(), canonical_values.size());
        for (std::size_t i = 0; i < alias_values.size(); ++i) {
            EXPECT_NEAR(alias_values[i], canonical_values[i], 1e-12);
        }
    }
}

TEST(BasisFactory, UnsupportedCombinationThrows) {
    BasisRequest req{ElementType::Tetra4, BasisType::Serendipity, 1, Continuity::C0, FieldType::Scalar};
    EXPECT_THROW(auto b = BasisFactory::create(req), svmp::FE::FEException);
}

TEST(BasisFactory, HigherOrderSerendipityRequestsRemainUnsupported) {
    for (const ElementType type : {ElementType::Quad4, ElementType::Hex8, ElementType::Wedge15, ElementType::Pyramid13}) {
        BasisRequest req{type, BasisType::Serendipity, 3, Continuity::C0, FieldType::Scalar};
        EXPECT_THROW((void)BasisFactory::create(req), svmp::FE::FEException);
    }
}

TEST(BasisFactory, ExplicitBDMRequestsHonorNarrowedScope) {
    {
        BasisRequest req{ElementType::Triangle3, BasisType::BDM, 1, Continuity::H_div, FieldType::Vector};
        auto basis = BasisFactory::create(req);
        EXPECT_EQ(basis->basis_type(), BasisType::BDM);
        EXPECT_EQ(basis->element_type(), ElementType::Triangle3);
    }
    {
        BasisRequest req{ElementType::Quad4, BasisType::BDM, 1, Continuity::H_div, FieldType::Vector};
        auto basis = BasisFactory::create(req);
        EXPECT_EQ(basis->basis_type(), BasisType::BDM);
        EXPECT_EQ(basis->element_type(), ElementType::Quad4);
    }
    {
        BasisRequest req{ElementType::Quad4, BasisType::BDM, 2, Continuity::H_div, FieldType::Vector};
        EXPECT_THROW((void)BasisFactory::create(req), svmp::FE::FEException);
    }
    {
        BasisRequest req{ElementType::Tetra4, BasisType::BDM, 1, Continuity::H_div, FieldType::Vector};
        EXPECT_THROW((void)BasisFactory::create(req), svmp::FE::FEException);
    }
}

TEST(BasisFactory, L2ReturnsExpectedBasis) {
    // L2 (discontinuous) should return the same basis as C0 -- DOF ownership
    // is handled at the Space/Element level, not in the basis shape functions.
    {
        BasisRequest req{ElementType::Triangle3, BasisType::Lagrange, 2, Continuity::L2, FieldType::Scalar};
        auto basis = BasisFactory::create(req);
        EXPECT_EQ(basis->basis_type(), BasisType::Lagrange);
        EXPECT_EQ(basis->element_type(), ElementType::Triangle3);
        EXPECT_EQ(basis->order(), 2);
        EXPECT_EQ(basis->size(), 6u);
    }
    {
        BasisRequest req{ElementType::Quad4, BasisType::Hierarchical, 3, Continuity::L2, FieldType::Scalar};
        auto basis = BasisFactory::create(req);
        EXPECT_EQ(basis->basis_type(), BasisType::Hierarchical);
        EXPECT_EQ(basis->element_type(), ElementType::Quad4);
        EXPECT_EQ(basis->order(), 3);
    }
}

TEST(BasisFactory, CreatesTensorBSplineFromDescriptor) {
    BasisRequest req{ElementType::Quad4, BasisType::BSpline, 2, Continuity::C0, FieldType::Scalar};
    req.knot_vector = make_open_uniform_knots(2, 5);

    auto basis = BasisFactory::create(req);

    EXPECT_EQ(basis->basis_type(), BasisType::BSpline);
    EXPECT_EQ(basis->element_type(), ElementType::Quad4);
    EXPECT_EQ(basis->size(), 25u);
}

TEST(BasisFactory, CreatesAnisotropicTensorBSplineFromDescriptor) {
    BasisRequest req{ElementType::Quad4, BasisType::BSpline, 1, Continuity::C0, FieldType::Scalar};
    req.axis_orders = {1, 2};
    req.axis_knot_vectors = {
        make_open_uniform_knots(1, 4),
        make_open_uniform_knots(2, 5)
    };

    auto basis = BasisFactory::create(req);

    EXPECT_EQ(basis->basis_type(), BasisType::BSpline);
    EXPECT_EQ(basis->element_type(), ElementType::Quad4);
    EXPECT_EQ(basis->size(), 20u);
}

TEST(BasisFactory, MissingSplineParametersThrows) {
    BasisRequest req{ElementType::Line2, BasisType::BSpline, 2, Continuity::C0, FieldType::Scalar};
    EXPECT_THROW(BasisFactory::create(req), svmp::FE::FEException);
}

TEST(BasisFactory, InvalidSplineWeightsThrow) {
    BasisRequest req{ElementType::Line2, BasisType::BSpline, 2, Continuity::C0, FieldType::Scalar};
    req.knot_vector = make_open_uniform_knots(2, 5);
    req.weights = {Real(1), Real(1), Real(1), Real(1), Real(1)};
    EXPECT_THROW(BasisFactory::create(req), svmp::FE::FEException);
}

TEST(BasisFactory, InvalidNURBSWithoutWeightsThrows) {
    BasisRequest req{ElementType::Line2, BasisType::NURBS, 2, Continuity::C0, FieldType::Scalar};
    req.knot_vector = make_open_uniform_knots(2, 5);
    EXPECT_THROW(BasisFactory::create(req), svmp::FE::FEException);
}

TEST(BasisFactory, CreatesMultiDimensionalNURBSFromDescriptor) {
    BasisRequest req{ElementType::Quad4, BasisType::NURBS, 2, Continuity::C0, FieldType::Scalar};
    req.axis_orders = {2, 1};
    req.axis_knot_vectors = {
        make_open_uniform_knots(2, 5),
        make_open_uniform_knots(1, 4)
    };
    req.tensor_extents = {5, 4};
    req.weights.resize(20u, Real(1));
    req.weights[6] = Real(0.75);
    req.weights[13] = Real(1.5);

    auto basis = BasisFactory::create(req);

    ASSERT_NE(basis, nullptr);
    EXPECT_EQ(basis->basis_type(), BasisType::NURBS);
    EXPECT_EQ(basis->element_type(), ElementType::Quad4);
    EXPECT_EQ(basis->size(), 20u);

    std::vector<Real> values;
    basis->evaluate_values({Real(0.23), Real(0.61), Real(0)}, values);
    ASSERT_EQ(values.size(), basis->size());
    EXPECT_NEAR(std::accumulate(values.begin(), values.end(), Real(0)), Real(1), 1e-12);
}

TEST(BasisFactory, FactoryAndDirectTensorNURBSConstructionAgree) {
    BasisRequest req{ElementType::Quad4, BasisType::NURBS, 2, Continuity::C0, FieldType::Scalar};
    req.axis_orders = {2, 1};
    req.axis_knot_vectors = {
        make_open_uniform_knots(2, 5),
        make_open_uniform_knots(1, 4)
    };
    req.tensor_extents = {5, 4};
    req.weights.resize(20u, Real(1));
    req.weights[5] = Real(0.6);
    req.weights[9] = Real(1.4);
    req.weights[17] = Real(0.8);

    auto from_factory = BasisFactory::create(req);
    NURBSTensorBasis direct(
        BSplineBasis(2, req.axis_knot_vectors[0]),
        BSplineBasis(1, req.axis_knot_vectors[1]),
        req.weights,
        req.tensor_extents);

    const math::Vector<Real, 3> xi{Real(0.31), Real(0.42), Real(0)};
    std::vector<Real> factory_values;
    std::vector<Real> direct_values;
    std::vector<Gradient> factory_gradients;
    std::vector<Gradient> direct_gradients;
    from_factory->evaluate_values(xi, factory_values);
    from_factory->evaluate_gradients(xi, factory_gradients);
    direct.evaluate_values(xi, direct_values);
    direct.evaluate_gradients(xi, direct_gradients);

    ASSERT_EQ(factory_values.size(), direct_values.size());
    ASSERT_EQ(factory_gradients.size(), direct_gradients.size());
    for (std::size_t i = 0; i < factory_values.size(); ++i) {
        EXPECT_NEAR(factory_values[i], direct_values[i], 1e-14);
        EXPECT_NEAR(factory_gradients[i][0], direct_gradients[i][0], 1e-12);
        EXPECT_NEAR(factory_gradients[i][1], direct_gradients[i][1], 1e-12);
    }
}

TEST(BasisFactory, FactoryAndDirectSplineConstructionAgree) {
    BasisRequest req{ElementType::Line2, BasisType::BSpline, 2, Continuity::C0, FieldType::Scalar};
    req.knot_vector = make_open_uniform_knots(2, 6);
    auto from_factory = BasisFactory::create(req);
    BSplineBasis direct(2, make_open_uniform_knots(2, 6));

    const math::Vector<Real, 3> xi{Real(0.2), Real(0), Real(0)};
    std::vector<Real> factory_values;
    std::vector<Real> direct_values;
    from_factory->evaluate_values(xi, factory_values);
    direct.evaluate_values(xi, direct_values);

    ASSERT_EQ(factory_values.size(), direct_values.size());
    for (std::size_t i = 0; i < factory_values.size(); ++i) {
        EXPECT_NEAR(factory_values[i], direct_values[i], 1e-14);
    }
}

TEST(BasisFactory, FactoryCreatedSplineCacheIdentityMatchesDirectConstruction) {
    BasisRequest req{ElementType::Line2, BasisType::BSpline, 2, Continuity::C0, FieldType::Scalar};
    req.knot_vector = make_open_uniform_knots(2, 6);
    auto from_factory = BasisFactory::create(req);
    BSplineBasis direct(2, make_open_uniform_knots(2, 6));
    GaussQuadrature1D quad(4);

    auto& cache = BasisCache::instance();
    cache.clear();
    const auto& entry_factory = cache.get_or_compute(*from_factory, quad, true, false);
    const auto& entry_direct = cache.get_or_compute(direct, quad, true, false);

    EXPECT_EQ(&entry_factory, &entry_direct);
    EXPECT_EQ(cache.size(), 1u);
}

TEST(BasisFactory, CreatesRegisteredCustomBasis) {
    BasisFactory::clear_custom_registry_for_tests();

    BasisFactory::register_custom(
        "test-linear",
        [](const BasisRequest& req) -> std::shared_ptr<BasisFunction> {
            return std::make_shared<TestCustomScalarBasis>(req.order.value_or(1));
        });

    BasisRequest req{ElementType::Line2, BasisType::Custom, 3, Continuity::C0, FieldType::Scalar};
    req.custom_id = "test-linear";

    auto basis = BasisFactory::create(req);
    ASSERT_NE(basis, nullptr);
    EXPECT_EQ(basis->basis_type(), BasisType::Custom);
    EXPECT_EQ(basis->element_type(), ElementType::Line2);
    EXPECT_EQ(basis->order(), 3);

    std::vector<Real> values;
    basis->evaluate_values({Real(0.25), Real(0), Real(0)}, values);
    ASSERT_EQ(values.size(), 2u);
    EXPECT_NEAR(values[0] + values[1], Real(1), 1e-12);

    BasisFactory::unregister_custom("test-linear");
    BasisFactory::clear_custom_registry_for_tests();
}

TEST(BasisFactory, DuplicateCustomRegistrationThrows) {
    BasisFactory::clear_custom_registry_for_tests();
    BasisFactory::register_custom(
        "duplicate",
        [](const BasisRequest& req) -> std::shared_ptr<BasisFunction> {
            return std::make_shared<TestCustomScalarBasis>(req.order.value_or(1));
        });

    EXPECT_THROW(
        BasisFactory::register_custom(
            "duplicate",
            [](const BasisRequest& req) -> std::shared_ptr<BasisFunction> {
                return std::make_shared<TestCustomScalarBasis>(req.order.value_or(1));
            }),
        svmp::FE::FEException);

    BasisFactory::clear_custom_registry_for_tests();
}

TEST(BasisFactory, UnknownCustomBasisThrows) {
    BasisFactory::clear_custom_registry_for_tests();

    BasisRequest req{ElementType::Line2, BasisType::Custom, 1, Continuity::C0, FieldType::Scalar};
    req.custom_id = "missing";
    EXPECT_THROW(BasisFactory::create(req), svmp::FE::FEException);
}
