/**
 * @file test_BasisCacheFactory.cpp
 * @brief Tests for basis cache and factory
 */

#include <gtest/gtest.h>
#include "FE/Basis/BasisCache.h"
#include "FE/Basis/BasisFactory.h"
#include "FE/Basis/BSplineBasis.h"
#include "FE/Basis/CompatibleTensorVectorBasis.h"
#include "FE/Basis/NURBSTensorBasis.h"
#include "FE/Basis/LagrangeBasis.h"
#include "FE/Basis/SerendipityBasis.h"
#include "FE/Basis/TensorBasis.h"
#include "FE/Basis/VectorBasis.h"
#include "FE/Quadrature/GaussQuadrature.h"
#include "FE/Quadrature/QuadrilateralQuadrature.h"
#include "FE/Quadrature/QuadratureRule.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
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

CustomQuadratureRule make_pyramid_quadrature_with_apex() {
    return CustomQuadratureRule(
        svmp::CellFamily::Pyramid, 3, 4,
        {
            QuadPoint{Real(0), Real(0), Real(1)},
            QuadPoint{Real(0.08), Real(-0.06), Real(0.35)},
            QuadPoint{Real(-0.12), Real(0.1), Real(0.5)}
        },
        {Real(0.2), Real(0.5), Real(0.6333333333333333)});
}

using SplineTensorBasis = TensorProductBasis<BSplineBasis>;

std::shared_ptr<BasisFunction> make_quad_spline_component(Real interior = Real(0.5)) {
    std::vector<Real> knots{Real(0), Real(0), Real(0), interior, Real(1), Real(1), Real(1)};
    return std::make_shared<SplineTensorBasis>(BSplineBasis(2, knots), BSplineBasis(2, knots));
}

std::shared_ptr<BasisFunction> make_hex_spline_component() {
    const auto knots = make_open_uniform_knots(2, 4);
    return std::make_shared<SplineTensorBasis>(
        BSplineBasis(2, knots),
        BSplineBasis(2, knots),
        BSplineBasis(2, knots));
}

std::shared_ptr<BasisFunction> make_quad_nurbs_component(Real center_weight) {
    const auto knots = make_open_uniform_knots(2, 4);
    std::vector<Real> weights(16u, Real(1));
    weights[5] = center_weight;
    weights[10] = Real(2) - center_weight * Real(0.25);
    return std::make_shared<NURBSTensorBasis>(
        BSplineBasis(2, knots),
        BSplineBasis(2, knots),
        std::move(weights));
}

std::vector<DofAssociation> compatible_associations(
    const std::vector<std::shared_ptr<BasisFunction>>& components) {
    std::size_t size = 0;
    for (const auto& component : components) {
        size += component->size();
    }
    return std::vector<DofAssociation>(size);
}

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

class TaggedCustomScalarBasis final : public BasisFunction {
public:
    explicit TaggedCustomScalarBasis(int tag)
        : tag_(tag) {}

    BasisType basis_type() const noexcept override { return BasisType::Custom; }
    ElementType element_type() const noexcept override { return ElementType::Line2; }
    int dimension() const noexcept override { return 1; }
    int order() const noexcept override { return 1; }
    std::size_t size() const noexcept override { return 2u; }

    std::string cache_identity() const override {
        return BasisFunction::cache_identity() + "|tag=" + std::to_string(tag_);
    }

    void evaluate_values(const math::Vector<Real, 3>& xi,
                         std::vector<Real>& values) const override {
        values.resize(2u);
        const Real shift = Real(tag_) * Real(0.01);
        values[0] = Real(0.5) * (Real(1) - xi[0]) + shift;
        values[1] = Real(0.5) * (Real(1) + xi[0]) - shift;
    }

private:
    int tag_{0};
};

class StructuredIdentityScalarBasis final : public BasisFunction {
public:
    explicit StructuredIdentityScalarBasis(int tag)
        : tag_(tag) {}

    BasisType basis_type() const noexcept override { return BasisType::Custom; }
    ElementType element_type() const noexcept override { return ElementType::Line2; }
    int dimension() const noexcept override { return 1; }
    int order() const noexcept override { return 1; }
    std::size_t size() const noexcept override { return 2u; }

    bool cache_identity_words(std::vector<std::uint64_t>& words) const override {
        words.push_back(0x7374727563746964ULL);
        words.push_back(static_cast<std::uint64_t>(tag_));
        return true;
    }

    std::string cache_identity() const override {
        ++string_identity_calls;
        return BasisFunction::cache_identity() + "|structured-tag=" + std::to_string(tag_);
    }

    void evaluate_values(const math::Vector<Real, 3>& xi,
                         std::vector<Real>& values) const override {
        values.resize(2u);
        const Real shift = Real(tag_) * Real(0.125);
        values[0] = Real(1) - xi[0] + shift;
        values[1] = xi[0] - shift;
    }

    mutable std::size_t string_identity_calls{0};

private:
    int tag_{0};
};

class StructuredIdentityQuadScalarBasis final : public BasisFunction {
public:
    explicit StructuredIdentityQuadScalarBasis(int tag)
        : tag_(tag) {}

    BasisType basis_type() const noexcept override { return BasisType::Custom; }
    ElementType element_type() const noexcept override { return ElementType::Quad4; }
    int dimension() const noexcept override { return 2; }
    int order() const noexcept override { return 1; }
    std::size_t size() const noexcept override { return 1u; }

    bool cache_identity_words(std::vector<std::uint64_t>& words) const override {
        words.push_back(0x7374727175616469ULL);
        words.push_back(static_cast<std::uint64_t>(tag_));
        return true;
    }

    std::string cache_identity() const override {
        ++string_identity_calls;
        return BasisFunction::cache_identity() + "|structured-quad-tag=" + std::to_string(tag_);
    }

    void evaluate_values(const math::Vector<Real, 3>& xi,
                         std::vector<Real>& values) const override {
        values.resize(1u);
        values[0] = Real(1) + Real(tag_) * Real(0.25) + xi[0] - Real(0.5) * xi[1];
    }

    mutable std::size_t string_identity_calls{0};

private:
    int tag_{0};
};

class CountingVectorBasis final : public VectorBasisFunction {
public:
    BasisType basis_type() const noexcept override { return BasisType::Custom; }
    ElementType element_type() const noexcept override { return ElementType::Quad4; }
    int dimension() const noexcept override { return 2; }
    int order() const noexcept override { return 1; }
    std::size_t size() const noexcept override { return 2u; }
    bool cache_identity_is_structural() const noexcept override { return true; }
    bool supports_curl() const noexcept override { return true; }
    bool supports_divergence() const noexcept override { return true; }

    void evaluate_vector_values(const math::Vector<Real, 3>& xi,
                                std::vector<math::Vector<Real, 3>>& values) const override {
        ++value_calls;
        values.resize(2u);
        values[0] = math::Vector<Real, 3>{Real(1) + xi[0], Real(2) + xi[1], Real(3) + xi[2]};
        values[1] = math::Vector<Real, 3>{-xi[0], xi[1], Real(2) * xi[2]};
    }

    void evaluate_vector_jacobians(const math::Vector<Real, 3>&,
                                   std::vector<VectorJacobian>& jacobians) const override {
        ++jacobian_calls;
        jacobians.assign(2u, VectorJacobian{});
        jacobians[0](0, 0) = Real(1);
        jacobians[0](1, 1) = Real(1);
        jacobians[0](2, 2) = Real(1);
        jacobians[1](0, 0) = Real(-1);
        jacobians[1](1, 1) = Real(1);
        jacobians[1](2, 2) = Real(2);
    }

    void evaluate_curl(const math::Vector<Real, 3>&,
                       std::vector<math::Vector<Real, 3>>& curl) const override {
        ++curl_calls;
        curl.assign(2u, math::Vector<Real, 3>{Real(99), Real(99), Real(99)});
    }

    void evaluate_divergence(const math::Vector<Real, 3>&,
                             std::vector<Real>& divergence) const override {
        ++divergence_calls;
        divergence.assign(2u, Real(-99));
    }

    mutable std::size_t value_calls{0};
    mutable std::size_t jacobian_calls{0};
    mutable std::size_t curl_calls{0};
    mutable std::size_t divergence_calls{0};
};

class SlowBatchScalarBasis final : public BasisFunction {
public:
    BasisType basis_type() const noexcept override { return BasisType::Custom; }
    ElementType element_type() const noexcept override { return ElementType::Line2; }
    int dimension() const noexcept override { return 1; }
    int order() const noexcept override { return 1; }
    std::size_t size() const noexcept override { return 2u; }
    bool cache_identity_is_structural() const noexcept override { return true; }

    void evaluate_values(const math::Vector<Real, 3>& xi,
                         std::vector<Real>& values) const override {
        values.resize(2u);
        values[0] = Real(1) - xi[0];
        values[1] = xi[0];
    }

    void evaluate_at_quadrature_points(const std::vector<math::Vector<Real, 3>>& points,
                                       Real* values_out,
                                       Real* gradients_out,
                                       Real* hessians_out) const override {
        ++batch_calls;
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
        for (std::size_t q = 0; q < points.size(); ++q) {
            if (values_out) {
                values_out[q] = Real(1) - points[q][0];
                values_out[points.size() + q] = points[q][0];
            }
            if (gradients_out) {
                gradients_out[q] = Real(-1);
                gradients_out[3u * points.size() + q] = Real(1);
            }
            if (hessians_out) {
                for (std::size_t i = 0; i < 18u * points.size(); ++i) {
                    hessians_out[i] = Real(0);
                }
            }
        }
    }

    void fill_scalar_cache_entry(const std::vector<math::Vector<Real, 3>>& points,
                                 std::size_t output_stride,
                                 Real* values_out,
                                 Real* gradients_out,
                                 Real* hessians_out) const override {
        ++batch_calls;
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
        BasisFunction::fill_scalar_cache_entry(points,
                                               output_stride,
                                               values_out,
                                               gradients_out,
                                               hessians_out);
    }

    mutable std::atomic<std::size_t> batch_calls{0};
};

class BlockingBatchScalarBasis final : public BasisFunction {
public:
    BasisType basis_type() const noexcept override { return BasisType::Custom; }
    ElementType element_type() const noexcept override { return ElementType::Line2; }
    int dimension() const noexcept override { return 1; }
    int order() const noexcept override { return 1; }
    std::size_t size() const noexcept override { return 2u; }
    bool cache_identity_is_structural() const noexcept override { return true; }

    void evaluate_values(const math::Vector<Real, 3>& xi,
                         std::vector<Real>& values) const override {
        values.resize(2u);
        values[0] = Real(1) - xi[0];
        values[1] = xi[0];
    }

    void fill_scalar_cache_entry(const std::vector<math::Vector<Real, 3>>& points,
                                 std::size_t output_stride,
                                 Real* values_out,
                                 Real* gradients_out,
                                 Real* hessians_out) const override {
        ++batch_calls;
        entered.store(true, std::memory_order_release);
        while (!release.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        BasisFunction::fill_scalar_cache_entry(points,
                                               output_stride,
                                               values_out,
                                               gradients_out,
                                               hessians_out);
    }

    mutable std::atomic<std::size_t> batch_calls{0};
    mutable std::atomic<bool> entered{false};
    mutable std::atomic<bool> release{false};
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

TEST(BasisCache, PrewarmPopulatesReusableEntry) {
    LagrangeBasis basis(ElementType::Line2, 2);
    GaussQuadrature1D quad(4);

    auto& cache = BasisCache::instance();
    cache.clear();
    const auto& warmed = cache.prewarm(basis, quad, true, false);
    const auto& reused = cache.get_or_compute(basis, quad, true, false);

    EXPECT_EQ(&warmed, &reused);
    EXPECT_EQ(cache.size(), 1u);
}

TEST(BasisCache, PrewarmHandleMatchesReusableEntry) {
    LagrangeBasis basis(ElementType::Line2, 2);
    GaussQuadrature1D quad(4);

    auto& cache = BasisCache::instance();
    cache.clear();
    auto handle = cache.prewarm_handle(basis, quad, true, false);
    const auto& reused = cache.get_or_compute(basis, quad, true, false);

    ASSERT_TRUE(handle.valid());
    EXPECT_EQ(&handle.entry(), &reused);
    EXPECT_EQ(cache.size(), 1u);
}

TEST(BasisCache, PrewarmHandleKeepsEntryAliveAfterClear) {
    LagrangeBasis basis(ElementType::Line2, 2);
    GaussQuadrature1D quad(4);

    auto& cache = BasisCache::instance();
    cache.clear();
    auto handle = cache.prewarm_handle(basis, quad, true, false);
    const auto* entry_before_clear = &handle.entry();

    cache.clear();

    EXPECT_EQ(cache.size(), 0u);
    EXPECT_EQ(&handle.entry(), entry_before_clear);
    EXPECT_EQ(handle.entry().num_qpts, quad.num_points());
    EXPECT_EQ(handle.entry().num_dofs, basis.size());
}

TEST(BasisCache, ConcurrentColdMissComputesKeyOnce) {
    SlowBatchScalarBasis basis;
    GaussQuadrature1D quad(4);

    auto& cache = BasisCache::instance();
    cache.clear();

    constexpr std::size_t num_threads = 8u;
    std::vector<const BasisCacheEntry*> entries(num_threads, nullptr);
    std::vector<std::thread> workers;
    workers.reserve(num_threads);
    for (std::size_t i = 0; i < num_threads; ++i) {
        workers.emplace_back([&, i] {
            entries[i] = &cache.get_or_compute(basis, quad, false, false);
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }

    EXPECT_EQ(basis.batch_calls.load(), 1u);
    ASSERT_NE(entries.front(), nullptr);
    for (const auto* entry : entries) {
        EXPECT_EQ(entry, entries.front());
    }
    EXPECT_EQ(cache.size(), 1u);
}

TEST(BasisCache, ConcurrentPrewarmHandleComputesKeyOnce) {
    SlowBatchScalarBasis basis;
    GaussQuadrature1D quad(4);

    auto& cache = BasisCache::instance();
    cache.clear();

    constexpr std::size_t num_threads = 8u;
    std::vector<BasisCacheHandle> handles(num_threads);
    std::vector<std::thread> workers;
    workers.reserve(num_threads);
    for (std::size_t i = 0; i < num_threads; ++i) {
        workers.emplace_back([&, i] {
            handles[i] = cache.prewarm_handle(basis, quad, false, false);
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }

    EXPECT_EQ(basis.batch_calls.load(), 1u);
    ASSERT_TRUE(handles.front().valid());
    const auto* first_entry = &handles.front().entry();
    for (const auto& handle : handles) {
        ASSERT_TRUE(handle.valid());
        EXPECT_EQ(&handle.entry(), first_entry);
    }
    EXPECT_EQ(cache.size(), 1u);
}

TEST(BasisCache, ClearKeepsInFlightComputationPublishable) {
    BlockingBatchScalarBasis basis;
    GaussQuadrature1D quad(4);

    auto& cache = BasisCache::instance();
    cache.clear();

    const BasisCacheEntry* entry = nullptr;
    std::thread worker([&] {
        entry = &cache.get_or_compute(basis, quad, false, false);
    });

    while (!basis.entered.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    cache.clear();
    EXPECT_EQ(cache.size(), 0u);

    basis.release.store(true, std::memory_order_release);
    worker.join();

    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->num_qpts, quad.num_points());
    EXPECT_EQ(entry->num_dofs, basis.size());
    EXPECT_EQ(basis.batch_calls.load(), 1u);
    EXPECT_EQ(cache.size(), 1u);
}

TEST(BasisCache, UsesStructuralKeysStructuredPayloadsAndStringFallbacks) {
    LagrangeBasis lagrange(ElementType::Line2, 2);
    RaviartThomasBasis rt(ElementType::Quad4, 0);
    BSplineBasis spline(2, make_open_uniform_knots(2, 5));
    TaggedCustomScalarBasis custom_a(1);
    TaggedCustomScalarBasis custom_b(2);
    GaussQuadrature1D quad(3);
    std::vector<std::uint64_t> spline_words;

    EXPECT_TRUE(lagrange.cache_identity_is_structural());
    EXPECT_TRUE(rt.cache_identity_is_structural());
    EXPECT_FALSE(spline.cache_identity_is_structural());
    EXPECT_TRUE(spline.cache_identity_words(spline_words));
    EXPECT_FALSE(spline_words.empty());
    EXPECT_FALSE(custom_a.cache_identity_is_structural());

    auto& cache = BasisCache::instance();
    cache.clear();
    const auto& entry_a = cache.get_or_compute(custom_a, quad, false, false);
    const auto& entry_b = cache.get_or_compute(custom_b, quad, false, false);
    EXPECT_NE(&entry_a, &entry_b);
    ASSERT_EQ(cache.size(), 2u);
    EXPECT_NE(entry_a.scalar_values, entry_b.scalar_values);
}

TEST(BasisCache, StructuredIdentityAvoidsStringFallbackAndSeparatesCustomBases) {
    StructuredIdentityScalarBasis custom_a(1);
    StructuredIdentityScalarBasis custom_b(2);
    GaussQuadrature1D quad(3);

    auto& cache = BasisCache::instance();
    cache.clear();
    const auto& entry_a = cache.get_or_compute(custom_a, quad, false, false);
    const auto& entry_a_again = cache.get_or_compute(custom_a, quad, false, false);
    const auto& entry_b = cache.get_or_compute(custom_b, quad, false, false);

    EXPECT_EQ(&entry_a, &entry_a_again);
    EXPECT_NE(&entry_a, &entry_b);
    EXPECT_EQ(custom_a.string_identity_calls, 0u);
    EXPECT_EQ(custom_b.string_identity_calls, 0u);
    EXPECT_NE(entry_a.scalar_values, entry_b.scalar_values);
}

TEST(BasisCache, TensorProductStructuredIdentityAvoidsAxisStringFallback) {
    StructuredIdentityScalarBasis bx(1);
    StructuredIdentityScalarBasis by(2);
    TensorProductBasis<StructuredIdentityScalarBasis> basis(bx, by);
    QuadrilateralQuadrature quad(2);

    auto& cache = BasisCache::instance();
    cache.clear();
    const auto& entry = cache.get_or_compute(basis, quad, false, false);
    const auto& entry_again = cache.get_or_compute(basis, quad, false, false);

    EXPECT_EQ(&entry, &entry_again);
    EXPECT_EQ(basis.axis_basis(0).string_identity_calls, 0u);
    EXPECT_EQ(basis.axis_basis(1).string_identity_calls, 0u);
    ASSERT_EQ(entry.scalar_values.size(), basis.size() * quad.num_points());
}

TEST(BasisCache, CompatibleTensorVectorStructuredIdentityAvoidsComponentStringFallback) {
    auto bx = std::make_shared<StructuredIdentityQuadScalarBasis>(1);
    auto by = std::make_shared<StructuredIdentityQuadScalarBasis>(2);
    const std::vector<DofAssociation> associations(2u);
    CompatibleTensorVectorBasis basis(CompatibleTensorVectorBasis::Family::HDiv,
                                      BasisType::Custom,
                                      bx,
                                      by,
                                      associations,
                                      1);
    const auto bx_identity_calls_after_construction = bx->string_identity_calls;
    const auto by_identity_calls_after_construction = by->string_identity_calls;
    QuadrilateralQuadrature quad(2);

    auto& cache = BasisCache::instance();
    cache.clear();
    const auto& entry = cache.get_or_compute(basis, quad, false, false);
    const auto& entry_again = cache.get_or_compute(basis, quad, false, false);

    EXPECT_EQ(&entry, &entry_again);
    EXPECT_EQ(bx->string_identity_calls, bx_identity_calls_after_construction);
    EXPECT_EQ(by->string_identity_calls, by_identity_calls_after_construction);
    ASSERT_EQ(entry.vector_values_xyz.size(), basis.size() * 3u * quad.num_points());
}

TEST(BasisCache, CompatibleTensorVectorStructuredIdentitySeparatesComponentKnots) {
    std::vector<std::shared_ptr<BasisFunction>> components_a{
        make_quad_spline_component(Real(0.4)),
        make_quad_spline_component(Real(0.5))
    };
    std::vector<std::shared_ptr<BasisFunction>> components_b{
        make_quad_spline_component(Real(0.6)),
        make_quad_spline_component(Real(0.5))
    };
    CompatibleTensorVectorBasis basis_a(
        CompatibleTensorVectorBasis::Family::HDiv,
        BasisType::BSpline,
        components_a,
        compatible_associations(components_a),
        2,
        ElementType::Quad4);
    CompatibleTensorVectorBasis basis_b(
        CompatibleTensorVectorBasis::Family::HDiv,
        BasisType::BSpline,
        components_b,
        compatible_associations(components_b),
        2,
        ElementType::Quad4);
    QuadrilateralQuadrature quad(3);

    auto& cache = BasisCache::instance();
    cache.clear();
    const auto& entry_a = cache.get_or_compute(basis_a, quad, false, false);
    const auto& entry_b = cache.get_or_compute(basis_b, quad, false, false);

    EXPECT_NE(&entry_a, &entry_b);
    EXPECT_EQ(cache.size(), 2u);
}

TEST(BasisCache, CompatibleTensorVectorStructuredIdentitySeparatesNurbsWeights) {
    std::vector<std::shared_ptr<BasisFunction>> components_a{
        make_quad_nurbs_component(Real(1.25)),
        make_quad_nurbs_component(Real(1.0))
    };
    std::vector<std::shared_ptr<BasisFunction>> components_b{
        make_quad_nurbs_component(Real(1.5)),
        make_quad_nurbs_component(Real(1.0))
    };
    CompatibleTensorVectorBasis basis_a(
        CompatibleTensorVectorBasis::Family::HCurl,
        BasisType::NURBS,
        components_a,
        compatible_associations(components_a),
        2,
        ElementType::Quad4);
    CompatibleTensorVectorBasis basis_b(
        CompatibleTensorVectorBasis::Family::HCurl,
        BasisType::NURBS,
        components_b,
        compatible_associations(components_b),
        2,
        ElementType::Quad4);
    QuadrilateralQuadrature quad(3);

    auto& cache = BasisCache::instance();
    cache.clear();
    const auto& entry_a = cache.get_or_compute(basis_a, quad, false, false);
    const auto& entry_b = cache.get_or_compute(basis_b, quad, false, false);

    EXPECT_NE(&entry_a, &entry_b);
    EXPECT_EQ(cache.size(), 2u);
}

TEST(BasisCache, CompatibleTensorVectorStructuredIdentitySeparatesFamilyAndElement) {
    std::vector<std::shared_ptr<BasisFunction>> components{
        make_quad_spline_component(),
        make_quad_spline_component()
    };
    CompatibleTensorVectorBasis hdiv_basis(
        CompatibleTensorVectorBasis::Family::HDiv,
        BasisType::BSpline,
        components,
        compatible_associations(components),
        2,
        ElementType::Quad4);
    CompatibleTensorVectorBasis hcurl_basis(
        CompatibleTensorVectorBasis::Family::HCurl,
        BasisType::BSpline,
        components,
        compatible_associations(components),
        2,
        ElementType::Quad4);

    std::vector<std::shared_ptr<BasisFunction>> hex_components{
        make_hex_spline_component(),
        make_hex_spline_component(),
        make_hex_spline_component()
    };
    CompatibleTensorVectorBasis hex_basis(
        CompatibleTensorVectorBasis::Family::HDiv,
        BasisType::BSpline,
        hex_components,
        compatible_associations(hex_components),
        2,
        ElementType::Hex8);

    std::vector<std::uint64_t> hdiv_words;
    std::vector<std::uint64_t> hcurl_words;
    std::vector<std::uint64_t> hex_words;
    EXPECT_TRUE(hdiv_basis.cache_identity_words(hdiv_words));
    EXPECT_TRUE(hcurl_basis.cache_identity_words(hcurl_words));
    EXPECT_TRUE(hex_basis.cache_identity_words(hex_words));
    EXPECT_NE(hdiv_words, hcurl_words);
    EXPECT_NE(hdiv_words, hex_words);
}

TEST(BasisCache, VectorBasisPopulatesVectorValues) {
    RaviartThomasBasis basis(ElementType::Quad4, 0);
    QuadrilateralQuadrature quad(2);

    auto& cache = BasisCache::instance();
    cache.clear();
    const auto& entry = cache.get_or_compute(basis, quad, false, false);

    EXPECT_TRUE(entry.scalar_values.empty());
    ASSERT_EQ(entry.vector_values_xyz.size(), basis.size() * 3u * quad.num_points());

    for (std::size_t qp = 0; qp < quad.num_points(); ++qp) {
        for (std::size_t dof = 0; dof < basis.size(); ++dof) {
            const auto v = entry.vectorValue(dof, qp);
            for (std::size_t component = 0; component < 3u; ++component) {
                EXPECT_DOUBLE_EQ(v[component],
                                 entry.vector_values_xyz[(dof * 3u + component) * quad.num_points() + qp]);
            }
        }
    }
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

TEST(BasisCache, VectorDerivativeStorageMatchesDirectEvaluation) {
    QuadrilateralQuadrature quad(2);
    auto& cache = BasisCache::instance();
    cache.clear();

    {
        RaviartThomasBasis basis(ElementType::Quad4, 0);
        const auto& entry = cache.get_or_compute(basis, quad, true, false);
        ASSERT_EQ(entry.vector_jacobians.size(), basis.size() * 9u * quad.num_points());
        ASSERT_EQ(entry.vector_divergence.size(), basis.size() * quad.num_points());
        EXPECT_TRUE(entry.vector_curls_xyz.empty());
        ASSERT_EQ(entry.vectorJacobiansForDof(0u).size(), 9u * quad.num_points());
        ASSERT_EQ(entry.vectorJacobiansForDofComponentDerivative(0u, 0u, 0u).size(),
                  quad.num_points());
        ASSERT_EQ(entry.vectorDivergenceForDof(0u).size(), quad.num_points());

        std::vector<VectorJacobian> jacobians;
        std::vector<Real> divergence;
        for (std::size_t qp = 0; qp < quad.num_points(); ++qp) {
            basis.evaluate_vector_jacobians(quad.points()[qp], jacobians);
            basis.evaluate_divergence(quad.points()[qp], divergence);
            for (std::size_t dof = 0; dof < basis.size(); ++dof) {
                for (std::size_t component = 0; component < 3u; ++component) {
                    for (std::size_t derivative = 0; derivative < 3u; ++derivative) {
                        EXPECT_DOUBLE_EQ(entry.vectorJacobianValue(dof, component, derivative, qp),
                                         jacobians[dof](component, derivative));
                    }
                }
                EXPECT_DOUBLE_EQ(entry.vectorDivergenceValue(dof, qp), divergence[dof]);
            }
        }
    }

    {
        NedelecBasis basis(ElementType::Quad4, 0);
        const auto& entry = cache.get_or_compute(basis, quad, true, false);
        ASSERT_EQ(entry.vector_jacobians.size(), basis.size() * 9u * quad.num_points());
        ASSERT_EQ(entry.vector_curls_xyz.size(), basis.size() * 3u * quad.num_points());
        EXPECT_TRUE(entry.vector_divergence.empty());
        ASSERT_EQ(entry.vectorValuesForDof(0u).size(), 3u * quad.num_points());
        ASSERT_EQ(entry.vectorCurlsForDof(0u).size(), 3u * quad.num_points());
        ASSERT_EQ(entry.vectorCurlsForDofComponent(0u, 2u).size(), quad.num_points());

        std::vector<VectorJacobian> jacobians;
        std::vector<math::Vector<Real, 3>> curls;
        for (std::size_t qp = 0; qp < quad.num_points(); ++qp) {
            basis.evaluate_vector_jacobians(quad.points()[qp], jacobians);
            basis.evaluate_curl(quad.points()[qp], curls);
            for (std::size_t dof = 0; dof < basis.size(); ++dof) {
                for (std::size_t component = 0; component < 3u; ++component) {
                    for (std::size_t derivative = 0; derivative < 3u; ++derivative) {
                        EXPECT_DOUBLE_EQ(entry.vectorJacobianValue(dof, component, derivative, qp),
                                         jacobians[dof](component, derivative));
                    }
                    EXPECT_DOUBLE_EQ(entry.vectorCurlValue(dof, component, qp),
                                     curls[dof][component]);
                }
            }
        }
    }
}

TEST(BasisCache, VectorStridedPathDerivesCurlAndDivergenceFromJacobians) {
    CountingVectorBasis basis;
    QuadrilateralQuadrature quad(2);

    auto& cache = BasisCache::instance();
    cache.clear();
    const auto& entry = cache.get_or_compute(basis, quad, true, false);

    EXPECT_EQ(basis.value_calls, quad.num_points());
    EXPECT_EQ(basis.jacobian_calls, quad.num_points());
    EXPECT_EQ(basis.curl_calls, 0u);
    EXPECT_EQ(basis.divergence_calls, 0u);

    ASSERT_EQ(entry.vector_curls_xyz.size(), basis.size() * 3u * quad.num_points());
    ASSERT_EQ(entry.vector_divergence.size(), basis.size() * quad.num_points());
    for (std::size_t qp = 0; qp < quad.num_points(); ++qp) {
        for (std::size_t component = 0; component < 3u; ++component) {
            EXPECT_DOUBLE_EQ(entry.vectorCurlValue(0u, component, qp), Real(0));
            EXPECT_DOUBLE_EQ(entry.vectorCurlValue(1u, component, qp), Real(0));
        }
        EXPECT_DOUBLE_EQ(entry.vectorDivergenceValue(0u, qp), Real(3));
        EXPECT_DOUBLE_EQ(entry.vectorDivergenceValue(1u, qp), Real(2));
    }
}

TEST(BasisCache, GradientAndHessianFlagsRespected) {
    LagrangeBasis basis(ElementType::Line2, 2);
    GaussQuadrature1D quad(3);
    auto& cache = BasisCache::instance();
    cache.clear();

    const auto& entry_hess = cache.get_or_compute(basis, quad, false, true);
    EXPECT_TRUE(entry_hess.gradients.empty());
    ASSERT_EQ(entry_hess.hessians.size(), basis.size() * 9u * quad.num_points());

    const auto& entry_grad = cache.get_or_compute(basis, quad, true, false);
    ASSERT_EQ(entry_grad.gradients.size(), basis.size() * 3u * quad.num_points());
    EXPECT_TRUE(entry_grad.hessians.empty());
}

TEST(BasisCache, GradientAndHessianStorageIsDofMajorContiguous) {
    LagrangeBasis basis(ElementType::Quad4, 2);
    QuadrilateralQuadrature quad(3);

    auto& cache = BasisCache::instance();
    cache.clear();
    const auto& entry = cache.get_or_compute(basis, quad, true, true);

    ASSERT_EQ(entry.scalar_values.size(), basis.size() * quad.num_points());
    ASSERT_EQ(entry.gradients.size(), basis.size() * 3u * quad.num_points());
    ASSERT_EQ(entry.hessians.size(), basis.size() * 9u * quad.num_points());

    for (std::size_t dof = 0; dof < basis.size(); ++dof) {
        const auto values = entry.scalarValuesForDof(dof);
        ASSERT_EQ(values.size(), quad.num_points());
        EXPECT_EQ(values.data(), entry.scalar_values.data() + dof * quad.num_points());

        const auto gx = entry.gradientsForDofComponent(dof, 0u);
        ASSERT_EQ(gx.size(), quad.num_points());
        EXPECT_EQ(gx.data(), entry.gradients.data() + (dof * 3u) * quad.num_points());

        const auto grad_block = entry.gradientsForDof(dof);
        ASSERT_EQ(grad_block.size(), 3u * quad.num_points());
        EXPECT_EQ(grad_block.data(), entry.gradients.data() + dof * 3u * quad.num_points());

        const auto hxx = entry.hessiansForDofComponent(dof, 0u, 0u);
        ASSERT_EQ(hxx.size(), quad.num_points());
        EXPECT_EQ(hxx.data(), entry.hessians.data() + (dof * 9u) * quad.num_points());

        const auto hess_block = entry.hessiansForDof(dof);
        ASSERT_EQ(hess_block.size(), 9u * quad.num_points());
        EXPECT_EQ(hess_block.data(), entry.hessians.data() + dof * 9u * quad.num_points());
    }
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

TEST(BasisCache, SerendipityGeometryModeDoesNotAliasFieldMode) {
    SerendipityBasis field_basis(ElementType::Hex20, 2, false);
    SerendipityBasis geometry_basis(ElementType::Hex20, 2, true);
    const math::Vector<Real, 3> xi{Real(0.25), Real(-0.35), Real(0.1)};
    CustomQuadratureRule quad(svmp::CellFamily::Hex, 3, 3, {xi}, {Real(1)});

    auto& cache = BasisCache::instance();
    cache.clear();
    const auto& field_entry = cache.get_or_compute(field_basis, quad, false, false);
    const auto& geometry_entry = cache.get_or_compute(geometry_basis, quad, false, false);

    EXPECT_NE(&field_entry, &geometry_entry);
    EXPECT_EQ(cache.size(), 2u);

    std::vector<Real> field_values;
    std::vector<Real> geometry_values;
    field_basis.evaluate_values(xi, field_values);
    geometry_basis.evaluate_values(xi, geometry_values);
    ASSERT_EQ(field_values.size(), geometry_values.size());
    ASSERT_EQ(field_values.size(), field_entry.num_dofs);
    ASSERT_EQ(geometry_values.size(), geometry_entry.num_dofs);

    Real max_difference = Real(0);
    for (std::size_t dof = 0; dof < field_values.size(); ++dof) {
        EXPECT_NEAR(field_entry.scalarValue(dof, 0), field_values[dof], Real(1e-12));
        EXPECT_NEAR(geometry_entry.scalarValue(dof, 0), geometry_values[dof], Real(1e-12));
        max_difference = std::max(max_difference, std::abs(field_values[dof] - geometry_values[dof]));
    }
    EXPECT_GT(max_difference, Real(1e-8));
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

TEST(BasisCache, BSplineSignedZeroKnotsReuseEntries) {
    BSplineBasis basis_negative_zero(1, {Real(-0.0), Real(0), Real(0.5), Real(1), Real(1)});
    BSplineBasis basis_positive_zero(1, {Real(0.0), Real(0), Real(0.5), Real(1), Real(1)});
    GaussQuadrature1D quad(4);

    auto& cache = BasisCache::instance();
    cache.clear();
    const auto& entry_negative_zero = cache.get_or_compute(basis_negative_zero, quad, true, false);
    const auto& entry_positive_zero = cache.get_or_compute(basis_positive_zero, quad, true, false);

    EXPECT_EQ(&entry_negative_zero, &entry_positive_zero);
    EXPECT_EQ(cache.size(), 1u);
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
    const auto& entry = cache.get_or_compute(basis, quad, true, true);

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
            EXPECT_NEAR(entry.scalarValue(dof, qp), values[dof], 1e-14);
            for (std::size_t component = 0; component < 3u; ++component) {
                EXPECT_NEAR(entry.gradientValue(dof, component, qp),
                            gradients[dof][component],
                            1e-12);
            }
            for (std::size_t row = 0; row < 3u; ++row) {
                for (std::size_t col = 0; col < 3u; ++col) {
                    EXPECT_NEAR(entry.hessianValue(dof, row, col, qp),
                                hessians[dof](row, col),
                                1e-10);
                }
            }
        }
    }
}

TEST(BasisCache, CachedSplineInactiveRowsRemainZero) {
    BSplineBasis basis(2, make_open_uniform_knots(2, 8));
    GaussQuadrature1D quad(5);

    auto& cache = BasisCache::instance();
    cache.clear();
    const auto& entry = cache.get_or_compute(basis, quad, true, true);

    for (std::size_t qp = 0; qp < quad.num_points(); ++qp) {
        std::vector<Real> active_values;
        const auto active = basis.evaluate_active_support(quad.point(qp), active_values);
        const std::size_t active_end = active.first_index + active.count;
        for (std::size_t dof = 0; dof < basis.size(); ++dof) {
            if (dof >= active.first_index && dof < active_end) {
                continue;
            }
            EXPECT_DOUBLE_EQ(entry.scalarValue(dof, qp), Real(0));
            for (std::size_t component = 0; component < 3u; ++component) {
                EXPECT_DOUBLE_EQ(entry.gradientValue(dof, component, qp), Real(0));
            }
            for (std::size_t row = 0; row < 3u; ++row) {
                for (std::size_t col = 0; col < 3u; ++col) {
                    EXPECT_DOUBLE_EQ(entry.hessianValue(dof, row, col, qp), Real(0));
                }
            }
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

TEST(BasisCache, CachedTensorNURBSMatchesDirectEvaluation) {
    BSplineBasis bx(2, make_open_uniform_knots(2, 5));
    BSplineBasis by(2, make_open_uniform_knots(2, 5));
    std::vector<Real> weights(25u, Real(1));
    for (std::size_t i = 0; i < weights.size(); ++i) {
        weights[i] += Real(0.04) * static_cast<Real>((i * 3u) % 7u);
    }
    NURBSTensorBasis basis(bx, by, weights, {5, 5});
    QuadrilateralQuadrature quad(4);

    auto& cache = BasisCache::instance();
    cache.clear();
    const auto& entry = cache.get_or_compute(basis, quad, true, true);

    for (std::size_t qp = 0; qp < quad.num_points(); ++qp) {
        std::vector<Real> values;
        std::vector<Gradient> gradients;
        std::vector<Hessian> hessians;
        basis.evaluate_all(quad.point(qp), values, gradients, hessians);

        ASSERT_EQ(values.size(), basis.size());
        ASSERT_EQ(gradients.size(), basis.size());
        ASSERT_EQ(hessians.size(), basis.size());
        for (std::size_t dof = 0; dof < basis.size(); ++dof) {
            EXPECT_NEAR(entry.scalarValue(dof, qp), values[dof], 1e-14);
            for (std::size_t component = 0; component < 3u; ++component) {
                EXPECT_NEAR(entry.gradientValue(dof, component, qp),
                            gradients[dof][component],
                            1e-12);
            }
            for (std::size_t row = 0; row < 3u; ++row) {
                for (std::size_t col = 0; col < 3u; ++col) {
                    EXPECT_NEAR(entry.hessianValue(dof, row, col, qp),
                                hessians[dof](row, col),
                                1e-11);
                }
            }
        }
    }
}

TEST(BasisCache, CachedTensorNURBSInactiveRowsRemainZero) {
    BSplineBasis bx(2, make_open_uniform_knots(2, 6));
    BSplineBasis by(2, make_open_uniform_knots(2, 6));
    std::vector<Real> weights(36u, Real(1));
    for (std::size_t i = 0; i < weights.size(); ++i) {
        weights[i] += Real(0.03) * static_cast<Real>((i * 5u) % 11u);
    }
    NURBSTensorBasis basis(bx, by, weights, {6, 6});
    QuadrilateralQuadrature quad(4);

    auto& cache = BasisCache::instance();
    cache.clear();
    const auto& entry = cache.get_or_compute(basis, quad, true, true);

    const auto& axis_sizes = basis.axis_sizes();
    const std::size_t nx = axis_sizes[0];
    for (std::size_t qp = 0; qp < quad.num_points(); ++qp) {
        const auto active = basis.active_tensor_support(quad.point(qp));
        const std::size_t ix_begin = active.first_indices[0];
        const std::size_t iy_begin = active.first_indices[1];
        const std::size_t ix_end = ix_begin + active.counts[0];
        const std::size_t iy_end = iy_begin + active.counts[1];
        for (std::size_t dof = 0; dof < basis.size(); ++dof) {
            const std::size_t ix = dof % nx;
            const std::size_t iy = dof / nx;
            if (ix >= ix_begin && ix < ix_end && iy >= iy_begin && iy < iy_end) {
                continue;
            }
            EXPECT_DOUBLE_EQ(entry.scalarValue(dof, qp), Real(0));
            for (std::size_t component = 0; component < 3u; ++component) {
                EXPECT_DOUBLE_EQ(entry.gradientValue(dof, component, qp), Real(0));
            }
            for (std::size_t row = 0; row < 3u; ++row) {
                for (std::size_t col = 0; col < 3u; ++col) {
                    EXPECT_DOUBLE_EQ(entry.hessianValue(dof, row, col, qp), Real(0));
                }
            }
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

TEST(BasisCache, QuadratureSignedZeroCoordinatesProduceDifferentEntries) {
    LagrangeBasis basis(ElementType::Line2, 1);
    CustomQuadratureRule quad_negative_zero(
        svmp::CellFamily::Line, 1, 1,
        {QuadPoint{Real(-0.0), Real(0), Real(0)}},
        {Real(2)});
    CustomQuadratureRule quad_positive_zero(
        svmp::CellFamily::Line, 1, 1,
        {QuadPoint{Real(0.0), Real(0), Real(0)}},
        {Real(2)});

    auto& cache = BasisCache::instance();
    cache.clear();
    const auto& entry_negative_zero =
        cache.get_or_compute(basis, quad_negative_zero, true, false);
    const auto& entry_positive_zero =
        cache.get_or_compute(basis, quad_positive_zero, true, false);

    EXPECT_NE(&entry_negative_zero, &entry_positive_zero);
    EXPECT_EQ(cache.size(), 2u);
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
                EXPECT_NEAR(entry_grad.gradientValue(dof, sd, qp), gradients[dof][sd], 1e-10);
                for (int e = 0; e < basis.dimension(); ++e) {
                    const std::size_t se = static_cast<std::size_t>(e);
                    EXPECT_NEAR(entry_hess.hessianValue(dof, sd, se, qp), hessians[dof](sd, se), 1e-8);
                }
            }
        }
    }
}

TEST(BasisCache, PyramidApexValueOnlyCacheMatchesDirectEvaluation) {
    const auto quad = make_pyramid_quadrature_with_apex();

    LagrangeBasis canonical(ElementType::Pyramid5, 2);
    LagrangeBasis alias(ElementType::Pyramid14, 2);
    SerendipityBasis pyramid13(ElementType::Pyramid13, 2);

    auto& cache = BasisCache::instance();
    cache.clear();

    const auto& entry_canonical = cache.get_or_compute(canonical, quad, false, false);
    const auto& entry_alias = cache.get_or_compute(alias, quad, false, false);
    const auto& entry_serendipity = cache.get_or_compute(pyramid13, quad, false, false);

    EXPECT_EQ(&entry_canonical, &entry_alias);
    EXPECT_EQ(cache.size(), 2u);

    for (std::size_t qp = 0; qp < quad.num_points(); ++qp) {
        std::vector<Real> canonical_values;
        std::vector<Real> alias_values;
        std::vector<Real> ser_values;
        canonical.evaluate_values(quad.point(qp), canonical_values);
        alias.evaluate_values(quad.point(qp), alias_values);
        pyramid13.evaluate_values(quad.point(qp), ser_values);

        ASSERT_EQ(canonical_values.size(), canonical.size());
        ASSERT_EQ(alias_values.size(), alias.size());
        ASSERT_EQ(ser_values.size(), pyramid13.size());

        for (std::size_t i = 0; i < canonical.size(); ++i) {
            EXPECT_NEAR(entry_canonical.scalarValue(i, qp), canonical_values[i], 1e-12);
            EXPECT_NEAR(entry_alias.scalarValue(i, qp), alias_values[i], 1e-12);
        }
        for (std::size_t i = 0; i < pyramid13.size(); ++i) {
            EXPECT_NEAR(entry_serendipity.scalarValue(i, qp), ser_values[i], 1e-12);
        }
    }
}

TEST(BasisCache, PyramidApexGradientConstructionThrowsAndDoesNotPoisonCache) {
    const auto quad = make_pyramid_quadrature_with_apex();
    auto& cache = BasisCache::instance();

    {
        LagrangeBasis basis(ElementType::Pyramid5, 2);
        cache.clear();
        EXPECT_THROW((void)cache.get_or_compute(basis, quad, true, false), BasisEvaluationException);
        EXPECT_EQ(cache.size(), 0u);

        const auto& values = cache.get_or_compute(basis, quad, false, false);
        EXPECT_EQ(cache.size(), 1u);
        EXPECT_EQ(values.num_qpts, quad.num_points());

        EXPECT_THROW((void)cache.get_or_compute(basis, quad, true, false), BasisEvaluationException);
        EXPECT_EQ(cache.size(), 1u);
    }

    {
        LagrangeBasis basis(ElementType::Pyramid14, 2);
        cache.clear();
        EXPECT_THROW((void)cache.get_or_compute(basis, quad, true, false), BasisEvaluationException);
        EXPECT_EQ(cache.size(), 0u);
    }

    {
        SerendipityBasis basis(ElementType::Pyramid13, 2);
        cache.clear();
        EXPECT_THROW((void)cache.get_or_compute(basis, quad, true, false), BasisEvaluationException);
        EXPECT_EQ(cache.size(), 0u);

        const auto& values = cache.get_or_compute(basis, quad, false, false);
        EXPECT_EQ(cache.size(), 1u);
        EXPECT_EQ(values.num_qpts, quad.num_points());

        EXPECT_THROW((void)cache.get_or_compute(basis, quad, true, false), BasisEvaluationException);
        EXPECT_EQ(cache.size(), 1u);
    }
}

TEST(BasisCache, PrewarmHandlePropagatesFailedComputationAndDoesNotPoisonCache) {
    const auto quad = make_pyramid_quadrature_with_apex();
    LagrangeBasis basis(ElementType::Pyramid5, 2);
    auto& cache = BasisCache::instance();

    cache.clear();
    EXPECT_THROW((void)cache.prewarm_handle(basis, quad, true, false),
                 BasisEvaluationException);
    EXPECT_EQ(cache.size(), 0u);

    const auto& values = cache.get_or_compute(basis, quad, false, false);
    EXPECT_EQ(cache.size(), 1u);
    EXPECT_EQ(values.num_qpts, quad.num_points());

    EXPECT_THROW((void)cache.prewarm_handle(basis, quad, true, false),
                 BasisEvaluationException);
    EXPECT_EQ(cache.size(), 1u);
}

TEST(BasisCache, PyramidApexHessianConstructionThrowsAndDoesNotPoisonCache) {
    const auto quad = make_pyramid_quadrature_with_apex();
    auto& cache = BasisCache::instance();

    {
        LagrangeBasis basis(ElementType::Pyramid5, 2);
        cache.clear();
        EXPECT_THROW((void)cache.get_or_compute(basis, quad, false, true), BasisEvaluationException);
        EXPECT_EQ(cache.size(), 0u);

        const auto& values = cache.get_or_compute(basis, quad, false, false);
        EXPECT_EQ(cache.size(), 1u);
        EXPECT_EQ(values.num_qpts, quad.num_points());

        EXPECT_THROW((void)cache.get_or_compute(basis, quad, false, true), BasisEvaluationException);
        EXPECT_EQ(cache.size(), 1u);
    }

    {
        LagrangeBasis basis(ElementType::Pyramid14, 2);
        cache.clear();
        EXPECT_THROW((void)cache.get_or_compute(basis, quad, false, true), BasisEvaluationException);
        EXPECT_EQ(cache.size(), 0u);
    }

    {
        SerendipityBasis basis(ElementType::Pyramid13, 2);
        cache.clear();
        EXPECT_THROW((void)cache.get_or_compute(basis, quad, false, true), BasisEvaluationException);
        EXPECT_EQ(cache.size(), 0u);

        const auto& values = cache.get_or_compute(basis, quad, false, false);
        EXPECT_EQ(cache.size(), 1u);
        EXPECT_EQ(values.num_qpts, quad.num_points());

        EXPECT_THROW((void)cache.get_or_compute(basis, quad, false, true), BasisEvaluationException);
        EXPECT_EQ(cache.size(), 1u);
    }
}

TEST(BasisFactory, CreatesVectorConformingBasis) {
    BasisRequest req{ElementType::Quad4, BasisType::Lagrange, 0, Continuity::H_div, FieldType::Vector};
    auto basis = basis_factory::create(req);
    EXPECT_TRUE(basis->is_vector_valued());
    EXPECT_EQ(basis->element_type(), ElementType::Quad4);
}

TEST(BasisFactory, CreatesCompatibleQuadVectorSplineAndNurbsBases) {
    BasisRequest spline_req{ElementType::Quad4, BasisType::BSpline, 2, Continuity::H_curl, FieldType::Vector};
    spline_req.axis_orders = {2, 2};
    spline_req.axis_knot_vectors = {
        make_open_uniform_knots(2, 4),
        make_open_uniform_knots(2, 4)
    };

    auto spline_basis = basis_factory::create(spline_req);
    ASSERT_TRUE(spline_basis);
    EXPECT_TRUE(spline_basis->is_vector_valued());
    EXPECT_EQ(spline_basis->basis_type(), BasisType::BSpline);
    EXPECT_EQ(spline_basis->element_type(), ElementType::Quad4);
    EXPECT_EQ(spline_basis->size(), 24u);

    BasisRequest nurbs_req{ElementType::Quad4, BasisType::NURBS, 2, Continuity::H_div, FieldType::Vector};
    nurbs_req.axis_orders = {2, 2};
    nurbs_req.axis_knot_vectors = spline_req.axis_knot_vectors;
    nurbs_req.tensor_extents = {4, 4};
    nurbs_req.weights.assign(16u, Real(1));
    nurbs_req.weights[5] = Real(0.8);
    nurbs_req.weights[10] = Real(1.25);

    auto nurbs_basis = basis_factory::create(nurbs_req);
    ASSERT_TRUE(nurbs_basis);
    EXPECT_TRUE(nurbs_basis->is_vector_valued());
    EXPECT_EQ(nurbs_basis->basis_type(), BasisType::NURBS);
    EXPECT_EQ(nurbs_basis->element_type(), ElementType::Quad4);
    EXPECT_EQ(nurbs_basis->size(), 24u);

    std::vector<math::Vector<Real, 3>> values;
    std::vector<Real> divergence;
    nurbs_basis->evaluate_vector_values({Real(0.15), Real(-0.35), Real(0)}, values);
    nurbs_basis->evaluate_divergence({Real(0.15), Real(-0.35), Real(0)}, divergence);
    ASSERT_EQ(values.size(), nurbs_basis->size());
    ASSERT_EQ(divergence.size(), nurbs_basis->size());
}

TEST(BasisFactory, CreatesCompatibleHexVectorSplineAndNurbsBases) {
    BasisRequest spline_req{ElementType::Hex8, BasisType::BSpline, 2, Continuity::H_curl, FieldType::Vector};
    spline_req.axis_orders = {2, 2, 2};
    spline_req.axis_knot_vectors = {
        make_open_uniform_knots(2, 4),
        make_open_uniform_knots(2, 4),
        make_open_uniform_knots(2, 4)
    };

    auto spline_basis = basis_factory::create(spline_req);
    ASSERT_TRUE(spline_basis);
    EXPECT_TRUE(spline_basis->is_vector_valued());
    EXPECT_EQ(spline_basis->basis_type(), BasisType::BSpline);
    EXPECT_EQ(spline_basis->element_type(), ElementType::Hex8);
    EXPECT_EQ(spline_basis->dimension(), 3);
    EXPECT_EQ(spline_basis->size(), 144u);

    std::vector<math::Vector<Real, 3>> curl;
    spline_basis->evaluate_curl({Real(0.15), Real(-0.35), Real(0.2)}, curl);
    ASSERT_EQ(curl.size(), spline_basis->size());

    auto* spline_vector_basis = dynamic_cast<VectorBasisFunction*>(spline_basis.get());
    ASSERT_NE(spline_vector_basis, nullptr);
    const auto spline_assoc = spline_vector_basis->dof_associations();
    ASSERT_EQ(spline_assoc.size(), spline_basis->size());
    EXPECT_EQ(std::count_if(spline_assoc.begin(), spline_assoc.end(), [](const DofAssociation& assoc) {
        return assoc.entity_type == DofEntity::Edge;
    }), 36);
    EXPECT_EQ(std::count_if(spline_assoc.begin(), spline_assoc.end(), [](const DofAssociation& assoc) {
        return assoc.entity_type == DofEntity::Face;
    }), 72);
    EXPECT_EQ(std::count_if(spline_assoc.begin(), spline_assoc.end(), [](const DofAssociation& assoc) {
        return assoc.entity_type == DofEntity::Interior;
    }), 36);

    BasisRequest nurbs_req{ElementType::Hex8, BasisType::NURBS, 2, Continuity::H_div, FieldType::Vector};
    nurbs_req.axis_orders = {2, 2, 2};
    nurbs_req.axis_knot_vectors = spline_req.axis_knot_vectors;
    nurbs_req.tensor_extents = {4, 4, 4};
    nurbs_req.weights.assign(64u, Real(1));
    nurbs_req.weights[5] = Real(0.8);
    nurbs_req.weights[21] = Real(1.25);
    nurbs_req.weights[42] = Real(1.1);

    auto nurbs_basis = basis_factory::create(nurbs_req);
    ASSERT_TRUE(nurbs_basis);
    EXPECT_TRUE(nurbs_basis->is_vector_valued());
    EXPECT_EQ(nurbs_basis->basis_type(), BasisType::NURBS);
    EXPECT_EQ(nurbs_basis->element_type(), ElementType::Hex8);
    EXPECT_EQ(nurbs_basis->dimension(), 3);
    EXPECT_EQ(nurbs_basis->size(), 108u);

    std::vector<math::Vector<Real, 3>> values;
    std::vector<Real> divergence;
    nurbs_basis->evaluate_vector_values({Real(0.15), Real(-0.35), Real(0.2)}, values);
    nurbs_basis->evaluate_divergence({Real(0.15), Real(-0.35), Real(0.2)}, divergence);
    ASSERT_EQ(values.size(), nurbs_basis->size());
    ASSERT_EQ(divergence.size(), nurbs_basis->size());

    auto* nurbs_vector_basis = dynamic_cast<VectorBasisFunction*>(nurbs_basis.get());
    ASSERT_NE(nurbs_vector_basis, nullptr);
    const auto nurbs_assoc = nurbs_vector_basis->dof_associations();
    ASSERT_EQ(nurbs_assoc.size(), nurbs_basis->size());
    EXPECT_EQ(std::count_if(nurbs_assoc.begin(), nurbs_assoc.end(), [](const DofAssociation& assoc) {
        return assoc.entity_type == DofEntity::Face;
    }), 54);
    EXPECT_EQ(std::count_if(nurbs_assoc.begin(), nurbs_assoc.end(), [](const DofAssociation& assoc) {
        return assoc.entity_type == DofEntity::Interior;
    }), 54);
}

TEST(BasisFactory, DefaultHDivOrderOneOnTwoDimensionalCellsUsesRaviartThomas) {
    BasisRequest quad_req{ElementType::Quad4, BasisType::Lagrange, 1, Continuity::H_div, FieldType::Vector};
    auto quad_basis = basis_factory::create(quad_req);
    EXPECT_EQ(quad_basis->basis_type(), BasisType::RaviartThomas);

    BasisRequest tri_req{ElementType::Triangle3, BasisType::Lagrange, 1, Continuity::H_div, FieldType::Vector};
    auto tri_basis = basis_factory::create(tri_req);
    EXPECT_EQ(tri_basis->basis_type(), BasisType::RaviartThomas);
}

TEST(BasisFactory, CreatesHDivHigherOrderOnQuad) {
    BasisRequest req{ElementType::Quad4, BasisType::Lagrange, 2, Continuity::H_div, FieldType::Vector};
    auto basis = basis_factory::create(req);
    EXPECT_TRUE(basis->is_vector_valued());
    EXPECT_EQ(basis->basis_type(), BasisType::RaviartThomas);
    EXPECT_EQ(basis->element_type(), ElementType::Quad4);
    EXPECT_EQ(basis->order(), 2);
    EXPECT_EQ(basis->size(), 24u); // 2*(k+1)*(k+2) with k=2
}

TEST(BasisFactory, CreatesHDivHigherOrderOnTriangle) {
    BasisRequest req{ElementType::Triangle3, BasisType::Lagrange, 2, Continuity::H_div, FieldType::Vector};
    auto basis = basis_factory::create(req);
    EXPECT_TRUE(basis->is_vector_valued());
    EXPECT_EQ(basis->basis_type(), BasisType::RaviartThomas);
    EXPECT_EQ(basis->element_type(), ElementType::Triangle3);
    EXPECT_EQ(basis->order(), 2);
    EXPECT_EQ(basis->size(), 15u); // (k+1)*(k+3) with k=2
}

TEST(BasisFactory, CreatesHCurlHigherOrderOnQuad) {
    BasisRequest req{ElementType::Quad4, BasisType::Lagrange, 1, Continuity::H_curl, FieldType::Vector};
    auto basis = basis_factory::create(req);
    EXPECT_TRUE(basis->is_vector_valued());
    EXPECT_EQ(basis->basis_type(), BasisType::Nedelec);
    EXPECT_EQ(basis->element_type(), ElementType::Quad4);
    EXPECT_EQ(basis->order(), 1);
    EXPECT_EQ(basis->size(), 12u); // 2*(k+1)*(k+2) with k=1
}

TEST(BasisFactory, CreatesHCurlHigherOrderOnTriangle) {
    BasisRequest req{ElementType::Triangle3, BasisType::Lagrange, 1, Continuity::H_curl, FieldType::Vector};
    auto basis = basis_factory::create(req);
    EXPECT_TRUE(basis->is_vector_valued());
    EXPECT_EQ(basis->basis_type(), BasisType::Nedelec);
    EXPECT_EQ(basis->element_type(), ElementType::Triangle3);
    EXPECT_EQ(basis->order(), 1);
    EXPECT_EQ(basis->size(), 8u); // (k+1)*(k+3) with k=1
}

TEST(BasisFactory, CreatesHCurlHigherOrderOnTetra) {
    BasisRequest req{ElementType::Tetra4, BasisType::Lagrange, 1, Continuity::H_curl, FieldType::Vector};
    auto basis = basis_factory::create(req);
    EXPECT_TRUE(basis->is_vector_valued());
    EXPECT_EQ(basis->basis_type(), BasisType::Nedelec);
    EXPECT_EQ(basis->element_type(), ElementType::Tetra4);
    EXPECT_EQ(basis->order(), 1);
    EXPECT_EQ(basis->size(), 20u); // (k+1)*(k+3)*(k+4)/2 with k=1
}

TEST(BasisFactory, NegativeHDivOrderThrowsInvalidArgumentBeforeConstruction) {
    BasisRequest req{ElementType::Quad4, BasisType::Lagrange, -1, Continuity::H_div, FieldType::Vector};

    try {
        (void)basis_factory::create(req);
        FAIL() << "Expected FEException";
    } catch (const FEException& e) {
        EXPECT_EQ(e.status(), FEStatus::InvalidArgument);
    }
}

TEST(BasisFactory, NegativeHCurlOrderThrowsInvalidArgumentBeforeConstruction) {
    BasisRequest req{ElementType::Quad4, BasisType::Lagrange, -1, Continuity::H_curl, FieldType::Vector};

    try {
        (void)basis_factory::create(req);
        FAIL() << "Expected FEException";
    } catch (const FEException& e) {
        EXPECT_EQ(e.status(), FEStatus::InvalidArgument);
    }
}

TEST(BasisFactory, CreatesScalarBasesByType) {
    {
        BasisRequest req{ElementType::Line2, BasisType::Lagrange, 2, Continuity::C0, FieldType::Scalar};
        auto basis = basis_factory::create(req);
        EXPECT_EQ(basis->basis_type(), BasisType::Lagrange);
        EXPECT_EQ(basis->size(), 3u);
    }
    {
        BasisRequest req{ElementType::Line2, BasisType::Hierarchical, 3, Continuity::C0, FieldType::Scalar};
        auto basis = basis_factory::create(req);
        EXPECT_EQ(basis->basis_type(), BasisType::Hierarchical);
        EXPECT_EQ(basis->size(), 4u);
    }
    {
        BasisRequest req{ElementType::Line2, BasisType::Spectral, 3, Continuity::C0, FieldType::Scalar};
        auto basis = basis_factory::create(req);
        EXPECT_EQ(basis->basis_type(), BasisType::Spectral);
        EXPECT_EQ(basis->size(), 4u);
    }
    {
        BasisRequest req{ElementType::Triangle3, BasisType::Bernstein, 2, Continuity::C0, FieldType::Scalar};
        auto basis = basis_factory::create(req);
        EXPECT_EQ(basis->basis_type(), BasisType::Bernstein);
        EXPECT_EQ(basis->element_type(), ElementType::Triangle3);
    }
    {
        BasisRequest req{ElementType::Quad4, BasisType::Serendipity, 2, Continuity::C0, FieldType::Scalar};
        auto basis = basis_factory::create(req);
        EXPECT_EQ(basis->basis_type(), BasisType::Serendipity);
        EXPECT_EQ(basis->size(), 8u);
    }
    {
        BasisRequest req{ElementType::Line2, BasisType::BSpline, 2, Continuity::C0, FieldType::Scalar};
        req.knot_vector = make_open_uniform_knots(2, 6);
        auto basis = basis_factory::create(req);
        EXPECT_EQ(basis->basis_type(), BasisType::BSpline);
        EXPECT_EQ(basis->size(), 6u);
    }
    {
        BasisRequest req{ElementType::Line2, BasisType::NURBS, 2, Continuity::C0, FieldType::Scalar};
        req.knot_vector = make_open_uniform_knots(2, 5);
        req.weights = {Real(1), Real(0.5), Real(2), Real(0.7), Real(1)};
        auto basis = basis_factory::create(req);
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
        auto basis = basis_factory::create(req);
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
        auto alias_basis = basis_factory::create(alias_req);
        auto canonical_basis = basis_factory::create(canonical_req);

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
    EXPECT_THROW(auto b = basis_factory::create(req), svmp::FE::FEException);
}

TEST(BasisFactory, HigherOrderSerendipityRequestsRemainUnsupported) {
    {
        BasisRequest req{ElementType::Quad4, BasisType::Serendipity, 4, Continuity::C0, FieldType::Scalar};
        auto basis = basis_factory::create(req);
        ASSERT_NE(basis, nullptr);
        EXPECT_EQ(basis->basis_type(), BasisType::Serendipity);
        EXPECT_EQ(basis->element_type(), ElementType::Quad4);
        EXPECT_EQ(basis->order(), 4);
        EXPECT_EQ(basis->size(), 17u);
    }

    for (const ElementType type : {ElementType::Hex8, ElementType::Wedge15, ElementType::Pyramid13}) {
        BasisRequest req{type, BasisType::Serendipity, 3, Continuity::C0, FieldType::Scalar};
        EXPECT_THROW((void)basis_factory::create(req), svmp::FE::FEException);
    }
}

TEST(BasisFactory, ScalarSplineAndNurbsRemainTensorProductOnly) {
    for (const ElementType type : {ElementType::Triangle3,
                                   ElementType::Tetra4,
                                   ElementType::Wedge6,
                                   ElementType::Pyramid5}) {
        BasisRequest spline_req{type, BasisType::BSpline, 2, Continuity::C0, FieldType::Scalar};
        EXPECT_THROW((void)basis_factory::create(spline_req), svmp::FE::FEException);

        BasisRequest nurbs_req{type, BasisType::NURBS, 2, Continuity::C0, FieldType::Scalar};
        nurbs_req.weights = {Real(1)};
        EXPECT_THROW((void)basis_factory::create(nurbs_req), svmp::FE::FEException);
    }
}

TEST(BasisFactory, CompatibleVectorSplineAndNurbsRemainQuadHexOnly) {
    for (const ElementType type : {ElementType::Line2,
                                   ElementType::Triangle3,
                                   ElementType::Tetra4,
                                   ElementType::Wedge6,
                                   ElementType::Pyramid5}) {
        BasisRequest hcurl_spline{type, BasisType::BSpline, 2, Continuity::H_curl, FieldType::Vector};
        EXPECT_THROW((void)basis_factory::create(hcurl_spline), svmp::FE::FEException);

        BasisRequest hdiv_nurbs{type, BasisType::NURBS, 2, Continuity::H_div, FieldType::Vector};
        EXPECT_THROW((void)basis_factory::create(hdiv_nurbs), svmp::FE::FEException);
    }
}

TEST(BasisFactory, C1ContinuityRemainsCubicTensorProductHermiteOnly) {
    for (const ElementType type : {ElementType::Line2, ElementType::Quad4, ElementType::Hex8}) {
        BasisRequest req{type, BasisType::Lagrange, 3, Continuity::C1, FieldType::Scalar};
        auto basis = basis_factory::create(req);
        ASSERT_NE(basis, nullptr);
        EXPECT_EQ(basis->basis_type(), BasisType::Hermite);
        EXPECT_EQ(basis->element_type(), type);
        EXPECT_EQ(basis->order(), 3);
    }

    {
        BasisRequest req{ElementType::Triangle3, BasisType::Lagrange, 3, Continuity::C1, FieldType::Scalar};
        EXPECT_THROW((void)basis_factory::create(req), svmp::FE::FEException);
    }
    {
        BasisRequest req{ElementType::Wedge6, BasisType::Lagrange, 3, Continuity::C1, FieldType::Scalar};
        EXPECT_THROW((void)basis_factory::create(req), svmp::FE::FEException);
    }
    {
        BasisRequest req{ElementType::Hex8, BasisType::Lagrange, 5, Continuity::C1, FieldType::Scalar};
        EXPECT_THROW((void)basis_factory::create(req), svmp::FE::FEException);
    }
}

TEST(BasisFactory, ExplicitBDMRequestsHonorExpandedSimplexScope) {
    {
        BasisRequest req{ElementType::Triangle3, BasisType::BDM, 1, Continuity::H_div, FieldType::Vector};
        auto basis = basis_factory::create(req);
        EXPECT_EQ(basis->basis_type(), BasisType::BDM);
        EXPECT_EQ(basis->element_type(), ElementType::Triangle3);
        EXPECT_EQ(basis->size(), 6u);
    }
    {
        BasisRequest req{ElementType::Triangle3, BasisType::BDM, 2, Continuity::H_div, FieldType::Vector};
        auto basis = basis_factory::create(req);
        EXPECT_EQ(basis->basis_type(), BasisType::BDM);
        EXPECT_EQ(basis->element_type(), ElementType::Triangle3);
        EXPECT_EQ(basis->size(), 12u);
    }
    {
        BasisRequest req{ElementType::Quad4, BasisType::BDM, 1, Continuity::H_div, FieldType::Vector};
        auto basis = basis_factory::create(req);
        EXPECT_EQ(basis->basis_type(), BasisType::BDM);
        EXPECT_EQ(basis->element_type(), ElementType::Quad4);
    }
    {
        BasisRequest req{ElementType::Tetra4, BasisType::BDM, 1, Continuity::H_div, FieldType::Vector};
        auto basis = basis_factory::create(req);
        EXPECT_EQ(basis->basis_type(), BasisType::BDM);
        EXPECT_EQ(basis->element_type(), ElementType::Tetra4);
        EXPECT_EQ(basis->size(), 12u);
    }
    {
        BasisRequest req{ElementType::Quad4, BasisType::BDM, 2, Continuity::H_div, FieldType::Vector};
        EXPECT_THROW((void)basis_factory::create(req), svmp::FE::FEException);
    }
    {
        BasisRequest req{ElementType::Hex8, BasisType::BDM, 1, Continuity::H_div, FieldType::Vector};
        EXPECT_THROW((void)basis_factory::create(req), svmp::FE::FEException);
    }
    {
        BasisRequest req{ElementType::Wedge6, BasisType::BDM, 1, Continuity::H_div, FieldType::Vector};
        EXPECT_THROW((void)basis_factory::create(req), svmp::FE::FEException);
    }
    {
        BasisRequest req{ElementType::Pyramid5, BasisType::BDM, 1, Continuity::H_div, FieldType::Vector};
        EXPECT_THROW((void)basis_factory::create(req), svmp::FE::FEException);
    }
}

TEST(BasisFactory, L2ReturnsExpectedBasis) {
    // L2 (discontinuous) should return the same basis as C0 -- DOF ownership
    // is handled at the Space/Element level, not in the basis shape functions.
    {
        BasisRequest req{ElementType::Triangle3, BasisType::Lagrange, 2, Continuity::L2, FieldType::Scalar};
        auto basis = basis_factory::create(req);
        EXPECT_EQ(basis->basis_type(), BasisType::Lagrange);
        EXPECT_EQ(basis->element_type(), ElementType::Triangle3);
        EXPECT_EQ(basis->order(), 2);
        EXPECT_EQ(basis->size(), 6u);
    }
    {
        BasisRequest req{ElementType::Quad4, BasisType::Hierarchical, 3, Continuity::L2, FieldType::Scalar};
        auto basis = basis_factory::create(req);
        EXPECT_EQ(basis->basis_type(), BasisType::Hierarchical);
        EXPECT_EQ(basis->element_type(), ElementType::Quad4);
        EXPECT_EQ(basis->order(), 3);
    }
}

TEST(BasisFactory, CreatesTensorBSplineFromDescriptor) {
    BasisRequest req{ElementType::Quad4, BasisType::BSpline, 2, Continuity::C0, FieldType::Scalar};
    req.knot_vector = make_open_uniform_knots(2, 5);

    auto basis = basis_factory::create(req);

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

    auto basis = basis_factory::create(req);

    EXPECT_EQ(basis->basis_type(), BasisType::BSpline);
    EXPECT_EQ(basis->element_type(), ElementType::Quad4);
    EXPECT_EQ(basis->size(), 20u);
}

TEST(BasisFactory, MissingSplineParametersThrows) {
    BasisRequest req{ElementType::Line2, BasisType::BSpline, 2, Continuity::C0, FieldType::Scalar};
    EXPECT_THROW((void)basis_factory::create(req), svmp::FE::FEException);
}

TEST(BasisFactory, InvalidSplineWeightsThrow) {
    BasisRequest req{ElementType::Line2, BasisType::BSpline, 2, Continuity::C0, FieldType::Scalar};
    req.knot_vector = make_open_uniform_knots(2, 5);
    req.weights = {Real(1), Real(1), Real(1), Real(1), Real(1)};
    EXPECT_THROW((void)basis_factory::create(req), svmp::FE::FEException);
}

TEST(BasisFactory, InvalidNURBSWithoutWeightsThrows) {
    BasisRequest req{ElementType::Line2, BasisType::NURBS, 2, Continuity::C0, FieldType::Scalar};
    req.knot_vector = make_open_uniform_knots(2, 5);
    EXPECT_THROW((void)basis_factory::create(req), svmp::FE::FEException);
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

    auto basis = basis_factory::create(req);

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

    auto from_factory = basis_factory::create(req);
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

TEST(BasisFactory, FactoryAndDirectHexNURBSConstructionAgree) {
    BasisRequest req{ElementType::Hex8, BasisType::NURBS, 1, Continuity::C0, FieldType::Scalar};
    req.axis_orders = {1, 2, 3};
    req.axis_knot_vectors = {
        make_open_uniform_knots(1, 4),
        make_open_uniform_knots(2, 5),
        make_open_uniform_knots(3, 6)
    };
    req.tensor_extents = {4, 5, 6};
    req.weights.reserve(120u);
    for (int k = 0; k < 6; ++k) {
        for (int j = 0; j < 5; ++j) {
            for (int i = 0; i < 4; ++i) {
                req.weights.push_back(Real(0.75) + Real(0.05) * Real((i + 2 * j + 3 * k) % 9));
            }
        }
    }

    auto from_factory = basis_factory::create(req);
    NURBSTensorBasis direct(
        BSplineBasis(1, req.axis_knot_vectors[0]),
        BSplineBasis(2, req.axis_knot_vectors[1]),
        BSplineBasis(3, req.axis_knot_vectors[2]),
        req.weights,
        req.tensor_extents);

    const math::Vector<Real, 3> xi{Real(0.13), Real(-0.37), Real(0.48)};
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
        for (std::size_t d = 0; d < 3u; ++d) {
            EXPECT_NEAR(factory_gradients[i][d], direct_gradients[i][d], 1e-12);
        }
    }
}

TEST(BasisFactory, SplineAxisDescriptorSizeMismatchesThrow) {
    BasisRequest order_mismatch{ElementType::Quad4, BasisType::BSpline, 1, Continuity::C0, FieldType::Scalar};
    order_mismatch.axis_orders = {1};
    order_mismatch.axis_knot_vectors = {
        make_open_uniform_knots(1, 4),
        make_open_uniform_knots(1, 4)
    };
    EXPECT_THROW((void)basis_factory::create(order_mismatch), svmp::FE::FEException);

    BasisRequest knots_mismatch{ElementType::Quad4, BasisType::BSpline, 1, Continuity::C0, FieldType::Scalar};
    knots_mismatch.axis_orders = {1, 1};
    knots_mismatch.axis_knot_vectors = {make_open_uniform_knots(1, 4)};
    EXPECT_THROW((void)basis_factory::create(knots_mismatch), svmp::FE::FEException);

    BasisRequest extents_mismatch{ElementType::Quad4, BasisType::NURBS, 1, Continuity::C0, FieldType::Scalar};
    extents_mismatch.axis_orders = {1, 1};
    extents_mismatch.axis_knot_vectors = {
        make_open_uniform_knots(1, 4),
        make_open_uniform_knots(1, 4)
    };
    extents_mismatch.tensor_extents = {4};
    extents_mismatch.weights.resize(16u, Real(1));
    EXPECT_THROW((void)basis_factory::create(extents_mismatch), svmp::FE::FEException);
}

TEST(BasisFactory, FactoryAndDirectSplineConstructionAgree) {
    BasisRequest req{ElementType::Line2, BasisType::BSpline, 2, Continuity::C0, FieldType::Scalar};
    req.knot_vector = make_open_uniform_knots(2, 6);
    auto from_factory = basis_factory::create(req);
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
    auto from_factory = basis_factory::create(req);
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
    basis_factory::clear_custom_registry_for_tests();

    basis_factory::register_custom(
        "test-linear",
        [](const BasisRequest& req) -> std::shared_ptr<BasisFunction> {
            return std::make_shared<TestCustomScalarBasis>(req.order.value_or(1));
        });

    BasisRequest req{ElementType::Line2, BasisType::Custom, 3, Continuity::C0, FieldType::Scalar};
    req.custom_id = "test-linear";

    auto basis = basis_factory::create(req);
    ASSERT_NE(basis, nullptr);
    EXPECT_EQ(basis->basis_type(), BasisType::Custom);
    EXPECT_EQ(basis->element_type(), ElementType::Line2);
    EXPECT_EQ(basis->order(), 3);

    std::vector<Real> values;
    basis->evaluate_values({Real(0.25), Real(0), Real(0)}, values);
    ASSERT_EQ(values.size(), 2u);
    EXPECT_NEAR(values[0] + values[1], Real(1), 1e-12);

    basis_factory::unregister_custom("test-linear");
    basis_factory::clear_custom_registry_for_tests();
}

TEST(BasisFactory, NamespaceApiCreatesBasis) {
    BasisRequest req{ElementType::Quad4, BasisType::Lagrange, 2, Continuity::C0, FieldType::Scalar};

    const auto basis = basis_factory::create(req);

    ASSERT_NE(basis, nullptr);
    EXPECT_EQ(basis->element_type(), ElementType::Quad4);
    EXPECT_EQ(basis->basis_type(), BasisType::Lagrange);
    EXPECT_EQ(basis->order(), 2);
    EXPECT_EQ(basis->size(), 9u);
}

TEST(BasisFactory, DuplicateCustomRegistrationThrows) {
    basis_factory::clear_custom_registry_for_tests();
    basis_factory::register_custom(
        "duplicate",
        [](const BasisRequest& req) -> std::shared_ptr<BasisFunction> {
            return std::make_shared<TestCustomScalarBasis>(req.order.value_or(1));
        });

    EXPECT_THROW(
        basis_factory::register_custom(
            "duplicate",
            [](const BasisRequest& req) -> std::shared_ptr<BasisFunction> {
                return std::make_shared<TestCustomScalarBasis>(req.order.value_or(1));
            }),
        svmp::FE::FEException);

    basis_factory::clear_custom_registry_for_tests();
}

TEST(BasisFactory, UnknownCustomBasisThrows) {
    basis_factory::clear_custom_registry_for_tests();

    BasisRequest req{ElementType::Line2, BasisType::Custom, 1, Continuity::C0, FieldType::Scalar};
    req.custom_id = "missing";
    EXPECT_THROW((void)basis_factory::create(req), svmp::FE::FEException);
}
