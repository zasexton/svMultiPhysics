/**
 * @file test_ElementCacheThreadSafety.cpp
 * @brief Multithreaded stress tests for ElementCache (BasisCache + JacobianCache)
 */

#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

#include "FE/Basis/LagrangeBasis.h"
#include "FE/Elements/ElementCache.h"
#include "FE/Elements/LagrangeElement.h"
#include "FE/Geometry/IsoparametricMapping.h"

using namespace svmp::FE;
using namespace svmp::FE::elements;

TEST(ElementCacheThreadSafety, ConcurrentGetAndBatchGet) {
    ElementCache::instance().clear();

    // Use a higher-order element to exercise larger cached tables.
    LagrangeElement elem(ElementType::Hex27, 2);

    // Linear geometry mapping (affine) for the reference hex.
    auto geom_basis = std::make_shared<basis::LagrangeBasis>(ElementType::Hex8, 1);
    auto geom_nodes = geom_basis->nodes();
    for (auto& n : geom_nodes) {
        n[0] = Real(2.0) * n[0] + Real(0.15) * n[1];
        n[1] = Real(1.1) * n[1] + Real(0.10) * n[2];
        n[2] = Real(0.9) * n[2];
    }
    geometry::IsoparametricMapping mapping(geom_basis, geom_nodes);

    const auto quad = elem.quadrature();
    ASSERT_TRUE(quad);

    std::atomic<bool> ok{true};

    auto worker = [&]() {
        for (int iter = 0; iter < 2000; ++iter) {
            const auto entry = ElementCache::instance().get(elem, mapping);
            if (!entry.basis || !entry.jacobian) {
                ok.store(false, std::memory_order_relaxed);
                return;
            }
            if (entry.jacobian->detJ.size() != quad->num_points()) {
                ok.store(false, std::memory_order_relaxed);
                return;
            }
            if (entry.basis->num_qpts != quad->num_points()) {
                ok.store(false, std::memory_order_relaxed);
                return;
            }
            if (entry.basis->num_dofs > 0 &&
                entry.basis->scalarValuesForDof(0).size() != quad->num_points()) {
                ok.store(false, std::memory_order_relaxed);
                return;
            }
        }
    };

    std::vector<std::thread> threads;
    unsigned nthreads_u = std::thread::hardware_concurrency();
    if (nthreads_u == 0u) {
        nthreads_u = 4u;
    }
    nthreads_u = std::min(8u, std::max(2u, nthreads_u));
    const int nthreads = static_cast<int>(nthreads_u);
    threads.reserve(static_cast<std::size_t>(nthreads));
    for (int t = 0; t < nthreads; ++t) {
        threads.emplace_back(worker);
    }
    for (auto& th : threads) {
        th.join();
    }

    // Also exercise get_batch on the fully-populated caches.
    std::vector<const Element*> elements(8, &elem);
    std::vector<const geometry::GeometryMapping*> mappings(8, &mapping);
    const auto batch = ElementCache::instance().get_batch(elements, mappings, BatchEvaluationHints{8});
    ASSERT_EQ(batch.size(), elements.size());
    for (const auto& e : batch) {
        EXPECT_TRUE(e.basis);
        EXPECT_TRUE(e.jacobian);
    }

    EXPECT_TRUE(ok.load(std::memory_order_relaxed));
}
