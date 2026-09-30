/**
 * @file test_ReferenceElement.cpp
 * @brief Unit tests for ReferenceElement topology and measures
 */

#include <gtest/gtest.h>

#include "FE/Elements/ReferenceElement.h"
#include "FE/Basis/NodeOrderingConventions.h"
#include "FE/Quadrature/QuadratureFactory.h"

#include <cstring>
#include <thread>
#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::elements;

namespace {

void check_consistency(ElementType type) {
    ReferenceElement ref = ReferenceElement::create(type);

    EXPECT_EQ(ref.type(), type);
    EXPECT_EQ(ref.dimension(), element_dimension(type));

    // Node count should match NodeOrderingConventions
    const std::size_t nn = basis::ReferenceNodeLayout::num_nodes(type);
    EXPECT_EQ(ref.num_nodes(), nn);

    // Reference measure should agree with QuadratureFactory
    using svmp::FE::quadrature::QuadratureFactory;

    auto quad = QuadratureFactory::create(type, 1);
    EXPECT_NEAR(static_cast<double>(ref.reference_measure()),
                static_cast<double>(quad->reference_measure()),
                1e-13);
}

} // namespace

TEST(ReferenceElement, BasicLinearTypes) {
    check_consistency(ElementType::Line2);
    check_consistency(ElementType::Triangle3);
    check_consistency(ElementType::Quad4);
    check_consistency(ElementType::Tetra4);
    check_consistency(ElementType::Hex8);
    check_consistency(ElementType::Wedge6);
    check_consistency(ElementType::Pyramid5);
}

TEST(ReferenceElement, HighOrderVariantsShareTopology) {
    ReferenceElement tet4 = ReferenceElement::create(ElementType::Tetra4);
    ReferenceElement tet10 = ReferenceElement::create(ElementType::Tetra10);

    EXPECT_EQ(tet4.num_faces(), tet10.num_faces());
    EXPECT_EQ(tet4.num_edges(), tet10.num_edges());

    ReferenceElement hex8 = ReferenceElement::create(ElementType::Hex8);
    ReferenceElement hex27 = ReferenceElement::create(ElementType::Hex27);
    EXPECT_EQ(hex8.num_faces(), hex27.num_faces());
    EXPECT_EQ(hex8.num_edges(), hex27.num_edges());
}

TEST(ReferenceElement, TetraConnectivity) {
    ReferenceElement ref = ReferenceElement::create(ElementType::Tetra4);
    ASSERT_EQ(ref.num_faces(), 4u);
    ASSERT_EQ(ref.num_edges(), 6u);

    // Each face of a tetrahedron is a triangle
    for (std::size_t f = 0; f < ref.num_faces(); ++f) {
        EXPECT_EQ(ref.face_nodes(f).size(), 3u);
    }
}

TEST(ReferenceElement, HexConnectivity) {
    ReferenceElement ref = ReferenceElement::create(ElementType::Hex8);
    EXPECT_EQ(ref.num_faces(), 6u);
    EXPECT_EQ(ref.num_edges(), 12u);

    // Bottom face expected to be (0,1,2,3)
    const auto& bottom = ref.face_nodes(0);
    ASSERT_EQ(bottom.size(), 4u);
    EXPECT_EQ(bottom[0], 0u);
    EXPECT_EQ(bottom[1], 1u);
    EXPECT_EQ(bottom[2], 2u);
    EXPECT_EQ(bottom[3], 3u);
}

TEST(ReferenceElement, WedgeConnectivity) {
    // Wedge6: 2 triangle faces + 3 quad faces
    ReferenceElement ref6 = ReferenceElement::create(ElementType::Wedge6);
    ASSERT_EQ(ref6.num_faces(), 5u);

    // Bottom and top triangular faces
    const auto& f0 = ref6.face_nodes(0);
    const auto& f1 = ref6.face_nodes(1);
    ASSERT_EQ(f0.size(), 3u);
    ASSERT_EQ(f1.size(), 3u);
    EXPECT_EQ(f0[0], 0u);
    EXPECT_EQ(f0[1], 1u);
    EXPECT_EQ(f0[2], 2u);
    EXPECT_EQ(f1[0], 3u);
    EXPECT_EQ(f1[1], 4u);
    EXPECT_EQ(f1[2], 5u);

    // Three quadrilateral faces
    const auto& fq2 = ref6.face_nodes(2);
    const auto& fq3 = ref6.face_nodes(3);
    const auto& fq4 = ref6.face_nodes(4);
    ASSERT_EQ(fq2.size(), 4u);
    ASSERT_EQ(fq3.size(), 4u);
    ASSERT_EQ(fq4.size(), 4u);
    // Check one quad explicitly: face 2 = {0,1,4,3}
    EXPECT_EQ(fq2[0], 0u);
    EXPECT_EQ(fq2[1], 1u);
    EXPECT_EQ(fq2[2], 4u);
    EXPECT_EQ(fq2[3], 3u);

    // Wedge18 should share the same face connectivity as Wedge6
    ReferenceElement ref18 = ReferenceElement::create(ElementType::Wedge18);
    ASSERT_EQ(ref18.num_faces(), ref6.num_faces());
    for (std::size_t i = 0; i < ref6.num_faces(); ++i) {
        const auto& f6 = ref6.face_nodes(i);
        const auto& f18 = ref18.face_nodes(i);
        ASSERT_EQ(f6.size(), f18.size());
        for (std::size_t j = 0; j < f6.size(); ++j) {
            EXPECT_EQ(f6[j], f18[j]);
        }
    }
}

TEST(ReferenceElement, PyramidConnectivity) {
    ReferenceElement ref = ReferenceElement::create(ElementType::Pyramid5);
    ASSERT_EQ(ref.num_faces(), 5u);

    // Base quad face
    const auto& base = ref.face_nodes(0);
    ASSERT_EQ(base.size(), 4u);
    EXPECT_EQ(base[0], 0u);
    EXPECT_EQ(base[1], 1u);
    EXPECT_EQ(base[2], 2u);
    EXPECT_EQ(base[3], 3u);

    // Four triangular side faces include the apex node 4
    for (std::size_t f = 1; f < ref.num_faces(); ++f) {
        const auto& fn = ref.face_nodes(f);
        ASSERT_EQ(fn.size(), 3u);
        bool has_apex = (fn[0] == 4u) || (fn[1] == 4u) || (fn[2] == 4u);
        EXPECT_TRUE(has_apex);
    }
}

TEST(ReferenceElement, SharedInstanceIsBuiltOncePerTypeAndMatchesCreate) {
    const ElementType types[] = {
        ElementType::Line2, ElementType::Triangle3, ElementType::Quad4,
        ElementType::Tetra4, ElementType::Hex8, ElementType::Wedge6,
        ElementType::Pyramid5, ElementType::Line3, ElementType::Triangle6,
        ElementType::Quad9, ElementType::Quad8, ElementType::Tetra10,
        ElementType::Hex27, ElementType::Hex20, ElementType::Wedge15,
        ElementType::Wedge18, ElementType::Pyramid13, ElementType::Pyramid14};
    for (const auto type : types) {
        const ReferenceElement& shared = ReferenceElement::shared(type);
        EXPECT_EQ(&shared, &ReferenceElement::shared(type));
        const ReferenceElement created = ReferenceElement::create(type);
        EXPECT_EQ(created.type(), shared.type());
        EXPECT_EQ(created.dimension(), shared.dimension());
        EXPECT_EQ(created.num_nodes(), shared.num_nodes());
        // The cached measure must be the bitwise value of a fresh build.
        const Real measure = created.reference_measure();
        const Real cached = shared.reference_measure();
        EXPECT_EQ(std::memcmp(&measure, &cached, sizeof(Real)), 0);
        ASSERT_EQ(created.num_edges(), shared.num_edges());
        for (std::size_t e = 0; e < created.num_edges(); ++e) {
            EXPECT_EQ(created.edge_nodes(e), shared.edge_nodes(e));
        }
        ASSERT_EQ(created.num_faces(), shared.num_faces());
        for (std::size_t f = 0; f < created.num_faces(); ++f) {
            EXPECT_EQ(created.face_nodes(f), shared.face_nodes(f));
        }
    }
}

TEST(ReferenceElement, SharedLookupIsThreadSafe) {
    constexpr int n_threads = 8;
    std::vector<const ReferenceElement*> seen(n_threads, nullptr);
    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; ++t) {
        threads.emplace_back([t, &seen]() {
            const ReferenceElement* first = nullptr;
            for (int i = 0; i < 1000; ++i) {
                const auto& ref = ReferenceElement::shared(ElementType::Wedge18);
                if (first == nullptr) {
                    first = &ref;
                }
                if (&ref != first) {
                    first = nullptr;
                    break;
                }
            }
            seen[static_cast<std::size_t>(t)] = first;
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    for (const auto* ptr : seen) {
        EXPECT_EQ(ptr, &ReferenceElement::shared(ElementType::Wedge18));
    }
}

TEST(ReferenceElement, UnknownTypeKeepsFailingAndIsNotCached) {
    EXPECT_THROW((void)ReferenceElement::shared(ElementType::Unknown), FEException);
    EXPECT_THROW((void)ReferenceElement::shared(ElementType::Unknown), FEException);
    EXPECT_THROW((void)ReferenceElement::create(ElementType::Unknown), FEException);
}
