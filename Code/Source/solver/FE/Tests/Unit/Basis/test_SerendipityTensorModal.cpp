/**
 * @file test_SerendipityTensorModal.cpp
 * @brief Tests for serendipity, tensor-product, and modal transforms
 */

#include <gtest/gtest.h>
#include "FE/Basis/SerendipityBasis.h"
#include "FE/Basis/TensorBasis.h"
#include "FE/Basis/LagrangeBasis.h"
#include "FE/Basis/HierarchicalBasis.h"
#include "FE/Basis/ModalTransform.h"
#include "FE/Basis/NodeOrderingConventions.h"
#include "FE/Core/FEException.h"

#include <array>
#include <cmath>
#include <exception>
#include <numeric>
#include <thread>
#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::basis;

namespace {

std::size_t expected_quad_serendipity_size(int order) {
    return static_cast<std::size_t>((order * order + 3 * order + 6) / 2);
}

void expect_quad_serendipity_nodal(int order) {
    SerendipityBasis basis(ElementType::Quad4, order);
    ASSERT_EQ(basis.size(), expected_quad_serendipity_size(order));

    const auto& nodes = basis.nodes();
    ASSERT_EQ(nodes.size(), basis.size());

    for (std::size_t j = 0; j < nodes.size(); ++j) {
        std::vector<Real> vals;
        basis.evaluate_values(nodes[j], vals);
        ASSERT_EQ(vals.size(), basis.size());
        for (std::size_t i = 0; i < vals.size(); ++i) {
            if (i == j) {
                EXPECT_NEAR(vals[i], 1.0, 1e-10);
            } else {
                EXPECT_NEAR(vals[i], 0.0, 1e-10);
            }
        }
    }

    const svmp::FE::math::Vector<Real, 3> xi{Real(0.17), Real(-0.31), Real(0)};
    std::vector<Real> vals;
    std::vector<Gradient> grads;
    basis.evaluate_values(xi, vals);
    basis.evaluate_gradients(xi, grads);

    Real sum = Real(0);
    Gradient grad_sum{};
    for (std::size_t i = 0; i < vals.size(); ++i) {
        sum += vals[i];
        grad_sum[0] += grads[i][0];
        grad_sum[1] += grads[i][1];
        grad_sum[2] += grads[i][2];
    }

    EXPECT_NEAR(sum, 1.0, 1e-10);
    EXPECT_NEAR(grad_sum[0], 0.0, 1e-10);
    EXPECT_NEAR(grad_sum[1], 0.0, 1e-10);
}

struct TensorStridedRequest {
    bool values;
    bool gradients;
    bool hessians;
};

void expect_tensor_strided_matches_pointwise(
    const TensorProductBasis<LagrangeBasis>& basis,
    const std::vector<svmp::FE::math::Vector<Real, 3>>& points,
    const TensorStridedRequest& request) {
    const std::size_t stride = points.size() + 5u;
    constexpr Real sentinel = Real(-991.5);

    std::vector<Real> values(request.values ? basis.size() * stride : 0u, sentinel);
    std::vector<Real> gradients(request.gradients ? basis.size() * 3u * stride : 0u, sentinel);
    std::vector<Real> hessians(request.hessians ? basis.size() * 9u * stride : 0u, sentinel);

    basis.evaluate_at_quadrature_points_strided(
        points,
        stride,
        request.values ? values.data() : nullptr,
        request.gradients ? gradients.data() : nullptr,
        request.hessians ? hessians.data() : nullptr);

    for (std::size_t q = 0; q < points.size(); ++q) {
        if (request.values) {
            std::vector<Real> expected;
            basis.evaluate_values(points[q], expected);
            ASSERT_EQ(expected.size(), basis.size());
            for (std::size_t i = 0; i < basis.size(); ++i) {
                EXPECT_NEAR(values[i * stride + q], expected[i], Real(1e-12));
            }
        }
        if (request.gradients) {
            std::vector<Gradient> expected;
            basis.evaluate_gradients(points[q], expected);
            ASSERT_EQ(expected.size(), basis.size());
            for (std::size_t i = 0; i < basis.size(); ++i) {
                for (std::size_t c = 0; c < 3u; ++c) {
                    EXPECT_NEAR(gradients[(i * 3u + c) * stride + q],
                                expected[i][c],
                                Real(1e-12));
                }
            }
        }
        if (request.hessians) {
            std::vector<Hessian> expected;
            basis.evaluate_hessians(points[q], expected);
            ASSERT_EQ(expected.size(), basis.size());
            for (std::size_t i = 0; i < basis.size(); ++i) {
                for (std::size_t r = 0; r < 3u; ++r) {
                    for (std::size_t c = 0; c < 3u; ++c) {
                        EXPECT_NEAR(hessians[(i * 9u + r * 3u + c) * stride + q],
                                    expected[i](r, c),
                                    Real(1e-11));
                    }
                }
            }
        }
    }

    const auto expect_padding_untouched = [&](const std::vector<Real>& buffer,
                                              std::size_t rows) {
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t q = points.size(); q < stride; ++q) {
                EXPECT_EQ(buffer[row * stride + q], sentinel);
            }
        }
    };

    if (request.values) {
        expect_padding_untouched(values, basis.size());
    }
    if (request.gradients) {
        expect_padding_untouched(gradients, basis.size() * 3u);
    }
    if (request.hessians) {
        expect_padding_untouched(hessians, basis.size() * 9u);
    }
}

} // namespace

TEST(SerendipityBasis, QuadraticHasEightFunctions) {
    SerendipityBasis basis(ElementType::Quad4, 2);
    EXPECT_EQ(basis.size(), 8u);
    svmp::FE::math::Vector<Real, 3> xi{0.0, 0.0, 0.0};
    std::vector<Real> vals;
    basis.evaluate_values(xi, vals);
    double sum = 0.0;
    for (double v : vals) sum += v;
    EXPECT_NEAR(sum, 1.0, 1e-12);
}

TEST(SerendipityBasis, LinearHasFourFunctions) {
    SerendipityBasis basis(ElementType::Quad4, 1);
    EXPECT_EQ(basis.size(), 4u);
    svmp::FE::math::Vector<Real, 3> xi{0.1, -0.2, 0.0};
    std::vector<Real> vals;
    basis.evaluate_values(xi, vals);
    double sum = 0.0;
    for (double v : vals) sum += v;
    EXPECT_NEAR(sum, 1.0, 1e-12);
}

TEST(SerendipityBasis, HigherOrderQuadrilateralIsNodalAndPartitionsUnity) {
    for (int order : {3, 4, 5}) {
        expect_quad_serendipity_nodal(order);
    }
}

TEST(SerendipityBasis, Hex20FieldBasisNodalAndPartition) {
    SerendipityBasis basis(ElementType::Hex20, 2); // field mode (geometry_mode = false)
    EXPECT_EQ(basis.size(), 20u);

    // Canonical Hex20 reference nodes (VTK/Mesh ordering).
    std::vector<svmp::FE::math::Vector<Real,3>> nodes;
    nodes.reserve(basis.size());
    for (std::size_t i = 0; i < basis.size(); ++i) {
        nodes.push_back(ReferenceNodeLayout::get_node_coords(ElementType::Hex20, i));
    }

    // Nodal property: N_i(node_j) ≈ δ_ij
    for (std::size_t j = 0; j < nodes.size(); ++j) {
        std::vector<Real> vals;
        basis.evaluate_values(nodes[j], vals);
        ASSERT_EQ(vals.size(), 20u);
        for (std::size_t i = 0; i < vals.size(); ++i) {
            if (i == j) {
                EXPECT_NEAR(vals[i], 1.0, 1e-12);
            } else {
                EXPECT_NEAR(vals[i], 0.0, 1e-10);
            }
        }
    }

    // Partition of unity at an interior point
    svmp::FE::math::Vector<Real, 3> xi{Real(0.2), Real(-0.1), Real(0.3)};
    std::vector<Real> vals;
    basis.evaluate_values(xi, vals);
    double sum = 0.0;
    for (double v : vals) sum += v;
    EXPECT_NEAR(sum, 1.0, 1e-10);
}

TEST(SerendipityBasis, Quad9IsNotSupported) {
    EXPECT_THROW(SerendipityBasis(ElementType::Quad9, 2), svmp::FE::FEException);
}

TEST(SerendipityBasis, Wedge15FieldBasisNodalAndPartition) {
    SerendipityBasis basis(ElementType::Wedge15, 2); // field mode
    EXPECT_EQ(basis.size(), 15u);

    // Canonical Wedge15 nodes used for the polynomial generator:
    // bottom triangle (z = -1), top triangle (z = +1),
    // then mid-edges on bottom, top, and vertical edges.
    std::vector<svmp::FE::math::Vector<Real,3>> nodes = {
        // vertices: bottom z=-1
        {Real(0),   Real(0),   Real(-1)}, // v0
        {Real(1),   Real(0),   Real(-1)}, // v1
        {Real(0),   Real(1),   Real(-1)}, // v2
        // vertices: top z=+1
        {Real(0),   Real(0),   Real(1)},  // v3
        {Real(1),   Real(0),   Real(1)},  // v4
        {Real(0),   Real(1),   Real(1)},  // v5
        // bottom edge midpoints
        {Real(0.5), Real(0),   Real(-1)}, // e0: v0-v1
        {Real(0.5), Real(0.5), Real(-1)}, // e1: v1-v2
        {Real(0),   Real(0.5), Real(-1)}, // e2: v2-v0
        // top edge midpoints
        {Real(0.5), Real(0),   Real(1)},  // e3: v3-v4
        {Real(0.5), Real(0.5), Real(1)},  // e4: v4-v5
        {Real(0),   Real(0.5), Real(1)},  // e5: v5-v3
        // vertical edge midpoints
        {Real(0),   Real(0),   Real(0)},  // e6: v0-v3
        {Real(1),   Real(0),   Real(0)},  // e7: v1-v4
        {Real(0),   Real(1),   Real(0)},  // e8: v2-v5
    };

    ASSERT_EQ(nodes.size(), basis.size());

    // Nodal property: N_i(node_j) ≈ δ_ij
    for (std::size_t j = 0; j < nodes.size(); ++j) {
        std::vector<Real> vals;
        basis.evaluate_values(nodes[j], vals);
        ASSERT_EQ(vals.size(), basis.size());
        for (std::size_t i = 0; i < vals.size(); ++i) {
            if (i == j) {
                EXPECT_NEAR(vals[i], 1.0, 1e-10);
            } else {
                EXPECT_NEAR(vals[i], 0.0, 1e-8);
            }
        }
    }

    // Partition of unity at an interior point (0.2, 0.3, 0.1) with 0 <= x,y, x+y <= 1
    svmp::FE::math::Vector<Real, 3> xi{Real(0.2), Real(0.3), Real(0.1)};
    std::vector<Real> vals;
    basis.evaluate_values(xi, vals);
    double sum = 0.0;
    for (double v : vals) sum += v;
    EXPECT_NEAR(sum, 1.0, 1e-10);
}

TEST(SerendipityBasis, Pyramid13FieldBasisNodalAndPartition) {
    SerendipityBasis basis(ElementType::Pyramid13, 2); // field mode
    EXPECT_EQ(basis.size(), 13u);

    // Canonical Pyramid13 nodes used for the polynomial generator:
    // base square z=0 with vertices at (±1,±1,0), apex at (0,0,1),
    // then midpoints on base edges and edges to the apex.
    std::vector<svmp::FE::math::Vector<Real,3>> nodes = {
        // base vertices (z=0)
        {Real(-1), Real(-1), Real(0)}, // v0
        {Real(1),  Real(-1), Real(0)}, // v1
        {Real(1),  Real(1),  Real(0)}, // v2
        {Real(-1), Real(1),  Real(0)}, // v3
        // apex
        {Real(0),  Real(0),  Real(1)}, // v4
        // base edge midpoints
        {Real(0),  Real(-1), Real(0)}, // v0-v1
        {Real(1),  Real(0),  Real(0)}, // v1-v2
        {Real(0),  Real(1),  Real(0)}, // v2-v3
        {Real(-1), Real(0),  Real(0)}, // v3-v0
        // vertical edge midpoints
        {Real(-0.5), Real(-0.5), Real(0.5)}, // v0-v4
        {Real(0.5),  Real(-0.5), Real(0.5)}, // v1-v4
        {Real(0.5),  Real(0.5),  Real(0.5)}, // v2-v4
        {Real(-0.5), Real(0.5),  Real(0.5)}, // v3-v4
    };

    ASSERT_EQ(nodes.size(), basis.size());

    // Nodal property: N_i(node_j) ≈ δ_ij
    for (std::size_t j = 0; j < nodes.size(); ++j) {
        std::vector<Real> vals;
        basis.evaluate_values(nodes[j], vals);
        ASSERT_EQ(vals.size(), basis.size());
        for (std::size_t i = 0; i < vals.size(); ++i) {
            if (i == j) {
                EXPECT_NEAR(vals[i], 1.0, 1e-10);
            } else {
                EXPECT_NEAR(vals[i], 0.0, 1e-8);
            }
        }
    }

    // Partition of unity at an interior point
    svmp::FE::math::Vector<Real, 3> xi{Real(0.1), Real(-0.2), Real(0.4)};
    std::vector<Real> vals;
    basis.evaluate_values(xi, vals);
    double sum = 0.0;
    for (double v : vals) sum += v;
    EXPECT_NEAR(sum, 1.0, 1e-10);
}

TEST(SerendipityBasis, Pyramid14IsNotSupported) {
    EXPECT_THROW(SerendipityBasis(ElementType::Pyramid14, 2), svmp::FE::FEException);
}

TEST(TensorProductBasis, MatchesLagrangeOnQuad) {
    LagrangeBasis line(ElementType::Line2, 2);
    TensorProductBasis<LagrangeBasis> tensor(line, 2);
    LagrangeBasis quad(ElementType::Quad4, 2);

    svmp::FE::math::Vector<Real, 3> xi{0.25, -0.5, 0.0};
    std::vector<Real> tvals, qvals;
    tensor.evaluate_values(xi, tvals);
    quad.evaluate_values(xi, qvals);
    ASSERT_EQ(tvals.size(), qvals.size());
    for (std::size_t i = 0; i < tvals.size(); ++i) {
        EXPECT_NEAR(tvals[i], qvals[i], 1e-12);
    }
}

TEST(TensorProductBasis, MatchesLagrangeOnHex) {
    LagrangeBasis line(ElementType::Line2, 1);
    TensorProductBasis<LagrangeBasis> tensor(line, 3);
    LagrangeBasis hex(ElementType::Hex8, 1);

    svmp::FE::math::Vector<Real, 3> xi{0.25, -0.5, 0.3};
    std::vector<Real> tvals, hvals;
    tensor.evaluate_values(xi, tvals);
    hex.evaluate_values(xi, hvals);
    ASSERT_EQ(tvals.size(), hvals.size());
    for (std::size_t i = 0; i < tvals.size(); ++i) {
        EXPECT_NEAR(tvals[i], hvals[i], 1e-12);
    }
}

TEST(TensorProductBasis, GradientsMatchLagrange) {
    LagrangeBasis line(ElementType::Line2, 1);
    TensorProductBasis<LagrangeBasis> tensor2(line, 2);
    TensorProductBasis<LagrangeBasis> tensor3(line, 3);
    LagrangeBasis quad(ElementType::Quad4, 1);
    LagrangeBasis hex(ElementType::Hex8, 1);

    // Quad
    {
        svmp::FE::math::Vector<Real, 3> xi{Real(0.3), Real(-0.2), Real(0)};
        std::vector<Real> tv, qv;
        std::vector<svmp::FE::basis::Gradient> tg, qg;
        tensor2.evaluate_values(xi, tv);
        tensor2.evaluate_gradients(xi, tg);
        quad.evaluate_values(xi, qv);
        quad.evaluate_gradients(xi, qg);
        ASSERT_EQ(tv.size(), qv.size());
        ASSERT_EQ(tg.size(), qg.size());
        for (std::size_t i = 0; i < tg.size(); ++i) {
            EXPECT_NEAR(tv[i], qv[i], 1e-12);
            EXPECT_NEAR(tg[i][0], qg[i][0], 1e-12);
            EXPECT_NEAR(tg[i][1], qg[i][1], 1e-12);
        }
    }

    // Hex
    {
        svmp::FE::math::Vector<Real, 3> xi{Real(-0.1), Real(0.25), Real(0.4)};
        std::vector<Real> tv, hv;
        std::vector<svmp::FE::basis::Gradient> tg, hg;
        tensor3.evaluate_values(xi, tv);
        tensor3.evaluate_gradients(xi, tg);
        hex.evaluate_values(xi, hv);
        hex.evaluate_gradients(xi, hg);
        ASSERT_EQ(tv.size(), hv.size());
        ASSERT_EQ(tg.size(), hg.size());
        for (std::size_t i = 0; i < tg.size(); ++i) {
            EXPECT_NEAR(tv[i], hv[i], 1e-12);
            EXPECT_NEAR(tg[i][0], hg[i][0], 1e-12);
            EXPECT_NEAR(tg[i][1], hg[i][1], 1e-12);
            EXPECT_NEAR(tg[i][2], hg[i][2], 1e-12);
        }
    }
}

TEST(TensorProductBasis, StridedEvaluationMatchesPointwise) {
    LagrangeBasis line(ElementType::Line2, 3);
    TensorProductBasis<LagrangeBasis> tensor2(line, 2);
    TensorProductBasis<LagrangeBasis> tensor3(line, 3);

    const std::vector<svmp::FE::math::Vector<Real, 3>> quad_points = {
        {Real(-0.7), Real(-0.25), Real(0)},
        {Real(0.2), Real(0.4), Real(0)},
        {Real(0.75), Real(-0.6), Real(0)},
    };
    const std::vector<svmp::FE::math::Vector<Real, 3>> hex_points = {
        {Real(-0.6), Real(-0.25), Real(0.1)},
        {Real(0.1), Real(0.45), Real(-0.35)},
        {Real(0.65), Real(-0.55), Real(0.7)},
    };
    const std::vector<TensorStridedRequest> requests = {
        {true, false, false},
        {false, true, false},
        {false, false, true},
        {true, true, false},
        {true, false, true},
        {false, true, true},
        {true, true, true},
    };

    for (const auto& request : requests) {
        SCOPED_TRACE(request.values ? "values" : "no values");
        SCOPED_TRACE(request.gradients ? "gradients" : "no gradients");
        SCOPED_TRACE(request.hessians ? "hessians" : "no hessians");
        expect_tensor_strided_matches_pointwise(tensor2, quad_points, request);
        expect_tensor_strided_matches_pointwise(tensor3, hex_points, request);
    }
}

TEST(TensorProductBasis, AnisotropicOrdersMatchManualConstruction) {
    LagrangeBasis bx(ElementType::Line2, 1); // 2 nodes
    LagrangeBasis by(ElementType::Line2, 2); // 3 nodes
    TensorProductBasis<LagrangeBasis> tensor(bx, by);

    svmp::FE::math::Vector<Real, 3> xi{Real(0.25), Real(-0.4), Real(0)};
    std::vector<Real> tvals;
    std::vector<svmp::FE::basis::Gradient> tgrads;
    tensor.evaluate_values(xi, tvals);
    tensor.evaluate_gradients(xi, tgrads);

    std::vector<Real> vx, vy;
    std::vector<svmp::FE::basis::Gradient> gx, gy;
    bx.evaluate_values({xi[0], Real(0), Real(0)}, vx);
    by.evaluate_values({xi[1], Real(0), Real(0)}, vy);
    bx.evaluate_gradients({xi[0], Real(0), Real(0)}, gx);
    by.evaluate_gradients({xi[1], Real(0), Real(0)}, gy);

    ASSERT_EQ(tvals.size(), vx.size() * vy.size());
    ASSERT_EQ(tgrads.size(), vx.size() * vy.size());

    std::size_t idx = 0;
    for (std::size_t j = 0; j < vy.size(); ++j) {
        for (std::size_t i = 0; i < vx.size(); ++i) {
            const Real manual = vx[i] * vy[j];
            EXPECT_NEAR(tvals[idx], manual, 1e-12);

            const Real gx_manual = gx[i][0] * vy[j];
            const Real gy_manual = vx[i] * gy[j][0];
            EXPECT_NEAR(tgrads[idx][0], gx_manual, 1e-12);
            EXPECT_NEAR(tgrads[idx][1], gy_manual, 1e-12);
            EXPECT_NEAR(tgrads[idx][2], 0.0, 1e-12);
            ++idx;
        }
    }
}

TEST(ModalTransform, RoundTrip) {
    HierarchicalBasis modal(ElementType::Line2, 3);
    LagrangeBasis nodal(ElementType::Line2, 3);
    ModalTransform transform(modal, nodal);

    std::vector<Real> modal_coeffs{1.0, -0.2, 0.3, 0.5};
    auto nodal_coeffs = transform.modal_to_nodal(modal_coeffs);
    auto recovered = transform.nodal_to_modal(nodal_coeffs);

    ASSERT_EQ(modal_coeffs.size(), recovered.size());
    for (std::size_t i = 0; i < recovered.size(); ++i) {
        EXPECT_NEAR(modal_coeffs[i], recovered[i], 1e-10);
    }
}

TEST(ModalTransform, RoundTripOnQuad) {
    const int order = 2;
    HierarchicalBasis modal(ElementType::Quad4, order);
    LagrangeBasis nodal(ElementType::Quad4, order);
    ModalTransform transform(modal, nodal);

    const std::size_t n = modal.size();
    std::vector<Real> modal_coeffs(n);
    for (std::size_t i = 0; i < n; ++i) {
        modal_coeffs[i] = static_cast<Real>(0.1 * (static_cast<double>(i) + 1.0));
    }

    auto nodal_coeffs = transform.modal_to_nodal(modal_coeffs);
    auto recovered = transform.nodal_to_modal(nodal_coeffs);

    ASSERT_EQ(modal_coeffs.size(), recovered.size());
    for (std::size_t i = 0; i < recovered.size(); ++i) {
        EXPECT_NEAR(modal_coeffs[i], recovered[i], 1e-9);
    }
}

TEST(ModalTransform, ConditionNumberFiniteAndReasonable) {
    const int order = 3;
    HierarchicalBasis modal(ElementType::Line2, order);
    LagrangeBasis nodal(ElementType::Line2, order);
    ModalTransform transform(modal, nodal);

    const Real cond = transform.condition_number();
    EXPECT_GT(cond, 1.0);
    EXPECT_LT(cond, Real(1e6));
}

TEST(ModalTransform, LazyInverseAccessorMatchesVandermondeSolve) {
    const int order = 3;
    HierarchicalBasis modal(ElementType::Line2, order);
    LagrangeBasis nodal(ElementType::Line2, order);
    ModalTransform transform(modal, nodal);

    const auto& V = transform.vandermonde();
    const auto& Vinv = transform.vandermonde_inverse();
    ASSERT_EQ(V.size(), Vinv.size());
    const std::size_t n = V.size();
    for (std::size_t row = 0; row < n; ++row) {
        ASSERT_EQ(V[row].size(), n);
        ASSERT_EQ(Vinv[row].size(), n);
    }

    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            Real product = Real(0);
            for (std::size_t k = 0; k < n; ++k) {
                product += V[row][k] * Vinv[k][col];
            }
            EXPECT_NEAR(product, row == col ? Real(1) : Real(0), Real(1.0e-10));
        }
    }
}

TEST(ModalTransform, LazyInverseAccessorIsThreadSafe) {
    const int order = 5;
    HierarchicalBasis modal(ElementType::Line2, order);
    LagrangeBasis nodal(ElementType::Line2, order);
    ModalTransform transform(modal, nodal);

    constexpr std::size_t num_threads = 8;
    std::array<const std::vector<std::vector<Real>>*, num_threads> inverses{};
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (std::size_t i = 0; i < num_threads; ++i) {
        threads.emplace_back([&transform, &inverses, i]() {
            inverses[i] = &transform.vandermonde_inverse();
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    ASSERT_NE(inverses[0], nullptr);
    const auto& V = transform.vandermonde();
    const auto& Vinv = *inverses[0];
    ASSERT_EQ(V.size(), Vinv.size());
    for (std::size_t i = 1; i < num_threads; ++i) {
        EXPECT_EQ(inverses[i], inverses[0]);
    }
    for (std::size_t row = 0; row < V.size(); ++row) {
        ASSERT_EQ(V[row].size(), V.size());
        ASSERT_EQ(Vinv[row].size(), V.size());
    }
}

TEST(ModalTransform, CachedConstructionIsThreadSafe) {
    constexpr std::size_t num_threads = 8;
    constexpr int order = 4;
    const std::vector<Real> modal_coeffs{Real(0.25), Real(-0.5), Real(0.75),
                                         Real(0.1), Real(-0.2)};

    HierarchicalBasis baseline_modal(ElementType::Line2, order);
    LagrangeBasis baseline_nodal(ElementType::Line2, order);
    ModalTransform baseline_transform(baseline_modal, baseline_nodal);
    const auto expected = baseline_transform.nodal_to_modal(
        baseline_transform.modal_to_nodal(modal_coeffs));

    std::array<std::vector<Real>, num_threads> observed;
    std::array<std::exception_ptr, num_threads> errors{};
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (std::size_t thread_index = 0; thread_index < num_threads; ++thread_index) {
        threads.emplace_back([&, thread_index]() {
            try {
                HierarchicalBasis modal(ElementType::Line2, order);
                LagrangeBasis nodal(ElementType::Line2, order);
                ModalTransform transform(modal, nodal);
                observed[thread_index] =
                    transform.nodal_to_modal(transform.modal_to_nodal(modal_coeffs));
            } catch (...) {
                errors[thread_index] = std::current_exception();
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    for (std::size_t thread_index = 0; thread_index < num_threads; ++thread_index) {
        ASSERT_FALSE(errors[thread_index]);
        ASSERT_EQ(observed[thread_index].size(), expected.size());
        for (std::size_t i = 0; i < expected.size(); ++i) {
            EXPECT_NEAR(observed[thread_index][i], expected[i], Real(1e-10))
                << "thread=" << thread_index << " entry=" << i;
        }
    }
}

TEST(ModalTransform, ThrowsOnMismatchedSizes) {
    HierarchicalBasis modal(ElementType::Line2, 2);
    LagrangeBasis nodal(ElementType::Line2, 3); // different size
    EXPECT_THROW(ModalTransform bad(modal, nodal), svmp::FE::FEException);
}

TEST(ModalTransform, RoundTripOnTriangle) {
    const int order = 3;
    HierarchicalBasis modal(ElementType::Triangle3, order);
    LagrangeBasis nodal(ElementType::Triangle3, order);
    ModalTransform transform(modal, nodal);

    const std::size_t n = modal.size();
    std::vector<Real> modal_coeffs(n);
    for (std::size_t i = 0; i < n; ++i) {
        modal_coeffs[i] = static_cast<Real>(0.1 * (static_cast<double>(i) + 1.0));
    }
    auto recovered = transform.nodal_to_modal(transform.modal_to_nodal(modal_coeffs));
    ASSERT_EQ(modal_coeffs.size(), recovered.size());
    for (std::size_t i = 0; i < recovered.size(); ++i) {
        EXPECT_NEAR(modal_coeffs[i], recovered[i], 1e-9)
            << "Triangle order=" << order << ", i=" << i;
    }
}

TEST(ModalTransform, RoundTripOnTetra) {
    const int order = 2;
    HierarchicalBasis modal(ElementType::Tetra4, order);
    LagrangeBasis nodal(ElementType::Tetra4, order);
    ModalTransform transform(modal, nodal);

    const std::size_t n = modal.size();
    std::vector<Real> modal_coeffs(n);
    for (std::size_t i = 0; i < n; ++i) {
        modal_coeffs[i] = static_cast<Real>(0.05 * (static_cast<double>(i) + 1.0));
    }
    auto recovered = transform.nodal_to_modal(transform.modal_to_nodal(modal_coeffs));
    ASSERT_EQ(modal_coeffs.size(), recovered.size());
    for (std::size_t i = 0; i < recovered.size(); ++i) {
        EXPECT_NEAR(modal_coeffs[i], recovered[i], 1e-8)
            << "Tetra order=" << order << ", i=" << i;
    }
}

TEST(ModalTransform, RoundTripOnHex) {
    const int order = 2;
    HierarchicalBasis modal(ElementType::Hex8, order);
    LagrangeBasis nodal(ElementType::Hex8, order);
    ModalTransform transform(modal, nodal);

    const std::size_t n = modal.size();
    std::vector<Real> modal_coeffs(n);
    for (std::size_t i = 0; i < n; ++i) {
        modal_coeffs[i] = static_cast<Real>(0.05 * (static_cast<double>(i) + 1.0));
    }
    auto recovered = transform.nodal_to_modal(transform.modal_to_nodal(modal_coeffs));
    ASSERT_EQ(modal_coeffs.size(), recovered.size());
    for (std::size_t i = 0; i < recovered.size(); ++i) {
        EXPECT_NEAR(modal_coeffs[i], recovered[i], 1e-9)
            << "Hex order=" << order << ", i=" << i;
    }
}

TEST(ModalTransform, RoundTripOnWedge) {
    const int order = 2;
    HierarchicalBasis modal(ElementType::Wedge6, order);
    LagrangeBasis nodal(ElementType::Wedge6, order);
    ModalTransform transform(modal, nodal);

    const std::size_t n = modal.size();
    std::vector<Real> modal_coeffs(n);
    for (std::size_t i = 0; i < n; ++i) {
        modal_coeffs[i] = static_cast<Real>(0.1 * (static_cast<double>(i) + 1.0));
    }
    auto recovered = transform.nodal_to_modal(transform.modal_to_nodal(modal_coeffs));
    ASSERT_EQ(modal_coeffs.size(), recovered.size());
    for (std::size_t i = 0; i < recovered.size(); ++i) {
        EXPECT_NEAR(modal_coeffs[i], recovered[i], 1e-8)
            << "Wedge order=" << order << ", i=" << i;
    }
}

TEST(ModalTransform, ConditionNumberGrowsWithOrder) {
    // Condition number should increase with polynomial order but remain bounded
    Real prev_cond = Real(1);
    for (int order = 2; order <= 6; ++order) {
        HierarchicalBasis modal(ElementType::Line2, order);
        LagrangeBasis nodal(ElementType::Line2, order);
        ModalTransform transform(modal, nodal);
        const Real cond = transform.condition_number();
        EXPECT_GE(cond, Real(1));
        EXPECT_LT(cond, Real(1e8))
            << "Condition number too large at order " << order;
        if (order > 2) {
            EXPECT_GE(cond, prev_cond * Real(0.5))
                << "Unexpected conditioning decrease at order " << order;
        }
        prev_cond = cond;
    }
}

// =============================================================================
// Serendipity Gradient Tests
// =============================================================================

TEST(SerendipityBasis, Pyramid13GradientSumZero) {
    // For a partition of unity, sum of gradients should be zero
    SerendipityBasis basis(ElementType::Pyramid13, 2);

    // Test at multiple interior points
    std::vector<svmp::FE::math::Vector<Real, 3>> test_points = {
        {Real(0.0), Real(0.0), Real(0.5)},
        {Real(0.1), Real(-0.2), Real(0.4)},
        {Real(-0.3), Real(0.2), Real(0.3)},
        {Real(0.5), Real(0.5), Real(0.2)},
    };

    for (const auto& xi : test_points) {
        std::vector<Gradient> grads;
        basis.evaluate_gradients(xi, grads);
        ASSERT_EQ(grads.size(), 13u);

        Gradient sum{};
        for (const auto& g : grads) {
            sum[0] += g[0];
            sum[1] += g[1];
            sum[2] += g[2];
        }
        EXPECT_NEAR(sum[0], 0.0, 1e-10) << "Failed at xi=" << xi[0] << "," << xi[1] << "," << xi[2];
        EXPECT_NEAR(sum[1], 0.0, 1e-10) << "Failed at xi=" << xi[0] << "," << xi[1] << "," << xi[2];
        EXPECT_NEAR(sum[2], 0.0, 1e-10) << "Failed at xi=" << xi[0] << "," << xi[1] << "," << xi[2];
    }
}

TEST(SerendipityBasis, Pyramid13GradientMatchesNumerical) {
    // Compare analytical gradients against finite differences
    SerendipityBasis basis(ElementType::Pyramid13, 2);
    const Real h = Real(1e-6);

    svmp::FE::math::Vector<Real, 3> xi{Real(0.1), Real(-0.2), Real(0.4)};

    // Analytical gradients
    std::vector<Gradient> grads;
    basis.evaluate_gradients(xi, grads);

    // Numerical gradients via central differences
    for (int d = 0; d < 3; ++d) {
        svmp::FE::math::Vector<Real, 3> xi_plus = xi;
        svmp::FE::math::Vector<Real, 3> xi_minus = xi;
        xi_plus[static_cast<std::size_t>(d)] += h;
        xi_minus[static_cast<std::size_t>(d)] -= h;

        std::vector<Real> vals_plus, vals_minus;
        basis.evaluate_values(xi_plus, vals_plus);
        basis.evaluate_values(xi_minus, vals_minus);

        for (std::size_t i = 0; i < grads.size(); ++i) {
            Real numerical = (vals_plus[i] - vals_minus[i]) / (Real(2) * h);
            EXPECT_NEAR(grads[i][static_cast<std::size_t>(d)], numerical, 1e-6)
                << "Basis " << i << ", direction " << d;
        }
    }
}

TEST(SerendipityBasis, Pyramid13GradientAtApexThrowsBecauseLimitIsNotUnique) {
    SerendipityBasis basis(ElementType::Pyramid13, 2);

    const svmp::FE::math::Vector<Real, 3> apex{Real(0), Real(0), Real(1)};
    std::vector<Gradient> grads;
    EXPECT_THROW(basis.evaluate_gradients(apex, grads), BasisEvaluationException);
}

TEST(SerendipityBasis, Pyramid13GradientRemainsFiniteNearApexAcrossDirections) {
    SerendipityBasis basis(ElementType::Pyramid13, 2);

    const Real eps = Real(1e-5);
    const std::array<svmp::FE::math::Vector<Real, 3>, 4> near_points = {{
        {Real(0),           Real(0),           Real(1) - eps},
        {Real(0.45) * eps,  Real(-0.25) * eps, Real(1) - eps},
        {Real(-0.30) * eps, Real(0.35) * eps,  Real(1) - eps},
        {Real(0.20) * eps,  Real(0.40) * eps,  Real(1) - eps},
    }};

    for (const auto& xi : near_points) {
        std::vector<Real> vals;
        basis.evaluate_values(xi, vals);
        ASSERT_EQ(vals.size(), 13u);

        Real value_sum = Real(0);
        for (const Real v : vals) {
            EXPECT_TRUE(std::isfinite(static_cast<double>(v)));
            value_sum += v;
        }
        EXPECT_NEAR(value_sum, 1.0, 1e-10);

        std::vector<Gradient> grads;
        basis.evaluate_gradients(xi, grads);
        ASSERT_EQ(grads.size(), 13u);

        Gradient grad_sum{};
        for (const auto& g : grads) {
            EXPECT_TRUE(std::isfinite(static_cast<double>(g[0])));
            EXPECT_TRUE(std::isfinite(static_cast<double>(g[1])));
            EXPECT_TRUE(std::isfinite(static_cast<double>(g[2])));
            grad_sum[0] += g[0];
            grad_sum[1] += g[1];
            grad_sum[2] += g[2];
        }

        EXPECT_NEAR(grad_sum[0], 0.0, 1e-10);
        EXPECT_NEAR(grad_sum[1], 0.0, 1e-10);
        EXPECT_NEAR(grad_sum[2], 0.0, 1e-10);
    }
}

TEST(SerendipityBasis, Hex20GradientSumZero) {
    SerendipityBasis basis(ElementType::Hex20, 2);

    std::vector<svmp::FE::math::Vector<Real, 3>> test_points = {
        {Real(0.0), Real(0.0), Real(0.0)},
        {Real(0.2), Real(-0.1), Real(0.3)},
        {Real(-0.4), Real(0.5), Real(-0.2)},
    };

    for (const auto& xi : test_points) {
        std::vector<Gradient> grads;
        basis.evaluate_gradients(xi, grads);
        ASSERT_EQ(grads.size(), 20u);

        Gradient sum{};
        for (const auto& g : grads) {
            sum[0] += g[0];
            sum[1] += g[1];
            sum[2] += g[2];
        }
        EXPECT_NEAR(sum[0], 0.0, 1e-10);
        EXPECT_NEAR(sum[1], 0.0, 1e-10);
        EXPECT_NEAR(sum[2], 0.0, 1e-10);
    }
}

TEST(SerendipityBasis, Hex20GradientMatchesNumerical) {
    SerendipityBasis basis(ElementType::Hex20, 2);
    const Real h = Real(1e-6);

    svmp::FE::math::Vector<Real, 3> xi{Real(0.2), Real(-0.1), Real(0.3)};

    std::vector<Gradient> grads;
    basis.evaluate_gradients(xi, grads);

    for (int d = 0; d < 3; ++d) {
        svmp::FE::math::Vector<Real, 3> xi_plus = xi;
        svmp::FE::math::Vector<Real, 3> xi_minus = xi;
        xi_plus[static_cast<std::size_t>(d)] += h;
        xi_minus[static_cast<std::size_t>(d)] -= h;

        std::vector<Real> vals_plus, vals_minus;
        basis.evaluate_values(xi_plus, vals_plus);
        basis.evaluate_values(xi_minus, vals_minus);

        for (std::size_t i = 0; i < grads.size(); ++i) {
            Real numerical = (vals_plus[i] - vals_minus[i]) / (Real(2) * h);
            EXPECT_NEAR(grads[i][static_cast<std::size_t>(d)], numerical, 1e-6)
                << "Basis " << i << ", direction " << d;
        }
    }
}

TEST(SerendipityBasis, Wedge15GradientSumZero) {
    SerendipityBasis basis(ElementType::Wedge15, 2);

    std::vector<svmp::FE::math::Vector<Real, 3>> test_points = {
        {Real(0.2), Real(0.3), Real(0.0)},
        {Real(0.1), Real(0.1), Real(-0.5)},
        {Real(0.4), Real(0.2), Real(0.3)},
    };

    for (const auto& xi : test_points) {
        std::vector<Gradient> grads;
        basis.evaluate_gradients(xi, grads);
        ASSERT_EQ(grads.size(), 15u);

        Gradient sum{};
        for (const auto& g : grads) {
            sum[0] += g[0];
            sum[1] += g[1];
            sum[2] += g[2];
        }
        EXPECT_NEAR(sum[0], 0.0, 1e-10);
        EXPECT_NEAR(sum[1], 0.0, 1e-10);
        EXPECT_NEAR(sum[2], 0.0, 1e-10);
    }
}

TEST(SerendipityBasis, Wedge15GradientMatchesNumerical) {
    SerendipityBasis basis(ElementType::Wedge15, 2);
    const Real h = Real(1e-6);

    svmp::FE::math::Vector<Real, 3> xi{Real(0.2), Real(0.3), Real(0.1)};

    std::vector<Gradient> grads;
    basis.evaluate_gradients(xi, grads);

    for (int d = 0; d < 3; ++d) {
        svmp::FE::math::Vector<Real, 3> xi_plus = xi;
        svmp::FE::math::Vector<Real, 3> xi_minus = xi;
        xi_plus[static_cast<std::size_t>(d)] += h;
        xi_minus[static_cast<std::size_t>(d)] -= h;

        std::vector<Real> vals_plus, vals_minus;
        basis.evaluate_values(xi_plus, vals_plus);
        basis.evaluate_values(xi_minus, vals_minus);

        for (std::size_t i = 0; i < grads.size(); ++i) {
            Real numerical = (vals_plus[i] - vals_minus[i]) / (Real(2) * h);
            EXPECT_NEAR(grads[i][static_cast<std::size_t>(d)], numerical, 1e-6)
                << "Basis " << i << ", direction " << d;
        }
    }
}

TEST(SerendipityBasis, Quad8NodalValuesAndPartitionOfUnity) {
    SerendipityBasis basis(ElementType::Quad8, 2);
    const auto& nodes = basis.nodes();
    ASSERT_EQ(nodes.size(), 8u);

    for (std::size_t node = 0; node < nodes.size(); ++node) {
        std::vector<Real> values;
        basis.evaluate_values(nodes[node], values);
        ASSERT_EQ(values.size(), 8u);
        for (std::size_t i = 0; i < values.size(); ++i) {
            const Real expected = (i == node) ? Real(1) : Real(0);
            EXPECT_NEAR(values[i], expected, 1e-12) << "node=" << node << " basis=" << i;
        }
    }

    const std::vector<svmp::FE::math::Vector<Real, 3>> test_points = {
        {Real(0.0), Real(0.0), Real(0.0)},
        {Real(0.3), Real(-0.5), Real(0.0)},
        {Real(-0.7), Real(0.4), Real(0.0)},
    };
    for (const auto& xi : test_points) {
        std::vector<Real> values;
        basis.evaluate_values(xi, values);
        const Real sum = std::accumulate(values.begin(), values.end(), Real(0));
        EXPECT_NEAR(sum, Real(1), 1e-12);
    }
}

TEST(SerendipityBasis, Quad8GradientSumZero) {
    SerendipityBasis basis(ElementType::Quad8, 2);

    std::vector<svmp::FE::math::Vector<Real, 3>> test_points = {
        {Real(0.0), Real(0.0), Real(0.0)},
        {Real(0.3), Real(-0.5), Real(0.0)},
        {Real(-0.7), Real(0.4), Real(0.0)},
    };

    for (const auto& xi : test_points) {
        std::vector<Gradient> grads;
        basis.evaluate_gradients(xi, grads);
        ASSERT_EQ(grads.size(), 8u);

        Gradient sum{};
        for (const auto& g : grads) {
            sum[0] += g[0];
            sum[1] += g[1];
        }
        EXPECT_NEAR(sum[0], 0.0, 1e-10);
        EXPECT_NEAR(sum[1], 0.0, 1e-10);
    }
}

TEST(SerendipityBasis, Quad8HessianSumZero) {
    SerendipityBasis basis(ElementType::Quad8, 2);

    const std::vector<svmp::FE::math::Vector<Real, 3>> test_points = {
        {Real(0.0), Real(0.0), Real(0.0)},
        {Real(0.3), Real(-0.5), Real(0.0)},
        {Real(-0.7), Real(0.4), Real(0.0)},
    };

    for (const auto& xi : test_points) {
        std::vector<Hessian> hessians;
        basis.evaluate_hessians(xi, hessians);
        ASSERT_EQ(hessians.size(), 8u);

        Hessian sum{};
        for (const auto& hessian : hessians) {
            for (std::size_t row = 0; row < 3u; ++row) {
                for (std::size_t col = 0; col < 3u; ++col) {
                    sum(row, col) += hessian(row, col);
                }
            }
        }
        for (std::size_t row = 0; row < 3u; ++row) {
            for (std::size_t col = 0; col < 3u; ++col) {
                EXPECT_NEAR(sum(row, col), Real(0), 1e-12)
                    << "row=" << row << " col=" << col;
            }
        }
    }
}

TEST(SerendipityBasis, Quad8GradientMatchesNumerical) {
    SerendipityBasis basis(ElementType::Quad8, 2);
    const Real h = Real(1e-6);

    svmp::FE::math::Vector<Real, 3> xi{Real(0.3), Real(-0.5), Real(0.0)};

    std::vector<Gradient> grads;
    basis.evaluate_gradients(xi, grads);

    for (int d = 0; d < 2; ++d) {
        svmp::FE::math::Vector<Real, 3> xi_plus = xi;
        svmp::FE::math::Vector<Real, 3> xi_minus = xi;
        xi_plus[static_cast<std::size_t>(d)] += h;
        xi_minus[static_cast<std::size_t>(d)] -= h;

        std::vector<Real> vals_plus, vals_minus;
        basis.evaluate_values(xi_plus, vals_plus);
        basis.evaluate_values(xi_minus, vals_minus);

        for (std::size_t i = 0; i < grads.size(); ++i) {
            Real numerical = (vals_plus[i] - vals_minus[i]) / (Real(2) * h);
            EXPECT_NEAR(grads[i][static_cast<std::size_t>(d)], numerical, 1e-6)
                << "Basis " << i << ", direction " << d;
        }
    }
}
