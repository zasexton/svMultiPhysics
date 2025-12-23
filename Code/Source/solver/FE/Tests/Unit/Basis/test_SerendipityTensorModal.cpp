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

using namespace svmp::FE;
using namespace svmp::FE::basis;

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

TEST(SerendipityBasis, Hex20FieldBasisNodalAndPartition) {
    SerendipityBasis basis(ElementType::Hex20, 2); // field mode (geometry_mode = false)
    EXPECT_EQ(basis.size(), 20u);

    // Canonical Hex20 reference nodes (VTK/Mesh ordering).
    std::vector<svmp::FE::math::Vector<Real,3>> nodes;
    nodes.reserve(basis.size());
    for (std::size_t i = 0; i < basis.size(); ++i) {
        nodes.push_back(NodeOrdering::get_node_coords(ElementType::Hex20, i));
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

TEST(ModalTransform, ThrowsOnMismatchedSizes) {
    HierarchicalBasis modal(ElementType::Line2, 2);
    LagrangeBasis nodal(ElementType::Line2, 3); // different size
    EXPECT_THROW(ModalTransform bad(modal, nodal), svmp::FE::FEException);
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
