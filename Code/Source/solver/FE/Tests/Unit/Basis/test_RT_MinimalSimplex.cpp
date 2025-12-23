/**
 * @file test_RT_MinimalSimplex.cpp
 * @brief Minimal RT0 spaces on triangles and tetrahedra: edge/face flux DOFs
 */

#include <gtest/gtest.h>
#include "FE/Basis/VectorBasis.h"
#include "FE/Quadrature/GaussQuadrature.h"
#include "FE/Quadrature/TriangleQuadrature.h"
#include "FE/Quadrature/TetrahedronQuadrature.h"

using namespace svmp::FE;
using namespace svmp::FE::basis;
using namespace svmp::FE::quadrature;

namespace {

// Outward (unnormalized) normals for reference Triangle3 (v0=(0,0), v1=(1,0), v2=(0,1))
// matching the construction in VectorBasis.cpp
static svmp::FE::math::Vector<Real,3> triangle_edge_normal(int edge_id) {
    using svmp::FE::math::Vector;
    switch (edge_id) {
        case 0: return Vector<Real,3>{Real(1), Real(1), Real(0)};   // opposite v0
        case 1: return Vector<Real,3>{Real(-1), Real(0), Real(0)};  // opposite v1
        default: return Vector<Real,3>{Real(0), Real(-1), Real(0)}; // opposite v2
    }
}

// Simple parameterizations of triangle edges for flux integration
static svmp::FE::math::Vector<Real,3> triangle_edge_point(int edge_id, Real s) {
    using svmp::FE::math::Vector;
    switch (edge_id) {
        case 0: // v1 -> v2
            return Vector<Real,3>{Real(1) - s, s, Real(0)};
        case 1: // v2 -> v0
            return Vector<Real,3>{Real(0), Real(1) - s, Real(0)};
        default: // edge 2: v0 -> v1
            return Vector<Real,3>{s, Real(0), Real(0)};
    }
}

static Real triangle_edge_length(int edge_id) {
    if (edge_id == 0) {
        return Real(std::sqrt(2.0));
    }
    return Real(1);
}

// Outward normals for reference Tetra4 using grad(lambda_i)
static svmp::FE::math::Vector<Real,3> tetra_face_normal(int face_id) {
    using svmp::FE::math::Vector;
    switch (face_id) {
        case 0: return Vector<Real,3>{Real(-1), Real(-1), Real(-1)}; // opposite v0
        case 1: return Vector<Real,3>{Real(1), Real(0), Real(0)};    // opposite v1
        case 2: return Vector<Real,3>{Real(0), Real(1), Real(0)};    // opposite v2
        default: return Vector<Real,3>{Real(0), Real(0), Real(1)};   // opposite v3
    }
}

// Face quadrature: use existing tetrahedral quadrature restricted to faces via barycentric maps.
// For minimal RT0 tests we just need relative accuracy, not exactness for higher degree.

} // namespace

TEST(RaviartThomasBasis, TriangleMinimalEdgeFluxDofs) {
    RaviartThomasBasis rt(ElementType::Triangle3, 0);
    constexpr int n_edges = 3;

    // Integrate v_i · n_j over each edge using 1D Gauss quadrature.
    const int q_order = 4;
    GaussQuadrature1D quad(q_order);

    for (int i = 0; i < n_edges; ++i) {
        for (int j = 0; j < n_edges; ++j) {
            double flux = 0.0;
            const auto n = triangle_edge_normal(j);
            const Real L = triangle_edge_length(j);

            for (std::size_t q = 0; q < quad.num_points(); ++q) {
                const Real s = quad.point(q)[0] * Real(0.5) + Real(0.5); // map [-1,1] -> [0,1]
                auto x = triangle_edge_point(j, s);
                std::vector<svmp::FE::math::Vector<Real,3>> vals;
                rt.evaluate_vector_values(x, vals);
                const auto& v = vals[static_cast<std::size_t>(i)];
                const Real vn = v[0] * n[0] + v[1] * n[1] + v[2] * n[2];
                const Real ds = Real(0.5) * quad.weight(q) * L;
                flux += static_cast<double>(vn * ds);
            }

            if (i == j) {
                EXPECT_NEAR(flux, 1.0, 1e-12);
            } else {
                EXPECT_NEAR(flux, 0.0, 1e-12);
            }
        }
    }
}

TEST(RaviartThomasBasis, TetraMinimalFaceFluxDofs) {
    RaviartThomasBasis rt(ElementType::Tetra4, 0);
    constexpr int n_faces = 4;

    // For each face, we parameterize using barycentric coordinates of the
    // corresponding triangle and reuse TriangleQuadrature.
    const int tri_order = 4;
    TriangleQuadrature tri(tri_order);

    // Reference tetra vertices
    using svmp::FE::math::Vector;
    const Vector<Real,3> v0{Real(0), Real(0), Real(0)};
    const Vector<Real,3> v1{Real(1), Real(0), Real(0)};
    const Vector<Real,3> v2{Real(0), Real(1), Real(0)};
    const Vector<Real,3> v3{Real(0), Real(0), Real(1)};

    auto face_vertices = [&](int face_id) {
        switch (face_id) {
            case 0: return std::array<Vector<Real,3>,3>{v1, v2, v3};
            case 1: return std::array<Vector<Real,3>,3>{v0, v2, v3};
            case 2: return std::array<Vector<Real,3>,3>{v0, v3, v1};
            default: return std::array<Vector<Real,3>,3>{v0, v1, v2};
        }
    };

    for (int i = 0; i < n_faces; ++i) {
        for (int j = 0; j < n_faces; ++j) {
            double flux = 0.0;
            const auto n = tetra_face_normal(j);
            const auto verts = face_vertices(j);
            const Vector<Real,3> e1 = verts[1] - verts[0];
            const Vector<Real,3> e2 = verts[2] - verts[0];
            const Vector<Real,3> cross = e1.cross(e2); // not normalized

            for (std::size_t q = 0; q < tri.num_points(); ++q) {
                const auto& pt = tri.point(q);
                const Real r = pt[0];
                const Real s = pt[1];
                const Real l0 = Real(1) - r - s;
                const Real l1 = r;
                const Real l2 = s;
                Vector<Real,3> x =
                    verts[0] * l0 +
                    verts[1] * l1 +
                    verts[2] * l2;

                std::vector<Vector<Real,3>> vals;
                rt.evaluate_vector_values(x, vals);
                const auto& v = vals[static_cast<std::size_t>(i)];

                const Real vn = v[0] * n[0] + v[1] * n[1] + v[2] * n[2];
                const Real dA = tri.weight(q) *
                                std::sqrt(static_cast<double>(cross[0]*cross[0] +
                                                              cross[1]*cross[1] +
                                                              cross[2]*cross[2]));
                flux += static_cast<double>(vn * dA);
            }

            if (i == j) {
                EXPECT_NEAR(flux, 1.0, 1e-10);
            } else {
                EXPECT_NEAR(flux, 0.0, 1e-10);
            }
        }
    }
}
