/**
 * @file test_TraceSpace.cpp
 * @brief Unit tests for TraceSpace face restriction utilities
 */

#include <gtest/gtest.h>

#include "FE/Basis/NodeOrderingConventions.h"
#include "FE/Basis/SerendipityBasis.h"
#include "FE/Quadrature/QuadratureFactory.h"
#include "FE/Spaces/H1Space.h"
#include "FE/Spaces/IsogeometricSpace.h"
#include "FE/Spaces/TraceSpace.h"

using namespace svmp::FE;
using namespace svmp::FE::spaces;

namespace {

FunctionSpace::Value xi2(Real x, Real y) {
    FunctionSpace::Value xi{};
    xi[0] = x;
    xi[1] = y;
    xi[2] = Real(0);
    return xi;
}

} // namespace

TEST(TraceSpace, FaceDofIndicesAndRestrictScatterRoundTrip) {
    auto h1 = std::make_shared<H1Space>(ElementType::Quad4, 2);
    TraceSpace trace(h1, /*face_id=*/0);

    const auto face_dofs = trace.face_dof_indices();
    ASSERT_EQ(face_dofs.size(), 3u);
    EXPECT_EQ(face_dofs[0], 0);
    EXPECT_EQ(face_dofs[1], 1);
    EXPECT_EQ(face_dofs[2], 4);

    std::vector<Real> element_values(h1->dofs_per_element());
    for (std::size_t i = 0; i < element_values.size(); ++i) {
        element_values[i] = Real(10) + Real(i);
    }

    const auto face_values = trace.restrict(element_values);
    ASSERT_EQ(face_values.size(), face_dofs.size());
    for (std::size_t i = 0; i < face_dofs.size(); ++i) {
        EXPECT_NEAR(face_values[i], element_values[static_cast<std::size_t>(face_dofs[i])], 1e-14);
    }

    std::vector<Real> scattered(element_values.size(), Real(0));
    trace.scatter(face_values, scattered);
    for (std::size_t i = 0; i < scattered.size(); ++i) {
        if (i == 0 || i == 1 || i == 4) {
            EXPECT_NEAR(scattered[i], element_values[i], 1e-14);
        } else {
            EXPECT_NEAR(scattered[i], 0.0, 1e-14);
        }
    }
}

TEST(TraceSpace, EvaluateFromFaceMatchesVolumeOnFace) {
    auto h1 = std::make_shared<H1Space>(ElementType::Quad4, 2);
    TraceSpace trace(h1, /*face_id=*/0);

    std::vector<Real> element_coeffs(h1->dofs_per_element());
    for (std::size_t i = 0; i < element_coeffs.size(); ++i) {
        element_coeffs[i] = Real(0.1) * Real(i + 1);
    }

    const auto face_coeffs = trace.restrict(element_coeffs);

    const auto xi = xi2(Real(0.33), Real(-1));
    const auto v_full = h1->evaluate(xi, element_coeffs);
    const auto v_face = trace.evaluate_from_face(xi, face_coeffs);

    EXPECT_NEAR(v_face[0], v_full[0], 1e-12);
    EXPECT_NEAR(v_face[1], v_full[1], 1e-12);
    EXPECT_NEAR(v_face[2], v_full[2], 1e-12);
}

TEST(TraceSpace, FaceRestrictionAccessorMatchesVolumeConfig) {
    auto h1 = std::make_shared<H1Space>(ElementType::Quad4, 2);
    TraceSpace trace(h1, /*face_id=*/0);

    const auto& fr = trace.face_restriction();
    EXPECT_EQ(fr.element_type(), h1->element_type());
    EXPECT_EQ(fr.polynomial_order(), h1->polynomial_order());
    EXPECT_EQ(fr.continuity(), h1->continuity());
}

TEST(TraceSpace, PrototypeElementTypeAndDofsMatchFace) {
    {
        auto h1 = std::make_shared<H1Space>(ElementType::Quad4, 2);
        TraceSpace trace(h1, /*face_id=*/0);
        EXPECT_EQ(trace.topological_dimension(), 1);
        EXPECT_EQ(trace.element_type(), ElementType::Line3);
        EXPECT_EQ(trace.dofs_per_element(), 3u);
    }

    {
        auto h1 = std::make_shared<H1Space>(ElementType::Hex8, 2);
        TraceSpace trace(h1, /*face_id=*/0);
        EXPECT_EQ(trace.topological_dimension(), 2);
        EXPECT_EQ(trace.element_type(), ElementType::Quad9);
        EXPECT_EQ(trace.dofs_per_element(), 9u);
    }

    {
        auto h1 = std::make_shared<H1Space>(ElementType::Tetra4, 2);
        TraceSpace trace(h1, /*face_id=*/0);
        EXPECT_EQ(trace.topological_dimension(), 2);
        EXPECT_EQ(trace.element_type(), ElementType::Triangle6);
        EXPECT_EQ(trace.dofs_per_element(), 6u);
    }
}

TEST(TraceSpace, EmbedFacePointMatchesQuadFace0Mapping) {
    auto h1 = std::make_shared<H1Space>(ElementType::Quad4, 1);
    TraceSpace trace(h1, /*face_id=*/0);

    FunctionSpace::Value xi_face{};
    xi_face[0] = Real(-1);
    auto xi_vol = trace.embed_face_point(xi_face);
    EXPECT_NEAR(xi_vol[0], -1.0, 1e-14);
    EXPECT_NEAR(xi_vol[1], -1.0, 1e-14);

    xi_face[0] = Real(1);
    xi_vol = trace.embed_face_point(xi_face);
    EXPECT_NEAR(xi_vol[0], 1.0, 1e-14);
    EXPECT_NEAR(xi_vol[1], -1.0, 1e-14);

    xi_face[0] = Real(0);
    xi_vol = trace.embed_face_point(xi_face);
    EXPECT_NEAR(xi_vol[0], 0.0, 1e-14);
    EXPECT_NEAR(xi_vol[1], -1.0, 1e-14);
}

TEST(TraceSpace, EmbedFaceVerticesMatchVolumeVerticesHex) {
    auto h1 = std::make_shared<H1Space>(ElementType::Hex8, 1);
    const int face_id = 0;
    TraceSpace trace(h1, face_id);

    const auto& topo = trace.face_restriction().topology();
    const auto& fverts = topo.face_vertices[static_cast<std::size_t>(face_id)];
    ASSERT_EQ(fverts.size(), 4u);

    for (std::size_t i = 0; i < fverts.size(); ++i) {
        const auto xi_face = basis::NodeOrdering::get_node_coords(ElementType::Quad4, i);
        const auto xi_vol = trace.embed_face_point(xi_face);
        const auto expected =
            basis::NodeOrdering::get_node_coords(h1->element_type(), static_cast<std::size_t>(fverts[i]));
        EXPECT_NEAR(xi_vol[0], expected[0], 1e-14);
        EXPECT_NEAR(xi_vol[1], expected[1], 1e-14);
        EXPECT_NEAR(xi_vol[2], expected[2], 1e-14);
    }
}

TEST(TraceSpace, InterpolateQuadraticPolynomialOnLineIsExact) {
    auto h1 = std::make_shared<H1Space>(ElementType::Quad4, 2);
    TraceSpace trace(h1, /*face_id=*/0);

    FunctionSpace::ValueFunction f = [](const FunctionSpace::Value& xi) {
        const Real x = xi[0];
        FunctionSpace::Value out{};
        out[0] = Real(1.25) + Real(2.0) * x - Real(0.5) * x * x;
        return out;
    };

    std::vector<Real> coeffs;
    trace.interpolate(f, coeffs);
    ASSERT_EQ(coeffs.size(), trace.dofs_per_element());

    auto check = [&](Real x) {
        FunctionSpace::Value xi{};
        xi[0] = x;
        const Real expected = Real(1.25) + Real(2.0) * x - Real(0.5) * x * x;
        const Real approx = trace.evaluate_scalar(xi, coeffs);
        EXPECT_NEAR(approx, expected, 1e-12);
    };

    check(Real(-0.7));
    check(Real(-0.2));
    check(Real(0.0));
    check(Real(0.3));
    check(Real(0.9));
}

TEST(TraceSpace, InterpolateQuadraticPolynomialOnTriangleIsExact) {
    auto h1 = std::make_shared<H1Space>(ElementType::Tetra4, 2);
    TraceSpace trace(h1, /*face_id=*/0);

    auto poly = [](const FunctionSpace::Value& xi) {
        const Real x = xi[0];
        const Real y = xi[1];
        return Real(1.0) + Real(0.2) * x + Real(0.3) * y +
               Real(0.1) * x * x + Real(0.05) * y * y + Real(0.4) * x * y;
    };

    FunctionSpace::ValueFunction f = [&](const FunctionSpace::Value& xi) {
        FunctionSpace::Value out{};
        out[0] = poly(xi);
        return out;
    };

    std::vector<Real> coeffs;
    trace.interpolate(f, coeffs);
    ASSERT_EQ(coeffs.size(), trace.dofs_per_element());

    const std::array<FunctionSpace::Value, 4> sample_points = {
        xi2(Real(0.1), Real(0.1)),
        xi2(Real(0.2), Real(0.3)),
        xi2(Real(0.4), Real(0.2)),
        xi2(Real(0.05), Real(0.6))
    };

    for (const auto& xi : sample_points) {
        ASSERT_LE(xi[0] + xi[1], Real(1.0));
        const Real expected = poly(xi);
        const Real approx = trace.evaluate_scalar(xi, coeffs);
        EXPECT_NEAR(approx, expected, 1e-12);
    }
}

TEST(TraceSpace, LiftedCoefficientsMatchVolumeTraceOnFace) {
    auto h1 = std::make_shared<H1Space>(ElementType::Hex8, 2);
    TraceSpace trace(h1, /*face_id=*/0);

    std::vector<Real> elem_coeffs(h1->dofs_per_element(), Real(0));
    for (std::size_t i = 0; i < elem_coeffs.size(); ++i) {
        elem_coeffs[i] = Real(0.01) * Real(i + 1);
    }

    const auto face_coeffs = trace.restrict(elem_coeffs);
    const auto lifted = trace.lift(face_coeffs);

    const std::array<FunctionSpace::Value, 3> face_points = {
        xi2(Real(-0.3), Real(0.2)),
        xi2(Real(0.1), Real(-0.4)),
        xi2(Real(0.75), Real(0.6))
    };

    for (const auto& xi_face : face_points) {
        const auto xi_vol = trace.embed_face_point(xi_face);
        const Real v_full = h1->evaluate_scalar(xi_vol, elem_coeffs);
        const Real v_lift = h1->evaluate_scalar(xi_vol, lifted);
        const Real v_face = trace.evaluate_scalar(xi_face, face_coeffs);
        EXPECT_NEAR(v_lift, v_full, 1e-12);
        EXPECT_NEAR(v_face, v_full, 1e-12);
    }
}

TEST(TraceSpace, SerendipityQuadFaceInterpolationRecoversCoefficients) {
    auto vol_basis = std::make_shared<basis::SerendipityBasis>(ElementType::Hex20, 2, false);
    const int qord = quadrature::QuadratureFactory::recommended_order(2, false);
    auto vol_quad = quadrature::QuadratureFactory::create(ElementType::Hex20, qord);
    auto volume = std::make_shared<IsogeometricSpace>(vol_basis, vol_quad, FieldType::Scalar, Continuity::C0);

    TraceSpace trace(volume, /*face_id=*/0);
    EXPECT_EQ(trace.element_type(), ElementType::Quad8);
    ASSERT_EQ(trace.dofs_per_element(), 8u);

    std::vector<Real> coeffs_known(trace.dofs_per_element(), Real(0));
    for (std::size_t i = 0; i < coeffs_known.size(); ++i) {
        coeffs_known[i] = Real(0.1) * Real(i + 1);
    }

    FunctionSpace::ValueFunction f = [&](const FunctionSpace::Value& xi_face) {
        return trace.evaluate(xi_face, coeffs_known);
    };

    std::vector<Real> coeffs_interp;
    trace.interpolate(f, coeffs_interp);
    ASSERT_EQ(coeffs_interp.size(), coeffs_known.size());

    for (std::size_t i = 0; i < coeffs_known.size(); ++i) {
        EXPECT_NEAR(coeffs_interp[i], coeffs_known[i], 1e-12);
    }
}
