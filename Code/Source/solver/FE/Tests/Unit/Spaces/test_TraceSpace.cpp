/**
 * @file test_TraceSpace.cpp
 * @brief Unit tests for TraceSpace face restriction utilities
 */

#include <gtest/gtest.h>

#include "FE/Basis/NodeOrderingConventions.h"
#include "FE/Basis/SerendipityBasis.h"
#include "FE/Elements/ElementTransform.h"
#include "FE/Spaces/HCurlSpace.h"
#include "FE/Spaces/HDivSpace.h"
#include "FE/Quadrature/QuadratureFactory.h"
#include "FE/Spaces/SpaceFactory.h"
#include "FE/Spaces/H1Space.h"
#include "FE/Spaces/GenericBasisSpace.h"
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

FunctionSpace::Value normalized_direction(const FunctionSpace::Value& v) {
    FunctionSpace::Value out = v;
    out.normalize();
    return out;
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

TEST(TraceSpace, HigherOrderSerendipityQuadEdgeTraceMatchesVolume) {
    auto h1 = std::make_shared<H1Space>(ElementType::Quad4, 4, BasisType::Serendipity);
    TraceSpace trace(h1, /*face_id=*/0);

    EXPECT_EQ(trace.topological_dimension(), 1);
    EXPECT_EQ(trace.element_type(), ElementType::Line2);
    EXPECT_EQ(trace.dofs_per_element(), 5u);

    const auto face_dofs = trace.face_dof_indices();
    ASSERT_EQ(face_dofs.size(), 5u);
    EXPECT_EQ(face_dofs[0], 0);
    EXPECT_EQ(face_dofs[1], 1);
    EXPECT_EQ(face_dofs[2], 4);
    EXPECT_EQ(face_dofs[3], 5);
    EXPECT_EQ(face_dofs[4], 6);

    std::vector<Real> element_coeffs(h1->dofs_per_element(), Real(0));
    for (std::size_t i = 0; i < element_coeffs.size(); ++i) {
        element_coeffs[i] = Real(0.05) * Real(i + 1);
    }

    const auto face_coeffs = trace.restrict(element_coeffs);
    ASSERT_EQ(face_coeffs.size(), face_dofs.size());

    for (std::size_t i = 0; i < face_dofs.size(); ++i) {
        EXPECT_NEAR(face_coeffs[i], element_coeffs[static_cast<std::size_t>(face_dofs[i])], 1e-14);
    }

    for (Real x : {Real(-0.8), Real(-0.25), Real(0.1), Real(0.7)}) {
        FunctionSpace::Value xi_face{};
        xi_face[0] = x;
        const auto xi_vol = trace.embed_face_point(xi_face);
        const Real v_trace = trace.evaluate_scalar(xi_face, face_coeffs);
        const Real v_volume = h1->evaluate_scalar(xi_vol, element_coeffs);
        EXPECT_NEAR(v_trace, v_volume, 1e-12);
    }
}

TEST(TraceSpace, BsplineQuadEdgeTraceRespectsFaceOrientationAndMatchesVolume) {
    SpaceRequest req;
    req.space_type = SpaceType::H1;
    req.element.element_type = ElementType::Quad4;
    req.element.basis_type = BasisType::BSpline;
    req.element.field_type = FieldType::Scalar;
    req.element.continuity = Continuity::C0;
    req.element.order = 2;
    req.element.axis_orders = {2, 1};
    req.element.axis_knot_vectors = {
        {Real(0), Real(0), Real(0), Real(0.5), Real(1), Real(1), Real(1)},
        {Real(0), Real(0), Real(0.35), Real(1), Real(1)}
    };

    auto h1 = SpaceFactory::create(req);
    ASSERT_TRUE(h1);

    TraceSpace trace(h1, /*face_id=*/2);
    EXPECT_EQ(trace.topological_dimension(), 1);
    EXPECT_EQ(trace.element_type(), ElementType::Line2);
    EXPECT_EQ(trace.dofs_per_element(), 4u);

    const std::vector<int> expected_face_dofs{11, 10, 9, 8};
    EXPECT_EQ(trace.face_dof_indices(), expected_face_dofs);

    std::vector<Real> element_coeffs(h1->dofs_per_element(), Real(0));
    for (std::size_t i = 0; i < element_coeffs.size(); ++i) {
        element_coeffs[i] = Real(0.125) * Real(i + 1);
    }

    const auto face_coeffs = trace.restrict(element_coeffs);
    ASSERT_EQ(face_coeffs.size(), expected_face_dofs.size());
    for (std::size_t i = 0; i < face_coeffs.size(); ++i) {
        EXPECT_NEAR(face_coeffs[i],
                    element_coeffs[static_cast<std::size_t>(expected_face_dofs[i])],
                    1e-14);
    }

    for (Real x : {Real(-0.8), Real(-0.2), Real(0.25), Real(0.75)}) {
        FunctionSpace::Value xi_face{};
        xi_face[0] = x;
        const auto xi_volume = trace.embed_face_point(xi_face);
        const Real face_value = trace.evaluate_scalar(xi_face, face_coeffs);
        const Real volume_value = h1->evaluate_scalar(xi_volume, element_coeffs);
        EXPECT_NEAR(face_value, volume_value, 1e-12);
    }
}

TEST(TraceSpace, NurbsHexFaceTraceMatchesVolume) {
    SpaceRequest req;
    req.space_type = SpaceType::H1;
    req.element.element_type = ElementType::Hex8;
    req.element.basis_type = BasisType::NURBS;
    req.element.field_type = FieldType::Scalar;
    req.element.continuity = Continuity::C0;
    req.element.order = 2;
    req.element.axis_orders = {2, 1, 1};
    req.element.axis_knot_vectors = {
        {Real(0), Real(0), Real(0), Real(0.5), Real(1), Real(1), Real(1)},
        {Real(0), Real(0), Real(0.4), Real(1), Real(1)},
        {Real(0), Real(0), Real(1), Real(1)}
    };
    req.element.weights = {
        Real(1.0), Real(1.1), Real(0.9), Real(1.0),
        Real(1.0), Real(0.95), Real(1.05), Real(1.0),
        Real(1.0), Real(1.0), Real(1.15), Real(0.85),
        Real(1.0), Real(0.9), Real(1.0), Real(1.0),
        Real(1.0), Real(1.2), Real(0.8), Real(1.0),
        Real(1.0), Real(1.0), Real(1.1), Real(0.9)
    };
    req.element.tensor_extents = {4, 3, 2};

    auto h1 = SpaceFactory::create(req);
    ASSERT_TRUE(h1);

    TraceSpace trace(h1, /*face_id=*/3);
    EXPECT_EQ(trace.topological_dimension(), 2);
    EXPECT_EQ(trace.element_type(), ElementType::Quad4);
    EXPECT_EQ(trace.dofs_per_element(), 6u);

    const std::vector<int> expected_face_dofs{3, 7, 11, 15, 19, 23};
    EXPECT_EQ(trace.face_dof_indices(), expected_face_dofs);

    std::vector<Real> element_coeffs(h1->dofs_per_element(), Real(0));
    for (std::size_t i = 0; i < element_coeffs.size(); ++i) {
        element_coeffs[i] = Real(0.05) * Real(i + 1);
    }

    const auto face_coeffs = trace.restrict(element_coeffs);
    ASSERT_EQ(face_coeffs.size(), expected_face_dofs.size());
    for (std::size_t i = 0; i < face_coeffs.size(); ++i) {
        EXPECT_NEAR(face_coeffs[i],
                    element_coeffs[static_cast<std::size_t>(expected_face_dofs[i])],
                    1e-14);
    }

    const std::array<FunctionSpace::Value, 4> sample_points = {
        xi2(Real(-0.7), Real(-0.6)),
        xi2(Real(-0.1), Real(0.4)),
        xi2(Real(0.3), Real(-0.2)),
        xi2(Real(0.8), Real(0.7))
    };

    for (const auto& xi_face : sample_points) {
        const auto xi_volume = trace.embed_face_point(xi_face);
        const Real face_value = trace.evaluate_scalar(xi_face, face_coeffs);
        const Real volume_value = h1->evaluate_scalar(xi_volume, element_coeffs);
        EXPECT_NEAR(face_value, volume_value, 1e-12);
    }

    const auto xi_corner0 = trace.embed_face_point(xi2(Real(-1), Real(-1)));
    EXPECT_NEAR(xi_corner0[0], Real(1), 1e-14);
    EXPECT_NEAR(xi_corner0[1], Real(-1), 1e-14);
    EXPECT_NEAR(xi_corner0[2], Real(-1), 1e-14);

    const auto xi_corner1 = trace.embed_face_point(xi2(Real(1), Real(1)));
    EXPECT_NEAR(xi_corner1[0], Real(1), 1e-14);
    EXPECT_NEAR(xi_corner1[1], Real(1), 1e-14);
    EXPECT_NEAR(xi_corner1[2], Real(1), 1e-14);
}

TEST(TraceSpace, HigherOrderHCurlQuadEdgeTangentialTraceMatchesVolume) {
    auto hcurl = std::make_shared<HCurlSpace>(ElementType::Quad4, 2);
    TraceSpace trace(hcurl, /*face_id=*/2);

    EXPECT_EQ(trace.trace_kind(), TraceKind::Tangential);
    EXPECT_EQ(trace.field_type(), FieldType::Scalar);
    EXPECT_EQ(trace.value_dimension(), 1);
    EXPECT_EQ(trace.topological_dimension(), 1);
    EXPECT_EQ(trace.element_type(), ElementType::Line2);
    EXPECT_EQ(trace.dofs_per_element(), 3u);

    const std::vector<int> expected_face_dofs{6, 7, 8};
    EXPECT_EQ(trace.face_dof_indices(), expected_face_dofs);

    std::vector<Real> element_coeffs(hcurl->dofs_per_element(), Real(0));
    for (std::size_t i = 0; i < element_coeffs.size(); ++i) {
        element_coeffs[i] = Real(0.08) * Real(i + 1);
    }

    const auto face_coeffs = trace.restrict(element_coeffs);
    ASSERT_EQ(face_coeffs.size(), expected_face_dofs.size());
    for (std::size_t i = 0; i < face_coeffs.size(); ++i) {
        EXPECT_NEAR(face_coeffs[i],
                    element_coeffs[static_cast<std::size_t>(expected_face_dofs[i])],
                    1e-14);
    }

    const auto tangent = normalized_direction(
        trace.embed_face_point(FunctionSpace::Value{Real(1), Real(0), Real(0)}) -
        trace.embed_face_point(FunctionSpace::Value{Real(-1), Real(0), Real(0)}));

    for (Real x : {Real(-0.8), Real(-0.35), Real(0.1), Real(0.7)}) {
        FunctionSpace::Value xi_face{};
        xi_face[0] = x;
        const auto xi_volume = trace.embed_face_point(xi_face);
        const auto volume_value = hcurl->evaluate(xi_volume, element_coeffs);
        const Real expected = volume_value.dot(tangent);
        const Real face_value = trace.evaluate_scalar(xi_face, face_coeffs);
        const Real from_volume = trace.evaluate_from_face(xi_volume, face_coeffs)[0];
        EXPECT_NEAR(face_value, expected, 1e-12);
        EXPECT_NEAR(from_volume, expected, 1e-12);
    }
}

TEST(TraceSpace, CompatibleVectorNurbsQuadEdgeTraceMatchesVolume) {
    SpaceRequest req;
    req.space_type = SpaceType::HDiv;
    req.element.element_type = ElementType::Quad4;
    req.element.basis_type = BasisType::NURBS;
    req.element.field_type = FieldType::Vector;
    req.element.continuity = Continuity::H_div;
    req.element.order = 2;
    req.element.axis_orders = {2, 2};
    req.element.axis_knot_vectors = {
        {Real(0), Real(0), Real(0), Real(0.5), Real(1), Real(1), Real(1)},
        {Real(0), Real(0), Real(0), Real(0.5), Real(1), Real(1), Real(1)}
    };
    req.element.tensor_extents = {4, 4};
    req.element.weights.assign(16u, Real(1));
    req.element.weights[5] = Real(0.85);
    req.element.weights[10] = Real(1.2);

    auto hdiv = std::dynamic_pointer_cast<FunctionSpace>(SpaceFactory::create(req));
    ASSERT_TRUE(hdiv);

    TraceSpace trace(hdiv, /*face_id=*/0);
    EXPECT_EQ(trace.trace_kind(), TraceKind::Normal);
    EXPECT_EQ(trace.field_type(), FieldType::Scalar);
    EXPECT_EQ(trace.topological_dimension(), 1);
    EXPECT_EQ(trace.element_type(), ElementType::Line2);
    EXPECT_EQ(trace.dofs_per_element(), 3u);

    std::vector<Real> element_coeffs(hdiv->dofs_per_element(), Real(0));
    for (std::size_t i = 0; i < element_coeffs.size(); ++i) {
        element_coeffs[i] = Real(0.03) * Real(i + 1);
    }

    const auto face_coeffs = trace.restrict(element_coeffs);
    ASSERT_EQ(face_coeffs.size(), trace.dofs_per_element());

    auto normal = elements::ElementTransform::reference_facet_normal(ElementType::Quad4, 0);
    normal.normalize();

    for (Real x : {Real(-0.8), Real(-0.25), Real(0.15), Real(0.7)}) {
        FunctionSpace::Value xi_face{};
        xi_face[0] = x;
        const auto xi_volume = trace.embed_face_point(xi_face);
        const auto volume_value = hdiv->evaluate(xi_volume, element_coeffs);
        const Real expected = volume_value.dot(normal);
        const Real face_value = trace.evaluate_scalar(xi_face, face_coeffs);
        const Real from_volume = trace.evaluate_from_face(xi_volume, face_coeffs)[0];
        EXPECT_NEAR(face_value, expected, 1e-10);
        EXPECT_NEAR(from_volume, expected, 1e-10);
    }
}

TEST(TraceSpace, HigherOrderHDivHexFaceNormalTraceMatchesVolume) {
    auto hdiv = std::make_shared<HDivSpace>(ElementType::Hex8, 2);
    TraceSpace trace(hdiv, /*face_id=*/3);

    EXPECT_EQ(trace.trace_kind(), TraceKind::Normal);
    EXPECT_EQ(trace.field_type(), FieldType::Scalar);
    EXPECT_EQ(trace.value_dimension(), 1);
    EXPECT_EQ(trace.topological_dimension(), 2);
    EXPECT_EQ(trace.element_type(), ElementType::Quad4);
    EXPECT_EQ(trace.dofs_per_element(), 9u);

    const std::vector<int> expected_face_dofs{27, 28, 29, 30, 31, 32, 33, 34, 35};
    EXPECT_EQ(trace.face_dof_indices(), expected_face_dofs);

    std::vector<Real> element_coeffs(hdiv->dofs_per_element(), Real(0));
    for (std::size_t i = 0; i < element_coeffs.size(); ++i) {
        element_coeffs[i] = Real(-0.2) + Real(0.03) * Real(i);
    }

    const auto face_coeffs = trace.restrict(element_coeffs);
    ASSERT_EQ(face_coeffs.size(), expected_face_dofs.size());
    for (std::size_t i = 0; i < face_coeffs.size(); ++i) {
        EXPECT_NEAR(face_coeffs[i],
                    element_coeffs[static_cast<std::size_t>(expected_face_dofs[i])],
                    1e-14);
    }

    auto normal = elements::ElementTransform::reference_facet_normal(ElementType::Hex8, 3);
    normal.normalize();

    const std::array<FunctionSpace::Value, 4> sample_points = {
        xi2(Real(-0.7), Real(-0.6)),
        xi2(Real(-0.1), Real(0.4)),
        xi2(Real(0.3), Real(-0.2)),
        xi2(Real(0.85), Real(0.65))
    };

    for (const auto& xi_face : sample_points) {
        const auto xi_volume = trace.embed_face_point(xi_face);
        const auto volume_value = hdiv->evaluate(xi_volume, element_coeffs);
        const Real expected = volume_value.dot(normal);
        const Real face_value = trace.evaluate_scalar(xi_face, face_coeffs);
        const Real from_volume = trace.evaluate_from_face(xi_volume, face_coeffs)[0];
        EXPECT_NEAR(face_value, expected, 1e-12);
        EXPECT_NEAR(from_volume, expected, 1e-12);
    }
}

TEST(TraceSpace, HDivNormalTraceInterpolationMatchesPrescribedScalarField) {
    auto hdiv = std::make_shared<HDivSpace>(ElementType::Hex8, 2);
    TraceSpace trace(hdiv, /*face_id=*/3);

    std::vector<Real> coeffs;
    trace.interpolate(
        [](const FunctionSpace::Value& xi) {
            FunctionSpace::Value out{};
            out[0] = Real(1.25) + Real(0.4) * xi[0] - Real(0.2) * xi[1];
            return out;
        },
        coeffs);

    ASSERT_EQ(coeffs.size(), trace.dofs_per_element());

    auto quad = trace.element().quadrature();
    ASSERT_TRUE(quad);
    for (std::size_t q = 0; q < quad->num_points(); ++q) {
        const auto xi = quad->point(q);
        const Real expected = Real(1.25) + Real(0.4) * xi[0] - Real(0.2) * xi[1];
        const Real approx = trace.evaluate_scalar(xi, coeffs);
        EXPECT_NEAR(approx, expected, 1e-10);
    }
}

TEST(TraceSpace, HigherOrderHCurlTetraFaceTangentialTraceMatchesVolume) {
    auto hcurl = std::make_shared<HCurlSpace>(ElementType::Tetra4, 1);
    TraceSpace trace(hcurl, /*face_id=*/2);

    EXPECT_EQ(trace.trace_kind(), TraceKind::Tangential);
    EXPECT_EQ(trace.field_type(), FieldType::Vector);
    EXPECT_EQ(trace.value_dimension(), 3);
    EXPECT_EQ(trace.topological_dimension(), 2);
    EXPECT_EQ(trace.element_type(), ElementType::Triangle3);
    EXPECT_EQ(trace.dofs_per_element(), 8u);

    const std::vector<int> expected_face_dofs{2, 3, 8, 9, 10, 11, 16, 17};
    EXPECT_EQ(trace.face_dof_indices(), expected_face_dofs);

    std::vector<Real> element_coeffs(hcurl->dofs_per_element(), Real(0));
    for (std::size_t i = 0; i < element_coeffs.size(); ++i) {
        element_coeffs[i] = Real(0.04) * Real(i + 1);
    }

    const auto face_coeffs = trace.restrict(element_coeffs);
    ASSERT_EQ(face_coeffs.size(), trace.dofs_per_element());

    auto normal = elements::ElementTransform::reference_facet_normal(ElementType::Tetra4, 2);
    normal.normalize();

    const std::array<FunctionSpace::Value, 4> sample_points = {
        xi2(Real(0.1), Real(0.1)),
        xi2(Real(0.2), Real(0.3)),
        xi2(Real(0.45), Real(0.15)),
        xi2(Real(0.15), Real(0.55))
    };

    for (const auto& xi_face : sample_points) {
        const auto xi_volume = trace.embed_face_point(xi_face);
        const auto volume_value = hcurl->evaluate(xi_volume, element_coeffs);
        const auto face_value = trace.evaluate(xi_face, face_coeffs);
        const auto from_volume = trace.evaluate_from_face(xi_volume, face_coeffs);
        const auto expected = volume_value - normal * volume_value.dot(normal);

        EXPECT_NEAR(face_value[0], expected[0], 1e-12);
        EXPECT_NEAR(face_value[1], expected[1], 1e-12);
        EXPECT_NEAR(face_value[2], expected[2], 1e-12);
        EXPECT_NEAR(from_volume[0], expected[0], 1e-12);
        EXPECT_NEAR(from_volume[1], expected[1], 1e-12);
        EXPECT_NEAR(from_volume[2], expected[2], 1e-12);
        EXPECT_NEAR(face_value.dot(normal), Real(0), 1e-12);
    }
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
        const auto xi_face = basis::ReferenceNodeLayout::get_node_coords(ElementType::Quad4, i);
        const auto xi_vol = trace.embed_face_point(xi_face);
        const auto expected =
            basis::ReferenceNodeLayout::get_node_coords(h1->element_type(), static_cast<std::size_t>(fverts[i]));
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
    auto volume = std::make_shared<GenericBasisSpace>(vol_basis, vol_quad, FieldType::Scalar, Continuity::C0);

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

TEST(TraceSpace, OrientedHDivNormalTraceMapReversesHigherOrderEdgeTrace)
{
    const auto map = TraceSpace::orientedHDivNormalTraceMap(ElementType::Line2,
                                                            /*trace_polynomial_order=*/2,
                                                            /*edge_orientation=*/-1);

    ASSERT_EQ(map.source_indices.size(), 3u);
    ASSERT_EQ(map.weights.size(), 3u);

    EXPECT_EQ(map.source_indices[0], 2);
    EXPECT_EQ(map.source_indices[1], 1);
    EXPECT_EQ(map.source_indices[2], 0);

    EXPECT_DOUBLE_EQ(map.weights[0], -1.0);
    EXPECT_DOUBLE_EQ(map.weights[1], -1.0);
    EXPECT_DOUBLE_EQ(map.weights[2], -1.0);
}

TEST(TraceSpace, OrientedHDivNormalTraceMapRotatesQuadrilateralFaceTrace)
{
    spaces::OrientationManager::FaceOrientation orientation;
    orientation.rotation = 1;
    orientation.reflection = false;
    orientation.sign = +1;
    orientation.vertex_perm = {1, 2, 3, 0};

    const auto map = TraceSpace::orientedHDivNormalTraceMap(ElementType::Quad4,
                                                            /*trace_polynomial_order=*/1,
                                                            orientation);

    ASSERT_EQ(map.source_indices.size(), 4u);
    ASSERT_EQ(map.weights.size(), 4u);

    EXPECT_EQ(map.source_indices[0], 1);
    EXPECT_EQ(map.source_indices[1], 2);
    EXPECT_EQ(map.source_indices[2], 3);
    EXPECT_EQ(map.source_indices[3], 0);

    for (double weight : map.weights) {
        EXPECT_DOUBLE_EQ(weight, 1.0);
    }
}
