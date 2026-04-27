/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "FE/Basis/LagrangeBasis.h"
#include "FE/Geometry/FrameGeometry.h"
#include "FE/Quadrature/QuadratureFactory.h"

#include <algorithm>
#include <array>
#include <cmath>

using namespace svmp::FE;
using namespace svmp::FE::geometry;

namespace {

void expectMatrixNear(const Matrix3x3& actual, const Matrix3x3& expected, Real tol)
{
    for (std::size_t r = 0; r < 3u; ++r) {
        for (std::size_t c = 0; c < 3u; ++c) {
            EXPECT_NEAR(actual[r][c], expected[r][c], tol);
        }
    }
}

} // namespace

TEST(FrameGeometry, LineCellFrameAndCoordinateSensitivities)
{
    const std::vector<Point3D> nodes{
        {0.0, 0.0, 0.0},
        {2.0, 0.0, 0.0},
    };
    const auto quad = quadrature::QuadratureFactory::create(ElementType::Line2, 2);

    const auto frame = evaluateCellFrame(ElementType::Line2, *quad, nodes);
    ASSERT_EQ(frame.points.size(), quad->num_points());
    ASSERT_EQ(frame.measures.size(), quad->num_points());

    const auto xi0 = quad->point(0);
    EXPECT_NEAR(frame.points[0][0], xi0[0] + Real(1), 1e-12);
    EXPECT_NEAR(frame.measures[0], Real(1), 1e-12);

    const auto sensitivity =
        finiteDifferenceCellGeometrySensitivity(ElementType::Line2, *quad, nodes);
    const Real N0 = Real(0.5) * (Real(1) - xi0[0]);
    const auto& dx_du0x = sensitivity.physical_points.at(0, 0, 0);
    const auto& dx_du0y = sensitivity.physical_points.at(0, 0, 1);
    EXPECT_NEAR(dx_du0x[0], N0, 5e-9);
    EXPECT_NEAR(dx_du0x[1], 0.0, 5e-9);
    EXPECT_NEAR(dx_du0y[0], 0.0, 5e-9);
    EXPECT_NEAR(dx_du0y[1], N0, 5e-9);
}

TEST(FrameGeometry, NonAffineQuadCellMeasuresVary)
{
    const std::vector<Point3D> nodes{
        {0.0, 0.0, 0.0},
        {2.0, 0.0, 0.0},
        {1.5, 1.0, 0.0},
        {0.0, 2.0, 0.0},
    };
    const auto quad = quadrature::QuadratureFactory::create(ElementType::Quad4, 3);

    const auto frame = evaluateCellFrame(ElementType::Quad4, *quad, nodes);
    ASSERT_GE(frame.measures.size(), 2u);

    Real min_measure = frame.measures.front();
    Real max_measure = frame.measures.front();
    for (const Real measure : frame.measures) {
        min_measure = std::min(min_measure, measure);
        max_measure = std::max(max_measure, measure);
        EXPECT_GT(measure, 0.0);
    }
    EXPECT_GT(max_measure - min_measure, 1e-3);

    const auto face_quad = quadrature::QuadratureFactory::create(ElementType::Line2, 2);
    const auto face = evaluateFaceFrame(ElementType::Quad4,
                                        /*local_face_id=*/1,
                                        ElementType::Line2,
                                        *face_quad,
                                        nodes);
    ASSERT_EQ(face.surface_jacobians.size(), face_quad->num_points());
    EXPECT_GT(face.surface_measures[0], 0.0);
    EXPECT_NEAR(face.normals[0][2], 0.0, 1e-12);
}

TEST(FrameGeometry, TetraFaceGeometryUsesSurfaceMeasureAndOutwardNormal)
{
    const std::vector<Point3D> nodes{
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    };
    const auto face_quad = quadrature::QuadratureFactory::create(ElementType::Triangle3, 2);

    const auto face = evaluateFaceFrame(ElementType::Tetra4,
                                        /*local_face_id=*/0,
                                        ElementType::Triangle3,
                                        *face_quad,
                                        nodes);
    ASSERT_EQ(face.surface_measures.size(), face_quad->num_points());

    Real integrated_area = 0.0;
    for (std::size_t q = 0; q < face_quad->num_points(); ++q) {
        integrated_area += face_quad->weight(q) * face.surface_measures[q];
        EXPECT_NEAR(face.normals[q][0], 0.0, 1e-12);
        EXPECT_NEAR(face.normals[q][1], 0.0, 1e-12);
        EXPECT_NEAR(face.normals[q][2], -1.0, 1e-12);
    }
    EXPECT_NEAR(integrated_area, 0.5, 1e-12);
}

TEST(FrameGeometry, NansonSurfaceTransformScalesMeasures)
{
    Matrix3x3 F{};
    F[0][0] = 2.0;
    F[1][1] = 3.0;
    F[2][2] = 4.0;

    const auto surface = nansonSurfaceTransform(Vector3D{0.0, 0.0, 1.0},
                                                /*reference_measure=*/5.0,
                                                F);
    EXPECT_NEAR(surface.measure, 30.0, 1e-12);
    EXPECT_NEAR(surface.normal[0], 0.0, 1e-12);
    EXPECT_NEAR(surface.normal[1], 0.0, 1e-12);
    EXPECT_NEAR(surface.normal[2], 1.0, 1e-12);
    EXPECT_NEAR(surface.oriented_measure_vector[2], 30.0, 1e-12);
}

TEST(FrameGeometry, ConfigurationTransformMultipliesCurrentAndReferenceInverse)
{
    Matrix3x3 J_cur{};
    J_cur[0][0] = 4.0;
    J_cur[1][1] = 9.0;
    J_cur[2][2] = 16.0;

    Matrix3x3 J_ref_inv{};
    J_ref_inv[0][0] = 0.5;
    J_ref_inv[1][1] = 1.0 / 3.0;
    J_ref_inv[2][2] = 0.25;

    Matrix3x3 expected{};
    expected[0][0] = 2.0;
    expected[1][1] = 3.0;
    expected[2][2] = 4.0;

    expectMatrixNear(configurationTransform(J_cur, J_ref_inv), expected, 1e-12);
}

TEST(FrameGeometry, FaceSensitivitiesIgnoreOppositeTetraNodeForFaceZero)
{
    const std::vector<Point3D> nodes{
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    };
    const auto face_quad = quadrature::QuadratureFactory::create(ElementType::Triangle3, 2);

    const auto sensitivity =
        finiteDifferenceFaceGeometrySensitivity(ElementType::Tetra4,
                                                /*local_face_id=*/0,
                                                ElementType::Triangle3,
                                                *face_quad,
                                                nodes);

    for (LocalIndex q = 0; q < static_cast<LocalIndex>(face_quad->num_points()); ++q) {
        for (int component = 0; component < 3; ++component) {
            const auto& dn = sensitivity.normals.at(q, /*node=*/3, component);
            EXPECT_NEAR(dn[0], 0.0, 1e-10);
            EXPECT_NEAR(dn[1], 0.0, 1e-10);
            EXPECT_NEAR(dn[2], 0.0, 1e-10);
            EXPECT_NEAR(sensitivity.measures.at(q, /*node=*/3, component), 0.0, 1e-10);
        }
    }
}

TEST(FrameGeometry, CellGeometrySensitivityMatchesFiniteDifferenceForTetra)
{
    const std::vector<Point3D> nodes{
        {0.0, 0.0, 0.0},
        {1.2, 0.1, 0.0},
        {0.2, 1.1, 0.1},
        {0.1, 0.2, 1.3},
    };
    const auto quad = quadrature::QuadratureFactory::create(ElementType::Tetra4, 2);

    const auto analytic = evaluateCellGeometrySensitivity(ElementType::Tetra4, *quad, nodes);
    const auto fd =
        finiteDifferenceCellGeometrySensitivity(ElementType::Tetra4, *quad, nodes, 1e-7);

    ASSERT_EQ(analytic.n_qpts, fd.n_qpts);
    ASSERT_EQ(analytic.n_nodes, fd.n_nodes);
    for (LocalIndex q = 0; q < analytic.n_qpts; ++q) {
        for (LocalIndex node = 0; node < analytic.n_nodes; ++node) {
            for (int component = 0; component < 3; ++component) {
                const auto& dx = analytic.physical_points.at(q, node, component);
                const auto& dx_fd = fd.physical_points.at(q, node, component);
                const auto& dJ = analytic.jacobians.at(q, node, component);
                const auto& dJ_fd = fd.jacobians.at(q, node, component);
                const auto& dJinv = analytic.inverse_jacobians.at(q, node, component);
                const auto& dJinv_fd = fd.inverse_jacobians.at(q, node, component);
                for (std::size_t d = 0; d < 3u; ++d) {
                    EXPECT_NEAR(dx[d], dx_fd[d], 2e-8);
                }
                expectMatrixNear(dJ, dJ_fd, 2e-8);
                expectMatrixNear(dJinv, dJinv_fd, 2e-7);
                EXPECT_NEAR(analytic.measures.at(q, node, component),
                            fd.measures.at(q, node, component),
                            2e-8);
            }
        }
    }
}

TEST(FrameGeometry, CellGeometrySensitivityCoversTriangleAndQuadTranslationModes)
{
    struct Case {
        ElementType type;
        std::vector<Point3D> nodes;
    };
    const std::array<Case, 2> cases{{
        {ElementType::Triangle3,
         {{0.0, 0.0, 0.0},
          {1.3, 0.1, 0.0},
          {0.2, 1.4, 0.0}}},
        {ElementType::Quad4,
         {{0.0, 0.0, 0.0},
          {1.5, 0.1, 0.0},
          {1.2, 1.1, 0.0},
          {-0.1, 1.3, 0.0}}},
    }};

    for (const auto& c : cases) {
        const auto quad = quadrature::QuadratureFactory::create(c.type, 2);
        const auto sensitivity = evaluateCellGeometrySensitivity(c.type, *quad, c.nodes);
        ASSERT_EQ(sensitivity.n_qpts, static_cast<LocalIndex>(quad->num_points()));
        ASSERT_EQ(sensitivity.n_nodes, static_cast<LocalIndex>(c.nodes.size()));

        for (LocalIndex q = 0; q < sensitivity.n_qpts; ++q) {
            for (int component = 0; component < 2; ++component) {
                Vector3D translated_point_derivative{};
                Real translated_measure_derivative = 0.0;
                for (LocalIndex node = 0; node < sensitivity.n_nodes; ++node) {
                    const auto& dx = sensitivity.physical_points.at(q, node, component);
                    for (std::size_t d = 0; d < 3u; ++d) {
                        translated_point_derivative[d] += dx[d];
                    }
                    translated_measure_derivative += sensitivity.measures.at(q, node, component);
                }
                EXPECT_NEAR(translated_point_derivative[static_cast<std::size_t>(component)],
                            1.0,
                            5.0e-8);
                EXPECT_NEAR(translated_point_derivative[static_cast<std::size_t>(1 - component)],
                            0.0,
                            5.0e-8);
                EXPECT_NEAR(translated_measure_derivative, 0.0, 5.0e-8);
            }
        }
    }
}

TEST(FrameGeometry, FaceGeometrySensitivityAnalyticMatchesFiniteDifferenceReference)
{
    const std::vector<Point3D> nodes{
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    };
    const auto face_quad = quadrature::QuadratureFactory::create(ElementType::Triangle3, 2);

    const auto evaluated = evaluateFaceGeometrySensitivity(ElementType::Tetra4,
                                                           /*local_face_id=*/0,
                                                           ElementType::Triangle3,
                                                           *face_quad,
                                                           nodes);
    const auto fd = finiteDifferenceFaceGeometrySensitivity(ElementType::Tetra4,
                                                            /*local_face_id=*/0,
                                                            ElementType::Triangle3,
                                                            *face_quad,
                                                            nodes);

    ASSERT_EQ(evaluated.n_qpts, fd.n_qpts);
    ASSERT_EQ(evaluated.n_nodes, fd.n_nodes);
    constexpr Real fd_reference_tol = 2.0e-9;
    for (LocalIndex q = 0; q < evaluated.n_qpts; ++q) {
        for (LocalIndex node = 0; node < evaluated.n_nodes; ++node) {
            for (int component = 0; component < 3; ++component) {
                const auto& dn = evaluated.normals.at(q, node, component);
                const auto& dn_fd = fd.normals.at(q, node, component);
                for (std::size_t d = 0; d < 3u; ++d) {
                    EXPECT_NEAR(dn[d], dn_fd[d], fd_reference_tol);
                }
                EXPECT_NEAR(evaluated.measures.at(q, node, component),
                            fd.measures.at(q, node, component),
                            fd_reference_tol);
            }
        }
    }
}

TEST(FrameGeometry, QuadFaceGeometrySensitivityMatchesFiniteDifferenceAndTranslationMode)
{
    const std::vector<Point3D> nodes{
        {0.0, 0.0, 0.0},
        {1.5, 0.2, 0.0},
        {1.3, 1.1, 0.0},
        {-0.1, 1.4, 0.0},
    };
    const auto face_quad = quadrature::QuadratureFactory::create(ElementType::Line2, 2);

    const auto evaluated = evaluateFaceGeometrySensitivity(ElementType::Quad4,
                                                           /*local_face_id=*/0,
                                                           ElementType::Line2,
                                                           *face_quad,
                                                           nodes);
    const auto fd = finiteDifferenceFaceGeometrySensitivity(ElementType::Quad4,
                                                            /*local_face_id=*/0,
                                                            ElementType::Line2,
                                                            *face_quad,
                                                            nodes);

    ASSERT_EQ(evaluated.n_qpts, fd.n_qpts);
    ASSERT_EQ(evaluated.n_nodes, fd.n_nodes);
    constexpr Real fd_reference_tol = 2.0e-9;
    for (LocalIndex q = 0; q < evaluated.n_qpts; ++q) {
        for (LocalIndex node = 0; node < evaluated.n_nodes; ++node) {
            for (int component = 0; component < 3; ++component) {
                const auto& dn = evaluated.normals.at(q, node, component);
                const auto& dn_fd = fd.normals.at(q, node, component);
                for (std::size_t d = 0; d < 3u; ++d) {
                    EXPECT_NEAR(dn[d], dn_fd[d], fd_reference_tol);
                }
                EXPECT_NEAR(evaluated.measures.at(q, node, component),
                            fd.measures.at(q, node, component),
                            fd_reference_tol);
            }
        }

        constexpr Real translation_tol = 5.0e-9;
        for (int component = 0; component < 2; ++component) {
            Vector3D translated_normal_derivative{};
            Real translated_measure_derivative = 0.0;
            for (LocalIndex node = 0; node < evaluated.n_nodes; ++node) {
                const auto& dn = evaluated.normals.at(q, node, component);
                for (std::size_t d = 0; d < 3u; ++d) {
                    translated_normal_derivative[d] += dn[d];
                }
                translated_measure_derivative += evaluated.measures.at(q, node, component);
            }
            EXPECT_NEAR(translated_normal_derivative[0], 0.0, translation_tol);
            EXPECT_NEAR(translated_normal_derivative[1], 0.0, translation_tol);
            EXPECT_NEAR(translated_normal_derivative[2], 0.0, translation_tol);
            EXPECT_NEAR(translated_measure_derivative, 0.0, translation_tol);
        }
    }
}

TEST(FrameGeometry, CurvedQuad9CurrentReferenceBoundaryNormalsAndMeasuresChange)
{
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 2);
    std::vector<Point3D> reference_nodes;
    reference_nodes.reserve(basis->nodes().size());
    for (const auto& node : basis->nodes()) {
        reference_nodes.push_back({node[0], node[1], node[2]});
    }
    auto current_nodes = reference_nodes;

    std::size_t bottom_mid = current_nodes.size();
    std::size_t center = current_nodes.size();
    for (std::size_t i = 0; i < current_nodes.size(); ++i) {
        if (std::abs(current_nodes[i][0]) < 1.0e-12 &&
            std::abs(current_nodes[i][1] + 1.0) < 1.0e-12) {
            bottom_mid = i;
        }
        if (std::abs(current_nodes[i][0]) < 1.0e-12 &&
            std::abs(current_nodes[i][1]) < 1.0e-12) {
            center = i;
        }
    }
    ASSERT_LT(bottom_mid, current_nodes.size());
    ASSERT_LT(center, current_nodes.size());

    current_nodes[bottom_mid][1] += 0.35;
    current_nodes[bottom_mid][2] += 0.10;
    current_nodes[center][2] += 0.40;

    const auto face_quad = quadrature::QuadratureFactory::create(ElementType::Line2, 3);
    const auto reference_face = evaluateFaceFrame(ElementType::Quad9,
                                                  /*local_face_id=*/0,
                                                  ElementType::Line2,
                                                  *face_quad,
                                                  reference_nodes);
    const auto current_face = evaluateFaceFrame(ElementType::Quad9,
                                                /*local_face_id=*/0,
                                                ElementType::Line2,
                                                *face_quad,
                                                current_nodes);

    ASSERT_EQ(reference_face.normals.size(), current_face.normals.size());
    ASSERT_EQ(reference_face.surface_measures.size(), current_face.surface_measures.size());
    bool saw_normal_change = false;
    bool saw_measure_change = false;
    for (std::size_t q = 0; q < reference_face.normals.size(); ++q) {
        Real normal_diff = 0.0;
        for (std::size_t d = 0; d < 3u; ++d) {
            normal_diff += std::abs(reference_face.normals[q][d] - current_face.normals[q][d]);
        }
        saw_normal_change = saw_normal_change || normal_diff > 1.0e-6;
        saw_measure_change =
            saw_measure_change ||
            std::abs(reference_face.surface_measures[q] - current_face.surface_measures[q]) > 1.0e-6;
        EXPECT_GT(current_face.surface_measures[q], 0.0);
    }
    EXPECT_TRUE(saw_normal_change);
    EXPECT_TRUE(saw_measure_change);
}

TEST(FrameGeometry, Quad9GeometrySensitivityCoversHighOrderControlPoints)
{
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 2);
    std::vector<Point3D> nodes;
    nodes.reserve(basis->nodes().size());
    for (const auto& node : basis->nodes()) {
        nodes.push_back({node[0], node[1], node[2]});
    }
    for (auto& node : nodes) {
        if (std::abs(node[0]) < 1.0e-12 && std::abs(node[1]) < 1.0e-12) {
            node[2] = 0.25;
        }
    }

    const auto quad = quadrature::QuadratureFactory::create(ElementType::Quad4, 3);
    const auto evaluated = evaluateCellGeometrySensitivity(ElementType::Quad9, *quad, nodes);
    const auto fd = finiteDifferenceCellGeometrySensitivity(ElementType::Quad9, *quad, nodes, 1.0e-7);

    ASSERT_EQ(evaluated.n_qpts, fd.n_qpts);
    ASSERT_EQ(evaluated.n_nodes, static_cast<LocalIndex>(nodes.size()));
    ASSERT_EQ(evaluated.n_nodes, fd.n_nodes);
    for (LocalIndex q = 0; q < evaluated.n_qpts; ++q) {
        for (LocalIndex node = 0; node < evaluated.n_nodes; ++node) {
            for (int component = 0; component < 3; ++component) {
                const auto& dx = evaluated.physical_points.at(q, node, component);
                const auto& dx_fd = fd.physical_points.at(q, node, component);
                const auto& dJ = evaluated.jacobians.at(q, node, component);
                const auto& dJ_fd = fd.jacobians.at(q, node, component);
                for (std::size_t d = 0; d < 3u; ++d) {
                    EXPECT_NEAR(dx[d], dx_fd[d], 3.0e-8);
                }
                expectMatrixNear(dJ, dJ_fd, 3.0e-8);
                EXPECT_NEAR(evaluated.measures.at(q, node, component),
                            fd.measures.at(q, node, component),
                            5.0e-7);
            }
        }
    }
}
