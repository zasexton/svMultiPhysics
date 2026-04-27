/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_GEOMETRY_FRAMEGEOMETRY_H
#define SVMP_FE_GEOMETRY_FRAMEGEOMETRY_H

/**
 * @file FrameGeometry.h
 * @brief Frame-explicit cell and face geometry preparation utilities.
 */

#include "Core/Types.h"
#include "Geometry/GeometryMapping.h"

#include <array>
#include <cstddef>
#include <span>
#include <vector>

namespace svmp {
namespace FE {

namespace quadrature {
class QuadratureRule;
}

namespace geometry {

using Point3D = std::array<Real, 3>;
using Vector3D = std::array<Real, 3>;
using Matrix3x3 = std::array<std::array<Real, 3>, 3>;

struct FrameGeometryData {
    std::vector<Point3D> points{};
    std::vector<Matrix3x3> jacobians{};
    std::vector<Matrix3x3> inverse_jacobians{};
    std::vector<Real> jacobian_determinants{};
    std::vector<Real> measures{};
};

struct FaceGeometryData {
    FrameGeometryData cell_geometry{};
    std::vector<Point3D> cell_reference_points{};
    std::vector<Point3D> face_reference_points{};
    std::vector<Real> canonical_to_reference_measures{};
    std::vector<Vector3D> normals{};
    std::vector<Real> surface_measures{};
    std::vector<Matrix3x3> surface_jacobians{};
};

struct SurfaceTransform {
    Vector3D normal{};
    Real measure{0.0};
    Vector3D oriented_measure_vector{};
};

struct NodalScalarSensitivity {
    LocalIndex n_qpts{0};
    LocalIndex n_nodes{0};
    std::vector<Real> values{};

    [[nodiscard]] Real at(LocalIndex q, LocalIndex node, int component) const;
};

struct NodalVectorSensitivity {
    LocalIndex n_qpts{0};
    LocalIndex n_nodes{0};
    std::vector<Vector3D> values{};

    [[nodiscard]] const Vector3D& at(LocalIndex q, LocalIndex node, int component) const;
};

struct NodalMatrixSensitivity {
    LocalIndex n_qpts{0};
    LocalIndex n_nodes{0};
    std::vector<Matrix3x3> values{};

    [[nodiscard]] const Matrix3x3& at(LocalIndex q, LocalIndex node, int component) const;
};

struct CellGeometrySensitivity {
    LocalIndex n_qpts{0};
    LocalIndex n_nodes{0};
    NodalVectorSensitivity physical_points{};
    NodalMatrixSensitivity jacobians{};
    NodalScalarSensitivity measures{};
    NodalMatrixSensitivity inverse_jacobians{};
};

struct FaceGeometrySensitivity {
    LocalIndex n_qpts{0};
    LocalIndex n_nodes{0};
    NodalVectorSensitivity normals{};
    NodalScalarSensitivity measures{};
};

[[nodiscard]] int defaultGeometryOrder(ElementType element_type) noexcept;

[[nodiscard]] std::vector<math::Vector<Real, 3>>
toGeometryNodes(std::span<const Point3D> coordinates);

[[nodiscard]] FrameGeometryData evaluateCellFrame(
    const GeometryMapping& mapping,
    const quadrature::QuadratureRule& quad_rule);

[[nodiscard]] FrameGeometryData evaluateCellFrame(
    ElementType cell_type,
    const quadrature::QuadratureRule& quad_rule,
    std::span<const Point3D> coordinates);

[[nodiscard]] FaceGeometryData evaluateFaceFrame(
    const GeometryMapping& mapping,
    ElementType cell_type,
    LocalIndex local_face_id,
    ElementType face_type,
    const quadrature::QuadratureRule& quad_rule,
    std::span<const LocalIndex> align_facet_to_reference = {});

[[nodiscard]] FaceGeometryData evaluateFaceFrame(
    ElementType cell_type,
    LocalIndex local_face_id,
    ElementType face_type,
    const quadrature::QuadratureRule& quad_rule,
    std::span<const Point3D> coordinates,
    std::span<const LocalIndex> align_facet_to_reference = {});

[[nodiscard]] Matrix3x3 configurationTransform(
    const Matrix3x3& current_jacobian,
    const Matrix3x3& reference_inverse_jacobian);

[[nodiscard]] SurfaceTransform nansonSurfaceTransform(
    const Vector3D& reference_normal,
    Real reference_measure,
    const Matrix3x3& deformation_gradient);

[[nodiscard]] SurfaceTransform surfaceTransformFromJacobianInverse(
    const Vector3D& reference_normal,
    Real reference_measure,
    const Matrix3x3& inverse_jacobian,
    Real jacobian_determinant);

/**
 * @brief Verification-only finite-difference reference for cell geometry sensitivities.
 *
 * Runtime geometry evaluation should use evaluateCellGeometrySensitivity().
 */
[[nodiscard]] CellGeometrySensitivity finiteDifferenceCellGeometrySensitivity(
    ElementType cell_type,
    const quadrature::QuadratureRule& quad_rule,
    std::span<const Point3D> coordinates,
    Real step = Real(1e-7));

[[nodiscard]] CellGeometrySensitivity evaluateCellGeometrySensitivity(
    const GeometryMapping& mapping,
    const quadrature::QuadratureRule& quad_rule);

[[nodiscard]] CellGeometrySensitivity evaluateCellGeometrySensitivity(
    ElementType cell_type,
    const quadrature::QuadratureRule& quad_rule,
    std::span<const Point3D> coordinates);

/**
 * @brief Verification-only finite-difference reference for face geometry sensitivities.
 *
 * Runtime geometry evaluation should use evaluateFaceGeometrySensitivity().
 */
[[nodiscard]] FaceGeometrySensitivity finiteDifferenceFaceGeometrySensitivity(
    ElementType cell_type,
    LocalIndex local_face_id,
    ElementType face_type,
    const quadrature::QuadratureRule& quad_rule,
    std::span<const Point3D> coordinates,
    std::span<const LocalIndex> align_facet_to_reference = {},
    Real step = Real(1e-7));

[[nodiscard]] FaceGeometrySensitivity evaluateFaceGeometrySensitivity(
    ElementType cell_type,
    LocalIndex local_face_id,
    ElementType face_type,
    const quadrature::QuadratureRule& quad_rule,
    std::span<const Point3D> coordinates,
    std::span<const LocalIndex> align_facet_to_reference = {});

} // namespace geometry
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_GEOMETRY_FRAMEGEOMETRY_H
