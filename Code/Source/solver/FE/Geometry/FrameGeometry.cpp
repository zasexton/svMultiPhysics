/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "FrameGeometry.h"

#include "Basis/BasisFunction.h"
#include "Core/FEException.h"
#include "Elements/ElementTransform.h"
#include "Geometry/GeometryFrameUtils.h"
#include "Geometry/MappingFactory.h"
#include "Quadrature/QuadratureRule.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>

namespace svmp {
namespace FE {
namespace geometry {

namespace {

[[nodiscard]] std::size_t sensitivityIndex(LocalIndex n_nodes,
                                           LocalIndex q,
                                           LocalIndex node,
                                           int component)
{
    FE_THROW_IF(component < 0 || component >= 3, FEException,
                "FrameGeometry sensitivity component out of range");
    return (static_cast<std::size_t>(q) * static_cast<std::size_t>(n_nodes) +
            static_cast<std::size_t>(node)) *
               3u +
           static_cast<std::size_t>(component);
}

[[nodiscard]] math::Vector<Real, 3> toVector(const Point3D& p)
{
    return math::Vector<Real, 3>{p[0], p[1], p[2]};
}

[[nodiscard]] Point3D toPoint(const math::Vector<Real, 3>& v)
{
    return Point3D{v[0], v[1], v[2]};
}

[[nodiscard]] Matrix3x3 toArray(const math::Matrix<Real, 3, 3>& m)
{
    Matrix3x3 out{};
    for (std::size_t r = 0; r < 3u; ++r) {
        for (std::size_t c = 0; c < 3u; ++c) {
            out[r][c] = m(r, c);
        }
    }
    return out;
}

[[nodiscard]] math::Matrix<Real, 3, 3> toMatrix(const Matrix3x3& m)
{
    math::Matrix<Real, 3, 3> out{};
    for (std::size_t r = 0; r < 3u; ++r) {
        for (std::size_t c = 0; c < 3u; ++c) {
            out(r, c) = m[r][c];
        }
    }
    return out;
}

[[nodiscard]] Matrix3x3 zeroMatrix() noexcept
{
    return Matrix3x3{};
}

[[nodiscard]] Matrix3x3 multiply(const Matrix3x3& a, const Matrix3x3& b) noexcept
{
    Matrix3x3 out{};
    for (std::size_t i = 0; i < 3u; ++i) {
        for (std::size_t j = 0; j < 3u; ++j) {
            Real value = 0.0;
            for (std::size_t k = 0; k < 3u; ++k) {
                value += a[i][k] * b[k][j];
            }
            out[i][j] = value;
        }
    }
    return out;
}

[[nodiscard]] Matrix3x3 negate(const Matrix3x3& m) noexcept
{
    Matrix3x3 out{};
    for (std::size_t i = 0; i < 3u; ++i) {
        for (std::size_t j = 0; j < 3u; ++j) {
            out[i][j] = -m[i][j];
        }
    }
    return out;
}

[[nodiscard]] Real traceProduct(const Matrix3x3& a, const Matrix3x3& b) noexcept
{
    Real value = 0.0;
    for (std::size_t i = 0; i < 3u; ++i) {
        for (std::size_t k = 0; k < 3u; ++k) {
            value += a[i][k] * b[k][i];
        }
    }
    return value;
}

[[nodiscard]] Real dot(const Vector3D& a, const Vector3D& b) noexcept
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

[[nodiscard]] Real norm(const Vector3D& v) noexcept
{
    return std::sqrt(dot(v, v));
}

[[nodiscard]] math::Vector<Real, 3> cross(const math::Vector<Real, 3>& a,
                                          const math::Vector<Real, 3>& b) noexcept
{
    return math::Vector<Real, 3>{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]};
}

[[nodiscard]] Real norm(const math::Vector<Real, 3>& v) noexcept
{
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

[[nodiscard]] Vector3D toArrayVector(const math::Vector<Real, 3>& v) noexcept
{
    return Vector3D{v[0], v[1], v[2]};
}

[[nodiscard]] math::Vector<Real, 3> toMathVector(const Vector3D& v) noexcept
{
    return math::Vector<Real, 3>{v[0], v[1], v[2]};
}

[[nodiscard]] math::Vector<Real, 3> matrixColumn(const math::Matrix<Real, 3, 3>& m,
                                                 std::size_t c) noexcept
{
    return math::Vector<Real, 3>{m(0, c), m(1, c), m(2, c)};
}

[[nodiscard]] math::Vector<Real, 3> unitOrZero(const math::Vector<Real, 3>& v) noexcept
{
    const Real n = norm(v);
    if (n <= detail::kDegenerateTol) {
        return math::Vector<Real, 3>{};
    }
    return v / n;
}

[[nodiscard]] math::Vector<Real, 3> projectedUnitDerivative(
    const math::Vector<Real, 3>& unit,
    const math::Vector<Real, 3>& dvalue,
    Real value_norm) noexcept
{
    if (value_norm <= detail::kDegenerateTol) {
        return math::Vector<Real, 3>{};
    }
    return (dvalue - unit * unit.dot(dvalue)) / value_norm;
}

[[nodiscard]] math::Vector<Real, 3> curveFrameAxis(
    const math::Vector<Real, 3>& t_unit) noexcept
{
    math::Vector<Real, 3> axis{Real(1), Real(0), Real(0)};
    if (std::abs(t_unit[0]) > Real(0.9)) {
        axis = math::Vector<Real, 3>{Real(0), Real(1), Real(0)};
        if (std::abs(t_unit[1]) > Real(0.9)) {
            axis = math::Vector<Real, 3>{Real(0), Real(0), Real(1)};
        }
    }
    return axis;
}

[[nodiscard]] Matrix3x3 embeddedFrameJacobianDerivative(
    const math::Matrix<Real, 3, 3>& J,
    int dim,
    const basis::Gradient& gradient,
    int component)
{
    Matrix3x3 dJ{};
    const auto c = static_cast<std::size_t>(component);
    for (int xi_dir = 0; xi_dir < dim; ++xi_dir) {
        dJ[c][static_cast<std::size_t>(xi_dir)] =
            gradient[static_cast<std::size_t>(xi_dir)];
    }

    if (dim == 1) {
        const auto t = matrixColumn(J, 0);
        const Real t_norm = norm(t);
        if (t_norm <= detail::kDegenerateTol) {
            return dJ;
        }
        const math::Vector<Real, 3> t_unit = t / t_norm;
        auto axis = curveFrameAxis(t_unit);
        auto n1_raw = t_unit.cross(axis);
        Real n1_norm = norm(n1_raw);
        if (n1_norm <= detail::kDegenerateTol) {
            axis = math::Vector<Real, 3>{Real(0), Real(0), Real(1)};
            n1_raw = t_unit.cross(axis);
            n1_norm = norm(n1_raw);
        }
        if (n1_norm <= detail::kDegenerateTol) {
            return dJ;
        }

        const math::Vector<Real, 3> n1 = n1_raw / n1_norm;
        const auto dt = math::Vector<Real, 3>{
            (component == 0) ? gradient[0] : Real(0),
            (component == 1) ? gradient[0] : Real(0),
            (component == 2) ? gradient[0] : Real(0)};
        const auto dt_unit = projectedUnitDerivative(t_unit, dt, t_norm);
        const auto dn1_raw = dt_unit.cross(axis);
        const auto dn1 = projectedUnitDerivative(n1, dn1_raw, n1_norm);
        const auto dn2 = dt_unit.cross(n1) + t_unit.cross(dn1);

        for (std::size_t r = 0; r < 3u; ++r) {
            dJ[r][1] = dn1[r];
            dJ[r][2] = dn2[r];
        }
    } else if (dim == 2) {
        const auto t0 = matrixColumn(J, 0);
        const auto t1 = matrixColumn(J, 1);
        const auto area_vec = t0.cross(t1);
        const Real area = norm(area_vec);
        if (area <= detail::kDegenerateTol) {
            return dJ;
        }

        const math::Vector<Real, 3> n = area_vec / area;
        const auto dt0 = math::Vector<Real, 3>{
            (component == 0) ? gradient[0] : Real(0),
            (component == 1) ? gradient[0] : Real(0),
            (component == 2) ? gradient[0] : Real(0)};
        const auto dt1 = math::Vector<Real, 3>{
            (component == 0) ? gradient[1] : Real(0),
            (component == 1) ? gradient[1] : Real(0),
            (component == 2) ? gradient[1] : Real(0)};
        const auto darea_vec = dt0.cross(t1) + t0.cross(dt1);
        const auto dn = projectedUnitDerivative(n, darea_vec, area);

        for (std::size_t r = 0; r < 3u; ++r) {
            dJ[r][2] = dn[r];
        }
    }

    return dJ;
}

[[nodiscard]] Real embeddedFrameMeasureDerivative(
    const math::Matrix<Real, 3, 3>& J,
    int dim,
    const basis::Gradient& gradient,
    int component) noexcept
{
    if (dim == 1) {
        const auto t = matrixColumn(J, 0);
        const Real t_norm = norm(t);
        if (t_norm <= detail::kDegenerateTol) {
            return Real(0);
        }
        const math::Vector<Real, 3> t_unit = t / t_norm;
        return t_unit[static_cast<std::size_t>(component)] * gradient[0];
    }

    if (dim == 2) {
        const auto t0 = matrixColumn(J, 0);
        const auto t1 = matrixColumn(J, 1);
        const auto area_vec = t0.cross(t1);
        const Real area = norm(area_vec);
        if (area <= detail::kDegenerateTol) {
            return Real(0);
        }
        const math::Vector<Real, 3> n = area_vec / area;
        const auto dt0 = math::Vector<Real, 3>{
            (component == 0) ? gradient[0] : Real(0),
            (component == 1) ? gradient[0] : Real(0),
            (component == 2) ? gradient[0] : Real(0)};
        const auto dt1 = math::Vector<Real, 3>{
            (component == 0) ? gradient[1] : Real(0),
            (component == 1) ? gradient[1] : Real(0),
            (component == 2) ? gradient[1] : Real(0)};
        const auto darea_vec = dt0.cross(t1) + t0.cross(dt1);
        return n.dot(darea_vec);
    }

    return Real(0);
}

[[nodiscard]] std::shared_ptr<GeometryMapping>
makeMapping(ElementType cell_type, std::span<const Point3D> coordinates)
{
    MappingRequest request;
    request.element_type = cell_type;
    request.geometry_order = defaultGeometryOrder(cell_type);
    request.use_affine = (request.geometry_order <= 1);
    return MappingFactory::create(request, toGeometryNodes(coordinates));
}

[[nodiscard]] math::Vector<Real, 3> canonicalToFacetCoordinates(
    ElementType face_type,
    const math::Vector<Real, 3>& canonical_point,
    std::span<const LocalIndex> align_facet_to_reference)
{
    if (face_type == ElementType::Line2) {
        Real t = (canonical_point[0] + Real(1)) * Real(0.5);
        if (align_facet_to_reference.size() == 2u) {
            const Real w_ref0 = Real(1) - t;
            const Real w_ref1 = t;
            const std::array<Real, 2> w_ref{w_ref0, w_ref1};
            std::array<Real, 2> w_local{Real(0), Real(0)};
            for (std::size_t j = 0; j < 2u; ++j) {
                const auto src = static_cast<std::size_t>(align_facet_to_reference[j]);
                FE_THROW_IF(src >= 2u, FEException,
                            "FrameGeometry: invalid line face alignment index");
                w_local[j] = w_ref[src];
            }
            t = w_local[1];
        }
        return math::Vector<Real, 3>{t, Real(0), Real(0)};
    }

    if (face_type == ElementType::Quad4) {
        Real s = (canonical_point[0] + Real(1)) * Real(0.5);
        Real t = (canonical_point[1] + Real(1)) * Real(0.5);
        if (align_facet_to_reference.size() == 4u) {
            const std::array<Real, 4> w_ref{
                (Real(1) - s) * (Real(1) - t),
                s * (Real(1) - t),
                s * t,
                (Real(1) - s) * t};
            std::array<Real, 4> w_local{Real(0), Real(0), Real(0), Real(0)};
            for (std::size_t j = 0; j < 4u; ++j) {
                const auto src = static_cast<std::size_t>(align_facet_to_reference[j]);
                FE_THROW_IF(src >= 4u, FEException,
                            "FrameGeometry: invalid quad face alignment index");
                w_local[j] = w_ref[src];
            }
            s = w_local[1] + w_local[2];
            t = w_local[2] + w_local[3];
        }
        return math::Vector<Real, 3>{s, t, Real(0)};
    }

    if (face_type == ElementType::Triangle3) {
        math::Vector<Real, 3> facet{
            canonical_point[0], canonical_point[1], Real(0)};
        if (align_facet_to_reference.size() == 3u) {
            const Real w_ref0 = Real(1) - canonical_point[0] - canonical_point[1];
            const Real w_ref1 = canonical_point[0];
            const Real w_ref2 = canonical_point[1];
            const std::array<Real, 3> w_ref{w_ref0, w_ref1, w_ref2};
            std::array<Real, 3> w_local{Real(0), Real(0), Real(0)};
            for (std::size_t j = 0; j < 3u; ++j) {
                const auto src = static_cast<std::size_t>(align_facet_to_reference[j]);
                FE_THROW_IF(src >= 3u, FEException,
                            "FrameGeometry: invalid triangle face alignment index");
                w_local[j] = w_ref[src];
            }
            facet = math::Vector<Real, 3>{w_local[1], w_local[2], Real(0)};
        }
        return facet;
    }

    FE_THROW(FEException, "FrameGeometry: unsupported face element type");
}

[[nodiscard]] Real canonicalFaceJacobianToReference(
    ElementType face_type,
    std::span<const math::Vector<Real, 3>> ref_face_coords,
    const math::Vector<Real, 3>& facet_coords)
{
    switch (face_type) {
        case ElementType::Line2: {
            FE_THROW_IF(ref_face_coords.size() < 2u, FEException,
                        "FrameGeometry(Line2): missing reference face vertices");
            const auto dx = ref_face_coords[1] - ref_face_coords[0];
            return Real(0.5) * norm(dx);
        }
        case ElementType::Triangle3: {
            FE_THROW_IF(ref_face_coords.size() < 3u, FEException,
                        "FrameGeometry(Triangle3): missing reference face vertices");
            (void)facet_coords;
            const auto e1 = ref_face_coords[1] - ref_face_coords[0];
            const auto e2 = ref_face_coords[2] - ref_face_coords[0];
            return norm(cross(e1, e2));
        }
        case ElementType::Quad4: {
            FE_THROW_IF(ref_face_coords.size() < 4u, FEException,
                        "FrameGeometry(Quad4): missing reference face vertices");
            const Real s = facet_coords[0];
            const Real t = facet_coords[1];
            math::Vector<Real, 3> dXds{};
            math::Vector<Real, 3> dXdt{};
            for (std::size_t i = 0; i < 3u; ++i) {
                dXds[i] = (Real(1) - t) * (ref_face_coords[1][i] - ref_face_coords[0][i]) +
                          t * (ref_face_coords[2][i] - ref_face_coords[3][i]);
                dXdt[i] = (Real(1) - s) * (ref_face_coords[3][i] - ref_face_coords[0][i]) +
                          s * (ref_face_coords[2][i] - ref_face_coords[1][i]);
            }
            return Real(0.25) * norm(cross(dXds, dXdt));
        }
        default:
            break;
    }
    return Real(1);
}

[[nodiscard]] math::Matrix<Real, 3, 3> canonicalToFacetJacobian(
    ElementType face_type,
    const math::Vector<Real, 3>& canonical_point,
    std::span<const LocalIndex> align_facet_to_reference)
{
    math::Matrix<Real, 3, 3> D{};
    if (face_type == ElementType::Line2) {
        if (align_facet_to_reference.size() == 2u) {
            const auto src = static_cast<std::size_t>(align_facet_to_reference[1]);
            FE_THROW_IF(src >= 2u, FEException,
                        "FrameGeometry: invalid line face alignment index");
            D(0, 0) = (src == 0u) ? Real(-0.5) : Real(0.5);
        } else {
            D(0, 0) = Real(0.5);
        }
        return D;
    }

    if (face_type == ElementType::Quad4) {
        const Real s = (canonical_point[0] + Real(1)) * Real(0.5);
        const Real t = (canonical_point[1] + Real(1)) * Real(0.5);
        if (align_facet_to_reference.size() == 4u) {
            const std::array<Real, 4> dw_ds{
                -(Real(1) - t), Real(1) - t, t, -t};
            const std::array<Real, 4> dw_dt{
                -(Real(1) - s), -s, s, Real(1) - s};
            const auto add_weight_derivative = [&](std::size_t local, int row) {
                const auto src = static_cast<std::size_t>(align_facet_to_reference[local]);
                FE_THROW_IF(src >= 4u, FEException,
                            "FrameGeometry: invalid quad face alignment index");
                const auto r = static_cast<std::size_t>(row);
                D(r, 0) += Real(0.5) * dw_ds[src];
                D(r, 1) += Real(0.5) * dw_dt[src];
            };
            add_weight_derivative(1u, 0);
            add_weight_derivative(2u, 0);
            add_weight_derivative(2u, 1);
            add_weight_derivative(3u, 1);
        } else {
            D(0, 0) = Real(0.5);
            D(1, 1) = Real(0.5);
        }
        return D;
    }

    if (face_type == ElementType::Triangle3) {
        if (align_facet_to_reference.size() == 3u) {
            const std::array<std::array<Real, 2>, 3> dw{{
                {{Real(-1), Real(-1)}},
                {{Real(1), Real(0)}},
                {{Real(0), Real(1)}}}};
            for (int row = 0; row < 2; ++row) {
                const auto src =
                    static_cast<std::size_t>(align_facet_to_reference[static_cast<std::size_t>(row + 1)]);
                FE_THROW_IF(src >= 3u, FEException,
                            "FrameGeometry: invalid triangle face alignment index");
                const auto r = static_cast<std::size_t>(row);
                D(r, 0) = dw[src][0];
                D(r, 1) = dw[src][1];
            }
        } else {
            D(0, 0) = Real(1);
            D(1, 1) = Real(1);
        }
        return D;
    }

    return D;
}

[[nodiscard]] math::Matrix<Real, 3, 3> referenceFacetJacobian(
    ElementType cell_type,
    LocalIndex local_face_id,
    const math::Vector<Real, 3>& facet_coords)
{
    auto [unused_vertices, coords] =
        elements::ElementTransform::facet_vertices(cell_type, static_cast<int>(local_face_id));
    (void)unused_vertices;

    math::Matrix<Real, 3, 3> D{};
    const int dim = element_dimension(cell_type);
    if (dim == 2 && coords.size() >= 2u) {
        for (std::size_t r = 0; r < 3u; ++r) {
            D(r, 0) = coords[1][r] - coords[0][r];
        }
    } else if (dim == 3 && coords.size() == 3u) {
        for (std::size_t r = 0; r < 3u; ++r) {
            D(r, 0) = coords[1][r] - coords[0][r];
            D(r, 1) = coords[2][r] - coords[0][r];
        }
    } else if (dim == 3 && coords.size() >= 4u) {
        const Real s = facet_coords[0];
        const Real t = facet_coords[1];
        for (std::size_t r = 0; r < 3u; ++r) {
            D(r, 0) = (Real(1) - t) * (coords[1][r] - coords[0][r]) +
                      t * (coords[2][r] - coords[3][r]);
            D(r, 1) = (Real(1) - s) * (coords[3][r] - coords[0][r]) +
                      s * (coords[2][r] - coords[1][r]);
        }
    }
    return D;
}

[[nodiscard]] math::Matrix<Real, 3, 3> referenceToCanonicalFaceJacobian(
    ElementType cell_type,
    LocalIndex local_face_id,
    ElementType face_type,
    const math::Vector<Real, 3>& canonical_point,
    std::span<const LocalIndex> align_facet_to_reference)
{
    const auto facet_coords =
        canonicalToFacetCoordinates(face_type, canonical_point, align_facet_to_reference);
    return referenceFacetJacobian(cell_type, local_face_id, facet_coords) *
           canonicalToFacetJacobian(face_type, canonical_point, align_facet_to_reference);
}

[[nodiscard]] Matrix3x3 analyticSurfaceJacobian(
    const GeometryMapping& mapping,
    ElementType cell_type,
    LocalIndex local_face_id,
    ElementType face_type,
    const math::Vector<Real, 3>& canonical_point,
    const Vector3D& normal,
    std::span<const LocalIndex> align_facet_to_reference)
{
    const int face_dim = (face_type == ElementType::Line2) ? 1 : 2;
    Matrix3x3 out{};

    const auto facet_coords =
        canonicalToFacetCoordinates(face_type, canonical_point, align_facet_to_reference);
    const auto xi = elements::ElementTransform::facet_to_reference(
        cell_type, static_cast<int>(local_face_id), facet_coords);
    const auto J = mapping.jacobian(xi);
    const auto dxi_dcanon =
        referenceToCanonicalFaceJacobian(cell_type,
                                         local_face_id,
                                         face_type,
                                         canonical_point,
                                         align_facet_to_reference);

    for (int d = 0; d < face_dim; ++d) {
        for (std::size_t r = 0; r < 3u; ++r) {
            Real value = Real(0);
            for (std::size_t j = 0; j < 3u; ++j) {
                value += J(r, j) * dxi_dcanon(j, static_cast<std::size_t>(d));
            }
            out[r][static_cast<std::size_t>(d)] = value;
        }
    }
    for (std::size_t r = 0; r < 3u; ++r) {
        out[r][2] = normal[r];
    }
    return out;
}

[[nodiscard]] FaceGeometryData evaluateFaceFrameImpl(
    const GeometryMapping& mapping,
    ElementType cell_type,
    LocalIndex local_face_id,
    ElementType face_type,
    const quadrature::QuadratureRule& quad_rule,
    std::span<const LocalIndex> align_facet_to_reference)
{
    FaceGeometryData data;
    const auto n_qpts = static_cast<LocalIndex>(quad_rule.num_points());
    data.cell_geometry.points.resize(static_cast<std::size_t>(n_qpts));
    data.cell_geometry.jacobians.resize(static_cast<std::size_t>(n_qpts));
    data.cell_geometry.inverse_jacobians.resize(static_cast<std::size_t>(n_qpts));
    data.cell_geometry.jacobian_determinants.resize(static_cast<std::size_t>(n_qpts));
    data.cell_geometry.measures.resize(static_cast<std::size_t>(n_qpts));
    data.cell_reference_points.resize(static_cast<std::size_t>(n_qpts));
    data.face_reference_points.resize(static_cast<std::size_t>(n_qpts));
    data.canonical_to_reference_measures.resize(static_cast<std::size_t>(n_qpts));
    data.normals.resize(static_cast<std::size_t>(n_qpts));
    data.surface_measures.resize(static_cast<std::size_t>(n_qpts));
    data.surface_jacobians.resize(static_cast<std::size_t>(n_qpts));

    const auto [unused_vertices, ref_face_coords] =
        elements::ElementTransform::facet_vertices(cell_type, static_cast<int>(local_face_id));
    (void)unused_vertices;

    const auto n_ref_math = elements::ElementTransform::reference_facet_normal(
        cell_type, static_cast<int>(local_face_id));
    const Vector3D n_ref{n_ref_math[0], n_ref_math[1], n_ref_math[2]};

    Vector3D cell_center{0.0, 0.0, 0.0};
    const auto& nodes = mapping.nodes();
    if (!nodes.empty()) {
        for (const auto& node : nodes) {
            cell_center[0] += node[0];
            cell_center[1] += node[1];
            cell_center[2] += node[2];
        }
        const Real inv_n = Real(1) / static_cast<Real>(nodes.size());
        cell_center[0] *= inv_n;
        cell_center[1] *= inv_n;
        cell_center[2] *= inv_n;
    }

    const auto& qpts = quad_rule.points();
    for (LocalIndex q = 0; q < n_qpts; ++q) {
        const auto qidx = static_cast<std::size_t>(q);
        const auto& canonical_point = qpts[qidx];
        const auto facet_coords =
            canonicalToFacetCoordinates(face_type, canonical_point, align_facet_to_reference);
        const auto xi = elements::ElementTransform::facet_to_reference(
            cell_type, static_cast<int>(local_face_id), facet_coords);

        data.cell_reference_points[qidx] = toPoint(xi);
        data.face_reference_points[qidx] = toPoint(facet_coords);

        const auto x = mapping.map_to_physical(xi);
        const auto J = mapping.jacobian(xi);
        const auto J_inv = mapping.jacobian_inverse(xi);
        const Real det_J = mapping.jacobian_determinant(xi);

        data.cell_geometry.points[qidx] = toPoint(x);
        data.cell_geometry.jacobians[qidx] = toArray(J);
        data.cell_geometry.inverse_jacobians[qidx] = toArray(J_inv);
        data.cell_geometry.jacobian_determinants[qidx] = det_J;
        data.cell_geometry.measures[qidx] = std::abs(det_J);

        const Real reference_measure =
            canonicalFaceJacobianToReference(face_type,
                                             std::span<const math::Vector<Real, 3>>(ref_face_coords),
                                             facet_coords);
        data.canonical_to_reference_measures[qidx] = reference_measure;

        auto surface = surfaceTransformFromJacobianInverse(n_ref,
                                                           reference_measure,
                                                           data.cell_geometry.inverse_jacobians[qidx],
                                                           det_J);

        const Vector3D to_center{
            cell_center[0] - x[0],
            cell_center[1] - x[1],
            cell_center[2] - x[2]};
        if (dot(to_center, surface.normal) > Real(0)) {
            for (std::size_t d = 0; d < 3u; ++d) {
                surface.normal[d] = -surface.normal[d];
                surface.oriented_measure_vector[d] = -surface.oriented_measure_vector[d];
            }
        }

        data.normals[qidx] = surface.normal;
        data.surface_measures[qidx] = surface.measure;
        data.surface_jacobians[qidx] = analyticSurfaceJacobian(mapping,
                                                               cell_type,
                                                               local_face_id,
                                                               face_type,
                                                               canonical_point,
                                                               surface.normal,
                                                               align_facet_to_reference);
    }

    return data;
}

} // namespace

Real NodalScalarSensitivity::at(LocalIndex q, LocalIndex node, int component) const
{
    return values.at(sensitivityIndex(n_nodes, q, node, component));
}

const Vector3D& NodalVectorSensitivity::at(LocalIndex q, LocalIndex node, int component) const
{
    return values.at(sensitivityIndex(n_nodes, q, node, component));
}

const Matrix3x3& NodalMatrixSensitivity::at(LocalIndex q, LocalIndex node, int component) const
{
    return values.at(sensitivityIndex(n_nodes, q, node, component));
}

int defaultGeometryOrder(ElementType element_type) noexcept
{
    switch (element_type) {
        case ElementType::Line3:
        case ElementType::Triangle6:
        case ElementType::Quad8:
        case ElementType::Quad9:
        case ElementType::Tetra10:
        case ElementType::Hex20:
        case ElementType::Hex27:
        case ElementType::Wedge15:
        case ElementType::Wedge18:
        case ElementType::Pyramid13:
        case ElementType::Pyramid14:
            return 2;
        default:
            return 1;
    }
}

std::vector<math::Vector<Real, 3>> toGeometryNodes(std::span<const Point3D> coordinates)
{
    std::vector<math::Vector<Real, 3>> nodes(coordinates.size());
    for (std::size_t i = 0; i < coordinates.size(); ++i) {
        nodes[i] = toVector(coordinates[i]);
    }
    return nodes;
}

FrameGeometryData evaluateCellFrame(const GeometryMapping& mapping,
                                    const quadrature::QuadratureRule& quad_rule)
{
    FrameGeometryData data;
    const auto n_qpts = static_cast<LocalIndex>(quad_rule.num_points());
    data.points.resize(static_cast<std::size_t>(n_qpts));
    data.jacobians.resize(static_cast<std::size_t>(n_qpts));
    data.inverse_jacobians.resize(static_cast<std::size_t>(n_qpts));
    data.jacobian_determinants.resize(static_cast<std::size_t>(n_qpts));
    data.measures.resize(static_cast<std::size_t>(n_qpts));

    const auto& qpts = quad_rule.points();
    for (LocalIndex q = 0; q < n_qpts; ++q) {
        const auto qidx = static_cast<std::size_t>(q);
        const auto& xi = qpts[qidx];
        const auto x = mapping.map_to_physical(xi);
        const auto J = mapping.jacobian(xi);
        const auto J_inv = mapping.jacobian_inverse(xi);
        const Real det_J = mapping.jacobian_determinant(xi);

        data.points[qidx] = toPoint(x);
        data.jacobians[qidx] = toArray(J);
        data.inverse_jacobians[qidx] = toArray(J_inv);
        data.jacobian_determinants[qidx] = det_J;
        data.measures[qidx] = std::abs(det_J);
    }

    return data;
}

FrameGeometryData evaluateCellFrame(ElementType cell_type,
                                    const quadrature::QuadratureRule& quad_rule,
                                    std::span<const Point3D> coordinates)
{
    const auto mapping = makeMapping(cell_type, coordinates);
    return evaluateCellFrame(*mapping, quad_rule);
}

FaceGeometryData evaluateFaceFrame(const GeometryMapping& mapping,
                                   ElementType cell_type,
                                   LocalIndex local_face_id,
                                   ElementType face_type,
                                   const quadrature::QuadratureRule& quad_rule,
                                   std::span<const LocalIndex> align_facet_to_reference)
{
    return evaluateFaceFrameImpl(mapping,
                                 cell_type,
                                 local_face_id,
                                 face_type,
                                 quad_rule,
                                 align_facet_to_reference);
}

FaceGeometryData evaluateFaceFrame(ElementType cell_type,
                                   LocalIndex local_face_id,
                                   ElementType face_type,
                                   const quadrature::QuadratureRule& quad_rule,
                                   std::span<const Point3D> coordinates,
                                   std::span<const LocalIndex> align_facet_to_reference)
{
    const auto mapping = makeMapping(cell_type, coordinates);
    return evaluateFaceFrame(*mapping,
                             cell_type,
                             local_face_id,
                             face_type,
                             quad_rule,
                             align_facet_to_reference);
}

Matrix3x3 configurationTransform(const Matrix3x3& current_jacobian,
                                 const Matrix3x3& reference_inverse_jacobian)
{
    Matrix3x3 out{};
    for (std::size_t r = 0; r < 3u; ++r) {
        for (std::size_t c = 0; c < 3u; ++c) {
            Real value = 0.0;
            for (std::size_t k = 0; k < 3u; ++k) {
                value += current_jacobian[r][k] * reference_inverse_jacobian[k][c];
            }
            out[r][c] = value;
        }
    }
    return out;
}

SurfaceTransform nansonSurfaceTransform(const Vector3D& reference_normal,
                                        Real reference_measure,
                                        const Matrix3x3& deformation_gradient)
{
    const auto F = toMatrix(deformation_gradient);
    const auto F_inv = F.inverse();
    return surfaceTransformFromJacobianInverse(reference_normal,
                                               reference_measure,
                                               toArray(F_inv),
                                               F.determinant());
}

SurfaceTransform surfaceTransformFromJacobianInverse(const Vector3D& reference_normal,
                                                     Real reference_measure,
                                                     const Matrix3x3& inverse_jacobian,
                                                     Real jacobian_determinant)
{
    SurfaceTransform out;
    for (std::size_t i = 0; i < 3u; ++i) {
        Real value = 0.0;
        for (std::size_t k = 0; k < 3u; ++k) {
            value += inverse_jacobian[k][i] * reference_normal[k];
        }
        out.oriented_measure_vector[i] = jacobian_determinant * value * reference_measure;
    }

    out.measure = norm(out.oriented_measure_vector);
    constexpr Real tol = Real(1e-14);
    if (out.measure > tol) {
        for (std::size_t i = 0; i < 3u; ++i) {
            out.normal[i] = out.oriented_measure_vector[i] / out.measure;
        }
    } else {
        out.normal = reference_normal;
        out.measure = Real(0);
    }
    return out;
}

CellGeometrySensitivity finiteDifferenceCellGeometrySensitivity(
    ElementType cell_type,
    const quadrature::QuadratureRule& quad_rule,
    std::span<const Point3D> coordinates,
    Real step)
{
    FE_THROW_IF(step <= Real(0), FEException,
                "FrameGeometry: finite-difference step must be positive");

    const auto n_qpts = static_cast<LocalIndex>(quad_rule.num_points());
    const auto n_nodes = static_cast<LocalIndex>(coordinates.size());
    const std::size_t total =
        static_cast<std::size_t>(n_qpts) * static_cast<std::size_t>(n_nodes) * 3u;

    CellGeometrySensitivity sensitivity;
    sensitivity.n_qpts = n_qpts;
    sensitivity.n_nodes = n_nodes;
    sensitivity.physical_points = {n_qpts, n_nodes, std::vector<Vector3D>(total)};
    sensitivity.jacobians = {n_qpts, n_nodes, std::vector<Matrix3x3>(total)};
    sensitivity.measures = {n_qpts, n_nodes, std::vector<Real>(total)};
    sensitivity.inverse_jacobians = {n_qpts, n_nodes, std::vector<Matrix3x3>(total)};

    std::vector<Point3D> plus(coordinates.begin(), coordinates.end());
    std::vector<Point3D> minus(coordinates.begin(), coordinates.end());
    const Real inv_2h = Real(0.5) / step;

    for (LocalIndex node = 0; node < n_nodes; ++node) {
        for (int component = 0; component < 3; ++component) {
            plus[static_cast<std::size_t>(node)][static_cast<std::size_t>(component)] += step;
            minus[static_cast<std::size_t>(node)][static_cast<std::size_t>(component)] -= step;

            const auto plus_geom = evaluateCellFrame(cell_type, quad_rule, plus);
            const auto minus_geom = evaluateCellFrame(cell_type, quad_rule, minus);

            for (LocalIndex q = 0; q < n_qpts; ++q) {
                const auto out = sensitivityIndex(n_nodes, q, node, component);
                const auto qidx = static_cast<std::size_t>(q);
                for (std::size_t d = 0; d < 3u; ++d) {
                    sensitivity.physical_points.values[out][d] =
                        (plus_geom.points[qidx][d] - minus_geom.points[qidx][d]) * inv_2h;
                }
                for (std::size_t r = 0; r < 3u; ++r) {
                    for (std::size_t c = 0; c < 3u; ++c) {
                        sensitivity.jacobians.values[out][r][c] =
                            (plus_geom.jacobians[qidx][r][c] -
                             minus_geom.jacobians[qidx][r][c]) *
                            inv_2h;
                        sensitivity.inverse_jacobians.values[out][r][c] =
                            (plus_geom.inverse_jacobians[qidx][r][c] -
                             minus_geom.inverse_jacobians[qidx][r][c]) *
                            inv_2h;
                    }
                }
                sensitivity.measures.values[out] =
                    (plus_geom.measures[qidx] - minus_geom.measures[qidx]) * inv_2h;
            }

            plus[static_cast<std::size_t>(node)][static_cast<std::size_t>(component)] -= step;
            minus[static_cast<std::size_t>(node)][static_cast<std::size_t>(component)] += step;
        }
    }

    return sensitivity;
}

CellGeometrySensitivity evaluateCellGeometrySensitivity(
    const GeometryMapping& mapping,
    const quadrature::QuadratureRule& quad_rule)
{
    const auto n_qpts = static_cast<LocalIndex>(quad_rule.num_points());
    const auto n_nodes = static_cast<LocalIndex>(mapping.num_nodes());
    const std::size_t total =
        static_cast<std::size_t>(n_qpts) * static_cast<std::size_t>(n_nodes) * 3u;

    CellGeometrySensitivity sensitivity;
    sensitivity.n_qpts = n_qpts;
    sensitivity.n_nodes = n_nodes;
    sensitivity.physical_points = {n_qpts, n_nodes, std::vector<Vector3D>(total)};
    sensitivity.jacobians = {n_qpts, n_nodes, std::vector<Matrix3x3>(total)};
    sensitivity.measures = {n_qpts, n_nodes, std::vector<Real>(total)};
    sensitivity.inverse_jacobians = {n_qpts, n_nodes, std::vector<Matrix3x3>(total)};

    const auto& geometry_basis = mapping.geometryBasis();
    FE_THROW_IF(static_cast<LocalIndex>(geometry_basis.size()) != n_nodes, FEException,
                "FrameGeometry: geometry basis size does not match mapping node count");

    std::vector<Real> values;
    std::vector<basis::Gradient> gradients;

    const auto& qpts = quad_rule.points();
    for (LocalIndex q = 0; q < n_qpts; ++q) {
        const auto& xi = qpts[static_cast<std::size_t>(q)];
        geometry_basis.evaluate_values(xi, values);
        geometry_basis.evaluate_gradients(xi, gradients);
        FE_THROW_IF(static_cast<LocalIndex>(values.size()) != n_nodes ||
                        static_cast<LocalIndex>(gradients.size()) != n_nodes,
                    FEException,
                    "FrameGeometry: geometry basis evaluation size mismatch");

        const auto J = mapping.jacobian(xi);
        const auto J_inv = toArray(mapping.jacobian_inverse(xi));
        const Real det_J = mapping.jacobian_determinant(xi);
        const Real measure_sign = (det_J < Real(0)) ? Real(-1) : Real(1);

        for (LocalIndex node = 0; node < n_nodes; ++node) {
            const auto nidx = static_cast<std::size_t>(node);
            for (int component = 0; component < 3; ++component) {
                const auto out = sensitivityIndex(n_nodes, q, node, component);

                sensitivity.physical_points.values[out] = Vector3D{};
                sensitivity.physical_points.values[out][static_cast<std::size_t>(component)] =
                    values[nidx];

                Matrix3x3 dJ = zeroMatrix();
                if (mapping.dimension() == 3) {
                    for (int xi_dir = 0; xi_dir < 3; ++xi_dir) {
                        dJ[static_cast<std::size_t>(component)][static_cast<std::size_t>(xi_dir)] =
                            gradients[nidx][static_cast<std::size_t>(xi_dir)];
                    }
                } else {
                    dJ = embeddedFrameJacobianDerivative(J,
                                                         mapping.dimension(),
                                                         gradients[nidx],
                                                         component);
                }
                sensitivity.jacobians.values[out] = dJ;

                if (mapping.dimension() == 3) {
                    const Real d_det = det_J * traceProduct(J_inv, dJ);
                    sensitivity.measures.values[out] = measure_sign * d_det;
                } else {
                    sensitivity.measures.values[out] =
                        embeddedFrameMeasureDerivative(J,
                                                       mapping.dimension(),
                                                       gradients[nidx],
                                                       component);
                }

                sensitivity.inverse_jacobians.values[out] =
                    negate(multiply(multiply(J_inv, dJ), J_inv));
            }
        }
    }

    return sensitivity;
}

CellGeometrySensitivity evaluateCellGeometrySensitivity(
    ElementType cell_type,
    const quadrature::QuadratureRule& quad_rule,
    std::span<const Point3D> coordinates)
{
    const auto mapping = makeMapping(cell_type, coordinates);
    return evaluateCellGeometrySensitivity(*mapping, quad_rule);
}

FaceGeometrySensitivity finiteDifferenceFaceGeometrySensitivity(
    ElementType cell_type,
    LocalIndex local_face_id,
    ElementType face_type,
    const quadrature::QuadratureRule& quad_rule,
    std::span<const Point3D> coordinates,
    std::span<const LocalIndex> align_facet_to_reference,
    Real step)
{
    FE_THROW_IF(step <= Real(0), FEException,
                "FrameGeometry: finite-difference step must be positive");

    const auto n_qpts = static_cast<LocalIndex>(quad_rule.num_points());
    const auto n_nodes = static_cast<LocalIndex>(coordinates.size());
    const std::size_t total =
        static_cast<std::size_t>(n_qpts) * static_cast<std::size_t>(n_nodes) * 3u;

    FaceGeometrySensitivity sensitivity;
    sensitivity.n_qpts = n_qpts;
    sensitivity.n_nodes = n_nodes;
    sensitivity.normals = {n_qpts, n_nodes, std::vector<Vector3D>(total)};
    sensitivity.measures = {n_qpts, n_nodes, std::vector<Real>(total)};

    std::vector<Point3D> plus(coordinates.begin(), coordinates.end());
    std::vector<Point3D> minus(coordinates.begin(), coordinates.end());
    const Real inv_2h = Real(0.5) / step;

    for (LocalIndex node = 0; node < n_nodes; ++node) {
        for (int component = 0; component < 3; ++component) {
            plus[static_cast<std::size_t>(node)][static_cast<std::size_t>(component)] += step;
            minus[static_cast<std::size_t>(node)][static_cast<std::size_t>(component)] -= step;

            const auto plus_geom = evaluateFaceFrame(cell_type,
                                                     local_face_id,
                                                     face_type,
                                                     quad_rule,
                                                     plus,
                                                     align_facet_to_reference);
            const auto minus_geom = evaluateFaceFrame(cell_type,
                                                      local_face_id,
                                                      face_type,
                                                      quad_rule,
                                                      minus,
                                                      align_facet_to_reference);

            for (LocalIndex q = 0; q < n_qpts; ++q) {
                const auto out = sensitivityIndex(n_nodes, q, node, component);
                const auto qidx = static_cast<std::size_t>(q);
                for (std::size_t d = 0; d < 3u; ++d) {
                    sensitivity.normals.values[out][d] =
                        (plus_geom.normals[qidx][d] - minus_geom.normals[qidx][d]) * inv_2h;
                }
                sensitivity.measures.values[out] =
                    (plus_geom.surface_measures[qidx] -
                     minus_geom.surface_measures[qidx]) *
                    inv_2h;
            }

            plus[static_cast<std::size_t>(node)][static_cast<std::size_t>(component)] -= step;
            minus[static_cast<std::size_t>(node)][static_cast<std::size_t>(component)] += step;
        }
    }

    return sensitivity;
}

FaceGeometrySensitivity evaluateFaceGeometrySensitivityAnalytic(
    const GeometryMapping& mapping,
    ElementType cell_type,
    LocalIndex local_face_id,
    ElementType face_type,
    const quadrature::QuadratureRule& quad_rule,
    std::span<const LocalIndex> align_facet_to_reference)
{
    const auto face = evaluateFaceFrame(mapping,
                                        cell_type,
                                        local_face_id,
                                        face_type,
                                        quad_rule,
                                        align_facet_to_reference);
    const auto n_qpts = static_cast<LocalIndex>(quad_rule.num_points());
    const auto n_nodes = static_cast<LocalIndex>(mapping.num_nodes());
    const std::size_t total =
        static_cast<std::size_t>(n_qpts) * static_cast<std::size_t>(n_nodes) * 3u;

    FaceGeometrySensitivity sensitivity;
    sensitivity.n_qpts = n_qpts;
    sensitivity.n_nodes = n_nodes;
    sensitivity.normals = {n_qpts, n_nodes, std::vector<Vector3D>(total)};
    sensitivity.measures = {n_qpts, n_nodes, std::vector<Real>(total)};

    const auto& geometry_basis = mapping.geometryBasis();
    FE_THROW_IF(static_cast<LocalIndex>(geometry_basis.size()) != n_nodes, FEException,
                "FrameGeometry: geometry basis size does not match mapping node count");

    std::vector<basis::Gradient> gradients;
    const int face_dim = (face_type == ElementType::Line2) ? 1 : 2;

    const auto& qpts = quad_rule.points();
    for (LocalIndex q = 0; q < n_qpts; ++q) {
        const auto qidx = static_cast<std::size_t>(q);
        const auto& canonical_point = qpts[qidx];
        const auto facet_coords =
            canonicalToFacetCoordinates(face_type, canonical_point, align_facet_to_reference);
        const auto xi = elements::ElementTransform::facet_to_reference(
            cell_type, static_cast<int>(local_face_id), facet_coords);
        const auto dxi_dcanon =
            referenceToCanonicalFaceJacobian(cell_type,
                                             local_face_id,
                                             face_type,
                                             canonical_point,
                                             align_facet_to_reference);

        geometry_basis.evaluate_gradients(xi, gradients);
        FE_THROW_IF(static_cast<LocalIndex>(gradients.size()) != n_nodes,
                    FEException,
                    "FrameGeometry: geometry basis gradient size mismatch");

        const auto J = mapping.jacobian(xi);
        const auto T0 = toMathVector(Vector3D{face.surface_jacobians[qidx][0][0],
                                             face.surface_jacobians[qidx][1][0],
                                             face.surface_jacobians[qidx][2][0]});
        const auto T1 = (face_dim == 2)
                            ? toMathVector(Vector3D{face.surface_jacobians[qidx][0][1],
                                                    face.surface_jacobians[qidx][1][1],
                                                    face.surface_jacobians[qidx][2][1]})
                            : math::Vector<Real, 3>{};
        const auto evaluated_normal = toMathVector(face.normals[qidx]);

        math::Vector<Real, 3> raw_vector{};
        Real raw_norm = Real(0);
        Real orientation_sign = Real(1);
        if (face_dim == 2) {
            raw_vector = T0.cross(T1);
            raw_norm = norm(raw_vector);
        } else {
            const auto cell_normal = matrixColumn(J, 2);
            raw_vector = cell_normal.cross(T0);
            raw_norm = norm(raw_vector);
        }
        if (raw_norm > detail::kDegenerateTol) {
            const math::Vector<Real, 3> raw_unit = raw_vector / raw_norm;
            orientation_sign = (raw_unit.dot(evaluated_normal) >= Real(0)) ? Real(1) : Real(-1);
        }

        for (LocalIndex node = 0; node < n_nodes; ++node) {
            const auto nidx = static_cast<std::size_t>(node);
            for (int component = 0; component < 3; ++component) {
                const auto out = sensitivityIndex(n_nodes, q, node, component);

                auto dT0 = math::Vector<Real, 3>{};
                auto dT1 = math::Vector<Real, 3>{};
                for (int xi_dir = 0; xi_dir < mapping.dimension(); ++xi_dir) {
                    const auto xidx = static_cast<std::size_t>(xi_dir);
                    dT0[static_cast<std::size_t>(component)] +=
                        gradients[nidx][xidx] * dxi_dcanon(xidx, 0);
                    if (face_dim == 2) {
                        dT1[static_cast<std::size_t>(component)] +=
                            gradients[nidx][xidx] * dxi_dcanon(xidx, 1);
                    }
                }

                math::Vector<Real, 3> draw{};
                Real dmeasure = Real(0);
                if (face_dim == 2) {
                    draw = dT0.cross(T1) + T0.cross(dT1);
                    if (raw_norm > detail::kDegenerateTol) {
                        const math::Vector<Real, 3> raw_unit = raw_vector / raw_norm;
                        dmeasure = raw_unit.dot(draw);
                    }
                } else {
                    const auto dJ =
                        embeddedFrameJacobianDerivative(J,
                                                        mapping.dimension(),
                                                        gradients[nidx],
                                                        component);
                    const auto dcell_normal =
                        math::Vector<Real, 3>{dJ[0][2], dJ[1][2], dJ[2][2]};
                    dmeasure = unitOrZero(T0).dot(dT0);
                    draw = dcell_normal.cross(T0) + matrixColumn(J, 2).cross(dT0);
                }

                math::Vector<Real, 3> dnormal{};
                if (raw_norm > detail::kDegenerateTol) {
                    const math::Vector<Real, 3> raw_unit = raw_vector / raw_norm;
                    dnormal = projectedUnitDerivative(raw_unit, draw, raw_norm) * orientation_sign;
                }

                sensitivity.normals.values[out] = toArrayVector(dnormal);
                sensitivity.measures.values[out] = dmeasure;
            }
        }
    }

    return sensitivity;
}

FaceGeometrySensitivity evaluateFaceGeometrySensitivity(
    ElementType cell_type,
    LocalIndex local_face_id,
    ElementType face_type,
    const quadrature::QuadratureRule& quad_rule,
    std::span<const Point3D> coordinates,
    std::span<const LocalIndex> align_facet_to_reference)
{
    const auto mapping = makeMapping(cell_type, coordinates);
    return evaluateFaceGeometrySensitivityAnalytic(*mapping,
                                                   cell_type,
                                                   local_face_id,
                                                   face_type,
                                                   quad_rule,
                                                   align_facet_to_reference);
}

} // namespace geometry
} // namespace FE
} // namespace svmp
