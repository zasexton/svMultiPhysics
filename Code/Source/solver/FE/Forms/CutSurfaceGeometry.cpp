/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/CutSurfaceGeometry.h"

#include "Basis/NodeOrderingConventions.h"
#include "Geometry/CutQuadratureMapping.h"
#include "Spaces/FunctionSpace.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace svmp::FE::forms::cut_surface {
namespace {

using Vector = math::Vector<Real, 3>;
using Matrix = math::Matrix<Real, 3, 3>;
constexpr Real reference_tolerance = 256 * std::numeric_limits<Real>::epsilon();

[[noreturn]] void unsupported(const char *cause)
{
    throw std::invalid_argument(std::string("differentiated cut surface: ") + cause);
}

Vector vectorOf(const std::array<Real, 3> &point)
{
    return {point[0], point[1], point[2]};
}

std::array<Real, 3> arrayOf(const Vector &point)
{
    return {point[0], point[1], point[2]};
}

std::size_t parentCornerCount(ElementType type)
{
    switch (type) {
        case ElementType::Triangle3:
        case ElementType::Triangle6:
            return 3;
        case ElementType::Quad4:
        case ElementType::Quad8:
        case ElementType::Quad9:
        case ElementType::Tetra4:
        case ElementType::Tetra10:
            return 4;
        default:
            unsupported("unsupported reference parent");
    }
}

struct EdgeIntersection {
    Vector point;
    std::vector<Vector> motion;
};

std::vector<EdgeIntersection> edgeIntersections(const geometry::CutQuadratureRule &rule, ElementType type,
                                                GlobalIndex cell, const spaces::FunctionSpace &field,
                                                std::span<const Real> coefficients,
                                                const spaces::FunctionSpace &trial, const Vector &normal)
{
    struct Corner {
        Vector point;
        Real signed_value;
        std::vector<Real> trial_values;
    };
    const auto &source = *rule.linear_corner_surface;
    if (source.count > source.edges.size()) unsupported("retained surface exceeds its edge storage");
    const bool active_wall = rule.provenance.parent_boundary_entity >= 0;
    const auto &field_basis = field.getElement(type, cell).basis();
    const auto &trial_basis = trial.getElement(type, cell).basis();
    const auto dof_count = trial.dofs_per_element(cell);
    std::vector<Corner> corners(parentCornerCount(type));
    for (std::size_t i = 0; i < corners.size(); ++i) {
        auto &corner = corners[i];
        corner.point = basis::ReferenceNodeLayout::get_node_coords(type, i);
        std::vector<Real> values;
        field_basis.evaluate_values(corner.point, values);
        trial_basis.evaluate_values(corner.point, corner.trial_values);
        if (values.size() != coefficients.size() || corner.trial_values.size() != dof_count) {
            unsupported("source and trial require scalar finite-element coefficients");
        }
        corner.signed_value =
            std::inner_product(values.begin(), values.end(), coefficients.begin(), -source.isovalue);
        if (std::abs(corner.signed_value) <= rule.provenance.coefficient_classification_band) {
            unsupported("interface meets a parent vertex; cut topology has no two-sided derivative");
        }
    }

    const Vector sample = vectorOf(rule.points.front().point);
    std::vector<EdgeIntersection> intersections;
    for (std::size_t index = 0; index < source.count; ++index) {
        const auto [first, last] = source.edges[index];
        if (first >= corners.size() || last >= corners.size() || (first == last && !active_wall)) {
            unsupported("retained vertex has no transverse parent edge");
        }
        const auto &a = corners[first];
        if (first == last) {
            const bool negative = rule.side == geometry::CutIntegrationSide::Negative;
            if ((a.signed_value < 0) != negative) unsupported("fixed wall corner no longer belongs to its phase");
            intersections.push_back({a.point, std::vector<Vector>(dof_count)});
            continue;
        }
        const auto &b = corners[last];
        const Real slope = b.signed_value - a.signed_value;
        if ((a.signed_value < 0) == (b.signed_value < 0)) {
            unsupported("source no longer crosses the retained parent edge");
        }
        const Real t = -a.signed_value / slope;
        const Vector edge = b.point - a.point;
        auto &root = intersections.emplace_back();
        root.point = a.point + t * edge;
        if ((!active_wall && normal.dot(edge) * slope <= 0) || std::abs(normal.dot(root.point - sample)) > reference_tolerance) {
            unsupported("source coefficients are stale for the retained edge intersections");
        }
        for (std::size_t j = 0; j < dof_count; ++j) {
            // Differentiate (1-t) phi_a + t phi_b = isovalue on the fixed parent edge.
            const Real level_set_seed = (1 - t) * a.trial_values[j] + t * b.trial_values[j];
            root.motion.emplace_back((-level_set_seed / slope) * edge);
        }
    }
    return intersections;
}

struct Triangle {
    std::size_t second; // The retained fan uses vertices (0, second, second+1).
    Real area;
    Vector normal;
};

std::vector<Triangle> triangleFan(const std::vector<EdgeIntersection> &roots)
{
    std::vector<Triangle> triangles;
    for (std::size_t i = 1; i + 1 < roots.size(); ++i) {
        const Vector ab = roots[i].point - roots[0].point;
        const Vector ac = roots[i + 1].point - roots[0].point;
        const Vector area_vector = math::cross(ab, ac);
        const Real length = area_vector.norm();
        if (!(length > 0)) {
            unsupported("collapsed triangle in retained quadrature fan");
        }
        triangles.push_back({i, length / 2, area_vector / length});
    }
    return triangles;
}

struct ReferenceVariation {
    Vector point;
    Vector normal;
    Real relative_weight = 0;
};

ReferenceVariation segmentVariation(const geometry::CutQuadraturePoint &sample, const Vector &normal,
                                    const std::vector<EdgeIntersection> &roots, std::size_t j)
{
    const auto &a = roots.front();
    const auto &b = roots.back();
    const Vector edge = b.point - a.point;
    const Vector edge_velocity = b.motion[j] - a.motion[j];
    const Real length_squared = edge.dot(edge);
    if (!(length_squared > 0)) {
        unsupported("collapsed retained segment");
    }
    const Real t = edge.dot(vectorOf(sample.point) - a.point) / length_squared;
    if (t < -reference_tolerance || t > 1 + reference_tolerance) {
        unsupported("sample lies outside retained segment");
    }
    return {a.motion[j] + t * edge_velocity, -normal.dot(edge_velocity) / length_squared * edge,
            edge.dot(edge_velocity) / length_squared};
}

ReferenceVariation polygonVariation(const geometry::CutQuadratureRule &rule, const Vector &normal,
                                    const std::vector<EdgeIntersection> &roots,
                                    const std::vector<Triangle> &triangles, std::size_t q, std::size_t j)
{
    const bool centroid = rule.points.size() == 1 && rule.provenance.parent_boundary_entity < 0;
    if (!centroid && (rule.points.size() % triangles.size() != 0 || rule.points.size() < triangles.size())) {
        unsupported("unsupported retained polygon quadrature");
    }
    const Vector sample = vectorOf(rule.points[q].point);
    const auto faces = std::span<const Triangle>(triangles);
    const auto active = centroid ? faces : faces.subspan(q / (rule.points.size() / triangles.size()), 1);
    const auto &a = roots[0];
    ReferenceVariation variation;
    Real total_area = 0;
    Vector expected_point;
    for (const auto &face : active) {
        const auto &b = roots[face.second];
        const auto &c = roots[face.second + 1];
        const Vector ab = b.point - a.point, ac = c.point - a.point;
        const Vector dab = b.motion[j] - a.motion[j], dac = c.motion[j] - a.motion[j];
        const Vector area_vector_rate = math::cross(dab, ac) + math::cross(ab, dac);
        const Real twice_area_rate = face.normal.dot(area_vector_rate);
        const Real area_rate = twice_area_rate / 2;
        const Real orientation = face.normal.dot(normal) > 0 ? 1 : -1;
        variation.normal = orientation / (2 * face.area) * (area_vector_rate - twice_area_rate * face.normal);
        total_area += face.area;
        variation.relative_weight += area_rate;
        if (centroid) {
            const Vector triangle_center = (a.point + b.point + c.point) / 3.;
            const Vector center_motion = (a.motion[j] + b.motion[j] + c.motion[j]) / 3.;
            expected_point += face.area * triangle_center;
            variation.point += face.area * center_motion + area_rate * (triangle_center - sample);
        } else {
            // Each retained sample has fixed barycentric coordinates in its moving fan triangle.
            if (rule.provenance.parent_boundary_entity < 0) {
                if (rule.points.size() != 3 * triangles.size()) unsupported("unsupported interior polygon quadrature");
                const auto& vertex = *std::array{&a, &b, &c}[q % 3];
                const Vector degree_two_sample = (a.point + b.point + c.point) / 6. + vertex.point / 2.;
                if (Vector(degree_two_sample - sample).norm() > reference_tolerance) {
                    unsupported("sample disagrees with retained polygon quadrature");
                }
            }
            const Vector offset = sample - a.point;
            const Real beta = math::cross(offset, ac).dot(face.normal) / (2 * face.area);
            const Real gamma = math::cross(ab, offset).dot(face.normal) / (2 * face.area);
            if (beta < -reference_tolerance || gamma < -reference_tolerance || beta + gamma > 1 + reference_tolerance) {
                unsupported("sample lies outside its retained fan triangle");
            }
            expected_point = a.point + beta * ab + gamma * ac;
            variation.point = a.motion[j] + beta * dab + gamma * dac;
        }
    }
    variation.relative_weight /= total_area;
    if (centroid) {
        expected_point /= total_area;
        variation.point /= total_area;
    }
    if (Vector(expected_point - sample).norm() > reference_tolerance) {
        unsupported("sample disagrees with retained fan triangle or polygon centroid");
    }
    return variation;
}

// The mapping Hessian differentiates the physical coordinate columns; the
// completed Jacobian of a 2D parent has its normalized surface normal in column 3.
Matrix mappingJacobianRate(const geometry::GeometryMapping::MappingHessian &hessian, const Matrix &jacobian,
                           int dimension, const Vector &reference_motion)
{
    Matrix rate;
    for (int i = 0; i < 3; ++i) {
        const Vector row = hessian[i] * reference_motion;
        for (int k = 0; k < dimension; ++k) {
            rate(i, k) = row[k];
        }
    }
    if (dimension == 2) {
        const Vector tangent0 = jacobian.column(0), tangent1 = jacobian.column(1);
        const Vector normal = jacobian.column(2);
        const Vector area_rate = math::cross(rate.column(0), tangent1) + math::cross(tangent0, rate.column(1));
        rate.set_column(2,
                        (area_rate - normal.dot(area_rate) * normal) / math::cross(tangent0, tangent1).norm());
    }
    return rate;
}

class PhysicalSample {
  public:
    PhysicalSample(const geometry::GeometryMapping &map, const geometry::CutQuadraturePoint &sample,
                   const Vector &reference_normal, int dim)
        : dimension(dim), reference_point(vectorOf(sample.point)), jacobian(map.jacobian(reference_point)),
          inverse(map.jacobian_inverse(reference_point)), hessian(map.mapping_hessian(reference_point)),
          covector(inverse.transpose() * reference_normal), normal_scale(covector.norm()),
          physical_tangent(jacobian * Vector{-reference_normal[1], reference_normal[0], 0})
    {
        if (!(normal_scale > 0)) {
            unsupported("singular physical geometry map");
        }
        normal = covector / normal_scale;
        weight = sample.weight * (dimension == 2
                                      ? physical_tangent.norm()
                                      : std::abs(map.jacobian_determinant(reference_point)) * normal_scale);
    }

    assembly::CutSurfaceSampleVariation differentiate(const ReferenceVariation &variation) const
    {
        assembly::CutSurfaceSampleVariation result;
        result.reference_point = arrayOf(variation.point);
        const Vector physical_motion = jacobian * variation.point;
        result.point = arrayOf(physical_motion);
        result.normal_speed = normal.dot(physical_motion);
        result.field_point = result.point;
        const Matrix jacobian_rate = mappingJacobianRate(hessian, jacobian, dimension, variation.point);
        const Matrix inverse_rate = (inverse * jacobian_rate * inverse) * Real{-1};
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                result.jacobian[row][col] = jacobian_rate(row, col);
                result.inverse_jacobian[row][col] = inverse_rate(row, col);
            }
        }
        result.relative_jacobian_determinant = (inverse * jacobian_rate).trace();
        // Differentiate n = c / |c|, c = J^{-T} n_ref, including the
        // motion of the quadrature point through the nonaffine parent.
        const Vector covector_rate =
            inverse.transpose() * Vector(variation.normal - jacobian_rate.transpose() * covector);
        const Real scale_rate = normal.dot(covector_rate);
        result.normal = arrayOf((covector_rate - scale_rate * normal) / normal_scale);
        if (dimension == 3) {
            result.relative_weight =
                variation.relative_weight + result.relative_jacobian_determinant + scale_rate / normal_scale;
        } else {
            // In 2D the native measure uses |J t_ref|, even for an embedded parent.
            const Vector reference_tangent_rate{-variation.normal[1], variation.normal[0], 0};
            const Vector reference_tangent = inverse * physical_tangent;
            const Vector tangent_rate = jacobian_rate * reference_tangent + jacobian * reference_tangent_rate;
            result.relative_weight = variation.relative_weight + physical_tangent.dot(tangent_rate) /
                                                                     physical_tangent.dot(physical_tangent);
            // A 2D field gradient uses the x and y components of J^{-T} grad_ref(u).
            // Subtracting the third Jacobian column preserves both reference components
            // of the motion. Choose its multiplier so the resulting z component is zero.
            const Vector parent_normal = jacobian.column(2);
            if (parent_normal[2] == 0) {
                unsupported("the parent tangent plane has a singular x-y projection for 2D field gradients");
            }
            const Real normal_motion = result.point[2] / parent_normal[2];
            result.field_point = {result.point[0] - normal_motion * parent_normal[0],
                                  result.point[1] - normal_motion * parent_normal[1], 0};
        }
        return result;
    }

    Real weight = 0;

  private:
    int dimension;
    Vector reference_point;
    Matrix jacobian, inverse;
    geometry::GeometryMapping::MappingHessian hessian;
    Vector covector;
    Real normal_scale;
    Vector physical_tangent, normal;
};

} // namespace

// Explicit quadrature on a smooth implicit zero set uses the physical
// level-set speed. A clipped linear-corner chord has its own retained motion.
Table implicitNormalSpeed(const assembly::AssemblyContext& context, FieldId field)
{
    Table result(context.numTrialDofs(), std::vector<assembly::CutSurfaceSampleVariation>(context.numQuadraturePoints()));
    for (LocalIndex q = 0; q < context.numQuadraturePoints(); ++q) {
        const auto gradient = context.fieldGradient(field, q);
        const Real magnitude = std::hypot(gradient[0], gradient[1], gradient[2]);
        if (!(magnitude > 0)) unsupported("implicit interface has zero level-set gradient");
        for (LocalIndex j = 0; j < context.numTrialDofs(); ++j) {
            result[j][q].normal_speed = -context.trialBasisValue(j, q) / magnitude;
        }
    }
    return result;
}

std::vector<Real> referenceFractionDerivative(const geometry::CutQuadratureRule& surface,
    ElementType type, GlobalIndex cell, const spaces::FunctionSpace& field, std::span<const Real> coefficients,
    const spaces::FunctionSpace& trial, Real signed_inverse_parent_measure)
{
    if (!surface.linear_corner_surface || surface.curved_geometry ||
        surface.frame != geometry::CutGeometryFrame::Reference || surface.points.empty()) {
        unsupported("cut fraction requires an original reference LinearCorner surface");
    }
    const Vector normal = vectorOf(surface.points.front().normal);
    if (std::abs(normal.norm() - 1) > reference_tolerance) unsupported("cut fraction requires a unit reference normal");
    const bool tet = type == ElementType::Tetra4 || type == ElementType::Tetra10;
    const auto count = surface.linear_corner_surface->count;
    if (tet ? (count != 3 && count != 4) : count != 2) unsupported("cut fraction surface has the wrong dimension");
    const auto roots = edgeIntersections(surface, type, cell, field, coefficients, trial, normal);
    std::vector<Real> derivative(trial.dofs_per_element(cell), Real{0});
    // The root velocities interpolate linearly on each segment or fan triangle. Their
    // mean normal component integrates exactly, independently of the physical mapping.
    if (roots.size() == 2) {
        const Real half_length = Vector(roots[1].point - roots[0].point).norm() / 2;
        for (std::size_t j = 0; j < derivative.size(); ++j) {
            derivative[j] = half_length * normal.dot(roots[0].motion[j] + roots[1].motion[j]);
        }
    } else {
        for (const auto& triangle : triangleFan(roots)) {
            for (std::size_t j = 0; j < derivative.size(); ++j) {
                derivative[j] += (triangle.area / 3) * normal.dot(
                    roots[0].motion[j] + roots[triangle.second].motion[j] + roots[triangle.second + 1].motion[j]);
            }
        }
    }
    for (auto& entry : derivative) entry *= signed_inverse_parent_measure;
    return derivative;
}

Table differentiateRule(const assembly::IMeshAccess &mesh, GlobalIndex cell,
                        const geometry::CutQuadratureRule &rule, const spaces::FunctionSpace &field,
                        std::span<const Real> coefficients, const spaces::FunctionSpace &trial)
{
    if (rule.kind != geometry::CutQuadratureKind::Interface || rule.curved_geometry ||
        rule.frame != geometry::CutGeometryFrame::Reference || !rule.linear_corner_surface ||
        rule.points.empty()) {
        unsupported(
            "requires a generated LinearCorner segment or planar polygon with differentiable quadrature");
    }
    const int dimension = mesh.dimension();
    const auto root_count = rule.linear_corner_surface->count;
    if (!((dimension == 2 && root_count == 2) || (dimension == 3 && (root_count == 3 || root_count == 4)))) {
        unsupported("invalid retained edge count or parent dimension");
    }
    const Vector reference_normal = vectorOf(rule.points.front().normal);
    if (std::abs(reference_normal.norm() - 1) > reference_tolerance) {
        unsupported("expected a unit reference normal");
    }
    const auto roots =
        edgeIntersections(rule, mesh.getCellType(cell), cell, field, coefficients, trial, reference_normal);
    const auto triangles = triangleFan(roots);
    const auto map = geometry::makeCutCellGeometryMapping(mesh, cell);
    const Vector plane_point = vectorOf(rule.points.front().point);
    std::vector<Real> weights(rule.points.size());
    Table result(trial.dofs_per_element(cell),
                 std::vector<assembly::CutSurfaceSampleVariation>(rule.points.size()));
    for (std::size_t q = 0; q < rule.points.size(); ++q) {
        const auto &sample = rule.points[q];
        if (Vector(vectorOf(sample.normal) - reference_normal).norm() > reference_tolerance ||
            std::abs(reference_normal.dot(vectorOf(sample.point) - plane_point)) > reference_tolerance ||
            !(sample.weight > 0)) {
            unsupported("nonplanar, unoriented or zero-weight reference sample");
        }
        const PhysicalSample physical(*map, sample, reference_normal, dimension);
        weights[q] = physical.weight;
        for (std::size_t j = 0; j < result.size(); ++j) {
            const auto reference = dimension == 2
                                       ? segmentVariation(sample, reference_normal, roots, j)
                                       : polygonVariation(rule, reference_normal, roots, triangles, q, j);
            result[j][q] = physical.differentiate(reference);
        }
    }
    for (auto &column : result) {
        Real measure_rate = 0;
        for (std::size_t q = 0; q < column.size(); ++q) {
            measure_rate += weights[q] * column[q].relative_weight;
        }
        for (auto &sample : column) {
            sample.facet_measure = measure_rate;
        }
    }
    return result;
}

} // namespace svmp::FE::forms::cut_surface
