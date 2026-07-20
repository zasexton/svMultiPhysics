#include "Geometry/CutQuadratureMapping.h"

#include "Assembly/Assembler.h"
#include "Geometry/MappingFactory.h"
#include "Quadrature/QuadratureFactory.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace svmp::FE::geometry {
namespace {

using Point = std::array<Real, 3>;

[[nodiscard]] CutGeometryJacobian toArray(
    const math::Matrix<Real, 3, 3>& matrix) noexcept
{
    CutGeometryJacobian result{};
    for (std::size_t i = 0; i < 3u; ++i) {
        for (std::size_t j = 0; j < 3u; ++j) {
            result[i][j] = matrix(i, j);
        }
    }
    return result;
}

[[nodiscard]] Point toArray(const math::Vector<Real, 3>& vector) noexcept
{
    return {{vector[0], vector[1], vector[2]}};
}

[[nodiscard]] math::Vector<Real, 3> toVector(const Point& point) noexcept
{
    return {point[0], point[1], point[2]};
}

[[nodiscard]] Real dot(const Point& a, const Point& b) noexcept
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

[[nodiscard]] Real norm(const Point& point) noexcept
{
    return std::sqrt(dot(point, point));
}

[[nodiscard]] Point normalized(Point point) noexcept
{
    const Real magnitude = norm(point);
    if (!(magnitude > Real{0.0}) || !std::isfinite(magnitude)) {
        return {{0.0, 0.0, 0.0}};
    }
    for (auto& value : point) {
        value /= magnitude;
    }
    return point;
}

[[nodiscard]] Point mapCovector(const CutGeometryJacobian& inverse_jacobian,
                                const Point& covector) noexcept
{
    Point result{{0.0, 0.0, 0.0}};
    for (std::size_t i = 0; i < 3u; ++i) {
        for (std::size_t j = 0; j < 3u; ++j) {
            result[i] += inverse_jacobian[j][i] * covector[j];
        }
    }
    return result;
}

[[nodiscard]] int resolvedGeometricDimension(const CutQuadratureRule& rule,
                                              int parent_dimension) noexcept
{
    if (rule.geometric_dimension >= 0) {
        return rule.geometric_dimension;
    }
    return rule.kind == CutQuadratureKind::Volume
               ? parent_dimension
               : parent_dimension - 1;
}

void requireFinitePoint(const Point& point, const char* label)
{
    if (!std::isfinite(point[0]) || !std::isfinite(point[1]) ||
        !std::isfinite(point[2])) {
        throw std::invalid_argument(std::string("non-finite ") + label +
                                    " in retained cut quadrature");
    }
}

} // namespace

std::shared_ptr<GeometryMapping> makeCutCellGeometryMapping(
    const assembly::IMeshAccess& mesh,
    GlobalIndex cell_id)
{
    if (cell_id < 0 || cell_id >= mesh.numCells()) {
        throw std::out_of_range(
            "retained cut quadrature parent cell is outside the local mesh");
    }
    std::vector<Point> coordinates;
    mesh.getCellCoordinates(cell_id, coordinates);
    if (coordinates.empty()) {
        throw std::invalid_argument(
            "retained cut quadrature parent cell has no geometry nodes");
    }
    std::vector<math::Vector<Real, 3>> nodes;
    nodes.reserve(coordinates.size());
    for (const auto& coordinate : coordinates) {
        requireFinitePoint(coordinate, "geometry node");
        nodes.push_back(toVector(coordinate));
    }
    MappingRequest request;
    request.element_type = mesh.getCellType(cell_id);
    request.geometry_order = mesh.getCellGeometryOrder(cell_id);
    request.use_affine = request.geometry_order <= 1;
    return MappingFactory::create(request, nodes);
}

Real physicalCellMeasureFromMapping(const assembly::IMeshAccess& mesh,
                                    GlobalIndex cell_id,
                                    int quadrature_order)
{
    const auto mapping = makeCutCellGeometryMapping(mesh, cell_id);
    const int geometry_order = std::max(1, mesh.getCellGeometryOrder(cell_id));
    const int resolved_order = quadrature_order >= 0
                                   ? quadrature_order
                                   : std::max(2, 2 * geometry_order);
    const auto quadrature = quadrature::QuadratureFactory::create(
        mesh.getCellType(cell_id), resolved_order);
    Real measure{0.0};
    for (std::size_t q = 0; q < quadrature->num_points(); ++q) {
        const Real determinant =
            mapping->jacobian_determinant(quadrature->point(q));
        const Real weight = quadrature->weight(q);
        if (!std::isfinite(determinant) || !std::isfinite(weight) ||
            !(weight > Real{0.0})) {
            throw std::invalid_argument(
                "cell measure mapping found a non-finite Jacobian or invalid quadrature weight");
        }
        measure += weight * std::abs(determinant);
    }
    if (!std::isfinite(measure) || !(measure > Real{0.0})) {
        throw std::invalid_argument("cell mapping produced an invalid physical measure");
    }
    return measure;
}

MappedCutQuadratureRule mapCutQuadratureRuleToPhysical(
    const assembly::IMeshAccess& mesh,
    const CutQuadratureRule& rule)
{
    if (rule.provenance.parent_entity < 0 || rule.points.empty()) {
        throw std::invalid_argument(
            "retained cut quadrature requires a parent cell and at least one point");
    }
    const auto cell = static_cast<GlobalIndex>(rule.provenance.parent_entity);
    const auto mapping = makeCutCellGeometryMapping(mesh, cell);
    const int parent_dimension = mesh.dimension();
    const int geometric_dimension =
        resolvedGeometricDimension(rule, parent_dimension);
    if (geometric_dimension < 0 || geometric_dimension > parent_dimension) {
        throw std::invalid_argument(
            "retained cut quadrature has an invalid geometric dimension");
    }

    MappedCutQuadratureRule mapped;
    mapped.kind = rule.kind;
    mapped.side = rule.side;
    mapped.geometric_dimension = geometric_dimension;
    mapped.parent_entity = rule.provenance.parent_entity;
    mapped.marker = rule.provenance.marker;
    mapped.source_stable_id = rule.provenance.source_stable_id;
    mapped.cut_topology_revision = rule.provenance.cut_topology_revision;
    mapped.source_value_revision = rule.provenance.source_value_revision;
    mapped.reference_measure = rule.measure;
    mapped.points.reserve(rule.points.size());

    for (const auto& point : rule.points) {
        if (!std::isfinite(point.weight) || !(point.weight > Real{0.0})) {
            throw std::invalid_argument(
                "retained cut quadrature has a non-positive or non-finite weight");
        }
        const Point reference =
            rule.frame == CutGeometryFrame::Reference ? point.point
                                                      : point.parent_coordinate;
        requireFinitePoint(reference, "reference point");
        const auto xi = toVector(reference);
        const auto jacobian = mapping->jacobian(xi);
        const Real determinant = jacobian.determinant();
        if (!std::isfinite(determinant) ||
            !(std::abs(determinant) >
              std::numeric_limits<Real>::min())) {
            throw std::invalid_argument(
                "retained cut quadrature maps through a singular or non-finite Jacobian");
        }
        const auto inverse = jacobian.inverse();

        MappedCutQuadraturePoint output;
        output.reference_point = reference;
        output.physical_point = toArray(mapping->map_to_physical(xi));
        output.jacobian = toArray(jacobian);
        output.inverse_jacobian = toArray(inverse);
        output.absolute_jacobian_determinant = std::abs(determinant);
        output.reference_weight = point.weight;

        if (rule.frame == CutGeometryFrame::Current) {
            requireFinitePoint(point.point, "current-frame point");
            output.physical_point = point.point;
            output.physical_weight = point.weight;
            output.normal = normalized(point.normal);
            output.boundary_normal = normalized(point.boundary_normal);
            output.tangent = normalized(point.tangent);
        } else if (geometric_dimension == parent_dimension) {
            output.physical_weight =
                point.weight * output.absolute_jacobian_determinant;
            output.normal = normalized(
                mapCovector(output.inverse_jacobian, point.normal));
        } else if (geometric_dimension == parent_dimension - 1) {
            const Point reference_normal =
                norm(point.boundary_normal) > Real{0.0}
                    ? point.boundary_normal
                    : point.normal;
            const auto mapped_normal =
                mapCovector(output.inverse_jacobian, reference_normal);
            const Real normal_scale = norm(mapped_normal);
            if (!(normal_scale > Real{0.0}) || !std::isfinite(normal_scale)) {
                throw std::invalid_argument(
                    "retained cut boundary has an invalid mapped normal");
            }
            output.physical_weight = point.weight *
                                     output.absolute_jacobian_determinant *
                                     normal_scale;
            output.normal = normalized(
                mapCovector(output.inverse_jacobian, point.normal));
            output.boundary_normal = normalized(mapped_normal);
        } else if (geometric_dimension == parent_dimension - 2) {
            const auto codimension_two =
                mapReferenceCutCodimensionTwoGeometry(
                    point,
                    parent_dimension,
                    output.jacobian,
                    output.inverse_jacobian);
            output.physical_weight = codimension_two.weight;
            output.normal = codimension_two.interface_normal;
            output.boundary_normal = codimension_two.boundary_normal;
            output.tangent = codimension_two.tangent;
        } else {
            throw std::invalid_argument(
                "retained cut quadrature mapping supports only codimension zero, one, or two");
        }
        if (!std::isfinite(output.physical_weight) ||
            !(output.physical_weight > Real{0.0})) {
            throw std::invalid_argument(
                "retained cut quadrature has a non-positive mapped weight");
        }
        mapped.physical_measure += output.physical_weight;
        mapped.points.push_back(std::move(output));
    }
    if (!std::isfinite(mapped.physical_measure) ||
        !(mapped.physical_measure > Real{0.0})) {
        throw std::invalid_argument(
            "retained cut quadrature has an invalid physical measure");
    }
    return mapped;
}

Real physicalCutQuadratureMeasure(const assembly::IMeshAccess& mesh,
                                  const CutQuadratureRule& rule)
{
    return mapCutQuadratureRuleToPhysical(mesh, rule).physical_measure;
}

} // namespace svmp::FE::geometry
