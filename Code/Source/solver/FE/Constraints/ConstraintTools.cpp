/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "ConstraintTools.h"
#include "MultiPointConstraint.h"
#include "PeriodicBC.h"
#include "Spaces/HDivSpace.h"
#include "Spaces/TraceSpace.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace svmp {
namespace FE {
namespace constraints {

namespace {

using Point3 = std::array<double, 3>;

double squaredDistance(const Point3& a, const Point3& b)
{
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    const double dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
}

Point3 applyTransform(const std::function<Point3(Point3)>& transform, const Point3& value)
{
    if (transform) {
        return transform(value);
    }
    return value;
}

Point3 normalizedNormal(const Point3& normal, const char* label)
{
    const double norm_sq = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
    FE_CHECK_ARG(norm_sq > 0.0, label);
    const double inv_norm = 1.0 / std::sqrt(norm_sq);
    return Point3{{normal[0] * inv_norm, normal[1] * inv_norm, normal[2] * inv_norm}};
}

ElementType traceElementTypeFromVertexCount(std::size_t n_vertices)
{
    switch (n_vertices) {
        case 2u:
            return ElementType::Line2;
        case 3u:
            return ElementType::Triangle3;
        case 4u:
            return ElementType::Quad4;
        default:
            throw FEException("ConstraintTools: unsupported boundary trace entity vertex count",
                              __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
}

std::optional<std::vector<int>> matchEntityVertices(std::span<const Point3> slave_vertices,
                                                    std::span<const Point3> master_vertices,
                                                    const std::function<Point3(Point3)>& point_transform,
                                                    double tolerance)
{
    if (slave_vertices.size() != master_vertices.size()) {
        return std::nullopt;
    }

    const double tol_sq = tolerance * tolerance;
    std::vector<int> matched(slave_vertices.size(), -1);
    std::vector<bool> used(master_vertices.size(), false);

    for (std::size_t i = 0; i < slave_vertices.size(); ++i) {
        const Point3 transformed = applyTransform(point_transform, slave_vertices[i]);
        double best_dist_sq = std::numeric_limits<double>::max();
        int best_j = -1;

        for (std::size_t j = 0; j < master_vertices.size(); ++j) {
            if (used[j]) {
                continue;
            }

            const double dist_sq = squaredDistance(transformed, master_vertices[j]);
            if (dist_sq < best_dist_sq) {
                best_dist_sq = dist_sq;
                best_j = static_cast<int>(j);
            }
        }

        if (best_j < 0 || best_dist_sq > tol_sq) {
            return std::nullopt;
        }

        used[static_cast<std::size_t>(best_j)] = true;
        matched[i] = best_j;
    }

    return matched;
}

double maxMatchedVertexDistance(std::span<const Point3> slave_vertices,
                                std::span<const Point3> master_vertices,
                                const std::function<Point3(Point3)>& point_transform,
                                std::span<const int> matched)
{
    double max_dist_sq = 0.0;
    for (std::size_t i = 0; i < slave_vertices.size(); ++i) {
        const Point3 transformed = applyTransform(point_transform, slave_vertices[i]);
        const int master_idx = matched[i];
        FE_CHECK_ARG(master_idx >= 0 &&
                         master_idx < static_cast<int>(master_vertices.size()),
                     "ConstraintTools: matched vertex index out of range");
        max_dist_sq = std::max(max_dist_sq,
                               squaredDistance(transformed, master_vertices[static_cast<std::size_t>(master_idx)]));
    }
    return std::sqrt(max_dist_sq);
}

spaces::TraceSpace::OrientedScalarTraceMap makeTraceOrientationMap(
    ElementType trace_element_type,
    int trace_polynomial_order,
    std::span<const int> matched_vertices)
{
    if (trace_element_type == ElementType::Line2) {
        FE_CHECK_ARG(matched_vertices.size() == 2u,
                     "ConstraintTools: line trace requires two matched vertices");
        FE_CHECK_ARG((matched_vertices[0] == 0 && matched_vertices[1] == 1) ||
                         (matched_vertices[0] == 1 && matched_vertices[1] == 0),
                     "ConstraintTools: invalid edge-vertex permutation for H(div) trace");
        const auto sign = (matched_vertices[0] == 0) ? +1 : -1;
        return spaces::TraceSpace::orientedHDivNormalTraceMap(trace_element_type,
                                                              trace_polynomial_order,
                                                              sign);
    }

    if (trace_element_type == ElementType::Triangle3) {
        FE_CHECK_ARG(matched_vertices.size() == 3u,
                     "ConstraintTools: triangular trace requires three matched vertices");
        const std::array<int, 3> local = {0, 1, 2};
        const std::array<int, 3> global = {
            matched_vertices[0], matched_vertices[1], matched_vertices[2]};
        const auto orientation = spaces::OrientationManager::triangle_face_orientation(local, global);
        return spaces::TraceSpace::orientedHDivNormalTraceMap(trace_element_type,
                                                              trace_polynomial_order,
                                                              orientation);
    }

    FE_CHECK_ARG(trace_element_type == ElementType::Quad4,
                 "ConstraintTools: unsupported H(div) trace element type");
    FE_CHECK_ARG(matched_vertices.size() == 4u,
                 "ConstraintTools: quadrilateral trace requires four matched vertices");
    const std::array<int, 4> local = {0, 1, 2, 3};
    const std::array<int, 4> global = {
        matched_vertices[0], matched_vertices[1], matched_vertices[2], matched_vertices[3]};
    const auto orientation = spaces::OrientationManager::quad_face_orientation(local, global);
    return spaces::TraceSpace::orientedHDivNormalTraceMap(trace_element_type,
                                                          trace_polynomial_order,
                                                          orientation);
}

double normalRelationWeight(const Point3& slave_normal,
                            const Point3& master_normal,
                            const TracePeriodicConstraintOptions& options)
{
    const Point3 slave_transformed =
        normalizedNormal(applyTransform(options.normal_transform, slave_normal),
                         "ConstraintTools: transformed slave normal is degenerate");
    const Point3 master_unit =
        normalizedNormal(master_normal, "ConstraintTools: master normal is degenerate");

    const double dot =
        slave_transformed[0] * master_unit[0] +
        slave_transformed[1] * master_unit[1] +
        slave_transformed[2] * master_unit[2];
    FE_CHECK_ARG(std::abs(dot) > 1e-10,
                 "ConstraintTools: periodic trace normals must be aligned or anti-aligned");

    double weight = (dot >= 0.0) ? 1.0 : -1.0;
    if (options.anti_periodic) {
        weight *= -1.0;
    }
    return weight;
}

std::vector<PeriodicPair> buildHDivTracePairs(const spaces::HDivSpace& space,
                                              std::span<const TraceBoundaryEntity> slave_entities,
                                              std::span<const TraceBoundaryEntity> master_entities,
                                              const std::function<Point3(Point3)>& point_transform,
                                              const TracePeriodicConstraintOptions& options)
{
    FE_CHECK_ARG(!slave_entities.empty(),
                 "ConstraintTools: H(div) trace periodic helper requires at least one slave entity");
    FE_CHECK_ARG(!master_entities.empty(),
                 "ConstraintTools: H(div) trace periodic helper requires at least one master entity");

    const int trace_order = space.polynomial_order();
    std::vector<PeriodicPair> pairs;
    std::vector<bool> used_master(master_entities.size(), false);

    for (const auto& slave : slave_entities) {
        FE_CHECK_ARG(!slave.vertices.empty(),
                     "ConstraintTools: slave trace entity must provide vertices");

        const ElementType trace_element_type = traceElementTypeFromVertexCount(slave.vertices.size());
        const auto expected_dofs =
            spaces::TraceSpace::hdivNormalTraceDofCount(trace_element_type, trace_order);
        FE_CHECK_ARG(slave.dofs.size() == expected_dofs,
                     "ConstraintTools: slave trace entity DOF count does not match H(div) trace size");

        std::optional<std::size_t> best_master;
        std::vector<int> best_matched_vertices;
        double best_distance = std::numeric_limits<double>::max();

        for (std::size_t master_idx = 0; master_idx < master_entities.size(); ++master_idx) {
            if (used_master[master_idx]) {
                continue;
            }
            const auto& master = master_entities[master_idx];
            if (master.vertices.size() != slave.vertices.size()) {
                continue;
            }
            if (master.dofs.size() != expected_dofs) {
                continue;
            }

            const auto matched_vertices =
                matchEntityVertices(slave.vertices, master.vertices, point_transform, options.tolerance);
            if (!matched_vertices.has_value()) {
                continue;
            }

            const double distance =
                maxMatchedVertexDistance(slave.vertices, master.vertices, point_transform, *matched_vertices);
            if (distance < best_distance) {
                best_master = master_idx;
                best_matched_vertices = *matched_vertices;
                best_distance = distance;
            }
        }

        FE_CHECK_ARG(best_master.has_value(),
                     "ConstraintTools: failed to match an H(div) slave trace entity to a master entity");

        used_master[*best_master] = true;
        const auto& master = master_entities[*best_master];
        const auto trace_map =
            makeTraceOrientationMap(trace_element_type, trace_order, best_matched_vertices);
        const double normal_weight =
            normalRelationWeight(slave.outward_normal, master.outward_normal, options);

        FE_CHECK_ARG(trace_map.source_indices.size() == master.dofs.size(),
                     "ConstraintTools: H(div) trace orientation map size mismatch");
        for (std::size_t i = 0; i < trace_map.source_indices.size(); ++i) {
            const int slave_local = trace_map.source_indices[i];
            FE_CHECK_ARG(slave_local >= 0 &&
                             slave_local < static_cast<int>(slave.dofs.size()),
                         "ConstraintTools: H(div) slave trace local index out of range");
            pairs.push_back(PeriodicPair{
                slave.dofs[static_cast<std::size_t>(slave_local)],
                master.dofs[i],
                normal_weight * static_cast<double>(trace_map.weights[i])});
        }
    }

    return pairs;
}

} // namespace

// ============================================================================
// Dirichlet constraint generation
// ============================================================================

void makeDirichletConstraints(
    std::span<const GlobalIndex> boundary_dofs,
    double value,
    AffineConstraints& constraints,
    [[maybe_unused]] const DirichletConstraintOptions& options)
{
    for (GlobalIndex dof : boundary_dofs) {
        constraints.addLine(dof);
        constraints.setInhomogeneity(dof, value);
    }
}

void makeDirichletConstraints(
    std::span<const GlobalIndex> boundary_dofs,
    std::span<const std::array<double, 3>> dof_coords,
    const std::function<double(double, double, double)>& value_func,
    AffineConstraints& constraints,
    [[maybe_unused]] const DirichletConstraintOptions& options)
{
    if (boundary_dofs.size() != dof_coords.size()) {
        CONSTRAINT_THROW("DOFs and coordinates must have same size");
    }

    for (std::size_t i = 0; i < boundary_dofs.size(); ++i) {
        GlobalIndex dof = boundary_dofs[i];
        const auto& coord = dof_coords[i];
        double value = value_func(coord[0], coord[1], coord[2]);

        constraints.addLine(dof);
        constraints.setInhomogeneity(dof, value);
    }
}

void makeDirichletConstraintsVector(
    std::span<const GlobalIndex> boundary_dofs,
    std::span<const std::array<double, 3>> dof_coords,
    const std::function<std::vector<double>(double, double, double)>& value_func,
    int num_components,
    AffineConstraints& constraints,
    const DirichletConstraintOptions& options)
{
    // Assumes DOFs are interleaved: [u0_x, u0_y, u0_z, u1_x, u1_y, u1_z, ...]
    std::size_t n_nodes = boundary_dofs.size() / static_cast<std::size_t>(num_components);

    if (n_nodes != dof_coords.size()) {
        CONSTRAINT_THROW("Number of nodes must match coordinate count");
    }

    for (std::size_t node = 0; node < n_nodes; ++node) {
        const auto& coord = dof_coords[node];
        std::vector<double> values = value_func(coord[0], coord[1], coord[2]);

        if (static_cast<int>(values.size()) != num_components) {
            CONSTRAINT_THROW("Value function must return num_components values");
        }

        for (int comp = 0; comp < num_components; ++comp) {
            if (options.component_mask.allSelected() || options.component_mask[static_cast<std::size_t>(comp)]) {
                GlobalIndex dof = boundary_dofs[node * static_cast<std::size_t>(num_components) + static_cast<std::size_t>(comp)];
                constraints.addLine(dof);
                constraints.setInhomogeneity(dof, values[static_cast<std::size_t>(comp)]);
            }
        }
    }
}

// ============================================================================
// Periodic constraint generation
// ============================================================================

void makePeriodicConstraints(
    std::span<const GlobalIndex> slave_dofs,
    std::span<const std::array<double, 3>> slave_coords,
    std::span<const GlobalIndex> master_dofs,
    std::span<const std::array<double, 3>> master_coords,
    const std::function<std::array<double, 3>(std::array<double, 3>)>& transform,
    AffineConstraints& constraints,
    const PeriodicConstraintOptions& options)
{
    // Find matching pairs
    auto matches = findMatchingPoints(slave_coords, master_coords, transform, options.tolerance);

    // Create periodic constraints for each match
    double sign = options.flip_sign ? -1.0 : 1.0;

    for (const auto& [slave_idx, master_idx] : matches) {
        GlobalIndex slave_dof = slave_dofs[slave_idx];
        GlobalIndex master_dof = master_dofs[master_idx];

        constraints.addLine(slave_dof);
        constraints.addEntry(slave_dof, master_dof, sign);
    }
}

void makePeriodicConstraintsTranslation(
    std::span<const GlobalIndex> slave_dofs,
    std::span<const std::array<double, 3>> slave_coords,
    std::span<const GlobalIndex> master_dofs,
    std::span<const std::array<double, 3>> master_coords,
    std::array<double, 3> translation,
    AffineConstraints& constraints,
    const PeriodicConstraintOptions& options)
{
    auto transform = [translation](std::array<double, 3> p) -> std::array<double, 3> {
        return {{p[0] + translation[0], p[1] + translation[1], p[2] + translation[2]}};
    };

    makePeriodicConstraints(slave_dofs, slave_coords, master_dofs, master_coords,
                            transform, constraints, options);
}

std::vector<PeriodicPair> makeHDivTracePeriodicPairs(
    const spaces::HDivSpace& space,
    std::span<const TraceBoundaryEntity> slave_entities,
    std::span<const TraceBoundaryEntity> master_entities,
    const std::function<std::array<double, 3>(std::array<double, 3>)>& point_transform,
    const TracePeriodicConstraintOptions& options)
{
    return buildHDivTracePairs(space, slave_entities, master_entities, point_transform, options);
}

std::vector<PeriodicPair> makeHDivTracePeriodicPairsTranslation(
    const spaces::HDivSpace& space,
    std::span<const TraceBoundaryEntity> slave_entities,
    std::span<const TraceBoundaryEntity> master_entities,
    std::array<double, 3> translation,
    const TracePeriodicConstraintOptions& options)
{
    auto transform = [translation](std::array<double, 3> p) -> std::array<double, 3> {
        return {{p[0] + translation[0], p[1] + translation[1], p[2] + translation[2]}};
    };
    return buildHDivTracePairs(space, slave_entities, master_entities, transform, options);
}

PeriodicBC makeHDivTracePeriodicBC(
    const spaces::HDivSpace& space,
    std::span<const TraceBoundaryEntity> slave_entities,
    std::span<const TraceBoundaryEntity> master_entities,
    const std::function<std::array<double, 3>(std::array<double, 3>)>& point_transform,
    const TracePeriodicConstraintOptions& options)
{
    return PeriodicBC(makeHDivTracePeriodicPairs(space,
                                                 slave_entities,
                                                 master_entities,
                                                 point_transform,
                                                 options));
}

PeriodicBC makeHDivTracePeriodicBCTranslation(
    const spaces::HDivSpace& space,
    std::span<const TraceBoundaryEntity> slave_entities,
    std::span<const TraceBoundaryEntity> master_entities,
    std::array<double, 3> translation,
    const TracePeriodicConstraintOptions& options)
{
    return PeriodicBC(makeHDivTracePeriodicPairsTranslation(space,
                                                            slave_entities,
                                                            master_entities,
                                                            translation,
                                                            options));
}

MultiPointConstraint makeHDivTracePeriodicMPC(
    const spaces::HDivSpace& space,
    std::span<const TraceBoundaryEntity> slave_entities,
    std::span<const TraceBoundaryEntity> master_entities,
    const std::function<std::array<double, 3>(std::array<double, 3>)>& point_transform,
    const TracePeriodicConstraintOptions& options)
{
    MultiPointConstraint mpc;
    for (const auto& pair : makeHDivTracePeriodicPairs(space,
                                                       slave_entities,
                                                       master_entities,
                                                       point_transform,
                                                       options)) {
        mpc.addConstraint(pair.slave_dof, pair.master_dof, pair.weight);
    }
    return mpc;
}

MultiPointConstraint makeHDivTracePeriodicMPCTranslation(
    const spaces::HDivSpace& space,
    std::span<const TraceBoundaryEntity> slave_entities,
    std::span<const TraceBoundaryEntity> master_entities,
    std::array<double, 3> translation,
    const TracePeriodicConstraintOptions& options)
{
    MultiPointConstraint mpc;
    for (const auto& pair : makeHDivTracePeriodicPairsTranslation(space,
                                                                  slave_entities,
                                                                  master_entities,
                                                                  translation,
                                                                  options)) {
        mpc.addConstraint(pair.slave_dof, pair.master_dof, pair.weight);
    }
    return mpc;
}

// ============================================================================
// Hanging node constraint generation
// ============================================================================

void makeHangingNodeConstraints1D(
    std::span<const GlobalIndex> hanging_dofs,
    std::span<const GlobalIndex> parent_dofs_1,
    std::span<const GlobalIndex> parent_dofs_2,
    AffineConstraints& constraints,
    [[maybe_unused]] const HangingNodeOptions& options)
{
    if (hanging_dofs.size() != parent_dofs_1.size() ||
        hanging_dofs.size() != parent_dofs_2.size()) {
        CONSTRAINT_THROW("Hanging and parent DOF arrays must have same size");
    }

    // For P1 elements, midpoint interpolation: u_h = 0.5*u_1 + 0.5*u_2
    for (std::size_t i = 0; i < hanging_dofs.size(); ++i) {
        constraints.addLine(hanging_dofs[i]);
        constraints.addEntry(hanging_dofs[i], parent_dofs_1[i], 0.5);
        constraints.addEntry(hanging_dofs[i], parent_dofs_2[i], 0.5);
    }
}

void makeHangingNodeConstraint2D(
    GlobalIndex hanging_dof,
    std::span<const GlobalIndex> parent_dofs,
    int order,
    AffineConstraints& constraints)
{
    if (order == 1) {
        // Linear interpolation: midpoint
        if (parent_dofs.size() != 2) {
            CONSTRAINT_THROW("P1 edge hanging node requires 2 parent DOFs");
        }
        constraints.addLine(hanging_dof);
        constraints.addEntry(hanging_dof, parent_dofs[0], 0.5);
        constraints.addEntry(hanging_dof, parent_dofs[1], 0.5);
    } else if (order == 2) {
        // Quadratic interpolation
        // For midpoint on quadratic edge with 3 DOFs:
        // u_h = 0.5*u_0 + 0.5*u_2 (vertices), midpoint is interpolated
        if (parent_dofs.size() >= 2) {
            constraints.addLine(hanging_dof);
            constraints.addEntry(hanging_dof, parent_dofs[0], 0.5);
            constraints.addEntry(hanging_dof, parent_dofs[parent_dofs.size()-1], 0.5);
        }
    } else {
        // General case: compute Lagrange interpolation weights
        // Simplified: assume midpoint interpolation
        constraints.addLine(hanging_dof);
        double weight = 1.0 / static_cast<double>(parent_dofs.size());
        for (GlobalIndex parent_dof : parent_dofs) {
            constraints.addEntry(hanging_dof, parent_dof, weight);
        }
    }
}

void makeHangingNodeConstraints3D(
    std::span<const GlobalIndex> hanging_dofs,
    std::span<const std::vector<GlobalIndex>> parent_dofs,
    std::span<const std::vector<double>> weights,
    AffineConstraints& constraints)
{
    if (hanging_dofs.size() != parent_dofs.size() ||
        hanging_dofs.size() != weights.size()) {
        CONSTRAINT_THROW("Hanging DOFs, parent DOFs, and weights must have same size");
    }

    for (std::size_t i = 0; i < hanging_dofs.size(); ++i) {
        const auto& parents = parent_dofs[i];
        const auto& w = weights[i];

        if (parents.size() != w.size()) {
            CONSTRAINT_THROW("Parent DOFs and weights must have same size for each hanging DOF");
        }

        constraints.addLine(hanging_dofs[i]);
        for (std::size_t j = 0; j < parents.size(); ++j) {
            if (std::abs(w[j]) > 1e-15) {
                constraints.addEntry(hanging_dofs[i], parents[j], w[j]);
            }
        }
    }
}

// ============================================================================
// Utility functions
// ============================================================================

std::vector<std::pair<std::size_t, std::size_t>> findMatchingPoints(
    std::span<const std::array<double, 3>> coords_a,
    std::span<const std::array<double, 3>> coords_b,
    const std::function<std::array<double, 3>(std::array<double, 3>)>& transform,
    double tolerance)
{
    std::vector<std::pair<std::size_t, std::size_t>> matches;
    matches.reserve(std::min(coords_a.size(), coords_b.size()));

    double tol_sq = tolerance * tolerance;

    for (std::size_t i = 0; i < coords_a.size(); ++i) {
        std::array<double, 3> point_a = coords_a[i];
        if (transform) {
            point_a = transform(point_a);
        }

        // Find closest point in B
        double best_dist_sq = std::numeric_limits<double>::max();
        std::size_t best_j = coords_b.size();

        for (std::size_t j = 0; j < coords_b.size(); ++j) {
            double dx = point_a[0] - coords_b[j][0];
            double dy = point_a[1] - coords_b[j][1];
            double dz = point_a[2] - coords_b[j][2];
            double dist_sq = dx*dx + dy*dy + dz*dz;

            if (dist_sq < best_dist_sq) {
                best_dist_sq = dist_sq;
                best_j = j;
            }
        }

        if (best_j < coords_b.size() && best_dist_sq < tol_sq) {
            matches.emplace_back(i, best_j);
        }
    }

    return matches;
}

std::vector<double> computeInterpolationWeights(
    std::array<double, 3> reference_point,
    ElementType cell_type,
    int order)
{
    std::vector<double> weights;

    // Simplified implementation for common cases
    double xi = reference_point[0];
    double eta = reference_point[1];
    double zeta = reference_point[2];

    switch (cell_type) {
        case ElementType::Line2:
            // Linear line: 2 nodes at xi = -1, 1
            weights = {0.5 * (1.0 - xi), 0.5 * (1.0 + xi)};
            break;

        case ElementType::Triangle3:
            // Linear triangle: barycentric coordinates
            // Assumes reference is [0,1] x [0,1] with node at (0,0), (1,0), (0,1)
            weights = {1.0 - xi - eta, xi, eta};
            break;

        case ElementType::Quad4:
            // Bilinear quad: 4 nodes at corners
            weights = {
                0.25 * (1.0 - xi) * (1.0 - eta),
                0.25 * (1.0 + xi) * (1.0 - eta),
                0.25 * (1.0 + xi) * (1.0 + eta),
                0.25 * (1.0 - xi) * (1.0 + eta)
            };
            break;

        case ElementType::Tetra4:
            // Linear tetrahedron: barycentric coordinates
            weights = {1.0 - xi - eta - zeta, xi, eta, zeta};
            break;

        case ElementType::Hex8:
            // Trilinear hex: 8 nodes at corners
            weights = {
                0.125 * (1.0 - xi) * (1.0 - eta) * (1.0 - zeta),
                0.125 * (1.0 + xi) * (1.0 - eta) * (1.0 - zeta),
                0.125 * (1.0 + xi) * (1.0 + eta) * (1.0 - zeta),
                0.125 * (1.0 - xi) * (1.0 + eta) * (1.0 - zeta),
                0.125 * (1.0 - xi) * (1.0 - eta) * (1.0 + zeta),
                0.125 * (1.0 + xi) * (1.0 - eta) * (1.0 + zeta),
                0.125 * (1.0 + xi) * (1.0 + eta) * (1.0 + zeta),
                0.125 * (1.0 - xi) * (1.0 + eta) * (1.0 + zeta)
            };
            break;

        default:
            // General case: uniform weights (placeholder)
            if (order > 0) {
                int n_nodes = (order + 1) * (order + 1);  // Rough estimate
                weights.resize(static_cast<std::size_t>(n_nodes), 1.0 / static_cast<double>(n_nodes));
            }
            break;
    }

    return weights;
}

std::vector<GlobalIndex> extractComponentDofs(
    std::span<const GlobalIndex> all_dofs,
    int component,
    int num_components)
{
    std::vector<GlobalIndex> result;
    result.reserve(all_dofs.size() / static_cast<std::size_t>(num_components));

    for (std::size_t i = static_cast<std::size_t>(component);
         i < all_dofs.size();
         i += static_cast<std::size_t>(num_components)) {
        result.push_back(all_dofs[i]);
    }

    return result;
}

void mergeConstraints(
    AffineConstraints& target,
    const AffineConstraints& source,
    bool overwrite)
{
    source.forEach([&](const AffineConstraints::ConstraintView& view) {
        if (target.isConstrained(view.slave_dof)) {
            if (!overwrite) {
                CONSTRAINT_THROW_DOF("Constraint conflict during merge", view.slave_dof);
            }
            // Remove existing (will be overwritten)
            // Note: AffineConstraints doesn't support removal, so this requires clear/rebuild
            // For now, just throw
            CONSTRAINT_THROW_DOF("Cannot overwrite existing constraint (not implemented)",
                                 view.slave_dof);
        }

        target.addLine(view.slave_dof);
        for (const auto& entry : view.entries) {
            target.addEntry(view.slave_dof, entry.master_dof, entry.weight);
        }
        target.setInhomogeneity(view.slave_dof, view.inhomogeneity);
    });
}

bool hasConstrainedDofs(
    std::span<const GlobalIndex> dofs,
    const AffineConstraints& constraints)
{
    return constraints.hasConstrainedDofs(dofs);
}

std::vector<GlobalIndex> filterUnconstrainedDofs(
    std::span<const GlobalIndex> dofs,
    const AffineConstraints& constraints)
{
    std::vector<GlobalIndex> result;
    result.reserve(dofs.size());

    for (GlobalIndex dof : dofs) {
        if (!constraints.isConstrained(dof)) {
            result.push_back(dof);
        }
    }

    return result;
}

std::unordered_set<GlobalIndex> getUnconstrainedDofSet(
    GlobalIndex n_dofs,
    const AffineConstraints& constraints)
{
    std::unordered_set<GlobalIndex> result;
    result.reserve(static_cast<std::size_t>(n_dofs));

    for (GlobalIndex dof = 0; dof < n_dofs; ++dof) {
        if (!constraints.isConstrained(dof)) {
            result.insert(dof);
        }
    }

    return result;
}

} // namespace constraints
} // namespace FE
} // namespace svmp
