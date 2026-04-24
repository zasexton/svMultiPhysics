/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "MovingMesh/GeometryRegularizationBackend.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

#include "Core/FEException.h"

#include "Mesh/Core/MeshBase.h"
#include "Mesh/Fields/MeshFields.h"
#include "Mesh/Mesh.h"
#include "Mesh/Motion/MotionQuality.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <sstream>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace svmp::FE::moving_mesh {
namespace {

using VertexId = std::size_t;

struct VertexConstraint {
    std::array<bool, 3> active{{false, false, false}};
    std::array<Real, 3> value{{0.0, 0.0, 0.0}};
};

struct ConstraintBuildResult {
    std::vector<VertexConstraint> constraints;
    std::size_t constrained_components{0};
    std::string message;
};

struct Graph {
    std::vector<std::vector<std::pair<VertexId, Real>>> adjacency;
};

struct LinearStats {
    bool success{true};
    int iterations{0};
    Real initial_residual{0.0};
    Real final_residual{0.0};
    std::string message;
};

[[nodiscard]] const std::vector<real_t>& coordinates(const MeshBase& mesh,
                                                     Configuration config)
{
    if ((config == Configuration::Current || config == Configuration::Deformed) &&
        mesh.has_current_coords()) {
        return mesh.X_cur();
    }
    return mesh.X_ref();
}

[[nodiscard]] std::array<Real, 3> vertex_coordinate(const MeshBase& mesh,
                                                    const std::vector<real_t>& x,
                                                    VertexId vertex)
{
    const int dim = mesh.dim();
    std::array<Real, 3> xyz{{0.0, 0.0, 0.0}};
    const auto base = vertex * static_cast<VertexId>(dim);
    for (int d = 0; d < dim; ++d) {
        xyz[static_cast<std::size_t>(d)] =
            static_cast<Real>(x[base + static_cast<VertexId>(d)]);
    }
    return xyz;
}

[[nodiscard]] Real squared_distance(const std::array<Real, 3>& a,
                                    const std::array<Real, 3>& b,
                                    int dim)
{
    Real value = 0.0;
    for (int d = 0; d < dim; ++d) {
        const Real diff = a[static_cast<std::size_t>(d)] -
                          b[static_cast<std::size_t>(d)];
        value += diff * diff;
    }
    return value;
}

[[nodiscard]] std::vector<index_t> boundary_faces_for_bc(const MeshBase& mesh,
                                                         const motion::MotionDirichletBC& bc)
{
    if (bc.boundary_label == INVALID_LABEL) {
        return mesh.boundary_faces();
    }
    return mesh.faces_with_label(bc.boundary_label);
}

[[nodiscard]] bool all_mask_entries_inactive(const std::array<bool, 3>& mask,
                                             int dim)
{
    for (int d = 0; d < dim; ++d) {
        if (mask[static_cast<std::size_t>(d)]) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] ConstraintBuildResult
build_constraints(const MeshBase& mesh,
                  const std::vector<real_t>& x,
                  const motion::MotionSolveRequest& request,
                  const GeometryRegularizationOptions& options)
{
    ConstraintBuildResult result;
    const auto n_vertices = mesh.n_vertices();
    const int dim = mesh.dim();
    result.constraints.resize(n_vertices);

    if (options.constraint_mode != GeometryConstraintMode::StrongDirichlet) {
        result.message = "geometry regularization supports strong mesh-displacement constraints only";
        return result;
    }

    if (!request.dirichlet_bcs || request.dirichlet_bcs->empty()) {
        result.message = "underconstrained mesh motion: no boundary displacement constraints were provided";
        return result;
    }

    for (const auto& bc : *request.dirichlet_bcs) {
        if (all_mask_entries_inactive(bc.component_mask, dim)) {
            result.message = "incomplete mesh-motion constraint: component mask has no active components";
            return result;
        }

        const auto faces = boundary_faces_for_bc(mesh, bc);
        if (faces.empty()) {
            std::ostringstream os;
            os << "incomplete mesh-motion constraint: boundary label "
               << bc.boundary_label << " matched no boundary faces";
            result.message = os.str();
            return result;
        }

        for (const index_t face : faces) {
            auto [verts, n_face_vertices] = mesh.face_vertices_span(face);
            for (std::size_t fv = 0; fv < n_face_vertices; ++fv) {
                const auto vertex = static_cast<VertexId>(verts[fv]);
                if (vertex >= n_vertices) {
                    result.message = "mesh-motion constraint references an invalid boundary vertex";
                    return result;
                }

                const auto xyz = vertex_coordinate(mesh, x, vertex);
                const auto value = bc.value
                    ? bc.value({{static_cast<real_t>(xyz[0]),
                                 static_cast<real_t>(xyz[1]),
                                 static_cast<real_t>(xyz[2])}},
                               request.dt,
                               request.step_scale)
                    : std::array<real_t, 3>{{0.0, 0.0, 0.0}};

                auto& constraint = result.constraints[vertex];
                for (int d = 0; d < dim; ++d) {
                    const auto di = static_cast<std::size_t>(d);
                    if (!bc.component_mask[di]) {
                        continue;
                    }
                    const Real next = static_cast<Real>(value[di]);
                    if (!std::isfinite(next)) {
                        result.message = "mesh-motion constraint returned a non-finite displacement";
                        return result;
                    }
                    if (constraint.active[di]) {
                        const Real diff = std::abs(constraint.value[di] - next);
                        if (diff > options.constraint_tolerance) {
                            std::ostringstream os;
                            os << "conflicting mesh-motion constraints at vertex "
                               << vertex << ", component " << d;
                            result.message = os.str();
                            return result;
                        }
                    } else {
                        constraint.active[di] = true;
                        constraint.value[di] = next;
                        ++result.constrained_components;
                    }
                }
            }
        }
    }

    return result;
}

[[nodiscard]] std::vector<Real>
boundary_distance_weights(const MeshBase& mesh,
                          const std::vector<real_t>& x,
                          const std::vector<VertexConstraint>& constraints,
                          Real floor)
{
    const auto n_vertices = mesh.n_vertices();
    const int dim = mesh.dim();
    std::vector<VertexId> constrained_vertices;
    constrained_vertices.reserve(n_vertices);
    for (VertexId v = 0; v < n_vertices; ++v) {
        const auto& c = constraints[v];
        if (c.active[0] || c.active[1] || c.active[2]) {
            constrained_vertices.push_back(v);
        }
    }

    std::vector<Real> weights(n_vertices, Real(1.0));
    if (constrained_vertices.empty()) {
        return weights;
    }

    const Real safe_floor = std::max(floor, Real(1.0e-14));
    for (VertexId v = 0; v < n_vertices; ++v) {
        const auto xv = vertex_coordinate(mesh, x, v);
        Real best = std::numeric_limits<Real>::infinity();
        for (const VertexId b : constrained_vertices) {
            const auto xb = vertex_coordinate(mesh, x, b);
            best = std::min(best, std::sqrt(squared_distance(xv, xb, dim)));
        }
        weights[v] = Real(1.0) / std::max(best, safe_floor);
    }
    return weights;
}

[[nodiscard]] std::vector<Real>
vertex_field_weights(const MeshBase& mesh,
                     const GeometryRegularizationOptions& options,
                     std::string& message)
{
    if (options.vertex_weight_field.empty()) {
        message = "vertex-field weighting requires a vertex_weight_field name";
        return {};
    }
    if (!mesh.has_field(EntityKind::Vertex, options.vertex_weight_field)) {
        message = "vertex-field weighting could not find field '" +
                  options.vertex_weight_field + "'";
        return {};
    }

    const auto handle = mesh.field_handle(EntityKind::Vertex, options.vertex_weight_field);
    if (mesh.field_type(handle) != FieldScalarType::Float64 ||
        mesh.field_components(handle) != 1u ||
        mesh.field_entity_count(handle) != mesh.n_vertices()) {
        message = "vertex-field weighting requires a scalar Float64 vertex field";
        return {};
    }

    const auto* data = mesh.field_data_as<real_t>(handle);
    if (!data) {
        message = "vertex-field weighting field has no data";
        return {};
    }

    std::vector<Real> weights(mesh.n_vertices(), Real(1.0));
    for (VertexId v = 0; v < mesh.n_vertices(); ++v) {
        const Real w = static_cast<Real>(data[v]);
        if (!std::isfinite(w) || w < options.minimum_weight) {
            message = "vertex-field weighting requires finite positive weights";
            return {};
        }
        weights[v] = w;
    }
    return weights;
}

[[nodiscard]] Graph
build_graph(const MeshBase& mesh,
            const std::vector<real_t>& x,
            const std::vector<VertexConstraint>& constraints,
            const GeometryRegularizationOptions& options,
            std::string& message)
{
    const auto n_vertices = mesh.n_vertices();
    const int dim = mesh.dim();
    std::vector<std::unordered_map<VertexId, Real>> accum(n_vertices);

    std::vector<Real> vertex_weights;
    if (options.weight_mode == GeometryRegularizationWeightMode::BoundaryDistance) {
        vertex_weights = boundary_distance_weights(mesh, x, constraints, options.boundary_distance_floor);
    } else if (options.weight_mode == GeometryRegularizationWeightMode::VertexField) {
        vertex_weights = vertex_field_weights(mesh, options, message);
        if (!message.empty()) {
            return {};
        }
    }

    const Real minimum_weight = std::max(options.minimum_weight, Real(1.0e-16));
    const Real stiffness = std::max(options.artificial_stiffness_scale, minimum_weight);
    const Real blend = std::max(options.artificial_pseudo_elastic_blend, Real(0.0));

    for (index_t cell = 0; cell < static_cast<index_t>(mesh.n_cells()); ++cell) {
        auto [verts, n_cell_vertices] = mesh.cell_vertices_span(cell);
        for (std::size_t a = 0; a < n_cell_vertices; ++a) {
            const auto va = static_cast<VertexId>(verts[a]);
            if (va >= n_vertices) {
                message = "mesh connectivity references an invalid vertex";
                return {};
            }
            const auto xa = vertex_coordinate(mesh, x, va);
            for (std::size_t b = a + 1; b < n_cell_vertices; ++b) {
                const auto vb = static_cast<VertexId>(verts[b]);
                if (vb >= n_vertices || va == vb) {
                    continue;
                }

                const auto xb = vertex_coordinate(mesh, x, vb);
                const Real length2 = std::max(squared_distance(xa, xb, dim), minimum_weight);
                Real weight = stiffness;

                switch (options.weight_mode) {
                    case GeometryRegularizationWeightMode::Uniform:
                        break;
                    case GeometryRegularizationWeightMode::ElementSize:
                        weight /= length2;
                        break;
                    case GeometryRegularizationWeightMode::BoundaryDistance:
                    case GeometryRegularizationWeightMode::VertexField:
                        weight *= Real(0.5) * (vertex_weights[va] + vertex_weights[vb]);
                        break;
                }

                if (options.model == GeometryRegularizationModel::ArtificialPseudoElastic) {
                    weight *= Real(1.0) + blend / (Real(1.0) + length2);
                }

                weight = std::max(weight, minimum_weight);
                accum[va][vb] += weight;
                accum[vb][va] += weight;
            }
        }
    }

    Graph graph;
    graph.adjacency.resize(n_vertices);
    for (VertexId v = 0; v < n_vertices; ++v) {
        graph.adjacency[v].reserve(accum[v].size());
        for (const auto& entry : accum[v]) {
            graph.adjacency[v].push_back(entry);
        }
        std::sort(graph.adjacency[v].begin(),
                  graph.adjacency[v].end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
    }

    return graph;
}

[[nodiscard]] bool
has_constraint_in_component(const Graph& graph,
                            const std::vector<VertexConstraint>& constraints,
                            VertexId seed,
                            int component)
{
    std::vector<char> visited(graph.adjacency.size(), 0);
    std::queue<VertexId> q;
    q.push(seed);
    visited[seed] = 1;
    while (!q.empty()) {
        const VertexId v = q.front();
        q.pop();
        if (constraints[v].active[static_cast<std::size_t>(component)]) {
            return true;
        }
        for (const auto& [neighbor, weight] : graph.adjacency[v]) {
            (void)weight;
            if (!visited[neighbor]) {
                visited[neighbor] = 1;
                q.push(neighbor);
            }
        }
    }
    return false;
}

[[nodiscard]] LinearStats
solve_component(const Graph& graph,
                const std::vector<VertexConstraint>& constraints,
                int component,
                const GeometryRegularizationOptions& options,
                std::vector<Real>& solution)
{
    LinearStats stats;
    const auto n_vertices = graph.adjacency.size();
    const auto ci = static_cast<std::size_t>(component);
    solution.assign(n_vertices, Real(0.0));

    std::vector<int> unknown_index(n_vertices, -1);
    std::vector<VertexId> unknown_vertices;
    unknown_vertices.reserve(n_vertices);

    for (VertexId v = 0; v < n_vertices; ++v) {
        if (constraints[v].active[ci]) {
            solution[v] = constraints[v].value[ci];
        } else {
            unknown_index[v] = static_cast<int>(unknown_vertices.size());
            unknown_vertices.push_back(v);
        }
    }

    const auto n_unknowns = unknown_vertices.size();
    if (n_unknowns == 0u) {
        return stats;
    }

    for (const VertexId v : unknown_vertices) {
        if (!has_constraint_in_component(graph, constraints, v, component)) {
            stats.success = false;
            std::ostringstream os;
            os << "underconstrained mesh motion: connected component containing vertex "
               << v << " has no constraint for component " << component;
            stats.message = os.str();
            return stats;
        }
    }

    std::vector<Real> b(n_unknowns, Real(0.0));
    std::vector<Real> diagonal(n_unknowns, Real(0.0));
    for (std::size_t row = 0; row < n_unknowns; ++row) {
        const VertexId v = unknown_vertices[row];
        for (const auto& [neighbor, weight] : graph.adjacency[v]) {
            diagonal[row] += weight;
            if (constraints[neighbor].active[ci]) {
                b[row] += weight * constraints[neighbor].value[ci];
            }
        }
        if (!(diagonal[row] > Real(0.0))) {
            stats.success = false;
            stats.message = "underconstrained mesh motion: free vertex has no graph neighbors";
            return stats;
        }
    }

    auto apply = [&](const std::vector<Real>& x, std::vector<Real>& y) {
        std::fill(y.begin(), y.end(), Real(0.0));
        for (std::size_t row = 0; row < n_unknowns; ++row) {
            const VertexId v = unknown_vertices[row];
            Real value = diagonal[row] * x[row];
            for (const auto& [neighbor, weight] : graph.adjacency[v]) {
                const int col = unknown_index[neighbor];
                if (col >= 0) {
                    value -= weight * x[static_cast<std::size_t>(col)];
                }
            }
            y[row] = value;
        }
    };

    auto dot = [](const std::vector<Real>& a, const std::vector<Real>& b_vec) {
        Real value = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            value += a[i] * b_vec[i];
        }
        return value;
    };

    std::vector<Real> x_unknown(n_unknowns, Real(0.0));
    std::vector<Real> ax(n_unknowns, Real(0.0));
    std::vector<Real> r(n_unknowns, Real(0.0));
    std::vector<Real> p(n_unknowns, Real(0.0));
    std::vector<Real> ap(n_unknowns, Real(0.0));

    apply(x_unknown, ax);
    for (std::size_t i = 0; i < n_unknowns; ++i) {
        r[i] = b[i] - ax[i];
        p[i] = r[i];
    }

    Real rs_old = dot(r, r);
    stats.initial_residual = std::sqrt(std::max(rs_old, Real(0.0)));
    Real tolerance = std::max(options.absolute_tolerance,
                              options.relative_tolerance * stats.initial_residual);
    if (stats.initial_residual <= tolerance) {
        stats.final_residual = stats.initial_residual;
        for (std::size_t row = 0; row < n_unknowns; ++row) {
            solution[unknown_vertices[row]] = x_unknown[row];
        }
        return stats;
    }

    for (int iter = 0; iter < options.max_linear_iterations; ++iter) {
        apply(p, ap);
        const Real denom = dot(p, ap);
        if (!(std::abs(denom) > std::numeric_limits<Real>::epsilon())) {
            stats.success = false;
            stats.message = "linear geometry-regularization solve encountered a singular search direction";
            return stats;
        }

        const Real alpha = rs_old / denom;
        for (std::size_t i = 0; i < n_unknowns; ++i) {
            x_unknown[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        const Real rs_new = dot(r, r);
        stats.iterations = iter + 1;
        stats.final_residual = std::sqrt(std::max(rs_new, Real(0.0)));
        if (stats.final_residual <= tolerance) {
            for (std::size_t row = 0; row < n_unknowns; ++row) {
                solution[unknown_vertices[row]] = x_unknown[row];
            }
            return stats;
        }

        const Real beta = rs_new / rs_old;
        for (std::size_t i = 0; i < n_unknowns; ++i) {
            p[i] = r[i] + beta * p[i];
        }
        rs_old = rs_new;
    }

    stats.success = false;
    stats.message = "linear geometry-regularization solve did not converge";
    return stats;
}

[[nodiscard]] Real max_constraint_violation(const std::vector<VertexConstraint>& constraints,
                                            std::span<const Real> values,
                                            int dim)
{
    Real max_violation = 0.0;
    for (VertexId v = 0; v < constraints.size(); ++v) {
        const auto& constraint = constraints[v];
        const auto base = v * static_cast<VertexId>(dim);
        for (int d = 0; d < dim; ++d) {
            const auto di = static_cast<std::size_t>(d);
            if (constraint.active[di]) {
                max_violation = std::max(
                    max_violation,
                    std::abs(values[base + di] - constraint.value[di]));
            }
        }
    }
    return max_violation;
}

[[nodiscard]] motion::MotionSolveResult failure_result(GeometryRegularizationDiagnostics& diag,
                                                       std::string message)
{
    diag.success = false;
    diag.message = std::move(message);
    motion::MotionSolveResult result;
    result.success = false;
    result.wrote_velocity = false;
    result.message = diag.message;
    return result;
}

} // namespace

const char* to_string(GeometryRegularizationModel model) noexcept
{
    switch (model) {
        case GeometryRegularizationModel::Harmonic:
            return "harmonic_geometry_regularization";
        case GeometryRegularizationModel::ArtificialPseudoElastic:
            return "artificial_pseudo_elastic_geometry_regularization";
    }
    return "unknown_geometry_regularization_model";
}

const char* to_string(GeometryRegularizationWeightMode mode) noexcept
{
    switch (mode) {
        case GeometryRegularizationWeightMode::Uniform:
            return "uniform_artificial_weight";
        case GeometryRegularizationWeightMode::ElementSize:
            return "element_size_artificial_weight";
        case GeometryRegularizationWeightMode::BoundaryDistance:
            return "boundary_distance_artificial_weight";
        case GeometryRegularizationWeightMode::VertexField:
            return "vertex_field_artificial_weight";
    }
    return "unknown_artificial_weight";
}

const char* to_string(GeometryConstraintMode mode) noexcept
{
    switch (mode) {
        case GeometryConstraintMode::StrongDirichlet:
            return "strong_mesh_displacement_constraint";
    }
    return "unknown_mesh_motion_constraint";
}

std::vector<std::string> geometry_regularization_option_names()
{
    return {
        "model",
        "weight_mode",
        "constraint_mode",
        "max_linear_iterations",
        "relative_tolerance",
        "absolute_tolerance",
        "constraint_tolerance",
        "artificial_stiffness_scale",
        "artificial_pseudo_elastic_blend",
        "boundary_distance_floor",
        "minimum_weight",
        "vertex_weight_field",
        "write_velocity",
    };
}

std::vector<std::string> geometry_regularization_diagnostic_field_names()
{
    return {
        "backend_name",
        "model",
        "weight_mode",
        "constraint_mode",
        "linear_iterations",
        "nonlinear_iterations",
        "initial_residual_norm",
        "final_residual_norm",
        "max_constraint_violation",
        "accepted_step_scale",
        "minimum_quality_jacobian",
        "minimum_quality_angle_degrees",
        "maximum_quality_skewness",
    };
}

void validate_geometry_regularization_options(const GeometryRegularizationOptions& options)
{
    if (options.max_linear_iterations <= 0) {
        throw std::invalid_argument("geometry regularization requires max_linear_iterations > 0");
    }
    if (!(options.relative_tolerance >= Real(0.0)) ||
        !(options.absolute_tolerance >= Real(0.0)) ||
        !(options.constraint_tolerance >= Real(0.0))) {
        throw std::invalid_argument("geometry regularization tolerances must be nonnegative");
    }
    if (!(options.relative_tolerance > Real(0.0)) &&
        !(options.absolute_tolerance > Real(0.0))) {
        throw std::invalid_argument("geometry regularization requires a positive solve tolerance");
    }
    if (!(options.artificial_stiffness_scale > Real(0.0))) {
        throw std::invalid_argument("geometry regularization requires positive artificial_stiffness_scale");
    }
    if (!(options.artificial_pseudo_elastic_blend >= Real(0.0))) {
        throw std::invalid_argument("geometry regularization requires nonnegative artificial_pseudo_elastic_blend");
    }
    if (!(options.boundary_distance_floor > Real(0.0))) {
        throw std::invalid_argument("geometry regularization requires positive boundary_distance_floor");
    }
    if (!(options.minimum_weight > Real(0.0))) {
        throw std::invalid_argument("geometry regularization requires positive minimum_weight");
    }
    if (options.weight_mode == GeometryRegularizationWeightMode::VertexField &&
        options.vertex_weight_field.empty()) {
        throw std::invalid_argument("vertex-field geometry regularization requires vertex_weight_field");
    }
}

GeometryRegularizationMotionBackend::GeometryRegularizationMotionBackend(
    GeometryRegularizationOptions options)
    : options_(std::move(options))
{
    validate_geometry_regularization_options(options_);
    diagnostics_.backend_name = name();
    diagnostics_.model = to_string(options_.model);
    diagnostics_.weight_mode = to_string(options_.weight_mode);
    diagnostics_.constraint_mode = to_string(options_.constraint_mode);
}

const char* GeometryRegularizationMotionBackend::name() const noexcept
{
    return "FE::MovingMesh::GeometryRegularizationMotionBackend";
}

motion::MotionSolveResult
GeometryRegularizationMotionBackend::solve(const motion::MotionSolveRequest& request)
{
    diagnostics_ = {};
    diagnostics_.backend_name = name();
    diagnostics_.model = to_string(options_.model);
    diagnostics_.weight_mode = to_string(options_.weight_mode);
    diagnostics_.constraint_mode = to_string(options_.constraint_mode);
    diagnostics_.accepted_step_scale = static_cast<Real>(request.step_scale);

    if (!request.displacement.valid()) {
        return failure_result(diagnostics_, "geometry regularization requires a valid displacement field view");
    }

    MeshBase& mesh = request.mesh.local_mesh();
    const int dim = mesh.dim();
    const auto n_vertices = mesh.n_vertices();
    diagnostics_.vertices = n_vertices;

    if (dim <= 0 || dim > 3) {
        return failure_result(diagnostics_, "geometry regularization requires a 1D, 2D, or 3D mesh");
    }
    if (request.displacement.n_entities != n_vertices ||
        request.displacement.components < static_cast<std::size_t>(dim)) {
        return failure_result(diagnostics_, "geometry regularization displacement view does not match the mesh");
    }

    const auto& x = coordinates(mesh, request.geometry_config);
    if (x.size() < n_vertices * static_cast<std::size_t>(dim)) {
        return failure_result(diagnostics_, "geometry regularization coordinate storage does not match the mesh");
    }

    const auto quality = motion::evaluate_motion_quality(request.mesh, request.geometry_config);
    diagnostics_.minimum_quality_jacobian = quality.min_jacobian;
    diagnostics_.minimum_quality_angle_degrees = quality.min_angle_deg;
    diagnostics_.maximum_quality_skewness = quality.max_skewness;

    const auto constraint_result = build_constraints(mesh, x, request, options_);
    if (!constraint_result.message.empty()) {
        return failure_result(diagnostics_, constraint_result.message);
    }
    diagnostics_.constrained_components = constraint_result.constrained_components;

    std::string graph_message;
    const Graph graph = build_graph(mesh, x, constraint_result.constraints, options_, graph_message);
    if (!graph_message.empty()) {
        return failure_result(diagnostics_, graph_message);
    }
    if (graph.adjacency.size() != n_vertices) {
        return failure_result(diagnostics_, "geometry regularization failed to assemble the mesh graph");
    }

    const std::size_t disp_components = request.displacement.components;
    std::fill(request.displacement.data,
              request.displacement.data + n_vertices * disp_components,
              real_t(0.0));

    std::vector<Real> step_values(n_vertices * static_cast<std::size_t>(dim), Real(0.0));
    std::vector<Real> component_solution;
    for (int d = 0; d < dim; ++d) {
        const LinearStats stats = solve_component(graph, constraint_result.constraints, d, options_, component_solution);
        diagnostics_.linear_iterations += stats.iterations;
        diagnostics_.initial_residual_norm =
            std::max(diagnostics_.initial_residual_norm, stats.initial_residual);
        diagnostics_.final_residual_norm =
            std::max(diagnostics_.final_residual_norm, stats.final_residual);
        if (!stats.success) {
            return failure_result(diagnostics_, stats.message);
        }

        for (VertexId v = 0; v < n_vertices; ++v) {
            step_values[v * static_cast<VertexId>(dim) + static_cast<VertexId>(d)] =
                component_solution[v];
            request.displacement.data[v * disp_components + static_cast<VertexId>(d)] =
                static_cast<real_t>(component_solution[v]);
        }
    }

    diagnostics_.free_components =
        n_vertices * static_cast<std::size_t>(dim) - diagnostics_.constrained_components;
    diagnostics_.max_constraint_violation =
        max_constraint_violation(constraint_result.constraints,
                                 std::span<const Real>(step_values.data(), step_values.size()),
                                 dim);

    if (diagnostics_.max_constraint_violation > options_.constraint_tolerance) {
        return failure_result(diagnostics_, "geometry regularization failed to satisfy mesh-motion constraints");
    }

    motion::MotionSolveResult result;
    result.success = true;
    result.wrote_velocity = false;

    if (options_.write_velocity && request.velocity.valid() &&
        request.velocity.n_entities == n_vertices &&
        request.velocity.components >= static_cast<std::size_t>(dim) &&
        request.dt > 0.0 && request.step_scale > 0.0) {
        const std::size_t vel_components = request.velocity.components;
        std::fill(request.velocity.data,
                  request.velocity.data + n_vertices * vel_components,
                  real_t(0.0));
        const Real inv_substep_dt =
            Real(1.0) / (static_cast<Real>(request.dt) * static_cast<Real>(request.step_scale));
        for (VertexId v = 0; v < n_vertices; ++v) {
            for (int d = 0; d < dim; ++d) {
                request.velocity.data[v * vel_components + static_cast<VertexId>(d)] =
                    request.displacement.data[v * disp_components + static_cast<VertexId>(d)] *
                    static_cast<real_t>(inv_substep_dt);
            }
        }
        result.wrote_velocity = true;
    }

    diagnostics_.success = true;
    std::ostringstream os;
    os << "geometry regularization accepted step_scale=" << request.step_scale
       << ", linear_iterations=" << diagnostics_.linear_iterations
       << ", final_residual=" << diagnostics_.final_residual_norm
       << ", max_constraint_violation=" << diagnostics_.max_constraint_violation;
    diagnostics_.message = os.str();
    result.message = diagnostics_.message;
    return result;
}

std::shared_ptr<motion::IMotionBackend>
make_geometry_regularization_motion_backend(GeometryRegularizationOptions options)
{
    return std::make_shared<GeometryRegularizationMotionBackend>(std::move(options));
}

} // namespace svmp::FE::moving_mesh

#endif // defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
