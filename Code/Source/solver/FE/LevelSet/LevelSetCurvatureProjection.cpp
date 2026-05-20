#include "LevelSet/LevelSetCurvatureProjection.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <queue>
#include <span>
#include <stdexcept>

namespace svmp::FE::level_set {
namespace {

[[nodiscard]] Real dot(const std::array<Real, 3>& a,
                       const std::array<Real, 3>& b) noexcept
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

[[nodiscard]] Real norm(const std::array<Real, 3>& a) noexcept
{
    return std::sqrt(dot(a, a));
}

[[nodiscard]] std::size_t fitSize(int dim)
{
    return dim == 2 ? 5u : 9u;
}

struct FitObservation {
    std::array<Real, 9> row{};
    Real rhs{0.0};
    Real weight{1.0};
};

struct FitResidualMetrics {
    Real rms{0.0};
    Real normalized{0.0};
};

void accumulateSymmetricNormalEquations(
    const std::array<Real, 9>& row,
    std::size_t n,
    Real rhs,
    Real weight,
    std::array<std::array<Real, 9>, 9>& ata,
    std::array<Real, 9>& atb)
{
    for (std::size_t i = 0; i < n; ++i) {
        const Real wi = weight * row[i];
        atb[i] += wi * rhs;
        for (std::size_t j = 0; j < n; ++j) {
            ata[i][j] += wi * row[j];
        }
    }
}

[[nodiscard]] FitResidualMetrics computeFitResidualMetrics(
    std::span<const FitObservation> observations,
    const std::array<Real, 9>& coefficients,
    std::size_t n,
    Real scale_floor)
{
    Real weighted_residual2 = Real{0.0};
    Real weighted_rhs2 = Real{0.0};
    Real weight_sum = Real{0.0};
    for (const auto& obs : observations) {
        Real predicted = Real{0.0};
        for (std::size_t i = 0; i < n; ++i) {
            predicted += obs.row[i] * coefficients[i];
        }
        const Real residual = predicted - obs.rhs;
        weighted_residual2 += obs.weight * residual * residual;
        weighted_rhs2 += obs.weight * obs.rhs * obs.rhs;
        weight_sum += obs.weight;
    }
    if (!(weight_sum > Real{0.0}) || !std::isfinite(weight_sum)) {
        return {};
    }

    const Real rms = std::sqrt(weighted_residual2 / weight_sum);
    const Real rhs_rms = std::sqrt(weighted_rhs2 / weight_sum);
    const Real normalized =
        rms / std::max(std::max(rhs_rms, scale_floor), Real{1.0e-300});
    if (!std::isfinite(rms) || !std::isfinite(normalized)) {
        return {std::numeric_limits<Real>::infinity(),
                std::numeric_limits<Real>::infinity()};
    }
    return {rms, normalized};
}

[[nodiscard]] bool solveDenseSystem(std::array<std::array<Real, 9>, 9> a,
                                    std::array<Real, 9> b,
                                    std::size_t n,
                                    Real tolerance,
                                    std::array<Real, 9>& x)
{
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t pivot = k;
        Real pivot_abs = std::abs(a[k][k]);
        for (std::size_t r = k + 1u; r < n; ++r) {
            const Real candidate = std::abs(a[r][k]);
            if (candidate > pivot_abs) {
                pivot = r;
                pivot_abs = candidate;
            }
        }
        if (!(pivot_abs > tolerance) || !std::isfinite(pivot_abs)) {
            return false;
        }
        if (pivot != k) {
            std::swap(a[pivot], a[k]);
            std::swap(b[pivot], b[k]);
        }

        const Real diag = a[k][k];
        for (std::size_t c = k; c < n; ++c) {
            a[k][c] /= diag;
        }
        b[k] /= diag;

        for (std::size_t r = 0; r < n; ++r) {
            if (r == k) {
                continue;
            }
            const Real factor = a[r][k];
            if (factor == Real{0.0}) {
                continue;
            }
            for (std::size_t c = k; c < n; ++c) {
                a[r][c] -= factor * a[k][c];
            }
            b[r] -= factor * b[k];
        }
    }

    x.fill(Real{0.0});
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(b[i])) {
            return false;
        }
        x[i] = b[i];
    }
    return true;
}

[[nodiscard]] std::vector<std::vector<GlobalIndex>> buildVertexAdjacency(
    const assembly::IMeshAccess& mesh)
{
    const auto n_vertices = static_cast<std::size_t>(mesh.numVertices());
    std::vector<std::vector<GlobalIndex>> adjacency(n_vertices);
    std::vector<GlobalIndex> nodes;
    mesh.forEachCell([&](GlobalIndex cell) {
        mesh.getCellNodes(cell, nodes);
        for (const auto a : nodes) {
            if (a < 0 || static_cast<std::size_t>(a) >= n_vertices) {
                continue;
            }
            auto& row = adjacency[static_cast<std::size_t>(a)];
            for (const auto b : nodes) {
                if (b == a || b < 0 ||
                    static_cast<std::size_t>(b) >= n_vertices) {
                    continue;
                }
                row.push_back(b);
            }
        }
    });
    for (auto& row : adjacency) {
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
    }
    return adjacency;
}

[[nodiscard]] std::vector<std::vector<std::size_t>> buildVertexSupplementalSampleAdjacency(
    const assembly::IMeshAccess& mesh,
    std::span<const LevelSetCurvatureProjectionSample> samples)
{
    const auto n_vertices = static_cast<std::size_t>(mesh.numVertices());
    std::vector<std::vector<std::size_t>> sample_adjacency(n_vertices);
    std::vector<GlobalIndex> nodes;
    for (std::size_t sample_index = 0; sample_index < samples.size();
         ++sample_index) {
        const auto& sample = samples[sample_index];
        if (sample.parent_cell >= static_cast<MeshIndex>(0) &&
            sample.parent_cell < mesh.numCells()) {
            mesh.getCellNodes(static_cast<GlobalIndex>(sample.parent_cell), nodes);
            for (const auto node : nodes) {
                if (node < 0 || static_cast<std::size_t>(node) >= n_vertices) {
                    continue;
                }
                sample_adjacency[static_cast<std::size_t>(node)].push_back(
                    sample_index);
            }
            continue;
        }

        Real best_distance2 = std::numeric_limits<Real>::infinity();
        GlobalIndex best_vertex = static_cast<GlobalIndex>(-1);
        for (GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
            const auto x = mesh.getNodeCoordinates(vertex);
            const std::array<Real, 3> dx{
                x[0] - sample.coordinate[0],
                x[1] - sample.coordinate[1],
                x[2] - sample.coordinate[2],
            };
            const Real distance2 = dot(dx, dx);
            if (distance2 < best_distance2) {
                best_distance2 = distance2;
                best_vertex = vertex;
            }
        }
        if (best_vertex >= 0) {
            sample_adjacency[static_cast<std::size_t>(best_vertex)].push_back(
                sample_index);
        }
    }

    for (auto& row : sample_adjacency) {
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
    }
    return sample_adjacency;
}

void mixSignature(std::uint64_t& seed, std::uint64_t value) noexcept
{
    seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
}

[[nodiscard]] std::uint64_t realBitsForSignature(Real value) noexcept
{
    std::uint64_t bits = 0u;
    static_assert(sizeof(value) <= sizeof(bits),
                  "level-set curvature projection signature expects Real <= 64 bits");
    std::memcpy(&bits, &value, sizeof(value));
    return bits;
}

[[nodiscard]] std::uint64_t supplementalSampleAdjacencySignature(
    std::span<const LevelSetCurvatureProjectionSample> samples) noexcept
{
    std::uint64_t seed = 0xcbf29ce484222325ull;
    mixSignature(seed, static_cast<std::uint64_t>(samples.size()));
    for (const auto& sample : samples) {
        mixSignature(seed, static_cast<std::uint64_t>(sample.parent_cell));
        for (const auto coordinate : sample.coordinate) {
            mixSignature(seed, realBitsForSignature(coordinate));
        }
    }
    return seed;
}

[[nodiscard]] bool workspaceMatchesMesh(
    const LevelSetCurvatureProjectionWorkspace& workspace,
    const assembly::IMeshAccess& mesh) noexcept
{
    return workspace.mesh_vertices == mesh.numVertices() &&
           workspace.mesh_cells == mesh.numCells() &&
           workspace.mesh_dimension == mesh.dimension();
}

void recordWorkspaceMeshIdentity(LevelSetCurvatureProjectionWorkspace& workspace,
                                 const assembly::IMeshAccess& mesh) noexcept
{
    workspace.mesh_vertices = mesh.numVertices();
    workspace.mesh_cells = mesh.numCells();
    workspace.mesh_dimension = mesh.dimension();
    workspace.mesh_revision_tracking_available =
        mesh.revisionTrackingAvailable();
    workspace.mesh_geometry_revision =
        workspace.mesh_revision_tracking_available ? mesh.geometryRevision() : 0u;
    workspace.mesh_topology_revision =
        workspace.mesh_revision_tracking_available ? mesh.topologyRevision() : 0u;
    workspace.mesh_ownership_revision =
        workspace.mesh_revision_tracking_available ? mesh.ownershipRevision() : 0u;
    workspace.mesh_numbering_revision =
        workspace.mesh_revision_tracking_available ? mesh.numberingRevision() : 0u;
    workspace.mesh_coordinate_configuration_key =
        workspace.mesh_revision_tracking_available
            ? mesh.coordinateConfigurationKey()
            : 0u;
}

[[nodiscard]] bool workspaceMatchesTopology(
    const LevelSetCurvatureProjectionWorkspace& workspace,
    const assembly::IMeshAccess& mesh) noexcept
{
    if (!workspaceMatchesMesh(workspace, mesh)) {
        return false;
    }
    if (!mesh.revisionTrackingAvailable()) {
        return true;
    }
    return workspace.mesh_revision_tracking_available &&
           workspace.mesh_topology_revision == mesh.topologyRevision() &&
           workspace.mesh_ownership_revision == mesh.ownershipRevision() &&
           workspace.mesh_numbering_revision == mesh.numberingRevision();
}

[[nodiscard]] bool workspaceMatchesSampleGeometry(
    const LevelSetCurvatureProjectionWorkspace& workspace,
    const assembly::IMeshAccess& mesh) noexcept
{
    if (!workspaceMatchesTopology(workspace, mesh) ||
        !mesh.revisionTrackingAvailable()) {
        return false;
    }
    return workspace.mesh_revision_tracking_available &&
           workspace.mesh_geometry_revision == mesh.geometryRevision() &&
           workspace.mesh_coordinate_configuration_key ==
               mesh.coordinateConfigurationKey();
}

[[nodiscard]] const std::vector<std::vector<GlobalIndex>>&
cachedVertexAdjacency(const assembly::IMeshAccess& mesh,
                      LevelSetCurvatureProjectionWorkspace* workspace,
                      LevelSetCurvatureProjectionResult& result,
                      std::vector<std::vector<GlobalIndex>>& local_adjacency)
{
    if (workspace != nullptr &&
        workspace->vertex_adjacency_valid &&
        workspaceMatchesTopology(*workspace, mesh)) {
        result.reused_vertex_adjacency = true;
        result.vertex_adjacency_builds = workspace->vertex_adjacency_builds;
        return workspace->vertex_adjacency;
    }

    auto adjacency = buildVertexAdjacency(mesh);
    if (workspace != nullptr) {
        recordWorkspaceMeshIdentity(*workspace, mesh);
        workspace->vertex_adjacency = std::move(adjacency);
        workspace->vertex_adjacency_valid = true;
        workspace->sample_adjacency_valid = false;
        ++workspace->vertex_adjacency_builds;
        result.vertex_adjacency_builds = workspace->vertex_adjacency_builds;
        return workspace->vertex_adjacency;
    }

    local_adjacency = std::move(adjacency);
    result.vertex_adjacency_builds = 1u;
    return local_adjacency;
}

[[nodiscard]] const std::vector<std::vector<std::size_t>>&
cachedSampleAdjacency(
    const assembly::IMeshAccess& mesh,
    std::span<const LevelSetCurvatureProjectionSample> supplemental_samples,
    LevelSetCurvatureProjectionWorkspace* workspace,
    LevelSetCurvatureProjectionResult& result,
    std::vector<std::vector<std::size_t>>& local_sample_adjacency)
{
    const auto sample_signature =
        supplementalSampleAdjacencySignature(supplemental_samples);
    if (workspace != nullptr &&
        workspace->sample_adjacency_valid &&
        workspaceMatchesSampleGeometry(*workspace, mesh) &&
        workspace->sample_signature == sample_signature) {
        result.reused_sample_adjacency = true;
        result.sample_adjacency_builds = workspace->sample_adjacency_builds;
        return workspace->sample_adjacency;
    }

    auto sample_adjacency =
        buildVertexSupplementalSampleAdjacency(mesh, supplemental_samples);
    if (workspace != nullptr) {
        recordWorkspaceMeshIdentity(*workspace, mesh);
        workspace->sample_signature = sample_signature;
        workspace->sample_adjacency = std::move(sample_adjacency);
        workspace->sample_adjacency_valid = true;
        ++workspace->sample_adjacency_builds;
        result.sample_adjacency_builds = workspace->sample_adjacency_builds;
        return workspace->sample_adjacency;
    }

    local_sample_adjacency = std::move(sample_adjacency);
    result.sample_adjacency_builds = 1u;
    return local_sample_adjacency;
}

[[nodiscard]] std::vector<GlobalIndex> collectNeighbors(
    GlobalIndex center,
    const std::vector<std::vector<GlobalIndex>>& adjacency,
    int max_rings)
{
    const auto n_vertices = adjacency.size();
    if (center < 0 || static_cast<std::size_t>(center) >= n_vertices) {
        return {};
    }

    std::vector<unsigned char> visited(n_vertices, 0u);
    std::queue<std::pair<GlobalIndex, int>> queue;
    visited[static_cast<std::size_t>(center)] = 1u;
    queue.push({center, 0});

    std::vector<GlobalIndex> result;
    while (!queue.empty()) {
        const auto [vertex, ring] = queue.front();
        queue.pop();
        if (ring >= max_rings) {
            continue;
        }
        for (const auto neighbor :
             adjacency[static_cast<std::size_t>(vertex)]) {
            const auto idx = static_cast<std::size_t>(neighbor);
            if (visited[idx] != 0u) {
                continue;
            }
            visited[idx] = 1u;
            result.push_back(neighbor);
            queue.push({neighbor, ring + 1});
        }
    }
    return result;
}

[[nodiscard]] std::array<Real, 9> quadraticRow(const std::array<Real, 3>& dx,
                                               int dim) noexcept
{
    std::array<Real, 9> row{};
    row[0] = dx[0];
    row[1] = dx[1];
    if (dim == 2) {
        row[2] = Real{0.5} * dx[0] * dx[0];
        row[3] = dx[0] * dx[1];
        row[4] = Real{0.5} * dx[1] * dx[1];
        return row;
    }
    row[2] = dx[2];
    row[3] = Real{0.5} * dx[0] * dx[0];
    row[4] = dx[0] * dx[1];
    row[5] = dx[0] * dx[2];
    row[6] = Real{0.5} * dx[1] * dx[1];
    row[7] = dx[1] * dx[2];
    row[8] = Real{0.5} * dx[2] * dx[2];
    return row;
}

[[nodiscard]] Real curvatureFromFit(const std::array<Real, 9>& c,
                                    int dim,
                                    Real gradient_tolerance,
                                    bool& small_gradient) noexcept
{
    std::array<Real, 3> g{c[0], c[1], dim == 2 ? Real{0.0} : c[2]};
    std::array<std::array<Real, 3>, 3> h{};
    if (dim == 2) {
        h[0][0] = c[2];
        h[0][1] = c[3];
        h[1][0] = c[3];
        h[1][1] = c[4];
    } else {
        h[0][0] = c[3];
        h[0][1] = c[4];
        h[0][2] = c[5];
        h[1][0] = c[4];
        h[1][1] = c[6];
        h[1][2] = c[7];
        h[2][0] = c[5];
        h[2][1] = c[7];
        h[2][2] = c[8];
    }

    const Real g_norm = norm(g);
    if (!(g_norm > gradient_tolerance) || !std::isfinite(g_norm)) {
        small_gradient = true;
        return Real{0.0};
    }

    const Real trace_h = h[0][0] + h[1][1] + (dim == 2 ? Real{0.0} : h[2][2]);
    Real ghg = Real{0.0};
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            ghg += g[static_cast<std::size_t>(i)] *
                   h[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] *
                   g[static_cast<std::size_t>(j)];
        }
    }
    return ((g_norm * g_norm) * trace_h - ghg) /
           (g_norm * g_norm * g_norm);
}

void smoothCurvatureOnVertexGraph(
    const std::vector<std::vector<GlobalIndex>>& adjacency,
    int iterations,
    Real relaxation,
    std::vector<Real>& curvature,
    LevelSetCurvatureProjectionResult& result)
{
    if (iterations <= 0 || !(relaxation > Real{0.0})) {
        return;
    }

    std::vector<Real> current = curvature;
    std::vector<Real> next = current;
    Real total_abs_update = Real{0.0};
    std::size_t update_count = 0u;

    for (int iter = 0; iter < iterations; ++iter) {
        Real iteration_max_update = Real{0.0};
        for (std::size_t vertex = 0; vertex < current.size(); ++vertex) {
            const auto& neighbors = adjacency[vertex];
            if (neighbors.empty()) {
                next[vertex] = current[vertex];
                continue;
            }

            Real sum = Real{0.0};
            std::size_t count = 0u;
            for (const auto neighbor : neighbors) {
                const auto index = static_cast<std::size_t>(neighbor);
                if (index >= current.size() || !std::isfinite(current[index])) {
                    continue;
                }
                sum += current[index];
                ++count;
            }
            if (count == 0u || !std::isfinite(current[vertex])) {
                next[vertex] = current[vertex];
                continue;
            }

            const Real average = sum / static_cast<Real>(count);
            next[vertex] =
                current[vertex] + relaxation * (average - current[vertex]);
            if (!std::isfinite(next[vertex])) {
                next[vertex] = current[vertex];
                continue;
            }
            const Real update = std::abs(next[vertex] - current[vertex]);
            total_abs_update += update;
            iteration_max_update = std::max(iteration_max_update, update);
            ++update_count;
        }
        current.swap(next);
        result.smoothing_max_abs_update =
            std::max(result.smoothing_max_abs_update, iteration_max_update);
        ++result.smoothing_iterations_applied;
    }

    if (update_count > 0u) {
        result.smoothing_mean_abs_update =
            total_abs_update / static_cast<Real>(update_count);
    }
    curvature = std::move(current);
}

} // namespace

LevelSetCurvatureProjectionResult projectLevelSetMeanCurvatureToVertices(
    const assembly::IMeshAccess& mesh,
    std::span<const Real> level_set_vertex_values,
    std::span<const LevelSetCurvatureProjectionSample> supplemental_samples,
    const LevelSetCurvatureProjectionOptions& options,
    std::vector<Real>& curvature_vertex_values,
    LevelSetCurvatureProjectionWorkspace* workspace);

LevelSetCurvatureProjectionResult projectLevelSetMeanCurvatureToVertices(
    const assembly::IMeshAccess& mesh,
    std::span<const Real> level_set_vertex_values,
    const LevelSetCurvatureProjectionOptions& options,
    std::vector<Real>& curvature_vertex_values)
{
    return projectLevelSetMeanCurvatureToVertices(
        mesh,
        level_set_vertex_values,
        std::span<const LevelSetCurvatureProjectionSample>{},
        options,
        curvature_vertex_values);
}

LevelSetCurvatureProjectionResult projectLevelSetMeanCurvatureToVertices(
    const assembly::IMeshAccess& mesh,
    std::span<const Real> level_set_vertex_values,
    std::span<const LevelSetCurvatureProjectionSample> supplemental_samples,
    const LevelSetCurvatureProjectionOptions& options,
    std::vector<Real>& curvature_vertex_values)
{
    return projectLevelSetMeanCurvatureToVertices(
        mesh,
        level_set_vertex_values,
        supplemental_samples,
        options,
        curvature_vertex_values,
        /*workspace=*/nullptr);
}

LevelSetCurvatureProjectionResult projectLevelSetMeanCurvatureToVertices(
    const assembly::IMeshAccess& mesh,
    std::span<const Real> level_set_vertex_values,
    std::span<const LevelSetCurvatureProjectionSample> supplemental_samples,
    const LevelSetCurvatureProjectionOptions& options,
    std::vector<Real>& curvature_vertex_values,
    LevelSetCurvatureProjectionWorkspace& workspace)
{
    return projectLevelSetMeanCurvatureToVertices(
        mesh,
        level_set_vertex_values,
        supplemental_samples,
        options,
        curvature_vertex_values,
        &workspace);
}

LevelSetCurvatureProjectionResult projectLevelSetMeanCurvatureToVertices(
    const assembly::IMeshAccess& mesh,
    std::span<const Real> level_set_vertex_values,
    std::span<const LevelSetCurvatureProjectionSample> supplemental_samples,
    const LevelSetCurvatureProjectionOptions& options,
    std::vector<Real>& curvature_vertex_values,
    LevelSetCurvatureProjectionWorkspace* workspace)
{
    const auto n_vertices = static_cast<std::size_t>(mesh.numVertices());
    if (level_set_vertex_values.size() != n_vertices) {
        throw std::invalid_argument(
            "level-set curvature projection requires one level-set value per mesh vertex");
    }
    const int dim = mesh.dimension();
    if (dim != 2 && dim != 3) {
        throw std::invalid_argument(
            "level-set curvature projection supports two- and three-dimensional meshes");
    }
    if (!(options.gradient_tolerance > Real{0.0}) ||
        !(options.normal_equation_tolerance > Real{0.0}) ||
        options.max_normalized_fit_residual < Real{0.0} ||
        !std::isfinite(options.max_normalized_fit_residual) ||
        options.smoothing_iterations < 0 ||
        options.smoothing_relaxation < Real{0.0} ||
        options.smoothing_relaxation > Real{1.0} ||
        !std::isfinite(options.smoothing_relaxation)) {
        throw std::invalid_argument(
            "level-set curvature projection requires positive tolerances, a nonnegative residual limit, nonnegative smoothing iterations, and smoothing relaxation in [0,1]");
    }

    LevelSetCurvatureProjectionResult result;
    result.vertices = n_vertices;
    result.supplemental_samples = supplemental_samples.size();
    curvature_vertex_values.assign(n_vertices, Real{0.0});
    if (n_vertices == 0u) {
        result.diagnostic = "level-set curvature projection received an empty mesh";
        return result;
    }

    std::vector<std::vector<GlobalIndex>> local_adjacency;
    const auto& adjacency =
        cachedVertexAdjacency(mesh, workspace, result, local_adjacency);
    std::vector<std::vector<std::size_t>> local_sample_adjacency;
    const auto& sample_adjacency =
        cachedSampleAdjacency(mesh,
                              supplemental_samples,
                              workspace,
                              result,
                              local_sample_adjacency);
    const auto n_fit = fitSize(dim);
    const int rings = std::max(1, options.max_neighbor_rings);
    std::vector<unsigned char> fitted(n_vertices, 0u);

    for (GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
        const auto center = mesh.getNodeCoordinates(vertex);
        const auto neighbors = collectNeighbors(vertex, adjacency, rings);

        std::array<std::array<Real, 9>, 9> ata{};
        std::array<Real, 9> atb{};
        std::vector<FitObservation> observations;
        std::size_t rows = 0u;
        const auto center_value =
            level_set_vertex_values[static_cast<std::size_t>(vertex)];
        if (!std::isfinite(center_value)) {
            throw std::invalid_argument(
                "level-set curvature projection received a non-finite level-set value");
        }
        for (const auto neighbor : neighbors) {
            const auto neighbor_index = static_cast<std::size_t>(neighbor);
            const auto x = mesh.getNodeCoordinates(neighbor);
            std::array<Real, 3> dx{
                x[0] - center[0],
                x[1] - center[1],
                dim == 2 ? Real{0.0} : x[2] - center[2],
            };
            const Real distance2 = dot(dx, dx);
            if (!(distance2 > Real{0.0}) || !std::isfinite(distance2)) {
                continue;
            }
            const Real rhs = level_set_vertex_values[neighbor_index] -
                             center_value;
            if (!std::isfinite(rhs)) {
                throw std::invalid_argument(
                    "level-set curvature projection received a non-finite level-set value");
            }
            const Real weight = Real{1.0} / std::max(distance2, Real{1.0e-24});
            const auto row = quadraticRow(dx, dim);
            accumulateSymmetricNormalEquations(
                row, n_fit, rhs, weight, ata, atb);
            observations.push_back(FitObservation{row, rhs, weight});
            ++rows;
        }
        std::vector<std::size_t> sample_indices =
            sample_adjacency[static_cast<std::size_t>(vertex)];
        for (const auto neighbor : neighbors) {
            const auto& neighbor_samples =
                sample_adjacency[static_cast<std::size_t>(neighbor)];
            sample_indices.insert(sample_indices.end(),
                                  neighbor_samples.begin(),
                                  neighbor_samples.end());
        }
        std::sort(sample_indices.begin(), sample_indices.end());
        sample_indices.erase(
            std::unique(sample_indices.begin(), sample_indices.end()),
            sample_indices.end());

        std::size_t supplemental_rows = 0u;
        for (const auto sample_index : sample_indices) {
            if (sample_index >= supplemental_samples.size()) {
                continue;
            }
            const auto& sample = supplemental_samples[sample_index];
            if (!std::isfinite(sample.value) ||
                !std::isfinite(sample.coordinate[0]) ||
                !std::isfinite(sample.coordinate[1]) ||
                !std::isfinite(sample.coordinate[2])) {
                throw std::invalid_argument(
                    "level-set curvature projection received a non-finite supplemental sample");
            }
            std::array<Real, 3> dx{
                sample.coordinate[0] - center[0],
                sample.coordinate[1] - center[1],
                dim == 2 ? Real{0.0} : sample.coordinate[2] - center[2],
            };
            const Real distance2 = dot(dx, dx);
            if (!(distance2 > Real{0.0}) || !std::isfinite(distance2)) {
                continue;
            }
            const Real rhs = sample.value - center_value;
            const Real weight = Real{1.0} / std::max(distance2, Real{1.0e-24});
            const auto row = quadraticRow(dx, dim);
            accumulateSymmetricNormalEquations(
                row, n_fit, rhs, weight, ata, atb);
            observations.push_back(FitObservation{row, rhs, weight});
            ++rows;
            ++supplemental_rows;
        }
        result.supplemental_sample_rows += supplemental_rows;
        if (supplemental_rows > 0u) {
            ++result.vertices_with_supplemental_samples;
        }
        if (rows < n_fit) {
            ++result.insufficient_stencil_vertices;
            continue;
        }

        std::array<Real, 9> coefficients{};
        if (!solveDenseSystem(
                ata, atb, n_fit, options.normal_equation_tolerance,
                coefficients)) {
            ++result.singular_stencil_vertices;
            continue;
        }
        const auto residual = computeFitResidualMetrics(
            std::span<const FitObservation>(observations.data(),
                                            observations.size()),
            coefficients,
            n_fit,
            options.gradient_tolerance);
        if (!(std::isfinite(residual.rms) &&
              std::isfinite(residual.normalized))) {
            ++result.singular_stencil_vertices;
            continue;
        }
        if (options.max_normalized_fit_residual > Real{0.0} &&
            residual.normalized > options.max_normalized_fit_residual) {
            ++result.fit_residual_failure_vertices;
            continue;
        }

        bool small_gradient = false;
        const Real kappa = curvatureFromFit(
            coefficients, dim, options.gradient_tolerance, small_gradient);
        if (small_gradient) {
            ++result.small_gradient_vertices;
            continue;
        }
        if (!std::isfinite(kappa)) {
            ++result.singular_stencil_vertices;
            continue;
        }
        curvature_vertex_values[static_cast<std::size_t>(vertex)] = kappa;
        fitted[static_cast<std::size_t>(vertex)] = 1u;
        ++result.fitted_vertices;
        result.mean_fit_rms_residual += residual.rms;
        result.mean_normalized_fit_residual += residual.normalized;
        result.max_fit_rms_residual =
            std::max(result.max_fit_rms_residual, residual.rms);
        result.max_normalized_fit_residual =
            std::max(result.max_normalized_fit_residual, residual.normalized);
    }

    std::vector<unsigned char> recovered = fitted;
    std::vector<GlobalIndex> pending;
    pending.reserve(n_vertices);
    for (GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
        const auto index = static_cast<std::size_t>(vertex);
        if (recovered[index] == 0u) {
            pending.push_back(vertex);
        }
    }

    std::vector<GlobalIndex> next_pending;
    next_pending.reserve(pending.size());
    while (!pending.empty()) {
        bool made_progress = false;
        next_pending.clear();
        for (const auto vertex : pending) {
            const auto index = static_cast<std::size_t>(vertex);
            Real sum = Real{0.0};
            std::size_t count = 0u;
            for (const auto neighbor : adjacency[index]) {
                const auto neighbor_index = static_cast<std::size_t>(neighbor);
                if (recovered[neighbor_index] == 0u) {
                    continue;
                }
                sum += curvature_vertex_values[neighbor_index];
                ++count;
            }
            if (count > 0u) {
                curvature_vertex_values[index] = sum / static_cast<Real>(count);
                recovered[index] = 1u;
                ++result.fallback_vertices;
                made_progress = true;
            } else {
                next_pending.push_back(vertex);
            }
        }
        if (!made_progress) {
            break;
        }
        pending.swap(next_pending);
    }
    for (const auto vertex : pending) {
        const auto index = static_cast<std::size_t>(vertex);
        if (recovered[index] == 0u) {
            curvature_vertex_values[index] = Real{0.0};
            ++result.zero_fallback_vertices;
        }
    }

    if (result.fitted_vertices == 0u) {
        result.diagnostic = result.fit_residual_failure_vertices > 0u
            ? "level-set curvature projection exceeded the normalized fit residual limit"
            : "level-set curvature projection could not fit any vertex stencil";
        return result;
    }

    const auto fitted_count = static_cast<Real>(result.fitted_vertices);
    result.mean_fit_rms_residual /= fitted_count;
    result.mean_normalized_fit_residual /= fitted_count;

    smoothCurvatureOnVertexGraph(
        adjacency,
        options.smoothing_iterations,
        options.smoothing_relaxation,
        curvature_vertex_values,
        result);

    result.min_curvature = std::numeric_limits<Real>::infinity();
    result.max_curvature = -std::numeric_limits<Real>::infinity();
    for (const auto kappa : curvature_vertex_values) {
        result.min_curvature = std::min(result.min_curvature, kappa);
        result.max_curvature = std::max(result.max_curvature, kappa);
        result.max_abs_curvature =
            std::max(result.max_abs_curvature, std::abs(kappa));
    }
    if (!std::isfinite(result.min_curvature)) {
        result.min_curvature = Real{0.0};
    }
    if (!std::isfinite(result.max_curvature)) {
        result.max_curvature = Real{0.0};
    }
    if (result.fit_residual_failure_vertices > 0u) {
        result.diagnostic =
            "level-set curvature projection used neighbor fallback for stencils "
            "that exceeded the normalized fit residual limit";
    }
    result.success = true;
    return result;
}

} // namespace svmp::FE::level_set
