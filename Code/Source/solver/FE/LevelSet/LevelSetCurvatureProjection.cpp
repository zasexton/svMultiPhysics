#include "LevelSet/LevelSetCurvatureProjection.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <queue>
#include <span>
#include <stdexcept>
#include <string>

namespace svmp::FE::level_set {
namespace {

[[nodiscard]] std::string normalizedSmoothingToken(std::string_view value)
{
    std::string token(value);
    token.erase(token.begin(),
                std::find_if(token.begin(), token.end(), [](unsigned char c) {
                    return !std::isspace(c);
                }));
    token.erase(std::find_if(token.rbegin(), token.rend(), [](unsigned char c) {
                    return !std::isspace(c);
                }).base(),
                token.end());
    std::transform(token.begin(), token.end(), token.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    token.erase(std::remove_if(token.begin(), token.end(),
                               [](unsigned char c) {
                                   return c == '_' || c == '-' ||
                                          std::isspace(c);
                               }),
                token.end());
    return token;
}

} // namespace

const char* levelSetCurvatureSmoothingModeName(
    LevelSetCurvatureSmoothingMode mode) noexcept
{
    switch (mode) {
        case LevelSetCurvatureSmoothingMode::LocalGraph:
            return "local_graph";
        case LevelSetCurvatureSmoothingMode::MassStiffnessOperator:
            return "mass_stiffness_operator";
    }
    return "unknown";
}

LevelSetCurvatureSmoothingMode parseLevelSetCurvatureSmoothingMode(
    std::string_view value)
{
    const auto token = normalizedSmoothingToken(value);
    if (token.empty() || token == "local" || token == "graph" ||
        token == "localgraph") {
        return LevelSetCurvatureSmoothingMode::LocalGraph;
    }
    if (token == "global" || token == "operator" ||
        token == "operatorprojection" || token == "massstiffness" ||
        token == "massstiffnessoperator" || token == "helmholtz" ||
        token == "helmholtzprojection" || token == "feprojection") {
        return LevelSetCurvatureSmoothingMode::MassStiffnessOperator;
    }
    throw std::invalid_argument(
        "level-set curvature projection smoothing mode '" +
        std::string(value) +
        "' must be local_graph or mass_stiffness_operator");
}

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
        if (sample.parent_cell < static_cast<MeshIndex>(0)) {
            for (const auto coordinate : sample.coordinate) {
                mixSignature(seed, realBitsForSignature(coordinate));
            }
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

struct WeightedNeighbor {
    std::size_t vertex{0};
    Real weight{0.0};
};

[[nodiscard]] std::array<Real, 3> subtract(const std::array<Real, 3>& a,
                                           const std::array<Real, 3>& b) noexcept
{
    return {{a[0] - b[0], a[1] - b[1], a[2] - b[2]}};
}

[[nodiscard]] std::array<Real, 3> cross(const std::array<Real, 3>& a,
                                        const std::array<Real, 3>& b) noexcept
{
    return {{a[1] * b[2] - a[2] * b[1],
             a[2] * b[0] - a[0] * b[2],
             a[0] * b[1] - a[1] * b[0]}};
}

[[nodiscard]] Real tripleProduct(const std::array<Real, 3>& a,
                                 const std::array<Real, 3>& b,
                                 const std::array<Real, 3>& c) noexcept
{
    return dot(a, cross(b, c));
}

[[nodiscard]] Real triangleMeasure(const std::array<Real, 3>& a,
                                   const std::array<Real, 3>& b,
                                   const std::array<Real, 3>& c) noexcept
{
    return Real{0.5} * norm(cross(subtract(b, a), subtract(c, a)));
}

[[nodiscard]] Real tetraMeasure(const std::array<Real, 3>& a,
                                const std::array<Real, 3>& b,
                                const std::array<Real, 3>& c,
                                const std::array<Real, 3>& d) noexcept
{
    return std::abs(
               tripleProduct(subtract(b, a), subtract(c, a), subtract(d, a))) /
           Real{6.0};
}

[[nodiscard]] std::size_t primaryVertexCount(ElementType type,
                                             std::size_t nodes) noexcept
{
    switch (type) {
        case ElementType::Triangle3:
        case ElementType::Triangle6:
            return std::min<std::size_t>(3u, nodes);
        case ElementType::Quad4:
        case ElementType::Quad8:
        case ElementType::Quad9:
            return std::min<std::size_t>(4u, nodes);
        case ElementType::Tetra4:
        case ElementType::Tetra10:
            return std::min<std::size_t>(4u, nodes);
        case ElementType::Hex8:
        case ElementType::Hex20:
        case ElementType::Hex27:
            return std::min<std::size_t>(8u, nodes);
        case ElementType::Wedge6:
        case ElementType::Wedge15:
        case ElementType::Wedge18:
            return std::min<std::size_t>(6u, nodes);
        case ElementType::Pyramid5:
        case ElementType::Pyramid13:
        case ElementType::Pyramid14:
            return std::min<std::size_t>(5u, nodes);
        default:
            return nodes;
    }
}

[[nodiscard]] Real estimateCellMeasure(
    const assembly::IMeshAccess& mesh,
    GlobalIndex cell,
    std::span<const GlobalIndex> nodes,
    int dim)
{
    const auto primary =
        primaryVertexCount(mesh.getCellType(cell), nodes.size());
    if (primary == 0u) {
        return Real{0.0};
    }

    std::vector<std::array<Real, 3>> x(primary);
    for (std::size_t i = 0; i < primary; ++i) {
        if (nodes[i] < 0) {
            return Real{0.0};
        }
        x[i] = mesh.getNodeCoordinates(nodes[i]);
    }

    Real measure = Real{0.0};
    if (dim == 2) {
        if (primary == 3u) {
            measure = triangleMeasure(x[0], x[1], x[2]);
        } else if (primary >= 4u) {
            for (std::size_t i = 1u; i + 1u < primary; ++i) {
                measure += triangleMeasure(x[0], x[i], x[i + 1u]);
            }
        }
    } else {
        if (primary == 4u) {
            measure = tetraMeasure(x[0], x[1], x[2], x[3]);
        } else if (primary >= 8u) {
            measure += tetraMeasure(x[0], x[1], x[3], x[4]);
            measure += tetraMeasure(x[1], x[2], x[3], x[6]);
            measure += tetraMeasure(x[1], x[4], x[5], x[6]);
            measure += tetraMeasure(x[3], x[4], x[6], x[7]);
            measure += tetraMeasure(x[1], x[3], x[4], x[6]);
        } else if (primary >= 5u) {
            for (std::size_t i = 1u; i + 2u < primary; ++i) {
                measure += tetraMeasure(x[0], x[i], x[i + 1u], x[i + 2u]);
            }
        }
    }
    return std::isfinite(measure) && measure > Real{0.0}
        ? measure
        : Real{0.0};
}

void assembleLumpedMass(
    const assembly::IMeshAccess& mesh,
    std::span<const unsigned char> active_vertices,
    std::vector<Real>& mass)
{
    std::fill(mass.begin(), mass.end(), Real{0.0});
    std::vector<GlobalIndex> nodes;
    mesh.forEachCell([&](GlobalIndex cell) {
        mesh.getCellNodes(cell, nodes);
        const Real measure =
            estimateCellMeasure(mesh,
                                cell,
                                std::span<const GlobalIndex>(nodes.data(),
                                                             nodes.size()),
                                mesh.dimension());
        if (!(measure > Real{0.0})) {
            return;
        }

        std::size_t active_count = 0u;
        for (const auto node : nodes) {
            const auto index = static_cast<std::size_t>(node);
            if (node >= 0 && index < mass.size() &&
                active_vertices[index] != 0u) {
                ++active_count;
            }
        }
        if (active_count == 0u) {
            return;
        }
        const Real lump = measure / static_cast<Real>(active_count);
        for (const auto node : nodes) {
            const auto index = static_cast<std::size_t>(node);
            if (node >= 0 && index < mass.size() &&
                active_vertices[index] != 0u) {
                mass[index] += lump;
            }
        }
    });

    for (std::size_t vertex = 0; vertex < mass.size(); ++vertex) {
        if (active_vertices[vertex] != 0u &&
            (!(mass[vertex] > Real{0.0}) || !std::isfinite(mass[vertex]))) {
            mass[vertex] = Real{1.0};
        }
    }
}

[[nodiscard]] Real vectorNorm2(std::span<const Real> values) noexcept
{
    Real sum = Real{0.0};
    for (const auto value : values) {
        sum += value * value;
    }
    return sum;
}

void applyMassStiffnessOperator(
    std::span<const Real> mass,
    const std::vector<std::vector<WeightedNeighbor>>& stiffness,
    std::span<const Real> stiffness_diag,
    Real strength,
    std::span<const Real> x,
    std::vector<Real>& y)
{
    y.assign(x.size(), Real{0.0});
    for (std::size_t row = 0; row < x.size(); ++row) {
        Real value =
            mass[row] * x[row] + strength * stiffness_diag[row] * x[row];
        for (const auto& entry : stiffness[row]) {
            value -= strength * entry.weight * x[entry.vertex];
        }
        y[row] = value;
    }
}

[[nodiscard]] bool solveMassStiffnessOperatorCG(
    std::span<const Real> mass,
    const std::vector<std::vector<WeightedNeighbor>>& stiffness,
    std::span<const Real> stiffness_diag,
    Real strength,
    std::span<const Real> rhs,
    std::span<const Real> initial_guess,
    std::vector<Real>& solution)
{
    const auto n = rhs.size();
    solution.assign(initial_guess.begin(), initial_guess.end());
    std::vector<Real> applied;
    std::vector<Real> residual(n, Real{0.0});
    std::vector<Real> direction(n, Real{0.0});
    std::vector<Real> next_applied;

    applyMassStiffnessOperator(
        mass, stiffness, stiffness_diag, strength, solution, applied);
    for (std::size_t i = 0; i < n; ++i) {
        residual[i] = rhs[i] - applied[i];
        direction[i] = residual[i];
    }

    Real rr = vectorNorm2(residual);
    const Real rhs_norm = std::sqrt(std::max(vectorNorm2(rhs), Real{0.0}));
    const Real tolerance =
        Real{1.0e-12} * std::max(rhs_norm, Real{1.0});
    if (std::sqrt(std::max(rr, Real{0.0})) <= tolerance) {
        return true;
    }

    const std::size_t max_iterations =
        std::max<std::size_t>(100u, 4u * std::max<std::size_t>(1u, n));
    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
        applyMassStiffnessOperator(
            mass, stiffness, stiffness_diag, strength, direction, next_applied);
        Real p_ap = Real{0.0};
        for (std::size_t i = 0; i < n; ++i) {
            p_ap += direction[i] * next_applied[i];
        }
        if (!(p_ap > Real{0.0}) || !std::isfinite(p_ap)) {
            return false;
        }
        const Real alpha = rr / p_ap;
        for (std::size_t i = 0; i < n; ++i) {
            solution[i] += alpha * direction[i];
            residual[i] -= alpha * next_applied[i];
        }
        const Real rr_next = vectorNorm2(residual);
        if (!std::isfinite(rr_next)) {
            return false;
        }
        if (std::sqrt(std::max(rr_next, Real{0.0})) <= tolerance) {
            return true;
        }
        const Real beta = rr_next / rr;
        for (std::size_t i = 0; i < n; ++i) {
            direction[i] = residual[i] + beta * direction[i];
        }
        rr = rr_next;
    }
    return false;
}

void smoothCurvatureOnVertexGraph(
    const std::vector<std::vector<GlobalIndex>>& adjacency,
    std::span<const unsigned char> active_vertices,
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
            if (!active_vertices.empty() && active_vertices[vertex] == 0u) {
                next[vertex] = current[vertex];
                continue;
            }
            const auto& neighbors = adjacency[vertex];
            if (neighbors.empty()) {
                next[vertex] = current[vertex];
                continue;
            }

            Real sum = Real{0.0};
            std::size_t count = 0u;
            for (const auto neighbor : neighbors) {
                const auto index = static_cast<std::size_t>(neighbor);
                if (index >= current.size() ||
                    (!active_vertices.empty() &&
                     active_vertices[index] == 0u) ||
                    !std::isfinite(current[index])) {
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

void smoothCurvatureWithMassStiffnessOperator(
    const assembly::IMeshAccess& mesh,
    const std::vector<std::vector<GlobalIndex>>& adjacency,
    std::span<const unsigned char> active_vertices,
    int iterations,
    Real relaxation,
    std::vector<Real>& curvature,
    LevelSetCurvatureProjectionResult& result)
{
    if (iterations <= 0 || !(relaxation > Real{0.0})) {
        return;
    }

    const auto n_vertices = curvature.size();
    std::vector<std::size_t> active;
    active.reserve(n_vertices);
    std::vector<std::size_t> local_index(n_vertices,
                                         std::numeric_limits<std::size_t>::max());
    for (std::size_t vertex = 0; vertex < n_vertices; ++vertex) {
        if ((active_vertices.empty() || active_vertices[vertex] != 0u) &&
            std::isfinite(curvature[vertex])) {
            local_index[vertex] = active.size();
            active.push_back(vertex);
        }
    }
    if (active.size() <= 1u) {
        return;
    }

    std::vector<Real> full_mass(n_vertices, Real{0.0});
    assembleLumpedMass(mesh, active_vertices, full_mass);
    std::vector<std::size_t> degree(n_vertices, 0u);
    for (const auto vertex : active) {
        for (const auto neighbor : adjacency[vertex]) {
            const auto neighbor_index = static_cast<std::size_t>(neighbor);
            if (neighbor >= 0 && neighbor_index < n_vertices &&
                local_index[neighbor_index] !=
                    std::numeric_limits<std::size_t>::max()) {
                ++degree[vertex];
            }
        }
    }

    std::vector<Real> mass(active.size(), Real{1.0});
    for (std::size_t i = 0; i < active.size(); ++i) {
        mass[i] = full_mass[active[i]];
    }

    std::vector<std::vector<WeightedNeighbor>> stiffness(active.size());
    std::vector<Real> stiffness_diag(active.size(), Real{0.0});
    Real edge_length2_sum = Real{0.0};
    std::size_t edge_count = 0u;
    for (const auto vertex : active) {
        const auto row = local_index[vertex];
        const auto x = mesh.getNodeCoordinates(static_cast<GlobalIndex>(vertex));
        for (const auto neighbor : adjacency[vertex]) {
            if (neighbor < 0) {
                continue;
            }
            const auto neighbor_index = static_cast<std::size_t>(neighbor);
            if (neighbor_index <= vertex || neighbor_index >= n_vertices) {
                continue;
            }
            const auto col = local_index[neighbor_index];
            if (col == std::numeric_limits<std::size_t>::max()) {
                continue;
            }
            const auto y =
                mesh.getNodeCoordinates(static_cast<GlobalIndex>(neighbor_index));
            const Real edge_length2 = dot(subtract(y, x), subtract(y, x));
            if (!(edge_length2 > Real{0.0}) ||
                !std::isfinite(edge_length2)) {
                continue;
            }
            const Real vertex_share =
                mass[row] / static_cast<Real>(std::max<std::size_t>(
                                1u, degree[vertex]));
            const Real neighbor_share =
                mass[col] / static_cast<Real>(std::max<std::size_t>(
                                1u, degree[neighbor_index]));
            const Real weight =
                Real{0.5} * (vertex_share + neighbor_share) / edge_length2;
            if (!(weight > Real{0.0}) || !std::isfinite(weight)) {
                continue;
            }
            stiffness[row].push_back(WeightedNeighbor{col, weight});
            stiffness[col].push_back(WeightedNeighbor{row, weight});
            stiffness_diag[row] += weight;
            stiffness_diag[col] += weight;
            edge_length2_sum += edge_length2;
            ++edge_count;
        }
    }
    result.smoothing_operator_edges = edge_count;
    if (edge_count == 0u) {
        return;
    }

    const Real mean_edge_length2 =
        edge_length2_sum / static_cast<Real>(edge_count);
    const Real strength = relaxation * mean_edge_length2;
    if (!(strength > Real{0.0}) || !std::isfinite(strength)) {
        return;
    }

    std::vector<Real> current(active.size(), Real{0.0});
    for (std::size_t i = 0; i < active.size(); ++i) {
        current[i] = curvature[active[i]];
    }

    std::vector<Real> rhs(active.size(), Real{0.0});
    std::vector<Real> next(active.size(), Real{0.0});
    Real total_abs_update = Real{0.0};
    std::size_t update_count = 0u;
    for (int iter = 0; iter < iterations; ++iter) {
        for (std::size_t i = 0; i < active.size(); ++i) {
            rhs[i] = mass[i] * current[i];
        }
        if (!solveMassStiffnessOperatorCG(
                std::span<const Real>(mass.data(), mass.size()),
                stiffness,
                std::span<const Real>(stiffness_diag.data(),
                                      stiffness_diag.size()),
                strength,
                std::span<const Real>(rhs.data(), rhs.size()),
                std::span<const Real>(current.data(), current.size()),
                next)) {
            break;
        }

        Real iteration_max_update = Real{0.0};
        for (std::size_t i = 0; i < active.size(); ++i) {
            if (!std::isfinite(next[i])) {
                next[i] = current[i];
            }
            const Real update = std::abs(next[i] - current[i]);
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
    for (std::size_t i = 0; i < active.size(); ++i) {
        curvature[active[i]] = current[i];
    }
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
        !(options.supplemental_sample_weight > Real{0.0}) ||
        !std::isfinite(options.supplemental_sample_weight) ||
        options.narrow_band_width < Real{0.0} ||
        !std::isfinite(options.narrow_band_width) ||
        options.smoothing_iterations < 0 ||
        options.smoothing_relaxation < Real{0.0} ||
        options.smoothing_relaxation > Real{1.0} ||
        !std::isfinite(options.smoothing_relaxation)) {
        throw std::invalid_argument(
            "level-set curvature projection requires positive tolerances, a nonnegative residual limit, a positive supplemental sample weight, a nonnegative narrow-band width, nonnegative smoothing iterations, and smoothing relaxation in [0,1]");
    }
    for (const auto value : level_set_vertex_values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "level-set curvature projection received a non-finite level-set value");
        }
    }

    LevelSetCurvatureProjectionResult result;
    result.vertices = n_vertices;
    result.supplemental_samples = supplemental_samples.size();
    result.supplemental_sample_weight = options.supplemental_sample_weight;
    result.narrow_band_width = options.narrow_band_width;
    result.smoothing_mode = options.smoothing_mode;
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
    const bool use_narrow_band = options.narrow_band_width > Real{0.0};
    std::vector<unsigned char> active_vertices(n_vertices, 1u);
    if (use_narrow_band) {
        active_vertices.assign(n_vertices, 0u);
        for (GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
            const auto index = static_cast<std::size_t>(vertex);
            const Real distance_to_interface =
                std::abs(level_set_vertex_values[index] - options.isovalue);
            if (distance_to_interface <= options.narrow_band_width ||
                !sample_adjacency[index].empty()) {
                active_vertices[index] = 1u;
            }
        }
    }
    result.narrow_band_vertices =
        static_cast<std::size_t>(std::count(active_vertices.begin(),
                                            active_vertices.end(),
                                            static_cast<unsigned char>(1u)));
    result.skipped_far_vertices = n_vertices - result.narrow_band_vertices;
    std::vector<unsigned char> fitted(n_vertices, 0u);

    for (GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
        if (active_vertices[static_cast<std::size_t>(vertex)] == 0u) {
            continue;
        }
        const auto center = mesh.getNodeCoordinates(vertex);
        const auto neighbors = collectNeighbors(vertex, adjacency, rings);

        std::array<std::array<Real, 9>, 9> ata{};
        std::array<Real, 9> atb{};
        std::vector<FitObservation> observations;
        std::size_t rows = 0u;
        const auto center_value =
            level_set_vertex_values[static_cast<std::size_t>(vertex)];
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
            const Real weight = options.supplemental_sample_weight /
                                std::max(distance2, Real{1.0e-24});
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
        if (active_vertices[index] != 0u && recovered[index] == 0u) {
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
        result.diagnostic = result.narrow_band_vertices == 0u
            ? "level-set curvature projection found no vertices in the requested narrow band"
            : result.fit_residual_failure_vertices > 0u
            ? "level-set curvature projection exceeded the normalized fit residual limit"
            : "level-set curvature projection could not fit any vertex stencil";
        return result;
    }
    if (options.max_neighbor_fallback_vertices >= 0 &&
        result.fallback_vertices >
            static_cast<std::size_t>(options.max_neighbor_fallback_vertices)) {
        result.diagnostic =
            "level-set curvature projection neighbor fallback vertices " +
            std::to_string(result.fallback_vertices) +
            " exceed configured limit " +
            std::to_string(options.max_neighbor_fallback_vertices);
        return result;
    }
    if (options.max_zero_fallback_vertices >= 0 &&
        result.zero_fallback_vertices >
            static_cast<std::size_t>(options.max_zero_fallback_vertices)) {
        result.diagnostic =
            "level-set curvature projection zero fallback vertices " +
            std::to_string(result.zero_fallback_vertices) +
            " exceed configured limit " +
            std::to_string(options.max_zero_fallback_vertices);
        return result;
    }

    const auto fitted_count = static_cast<Real>(result.fitted_vertices);
    result.mean_fit_rms_residual /= fitted_count;
    result.mean_normalized_fit_residual /= fitted_count;

    const auto active_span =
        std::span<const unsigned char>(active_vertices.data(),
                                       active_vertices.size());
    switch (options.smoothing_mode) {
        case LevelSetCurvatureSmoothingMode::LocalGraph:
            smoothCurvatureOnVertexGraph(
                adjacency,
                active_span,
                options.smoothing_iterations,
                options.smoothing_relaxation,
                curvature_vertex_values,
                result);
            break;
        case LevelSetCurvatureSmoothingMode::MassStiffnessOperator:
            smoothCurvatureWithMassStiffnessOperator(
                mesh,
                adjacency,
                active_span,
                options.smoothing_iterations,
                options.smoothing_relaxation,
                curvature_vertex_values,
                result);
            break;
    }

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
