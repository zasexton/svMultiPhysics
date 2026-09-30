/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Application/Core/LevelSetPdeVelocityExtension.h"

#include "FE/Assembly/GlobalSystemView.h"
#include "FE/Backends/Interfaces/BackendFactory.h"
#include "FE/Backends/Interfaces/BackendKind.h"
#include "FE/Backends/Interfaces/GenericMatrix.h"
#include "FE/Backends/Interfaces/GenericVector.h"
#include "FE/Backends/Interfaces/LinearSolver.h"
#include "FE/Backends/Utils/BackendOptions.h"
#include "FE/Sparsity/SparsityPattern.h"
#include "Mesh/Topology/DistributedTopology.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <exception>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>

#ifdef MESH_HAS_MPI
#include <mpi.h>
#endif

namespace application::core {
namespace {

constexpr std::uint8_t kKnownFlag = 1u;
constexpr std::uint8_t kExtensionFlag = 2u;

[[nodiscard]] std::string normalizedMethodToken(std::string_view raw)
{
  std::string token;
  token.reserve(raw.size());
  for (const char character : raw) {
    if (character == '_' || character == '-' ||
        std::isspace(static_cast<unsigned char>(character))) {
      continue;
    }
    token.push_back(static_cast<char>(
        std::tolower(static_cast<unsigned char>(character))));
  }
  return token;
}

[[nodiscard]] std::vector<svmp::gid_t> localVertexGids(const svmp::Mesh& mesh)
{
  const auto n = mesh.n_vertices();
  const auto& gids = mesh.local_mesh().vertex_gids();
  std::vector<svmp::gid_t> out(n);
  for (std::size_t v = 0; v < n; ++v) {
    out[v] = (gids.size() == n && gids[v] != svmp::INVALID_GID)
                 ? gids[v]
                 : static_cast<svmp::gid_t>(v);
  }
  return out;
}

[[nodiscard]] std::vector<svmp::gid_t> localCellGids(const svmp::Mesh& mesh)
{
  const auto& local_mesh = mesh.local_mesh();
  const auto n = static_cast<std::size_t>(local_mesh.n_cells());
  const auto& gids = local_mesh.cell_gids();
  std::vector<svmp::gid_t> out(n);
  for (std::size_t c = 0; c < n; ++c) {
    out[c] = (gids.size() == n && gids[c] != svmp::INVALID_GID)
                 ? gids[c]
                 : static_cast<svmp::gid_t>(c);
  }
  return out;
}

template <typename T>
[[nodiscard]] std::vector<T> allGatherV(const std::vector<T>& local,
                                        const svmp::MeshComm& comm)
{
  if (!comm.is_parallel()) {
    return local;
  }
#ifdef MESH_HAS_MPI
  static_assert(std::is_trivially_copyable_v<T>);
  if (local.size() * sizeof(T) >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(
        "PDE velocity extension: gathered record block exceeds MPI int range");
  }
  const int local_bytes = static_cast<int>(local.size() * sizeof(T));
  std::vector<int> counts(static_cast<std::size_t>(comm.size()), 0);
  MPI_Allgather(&local_bytes, 1, MPI_INT, counts.data(), 1, MPI_INT,
                comm.native());
  std::vector<int> displacements(counts.size(), 0);
  long long total = 0;
  for (std::size_t rank = 0; rank < counts.size(); ++rank) {
    displacements[rank] = static_cast<int>(total);
    total += counts[rank];
    if (total > std::numeric_limits<int>::max()) {
      throw std::runtime_error(
          "PDE velocity extension: gathered record total exceeds MPI int range");
    }
  }
  std::vector<T> gathered(static_cast<std::size_t>(total) / sizeof(T));
  MPI_Allgatherv(local.empty() ? nullptr : local.data(), local_bytes,
                 MPI_BYTE, gathered.empty() ? nullptr : gathered.data(),
                 counts.data(), displacements.data(), MPI_BYTE,
                 comm.native());
  return gathered;
#else
  return local;
#endif
}

// Union of a vertex mask over the communicator (by global vertex ID).
void unionVertexMask(const svmp::Mesh& mesh,
                     const svmp::MeshComm& comm,
                     std::span<const svmp::gid_t> gids,
                     std::vector<std::uint8_t>& mask)
{
  if (!comm.is_parallel()) {
    return;
  }
  std::vector<svmp::gid_t> marked;
  for (std::size_t v = 0; v < mask.size(); ++v) {
    if (mask[v] != 0u) {
      marked.push_back(gids[v]);
    }
  }
  const auto all = allGatherV(marked, comm);
  std::unordered_map<svmp::gid_t, std::size_t> local_by_gid;
  local_by_gid.reserve(gids.size());
  for (std::size_t v = 0; v < gids.size(); ++v) {
    local_by_gid.emplace(gids[v], v);
  }
  for (const auto gid : all) {
    if (const auto it = local_by_gid.find(gid); it != local_by_gid.end()) {
      mask[it->second] = 1u;
    }
  }
  (void)mesh;
}

[[nodiscard]] bool ownsVertex(const svmp::Mesh& mesh,
                              const svmp::MeshComm& comm,
                              std::size_t vertex)
{
  if (!comm.is_parallel()) {
    return true;
  }
  return mesh.owner_rank_vertex(static_cast<svmp::index_t>(vertex)) ==
         comm.rank();
}

[[nodiscard]] double globalMax(double value, const svmp::MeshComm& comm)
{
#ifdef MESH_HAS_MPI
  if (comm.is_parallel()) {
    double global = value;
    MPI_Allreduce(&value, &global, 1, MPI_DOUBLE, MPI_MAX, comm.native());
    return global;
  }
#endif
  (void)comm;
  return value;
}

[[nodiscard]] std::size_t globalSum(std::size_t value,
                                    const svmp::MeshComm& comm)
{
#ifdef MESH_HAS_MPI
  if (comm.is_parallel()) {
    auto local = static_cast<unsigned long long>(value);
    unsigned long long global = local;
    MPI_Allreduce(&local, &global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                  comm.native());
    return static_cast<std::size_t>(global);
  }
#endif
  (void)comm;
  return value;
}

// Gradients of the P1 barycentric functions and the measure of an affine
// simplex.  Returns false for a degenerate cell.
bool simplexGradients(const std::array<std::array<double, 3>, 4>& x,
                      int dim,
                      std::array<std::array<double, 3>, 4>& grad,
                      double& measure)
{
  // Jacobian columns e_k = x_{k+1} - x_0; gradients of lambda_{k+1} are the
  // rows of J^{-1}; lambda_0 = 1 - sum.
  std::array<std::array<double, 3>, 3> J{};
  for (int k = 0; k < dim; ++k) {
    for (int d = 0; d < dim; ++d) {
      J[d][k] = x[k + 1][d] - x[0][d];
    }
  }
  std::array<std::array<double, 3>, 3> inv{};
  double det = 0.0;
  if (dim == 2) {
    det = J[0][0] * J[1][1] - J[0][1] * J[1][0];
    if (!(std::abs(det) > 0.0) || !std::isfinite(det)) {
      return false;
    }
    inv[0][0] = J[1][1] / det;
    inv[0][1] = -J[0][1] / det;
    inv[1][0] = -J[1][0] / det;
    inv[1][1] = J[0][0] / det;
    measure = 0.5 * std::abs(det);
  } else {
    det = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) -
          J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
          J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
    if (!(std::abs(det) > 0.0) || !std::isfinite(det)) {
      return false;
    }
    inv[0][0] = (J[1][1] * J[2][2] - J[1][2] * J[2][1]) / det;
    inv[0][1] = (J[0][2] * J[2][1] - J[0][1] * J[2][2]) / det;
    inv[0][2] = (J[0][1] * J[1][2] - J[0][2] * J[1][1]) / det;
    inv[1][0] = (J[1][2] * J[2][0] - J[1][0] * J[2][2]) / det;
    inv[1][1] = (J[0][0] * J[2][2] - J[0][2] * J[2][0]) / det;
    inv[1][2] = (J[0][2] * J[1][0] - J[0][0] * J[1][2]) / det;
    inv[2][0] = (J[1][0] * J[2][1] - J[1][1] * J[2][0]) / det;
    inv[2][1] = (J[0][1] * J[2][0] - J[0][0] * J[2][1]) / det;
    inv[2][2] = (J[0][0] * J[1][1] - J[0][1] * J[1][0]) / det;
    measure = std::abs(det) / 6.0;
  }
  for (auto& g : grad) {
    g = {0.0, 0.0, 0.0};
  }
  for (int k = 0; k < dim; ++k) {
    for (int d = 0; d < dim; ++d) {
      grad[k + 1][d] = inv[k][d];
      grad[0][d] -= inv[k][d];
    }
  }
  return true;
}

// One gathered cell: global cell ID, vertex count, vertex GIDs, element matrix.
struct CellRecord {
  std::int64_t cell_gid{0};
  std::int64_t count{0};
  std::array<std::int64_t, 4> vertex_gid{0, 0, 0, 0};
  std::array<double, 16> matrix{};
};

// One gathered vertex (owned copy only).
struct VertexRecord {
  std::int64_t gid{0};
  std::uint8_t flags{0u};
  std::array<std::uint8_t, 3> wall{0u, 0u, 0u};
  std::array<double, 3> value{0.0, 0.0, 0.0};
};

} // namespace

std::string_view pdeVelocityExtensionOperatorName(
    PdeVelocityExtensionOperator op) noexcept
{
  switch (op) {
  case PdeVelocityExtensionOperator::LeastSquaresNormal:
    return "pde_normal";
  case PdeVelocityExtensionOperator::Harmonic:
    return "pde_harmonic";
  }
  return "unknown";
}

std::optional<PdeVelocityExtensionOperator>
pdeVelocityExtensionOperatorFromToken(std::string_view token)
{
  const auto normalized = normalizedMethodToken(token);
  if (normalized == "pdenormal" || normalized == "pdeleastsquaresnormal") {
    return PdeVelocityExtensionOperator::LeastSquaresNormal;
  }
  if (normalized == "pdeharmonic") {
    return PdeVelocityExtensionOperator::Harmonic;
  }
  return std::nullopt;
}

PdeVelocityExtensionReport extendVelocityByPde(
    const svmp::Mesh& mesh,
    const svmp::MeshComm& comm,
    std::span<const double> level_set,
    std::span<const double> source,
    std::size_t source_components,
    std::span<const std::uint8_t> known,
    std::size_t target_components,
    std::span<const WallVelocityExtensionConstraint> walls,
    const PdeVelocityExtensionOptions& options,
    std::vector<double>& extended,
    std::vector<svmp::FE::level_set::VelocityExtensionConstraintRow>* rows)
{
  const auto n_vertices = mesh.n_vertices();
  const int dim = mesh.dim();
  if (dim != 2 && dim != 3) {
    throw std::invalid_argument(
        "PDE velocity extension requires a 2D or 3D mesh");
  }
  if (level_set.size() != n_vertices || known.size() != n_vertices ||
      source_components == 0u || target_components == 0u ||
      target_components > 3u ||
      source.size() < n_vertices * source_components) {
    throw std::invalid_argument(
        "PDE velocity extension received incompatible input sizes");
  }
  const auto copy_components = std::min(source_components, target_components);
  if (options.enforce_wall_impermeability && walls.empty()) {
    throw std::invalid_argument(
        "PDE velocity extension wall impermeability requires at least one "
        "strong zero-velocity wall mask");
  }

  const auto& local_mesh = mesh.local_mesh();
  const auto& coords = mesh.X_ref();
  const auto vertex_gids = localVertexGids(mesh);
  const auto cell_gids = localCellGids(mesh);
  const auto n_cells = static_cast<std::size_t>(local_mesh.n_cells());

  // ---- extension domain --------------------------------------------------
  // Make the known set communicator-consistent: a vertex is known if any
  // rank marks it (a ghost may not see every cut cell of its star).
  std::vector<std::uint8_t> known_mask(known.begin(), known.end());
  unionVertexMask(mesh, comm, vertex_gids, known_mask);
  std::vector<std::uint8_t> domain(n_vertices, 0u);
  if (options.band_layers <= 0) {
    for (std::size_t v = 0; v < n_vertices; ++v) {
      domain[v] = known_mask[v] == 0u ? 1u : 0u;
    }
  } else {
    std::vector<std::uint8_t> reached(known_mask);
    for (int layer = 0; layer < options.band_layers; ++layer) {
      std::vector<std::uint8_t> next(reached);
      for (std::size_t c = 0; c < n_cells; ++c) {
        auto [cv, count] =
            local_mesh.cell_vertices_span(static_cast<svmp::index_t>(c));
        bool touches = false;
        for (std::size_t i = 0; i < count; ++i) {
          touches = touches || reached[static_cast<std::size_t>(cv[i])] != 0u;
        }
        if (!touches) {
          continue;
        }
        for (std::size_t i = 0; i < count; ++i) {
          next[static_cast<std::size_t>(cv[i])] = 1u;
        }
      }
      unionVertexMask(mesh, comm, vertex_gids, next);
      reached.swap(next);
    }
    for (std::size_t v = 0; v < n_vertices; ++v) {
      domain[v] = (reached[v] != 0u && known_mask[v] == 0u) ? 1u : 0u;
    }
  }

  // ---- wall component masks ------------------------------------------------
  std::array<std::vector<std::uint8_t>, 3> wall_mask{
      std::vector<std::uint8_t>(n_vertices, 0u),
      std::vector<std::uint8_t>(n_vertices, 0u),
      std::vector<std::uint8_t>(n_vertices, 0u)};
  if (options.enforce_wall_impermeability) {
    auto boundary_faces = svmp::DistributedTopology::global_boundary_faces(
        mesh, /*owned_only=*/false);
    if (boundary_faces.empty()) {
      boundary_faces = local_mesh.boundary_faces();
    }
    for (const auto& wall : walls) {
      if (wall.project_boundary_normal) {
        throw std::invalid_argument(
            "PDE velocity extension supports only axis-aligned strong "
            "zero-velocity component masks, not boundary-normal projection");
      }
      bool any = false;
      for (int d = 0; d < dim; ++d) {
        any = any || wall.constrained_components[static_cast<std::size_t>(d)];
      }
      if (!any) {
        throw std::invalid_argument(
            "PDE velocity extension received an empty wall component mask");
      }
    }
    for (const auto face : boundary_faces) {
      if (face < 0 || face >= local_mesh.n_faces()) {
        continue;
      }
      const auto label = local_mesh.boundary_label(face);
      const auto raw_normal = local_mesh.face_normal(face);
      const double norm = std::sqrt(raw_normal[0] * raw_normal[0] +
                                    raw_normal[1] * raw_normal[1] +
                                    raw_normal[2] * raw_normal[2]);
      for (const auto& wall : walls) {
        if (wall.boundary_label != svmp::INVALID_LABEL &&
            wall.boundary_label != label) {
          continue;
        }
        // A mask is impermeable only if it constrains the wall normal.
        if (norm > 0.0 && std::isfinite(norm)) {
          double unconstrained2 = 0.0;
          for (int d = 0; d < dim; ++d) {
            if (!wall.constrained_components[static_cast<std::size_t>(d)]) {
              const double nd = raw_normal[static_cast<std::size_t>(d)] / norm;
              unconstrained2 += nd * nd;
            }
          }
          if (unconstrained2 > 1.0e-20) {
            throw std::runtime_error(
                "PDE velocity extension received a strong zero-velocity "
                "component mask that does not constrain the wall-normal "
                "direction");
          }
        }
        auto [fv, count] = local_mesh.face_vertices_span(face);
        for (std::size_t i = 0; fv != nullptr && i < count; ++i) {
          if (fv[i] < 0 || static_cast<std::size_t>(fv[i]) >= n_vertices) {
            continue;
          }
          for (int d = 0; d < dim; ++d) {
            if (wall.constrained_components[static_cast<std::size_t>(d)]) {
              wall_mask[static_cast<std::size_t>(d)]
                       [static_cast<std::size_t>(fv[i])] = 1u;
            }
          }
        }
      }
    }
    for (int d = 0; d < dim; ++d) {
      unionVertexMask(mesh, comm, vertex_gids,
                      wall_mask[static_cast<std::size_t>(d)]);
    }
  }

  // ---- local records -------------------------------------------------------
  std::vector<CellRecord> local_cells;
  for (std::size_t c = 0; c < n_cells; ++c) {
    const auto cell = static_cast<svmp::index_t>(c);
    if (!mesh.is_owned_cell(cell)) {
      continue;
    }
    auto [cv, count] = local_mesh.cell_vertices_span(cell);
    if (cv == nullptr || count == 0u) {
      continue;
    }
    bool has_domain = false;
    bool inside = true;
    for (std::size_t i = 0; i < count; ++i) {
      const auto v = static_cast<std::size_t>(cv[i]);
      has_domain = has_domain || domain[v] != 0u;
      inside = inside && (domain[v] != 0u || known_mask[v] != 0u);
    }
    if (!has_domain || !inside) {
      continue;
    }
    const auto shape = local_mesh.cell_shape(cell);
    const bool simplex =
        shape.order == 1 &&
        ((dim == 2 && shape.family == svmp::CellFamily::Triangle &&
          count == 3u) ||
         (dim == 3 && shape.family == svmp::CellFamily::Tetra &&
          count == 4u));
    if (!simplex) {
      throw std::runtime_error(
          "PDE velocity extension requires affine Triangle3 (2D) or Tetra4 "
          "(3D) cells in the dry region");
    }
    std::array<std::array<double, 3>, 4> x{};
    for (std::size_t i = 0; i < count; ++i) {
      const auto v = static_cast<std::size_t>(cv[i]);
      for (int d = 0; d < dim; ++d) {
        x[i][static_cast<std::size_t>(d)] = static_cast<double>(
            coords[v * static_cast<std::size_t>(dim) +
                   static_cast<std::size_t>(d)]);
      }
    }
    std::array<std::array<double, 3>, 4> grad{};
    double measure = 0.0;
    if (!simplexGradients(x, dim, grad, measure)) {
      throw std::runtime_error(
          "PDE velocity extension found a degenerate cell");
    }
    CellRecord record;
    record.cell_gid = static_cast<std::int64_t>(cell_gids[c]);
    record.count = static_cast<std::int64_t>(count);
    for (std::size_t i = 0; i < count; ++i) {
      record.vertex_gid[i] = static_cast<std::int64_t>(
          vertex_gids[static_cast<std::size_t>(cv[i])]);
    }
    if (options.op == PdeVelocityExtensionOperator::Harmonic) {
      for (std::size_t i = 0; i < count; ++i) {
        for (std::size_t j = 0; j < count; ++j) {
          double dot = 0.0;
          for (int d = 0; d < dim; ++d) {
            dot += grad[i][static_cast<std::size_t>(d)] *
                   grad[j][static_cast<std::size_t>(d)];
          }
          record.matrix[i * count + j] = measure * dot;
        }
      }
    } else {
      std::array<double, 3> gphi{0.0, 0.0, 0.0};
      for (std::size_t i = 0; i < count; ++i) {
        const double value = level_set[static_cast<std::size_t>(cv[i])];
        if (!std::isfinite(value)) {
          throw std::runtime_error(
              "PDE velocity extension found a non-finite level set");
        }
        for (int d = 0; d < dim; ++d) {
          gphi[static_cast<std::size_t>(d)] +=
              value * grad[i][static_cast<std::size_t>(d)];
        }
      }
      const double gnorm = std::sqrt(gphi[0] * gphi[0] + gphi[1] * gphi[1] +
                                     gphi[2] * gphi[2]);
      if (!(gnorm > 0.0) || !std::isfinite(gnorm)) {
        throw std::runtime_error(
            "PDE least-squares normal extension found a dry cell with a zero "
            "level-set gradient; the normal is undefined");
      }
      std::array<double, 4> ndg{0.0, 0.0, 0.0, 0.0};
      for (std::size_t i = 0; i < count; ++i) {
        for (int d = 0; d < dim; ++d) {
          ndg[i] += gphi[static_cast<std::size_t>(d)] / gnorm *
                    grad[i][static_cast<std::size_t>(d)];
        }
      }
      for (std::size_t i = 0; i < count; ++i) {
        for (std::size_t j = 0; j < count; ++j) {
          record.matrix[i * count + j] = measure * ndg[i] * ndg[j];
        }
      }
    }
    local_cells.push_back(record);
  }

  std::vector<VertexRecord> local_vertices;
  double max_known_speed = 0.0;
  for (std::size_t v = 0; v < n_vertices; ++v) {
    if (known_mask[v] != 0u) {
      double speed2 = 0.0;
      for (std::size_t c = 0; c < copy_components; ++c) {
        const double value = source[v * source_components + c];
        speed2 += value * value;
      }
      max_known_speed = std::max(max_known_speed, std::sqrt(speed2));
    }
    if ((known_mask[v] == 0u && domain[v] == 0u) || !ownsVertex(mesh, comm, v)) {
      continue;
    }
    VertexRecord record;
    record.gid = static_cast<std::int64_t>(vertex_gids[v]);
    record.flags = static_cast<std::uint8_t>(
        (known_mask[v] != 0u ? kKnownFlag : 0u) |
        (domain[v] != 0u ? kExtensionFlag : 0u));
    for (std::size_t c = 0; c < 3u; ++c) {
      record.wall[c] = wall_mask[c][v];
      if (c < copy_components) {
        const double value = source[v * source_components + c];
        if (known_mask[v] != 0u && !std::isfinite(value)) {
          throw std::runtime_error(
              "PDE velocity extension found a non-finite known velocity");
        }
        record.value[c] = std::isfinite(value) ? value : 0.0;
      }
    }
    local_vertices.push_back(record);
  }
  max_known_speed = globalMax(max_known_speed, comm);

  // ---- replicated global system -------------------------------------------
  auto cells = allGatherV(local_cells, comm);
  auto vertices = allGatherV(local_vertices, comm);
  std::sort(cells.begin(), cells.end(),
            [](const CellRecord& a, const CellRecord& b) {
              return a.cell_gid < b.cell_gid;
            });
  std::sort(vertices.begin(), vertices.end(),
            [](const VertexRecord& a, const VertexRecord& b) {
              return a.gid < b.gid;
            });
  std::unordered_map<std::int64_t, std::size_t> vertex_index;
  vertex_index.reserve(vertices.size());
  for (std::size_t i = 0; i < vertices.size(); ++i) {
    if (!vertex_index.emplace(vertices[i].gid, i).second) {
      throw std::runtime_error(
          "PDE velocity extension gathered duplicate vertex records");
    }
  }
  for (const auto& cell : cells) {
    for (std::int64_t i = 0; i < cell.count; ++i) {
      if (vertex_index.find(cell.vertex_gid[static_cast<std::size_t>(i)]) ==
          vertex_index.end()) {
        throw std::runtime_error(
            "PDE velocity extension found a cell vertex without an owned "
            "vertex record");
      }
    }
  }

  PdeVelocityExtensionReport report;
  report.extension_cells = cells.size();
  for (const auto& vertex : vertices) {
    report.known_vertices += (vertex.flags & kKnownFlag) != 0u ? 1u : 0u;
    report.extension_vertices +=
        (vertex.flags & kExtensionFlag) != 0u ? 1u : 0u;
  }
  report.max_known_speed = max_known_speed;

  // solution[i * 3 + c] for gathered vertex i.
  std::vector<double> solution(vertices.size() * 3u, 0.0);
  for (std::size_t i = 0; i < vertices.size(); ++i) {
    if ((vertices[i].flags & kKnownFlag) != 0u) {
      for (std::size_t c = 0; c < 3u; ++c) {
        solution[i * 3u + c] = vertices[i].value[c];
      }
    }
  }

  const auto factory = svmp::FE::backends::BackendFactory::create(
      svmp::FE::backends::BackendKind::Eigen);
  if (!factory) {
    throw std::runtime_error(
        "PDE velocity extension requires the FE Eigen backend");
  }
  svmp::FE::backends::SolverOptions solver_options;
  solver_options.method = svmp::FE::backends::SolverMethod::Direct;
  solver_options.rel_tol = 1.0e-10;

  for (std::size_t component = 0; component < copy_components; ++component) {
    std::vector<svmp::FE::GlobalIndex> unknown(vertices.size(), -1);
    svmp::FE::GlobalIndex n_unknown = 0;
    for (std::size_t i = 0; i < vertices.size(); ++i) {
      if ((vertices[i].flags & kExtensionFlag) == 0u) {
        continue;
      }
      if (vertices[i].wall[component] != 0u) {
        ++report.wall_fixed[component];
        solution[i * 3u + component] = 0.0;
        continue;
      }
      unknown[i] = n_unknown++;
    }
    report.unknowns[component] = static_cast<std::size_t>(n_unknown);
    if (n_unknown == 0) {
      continue;
    }

    svmp::FE::sparsity::SparsityPattern pattern(n_unknown, n_unknown);
    for (const auto& cell : cells) {
      const auto count = static_cast<std::size_t>(cell.count);
      for (std::size_t a = 0; a < count; ++a) {
        const auto ia = unknown[vertex_index.at(cell.vertex_gid[a])];
        if (ia < 0) {
          continue;
        }
        for (std::size_t b = 0; b < count; ++b) {
          const auto ib = unknown[vertex_index.at(cell.vertex_gid[b])];
          if (ib >= 0) {
            pattern.addEntry(ia, ib);
          }
        }
      }
    }
    for (svmp::FE::GlobalIndex i = 0; i < n_unknown; ++i) {
      pattern.addEntry(i, i);
    }
    pattern.finalize();

    auto A = factory->createMatrix(pattern);
    auto b = factory->createVector(n_unknown);
    auto x = factory->createVector(n_unknown);
    if (!A || !b || !x) {
      throw std::runtime_error(
          "PDE velocity extension could not create backend objects");
    }
    A->zero();
    b->zero();
    x->zero();
    std::vector<double> rhs(static_cast<std::size_t>(n_unknown), 0.0);
    {
      auto view = A->createAssemblyView();
      view->beginAssemblyPhase();
      for (const auto& cell : cells) {
        const auto count = static_cast<std::size_t>(cell.count);
        std::array<svmp::FE::GlobalIndex, 4> dofs{};
        std::array<std::size_t, 4> local{};
        std::size_t n_local = 0u;
        for (std::size_t a = 0; a < count; ++a) {
          const auto i = vertex_index.at(cell.vertex_gid[a]);
          const auto ia = unknown[i];
          if (ia >= 0) {
            dofs[n_local] = ia;
            local[n_local] = a;
            ++n_local;
            for (std::size_t bb = 0; bb < count; ++bb) {
              const auto j = vertex_index.at(cell.vertex_gid[bb]);
              if (unknown[j] < 0) {
                rhs[static_cast<std::size_t>(ia)] -=
                    cell.matrix[a * count + bb] * solution[j * 3u + component];
              }
            }
          }
        }
        if (n_local == 0u) {
          continue;
        }
        std::array<double, 16> block{};
        for (std::size_t p = 0; p < n_local; ++p) {
          for (std::size_t q = 0; q < n_local; ++q) {
            block[p * n_local + q] = cell.matrix[local[p] * count + local[q]];
          }
        }
        view->addMatrixEntries(
            std::span<const svmp::FE::GlobalIndex>(dofs.data(), n_local),
            std::span<const double>(block.data(), n_local * n_local),
            svmp::FE::assembly::AddMode::Add);
      }
      view->endAssemblyPhase();
      view->finalizeAssembly();
    }
    A->finalizeAssembly();
    {
      auto span = b->localSpan();
      std::copy(rhs.begin(), rhs.end(), span.begin());
    }

    auto solver = factory->createLinearSolver(solver_options);
    svmp::FE::backends::SolverReport solve_report;
    try {
      solve_report = solver->solve(*A, *x, *b);
    } catch (const std::exception& error) {
      throw std::runtime_error(
          std::string("PDE velocity extension (") +
          std::string(pdeVelocityExtensionOperatorName(options.op)) +
          ") could not solve its dry-region system; the operator is "
          "singular for this level set and known set: " + error.what());
    }
    auto r = factory->createVector(n_unknown);
    A->mult(*x, *r);
    double residual2 = 0.0;
    double rhs2 = 0.0;
    {
      const auto ax = r->localSpan();
      for (std::size_t i = 0; i < rhs.size(); ++i) {
        const double diff = rhs[i] - ax[i];
        residual2 += diff * diff;
        rhs2 += rhs[i] * rhs[i];
      }
    }
    const double relative_residual =
        rhs2 > 0.0 ? std::sqrt(residual2 / rhs2) : std::sqrt(residual2);
    report.max_relative_residual =
        std::max(report.max_relative_residual, relative_residual);
    if (!solve_report.converged || !std::isfinite(relative_residual) ||
        relative_residual > 1.0e-8) {
      throw std::runtime_error(
          std::string("PDE velocity extension (") +
          std::string(pdeVelocityExtensionOperatorName(options.op)) +
          ") failed its dry-region solve (relative residual " +
          std::to_string(relative_residual) + ")");
    }
    const auto xs = x->localSpan();
    for (std::size_t i = 0; i < vertices.size(); ++i) {
      if (unknown[i] >= 0) {
        solution[i * 3u + component] = xs[static_cast<std::size_t>(unknown[i])];
      }
    }
  }

  // ---- owner-local algebraic rows (monolithic coupling) ---------------------
  if (rows != nullptr) {
    rows->clear();
    std::unordered_map<std::int64_t, std::size_t> local_by_gid;
    local_by_gid.reserve(n_vertices);
    for (std::size_t v = 0; v < n_vertices; ++v) {
      local_by_gid.emplace(static_cast<std::int64_t>(vertex_gids[v]), v);
    }
    // Operator row of every owned dry vertex, accumulated in the canonical
    // (sorted cell) order used for the solve.
    std::unordered_map<std::int64_t, std::map<std::int64_t, double>> row_entries;
    for (std::size_t v = 0; v < n_vertices; ++v) {
      if (domain[v] != 0u && ownsVertex(mesh, comm, v)) {
        row_entries.emplace(static_cast<std::int64_t>(vertex_gids[v]),
                            std::map<std::int64_t, double>{});
      }
    }
    for (const auto& cell : cells) {
      const auto count = static_cast<std::size_t>(cell.count);
      for (std::size_t a = 0; a < count; ++a) {
        const auto found = row_entries.find(cell.vertex_gid[a]);
        if (found == row_entries.end()) {
          continue;
        }
        for (std::size_t b = 0; b < count; ++b) {
          found->second[cell.vertex_gid[b]] += cell.matrix[a * count + b];
        }
      }
    }
    for (std::size_t v = 0; v < n_vertices; ++v) {
      if (!ownsVertex(mesh, comm, v)) {
        continue;
      }
      for (std::size_t c = 0; c < target_components; ++c) {
        svmp::FE::level_set::VelocityExtensionConstraintRow row;
        row.vertex = static_cast<svmp::FE::GlobalIndex>(v);
        row.component = static_cast<int>(c);
        if (known_mask[v] != 0u) {
          if (c < copy_components) {
            row.dependencies.push_back(
                svmp::FE::level_set::VelocityExtensionDependency{
                    .field = svmp::FE::level_set::
                        VelocityExtensionDependencyField::SourceVelocity,
                    .vertex = static_cast<svmp::FE::GlobalIndex>(v),
                    .component = static_cast<int>(c),
                    .coefficient = 1.0});
          }
        } else if (domain[v] != 0u && c < copy_components &&
                   wall_mask[c][v] == 0u) {
          const auto gid = static_cast<std::int64_t>(vertex_gids[v]);
          const auto& entries = row_entries.at(gid);
          const auto diagonal = entries.find(gid);
          if (diagonal == entries.end() || !(diagonal->second > 0.0)) {
            throw std::runtime_error(
                std::string("PDE velocity extension (") +
                std::string(pdeVelocityExtensionOperatorName(options.op)) +
                ") found a dry vertex with a non-positive operator diagonal; "
                "its algebraic row is undefined");
          }
          for (const auto& [neighbor_gid, value] : entries) {
            if (neighbor_gid == gid) {
              continue;
            }
            const auto local = local_by_gid.find(neighbor_gid);
            if (local == local_by_gid.end()) {
              throw std::runtime_error(
                  "PDE velocity extension row depends on a vertex that is not "
                  "present on its owner rank (ghost layer too thin)");
            }
            row.dependencies.push_back(
                svmp::FE::level_set::VelocityExtensionDependency{
                    .field = svmp::FE::level_set::
                        VelocityExtensionDependencyField::ExtensionVelocity,
                    .vertex = static_cast<svmp::FE::GlobalIndex>(local->second),
                    .component = static_cast<int>(c),
                    .coefficient = -value / diagonal->second});
          }
        }
        rows->push_back(std::move(row));
      }
    }
  }

  // ---- scatter back ----------------------------------------------------------
  extended.assign(n_vertices * target_components, 0.0);
  double max_extended_speed = 0.0;
  double max_wall_normal = 0.0;
  std::size_t outside = 0u;
  for (std::size_t v = 0; v < n_vertices; ++v) {
    if (known_mask[v] != 0u) {
      for (std::size_t c = 0; c < copy_components; ++c) {
        extended[v * target_components + c] = source[v * source_components + c];
      }
      continue;
    }
    if (domain[v] == 0u) {
      outside += ownsVertex(mesh, comm, v) ? 1u : 0u;
      continue;
    }
    const auto found =
        vertex_index.find(static_cast<std::int64_t>(vertex_gids[v]));
    if (found == vertex_index.end()) {
      throw std::runtime_error(
          "PDE velocity extension lost a local dry vertex in the gathered "
          "system");
    }
    double speed2 = 0.0;
    for (std::size_t c = 0; c < copy_components; ++c) {
      const double value = solution[found->second * 3u + c];
      if (!std::isfinite(value)) {
        throw std::runtime_error(
            "PDE velocity extension produced a non-finite velocity");
      }
      extended[v * target_components + c] = value;
      speed2 += value * value;
      if (wall_mask[c][v] != 0u) {
        max_wall_normal = std::max(max_wall_normal, std::abs(value));
      }
    }
    max_extended_speed = std::max(max_extended_speed, std::sqrt(speed2));
  }
  report.outside_vertices = globalSum(outside, comm);
  report.max_extended_speed = globalMax(max_extended_speed, comm);
  report.max_wall_normal_velocity = globalMax(max_wall_normal, comm);

  if (report.max_extended_speed >
      kVelocityExtensionMaxWetToDryAmplification *
          std::max(report.max_known_speed, 1.0e-12)) {
    throw std::runtime_error(
        "PDE velocity extension amplified the known speed beyond the fixed "
        "wet-to-dry guard (max_known_speed=" +
        std::to_string(report.max_known_speed) + ", max_extended_speed=" +
        std::to_string(report.max_extended_speed) + ")");
  }
  return report;
}

} // namespace application::core
