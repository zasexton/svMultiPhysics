#include "Application/Core/LevelSetVelocityExtensionMap.h"

#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellTopology.h"
#include "Mesh/Topology/DistributedTopology.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#ifdef MESH_HAS_MPI
#include <mpi.h>
#endif

namespace application::core {
namespace {

template <typename CoordinateContainer>
std::array<double, 3> meshVertexPoint(const CoordinateContainer& coordinates,
                                      int dimension,
                                      std::size_t vertex)
{
  std::array<double, 3> point{0.0, 0.0, 0.0};
  for (int component = 0; component < dimension; ++component) {
    point[static_cast<std::size_t>(component)] =
        static_cast<double>(
            coordinates[vertex * static_cast<std::size_t>(dimension) +
                        static_cast<std::size_t>(component)]);
  }
  return point;
}

constexpr std::uint64_t kMapHashOffset = 1469598103934665603ull;
constexpr std::uint64_t kMapHashPrime = 1099511628211ull;

void hashMapWord(std::uint64_t& hash, std::uint64_t word) noexcept
{
  for (int byte = 0; byte < 8; ++byte) {
    hash ^= (word >> (8 * byte)) & 0xffu;
    hash *= kMapHashPrime;
  }
}

std::uint64_t hashLevelSetValues(std::span<const double> values) noexcept
{
  std::uint64_t hash = kMapHashOffset;
  hashMapWord(hash, static_cast<std::uint64_t>(values.size()));
  for (const auto value : values) {
    hashMapWord(hash, std::bit_cast<std::uint64_t>(value));
  }
  return hash;
}

std::uint64_t hashActiveSet(std::span<const std::uint8_t> values) noexcept
{
  std::uint64_t hash = kMapHashOffset;
  hashMapWord(hash, static_cast<std::uint64_t>(values.size()));
  for (const auto value : values) {
    hash ^= static_cast<std::uint64_t>(value);
    hash *= kMapHashPrime;
  }
  return hash;
}

} // namespace

std::uint64_t VelocityExtensionMapRevision::key() const noexcept
{
  std::uint64_t hash = kMapHashOffset;
  hashMapWord(hash, mesh_geometry);
  hashMapWord(hash, mesh_topology);
  hashMapWord(hash, mesh_ownership);
  hashMapWord(hash, mesh_numbering);
  hashMapWord(hash, free_surface_geometry);
  hashMapWord(hash, level_set_values);
  hashMapWord(hash, active_set);
  return hash == 0u ? 1u : hash;
}

std::string_view velocityExtensionRowDispositionName(
    VelocityExtensionRowDisposition disposition) noexcept
{
  switch (disposition) {
  case VelocityExtensionRowDisposition::TraceSeed:
    return "trace_seed";
  case VelocityExtensionRowDisposition::Regression:
    return "regression";
  case VelocityExtensionRowDisposition::BoundedFallback:
    return "bounded_fallback";
  case VelocityExtensionRowDisposition::OutsideBandZero:
    return "outside_band_zero";
  }
  return "unknown";
}

bool VelocityExtensionMapRevision::complete() const noexcept
{
  return level_set_values != 0u && active_set != 0u && key() != 0u;
}

VelocityExtensionMapRevision velocityExtensionMapRevision(
    std::uint64_t mesh_geometry,
    std::uint64_t mesh_topology,
    std::uint64_t mesh_ownership,
    std::uint64_t mesh_numbering,
    std::uint64_t free_surface_geometry,
    std::span<const double> level_set_values,
    std::span<const std::uint8_t> active_set)
{
  if (level_set_values.empty() ||
      active_set.size() != level_set_values.size()) {
    throw std::invalid_argument(
        "velocity-extension map revision requires matching nonempty level-set and active-set arrays");
  }
  for (std::size_t vertex = 0; vertex < level_set_values.size(); ++vertex) {
    if (!std::isfinite(level_set_values[vertex]) || active_set[vertex] > 1u) {
      throw std::invalid_argument(
          "velocity-extension map revision received invalid level-set or active-set data");
    }
  }
  return VelocityExtensionMapRevision{
      .mesh_geometry = mesh_geometry,
      .mesh_topology = mesh_topology,
      .mesh_ownership = mesh_ownership,
      .mesh_numbering = mesh_numbering,
      .free_surface_geometry = free_surface_geometry,
      .level_set_values = hashLevelSetValues(level_set_values),
      .active_set = hashActiveSet(active_set),
  };
}

VelocityExtensionMapSnapshot::VelocityExtensionMapSnapshot(
    VelocityExtensionMapRevision revision,
    std::size_t components,
    std::vector<double> preview,
    std::vector<svmp::FE::level_set::VelocityExtensionConstraintRow> rows,
    std::vector<std::int64_t> component_assignment,
    std::vector<VelocityExtensionGraphRowDiagnostic> row_diagnostics,
    WallCompatibleVelocityExtensionResult report,
    double wet_to_dry_amplification)
    : revision_(revision),
      components_(components),
      preview_(std::move(preview)),
      rows_(std::move(rows)),
      component_assignment_(std::move(component_assignment)),
      row_diagnostics_(std::move(row_diagnostics)),
      report_(report),
      wet_to_dry_amplification_(wet_to_dry_amplification)
{
  if (!revision_.complete() || components_ == 0u || preview_.empty() ||
      preview_.size() % components_ != 0u ||
      component_assignment_.size() != preview_.size() / components_ ||
      rows_.empty() || row_diagnostics_.empty() ||
      rows_.size() != row_diagnostics_.size() * components_ ||
      !std::isfinite(wet_to_dry_amplification_) ||
      wet_to_dry_amplification_ < 0.0) {
    throw std::invalid_argument(
        "velocity-extension map snapshot received incomplete or incompatible data");
  }
  std::set<svmp::FE::GlobalIndex> owner_vertices;
  std::size_t extended_rows = 0u;
  std::size_t outside_rows = 0u;
  std::size_t collision_rows = 0u;
  std::size_t regression_rows = 0u;
  std::size_t accepted_regression_rows = 0u;
  std::size_t fallback_rows = 0u;
  std::size_t condition_rejections = 0u;
  std::size_t coefficient_rejections = 0u;
  std::size_t wall_projected_rows = 0u;
  for (const auto& diagnostic : row_diagnostics_) {
    if (diagnostic.local_vertex == svmp::FE::INVALID_GLOBAL_INDEX ||
        diagnostic.local_vertex < 0 ||
        static_cast<std::size_t>(diagnostic.local_vertex) >=
            component_assignment_.size() ||
        diagnostic.global_vertex == svmp::INVALID_GID ||
        !owner_vertices.insert(diagnostic.local_vertex).second ||
        diagnostic.component_assignment !=
            component_assignment_[static_cast<std::size_t>(
                diagnostic.local_vertex)] ||
        diagnostic.band_layer < 0 ||
        diagnostic.reconstruction_dimension < 0 ||
        diagnostic.numerical_rank < 0 ||
        diagnostic.numerical_rank > diagnostic.reconstruction_dimension ||
        !std::isfinite(diagnostic.proposed_coefficient_sum) ||
        !std::isfinite(diagnostic.proposed_coefficient_l1) ||
        !std::isfinite(diagnostic.proposed_max_abs_coefficient) ||
        !std::isfinite(diagnostic.proposed_max_negative_coefficient) ||
        !std::isfinite(diagnostic.coefficient_sum) ||
        !std::isfinite(diagnostic.coefficient_l1) ||
        !std::isfinite(diagnostic.max_abs_coefficient) ||
        !std::isfinite(diagnostic.max_negative_coefficient) ||
        !std::isfinite(diagnostic.constant_reproduction_error) ||
        !std::isfinite(
            diagnostic.max_tangential_linear_reproduction_error) ||
        !std::isfinite(diagnostic.extrapolation_distance) ||
        diagnostic.extrapolation_distance < 0.0 ||
        !std::isfinite(diagnostic.dependency_max_speed) ||
        diagnostic.dependency_max_speed < 0.0 ||
        !std::isfinite(diagnostic.preview_speed) ||
        diagnostic.preview_speed < 0.0 ||
        !std::isfinite(diagnostic.preview_amplification) ||
        diagnostic.preview_amplification < 0.0 ||
        diagnostic.preview_amplification >
            1.0 + kVelocityExtensionRowTolerance ||
        (!std::isfinite(diagnostic.condition_estimate) &&
         !diagnostic.condition_rejected)) {
      throw std::invalid_argument(
          "velocity-extension map snapshot received an invalid owner row diagnostic");
    }
    if (diagnostic.assigned) {
      if (std::abs(diagnostic.coefficient_sum - 1.0) >
              kVelocityExtensionRowTolerance ||
          diagnostic.coefficient_l1 >
              1.0 + kVelocityExtensionRowTolerance ||
          diagnostic.max_abs_coefficient >
              1.0 + kVelocityExtensionRowTolerance ||
          diagnostic.negative_weight_count != 0u ||
          diagnostic.max_negative_coefficient >
              kVelocityExtensionCoefficientTolerance ||
          diagnostic.dependencies.empty()) {
        throw std::invalid_argument(
            "velocity-extension map snapshot received an unbounded assigned graph row");
      }
    } else if (diagnostic.disposition !=
                   VelocityExtensionRowDisposition::OutsideBandZero ||
               !diagnostic.dependencies.empty()) {
      throw std::invalid_argument(
          "velocity-extension map snapshot received an invalid outside-band graph row");
    }
    for (const auto& dependency : diagnostic.dependencies) {
      if (dependency.local_vertex == svmp::FE::INVALID_GLOBAL_INDEX ||
          dependency.global_vertex == svmp::INVALID_GID ||
          !std::isfinite(dependency.coefficient)) {
        throw std::invalid_argument(
            "velocity-extension map snapshot received an invalid graph dependency");
      }
    }
    const bool reconstructed =
        diagnostic.disposition == VelocityExtensionRowDisposition::Regression ||
        diagnostic.disposition ==
            VelocityExtensionRowDisposition::BoundedFallback;
    if (reconstructed != diagnostic.regression_attempted ||
        diagnostic.regression_accepted !=
            (diagnostic.disposition ==
             VelocityExtensionRowDisposition::Regression) ||
        diagnostic.bounded_fallback_used !=
            (diagnostic.disposition ==
             VelocityExtensionRowDisposition::BoundedFallback) ||
        diagnostic.condition_rejected == diagnostic.coefficient_rejected &&
            diagnostic.bounded_fallback_used) {
      throw std::invalid_argument(
          "velocity-extension map snapshot received inconsistent reconstruction evidence");
    }
    extended_rows += reconstructed ? 1u : 0u;
    outside_rows += !diagnostic.assigned ? 1u : 0u;
    collision_rows += diagnostic.component_candidates > 1u ? 1u : 0u;
    regression_rows += diagnostic.regression_attempted ? 1u : 0u;
    accepted_regression_rows += diagnostic.regression_accepted ? 1u : 0u;
    fallback_rows += diagnostic.bounded_fallback_used ? 1u : 0u;
    condition_rejections += diagnostic.condition_rejected ? 1u : 0u;
    coefficient_rejections += diagnostic.coefficient_rejected ? 1u : 0u;
    wall_projected_rows += diagnostic.wall_projected ? 1u : 0u;
  }
  if (extended_rows != report_.extended_vertices ||
      outside_rows != report_.vertices_outside_band ||
      collision_rows != report_.component_collision_vertices ||
      regression_rows != report_.regression_candidate_rows ||
      accepted_regression_rows != report_.regression_accepted_rows ||
      fallback_rows != report_.bounded_fallback_rows ||
      condition_rejections != report_.condition_rejected_rows ||
      coefficient_rejections != report_.coefficient_rejected_rows ||
      wall_projected_rows != report_.wall_projected_vertices) {
    throw std::invalid_argument(
        "velocity-extension map snapshot owner rows disagree with the aggregate report");
  }
}

bool solveSmallDenseSystem(
    std::array<std::array<double, 4>, 4> matrix,
    std::array<double, 4> rhs,
    int size,
    std::array<double, 4>& solution)
{
  solution.fill(0.0);
  for (int column = 0; column < size; ++column) {
    int pivot = column;
    double pivot_magnitude =
        std::abs(matrix[static_cast<std::size_t>(column)]
                       [static_cast<std::size_t>(column)]);
    for (int row = column + 1; row < size; ++row) {
      const double magnitude =
          std::abs(matrix[static_cast<std::size_t>(row)]
                         [static_cast<std::size_t>(column)]);
      if (magnitude > pivot_magnitude) {
        pivot = row;
        pivot_magnitude = magnitude;
      }
    }
    double row_scale = 0.0;
    for (int entry = column; entry < size; ++entry) {
      row_scale = std::max(
          row_scale,
          std::abs(matrix[static_cast<std::size_t>(pivot)]
                         [static_cast<std::size_t>(entry)]));
    }
    if (!(pivot_magnitude > 1.0e-12 * std::max(row_scale, 1.0))) {
      return false;
    }
    if (pivot != column) {
      std::swap(matrix[static_cast<std::size_t>(pivot)],
                matrix[static_cast<std::size_t>(column)]);
      std::swap(rhs[static_cast<std::size_t>(pivot)],
                rhs[static_cast<std::size_t>(column)]);
    }
    const double diagonal =
        matrix[static_cast<std::size_t>(column)]
              [static_cast<std::size_t>(column)];
    for (int row = column + 1; row < size; ++row) {
      const double factor =
          matrix[static_cast<std::size_t>(row)]
                [static_cast<std::size_t>(column)] /
          diagonal;
      for (int entry = column; entry < size; ++entry) {
        matrix[static_cast<std::size_t>(row)]
              [static_cast<std::size_t>(entry)] -=
            factor * matrix[static_cast<std::size_t>(column)]
                           [static_cast<std::size_t>(entry)];
      }
      rhs[static_cast<std::size_t>(row)] -=
          factor * rhs[static_cast<std::size_t>(column)];
    }
  }
  for (int row = size - 1; row >= 0; --row) {
    double value = rhs[static_cast<std::size_t>(row)];
    for (int column = row + 1; column < size; ++column) {
      value -= matrix[static_cast<std::size_t>(row)]
                     [static_cast<std::size_t>(column)] *
               solution[static_cast<std::size_t>(column)];
    }
    const double diagonal =
        matrix[static_cast<std::size_t>(row)]
              [static_cast<std::size_t>(row)];
    if (!(std::abs(diagonal) > 0.0) || !std::isfinite(diagonal)) {
      return false;
    }
    solution[static_cast<std::size_t>(row)] = value / diagonal;
  }
  return std::all_of(
      solution.begin(),
      solution.begin() + size,
      [](double value) { return std::isfinite(value); });
}

SymmetricRankConditionEstimate estimateSymmetricRankAndCondition(
    const std::array<std::array<double, 4>, 4>& matrix,
    int size)
{
  if (size <= 0 || size > 4) {
    return SymmetricRankConditionEstimate{
        .numerical_rank = 0,
        .condition_estimate = std::numeric_limits<double>::infinity(),
    };
  }

  std::array<std::array<double, 4>, 4> scaled{};
  for (int row = 0; row < size; ++row) {
    const double diagonal =
        matrix[static_cast<std::size_t>(row)]
              [static_cast<std::size_t>(row)];
    if (!(diagonal > 0.0) || !std::isfinite(diagonal)) {
      return SymmetricRankConditionEstimate{
          .numerical_rank = 0,
          .condition_estimate = std::numeric_limits<double>::infinity(),
      };
    }
    for (int column = 0; column < size; ++column) {
      const double other_diagonal =
          matrix[static_cast<std::size_t>(column)]
                [static_cast<std::size_t>(column)];
      if (!(other_diagonal > 0.0) || !std::isfinite(other_diagonal)) {
        return SymmetricRankConditionEstimate{
            .numerical_rank = 0,
            .condition_estimate = std::numeric_limits<double>::infinity(),
        };
      }
      const double value =
          matrix[static_cast<std::size_t>(row)]
                [static_cast<std::size_t>(column)] /
          std::sqrt(diagonal * other_diagonal);
      if (!std::isfinite(value)) {
        return SymmetricRankConditionEstimate{
            .numerical_rank = 0,
            .condition_estimate = std::numeric_limits<double>::infinity(),
        };
      }
      scaled[static_cast<std::size_t>(row)]
            [static_cast<std::size_t>(column)] = value;
    }
  }

  // Jacobi diagonalization is inexpensive and deterministic for these
  // one- through three-dimensional weighted Gram matrices.
  for (int sweep = 0; sweep < 64; ++sweep) {
    int pivot_row = 0;
    int pivot_column = 0;
    double max_off_diagonal = 0.0;
    for (int row = 0; row < size; ++row) {
      for (int column = row + 1; column < size; ++column) {
        const double magnitude = std::abs(
            scaled[static_cast<std::size_t>(row)]
                  [static_cast<std::size_t>(column)]);
        if (magnitude > max_off_diagonal) {
          max_off_diagonal = magnitude;
          pivot_row = row;
          pivot_column = column;
        }
      }
    }
    if (max_off_diagonal <= 64.0 * std::numeric_limits<double>::epsilon()) {
      break;
    }
    const auto p = static_cast<std::size_t>(pivot_row);
    const auto q = static_cast<std::size_t>(pivot_column);
    const double app = scaled[p][p];
    const double aqq = scaled[q][q];
    const double apq = scaled[p][q];
    const double angle = 0.5 * std::atan2(2.0 * apq, aqq - app);
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    for (int index = 0; index < size; ++index) {
      const auto k = static_cast<std::size_t>(index);
      if (k == p || k == q) {
        continue;
      }
      const double akp = scaled[k][p];
      const double akq = scaled[k][q];
      scaled[k][p] = scaled[p][k] = cosine * akp - sine * akq;
      scaled[k][q] = scaled[q][k] = sine * akp + cosine * akq;
    }
    scaled[p][p] = cosine * cosine * app -
                   2.0 * sine * cosine * apq + sine * sine * aqq;
    scaled[q][q] = sine * sine * app +
                   2.0 * sine * cosine * apq + cosine * cosine * aqq;
    scaled[p][q] = scaled[q][p] = 0.0;
  }

  std::array<double, 4> eigenvalues{};
  double maximum = 0.0;
  for (int index = 0; index < size; ++index) {
    const double eigenvalue =
        scaled[static_cast<std::size_t>(index)]
              [static_cast<std::size_t>(index)];
    if (!std::isfinite(eigenvalue)) {
      return SymmetricRankConditionEstimate{
          .numerical_rank = 0,
          .condition_estimate = std::numeric_limits<double>::infinity(),
      };
    }
    eigenvalues[static_cast<std::size_t>(index)] = eigenvalue;
    maximum = std::max(maximum, eigenvalue);
  }
  if (!(maximum > 0.0)) {
    return SymmetricRankConditionEstimate{
        .numerical_rank = 0,
        .condition_estimate = std::numeric_limits<double>::infinity(),
    };
  }
  const double rank_tolerance =
      maximum * 64.0 * std::numeric_limits<double>::epsilon();
  int numerical_rank = 0;
  double minimum_retained = std::numeric_limits<double>::infinity();
  for (int index = 0; index < size; ++index) {
    const double eigenvalue = eigenvalues[static_cast<std::size_t>(index)];
    if (eigenvalue > rank_tolerance) {
      ++numerical_rank;
      minimum_retained = std::min(minimum_retained, eigenvalue);
    }
  }
  return SymmetricRankConditionEstimate{
      .numerical_rank = numerical_rank,
      .condition_estimate =
          numerical_rank == size
              ? maximum / minimum_retained
              : std::numeric_limits<double>::infinity(),
  };
}

double estimateSymmetricConditionNumber(
    const std::array<std::array<double, 4>, 4>& matrix,
    int size)
{
  return estimateSymmetricRankAndCondition(matrix, size).condition_estimate;
}

bool ownsVelocityExtensionVertex(const svmp::Mesh& mesh,
                                 std::size_t vertex,
                                 const svmp::MeshComm& comm)
{
  if (!comm.is_parallel()) {
    return true;
  }
  if (vertex >= mesh.n_vertices()) {
    throw std::runtime_error(
        "velocity-extension ownership check received an invalid local vertex");
  }
  const auto owner = mesh.owner_rank_vertex(
      static_cast<svmp::index_t>(vertex));
  if (owner < 0 || owner >= comm.size()) {
    throw std::runtime_error(
        "velocity-extension graph requires valid vertex owner ranks on the active communicator");
  }
  return owner == comm.rank();
}

svmp::gid_t velocityExtensionVertexGlobalIdentity(const svmp::Mesh& mesh,
                                                  std::size_t vertex)
{
  if (vertex >= mesh.n_vertices()) {
    throw std::runtime_error(
        "velocity-extension row diagnostic received an invalid local vertex");
  }
  const auto& vertex_gids = mesh.local_mesh().vertex_gids();
  if (vertex_gids.size() == mesh.n_vertices() &&
      vertex_gids[vertex] != svmp::INVALID_GID) {
    return vertex_gids[vertex];
  }
  return static_cast<svmp::gid_t>(vertex);
}

std::size_t globalOwnedVelocityExtensionMaskCount(
    const svmp::Mesh& mesh,
    const svmp::MeshComm& comm,
    std::span<const std::uint8_t> mask)
{
  if (mask.size() != mesh.n_vertices()) {
    throw std::runtime_error(
        "velocity-extension global mask count received an incompatible mask");
  }
  std::uint64_t global_count = 0u;
  for (std::size_t vertex = 0; vertex < mask.size(); ++vertex) {
    if (mask[vertex] != 0u &&
        ownsVelocityExtensionVertex(mesh, vertex, comm)) {
      ++global_count;
    }
  }
#ifdef MESH_HAS_MPI
  if (comm.is_parallel()) {
    const auto local_count = global_count;
    MPI_Allreduce(&local_count,
                  &global_count,
                  1,
                  MPI_UINT64_T,
                  MPI_SUM,
                  comm.native());
  }
#endif
  if (global_count >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error(
        "velocity-extension global mask count exceeds size_t");
  }
  return static_cast<std::size_t>(global_count);
}

std::size_t globalVelocityExtensionGeometrySampleCount(
    std::size_t local_count,
    const svmp::MeshComm& comm)
{
  if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
    if (local_count >
        static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max())) {
      throw std::runtime_error(
          "velocity-extension geometry sample count exceeds uint64_t");
    }
  }
  std::uint64_t global_count = static_cast<std::uint64_t>(local_count);
#ifdef MESH_HAS_MPI
  if (comm.is_parallel()) {
    const auto communicator_size = static_cast<std::uint64_t>(comm.size());
    if (communicator_size == 0u ||
        global_count >
            std::numeric_limits<std::uint64_t>::max() / communicator_size) {
      throw std::runtime_error(
          "velocity-extension global geometry sample count would overflow uint64_t");
    }
    const auto bounded_local_count = global_count;
    MPI_Allreduce(&bounded_local_count,
                  &global_count,
                  1,
                  MPI_UINT64_T,
                  MPI_SUM,
                  comm.native());
  }
#else
  (void)comm;
#endif
  if (global_count >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error(
        "velocity-extension global geometry sample count exceeds size_t");
  }
  return static_cast<std::size_t>(global_count);
}

std::size_t markVelocityExtensionTraceSupportCells(
    const svmp::Mesh& mesh,
    std::span<const svmp::FE::MeshIndex> cells,
    std::vector<std::uint8_t>& trace_support)
{
  if (trace_support.size() != mesh.n_vertices()) {
    throw std::runtime_error(
        "velocity-extension trace support received an incompatible vertex mask");
  }
  const auto& local_mesh = mesh.local_mesh();
  std::size_t newly_marked = 0u;
  for (const auto raw_cell : cells) {
    if (raw_cell < static_cast<svmp::FE::MeshIndex>(0) ||
        raw_cell >= static_cast<svmp::FE::MeshIndex>(local_mesh.n_cells())) {
      throw std::runtime_error(
          "velocity-extension trace support received an invalid retained cut/interface cell");
    }
    auto [cell_vertices, count] = local_mesh.cell_vertices_span(
        static_cast<svmp::index_t>(raw_cell));
    if (cell_vertices == nullptr || count == 0u) {
      throw std::runtime_error(
          "velocity-extension trace support found a retained cut/interface cell without vertices");
    }
    for (std::size_t local = 0; local < count; ++local) {
      if (cell_vertices[local] < 0 ||
          static_cast<std::size_t>(cell_vertices[local]) >= mesh.n_vertices()) {
        throw std::runtime_error(
            "velocity-extension trace support found an invalid cut-cell vertex");
      }
      const auto vertex = static_cast<std::size_t>(cell_vertices[local]);
      if (trace_support[vertex] == 0u) {
        trace_support[vertex] = 1u;
        ++newly_marked;
      }
    }
  }
  return newly_marked;
}

std::vector<svmp::FE::MeshIndex> nodalVelocityExtensionInterfaceCells(
    const svmp::Mesh& mesh,
    std::span<const double> phi,
    double isovalue)
{
  if (phi.size() != mesh.n_vertices() || !std::isfinite(isovalue)) {
    throw std::runtime_error(
        "velocity-extension nodal interface search received incompatible data");
  }
  constexpr double zero_tolerance = 1.0e-12;
  const auto& local_mesh = mesh.local_mesh();
  std::vector<svmp::FE::MeshIndex> cells;
  for (svmp::index_t cell = 0; cell < local_mesh.n_cells(); ++cell) {
    auto [cell_vertices, count] = local_mesh.cell_vertices_span(cell);
    if (cell_vertices == nullptr || count == 0u) {
      continue;
    }
    bool negative = false;
    bool positive = false;
    bool on_interface = false;
    for (std::size_t local = 0; local < count; ++local) {
      if (cell_vertices[local] < 0 ||
          static_cast<std::size_t>(cell_vertices[local]) >= phi.size()) {
        throw std::runtime_error(
            "velocity-extension nodal interface search found an invalid cell vertex");
      }
      const double value =
          phi[static_cast<std::size_t>(cell_vertices[local])] - isovalue;
      if (!std::isfinite(value)) {
        throw std::runtime_error(
            "velocity-extension nodal interface search found a non-finite level set");
      }
      negative = negative || value < -zero_tolerance;
      positive = positive || value > zero_tolerance;
      on_interface = on_interface || std::abs(value) <= zero_tolerance;
    }
    if ((negative && positive) || on_interface) {
      cells.push_back(static_cast<svmp::FE::MeshIndex>(cell));
    }
  }
  return cells;
}

std::size_t synchronizeVelocityExtensionTraceSupportMask(
    const svmp::Mesh& mesh,
    const svmp::MeshComm& comm,
    std::vector<std::uint8_t>& trace_support)
{
  if (trace_support.size() != mesh.n_vertices()) {
    throw std::runtime_error(
        "velocity-extension trace-support synchronization received an incompatible mask");
  }
  if (!comm.is_parallel()) {
    return globalOwnedVelocityExtensionMaskCount(
        mesh, comm, trace_support);
  }

#ifdef MESH_HAS_MPI
  const auto& vertex_gids = mesh.local_mesh().vertex_gids();
  if (vertex_gids.size() != mesh.n_vertices()) {
    throw std::runtime_error(
        "parallel velocity-extension trace support requires one global ID per local vertex");
  }
  std::vector<std::int64_t> local_gids;
  local_gids.reserve(trace_support.size());
  for (std::size_t vertex = 0; vertex < trace_support.size(); ++vertex) {
    if (trace_support[vertex] == 0u) {
      continue;
    }
    if (vertex_gids[vertex] == svmp::INVALID_GID) {
      throw std::runtime_error(
          "parallel velocity-extension trace support found an invalid vertex global ID");
    }
    local_gids.push_back(static_cast<std::int64_t>(vertex_gids[vertex]));
  }
  if (local_gids.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(
        "velocity-extension trace-support synchronization record count overflow");
  }
  const int local_count = static_cast<int>(local_gids.size());
  std::vector<int> counts(static_cast<std::size_t>(comm.size()), 0);
  MPI_Allgather(&local_count, 1, MPI_INT,
                counts.data(), 1, MPI_INT, comm.native());
  std::vector<int> displacements(counts.size(), 0);
  int total_count = 0;
  for (std::size_t rank = 0; rank < counts.size(); ++rank) {
    if (counts[rank] < 0 ||
        counts[rank] > std::numeric_limits<int>::max() - total_count) {
      throw std::runtime_error(
          "velocity-extension trace-support synchronization displacement overflow");
    }
    displacements[rank] = total_count;
    total_count += counts[rank];
  }
  std::vector<std::int64_t> gathered_gids(
      static_cast<std::size_t>(total_count), -1);
#ifdef MPI_INT64_T
  const MPI_Datatype gid_type = MPI_INT64_T;
#else
  const MPI_Datatype gid_type = MPI_LONG_LONG;
#endif
  MPI_Allgatherv(local_gids.empty() ? nullptr : local_gids.data(),
                 local_count,
                 gid_type,
                 gathered_gids.empty() ? nullptr : gathered_gids.data(),
                 counts.data(),
                 displacements.data(),
                 gid_type,
                 comm.native());

  std::unordered_map<svmp::gid_t, std::size_t> local_vertex_by_gid;
  local_vertex_by_gid.reserve(vertex_gids.size());
  for (std::size_t vertex = 0; vertex < vertex_gids.size(); ++vertex) {
    if (vertex_gids[vertex] == svmp::INVALID_GID ||
        !local_vertex_by_gid.emplace(vertex_gids[vertex], vertex).second) {
      throw std::runtime_error(
          "velocity-extension trace support requires unique valid local vertex global IDs");
    }
  }
  for (const auto raw_gid : gathered_gids) {
    const auto local =
        local_vertex_by_gid.find(static_cast<svmp::gid_t>(raw_gid));
    if (local != local_vertex_by_gid.end()) {
      trace_support[local->second] = 1u;
    }
  }
  return globalOwnedVelocityExtensionMaskCount(
      mesh, comm, trace_support);
#else
  return globalOwnedVelocityExtensionMaskCount(
      mesh, comm, trace_support);
#endif
}

constexpr std::int64_t kInvalidVelocityExtensionComponent =
    std::numeric_limits<std::int64_t>::max();

std::vector<std::size_t> synchronizeVelocityExtensionComponentLabels(
    const svmp::Mesh& mesh,
    const svmp::MeshComm& comm,
    std::span<const std::size_t> changed_owned_vertices,
    std::vector<std::int64_t>& component_labels)
{
  if (!comm.is_parallel()) {
    return std::vector<std::size_t>(changed_owned_vertices.begin(),
                                    changed_owned_vertices.end());
  }
  if (component_labels.size() != mesh.n_vertices()) {
    throw std::runtime_error(
        "velocity-extension component synchronization received an incompatible label array");
  }

#ifdef MESH_HAS_MPI
  const auto& vertex_gids = mesh.local_mesh().vertex_gids();
  if (vertex_gids.size() != mesh.n_vertices()) {
    throw std::runtime_error(
        "parallel velocity-extension components require one valid global ID per local vertex");
  }
  if (changed_owned_vertices.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(
        "velocity-extension component synchronization record count overflow");
  }
  const int local_count = static_cast<int>(changed_owned_vertices.size());
  std::vector<int> counts(static_cast<std::size_t>(comm.size()), 0);
  MPI_Allgather(&local_count, 1, MPI_INT,
                counts.data(), 1, MPI_INT, comm.native());
  std::vector<int> displacements(counts.size(), 0);
  int total_count = 0;
  for (std::size_t rank = 0; rank < counts.size(); ++rank) {
    if (counts[rank] < 0 ||
        counts[rank] > std::numeric_limits<int>::max() - total_count) {
      throw std::runtime_error(
          "velocity-extension component synchronization displacement overflow");
    }
    displacements[rank] = total_count;
    total_count += counts[rank];
  }

  std::vector<std::int64_t> send_gids(changed_owned_vertices.size(), -1);
  std::vector<std::int64_t> send_labels(
      changed_owned_vertices.size(), kInvalidVelocityExtensionComponent);
  for (std::size_t record = 0; record < changed_owned_vertices.size();
       ++record) {
    const auto vertex = changed_owned_vertices[record];
    if (vertex >= mesh.n_vertices() ||
        !ownsVelocityExtensionVertex(mesh, vertex, comm) ||
        component_labels[vertex] == kInvalidVelocityExtensionComponent) {
      throw std::runtime_error(
          "velocity-extension component synchronization attempted to publish an invalid owner label");
    }
    if (vertex_gids[vertex] == svmp::INVALID_GID) {
      throw std::runtime_error(
          "velocity-extension component synchronization encountered an invalid vertex global ID");
    }
    send_gids[record] = static_cast<std::int64_t>(vertex_gids[vertex]);
    send_labels[record] = component_labels[vertex];
  }

  std::vector<std::int64_t> gathered_gids(
      static_cast<std::size_t>(total_count), -1);
  std::vector<std::int64_t> gathered_labels(
      static_cast<std::size_t>(total_count),
      kInvalidVelocityExtensionComponent);
#ifdef MPI_INT64_T
  const MPI_Datatype component_type = MPI_INT64_T;
#else
  const MPI_Datatype component_type = MPI_LONG_LONG;
#endif
  MPI_Allgatherv(send_gids.empty() ? nullptr : send_gids.data(),
                 local_count,
                 component_type,
                 gathered_gids.empty() ? nullptr : gathered_gids.data(),
                 counts.data(),
                 displacements.data(),
                 component_type,
                 comm.native());
  MPI_Allgatherv(send_labels.empty() ? nullptr : send_labels.data(),
                 local_count,
                 component_type,
                 gathered_labels.empty() ? nullptr : gathered_labels.data(),
                 counts.data(),
                 displacements.data(),
                 component_type,
                 comm.native());

  std::unordered_map<svmp::gid_t, std::size_t> local_vertex_by_gid;
  local_vertex_by_gid.reserve(vertex_gids.size());
  for (std::size_t vertex = 0; vertex < vertex_gids.size(); ++vertex) {
    if (vertex_gids[vertex] == svmp::INVALID_GID ||
        !local_vertex_by_gid.emplace(vertex_gids[vertex], vertex).second) {
      throw std::runtime_error(
          "velocity-extension components require unique valid local vertex global IDs");
    }
  }
  std::set<svmp::gid_t> published_gids;
  std::vector<std::size_t> local_changed_vertices;
  for (int record = 0; record < total_count; ++record) {
    const auto gid = static_cast<svmp::gid_t>(
        gathered_gids[static_cast<std::size_t>(record)]);
    const auto label = gathered_labels[static_cast<std::size_t>(record)];
    if (!published_gids.insert(gid).second) {
      throw std::runtime_error(
          "velocity-extension components received duplicate owner records for one global vertex");
    }
    if (label == kInvalidVelocityExtensionComponent) {
      throw std::runtime_error(
          "velocity-extension components received an invalid synchronized label");
    }
    const auto local = local_vertex_by_gid.find(gid);
    if (local == local_vertex_by_gid.end()) {
      continue;
    }
    component_labels[local->second] = label;
    local_changed_vertices.push_back(local->second);
  }
  std::sort(local_changed_vertices.begin(), local_changed_vertices.end());
  local_changed_vertices.erase(
      std::unique(local_changed_vertices.begin(),
                  local_changed_vertices.end()),
      local_changed_vertices.end());
  return local_changed_vertices;
#else
  (void)mesh;
  (void)component_labels;
  return std::vector<std::size_t>(changed_owned_vertices.begin(),
                                  changed_owned_vertices.end());
#endif
}

std::vector<std::int64_t> identifyVelocityExtensionActiveComponents(
    const svmp::Mesh& mesh,
    const svmp::MeshComm& comm,
    std::span<const std::uint8_t> active,
    const std::vector<std::vector<std::size_t>>& neighbors)
{
  if (active.size() != mesh.n_vertices() ||
      neighbors.size() != mesh.n_vertices()) {
    throw std::runtime_error(
        "velocity-extension component identification received incompatible graph data");
  }
  std::vector<std::int64_t> labels(
      mesh.n_vertices(), kInvalidVelocityExtensionComponent);
  std::vector<std::size_t> owned_active;
  std::vector<std::uint8_t> owned_active_flag(mesh.n_vertices(), 0u);
  owned_active.reserve(mesh.n_vertices());
  const auto& vertex_gids = mesh.local_mesh().vertex_gids();
  for (std::size_t vertex = 0; vertex < mesh.n_vertices(); ++vertex) {
    if (active[vertex] == 0u ||
        !ownsVelocityExtensionVertex(mesh, vertex, comm)) {
      continue;
    }
    if (comm.is_parallel()) {
      if (vertex_gids.size() != mesh.n_vertices() ||
          vertex_gids[vertex] == svmp::INVALID_GID) {
        throw std::runtime_error(
            "parallel velocity-extension component identification requires valid vertex global IDs");
      }
      labels[vertex] = static_cast<std::int64_t>(vertex_gids[vertex]);
    } else {
      labels[vertex] = static_cast<std::int64_t>(vertex);
    }
    owned_active.push_back(vertex);
    owned_active_flag[vertex] = 1u;
  }
  synchronizeVelocityExtensionComponentLabels(
      mesh, comm, owned_active, labels);

  std::uint64_t global_active_count =
      static_cast<std::uint64_t>(owned_active.size());
#ifdef MESH_HAS_MPI
  if (comm.is_parallel()) {
    const auto local_active_count = global_active_count;
    MPI_Allreduce(&local_active_count,
                  &global_active_count,
                  1,
                  MPI_UINT64_T,
                  MPI_SUM,
                  comm.native());
  }
#endif
  if (global_active_count == 0u) {
    return labels;
  }

  // Distributed minimum-label relaxation identifies connected components of
  // the active P1 vertex graph.  Each rank first collapses every complete
  // local owned/ghost component, then synchronizes changed owner labels.  A
  // component crossing partitions therefore receives the same deterministic
  // minimum global-vertex label without one collective per mesh edge.
  for (std::uint64_t sweep = 0u; sweep <= global_active_count; ++sweep) {
    std::vector<std::size_t> changed_owned;
    std::vector<std::uint8_t> visited(mesh.n_vertices(), 0u);
    std::vector<std::size_t> stack;
    std::vector<std::size_t> local_component;
    for (std::size_t seed = 0; seed < mesh.n_vertices(); ++seed) {
      if (active[seed] == 0u || visited[seed] != 0u) {
        continue;
      }
      stack.clear();
      local_component.clear();
      stack.push_back(seed);
      visited[seed] = 1u;
      auto minimum_label = kInvalidVelocityExtensionComponent;
      while (!stack.empty()) {
        const auto vertex = stack.back();
        stack.pop_back();
        local_component.push_back(vertex);
        minimum_label = std::min(minimum_label, labels[vertex]);
        for (const auto neighbor : neighbors[vertex]) {
          if (active[neighbor] == 0u || visited[neighbor] != 0u) {
            continue;
          }
          visited[neighbor] = 1u;
          stack.push_back(neighbor);
        }
      }
      if (minimum_label == kInvalidVelocityExtensionComponent) {
        throw std::runtime_error(
            "velocity-extension active component has no synchronized owner label");
      }
      for (const auto vertex : local_component) {
        if (owned_active_flag[vertex] != 0u &&
            minimum_label < labels[vertex]) {
          labels[vertex] = minimum_label;
          changed_owned.push_back(vertex);
        }
      }
    }
    synchronizeVelocityExtensionComponentLabels(
        mesh, comm, changed_owned, labels);

    std::uint64_t global_changed =
        static_cast<std::uint64_t>(changed_owned.size());
#ifdef MESH_HAS_MPI
    if (comm.is_parallel()) {
      const auto local_changed = global_changed;
      MPI_Allreduce(&local_changed,
                    &global_changed,
                    1,
                    MPI_UINT64_T,
                    MPI_SUM,
                    comm.native());
    }
#endif
    if (global_changed == 0u) {
      return labels;
    }
  }
  throw std::runtime_error(
      "velocity-extension active-component labels did not converge within the global active-vertex bound");
}

std::vector<std::size_t> synchronizeVelocityExtensionLayer(
    const svmp::Mesh& mesh,
    const svmp::MeshComm& comm,
    std::span<const std::size_t> newly_owned_vertices,
    std::size_t target_components,
    std::vector<std::uint8_t>& assigned,
    std::vector<double>& extended,
    std::vector<double>& extension_distance)
{
  if (target_components == 0u || assigned.size() != mesh.n_vertices() ||
      extended.size() != mesh.n_vertices() * target_components ||
      extension_distance.size() != mesh.n_vertices()) {
    throw std::runtime_error(
        "velocity-extension layer synchronization received incompatible arrays");
  }
  if (!comm.is_parallel()) {
    return std::vector<std::size_t>(newly_owned_vertices.begin(),
                                    newly_owned_vertices.end());
  }

#ifdef MESH_HAS_MPI
  const auto record_width = target_components + 1u;
  const auto& vertex_gids = mesh.local_mesh().vertex_gids();
  if (vertex_gids.size() != mesh.n_vertices()) {
    throw std::runtime_error(
        "parallel velocity-extension graph requires one valid global ID per local vertex");
  }
  if (newly_owned_vertices.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(
        "velocity-extension layer has too many newly assigned vertices for MPI exchange");
  }
  const int local_count = static_cast<int>(newly_owned_vertices.size());
  std::vector<int> counts(static_cast<std::size_t>(comm.size()), 0);
  MPI_Allgather(&local_count, 1, MPI_INT,
                counts.data(), 1, MPI_INT, comm.native());

  std::vector<int> displacements(counts.size(), 0);
  std::vector<int> value_counts(counts.size(), 0);
  std::vector<int> value_displacements(counts.size(), 0);
  int total_count = 0;
  for (std::size_t rank = 0; rank < counts.size(); ++rank) {
    if (counts[rank] < 0 ||
        record_width >
            static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        counts[rank] >
            std::numeric_limits<int>::max() /
                static_cast<int>(record_width)) {
      throw std::runtime_error(
          "velocity-extension layer MPI record count overflow");
    }
    if (counts[rank] > std::numeric_limits<int>::max() - total_count ||
        total_count >
            std::numeric_limits<int>::max() /
                static_cast<int>(record_width)) {
      throw std::runtime_error(
          "velocity-extension layer MPI displacement overflow");
    }
    displacements[rank] = total_count;
    value_displacements[rank] =
        total_count * static_cast<int>(record_width);
    value_counts[rank] =
        counts[rank] * static_cast<int>(record_width);
    total_count += counts[rank];
    if (total_count >
        std::numeric_limits<int>::max() /
            static_cast<int>(record_width)) {
      throw std::runtime_error(
          "velocity-extension layer MPI value extent overflow");
    }
  }

  std::vector<std::int64_t> send_gids(newly_owned_vertices.size(), -1);
  std::vector<double> send_values(
      newly_owned_vertices.size() * record_width, 0.0);
  for (std::size_t record = 0; record < newly_owned_vertices.size(); ++record) {
    const auto vertex = newly_owned_vertices[record];
    if (vertex >= mesh.n_vertices() ||
        !ownsVelocityExtensionVertex(mesh, vertex, comm) ||
        assigned[vertex] == 0u) {
      throw std::runtime_error(
          "velocity-extension layer attempted to publish a non-owned or unassigned vertex");
    }
    const auto gid = vertex_gids[vertex];
    if (gid == svmp::INVALID_GID) {
      throw std::runtime_error(
          "velocity-extension layer encountered an invalid vertex global ID");
    }
    send_gids[record] = static_cast<std::int64_t>(gid);
    for (std::size_t component = 0; component < target_components;
         ++component) {
      const double value =
          extended[vertex * target_components + component];
      if (!std::isfinite(value)) {
        throw std::runtime_error(
            "velocity-extension layer encountered a non-finite owner value");
      }
      send_values[record * record_width + component] = value;
    }
    const double distance = extension_distance[vertex];
    if (!std::isfinite(distance)) {
      throw std::runtime_error(
          "velocity-extension layer encountered an invalid owner extrapolation distance");
    }
    send_values[record * record_width + target_components] = distance;
  }

  std::vector<std::int64_t> gathered_gids(
      static_cast<std::size_t>(total_count), -1);
  std::vector<double> gathered_values(
      static_cast<std::size_t>(total_count) * record_width, 0.0);
#ifdef MPI_INT64_T
  const MPI_Datatype gid_type = MPI_INT64_T;
#else
  const MPI_Datatype gid_type = MPI_LONG_LONG;
#endif
  MPI_Allgatherv(send_gids.empty() ? nullptr : send_gids.data(),
                 local_count,
                 gid_type,
                 gathered_gids.empty() ? nullptr : gathered_gids.data(),
                 counts.data(),
                 displacements.data(),
                 gid_type,
                 comm.native());
  MPI_Allgatherv(send_values.empty() ? nullptr : send_values.data(),
                 local_count * static_cast<int>(record_width),
                 MPI_DOUBLE,
                 gathered_values.empty() ? nullptr : gathered_values.data(),
                 value_counts.data(),
                 value_displacements.data(),
                 MPI_DOUBLE,
                 comm.native());

  std::unordered_map<svmp::gid_t, std::size_t> local_vertex_by_gid;
  local_vertex_by_gid.reserve(vertex_gids.size());
  for (std::size_t vertex = 0; vertex < vertex_gids.size(); ++vertex) {
    if (vertex_gids[vertex] == svmp::INVALID_GID ||
        !local_vertex_by_gid.emplace(vertex_gids[vertex], vertex).second) {
      throw std::runtime_error(
          "velocity-extension layer requires unique valid local vertex global IDs");
    }
  }
  std::set<svmp::gid_t> published_gids;
  std::vector<std::size_t> local_frontier;
  for (int record = 0; record < total_count; ++record) {
    const auto gid = static_cast<svmp::gid_t>(
        gathered_gids[static_cast<std::size_t>(record)]);
    if (!published_gids.insert(gid).second) {
      throw std::runtime_error(
          "velocity-extension layer received duplicate owner records for one global vertex");
    }
    const auto local = local_vertex_by_gid.find(gid);
    if (local == local_vertex_by_gid.end()) {
      continue;
    }
    const auto vertex = local->second;
    for (std::size_t component = 0; component < target_components;
         ++component) {
      const double value = gathered_values[
          static_cast<std::size_t>(record) * record_width + component];
      if (!std::isfinite(value)) {
        throw std::runtime_error(
            "velocity-extension layer received a non-finite synchronized value");
      }
      extended[vertex * target_components + component] = value;
    }
    const double distance = gathered_values[
        static_cast<std::size_t>(record) * record_width + target_components];
    if (!std::isfinite(distance)) {
      throw std::runtime_error(
          "velocity-extension layer received an invalid extrapolation distance");
    }
    extension_distance[vertex] = distance;
    assigned[vertex] = 1u;
    local_frontier.push_back(vertex);
  }
  std::sort(local_frontier.begin(), local_frontier.end());
  local_frontier.erase(
      std::unique(local_frontier.begin(), local_frontier.end()),
      local_frontier.end());
  return local_frontier;
#else
  (void)mesh;
  (void)target_components;
  (void)assigned;
  (void)extended;
  (void)extension_distance;
  return std::vector<std::size_t>(newly_owned_vertices.begin(),
                                  newly_owned_vertices.end());
#endif
}

std::vector<std::vector<std::size_t>> velocityExtensionEdgeAdjacency(
    const svmp::Mesh& mesh)
{
  const auto vertex_count = mesh.n_vertices();
  const auto& local_mesh = mesh.local_mesh();
  std::vector<std::vector<std::size_t>> neighbors(vertex_count);

  const auto add_edge = [&](svmp::index_t raw_a, svmp::index_t raw_b) {
    if (raw_a < 0 || raw_b < 0 || raw_a == raw_b ||
        static_cast<std::size_t>(raw_a) >= vertex_count ||
        static_cast<std::size_t>(raw_b) >= vertex_count) {
      return;
    }
    const auto a = static_cast<std::size_t>(raw_a);
    const auto b = static_cast<std::size_t>(raw_b);
    neighbors[a].push_back(b);
    neighbors[b].push_back(a);
  };

  for (svmp::index_t cell = 0; cell < local_mesh.n_cells(); ++cell) {
    const auto [cell_vertices, count] = local_mesh.cell_vertices_span(cell);
    if (cell_vertices == nullptr || count < 2u) {
      continue;
    }
    const auto& shape = local_mesh.cell_shape(cell);
    const auto corner_count =
        shape.num_corners > 0
            ? std::min(count, static_cast<std::size_t>(shape.num_corners))
            : count;
    const auto add_edge_chain = [&](std::span<const svmp::index_t> chain) {
      for (std::size_t index = 0; index + 1u < chain.size(); ++index) {
        add_edge(chain[index], chain[index + 1u]);
      }
    };
    if (shape.family == svmp::CellFamily::Line) {
      const auto edge_dofs = local_mesh.cell_edge_geometry_dofs(cell, 0);
      if (edge_dofs.size() < 2u) {
        throw std::runtime_error(
            "velocity-extension line topology returned fewer than two edge nodes");
      }
      add_edge_chain(edge_dofs);
      continue;
    }

    if (shape.family == svmp::CellFamily::Polygon) {
      const auto edge_view = svmp::CellTopology::get_polygon_edges_view(
          static_cast<int>(corner_count));
      for (int edge = 0; edge < edge_view.edge_count; ++edge) {
        const auto local_a = edge_view.pairs_flat[2 * edge];
        const auto local_b = edge_view.pairs_flat[2 * edge + 1];
        if (local_a < 0 || local_b < 0 ||
            static_cast<std::size_t>(local_a) >= corner_count ||
            static_cast<std::size_t>(local_b) >= corner_count) {
          throw std::runtime_error(
              "velocity-extension polygon topology returned an invalid edge");
        }
        const auto edge_dofs =
            local_mesh.cell_edge_geometry_dofs(cell, edge);
        if (edge_dofs.size() >= 2u) {
          add_edge_chain(edge_dofs);
        } else {
          add_edge(cell_vertices[static_cast<std::size_t>(local_a)],
                   cell_vertices[static_cast<std::size_t>(local_b)]);
        }
      }
      continue;
    }

    const auto edge_view = svmp::CellTopology::get_edges_view(shape.family);
    if (edge_view.edge_count == 0) {
      throw std::runtime_error(
          "velocity-extension graph requires explicit mesh edges for every non-point cell");
    }
    for (int edge = 0; edge < edge_view.edge_count; ++edge) {
      const auto local_a = edge_view.pairs_flat[2 * edge];
      const auto local_b = edge_view.pairs_flat[2 * edge + 1];
      if (local_a < 0 || local_b < 0 ||
          static_cast<std::size_t>(local_a) >= corner_count ||
          static_cast<std::size_t>(local_b) >= corner_count) {
        throw std::runtime_error(
            "velocity-extension cell topology returned an invalid edge");
      }
      const auto edge_dofs = local_mesh.cell_edge_geometry_dofs(cell, edge);
      if (edge_dofs.size() >= 2u) {
        add_edge_chain(edge_dofs);
      } else {
        add_edge(cell_vertices[static_cast<std::size_t>(local_a)],
                 cell_vertices[static_cast<std::size_t>(local_b)]);
      }
    }
  }

  for (auto& adjacency : neighbors) {
    std::sort(adjacency.begin(), adjacency.end());
    adjacency.erase(std::unique(adjacency.begin(), adjacency.end()),
                    adjacency.end());
  }
  return neighbors;
}

[[maybe_unused]] WallCompatibleVelocityExtensionResult
extendVelocityInLevelSetNormalBand(
    const svmp::Mesh& mesh,
    const svmp::MeshComm& comm,
    std::span<const double> phi,
    std::span<const double> source_velocity,
    std::size_t source_components,
    std::span<const std::uint8_t> active,
    std::size_t target_components,
    std::size_t copy_components,
    int band_layers,
    bool enforce_wall_impermeability,
    std::span<const WallVelocityExtensionConstraint> wall_constraints,
    std::vector<double>& extended,
    std::vector<svmp::FE::level_set::VelocityExtensionConstraintRow>*
        constraint_rows,
    std::vector<std::int64_t>* component_assignment,
    std::vector<VelocityExtensionGraphRowDiagnostic>* row_diagnostics)
{
  const auto vertex_count = mesh.n_vertices();
  const int dimension = mesh.dim();
  if (phi.size() != vertex_count || active.size() != vertex_count ||
      source_velocity.size() < vertex_count * source_components ||
      target_components == 0u || copy_components > target_components ||
      band_layers <= 0) {
    throw std::invalid_argument(
        "wall-compatible level-set velocity extension received incompatible input");
  }
  if (enforce_wall_impermeability &&
      (copy_components < static_cast<std::size_t>(dimension) ||
       wall_constraints.empty())) {
    throw std::invalid_argument(
        "wall-compatible level-set velocity extension requires one explicit "
        "strong zero-velocity component mask per impermeable wall and all "
        "physical velocity components");
  }

  extended.assign(vertex_count * target_components, 0.0);
  if (constraint_rows != nullptr) {
    constraint_rows->clear();
    constraint_rows->reserve(vertex_count * target_components);
  }
  if (row_diagnostics != nullptr) {
    row_diagnostics->clear();
    row_diagnostics->reserve(vertex_count);
  }
  const auto& coordinates = mesh.X_ref();
  const auto& local_mesh = mesh.local_mesh();
  const auto neighbors = velocityExtensionEdgeAdjacency(mesh);
  auto extension_component = identifyVelocityExtensionActiveComponents(
      mesh, comm, active, neighbors);

  std::vector<std::vector<std::array<double, 3>>> wall_normals(vertex_count);
  std::vector<std::vector<std::array<double, 3>>>
      wall_projection_normals(vertex_count);
  std::vector<std::array<bool, 3>> wall_component_masks(
      vertex_count, std::array<bool, 3>{false, false, false});
  if (enforce_wall_impermeability) {
    auto boundary_faces =
        svmp::DistributedTopology::global_boundary_faces(mesh,
                                                         /*owned_only=*/false);
    if (boundary_faces.empty()) {
      boundary_faces = local_mesh.boundary_faces();
    }
    for (const auto face : boundary_faces) {
      if (face < 0 || face >= local_mesh.n_faces()) {
        continue;
      }
      const auto label = local_mesh.boundary_label(face);
      std::vector<const WallVelocityExtensionConstraint*>
          matching_constraints;
      for (const auto& constraint : wall_constraints) {
        if (constraint.boundary_label == svmp::INVALID_LABEL ||
            constraint.boundary_label == label) {
          matching_constraints.push_back(&constraint);
        }
      }
      if (matching_constraints.empty()) {
        continue;
      }
      const auto raw_normal = local_mesh.face_normal(face);
      const double norm = std::sqrt(raw_normal[0] * raw_normal[0] +
                                    raw_normal[1] * raw_normal[1] +
                                    raw_normal[2] * raw_normal[2]);
      if (!(norm > 0.0) || !std::isfinite(norm)) {
        continue;
      }
      const std::array<double, 3> normal{{raw_normal[0] / norm,
                                          raw_normal[1] / norm,
                                          raw_normal[2] / norm}};
      for (const auto* constraint : matching_constraints) {
        if (constraint->project_boundary_normal) {
          continue;
        }
        bool has_constrained_component = false;
        double unconstrained_normal2 = 0.0;
        for (int d = 0; d < dimension; ++d) {
          const auto component = static_cast<std::size_t>(d);
          has_constrained_component =
              has_constrained_component ||
              constraint->constrained_components[component];
          if (!constraint->constrained_components[component]) {
            unconstrained_normal2 += normal[component] * normal[component];
          }
        }
        if (!has_constrained_component) {
          throw std::runtime_error(
              "wall-compatible level-set velocity extension received an "
              "empty strong Dirichlet component mask");
        }
        // A component mask is impermeable only when it spans the boundary
        // normal.  Reject a mismatched Effective_direction instead of
        // silently changing the Navier--Stokes boundary condition.
        if (unconstrained_normal2 > 1.0e-20) {
          throw std::runtime_error(
              "wall-compatible level-set velocity extension received a "
              "strong zero-velocity component mask that does not constrain "
              "the wall-normal direction");
        }
      }
      auto [face_vertices, count] = local_mesh.face_vertices_span(face);
      for (std::size_t i = 0; face_vertices != nullptr && i < count; ++i) {
        if (face_vertices[i] < 0 ||
            static_cast<std::size_t>(face_vertices[i]) >= vertex_count) {
          continue;
        }
        const auto vertex = static_cast<std::size_t>(face_vertices[i]);
        wall_normals[vertex].push_back(normal);
        for (const auto* constraint : matching_constraints) {
          if (constraint->project_boundary_normal) {
            wall_projection_normals[vertex].push_back(normal);
            continue;
          }
          for (int d = 0; d < dimension; ++d) {
            const auto component = static_cast<std::size_t>(d);
            wall_component_masks[vertex][component] =
                wall_component_masks[vertex][component] ||
                constraint->constrained_components[component];
          }
        }
      }
    }
  }

  auto wall_velocity_projection = [&](std::size_t vertex) {
    const auto& normals = wall_projection_normals[vertex];
    std::vector<std::array<double, 3>> basis;
    basis.reserve(normals.size());
    for (auto normal : normals) {
      for (const auto& q : basis) {
        const double projection = normal[0] * q[0] +
                                  normal[1] * q[1] +
                                  normal[2] * q[2];
        for (int d = 0; d < dimension; ++d) {
          normal[static_cast<std::size_t>(d)] -=
              projection * q[static_cast<std::size_t>(d)];
        }
      }
      double length2 = 0.0;
      for (int d = 0; d < dimension; ++d) {
        length2 += normal[static_cast<std::size_t>(d)] *
                   normal[static_cast<std::size_t>(d)];
      }
      if (length2 <= 1.0e-20) {
        continue;
      }
      const double inverse_length = 1.0 / std::sqrt(length2);
      for (int d = 0; d < dimension; ++d) {
        normal[static_cast<std::size_t>(d)] *= inverse_length;
      }
      basis.push_back(normal);
    }
    std::array<std::array<double, 3>, 3> projection{};
    for (int row = 0; row < dimension; ++row) {
      projection[static_cast<std::size_t>(row)]
                [static_cast<std::size_t>(row)] = 1.0;
    }
    for (const auto& normal : basis) {
      for (int row = 0; row < dimension; ++row) {
        for (int column = 0; column < dimension; ++column) {
          projection[static_cast<std::size_t>(row)]
                    [static_cast<std::size_t>(column)] -=
              normal[static_cast<std::size_t>(row)] *
              normal[static_cast<std::size_t>(column)];
        }
      }
    }
    bool constrained = !basis.empty();
    for (int d = 0; d < dimension; ++d) {
      const auto component = static_cast<std::size_t>(d);
      if (wall_component_masks[vertex][component]) {
        projection[component].fill(0.0);
        constrained = true;
      }
    }
    return std::make_pair(projection, constrained);
  };

  auto apply_wall_velocity_constraints = [&](std::size_t vertex) {
    const auto [projection, constrained] =
        wall_velocity_projection(vertex);
    std::array<double, 3> unconstrained{};
    for (std::size_t component = 0;
         component < copy_components &&
         component < static_cast<std::size_t>(dimension);
         ++component) {
      unconstrained[component] =
          extended[vertex * target_components + component];
    }
    for (int row = 0; row < dimension; ++row) {
      double value = 0.0;
      for (int column = 0; column < dimension; ++column) {
        value += projection[static_cast<std::size_t>(row)]
                           [static_cast<std::size_t>(column)] *
                 unconstrained[static_cast<std::size_t>(column)];
      }
      extended[vertex * target_components +
               static_cast<std::size_t>(row)] = value;
    }
    return constrained;
  };

  WallCompatibleVelocityExtensionResult result;
  std::vector<std::uint8_t> assigned(vertex_count, 0u);
  std::vector<double> extension_distance(
      vertex_count, std::numeric_limits<double>::infinity());
  std::vector<std::size_t> frontier;
  frontier.reserve(vertex_count);
  for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
    if (active[vertex] == 0u ||
        !ownsVelocityExtensionVertex(mesh, vertex, comm)) {
      continue;
    }
    for (std::size_t c = 0; c < copy_components; ++c) {
      const double value =
          source_velocity[vertex * source_components + c];
      if (!std::isfinite(value)) {
        throw std::runtime_error(
            "wall-compatible level-set velocity extension found a non-finite source velocity on an exact trace/active seed");
      }
      extended[vertex * target_components + c] = value;
    }
    assigned[vertex] = 1u;
    // The input is oriented so the retained side is negative.  Retaining its
    // signed seed offset reconstructs distance from the interface: a wet seed
    // contributes the edge path minus its interior distance, while a promoted
    // dry trace seed contributes the path plus its exterior distance.
    extension_distance[vertex] = phi[vertex];
    frontier.push_back(vertex);
    if (row_diagnostics != nullptr) {
      double preview_speed2 = 0.0;
      for (std::size_t component = 0; component < copy_components;
           ++component) {
        const double value =
            extended[vertex * target_components + component];
        preview_speed2 += value * value;
      }
      VelocityExtensionGraphRowDiagnostic diagnostic{
          .local_vertex = static_cast<svmp::FE::GlobalIndex>(vertex),
          .global_vertex =
              velocityExtensionVertexGlobalIdentity(mesh, vertex),
          .disposition = VelocityExtensionRowDisposition::TraceSeed,
          .component_assignment = extension_component[vertex],
          .component_candidates = 1u,
          .band_layer = 0,
          .reconstruction_dimension = 1,
          .numerical_rank = 1,
          .assigned = true,
          .regression_attempted = false,
          .regression_accepted = false,
          .bounded_fallback_used = false,
          .condition_rejected = false,
          .coefficient_rejected = false,
          .wall_projected = false,
          .condition_estimate = 1.0,
          .proposed_coefficient_sum = 1.0,
          .proposed_coefficient_l1 = 1.0,
          .proposed_max_abs_coefficient = 1.0,
          .proposed_negative_weight_count = 0u,
          .proposed_max_negative_coefficient = 0.0,
          .coefficient_sum = 1.0,
          .coefficient_l1 = 1.0,
          .max_abs_coefficient = 1.0,
          .negative_weight_count = 0u,
          .max_negative_coefficient = 0.0,
          .constant_reproduction_error = 0.0,
          .max_tangential_linear_reproduction_error = 0.0,
          .extrapolation_distance = std::abs(extension_distance[vertex]),
          .dependency_max_speed = std::sqrt(preview_speed2),
          .preview_speed = std::sqrt(preview_speed2),
          .preview_amplification = preview_speed2 > 0.0 ? 1.0 : 0.0,
      };
      diagnostic.dependencies.push_back(VelocityExtensionGraphDependency{
          .local_vertex = static_cast<svmp::FE::GlobalIndex>(vertex),
          .global_vertex =
              velocityExtensionVertexGlobalIdentity(mesh, vertex),
          .coefficient = 1.0,
      });
      row_diagnostics->push_back(std::move(diagnostic));
    }
    if (constraint_rows != nullptr) {
      for (std::size_t component = 0; component < target_components;
           ++component) {
        svmp::FE::level_set::VelocityExtensionConstraintRow row{
            .vertex = static_cast<svmp::FE::GlobalIndex>(vertex),
            .component = static_cast<int>(component),
        };
        if (component < copy_components) {
          row.dependencies.push_back(
              svmp::FE::level_set::VelocityExtensionDependency{
                  .field = svmp::FE::level_set::
                      VelocityExtensionDependencyField::SourceVelocity,
                  .vertex = static_cast<svmp::FE::GlobalIndex>(vertex),
                  .component = static_cast<int>(component),
                  .coefficient = 1.0,
              });
        }
        constraint_rows->push_back(std::move(row));
      }
    }
  }
  frontier = synchronizeVelocityExtensionLayer(
      mesh,
      comm,
      std::span<const std::size_t>(frontier),
      target_components,
      assigned,
      extended,
      extension_distance);

  const auto velocity_magnitude = [&](std::size_t vertex) {
    double magnitude2 = 0.0;
    for (std::size_t component = 0; component < copy_components;
         ++component) {
      const double value =
          extended[vertex * target_components + component];
      if (!std::isfinite(value)) {
        throw std::runtime_error(
            "wall-compatible level-set velocity extension produced a non-finite value");
      }
      magnitude2 += value * value;
    }
    return std::sqrt(magnitude2);
  };
  std::vector<std::uint8_t> candidate_flag(vertex_count, 0u);
  // Every communicator rank participates in every layer exchange. A rank can
  // have an empty local frontier while a remote owner is still advancing the
  // same global BFS band toward its one-layer halo.
  for (int layer = 1; layer <= band_layers; ++layer) {
    std::vector<std::size_t> candidates;
    for (const auto source : frontier) {
      for (const auto candidate : neighbors[source]) {
        if (assigned[candidate] != 0u || active[candidate] != 0u ||
            !ownsVelocityExtensionVertex(mesh, candidate, comm) ||
            candidate_flag[candidate] != 0u) {
          continue;
        }
        candidate_flag[candidate] = 1u;
        candidates.push_back(candidate);
      }
    }
    std::sort(candidates.begin(), candidates.end());
    std::vector<std::vector<double>> candidate_values(
        candidates.size(), std::vector<double>(copy_components, 0.0));
    std::vector<double> candidate_distances(
        candidates.size(), std::numeric_limits<double>::infinity());
    std::vector<std::vector<std::pair<std::size_t, double>>>
        candidate_dependencies(candidates.size());
    std::vector<VelocityExtensionGraphRowDiagnostic> candidate_diagnostics(
        candidates.size());
    for (std::size_t index = 0; index < candidates.size(); ++index) {
      const auto vertex = candidates[index];
      const auto point = meshVertexPoint(coordinates, dimension, vertex);
      auto selected_component = kInvalidVelocityExtensionComponent;
      std::map<std::int64_t, double> component_geometric_distance;
      for (const auto source : neighbors[vertex]) {
        if (assigned[source] == 0u) {
          continue;
        }
        const auto source_component = extension_component[source];
        if (source_component == kInvalidVelocityExtensionComponent) {
          throw std::runtime_error(
              "wall-compatible level-set velocity extension encountered an assigned vertex without a component label");
        }
        const auto source_point =
            meshVertexPoint(coordinates, dimension, source);
        double distance2 = 0.0;
        for (int d = 0; d < dimension; ++d) {
          const double delta =
              source_point[static_cast<std::size_t>(d)] -
              point[static_cast<std::size_t>(d)];
          distance2 += delta * delta;
        }
        if (!(distance2 > 0.0) || !std::isfinite(distance2)) {
          continue;
        }
        const double geometric_distance =
            extension_distance[source] + std::sqrt(distance2);
        if (!std::isfinite(geometric_distance)) {
          throw std::runtime_error(
              "wall-compatible level-set velocity extension found a non-finite component distance");
        }
        const auto [position, inserted] =
            component_geometric_distance.emplace(source_component,
                                                 geometric_distance);
        if (!inserted) {
          position->second =
              std::min(position->second, geometric_distance);
        }
      }
      if (component_geometric_distance.empty()) {
        throw std::runtime_error(
            "wall-compatible level-set velocity extension encountered an unseeded graph layer");
      }
      if (component_geometric_distance.size() > 1u) {
        ++result.component_collision_vertices;
      }
      double selected_distance = std::numeric_limits<double>::infinity();
      for (const auto& [component, distance] :
           component_geometric_distance) {
        if (distance < selected_distance) {
          selected_component = component;
          selected_distance = distance;
        }
      }
      for (const auto& [component, distance] :
           component_geometric_distance) {
        if (component == selected_component) {
          continue;
        }
        const double tie_tolerance =
            64.0 * std::numeric_limits<double>::epsilon() *
            std::max({1.0, std::abs(selected_distance),
                      std::abs(distance)});
        if (std::abs(distance - selected_distance) <= tie_tolerance) {
          throw std::runtime_error(
              "wall-compatible level-set velocity extension encountered an unresolved equidistant active-component collision");
        }
      }
      double weight_sum = 0.0;
      std::array<std::array<double, 4>, 4> gradient_matrix{};
      std::array<double, 4> gradient_rhs{};
      for (const auto neighbor : neighbors[vertex]) {
        if (assigned[neighbor] == 0u ||
            extension_component[neighbor] != selected_component) {
          continue;
        }
        const auto neighbor_point =
            meshVertexPoint(coordinates, dimension, neighbor);
        std::array<double, 3> delta{};
        double distance2 = 0.0;
        for (int d = 0; d < dimension; ++d) {
          delta[static_cast<std::size_t>(d)] =
              neighbor_point[static_cast<std::size_t>(d)] -
              point[static_cast<std::size_t>(d)];
          distance2 += delta[static_cast<std::size_t>(d)] *
                       delta[static_cast<std::size_t>(d)];
        }
        if (!(distance2 > 0.0)) {
          continue;
        }
        const double weight = 1.0 / distance2;
        const double delta_phi = phi[neighbor] - phi[vertex];
        for (int row = 0; row < dimension; ++row) {
          gradient_rhs[static_cast<std::size_t>(row)] +=
              weight * delta[static_cast<std::size_t>(row)] * delta_phi;
          for (int column = 0; column < dimension; ++column) {
            gradient_matrix[static_cast<std::size_t>(row)]
                           [static_cast<std::size_t>(column)] +=
                weight * delta[static_cast<std::size_t>(row)] *
                delta[static_cast<std::size_t>(column)];
          }
        }
      }
      std::array<double, 4> gradient{};
      const bool gradient_available = solveSmallDenseSystem(
          gradient_matrix, gradient_rhs, dimension, gradient);
      double gradient_norm2 = 0.0;
      for (int d = 0; d < dimension; ++d) {
        gradient_norm2 += gradient[static_cast<std::size_t>(d)] *
                          gradient[static_cast<std::size_t>(d)];
      }
      std::vector<std::array<double, 3>> tangent_basis;
      if (gradient_available && gradient_norm2 > 1.0e-24) {
        const double inverse_gradient_norm = 1.0 / std::sqrt(gradient_norm2);
        std::array<double, 3> normal{};
        for (int d = 0; d < dimension; ++d) {
          normal[static_cast<std::size_t>(d)] =
              gradient[static_cast<std::size_t>(d)] * inverse_gradient_norm;
        }
        if (dimension == 2) {
          tangent_basis.push_back(
              {{-normal[1], normal[0], 0.0}});
        } else if (dimension == 3) {
          std::array<double, 3> axis{{1.0, 0.0, 0.0}};
          if (std::abs(normal[0]) > std::abs(normal[1])) {
            axis = std::abs(normal[1]) <= std::abs(normal[2])
                       ? std::array<double, 3>{{0.0, 1.0, 0.0}}
                       : std::array<double, 3>{{0.0, 0.0, 1.0}};
          } else if (std::abs(normal[0]) <= std::abs(normal[2])) {
            axis = {{1.0, 0.0, 0.0}};
          } else {
            axis = {{0.0, 0.0, 1.0}};
          }
          std::array<double, 3> tangent0{{
              normal[1] * axis[2] - normal[2] * axis[1],
              normal[2] * axis[0] - normal[0] * axis[2],
              normal[0] * axis[1] - normal[1] * axis[0]}};
          double tangent0_norm2 = tangent0[0] * tangent0[0] +
                                  tangent0[1] * tangent0[1] +
                                  tangent0[2] * tangent0[2];
          if (tangent0_norm2 > 1.0e-24) {
            const double inverse_tangent0_norm =
                1.0 / std::sqrt(tangent0_norm2);
            for (auto& value : tangent0) {
              value *= inverse_tangent0_norm;
            }
            tangent_basis.push_back(tangent0);
            tangent_basis.push_back({{
                normal[1] * tangent0[2] - normal[2] * tangent0[1],
                normal[2] * tangent0[0] - normal[0] * tangent0[2],
                normal[0] * tangent0[1] - normal[1] * tangent0[0]}});
          }
        }
      }

      const int regression_size =
          1 + static_cast<int>(tangent_basis.size());
      std::array<std::array<double, 4>, 4> regression_matrix{};
      struct WeightedRegressionNeighbor {
        std::size_t vertex{0u};
        double weight{0.0};
        std::array<double, 4> features{};
      };
      std::vector<WeightedRegressionNeighbor> regression_neighbors;
      for (const auto source : neighbors[vertex]) {
        if (assigned[source] == 0u ||
            extension_component[source] != selected_component) {
          continue;
        }
        const auto source_point =
            meshVertexPoint(coordinates, dimension, source);
        double distance2 = 0.0;
        for (int d = 0; d < dimension; ++d) {
          const double delta = point[static_cast<std::size_t>(d)] -
                               source_point[static_cast<std::size_t>(d)];
          distance2 += delta * delta;
        }
        if (!(distance2 > 0.0)) {
          continue;
        }
        const double level_set_change =
            std::abs(phi[vertex] - phi[source]);
        // Edges aligned with grad(phi) dominate the upwind extension; inverse
        // distance provides a deterministic fallback on flat/under-resolved
        // patches without a global nearest-component jump.
        const double weight =
            (level_set_change + 1.0e-12 * std::sqrt(distance2)) / distance2;
        weight_sum += weight;
        std::array<double, 4> features{};
        features[0] = 1.0;
        for (std::size_t tangent = 0;
             tangent < tangent_basis.size();
             ++tangent) {
          for (int d = 0; d < dimension; ++d) {
            features[tangent + 1u] +=
                (source_point[static_cast<std::size_t>(d)] -
                 point[static_cast<std::size_t>(d)]) *
                tangent_basis[tangent][static_cast<std::size_t>(d)];
          }
        }
        for (int row = 0; row < regression_size; ++row) {
          for (int column = 0; column < regression_size; ++column) {
            regression_matrix[static_cast<std::size_t>(row)]
                             [static_cast<std::size_t>(column)] +=
                weight * features[static_cast<std::size_t>(row)] *
                features[static_cast<std::size_t>(column)];
          }
        }
        regression_neighbors.push_back(WeightedRegressionNeighbor{
            .vertex = source,
            .weight = weight,
            .features = features,
        });
      }
      if (!(weight_sum > 0.0)) {
        throw std::runtime_error(
            "wall-compatible level-set velocity extension encountered an unseeded graph layer");
      }
      extension_component[vertex] = selected_component;
      std::array<double, 4> evaluation_rhs{};
      evaluation_rhs[0] = 1.0;
      std::array<double, 4> evaluation_weights{};
      const auto rank_condition = estimateSymmetricRankAndCondition(
          regression_matrix, regression_size);
      const double regression_condition = rank_condition.condition_estimate;
      if (std::isfinite(regression_condition)) {
        result.max_regression_condition = std::max(
            result.max_regression_condition, regression_condition);
      }
      ++result.regression_candidate_rows;
      const bool regression_available =
          rank_condition.numerical_rank == regression_size &&
          regression_condition <= kVelocityExtensionMaxRegressionCondition &&
          solveSmallDenseSystem(
          regression_matrix,
          evaluation_rhs,
          regression_size,
          evaluation_weights);
      std::vector<double> graph_coefficients;
      graph_coefficients.reserve(regression_neighbors.size());
      double regression_sum = 0.0;
      double regression_l1 = 0.0;
      double regression_max_abs = 0.0;
      double regression_max_negative = 0.0;
      std::size_t regression_negative_count = 0u;
      bool finite_coefficients = regression_available;
      for (const auto& neighbor : regression_neighbors) {
        double coefficient = 0.0;
        if (regression_available) {
          for (int feature = 0; feature < regression_size; ++feature) {
            coefficient +=
                evaluation_weights[static_cast<std::size_t>(feature)] *
                neighbor.weight *
                neighbor.features[static_cast<std::size_t>(feature)];
          }
        }
        finite_coefficients =
            finite_coefficients && std::isfinite(coefficient);
        regression_sum += coefficient;
        regression_l1 += std::abs(coefficient);
        regression_max_abs =
            std::max(regression_max_abs, std::abs(coefficient));
        regression_max_negative =
            std::max(regression_max_negative, std::max(0.0, -coefficient));
        regression_negative_count += coefficient < 0.0 ? 1u : 0u;
        graph_coefficients.push_back(coefficient);
      }
      const bool bounded_regression =
          finite_coefficients &&
          std::abs(regression_sum - 1.0) <=
              kVelocityExtensionRowTolerance &&
          regression_l1 <= 1.0 + kVelocityExtensionRowTolerance &&
          regression_max_abs <= 1.0 + kVelocityExtensionRowTolerance &&
          regression_max_negative <=
              kVelocityExtensionCoefficientTolerance;
      if (!bounded_regression) {
        ++result.bounded_fallback_rows;
        if (!regression_available) {
          ++result.condition_rejected_rows;
        } else {
          ++result.coefficient_rejected_rows;
        }
        graph_coefficients.clear();
        for (const auto& neighbor : regression_neighbors) {
          graph_coefficients.push_back(neighbor.weight / weight_sum);
        }
      } else {
        ++result.regression_accepted_rows;
      }

      double positive_sum = 0.0;
      for (auto& coefficient : graph_coefficients) {
        coefficient = std::max(0.0, coefficient);
        positive_sum += coefficient;
      }
      if (!(positive_sum > 0.0) || !std::isfinite(positive_sum)) {
        throw std::runtime_error(
            "wall-compatible level-set velocity extension could not construct a positive bounded graph row");
      }
      double row_sum = 0.0;
      double row_l1 = 0.0;
      double row_max_abs = 0.0;
      double row_max_negative = 0.0;
      std::size_t row_negative_count = 0u;
      for (auto& coefficient : graph_coefficients) {
        coefficient /= positive_sum;
        row_sum += coefficient;
        row_l1 += std::abs(coefficient);
        row_max_abs = std::max(row_max_abs, std::abs(coefficient));
        row_max_negative =
            std::max(row_max_negative, std::max(0.0, -coefficient));
        row_negative_count += coefficient < 0.0 ? 1u : 0u;
      }
      const double row_sum_error = std::abs(row_sum - 1.0);
      if (row_sum_error > kVelocityExtensionRowTolerance ||
          row_l1 > 1.0 + kVelocityExtensionRowTolerance ||
          row_max_abs > 1.0 + kVelocityExtensionRowTolerance ||
          row_max_negative > kVelocityExtensionCoefficientTolerance) {
        throw std::runtime_error(
            "wall-compatible level-set velocity extension constructed an unbounded graph row");
      }
      result.max_abs_graph_coefficient =
          std::max(result.max_abs_graph_coefficient, row_max_abs);
      result.max_graph_row_l1 =
          std::max(result.max_graph_row_l1, row_l1);
      result.max_graph_row_sum_error =
          std::max(result.max_graph_row_sum_error, row_sum_error);
      result.max_negative_graph_coefficient = std::max(
          result.max_negative_graph_coefficient, row_max_negative);
      result.max_constant_reproduction_error = std::max(
          result.max_constant_reproduction_error, row_sum_error);
      double maximum_linear_reproduction_error = 0.0;
      for (int feature = 1; feature < regression_size; ++feature) {
        double reproduction_error = 0.0;
        for (std::size_t neighbor_index = 0;
             neighbor_index < regression_neighbors.size();
             ++neighbor_index) {
          reproduction_error +=
              graph_coefficients[neighbor_index] *
              regression_neighbors[neighbor_index]
                  .features[static_cast<std::size_t>(feature)];
        }
        maximum_linear_reproduction_error = std::max(
            maximum_linear_reproduction_error,
            std::abs(reproduction_error));
      }
      result.max_linear_reproduction_error = std::max(
          result.max_linear_reproduction_error,
          maximum_linear_reproduction_error);

      auto& diagnostic = candidate_diagnostics[index];
      diagnostic.local_vertex =
          static_cast<svmp::FE::GlobalIndex>(vertex);
      diagnostic.global_vertex =
          velocityExtensionVertexGlobalIdentity(mesh, vertex);
      diagnostic.disposition = bounded_regression
          ? VelocityExtensionRowDisposition::Regression
          : VelocityExtensionRowDisposition::BoundedFallback;
      diagnostic.component_assignment = selected_component;
      diagnostic.component_candidates =
          component_geometric_distance.size();
      diagnostic.band_layer = layer;
      diagnostic.reconstruction_dimension = regression_size;
      diagnostic.numerical_rank = rank_condition.numerical_rank;
      diagnostic.assigned = true;
      diagnostic.regression_attempted = true;
      diagnostic.regression_accepted = bounded_regression;
      diagnostic.bounded_fallback_used = !bounded_regression;
      diagnostic.condition_rejected = !regression_available;
      diagnostic.coefficient_rejected =
          regression_available && !bounded_regression;
      diagnostic.condition_estimate = regression_condition;
      diagnostic.proposed_coefficient_sum = regression_sum;
      diagnostic.proposed_coefficient_l1 = regression_l1;
      diagnostic.proposed_max_abs_coefficient = regression_max_abs;
      diagnostic.proposed_negative_weight_count =
          regression_negative_count;
      diagnostic.proposed_max_negative_coefficient =
          regression_max_negative;
      diagnostic.coefficient_sum = row_sum;
      diagnostic.coefficient_l1 = row_l1;
      diagnostic.max_abs_coefficient = row_max_abs;
      diagnostic.negative_weight_count = row_negative_count;
      diagnostic.max_negative_coefficient = row_max_negative;
      diagnostic.constant_reproduction_error = row_sum_error;
      diagnostic.max_tangential_linear_reproduction_error =
          maximum_linear_reproduction_error;

      auto& dependencies = candidate_dependencies[index];
      dependencies.reserve(regression_neighbors.size());
      for (std::size_t neighbor_index = 0;
           neighbor_index < regression_neighbors.size(); ++neighbor_index) {
        const auto& neighbor = regression_neighbors[neighbor_index];
        const double coefficient = graph_coefficients[neighbor_index];
        dependencies.emplace_back(neighbor.vertex, coefficient);
        diagnostic.dependencies.push_back(VelocityExtensionGraphDependency{
            .local_vertex = static_cast<svmp::FE::GlobalIndex>(
                neighbor.vertex),
            .global_vertex = velocityExtensionVertexGlobalIdentity(
                mesh, neighbor.vertex),
            .coefficient = coefficient,
        });
        const auto neighbor_point =
            meshVertexPoint(coordinates, dimension, neighbor.vertex);
        double edge_distance2 = 0.0;
        for (int component = 0; component < dimension; ++component) {
          const double delta =
              point[static_cast<std::size_t>(component)] -
              neighbor_point[static_cast<std::size_t>(component)];
          edge_distance2 += delta * delta;
        }
        if (!(edge_distance2 > 0.0) ||
            !std::isfinite(extension_distance[neighbor.vertex])) {
          throw std::runtime_error(
              "wall-compatible level-set velocity extension found an invalid extrapolation path");
        }
        candidate_distances[index] = std::min(
            candidate_distances[index],
            extension_distance[neighbor.vertex] + std::sqrt(edge_distance2));
        for (std::size_t component = 0; component < copy_components;
             ++component) {
          candidate_values[index][component] +=
              coefficient *
              extended[neighbor.vertex * target_components + component];
        }
      }
    }
    for (std::size_t index = 0; index < candidates.size(); ++index) {
      const auto vertex = candidates[index];
      for (std::size_t c = 0; c < copy_components; ++c) {
        extended[vertex * target_components + c] =
            candidate_values[index][c];
      }
      assigned[vertex] = 1u;
      if (!std::isfinite(candidate_distances[index])) {
        throw std::runtime_error(
            "wall-compatible level-set velocity extension could not measure its extrapolation path");
      }
      extension_distance[vertex] = candidate_distances[index];
      auto& diagnostic = candidate_diagnostics[index];
      diagnostic.extrapolation_distance = extension_distance[vertex];
      result.max_extrapolation_distance = std::max(
          result.max_extrapolation_distance,
          std::max(0.0, extension_distance[vertex]));
      candidate_flag[vertex] = 0u;
      if (apply_wall_velocity_constraints(vertex)) {
        ++result.wall_projected_vertices;
        diagnostic.wall_projected = true;
      }
      double dependency_max_speed = 0.0;
      for (const auto& dependency : diagnostic.dependencies) {
        dependency_max_speed = std::max(
            dependency_max_speed,
            velocity_magnitude(
                static_cast<std::size_t>(dependency.local_vertex)));
      }
      diagnostic.dependency_max_speed = dependency_max_speed;
      diagnostic.preview_speed = velocity_magnitude(vertex);
      diagnostic.preview_amplification =
          diagnostic.preview_speed /
          std::max(diagnostic.dependency_max_speed, 1.0e-12);
      if (constraint_rows != nullptr) {
        const auto [projection, unused_constrained] =
            wall_velocity_projection(vertex);
        (void)unused_constrained;
        for (std::size_t row_component = 0;
             row_component < target_components; ++row_component) {
          svmp::FE::level_set::VelocityExtensionConstraintRow row{
              .vertex = static_cast<svmp::FE::GlobalIndex>(vertex),
              .component = static_cast<int>(row_component),
          };
          if (row_component < copy_components &&
              row_component < static_cast<std::size_t>(dimension)) {
            for (const auto& [dependency_vertex, graph_coefficient] :
                 candidate_dependencies[index]) {
              for (std::size_t dependency_component = 0;
                   dependency_component < copy_components &&
                   dependency_component <
                       static_cast<std::size_t>(dimension);
                   ++dependency_component) {
                row.dependencies.push_back(
                    svmp::FE::level_set::VelocityExtensionDependency{
                        .field = svmp::FE::level_set::
                            VelocityExtensionDependencyField::
                                ExtensionVelocity,
                        .vertex = static_cast<svmp::FE::GlobalIndex>(
                            dependency_vertex),
                        .component =
                            static_cast<int>(dependency_component),
                        .coefficient =
                            projection[row_component]
                                      [dependency_component] *
                            graph_coefficient,
                    });
              }
            }
          }
          constraint_rows->push_back(std::move(row));
        }
      }
      if (row_diagnostics != nullptr) {
        row_diagnostics->push_back(std::move(diagnostic));
      }
      ++result.extended_vertices;
    }
    synchronizeVelocityExtensionComponentLabels(
        mesh,
        comm,
        std::span<const std::size_t>(candidates),
        extension_component);
    frontier = synchronizeVelocityExtensionLayer(
        mesh,
        comm,
        std::span<const std::size_t>(candidates),
        target_components,
        assigned,
        extended,
        extension_distance);
  }

  for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
    if (active[vertex] == 0u && assigned[vertex] == 0u) {
      const bool owned = ownsVelocityExtensionVertex(mesh, vertex, comm);
      result.vertices_outside_band += owned ? 1u : 0u;
      if (constraint_rows != nullptr && owned) {
        for (std::size_t component = 0; component < target_components;
             ++component) {
          constraint_rows->push_back(
              svmp::FE::level_set::VelocityExtensionConstraintRow{
                  .vertex = static_cast<svmp::FE::GlobalIndex>(vertex),
                  .component = static_cast<int>(component),
              });
        }
      }
      if (row_diagnostics != nullptr && owned) {
        row_diagnostics->push_back(VelocityExtensionGraphRowDiagnostic{
            .local_vertex = static_cast<svmp::FE::GlobalIndex>(vertex),
            .global_vertex =
                velocityExtensionVertexGlobalIdentity(mesh, vertex),
            .disposition =
                VelocityExtensionRowDisposition::OutsideBandZero,
            .component_assignment = extension_component[vertex],
            .component_candidates = 0u,
            .band_layer = band_layers + 1,
            .reconstruction_dimension = 0,
            .numerical_rank = 0,
            .assigned = false,
        });
      }
    }
    if (active[vertex] != 0u) {
      result.max_seed_speed =
          std::max(result.max_seed_speed, velocity_magnitude(vertex));
      continue;
    }
    if (assigned[vertex] != 0u) {
      result.max_extended_speed =
          std::max(result.max_extended_speed, velocity_magnitude(vertex));
    }
    for (const auto& normal : wall_normals[vertex]) {
      double normal_velocity = 0.0;
      for (std::size_t c = 0;
           c < copy_components && c < static_cast<std::size_t>(dimension);
           ++c) {
        normal_velocity += extended[vertex * target_components + c] *
                           normal[c];
      }
      result.max_wall_normal_velocity =
          std::max(result.max_wall_normal_velocity,
                   std::abs(normal_velocity));
    }
  }
  const auto communicator_max = [&comm](double value) {
#ifdef MESH_HAS_MPI
    if (comm.is_parallel()) {
      double reduced = 0.0;
      MPI_Allreduce(&value, &reduced, 1, MPI_DOUBLE, MPI_MAX, comm.native());
      return reduced;
    }
#else
    (void)comm;
#endif
    return value;
  };
  result.max_wall_normal_velocity =
      communicator_max(result.max_wall_normal_velocity);
  result.max_regression_condition =
      communicator_max(result.max_regression_condition);
  result.max_abs_graph_coefficient =
      communicator_max(result.max_abs_graph_coefficient);
  result.max_graph_row_l1 = communicator_max(result.max_graph_row_l1);
  result.max_graph_row_sum_error =
      communicator_max(result.max_graph_row_sum_error);
  result.max_negative_graph_coefficient =
      communicator_max(result.max_negative_graph_coefficient);
  result.max_constant_reproduction_error =
      communicator_max(result.max_constant_reproduction_error);
  result.max_linear_reproduction_error =
      communicator_max(result.max_linear_reproduction_error);
  result.max_extrapolation_distance =
      communicator_max(result.max_extrapolation_distance);
  result.max_seed_speed = communicator_max(result.max_seed_speed);
  result.max_extended_speed = communicator_max(result.max_extended_speed);
  const double nonexpansive_tolerance =
      64.0 * std::numeric_limits<double>::epsilon() *
      std::max(1.0, result.max_seed_speed);
  if (result.max_extended_speed >
      result.max_seed_speed + nonexpansive_tolerance) {
    throw std::runtime_error(
        "wall-compatible level-set velocity extension violated its non-amplifying map invariant");
  }
  if (component_assignment != nullptr) {
    *component_assignment = std::move(extension_component);
  }
  if (row_diagnostics != nullptr) {
    std::sort(
        row_diagnostics->begin(),
        row_diagnostics->end(),
        [](const auto& lhs, const auto& rhs) {
          return lhs.global_vertex < rhs.global_vertex ||
                 (lhs.global_vertex == rhs.global_vertex &&
                  lhs.local_vertex < rhs.local_vertex);
        });
    const auto duplicate = std::adjacent_find(
        row_diagnostics->begin(),
        row_diagnostics->end(),
        [](const auto& lhs, const auto& rhs) {
          return lhs.local_vertex == rhs.local_vertex;
        });
    if (duplicate != row_diagnostics->end()) {
      throw std::runtime_error(
          "velocity-extension map produced duplicate owner row diagnostics");
    }
  }
  return result;
}

std::shared_ptr<const VelocityExtensionMapSnapshot>
buildVelocityExtensionMapSnapshot(
    const svmp::Mesh& mesh,
    const svmp::MeshComm& comm,
    VelocityExtensionMapRevision revision,
    std::span<const double> phi,
    std::span<const double> source_velocity,
    std::size_t source_components,
    std::span<const std::uint8_t> active,
    std::size_t target_components,
    std::size_t copy_components,
    int band_layers,
    bool enforce_wall_impermeability,
    std::span<const WallVelocityExtensionConstraint> wall_constraints)
{
  const auto expected_revision = velocityExtensionMapRevision(
      revision.mesh_geometry,
      revision.mesh_topology,
      revision.mesh_ownership,
      revision.mesh_numbering,
      revision.free_surface_geometry,
      phi,
      active);
  if (!revision.complete() || revision != expected_revision) {
    throw std::invalid_argument(
        "velocity-extension map build received a stale or incomplete revision identity");
  }

  std::vector<double> preview;
  std::vector<svmp::FE::level_set::VelocityExtensionConstraintRow> rows;
  std::vector<std::int64_t> component_assignment;
  std::vector<VelocityExtensionGraphRowDiagnostic> row_diagnostics;
  auto report = extendVelocityInLevelSetNormalBand(
      mesh,
      comm,
      phi,
      source_velocity,
      source_components,
      active,
      target_components,
      copy_components,
      band_layers,
      enforce_wall_impermeability,
      wall_constraints,
      preview,
      &rows,
      &component_assignment,
      &row_diagnostics);

  const double amplification =
      report.max_extended_speed / std::max(report.max_seed_speed, 1.0e-12);
  if (!std::isfinite(amplification) ||
      amplification > kVelocityExtensionMaxWetToDryAmplification) {
    throw std::runtime_error(
        "velocity-extension map preview exceeded its fixed amplification guard");
  }
  if (report.max_abs_graph_coefficient >
          1.0 + kVelocityExtensionRowTolerance ||
      report.max_graph_row_l1 > 1.0 + kVelocityExtensionRowTolerance ||
      report.max_graph_row_sum_error > kVelocityExtensionRowTolerance ||
      report.max_negative_graph_coefficient >
          kVelocityExtensionCoefficientTolerance) {
    throw std::runtime_error(
        "velocity-extension map failed its bounded-row acceptance guards");
  }

  return std::make_shared<const VelocityExtensionMapSnapshot>(
      revision,
      target_components,
      std::move(preview),
      std::move(rows),
      std::move(component_assignment),
      std::move(row_diagnostics),
      report,
      amplification);
}

// Compatibility overload for focused tests that predate production
// component-mask plumbing.  An empty label list means all boundary faces, as
// it did historically.  Production callers do not use this overload.
[[maybe_unused]] WallCompatibleVelocityExtensionResult
extendVelocityInLevelSetNormalBand(
    const svmp::Mesh& mesh,
    const svmp::MeshComm& comm,
    std::span<const double> phi,
    std::span<const double> source_velocity,
    std::size_t source_components,
    std::span<const std::uint8_t> active,
    std::size_t target_components,
    std::size_t copy_components,
    int band_layers,
    bool enforce_wall_impermeability,
    std::span<const svmp::label_t> wall_boundary_labels,
    std::vector<double>& extended)
{
  std::vector<WallVelocityExtensionConstraint> constraints;
  if (enforce_wall_impermeability) {
    if (wall_boundary_labels.empty()) {
      constraints.push_back(WallVelocityExtensionConstraint{
          .boundary_label = svmp::INVALID_LABEL,
          .project_boundary_normal = true});
    } else {
      constraints.reserve(wall_boundary_labels.size());
      for (const auto label : wall_boundary_labels) {
        constraints.push_back(WallVelocityExtensionConstraint{
            .boundary_label = label,
            .project_boundary_normal = true});
      }
    }
  }
  return extendVelocityInLevelSetNormalBand(
      mesh,
      comm,
      phi,
      source_velocity,
      source_components,
      active,
      target_components,
      copy_components,
      band_layers,
      enforce_wall_impermeability,
      std::span<const WallVelocityExtensionConstraint>(constraints),
      extended);
}

// Serial/unit-test convenience overload. Production callers must provide the
// active FE-system communicator explicitly so no collective can fall back to
// MPI_COMM_WORLD by accident.
[[maybe_unused]] WallCompatibleVelocityExtensionResult
extendVelocityInLevelSetNormalBand(
    const svmp::Mesh& mesh,
    std::span<const double> phi,
    std::span<const double> source_velocity,
    std::size_t source_components,
    std::span<const std::uint8_t> active,
    std::size_t target_components,
    std::size_t copy_components,
    int band_layers,
    bool enforce_wall_impermeability,
    std::span<const svmp::label_t> wall_boundary_labels,
    std::vector<double>& extended)
{
  return extendVelocityInLevelSetNormalBand(
      mesh,
      svmp::MeshComm::self(),
      phi,
      source_velocity,
      source_components,
      active,
      target_components,
      copy_components,
      band_layers,
      enforce_wall_impermeability,
      wall_boundary_labels,
      extended);
}

} // namespace application::core
