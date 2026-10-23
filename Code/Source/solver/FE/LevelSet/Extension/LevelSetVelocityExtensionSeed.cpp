/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "LevelSet/Extension/LevelSetVelocityExtensionSeed.h"

#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

#ifdef MESH_HAS_MPI
#include <mpi.h>
#endif

namespace svmp {
namespace FE {
namespace level_set {
namespace {

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

} // namespace level_set
} // namespace FE
} // namespace svmp
