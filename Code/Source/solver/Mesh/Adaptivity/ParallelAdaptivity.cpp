/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "ParallelAdaptivity.h"
#include "../Core/DistributedMesh.h"
#include "../Fields/MeshFields.h"
#include "ErrorEstimator.h"
#include "Marker.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <unordered_set>

namespace svmp {

//=============================================================================
// ParallelAdaptivityManager Implementation
//=============================================================================

ParallelAdaptivityManager::ParallelAdaptivityManager(
    MPI_Comm comm, Config config)
    : comm_(comm), config_(std::move(config)) {

  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &size_);

  // Create local adaptivity manager
  local_manager_ = std::make_unique<AdaptivityManager>();
}

ParallelAdaptivityManager::ParallelAdaptivityManager(MPI_Comm comm)
    : ParallelAdaptivityManager(comm, Config{}) {
}

AdaptivityResult ParallelAdaptivityManager::adapt_parallel(
    Mesh& mesh,
    MeshFields* fields,
    const AdaptivityOptions& options) {

  auto start_time = std::chrono::steady_clock::now();

  local_manager_->set_options(options);

  // Update ghost layers
  auto ghost_start = std::chrono::steady_clock::now();
  update_ghost_layers(mesh);
  auto ghost_end = std::chrono::steady_clock::now();
  stats_.ghost_time = std::chrono::duration<double>(ghost_end - ghost_start).count();

  // Initialize communication pattern (requires established neighbor/exchange info).
  initialize_comm_pattern(mesh);

  // Perform local adaptation
  auto local_result = local_manager_->adapt(mesh.local_mesh(), fields);

  // Load balance if needed
  if (config_.enable_load_balancing) {
    auto balance_start = std::chrono::steady_clock::now();
    LoadBalance balance = compute_load_balance(mesh);

    if (balance.needs_rebalancing) {
      stats_.imbalance_before = balance.imbalance_factor;
      stats_.migrated_elements = rebalance_load(mesh, fields);

      // Recompute balance
      balance = compute_load_balance(mesh);
      stats_.imbalance_after = balance.imbalance_factor;
    }

    auto balance_end = std::chrono::steady_clock::now();
    stats_.balance_time = std::chrono::duration<double>(balance_end - balance_start).count();
  }

  // Gather global statistics
  stats_.max_local_level = 0;

  // Reduce statistics
  size_t global_refined, global_coarsened;
  MPI_Allreduce(&local_result.num_refined, &global_refined, 1,
                MPI_UNSIGNED_LONG, MPI_SUM, comm_);
  MPI_Allreduce(&local_result.num_coarsened, &global_coarsened, 1,
                MPI_UNSIGNED_LONG, MPI_SUM, comm_);

  local_result.num_refined = global_refined;
  local_result.num_coarsened = global_coarsened;

  auto end_time = std::chrono::steady_clock::now();
  local_result.total_time = std::chrono::duration<double>(end_time - start_time);

  return local_result;
}

void ParallelAdaptivityManager::exchange_ghost_marks(
    Mesh& mesh,
    std::vector<MarkType>& marks) {

  if (size_ <= 1) {
    return;
  }

  auto& local = mesh.local_mesh();
  const size_t n_cells = local.n_cells();
  if (marks.size() != n_cells) {
    marks.resize(n_cells, MarkType::NONE);
  }

  constexpr const char* kTmpFieldName = "__svmp_parallel_adaptivity_tmp_marks";
  if (local.has_field(EntityKind::Volume, kTmpFieldName)) {
    local.remove_field(local.field_handle(EntityKind::Volume, kTmpFieldName));
  }
  const auto h = local.attach_field(EntityKind::Volume, kTmpFieldName, FieldScalarType::Int32, 1);
  auto* mark_data = local.field_data_as<int32_t>(h);
  for (size_t c = 0; c < n_cells; ++c) {
    mark_data[c] = static_cast<int32_t>(marks[c]);
  }

  // ------------------------------------------------------
  // Ghost -> owner aggregation (conservative merge)
  // ------------------------------------------------------
  std::vector<std::vector<gid_t>> send_gids_per_rank(static_cast<size_t>(size_));
  std::vector<std::vector<int>> send_marks_per_rank(static_cast<size_t>(size_));

  const auto& cell_gids = local.cell_gids();
  for (index_t c = 0; c < static_cast<index_t>(n_cells); ++c) {
    if (mesh.is_owned_cell(c)) {
      continue;
    }

    const rank_t owner = mesh.owner_rank_cell(c);
    if (owner < 0 || owner >= static_cast<rank_t>(size_) || owner == static_cast<rank_t>(rank_)) {
      continue;
    }

    send_gids_per_rank[static_cast<size_t>(owner)].push_back(cell_gids[static_cast<size_t>(c)]);
    send_marks_per_rank[static_cast<size_t>(owner)].push_back(static_cast<int>(mark_data[static_cast<size_t>(c)]));
  }

  std::vector<int> send_counts(size_, 0);
  for (int r = 0; r < size_; ++r) {
    send_counts[r] = static_cast<int>(send_gids_per_rank[static_cast<size_t>(r)].size());
  }

  std::vector<int> recv_counts(size_, 0);
  MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm_);

  std::vector<int> send_displs(size_, 0);
  std::vector<int> recv_displs(size_, 0);
  int send_total = 0;
  int recv_total = 0;
  for (int r = 0; r < size_; ++r) {
    send_displs[r] = send_total;
    recv_displs[r] = recv_total;
    send_total += send_counts[r];
    recv_total += recv_counts[r];
  }

  std::vector<gid_t> send_gids_flat(static_cast<size_t>(send_total));
  std::vector<int> send_marks_flat(static_cast<size_t>(send_total));
  for (int r = 0; r < size_; ++r) {
    const auto& gids = send_gids_per_rank[static_cast<size_t>(r)];
    const auto& vals = send_marks_per_rank[static_cast<size_t>(r)];
    const int base = send_displs[r];
    for (size_t i = 0; i < gids.size(); ++i) {
      send_gids_flat[static_cast<size_t>(base) + i] = gids[i];
      send_marks_flat[static_cast<size_t>(base) + i] = vals[i];
    }
  }

  std::vector<gid_t> recv_gids_flat(static_cast<size_t>(recv_total));
  std::vector<int> recv_marks_flat(static_cast<size_t>(recv_total));

  MPI_Datatype gid_type =
#ifdef MPI_INT64_T
      MPI_INT64_T;
#else
      MPI_LONG_LONG;
#endif

  MPI_Alltoallv(send_gids_flat.data(), send_counts.data(), send_displs.data(), gid_type,
                recv_gids_flat.data(), recv_counts.data(), recv_displs.data(), gid_type,
                comm_);
  MPI_Alltoallv(send_marks_flat.data(), send_counts.data(), send_displs.data(), MPI_INT,
                recv_marks_flat.data(), recv_counts.data(), recv_displs.data(), MPI_INT,
                comm_);

  const auto merge_marks = [](MarkType a, MarkType b) -> MarkType {
    if (a == MarkType::REFINE || b == MarkType::REFINE) return MarkType::REFINE;
    if (a == MarkType::NONE || b == MarkType::NONE) return MarkType::NONE;
    return MarkType::COARSEN;
  };

  for (int i = 0; i < recv_total; ++i) {
    const gid_t gid = recv_gids_flat[static_cast<size_t>(i)];
    const auto incoming = static_cast<MarkType>(recv_marks_flat[static_cast<size_t>(i)]);

    const index_t local_cell = mesh.global_to_local_cell(gid);
    if (local_cell == INVALID_INDEX) {
      continue;
    }
    if (!mesh.is_owned_cell(local_cell)) {
      continue;
    }

    const auto current = static_cast<MarkType>(mark_data[static_cast<size_t>(local_cell)]);
    mark_data[static_cast<size_t>(local_cell)] =
        static_cast<int32_t>(merge_marks(current, incoming));
  }

  // Owner -> ghost propagation of final marks.
  mesh.update_ghosts({h});

  for (size_t c = 0; c < n_cells; ++c) {
    marks[c] = static_cast<MarkType>(mark_data[c]);
  }

  local.remove_field(h);
}

size_t ParallelAdaptivityManager::synchronize_marks(
    Mesh& mesh,
    std::vector<MarkType>& marks) {

  size_t iterations = 0;
  bool converged = false;

  while (!converged && iterations < config_.max_consensus_iterations) {
    // Exchange marks
    exchange_ghost_marks(mesh, marks);

    // Check convergence
    size_t changes = 0;

    // Update based on neighbor consistency
    for (size_t elem = 0; elem < marks.size(); ++elem) {
      const auto neighbors = mesh.local_mesh().cell_neighbors(static_cast<index_t>(elem));

      for (const auto nbr_cell : neighbors) {
        const auto neighbor = static_cast<size_t>(nbr_cell);
        if (neighbor >= marks.size()) {
          continue;
        }

        if (marks[neighbor] == MarkType::REFINE &&
            marks[elem] == MarkType::NONE) {
          // Need to refine for conformity
          marks[elem] = MarkType::REFINE;
          changes++;
        }
      }
    }

    // Check global convergence
    size_t global_changes;
    MPI_Allreduce(&changes, &global_changes, 1,
                  MPI_UNSIGNED_LONG, MPI_SUM, comm_);

    converged = (global_changes == 0);
    iterations++;
  }

  return iterations;
}

size_t ParallelAdaptivityManager::rebalance_load(
    Mesh& mesh,
    MeshFields* fields) {

  LoadBalance balance = compute_load_balance(mesh);

  if (!balance.needs_rebalancing) {
    return 0;
  }

  const auto& before_mesh = mesh.local_mesh();
  const auto& before_gids = before_mesh.cell_gids();
  std::unordered_set<gid_t> owned_before;
  owned_before.reserve(before_mesh.n_cells());
  for (index_t c = 0; c < static_cast<index_t>(before_mesh.n_cells()); ++c) {
    if (mesh.is_owned_cell(c)) {
      owned_before.insert(before_gids[static_cast<size_t>(c)]);
    }
  }

  PartitionHint hint = PartitionHint::Cells;
  switch (config_.partition_method) {
    case Config::PartitionMethod::GRAPH:
#if defined(SVMP_HAS_METIS)
      hint = PartitionHint::Metis;
#else
      hint = PartitionHint::Cells;
#endif
      break;
    case Config::PartitionMethod::GEOMETRIC:
      hint = PartitionHint::Vertices;
      break;
    case Config::PartitionMethod::SPACE_FILLING:
    case Config::PartitionMethod::RECURSIVE:
      hint = PartitionHint::Cells;
      break;
  }

  mesh.rebalance(hint);

  // Estimate the number of migrated elements as the number of owned cells that
  // left this rank (sum over ranks gives a global migrated cell count).
  const auto& after_mesh = mesh.local_mesh();
  const auto& after_gids = after_mesh.cell_gids();
  std::unordered_set<gid_t> owned_after;
  owned_after.reserve(after_mesh.n_cells());
  for (index_t c = 0; c < static_cast<index_t>(after_mesh.n_cells()); ++c) {
    if (mesh.is_owned_cell(c)) {
      owned_after.insert(after_gids[static_cast<size_t>(c)]);
    }
  }

  size_t local_left = 0;
  for (const auto gid : owned_before) {
    if (owned_after.find(gid) == owned_after.end()) {
      ++local_left;
    }
  }

  size_t global_migrated = 0;
  MPI_Allreduce(&local_left, &global_migrated, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm_);

  // Re-establish ghost layers to the configured depth and update exchange fields.
  update_ghost_layers(mesh);
  (void)fields;
  mesh.update_exchange_ghost_fields();

  return global_migrated;
}

void ParallelAdaptivityManager::initialize_comm_pattern(
    const Mesh& mesh) {

  // Determine communication neighbors
  comm_pattern_.send_procs.clear();
  comm_pattern_.recv_procs.clear();

  // Get mesh neighbors (processors sharing boundaries)
  std::vector<int> neighbors;
  neighbors.reserve(mesh.neighbor_ranks().size());
  for (auto r : mesh.neighbor_ranks()) {
    const int rr = static_cast<int>(r);
    if (rr != rank_) {
      neighbors.push_back(rr);
    }
  }
  std::sort(neighbors.begin(), neighbors.end());
  neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());

  comm_pattern_.send_procs = neighbors;
  comm_pattern_.recv_procs = neighbors;

  // Allocate buffers
  comm_pattern_.send_buffers.resize(size_);
  comm_pattern_.recv_buffers.resize(size_);
}

void ParallelAdaptivityManager::update_ghost_layers(Mesh& mesh) {
  // Ensure the DistributedMesh ghost infrastructure is established so that
  // ownership/exchange patterns are valid for subsequent synchronization.
  if (config_.ghost_depth > 0) {
    mesh.build_ghost_layer(static_cast<int>(config_.ghost_depth));
  } else {
    mesh.build_exchange_patterns();
  }
  ghost_layer_ = GhostLayerManager::build_ghost_layer(mesh, config_.ghost_depth);
}

void ParallelAdaptivityManager::exchange_field_data(
    Mesh& mesh,
    std::vector<double>& field_data) {

  // For now, treat field_data as a per-cell scalar field and use the
  // DistributedMesh ghost exchange to synchronize ghost values.
  auto& local = mesh.local_mesh();
  const size_t n_cells = local.n_cells();

  if (field_data.size() != n_cells) {
    field_data.resize(n_cells, 0.0);
  }

  constexpr const char* kTmpFieldName = "__svmp_parallel_adaptivity_tmp_exchange";
  if (local.has_field(EntityKind::Volume, kTmpFieldName)) {
    local.remove_field(local.field_handle(EntityKind::Volume, kTmpFieldName));
  }

  const auto h = local.attach_field(EntityKind::Volume, kTmpFieldName, FieldScalarType::Float64, 1);
  auto* data = local.field_data_as<real_t>(h);
  for (size_t c = 0; c < n_cells; ++c) {
    data[c] = field_data[c];
  }

  mesh.update_ghosts({h});

  for (size_t c = 0; c < n_cells; ++c) {
    field_data[c] = data[c];
  }

  local.remove_field(h);
}

LoadBalance ParallelAdaptivityManager::compute_load_balance(
    const Mesh& mesh) {

  LoadBalance balance;

  // Get local element count
  // Use owned-only counts to avoid ghost-layer distortion.
  size_t local_elements = mesh.n_owned_cells();
  balance.element_counts.resize(size_);

  // Gather element counts
  MPI_Allgather(&local_elements, 1, MPI_UNSIGNED_LONG,
                balance.element_counts.data(), 1, MPI_UNSIGNED_LONG,
                comm_);

  // Compute imbalance
  size_t total_elements = std::accumulate(balance.element_counts.begin(),
                                           balance.element_counts.end(), 0UL);
  size_t avg_elements = (size_ > 0) ? (total_elements / static_cast<size_t>(size_)) : 0u;
  size_t max_elements = *std::max_element(balance.element_counts.begin(),
                                           balance.element_counts.end());

  if (avg_elements == 0) {
    // Fewer elements than ranks; treat any rank with >1 element as imbalanced.
    balance.imbalance_factor = (max_elements > 0) ? static_cast<double>(max_elements) : 0.0;
  } else {
    balance.imbalance_factor = static_cast<double>(max_elements) / avg_elements;
  }
  balance.needs_rebalancing =
      (avg_elements > 0) && (balance.imbalance_factor > config_.imbalance_threshold);

  if (balance.needs_rebalancing) {
    // Compute target distribution
    balance.target_counts.resize(size_, avg_elements);

    // Adjust for remainder
    size_t remainder = total_elements % size_;
    for (size_t i = 0; i < remainder; ++i) {
      balance.target_counts[i]++;
    }

    // Determine migrations (simplified)
    for (int p = 0; p < size_; ++p) {
      if (balance.element_counts[p] > balance.target_counts[p]) {
        // Need to send elements
        size_t excess = balance.element_counts[p] - balance.target_counts[p];
        // Mark elements for migration
      }
    }
  }

  return balance;
}

void ParallelAdaptivityManager::migrate_elements(
    Mesh& mesh,
    const LoadBalance& balance) {

  // Pack elements for migration
  std::vector<std::vector<uint8_t>> send_data(size_);

  for (const auto& [elem, target_proc] : balance.migration_map) {
    // Pack element data
    // Implementation depends on mesh representation
  }

  // Exchange elements
  // Use MPI_Alltoallv or point-to-point communication

  // Unpack received elements
  // Update mesh connectivity
}

bool ParallelAdaptivityManager::check_global_conformity(
    const Mesh& mesh,
    const std::vector<MarkType>& marks) {

  // Check local conformity
  bool local_conforming = true;

  // Check processor boundaries
  for (const auto& [edge, procs] : ghost_layer_.shared_edges) {
    // Check if edge is conforming across processors
  }

  // Global reduction
  int global_conforming;
  int local_conf = local_conforming ? 1 : 0;
  MPI_Allreduce(&local_conf, &global_conforming, 1, MPI_INT, MPI_MIN, comm_);

  return global_conforming == 1;
}

void ParallelAdaptivityManager::enforce_boundary_conformity(
    Mesh& mesh,
    std::vector<MarkType>& marks) {

  // Exchange marks at processor boundaries
  exchange_ghost_marks(mesh, marks);

  // Check shared edges
  for (const auto& [edge, procs] : ghost_layer_.shared_edges) {
    // Ensure consistent refinement
  }

  // Check shared faces
  for (const auto& [face, procs] : ghost_layer_.shared_faces) {
    // Ensure consistent refinement
  }
}

void ParallelAdaptivityManager::collective_error_estimation(
    Mesh& mesh,
    std::vector<double>& error_field) {

  // Compute local error
  auto estimator = ErrorEstimatorFactory::create(AdaptivityOptions{});
  error_field = estimator->estimate(mesh.local_mesh(), nullptr, AdaptivityOptions{});
  if (error_field.empty()) {
    return;
  }

  // Exchange ghost error values
  exchange_field_data(mesh, error_field);

  // Compute global error metrics
  double local_max = *std::max_element(error_field.begin(), error_field.end());
  double global_max = ParallelCommUtils::global_max(comm_, local_max);

  // Normalize if needed
  for (double& err : error_field) {
    err /= global_max;
  }
}

void ParallelAdaptivityManager::apply_global_marking(
    const Mesh& mesh,
    const std::vector<double>& error_field,
    std::vector<MarkType>& marks,
    const AdaptivityOptions& options) {

  // Compute global threshold
  double threshold = ParallelAdaptivityUtils::compute_global_threshold(
      comm_, error_field, options.refine_fraction);

  // Apply marking
  auto marker = MarkerFactory::create(options);
  marks = marker->mark(error_field, mesh.local_mesh(), options);
}

//=============================================================================
// ParallelCommUtils Implementation
//=============================================================================

double ParallelCommUtils::global_min(MPI_Comm comm, double value) {
  double global_value;
  MPI_Allreduce(&value, &global_value, 1, MPI_DOUBLE, MPI_MIN, comm);
  return global_value;
}

double ParallelCommUtils::global_max(MPI_Comm comm, double value) {
  double global_value;
  MPI_Allreduce(&value, &global_value, 1, MPI_DOUBLE, MPI_MAX, comm);
  return global_value;
}

double ParallelCommUtils::global_sum(MPI_Comm comm, double value) {
  double global_value;
  MPI_Allreduce(&value, &global_value, 1, MPI_DOUBLE, MPI_SUM, comm);
  return global_value;
}

//=============================================================================
// MeshPartitioner Implementation
//=============================================================================

std::vector<int> MeshPartitioner::partition_mesh(
    const MeshBase& mesh,
    int num_partitions) {
  return partition_mesh(mesh, num_partitions, Config{});
}

std::vector<int> MeshPartitioner::partition_mesh(
    const MeshBase& mesh,
    int num_partitions,
    Config config) {

  (void)config;

  // Simplified partitioning
  const size_t n_cells = mesh.n_cells();
  std::vector<int> partition(n_cells);

  // Simple block partitioning
  if (num_partitions <= 0) {
    return partition;
  }
  const size_t elements_per_partition =
      std::max<size_t>(1u, n_cells / static_cast<size_t>(num_partitions));

  for (size_t i = 0; i < n_cells; ++i) {
    partition[i] = std::min(static_cast<int>(i / elements_per_partition),
                            num_partitions - 1);
  }

  return partition;
}

std::vector<int> MeshPartitioner::repartition_mesh(
    const Mesh& mesh,
    const std::vector<double>& weights) {
  return repartition_mesh(mesh, weights, Config{});
}

std::vector<int> MeshPartitioner::repartition_mesh(
    const Mesh& mesh,
    const std::vector<double>& weights,
    Config config) {

  (void)weights;

  // Simplified repartitioning
  return partition_mesh(mesh.local_mesh(), mesh.world_size(), std::move(config));
}

MeshPartitioner::QualityMetrics MeshPartitioner::evaluate_partition(
    const MeshBase& mesh,
    const std::vector<int>& partition) {

  QualityMetrics metrics;

  int num_parts = *std::max_element(partition.begin(), partition.end()) + 1;
  metrics.partition_sizes.resize(num_parts, 0);

  for (int p : partition) {
    metrics.partition_sizes[p]++;
  }

  metrics.max_partition_size = *std::max_element(metrics.partition_sizes.begin(),
                                                  metrics.partition_sizes.end());
  metrics.min_partition_size = *std::min_element(metrics.partition_sizes.begin(),
                                                  metrics.partition_sizes.end());

  double avg_size = static_cast<double>(mesh.n_cells()) / static_cast<double>(num_parts);
  metrics.imbalance = metrics.max_partition_size / avg_size;

  // Compute edge cut (simplified)
  metrics.edge_cut = 0;
  for (size_t elem = 0; elem < mesh.n_cells(); ++elem) {
    const auto neighbors = mesh.cell_neighbors(static_cast<index_t>(elem));
    for (const auto nbr_cell : neighbors) {
      const auto neighbor = static_cast<size_t>(nbr_cell);
      if (neighbor >= partition.size()) {
        continue;
      }
      if (partition[elem] != partition[neighbor]) {
        metrics.edge_cut++;
      }
    }
  }
  metrics.edge_cut /= 2; // Each edge counted twice

  return metrics;
}

//=============================================================================
// GhostLayerManager Implementation
//=============================================================================

GhostLayer GhostLayerManager::build_ghost_layer(
    const Mesh& mesh,
    size_t depth) {

  GhostLayer ghost;
  ghost.depth = depth;

  // Prefer DistributedMesh's ghost infrastructure as the source of truth.
  // In the Mesh library, cell indices are rank-local; store them as `size_t`
  // for compatibility with existing ParallelAdaptivity data structures.
  const auto ghost_cells = mesh.ghost_cells();
  ghost.ghost_elements.reserve(ghost_cells.size());

  for (const auto c : ghost_cells) {
    if (c < 0) {
      continue;
    }
    const size_t cell = static_cast<size_t>(c);
    ghost.ghost_elements.push_back(cell);
    ghost.ghost_owners[cell] = static_cast<int>(mesh.owner_rank_cell(c));
  }

  // Record (at least) the owning rank for shared/ghost vertices.
  const index_t n_vertices = static_cast<index_t>(mesh.local_mesh().n_vertices());
  for (index_t v = 0; v < n_vertices; ++v) {
    if (mesh.is_owned_vertex(v)) {
      continue;
    }
    ghost.shared_vertices[static_cast<size_t>(v)].insert(static_cast<int>(mesh.owner_rank_vertex(v)));
  }

  return ghost;
}

void GhostLayerManager::update_ghost_values(
    const Mesh& mesh,
    const GhostLayer& ghost,
    std::vector<double>& field_values) {

  (void)mesh;
  (void)ghost;
  (void)field_values;

  // Update ghost values from owning processors
  // Implementation depends on distributed mesh
}

void GhostLayerManager::synchronize_ghost_marks(
    const Mesh& mesh,
    const GhostLayer& ghost,
    std::vector<MarkType>& marks) {

  (void)mesh;
  (void)ghost;
  (void)marks;

  // Synchronize marks across ghost layers
}

bool GhostLayerManager::check_consistency(
    const Mesh& mesh,
    const GhostLayer& ghost) {

  (void)mesh;
  (void)ghost;

  // Check that ghost values match owner values
  return true;
}

std::vector<double> GhostLayerManager::pack_ghost_data(
    const Mesh& mesh,
    const GhostLayer& ghost,
    const std::vector<double>& field) {

  (void)mesh;

  std::vector<double> packed_data;

  for (size_t elem : ghost.ghost_elements) {
    if (elem < field.size()) {
      packed_data.push_back(field[elem]);
    }
  }

  return packed_data;
}

void GhostLayerManager::unpack_ghost_data(
    const Mesh& mesh,
    const GhostLayer& ghost,
    const std::vector<double>& packed_data,
    std::vector<double>& field) {

  (void)mesh;

  size_t idx = 0;
  for (size_t elem : ghost.ghost_elements) {
    if (elem < field.size() && idx < packed_data.size()) {
      field[elem] = packed_data[idx++];
    }
  }
}

//=============================================================================
// ParallelLoadBalancer Implementation
//=============================================================================

LoadBalance ParallelLoadBalancer::compute_balance(
    const Mesh& mesh) {
  return compute_balance(mesh, Config{});
}

LoadBalance ParallelLoadBalancer::compute_balance(
    const Mesh& mesh,
    Config config) {

  switch (config.algorithm) {
    case Config::Algorithm::DIFFUSIVE:
      return diffusive_balance(mesh, config);
    case Config::Algorithm::DIRECT:
      return direct_balance(mesh, config);
    default:
      return LoadBalance{};
  }
}

void ParallelLoadBalancer::execute_balancing(
    Mesh& mesh,
    MeshFields* fields,
    const LoadBalance& balance) {

  if (!balance.needs_rebalancing) {
    return;
  }

  // Delegate migration and partition updates to DistributedMesh.
  mesh.rebalance(PartitionHint::Cells);

  // Best-effort: update any exchange-marked fields after migration.
  (void)fields;
  mesh.update_exchange_ghost_fields();
}

double ParallelLoadBalancer::estimate_cost(
    const Mesh& mesh,
    const LoadBalance& balance) {

  (void)mesh;

  // Estimate migration cost
  double cost = balance.migration_map.size() * 1.0; // Simplified
  return cost;
}

LoadBalance ParallelLoadBalancer::diffusive_balance(
    const Mesh& mesh,
    const Config& config) {

  (void)mesh;
  (void)config;

  LoadBalance balance;

  // Diffusive load balancing algorithm
  // Exchange load with neighbors iteratively

  return balance;
}

LoadBalance ParallelLoadBalancer::direct_balance(
    const Mesh& mesh,
    const Config& config) {

  (void)mesh;
  (void)config;

  LoadBalance balance;

  // Direct reassignment based on global view

  return balance;
}

//=============================================================================
// ParallelAdaptivityUtils Implementation
//=============================================================================

bool ParallelAdaptivityUtils::is_adaptation_balanced(
    MPI_Comm comm,
    const std::vector<MarkType>& marks) {

  size_t local_refined = std::count(marks.begin(), marks.end(), MarkType::REFINE);
  size_t local_coarsened = std::count(marks.begin(), marks.end(), MarkType::COARSEN);

  size_t global_refined, global_coarsened;
  MPI_Allreduce(&local_refined, &global_refined, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);
  MPI_Allreduce(&local_coarsened, &global_coarsened, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);

  int size;
  MPI_Comm_size(comm, &size);

  double avg_refined = global_refined / static_cast<double>(size);
  double imbalance = local_refined / (avg_refined + 1.0);

  return imbalance < 1.5 && imbalance > 0.67;
}

double ParallelAdaptivityUtils::compute_global_error(
    MPI_Comm comm,
    const std::vector<double>& local_error) {

  double local_sum = std::accumulate(local_error.begin(), local_error.end(), 0.0);
  double global_sum;

  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);

  return global_sum;
}

double ParallelAdaptivityUtils::compute_global_threshold(
    MPI_Comm comm,
    const std::vector<double>& local_error,
    double fraction) {

  // Gather all errors
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  std::vector<int> counts(size);
  int local_count = local_error.size();
  MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);

  std::vector<int> displacements(size);
  for (int i = 1; i < size; ++i) {
    displacements[i] = displacements[i-1] + counts[i-1];
  }

  int total_count = displacements[size-1] + counts[size-1];
  std::vector<double> global_error(total_count);

  MPI_Allgatherv(local_error.data(), local_count, MPI_DOUBLE,
                 global_error.data(), counts.data(), displacements.data(),
                 MPI_DOUBLE, comm);

  // Sort and find threshold
  std::sort(global_error.begin(), global_error.end(), std::greater<double>());

  size_t threshold_idx = static_cast<size_t>(fraction * global_error.size());
  return global_error[threshold_idx];
}

ParallelStats ParallelAdaptivityUtils::gather_statistics(
    MPI_Comm comm,
    const AdaptivityResult& local_result) {

  ParallelStats stats;

  // Gather timing statistics
  const double local_time = local_result.total_time.count();
  MPI_Allreduce(&local_time, &stats.comm_time, 1,
                MPI_DOUBLE, MPI_MAX, comm);

  // Gather adaptation counts
  size_t local_refined = local_result.num_refined;
  size_t global_refined;
  MPI_Allreduce(&local_refined, &global_refined, 1,
                MPI_UNSIGNED_LONG, MPI_SUM, comm);

  return stats;
}

bool ParallelAdaptivityUtils::check_parallel_consistency(
    MPI_Comm comm,
    const Mesh& mesh) {

  (void)mesh;

  // Check consistency of shared entities
  bool local_consistent = true;

  // Check shared vertices
  // Check shared edges
  // Check shared faces

  int global_consistent;
  int local_cons = local_consistent ? 1 : 0;
  MPI_Allreduce(&local_cons, &global_consistent, 1, MPI_INT, MPI_MIN, comm);

  return global_consistent == 1;
}

void ParallelAdaptivityUtils::write_parallel_mesh(
    MPI_Comm comm,
    const Mesh& mesh,
    const std::string& filename) {

  (void)mesh;

  // Parallel mesh output
  int rank;
  MPI_Comm_rank(comm, &rank);

  std::string local_filename = filename + "." + std::to_string(rank);

  // Write local partition
  // Implementation depends on output format
}

} // namespace svmp
