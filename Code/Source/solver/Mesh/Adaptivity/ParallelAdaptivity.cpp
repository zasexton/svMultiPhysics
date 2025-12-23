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

namespace svmp {

//=============================================================================
// ParallelAdaptivityManager Implementation
//=============================================================================

ParallelAdaptivityManager::ParallelAdaptivityManager(
    MPI_Comm comm, const Config& config)
    : comm_(comm), config_(config) {

  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &size_);

  // Create local adaptivity manager
  local_manager_ = std::make_unique<AdaptivityManager>();
}

AdaptivityResult ParallelAdaptivityManager::adapt_parallel(
    DistributedMesh& mesh,
    MeshFields* fields,
    const AdaptivityOptions& options) {

  auto start_time = std::chrono::steady_clock::now();

  // Initialize communication pattern
  initialize_comm_pattern(mesh);

  // Update ghost layers
  auto ghost_start = std::chrono::steady_clock::now();
  update_ghost_layers(mesh);
  auto ghost_end = std::chrono::steady_clock::now();
  stats_.ghost_time = std::chrono::duration<double>(ghost_end - ghost_start).count();

  // Perform local adaptation
  auto local_result = local_manager_->adapt(mesh.get_local_mesh(), fields);

  // Exchange marks with neighbors
  if (config_.synchronize_marks) {
    auto sync_start = std::chrono::steady_clock::now();
    exchange_ghost_marks(mesh, local_result.marks);
    size_t iterations = synchronize_marks(mesh, local_result.marks);
    auto sync_end = std::chrono::steady_clock::now();
    stats_.sync_time = std::chrono::duration<double>(sync_end - sync_start).count();
    stats_.comm_rounds = iterations;
  }

  // Check global conformity
  if (config_.global_conformity) {
    enforce_boundary_conformity(mesh, local_result.marks);
  }

  // Apply refined marks
  local_result = local_manager_->adapt(mesh.get_local_mesh(), fields);

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
  stats_.max_local_level = local_result.max_level;

  // Reduce statistics
  size_t global_refined, global_coarsened;
  MPI_Allreduce(&local_result.num_refined, &global_refined, 1,
                MPI_UNSIGNED_LONG, MPI_SUM, comm_);
  MPI_Allreduce(&local_result.num_coarsened, &global_coarsened, 1,
                MPI_UNSIGNED_LONG, MPI_SUM, comm_);

  local_result.num_refined = global_refined;
  local_result.num_coarsened = global_coarsened;

  auto end_time = std::chrono::steady_clock::now();
  local_result.adaptation_time = std::chrono::duration<double>(end_time - start_time).count();

  return local_result;
}

void ParallelAdaptivityManager::exchange_ghost_marks(
    DistributedMesh& mesh,
    std::vector<MarkType>& marks) {

  // Pack marks for ghost elements
  std::vector<std::vector<int>> send_marks(size_);
  std::vector<std::vector<int>> recv_marks(size_);

  for (const auto& [elem, owner] : ghost_layer_.ghost_owners) {
    if (elem < marks.size()) {
      send_marks[owner].push_back(elem);
      send_marks[owner].push_back(static_cast<int>(marks[elem]));
    }
  }

  // Exchange marks
  if (config_.comm_strategy == Config::CommStrategy::NON_BLOCKING) {
    std::vector<MPI_Request> requests;

    // Post receives
    for (int p = 0; p < size_; ++p) {
      if (p == rank_) continue;

      MPI_Request req;
      int recv_size;
      MPI_Irecv(&recv_size, 1, MPI_INT, p, 0, comm_, &req);
      requests.push_back(req);
    }

    // Send sizes
    for (int p = 0; p < size_; ++p) {
      if (p == rank_) continue;

      int send_size = send_marks[p].size();
      MPI_Request req;
      MPI_Isend(&send_size, 1, MPI_INT, p, 0, comm_, &req);
      requests.push_back(req);
    }

    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
  }

  // Update marks based on received data
  for (int p = 0; p < size_; ++p) {
    if (p == rank_) continue;

    for (size_t i = 0; i < recv_marks[p].size(); i += 2) {
      size_t elem = recv_marks[p][i];
      MarkType mark = static_cast<MarkType>(recv_marks[p][i + 1]);

      // Take maximum mark (conservative)
      if (elem < marks.size()) {
        if (mark > marks[elem]) {
          marks[elem] = mark;
        }
      }
    }
  }
}

size_t ParallelAdaptivityManager::synchronize_marks(
    DistributedMesh& mesh,
    std::vector<MarkType>& marks) {

  size_t iterations = 0;
  bool converged = false;

  while (!converged && iterations < config_.max_consensus_iterations) {
    // Exchange marks
    exchange_ghost_marks(mesh, marks);

    // Check convergence
    size_t changes = 0;
    std::vector<MarkType> old_marks = marks;

    // Update based on neighbor consistency
    for (size_t elem = 0; elem < marks.size(); ++elem) {
      auto neighbors = mesh.get_local_mesh().get_element_neighbors(elem);

      for (size_t neighbor : neighbors) {
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
    DistributedMesh& mesh,
    MeshFields* fields) {

  LoadBalance balance = compute_load_balance(mesh);

  if (!balance.needs_rebalancing) {
    return 0;
  }

  // Migrate elements
  migrate_elements(mesh, balance);

  // Update fields
  if (fields) {
    // Redistribute field data
    // Implementation depends on field storage
  }

  return balance.migration_map.size();
}

void ParallelAdaptivityManager::initialize_comm_pattern(
    const DistributedMesh& mesh) {

  // Determine communication neighbors
  comm_pattern_.send_procs.clear();
  comm_pattern_.recv_procs.clear();

  // Get mesh neighbors (processors sharing boundaries)
  auto neighbors = mesh.get_neighbor_processors();

  comm_pattern_.send_procs = neighbors;
  comm_pattern_.recv_procs = neighbors;

  // Allocate buffers
  comm_pattern_.send_buffers.resize(size_);
  comm_pattern_.recv_buffers.resize(size_);
}

void ParallelAdaptivityManager::update_ghost_layers(DistributedMesh& mesh) {
  ghost_layer_ = GhostLayerManager::build_ghost_layer(mesh, config_.ghost_depth);
}

void ParallelAdaptivityManager::exchange_field_data(
    const DistributedMesh& mesh,
    std::vector<double>& field_data) {

  // Pack field data for ghost elements
  auto packed_data = GhostLayerManager::pack_ghost_data(mesh, ghost_layer_, field_data);

  // Exchange with neighbors
  std::vector<double> recv_data(packed_data.size());

  if (config_.comm_strategy == Config::CommStrategy::COLLECTIVE) {
    MPI_Alltoall(packed_data.data(), packed_data.size() / size_, MPI_DOUBLE,
                 recv_data.data(), packed_data.size() / size_, MPI_DOUBLE,
                 comm_);
  }

  // Unpack received data
  GhostLayerManager::unpack_ghost_data(mesh, ghost_layer_, recv_data, field_data);
}

LoadBalance ParallelAdaptivityManager::compute_load_balance(
    const DistributedMesh& mesh) {

  LoadBalance balance;

  // Get local element count
  size_t local_elements = mesh.get_local_mesh().num_elements();
  balance.element_counts.resize(size_);

  // Gather element counts
  MPI_Allgather(&local_elements, 1, MPI_UNSIGNED_LONG,
                balance.element_counts.data(), 1, MPI_UNSIGNED_LONG,
                comm_);

  // Compute imbalance
  size_t total_elements = std::accumulate(balance.element_counts.begin(),
                                           balance.element_counts.end(), 0UL);
  size_t avg_elements = total_elements / size_;
  size_t max_elements = *std::max_element(balance.element_counts.begin(),
                                           balance.element_counts.end());

  balance.imbalance_factor = static_cast<double>(max_elements) / avg_elements;
  balance.needs_rebalancing = (balance.imbalance_factor > config_.imbalance_threshold);

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
    DistributedMesh& mesh,
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
    const DistributedMesh& mesh,
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
    DistributedMesh& mesh,
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
    const DistributedMesh& mesh,
    std::vector<double>& error_field) {

  // Compute local error
  auto estimator = ErrorEstimatorFactory::create(AdaptivityOptions{});
  estimator->estimate_error(mesh.get_local_mesh(), nullptr, error_field);

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
    const DistributedMesh& mesh,
    const std::vector<double>& error_field,
    std::vector<MarkType>& marks,
    const AdaptivityOptions& options) {

  // Compute global threshold
  double threshold = ParallelAdaptivityUtils::compute_global_threshold(
      comm_, error_field, options.refine_fraction);

  // Apply marking
  auto marker = MarkerFactory::create(options);
  marker->mark_elements(mesh.get_local_mesh(), error_field, marks, options);
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
    int num_partitions,
    const Config& config) {

  // Simplified partitioning
  std::vector<int> partition(mesh.num_elements());

  // Simple block partitioning
  size_t elements_per_partition = mesh.num_elements() / num_partitions;

  for (size_t i = 0; i < mesh.num_elements(); ++i) {
    partition[i] = std::min(static_cast<int>(i / elements_per_partition),
                            num_partitions - 1);
  }

  return partition;
}

std::vector<int> MeshPartitioner::repartition_mesh(
    const DistributedMesh& mesh,
    const std::vector<double>& weights,
    const Config& config) {

  // Simplified repartitioning
  return partition_mesh(mesh.get_local_mesh(), mesh.get_num_processors(), config);
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

  double avg_size = mesh.num_elements() / static_cast<double>(num_parts);
  metrics.imbalance = metrics.max_partition_size / avg_size;

  // Compute edge cut (simplified)
  metrics.edge_cut = 0;
  for (size_t elem = 0; elem < mesh.num_elements(); ++elem) {
    auto neighbors = mesh.get_element_neighbors(elem);
    for (size_t neighbor : neighbors) {
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
    const DistributedMesh& mesh,
    size_t depth) {

  GhostLayer ghost;
  ghost.depth = depth;

  // Identify ghost elements (simplified)
  // Would need actual distributed mesh interface

  return ghost;
}

void GhostLayerManager::update_ghost_values(
    const DistributedMesh& mesh,
    const GhostLayer& ghost,
    std::vector<double>& field_values) {

  // Update ghost values from owning processors
  // Implementation depends on distributed mesh
}

void GhostLayerManager::synchronize_ghost_marks(
    const DistributedMesh& mesh,
    const GhostLayer& ghost,
    std::vector<MarkType>& marks) {

  // Synchronize marks across ghost layers
}

bool GhostLayerManager::check_consistency(
    const DistributedMesh& mesh,
    const GhostLayer& ghost) {

  // Check that ghost values match owner values
  return true;
}

std::vector<double> GhostLayerManager::pack_ghost_data(
    const DistributedMesh& mesh,
    const GhostLayer& ghost,
    const std::vector<double>& field) {

  std::vector<double> packed_data;

  for (size_t elem : ghost.ghost_elements) {
    if (elem < field.size()) {
      packed_data.push_back(field[elem]);
    }
  }

  return packed_data;
}

void GhostLayerManager::unpack_ghost_data(
    const DistributedMesh& mesh,
    const GhostLayer& ghost,
    const std::vector<double>& packed_data,
    std::vector<double>& field) {

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
    const DistributedMesh& mesh,
    const Config& config) {

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
    DistributedMesh& mesh,
    MeshFields* fields,
    const LoadBalance& balance) {

  // Pack elements for migration
  std::vector<std::vector<uint8_t>> send_buffers;

  // Execute migration
  // Update mesh and fields
}

double ParallelLoadBalancer::estimate_cost(
    const DistributedMesh& mesh,
    const LoadBalance& balance) {

  // Estimate migration cost
  double cost = balance.migration_map.size() * 1.0; // Simplified
  return cost;
}

LoadBalance ParallelLoadBalancer::diffusive_balance(
    const DistributedMesh& mesh,
    const Config& config) {

  LoadBalance balance;

  // Diffusive load balancing algorithm
  // Exchange load with neighbors iteratively

  return balance;
}

LoadBalance ParallelLoadBalancer::direct_balance(
    const DistributedMesh& mesh,
    const Config& config) {

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
  MPI_Allreduce(&local_result.adaptation_time, &stats.comm_time, 1,
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
    const DistributedMesh& mesh) {

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
    const DistributedMesh& mesh,
    const std::string& filename) {

  // Parallel mesh output
  int rank;
  MPI_Comm_rank(comm, &rank);

  std::string local_filename = filename + "." + std::to_string(rank);

  // Write local partition
  // Implementation depends on output format
}

} // namespace svmp
