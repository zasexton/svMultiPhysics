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

#ifndef SVMP_DISTRIBUTED_MESH_H
#define SVMP_DISTRIBUTED_MESH_H

#include "MeshBase.h"
#include <mpi.h>
#include <memory>
#include <unordered_set>
#include <vector>

namespace svmp {

// ========================
// Distributed mesh wrapper
// ========================
// This class adds MPI/parallel functionality to MeshBase through composition.
// It owns a MeshBase and manages all distributed aspects like ownership,
// ghost layers, communication patterns, and parallel I/O.

class DistributedMesh {
public:
  // ---- Constructors
  DistributedMesh();
  explicit DistributedMesh(MPI_Comm comm);
  explicit DistributedMesh(std::shared_ptr<MeshBase> local_mesh, MPI_Comm comm = MPI_COMM_WORLD);

  // ---- Access to underlying mesh
  MeshBase& local_mesh() { return *local_mesh_; }
  const MeshBase& local_mesh() const { return *local_mesh_; }
  std::shared_ptr<MeshBase> local_mesh_ptr() { return local_mesh_; }
  std::shared_ptr<const MeshBase> local_mesh_ptr() const { return local_mesh_; }

  // ---- MPI info
  MPI_Comm mpi_comm() const noexcept { return comm_; }
  rank_t rank() const noexcept { return my_rank_; }
  int world_size() const noexcept { return world_size_; }
  const std::unordered_set<rank_t>& neighbor_ranks() const noexcept { return neighbor_ranks_; }

  void set_mpi_comm(MPI_Comm comm);

  // ---- Ownership & ghosting
  bool is_owned_cell(index_t i) const;
  bool is_ghost_cell(index_t i) const;
  bool is_shared_cell(index_t i) const;
  rank_t owner_rank_cell(index_t i) const;

  bool is_owned_vertex(index_t i) const;
  bool is_ghost_vertex(index_t i) const;
  bool is_shared_vertex(index_t i) const;
  rank_t owner_rank_vertex(index_t i) const;

  bool is_owned_face(index_t i) const;
  bool is_ghost_face(index_t i) const;
  bool is_shared_face(index_t i) const;
  rank_t owner_rank_face(index_t i) const;

  void set_ownership(index_t entity_id, EntityKind kind, Ownership ownership, rank_t owner_rank = -1);

  // ---- Ghost layer construction
  void build_ghost_layer(int levels);
  void clear_ghosts();
  void update_ghosts(const std::vector<FieldHandle>& fields);

  // ---- Migration & load balancing
  void migrate(const std::vector<rank_t>& new_owner_rank_per_cell);
  void rebalance(PartitionHint hint, const std::unordered_map<std::string,std::string>& options = {});

  // ---- Partition quality metrics
  struct PartitionMetrics {
    // Load balance metrics
    double load_imbalance_factor;  // Max_load / Avg_load - 1.0
    size_t min_cells_per_rank;
    size_t max_cells_per_rank;
    size_t avg_cells_per_rank;

    // Communication metrics
    size_t total_edge_cuts;        // Number of cell-cell edges crossing ranks
    size_t total_shared_faces;     // Number of faces shared between ranks
    size_t total_ghost_cells;      // Total ghost cells across all ranks
    double avg_neighbors_per_rank; // Average number of neighboring ranks

    // Memory metrics
    size_t min_memory_per_rank;    // Bytes
    size_t max_memory_per_rank;    // Bytes
    double memory_imbalance_factor;

    // Migration metrics (if applicable)
    size_t cells_to_migrate;       // Number of cells that would move
    size_t migration_volume;       // Total bytes to transfer
  };

  PartitionMetrics compute_partition_quality() const;

  // ---- Parallel I/O
  static DistributedMesh load_parallel(const MeshIOOptions& opts, MPI_Comm comm);
  void save_parallel(const MeshIOOptions& opts) const;

  // ---- Global reductions
  size_t global_n_vertices() const;
  size_t global_n_cells() const;
  size_t global_n_faces() const;

  BoundingBox global_bounding_box() const;

  // ---- Distributed search
  PointLocateResult locate_point_global(const std::array<real_t,3>& x,
                                        Configuration cfg = Configuration::Reference) const;

  // ---- Communication patterns
  struct ExchangePattern {
    std::vector<rank_t> send_ranks;
    std::vector<std::vector<index_t>> send_lists; // entities to send per rank
    std::vector<rank_t> recv_ranks;
    std::vector<std::vector<index_t>> recv_lists; // entities to recv per rank
  };

  const ExchangePattern& vertex_exchange_pattern() const { return vertex_exchange_; }
  const ExchangePattern& cell_exchange_pattern() const { return cell_exchange_; }

  void build_exchange_patterns();

private:
  // Local mesh (owned)
  std::shared_ptr<MeshBase> local_mesh_;

  // MPI communicator and info
  MPI_Comm comm_ = MPI_COMM_SELF;
  rank_t my_rank_ = 0;
  int world_size_ = 1;
  std::unordered_set<rank_t> neighbor_ranks_;

  // Per-entity ownership
  std::vector<Ownership> vertex_owner_;
  std::vector<Ownership> face_owner_;
  std::vector<Ownership> cell_owner_;

  // Owner ranks (for shared/ghost entities)
  std::vector<rank_t> cell_owner_rank_;
  std::vector<rank_t> face_owner_rank_;
  std::vector<rank_t> vertex_owner_rank_;

  // Communication patterns
  ExchangePattern vertex_exchange_;
  ExchangePattern cell_exchange_;
  ExchangePattern face_exchange_;

  // Ghost layer metadata
  int ghost_levels_ = 0;
  std::unordered_set<index_t> ghost_vertices_;
  std::unordered_set<index_t> ghost_cells_;
  std::unordered_set<index_t> ghost_faces_;

  // Helper methods
  void exchange_entity_data(EntityKind kind, const void* send_data, void* recv_data,
                            size_t bytes_per_entity, const ExchangePattern& pattern);
  void gather_shared_entities();
  void sync_ghost_metadata();
  void synchronize_field_data(EntityKind kind, const std::string& field_name);
};

// ========================
// Template specialization for compile-time dimension
// ========================
template <int Dim>
class DistributedMesh_t {
public:
  explicit DistributedMesh_t(std::shared_ptr<DistributedMesh> dmesh)
    : dmesh_(std::move(dmesh))
  {
    if (!dmesh_) throw std::invalid_argument("DistributedMesh_t: null distributed mesh");
    if (dmesh_->local_mesh().dim() != 0 && dmesh_->local_mesh().dim() != Dim) {
      throw std::invalid_argument("DistributedMesh_t: dimension mismatch");
    }
  }

  int dim() const noexcept { return Dim; }
  DistributedMesh& dist_mesh() { return *dmesh_; }
  const DistributedMesh& dist_mesh() const { return *dmesh_; }

  // TODO: Implement when Mesh<Dim> wrapper is available
  // Mesh<Dim> local_mesh() {
  //   return Mesh<Dim>(dmesh_->local_mesh_ptr());
  // }

private:
  std::shared_ptr<DistributedMesh> dmesh_;
};

} // namespace svmp

#endif // SVMP_DISTRIBUTED_MESH_H
