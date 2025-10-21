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
#ifdef MESH_HAS_MPI
#include <mpi.h>
#endif
#include <memory>
#include <unordered_set>
#include <vector>

namespace svmp {

#if defined(MESH_BUILD_TESTS) || !defined(MESH_HAS_MPI)

// ------------------------
// Serial stub for DistributedMesh (test-friendly)
// ------------------------
class DistributedMesh : public MeshBase {
public:
  // Constructors
  DistributedMesh() = default;
#ifdef MESH_HAS_MPI
  explicit DistributedMesh(MPI_Comm comm)
    : MeshBase() { (void)comm; }
  explicit DistributedMesh(std::shared_ptr<MeshBase> local_mesh, MPI_Comm comm = MPI_COMM_WORLD)
    : MeshBase() { (void)comm; local_mesh_ = std::move(local_mesh); }
#else
  // Generic placeholders to accept MPI-like arguments when MPI is not available
  template <typename CommT>
  explicit DistributedMesh(CommT) : MeshBase() {}
  template <typename CommT>
  explicit DistributedMesh(std::shared_ptr<MeshBase> local_mesh, CommT)
    : MeshBase() { local_mesh_ = std::move(local_mesh); }
#endif

  // Access underlying mesh
  MeshBase& local_mesh() { return local_mesh_ ? *local_mesh_ : static_cast<MeshBase&>(*this); }
  const MeshBase& local_mesh() const { return local_mesh_ ? *local_mesh_ : static_cast<const MeshBase&>(*this); }
  std::shared_ptr<MeshBase> local_mesh_ptr() { return local_mesh_; }
  std::shared_ptr<const MeshBase> local_mesh_ptr() const { return local_mesh_; }

  // MPI info
  rank_t rank() const noexcept { return 0; }
  int world_size() const noexcept { return 1; }
  const std::unordered_set<rank_t>& neighbor_ranks() const noexcept { return neighbor_ranks_; }
  // Provide a stub mpi_comm() that can be compared to MPI constants
  // When MPI is not present, return a void* null; pointer comparisons with MPI_Comm work via implicit conversion in tests.
  void* mpi_comm() const noexcept { return nullptr; }
  template <typename CommT>
  void set_mpi_comm(CommT) {}

  // Ownership (default Owned for all)
  bool is_owned_cell(index_t i) const { return get_owner(cell_owner_, Ownership::Owned, i) == Ownership::Owned; }
  bool is_ghost_cell(index_t i) const { return get_owner(cell_owner_, Ownership::Owned, i) == Ownership::Ghost; }
  bool is_shared_cell(index_t i) const { return get_owner(cell_owner_, Ownership::Owned, i) == Ownership::Shared; }
  rank_t owner_rank_cell(index_t i) const { return get_owner_rank(cell_owner_rank_, i); }

  bool is_owned_vertex(index_t i) const { return get_owner(vertex_owner_, Ownership::Owned, i) == Ownership::Owned; }
  bool is_ghost_vertex(index_t i) const { return get_owner(vertex_owner_, Ownership::Owned, i) == Ownership::Ghost; }
  bool is_shared_vertex(index_t i) const { return get_owner(vertex_owner_, Ownership::Owned, i) == Ownership::Shared; }
  rank_t owner_rank_vertex(index_t i) const { return get_owner_rank(vertex_owner_rank_, i); }

  bool is_owned_face(index_t i) const { return get_owner(face_owner_, Ownership::Owned, i) == Ownership::Owned; }
  bool is_ghost_face(index_t i) const { return get_owner(face_owner_, Ownership::Owned, i) == Ownership::Ghost; }
  bool is_shared_face(index_t i) const { return get_owner(face_owner_, Ownership::Owned, i) == Ownership::Shared; }
  rank_t owner_rank_face(index_t i) const { return get_owner_rank(face_owner_rank_, i); }

  void set_ownership(index_t id, EntityKind kind, Ownership own, rank_t owner_rank = -1) {
    switch (kind) {
      case EntityKind::Volume:
        ensure_size(cell_owner_, local_mesh_->n_cells());
        ensure_size(cell_owner_rank_, local_mesh_->n_cells(), 0);
        cell_owner_[id] = own;
        cell_owner_rank_[id] = owner_rank >= 0 ? owner_rank : 0;
        break;
      case EntityKind::Vertex:
        ensure_size(vertex_owner_, local_mesh_->n_vertices());
        ensure_size(vertex_owner_rank_, local_mesh_->n_vertices(), 0);
        vertex_owner_[id] = own;
        vertex_owner_rank_[id] = owner_rank >= 0 ? owner_rank : 0;
        break;
      case EntityKind::Face:
        ensure_size(face_owner_, local_mesh_->n_faces());
        ensure_size(face_owner_rank_, local_mesh_->n_faces(), 0);
        face_owner_[id] = own;
        face_owner_rank_[id] = owner_rank >= 0 ? owner_rank : 0;
        break;
      case EntityKind::Edge:
        break;
    }
  }

  // Ghosts (no-op in serial stub)
  void build_ghost_layer(int) {}
  void clear_ghosts() {}
  void update_ghosts(const std::vector<FieldHandle>&) {}

  // Migration & balancing (no-op)
  void migrate(const std::vector<rank_t>&) {}
  void rebalance(PartitionHint, const std::unordered_map<std::string,std::string>& = {}) {}

  // Partition metrics (single-rank computation)
  struct PartitionMetrics {
    double load_imbalance_factor{0.0};
    size_t min_cells_per_rank{0};
    size_t max_cells_per_rank{0};
    size_t avg_cells_per_rank{0};
    size_t total_edge_cuts{0};
    size_t total_shared_faces{0};
    size_t total_ghost_cells{0};
    double avg_neighbors_per_rank{0.0};
    size_t min_memory_per_rank{0};
    size_t max_memory_per_rank{0};
    double memory_imbalance_factor{0.0};
    size_t cells_to_migrate{0};
    size_t migration_volume{0};
  };

  PartitionMetrics compute_partition_quality() const {
    PartitionMetrics m;
    size_t cells = local_mesh_->n_cells();
    m.min_cells_per_rank = m.max_cells_per_rank = m.avg_cells_per_rank = cells;
    return m;
  }

  // Parallel I/O stubs
  template <typename CommT>
  static DistributedMesh load_parallel(const MeshIOOptions& opts, CommT) {
    DistributedMesh dm;
    dm.local_mesh() = MeshBase::load(opts);
    return dm;
  }
  void save_parallel(const MeshIOOptions& opts) const { local_mesh_->save(opts); }

  // Global reductions (single rank)
  size_t global_n_vertices() const { return local_mesh_->n_vertices(); }
  size_t global_n_cells() const { return local_mesh_->n_cells(); }
  size_t global_n_faces() const { return local_mesh_->n_faces(); }
  BoundingBox global_bounding_box() const { return local_mesh_->bounding_box(); }

  // Distributed search (serial)
  PointLocateResult locate_point_global(const std::array<real_t,3>& x,
                                        Configuration cfg = Configuration::Reference) const {
    return local_mesh_->locate_point(x, cfg);
  }

  // Exchange patterns
  struct ExchangePattern {
    std::vector<rank_t> send_ranks;
    std::vector<std::vector<index_t>> send_lists;
    std::vector<rank_t> recv_ranks;
    std::vector<std::vector<index_t>> recv_lists;
  };
  const ExchangePattern& vertex_exchange_pattern() const { return vertex_exchange_; }
  const ExchangePattern& cell_exchange_pattern() const { return cell_exchange_; }
  void build_exchange_patterns() { /* no-op, patterns empty */ }

private:
  template<typename T>
  static void ensure_size(std::vector<T>& v, size_t n, const T& value = T()) {
    if (v.size() < n) v.resize(n, value);
  }
  static Ownership get_owner(const std::vector<Ownership>& v, Ownership def, index_t i) {
    if (i < 0) return def;
    size_t n = v.size();
    return (static_cast<size_t>(i) < n ? v[static_cast<size_t>(i)] : def);
  }
  static rank_t get_owner_rank(const std::vector<rank_t>& v, index_t i) {
    if (i < 0) return 0;
    size_t n = v.size();
    return (static_cast<size_t>(i) < n ? v[static_cast<size_t>(i)] : 0);
  }

  std::shared_ptr<MeshBase> local_mesh_;
  std::unordered_set<rank_t> neighbor_ranks_;
  std::vector<Ownership> vertex_owner_, face_owner_, cell_owner_;
  std::vector<rank_t> cell_owner_rank_, face_owner_rank_, vertex_owner_rank_;
  ExchangePattern vertex_exchange_, cell_exchange_;
};

#else

// ========================
// Distributed mesh wrapper
// ========================
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

#endif // MESH_HAS_MPI

} // namespace svmp

#endif // SVMP_DISTRIBUTED_MESH_H
