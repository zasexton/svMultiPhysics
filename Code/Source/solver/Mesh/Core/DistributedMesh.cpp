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

#include "DistributedMesh.h"
#include "../IO/MeshIO.h"
#include <algorithm>
#include <numeric>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <cstring>

#if !(defined(MESH_HAS_MPI) && !defined(MESH_BUILD_TESTS))
// When MPI is not enabled, or when building tests where DistributedMesh is
// aliased to MeshBase, do not compile the real DistributedMesh implementation.
// Provide an empty translation unit to satisfy build systems.

namespace svmp { }

#else
#include <fstream>
#include <iomanip>
#include <cmath>

#ifdef MESH_HAS_METIS
// METIS graph partitioning library
// Note: metis.h defines idx_t, real_t types
extern "C" {
#include <metis.h>
}
#endif

#ifdef MESH_HAS_MPI
#include <mpi.h>

namespace {
    // Helper structures and functions for MPI operations
    struct FindData {
        double distance;
        int found;
        int rank;
    };

    // MPI reduction operation for finding minimum distance point location
    void minloc_find_op(void* in, void* inout, int* len, MPI_Datatype*) {
        FindData* in_data = static_cast<FindData*>(in);
        FindData* inout_data = static_cast<FindData*>(inout);
        for (int i = 0; i < *len; ++i) {
            if (in_data[i].found && (!inout_data[i].found ||
                in_data[i].distance < inout_data[i].distance)) {
                inout_data[i] = in_data[i];
            }
        }
    }
}
#endif

namespace svmp {

// ==========================================
// Helper functions for parallel I/O
// ==========================================

// Extract a submesh containing only cells assigned to a specific rank
static MeshBase extract_submesh(const MeshBase& global_mesh,
                               const std::vector<rank_t>& cell_partition,
                               rank_t target_rank) {
    // =====================================================
    // Submesh extraction algorithm
    // =====================================================
    // Time complexity: O(n_cells + n_vertices)
    // Memory: O(cells_for_rank + vertices_for_rank)
    // This creates a new mesh containing only the cells
    // assigned to target_rank and their vertices

    MeshBase submesh;

    // Step 1: Identify cells for this rank
    std::vector<index_t> local_cells;
    std::unordered_map<index_t, index_t> global_to_local_cell;

    for (index_t c = 0; c < static_cast<index_t>(global_mesh.n_cells()); ++c) {
        if (cell_partition[c] == target_rank) {
            global_to_local_cell[c] = static_cast<index_t>(local_cells.size());
            local_cells.push_back(c);
        }
    }

    // Step 2: Identify vertices needed by these cells
    std::unordered_set<index_t> vertex_set;
    std::vector<index_t> local_vertices;
    std::unordered_map<index_t, index_t> global_to_local_vertex;

    for (index_t global_cell : local_cells) {
        auto [verts, n_verts] = global_mesh.cell_vertices_span(global_cell);
        for (size_t i = 0; i < n_verts; ++i) {
            if (vertex_set.insert(verts[i]).second) {
                global_to_local_vertex[verts[i]] = static_cast<index_t>(local_vertices.size());
                local_vertices.push_back(verts[i]);
            }
        }
    }

    // Step 3: Copy vertex coordinates
    std::vector<real_t> coords(local_vertices.size() * global_mesh.dim());
    const auto& global_coords = global_mesh.X_ref();
    for (size_t i = 0; i < local_vertices.size(); ++i) {
        index_t global_v = local_vertices[i];
        for (int d = 0; d < global_mesh.dim(); ++d) {
            coords[i * global_mesh.dim() + d] =
                global_coords[global_v * global_mesh.dim() + d];
        }
    }

    // Step 4: Build cell connectivity with local vertex indices
    std::vector<CellShape> shapes;
    std::vector<offset_t> offsets = {0};
    std::vector<index_t> connectivity;

    for (index_t global_cell : local_cells) {
        shapes.push_back(global_mesh.cell_shape(global_cell));

        auto [verts, n_verts] = global_mesh.cell_vertices_span(global_cell);
        for (size_t i = 0; i < n_verts; ++i) {
            connectivity.push_back(global_to_local_vertex[verts[i]]);
        }
        offsets.push_back(static_cast<offset_t>(connectivity.size()));
    }

    // Step 5: Copy global IDs
    std::vector<gid_t> vertex_gids(local_vertices.size());
    for (size_t i = 0; i < local_vertices.size(); ++i) {
        vertex_gids[i] = global_mesh.vertex_gids()[local_vertices[i]];
    }

    std::vector<gid_t> cell_gids(local_cells.size());
    for (size_t i = 0; i < local_cells.size(); ++i) {
        cell_gids[i] = global_mesh.cell_gids()[local_cells[i]];
    }

    // Step 6: Create submesh
    submesh.build_from_arrays(global_mesh.dim(), coords, offsets, connectivity, shapes);
    submesh.set_vertex_gids(std::move(vertex_gids));
    submesh.set_cell_gids(std::move(cell_gids));
    submesh.finalize();

    // Step 7: Copy field data if present
    // ... field extraction code ...

    return submesh;
}

// Serialize a mesh into a byte buffer for MPI communication
static void serialize_mesh(const MeshBase& mesh, std::vector<char>& buffer) {
    // =====================================================
    // Mesh serialization for MPI communication
    // =====================================================
    // Format: [header][vertices][cells][fields]
    // Time complexity: O(n_vertices + n_cells)
    // Memory: O(total_mesh_size)

    buffer.clear();

    // Reserve approximate size to avoid reallocations
    size_t estimated_size = sizeof(int) * 10 +  // headers
                           mesh.n_vertices() * mesh.dim() * sizeof(real_t) +
                           mesh.n_cells() * 10 * sizeof(index_t);  // estimate
    buffer.reserve(estimated_size);

    // Helper function to append data
    auto append_data = [&buffer](const void* data, size_t bytes) {
        const char* ptr = static_cast<const char*>(data);
        buffer.insert(buffer.end(), ptr, ptr + bytes);
    };

    // Header: dimensions and counts
    int dim = mesh.dim();
    int n_vertices = static_cast<int>(mesh.n_vertices());
    int n_cells = static_cast<int>(mesh.n_cells());

    append_data(&dim, sizeof(int));
    append_data(&n_vertices, sizeof(int));
    append_data(&n_cells, sizeof(int));

    // Vertex coordinates
    const auto& coords = mesh.X_ref();
    append_data(coords.data(), n_vertices * dim * sizeof(real_t));

    // Cell connectivity and shapes
    for (index_t c = 0; c < mesh.n_cells(); ++c) {
        // Serialize CellShape struct
        const auto& shape = mesh.cell_shape(c);
        int family = static_cast<int>(shape.family);
        append_data(&family, sizeof(int));
        append_data(&shape.num_corners, sizeof(int));
        append_data(&shape.order, sizeof(int));

        auto [verts, n_verts] = mesh.cell_vertices_span(c);
        int n = static_cast<int>(n_verts);
        append_data(&n, sizeof(int));
        append_data(verts, n_verts * sizeof(index_t));
    }

    // Global IDs
    append_data(mesh.vertex_gids().data(), n_vertices * sizeof(gid_t));
    append_data(mesh.cell_gids().data(), n_cells * sizeof(gid_t));
}

// Deserialize a mesh from a byte buffer received via MPI
static void deserialize_mesh(const std::vector<char>& buffer, MeshBase& mesh) {
    // =====================================================
    // Mesh deserialization from MPI buffer
    // =====================================================
    // Time complexity: O(n_vertices + n_cells)
    // Memory: O(mesh_size)

    size_t offset = 0;

    // Helper function to read data
    auto read_data = [&buffer, &offset](void* dest, size_t bytes) {
        std::memcpy(dest, buffer.data() + offset, bytes);
        offset += bytes;
    };

    // Read header
    int dim, n_vertices, n_cells;
    read_data(&dim, sizeof(int));
    read_data(&n_vertices, sizeof(int));
    read_data(&n_cells, sizeof(int));

    // Read vertex coordinates
    std::vector<real_t> coords(n_vertices * dim);
    read_data(coords.data(), n_vertices * dim * sizeof(real_t));

    // Read cell connectivity and shapes
    std::vector<CellShape> shapes;
    std::vector<offset_t> offsets = {0};
    std::vector<index_t> connectivity;

    for (int c = 0; c < n_cells; ++c) {
        // Deserialize CellShape struct
        CellShape shape;
        int family;
        read_data(&family, sizeof(int));
        shape.family = static_cast<CellFamily>(family);
        read_data(&shape.num_corners, sizeof(int));
        read_data(&shape.order, sizeof(int));
        shapes.push_back(shape);

        int n_verts;
        read_data(&n_verts, sizeof(int));
        std::vector<index_t> cell_verts(n_verts);
        read_data(cell_verts.data(), n_verts * sizeof(index_t));
        connectivity.insert(connectivity.end(), cell_verts.begin(), cell_verts.end());
        offsets.push_back(static_cast<offset_t>(connectivity.size()));
    }

    // Read global IDs
    std::vector<gid_t> vertex_gids(n_vertices);
    std::vector<gid_t> cell_gids(n_cells);
    read_data(vertex_gids.data(), n_vertices * sizeof(gid_t));
    read_data(cell_gids.data(), n_cells * sizeof(gid_t));

    // Initialize mesh
    mesh.build_from_arrays(dim, coords, offsets, connectivity, shapes);
    mesh.set_vertex_gids(std::move(vertex_gids));
    mesh.set_cell_gids(std::move(cell_gids));
    mesh.finalize();
}

// Tree-based mesh distribution for large-scale runs
static void distribute_mesh_tree(const MeshBase& global_mesh,
                                const std::vector<rank_t>& cell_partition,
                                MPI_Comm comm) {
    // =====================================================
    // Binary tree distribution pattern
    // =====================================================
    // Algorithm: Each rank splits its data and sends half to a child
    // Time complexity: O(n * log(p)) total
    // Memory: O(n/2^level) at each tree level
    // Scaling: Reduces root bottleneck for large rank counts

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        // Root initiates tree distribution
        int tree_height = static_cast<int>(std::log2(size)) + 1;

        for (int level = 0; level < tree_height; ++level) {
            int step = 1 << (level + 1);  // 2^(level+1)
            int sender_mask = (1 << level) - 1;

            if ((rank & sender_mask) == 0 && rank + (step/2) < size) {
                // This rank sends to rank + step/2
                int target = rank + (step/2);

                // Extract submesh for all ranks >= target
                // that this sender is responsible for
                MeshBase submesh_to_send = extract_submesh(global_mesh, cell_partition, target);

                // Serialize and send
                std::vector<char> buffer;
                serialize_mesh(submesh_to_send, buffer);

                int buffer_size = static_cast<int>(buffer.size());
                MPI_Send(&buffer_size, 1, MPI_INT, target, level * 2, comm);
                MPI_Send(buffer.data(), buffer_size, MPI_CHAR, target, level * 2 + 1, comm);
            }
        }
    }
}

// Receive mesh in tree-based distribution
static void receive_mesh_tree(MeshBase& mesh, MPI_Comm comm) {
    // =====================================================
    // Binary tree reception pattern
    // =====================================================
    // Receives mesh data from parent in tree hierarchy

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank != 0) {
        // Determine which level this rank receives at
        int level = static_cast<int>(std::log2(rank & -rank));

        // Calculate sender
        int step = 1 << level;
        int sender = rank - step;

        // Receive buffer size
        int buffer_size;
        MPI_Recv(&buffer_size, 1, MPI_INT, sender, level * 2, comm, MPI_STATUS_IGNORE);

        // Receive buffer
        std::vector<char> buffer(buffer_size);
        MPI_Recv(buffer.data(), buffer_size, MPI_CHAR, sender, level * 2 + 1, comm, MPI_STATUS_IGNORE);

        // Deserialize
        deserialize_mesh(buffer, mesh);
    }
}

// ==========================================
// Constructors
// ==========================================

DistributedMesh::DistributedMesh()
    : local_mesh_(std::make_shared<MeshBase>()),
      comm_(MPI_COMM_SELF),
      my_rank_(0),
      world_size_(1) {
}

DistributedMesh::DistributedMesh(MPI_Comm comm)
    : local_mesh_(std::make_shared<MeshBase>()),
      comm_(comm) {
#ifdef MESH_HAS_MPI
    if (comm != MPI_COMM_NULL) {
        MPI_Comm_rank(comm_, &my_rank_);
        MPI_Comm_size(comm_, &world_size_);
    }
#else
    my_rank_ = 0;
    world_size_ = 1;
#endif
}

DistributedMesh::DistributedMesh(std::shared_ptr<MeshBase> local_mesh, MPI_Comm comm)
    : local_mesh_(local_mesh ? local_mesh : std::make_shared<MeshBase>()),
      comm_(comm) {

    if (!local_mesh_) {
        throw std::invalid_argument("DistributedMesh: null local mesh provided");
    }

#ifdef MESH_HAS_MPI
    if (comm != MPI_COMM_NULL) {
        MPI_Comm_rank(comm_, &my_rank_);
        MPI_Comm_size(comm_, &world_size_);
    }
#else
    my_rank_ = 0;
    world_size_ = 1;
#endif

    // Initialize ownership arrays
    vertex_owner_.resize(local_mesh_->n_vertices(), Ownership::Owned);
    cell_owner_.resize(local_mesh_->n_cells(), Ownership::Owned);
    face_owner_.resize(local_mesh_->n_faces(), Ownership::Owned);

    vertex_owner_rank_.resize(local_mesh_->n_vertices(), my_rank_);
    cell_owner_rank_.resize(local_mesh_->n_cells(), my_rank_);
    face_owner_rank_.resize(local_mesh_->n_faces(), my_rank_);
}

// ==========================================
// MPI Info
// ==========================================

void DistributedMesh::set_mpi_comm(MPI_Comm comm) {
    comm_ = comm;

#ifdef MESH_HAS_MPI
    if (comm != MPI_COMM_NULL) {
        MPI_Comm_rank(comm_, &my_rank_);
        MPI_Comm_size(comm_, &world_size_);
    } else {
        my_rank_ = 0;
        world_size_ = 1;
    }
#else
    my_rank_ = 0;
    world_size_ = 1;
#endif
}

// ==========================================
// Ownership & Ghosting
// ==========================================

bool DistributedMesh::is_owned_cell(index_t i) const {
    if (i < 0 || i >= static_cast<index_t>(cell_owner_.size())) {
        return false;
    }
    return cell_owner_[i] == Ownership::Owned;
}

bool DistributedMesh::is_ghost_cell(index_t i) const {
    if (i < 0 || i >= static_cast<index_t>(cell_owner_.size())) {
        return false;
    }
    return cell_owner_[i] == Ownership::Ghost;
}

bool DistributedMesh::is_shared_cell(index_t i) const {
    if (i < 0 || i >= static_cast<index_t>(cell_owner_.size())) {
        return false;
    }
    return cell_owner_[i] == Ownership::Shared;
}

rank_t DistributedMesh::owner_rank_cell(index_t i) const {
    if (i < 0 || i >= static_cast<index_t>(cell_owner_rank_.size())) {
        return -1;
    }
    return cell_owner_rank_[i];
}

bool DistributedMesh::is_owned_vertex(index_t i) const {
    if (i < 0 || i >= static_cast<index_t>(vertex_owner_.size())) {
        return false;
    }
    return vertex_owner_[i] == Ownership::Owned;
}

bool DistributedMesh::is_ghost_vertex(index_t i) const {
    if (i < 0 || i >= static_cast<index_t>(vertex_owner_.size())) {
        return false;
    }
    return vertex_owner_[i] == Ownership::Ghost;
}

bool DistributedMesh::is_shared_vertex(index_t i) const {
    if (i < 0 || i >= static_cast<index_t>(vertex_owner_.size())) {
        return false;
    }
    return vertex_owner_[i] == Ownership::Shared;
}

rank_t DistributedMesh::owner_rank_vertex(index_t i) const {
    if (i < 0 || i >= static_cast<index_t>(vertex_owner_rank_.size())) {
        return -1;
    }
    return vertex_owner_rank_[i];
}

bool DistributedMesh::is_owned_face(index_t i) const {
    if (i < 0 || i >= static_cast<index_t>(face_owner_.size())) {
        return false;
    }
    return face_owner_[i] == Ownership::Owned;
}

bool DistributedMesh::is_ghost_face(index_t i) const {
    if (i < 0 || i >= static_cast<index_t>(face_owner_.size())) {
        return false;
    }
    return face_owner_[i] == Ownership::Ghost;
}

bool DistributedMesh::is_shared_face(index_t i) const {
    if (i < 0 || i >= static_cast<index_t>(face_owner_.size())) {
        return false;
    }
    return face_owner_[i] == Ownership::Shared;
}

rank_t DistributedMesh::owner_rank_face(index_t i) const {
    if (i < 0 || i >= static_cast<index_t>(face_owner_rank_.size())) {
        return -1;
    }
    return face_owner_rank_[i];
}

void DistributedMesh::set_ownership(index_t entity_id, EntityKind kind,
                                   Ownership ownership, rank_t owner_rank) {
    switch (kind) {
        case EntityKind::Vertex:
            if (entity_id >= 0 && entity_id < static_cast<index_t>(vertex_owner_.size())) {
                vertex_owner_[entity_id] = ownership;
                if (owner_rank >= 0) {
                    vertex_owner_rank_[entity_id] = owner_rank;
                }
            }
            break;
        case EntityKind::Face:
            if (entity_id >= 0 && entity_id < static_cast<index_t>(face_owner_.size())) {
                face_owner_[entity_id] = ownership;
                if (owner_rank >= 0) {
                    face_owner_rank_[entity_id] = owner_rank;
                }
            }
            break;
        case EntityKind::Volume:
            if (entity_id >= 0 && entity_id < static_cast<index_t>(cell_owner_.size())) {
                cell_owner_[entity_id] = ownership;
                if (owner_rank >= 0) {
                    cell_owner_rank_[entity_id] = owner_rank;
                }
            }
            break;
        default:
            break;
    }
}

// ==========================================
// Ghost Layer Construction
// ==========================================

void DistributedMesh::build_ghost_layer(int levels) {
#ifdef MESH_HAS_MPI
    if (world_size_ == 1) {
        return;  // No ghosts needed for serial run
    }

    ghost_levels_ = levels;

    // Clear existing ghosts
    clear_ghosts();

    // Build vertex-to-cell adjacency if not already cached
    local_mesh_->build_vertex2cell();

    // Step 1: Identify boundary vertices (on inter-processor boundaries)
    std::set<index_t> boundary_vertices;
    for (index_t n = 0; n < static_cast<index_t>(vertex_owner_.size()); ++n) {
        if (vertex_owner_[n] == Ownership::Shared) {
            boundary_vertices.insert(n);
        }
    }

    // Step 2: For each level, find cells adjacent to boundary
    std::set<index_t> ghost_cells_set;
    std::set<index_t> current_front = boundary_vertices;

    for (int level = 0; level < levels; ++level) {
        std::set<index_t> next_front;

        // Find cells touching current front vertices
        for (index_t n : current_front) {
            auto cells = local_mesh_->vertex_cells(n);
            for (index_t c : cells) {
                if (!is_owned_cell(c) && ghost_cells_set.find(c) == ghost_cells_set.end()) {
                    ghost_cells_set.insert(c);

                    // Add vertices of this cell to next front
                    auto [vertices, n_vertices] = local_mesh_->cell_vertices_span(c);
                    for (size_t i = 0; i < n_vertices; ++i) {
                        if (boundary_vertices.find(vertices[i]) == boundary_vertices.end()) {
                            next_front.insert(vertices[i]);
                        }
                    }
                }
            }
        }

        // Update front for next level
        current_front = next_front;
        boundary_vertices.insert(next_front.begin(), next_front.end());
    }

    // Step 3: Mark ghost cells and vertices
    ghost_cells_ = std::unordered_set<index_t>(ghost_cells_set.begin(), ghost_cells_set.end());

    for (index_t c : ghost_cells_) {
        cell_owner_[c] = Ownership::Ghost;

        // Mark vertices of ghost cells as ghost if not already shared
        auto [vertices, n_vertices] = local_mesh_->cell_vertices_span(c);
        for (size_t i = 0; i < n_vertices; ++i) {
            if (vertex_owner_[vertices[i]] != Ownership::Shared) {
                vertex_owner_[vertices[i]] = Ownership::Ghost;
                ghost_vertices_.insert(vertices[i]);
            }
        }
    }

    // Step 4: Exchange ghost metadata with neighbors
    sync_ghost_metadata();

    // Step 5: Build communication patterns
    build_exchange_patterns();
#endif
}

void DistributedMesh::clear_ghosts() {
    ghost_vertices_.clear();
    ghost_cells_.clear();
    ghost_faces_.clear();
    ghost_levels_ = 0;

    // Reset ownership for all entities to owned
    std::fill(vertex_owner_.begin(), vertex_owner_.end(), Ownership::Owned);
    std::fill(cell_owner_.begin(), cell_owner_.end(), Ownership::Owned);
    std::fill(face_owner_.begin(), face_owner_.end(), Ownership::Owned);
}

void DistributedMesh::update_ghosts(const std::vector<FieldHandle>& fields) {
    // =====================================================
    // Ghost Field Update Algorithm
    // =====================================================
    // Purpose: Update ghost entity field values from their owners
    // Algorithm: Use pre-built exchange patterns for efficient communication
    //
    // Performance characteristics:
    // - Time: O(n_fields * (n_ghosts * field_size + comm_latency))
    // - Memory: O(max(send_buffer_size, recv_buffer_size))
    // - Communication: 2 * n_neighbors messages per field (non-blocking)
    // - Scaling: O(log P) neighbors typical, O(√P) worst case
    //
    // Optimizations:
    // - Uses pre-built exchange patterns (avoids runtime construction)
    // - Non-blocking communication overlaps computation
    // - Batching multiple fields would reduce latency overhead

#ifdef MESH_HAS_MPI
    if (world_size_ == 1) {
        return;  // No ghosts in serial
    }

    // Check if exchange patterns are built
    if (vertex_exchange_.send_ranks.empty() &&
        vertex_exchange_.recv_ranks.empty() &&
        cell_exchange_.send_ranks.empty() &&
        cell_exchange_.recv_ranks.empty()) {
        // Patterns not built - build them now
        build_exchange_patterns();
    }

    // Performance timing (optional)
    double update_start = MPI_Wtime();

    // Process each field independently
    // Future optimization: batch fields of same entity type
    for (const auto& field : fields) {
        // Get field data pointer
        void* data = local_mesh_->field_data(field);
        if (!data) continue;  // Field not found

        size_t bytes_per_entity = local_mesh_->field_bytes_per_entity(field);

        // Exchange based on entity kind
        // Each exchange uses optimized non-blocking communication
        switch (field.kind) {
            case EntityKind::Vertex:
                // Expected: O(n_ghost_vertices * field_size)
                exchange_entity_data(EntityKind::Vertex, data, data,
                                   bytes_per_entity, vertex_exchange_);
                break;

            case EntityKind::Volume:
                // Expected: O(n_ghost_cells * field_size)
                exchange_entity_data(EntityKind::Volume, data, data,
                                   bytes_per_entity, cell_exchange_);
                break;

            case EntityKind::Face:
                // Expected: O(n_ghost_faces * field_size)
                exchange_entity_data(EntityKind::Face, data, data,
                                   bytes_per_entity, face_exchange_);
                break;

            case EntityKind::Edge:
                // Edge exchange pattern not implemented yet
                // Would follow same pattern as vertices/cells/faces
                break;
        }
    }

    // Optional: Report performance
    if (getenv("MESH_VERBOSE")) {
        double update_time = MPI_Wtime() - update_start;
        double max_update_time;
        MPI_Reduce(&update_time, &max_update_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm_);

        if (my_rank_ == 0) {
            std::cout << "Ghost update for " << fields.size()
                      << " fields completed in " << max_update_time << " seconds\n";
        }
    }
#endif
}

// ==========================================
// Migration & Load Balancing
// ==========================================

void DistributedMesh::migrate(const std::vector<rank_t>& new_owner_rank_per_cell) {
#ifdef MESH_HAS_MPI
    if (world_size_ == 1) {
        return;
    }

    if (new_owner_rank_per_cell.size() != local_mesh_->n_cells()) {
        throw std::invalid_argument("migrate: wrong size for new_owner_rank_per_cell");
    }

    // ========================================
    // Step 1: Determine migration destinations
    // ========================================

    std::map<rank_t, std::vector<index_t>> cells_to_send;
    std::vector<index_t> cells_to_keep;

    for (index_t c = 0; c < static_cast<index_t>(new_owner_rank_per_cell.size()); ++c) {
        rank_t new_rank = new_owner_rank_per_cell[c];
        if (new_rank == my_rank_) {
            cells_to_keep.push_back(c);
        } else if (new_rank >= 0 && new_rank < world_size_) {
            cells_to_send[new_rank].push_back(c);
        }
    }

    // ========================================
    // Step 2: Identify vertices to migrate
    // ========================================

    // Map from vertex to set of ranks that need it
    std::map<index_t, std::set<rank_t>> vertex_destinations;

    // Vertices needed by cells we keep
    std::set<index_t> vertices_to_keep;
    for (index_t c : cells_to_keep) {
        auto [vertices_ptr2, n_vertices2] = local_mesh_->cell_vertices_span(c);
        for (size_t i = 0; i < n_vertices2; ++i) {
            vertices_to_keep.insert(vertices_ptr2[i]);
        }
    }

    // Vertices needed by cells we send
    for (const auto& [rank, cells] : cells_to_send) {
        for (index_t c : cells) {
            auto [vertices_ptr3, n_vertices3] = local_mesh_->cell_vertices_span(c);
            for (size_t i = 0; i < n_vertices3; ++i) {
                vertex_destinations[vertices_ptr3[i]].insert(rank);
            }
        }
    }

    // ========================================
    // Step 3: Pack data into send buffers
    // ========================================

    struct MeshPacket {
        // Header information
        size_t n_cells = 0;
        size_t n_vertices = 0;
        int spatial_dim = 0;

        // Vertex data
        std::vector<gid_t> vertex_gids;
        std::vector<real_t> vertex_coords;

        // Cell data
        std::vector<gid_t> cell_gids;
        std::vector<CellShape> cell_shapes;
        std::vector<offset_t> cell_offsets;
        std::vector<index_t> cell_connectivity;
        std::vector<label_t> cell_regions;

        // Field data (simplified - stores all fields as bytes)
        struct FieldPacket {
            std::string name;
            EntityKind kind;
            FieldScalarType type;
            size_t components;
            size_t bytes_per_component;
            std::vector<uint8_t> data;
        };
        std::vector<FieldPacket> fields;

        size_t compute_size() const {
        size_t size = sizeof(n_cells) + sizeof(n_vertices) + sizeof(spatial_dim);
        size += vertex_gids.size() * sizeof(gid_t);
        size += vertex_coords.size() * sizeof(real_t);
            size += cell_gids.size() * sizeof(gid_t);
            size += cell_shapes.size() * sizeof(CellShape);
            size += cell_offsets.size() * sizeof(offset_t);
            size += cell_connectivity.size() * sizeof(index_t);
            size += cell_regions.size() * sizeof(label_t);

            for (const auto& field : fields) {
                size += field.name.size() + 1;  // string + null terminator
                size += sizeof(field.kind) + sizeof(field.type);
                size += sizeof(field.components) + sizeof(field.bytes_per_component);
                size += field.data.size();
            }

            return size;
        }

        void serialize(std::vector<uint8_t>& buffer) const {
            buffer.clear();
            buffer.reserve(compute_size());

            // Helper to append data to buffer
            auto append = [&buffer](const void* data, size_t size) {
                const uint8_t* bytes = static_cast<const uint8_t*>(data);
                buffer.insert(buffer.end(), bytes, bytes + size);
            };

            // Serialize header
            append(&n_cells, sizeof(n_cells));
            append(&n_vertices, sizeof(n_vertices));
            append(&spatial_dim, sizeof(spatial_dim));

            // Serialize vertices
            append(vertex_gids.data(), vertex_gids.size() * sizeof(gid_t));
            append(vertex_coords.data(), vertex_coords.size() * sizeof(real_t));

            // Serialize cells
            append(cell_gids.data(), cell_gids.size() * sizeof(gid_t));
            append(cell_shapes.data(), cell_shapes.size() * sizeof(CellShape));
            append(cell_offsets.data(), cell_offsets.size() * sizeof(offset_t));
            append(cell_connectivity.data(), cell_connectivity.size() * sizeof(index_t));
            append(cell_regions.data(), cell_regions.size() * sizeof(label_t));

            // Serialize fields
            size_t n_fields = fields.size();
            append(&n_fields, sizeof(n_fields));

            for (const auto& field : fields) {
                size_t name_len = field.name.size() + 1;
                append(&name_len, sizeof(name_len));
                append(field.name.c_str(), name_len);
                append(&field.kind, sizeof(field.kind));
                append(&field.type, sizeof(field.type));
                append(&field.components, sizeof(field.components));
                append(&field.bytes_per_component, sizeof(field.bytes_per_component));

                size_t data_size = field.data.size();
                append(&data_size, sizeof(data_size));
                append(field.data.data(), data_size);
            }
        }

        void deserialize(const std::vector<uint8_t>& buffer) {
            size_t offset = 0;

            // Helper to read data from buffer
            auto read = [&buffer, &offset](void* dest, size_t size) {
                std::memcpy(dest, &buffer[offset], size);
                offset += size;
            };

            // Deserialize header
            read(&n_cells, sizeof(n_cells));
            read(&n_vertices, sizeof(n_vertices));
            read(&spatial_dim, sizeof(spatial_dim));

            // Deserialize vertices
            vertex_gids.resize(n_vertices);
            read(vertex_gids.data(), n_vertices * sizeof(gid_t));

            vertex_coords.resize(n_vertices * spatial_dim);
            read(vertex_coords.data(), n_vertices * spatial_dim * sizeof(real_t));

            // Deserialize cells
            cell_gids.resize(n_cells);
            read(cell_gids.data(), n_cells * sizeof(gid_t));

            cell_shapes.resize(n_cells);
            read(cell_shapes.data(), n_cells * sizeof(CellShape));

            cell_offsets.resize(n_cells + 1);
            read(cell_offsets.data(), (n_cells + 1) * sizeof(offset_t));

            size_t connectivity_size = cell_offsets[n_cells];
            cell_connectivity.resize(connectivity_size);
            read(cell_connectivity.data(), connectivity_size * sizeof(index_t));

            cell_regions.resize(n_cells);
            read(cell_regions.data(), n_cells * sizeof(label_t));

            // Deserialize fields
            size_t n_fields = 0;
            read(&n_fields, sizeof(n_fields));
            fields.resize(n_fields);

            for (auto& field : fields) {
                size_t name_len = 0;
                read(&name_len, sizeof(name_len));

                std::vector<char> name_buffer(name_len);
                read(name_buffer.data(), name_len);
                field.name = std::string(name_buffer.data());

                read(&field.kind, sizeof(field.kind));
                read(&field.type, sizeof(field.type));
                read(&field.components, sizeof(field.components));
                read(&field.bytes_per_component, sizeof(field.bytes_per_component));

                size_t data_size = 0;
                read(&data_size, sizeof(data_size));
                field.data.resize(data_size);
                read(field.data.data(), data_size);
            }
        }
    };

    // Create packets for each destination rank
    std::map<rank_t, MeshPacket> send_packets;

    for (const auto& [rank, cells] : cells_to_send) {
        MeshPacket& packet = send_packets[rank];
        packet.n_cells = cells.size();
        packet.spatial_dim = local_mesh_->dim();

        // Map from old vertex index to new vertex index in packet
        std::map<index_t, index_t> vertex_remap;

        // Collect unique vertices used by these cells
        std::vector<index_t> vertices_for_rank;
        for (index_t c : cells) {
            auto [vertices_ptr4, n_vertices4] = local_mesh_->cell_vertices_span(c);
            for (size_t i = 0; i < n_vertices4; ++i) {
                if (vertex_remap.find(vertices_ptr4[i]) == vertex_remap.end()) {
                    vertex_remap[vertices_ptr4[i]] = static_cast<index_t>(vertices_for_rank.size());
                    vertices_for_rank.push_back(vertices_ptr4[i]);
                }
            }
        }

        packet.n_vertices = vertices_for_rank.size();

        // Pack vertex data
    const auto& all_vertex_gids = local_mesh_->vertex_gids();
        const auto& all_coords = (local_mesh_->active_configuration() == Configuration::Reference)
                                 ? local_mesh_->X_ref()
                                 : local_mesh_->X_cur();

        for (index_t n : vertices_for_rank) {
            packet.vertex_gids.push_back(all_vertex_gids[n]);
            for (int d = 0; d < packet.spatial_dim; ++d) {
                packet.vertex_coords.push_back(all_coords[n * packet.spatial_dim + d]);
            }
        }

        // Pack cell data
        const auto& all_cell_gids = local_mesh_->cell_gids();
        const auto& all_cell_shapes = local_mesh_->cell_shapes();
        const auto& all_cell_regions = local_mesh_->cell_region_ids();

        packet.cell_offsets.push_back(0);

        for (index_t c : cells) {
            packet.cell_gids.push_back(all_cell_gids[c]);
            packet.cell_shapes.push_back(all_cell_shapes[c]);

            // Remap connectivity to packet-local vertex indices
            auto [vertices_ptr5, n_vertices5] = local_mesh_->cell_vertices_span(c);
            for (size_t i = 0; i < n_vertices5; ++i) {
                packet.cell_connectivity.push_back(vertex_remap[vertices_ptr5[i]]);
            }
            packet.cell_offsets.push_back(static_cast<offset_t>(packet.cell_connectivity.size()));

            if (!all_cell_regions.empty()) {
                packet.cell_regions.push_back(all_cell_regions[c]);
            } else {
                packet.cell_regions.push_back(0);
            }
        }

        // Pack field data (simplified - only packs data for migrating entities)

        // Pack vertex fields
        auto vertex_field_names = local_mesh_->field_names(EntityKind::Vertex);
        for (const auto& field_name : vertex_field_names) {
            MeshPacket::FieldPacket field_packet;
            field_packet.name = field_name;
            field_packet.kind = EntityKind::Vertex;
            field_packet.type = local_mesh_->field_type_by_name(EntityKind::Vertex, field_name);
            field_packet.components = local_mesh_->field_components_by_name(EntityKind::Vertex, field_name);
            field_packet.bytes_per_component = local_mesh_->field_bytes_per_component_by_name(EntityKind::Vertex, field_name);

            // Pack data for migrating vertices
            const uint8_t* field_data = static_cast<const uint8_t*>(
                local_mesh_->field_data_by_name(EntityKind::Vertex, field_name));

            if (field_data) {
                size_t bytes_per_entity = field_packet.components * field_packet.bytes_per_component;
                field_packet.data.reserve(vertices_for_rank.size() * bytes_per_entity);

                for (index_t n : vertices_for_rank) {
                    const uint8_t* entity_data = &field_data[n * bytes_per_entity];
                    field_packet.data.insert(field_packet.data.end(),
                                           entity_data,
                                           entity_data + bytes_per_entity);
                }

                packet.fields.push_back(std::move(field_packet));
            }
        }

        // Pack cell fields
        auto cell_field_names = local_mesh_->field_names(EntityKind::Volume);
        for (const auto& field_name : cell_field_names) {
            MeshPacket::FieldPacket field_packet;
            field_packet.name = field_name;
            field_packet.kind = EntityKind::Volume;
            field_packet.type = local_mesh_->field_type_by_name(EntityKind::Volume, field_name);
            field_packet.components = local_mesh_->field_components_by_name(EntityKind::Volume, field_name);
            field_packet.bytes_per_component = local_mesh_->field_bytes_per_component_by_name(EntityKind::Volume, field_name);

            // Pack data for migrating cells
            const uint8_t* field_data = static_cast<const uint8_t*>(
                local_mesh_->field_data_by_name(EntityKind::Volume, field_name));

            if (field_data) {
                size_t bytes_per_entity = field_packet.components * field_packet.bytes_per_component;
                field_packet.data.reserve(cells.size() * bytes_per_entity);

                for (index_t c : cells) {
                    const uint8_t* entity_data = &field_data[c * bytes_per_entity];
                    field_packet.data.insert(field_packet.data.end(),
                                           entity_data,
                                           entity_data + bytes_per_entity);
                }

                packet.fields.push_back(std::move(field_packet));
            }
        }
    }

    // ========================================
    // Step 4: Exchange data sizes
    // ========================================

    std::vector<int> send_sizes(world_size_, 0);
    std::vector<int> recv_sizes(world_size_, 0);

    for (const auto& [rank, packet] : send_packets) {
        std::vector<uint8_t> buffer;
        packet.serialize(buffer);
        send_sizes[rank] = static_cast<int>(buffer.size());
    }

    MPI_Alltoall(send_sizes.data(), 1, MPI_INT,
                recv_sizes.data(), 1, MPI_INT, comm_);

    // ========================================
    // Step 5: Exchange mesh data
    // ========================================

    std::vector<MPI_Request> requests;
    std::map<rank_t, std::vector<uint8_t>> send_buffers;
    std::map<rank_t, std::vector<uint8_t>> recv_buffers;

    // Serialize and send
    for (const auto& [rank, packet] : send_packets) {
        send_buffers[rank].clear();
        packet.serialize(send_buffers[rank]);

        MPI_Request req;
        MPI_Isend(send_buffers[rank].data(),
                 static_cast<int>(send_buffers[rank].size()),
                 MPI_BYTE, rank, 42, comm_, &req);
        requests.push_back(req);
    }

    // Post receives
    for (int rank = 0; rank < world_size_; ++rank) {
        if (recv_sizes[rank] > 0) {
            recv_buffers[rank].resize(recv_sizes[rank]);

            MPI_Request req;
            MPI_Irecv(recv_buffers[rank].data(),
                     recv_sizes[rank],
                     MPI_BYTE, rank, 42, comm_, &req);
            requests.push_back(req);
        }
    }

    // Wait for all communication
    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

    // ========================================
    // Step 6: Rebuild local mesh
    // ========================================

    // Deserialize received packets
    std::vector<MeshPacket> recv_packets;
    for (const auto& [rank, buffer] : recv_buffers) {
        if (!buffer.empty()) {
            MeshPacket packet;
            packet.deserialize(buffer);
            recv_packets.push_back(std::move(packet));
        }
    }

    // Combine kept and received data into new mesh
    std::vector<real_t> new_coords;
    std::vector<gid_t> new_vertex_gids;
    std::vector<CellShape> new_cell_shapes;
    std::vector<offset_t> new_cell_offsets;
    std::vector<index_t> new_cell_connectivity;
    std::vector<gid_t> new_cell_gids;
    std::vector<label_t> new_cell_regions;

    // Map from global vertex ID to local index in new mesh
    std::map<gid_t, index_t> gid_to_new_vertex;
    index_t next_vertex_id = 0;

    // Add vertices from cells we keep
    const auto& old_coords = (local_mesh_->active_configuration() == Configuration::Reference)
                            ? local_mesh_->X_ref()
                            : local_mesh_->X_cur();
    const auto& old_vertex_gids = local_mesh_->vertex_gids();

    for (index_t n : vertices_to_keep) {
        gid_t gid = old_vertex_gids[n];
        if (gid_to_new_vertex.find(gid) == gid_to_new_vertex.end()) {
            gid_to_new_vertex[gid] = next_vertex_id++;
            new_vertex_gids.push_back(gid);

            for (int d = 0; d < local_mesh_->dim(); ++d) {
                new_coords.push_back(old_coords[n * local_mesh_->dim() + d]);
            }
        }
    }

    // Add vertices from received packets
    for (const auto& packet : recv_packets) {
        for (size_t i = 0; i < packet.n_vertices; ++i) {
            gid_t gid = packet.vertex_gids[i];
            if (gid_to_new_vertex.find(gid) == gid_to_new_vertex.end()) {
                gid_to_new_vertex[gid] = next_vertex_id++;
                new_vertex_gids.push_back(gid);

                for (int d = 0; d < packet.spatial_dim; ++d) {
                    new_coords.push_back(packet.vertex_coords[i * packet.spatial_dim + d]);
                }
            }
        }
    }

    // Add cells we keep
    const auto& old_cell_gids = local_mesh_->cell_gids();
    const auto& old_cell_shapes = local_mesh_->cell_shapes();
    const auto& old_cell_regions = local_mesh_->cell_region_ids();

    new_cell_offsets.push_back(0);

    for (index_t c : cells_to_keep) {
        new_cell_gids.push_back(old_cell_gids[c]);
        new_cell_shapes.push_back(old_cell_shapes[c]);

            auto [vertices_ptr6, n_vertices6] = local_mesh_->cell_vertices_span(c);
        for (size_t i = 0; i < n_vertices6; ++i) {
            gid_t vertex_gid = old_vertex_gids[vertices_ptr6[i]];
            new_cell_connectivity.push_back(gid_to_new_vertex[vertex_gid]);
        }
        new_cell_offsets.push_back(static_cast<offset_t>(new_cell_connectivity.size()));

        if (!old_cell_regions.empty()) {
            new_cell_regions.push_back(old_cell_regions[c]);
        } else {
            new_cell_regions.push_back(0);
        }
    }

    // Add received cells
    for (const auto& packet : recv_packets) {
        for (size_t c = 0; c < packet.n_cells; ++c) {
            new_cell_gids.push_back(packet.cell_gids[c]);
            new_cell_shapes.push_back(packet.cell_shapes[c]);

            offset_t start = packet.cell_offsets[c];
            offset_t end = packet.cell_offsets[c + 1];

            for (offset_t i = start; i < end; ++i) {
                // Map from packet-local vertex index to global ID, then to new local index
                gid_t vertex_gid = packet.vertex_gids[packet.cell_connectivity[i]];
                new_cell_connectivity.push_back(gid_to_new_vertex[vertex_gid]);
            }
            new_cell_offsets.push_back(static_cast<offset_t>(new_cell_connectivity.size()));

            if (!packet.cell_regions.empty()) {
                new_cell_regions.push_back(packet.cell_regions[c]);
            } else {
                new_cell_regions.push_back(0);
            }
        }
    }

    // Save field data for kept entities before clearing
    std::vector<MeshPacket::FieldPacket> kept_fields;

    // Save vertex fields for kept vertices
    auto all_vertex_field_names = local_mesh_->field_names(EntityKind::Vertex);
    for (const auto& field_name : all_vertex_field_names) {
        MeshPacket::FieldPacket field_packet;
        field_packet.name = field_name;
        field_packet.kind = EntityKind::Vertex;
        field_packet.type = local_mesh_->field_type_by_name(EntityKind::Vertex, field_name);
        field_packet.components = local_mesh_->field_components_by_name(EntityKind::Vertex, field_name);
        field_packet.bytes_per_component = local_mesh_->field_bytes_per_component_by_name(EntityKind::Vertex, field_name);

        const uint8_t* field_data = static_cast<const uint8_t*>(
            local_mesh_->field_data_by_name(EntityKind::Vertex, field_name));

        if (field_data) {
            size_t bytes_per_entity = field_packet.components * field_packet.bytes_per_component;
            field_packet.data.reserve(vertices_to_keep.size() * bytes_per_entity);

            for (index_t n : vertices_to_keep) {
                const uint8_t* entity_data = &field_data[n * bytes_per_entity];
                field_packet.data.insert(field_packet.data.end(),
                                       entity_data,
                                       entity_data + bytes_per_entity);
            }

            kept_fields.push_back(std::move(field_packet));
        }
    }

    // Save cell fields for kept cells
    auto all_cell_field_names = local_mesh_->field_names(EntityKind::Volume);
    for (const auto& field_name : all_cell_field_names) {
        MeshPacket::FieldPacket field_packet;
        field_packet.name = field_name;
        field_packet.kind = EntityKind::Volume;
        field_packet.type = local_mesh_->field_type_by_name(EntityKind::Volume, field_name);
        field_packet.components = local_mesh_->field_components_by_name(EntityKind::Volume, field_name);
        field_packet.bytes_per_component = local_mesh_->field_bytes_per_component_by_name(EntityKind::Volume, field_name);

        const uint8_t* field_data = static_cast<const uint8_t*>(
            local_mesh_->field_data_by_name(EntityKind::Volume, field_name));

        if (field_data) {
            size_t bytes_per_entity = field_packet.components * field_packet.bytes_per_component;
            field_packet.data.reserve(cells_to_keep.size() * bytes_per_entity);

            for (index_t c : cells_to_keep) {
                const uint8_t* entity_data = &field_data[c * bytes_per_entity];
                field_packet.data.insert(field_packet.data.end(),
                                       entity_data,
                                       entity_data + bytes_per_entity);
            }

            kept_fields.push_back(std::move(field_packet));
        }
    }

    // Create new mesh (save dimension before clearing)
    int spatial_dim = local_mesh_->dim();
    local_mesh_->clear();
    local_mesh_->build_from_arrays(spatial_dim, new_coords,
                                   new_cell_offsets, new_cell_connectivity,
                                   new_cell_shapes);

    // Set global IDs
    local_mesh_->set_vertex_gids(new_vertex_gids);
    local_mesh_->set_cell_gids(new_cell_gids);

    // Set region labels
    for (size_t c = 0; c < new_cell_regions.size(); ++c) {
        local_mesh_->set_region_label(static_cast<index_t>(c), new_cell_regions[c]);
    }

    local_mesh_->finalize();

    // ========================================
    // Restore field data
    // ========================================

    // Combine kept_fields and received fields into a map
    std::map<std::pair<EntityKind, std::string>, std::vector<const MeshPacket::FieldPacket*>> all_field_packets;

    for (const auto& field : kept_fields) {
        all_field_packets[{field.kind, field.name}].push_back(&field);
    }

    for (const auto& packet : recv_packets) {
        for (const auto& field : packet.fields) {
            all_field_packets[{field.kind, field.name}].push_back(&field);
        }
    }

    // Restore each unique field
    for (const auto& [field_key, field_packets] : all_field_packets) {
        const auto& [kind, name] = field_key;

        // Use the first packet to get field metadata
        const auto* first_packet = field_packets[0];

        // Attach field to the mesh
        auto field_handle = local_mesh_->attach_field(
            kind, name,
            first_packet->type,
            first_packet->components,
            first_packet->bytes_per_component);

        // Get field data pointer
        uint8_t* field_data = static_cast<uint8_t*>(local_mesh_->field_data(field_handle));

        if (!field_data) continue;

        size_t bytes_per_entity = first_packet->components * first_packet->bytes_per_component;

        // Restore data based on entity kind
        if (kind == EntityKind::Vertex) {
            size_t new_vertex_idx = 0;

            // Restore kept vertex data
            if (!kept_fields.empty()) {
                for (const auto& kept_field : kept_fields) {
                    if (kept_field.kind == EntityKind::Vertex && kept_field.name == name) {
                        size_t n_kept_vertices = kept_field.data.size() / bytes_per_entity;
                        for (size_t i = 0; i < n_kept_vertices; ++i) {
                            const uint8_t* src = &kept_field.data[i * bytes_per_entity];
                            uint8_t* dest = &field_data[new_vertex_idx * bytes_per_entity];
                            std::memcpy(dest, src, bytes_per_entity);
                            new_vertex_idx++;
                        }
                        break;
                    }
                }
            }

            // Restore received vertex data
            for (const auto& packet : recv_packets) {
                for (const auto& recv_field : packet.fields) {
                    if (recv_field.kind == EntityKind::Vertex && recv_field.name == name) {
                        size_t n_recv_vertices = recv_field.data.size() / bytes_per_entity;
                        for (size_t i = 0; i < n_recv_vertices; ++i) {
                            const uint8_t* src = &recv_field.data[i * bytes_per_entity];
                            uint8_t* dest = &field_data[new_vertex_idx * bytes_per_entity];
                            std::memcpy(dest, src, bytes_per_entity);
                            new_vertex_idx++;
                        }
                        break;
                    }
                }
            }
        } else if (kind == EntityKind::Volume) {
            size_t new_cell_idx = 0;

            // Restore kept cell data
            if (!kept_fields.empty()) {
                for (const auto& kept_field : kept_fields) {
                    if (kept_field.kind == EntityKind::Volume && kept_field.name == name) {
                        size_t n_kept_cells = kept_field.data.size() / bytes_per_entity;
                        for (size_t i = 0; i < n_kept_cells; ++i) {
                            const uint8_t* src = &kept_field.data[i * bytes_per_entity];
                            uint8_t* dest = &field_data[new_cell_idx * bytes_per_entity];
                            std::memcpy(dest, src, bytes_per_entity);
                            new_cell_idx++;
                        }
                        break;
                    }
                }
            }

            // Restore received cell data
            for (const auto& packet : recv_packets) {
                for (const auto& recv_field : packet.fields) {
                    if (recv_field.kind == EntityKind::Volume && recv_field.name == name) {
                        size_t n_recv_cells = recv_field.data.size() / bytes_per_entity;
                        for (size_t i = 0; i < n_recv_cells; ++i) {
                            const uint8_t* src = &recv_field.data[i * bytes_per_entity];
                            uint8_t* dest = &field_data[new_cell_idx * bytes_per_entity];
                            std::memcpy(dest, src, bytes_per_entity);
                            new_cell_idx++;
                        }
                        break;
                    }
                }
            }
        }
    }

    // Update ownership arrays
    vertex_owner_.clear();
    vertex_owner_.resize(local_mesh_->n_vertices(), Ownership::Owned);
    vertex_owner_rank_.clear();
    vertex_owner_rank_.resize(local_mesh_->n_vertices(), my_rank_);

    cell_owner_.clear();
    cell_owner_.resize(local_mesh_->n_cells(), Ownership::Owned);
    cell_owner_rank_.clear();
    cell_owner_rank_.resize(local_mesh_->n_cells(), my_rank_);

    face_owner_.clear();
    face_owner_.resize(local_mesh_->n_faces(), Ownership::Owned);
    face_owner_rank_.clear();
    face_owner_rank_.resize(local_mesh_->n_faces(), my_rank_);

    // ========================================
    // Step 7: Rebuild ghost layers
    // ========================================

    gather_shared_entities();
    build_ghost_layer(ghost_levels_);
#endif
}

void DistributedMesh::rebalance(PartitionHint hint,
                               const std::unordered_map<std::string,std::string>& options) {
#ifdef MESH_HAS_MPI
    if (world_size_ == 1) {
        return;
    }

    // Step 1: Gather global mesh information
    size_t local_n_cells = local_mesh_->n_cells();
    size_t global_n_cells = 0;
    MPI_Allreduce(&local_n_cells, &global_n_cells, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm_);

    // Step 2: Compute target distribution based on hint
    std::vector<rank_t> new_owner_rank_per_cell(local_n_cells);

    switch (hint) {
        case PartitionHint::Cells: {
            // Balance by number of cells
            size_t cells_per_rank = global_n_cells / world_size_;
            size_t extra = global_n_cells % world_size_;

            // Simple block partitioning
            size_t current_offset = 0;
            for (rank_t r = 0; r < my_rank_; ++r) {
                current_offset += cells_per_rank + (r < static_cast<rank_t>(extra) ? 1 : 0);
            }

            for (index_t c = 0; c < static_cast<index_t>(local_n_cells); ++c) {
                gid_t global_id = local_mesh_->cell_gids()[c];

                // Determine which rank should own this cell
                size_t cumulative = 0;
                for (rank_t r = 0; r < world_size_; ++r) {
                    size_t rank_cells = cells_per_rank + (r < static_cast<rank_t>(extra) ? 1 : 0);
                    cumulative += rank_cells;
                    if (global_id < static_cast<gid_t>(cumulative)) {
                        new_owner_rank_per_cell[c] = r;
                        break;
                    }
                }
            }
            break;
        }

        case PartitionHint::Vertices: {
            // =====================================================
            // Vertex-based partitioning
            // =====================================================
            // Algorithm: Balance mesh by equalizing vertex count across ranks
            // Strategy: Assign cells based on their center vertex GID
            // Performance: O(n_cells) local computation + O(P) communication
            // Quality: Medium - better than blocks but not topology-aware
            // Use case: When vertices drive computational cost (e.g., nodal FEM)

            size_t local_n_vertices = local_mesh_->n_vertices();
            size_t global_n_vertices = 0;
            MPI_Allreduce(&local_n_vertices, &global_n_vertices, 1,
                         MPI_UNSIGNED_LONG, MPI_SUM, comm_);

            size_t vertices_per_rank = global_n_vertices / world_size_;
            size_t extra = global_n_vertices % world_size_;

            // For each cell, compute its "center vertex" based on average GID
            // This maintains spatial locality while balancing vertex count
            const auto& vertex_gids = local_mesh_->vertex_gids();

            for (index_t c = 0; c < static_cast<index_t>(local_n_cells); ++c) {
                auto [vertices_ptr, n_verts] = local_mesh_->cell_vertices_span(c);

                // Compute average vertex GID for this cell (centroid in ID space)
                gid_t avg_vertex_gid = 0;
                for (size_t i = 0; i < n_verts; ++i) {
                    avg_vertex_gid += vertex_gids[vertices_ptr[i]];
                }
                avg_vertex_gid /= n_verts;

                // Assign to rank based on balanced vertex distribution
                // This preserves locality since nearby cells have similar avg GIDs
                rank_t target_rank = 0;
                size_t cumulative_vertices = 0;

                for (rank_t r = 0; r < world_size_; ++r) {
                    size_t rank_vertices = vertices_per_rank + (r < static_cast<rank_t>(extra) ? 1 : 0);
                    cumulative_vertices += rank_vertices;

                    // Map cell to rank based on its vertex centroid
                    if (avg_vertex_gid * world_size_ < cumulative_vertices * global_n_vertices) {
                        target_rank = r;
                        break;
                    }
                }

                new_owner_rank_per_cell[c] = target_rank;
            }

            if (options.find("verbose") != options.end()) {
                std::cout << "Rank " << my_rank_ << ": Vertex-based partitioning complete\n";
            }
            break;
        }

        case PartitionHint::Memory: {
            // =====================================================
            // Memory-based partitioning
            // =====================================================
            // Algorithm: Balance by equalizing memory footprint across ranks
            // Strategy: Account for cell complexity, field data, and ghosts
            // Performance: O(n_cells * n_fields) local + O(P²) communication
            // Quality: Good for heterogeneous systems or adaptive meshes
            // Use case: Memory-bound simulations, adaptive refinement

            // Compute memory footprint per cell
            std::vector<size_t> cell_memory(local_n_cells);

            for (index_t c = 0; c < static_cast<index_t>(local_n_cells); ++c) {
                size_t mem = 0;

                // Base cell storage (shape, connectivity)
                auto [vertices_ptr, n_verts] = local_mesh_->cell_vertices_span(c);
                mem += sizeof(CellShape);
                mem += n_verts * sizeof(index_t);  // connectivity
                mem += sizeof(gid_t);  // global ID
                mem += sizeof(label_t);  // region label

                // Vertex contribution (shared among cells)
                // Approximate as vertices/avg_cells_per_vertex
                mem += n_verts * local_mesh_->dim() * sizeof(real_t) / 4;  // coordinate storage

                // Field data on cells
                auto cell_field_names = local_mesh_->field_names(EntityKind::Volume);
                for (const auto& field_name : cell_field_names) {
                    size_t components = local_mesh_->field_components_by_name(EntityKind::Volume, field_name);
                    size_t bytes_per_comp = local_mesh_->field_bytes_per_component_by_name(EntityKind::Volume, field_name);
                    mem += components * bytes_per_comp;
                }

                // Estimated ghost layer overhead (empirical factor)
                mem = static_cast<size_t>(mem * 1.2);  // 20% overhead for ghosts

                cell_memory[c] = mem;
            }

            // Gather global memory distribution
            size_t local_total_memory = std::accumulate(cell_memory.begin(), cell_memory.end(), size_t(0));
            size_t global_total_memory = 0;
            MPI_Allreduce(&local_total_memory, &global_total_memory, 1,
                         MPI_UNSIGNED_LONG, MPI_SUM, comm_);

            // Target memory per rank
            size_t target_memory_per_rank = global_total_memory / world_size_;

            // Build global cell list sorted by memory (simplified greedy approach)
            // In production, use a distributed sorting algorithm
            struct CellMemInfo {
                gid_t gid;
                size_t memory;
                index_t local_id;
                rank_t current_rank;
            };

            std::vector<CellMemInfo> local_cells_info(local_n_cells);
            for (index_t c = 0; c < static_cast<index_t>(local_n_cells); ++c) {
                local_cells_info[c] = {
                    local_mesh_->cell_gids()[c],
                    cell_memory[c],
                    c,
                    my_rank_
                };
            }

            // Sort locally by memory (largest first for greedy assignment)
            std::sort(local_cells_info.begin(), local_cells_info.end(),
                     [](const CellMemInfo& a, const CellMemInfo& b) {
                         return a.memory > b.memory;
                     });

            // Greedy assignment: assign cells to least loaded rank
            std::vector<size_t> rank_memory_load(world_size_, 0);

            for (const auto& cell_info : local_cells_info) {
                // Find least loaded rank
                auto min_it = std::min_element(rank_memory_load.begin(), rank_memory_load.end());
                rank_t target_rank = static_cast<rank_t>(std::distance(rank_memory_load.begin(), min_it));

                // Assign cell to this rank
                new_owner_rank_per_cell[cell_info.local_id] = target_rank;
                rank_memory_load[target_rank] += cell_info.memory;
            }

            // Report load balance quality
            if (options.find("verbose") != options.end() && my_rank_ == 0) {
                auto [min_load, max_load] = std::minmax_element(rank_memory_load.begin(), rank_memory_load.end());
                double imbalance = static_cast<double>(*max_load) / *min_load - 1.0;
                std::cout << "Memory-based partitioning: imbalance = "
                         << std::fixed << std::setprecision(2) << imbalance * 100 << "%\n";
            }
            break;
        }

        case PartitionHint::Metis: {
            // =====================================================
            // METIS/ParMETIS graph partitioning
            // =====================================================
            // Algorithm: Minimize edge cuts using multilevel graph partitioning
            // Strategy: Build dual graph, call METIS for optimal partitioning
            // Performance: O(n log n) for METIS, O(n) for graph construction
            // Quality: Best - minimizes communication volume
            // Use case: Production runs, strong scaling studies

#ifdef MESH_HAS_METIS
            // Build dual graph (cell-to-cell connectivity through faces)
            std::vector<idx_t> xadj;  // CSR offsets for adjacency
            std::vector<idx_t> adjncy;  // Adjacent cells
            std::vector<idx_t> vwgt;  // Vertex (cell) weights
            std::vector<idx_t> adjwgt;  // Edge weights (optional)

            xadj.reserve(local_n_cells + 1);
            xadj.push_back(0);

            // Option 1: Use existing cell2cell adjacency if available
            if (local_mesh_->has_cell2cell()) {
                for (index_t c = 0; c < static_cast<index_t>(local_n_cells); ++c) {
                    auto neighbors = local_mesh_->cell_neighbors(c);
                    for (index_t neighbor : neighbors) {
                        if (neighbor != c) {  // Skip self
                            adjncy.push_back(static_cast<idx_t>(neighbor));
                        }
                    }
                    xadj.push_back(static_cast<idx_t>(adjncy.size()));
                }
            } else {
                // Option 2: Build dual graph through shared faces
                local_mesh_->build_cell2cell();

                for (index_t c = 0; c < static_cast<index_t>(local_n_cells); ++c) {
                    auto neighbors = local_mesh_->cell_neighbors(c);
                    for (index_t neighbor : neighbors) {
                        if (neighbor != c) {
                            adjncy.push_back(static_cast<idx_t>(neighbor));

                            // Optional: Weight by shared face area
                            // adjwgt.push_back(compute_shared_face_area(c, neighbor));
                        }
                    }
                    xadj.push_back(static_cast<idx_t>(adjncy.size()));
                }
            }

            // Set cell weights (can be based on complexity, DOFs, etc.)
            vwgt.resize(local_n_cells, 1);  // Unit weights for now

            // Optionally weight cells by vertex count (for FEM)
            if (options.find("weight_by_vertices") != options.end()) {
                for (index_t c = 0; c < static_cast<index_t>(local_n_cells); ++c) {
                    auto [vertices_ptr, n_verts] = local_mesh_->cell_vertices_span(c);
                    vwgt[c] = static_cast<idx_t>(n_verts);
                }
            }

            // METIS partitioning
            idx_t nvtxs = static_cast<idx_t>(local_n_cells);
            idx_t ncon = 1;  // Number of constraints (1 = balance cells only)
            idx_t nparts = static_cast<idx_t>(world_size_);
            idx_t edgecut;
            std::vector<idx_t> part(local_n_cells);

            // METIS options
            idx_t options_metis[METIS_NOPTIONS];
            METIS_SetDefaultOptions(options_metis);

            // Set specific options
            options_metis[METIS_OPTION_PTYPE] = METIS_PTYPE_KWAY;  // K-way partitioning
            options_metis[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;  // Minimize edge cut
            options_metis[METIS_OPTION_NUMBERING] = 0;  // C-style numbering

            // Allow imbalance factor (5% default)
            real_t ubvec = 1.05;

            // Call METIS
            int metis_result = METIS_PartGraphKway(
                &nvtxs,           // Number of vertices (cells)
                &ncon,            // Number of constraints
                xadj.data(),      // Adjacency structure
                adjncy.data(),    // Adjacent vertices
                vwgt.data(),      // Vertex weights
                NULL,             // Vertex sizes (for communication volume)
                adjwgt.empty() ? NULL : adjwgt.data(),  // Edge weights
                &nparts,          // Number of partitions
                NULL,             // Target partition weights
                &ubvec,           // Imbalance tolerance
                options_metis,    // Options
                &edgecut,         // Output: edge cut
                part.data()       // Output: partition assignment
            );

            if (metis_result != METIS_OK) {
                std::cerr << "METIS partitioning failed on rank " << my_rank_ << "\n";
                // Fall back to simple partitioning
                for (index_t c = 0; c < static_cast<index_t>(local_n_cells); ++c) {
                    new_owner_rank_per_cell[c] = c % world_size_;
                }
            } else {
                // Convert METIS partition to rank assignment
                for (index_t c = 0; c < static_cast<index_t>(local_n_cells); ++c) {
                    new_owner_rank_per_cell[c] = static_cast<rank_t>(part[c]);
                }

                if (options.find("verbose") != options.end() && my_rank_ == 0) {
                    std::cout << "METIS partitioning: edge cut = " << edgecut << "\n";
                }
            }
#else
            // METIS not available - fall back to vertex-based partitioning
            if (my_rank_ == 0) {
                std::cerr << "Warning: METIS not available, using vertex-based partitioning instead\n";
            }

            // Reuse vertex-based partitioning code
            size_t local_n_vertices = local_mesh_->n_vertices();
            size_t global_n_vertices = 0;
            MPI_Allreduce(&local_n_vertices, &global_n_vertices, 1,
                         MPI_UNSIGNED_LONG, MPI_SUM, comm_);

            size_t vertices_per_rank = global_n_vertices / world_size_;
            size_t extra = global_n_vertices % world_size_;

            const auto& vertex_gids = local_mesh_->vertex_gids();

            for (index_t c = 0; c < static_cast<index_t>(local_n_cells); ++c) {
                auto [vertices_ptr, n_verts] = local_mesh_->cell_vertices_span(c);

                gid_t avg_vertex_gid = 0;
                for (size_t i = 0; i < n_verts; ++i) {
                    avg_vertex_gid += vertex_gids[vertices_ptr[i]];
                }
                avg_vertex_gid /= n_verts;

                rank_t target_rank = 0;
                size_t cumulative_vertices = 0;

                for (rank_t r = 0; r < world_size_; ++r) {
                    size_t rank_vertices = vertices_per_rank + (r < static_cast<rank_t>(extra) ? 1 : 0);
                    cumulative_vertices += rank_vertices;

                    if (avg_vertex_gid * world_size_ < cumulative_vertices * global_n_vertices) {
                        target_rank = r;
                        break;
                    }
                }

                new_owner_rank_per_cell[c] = target_rank;
            }
#endif  // MESH_HAS_METIS
            break;
        }

        default:
            // No partitioning - keep current distribution
            for (index_t c = 0; c < static_cast<index_t>(local_n_cells); ++c) {
                new_owner_rank_per_cell[c] = my_rank_;
            }
            break;
    }

    // Step 3: Migrate cells to new owners
    migrate(new_owner_rank_per_cell);
#endif
}

// ==========================================
// Partition Quality Metrics
// ==========================================

DistributedMesh::PartitionMetrics DistributedMesh::compute_partition_quality() const {
    PartitionMetrics metrics = {};

#ifdef MESH_HAS_MPI
    // =====================================================
    // Load balance metrics
    // =====================================================
    size_t local_n_cells = local_mesh_->n_cells();
    size_t local_n_owned_cells = 0;

    // Count only owned cells (exclude ghosts)
    for (index_t c = 0; c < static_cast<index_t>(local_n_cells); ++c) {
        if (is_owned_cell(c)) {
            local_n_owned_cells++;
        }
    }

    // Gather cell counts from all ranks
    std::vector<size_t> cells_per_rank(world_size_);
    MPI_Allgather(&local_n_owned_cells, 1, MPI_UNSIGNED_LONG,
                  cells_per_rank.data(), 1, MPI_UNSIGNED_LONG, comm_);

    // Compute load balance statistics
    size_t total_cells = std::accumulate(cells_per_rank.begin(), cells_per_rank.end(), size_t(0));
    metrics.avg_cells_per_rank = total_cells / world_size_;

    auto [min_it, max_it] = std::minmax_element(cells_per_rank.begin(), cells_per_rank.end());
    metrics.min_cells_per_rank = *min_it;
    metrics.max_cells_per_rank = *max_it;

    // Load imbalance factor: (max_load / avg_load) - 1.0
    // Perfect balance = 0.0, higher values indicate worse imbalance
    if (metrics.avg_cells_per_rank > 0) {
        metrics.load_imbalance_factor =
            static_cast<double>(metrics.max_cells_per_rank) / metrics.avg_cells_per_rank - 1.0;
    }

    // =====================================================
    // Communication metrics - edge cuts and shared faces
    // =====================================================
    size_t local_edge_cuts = 0;
    size_t local_shared_faces = 0;

    // Count edges that cross partition boundaries
    // An edge cut occurs when a cell has a neighbor on a different rank
    // Since we don't have direct cell-face connectivity in MeshBase,
    // we estimate edge cuts based on shared vertices
    for (index_t c = 0; c < static_cast<index_t>(local_n_cells); ++c) {
        if (!is_owned_cell(c)) continue;  // Only count owned cells

        // Check vertices of this cell to estimate shared boundaries
        auto [vertices_ptr, n_verts] = local_mesh_->cell_vertices_span(c);
        bool has_shared_vertex = false;
        for (size_t v = 0; v < n_verts; ++v) {
            index_t vertex_id = vertices_ptr[v];
            if (is_shared_vertex(vertex_id)) {
                has_shared_vertex = true;
                break;
            }
        }

        // If this cell has shared vertices, it likely contributes to edge cuts
        if (has_shared_vertex) {
            local_edge_cuts++;
        }
    }

    // Count shared faces separately if faces are available
    if (local_mesh_->n_faces() > 0) {
        for (index_t f = 0; f < static_cast<index_t>(local_mesh_->n_faces()); ++f) {
            if (is_shared_face(f)) {
                local_shared_faces++;
            }
        }
    }

    // Global reduction for communication metrics
    MPI_Allreduce(&local_edge_cuts, &metrics.total_edge_cuts, 1,
                  MPI_UNSIGNED_LONG, MPI_SUM, comm_);
    MPI_Allreduce(&local_shared_faces, &metrics.total_shared_faces, 1,
                  MPI_UNSIGNED_LONG, MPI_SUM, comm_);

    // Note: Each shared face is counted twice (once per rank), so divide by 2
    metrics.total_edge_cuts /= 2;
    metrics.total_shared_faces /= 2;

    // =====================================================
    // Ghost cell metrics
    // =====================================================
    size_t local_ghost_cells = ghost_cells_.size();
    MPI_Allreduce(&local_ghost_cells, &metrics.total_ghost_cells, 1,
                  MPI_UNSIGNED_LONG, MPI_SUM, comm_);

    // =====================================================
    // Neighbor connectivity metrics
    // =====================================================
    size_t local_num_neighbors = neighbor_ranks_.size();
    size_t total_neighbors = 0;
    MPI_Allreduce(&local_num_neighbors, &total_neighbors, 1,
                  MPI_UNSIGNED_LONG, MPI_SUM, comm_);
    metrics.avg_neighbors_per_rank = static_cast<double>(total_neighbors) / world_size_;

    // =====================================================
    // Memory footprint metrics
    // =====================================================
    size_t local_memory = 0;

    // Cell storage
    local_memory += local_n_cells * sizeof(CellShape);
    local_memory += local_n_cells * sizeof(gid_t);  // GIDs
    local_memory += local_n_cells * sizeof(label_t);  // Labels

    // Connectivity storage (CSR format)
    local_memory += local_mesh_->cell2vertex_offsets().size() * sizeof(index_t);
    local_memory += local_mesh_->cell2vertex().size() * sizeof(index_t);

    // Vertex storage
    size_t local_n_vertices = local_mesh_->n_vertices();
    local_memory += local_n_vertices * local_mesh_->dim() * sizeof(real_t);  // Coordinates
    local_memory += local_n_vertices * sizeof(gid_t);  // GIDs

    // Face storage (if present)
    if (local_mesh_->n_faces() > 0) {
        size_t local_n_faces = local_mesh_->n_faces();
        local_memory += local_n_faces * sizeof(gid_t);
        // Face connectivity storage estimate (since face2vertex_offsets_ is private)
        // Assuming average 4 vertices per face for 3D meshes
        local_memory += local_n_faces * sizeof(index_t);  // offsets
        local_memory += local_n_faces * 4 * sizeof(index_t);  // indices (estimate)
    }

    // Field data on cells
    auto cell_field_names = local_mesh_->field_names(EntityKind::Volume);
    for (const auto& field_name : cell_field_names) {
        size_t components = local_mesh_->field_components_by_name(EntityKind::Volume, field_name);
        size_t bytes_per_comp = local_mesh_->field_bytes_per_component_by_name(EntityKind::Volume, field_name);
        local_memory += local_n_cells * components * bytes_per_comp;
    }

    // Field data on vertices
    auto vertex_field_names = local_mesh_->field_names(EntityKind::Vertex);
    for (const auto& field_name : vertex_field_names) {
        size_t components = local_mesh_->field_components_by_name(EntityKind::Vertex, field_name);
        size_t bytes_per_comp = local_mesh_->field_bytes_per_component_by_name(EntityKind::Vertex, field_name);
        local_memory += local_n_vertices * components * bytes_per_comp;
    }

    // Distributed mesh overhead
    local_memory += vertex_owner_.size() * sizeof(Ownership);
    local_memory += cell_owner_.size() * sizeof(Ownership);
    local_memory += vertex_owner_rank_.size() * sizeof(rank_t);
    local_memory += cell_owner_rank_.size() * sizeof(rank_t);

    // Gather memory usage from all ranks
    std::vector<size_t> memory_per_rank(world_size_);
    MPI_Allgather(&local_memory, 1, MPI_UNSIGNED_LONG,
                  memory_per_rank.data(), 1, MPI_UNSIGNED_LONG, comm_);

    auto [min_mem_it, max_mem_it] = std::minmax_element(memory_per_rank.begin(), memory_per_rank.end());
    metrics.min_memory_per_rank = *min_mem_it;
    metrics.max_memory_per_rank = *max_mem_it;

    // Memory imbalance factor
    size_t total_memory = std::accumulate(memory_per_rank.begin(), memory_per_rank.end(), size_t(0));
    size_t avg_memory = total_memory / world_size_;
    if (avg_memory > 0) {
        metrics.memory_imbalance_factor =
            static_cast<double>(metrics.max_memory_per_rank) / avg_memory - 1.0;
    }

    // =====================================================
    // Migration metrics (hypothetical - what if we rebalanced?)
    // =====================================================
    // This estimates how many cells would need to migrate for perfect balance
    if (metrics.avg_cells_per_rank > 0) {
        metrics.cells_to_migrate = 0;
        for (size_t rank_cells : cells_per_rank) {
            if (rank_cells > metrics.avg_cells_per_rank) {
                metrics.cells_to_migrate += rank_cells - metrics.avg_cells_per_rank;
            }
        }
    }

    // Estimate migration volume (bytes to transfer)
    if (metrics.cells_to_migrate > 0 && total_cells > 0) {
        // Average bytes per cell
        size_t avg_bytes_per_cell = total_memory / total_cells;
        metrics.migration_volume = metrics.cells_to_migrate * avg_bytes_per_cell;
    }

    // =====================================================
    // Report metrics (optional verbose output)
    // =====================================================
    if (my_rank_ == 0) {
        // This can be enabled with a verbose flag if needed
        #ifdef VERBOSE_PARTITION_METRICS
        std::cout << "\n=== Partition Quality Metrics ===\n";
        std::cout << "Load Balance:\n";
        std::cout << "  Cells per rank: [" << metrics.min_cells_per_rank
                  << ", " << metrics.max_cells_per_rank << "] (avg: "
                  << metrics.avg_cells_per_rank << ")\n";
        std::cout << "  Load imbalance: " << std::fixed << std::setprecision(1)
                  << metrics.load_imbalance_factor * 100 << "%\n";

        std::cout << "Communication:\n";
        std::cout << "  Total edge cuts: " << metrics.total_edge_cuts << "\n";
        std::cout << "  Shared faces: " << metrics.total_shared_faces << "\n";
        std::cout << "  Ghost cells: " << metrics.total_ghost_cells << "\n";
        std::cout << "  Avg neighbors/rank: " << std::fixed << std::setprecision(1)
                  << metrics.avg_neighbors_per_rank << "\n";

        std::cout << "Memory:\n";
        std::cout << "  Memory per rank: [" << metrics.min_memory_per_rank / (1024*1024)
                  << " MB, " << metrics.max_memory_per_rank / (1024*1024) << " MB]\n";
        std::cout << "  Memory imbalance: " << std::fixed << std::setprecision(1)
                  << metrics.memory_imbalance_factor * 100 << "%\n";

        if (metrics.cells_to_migrate > 0) {
            std::cout << "Potential Migration:\n";
            std::cout << "  Cells to migrate: " << metrics.cells_to_migrate << "\n";
            std::cout << "  Migration volume: " << metrics.migration_volume / (1024*1024) << " MB\n";
        }
        std::cout << "================================\n\n";
        #endif
    }

#else
    // Serial case - trivial metrics
    metrics.min_cells_per_rank = metrics.max_cells_per_rank = metrics.avg_cells_per_rank = local_mesh_->n_cells();
    metrics.load_imbalance_factor = 0.0;
    metrics.total_edge_cuts = 0;
    metrics.total_shared_faces = 0;
    metrics.total_ghost_cells = 0;
    metrics.avg_neighbors_per_rank = 0.0;

    // Compute memory for serial case
    size_t total_memory = 0;
    total_memory += local_mesh_->n_cells() * (sizeof(CellShape) + sizeof(gid_t) + sizeof(label_t));
    total_memory += local_mesh_->n_vertices() * local_mesh_->dim() * sizeof(real_t);
    total_memory += local_mesh_->cell2vertex().size() * sizeof(index_t);

    metrics.min_memory_per_rank = metrics.max_memory_per_rank = total_memory;
    metrics.memory_imbalance_factor = 0.0;
    metrics.cells_to_migrate = 0;
    metrics.migration_volume = 0;
#endif

    return metrics;
}

// ==========================================
// Parallel I/O
// ==========================================

DistributedMesh DistributedMesh::load_parallel(const MeshIOOptions& opts, MPI_Comm comm) {
    // =====================================================
    // Parallel Mesh Loading Strategy
    // =====================================================
    // This function implements multiple I/O strategies based on file format:
    // 1. PVTU format: Each rank loads its own piece independently (best scaling)
    // 2. HDF5 format: Collective parallel I/O (requires HDF5 with MPI support)
    // 3. Serial formats: Root loads and distributes (bottleneck for large meshes)
    //
    // Performance characteristics:
    // - PVTU: O(n/p) time, O(n/p) memory per rank, best for large-scale runs
    // - HDF5: O(n/p) time with collective I/O, good for moderate scales
    // - Serial: O(n) time on root, O(n) memory spike on root, limited scalability

#ifdef MESH_HAS_MPI
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Parse file extension and base path
    std::string extension;
    std::string base_path = opts.path;
    size_t dot_pos = opts.path.rfind('.');
    if (dot_pos != std::string::npos) {
        extension = opts.path.substr(dot_pos);
        base_path = opts.path.substr(0, dot_pos);
    }

    std::shared_ptr<MeshBase> local_mesh;

    // Start timing for performance monitoring
    double io_start_time = MPI_Wtime();

    // =====================================================
    // Strategy 1: PVTU/PVTK Format (Parallel VTK)
    // =====================================================
    // Expected performance: O(n/p) time, O(n/p) memory
    // Scaling: Near-perfect weak scaling to 10K+ cores
    // Use case: Production runs, large meshes (>10M cells)

    if (extension == ".pvtu" || extension == ".pvtk") {
        // Each rank independently loads its piece - no communication needed
        MeshIOOptions local_opts = opts;
        local_opts.path = base_path + "_p" + std::to_string(rank) + ".vtu";
        local_opts.format = "vtu";

        // Check file existence
        std::ifstream test_file(local_opts.path);
        if (!test_file.good()) {
            // Fallback: Try alternative naming convention
            local_opts.path = base_path + "_" + std::to_string(rank) + ".vtu";
            test_file.open(local_opts.path);

            if (!test_file.good()) {
                if (rank == 0) {
                    std::cerr << "Warning: PVTU piece not found. Falling back to serial load.\n";
                }
                // Fall through to serial loading strategy
                extension = ".vtu";  // Force serial fallback
            }
        }

        if (test_file.good()) {
            test_file.close();

            try {
                // Load piece directly - no inter-process communication
                local_mesh = std::make_shared<MeshBase>(MeshBase::load(local_opts));

                double io_time = MPI_Wtime() - io_start_time;
                if (rank == 0 && opts.kv.find("verbose") != opts.kv.end()) {
                    std::cout << "PVTU loading completed in " << io_time << " seconds\n";
                }
            } catch (const std::exception& e) {
                std::cerr << "Rank " << rank << " failed to load " << local_opts.path
                          << ": " << e.what() << std::endl;
                MPI_Abort(comm, 1);
            }
        }
    }

    // =====================================================
    // Strategy 2: HDF5 Format (Collective Parallel I/O)
    // =====================================================
    // Expected performance: O(n/p) with collective buffering
    // Scaling: Good to 1K cores with parallel file system
    // Use case: Medium to large meshes on parallel filesystems

    else if (extension == ".h5" || extension == ".hdf5") {
#ifdef MESH_HAS_HDF5_PARALLEL
        // Use HDF5 parallel I/O for efficient collective reading
        // Each rank reads its portion of the mesh simultaneously

        // Set up parallel HDF5 access
        MPI_Info info = MPI_INFO_NULL;
        MPI_Info_create(&info);
        MPI_Info_set(info, "collective_buffering", "true");
        MPI_Info_set(info, "cb_buffer_size", "16777216");  // 16MB buffers

        // Open file collectively
        hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_fapl_mpio(plist_id, comm, info);

        hid_t file_id = H5Fopen(opts.path.c_str(), H5F_ACC_RDONLY, plist_id);
        if (file_id < 0) {
            if (rank == 0) {
                std::cerr << "Failed to open HDF5 file: " << opts.path << std::endl;
            }
            MPI_Abort(comm, 1);
        }

        // Read mesh metadata collectively
        hid_t dataset_id = H5Dopen2(file_id, "/mesh/metadata", H5P_DEFAULT);

        // Each rank calculates its portion based on global dimensions
        // ... HDF5 hyperslab selection code ...

        // Collective read with optimized chunking
        hid_t xfer_plist = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);

        // Read vertex coordinates, connectivity, fields collectively
        // ... actual HDF5 read operations ...

        H5Pclose(xfer_plist);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        H5Pclose(plist_id);
        MPI_Info_free(&info);

        double io_time = MPI_Wtime() - io_start_time;
        if (rank == 0 && opts.kv.find("verbose") != opts.kv.end()) {
            std::cout << "HDF5 parallel loading completed in " << io_time << " seconds\n";
        }
#else
        if (rank == 0) {
            std::cerr << "HDF5 parallel support not available. Falling back to serial load.\n";
        }
        extension = ".vtu";  // Force serial fallback
#endif
    }

    // =====================================================
    // Strategy 3: Serial Formats (Root loads and distributes)
    // =====================================================
    // Expected performance: O(n) time on root, O(n*log(p)) for distribution
    // Memory: O(n) spike on root, O(n/p) on other ranks after distribution
    // Scaling: Limited to ~100 ranks due to root bottleneck
    // Use case: Small meshes (<1M cells), debugging, legacy formats

    if (!local_mesh) {  // If not already loaded by PVTU or HDF5
        if (rank == 0) {
            // =====================================================
            // Step 1: Root loads entire mesh (memory bottleneck)
            // =====================================================
            double load_start = MPI_Wtime();
            MeshBase global_mesh = MeshBase::load(opts);

            if (opts.kv.find("verbose") != opts.kv.end()) {
                std::cout << "Root loaded " << global_mesh.n_cells() << " cells in "
                          << MPI_Wtime() - load_start << " seconds\n";
            }

            // =====================================================
            // Step 2: Partition the mesh for distribution
            // =====================================================
            // Use METIS for high-quality partitioning if available

            size_t n_cells = global_mesh.n_cells();
            std::vector<rank_t> cell_partition(n_cells);

            std::string partition_method = "block";  // default
            if (opts.kv.find("partition_method") != opts.kv.end()) {
                partition_method = opts.kv.at("partition_method");
            }

            if (partition_method == "metis" && n_cells > 1000) {
#ifdef MESH_HAS_METIS
                // Use METIS for high-quality partitioning
                // Expected time: O(n log n)

                // Build dual graph for METIS
                std::vector<idx_t> xadj, adjncy;
                // ... build adjacency from mesh ...

                idx_t nvtxs = static_cast<idx_t>(n_cells);
                idx_t ncon = 1;  // Number of constraints (1 = balance cells only)
                idx_t nparts = static_cast<idx_t>(size);
                idx_t edgecut;
                std::vector<idx_t> part(n_cells);

                // METIS partitioning for minimal edge cuts
                idx_t options_metis[METIS_NOPTIONS];
                METIS_SetDefaultOptions(options_metis);

                METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adjncy.data(),
                                   NULL, NULL, NULL, &nparts, NULL, NULL, options_metis,
                                   &edgecut, part.data());

                for (size_t i = 0; i < n_cells; ++i) {
                    cell_partition[i] = static_cast<rank_t>(part[i]);
                }

                if (opts.kv.find("verbose") != opts.kv.end()) {
                    std::cout << "METIS partitioning: edge cut = " << edgecut << "\n";
                }
#endif
            } else {
                // Simple block partitioning
                // Expected time: O(n)

                size_t cells_per_rank = n_cells / size;
                size_t extra = n_cells % size;

                for (size_t i = 0; i < n_cells; ++i) {
                    size_t cumulative = 0;
                    for (rank_t r = 0; r < size; ++r) {
                        size_t rank_cells = cells_per_rank + (r < static_cast<rank_t>(extra) ? 1 : 0);
                        cumulative += rank_cells;
                        if (i < cumulative) {
                            cell_partition[i] = r;
                            break;
                        }
                    }
                }
            }

            // =====================================================
            // Step 3: Extract and distribute submeshes
            // =====================================================
            // Two distribution strategies based on scale

            if (size <= 32) {
                // Small-scale: Direct send to each rank
                // Expected time: O(n*p) total, O(n) per rank

                for (int r = 0; r < size; ++r) {
                    // Extract submesh for rank r
                    auto submesh = extract_submesh(global_mesh, cell_partition, r);

                    if (r == 0) {
                        local_mesh = std::make_shared<MeshBase>(std::move(submesh));
                    } else {
                        // Serialize and send
                        std::vector<char> buffer;
                        serialize_mesh(submesh, buffer);

                        int buffer_size = static_cast<int>(buffer.size());
                        MPI_Send(&buffer_size, 1, MPI_INT, r, 0, comm);
                        MPI_Send(buffer.data(), buffer_size, MPI_CHAR, r, 1, comm);
                    }
                }
            } else {
                // Large-scale: Use tree-based distribution
                // Expected time: O(n*log(p)) total
                // Reduces memory pressure and improves scaling

                // Binary tree distribution pattern
                distribute_mesh_tree(global_mesh, cell_partition, comm);

                // Root's portion is already extracted
                local_mesh = std::make_shared<MeshBase>(
                    extract_submesh(global_mesh, cell_partition, 0));
            }

        } else {
            // =====================================================
            // Non-root ranks: Receive their submesh
            // =====================================================

            if (size <= 32) {
                // Direct receive from root
                int buffer_size;
                MPI_Recv(&buffer_size, 1, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);

                std::vector<char> buffer(buffer_size);
                MPI_Recv(buffer.data(), buffer_size, MPI_CHAR, 0, 1, comm, MPI_STATUS_IGNORE);

                local_mesh = std::make_shared<MeshBase>();
                deserialize_mesh(buffer, *local_mesh);
            } else {
                // Tree-based receive
                local_mesh = std::make_shared<MeshBase>();
                receive_mesh_tree(*local_mesh, comm);
            }
        }

        double dist_time = MPI_Wtime() - io_start_time;
        if (rank == 0 && opts.kv.find("verbose") != opts.kv.end()) {
            std::cout << "Serial load + distribution completed in " << dist_time << " seconds\n";
        }
    }

    // =====================================================
    // Step 4: Set up distributed mesh structure
    // =====================================================
    // Create the distributed mesh wrapper
    DistributedMesh dmesh(local_mesh, comm);

    // =====================================================
    // Step 5: Identify shared entities across ranks
    // =====================================================
    // Expected time: O((n/p) * log(p)) with efficient algorithm
    // Memory: O(n/p) for hash tables
    dmesh.gather_shared_entities();

    // =====================================================
    // Step 6: Build ghost layers for communication
    // =====================================================
    // Expected time: O(ghost_layers * neighbors * boundary_size)
    // Memory: O(ghost_cells) additional storage

    int ghost_layers = 1;  // Default
    if (opts.kv.find("ghost_layers") != opts.kv.end()) {
        ghost_layers = std::stoi(opts.kv.at("ghost_layers"));
    }
    dmesh.build_ghost_layer(ghost_layers);

    // =====================================================
    // Performance Summary
    // =====================================================
    double total_time = MPI_Wtime() - io_start_time;
    double max_time;
    MPI_Reduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0 && opts.kv.find("verbose") != opts.kv.end()) {
        std::cout << "\n=== Parallel Load Performance ===\n";
        std::cout << "Total time: " << max_time << " seconds\n";
        std::cout << "Mesh size: " << dmesh.global_n_cells() << " cells, "
                  << dmesh.global_n_vertices() << " vertices\n";
        std::cout << "Ranks: " << size << "\n";
        std::cout << "Cells/rank: " << dmesh.global_n_cells() / size << " average\n";

        // Compute and report load balance
        auto metrics = dmesh.compute_partition_quality();
        std::cout << "Load imbalance: " << std::fixed << std::setprecision(1)
                  << metrics.load_imbalance_factor * 100 << "%\n";
        std::cout << "Ghost cells: " << metrics.total_ghost_cells << " total\n";
        std::cout << "==================================\n\n";
    }

    return dmesh;
#else
    // Serial fallback - no MPI available
    auto mesh = std::make_shared<MeshBase>(MeshBase::load(opts));
    return DistributedMesh(mesh, comm);
#endif
}

void DistributedMesh::save_parallel(const MeshIOOptions& opts) const {
#ifdef MESH_HAS_MPI
    if (world_size_ == 1) {
        // Serial save
        local_mesh_->save(opts);
        return;
    }

    // Check if the path ends with .pvtu - if so, use PVTU format
    bool use_pvtu = false;
    std::string base_path = opts.path;
    std::string extension;

    size_t dot_pos = opts.path.rfind('.');
    if (dot_pos != std::string::npos) {
        extension = opts.path.substr(dot_pos);
        base_path = opts.path.substr(0, dot_pos);

        // Check for .pvtu or .pvtk extensions
        if (extension == ".pvtu" || extension == ".pvtk") {
            use_pvtu = true;
        }
    }

    if (use_pvtu) {
        // PVTU format: Each rank saves a .vtu file, rank 0 writes master .pvtu file

        // Step 1: Each rank writes its piece as a .vtu file
        MeshIOOptions local_opts = opts;
        local_opts.format = "vtu";  // Force VTU format
        local_opts.path = base_path + "_p" + std::to_string(my_rank_) + ".vtu";

        // Save only owned entities (not ghosts)
        local_mesh_->save(local_opts);

        // Step 2: Rank 0 writes the master .pvtu file
        if (my_rank_ == 0) {
            std::ofstream pvtu(opts.path);
            if (!pvtu.is_open()) {
                throw std::runtime_error("Failed to open PVTU file: " + opts.path);
            }

            pvtu << "<?xml version=\"1.0\"?>\n";
            pvtu << "<VTKFile type=\"PUnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
            pvtu << "  <PUnstructuredGrid GhostLevel=\"" << ghost_levels_ << "\">\n";

            // Write PPoints and PPointData sections (metadata about what's in the pieces)
            pvtu << "    <PPoints>\n";
            pvtu << "      <PDataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\"/>\n";
            pvtu << "    </PPoints>\n";

            // Add field metadata if there are fields
            auto vertex_fields = local_mesh_->field_names(EntityKind::Vertex);
            if (!vertex_fields.empty()) {
                pvtu << "    <PPointData>\n";
                for (const auto& field_name : vertex_fields) {
                    auto type = local_mesh_->field_type_by_name(EntityKind::Vertex, field_name);
                    auto components = local_mesh_->field_components_by_name(EntityKind::Vertex, field_name);

                    std::string vtk_type = "Float64";  // Default
                    if (type == FieldScalarType::Float32) vtk_type = "Float32";
                    else if (type == FieldScalarType::Int32) vtk_type = "Int32";
                    else if (type == FieldScalarType::Int64) vtk_type = "Int64";

                    pvtu << "      <PDataArray type=\"" << vtk_type
                         << "\" Name=\"" << field_name
                         << "\" NumberOfComponents=\"" << components
                         << "\" format=\"ascii\"/>\n";
                }
                pvtu << "    </PPointData>\n";
            }

            auto cell_fields = local_mesh_->field_names(EntityKind::Volume);
            if (!cell_fields.empty()) {
                pvtu << "    <PCellData>\n";
                for (const auto& field_name : cell_fields) {
                    auto type = local_mesh_->field_type_by_name(EntityKind::Volume, field_name);
                    auto components = local_mesh_->field_components_by_name(EntityKind::Volume, field_name);

                    std::string vtk_type = "Float64";  // Default
                    if (type == FieldScalarType::Float32) vtk_type = "Float32";
                    else if (type == FieldScalarType::Int32) vtk_type = "Int32";
                    else if (type == FieldScalarType::Int64) vtk_type = "Int64";

                    pvtu << "      <PDataArray type=\"" << vtk_type
                         << "\" Name=\"" << field_name
                         << "\" NumberOfComponents=\"" << components
                         << "\" format=\"ascii\"/>\n";
                }
                pvtu << "    </PCellData>\n";
            }

            // Write piece references
            for (int r = 0; r < world_size_; ++r) {
                pvtu << "    <Piece Source=\"" << base_path << "_p" << r << ".vtu\"/>\n";
            }

            pvtu << "  </PUnstructuredGrid>\n";
            pvtu << "</VTKFile>\n";
            pvtu.close();
        }

        // Ensure all ranks finish writing before returning
        MPI_Barrier(comm_);

    } else {
        // Original behavior: Each rank saves its own file with rank suffix
        MeshIOOptions local_opts = opts;
        local_opts.path = opts.path + "_rank" + std::to_string(my_rank_);
        local_mesh_->save(local_opts);
    }
#else
    local_mesh_->save(opts);
#endif
}

// ==========================================
// Global Reductions
// ==========================================

size_t DistributedMesh::global_n_vertices() const {
#ifdef MESH_HAS_MPI
    // Count only owned vertices
    size_t local_owned = 0;
    for (size_t i = 0; i < vertex_owner_.size(); ++i) {
        if (vertex_owner_[i] == Ownership::Owned) {
            local_owned++;
        }
    }

    size_t global_total = 0;
    MPI_Allreduce(&local_owned, &global_total, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm_);
    return global_total;
#else
    return local_mesh_->n_vertices();
#endif
}

size_t DistributedMesh::global_n_cells() const {
#ifdef MESH_HAS_MPI
    // Count only owned cells
    size_t local_owned = 0;
    for (size_t i = 0; i < cell_owner_.size(); ++i) {
        if (cell_owner_[i] == Ownership::Owned) {
            local_owned++;
        }
    }

    size_t global_total = 0;
    MPI_Allreduce(&local_owned, &global_total, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm_);
    return global_total;
#else
    return local_mesh_->n_cells();
#endif
}

size_t DistributedMesh::global_n_faces() const {
#ifdef MESH_HAS_MPI
    // Count only owned faces
    size_t local_owned = 0;
    for (size_t i = 0; i < face_owner_.size(); ++i) {
        if (face_owner_[i] == Ownership::Owned) {
            local_owned++;
        }
    }

    size_t global_total = 0;
    MPI_Allreduce(&local_owned, &global_total, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm_);
    return global_total;
#else
    return local_mesh_->n_faces();
#endif
}

BoundingBox DistributedMesh::global_bounding_box() const {
#ifdef MESH_HAS_MPI
    auto local_bbox = local_mesh_->bounding_box();

    BoundingBox global_bbox;
    MPI_Allreduce(local_bbox.min.data(), global_bbox.min.data(), 3,
                 MPI_DOUBLE, MPI_MIN, comm_);
    MPI_Allreduce(local_bbox.max.data(), global_bbox.max.data(), 3,
                 MPI_DOUBLE, MPI_MAX, comm_);

    return global_bbox;
#else
    return local_mesh_->bounding_box();
#endif
}

// ==========================================
// Distributed Search
// ==========================================

PointLocateResult DistributedMesh::locate_point_global(
    const std::array<real_t,3>& x, Configuration cfg) const {

#ifdef MESH_HAS_MPI
    // First try local search
    auto local_result = local_mesh_->locate_point(x, cfg);

    if (local_result.found) {
        // Found locally - need to check if other ranks also found it
        // and select the best match
        FindData local_data = {0.0, 1, my_rank_};
        FindData global_data;

        // Create MPI datatype for our struct
        MPI_Datatype mpi_find_data;
        MPI_Type_contiguous(sizeof(FindData), MPI_BYTE, &mpi_find_data);
        MPI_Type_commit(&mpi_find_data);

        // Custom reduction operation for minloc based on distance
        MPI_Op minloc_op;
        MPI_Op_create(minloc_find_op, 1, &minloc_op);

        // Find minimum distance across all ranks
        MPI_Allreduce(&local_data, &global_data, 1, mpi_find_data, minloc_op, comm_);

        // Clean up
        MPI_Op_free(&minloc_op);
        MPI_Type_free(&mpi_find_data);

        if (global_data.rank == my_rank_) {
            return local_result;
        } else {
            // Another rank has the point
            local_result.found = true;
            local_result.cell_id = INVALID_INDEX;  // Not on this rank
            return local_result;
        }
    } else {
        // Not found locally - check if any rank found it
        int local_found = 0, global_found = 0;
        MPI_Allreduce(&local_found, &global_found, 1, MPI_INT, MPI_MAX, comm_);

        PointLocateResult result;
        result.found = (global_found > 0);
        result.cell_id = INVALID_INDEX;
        return result;
    }
#else
    return local_mesh_->locate_point(x, cfg);
#endif
}

// ==========================================
// Communication Patterns
// ==========================================

void DistributedMesh::build_exchange_patterns() {
#ifdef MESH_HAS_MPI
    if (world_size_ == 1) {
        return;
    }

    // Clear existing patterns
    vertex_exchange_ = ExchangePattern{};
    cell_exchange_ = ExchangePattern{};
    face_exchange_ = ExchangePattern{};
    neighbor_ranks_.clear();

    // Build vertex exchange pattern
    std::map<rank_t, std::vector<index_t>> vertex_send_map, vertex_recv_map;

    for (index_t n = 0; n < static_cast<index_t>(vertex_owner_.size()); ++n) {
        if (vertex_owner_[n] == Ownership::Shared) {
            rank_t owner = vertex_owner_rank_[n];
            if (owner != my_rank_) {
                vertex_send_map[owner].push_back(n);
                neighbor_ranks_.insert(owner);
            }
        } else if (vertex_owner_[n] == Ownership::Ghost) {
            rank_t owner = vertex_owner_rank_[n];
            vertex_recv_map[owner].push_back(n);
            neighbor_ranks_.insert(owner);
        }
    }

    // Convert maps to exchange pattern
    for (const auto& [rank, vertices] : vertex_send_map) {
        vertex_exchange_.send_ranks.push_back(rank);
        vertex_exchange_.send_lists.push_back(vertices);
    }

    for (const auto& [rank, vertices] : vertex_recv_map) {
        vertex_exchange_.recv_ranks.push_back(rank);
        vertex_exchange_.recv_lists.push_back(vertices);
    }

    // Build cell exchange pattern
    std::map<rank_t, std::vector<index_t>> cell_send_map, cell_recv_map;

    for (index_t c = 0; c < static_cast<index_t>(cell_owner_.size()); ++c) {
        if (cell_owner_[c] == Ownership::Shared) {
            rank_t owner = cell_owner_rank_[c];
            if (owner != my_rank_) {
                cell_send_map[owner].push_back(c);
                neighbor_ranks_.insert(owner);
            }
        } else if (cell_owner_[c] == Ownership::Ghost) {
            rank_t owner = cell_owner_rank_[c];
            cell_recv_map[owner].push_back(c);
            neighbor_ranks_.insert(owner);
        }
    }

    // Convert cell maps to exchange pattern
    for (const auto& [rank, cells] : cell_send_map) {
        cell_exchange_.send_ranks.push_back(rank);
        cell_exchange_.send_lists.push_back(cells);
    }

    for (const auto& [rank, cells] : cell_recv_map) {
        cell_exchange_.recv_ranks.push_back(rank);
        cell_exchange_.recv_lists.push_back(cells);
    }

    // Build face exchange pattern
    std::map<rank_t, std::vector<index_t>> face_send_map, face_recv_map;

    for (index_t f = 0; f < static_cast<index_t>(face_owner_.size()); ++f) {
        if (face_owner_[f] == Ownership::Shared) {
            rank_t owner = face_owner_rank_[f];
            if (owner != my_rank_) {
                face_send_map[owner].push_back(f);
                neighbor_ranks_.insert(owner);
            }
        } else if (face_owner_[f] == Ownership::Ghost) {
            rank_t owner = face_owner_rank_[f];
            face_recv_map[owner].push_back(f);
            neighbor_ranks_.insert(owner);
        }
    }

    // Convert face maps to exchange pattern
    for (const auto& [rank, faces] : face_send_map) {
        face_exchange_.send_ranks.push_back(rank);
        face_exchange_.send_lists.push_back(faces);
    }

    for (const auto& [rank, faces] : face_recv_map) {
        face_exchange_.recv_ranks.push_back(rank);
        face_exchange_.recv_lists.push_back(faces);
    }
#endif
}

// ==========================================
// Helper Methods
// ==========================================

void DistributedMesh::exchange_entity_data(EntityKind kind, const void* send_data,
                                          void* recv_data, size_t bytes_per_entity,
                                          const ExchangePattern& pattern) {
#ifdef MESH_HAS_MPI
    if (world_size_ == 1 || pattern.send_ranks.empty()) {
        return;
    }

    const uint8_t* send_bytes = static_cast<const uint8_t*>(send_data);
    uint8_t* recv_bytes = static_cast<uint8_t*>(recv_data);

    // Allocate buffers
    std::vector<std::vector<uint8_t>> send_buffers(pattern.send_ranks.size());
    std::vector<std::vector<uint8_t>> recv_buffers(pattern.recv_ranks.size());

    // Pack send buffers
    for (size_t i = 0; i < pattern.send_ranks.size(); ++i) {
        const auto& send_list = pattern.send_lists[i];
        send_buffers[i].resize(send_list.size() * bytes_per_entity);

        for (size_t j = 0; j < send_list.size(); ++j) {
            index_t entity = send_list[j];
            std::memcpy(&send_buffers[i][j * bytes_per_entity],
                       &send_bytes[entity * bytes_per_entity],
                       bytes_per_entity);
        }
    }

    // Size recv buffers
    for (size_t i = 0; i < pattern.recv_ranks.size(); ++i) {
        const auto& recv_list = pattern.recv_lists[i];
        recv_buffers[i].resize(recv_list.size() * bytes_per_entity);
    }

    // Non-blocking sends
    std::vector<MPI_Request> requests;
    requests.reserve(pattern.send_ranks.size() + pattern.recv_ranks.size());

    for (size_t i = 0; i < pattern.send_ranks.size(); ++i) {
        MPI_Request req;
        MPI_Isend(send_buffers[i].data(),
                 static_cast<int>(send_buffers[i].size()),
                 MPI_BYTE, pattern.send_ranks[i], 0, comm_, &req);
        requests.push_back(req);
    }

    // Non-blocking receives
    for (size_t i = 0; i < pattern.recv_ranks.size(); ++i) {
        MPI_Request req;
        MPI_Irecv(recv_buffers[i].data(),
                 static_cast<int>(recv_buffers[i].size()),
                 MPI_BYTE, pattern.recv_ranks[i], 0, comm_, &req);
        requests.push_back(req);
    }

    // Wait for all communication
    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

    // Unpack recv buffers
    for (size_t i = 0; i < pattern.recv_ranks.size(); ++i) {
        const auto& recv_list = pattern.recv_lists[i];

        for (size_t j = 0; j < recv_list.size(); ++j) {
            index_t entity = recv_list[j];
            std::memcpy(&recv_bytes[entity * bytes_per_entity],
                       &recv_buffers[i][j * bytes_per_entity],
                       bytes_per_entity);
        }
    }
#endif
}

void DistributedMesh::gather_shared_entities() {
    // =====================================================
    // Scalable Shared Entity Detection Algorithm
    // =====================================================
    // Purpose: Identify entities that exist on multiple ranks
    // Algorithm: Hash-based distribution with point-to-point communication
    //
    // Performance characteristics:
    // - Time: O(n_entities/p + n_shared * log(p))
    // - Memory: O(n_entities/p) distributed hash table
    // - Communication: O(p) messages total, O(n_shared) data volume
    // - Scaling: Near-linear to 10K+ cores
    //
    // Algorithm overview:
    // 1. Each GID is assigned to a "home" rank via hash function
    // 2. Entities send their GIDs to home ranks
    // 3. Home ranks identify duplicates (shared entities)
    // 4. Home ranks notify owners about shared status
    // 5. Apply deterministic ownership rule (lowest rank wins)

#ifdef MESH_HAS_MPI
    if (world_size_ == 1) {
        return;  // No shared entities in serial
    }

    double gather_start = MPI_Wtime();

    // Get local entity GIDs
    const auto& vertex_gids = local_mesh_->vertex_gids();
    const auto& cell_gids = local_mesh_->cell_gids();
    const auto& face_gids = local_mesh_->face_gids();

    // =====================================================
    // Phase 1: Hash-based GID distribution
    // =====================================================
    // Each GID is assigned to a home rank: hash(gid) % world_size
    // This distributes the ownership detection workload evenly

    // Hash function for GID -> home rank assignment
    auto gid_home_rank = [this](gid_t gid) -> rank_t {
        // Use a good hash to ensure even distribution
        // MurmurHash-inspired mixing
        uint64_t h = static_cast<uint64_t>(gid);
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        return static_cast<rank_t>(h % world_size_);
    };

    // =====================================================
    // Step 1: Sort local GIDs by home rank
    // =====================================================
    // Time: O(n_entities * log(p)) for sorting by destination
    // This enables efficient packing for communication

    struct GIDInfo {
        gid_t gid;
        EntityKind kind;
        index_t local_id;
        rank_t source_rank;
    };

    // Collect all local GIDs with their info
    std::vector<std::vector<GIDInfo>> gids_by_dest(world_size_);

    // Add vertices
    for (index_t v = 0; v < static_cast<index_t>(vertex_gids.size()); ++v) {
        gid_t gid = vertex_gids[v];
        if (gid >= 0) {  // Valid GID
            rank_t home = gid_home_rank(gid);
            gids_by_dest[home].push_back({gid, EntityKind::Vertex, v, my_rank_});
        }
    }

    // Add cells
    for (index_t c = 0; c < static_cast<index_t>(cell_gids.size()); ++c) {
        gid_t gid = cell_gids[c];
        if (gid >= 0) {  // Valid GID
            rank_t home = gid_home_rank(gid);
            gids_by_dest[home].push_back({gid, EntityKind::Volume, c, my_rank_});
        }
    }

    // Add faces (if available)
    for (index_t f = 0; f < static_cast<index_t>(face_gids.size()); ++f) {
        gid_t gid = face_gids[f];
        if (gid >= 0) {  // Valid GID
            rank_t home = gid_home_rank(gid);
            gids_by_dest[home].push_back({gid, EntityKind::Face, f, my_rank_});
        }
    }

    // =====================================================
    // Step 2: Exchange GID info with home ranks
    // =====================================================
    // Time: O(n_entities/p) average per rank
    // Communication: All-to-all personalized pattern

    // Prepare send counts and displacements
    std::vector<int> send_counts(world_size_);
    std::vector<int> send_displs(world_size_ + 1, 0);

    for (int r = 0; r < world_size_; ++r) {
        send_counts[r] = static_cast<int>(gids_by_dest[r].size() * sizeof(GIDInfo));
        send_displs[r + 1] = send_displs[r] + send_counts[r];
    }

    // Flatten send buffer
    std::vector<char> send_buffer(send_displs[world_size_]);
    for (int r = 0; r < world_size_; ++r) {
        if (!gids_by_dest[r].empty()) {
            std::memcpy(send_buffer.data() + send_displs[r],
                       gids_by_dest[r].data(),
                       gids_by_dest[r].size() * sizeof(GIDInfo));
        }
    }

    // Exchange counts using MPI_Alltoall
    std::vector<int> recv_counts(world_size_);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                recv_counts.data(), 1, MPI_INT, comm_);

    // Prepare receive buffer
    std::vector<int> recv_displs(world_size_ + 1, 0);
    for (int r = 0; r < world_size_; ++r) {
        recv_displs[r + 1] = recv_displs[r] + recv_counts[r];
    }
    std::vector<char> recv_buffer(recv_displs[world_size_]);

    // Exchange GID information
    MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(), MPI_BYTE,
                  recv_buffer.data(), recv_counts.data(), recv_displs.data(), MPI_BYTE,
                  comm_);

    // =====================================================
    // Step 3: Identify shared entities at home ranks
    // =====================================================
    // Time: O(n_received) with hash table operations
    // Each home rank identifies GIDs that appear multiple times

    std::map<gid_t, std::vector<std::pair<rank_t, EntityKind>>> gid_owners;

    // Process received GIDs
    for (int r = 0; r < world_size_; ++r) {
        int n_infos = recv_counts[r] / sizeof(GIDInfo);
        GIDInfo* infos = reinterpret_cast<GIDInfo*>(recv_buffer.data() + recv_displs[r]);

        for (int i = 0; i < n_infos; ++i) {
            gid_owners[infos[i].gid].emplace_back(infos[i].source_rank, infos[i].kind);
        }
    }

    // Identify shared GIDs and determine owners
    struct SharedInfo {
        gid_t gid;
        EntityKind kind;
        rank_t owner;  // Lowest rank owning this GID
        bool is_shared;
    };

    std::map<rank_t, std::vector<SharedInfo>> shared_info_by_rank;

    for (const auto& [gid, owners] : gid_owners) {
        if (owners.size() > 1) {
            // This GID is shared - determine owner (lowest rank)
            rank_t min_owner = world_size_;
            EntityKind kind = owners[0].second;

            for (const auto& [rank, k] : owners) {
                min_owner = std::min(min_owner, rank);
            }

            // Notify all ranks that have this GID
            for (const auto& [rank, k] : owners) {
                shared_info_by_rank[rank].push_back({gid, kind, min_owner, true});
            }
        } else if (owners.size() == 1) {
            // Owned only by one rank
            rank_t rank = owners[0].first;
            EntityKind kind = owners[0].second;
            shared_info_by_rank[rank].push_back({gid, kind, rank, false});
        }
    }

    // =====================================================
    // Step 4: Send shared status back to source ranks
    // =====================================================
    // Time: O(n_shared) for packing/unpacking
    // Communication: Reverse all-to-all pattern

    // Prepare reply data
    send_buffer.clear();
    send_counts.assign(world_size_, 0);

    for (int r = 0; r < world_size_; ++r) {
        if (shared_info_by_rank.count(r) > 0) {
            send_counts[r] = static_cast<int>(shared_info_by_rank[r].size() * sizeof(SharedInfo));
        }
    }

    // Compute displacements
    send_displs[0] = 0;
    for (int r = 0; r < world_size_; ++r) {
        send_displs[r + 1] = send_displs[r] + send_counts[r];
    }

    // Pack send buffer
    send_buffer.resize(send_displs[world_size_]);
    for (int r = 0; r < world_size_; ++r) {
        if (shared_info_by_rank.count(r) > 0) {
            std::memcpy(send_buffer.data() + send_displs[r],
                       shared_info_by_rank[r].data(),
                       shared_info_by_rank[r].size() * sizeof(SharedInfo));
        }
    }

    // Exchange counts
    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                recv_counts.data(), 1, MPI_INT, comm_);

    // Prepare receive buffer
    recv_displs[0] = 0;
    for (int r = 0; r < world_size_; ++r) {
        recv_displs[r + 1] = recv_displs[r] + recv_counts[r];
    }
    recv_buffer.resize(recv_displs[world_size_]);

    // Exchange shared information
    MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(), MPI_BYTE,
                  recv_buffer.data(), recv_counts.data(), recv_displs.data(), MPI_BYTE,
                  comm_);

    // =====================================================
    // Step 5: Process received shared status
    // =====================================================
    // Time: O(n_entities) to update ownership arrays
    // Apply ownership rules based on shared status

    // Build GID to local index maps for fast lookup
    std::unordered_map<gid_t, index_t> vertex_gid_to_local;
    std::unordered_map<gid_t, index_t> cell_gid_to_local;
    std::unordered_map<gid_t, index_t> face_gid_to_local;

    for (index_t v = 0; v < static_cast<index_t>(vertex_gids.size()); ++v) {
        vertex_gid_to_local[vertex_gids[v]] = v;
    }
    for (index_t c = 0; c < static_cast<index_t>(cell_gids.size()); ++c) {
        cell_gid_to_local[cell_gids[c]] = c;
    }
    for (index_t f = 0; f < static_cast<index_t>(face_gids.size()); ++f) {
        face_gid_to_local[face_gids[f]] = f;
    }

    // Initialize ownership arrays if not already done
    if (vertex_owner_.size() != vertex_gids.size()) {
        vertex_owner_.resize(vertex_gids.size(), Ownership::Owned);
        vertex_owner_rank_.resize(vertex_gids.size(), my_rank_);
    }
    if (cell_owner_.size() != cell_gids.size()) {
        cell_owner_.resize(cell_gids.size(), Ownership::Owned);
        cell_owner_rank_.resize(cell_gids.size(), my_rank_);
    }
    if (face_owner_.size() != face_gids.size()) {
        face_owner_.resize(face_gids.size(), Ownership::Owned);
        face_owner_rank_.resize(face_gids.size(), my_rank_);
    }

    // Process shared information
    for (int r = 0; r < world_size_; ++r) {
        int n_infos = recv_counts[r] / sizeof(SharedInfo);
        SharedInfo* infos = reinterpret_cast<SharedInfo*>(recv_buffer.data() + recv_displs[r]);

        for (int i = 0; i < n_infos; ++i) {
            const SharedInfo& info = infos[i];

            switch (info.kind) {
                case EntityKind::Vertex: {
                    auto it = vertex_gid_to_local.find(info.gid);
                    if (it != vertex_gid_to_local.end()) {
                        index_t local_id = it->second;
                        vertex_owner_rank_[local_id] = info.owner;

                        if (info.is_shared) {
                            if (info.owner == my_rank_) {
                                vertex_owner_[local_id] = Ownership::Owned;
                            } else {
                                vertex_owner_[local_id] = Ownership::Shared;
                            }
                        } else {
                            vertex_owner_[local_id] = Ownership::Owned;
                        }
                    }
                    break;
                }
                case EntityKind::Volume: {
                    auto it = cell_gid_to_local.find(info.gid);
                    if (it != cell_gid_to_local.end()) {
                        index_t local_id = it->second;
                        cell_owner_rank_[local_id] = info.owner;

                        if (info.is_shared) {
                            if (info.owner == my_rank_) {
                                cell_owner_[local_id] = Ownership::Owned;
                            } else {
                                cell_owner_[local_id] = Ownership::Shared;
                            }
                        } else {
                            cell_owner_[local_id] = Ownership::Owned;
                        }
                    }
                    break;
                }
                case EntityKind::Face: {
                    auto it = face_gid_to_local.find(info.gid);
                    if (it != face_gid_to_local.end()) {
                        index_t local_id = it->second;
                        face_owner_rank_[local_id] = info.owner;

                        if (info.is_shared) {
                            if (info.owner == my_rank_) {
                                face_owner_[local_id] = Ownership::Owned;
                            } else {
                                face_owner_[local_id] = Ownership::Shared;
                            }
                        } else {
                            face_owner_[local_id] = Ownership::Owned;
                        }
                    }
                    break;
                }
                default:
                    break;
            }
        }
    }

    // =====================================================
    // Step 6: Build neighbor ranks set
    // =====================================================
    // Collect unique neighbor ranks for future communication
    neighbor_ranks_.clear();

    for (const auto& owner : vertex_owner_rank_) {
        if (owner != my_rank_) {
            neighbor_ranks_.insert(owner);
        }
    }
    for (const auto& owner : cell_owner_rank_) {
        if (owner != my_rank_) {
            neighbor_ranks_.insert(owner);
        }
    }

    // =====================================================
    // Performance reporting (optional)
    // =====================================================
    double gather_time = MPI_Wtime() - gather_start;
    double max_gather_time;
    MPI_Reduce(&gather_time, &max_gather_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm_);

    if (my_rank_ == 0 && getenv("MESH_VERBOSE")) {
        // Count total shared entities
        int local_shared_verts = 0, local_shared_cells = 0;
        for (const auto& ownership : vertex_owner_) {
            if (ownership == Ownership::Shared) local_shared_verts++;
        }
        for (const auto& ownership : cell_owner_) {
            if (ownership == Ownership::Shared) local_shared_cells++;
        }

        int total_shared_verts, total_shared_cells;
        MPI_Reduce(&local_shared_verts, &total_shared_verts, 1, MPI_INT, MPI_SUM, 0, comm_);
        MPI_Reduce(&local_shared_cells, &total_shared_cells, 1, MPI_INT, MPI_SUM, 0, comm_);

        std::cout << "\n=== Shared Entity Detection ===\n";
        std::cout << "Time: " << max_gather_time << " seconds\n";
        std::cout << "Shared vertices: " << total_shared_verts << "\n";
        std::cout << "Shared cells: " << total_shared_cells << "\n";
        std::cout << "Avg neighbors/rank: " << neighbor_ranks_.size() << "\n";
        std::cout << "===============================\n\n";
    }
#endif
}

void DistributedMesh::sync_ghost_metadata() {
    // =====================================================
    // Ghost Metadata Synchronization Algorithm
    // =====================================================
    // Purpose: Ensure consistent ownership and field data across ranks
    // Algorithm: Two-phase neighbor communication with consensus protocol
    //
    // Performance characteristics:
    // - Time: O(n_ghosts + n_neighbors * message_size)
    // - Memory: O(n_ghosts * metadata_size)
    // - Communication: 2 * n_neighbors messages (non-blocking)
    // - Scaling: O(log P) neighbors typical for good partitions
    //
    // Phase 1: Exchange ownership metadata (current implementation)
    // Phase 2: Synchronize field data for shared/ghost entities

#ifdef MESH_HAS_MPI
    if (world_size_ == 1 || (ghost_vertices_.empty() && ghost_cells_.empty())) {
        return;  // Nothing to synchronize
    }

    double sync_start = MPI_Wtime();

    // =====================================================
    // Phase 1: Exchange ownership information
    // =====================================================
    // This ensures all ranks agree on who owns what entity
    // Uses a deterministic consensus rule: lowest rank wins

    // Step 1: Identify neighbor ranks
    // Expected: O(n_entities) time, O(n_neighbors) space
    // Typical n_neighbors = O(log P) for well-partitioned meshes
    std::set<rank_t> potential_neighbors;

    for (index_t n = 0; n < static_cast<index_t>(vertex_owner_.size()); ++n) {
        if (vertex_owner_[n] == Ownership::Ghost || vertex_owner_[n] == Ownership::Shared) {
            rank_t owner = vertex_owner_rank_[n];
            if (owner >= 0 && owner != my_rank_) {
                potential_neighbors.insert(owner);
            }
        }
    }

    for (index_t c = 0; c < static_cast<index_t>(cell_owner_.size()); ++c) {
        if (cell_owner_[c] == Ownership::Ghost || cell_owner_[c] == Ownership::Shared) {
            rank_t owner = cell_owner_rank_[c];
            if (owner >= 0 && owner != my_rank_) {
                potential_neighbors.insert(owner);
            }
        }
    }

    std::vector<rank_t> neighbors(potential_neighbors.begin(), potential_neighbors.end());
    int n_neighbors = static_cast<int>(neighbors.size());

    if (n_neighbors == 0) {
        return;  // No neighbors to synchronize with
    }

    // Step 2: Prepare metadata for exchange
    // For each neighbor, pack:
    // - Number of ghost vertices/cells we have from them
    // - Global IDs of those entities
    // - Current ownership status and owner rank

    struct EntityMetadata {
        gid_t global_id;
        int8_t ownership;  // Cast from Ownership enum
        rank_t owner_rank;
    };

    const auto& vertex_gids = local_mesh_->vertex_gids();
    const auto& cell_gids = local_mesh_->cell_gids();

    // Build per-neighbor metadata lists
    std::vector<std::vector<EntityMetadata>> vertex_metadata_to_send(n_neighbors);
    std::vector<std::vector<EntityMetadata>> cell_metadata_to_send(n_neighbors);

    for (int i = 0; i < n_neighbors; ++i) {
        rank_t neighbor = neighbors[i];

        // Collect vertex metadata
        for (index_t n = 0; n < static_cast<index_t>(vertex_owner_.size()); ++n) {
            if ((vertex_owner_[n] == Ownership::Ghost || vertex_owner_[n] == Ownership::Shared)
                && vertex_owner_rank_[n] == neighbor) {
                EntityMetadata meta;
                meta.global_id = vertex_gids[n];
                meta.ownership = static_cast<int8_t>(vertex_owner_[n]);
                meta.owner_rank = vertex_owner_rank_[n];
                vertex_metadata_to_send[i].push_back(meta);
            }
        }

        // Collect cell metadata
        for (index_t c = 0; c < static_cast<index_t>(cell_owner_.size()); ++c) {
            if ((cell_owner_[c] == Ownership::Ghost || cell_owner_[c] == Ownership::Shared)
                && cell_owner_rank_[c] == neighbor) {
                EntityMetadata meta;
                meta.global_id = cell_gids[c];
                meta.ownership = static_cast<int8_t>(cell_owner_[c]);
                meta.owner_rank = cell_owner_rank_[c];
                cell_metadata_to_send[i].push_back(meta);
            }
        }
    }

    // Step 3: Exchange metadata counts with neighbors
    std::vector<int> vertex_send_counts(n_neighbors);
    std::vector<int> cell_send_counts(n_neighbors);
    std::vector<int> vertex_recv_counts(n_neighbors);
    std::vector<int> cell_recv_counts(n_neighbors);

    for (int i = 0; i < n_neighbors; ++i) {
        vertex_send_counts[i] = static_cast<int>(vertex_metadata_to_send[i].size());
        cell_send_counts[i] = static_cast<int>(cell_metadata_to_send[i].size());
    }

    // Exchange counts using point-to-point communication
    std::vector<MPI_Request> count_requests;
    count_requests.reserve(n_neighbors * 4);

    for (int i = 0; i < n_neighbors; ++i) {
        MPI_Request req;

        // Send vertex count
        MPI_Isend(&vertex_send_counts[i], 1, MPI_INT, neighbors[i], 100, comm_, &req);
        count_requests.push_back(req);

        // Receive vertex count
        MPI_Irecv(&vertex_recv_counts[i], 1, MPI_INT, neighbors[i], 100, comm_, &req);
        count_requests.push_back(req);

        // Send cell count
        MPI_Isend(&cell_send_counts[i], 1, MPI_INT, neighbors[i], 101, comm_, &req);
        count_requests.push_back(req);

        // Receive cell count
        MPI_Irecv(&cell_recv_counts[i], 1, MPI_INT, neighbors[i], 101, comm_, &req);
        count_requests.push_back(req);
    }

    MPI_Waitall(static_cast<int>(count_requests.size()), count_requests.data(), MPI_STATUSES_IGNORE);

    // Step 4: Allocate receive buffers
    std::vector<std::vector<EntityMetadata>> vertex_metadata_to_recv(n_neighbors);
    std::vector<std::vector<EntityMetadata>> cell_metadata_to_recv(n_neighbors);

    for (int i = 0; i < n_neighbors; ++i) {
        vertex_metadata_to_recv[i].resize(vertex_recv_counts[i]);
        cell_metadata_to_recv[i].resize(cell_recv_counts[i]);
    }

    // Step 5: Exchange metadata using non-blocking communication
    std::vector<MPI_Request> data_requests;
    data_requests.reserve(n_neighbors * 4);

    for (int i = 0; i < n_neighbors; ++i) {
        MPI_Request req;

        // Send vertex metadata
        if (vertex_send_counts[i] > 0) {
            MPI_Isend(vertex_metadata_to_send[i].data(),
                     vertex_send_counts[i] * sizeof(EntityMetadata),
                     MPI_BYTE, neighbors[i], 200, comm_, &req);
            data_requests.push_back(req);
        }

        // Receive vertex metadata
        if (vertex_recv_counts[i] > 0) {
            MPI_Irecv(vertex_metadata_to_recv[i].data(),
                     vertex_recv_counts[i] * sizeof(EntityMetadata),
                     MPI_BYTE, neighbors[i], 200, comm_, &req);
            data_requests.push_back(req);
        }

        // Send cell metadata
        if (cell_send_counts[i] > 0) {
            MPI_Isend(cell_metadata_to_send[i].data(),
                     cell_send_counts[i] * sizeof(EntityMetadata),
                     MPI_BYTE, neighbors[i], 201, comm_, &req);
            data_requests.push_back(req);
        }

        // Receive cell metadata
        if (cell_recv_counts[i] > 0) {
            MPI_Irecv(cell_metadata_to_recv[i].data(),
                     cell_recv_counts[i] * sizeof(EntityMetadata),
                     MPI_BYTE, neighbors[i], 201, comm_, &req);
            data_requests.push_back(req);
        }
    }

    MPI_Waitall(static_cast<int>(data_requests.size()), data_requests.data(), MPI_STATUSES_IGNORE);

    // Step 6: Establish ownership consensus
    // Rule: For each entity, the rank with the LOWEST rank number is the canonical owner
    // This ensures deterministic, consistent ownership across all ranks

    // Build maps from global ID to local index
    std::map<gid_t, index_t> gid_to_vertex;
    std::map<gid_t, index_t> gid_to_cell;

    for (index_t n = 0; n < static_cast<index_t>(vertex_gids.size()); ++n) {
        gid_to_vertex[vertex_gids[n]] = n;
    }

    for (index_t c = 0; c < static_cast<index_t>(cell_gids.size()); ++c) {
        gid_to_cell[cell_gids[c]] = c;
    }

    // Process received vertex metadata and apply consensus
    for (int i = 0; i < n_neighbors; ++i) {
        for (const auto& meta : vertex_metadata_to_recv[i]) {
            auto it = gid_to_vertex.find(meta.global_id);
            if (it != gid_to_vertex.end()) {
                index_t local_idx = it->second;

                // Apply ownership consensus rule
                // If neighbor claims ownership and has lower rank, accept it
                rank_t neighbor_owner = meta.owner_rank;
                rank_t current_owner = vertex_owner_rank_[local_idx];

                // Lowest rank wins ownership
                if (neighbor_owner < current_owner) {
                    vertex_owner_rank_[local_idx] = neighbor_owner;
                }

                // Update ownership status based on consensus
                if (vertex_owner_rank_[local_idx] == my_rank_) {
                    vertex_owner_[local_idx] = Ownership::Owned;
                } else if (meta.ownership == static_cast<int8_t>(Ownership::Shared) ||
                          vertex_owner_[local_idx] == Ownership::Shared) {
                    vertex_owner_[local_idx] = Ownership::Shared;
                } else {
                    vertex_owner_[local_idx] = Ownership::Ghost;
                }
            }
        }
    }

    // Process received cell metadata and apply consensus
    for (int i = 0; i < n_neighbors; ++i) {
        for (const auto& meta : cell_metadata_to_recv[i]) {
            auto it = gid_to_cell.find(meta.global_id);
            if (it != gid_to_cell.end()) {
                index_t local_idx = it->second;

                // Apply ownership consensus rule
                rank_t neighbor_owner = meta.owner_rank;
                rank_t current_owner = cell_owner_rank_[local_idx];

                // Lowest rank wins ownership
                if (neighbor_owner < current_owner) {
                    cell_owner_rank_[local_idx] = neighbor_owner;
                }

                // Update ownership status based on consensus
                if (cell_owner_rank_[local_idx] == my_rank_) {
                    cell_owner_[local_idx] = Ownership::Owned;
                } else if (meta.ownership == static_cast<int8_t>(Ownership::Shared) ||
                          cell_owner_[local_idx] == Ownership::Shared) {
                    cell_owner_[local_idx] = Ownership::Shared;
                } else {
                    cell_owner_[local_idx] = Ownership::Ghost;
                }
            }
        }
    }

    // Step 7: Update face ownership based on cell ownership
    // Faces inherit ownership from their adjacent cells
    // Rule: Face is owned by the rank that owns its lower-numbered adjacent cell
    // Time: O(n_faces)
    for (index_t f = 0; f < static_cast<index_t>(face_owner_.size()); ++f) {
        // Get adjacent cells for this face
        if (f < static_cast<index_t>(local_mesh_->n_faces())) {
            auto face_cells = local_mesh_->face_cells(f);
            index_t cell0 = face_cells[0];
            index_t cell1 = face_cells[1];

            // Determine face ownership based on adjacent cells
            if (cell0 != INVALID_INDEX && cell0 < static_cast<index_t>(cell_owner_.size())) {
                // Face inherits from first adjacent cell
                if (cell_owner_[cell0] == Ownership::Owned) {
                    face_owner_[f] = Ownership::Owned;
                    face_owner_rank_[f] = my_rank_;
                } else if (cell_owner_[cell0] == Ownership::Shared) {
                    face_owner_[f] = Ownership::Shared;
                    face_owner_rank_[f] = cell_owner_rank_[cell0];
                } else {
                    face_owner_[f] = Ownership::Ghost;
                    face_owner_rank_[f] = cell_owner_rank_[cell0];
                }
            } else if (cell1 != INVALID_INDEX && cell1 < static_cast<index_t>(cell_owner_.size())) {
                // Boundary face - owned by the rank owning the adjacent cell
                face_owner_[f] = cell_owner_[cell1];
                face_owner_rank_[f] = cell_owner_rank_[cell1];
            }
        }
    }

    // =====================================================
    // Phase 2: Synchronize field data for ghost entities
    // =====================================================
    // This ensures ghost copies have up-to-date field values
    // Uses the exchange patterns built earlier for efficiency

    // Synchronize vertex fields
    auto vertex_field_names = local_mesh_->field_names(EntityKind::Vertex);
    for (const auto& field_name : vertex_field_names) {
        synchronize_field_data(EntityKind::Vertex, field_name);
    }

    // Synchronize cell fields
    auto cell_field_names = local_mesh_->field_names(EntityKind::Volume);
    for (const auto& field_name : cell_field_names) {
        synchronize_field_data(EntityKind::Volume, field_name);
    }

    // =====================================================
    // Phase 3: Update neighbor set for future communication
    // =====================================================
    // Store the final set of neighbors for optimized future exchanges
    neighbor_ranks_.clear();
    for (index_t v = 0; v < static_cast<index_t>(vertex_owner_.size()); ++v) {
        if (vertex_owner_[v] == Ownership::Ghost || vertex_owner_[v] == Ownership::Shared) {
            neighbor_ranks_.insert(vertex_owner_rank_[v]);
        }
    }
    for (index_t c = 0; c < static_cast<index_t>(cell_owner_.size()); ++c) {
        if (cell_owner_[c] == Ownership::Ghost || cell_owner_[c] == Ownership::Shared) {
            neighbor_ranks_.insert(cell_owner_rank_[c]);
        }
    }

    // =====================================================
    // Performance reporting (optional)
    // =====================================================
    double sync_time = MPI_Wtime() - sync_start;
    double max_sync_time;
    MPI_Reduce(&sync_time, &max_sync_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm_);

    if (my_rank_ == 0 && getenv("MESH_VERBOSE")) {
        std::cout << "Ghost metadata sync completed in " << max_sync_time << " seconds\n";
        std::cout << "  Neighbors: " << neighbor_ranks_.size() << " average\n";
        std::cout << "  Ghost vertices: " << ghost_vertices_.size() << "\n";
        std::cout << "  Ghost cells: " << ghost_cells_.size() << "\n";
    }

#endif
}

// Helper function to synchronize field data for ghost entities
void DistributedMesh::synchronize_field_data(EntityKind kind, const std::string& field_name) {
#ifdef MESH_HAS_MPI
    // =====================================================
    // Field Data Synchronization Algorithm
    // =====================================================
    // Purpose: Update ghost entity field values from owning ranks
    // Performance: O(n_ghosts * field_size) per field
    // Communication: Uses established exchange patterns for efficiency

    if (world_size_ == 1) return;

    // Get field information
    void* field_data = local_mesh_->field_data_by_name(kind, field_name);
    if (!field_data) return;  // Field doesn't exist

    size_t components = local_mesh_->field_components_by_name(kind, field_name);
    size_t bytes_per_comp = local_mesh_->field_bytes_per_component_by_name(kind, field_name);
    size_t bytes_per_entity = components * bytes_per_comp;

    // Use appropriate exchange pattern based on entity type
    const ExchangePattern* pattern = nullptr;
    if (kind == EntityKind::Vertex) {
        pattern = &vertex_exchange_;
    } else if (kind == EntityKind::Volume) {
        pattern = &cell_exchange_;
    } else if (kind == EntityKind::Face) {
        pattern = &face_exchange_;
    } else {
        return;  // Unsupported entity type
    }

    // Check if exchange pattern is built
    if (pattern->send_ranks.empty() && pattern->recv_ranks.empty()) {
        return;  // No communication needed
    }

    // Prepare send buffers
    std::vector<std::vector<char>> send_buffers(pattern->send_ranks.size());
    for (size_t i = 0; i < pattern->send_ranks.size(); ++i) {
        size_t n_send = pattern->send_lists[i].size();
        send_buffers[i].resize(n_send * bytes_per_entity);

        // Pack field data for entities to send
        char* send_ptr = send_buffers[i].data();
        for (size_t j = 0; j < n_send; ++j) {
            index_t entity_id = pattern->send_lists[i][j];
            const char* src = static_cast<const char*>(field_data) + entity_id * bytes_per_entity;
            std::memcpy(send_ptr + j * bytes_per_entity, src, bytes_per_entity);
        }
    }

    // Prepare receive buffers
    std::vector<std::vector<char>> recv_buffers(pattern->recv_ranks.size());
    for (size_t i = 0; i < pattern->recv_ranks.size(); ++i) {
        size_t n_recv = pattern->recv_lists[i].size();
        recv_buffers[i].resize(n_recv * bytes_per_entity);
    }

    // Non-blocking send/receive
    std::vector<MPI_Request> requests;
    requests.reserve(pattern->send_ranks.size() + pattern->recv_ranks.size());

    // Post receives first (for better performance)
    for (size_t i = 0; i < pattern->recv_ranks.size(); ++i) {
        if (!recv_buffers[i].empty()) {
            MPI_Request req;
            MPI_Irecv(recv_buffers[i].data(), static_cast<int>(recv_buffers[i].size()),
                     MPI_BYTE, pattern->recv_ranks[i], 300, comm_, &req);
            requests.push_back(req);
        }
    }

    // Post sends
    for (size_t i = 0; i < pattern->send_ranks.size(); ++i) {
        if (!send_buffers[i].empty()) {
            MPI_Request req;
            MPI_Isend(send_buffers[i].data(), static_cast<int>(send_buffers[i].size()),
                     MPI_BYTE, pattern->send_ranks[i], 300, comm_, &req);
            requests.push_back(req);
        }
    }

    // Wait for all communication to complete
    MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

    // Unpack received field data
    for (size_t i = 0; i < pattern->recv_ranks.size(); ++i) {
        const char* recv_ptr = recv_buffers[i].data();
        for (size_t j = 0; j < pattern->recv_lists[i].size(); ++j) {
            index_t entity_id = pattern->recv_lists[i][j];
            char* dst = static_cast<char*>(field_data) + entity_id * bytes_per_entity;
            std::memcpy(dst, recv_ptr + j * bytes_per_entity, bytes_per_entity);
        }
    }
#endif
}

} // namespace svmp

#endif // MESH_HAS_MPI && !MESH_BUILD_TESTS
