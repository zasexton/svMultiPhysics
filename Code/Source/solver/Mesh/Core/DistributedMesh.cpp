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
#include <iostream>
#include <cstring>

#ifdef MESH_HAS_MPI
#include <mpi.h>
#endif

namespace svmp {

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
#ifdef MESH_HAS_MPI
    if (world_size_ == 1) {
        return;
    }

    for (const auto& field : fields) {
        // Get field data
        void* data = local_mesh_->field_data(field);
        if (!data) continue;

        size_t bytes_per_entity = local_mesh_->field_bytes_per_entity(field);

        // Exchange based on entity kind
        switch (field.kind) {
            case EntityKind::Vertex:
                exchange_entity_data(EntityKind::Vertex, data, data,
                                   bytes_per_entity, vertex_exchange_);
                break;
            case EntityKind::Volume:
                exchange_entity_data(EntityKind::Volume, data, data,
                                   bytes_per_entity, cell_exchange_);
                break;
            case EntityKind::Face:
                exchange_entity_data(EntityKind::Face, data, data,
                                   bytes_per_entity, face_exchange_);
                break;
            default:
                break;
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

        case PartitionHint::Vertices:
            // Balance by number of vertices
            // TODO: Implement vertex-based partitioning
            break;

        case PartitionHint::Memory:
            // Balance by memory usage
            // TODO: Implement memory-based partitioning
            break;

        case PartitionHint::Metis:
            // Use METIS graph partitioning
            // TODO: Implement METIS-based partitioning
            break;

        default:
            break;
    }

    // Step 3: Migrate cells to new owners
    migrate(new_owner_rank_per_cell);
#endif
}

// ==========================================
// Parallel I/O
// ==========================================

DistributedMesh DistributedMesh::load_parallel(const MeshIOOptions& opts, MPI_Comm comm) {
#ifdef MESH_HAS_MPI
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // For now, rank 0 loads and distributes
    // TODO: Implement true parallel I/O with MPI-IO or parallel HDF5

    std::shared_ptr<MeshBase> local_mesh;

    if (rank == 0) {
        // Load entire mesh on rank 0
        MeshBase global_mesh = MeshBase::load(opts);

        // Simple block partitioning
        size_t n_cells = global_mesh.n_cells();
        size_t cells_per_rank = n_cells / size;
        size_t extra = n_cells % size;

        // Create local mesh for rank 0
        size_t start = 0;
        size_t end = cells_per_rank + (0 < extra ? 1 : 0);

        // Extract submesh for rank 0
        // TODO: Implement submesh extraction based on cell range
        local_mesh = std::make_shared<MeshBase>(std::move(global_mesh));

        // Send submeshes to other ranks
        for (int r = 1; r < size; ++r) {
            start = end;
            end = start + cells_per_rank + (r < static_cast<int>(extra) ? 1 : 0);

            // Pack and send submesh data
            // TODO: Implement mesh serialization and MPI send
        }
    } else {
        // Receive submesh from rank 0
        // TODO: Implement mesh deserialization and MPI recv
        local_mesh = std::make_shared<MeshBase>();
    }

    // Create distributed mesh
    DistributedMesh dmesh(local_mesh, comm);

    // Build ghost layers
    dmesh.build_ghost_layer(1);

    return dmesh;
#else
    // Serial fallback
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

    // Parallel save options:
    // 1. Each rank saves its own file with rank suffix
    // 2. Use parallel HDF5
    // 3. Gather to rank 0 and save

    // Option 1: Each rank saves its own file
    MeshIOOptions local_opts = opts;
    local_opts.path = opts.path + "_rank" + std::to_string(my_rank_);
    local_mesh_->save(local_opts);

    // TODO: Implement options 2 and 3
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
        struct FindData {
            double distance;
            int found;
            int rank;
        } local_data = {0.0, 1, my_rank_},
          global_data;

        // Create MPI datatype for our struct
        MPI_Datatype mpi_find_data;
        MPI_Type_contiguous(sizeof(FindData), MPI_BYTE, &mpi_find_data);
        MPI_Type_commit(&mpi_find_data);

        // Custom reduction operation for minloc based on distance
        MPI_Op minloc_op;
        MPI_Op_create([](void* in, void* inout, int* len, MPI_Datatype*) {
            FindData* in_data = static_cast<FindData*>(in);
            FindData* inout_data = static_cast<FindData*>(inout);
            for (int i = 0; i < *len; ++i) {
                if (in_data[i].distance < inout_data[i].distance) {
                    inout_data[i] = in_data[i];
                }
            }
        }, 1, &minloc_op);

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

    // Similarly build cell and face exchange patterns
    // TODO: Implement cell and face exchange pattern building
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
#ifdef MESH_HAS_MPI
    if (world_size_ == 1) {
        return;
    }

    // Identify shared entities based on global IDs
    // This requires communication to determine which entities
    // exist on multiple ranks

    // Step 1: Gather global IDs of all local vertices
    const auto& vertex_gids_alias = local_mesh_->vertex_gids();

    // Step 2: Create map of GID to owning ranks
    std::map<gid_t, std::set<rank_t>> gid_to_ranks;

    // For simplicity, use all-to-all communication
    // In production, use more scalable neighbor-only communication
    std::vector<int> local_gids_count(1, static_cast<int>(vertex_gids_alias.size()));
    std::vector<int> all_gids_counts(world_size_);

    MPI_Allgather(local_gids_count.data(), 1, MPI_INT,
                 all_gids_counts.data(), 1, MPI_INT, comm_);

    // Gather all GIDs (simplified - not scalable for large meshes)
    // TODO: Implement scalable shared entity detection

    // Step 3: Mark shared entities
    for (index_t n = 0; n < static_cast<index_t>(vertex_gids_alias.size()); ++n) {
        gid_t gid = vertex_gids_alias[n];
        // If this GID exists on multiple ranks, mark as shared
        // TODO: Complete implementation
    }
#endif
}

void DistributedMesh::sync_ghost_metadata() {
#ifdef MESH_HAS_MPI
    if (world_size_ == 1 || (ghost_vertices_.empty() && ghost_cells_.empty())) {
        return;
    }

    // =====================================================
    // Exchange ownership information for ghost entities
    // This ensures all ranks agree on who owns what
    // =====================================================

    // Step 1: Identify neighbor ranks by examining ghost entity ownership
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
    for (index_t f = 0; f < static_cast<index_t>(face_owner_.size()); ++f) {
        // Find cells adjacent to this face
        // For now, use a simple heuristic: if any adjacent cell is owned, face is owned
        // This is a simplified implementation - production code would need proper face-to-cell adjacency

        // Default: inherit from current setting or mark as owned if uncertain
        if (face_owner_[f] == Ownership::Ghost) {
            // Keep ghost faces as ghost for now
            // TODO: Implement proper face ownership based on adjacent cells
        }
    }

    // Step 8: Final synchronization barrier to ensure all ranks have completed consensus
    MPI_Barrier(comm_);

#endif
}

} // namespace svmp
