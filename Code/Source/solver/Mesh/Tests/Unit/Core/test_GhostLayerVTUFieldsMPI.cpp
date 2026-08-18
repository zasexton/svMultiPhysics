/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_GhostLayerVTUFieldsMPI.cpp
 * @brief Regression for codim1-deferred halo construction and VTU metadata.
 *
 * The fixture starts with no materialized faces, matching MeshTranslator's
 * parallel-load options.  One face-neighbor ghost layer must be built and all
 * imported point/cell fields and metadata must agree with their owners.  The
 * four-rank form also records the intentional limitation that a face halo does
 * not complete every vertex star.
 */

#include "../../../Core/DistributedMesh.h"
#include "../../../Core/MeshBase.h"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace svmp::test {

#define ASSERT(cond)                                                                               \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__ << " : " #cond "\n";     \
      MPI_Abort(MPI_COMM_WORLD, 1);                                                                \
    }                                                                                              \
  } while (0)

#define ASSERT_EQ(a, b) ASSERT((a) == (b))
#define ASSERT_NEAR(a, b, tol) ASSERT(std::abs((a) - (b)) <= (tol))

static rank_t vertex_owner(gid_t gid) {
  if (gid >= 100) {
    return static_cast<rank_t>(gid - 100);
  }
  return 0;
}

static gid_t special_vertex_gid(rank_t rank) {
  return rank == 0 ? gid_t{3} : static_cast<gid_t>(100 + rank);
}

static std::shared_ptr<MeshBase> make_partition_without_faces(int rank) {
  auto mesh = std::make_shared<MeshBase>();
  std::vector<gid_t> vertex_gids;
  std::vector<real_t> coords;

  if (rank == 0) {
    vertex_gids = {0, 1, 2, 3};
    coords = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
  } else if (rank == 1) {
    vertex_gids = {0, 1, 2, 101};
    coords = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, -1};
  } else if (rank == 2) {
    vertex_gids = {0, 1, 3, 102};
    coords = {0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1};
  } else {
    vertex_gids = {0, 2, 3, 103};
    coords = {0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};
  }

  mesh->build_from_arrays(3,
                          coords,
                          std::vector<offset_t>{0, 4},
                          std::vector<index_t>{0, 1, 2, 3},
                          std::vector<CellShape>{{CellFamily::Tetra, 4, 1}});
  mesh->set_vertex_gids(std::move(vertex_gids));
  mesh->set_cell_gids(
      std::vector<gid_t>{static_cast<gid_t>(1000 + rank)});
  MeshFinalizeOptions options{};
  options.codim1_storage = MeshCodim1StorageMode::None;
  options.edge_storage = false;
  mesh->finalize(options);
  ASSERT_EQ(mesh->n_faces(), 0u);
  ASSERT_EQ(mesh->codim1_storage_mode(), MeshCodim1StorageMode::None);
  return mesh;
}

static void attach_vtu_state(MeshBase& mesh, int rank) {
  const auto region = static_cast<label_t>(300 + rank);
  mesh.register_label("region_" + std::to_string(rank), region);
  mesh.set_region_label(0, region);
  mesh.set_refinement_level(0, static_cast<size_t>(rank + 1));

  mesh.register_label("shared_vertex", 10);
  const auto shared = mesh.global_to_local_vertex(0);
  ASSERT(shared != INVALID_INDEX);
  mesh.set_vertex_label(shared, 10);

  const auto special_gid = special_vertex_gid(rank);
  const auto special = mesh.global_to_local_vertex(special_gid);
  ASSERT(special != INVALID_INDEX);
  const auto special_label = static_cast<label_t>(20 + rank);
  mesh.register_label("special_vertex_" + std::to_string(rank), special_label);
  mesh.set_vertex_label(special, special_label);

  mesh.add_to_set(EntityKind::Vertex, "tagged_vertices", shared);
  mesh.add_to_set(EntityKind::Vertex, "tagged_vertices", special);
  mesh.add_to_set(EntityKind::Volume, "material_cells", 0);

  const auto phi_handle =
      mesh.attach_field(EntityKind::Vertex, "phi", FieldScalarType::Float64, 1);
  const auto pressure_handle =
      mesh.attach_field(EntityKind::Vertex, "Pressure", FieldScalarType::Float64, 1);
  const auto node_id_handle =
      mesh.attach_field(EntityKind::Vertex, "GlobalNodeID", FieldScalarType::Int32, 1);
  const auto local_handle =
      mesh.attach_field(EntityKind::Vertex, "local_marker", FieldScalarType::Float64, 1);
  const auto accumulate_handle =
      mesh.attach_field(EntityKind::Vertex, "accumulate_marker", FieldScalarType::Float64, 1);

  auto pressure_descriptor = FieldDescriptor::scalar(EntityKind::Vertex);
  pressure_descriptor.ghost_policy = FieldGhostPolicy::Exchange;
  mesh.set_field_descriptor(pressure_handle, pressure_descriptor);
  auto local_descriptor = FieldDescriptor::scalar(EntityKind::Vertex);
  local_descriptor.ghost_policy = FieldGhostPolicy::None;
  mesh.set_field_descriptor(local_handle, local_descriptor);
  auto accumulate_descriptor = FieldDescriptor::scalar(EntityKind::Vertex);
  accumulate_descriptor.ghost_policy = FieldGhostPolicy::Accumulate;
  mesh.set_field_descriptor(accumulate_handle, accumulate_descriptor);

  auto* phi = mesh.field_data_as<real_t>(phi_handle);
  auto* pressure = mesh.field_data_as<real_t>(pressure_handle);
  auto* node_ids = mesh.field_data_as<std::int32_t>(node_id_handle);
  auto* local = mesh.field_data_as<real_t>(local_handle);
  auto* accumulate = mesh.field_data_as<real_t>(accumulate_handle);
  ASSERT(phi && pressure && node_ids && local && accumulate);
  for (index_t vertex = 0;
       vertex < static_cast<index_t>(mesh.n_vertices()); ++vertex) {
    const auto gid = mesh.vertex_gids()[static_cast<size_t>(vertex)];
    phi[vertex] = static_cast<real_t>(gid + 1000 * rank);
    pressure[vertex] = static_cast<real_t>(2 * gid + 100 * rank);
    node_ids[vertex] = static_cast<std::int32_t>(gid);
    local[vertex] = static_cast<real_t>(3 * gid + 1);
    accumulate[vertex] = static_cast<real_t>(4 * gid + 2);
  }

  const auto element_id_handle =
      mesh.attach_field(EntityKind::Volume,
                        "GlobalElementID",
                        FieldScalarType::Int64,
                        1);
  const auto material_handle =
      mesh.attach_field(EntityKind::Volume,
                        "cell_material",
                        FieldScalarType::Float64,
                        1);
  auto material_descriptor = FieldDescriptor::scalar(EntityKind::Volume);
  material_descriptor.ghost_policy = FieldGhostPolicy::None;
  mesh.set_field_descriptor(material_handle, material_descriptor);
  auto* element_ids = mesh.field_data_as<std::int64_t>(element_id_handle);
  auto* material = mesh.field_data_as<real_t>(material_handle);
  ASSERT(element_ids && material);
  element_ids[0] = static_cast<std::int64_t>(mesh.cell_gids()[0]);
  material[0] = static_cast<real_t>(mesh.cell_gids()[0]) + real_t{0.5};

  std::vector<real_t> current = mesh.X_ref();
  for (size_t vertex = 0; vertex < mesh.n_vertices(); ++vertex) {
    current[3 * vertex + 0] += real_t{0.01} * rank;
    current[3 * vertex + 1] += real_t{0.02} * rank;
    current[3 * vertex + 2] += real_t{0.03} * rank;
  }
  mesh.set_current_coords(current);
  mesh.use_current_configuration();
}

static bool set_contains(const MeshBase& mesh,
                         EntityKind kind,
                         const std::string& name,
                         index_t id) {
  const auto& members = mesh.get_set(kind, name);
  return std::find(members.begin(), members.end(), id) != members.end();
}

static void verify_halo(DistributedMesh& dmesh, int rank, int world) {
  const auto& mesh = dmesh.local_mesh();
  ASSERT(mesh.n_faces() > 0);
  ASSERT_EQ(mesh.codim1_storage_mode(), MeshCodim1StorageMode::Full);
  ASSERT_EQ(mesh.n_cells(), rank == 0 ? static_cast<size_t>(world) : 2u);
  ASSERT(mesh.has_current_coords());
  ASSERT_EQ(mesh.active_configuration(), Configuration::Current);

  const auto phi_handle = mesh.field_handle(EntityKind::Vertex, "phi");
  const auto pressure_handle = mesh.field_handle(EntityKind::Vertex, "Pressure");
  const auto node_id_handle = mesh.field_handle(EntityKind::Vertex, "GlobalNodeID");
  const auto local_handle = mesh.field_handle(EntityKind::Vertex, "local_marker");
  const auto accumulate_handle =
      mesh.field_handle(EntityKind::Vertex, "accumulate_marker");
  ASSERT(phi_handle.id && pressure_handle.id && node_id_handle.id &&
         local_handle.id && accumulate_handle.id);
  ASSERT(mesh.field_descriptor(phi_handle) == nullptr);
  ASSERT(mesh.field_descriptor(node_id_handle) == nullptr);
  ASSERT(mesh.field_descriptor(pressure_handle));
  ASSERT_EQ(mesh.field_descriptor(pressure_handle)->ghost_policy,
            FieldGhostPolicy::Exchange);
  ASSERT(mesh.field_descriptor(local_handle));
  ASSERT_EQ(mesh.field_descriptor(local_handle)->ghost_policy,
            FieldGhostPolicy::None);
  ASSERT(mesh.field_descriptor(accumulate_handle));
  ASSERT_EQ(mesh.field_descriptor(accumulate_handle)->ghost_policy,
            FieldGhostPolicy::Accumulate);

  const auto* phi = mesh.field_data_as<const real_t>(phi_handle);
  const auto* pressure = mesh.field_data_as<const real_t>(pressure_handle);
  const auto* node_ids = mesh.field_data_as<const std::int32_t>(node_id_handle);
  const auto* local = mesh.field_data_as<const real_t>(local_handle);
  const auto* accumulate = mesh.field_data_as<const real_t>(accumulate_handle);
  ASSERT(phi && pressure && node_ids && local && accumulate);
  for (index_t vertex = 0;
       vertex < static_cast<index_t>(mesh.n_vertices()); ++vertex) {
    const auto gid = mesh.vertex_gids()[static_cast<size_t>(vertex)];
    const auto owner = vertex_owner(gid);
    ASSERT_EQ(dmesh.owner_rank_vertex(vertex), owner);
    ASSERT_NEAR(phi[vertex], static_cast<real_t>(gid + 1000 * owner), 1e-12);
    ASSERT_NEAR(pressure[vertex], static_cast<real_t>(2 * gid + 100 * owner), 1e-12);
    ASSERT_EQ(node_ids[vertex], static_cast<std::int32_t>(gid));
    ASSERT_NEAR(local[vertex], static_cast<real_t>(3 * gid + 1), 1e-12);
    ASSERT_NEAR(accumulate[vertex], static_cast<real_t>(4 * gid + 2), 1e-12);

    const auto offset = static_cast<size_t>(vertex) * 3u;
    ASSERT_NEAR(mesh.X_cur()[offset + 0],
                mesh.X_ref()[offset + 0] + real_t{0.01} * owner,
                1e-12);
    ASSERT_NEAR(mesh.X_cur()[offset + 1],
                mesh.X_ref()[offset + 1] + real_t{0.02} * owner,
                1e-12);
    ASSERT_NEAR(mesh.X_cur()[offset + 2],
                mesh.X_ref()[offset + 2] + real_t{0.03} * owner,
                1e-12);
  }

  const auto shared = mesh.global_to_local_vertex(0);
  ASSERT(shared != INVALID_INDEX);
  ASSERT_EQ(mesh.vertex_label(shared), 10);
  ASSERT(set_contains(mesh, EntityKind::Vertex, "tagged_vertices", shared));

  const auto element_id_handle =
      mesh.field_handle(EntityKind::Volume, "GlobalElementID");
  const auto material_handle =
      mesh.field_handle(EntityKind::Volume, "cell_material");
  ASSERT(element_id_handle.id && material_handle.id);
  ASSERT(mesh.field_descriptor(element_id_handle) == nullptr);
  ASSERT(mesh.field_descriptor(material_handle));
  ASSERT_EQ(mesh.field_descriptor(material_handle)->ghost_policy,
            FieldGhostPolicy::None);
  const auto* element_ids =
      mesh.field_data_as<const std::int64_t>(element_id_handle);
  const auto* material = mesh.field_data_as<const real_t>(material_handle);
  ASSERT(element_ids && material);

  size_t ghost_cells = 0;
  for (index_t cell = 0;
       cell < static_cast<index_t>(mesh.n_cells()); ++cell) {
    const auto gid = mesh.cell_gids()[static_cast<size_t>(cell)];
    const auto owner = static_cast<rank_t>(gid - 1000);
    ASSERT_EQ(dmesh.owner_rank_cell(cell), owner);
    ASSERT_EQ(element_ids[cell], static_cast<std::int64_t>(gid));
    ASSERT_NEAR(material[cell], static_cast<real_t>(gid) + real_t{0.5}, 1e-12);
    ASSERT_EQ(mesh.region_label(cell), static_cast<label_t>(300 + owner));
    ASSERT_EQ(mesh.refinement_level(cell), static_cast<size_t>(owner + 1));
    ASSERT_EQ(mesh.label_from_name("region_" + std::to_string(owner)),
              static_cast<label_t>(300 + owner));
    ASSERT(set_contains(mesh, EntityKind::Volume, "material_cells", cell));
    if (owner != rank) {
      ASSERT(dmesh.is_ghost_cell(cell));
      ++ghost_cells;
    }

    const auto special_gid = special_vertex_gid(owner);
    const auto special = mesh.global_to_local_vertex(special_gid);
    ASSERT(special != INVALID_INDEX);
    ASSERT_EQ(mesh.vertex_label(special), static_cast<label_t>(20 + owner));
    ASSERT_EQ(mesh.label_from_name("special_vertex_" + std::to_string(owner)),
              static_cast<label_t>(20 + owner));
    ASSERT(set_contains(mesh, EntityKind::Vertex, "tagged_vertices", special));
  }
  ASSERT_EQ(ghost_cells, mesh.n_cells() - 1u);

  size_t incident_at_zero = 0;
  for (index_t cell = 0;
       cell < static_cast<index_t>(mesh.n_cells()); ++cell) {
    const auto [nodes, node_count] = mesh.cell_vertices_span(cell);
    bool contains_zero = false;
    for (size_t node = 0; node < node_count; ++node) {
      const auto vertex = nodes[node];
      contains_zero = contains_zero ||
                      mesh.vertex_gids()[static_cast<size_t>(vertex)] == 0;
    }
    incident_at_zero += contains_zero ? 1u : 0u;
  }
  if (world == 4) {
    ASSERT_EQ(incident_at_zero, rank == 0 ? 4u : 2u);
    if (rank != 0) {
      ASSERT(incident_at_zero < 4u); // one face layer is not a vertex-star halo
    }
  }
}

} // namespace svmp::test

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int world = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  if (world != 2 && world != 4) {
    if (rank == 0) {
      std::cout << "Skipping ghost VTU-field test (requires 2 or 4 ranks)\n";
    }
    MPI_Finalize();
    return 0;
  }

  auto local_mesh = svmp::test::make_partition_without_faces(rank);
  svmp::test::attach_vtu_state(*local_mesh, rank);
  svmp::DistributedMesh dmesh(local_mesh, MPI_COMM_WORLD);
  dmesh.build_ghost_layer(1);
  svmp::test::verify_halo(dmesh, rank, world);

  dmesh.clear_ghosts();
  ASSERT_EQ(dmesh.local_mesh().n_cells(), 1u);
  ASSERT(dmesh.local_mesh().has_field(svmp::EntityKind::Vertex, "phi"));
  ASSERT(dmesh.local_mesh().has_current_coords());
  ASSERT_EQ(dmesh.local_mesh().active_configuration(), svmp::Configuration::Current);

  if (rank == 0) {
    std::cout << "Ghost layer VTU field/metadata preservation test PASSED ("
              << world << " ranks)\n";
  }
  MPI_Finalize();
  return 0;
}
