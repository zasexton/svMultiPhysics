#include "Mesh.h"
#include "Search/CutCell.h"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace svmp::test {

#define ASSERT_MPI(cond)                                                                           \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__ << " : " #cond "\n";     \
      MPI_Abort(MPI_COMM_WORLD, 1);                                                                \
    }                                                                                              \
  } while (0)

static index_t strip_vertex_lid(int x_plane, int y, int z)
{
  return static_cast<index_t>(x_plane * 4 + (y + 2 * z));
}

static void build_hex_strip_global_arrays(int n_cells,
                                         std::vector<real_t>& coords,
                                         std::vector<offset_t>& offsets,
                                         std::vector<index_t>& conn,
                                         std::vector<CellShape>& shapes)
{
  const int n_planes = n_cells + 1;
  const int n_vertices = 4 * n_planes;
  coords.clear();
  coords.reserve(static_cast<size_t>(n_vertices) * 3u);
  for (int x_plane = 0; x_plane < n_planes; ++x_plane) {
    for (int z = 0; z <= 1; ++z) {
      for (int y = 0; y <= 1; ++y) {
        coords.push_back(static_cast<real_t>(x_plane));
        coords.push_back(static_cast<real_t>(y));
        coords.push_back(static_cast<real_t>(z));
      }
    }
  }

  offsets.assign(static_cast<size_t>(n_cells) + 1u, 0);
  conn.clear();
  conn.reserve(static_cast<size_t>(n_cells) * 8u);
  shapes.assign(static_cast<size_t>(n_cells), CellShape{CellFamily::Hex, 8, 1});
  offsets[0] = 0;
  for (int c = 0; c < n_cells; ++c) {
    const int x0 = c;
    const int x1 = c + 1;
    conn.push_back(strip_vertex_lid(x0, 0, 0));
    conn.push_back(strip_vertex_lid(x1, 0, 0));
    conn.push_back(strip_vertex_lid(x1, 1, 0));
    conn.push_back(strip_vertex_lid(x0, 1, 0));
    conn.push_back(strip_vertex_lid(x0, 0, 1));
    conn.push_back(strip_vertex_lid(x1, 0, 1));
    conn.push_back(strip_vertex_lid(x1, 1, 1));
    conn.push_back(strip_vertex_lid(x0, 1, 1));
    offsets[static_cast<size_t>(c) + 1u] = static_cast<offset_t>(conn.size());
  }
}

static Mesh build_partitioned_hex_strip_cells(int n_cells,
                                              const std::string& partition_method = "block")
{
  std::vector<real_t> coords;
  std::vector<offset_t> offsets;
  std::vector<index_t> conn;
  std::vector<CellShape> shapes;
  build_hex_strip_global_arrays(n_cells, coords, offsets, conn, shapes);

  std::unordered_map<std::string, std::string> options;
  options["partition_method"] = partition_method;
  if (partition_method == "parmetis") {
    options["parmetis_algorithm"] = "mesh";
  }

  Mesh mesh(MeshComm::world());
  mesh.build_from_arrays_global_and_partition(3,
                                             coords,
                                             offsets,
                                             conn,
                                             shapes,
                                             PartitionHint::Cells,
                                             /*ghost_layers=*/1,
                                             options);
  return mesh;
}

static Mesh build_partitioned_hex_strip(int world_size)
{
  return build_partitioned_hex_strip_cells(world_size);
}

static std::string cut_entity_identity_key(const search::CutDistributedEntityRecord& entity)
{
  return std::to_string(entity.stable_id) + ":" +
         std::to_string(static_cast<long long>(entity.parent_gid)) + ":" +
         std::to_string(static_cast<int>(entity.kind)) + ":" +
         std::to_string(static_cast<int>(entity.side));
}

static std::map<std::string, rank_t> exchanged_owner_map(
    const search::CutDistributedExchangePacket& packet)
{
  std::map<std::string, rank_t> out;
  for (const auto& entity : packet.entities) {
    if (entity.kind == search::CutTopologyEntityKind::SideRegion &&
        entity.measure <= 0.0) {
      continue;
    }
    out[cut_entity_identity_key(entity)] = entity.owner_rank;
  }
  return out;
}

static int local_owned_cell_count(const Mesh& mesh)
{
  int count = 0;
  for (index_t c = 0; c < static_cast<index_t>(mesh.local_mesh().n_cells()); ++c) {
    if (mesh.is_owned_cell(c)) {
      ++count;
    }
  }
  return count;
}

static std::set<rank_t> expected_sparse_cut_neighbors(const Mesh& mesh,
                                                      const search::CutDistributedExchangePacket& packet)
{
  std::set<rank_t> expected;
  const rank_t rank = mesh.rank();
  const int size = mesh.world_size();
  for (const auto neighbor : mesh.neighbor_ranks()) {
    if (neighbor >= 0 && neighbor < size && neighbor != rank) {
      expected.insert(neighbor);
    }
  }
  for (const auto& entity : packet.entities) {
    if (entity.owner_rank >= 0 &&
        entity.owner_rank < size &&
        entity.owner_rank != rank) {
      expected.insert(entity.owner_rank);
    }
  }
  return expected;
}

static void assert_sparse_neighbor_contract(const Mesh& mesh,
                                            const search::CutDistributedState& state)
{
  const auto expected = expected_sparse_cut_neighbors(mesh, state.local_packet);
  const std::set<rank_t> actual(state.communication_neighbors.begin(),
                                state.communication_neighbors.end());
  ASSERT_MPI(actual == expected);
  for (const auto rank : state.received_neighbor_ranks) {
    ASSERT_MPI(actual.find(rank) != actual.end());
  }
  for (const auto& entity : state.imported_records) {
    ASSERT_MPI(entity.owner_rank != mesh.rank());
    ASSERT_MPI(actual.find(entity.owner_rank) != actual.end());
  }
}

static void assert_positive_ghost_side_region_payloads(const search::CutDistributedState& state,
                                                       MPI_Comm comm)
{
  int local_ghost_payloads = 0;
  for (const auto& entity : state.ghost_records) {
    if (entity.kind == search::CutTopologyEntityKind::SideRegion &&
        entity.measure > 0.0) {
      ASSERT_MPI(entity.closed_topology);
      ASSERT_MPI(!entity.vertex_ids.empty());
      ASSERT_MPI(!entity.face_ids.empty());
      ++local_ghost_payloads;
    }
  }
  int global_ghost_payloads = 0;
  MPI_Allreduce(&local_ghost_payloads, &global_ghost_payloads, 1, MPI_INT, MPI_SUM, comm);
  ASSERT_MPI(global_ghost_payloads > 0);
}

static void assert_cut_state_valid_for_partition_method(Mesh& mesh,
                                                        const std::string& partition_method,
                                                        MPI_Comm comm)
{
  search::EmbeddedGeometryDescriptor embedded;
  embedded.kind = search::EmbeddedGeometryKind::Plane;
  embedded.origin = {{0.0, 0.5, 0.0}};
  embedded.normal = {{0.0, 1.0, 0.0}};
  embedded.geometry_epoch = 21;
  embedded.provenance.persistent_id = "partitioner-" + partition_method + "-all-cell-plane";

  search::CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;

  const auto map = search::classify_embedded_geometry(mesh, embedded, options);
  const auto topology = search::reconstruct_cut_topology(mesh.local_mesh(), map);
  const auto state = search::build_distributed_cut_state(mesh, map, topology);
  const auto diag = search::diagnose_distributed_cut_state(state);
  if (!diag.ok) {
    for (const auto& message : diag.messages) {
      std::cerr << "Partition method '" << partition_method
                << "' distributed cut-state diagnostic: " << message << "\n";
    }
  }
  ASSERT_MPI(diag.ok);
  ASSERT_MPI(state.valid_for(mesh, map, topology));
  ASSERT_MPI(state.neighbor_sparse_exchange);
  assert_sparse_neighbor_contract(mesh, state);

  int local_owned_cuts = 0;
  int local_cut_records = 0;
  for (const auto& record : map.cells) {
    if (record.classification == search::CutClassification::Cut) {
      ++local_cut_records;
      if (record.owner_rank == mesh.rank()) {
        ++local_owned_cuts;
      }
    }
  }
  int global_owned_cuts = 0;
  int global_cut_records = 0;
  MPI_Allreduce(&local_owned_cuts, &global_owned_cuts, 1, MPI_INT, MPI_SUM, comm);
  MPI_Allreduce(&local_cut_records, &global_cut_records, 1, MPI_INT, MPI_SUM, comm);
  ASSERT_MPI(global_owned_cuts > 0);
  ASSERT_MPI(global_cut_records >= global_owned_cuts);

  int local_neighbor_count = static_cast<int>(state.communication_neighbors.size());
  int global_neighbor_count = 0;
  MPI_Allreduce(&local_neighbor_count, &global_neighbor_count, 1, MPI_INT, MPI_SUM, comm);
  ASSERT_MPI(global_neighbor_count > 0);

  int local_exchanged_entities = static_cast<int>(state.exchanged_packet.entities.size());
  int global_exchanged_entities = 0;
  MPI_Allreduce(&local_exchanged_entities, &global_exchanged_entities, 1, MPI_INT, MPI_SUM, comm);
  ASSERT_MPI(global_exchanged_entities > 0);

  assert_positive_ghost_side_region_payloads(state, comm);

  const auto repeat_state = search::build_distributed_cut_state(mesh, map, topology);
  ASSERT_MPI(repeat_state.neighbor_sparse_exchange == state.neighbor_sparse_exchange);
  ASSERT_MPI(repeat_state.communication_neighbors == state.communication_neighbors);
  ASSERT_MPI(repeat_state.exchanged_packet.revision_key == state.exchanged_packet.revision_key);
  ASSERT_MPI(repeat_state.exchanged_packet.entities.size() == state.exchanged_packet.entities.size());

  std::vector<rank_t> rotated_owner(mesh.local_mesh().n_cells(), mesh.rank());
  for (index_t c = 0; c < static_cast<index_t>(mesh.local_mesh().n_cells()); ++c) {
    rotated_owner[static_cast<std::size_t>(c)] = mesh.owner_rank_cell(c);
    if (mesh.is_owned_cell(c)) {
      rotated_owner[static_cast<std::size_t>(c)] =
          static_cast<rank_t>((mesh.rank() + 1) % mesh.world_size());
    }
  }
  mesh.migrate(rotated_owner);
  ASSERT_MPI(!state.valid_for(mesh, map, topology));

  const auto migrated_map = search::classify_embedded_geometry(mesh, embedded, options);
  const auto migrated_topology = search::reconstruct_cut_topology(mesh.local_mesh(), migrated_map);
  const auto migrated_state =
      search::build_distributed_cut_state(mesh, migrated_map, migrated_topology);
  const auto migrated_diag = search::diagnose_distributed_cut_state(migrated_state);
  if (!migrated_diag.ok) {
    for (const auto& message : migrated_diag.messages) {
      std::cerr << "Partition method '" << partition_method
                << "' migrated cut-state diagnostic: " << message << "\n";
    }
  }
  ASSERT_MPI(migrated_diag.ok);
  ASSERT_MPI(migrated_state.valid_for(mesh, migrated_map, migrated_topology));
  ASSERT_MPI(migrated_state.neighbor_sparse_exchange);
  assert_sparse_neighbor_contract(mesh, migrated_state);
}

static void run_cut_cell_partition_method_qualification(MPI_Comm comm,
                                                        int rank,
                                                        int size,
                                                        const std::string& partition_method)
{
  const int n_cells = std::max(12, 4 * size);
  auto mesh = build_partitioned_hex_strip_cells(n_cells, partition_method);
  assert_cut_state_valid_for_partition_method(mesh, partition_method, comm);

  if (rank == 0) {
    std::cout << "CutCell sparse exchange partition_method=" << partition_method
              << " qualified\n";
  }
}

static void run_cut_cell_mpi_test(MPI_Comm comm, int rank, int size)
{
  auto mesh = build_partitioned_hex_strip(size);
  search::EmbeddedGeometryDescriptor embedded;
  embedded.kind = search::EmbeddedGeometryKind::Plane;
  embedded.origin = {{1.0, 0.0, 0.0}};
  embedded.normal = {{1.0, 0.0, 0.0}};
  embedded.geometry_epoch = 4;
  embedded.provenance.persistent_id = "partition-crossing-plane";

  search::CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;

  const auto map = search::classify_embedded_geometry(mesh, embedded, options);
  const auto topology = search::reconstruct_cut_topology(mesh.local_mesh(), map);
  const auto packet = search::make_distributed_cut_exchange_packet(map, topology);
  const auto state = search::build_distributed_cut_state(mesh, map, topology);
  const auto state_diag = search::diagnose_distributed_cut_state(state);
  if (!state_diag.ok) {
    for (const auto& message : state_diag.messages) {
      std::cerr << "Distributed cut-state diagnostic: " << message << "\n";
    }
  }
  ASSERT_MPI(state_diag.ok);
  ASSERT_MPI(state.valid_for(mesh, map, topology));
  ASSERT_MPI(state.neighbor_sparse_exchange);
  ASSERT_MPI(static_cast<int>(state.communication_neighbors.size()) < size);

  const int local_exchanged_entities = static_cast<int>(state.exchanged_packet.entities.size());
  int global_exchanged_entities = 0;
  MPI_Allreduce(&local_exchanged_entities, &global_exchanged_entities, 1, MPI_INT, MPI_SUM, comm);
  ASSERT_MPI(global_exchanged_entities > 0);

  const int local_sparse_nonparticipant =
      (state.local_packet.entities.empty() &&
       state.exchanged_packet.entities.empty() &&
       state.received_neighbor_ranks.empty()) ? 1 : 0;
  int global_sparse_nonparticipants = 0;
  MPI_Allreduce(&local_sparse_nonparticipant,
                &global_sparse_nonparticipants,
                1,
                MPI_INT,
                MPI_SUM,
                comm);
  if (size >= 4) {
    ASSERT_MPI(global_sparse_nonparticipants > 0);
  }

  int local_owned_cut = 0;
  for (const auto& record : map.cells) {
    if (record.classification == search::CutClassification::Cut &&
        record.owner_rank == rank) {
      ++local_owned_cut;
    }
    ASSERT_MPI(record.owner_rank >= 0);
    ASSERT_MPI(record.owner_rank < size);
  }
  for (const auto& entity : packet.entities) {
    ASSERT_MPI(entity.owner_rank >= 0);
    ASSERT_MPI(entity.owner_rank < size);
    ASSERT_MPI(entity.stable_id != 0);
    if (entity.kind == search::CutTopologyEntityKind::SideRegion &&
        entity.measure > 0.0) {
      ASSERT_MPI(entity.parent_measure > 0.0);
      ASSERT_MPI(entity.volume_fraction > 0.0);
      ASSERT_MPI(entity.volume_fraction <= 1.0 + 1.0e-12);
      ASSERT_MPI(entity.closed_topology);
      ASSERT_MPI(!entity.vertex_ids.empty());
      ASSERT_MPI(!entity.face_ids.empty());
    }
  }

  int global_owned_cut = 0;
  MPI_Allreduce(&local_owned_cut, &global_owned_cut, 1, MPI_INT, MPI_SUM, comm);
  ASSERT_MPI(global_owned_cut >= 2);
  int local_packet_entities = static_cast<int>(packet.entities.size());
  int global_packet_entities = 0;
  MPI_Allreduce(&local_packet_entities, &global_packet_entities, 1, MPI_INT, MPI_SUM, comm);
  ASSERT_MPI(global_packet_entities > 0);
  int local_payload_regions = 0;
  for (const auto& entity : packet.entities) {
    if (entity.kind == search::CutTopologyEntityKind::SideRegion &&
        entity.measure > 0.0 &&
        entity.closed_topology &&
        !entity.vertex_ids.empty() &&
        !entity.face_ids.empty()) {
      ++local_payload_regions;
    }
  }
  int global_payload_regions = 0;
  MPI_Allreduce(&local_payload_regions, &global_payload_regions, 1, MPI_INT, MPI_SUM, comm);
  ASSERT_MPI(global_payload_regions > 0);

  int local_exchanged_side_payloads = 0;
  for (const auto& entity : state.exchanged_packet.entities) {
    if (entity.kind == search::CutTopologyEntityKind::SideRegion &&
        entity.measure > 0.0 &&
        entity.closed_topology &&
        !entity.vertex_ids.empty() &&
        !entity.face_ids.empty()) {
      ++local_exchanged_side_payloads;
    }
  }
  int global_exchanged_side_payloads = 0;
  MPI_Allreduce(&local_exchanged_side_payloads,
                &global_exchanged_side_payloads,
                1,
                MPI_INT,
                MPI_SUM,
                comm);
  ASSERT_MPI(global_exchanged_side_payloads > 0);

  for (const auto& entity : state.ghost_records) {
    ASSERT_MPI(entity.owner_rank != rank);
  }
  assert_positive_ghost_side_region_payloads(state, comm);

  const auto repeat_state = search::build_distributed_cut_state(mesh, map, topology);
  ASSERT_MPI(repeat_state.exchanged_packet.revision_key == state.exchanged_packet.revision_key);
  ASSERT_MPI(repeat_state.exchanged_packet.entities.size() == state.exchanged_packet.entities.size());

  const auto owners_before_migration = exchanged_owner_map(state.exchanged_packet);
  const int local_owner_rows_before = static_cast<int>(owners_before_migration.size());
  int global_owner_rows_before = 0;
  MPI_Allreduce(&local_owner_rows_before, &global_owner_rows_before, 1, MPI_INT, MPI_SUM, comm);
  ASSERT_MPI(global_owner_rows_before > 0);

  std::vector<rank_t> new_owner(mesh.local_mesh().n_cells(), static_cast<rank_t>(rank));
  for (index_t c = 0; c < static_cast<index_t>(mesh.local_mesh().n_cells()); ++c) {
    new_owner[static_cast<std::size_t>(c)] = mesh.owner_rank_cell(c);
    if (mesh.is_owned_cell(c)) {
      new_owner[static_cast<std::size_t>(c)] = static_cast<rank_t>((rank + 1) % size);
    }
  }

  mesh.migrate(new_owner);
  ASSERT_MPI(!state.valid_for(mesh, map, topology));

  const auto migrated_map = search::classify_embedded_geometry(mesh, embedded, options);
  const auto migrated_topology = search::reconstruct_cut_topology(mesh.local_mesh(), migrated_map);
  const auto migrated_state = search::build_distributed_cut_state(mesh, migrated_map, migrated_topology);
  const auto migrated_diag = search::diagnose_distributed_cut_state(migrated_state);
  if (!migrated_diag.ok) {
    for (const auto& message : migrated_diag.messages) {
      std::cerr << "Migrated distributed cut-state diagnostic: " << message << "\n";
    }
  }
  ASSERT_MPI(migrated_diag.ok);
  ASSERT_MPI(migrated_state.valid_for(mesh, migrated_map, migrated_topology));

  const auto owners_after_migration = exchanged_owner_map(migrated_state.exchanged_packet);
  int local_changed_owner = 0;
  int local_common_owner_rows = 0;
  for (const auto& [key, before_owner] : owners_before_migration) {
    const auto it = owners_after_migration.find(key);
    if (it == owners_after_migration.end()) {
      continue;
    }
    ++local_common_owner_rows;
    if (it->second != before_owner) {
      local_changed_owner = 1;
    }
  }
  int global_common_owner_rows = 0;
  MPI_Allreduce(&local_common_owner_rows, &global_common_owner_rows, 1, MPI_INT, MPI_SUM, comm);
  ASSERT_MPI(global_common_owner_rows > 0);
  int global_changed_owner = 0;
  MPI_Allreduce(&local_changed_owner, &global_changed_owner, 1, MPI_INT, MPI_MAX, comm);
  ASSERT_MPI(global_changed_owner == 1);

  assert_positive_ghost_side_region_payloads(migrated_state, comm);
}

static void run_cut_cell_rebalance_mpi_test(MPI_Comm comm, int rank, int size)
{
  const int n_cells = std::max(4, 2 * size);
  auto mesh = build_partitioned_hex_strip_cells(n_cells);

  std::vector<rank_t> all_to_zero(mesh.local_mesh().n_cells(), static_cast<rank_t>(rank));
  for (index_t c = 0; c < static_cast<index_t>(mesh.local_mesh().n_cells()); ++c) {
    all_to_zero[static_cast<std::size_t>(c)] = mesh.owner_rank_cell(c);
    if (mesh.is_owned_cell(c)) {
      all_to_zero[static_cast<std::size_t>(c)] = 0;
    }
  }

  mesh.migrate(all_to_zero);

  const int local_owned_before = local_owned_cell_count(mesh);
  int global_owned_before = 0;
  int max_owned_before = 0;
  MPI_Allreduce(&local_owned_before, &global_owned_before, 1, MPI_INT, MPI_SUM, comm);
  MPI_Allreduce(&local_owned_before, &max_owned_before, 1, MPI_INT, MPI_MAX, comm);
  ASSERT_MPI(global_owned_before == n_cells);
  ASSERT_MPI(max_owned_before == n_cells);

  search::EmbeddedGeometryDescriptor embedded;
  embedded.kind = search::EmbeddedGeometryKind::Plane;
  embedded.origin = {{static_cast<real_t>(n_cells / 2), 0.0, 0.0}};
  embedded.normal = {{1.0, 0.0, 0.0}};
  embedded.geometry_epoch = 7;
  embedded.provenance.persistent_id = "rebalance-crossing-plane";

  search::CutClassificationOptions options;
  options.classify_faces = false;
  options.classify_edges = false;

  const auto map = search::classify_embedded_geometry(mesh, embedded, options);
  const auto topology = search::reconstruct_cut_topology(mesh.local_mesh(), map);
  const auto state = search::build_distributed_cut_state(mesh, map, topology);
  const auto state_diag = search::diagnose_distributed_cut_state(state);
  if (!state_diag.ok) {
    for (const auto& message : state_diag.messages) {
      std::cerr << "Imbalanced distributed cut-state diagnostic: " << message << "\n";
    }
  }
  ASSERT_MPI(state_diag.ok);
  ASSERT_MPI(state.valid_for(mesh, map, topology));
  ASSERT_MPI(state.neighbor_sparse_exchange);
  const auto owners_before_rebalance = exchanged_owner_map(state.exchanged_packet);
  const int local_rebalance_owner_rows_before = static_cast<int>(owners_before_rebalance.size());
  int global_rebalance_owner_rows_before = 0;
  MPI_Allreduce(&local_rebalance_owner_rows_before,
                &global_rebalance_owner_rows_before,
                1,
                MPI_INT,
                MPI_SUM,
                comm);
  ASSERT_MPI(global_rebalance_owner_rows_before > 0);
  int local_max_owner_before = -1;
  for (const auto& [key, owner] : owners_before_rebalance) {
    (void)key;
    local_max_owner_before = std::max(local_max_owner_before, static_cast<int>(owner));
  }
  int global_max_owner_before = -1;
  MPI_Allreduce(&local_max_owner_before,
                &global_max_owner_before,
                1,
                MPI_INT,
                MPI_MAX,
                comm);

  mesh.rebalance(PartitionHint::Cells);
  ASSERT_MPI(!state.valid_for(mesh, map, topology));

  const int local_owned_after = local_owned_cell_count(mesh);
  int global_owned_after = 0;
  int max_owned_after = 0;
  MPI_Allreduce(&local_owned_after, &global_owned_after, 1, MPI_INT, MPI_SUM, comm);
  MPI_Allreduce(&local_owned_after, &max_owned_after, 1, MPI_INT, MPI_MAX, comm);
  ASSERT_MPI(global_owned_after == n_cells);
  ASSERT_MPI(max_owned_after < max_owned_before);

  const auto rebalanced_map = search::classify_embedded_geometry(mesh, embedded, options);
  const auto rebalanced_topology = search::reconstruct_cut_topology(mesh.local_mesh(), rebalanced_map);
  const auto rebalanced_state =
      search::build_distributed_cut_state(mesh, rebalanced_map, rebalanced_topology);
  const auto rebalanced_diag = search::diagnose_distributed_cut_state(rebalanced_state);
  if (!rebalanced_diag.ok) {
    for (const auto& message : rebalanced_diag.messages) {
      std::cerr << "Rebalanced distributed cut-state diagnostic: " << message << "\n";
    }
  }
  ASSERT_MPI(rebalanced_diag.ok);
  ASSERT_MPI(rebalanced_state.valid_for(mesh, rebalanced_map, rebalanced_topology));

  const auto owners_after_rebalance = exchanged_owner_map(rebalanced_state.exchanged_packet);
  int local_max_owner_after = -1;
  for (const auto& [key, owner] : owners_after_rebalance) {
    (void)key;
    local_max_owner_after = std::max(local_max_owner_after, static_cast<int>(owner));
  }
  int global_max_owner_after = -1;
  MPI_Allreduce(&local_max_owner_after,
                &global_max_owner_after,
                1,
                MPI_INT,
                MPI_MAX,
                comm);
  ASSERT_MPI(global_max_owner_after > global_max_owner_before);
  assert_positive_ghost_side_region_payloads(rebalanced_state, comm);
}

} // namespace svmp::test

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size >= 2) {
    svmp::test::run_cut_cell_mpi_test(MPI_COMM_WORLD, rank, size);
    svmp::test::run_cut_cell_rebalance_mpi_test(MPI_COMM_WORLD, rank, size);
    svmp::test::run_cut_cell_partition_method_qualification(MPI_COMM_WORLD,
                                                            rank,
                                                            size,
                                                            "block");
#if defined(SVMP_HAS_METIS)
    svmp::test::run_cut_cell_partition_method_qualification(MPI_COMM_WORLD,
                                                            rank,
                                                            size,
                                                            "metis");
#else
    if (rank == 0) {
      std::cerr << "Skipping CutCell METIS sparse exchange qualification: "
                << "SVMP_HAS_METIS not enabled.\n";
    }
#endif
#if defined(SVMP_HAS_PARMETIS)
    svmp::test::run_cut_cell_partition_method_qualification(MPI_COMM_WORLD,
                                                            rank,
                                                            size,
                                                            "parmetis");
#else
    if (rank == 0) {
      std::cerr << "Skipping CutCell ParMETIS sparse exchange qualification: "
                << "SVMP_HAS_PARMETIS not enabled.\n";
    }
#endif
  }

  if (rank == 0) {
    std::cout << "CutCell MPI tests PASSED\n";
  }
  MPI_Finalize();
  return 0;
}
