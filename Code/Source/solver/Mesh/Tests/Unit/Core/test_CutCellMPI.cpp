#include "Mesh.h"
#include "Search/CutCell.h"

#include <mpi.h>

#include <cmath>
#include <iostream>
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

static Mesh build_partitioned_hex_strip(int world_size)
{
  std::vector<real_t> coords;
  std::vector<offset_t> offsets;
  std::vector<index_t> conn;
  std::vector<CellShape> shapes;
  build_hex_strip_global_arrays(world_size, coords, offsets, conn, shapes);

  Mesh mesh(MeshComm::world());
  mesh.build_from_arrays_global_and_partition(3,
                                             coords,
                                             offsets,
                                             conn,
                                             shapes,
                                             PartitionHint::Cells,
                                             /*ghost_layers=*/1,
                                             {{"partition_method", "block"}});
  return mesh;
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
  int local_owned_cut = 0;
  for (const auto& record : map.cells) {
    if (record.classification == search::CutClassification::Cut &&
        record.owner_rank == rank) {
      ++local_owned_cut;
    }
    ASSERT_MPI(record.owner_rank >= 0);
    ASSERT_MPI(record.owner_rank < size);
  }

  int global_owned_cut = 0;
  MPI_Allreduce(&local_owned_cut, &global_owned_cut, 1, MPI_INT, MPI_SUM, comm);
  ASSERT_MPI(global_owned_cut >= 2);
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
  }

  if (rank == 0) {
    std::cout << "CutCell MPI tests PASSED\n";
  }
  MPI_Finalize();
  return 0;
}
