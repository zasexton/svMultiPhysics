#include "../../../Search/MultiMeshInterface.h"
#include "../../../Core/DistributedMesh.h"
#include "../../../Topology/CellShape.h"

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <vector>

namespace {

constexpr svmp::label_t kSourceLabel = 91;
constexpr svmp::label_t kTargetLabel = 92;

#define ASSERT_TRUE_MPI(cond)                                                                  \
  do {                                                                                         \
    if (!(cond)) {                                                                             \
      std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__ << " : " #cond       \
                << "\n";                                                                       \
      MPI_Abort(MPI_COMM_WORLD, 1);                                                            \
    }                                                                                          \
  } while (0)

svmp::CellShape quad_shape() {
  return {svmp::CellFamily::Quad, 4, 1};
}

svmp::CellShape line_shape() {
  return {svmp::CellFamily::Line, 2, 1};
}

void build_local_side(svmp::DistributedMesh& mesh,
                      double x0,
                      double x1,
                      svmp::label_t label,
                      bool use_right_face) {
  const std::vector<svmp::real_t> x_ref = {
      x0, 0.0, 0.0,
      x1, 0.0, 0.0,
      x1, 1.0, 0.0,
      x0, 1.0, 0.0};
  const std::vector<svmp::offset_t> cell_offsets = {0, 4};
  const std::vector<svmp::index_t> cells = {0, 1, 2, 3};
  mesh.build_from_arrays(3, x_ref, cell_offsets, cells, {quad_shape()});

  const std::vector<svmp::CellShape> face_shapes(1, line_shape());
  const std::vector<svmp::offset_t> face_offsets = {0, 2};
  const std::vector<svmp::index_t> faces = use_right_face
      ? std::vector<svmp::index_t>{1, 2}
      : std::vector<svmp::index_t>{3, 0};
  const std::vector<std::array<svmp::index_t, 2>> face2cell = {{{0, svmp::INVALID_INDEX}}};
  mesh.set_faces_from_arrays(face_shapes, face_offsets, faces, face2cell);
  mesh.set_face_gids({static_cast<svmp::gid_t>(1000 + mesh.rank())});
  mesh.set_boundary_label(0, label);
}

} // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  svmp::DistributedMesh source(MPI_COMM_WORLD);
  svmp::DistributedMesh target(MPI_COMM_WORLD);
  const int rank = static_cast<int>(source.rank());
  const int size = source.world_size();

  build_local_side(source, 0.0, 1.0, kSourceLabel, true);
  build_local_side(target, 1.1, 1.4, kTargetLabel, false);

  const svmp::rank_t remote_owner = static_cast<svmp::rank_t>((rank + 1) % size);
  source.set_ownership(0, svmp::EntityKind::Face, svmp::Ownership::Owned, source.rank());
  target.set_ownership(0, svmp::EntityKind::Face, svmp::Ownership::Ghost, remote_owner);

  svmp::search::InterfaceSearchRegistry registry;
  registry.register_interface(
      "mpi-wall",
      svmp::search::InterfaceSideSpec::from_distributed_mesh(source, kSourceLabel),
      svmp::search::InterfaceSideSpec::from_distributed_mesh(target, kTargetLabel),
      1.0);

  auto map = registry.build_trial_map("mpi-wall");
  ASSERT_TRUE_MPI(map.pairs.size() == 1u);
  const auto& pair = map.pairs.front();
  ASSERT_TRUE_MPI(pair.source_local_rank == source.rank());
  ASSERT_TRUE_MPI(pair.target_local_rank == target.rank());
  ASSERT_TRUE_MPI(pair.source_owner_rank == source.rank());
  ASSERT_TRUE_MPI(pair.target_owner_rank == remote_owner);
  if (size > 1) {
    ASSERT_TRUE_MPI(pair.source_owner_rank != pair.target_owner_rank);
  }
  ASSERT_TRUE_MPI(std::abs(pair.distance - 0.1) < 1.0e-12);
  ASSERT_TRUE_MPI(map.valid_for_current_revisions());

  registry.commit_map(map);
  ASSERT_TRUE_MPI(registry.committed_map_valid("mpi-wall"));

  target.set_ownership(0, svmp::EntityKind::Face, svmp::Ownership::Owned, target.rank());
  ASSERT_TRUE_MPI(!registry.committed_map_valid("mpi-wall"));

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    std::cout << "MultiMeshInterface MPI tests PASSED\n";
  }
  MPI_Finalize();
  return 0;
}
