#include "../../../Search/ContactProximity.h"
#include "../../../Core/DistributedMesh.h"
#include "../../../Topology/CellShape.h"

#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

constexpr svmp::label_t kSlaveLabel = 601;
constexpr svmp::label_t kMasterLabel = 602;

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

void build_local_surface(svmp::DistributedMesh& mesh,
                         double x,
                         svmp::label_t label,
                         svmp::gid_t face_gid) {
  const std::vector<svmp::real_t> x_ref = {
      x, 0.0, 0.0,
      x, 1.0, 0.0,
      x, 1.0, 1.0,
      x, 0.0, 1.0};
  mesh.build_from_arrays(3, x_ref, {0}, {}, {});
  mesh.set_faces_from_arrays({quad_shape()}, {0, 4}, {0, 1, 2, 3},
                             {{{svmp::INVALID_INDEX, svmp::INVALID_INDEX}}});
  mesh.set_face_gids({face_gid});
  mesh.set_boundary_label(0, label);
}

} // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  svmp::DistributedMesh slave(MPI_COMM_WORLD);
  svmp::DistributedMesh master(MPI_COMM_WORLD);
  const svmp::rank_t rank = slave.rank();
  const int size = slave.world_size();

  build_local_surface(slave, 0.0, kSlaveLabel, static_cast<svmp::gid_t>(7000 + rank));
  build_local_surface(master, 0.2, kMasterLabel, static_cast<svmp::gid_t>(8000 + rank));

  const svmp::rank_t remote_owner = static_cast<svmp::rank_t>((rank + 1) % size);
  slave.set_ownership(0, svmp::EntityKind::Face, svmp::Ownership::Owned, rank);
  master.set_ownership(0, svmp::EntityKind::Face, svmp::Ownership::Ghost, remote_owner);

  svmp::search::ContactCandidateOptions options;
  options.search_radius = 1.0;
  options.activation_distance = 0.3;
  options.only_nearest_per_source = true;
  options.remove_duplicate_pairs = true;

  svmp::search::ContactProximityRegistry registry;
  registry.register_contact(
      "mpi-contact",
      svmp::search::ContactSurfaceSpec::from_distributed_mesh(
          slave, kSlaveLabel, svmp::search::ContactEntityKind::Surface,
          svmp::Configuration::Reference, "slave"),
      svmp::search::ContactSurfaceSpec::from_distributed_mesh(
          master, kMasterLabel, svmp::search::ContactEntityKind::Surface,
          svmp::Configuration::Reference, "master"),
      options);

  auto map = registry.build_trial_map("mpi-contact");
  ASSERT_TRUE_MPI(map.pairs.size() == 1u);
  const auto& pair = map.pairs.front();
  ASSERT_TRUE_MPI(pair.provenance.source_local_rank == rank);
  ASSERT_TRUE_MPI(pair.provenance.target_local_rank == rank);
  ASSERT_TRUE_MPI(pair.provenance.source_owner_rank == rank);
  ASSERT_TRUE_MPI(pair.provenance.target_owner_rank == remote_owner);
  ASSERT_TRUE_MPI(pair.provenance.canonical_owner_rank == std::min(rank, remote_owner));
  ASSERT_TRUE_MPI(pair.state == svmp::search::ContactPairState::Active);
  ASSERT_TRUE_MPI(std::abs(pair.projection.unsigned_gap - 0.2) < 1.0e-12);
  ASSERT_TRUE_MPI(map.valid_for_current_revisions());

  registry.commit_map(map);
  ASSERT_TRUE_MPI(registry.committed_map_valid("mpi-contact"));

  master.set_ownership(0, svmp::EntityKind::Face, svmp::Ownership::Owned, rank);
  ASSERT_TRUE_MPI(!registry.committed_map_valid("mpi-contact"));

  svmp::DistributedMesh duplicate_slave(MPI_COMM_WORLD);
  svmp::DistributedMesh duplicate_master(MPI_COMM_WORLD);
  build_local_surface(duplicate_slave, 0.0, kSlaveLabel, 9000);
  build_local_surface(duplicate_master, 0.2, kMasterLabel, 9100);
  duplicate_slave.set_ownership(0, svmp::EntityKind::Face,
                                rank == 0 ? svmp::Ownership::Owned : svmp::Ownership::Ghost,
                                0);
  duplicate_master.set_ownership(
      0, svmp::EntityKind::Face,
      rank == static_cast<svmp::rank_t>(size - 1) ? svmp::Ownership::Owned : svmp::Ownership::Ghost,
      static_cast<svmp::rank_t>(size - 1));

  svmp::search::ContactProximityRegistry duplicate_registry;
  duplicate_registry.register_contact(
      "replicated-cross-rank-contact",
      svmp::search::ContactSurfaceSpec::from_distributed_mesh(
          duplicate_slave, kSlaveLabel, svmp::search::ContactEntityKind::Surface,
          svmp::Configuration::Reference, "replicated-slave"),
      svmp::search::ContactSurfaceSpec::from_distributed_mesh(
          duplicate_master, kMasterLabel, svmp::search::ContactEntityKind::Surface,
          svmp::Configuration::Reference, "replicated-master"),
      options);

  const auto duplicate_map = duplicate_registry.build_trial_map("replicated-cross-rank-contact");
  ASSERT_TRUE_MPI(duplicate_map.pairs.size() == 1u);
  const auto& duplicate_pair = duplicate_map.pairs.front();
  ASSERT_TRUE_MPI(duplicate_pair.provenance.source_owner_rank == 0);
  ASSERT_TRUE_MPI(duplicate_pair.provenance.target_owner_rank == static_cast<svmp::rank_t>(size - 1));
  ASSERT_TRUE_MPI(duplicate_pair.provenance.canonical_owner_rank == 0);

  const unsigned long long local_pair_id =
      static_cast<unsigned long long>(duplicate_pair.provenance.pair_id);
  unsigned long long min_pair_id = 0;
  unsigned long long max_pair_id = 0;
  MPI_Allreduce(&local_pair_id, &min_pair_id, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&local_pair_id, &max_pair_id, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
  ASSERT_TRUE_MPI(min_pair_id == max_pair_id);

  const int local_duplicate_count = static_cast<int>(duplicate_map.pairs.size());
  int global_duplicate_count = 0;
  MPI_Allreduce(&local_duplicate_count, &global_duplicate_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  ASSERT_TRUE_MPI(global_duplicate_count == size);

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    std::cout << "ContactProximity MPI tests PASSED\n";
  }
  MPI_Finalize();
  return 0;
}
