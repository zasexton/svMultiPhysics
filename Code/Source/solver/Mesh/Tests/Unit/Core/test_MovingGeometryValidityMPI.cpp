#include "../../../Core/DistributedMesh.h"
#include "../../../Core/MeshBase.h"
#include "../../../Topology/CellShape.h"
#include "../../../Validation/MovingGeometryValidity.h"

#include <mpi.h>

#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

namespace {

#define ASSERT_TRUE_MPI(cond)                                                                        \
  do {                                                                                               \
    if (!(cond)) {                                                                                   \
      std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__ << " : " #cond "\n";       \
      MPI_Abort(comm, 1);                                                                            \
    }                                                                                                \
  } while (0)

std::shared_ptr<svmp::MeshBase> make_rank_face(int rank)
{
  auto mesh = std::make_shared<svmp::MeshBase>();
  const double x = 0.0;
  const std::vector<svmp::real_t> coords = {
      x, 0.0, 0.0,
      x, 1.0, 0.0,
      x, 1.0, 1.0,
      x, 0.0, 1.0};
  mesh->build_from_arrays(3, coords, {0}, {}, {});
  mesh->set_faces_from_arrays({{svmp::CellFamily::Quad, 4, 1}}, {0, 4}, {0, 1, 2, 3},
                              {{{svmp::INVALID_INDEX, svmp::INVALID_INDEX}}});
  mesh->set_face_gids({static_cast<svmp::gid_t>(1000 + rank)});
  mesh->set_boundary_label(0, rank == 0 ? 21 : 22);
  return mesh;
}

void run_cross_rank_validity_test(MPI_Comm comm, int rank)
{
  svmp::DistributedMesh mesh(make_rank_face(rank), comm);

  auto policy = svmp::validation::MovingGeometryValidity::contact_policy();
  policy.checks.clear();
  svmp::validation::MovingGeometryCheckSpec min_sep;
  min_sep.check = svmp::validation::MovingGeometryCheck::MinimumSeparation;
  min_sep.name = "mpi-min-separation";
  min_sep.threshold = 0.10;
  min_sep.action = svmp::validation::ValidityAction::Reject;
  min_sep.label_pairs.push_back({21, 22, true});
  policy.checks.push_back(min_sep);

  svmp::validation::MovingGeometryCheckSpec intersection;
  intersection.check = svmp::validation::MovingGeometryCheck::BoundarySelfIntersection;
  intersection.name = "mpi-self-intersection";
  intersection.action = svmp::validation::ValidityAction::Reject;
  intersection.label_pairs.push_back({21, 22, true});
  policy.checks.push_back(intersection);

  const auto report = svmp::validation::MovingGeometryValidity::evaluate(mesh, policy);
  ASSERT_TRUE_MPI(!report.passed);

  int min_sep_count = 0;
  int self_intersection_count = 0;
  for (const auto& failure : report.failures) {
    if (failure.check_name == "mpi-min-separation") {
      ++min_sep_count;
      ASSERT_TRUE_MPI(failure.global_ids.size() == 2);
      ASSERT_TRUE_MPI(failure.owner_rank == 0);
      ASSERT_TRUE_MPI(failure.labels.size() == 2);
    } else if (failure.check_name == "mpi-self-intersection") {
      ++self_intersection_count;
      ASSERT_TRUE_MPI(failure.global_ids.size() == 2);
      ASSERT_TRUE_MPI(failure.owner_rank == 0);
    }
  }
  ASSERT_TRUE_MPI(min_sep_count == 1);
  ASSERT_TRUE_MPI(self_intersection_count == 1);
}

} // namespace

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int world_rank = 0;
  int world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm split_comm = MPI_COMM_NULL;
  if (world_size > 2) {
    const int color = (world_rank < 2) ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &split_comm);
    if (color == MPI_UNDEFINED) {
      MPI_Finalize();
      return 0;
    }
    comm = split_comm;
  }

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (size >= 2) {
    run_cross_rank_validity_test(comm, rank);
  }

  if (split_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&split_comm);
  }
  MPI_Finalize();
  return 0;
}
