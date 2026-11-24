#include <gtest/gtest.h>

#include "Application/Translators/LevelSetEquationTranslator.h"
#include "FE/Systems/FESystem.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Physics/Core/EquationModuleInput.h"

#include <mpi.h>

#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

/// One unit quadrilateral whose vertices carry the global ids first_id + 0..3.
std::shared_ptr<svmp::Mesh> unitQuadWithVertexIds(svmp::gid_t first_id)
{
  auto base = std::make_shared<svmp::MeshBase>();
  svmp::CellShape shape{};
  shape.family = svmp::CellFamily::Quad;
  shape.num_corners = 4;
  shape.order = 1;
  base->build_from_arrays(2, {0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0}, {0, 4}, {0, 1, 2, 3}, {shape});
  base->set_vertex_gids({first_id, first_id + 1, first_id + 2, first_id + 3});
  base->finalize();
  return svmp::create_mesh(std::move(base));
}

/// Translates a level-set inflow boundary that reads the table @p file_path on
/// the mesh @p mesh.
void translateInflowTable(const std::shared_ptr<svmp::Mesh>& mesh, const std::string& file_path)
{
  svmp::Physics::EquationModuleInput input{};
  input.equation_type = "level_set";
  input.mesh_name = "quad";
  input.mesh = mesh->local_mesh_ptr();
  input.equation_params["Level_set_field_name"] = svmp::Physics::ParameterValue{true, "phi"};
  input.equation_params["Velocity_source"] = svmp::Physics::ParameterValue{true, "constant"};
  input.equation_params["Constant_velocity"] = svmp::Physics::ParameterValue{true, "1.0 0.0 0.0"};
  svmp::Physics::BoundaryConditionInput inflow{};
  inflow.name = "inlet";
  inflow.boundary_marker = 4;
  inflow.params["Type"] = svmp::Physics::ParameterValue{true, "LevelSetInflow"};
  inflow.params["Temporal_and_spatial_values_file_path"] = svmp::Physics::ParameterValue{true, file_path};
  input.boundary_conditions.push_back(std::move(inflow));
  svmp::FE::systems::FESystem system(mesh);
  (void)application::translators::level_set::createModule(input, system);
}

} // namespace

// A temporal and spatial inflow table on a run where the second rank holds
// none of the table's nodes (its part of the mesh does not touch the inflow
// face): every rank translates the boundary (the second rank threw "did not
// match any local mesh vertices").  A table that matches no vertex on any
// rank is still refused, on every rank.
TEST(LevelSetInflowTableMPI, ARankWithoutTheTableNodesTranslatesTheBoundary)
{
  int size = 1;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (size != 2) {
    GTEST_SKIP() << "Run with exactly 2 MPI ranks";
  }
  // Nodes 1..4 (one-based) are the vertices with global ids 0..3.
  const std::string file_path = "level_set_inflow_table_mpi_" + std::to_string(rank) + ".dat";
  {
    std::ofstream out(file_path);
    out << "1 2 4\n0.0\n1.0\n";
    for (int node = 1; node <= 4; ++node) {
      out << node << "\n" << static_cast<double>(node) << "\n" << static_cast<double>(node) + 0.5 << "\n";
    }
  }

  // Rank 0 holds the table's nodes, rank 1 holds vertices 100..103.
  std::string failure;
  try {
    translateInflowTable(unitQuadWithVertexIds(rank == 0 ? 0 : 100), file_path);
  } catch (const std::exception& error) {
    failure = error.what();
  }
  EXPECT_TRUE(failure.empty()) << "rank " << rank << ": " << failure;

  // No rank holds a node of the table.
  std::string refusal;
  try {
    translateInflowTable(unitQuadWithVertexIds(200 + 10 * static_cast<svmp::gid_t>(rank)), file_path);
  } catch (const std::exception& error) {
    refusal = error.what();
  }
  EXPECT_NE(refusal.find("did not match any mesh vertex"), std::string::npos) << "rank " << rank << ": " << refusal;
  std::remove(file_path.c_str());
}
