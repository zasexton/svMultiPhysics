#include <gtest/gtest.h>

#include "Application/Core/LevelSetMaintenanceHistory.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"

#include <memory>
#include <utility>
#include <vector>

namespace {

std::shared_ptr<svmp::Mesh> buildSingleQuadMesh()
{
  auto base = std::make_shared<svmp::MeshBase>();

  const std::vector<svmp::real_t> x_ref = {
      0.0, 0.0,
      1.0, 0.0,
      1.0, 1.0,
      0.0, 1.0,
  };
  const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 4};
  const std::vector<svmp::index_t> cell2vertex = {0, 1, 2, 3};

  svmp::CellShape shape{};
  shape.family = svmp::CellFamily::Quad;
  shape.num_corners = 4;
  shape.order = 1;
  base->build_from_arrays(
      /*spatial_dim=*/2,
      x_ref,
      cell2vertex_offsets,
      cell2vertex,
      {shape});
  base->finalize();
  return svmp::create_mesh(std::move(base));
}

[[nodiscard]] std::pair<std::size_t, std::size_t> fieldRange(
    const svmp::FE::systems::FESystem& system,
    svmp::FE::FieldId field)
{
  return {
      static_cast<std::size_t>(system.fieldDofOffset(field)),
      static_cast<std::size_t>(system.fieldDofHandler(field).getNumDofs())};
}

} // namespace

TEST(LevelSetMaintenanceHistory, CopiesOnlyRequestedFieldDofs)
{
  auto mesh = buildSingleQuadMesh();
  auto space =
      std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Quad4,
                                                  /*order=*/1);

  svmp::FE::systems::FESystem system(mesh);
  const auto pressure = system.addField(
      svmp::FE::systems::FieldSpec{.name = "Pressure",
                                   .space = space,
                                   .components = 1});
  const auto phi = system.addField(
      svmp::FE::systems::FieldSpec{.name = "phi",
                                   .space = space,
                                   .components = 1});
  ASSERT_NO_THROW(system.setup());

  const auto n_dofs =
      static_cast<std::size_t>(system.dofHandler().getNumDofs());
  std::vector<svmp::FE::Real> source(n_dofs, svmp::FE::Real{0.0});
  std::vector<svmp::FE::Real> target(n_dofs, svmp::FE::Real{0.0});
  for (std::size_t i = 0; i < n_dofs; ++i) {
    source[i] = svmp::FE::Real{100.0} + static_cast<svmp::FE::Real>(i);
    target[i] = svmp::FE::Real{10.0} + static_cast<svmp::FE::Real>(i);
  }
  const auto original_target = target;

  const auto copied =
      application::core::copyFieldDofsIntoFeOrderedSolution(
          system, phi, source, target);
  const auto [pressure_offset, pressure_count] = fieldRange(system, pressure);
  const auto [phi_offset, phi_count] = fieldRange(system, phi);

  EXPECT_EQ(copied, phi_count);
  for (std::size_t i = 0; i < pressure_count; ++i) {
    EXPECT_EQ(target[pressure_offset + i], original_target[pressure_offset + i]);
  }
  for (std::size_t i = 0; i < phi_count; ++i) {
    EXPECT_EQ(target[phi_offset + i], source[phi_offset + i]);
  }
}

TEST(LevelSetMaintenanceHistory, RejectsMismatchedSolutionSizes)
{
  auto mesh = buildSingleQuadMesh();
  auto space =
      std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Quad4,
                                                  /*order=*/1);

  svmp::FE::systems::FESystem system(mesh);
  const auto phi = system.addField(
      svmp::FE::systems::FieldSpec{.name = "phi",
                                   .space = space,
                                   .components = 1});
  ASSERT_NO_THROW(system.setup());

  std::vector<svmp::FE::Real> source(4u, svmp::FE::Real{1.0});
  std::vector<svmp::FE::Real> target(3u, svmp::FE::Real{0.0});
  EXPECT_THROW(
      (void)application::core::copyFieldDofsIntoFeOrderedSolution(
          system, phi, source, target),
      std::invalid_argument);
}
