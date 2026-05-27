#include <gtest/gtest.h>

#include "Application/Core/LevelSetCurvatureSamples.h"
#include "Application/Core/LevelSetMaintenanceHistory.h"
#include "FE/Assembly/CutIntegrationContext.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"

#include <cmath>
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

TEST(LevelSetMaintenanceHistory, CopiesHighOrderFieldDofsToCurrentAndPrevious)
{
  auto mesh = buildSingleQuadMesh();
  auto q1_space =
      std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Quad4,
                                                  /*order=*/1);
  auto q2_space =
      std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Quad4,
                                                  /*order=*/2);

  svmp::FE::systems::FESystem system(mesh);
  const auto pressure = system.addField(
      svmp::FE::systems::FieldSpec{.name = "Pressure",
                                   .space = q1_space,
                                   .components = 1});
  const auto phi = system.addField(
      svmp::FE::systems::FieldSpec{.name = "phi",
                                   .space = q2_space,
                                   .components = 1});
  ASSERT_NO_THROW(system.setup());

  const auto n_dofs =
      static_cast<std::size_t>(system.dofHandler().getNumDofs());
  std::vector<svmp::FE::Real> repaired(n_dofs, svmp::FE::Real{0.0});
  std::vector<svmp::FE::Real> current(n_dofs, svmp::FE::Real{0.0});
  std::vector<svmp::FE::Real> previous(n_dofs, svmp::FE::Real{0.0});
  for (std::size_t i = 0; i < n_dofs; ++i) {
    repaired[i] = svmp::FE::Real{200.0} + static_cast<svmp::FE::Real>(i);
    current[i] = svmp::FE::Real{20.0} + static_cast<svmp::FE::Real>(i);
    previous[i] = svmp::FE::Real{-20.0} - static_cast<svmp::FE::Real>(i);
  }
  const auto original_current = current;
  const auto original_previous = previous;

  const auto copied_current =
      application::core::copyFieldDofsIntoFeOrderedSolution(
          system, phi, repaired, current);
  const auto copied_previous =
      application::core::copyFieldDofsIntoFeOrderedSolution(
          system, phi, repaired, previous);
  const auto [pressure_offset, pressure_count] = fieldRange(system, pressure);
  const auto [phi_offset, phi_count] = fieldRange(system, phi);

  EXPECT_GT(phi_count, 4u);
  EXPECT_EQ(copied_current, phi_count);
  EXPECT_EQ(copied_previous, phi_count);
  for (std::size_t i = 0; i < pressure_count; ++i) {
    EXPECT_EQ(current[pressure_offset + i],
              original_current[pressure_offset + i]);
    EXPECT_EQ(previous[pressure_offset + i],
              original_previous[pressure_offset + i]);
  }
  for (std::size_t i = 0; i < phi_count; ++i) {
    EXPECT_EQ(current[phi_offset + i], repaired[phi_offset + i]);
    EXPECT_EQ(previous[phi_offset + i], repaired[phi_offset + i]);
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

TEST(LevelSetCurvatureSamples,
     CollectsGeneratedCutVolumeQuadratureSamplesForActiveSide)
{
  auto mesh = buildSingleQuadMesh();
  auto space =
      std::make_shared<svmp::FE::spaces::H1Space>(svmp::FE::ElementType::Quad4,
                                                  /*order=*/1);

  svmp::FE::systems::FESystem system(mesh);
  const auto phi = system.addField(
      svmp::FE::systems::FieldSpec{
          .name = "phi",
          .space = space,
          .components = 1,
          .source_kind =
              svmp::FE::systems::FieldSourceKind::PrescribedData});
  ASSERT_NO_THROW(system.setup());
  const std::vector<svmp::FE::Real> prescribed_coefficients(
      4u, svmp::FE::Real{7.0});
  system.setPrescribedFieldCoefficients(phi, prescribed_coefficients);

  constexpr int marker = 42;
  auto cut_context =
      std::make_shared<svmp::FE::assembly::CutIntegrationContext>();

  svmp::FE::geometry::CutQuadratureRule rule;
  rule.kind = svmp::FE::geometry::CutQuadratureKind::Volume;
  rule.side = svmp::FE::geometry::CutIntegrationSide::Negative;
  rule.frame = svmp::FE::geometry::CutGeometryFrame::Reference;
  rule.provenance.parent_entity = 0;
  rule.provenance.marker = marker;
  rule.measure = svmp::FE::Real{0.5};
  rule.parent_measure = svmp::FE::Real{1.0};
  rule.volume_fraction = svmp::FE::Real{0.5};
  rule.full_cell_equivalent = false;

  svmp::FE::geometry::CutQuadraturePoint qp;
  qp.point = {{svmp::FE::Real{0.25}, svmp::FE::Real{0.25}, svmp::FE::Real{0.0}}};
  qp.parent_coordinate = qp.point;
  qp.weight = svmp::FE::Real{0.5};
  rule.points.push_back(qp);

  svmp::FE::assembly::CutCellAssemblyMetadata metadata;
  metadata.parent_entity = 0;
  metadata.side = svmp::FE::geometry::CutIntegrationSide::Negative;
  metadata.volume_fraction = rule.volume_fraction;
  cut_context->addGeneratedVolumeRule(marker, metadata, rule);
  system.setCutIntegrationContext(cut_context);

  const svmp::FE::systems::SystemStateView state;
  const auto negative_samples =
      application::core::collectLevelSetCurvatureCutVolumeSupplementalSamples(
          system,
          state,
          phi,
          marker,
          svmp::FE::geometry::CutIntegrationSide::Negative);
  ASSERT_EQ(negative_samples.size(), 1u);
  EXPECT_EQ(negative_samples.front().parent_cell, 0);
  EXPECT_NEAR(negative_samples.front().value, svmp::FE::Real{7.0}, 1.0e-12);
  EXPECT_TRUE(std::isfinite(negative_samples.front().coordinate[0]));
  EXPECT_TRUE(std::isfinite(negative_samples.front().coordinate[1]));
  EXPECT_TRUE(std::isfinite(negative_samples.front().coordinate[2]));

  const auto positive_samples =
      application::core::collectLevelSetCurvatureCutVolumeSupplementalSamples(
          system,
          state,
          phi,
          marker,
          svmp::FE::geometry::CutIntegrationSide::Positive);
  EXPECT_TRUE(positive_samples.empty());
}
