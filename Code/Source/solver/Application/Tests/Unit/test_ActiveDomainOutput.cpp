#include <gtest/gtest.h>

#include "Application/Core/ActiveDomainOutput.h"
#include "FE/Assembly/MeshAccess.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"

#include <memory>
#include <map>
#include <vector>

namespace {

std::shared_ptr<svmp::Mesh> makeTwoQuadCellMesh()
{
  auto base = std::make_shared<svmp::MeshBase>();

  const std::vector<svmp::real_t> x_ref = {
      0.0, 0.0,
      1.0, 0.0,
      2.0, 0.0,
      0.0, 1.0,
      1.0, 1.0,
      2.0, 1.0,
  };
  const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 4, 8};
  const std::vector<svmp::index_t> cell2vertex = {
      0, 1, 4, 3,
      1, 2, 5, 4,
  };

  svmp::CellShape quad{};
  quad.family = svmp::CellFamily::Quad;
  quad.num_corners = 4;
  quad.order = 1;
  base->build_from_arrays(
      /*spatial_dim=*/2,
      x_ref,
      cell2vertex_offsets,
      cell2vertex,
      {quad, quad});
  base->finalize();
  return svmp::create_mesh(std::move(base));
}

std::shared_ptr<svmp::Mesh> makeSingleQuadCellMesh(
    const std::vector<svmp::real_t>& x_ref)
{
  auto base = std::make_shared<svmp::MeshBase>();

  const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 4};
  const std::vector<svmp::index_t> cell2vertex = {0, 1, 2, 3};

  svmp::CellShape quad{};
  quad.family = svmp::CellFamily::Quad;
  quad.num_corners = 4;
  quad.order = 1;
  base->build_from_arrays(
      /*spatial_dim=*/2,
      x_ref,
      cell2vertex_offsets,
      cell2vertex,
      {quad});
  base->finalize();
  return svmp::create_mesh(std::move(base));
}

} // namespace

TEST(ActiveDomainOutput, WritesWetVolumeFractionForCutCell)
{
  auto mesh = makeTwoQuadCellMesh();

  svmp::FE::geometry::CutQuadratureRule cut_rule;
  cut_rule.kind = svmp::FE::geometry::CutQuadratureKind::Volume;
  cut_rule.side = svmp::FE::geometry::CutIntegrationSide::Negative;
  cut_rule.volume_fraction = 0.375;
  cut_rule.provenance.parent_entity = 0;

  const std::vector<const svmp::FE::geometry::CutQuadratureRule*> rules = {
      &cut_rule,
  };
  const auto fields_written =
      application::core::writeWetVolumeFractionField(
          *mesh, "WetVolumeFraction", rules);

  EXPECT_EQ(fields_written, 1u);
  ASSERT_TRUE(mesh->has_field(svmp::EntityKind::Volume, "WetVolumeFraction"));
  const auto handle =
      mesh->field_handle(svmp::EntityKind::Volume, "WetVolumeFraction");
  ASSERT_EQ(mesh->field_type(handle), svmp::FieldScalarType::Float64);
  ASSERT_EQ(mesh->field_components(handle), 1u);
  const auto* data = static_cast<const double*>(mesh->field_data(handle));
  ASSERT_NE(data, nullptr);
  EXPECT_NEAR(data[0], 0.375, 1.0e-12);
}

TEST(ActiveDomainOutput, WetFractionsMatchGeneratedCutVolumeMetadata)
{
  auto mesh = makeTwoQuadCellMesh();

  std::vector<svmp::FE::geometry::CutQuadratureRule> generated_rules(3);
  generated_rules[0].kind = svmp::FE::geometry::CutQuadratureKind::Volume;
  generated_rules[0].side = svmp::FE::geometry::CutIntegrationSide::Negative;
  generated_rules[0].volume_fraction = 0.25;
  generated_rules[0].provenance.parent_entity = 0;

  generated_rules[1].kind = svmp::FE::geometry::CutQuadratureKind::Volume;
  generated_rules[1].side = svmp::FE::geometry::CutIntegrationSide::Negative;
  generated_rules[1].volume_fraction = 0.5;
  generated_rules[1].provenance.parent_entity = 1;

  generated_rules[2].kind = svmp::FE::geometry::CutQuadratureKind::Volume;
  generated_rules[2].side = svmp::FE::geometry::CutIntegrationSide::Negative;
  generated_rules[2].volume_fraction = 0.25;
  generated_rules[2].provenance.parent_entity = 1;

  const std::vector<const svmp::FE::geometry::CutQuadratureRule*> rules = {
      &generated_rules[0],
      &generated_rules[1],
      &generated_rules[2],
  };
  const auto expected =
      application::core::collectWetVolumeFractions(mesh->n_cells(), rules);
  ASSERT_EQ(expected.size(), mesh->n_cells());
  EXPECT_DOUBLE_EQ(expected[0], 0.25);
  EXPECT_DOUBLE_EQ(expected[1], 0.75);

  const auto fields_written =
      application::core::writeWetVolumeFractionField(
          *mesh, "WetVolumeFraction", rules);

  EXPECT_EQ(fields_written, 1u);
  const auto handle =
      mesh->field_handle(svmp::EntityKind::Volume, "WetVolumeFraction");
  const auto* data = static_cast<const double*>(mesh->field_data(handle));
  ASSERT_NE(data, nullptr);
  for (std::size_t cell = 0; cell < expected.size(); ++cell) {
    EXPECT_DOUBLE_EQ(data[cell], expected[cell]);
  }
}

TEST(ActiveDomainOutput, WritesFullWetAndFullDryFractions)
{
  auto mesh = makeTwoQuadCellMesh();

  svmp::FE::geometry::CutQuadratureRule full_wet_rule;
  full_wet_rule.kind = svmp::FE::geometry::CutQuadratureKind::Volume;
  full_wet_rule.side = svmp::FE::geometry::CutIntegrationSide::Negative;
  full_wet_rule.volume_fraction = 1.0;
  full_wet_rule.provenance.parent_entity = 0;

  const std::vector<const svmp::FE::geometry::CutQuadratureRule*> rules = {
      &full_wet_rule,
  };
  const auto fields_written =
      application::core::writeWetVolumeFractionField(
          *mesh, "WetVolumeFraction", rules);

  EXPECT_EQ(fields_written, 1u);
  const auto handle =
      mesh->field_handle(svmp::EntityKind::Volume, "WetVolumeFraction");
  const auto* data = static_cast<const double*>(mesh->field_data(handle));
  ASSERT_NE(data, nullptr);
  EXPECT_DOUBLE_EQ(data[0], 1.0);
  EXPECT_DOUBLE_EQ(data[1], 0.0);
}

TEST(ActiveDomainOutput, TracksWetVolumeDriftAcrossAcceptedSteps)
{
  std::map<std::string, svmp::FE::Real> initial_wet_volume_by_key;
  const auto first =
      application::core::computeWetVolumeDrift(
          "phi|free_surface|101", 2.0, initial_wet_volume_by_key);
  const auto later =
      application::core::computeWetVolumeDrift(
          "phi|free_surface|101", 2.1, initial_wet_volume_by_key);
  const auto zero_initial =
      application::core::computeWetVolumeDrift(
          "phi|empty|102", 0.0, initial_wet_volume_by_key);
  const auto zero_relative =
      application::core::computeWetVolumeDrift(
          "phi|empty|102", 0.25, initial_wet_volume_by_key);

  EXPECT_DOUBLE_EQ(first.initial_wet_volume, 2.0);
  EXPECT_DOUBLE_EQ(first.wet_volume_drift, 0.0);
  EXPECT_DOUBLE_EQ(first.relative_wet_volume_drift, 0.0);
  EXPECT_DOUBLE_EQ(later.initial_wet_volume, 2.0);
  EXPECT_NEAR(later.wet_volume_drift, 0.1, 1.0e-12);
  EXPECT_NEAR(later.relative_wet_volume_drift, 0.05, 1.0e-12);
  EXPECT_DOUBLE_EQ(zero_initial.initial_wet_volume, 0.0);
  EXPECT_DOUBLE_EQ(zero_relative.wet_volume_drift, 0.25);
  EXPECT_DOUBLE_EQ(zero_relative.relative_wet_volume_drift, 0.0);
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

TEST(ActiveDomainOutput, CollectsPhysicalCutVolumeMeasureOnScaledQuad)
{
  auto mesh = makeSingleQuadCellMesh({
      0.0, 0.0,
      2.0, 0.0,
      2.0, 3.0,
      0.0, 3.0,
  });
  svmp::FE::assembly::MeshAccess mesh_access(*mesh);

  svmp::FE::geometry::CutQuadratureRule cut_rule;
  cut_rule.kind = svmp::FE::geometry::CutQuadratureKind::Volume;
  cut_rule.side = svmp::FE::geometry::CutIntegrationSide::Negative;
  cut_rule.measure = 2.0;
  cut_rule.parent_measure = 4.0;
  cut_rule.volume_fraction = 0.5;
  cut_rule.frame = svmp::FE::geometry::CutGeometryFrame::Reference;
  cut_rule.provenance.frame = svmp::FE::geometry::CutGeometryFrame::Reference;
  cut_rule.provenance.parent_entity = 0;
  cut_rule.points.push_back(
      {{{0.0, 0.0, 0.0}}, {{0.0, 0.0, 1.0}}, 2.0});

  const std::vector<const svmp::FE::geometry::CutQuadratureRule*> rules = {
      &cut_rule,
  };
  const auto summary =
      application::core::collectCutVolumeMeasures(mesh_access, rules);

  EXPECT_EQ(summary.rule_count, 1u);
  EXPECT_EQ(summary.physical_rule_count, 1u);
  EXPECT_EQ(summary.skipped_physical_rule_count, 0u);
  EXPECT_NEAR(summary.reference_measure, 2.0, 1.0e-12);
  EXPECT_NEAR(summary.physical_measure, 3.0, 1.0e-12);

  const auto wet_fraction =
      application::core::collectWetVolumeFractions(mesh->n_cells(), rules);
  ASSERT_EQ(wet_fraction.size(), 1u);
  EXPECT_DOUBLE_EQ(wet_fraction[0], 0.5);
}

TEST(ActiveDomainOutput, CollectsPhysicalFullCellMeasureOnDistortedQuad)
{
  auto mesh = makeSingleQuadCellMesh({
      0.0, 0.0,
      2.0, 0.0,
      2.5, 3.0,
      0.0, 2.0,
  });
  svmp::FE::assembly::MeshAccess mesh_access(*mesh);

  svmp::FE::geometry::CutQuadratureRule full_rule;
  full_rule.kind = svmp::FE::geometry::CutQuadratureKind::Volume;
  full_rule.side = svmp::FE::geometry::CutIntegrationSide::Negative;
  full_rule.measure = 4.0;
  full_rule.parent_measure = 4.0;
  full_rule.volume_fraction = 1.0;
  full_rule.full_cell_equivalent = true;
  full_rule.frame = svmp::FE::geometry::CutGeometryFrame::Reference;
  full_rule.provenance.frame = svmp::FE::geometry::CutGeometryFrame::Reference;
  full_rule.provenance.parent_entity = 0;
  full_rule.points.push_back(
      {{{0.0, 0.0, 0.0}}, {{0.0, 0.0, 1.0}}, 4.0});

  const std::vector<const svmp::FE::geometry::CutQuadratureRule*> rules = {
      &full_rule,
  };
  const auto summary =
      application::core::collectCutVolumeMeasures(mesh_access, rules);

  EXPECT_EQ(summary.rule_count, 1u);
  EXPECT_EQ(summary.physical_rule_count, 1u);
  EXPECT_EQ(summary.skipped_physical_rule_count, 0u);
  EXPECT_NEAR(summary.reference_measure, 4.0, 1.0e-12);
  EXPECT_NEAR(summary.physical_measure, 5.5, 1.0e-12);

  const auto wet_fraction =
      application::core::collectWetVolumeFractions(mesh->n_cells(), rules);
  ASSERT_EQ(wet_fraction.size(), 1u);
  EXPECT_DOUBLE_EQ(wet_fraction[0], 1.0);
}

#endif
