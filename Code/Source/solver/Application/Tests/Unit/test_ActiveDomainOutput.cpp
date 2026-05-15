#include <gtest/gtest.h>

#include "Application/Core/ActiveDomainOutput.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"

#include <memory>
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
