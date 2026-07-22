#include <gtest/gtest.h>

#include "Application/Core/LevelSetCurvatureSamples.h"
#include "Application/Core/LevelSetMaintenanceHistory.h"
#include "FE/Assembly/CutIntegrationContext.h"
#include "FE/Assembly/MeshAccess.h"
#include "FE/Basis/NodeOrderingConventions.h"
#include "FE/Interfaces/FreeSurfaceGeometrySnapshot.h"
#include "FE/Interfaces/LevelSetInterfaceDomain.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
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

std::shared_ptr<svmp::Mesh> buildWarpedBiquadraticQuadMesh()
{
  auto base = std::make_shared<svmp::MeshBase>();

  const std::vector<svmp::real_t> x_ref = {
      0.0, 0.0,
      2.0, 0.0,
      2.0, 1.0,
      0.0, 1.0,
      1.0, -0.2,
      2.2, 0.5,
      1.0, 1.3,
      -0.1, 0.5,
      1.4, 0.35,
  };
  const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 9};
  const std::vector<svmp::index_t> cell2vertex = {
      0, 1, 2, 3, 4, 5, 6, 7, 8};

  svmp::CellShape shape{};
  shape.family = svmp::CellFamily::Quad;
  shape.num_corners = 4;
  shape.order = 2;
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
          svmp::FE::geometry::CutIntegrationSide::Negative,
          /*evaluated_state_source_revision=*/0u);
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
          svmp::FE::geometry::CutIntegrationSide::Positive,
          /*evaluated_state_source_revision=*/0u);
  EXPECT_TRUE(positive_samples.empty());
}

TEST(LevelSetCurvatureSamples,
     MapsWarpedHighOrderValueAndCoordinateAtIdenticalReferencePoint)
{
  const auto mesh = buildWarpedBiquadraticQuadMesh();
  const svmp::FE::assembly::MeshAccess access(*mesh);
  const std::array<svmp::FE::Real, 3> xi{{0.0, 0.0, 0.0}};

  const auto physical =
      application::core::mapLevelSetCurvatureReferenceSampleToPhysical(
          access, 0, xi);
  ASSERT_TRUE(physical.has_value());
  EXPECT_NEAR((*physical)[0], 1.4, 1.0e-13);
  EXPECT_NEAR((*physical)[1], 0.35, 1.0e-13);
  EXPECT_NEAR((*physical)[2], 0.0, 1.0e-13);

  std::vector<std::array<svmp::FE::Real, 3>> nodes;
  access.getCellCoordinates(0, nodes);
  ASSERT_EQ(nodes.size(), 9u);
  std::array<svmp::FE::Real, 3> nodal_average{{0.0, 0.0, 0.0}};
  for (const auto& node : nodes) {
    for (std::size_t d = 0; d < nodal_average.size(); ++d) {
      nodal_average[d] += node[d] / static_cast<svmp::FE::Real>(nodes.size());
    }
  }
  EXPECT_GT(std::abs((*physical)[0] - nodal_average[0]), 0.25);
  EXPECT_GT(std::abs((*physical)[1] - nodal_average[1]), 0.1);
}

TEST(LevelSetCurvatureSamples,
     Q3AuthoritativeInterfacePairsSampleWithSnapshotRevision)
{
  constexpr int marker = 943;
  constexpr svmp::FE::Real first_root = svmp::FE::Real{0.55};
  constexpr svmp::FE::Real second_root = svmp::FE::Real{0.70};
  constexpr std::uint64_t source_value_revision = 37u;
  const auto polynomial = [](svmp::FE::Real xi) {
    return (xi - first_root) * (xi - second_root);
  };

  auto mesh = buildSingleQuadMesh();
  auto q3_space =
      std::make_shared<svmp::FE::spaces::H1Space>(
          svmp::FE::ElementType::Quad4, /*order=*/3);
  svmp::FE::systems::FESystem system(mesh);
  const auto phi = system.addField(
      svmp::FE::systems::FieldSpec{
          .name = "phi",
          .space = q3_space,
          .components = 1});
  ASSERT_NO_THROW(system.setup());

  const auto q3_nodes =
      svmp::FE::basis::ReferenceNodeLayout::get_lagrange_node_coords(
          svmp::FE::ElementType::Quad4, /*order=*/3);
  const auto cell_dofs = system.fieldDofHandler(phi).getCellDofs(0);
  ASSERT_EQ(cell_dofs.size(), q3_nodes.size());
  const auto offset = system.fieldDofOffset(phi);
  std::vector<svmp::FE::Real> solution(
      static_cast<std::size_t>(system.dofHandler().getNumDofs()),
      svmp::FE::Real{0.0});
  std::vector<svmp::FE::Real> coefficients;
  coefficients.reserve(q3_nodes.size());
  for (std::size_t i = 0; i < q3_nodes.size(); ++i) {
    const auto value = polynomial(q3_nodes[i][0]);
    ASSERT_GT(value, svmp::FE::Real{0.0});
    coefficients.push_back(value);
    solution[static_cast<std::size_t>(offset + cell_dofs[i])] = value;
  }

  svmp::FE::spaces::FunctionSpace::Value center{};
  const auto center_value = q3_space->evaluate_scalar(center, coefficients);
  ASSERT_GT(center_value, svmp::FE::Real{0.0});
  ASSERT_NEAR(center_value, polynomial(svmp::FE::Real{0.0}), 1.0e-13);
  ASSERT_LT(polynomial(svmp::FE::Real{0.60}), svmp::FE::Real{0.0});

  svmp::FE::interfaces::CutInterfaceDomainRequest interface_request;
  interface_request.source =
      svmp::FE::interfaces::LevelSetInterfaceSource::fromField(
          phi, system.dofLayoutRevision(), source_value_revision);
  interface_request.interface_marker = marker;
  interface_request.interface_quadrature_order = 3;
  interface_request.volume_quadrature_order = 3;
  interface_request.achieved_interface_quadrature_order = 3;
  interface_request.implicit_geometry_mode = "HighOrderImplicit";
  interface_request.implicit_quadrature_backend = "SayeHyperrectangle";
  interface_request.implicit_fallback_policy = "Fail";
  interface_request.implicit_fallback_status = "None";
  interface_request.mesh_geometry_revision =
      system.meshAccess().geometryRevision();
  interface_request.mesh_topology_revision =
      system.meshAccess().topologyRevision();
  interface_request.ownership_revision =
      system.meshAccess().ownershipRevision();
  interface_request.quadrature_policy_key = 73u;

  svmp::FE::interfaces::LevelSetInterfaceDomain interface_domain(
      interface_request);
  svmp::FE::interfaces::CutInterfaceFragment fragment;
  fragment.parent_cell = 0;
  fragment.parent_cell_global_id = 0;
  fragment.owner_rank = 0;
  fragment.kind =
      svmp::FE::interfaces::CutInterfaceFragmentKind::Segment;
  fragment.measure = svmp::FE::Real{2.0};
  fragment.normal = {{svmp::FE::Real{-1.0}, 0.0, 0.0}};
  fragment.min_gradient_norm = second_root - first_root;
  fragment.root_polished = true;
  fragment.implicit_quadrature_backend = "SayeHyperrectangle";
  fragment.implicit_fallback_status = "None";
  fragment.vertices = {
      svmp::FE::interfaces::CutInterfaceVertex{
          .point = {{first_root, -1.0, 0.0}},
          .parent_coordinate = {{first_root, -1.0, 0.0}}},
      svmp::FE::interfaces::CutInterfaceVertex{
          .point = {{first_root, 1.0, 0.0}},
          .parent_coordinate = {{first_root, 1.0, 0.0}}},
  };
  interface_domain.addFragment(std::move(fragment));

  constexpr svmp::FE::Real parent_measure = svmp::FE::Real{4.0};
  const auto negative_measure =
      svmp::FE::Real{2.0} * (second_root - first_root);
  const auto positive_measure = parent_measure - negative_measure;
  const auto add_volume_region =
      [&](svmp::FE::geometry::CutIntegrationSide side,
          svmp::FE::Real measure,
          svmp::FE::Real centroid_x) {
    svmp::FE::interfaces::CutInterfaceVolumeRegion region;
    region.parent_cell = 0;
    region.parent_cell_global_id = 0;
    region.owner_rank = 0;
    region.side = side;
    region.centroid = {{centroid_x, 0.0, 0.0}};
    region.normal =
        side == svmp::FE::geometry::CutIntegrationSide::Negative
            ? std::array<svmp::FE::Real, 3>{{-1.0, 0.0, 0.0}}
            : std::array<svmp::FE::Real, 3>{{1.0, 0.0, 0.0}};
    region.parent_measure = parent_measure;
    region.measure = measure;
    region.volume_fraction = measure / parent_measure;
    region.min_level_set_value =
        side == svmp::FE::geometry::CutIntegrationSide::Negative
            ? polynomial(centroid_x)
            : svmp::FE::Real{0.0};
    region.max_level_set_value =
        side == svmp::FE::geometry::CutIntegrationSide::Negative
            ? svmp::FE::Real{0.0}
            : polynomial(svmp::FE::Real{-1.0});
    region.topology_id =
        side == svmp::FE::geometry::CutIntegrationSide::Negative
            ? "q3-negative-pocket"
            : "q3-positive-complement";
    region.achieved_quadrature_order = 0;
    interface_domain.addVolumeRegion(std::move(region));
  };
  add_volume_region(
      svmp::FE::geometry::CutIntegrationSide::Negative,
      negative_measure,
      svmp::FE::Real{0.5} * (first_root + second_root));
  add_volume_region(
      svmp::FE::geometry::CutIntegrationSide::Positive,
      positive_measure,
      (first_root * first_root - second_root * second_root) /
          positive_measure);

  const auto authoritative_interface_rules =
      interface_domain.interfaceQuadratureRules();
  ASSERT_FALSE(authoritative_interface_rules.empty());
  svmp::FE::Real maximum_root_residual = 0.0;
  for (const auto& rule : authoritative_interface_rules) {
    ASSERT_EQ(rule.provenance.parent_entity, 0);
    ASSERT_EQ(rule.provenance.source_value_revision,
              source_value_revision);
    ASSERT_GT(rule.provenance.cut_topology_revision, 0u);
    for (const auto& point : rule.points) {
      const auto xi = rule.frame ==
                              svmp::FE::geometry::CutGeometryFrame::Reference
                          ? point.point
                          : point.parent_coordinate;
      maximum_root_residual = std::max(
          maximum_root_residual, std::abs(polynomial(xi[0])));
    }
  }
  ASSERT_LT(maximum_root_residual, svmp::FE::Real{1.0e-8});

  svmp::FE::systems::SystemStateView state;
  state.u = solution;
  EXPECT_THROW(
      (void)application::core::
          collectLevelSetCurvatureHighOrderSupplementalSamples(
              system, state, phi, marker, source_value_revision),
      std::invalid_argument);

  svmp::FE::interfaces::FreeSurfaceGeometrySnapshotPolicy snapshot_policy;
  snapshot_policy.require_complete_exterior_boundary_partition = false;
  svmp::FE::interfaces::FreeSurfaceGeometryScalarEvaluator scalar;
  scalar.value = [q3_space, coefficients](
                     svmp::FE::GlobalIndex,
                     const std::array<svmp::FE::Real, 3>& xi,
                     const svmp::FE::geometry::CutQuadratureProvenance&) {
    svmp::FE::spaces::FunctionSpace::Value reference{};
    reference[0] = xi[0];
    reference[1] = xi[1];
    reference[2] = xi[2];
    return q3_space->evaluate_scalar(reference, coefficients);
  };
  scalar.reference_gradient = [q3_space, coefficients](
                                  svmp::FE::GlobalIndex,
                                  const std::array<svmp::FE::Real, 3>& xi,
                                  const svmp::FE::geometry::
                                      CutQuadratureProvenance&) {
    svmp::FE::spaces::FunctionSpace::Value reference{};
    reference[0] = xi[0];
    reference[1] = xi[1];
    reference[2] = xi[2];
    const auto gradient =
        q3_space->evaluate_gradient(reference, coefficients);
    return std::array<svmp::FE::Real, 3>{
        gradient[0], gradient[1], gradient[2]};
  };
  const auto snapshot =
      svmp::FE::interfaces::buildFreeSurfaceGeometrySnapshot(
          std::move(interface_domain),
          {},
          {},
          system.meshAccess(),
          snapshot_policy,
          std::move(scalar),
          "q3-interior-root-curvature");
  ASSERT_TRUE(snapshot);
  ASSERT_GT(snapshot->revision().snapshot_revision_key, 0u);
  ASSERT_EQ(snapshot->revision().source_value_revision,
            source_value_revision);

  auto cut_context =
      std::make_shared<svmp::FE::assembly::CutIntegrationContext>();
  cut_context->addFreeSurfaceGeometrySnapshot(
      snapshot, svmp::FE::geometry::CutIntegrationSide::Negative);
  system.setCutIntegrationContext(cut_context);

  EXPECT_THROW(
      (void)application::core::
          collectLevelSetCurvatureHighOrderSupplementalSamples(
              system,
              state,
              phi,
              marker,
              source_value_revision + 1u),
      std::invalid_argument);
  const auto samples =
      application::core::
          collectLevelSetCurvatureHighOrderSupplementalSamples(
              system, state, phi, marker, source_value_revision);
  ASSERT_EQ(samples.size(), 1u);
  const auto& sample = samples.front();
  EXPECT_EQ(sample.parent_cell, 0);
  EXPECT_NEAR(sample.coordinate[0], svmp::FE::Real{0.5}, 1.0e-13);
  EXPECT_NEAR(sample.coordinate[1], svmp::FE::Real{0.5}, 1.0e-13);
  EXPECT_NEAR(sample.coordinate[2], svmp::FE::Real{0.0}, 1.0e-13);
  EXPECT_NEAR(sample.value, center_value, 1.0e-13);
  EXPECT_EQ(sample.free_surface_snapshot_revision_key,
            snapshot->revision().snapshot_revision_key);
  EXPECT_EQ(sample.source_value_revision, source_value_revision);
  EXPECT_GT(sample.cut_topology_revision, 0u);

  auto zero_topology_samples = samples;
  zero_topology_samples.front().cut_topology_revision = 0u;
  const std::vector<svmp::FE::Real> vertex_phi{
      polynomial(svmp::FE::Real{-1.0}),
      polynomial(svmp::FE::Real{1.0}),
      polynomial(svmp::FE::Real{1.0}),
      polynomial(svmp::FE::Real{-1.0}),
  };
  svmp::FE::level_set::LevelSetCurvatureProjectionOptions projection_options;
  std::vector<svmp::FE::Real> curvature;
  EXPECT_THROW(
      (void)svmp::FE::level_set::projectLevelSetMeanCurvatureToVertices(
          system.meshAccess(),
          vertex_phi,
          zero_topology_samples,
          projection_options,
          curvature),
      std::invalid_argument);

  RecordProperty("q3_curvature_paired_sample_count",
                 std::to_string(samples.size()));
  RecordProperty("q3_curvature_authoritative_interface_validation_count", "1");
  RecordProperty("q3_curvature_missing_snapshot_rejections", "1");
  RecordProperty("q3_curvature_stale_state_rejections", "1");
  RecordProperty("q3_curvature_zero_topology_rejections", "1");
  RecordProperty("q3_curvature_authoritative_root_residual",
                 ::testing::PrintToString(maximum_root_residual));
}
