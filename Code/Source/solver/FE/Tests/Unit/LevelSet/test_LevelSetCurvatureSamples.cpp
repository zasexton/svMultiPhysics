#include <gtest/gtest.h>

#include "LevelSet/LevelSetCurvatureSamples.h"

#include "FE/Assembly/CutIntegrationContext.h"
#include "FE/Assembly/MeshAccess.h"
#include "FE/Basis/NodeOrderingConventions.h"
#include "FE/Interfaces/FreeSurfaceGeometrySnapshot.h"
#include "FE/Interfaces/LevelSetInterfaceDomain.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Sample = svmp::FE::level_set::LevelSetCurvatureProjectionSample;
using Filter = svmp::FE::level_set::LevelSetCurvatureSampleDuplicateFilter;

/// The linear scan that collectLevelSetCurvatureSupplementalSamples used
/// before the hashed filter, verbatim apart from returning the index of the
/// first duplicate instead of acting on it.
std::optional<std::size_t> appendWithLinearScan(std::vector<Sample>& samples,
                                                const Sample& candidate)
{
  constexpr svmp::FE::Real duplicate_tol2 = svmp::FE::Real{1.0e-24};
  constexpr svmp::FE::Real duplicate_value_tol = svmp::FE::Real{1.0e-12};
  for (std::size_t index = 0u; index < samples.size(); ++index) {
    const auto& existing = samples[index];
    if (existing.parent_cell != candidate.parent_cell ||
        existing.generated_interface_geometry !=
            candidate.generated_interface_geometry) {
      continue;
    }
    const auto dx = existing.coordinate[0] - candidate.coordinate[0];
    const auto dy = existing.coordinate[1] - candidate.coordinate[1];
    const auto dz = existing.coordinate[2] - candidate.coordinate[2];
    const auto dist2 = dx * dx + dy * dy + dz * dz;
    if (dist2 <= duplicate_tol2 &&
        std::abs(existing.value - candidate.value) <= duplicate_value_tol) {
      return index;
    }
  }
  samples.push_back(candidate);
  return std::nullopt;
}

bool bitwiseEqual(const Sample& a, const Sample& b)
{
  return a.parent_cell == b.parent_cell &&
         std::memcmp(a.coordinate.data(), b.coordinate.data(),
                     sizeof(a.coordinate)) == 0 &&
         std::memcmp(&a.value, &b.value, sizeof(a.value)) == 0 &&
         a.free_surface_snapshot_revision_key ==
             b.free_surface_snapshot_revision_key &&
         a.source_value_revision == b.source_value_revision &&
         a.cut_topology_revision == b.cut_topology_revision &&
         a.generated_interface_geometry == b.generated_interface_geometry;
}

/// Random candidates with many exact repeats and perturbations on both sides
/// of the distance and value tolerances, spread over few cells so that
/// buckets hold many samples.
std::vector<Sample> randomCandidates(std::uint32_t seed, std::size_t count)
{
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> cell_dist(0, 11);
  std::uniform_int_distribution<int> flag_dist(0, 1);
  std::uniform_real_distribution<double> coord_dist(-1.0, 1.0);
  std::uniform_int_distribution<int> choice(0, 9);
  std::uniform_int_distribution<int> revision_dist(1, 3);
  // Per-coordinate offsets: 3 * (5e-13)^2 = 7.5e-25 is inside the squared
  // distance tolerance, 3 * (7e-13)^2 = 1.47e-24 is outside it.
  const double coordinate_offsets[] = {0.0, 1.0e-14, 5.0e-13, 5.8e-13,
                                       7.0e-13, 1.0e-12, 1.0e-9};
  const double value_offsets[] = {0.0, 1.0e-13, 9.0e-13, 1.0e-12,
                                  1.1e-12, 1.0e-9};
  std::uniform_int_distribution<std::size_t> coordinate_offset_dist(
      0u, std::size(coordinate_offsets) - 1u);
  std::uniform_int_distribution<std::size_t> value_offset_dist(
      0u, std::size(value_offsets) - 1u);

  std::vector<Sample> bases;
  std::vector<Sample> candidates;
  candidates.reserve(count);
  for (std::size_t i = 0u; i < count; ++i) {
    Sample sample{};
    const int kind = choice(rng);
    if (bases.empty() || kind < 3) {
      sample.parent_cell = static_cast<svmp::FE::MeshIndex>(cell_dist(rng));
      sample.coordinate = {coord_dist(rng), coord_dist(rng),
                           kind == 0 ? 0.0 : coord_dist(rng)};
      sample.value = kind == 1 ? 0.0 : coord_dist(rng);
      sample.generated_interface_geometry = flag_dist(rng) != 0;
      bases.push_back(sample);
    } else {
      std::uniform_int_distribution<std::size_t> base_dist(0u,
                                                           bases.size() - 1u);
      sample = bases[base_dist(rng)];
      if (kind >= 5) {
        for (auto& x : sample.coordinate) {
          const double offset = coordinate_offsets[coordinate_offset_dist(rng)];
          x += flag_dist(rng) != 0 ? offset : -offset;
        }
        const double offset = value_offsets[value_offset_dist(rng)];
        sample.value += flag_dist(rng) != 0 ? offset : -offset;
      }
      if (kind == 9) {
        // Same point in another cell or of the other kind is never a
        // duplicate.
        if (flag_dist(rng) != 0) {
          sample.parent_cell = static_cast<svmp::FE::MeshIndex>(cell_dist(rng));
        } else {
          sample.generated_interface_geometry =
              !sample.generated_interface_geometry;
        }
      }
    }
    sample.free_surface_snapshot_revision_key =
        static_cast<std::uint64_t>(revision_dist(rng));
    sample.source_value_revision = static_cast<std::uint64_t>(revision_dist(rng));
    sample.cut_topology_revision = static_cast<std::uint64_t>(revision_dist(rng));
    candidates.push_back(sample);
  }
  return candidates;
}

} // namespace

TEST(LevelSetCurvatureSampleDuplicateFilter,
     MatchesLinearScanOnRandomizedSamplesWithDuplicates)
{
  std::size_t total_duplicates = 0u;
  std::size_t total_appended = 0u;
  for (std::uint32_t seed = 1u; seed <= 40u; ++seed) {
    const auto candidates = randomCandidates(seed, 3000u);
    std::vector<Sample> reference;
    std::vector<Sample> hashed;
    Filter filter;
    for (std::size_t i = 0u; i < candidates.size(); ++i) {
      const auto expected = appendWithLinearScan(reference, candidates[i]);
      const auto* existing = filter.appendUnlessDuplicate(hashed, candidates[i]);
      ASSERT_EQ(expected.has_value(), existing != nullptr)
          << "seed " << seed << " candidate " << i;
      if (expected.has_value()) {
        // The first duplicate decides the snapshot-consistency check of the
        // caller, so it must be the same earlier sample.
        ASSERT_EQ(*expected, static_cast<std::size_t>(existing - hashed.data()))
            << "seed " << seed << " candidate " << i;
        ++total_duplicates;
      } else {
        ++total_appended;
      }
      ASSERT_EQ(reference.size(), hashed.size());
    }
    for (std::size_t i = 0u; i < reference.size(); ++i) {
      ASSERT_TRUE(bitwiseEqual(reference[i], hashed[i]))
          << "seed " << seed << " sample " << i;
    }
  }
  // The generator must exercise both outcomes heavily.
  EXPECT_GT(total_duplicates, 20000u);
  EXPECT_GT(total_appended, 20000u);
}

TEST(LevelSetCurvatureSampleDuplicateFilter, AppliesTheToleranceCriterion)
{
  Sample base{};
  base.parent_cell = 7;
  base.coordinate = {0.25, -0.5, 0.125};
  base.value = 0.75;

  auto nearby = base;
  nearby.coordinate[0] += 5.0e-13;
  nearby.value += 5.0e-13;
  EXPECT_TRUE(Filter::isDuplicate(base, nearby));

  auto far = base;
  far.coordinate[0] += 2.0e-12;
  EXPECT_FALSE(Filter::isDuplicate(base, far));

  auto other_value = base;
  other_value.value += 2.0e-12;
  EXPECT_FALSE(Filter::isDuplicate(base, other_value));

  auto other_cell = base;
  other_cell.parent_cell = 8;
  EXPECT_FALSE(Filter::isDuplicate(base, other_cell));

  auto other_kind = base;
  other_kind.generated_interface_geometry = true;
  EXPECT_FALSE(Filter::isDuplicate(base, other_kind));

  // Revision fields do not take part in the criterion.
  auto other_revision = base;
  other_revision.free_surface_snapshot_revision_key = 99u;
  other_revision.source_value_revision = 98u;
  other_revision.cut_topology_revision = 97u;
  EXPECT_TRUE(Filter::isDuplicate(base, other_revision));

  std::vector<Sample> samples;
  Filter filter;
  EXPECT_EQ(filter.appendUnlessDuplicate(samples, base), nullptr);
  EXPECT_EQ(filter.appendUnlessDuplicate(samples, far), nullptr);
  EXPECT_EQ(filter.appendUnlessDuplicate(samples, other_kind), nullptr);
  EXPECT_EQ(filter.appendUnlessDuplicate(samples, nearby), &samples[0]);
  EXPECT_EQ(filter.appendUnlessDuplicate(samples, other_revision),
            &samples[0]);
  ASSERT_EQ(samples.size(), 3u);
  EXPECT_TRUE(bitwiseEqual(samples[0], base));
  EXPECT_TRUE(bitwiseEqual(samples[1], far));
  EXPECT_TRUE(bitwiseEqual(samples[2], other_kind));
}

// The collection tests build FESystem instances on Mesh-backed cells.
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

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

std::shared_ptr<svmp::Mesh> buildTwoQuadStripMesh()
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
  const std::vector<svmp::index_t> cell2vertex = {0, 1, 4, 3,
                                                  1, 2, 5, 4};

  svmp::CellShape shape{};
  shape.family = svmp::CellFamily::Quad;
  shape.num_corners = 4;
  shape.order = 1;
  base->build_from_arrays(
      /*spatial_dim=*/2,
      x_ref,
      cell2vertex_offsets,
      cell2vertex,
      {shape, shape});
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

} // namespace

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
      svmp::FE::level_set::collectLevelSetCurvatureCutVolumeSupplementalSamples(
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
      svmp::FE::level_set::collectLevelSetCurvatureCutVolumeSupplementalSamples(
          system,
          state,
          phi,
          marker,
          svmp::FE::geometry::CutIntegrationSide::Positive,
          /*evaluated_state_source_revision=*/0u);
  EXPECT_TRUE(positive_samples.empty());
}

TEST(LevelSetCurvatureSamples,
     CutVolumeSamplesKeepFirstDuplicatePerCellInRuleOrder)
{
  auto mesh = buildTwoQuadStripMesh();
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
  const auto n_dofs =
      static_cast<std::size_t>(system.fieldDofHandler(phi).getNumDofs());
  ASSERT_EQ(n_dofs, 6u);
  std::vector<svmp::FE::Real> prescribed_coefficients(n_dofs);
  for (std::size_t i = 0; i < n_dofs; ++i) {
    prescribed_coefficients[i] =
        svmp::FE::Real{0.5} + svmp::FE::Real{0.25} * static_cast<svmp::FE::Real>(i);
  }
  system.setPrescribedFieldCoefficients(phi, prescribed_coefficients);

  constexpr int marker = 43;
  constexpr auto side = svmp::FE::geometry::CutIntegrationSide::Negative;
  auto cut_context =
      std::make_shared<svmp::FE::assembly::CutIntegrationContext>();
  const auto add_rule =
      [&](svmp::FE::MeshIndex cell,
          const std::vector<std::array<svmp::FE::Real, 3>>& points) {
        svmp::FE::geometry::CutQuadratureRule rule;
        rule.kind = svmp::FE::geometry::CutQuadratureKind::Volume;
        rule.side = side;
        rule.frame = svmp::FE::geometry::CutGeometryFrame::Reference;
        rule.provenance.parent_entity = cell;
        rule.provenance.marker = marker;
        rule.measure = svmp::FE::Real{0.5};
        rule.parent_measure = svmp::FE::Real{1.0};
        rule.volume_fraction = svmp::FE::Real{0.5};
        rule.full_cell_equivalent = false;
        for (const auto& point : points) {
          svmp::FE::geometry::CutQuadraturePoint qp;
          qp.point = point;
          qp.parent_coordinate = point;
          qp.weight = svmp::FE::Real{0.125};
          rule.points.push_back(qp);
        }
        svmp::FE::assembly::CutCellAssemblyMetadata metadata;
        metadata.parent_entity = cell;
        metadata.side = side;
        metadata.volume_fraction = rule.volume_fraction;
        cut_context->addGeneratedVolumeRule(marker, metadata, rule);
      };
  using P = std::array<svmp::FE::Real, 3>;
  const P a{{0.25, 0.25, 0.0}};
  const P b{{0.5, -0.25, 0.0}};
  const P edge_right{{1.0, 0.0, 0.0}};
  const P edge_left{{-1.0, 0.0, 0.0}};
  // Cell 0, first rule: three distinct points.
  add_rule(0, {a, b, edge_right});
  // Cell 0, second rule: an exact repeat, a repeat within the coordinate
  // tolerance, a new point, and a point just outside the tolerance.
  add_rule(0, {b,
               P{{0.25 + 1.0e-13, 0.25, 0.0}},
               P{{-0.5, 0.5, 0.0}},
               P{{0.25 + 1.0e-9, 0.25, 0.0}}});
  // Cell 1: the physical point shared with cell 0's right edge is a distinct
  // sample (other parent cell); its exact repeat in cell 1 is not.
  add_rule(1, {edge_left, edge_left});
  system.setCutIntegrationContext(cut_context);

  const svmp::FE::systems::SystemStateView state;
  const auto samples =
      svmp::FE::level_set::collectLevelSetCurvatureCutVolumeSupplementalSamples(
          system,
          state,
          phi,
          marker,
          side,
          /*evaluated_state_source_revision=*/0u);

  // Reference: the scan over all earlier samples, in rule and point order.
  const svmp::FE::assembly::MeshAccess access(*mesh);
  std::vector<svmp::FE::level_set::LevelSetCurvatureProjectionSample> expected;
  for (const auto* rule :
       cut_context->generatedVolumeRulesForMarkerAndSide(marker, side)) {
    const auto cell = rule->provenance.parent_entity;
    const auto cell_dofs = system.fieldDofHandler(phi).getCellDofs(cell);
    std::vector<svmp::FE::Real> coefficients;
    for (const auto dof : cell_dofs) {
      coefficients.push_back(
          prescribed_coefficients[static_cast<std::size_t>(dof)]);
    }
    for (const auto& point : rule->points) {
      const auto physical =
          svmp::FE::level_set::mapLevelSetCurvatureReferenceSampleToPhysical(
              access, cell, point.parent_coordinate);
      ASSERT_TRUE(physical.has_value());
      svmp::FE::spaces::FunctionSpace::Value xi{};
      xi[0] = point.parent_coordinate[0];
      xi[1] = point.parent_coordinate[1];
      xi[2] = point.parent_coordinate[2];
      const auto value = space->evaluate_scalar(xi, coefficients);
      bool duplicate = false;
      for (const auto& existing : expected) {
        if (existing.parent_cell != cell) {
          continue;
        }
        const auto dx = existing.coordinate[0] - (*physical)[0];
        const auto dy = existing.coordinate[1] - (*physical)[1];
        const auto dz = existing.coordinate[2] - (*physical)[2];
        if (dx * dx + dy * dy + dz * dz <= svmp::FE::Real{1.0e-24} &&
            std::abs(existing.value - value) <= svmp::FE::Real{1.0e-12}) {
          duplicate = true;
          break;
        }
      }
      if (!duplicate) {
        expected.push_back(
            svmp::FE::level_set::LevelSetCurvatureProjectionSample{
                .parent_cell = cell,
                .coordinate = *physical,
                .value = value});
      }
    }
  }

  ASSERT_EQ(expected.size(), 6u);
  ASSERT_EQ(samples.size(), expected.size());
  for (std::size_t i = 0; i < samples.size(); ++i) {
    EXPECT_EQ(samples[i].parent_cell, expected[i].parent_cell) << i;
    EXPECT_EQ(samples[i].coordinate, expected[i].coordinate) << i;
    EXPECT_EQ(samples[i].value, expected[i].value) << i;
    EXPECT_FALSE(samples[i].generated_interface_geometry) << i;
  }
  // The shared edge point appears once per parent cell.
  EXPECT_EQ(std::count_if(samples.begin(), samples.end(),
                          [&](const auto& sample) {
                            return sample.coordinate ==
                                   expected[2].coordinate;
                          }),
            2);
}

TEST(LevelSetCurvatureSamples,
     MapsWarpedHighOrderValueAndCoordinateAtIdenticalReferencePoint)
{
  const auto mesh = buildWarpedBiquadraticQuadMesh();
  const svmp::FE::assembly::MeshAccess access(*mesh);
  const std::array<svmp::FE::Real, 3> xi{{0.0, 0.0, 0.0}};

  const auto physical =
      svmp::FE::level_set::mapLevelSetCurvatureReferenceSampleToPhysical(
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
      (void)svmp::FE::level_set::
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
      (void)svmp::FE::level_set::
          collectLevelSetCurvatureHighOrderSupplementalSamples(
              system,
              state,
              phi,
              marker,
              source_value_revision + 1u),
      std::invalid_argument);
  const auto samples =
      svmp::FE::level_set::
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

#endif // SVMP_FE_WITH_MESH
