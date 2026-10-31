/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

// Reuse of full-cell volume records between successive free-surface geometry
// snapshot builds (FreeSurfaceGeometrySnapshotReuseCache).  A build that
// reuses the previous build's records must give the snapshot a build without
// the cache gives, bit for bit, for every storage policy and level-set
// motion, including motions that change the cut topology.

#include "Assembly/CutIntegrationContext.h"
#include "Core/DeterministicParallel.h"
#include "Dofs/DofHandler.h"
#include "Geometry/CutQuadratureMapping.h"
#include "Dofs/EntityDofMap.h"
#include "Interfaces/FreeSurfaceGeometrySnapshot.h"
#include "LevelSet/LevelSetCellEvaluator.h"
#include "LevelSet/LevelSetInterfaceLifecycle.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"

#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace FE = svmp::FE;
namespace interfaces = svmp::FE::interfaces;
namespace geometry = svmp::FE::geometry;
namespace level_set = svmp::FE::level_set;

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

using Point = std::array<FE::Real, 3>;

// Box [0,1]^3 split into six Kuhn tetrahedra per cube, slightly sheared so
// that the cell Jacobians differ between cells.
std::shared_ptr<svmp::Mesh> buildShearedTetraMesh(int n)
{
    auto base = std::make_shared<svmp::MeshBase>();
    const auto extent = static_cast<svmp::index_t>(n + 1);
    std::vector<svmp::real_t> x_ref;
    for (int k = 0; k <= n; ++k) {
        for (int j = 0; j <= n; ++j) {
            for (int i = 0; i <= n; ++i) {
                const auto x = static_cast<svmp::real_t>(i) / n;
                const auto y = static_cast<svmp::real_t>(j) / n;
                const auto z = static_cast<svmp::real_t>(k) / n;
                x_ref.push_back(x + svmp::real_t{0.07} * y * z);
                x_ref.push_back(y + svmp::real_t{0.05} * x * x);
                x_ref.push_back(z);
            }
        }
    }
    const auto vid = [&](int i, int j, int k) {
        return static_cast<svmp::index_t>((k * extent + j) * extent + i);
    };
    constexpr std::array<std::array<int, 3>, 6> paths{{
        {{0, 1, 2}}, {{0, 2, 1}}, {{1, 0, 2}},
        {{1, 2, 0}}, {{2, 0, 1}}, {{2, 1, 0}},
    }};
    std::vector<svmp::offset_t> offsets{0};
    std::vector<svmp::index_t> cell2vertex;
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                for (const auto& path : paths) {
                    std::array<int, 3> corner{{i, j, k}};
                    std::array<std::array<int, 3>, 4> tet{};
                    tet[0] = corner;
                    for (std::size_t step = 0; step < 3u; ++step) {
                        ++corner[static_cast<std::size_t>(path[step])];
                        tet[step + 1u] = corner;
                    }
                    std::array<std::array<int, 3>, 3> e{};
                    for (std::size_t q = 0; q < 3u; ++q) {
                        for (std::size_t c = 0; c < 3u; ++c) {
                            e[q][c] = tet[q + 1u][c] - tet[0][c];
                        }
                    }
                    const int det =
                        e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1]) -
                        e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0]) +
                        e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0]);
                    if (det < 0) {
                        std::swap(tet[1], tet[2]);
                    }
                    for (const auto& vertex : tet) {
                        cell2vertex.push_back(
                            vid(vertex[0], vertex[1], vertex[2]));
                    }
                    offsets.push_back(
                        static_cast<svmp::offset_t>(cell2vertex.size()));
                }
            }
        }
    }
    svmp::CellShape shape{};
    shape.family = svmp::CellFamily::Tetra;
    shape.num_corners = 4;
    shape.order = 1;
    std::vector<svmp::CellShape> shapes(offsets.size() - 1u, shape);
    base->build_from_arrays(/*spatial_dim=*/3, x_ref, offsets, cell2vertex,
                            shapes);
    base->finalize();
    return svmp::create_mesh(std::move(base));
}

// Square [0,1]^2 split into two triangles per cell, slightly sheared.
std::shared_ptr<svmp::Mesh> buildShearedTriangleMesh(int n)
{
    auto base = std::make_shared<svmp::MeshBase>();
    const auto extent = static_cast<svmp::index_t>(n + 1);
    std::vector<svmp::real_t> x_ref;
    for (int j = 0; j <= n; ++j) {
        for (int i = 0; i <= n; ++i) {
            const auto x = static_cast<svmp::real_t>(i) / n;
            const auto y = static_cast<svmp::real_t>(j) / n;
            x_ref.push_back(x + svmp::real_t{0.06} * y * y);
            x_ref.push_back(y + svmp::real_t{0.04} * x);
        }
    }
    const auto vid = [&](int i, int j) {
        return static_cast<svmp::index_t>(j * extent + i);
    };
    std::vector<svmp::offset_t> offsets{0};
    std::vector<svmp::index_t> cell2vertex;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            for (const auto& triangle :
                 {std::array<svmp::index_t, 3>{vid(i, j), vid(i + 1, j),
                                               vid(i + 1, j + 1)},
                  std::array<svmp::index_t, 3>{vid(i, j), vid(i + 1, j + 1),
                                               vid(i, j + 1)}}) {
                cell2vertex.insert(cell2vertex.end(), triangle.begin(),
                                   triangle.end());
                offsets.push_back(
                    static_cast<svmp::offset_t>(cell2vertex.size()));
            }
        }
    }
    svmp::CellShape shape{};
    shape.family = svmp::CellFamily::Triangle;
    shape.num_corners = 3;
    shape.order = 1;
    std::vector<svmp::CellShape> shapes(offsets.size() - 1u, shape);
    base->build_from_arrays(/*spatial_dim=*/2, x_ref, offsets, cell2vertex,
                            shapes);
    base->finalize();
    return svmp::create_mesh(std::move(base));
}

// One level-set field on a sheared mesh and one generated-interface
// lifecycle, so that successive builds carry new source value revisions
// exactly as successive outer passes do.  The assembly thread count of the
// system sets the geometry threads.
struct ReuseFixture {
    std::shared_ptr<svmp::Mesh> mesh;
    FE::systems::FESystem system;
    FE::FieldId phi{FE::INVALID_FIELD_ID};
    int dimension{3};
    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle{};
    std::vector<FE::Real> solution{};

    explicit ReuseFixture(int dim, int assembly_threads = 1)
        : mesh(dim == 3 ? buildShearedTetraMesh(6)
                        : buildShearedTriangleMesh(12)),
          system(mesh), dimension(dim)
    {
        phi = system.addField(FE::systems::FieldSpec{
            .name = "phi",
            .space = std::make_shared<FE::spaces::H1Space>(
                dim == 3 ? FE::ElementType::Tetra4
                         : FE::ElementType::Triangle3,
                /*order=*/1),
            .components = 1,
        });
        FE::systems::SetupOptions setup_options{};
        setup_options.assembly_options.num_threads = assembly_threads;
        system.setup(setup_options);
        solution.assign(
            static_cast<std::size_t>(system.dofHandler().getNumDofs()),
            FE::Real{0.0});
    }

    // Signed distance to a sphere (circle in 2D) centred at `centre`.
    void setLevelSet(const Point& centre, FE::Real radius)
    {
        const auto& dofs = system.fieldDofHandler(phi);
        const auto* map = dofs.getEntityDofMap();
        const auto offset =
            static_cast<std::size_t>(system.fieldDofOffset(phi));
        for (FE::GlobalIndex v = 0; v < map->numVertices(); ++v) {
            const auto x = system.meshAccess().getNodeCoordinates(v);
            FE::Real r2{0.0};
            for (int d = 0; d < dimension; ++d) {
                const auto dx = x[static_cast<std::size_t>(d)] -
                                centre[static_cast<std::size_t>(d)];
                r2 += dx * dx;
            }
            solution[offset + static_cast<std::size_t>(
                                  map->getVertexDofs(v).front())] =
                std::sqrt(r2) - radius;
        }
    }

    struct Pair {
        std::shared_ptr<const interfaces::FreeSurfaceGeometrySnapshot> reused;
        std::shared_ptr<const interfaces::FreeSurfaceGeometrySnapshot> full;
    };

    // One generated-interface rebuild, snapshotted with and without the
    // reuse cache.
    Pair build(interfaces::FreeSurfaceGeometrySnapshotReuseCache& cache,
               const interfaces::FreeSurfaceGeometrySnapshotPolicy& policy)
    {
        level_set::LevelSetGeneratedInterfaceOptions options{};
        options.level_set_field_name = "phi";
        options.domain_id = "snapshot_reuse";
        options.requested_interface_marker = 4712;
        options.quadrature_order = 1;
        options.interface_quadrature_order = 2;
        options.volume_quadrature_order = 2;
        const auto generated =
            lifecycle.build(system, options, solution);
        EXPECT_TRUE(generated.success) << generated.diagnostic;
        auto evaluator = std::make_shared<level_set::LevelSetCellEvaluator>(
            level_set::makeLevelSetCellEvaluator(system, phi, solution));
        const auto make_scalar =
            [](std::shared_ptr<level_set::LevelSetCellEvaluator> cell) {
                interfaces::FreeSurfaceGeometryScalarEvaluator result;
                result.value = [cell](FE::GlobalIndex id,
                                      const Point& xi,
                                      const geometry::CutQuadratureProvenance&) {
                    return cell->evaluateLinearCorner(id, xi).value;
                };
                result.reference_gradient =
                    [cell](FE::GlobalIndex id,
                           const Point& xi,
                           const geometry::CutQuadratureProvenance&) {
                        return cell->evaluateLinearCorner(id, xi)
                            .reference_gradient;
                    };
                return result;
            };
        auto scalar = make_scalar(evaluator);
        scalar.make_concurrent_copy = [evaluator, make_scalar]() {
            return make_scalar(
                std::make_shared<level_set::LevelSetCellEvaluator>(*evaluator));
        };
        Pair pair;
        // The reference: no cache, one thread.
        pair.full = interfaces::buildFreeSurfaceGeometrySnapshot(
            generated.domain, {}, {}, system.meshAccess(), policy, scalar,
            "snapshot_reuse", {}, nullptr, 1);
        pair.reused = interfaces::buildFreeSurfaceGeometrySnapshot(
            generated.domain, {}, {}, system.meshAccess(), policy, scalar,
            "snapshot_reuse", {}, &cache,
            FE::geometryThreadCount(system.assemblyThreadCount()));
        return pair;
    }
};

[[nodiscard]] std::size_t fullCellVolumeRecordCount(
    const interfaces::FreeSurfaceGeometrySnapshot& snapshot)
{
    std::size_t count = 0u;
    for (const auto& record : snapshot.rules()) {
        const bool volume =
            record.role ==
                interfaces::FreeSurfaceGeometryRuleRole::NegativeVolume ||
            record.role ==
                interfaces::FreeSurfaceGeometryRuleRole::PositiveVolume;
        count += volume && record.reference_rule.full_cell_equivalent ? 1u
                                                                       : 0u;
    }
    return count;
}

void expectIdenticalSnapshots(
    const ReuseFixture::Pair& pair,
    const interfaces::FreeSurfaceGeometrySnapshotReuseCache& cache,
    bool expect_reuse,
    const std::string& label)
{
    ASSERT_NE(pair.full, nullptr) << label;
    ASSERT_NE(pair.reused, nullptr) << label;
    EXPECT_EQ(interfaces::compareFreeSurfaceGeometrySnapshots(*pair.reused,
                                                              *pair.full),
              "")
        << label;
    // The ledger bytes too (it holds only counts and reals, no padding).
    EXPECT_EQ(std::memcmp(&pair.reused->ledger(),
                          &pair.full->ledger(),
                          sizeof(interfaces::FreeSurfaceGeometryValidationLedger)),
              0)
        << label;
    EXPECT_EQ(pair.reused->revision().snapshot_revision_key,
              pair.full->revision().snapshot_revision_key)
        << label;
    const auto& statistics = cache.lastBuild();
    const auto full_cells = fullCellVolumeRecordCount(*pair.full);
    EXPECT_EQ(statistics.full_cell_records_reused +
                  statistics.full_cell_records_built,
              full_cells)
        << label;
    if (expect_reuse) {
        EXPECT_TRUE(statistics.context_matched) << label;
        EXPECT_GT(statistics.full_cell_records_reused, 0u) << label;
    } else {
        EXPECT_EQ(statistics.full_cell_records_reused, 0u) << label;
    }
}

// The level set moves by small and large steps, stays put once, and the
// motion changes which cells are cut; every build is bitwise identical to a
// build without the cache.
void runMotionSequence(int dimension,
                       const interfaces::FreeSurfaceGeometrySnapshotPolicy& policy,
                       const std::string& label)
{
    ReuseFixture fixture(dimension);
    interfaces::FreeSurfaceGeometrySnapshotReuseCache cache;
    const std::vector<std::pair<Point, FE::Real>> motion{
        {{{0.531, 0.487, 0.462}}, FE::Real{0.29}},
        {{{0.5312, 0.4871, 0.4619}}, FE::Real{0.29}},   // small move
        {{{0.5312, 0.4871, 0.4619}}, FE::Real{0.29}},   // no move
        {{{0.5333, 0.4850, 0.4640}}, FE::Real{0.2903}}, // small move
        {{{0.5600, 0.4500, 0.5000}}, FE::Real{0.31}},   // new cut cells
        {{{0.5601, 0.4500, 0.5001}}, FE::Real{0.31}},
    };
    // The previous snapshot stays alive, as the installed integration
    // context keeps it in the solver.
    ReuseFixture::Pair previous;
    for (std::size_t step = 0; step < motion.size(); ++step) {
        fixture.setLevelSet(motion[step].first, motion[step].second);
        auto pair = fixture.build(cache, policy);
        expectIdenticalSnapshots(pair, cache, /*expect_reuse=*/step > 0u,
                                 label + " step " + std::to_string(step));
        previous = std::move(pair);
    }
}

TEST(FreeSurfaceSnapshotReuse, TetrahedraWithAllPointsStored)
{
    interfaces::FreeSurfaceGeometrySnapshotPolicy policy;
    policy.require_complete_exterior_boundary_partition = false;
    runMotionSequence(3, policy, "3D all points");
}

TEST(FreeSurfaceSnapshotReuse, TetrahedraWithDryCellsClassificationOnly)
{
    interfaces::FreeSurfaceGeometrySnapshotPolicy policy;
    policy.require_complete_exterior_boundary_partition = false;
    policy.classification_only_full_cell_side =
        geometry::CutIntegrationSide::Positive;
    runMotionSequence(3, policy, "3D dry compact");
}

TEST(FreeSurfaceSnapshotReuse, TetrahedraWithBothSidesClassificationOnly)
{
    interfaces::FreeSurfaceGeometrySnapshotPolicy policy;
    policy.require_complete_exterior_boundary_partition = false;
    policy.classification_only_full_cell_side =
        geometry::CutIntegrationSide::Positive;
    policy.classification_only_full_cells_on_both_sides = true;
    runMotionSequence(3, policy, "3D both compact");
}

TEST(FreeSurfaceSnapshotReuse, TrianglesWithDryCellsClassificationOnly)
{
    interfaces::FreeSurfaceGeometrySnapshotPolicy policy;
    policy.require_complete_exterior_boundary_partition = false;
    policy.classification_only_full_cell_side =
        geometry::CutIntegrationSide::Positive;
    runMotionSequence(2, policy, "2D dry compact");
}

TEST(FreeSurfaceSnapshotReuse, TrianglesWithAllPointsStored)
{
    interfaces::FreeSurfaceGeometrySnapshotPolicy policy;
    policy.require_complete_exterior_boundary_partition = false;
    runMotionSequence(2, policy, "2D all points");
}

// A different policy is a different reuse context: nothing is reused, the
// result is still identical, and the next build under the new policy reuses
// again.
TEST(FreeSurfaceSnapshotReuse, PolicyChangeStartsAFreshContext)
{
    ReuseFixture fixture(3);
    interfaces::FreeSurfaceGeometrySnapshotReuseCache cache;
    interfaces::FreeSurfaceGeometrySnapshotPolicy all_points;
    all_points.require_complete_exterior_boundary_partition = false;
    auto compact = all_points;
    compact.classification_only_full_cell_side =
        geometry::CutIntegrationSide::Positive;

    fixture.setLevelSet({{0.531, 0.487, 0.462}}, FE::Real{0.29});
    auto first = fixture.build(cache, all_points);
    expectIdenticalSnapshots(first, cache, false, "first build");
    fixture.setLevelSet({{0.5311, 0.487, 0.462}}, FE::Real{0.29});
    auto second = fixture.build(cache, compact);
    expectIdenticalSnapshots(second, cache, false, "policy change");
    EXPECT_FALSE(cache.lastBuild().context_matched);
    fixture.setLevelSet({{0.5312, 0.487, 0.462}}, FE::Real{0.29});
    auto third = fixture.build(cache, compact);
    expectIdenticalSnapshots(third, cache, true, "same policy again");

    cache.clear();
    EXPECT_TRUE(cache.empty());
    fixture.setLevelSet({{0.5313, 0.487, 0.462}}, FE::Real{0.29});
    auto fourth = fixture.build(cache, compact);
    expectIdenticalSnapshots(fourth, cache, false, "after clear");
}

// The cache does not keep the previous snapshot alive: once it is released,
// the next build reuses nothing (and is still identical).
TEST(FreeSurfaceSnapshotReuse, ReleasedPreviousSnapshotIsNotReused)
{
    ReuseFixture fixture(3);
    interfaces::FreeSurfaceGeometrySnapshotReuseCache cache;
    interfaces::FreeSurfaceGeometrySnapshotPolicy policy;
    policy.require_complete_exterior_boundary_partition = false;
    fixture.setLevelSet({{0.531, 0.487, 0.462}}, FE::Real{0.29});
    {
        const auto first = fixture.build(cache, policy);
        expectIdenticalSnapshots(first, cache, false, "first build");
    }
    fixture.setLevelSet({{0.5311, 0.487, 0.462}}, FE::Real{0.29});
    const auto second = fixture.build(cache, policy);
    expectIdenticalSnapshots(second, cache, false, "after release");
    EXPECT_FALSE(cache.lastBuild().context_matched);
    fixture.setLevelSet({{0.5312, 0.487, 0.462}}, FE::Real{0.29});
    const auto third = fixture.build(cache, policy);
    expectIdenticalSnapshots(third, cache, true, "previous alive");
}

// The reused records carry the source identities of their own build.
TEST(FreeSurfaceSnapshotReuse, ReusedRecordsCarryTheNewSourceRevision)
{
    ReuseFixture fixture(3);
    interfaces::FreeSurfaceGeometrySnapshotReuseCache cache;
    interfaces::FreeSurfaceGeometrySnapshotPolicy policy;
    policy.require_complete_exterior_boundary_partition = false;
    policy.classification_only_full_cell_side =
        geometry::CutIntegrationSide::Positive;
    fixture.setLevelSet({{0.531, 0.487, 0.462}}, FE::Real{0.29});
    const auto first = fixture.build(cache, policy);
    fixture.setLevelSet({{0.5311, 0.4871, 0.462}}, FE::Real{0.29});
    const auto second = fixture.build(cache, policy);
    ASSERT_GT(cache.lastBuild().full_cell_records_reused, 0u);
    EXPECT_NE(first.reused->revision().source_value_revision,
              second.reused->revision().source_value_revision);
    for (const auto& record : second.reused->rules()) {
        EXPECT_EQ(record.reference_rule.provenance.source_value_revision,
                  second.reused->revision().source_value_revision);
        EXPECT_EQ(record.physical_rule.source_value_revision,
                  second.reused->revision().source_value_revision);
        EXPECT_EQ(record.reference_rule.provenance
                      .free_surface_snapshot_revision_key,
                  second.reused->revision().snapshot_revision_key);
    }
}

// The value-only linear-corner evaluation the snapshot validation uses is
// the value of the full evaluation, bit for bit.
TEST(FreeSurfaceSnapshotReuse, LinearCornerValueMatchesFullEvaluationBitwise)
{
    for (const int dimension : {2, 3}) {
        ReuseFixture fixture(dimension);
        fixture.setLevelSet({{0.531, 0.487, 0.462}}, FE::Real{0.29});
        const auto evaluator = level_set::makeLevelSetCellEvaluator(
            fixture.system, fixture.phi, fixture.solution);
        std::size_t compared = 0u;
        for (FE::GlobalIndex cell = 0;
             cell < fixture.system.meshAccess().numCells(); ++cell) {
            for (int i = 0; i <= 4; ++i) {
                for (int j = 0; j <= 4 - i; ++j) {
                    for (int k = 0; k <= (dimension == 3 ? 4 - i - j : 0);
                         ++k) {
                        const Point xi{{FE::Real{0.23} * i + FE::Real{0.01},
                                        FE::Real{0.19} * j + FE::Real{0.02},
                                        dimension == 3
                                            ? FE::Real{0.17} * k +
                                                  FE::Real{0.03}
                                            : FE::Real{0.0}}};
                        const auto full =
                            evaluator.evaluateLinearCorner(cell, xi).value;
                        const auto value =
                            evaluator.evaluateLinearCornerValue(cell, xi);
                        ASSERT_EQ(std::memcmp(&full, &value, sizeof(FE::Real)),
                                  0)
                            << "dimension " << dimension << " cell " << cell;
                        ++compared;
                    }
                }
            }
        }
        EXPECT_GT(compared, 0u);
    }
}

// The measure of a classification-only context rule taken from its source
// region equals the measure of the rematerialized rule, bit for bit, and
// the import without points gives the rule counts and diagnostics of an
// import that releases the points afterwards.
TEST(FreeSurfaceSnapshotReuse, ClassificationOnlyContextRuleMeasuresMatchMaterialized)
{
    for (const int dimension : {2, 3}) {
        ReuseFixture fixture(dimension);
        interfaces::FreeSurfaceGeometrySnapshotReuseCache cache;
        interfaces::FreeSurfaceGeometrySnapshotPolicy policy;
        policy.require_complete_exterior_boundary_partition = false;
        policy.classification_only_full_cell_side =
            geometry::CutIntegrationSide::Positive;
        fixture.setLevelSet({{0.531, 0.487, 0.462}}, FE::Real{0.29});
        const auto pair = fixture.build(cache, policy);
        FE::assembly::CutIntegrationContext context;
        context.addFreeSurfaceGeometrySnapshot(
            pair.full, std::nullopt, geometry::CutIntegrationSide::Positive);
        const auto& mesh = fixture.system.meshAccess();
        std::size_t released = 0u;
        for (std::size_t i = 0; i < context.volumeRules().size(); ++i) {
            const auto& rule = context.volumeRules()[i];
            const auto expected = geometry::physicalCutQuadratureMeasure(
                mesh, context.materializedVolumeRule(i));
            const auto measured = context.physicalVolumeRuleMeasure(mesh, rule);
            ASSERT_EQ(std::memcmp(&expected, &measured, sizeof(FE::Real)), 0)
                << "dimension " << dimension << " rule " << i;
            released += context.volumeRuleIsClassificationOnly(i) ? 1u : 0u;
        }
        EXPECT_GT(released, 0u) << "dimension " << dimension;
        EXPECT_EQ(context.classificationOnlyVolumeRuleCount(), released);
    }
}

[[nodiscard]] bool sameRealArray(const Point& a, const Point& b) noexcept
{
    return std::memcmp(a.data(), b.data(), sizeof(Point)) == 0;
}

void expectSameDomains(const interfaces::LevelSetInterfaceDomain& a,
                       const interfaces::LevelSetInterfaceDomain& b,
                       const std::string& label)
{
    ASSERT_EQ(a.fragments().size(), b.fragments().size()) << label;
    for (std::size_t i = 0; i < a.fragments().size(); ++i) {
        const auto& x = a.fragments()[i];
        const auto& y = b.fragments()[i];
        EXPECT_EQ(x.parent_cell, y.parent_cell) << label;
        EXPECT_EQ(x.local_fragment_index, y.local_fragment_index) << label;
        EXPECT_EQ(x.topology_id, y.topology_id) << label;
        EXPECT_EQ(x.construction_observation, y.construction_observation)
            << label;
        EXPECT_TRUE(sameRealArray(x.normal, y.normal)) << label;
        EXPECT_EQ(std::memcmp(&x.measure, &y.measure, sizeof(FE::Real)), 0)
            << label;
        ASSERT_EQ(x.vertices.size(), y.vertices.size()) << label;
        for (std::size_t v = 0; v < x.vertices.size(); ++v) {
            EXPECT_TRUE(sameRealArray(x.vertices[v].point, y.vertices[v].point))
                << label;
        }
        ASSERT_EQ(x.quadrature_points.size(), y.quadrature_points.size())
            << label;
        for (std::size_t q = 0; q < x.quadrature_points.size(); ++q) {
            EXPECT_TRUE(sameRealArray(x.quadrature_points[q].point,
                                      y.quadrature_points[q].point))
                << label;
            EXPECT_EQ(std::memcmp(&x.quadrature_points[q].weight,
                                  &y.quadrature_points[q].weight,
                                  sizeof(FE::Real)),
                      0)
                << label;
        }
    }
    ASSERT_EQ(a.volumeRegions().size(), b.volumeRegions().size()) << label;
    for (std::size_t i = 0; i < a.volumeRegions().size(); ++i) {
        const auto& x = a.volumeRegions()[i];
        const auto& y = b.volumeRegions()[i];
        EXPECT_EQ(x.parent_cell, y.parent_cell) << label;
        EXPECT_EQ(x.local_region_index, y.local_region_index) << label;
        EXPECT_EQ(x.side, y.side) << label;
        EXPECT_EQ(x.topology_id, y.topology_id) << label;
        EXPECT_EQ(x.full_cell_equivalent, y.full_cell_equivalent) << label;
        EXPECT_EQ(std::memcmp(&x.measure, &y.measure, sizeof(FE::Real)), 0)
            << label;
        ASSERT_EQ(x.quadrature_points.size(), y.quadrature_points.size())
            << label;
        for (std::size_t q = 0; q < x.quadrature_points.size(); ++q) {
            EXPECT_TRUE(sameRealArray(x.quadrature_points[q].point,
                                      y.quadrature_points[q].point))
                << label;
            EXPECT_EQ(std::memcmp(&x.quadrature_points[q].weight,
                                  &y.quadrature_points[q].weight,
                                  sizeof(FE::Real)),
                      0)
                << label;
        }
    }
}

// The generated geometry, and the snapshots built from it, do not depend on
// the number of geometry threads, for full builds (first build, every cell)
// and incremental refreshes (only cut cells).
TEST(FreeSurfaceSnapshotReuse, GeneratedGeometryIsIndependentOfTheThreadCount)
{
    for (const int dimension : {2, 3}) {
        for (const int threads : {2, 3, 8}) {
            // Two systems on the same mesh, one per thread count.
            ReuseFixture fixture(dimension);
            ReuseFixture threaded(dimension, threads);
            interfaces::FreeSurfaceGeometrySnapshotReuseCache serial_cache;
            interfaces::FreeSurfaceGeometrySnapshotReuseCache threaded_cache;
            interfaces::FreeSurfaceGeometrySnapshotPolicy policy;
            policy.require_complete_exterior_boundary_partition = false;
            policy.classification_only_full_cell_side =
                geometry::CutIntegrationSide::Positive;
            const std::vector<std::pair<Point, FE::Real>> motion{
                {{{0.531, 0.487, 0.462}}, FE::Real{0.29}},
                {{{0.5312, 0.4871, 0.4619}}, FE::Real{0.29}},
                {{{0.5600, 0.4500, 0.5000}}, FE::Real{0.31}},
            };
            ReuseFixture::Pair serial_previous;
            ReuseFixture::Pair threaded_previous;
            for (std::size_t step = 0; step < motion.size(); ++step) {
                const std::string label =
                    std::to_string(dimension) + "D threads " +
                    std::to_string(threads) + " step " + std::to_string(step);
                fixture.setLevelSet(motion[step].first, motion[step].second);
                threaded.setLevelSet(motion[step].first, motion[step].second);
                auto a = fixture.build(serial_cache, policy);
                auto b = threaded.build(threaded_cache, policy);
                expectSameDomains(a.full->interfaceDomain(),
                                  b.full->interfaceDomain(),
                                  label);
                EXPECT_EQ(interfaces::compareFreeSurfaceGeometrySnapshots(
                              *a.full, *b.full),
                          "")
                    << label;
                // Threaded snapshot builds with reuse.
                EXPECT_EQ(interfaces::compareFreeSurfaceGeometrySnapshots(
                              *b.reused, *a.full),
                          "")
                    << label;
                serial_previous = std::move(a);
                threaded_previous = std::move(b);
            }
        }
    }
}

#endif

// ---------------------------------------------------------------------------
// The double-precision rejection pre-check of the represented-phase
// containment certificate (FreeSurfaceGeometrySnapshot.cpp) must never change
// a certificate.  Adversarial tetrahedra (reference, skewed, inverted, sliver,
// near-flat, flat, tiny, large) and points (vertices, edge midpoints, face
// points and points within rounding distance of a face on either side, far
// points, non-finite points) are tested with every tolerance class.
// ---------------------------------------------------------------------------

using Tetrahedron = std::array<std::array<FE::Real, 3>, 4>;

std::vector<Tetrahedron> adversarialTetrahedra()
{
    using R = FE::Real;
    return {
        Tetrahedron{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}},
        Tetrahedron{{{1, 0, 0}, {0, 0, 0}, {0, 1, 0}, {0, 0, 1}}},
        Tetrahedron{{{R{0.1}, R{0.2}, R{0.05}},
                     {R{0.9}, R{0.15}, R{0.1}},
                     {R{0.3}, R{0.8}, R{0.2}},
                     {R{0.25}, R{0.3}, R{0.95}}}},
        Tetrahedron{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {R{0.3}, R{0.3}, R{1.0e-9}}}},
        Tetrahedron{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {R{0.3}, R{0.3}, R{1.0e-15}}}},
        Tetrahedron{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {R{0.3}, R{0.3}, R{0.0}}}},
        Tetrahedron{{{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {0, 1, 0}}},
        Tetrahedron{{{R{0.5}, R{0.5}, R{0.5}},
                     {R{0.5} + R{1.0e-9}, R{0.5}, R{0.5}},
                     {R{0.5}, R{0.5} + R{1.0e-9}, R{0.5}},
                     {R{0.5}, R{0.5}, R{0.5} + R{1.0e-9}}}},
        Tetrahedron{{{-100, -50, 10}, {300, 20, -40}, {-20, 250, 30}, {40, 60, 500}}},
    };
}

std::vector<std::array<FE::Real, 3>> adversarialPoints(const Tetrahedron& tet,
                                                       std::uint64_t seed)
{
    using R = FE::Real;
    std::vector<std::array<R, 3>> points;
    const auto add = [&](std::array<R, 3> p) { points.push_back(p); };
    const auto combine = [&](const std::array<R, 4>& weights) {
        std::array<R, 3> p{{0, 0, 0}};
        for (std::size_t v = 0; v < 4; ++v) {
            for (std::size_t k = 0; k < 3; ++k) {
                p[k] += weights[v] * tet[v][k];
            }
        }
        return p;
    };
    for (const auto& v : tet) {
        add(v);
    }
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = i + 1; j < 4; ++j) {
            std::array<R, 4> w{{0, 0, 0, 0}};
            w[i] = R{0.5};
            w[j] = R{0.5};
            add(combine(w));
        }
    }
    add(combine({{R{0.25}, R{0.25}, R{0.25}, R{0.25}}}));
    const R offsets[] = {R{0},       R{1.0e-17}, R{1.0e-16}, R{1.0e-15},
                         R{1.0e-14}, R{1.0e-13}, R{1.0e-12}, R{1.0e-10},
                         R{1.0e-8},  R{1.0e-6},  R{1.0e-3},  R{0.1},
                         R{1.0}};
    for (std::size_t opposite = 0; opposite < 4; ++opposite) {
        std::array<R, 4> w{{0, 0, 0, 0}};
        for (std::size_t v = 0; v < 4; ++v) {
            w[v] = v == opposite ? R{0} : R{1.0} / R{3.0};
        }
        const auto centroid = combine(w);
        std::array<R, 3> inward{};
        for (std::size_t k = 0; k < 3; ++k) {
            inward[k] = tet[opposite][k] - centroid[k];
        }
        const R length = std::sqrt(inward[0] * inward[0] + inward[1] * inward[1] +
                                   inward[2] * inward[2]);
        for (const R offset : offsets) {
            for (const R sign : {R{1.0}, R{-1.0}}) {
                std::array<R, 3> p = centroid;
                for (std::size_t k = 0; k < 3; ++k) {
                    const R scale = length > R{0} ? sign * offset / length : R{0};
                    p[k] += scale * inward[k];
                }
                add(p);
            }
        }
    }
    std::array<R, 3> lo = tet[0];
    std::array<R, 3> hi = tet[0];
    for (const auto& v : tet) {
        for (std::size_t k = 0; k < 3; ++k) {
            lo[k] = std::min(lo[k], v[k]);
            hi[k] = std::max(hi[k], v[k]);
        }
    }
    std::uint64_t state = seed;
    const auto next_unit = [&state]() {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<R>(state >> 11) * (R{1.0} / R{9007199254740992.0});
    };
    for (int i = 0; i < 400; ++i) {
        std::array<R, 3> p{};
        for (std::size_t k = 0; k < 3; ++k) {
            const R span = std::max(hi[k] - lo[k], R{1.0e-12});
            p[k] = lo[k] - span + R{3.0} * span * next_unit();
        }
        add(p);
    }
    const R nan = std::numeric_limits<R>::quiet_NaN();
    const R inf = std::numeric_limits<R>::infinity();
    add({{nan, R{0.1}, R{0.1}}});
    add({{R{0.1}, inf, R{0.1}}});
    add({{-inf, R{0.1}, R{0.1}}});
    return points;
}

bool sameCoordinateBits(const std::array<FE::Real, 4>& a,
                        const std::array<FE::Real, 4>& b)
{
    return std::memcmp(a.data(), b.data(), sizeof(a)) == 0;
}

TEST(FreeSurfaceSnapshotContainmentPreCheck, KeepsEveryTetrahedronCertificate)
{
    const FE::Real tolerances[] = {FE::Real{0.0}, FE::Real{1.0e-14},
                                   FE::Real{1.0e-10}, FE::Real{1.0e-6},
                                   FE::Real{0.25}};
    std::size_t accepted = 0;
    std::size_t prechecked_rejections = 0;
    std::uint64_t seed = 12345u;
    for (const auto& tet : adversarialTetrahedra()) {
        interfaces::CutInterfaceReferenceSimplex simplex;
        simplex.vertex_count = 4u;
        simplex.vertices = tet;
        for (const auto& point : adversarialPoints(tet, ++seed)) {
            for (const FE::Real tolerance : tolerances) {
                std::array<FE::Real, 4> with{};
                std::array<FE::Real, 4> without{};
                const bool accepted_with =
                    interfaces::snapshot_detail::representedSimplexCoordinates(
                        simplex, point, 3, tolerance, with, true);
                const bool accepted_without =
                    interfaces::snapshot_detail::representedSimplexCoordinates(
                        simplex, point, 3, tolerance, without, false);
                ASSERT_EQ(accepted_with, accepted_without)
                    << "point (" << point[0] << ", " << point[1] << ", "
                    << point[2] << ") tolerance " << tolerance;
                if (accepted_without) {
                    ++accepted;
                    EXPECT_TRUE(sameCoordinateBits(with, without));
                }
                const FE::Real coordinate_tolerance = std::max(
                    tolerance,
                    FE::Real{128.0} * std::numeric_limits<FE::Real>::epsilon());
                if (interfaces::snapshot_detail::tetrahedronCertainlyRejectsPoint(
                        tet, point, coordinate_tolerance)) {
                    ++prechecked_rejections;
                    EXPECT_FALSE(accepted_without)
                        << "pre-check rejected a point the solve accepts";
                }
            }
        }
    }
    // Both branches are exercised.
    EXPECT_GT(accepted, 1000u);
    EXPECT_GT(prechecked_rejections, 5000u);
}

// The containment search of the represented-phase check takes the first
// subcell, in order, that contains the point.  With and without the pre-check
// it must match the same subcell with the same coordinates, including points
// on faces shared by two subcells (Kuhn decomposition of a box).
TEST(FreeSurfaceSnapshotContainmentPreCheck, KeepsTheMatchedSubcellOfASearch)
{
    using R = FE::Real;
    std::vector<interfaces::CutInterfaceReferenceSimplex> subcells;
    const int permutations[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2},
                                    {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
    const std::array<R, 3> size{{R{1.0}, R{0.7}, R{1.3}}};
    for (const auto& permutation : permutations) {
        interfaces::CutInterfaceReferenceSimplex simplex;
        simplex.vertex_count = 4u;
        std::array<R, 3> corner{{0, 0, 0}};
        simplex.vertices[0] = corner;
        for (int step = 0; step < 3; ++step) {
            const auto axis = static_cast<std::size_t>(permutation[step]);
            corner[axis] = size[axis];
            simplex.vertices[static_cast<std::size_t>(step + 1)] = corner;
        }
        subcells.push_back(simplex);
    }
    const auto search = [&](const std::array<R, 3>& point,
                            R tolerance,
                            bool allow_double_rejection,
                            std::array<R, 4>& coordinates) {
        for (std::size_t s = 0; s < subcells.size(); ++s) {
            std::array<R, 4> candidate{};
            if (interfaces::snapshot_detail::representedSimplexCoordinates(
                    subcells[s], point, 3, tolerance, candidate,
                    allow_double_rejection)) {
                coordinates = candidate;
                return static_cast<int>(s);
            }
        }
        return -1;
    };
    std::size_t matched = 0;
    const R perturbations[] = {R{0.0}, R{1.0e-16}, R{-1.0e-16}, R{1.0e-13},
                               R{-1.0e-13}, R{1.0e-9}, R{-1.0e-9}};
    for (int i = 0; i <= 8; ++i) {
        for (int j = 0; j <= 8; ++j) {
            for (int k = 0; k <= 8; ++k) {
                for (const R perturbation : perturbations) {
                    const std::array<R, 3> point{
                        {size[0] * static_cast<R>(i) / R{8} + perturbation,
                         size[1] * static_cast<R>(j) / R{8} - perturbation,
                         size[2] * static_cast<R>(k) / R{8} + perturbation}};
                    for (const R tolerance : {R{0.0}, R{1.0e-12}}) {
                        std::array<R, 4> with{};
                        std::array<R, 4> without{};
                        const int match_with = search(point, tolerance, true, with);
                        const int match_without =
                            search(point, tolerance, false, without);
                        ASSERT_EQ(match_with, match_without);
                        if (match_without >= 0) {
                            ++matched;
                            EXPECT_TRUE(sameCoordinateBits(with, without));
                        }
                    }
                }
            }
        }
    }
    EXPECT_GT(matched, 1000u);
}

} // namespace

namespace {

interfaces::FreeSurfaceGeometryRuleRecord partitionRecord(
    interfaces::FreeSurfaceGeometryRuleRole role,
    FE::GlobalIndex global_cell,
    FE::GlobalIndex local_parent,
    FE::Real measure,
    FE::Real parent_measure)
{
    interfaces::FreeSurfaceGeometryRuleRecord record;
    record.role = role;
    record.reference_rule.provenance.parent_entity_global_id = global_cell;
    record.reference_rule.provenance.parent_entity = local_parent;
    record.reference_rule.measure = measure;
    record.reference_rule.parent_measure = parent_measure;
    record.reference_rule.exact_polynomial_order = 1;
    record.classification_only = true;
    return record;
}

} // namespace

// validateVolumePartition() groups the volume records by global cell through
// a stable sort instead of a std::map.  The alias rule and its precedence over
// the partition check, the partition check and the ledger maximum are
// unchanged.
TEST(FreeSurfaceSnapshotValidation, VolumePartitionGroupsCellsAsTheKeyOrderedMap)
{
    using Role = interfaces::FreeSurfaceGeometryRuleRole;
    ReuseFixture fixture(3);
    const auto& mesh = fixture.system.meshAccess();
    const interfaces::FreeSurfaceGeometrySnapshotPolicy policy{};

    // Single classification-only full cells (no moment check), out of order,
    // with interface records interleaved; the partition errors differ in the
    // last bits so the maximum is order sensitive only through its value.
    std::vector<interfaces::FreeSurfaceGeometryRuleRecord> records;
    records.push_back(partitionRecord(Role::NegativeVolume, 40, 3, 0.125, 0.125));
    records.push_back(partitionRecord(Role::Interface, 40, 3, 0.5, 0.125));
    records.push_back(partitionRecord(Role::PositiveVolume, 7, 1, 0.3, 0.1 + 0.2));
    records.push_back(partitionRecord(Role::NegativeVolume, 12, 5, 1.0 / 3.0, 1.0 / 3.0));
    records.push_back(partitionRecord(Role::PositiveVolume, -3, 0, 0.7, 0.7));
    FE::Real expected_maximum = 0.0;
    for (const auto& record : records) {
        if (record.role == Role::Interface) {
            continue;
        }
        expected_maximum = std::max(
            expected_maximum,
            std::abs(record.reference_rule.measure - record.reference_rule.parent_measure));
    }
    interfaces::FreeSurfaceGeometryValidationLedger ledger;
    ASSERT_NO_THROW(interfaces::snapshot_detail::validateVolumePartition(
        records, mesh, policy, ledger));
    EXPECT_EQ(ledger.maximum_volume_partition_error, expected_maximum);

    // Cell 12 named by a second record with another local parent.
    auto aliased = records;
    aliased.push_back(partitionRecord(Role::NegativeVolume, 12, 6, 0.0, 1.0 / 3.0));

    // An alias is reported before a partition error of a lower cell.
    auto aliased_and_unbalanced = aliased;
    aliased_and_unbalanced.push_back(partitionRecord(Role::PositiveVolume, -9, 2, 0.5, 1.0));
    try {
        interfaces::FreeSurfaceGeometryValidationLedger alias_ledger;
        interfaces::snapshot_detail::validateVolumePartition(
            aliased_and_unbalanced, mesh, policy, alias_ledger);
        ADD_FAILURE() << "an aliased cell must be rejected";
    } catch (const std::invalid_argument& error) {
        EXPECT_NE(std::string(error.what()).find("aliases one global cell"),
                  std::string::npos)
            << error.what();
    }

    auto unbalanced = records;
    unbalanced.push_back(partitionRecord(Role::PositiveVolume, -9, 2, 0.5, 1.0));
    try {
        interfaces::FreeSurfaceGeometryValidationLedger unbalanced_ledger;
        interfaces::snapshot_detail::validateVolumePartition(
            unbalanced, mesh, policy, unbalanced_ledger);
        ADD_FAILURE() << "an unbalanced cell must be rejected";
    } catch (const std::invalid_argument& error) {
        EXPECT_NE(std::string(error.what()).find("do not partition"), std::string::npos)
            << error.what();
    }
}
