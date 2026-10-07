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

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
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
// exactly as successive outer passes do.
struct ReuseFixture {
    std::shared_ptr<svmp::Mesh> mesh;
    FE::systems::FESystem system;
    FE::FieldId phi{FE::INVALID_FIELD_ID};
    int dimension{3};
    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle{};
    std::vector<FE::Real> solution{};

    explicit ReuseFixture(int dim)
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
        system.setup();
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
               const interfaces::FreeSurfaceGeometrySnapshotPolicy& policy,
               level_set::LevelSetGeneratedInterfaceLifecycle* other_lifecycle =
                   nullptr)
    {
        auto& active_lifecycle =
            other_lifecycle != nullptr ? *other_lifecycle : lifecycle;
        level_set::LevelSetGeneratedInterfaceOptions options{};
        options.level_set_field_name = "phi";
        options.domain_id = "snapshot_reuse";
        options.requested_interface_marker = 4712;
        options.quadrature_order = 1;
        options.interface_quadrature_order = 2;
        options.volume_quadrature_order = 2;
        const auto generated =
            active_lifecycle.build(system, options, solution);
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
            "snapshot_reuse", {}, &cache, FE::geometryThreadCount());
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

// Sets SVMP_ASSEMBLY_THREADS for the lifetime of the object.
class ScopedGeometryThreads {
public:
    explicit ScopedGeometryThreads(int threads)
    {
        if (const char* old = std::getenv("SVMP_ASSEMBLY_THREADS")) {
            previous_ = old;
            had_previous_ = true;
        }
        ::setenv("SVMP_ASSEMBLY_THREADS", std::to_string(threads).c_str(), 1);
    }
    ~ScopedGeometryThreads()
    {
        if (had_previous_) {
            ::setenv("SVMP_ASSEMBLY_THREADS", previous_.c_str(), 1);
        } else {
            ::unsetenv("SVMP_ASSEMBLY_THREADS");
        }
    }
    ScopedGeometryThreads(const ScopedGeometryThreads&) = delete;
    ScopedGeometryThreads& operator=(const ScopedGeometryThreads&) = delete;

private:
    std::string previous_{};
    bool had_previous_{false};
};

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
            // One mesh and system; two lifecycles, one per thread count.
            ReuseFixture fixture(dimension);
            level_set::LevelSetGeneratedInterfaceLifecycle threaded_lifecycle;
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
                ReuseFixture::Pair a;
                ReuseFixture::Pair b;
                {
                    ScopedGeometryThreads one(1);
                    a = fixture.build(serial_cache, policy);
                }
                {
                    ScopedGeometryThreads many(threads);
                    b = fixture.build(threaded_cache, policy,
                                      &threaded_lifecycle);
                }
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

} // namespace
