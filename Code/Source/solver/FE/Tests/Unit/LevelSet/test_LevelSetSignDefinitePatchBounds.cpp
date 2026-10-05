#include "LevelSet/LevelSetSignDefinitePatchBounds.h"

#include "Dofs/DofHandler.h"
#include "Dofs/EntityDofMap.h"
#include "LevelSet/LevelSetVolume.h"
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
#include <functional>
#include <limits>
#include <memory>
#include <set>
#include <utility>
#include <vector>

namespace {

namespace FE = svmp::FE;
namespace level_set = svmp::FE::level_set;

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

using Point = std::array<FE::Real, 3>;
using ScalarFunction = std::function<FE::Real(const Point&)>;

// Unit square split into right triangles whose diagonal alternates with the
// cell parity, as in the free-surface benchmark meshes.
std::shared_ptr<svmp::Mesh> buildStructuredTriangleMesh(int n)
{
    auto base = std::make_shared<svmp::MeshBase>();
    const auto extent = static_cast<svmp::index_t>(n + 1);
    std::vector<svmp::real_t> x_ref;
    for (int j = 0; j <= n; ++j) {
        for (int i = 0; i <= n; ++i) {
            x_ref.push_back(static_cast<svmp::real_t>(i) / n);
            x_ref.push_back(static_cast<svmp::real_t>(j) / n);
        }
    }
    std::vector<svmp::offset_t> offsets{0};
    std::vector<svmp::index_t> cell2vertex;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const svmp::index_t a = j * extent + i;
            const svmp::index_t b = a + 1;
            const svmp::index_t d = a + extent;
            const svmp::index_t c = d + 1;
            if ((i + j) % 2 == 0) {
                cell2vertex.insert(cell2vertex.end(), {a, b, c, a, c, d});
            } else {
                cell2vertex.insert(cell2vertex.end(), {a, b, d, b, c, d});
            }
            offsets.push_back(static_cast<svmp::offset_t>(cell2vertex.size() - 3u));
            offsets.push_back(static_cast<svmp::offset_t>(cell2vertex.size()));
        }
    }
    svmp::CellShape shape{};
    shape.family = svmp::CellFamily::Triangle;
    shape.num_corners = 3;
    shape.order = 1;
    std::vector<svmp::CellShape> shapes(offsets.size() - 1u, shape);
    base->build_from_arrays(/*spatial_dim=*/2, x_ref, offsets, cell2vertex, shapes);
    base->finalize();
    return svmp::create_mesh(std::move(base));
}

// Unit cube split into six tetrahedra per hexahedron (Kuhn subdivision).
std::shared_ptr<svmp::Mesh> buildStructuredTetraMesh(int n)
{
    auto base = std::make_shared<svmp::MeshBase>();
    const auto extent = static_cast<svmp::index_t>(n + 1);
    std::vector<svmp::real_t> x_ref;
    for (int k = 0; k <= n; ++k) {
        for (int j = 0; j <= n; ++j) {
            for (int i = 0; i <= n; ++i) {
                x_ref.push_back(static_cast<svmp::real_t>(i) / n);
                x_ref.push_back(static_cast<svmp::real_t>(j) / n);
                x_ref.push_back(static_cast<svmp::real_t>(k) / n);
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
                    const int det = e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1]) -
                                    e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0]) +
                                    e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0]);
                    if (det < 0) {
                        std::swap(tet[1], tet[2]);
                    }
                    for (const auto& vertex : tet) {
                        cell2vertex.push_back(vid(vertex[0], vertex[1], vertex[2]));
                    }
                    offsets.push_back(static_cast<svmp::offset_t>(cell2vertex.size()));
                }
            }
        }
    }
    svmp::CellShape shape{};
    shape.family = svmp::CellFamily::Tetra;
    shape.num_corners = 4;
    shape.order = 1;
    std::vector<svmp::CellShape> shapes(offsets.size() - 1u, shape);
    base->build_from_arrays(/*spatial_dim=*/3, x_ref, offsets, cell2vertex, shapes);
    base->finalize();
    return svmp::create_mesh(std::move(base));
}

struct ScalarSystem {
    std::shared_ptr<svmp::Mesh> mesh;
    FE::systems::FESystem system;
    FE::FieldId phi{FE::INVALID_FIELD_ID};

    ScalarSystem(std::shared_ptr<svmp::Mesh> m, int dim)
        : mesh(std::move(m)), system(mesh)
    {
        const auto type = dim == 2 ? FE::ElementType::Triangle3 : FE::ElementType::Tetra4;
        phi = system.addField(FE::systems::FieldSpec{
            .name = "phi",
            .space = std::make_shared<FE::spaces::H1Space>(type, /*order=*/1),
            .components = 1,
        });
        system.setup();
    }

    [[nodiscard]] const FE::dofs::DofHandler& dofs() const
    {
        return system.fieldDofHandler(phi);
    }

    [[nodiscard]] std::vector<FE::Real> scalar(const ScalarFunction& f) const
    {
        const auto* map = dofs().getEntityDofMap();
        std::vector<FE::Real> out(static_cast<std::size_t>(dofs().getNumDofs()), 0.0);
        for (FE::GlobalIndex v = 0; v < map->numVertices(); ++v) {
            out[static_cast<std::size_t>(map->getVertexDofs(v).front())] =
                f(system.meshAccess().getNodeCoordinates(v));
        }
        return out;
    }

    [[nodiscard]] std::size_t dofAt(const Point& x) const
    {
        const auto* map = dofs().getEntityDofMap();
        for (FE::GlobalIndex v = 0; v < map->numVertices(); ++v) {
            const auto y = system.meshAccess().getNodeCoordinates(v);
            if (std::abs(y[0] - x[0]) + std::abs(y[1] - x[1]) + std::abs(y[2] - x[2]) < 1.0e-12) {
                return static_cast<std::size_t>(map->getVertexDofs(v).front());
            }
        }
        ADD_FAILURE() << "no vertex at the requested point";
        return 0u;
    }

    // Previous-state range of the patch of a node.
    [[nodiscard]] std::pair<FE::Real, FE::Real> patchRange(
        std::size_t dof, const std::vector<FE::Real>& values) const
    {
        FE::Real lo = std::numeric_limits<FE::Real>::infinity();
        FE::Real hi = -lo;
        const auto& access = system.meshAccess();
        access.forEachCell([&](FE::GlobalIndex cell) {
            const auto cell_dofs = dofs().getCellDofs(cell);
            if (std::find(cell_dofs.begin(), cell_dofs.end(),
                          static_cast<FE::GlobalIndex>(dof)) == cell_dofs.end()) {
                return;
            }
            for (const auto d : cell_dofs) {
                lo = std::min(lo, values[static_cast<std::size_t>(d)]);
                hi = std::max(hi, values[static_cast<std::size_t>(d)]);
            }
        });
        return {lo, hi};
    }

    [[nodiscard]] level_set::LevelSetSignDefinitePatchBoundsResult bound(
        const std::vector<FE::Real>& previous,
        const std::vector<FE::Real>& candidate,
        std::vector<FE::Real>& bounded) const
    {
        return level_set::boundLevelSetOnSignDefinitePatches(
            system.meshAccess(), dofs(), /*isovalue=*/0.0, /*tolerance=*/1.0e-12,
            previous, candidate, bounded);
    }

    [[nodiscard]] FE::Real negativeVolume(const std::vector<FE::Real>& values) const
    {
        const auto result = level_set::computeLevelSetCutCellVolume(
            system.meshAccess(), dofs(), level_set::LevelSetVolumeOptions{}, values);
        EXPECT_TRUE(result.success) << result.diagnostic;
        return result.negative_volume;
    }
};

FE::Real circle(const Point& x)
{
    return std::hypot(x[0] - 0.5031, x[1] - 0.4687) - 0.27;
}

} // namespace

TEST(LevelSetSignDefinitePatchBounds, LeavesAConsistentTranslationUnchangedAwayFromInflow)
{
    ScalarSystem s(buildStructuredTriangleMesh(16), 2);
    // A plane translated by less than a cell along +grad(phi) stays inside the
    // patch range of every node whose characteristic starts inside the mesh.
    // Only nodes on the inflow part of the boundary (x = 0 or y = 0), where
    // the value comes from outside, can leave it; the bound is meant for
    // impermeable or outflow boundaries.
    const auto plane = [](const Point& x) { return 0.8 * x[0] + 0.6 * x[1] - 0.5371; };
    const auto previous = s.scalar(plane);
    const auto candidate = s.scalar([&](const Point& x) { return plane(x) - 0.01; });
    std::vector<FE::Real> bounded;
    const auto result = s.bound(previous, candidate, bounded);
    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_GT(result.sign_definite_dofs, 0u);
    EXPECT_EQ(result.sign_changes_prevented, 0u);
    const auto* map = s.dofs().getEntityDofMap();
    for (FE::GlobalIndex v = 0; v < map->numVertices(); ++v) {
        const auto dof = static_cast<std::size_t>(map->getVertexDofs(v).front());
        const auto x = s.system.meshAccess().getNodeCoordinates(v);
        const bool inflow = x[0] < 1.0e-12 || x[1] < 1.0e-12;
        if (!inflow) {
            EXPECT_EQ(bounded[dof], candidate[dof]) << "x=" << x[0] << " y=" << x[1];
        }
    }
}

TEST(LevelSetSignDefinitePatchBounds, PreventsASpuriousCrossingAwayFromTheInterface)
{
    ScalarSystem s(buildStructuredTriangleMesh(16), 2);
    const auto previous = s.scalar(circle);
    auto candidate = previous;
    // A dry node two cells outside the interface is driven across the
    // isovalue, a wet node inside undershoots its patch without changing sign.
    const auto dry = s.dofAt({0.875, 0.5, 0.0});
    const auto wet = s.dofAt({0.5, 0.5, 0.0});
    ASSERT_GT(previous[dry], 1.5 / 16.0);
    ASSERT_LT(previous[wet], -1.5 / 16.0);
    candidate[dry] = -1.0e-3;
    candidate[wet] = previous[wet] - 0.05;
    std::vector<FE::Real> bounded;
    const auto result = s.bound(previous, candidate, bounded);
    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_TRUE(result.applied);
    EXPECT_EQ(result.bounded_dofs, 2u);
    EXPECT_EQ(result.sign_changes_prevented, 1u);
    EXPECT_EQ(bounded[dry], s.patchRange(dry, previous).first);
    EXPECT_GT(bounded[dry], 0.0);
    EXPECT_EQ(bounded[wet], s.patchRange(wet, previous).first);
    EXPECT_NEAR(result.max_abs_correction,
                std::max(std::abs(bounded[dry] - candidate[dry]),
                         std::abs(bounded[wet] - candidate[wet])),
                0.0);
    for (std::size_t i = 0; i < bounded.size(); ++i) {
        if (i != dry && i != wet) {
            EXPECT_EQ(bounded[i], candidate[i]);
        }
    }
    // The spurious island is gone; the true interface is untouched.
    EXPECT_NEAR(s.negativeVolume(bounded), s.negativeVolume(previous), 1.0e-15);
}

TEST(LevelSetSignDefinitePatchBounds, NeverChangesNodesOfCutCells)
{
    ScalarSystem s(buildStructuredTriangleMesh(16), 2);
    const FE::Real h = 1.0 / 16.0;
    const auto previous = s.scalar(circle);
    // Large overshoots everywhere: only nodes whose patch stays in one phase
    // may change, so the cut cells and the liquid measure are untouched.
    const auto candidate = s.scalar([&](const Point& x) {
        const FE::Real phi = circle(x);
        const FE::Real wiggle = 0.4 * h * std::sin(97.0 * x[0]) * std::cos(89.0 * x[1]);
        return std::abs(phi) < 1.5 * h ? phi + 0.02 * h * std::sin(31.0 * x[0]) : phi + wiggle;
    });
    std::vector<FE::Real> bounded;
    const auto result = s.bound(previous, candidate, bounded);
    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_TRUE(result.applied);
    EXPECT_EQ(result.sign_changes_prevented, 0u);

    std::set<std::size_t> cut_nodes;
    s.system.meshAccess().forEachCell([&](FE::GlobalIndex cell) {
        const auto cell_dofs = s.dofs().getCellDofs(cell);
        bool negative = false;
        bool positive = false;
        for (const auto d : cell_dofs) {
            for (const auto* values : {&previous, &candidate}) {
                const auto v = (*values)[static_cast<std::size_t>(d)];
                negative = negative || v < 0.0;
                positive = positive || v > 0.0;
            }
        }
        if (negative && positive) {
            for (const auto d : cell_dofs) {
                cut_nodes.insert(static_cast<std::size_t>(d));
            }
        }
    });
    ASSERT_FALSE(cut_nodes.empty());
    for (const auto d : cut_nodes) {
        EXPECT_EQ(bounded[d], candidate[d]);
    }
    for (std::size_t i = 0; i < bounded.size(); ++i) {
        const auto [lo, hi] = s.patchRange(i, previous);
        if (bounded[i] != candidate[i]) {
            EXPECT_EQ(cut_nodes.count(i), 0u);
            EXPECT_GE(bounded[i], lo);
            EXPECT_LE(bounded[i], hi);
        }
    }
    EXPECT_EQ(s.negativeVolume(bounded), s.negativeVolume(candidate));
}

TEST(LevelSetSignDefinitePatchBounds, LeavesANodeWhoseNeighbourChangesSign)
{
    ScalarSystem s(buildStructuredTriangleMesh(16), 2);
    const auto previous = s.scalar(circle);
    auto candidate = previous;
    // The interface moves into the patch of a dry node: a neighbour becomes
    // wet, so the node is next to the interface and keeps its candidate value
    // even though it leaves its previous patch range.
    const auto node = s.dofAt({0.875, 0.5, 0.0});
    const auto neighbour = s.dofAt({0.8125, 0.5, 0.0});
    candidate[neighbour] = -1.0e-3;
    candidate[node] = s.patchRange(node, previous).first - 0.01;
    std::vector<FE::Real> bounded;
    const auto result = s.bound(previous, candidate, bounded);
    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(bounded[node], candidate[node]);
    EXPECT_EQ(bounded[neighbour], candidate[neighbour]);
}

TEST(LevelSetSignDefinitePatchBounds, BoundsTetrahedralPatches)
{
    ScalarSystem s(buildStructuredTetraMesh(6), 3);
    const auto sphere = [](const Point& x) {
        return std::sqrt((x[0] - 0.51) * (x[0] - 0.51) + (x[1] - 0.47) * (x[1] - 0.47) +
                         (x[2] - 0.49) * (x[2] - 0.49)) - 0.3;
    };
    const auto previous = s.scalar(sphere);
    auto candidate = previous;
    const auto corner = s.dofAt({0.0, 0.0, 0.0});
    ASSERT_GT(previous[corner], 0.3);
    candidate[corner] = -0.01;
    std::vector<FE::Real> bounded;
    const auto result = s.bound(previous, candidate, bounded);
    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.bounded_dofs, 1u);
    EXPECT_EQ(result.sign_changes_prevented, 1u);
    EXPECT_EQ(bounded[corner], s.patchRange(corner, previous).first);
}

TEST(LevelSetSignDefinitePatchBounds, RejectsBadInput)
{
    ScalarSystem s(buildStructuredTriangleMesh(4), 2);
    const auto previous = s.scalar(circle);
    std::vector<FE::Real> bounded;
    auto short_candidate = previous;
    short_candidate.pop_back();
    EXPECT_FALSE(s.bound(previous, short_candidate, bounded).success);
    auto nan_candidate = previous;
    nan_candidate[3] = std::numeric_limits<FE::Real>::quiet_NaN();
    EXPECT_FALSE(s.bound(previous, nan_candidate, bounded).success);
    EXPECT_FALSE(level_set::boundLevelSetOnSignDefinitePatches(
                     s.system.meshAccess(), s.dofs(), 0.0, -1.0, previous, previous, bounded)
                     .success);
}

#endif
