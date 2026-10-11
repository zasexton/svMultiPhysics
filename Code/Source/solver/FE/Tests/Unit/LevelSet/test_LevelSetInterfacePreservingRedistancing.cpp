#include "LevelSet/LevelSetInterfacePreservingRedistancing.h"

#include "Dofs/DofHandler.h"
#include "Dofs/EntityDofMap.h"
#include "LevelSet/LevelSetKinematicReconciliation.h"
#include "LevelSet/LevelSetVolume.h"
#include "Spaces/H1Space.h"
#include "Spaces/SpaceFactory.h"
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


    [[nodiscard]] FE::Real negativeVolume(const std::vector<FE::Real>& values) const
    {
        const auto result = level_set::computeLevelSetCutCellVolume(
            system.meshAccess(), dofs(), level_set::LevelSetVolumeOptions{}, values);
        EXPECT_TRUE(result.success) << result.diagnostic;
        return result.negative_volume;
    }
};


// A cap of radius R on the bottom wall y = 0 with contact angle theta
// (through the liquid, phi < 0), scaled by a non-unit factor.
FE::Real cap(const Point& x, FE::Real theta, FE::Real scale)
{
    constexpr FE::Real R = 0.4;
    const FE::Real cx = 0.5031;
    const FE::Real cy = -R * std::cos(theta);
    return scale * (std::hypot(x[0] - cx, x[1] - cy) - R);
}

std::vector<FE::Real> wallCrossings(const ScalarSystem& s, const std::vector<FE::Real>& phi)
{
    std::vector<std::pair<FE::Real, FE::Real>> wall;  // (x, phi) on y = 0
    const auto* map = s.dofs().getEntityDofMap();
    for (FE::GlobalIndex v = 0; v < map->numVertices(); ++v) {
        const auto y = s.system.meshAccess().getNodeCoordinates(v);
        if (std::abs(y[1]) < 1.0e-14) {
            wall.emplace_back(y[0], phi[static_cast<std::size_t>(map->getVertexDofs(v).front())]);
        }
    }
    std::sort(wall.begin(), wall.end());
    std::vector<FE::Real> out;
    for (std::size_t i = 0; i + 1 < wall.size(); ++i) {
        const auto [xa, fa] = wall[i];
        const auto [xb, fb] = wall[i + 1];
        if ((fa < 0.0) != (fb < 0.0)) {
            out.push_back(xa + fa / (fa - fb) * (xb - xa));
        }
    }
    return out;
}

} // namespace

TEST(LevelSetInterfacePreservingRedistancing, ScaledPlaneBecomesItsExactDistance)
{
    ScalarSystem s(buildStructuredTriangleMesh(12), 2);
    // A tilted line crossing the bottom and top boundaries, scaled by 2.5.
    const auto plane = [](const Point& x) {
        return 0.8 * (x[0] - 0.437) + 0.6 * (x[1] - 0.211);
    };
    const auto scaled = s.scalar([&](const Point& x) { return 2.5 * plane(x); });
    const auto exact = s.scalar(plane);
    std::vector<FE::Real> out;
    const auto result = level_set::redistanceLevelSetToZeroPolygon(
        s.system.meshAccess(), s.dofs(), 0.0, scaled, out);
    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_TRUE(result.applied);
    EXPECT_EQ(result.open_ends, 2u);
    for (std::size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(out[i], exact[i], 1.0e-13) << "dof " << i;
    }
    const auto change = level_set::measureLevelSetZeroPolygonChange(
        s.system.meshAccess(), s.dofs(), 0.0, scaled, out);
    ASSERT_TRUE(change.success) << change.diagnostic;
    EXPECT_LT(change.max_crossing_shift, 1.0e-14);
    EXPECT_LT(change.max_open_end_shift, 1.0e-14);
    EXPECT_EQ(change.cells_with_changed_cut, 0u);
    // Already a distance: nothing changes beyond round-off.
    std::vector<FE::Real> again;
    const auto second = level_set::redistanceLevelSetToZeroPolygon(
        s.system.meshAccess(), s.dofs(), 0.0, out, again);
    ASSERT_TRUE(second.success);
    EXPECT_LT(second.max_abs_change, 1.0e-13);
}

TEST(LevelSetInterfacePreservingRedistancing, ContactCapKeepsSignsContactPointsAndAngleAndRestoresArea)
{
    for (const FE::Real degrees : {60.0, 120.0}) {
        ScalarSystem s(buildStructuredTriangleMesh(32), 2);
        const FE::Real theta = degrees * 3.14159265358979323846 / 180.0;
        // A distorted field with the zero set of the cap: steep near the
        // wall, flat above it.
        const auto distorted = s.scalar([&](const Point& x) {
            return cap(x, theta, 1.0) * (0.7 + 0.6 * std::exp(-20.0 * x[1]));
        });
        std::vector<FE::Real> out;
        const auto result = level_set::redistanceLevelSetToZeroPolygon(
            s.system.meshAccess(), s.dofs(), 0.0, distorted, out);
        ASSERT_TRUE(result.success) << result.diagnostic;
        EXPECT_EQ(result.open_ends, 2u) << degrees;
        for (std::size_t i = 0; i < out.size(); ++i) {
            ASSERT_EQ(out[i] < 0.0, distorted[i] < 0.0) << "sign of dof " << i;
        }
        const auto change = level_set::measureLevelSetZeroPolygonChange(
            s.system.meshAccess(), s.dofs(), 0.0, distorted, out);
        ASSERT_TRUE(change.success);
        EXPECT_EQ(change.cells_with_changed_cut, 0u);
        const FE::Real h = 1.0 / 32.0;
        EXPECT_LT(change.max_crossing_shift, 0.1 * h) << degrees;
        // The wall crossings move by much less than the crossing bound,
        // because the end segments are continued across the wall.
        const auto before = wallCrossings(s, distorted);
        const auto after = wallCrossings(s, out);
        ASSERT_EQ(before.size(), 2u);
        ASSERT_EQ(after.size(), 2u);
        for (std::size_t k = 0; k < 2u; ++k) {
            EXPECT_LT(std::abs(after[k] - before[k]), 0.02 * h) << degrees;
        }
        // Zero-velocity reconciliation restores the area to round-off level
        // relative to the raw redistancing change.
        const auto v0 = s.negativeVolume(distorted);
        const auto v1 = s.negativeVolume(out);
        const std::vector<FE::Real> zero_velocity(
            2u * out.size(), 0.0);
        auto velocity_space = FE::spaces::VectorSpace(
            FE::spaces::SpaceType::H1, FE::ElementType::Triangle3, /*order=*/1, 2);
        FE::systems::FESystem vsys(s.mesh);
        const auto u = vsys.addField(FE::systems::FieldSpec{
            .name = "u", .space = velocity_space,
            .components = velocity_space->value_dimension()});
        vsys.setup();
        std::vector<FE::Real> restored;
        const auto restoration = level_set::reconcileLevelSetWithKinematicFlux(
            s.system.meshAccess(), s.dofs(), vsys.fieldDofHandler(u), 0.0, 1.0e-12,
            1.0, distorted, out,
            std::vector<FE::Real>(static_cast<std::size_t>(vsys.fieldDofHandler(u).getNumDofs()), 0.0),
            std::vector<FE::Real>(static_cast<std::size_t>(vsys.fieldDofHandler(u).getNumDofs()), 0.0),
            restored);
        ASSERT_TRUE(restoration.success) << restoration.diagnostic;
        // A single large redistancing may need more than the restoration's
        // fixed iteration cap; the restored area is what must hold.
        EXPECT_EQ(restoration.degeneracy_frozen_dofs, 0u) << degrees;
        const auto v2 = s.negativeVolume(restored);
        EXPECT_LT(std::abs(v2 - v0), 0.1 * std::abs(v1 - v0) + 1.0e-12) << degrees;
        const auto restored_change = level_set::measureLevelSetZeroPolygonChange(
            s.system.meshAccess(), s.dofs(), 0.0, distorted, restored);
        ASSERT_TRUE(restored_change.success);
        EXPECT_LT(restored_change.max_crossing_shift, change.max_crossing_shift) << degrees;
        (void)zero_velocity;
    }
}

TEST(LevelSetInterfacePreservingRedistancing, RejectsTetrahedraOnEveryRank)
{
    ScalarSystem s(buildStructuredTetraMesh(2), 3);
    const auto phi = s.scalar([](const Point& x) { return x[0] - 0.41; });
    std::vector<FE::Real> out;
    const auto result = level_set::redistanceLevelSetToZeroPolygon(
        s.system.meshAccess(), s.dofs(), 0.0, phi, out);
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.diagnostic.find("2D"), std::string::npos) << result.diagnostic;
}

#endif
