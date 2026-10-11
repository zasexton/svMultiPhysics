#include "LevelSet/LevelSetInterfacePreservingRedistancing3D.h"

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
#include <iostream>
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
std::shared_ptr<svmp::Mesh> buildStructuredTriangleMesh2D(int n)
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


// A spherical cap of radius R on the bottom wall y = 0 with contact angle
// theta (through the liquid, phi < 0).
FE::Real cap(const Point& x, FE::Real theta)
{
    constexpr FE::Real R = 0.35;
    const FE::Real cy = -R * std::cos(theta);
    return std::sqrt((x[0] - 0.5031) * (x[0] - 0.5031) + (x[1] - cy) * (x[1] - cy) +
                     (x[2] - 0.4973) * (x[2] - 0.4973)) - R;
}

// Length-weighted mean through-liquid angle (degrees) of the P1 interface in
// the tetrahedra with a cut face on y = 0: cos(theta) = n_y.
FE::Real meanWallCellAngle(const ScalarSystem& s, const std::vector<FE::Real>& phi)
{
    const auto& access = s.system.meshAccess();
    FE::Real sum = 0.0;
    FE::Real count = 0.0;
    std::vector<FE::GlobalIndex> nodes;
    access.forEachCell([&](FE::GlobalIndex cell) {
        access.getCellNodes(cell, nodes);
        const auto* map = s.dofs().getEntityDofMap();
        std::array<Point, 4> x{};
        std::array<FE::Real, 4> f{};
        int on_wall = 0;
        bool below = false;
        bool above = false;
        for (std::size_t k = 0; k < 4u; ++k) {
            x[k] = access.getNodeCoordinates(nodes[k]);
            f[k] = phi[static_cast<std::size_t>(map->getVertexDofs(nodes[k]).front())];
            if (std::abs(x[k][1]) < 1.0e-14) {
                ++on_wall;
                below = below || f[k] < 0.0;
                above = above || f[k] >= 0.0;
            }
        }
        if (on_wall < 3 || !(below && above)) {
            return;
        }
        // Gradient of the linear interpolant: solve J^T g = df.
        FE::Real m[3][4];
        for (std::size_t r = 0; r < 3u; ++r) {
            for (std::size_t c = 0; c < 3u; ++c) {
                m[r][c] = x[r + 1u][c] - x[0][c];
            }
            m[r][3] = f[r + 1u] - f[0];
        }
        for (std::size_t p = 0; p < 3u; ++p) {
            std::size_t best = p;
            for (std::size_t r = p + 1u; r < 3u; ++r) {
                if (std::abs(m[r][p]) > std::abs(m[best][p])) {
                    best = r;
                }
            }
            for (std::size_t c = 0; c < 4u; ++c) {
                std::swap(m[p][c], m[best][c]);
            }
            for (std::size_t r = 0; r < 3u; ++r) {
                if (r != p) {
                    const FE::Real factor = m[r][p] / m[p][p];
                    for (std::size_t c = 0; c < 4u; ++c) {
                        m[r][c] -= factor * m[p][c];
                    }
                }
            }
        }
        const FE::Real g[3] = {m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2]};
        const FE::Real norm = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
        sum += std::acos(std::clamp(g[1] / norm, -1.0, 1.0)) * 180.0 / 3.14159265358979323846;
        count += 1.0;
    });
    EXPECT_GT(count, 0.0);
    return sum / count;
}

} // namespace

TEST(LevelSetInterfacePreservingRedistancing3D, ScaledPlaneBecomesItsExactDistance)
{
    ScalarSystem s(buildStructuredTetraMesh(6), 3);
    // A tilted plane crossing four faces of the cube, scaled by 2.5.
    const auto plane = [](const Point& x) {
        return 0.48 * (x[0] - 0.437) + 0.6 * (x[1] - 0.511) + 0.64 * (x[2] - 0.389);
    };
    const auto scaled = s.scalar([&](const Point& x) { return 2.5 * plane(x); });
    const auto exact = s.scalar(plane);
    std::vector<FE::Real> out;
    const auto result = level_set::redistanceLevelSetToZeroSurface(
        s.system.meshAccess(), s.dofs(), 0.0, scaled, out);
    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_TRUE(result.applied);
    EXPECT_GT(result.interface_segments, 0u);
    EXPECT_GT(result.open_ends, 0u);
    for (std::size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(out[i], exact[i], 1.0e-13) << "dof " << i;
    }
    const auto change = level_set::measureLevelSetZeroSurfaceChange(
        s.system.meshAccess(), s.dofs(), 0.0, scaled, out);
    ASSERT_TRUE(change.success) << change.diagnostic;
    EXPECT_LT(change.max_crossing_shift, 1.0e-14);
    EXPECT_LT(change.max_open_end_shift, 1.0e-14);
    EXPECT_EQ(change.cells_with_changed_cut, 0u);
    EXPECT_EQ(change.open_ends_first, result.open_ends);
    std::vector<FE::Real> again;
    const auto second = level_set::redistanceLevelSetToZeroSurface(
        s.system.meshAccess(), s.dofs(), 0.0, out, again);
    ASSERT_TRUE(second.success);
    EXPECT_LT(second.max_abs_change, 1.0e-13);
}

TEST(LevelSetInterfacePreservingRedistancing3D,
     ContactCapKeepsSignsContactLineAndAngleAndRestoresVolume)
{
    constexpr FE::Real R = 0.35;  // cap radius of cap()
    for (const FE::Real degrees : {60.0, 120.0}) {
        std::array<FE::Real, 2> raw_shift{};
        for (std::size_t level = 0; level < 2u; ++level) {
            const int n = level == 0u ? 12 : 24;
            const FE::Real h = 1.0 / n;
            ScalarSystem s(buildStructuredTetraMesh(n), 3);
            const FE::Real theta = degrees * 3.14159265358979323846 / 180.0;
            // A distorted field with the zero set of the cap: steeper near
            // the wall, flatter above it, on a scale the meshes resolve.
            const auto distorted = s.scalar([&](const Point& x) {
                return cap(x, theta) * (0.7 + 0.6 * std::exp(-4.0 * x[1]));
            });
            std::vector<FE::Real> out;
            const auto result = level_set::redistanceLevelSetToZeroSurface(
                s.system.meshAccess(), s.dofs(), 0.0, distorted, out);
            ASSERT_TRUE(result.success) << result.diagnostic;
            EXPECT_GT(result.open_ends, 0u) << degrees;
            for (std::size_t i = 0; i < out.size(); ++i) {
                ASSERT_EQ(out[i] < 0.0, distorted[i] < 0.0) << "sign of dof " << i;
            }
            const auto change = level_set::measureLevelSetZeroSurfaceChange(
                s.system.meshAccess(), s.dofs(), 0.0, distorted, out);
            ASSERT_TRUE(change.success) << change.diagnostic;
            EXPECT_EQ(change.cells_with_changed_cut, 0u);
            EXPECT_EQ(change.open_ends_first, change.open_ends_second);
            // The raw redistancing moves the crossings by the second-order
            // defect of the faceted surface, O(h^2/R).
            raw_shift[level] = change.max_crossing_shift;
            EXPECT_LT(change.max_crossing_shift, 2.0 * h * h / R) << degrees << " n=" << n;
            // Zero-velocity reconciliation restores the interface and the volume.
            auto velocity_space = FE::spaces::VectorSpace(
                FE::spaces::SpaceType::H1, FE::ElementType::Tetra4, /*order=*/1, 3);
            FE::systems::FESystem vsys(s.mesh);
            const auto u = vsys.addField(FE::systems::FieldSpec{
                .name = "u", .space = velocity_space,
                .components = velocity_space->value_dimension()});
            vsys.setup();
            const std::vector<FE::Real> zero(
                static_cast<std::size_t>(vsys.fieldDofHandler(u).getNumDofs()), 0.0);
            // The driver's restoration: reconcile at zero velocity; revert the
            // cell closure of every frozen DOF to its input value and repeat
            // until nothing new is reverted.
            std::vector<FE::Real> restored;
            std::vector<FE::Real> candidate = out;
            std::size_t passes = 0u;
            std::size_t reverted = 0u;
            level_set::LevelSetKinematicReconciliationResult restoration{};
            for (;;) {
                ++passes;
                restoration = level_set::reconcileLevelSetWithKinematicFlux(
                    s.system.meshAccess(), s.dofs(), vsys.fieldDofHandler(u), 0.0, 1.0e-12,
                    1.0, distorted, candidate, zero, zero, restored);
                ASSERT_TRUE(restoration.success) << restoration.diagnostic;
                if (!restoration.applied) {
                    restored = candidate;
                }
                bool changed = false;
                if (!restoration.degeneracy_frozen_dof_indices.empty()) {
                    for (const auto i : level_set::levelSetDofCellClosure3D(
                             s.system.meshAccess(), s.dofs(),
                             restoration.degeneracy_frozen_dof_indices)) {
                        if (candidate[i] != distorted[i]) {
                            candidate[i] = distorted[i];
                            ++reverted;
                            changed = true;
                        }
                    }
                }
                if (!changed) {
                    break;
                }
            }
            std::cout << "[ redistancing 3D ] restoration passes=" << passes
                      << " reverted=" << reverted << " converged=" << restoration.converged
                      << " iterations=" << restoration.iterations
                      << " frozen=" << restoration.degeneracy_frozen_dofs << std::endl;
            const auto v0 = s.negativeVolume(distorted);
            const auto v1 = s.negativeVolume(out);
            const auto v2 = s.negativeVolume(restored);
            EXPECT_LT(std::abs(v2 - v0), 0.1 * std::abs(v1 - v0) + 1.0e-12)
                << degrees << ": " << v0 << " " << v1 << " " << v2;
            const auto restored_change = level_set::measureLevelSetZeroSurfaceChange(
                s.system.meshAccess(), s.dofs(), 0.0, distorted, restored);
            ASSERT_TRUE(restored_change.success);
            EXPECT_EQ(restored_change.cells_with_changed_cut, 0u);
            // One large redistancing of a field far from a distance function
            // exceeds the restoration's iteration cap (the production calls
            // follow the one-cell trigger and converge), so the restored
            // crossings are only bounded here.
            EXPECT_LT(restored_change.max_crossing_shift, 0.2 * h) << degrees << " n=" << n;
            // The contact line and the discrete contact angle of the wall
            // cells stay put: the surface is continued across the wall at its
            // open edges, and the restoration takes back the crossings.
            EXPECT_LT(restored_change.max_open_end_shift, 0.05 * h) << degrees << " n=" << n;
            const auto angle_before = meanWallCellAngle(s, distorted);
            const auto angle_after = meanWallCellAngle(s, restored);
            std::cout << "[ redistancing 3D ] theta=" << degrees << " n=" << n
                      << " raw_crossing_shift/h=" << change.max_crossing_shift / h
                      << " raw_contact_line_shift/h=" << change.max_open_end_shift / h
                      << " restored_crossing_shift/h=" << restored_change.max_crossing_shift / h
                      << " restored_contact_line_shift/h=" << restored_change.max_open_end_shift / h
                      << " wall_cell_angle " << angle_before << " -> "
                      << meanWallCellAngle(s, out) << " (raw) -> " << angle_after
                      << " (restored)" << std::endl;
            // The angle of the distorted field's wall cells carries the P1
            // error of the distortion; the redistanced field must not move it
            // away from the cap angle.
            EXPECT_LT(std::abs(angle_after - degrees), std::abs(angle_before - degrees) + 1.0)
                << degrees << " n=" << n << ": " << angle_before << " -> " << angle_after;
            if (level == 0u) {
                // The closure of a vertex contains the vertices of its cells.
                const std::vector<std::size_t> seed{s.dofAt({0.5, 0.0, 0.5})};
                const auto closure = level_set::levelSetDofCellClosure3D(
                    s.system.meshAccess(), s.dofs(), seed);
                EXPECT_GT(closure.size(), 4u);
                EXPECT_TRUE(std::find(closure.begin(), closure.end(), seed.front()) !=
                            closure.end());
            }
        }
        // Second-order defect: halving h divides the raw shift by about 4.
        EXPECT_GT(raw_shift[0] / raw_shift[1], 2.5) << degrees;
    }
}

TEST(LevelSetInterfacePreservingRedistancing3D, MotionScaleSeesOnlyCutCellVertices)
{
    ScalarSystem s(buildStructuredTetraMesh(5), 3);
    const auto phi = s.scalar([](const Point& x) { return x[0] - 0.43; });
    auto velocity_space = FE::spaces::VectorSpace(
        FE::spaces::SpaceType::H1, FE::ElementType::Tetra4, /*order=*/1, 3);
    FE::systems::FESystem vsys(s.mesh);
    const auto u = vsys.addField(FE::systems::FieldSpec{
        .name = "u", .space = velocity_space,
        .components = velocity_space->value_dimension()});
    vsys.setup();
    const auto& udofs = vsys.fieldDofHandler(u);
    const auto* map = udofs.getEntityDofMap();
    std::vector<FE::Real> w(static_cast<std::size_t>(udofs.getNumDofs()), 0.0);
    FE::Real expected = 0.0;
    for (FE::GlobalIndex v = 0; v < map->numVertices(); ++v) {
        const auto x = s.system.meshAccess().getNodeCoordinates(v);
        // Large speed far from the interface, a known speed next to it.
        const bool near = std::abs(x[0] - 0.43) < 0.21;
        const FE::Real speed = near ? 0.2 + x[1] + x[2] : 50.0;
        if (near) {
            expected = std::max(expected, speed);
        }
        w[static_cast<std::size_t>(map->getVertexDofs(v)[2])] = speed;
    }
    const auto scale = level_set::levelSetInterfaceMotionScale3D(
        s.system.meshAccess(), s.dofs(), udofs, 0.0, phi, w);
    ASSERT_TRUE(scale.success) << scale.diagnostic;
    EXPECT_DOUBLE_EQ(scale.max_speed, expected);
    EXPECT_NEAR(scale.min_edge, 0.2, 1.0e-12);
}

TEST(LevelSetInterfacePreservingRedistancing3D, RejectsTrianglesOnEveryRank)
{
    ScalarSystem s(buildStructuredTriangleMesh2D(4), 2);
    const auto phi = s.scalar([](const Point& x) { return x[0] - 0.41; });
    std::vector<FE::Real> out;
    const auto result = level_set::redistanceLevelSetToZeroSurface(
        s.system.meshAccess(), s.dofs(), 0.0, phi, out);
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.diagnostic.find("3D"), std::string::npos) << result.diagnostic;
}

#endif
