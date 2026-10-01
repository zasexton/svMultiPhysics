#include "LevelSet/LevelSetKinematicReconciliation.h"

#include "Dofs/DofHandler.h"
#include "Dofs/EntityDofMap.h"
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
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace FE = svmp::FE;
namespace level_set = svmp::FE::level_set;

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

using Point = std::array<FE::Real, 3>;
using ScalarFunction = std::function<FE::Real(const Point&)>;
using VectorFunction = std::function<Point(const Point&)>;

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
    // Kuhn tetrahedra: paths from corner 0 to corner 7 of the unit cube.
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
                    // Positive orientation: swap two vertices if needed.
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

struct SimplexSystem {
    std::shared_ptr<svmp::Mesh> mesh;
    FE::systems::FESystem system;
    FE::FieldId phi{FE::INVALID_FIELD_ID};
    FE::FieldId velocity{FE::INVALID_FIELD_ID};
    int dimension{2};

    SimplexSystem(std::shared_ptr<svmp::Mesh> m, int dim)
        : mesh(std::move(m)), system(mesh), dimension(dim)
    {
        const auto type = dim == 2 ? FE::ElementType::Triangle3 : FE::ElementType::Tetra4;
        phi = system.addField(FE::systems::FieldSpec{
            .name = "phi",
            .space = std::make_shared<FE::spaces::H1Space>(type, /*order=*/1),
            .components = 1,
        });
        auto velocity_space =
            FE::spaces::VectorSpace(FE::spaces::SpaceType::H1, type, /*order=*/1, dim);
        velocity = system.addField(FE::systems::FieldSpec{
            .name = "w",
            .space = velocity_space,
            .components = velocity_space->value_dimension(),
        });
        system.setup();
    }

    [[nodiscard]] const FE::dofs::DofHandler& phiDofs() const
    {
        return system.fieldDofHandler(phi);
    }
    [[nodiscard]] const FE::dofs::DofHandler& velocityDofs() const
    {
        return system.fieldDofHandler(velocity);
    }

    [[nodiscard]] std::vector<FE::Real> scalar(const ScalarFunction& f) const
    {
        const auto* map = phiDofs().getEntityDofMap();
        std::vector<FE::Real> out(static_cast<std::size_t>(phiDofs().getNumDofs()), 0.0);
        for (FE::GlobalIndex v = 0; v < map->numVertices(); ++v) {
            out[static_cast<std::size_t>(map->getVertexDofs(v).front())] =
                f(system.meshAccess().getNodeCoordinates(v));
        }
        return out;
    }

    [[nodiscard]] std::vector<FE::Real> vector(const VectorFunction& f) const
    {
        const auto* map = velocityDofs().getEntityDofMap();
        std::vector<FE::Real> out(static_cast<std::size_t>(velocityDofs().getNumDofs()), 0.0);
        for (FE::GlobalIndex v = 0; v < map->numVertices(); ++v) {
            const auto value = f(system.meshAccess().getNodeCoordinates(v));
            const auto dofs = map->getVertexDofs(v);
            for (int d = 0; d < dimension; ++d) {
                out[static_cast<std::size_t>(dofs[static_cast<std::size_t>(d)])] =
                    value[static_cast<std::size_t>(d)];
            }
        }
        return out;
    }

    [[nodiscard]] level_set::LevelSetKinematicReconciliationResult reconcile(
        FE::Real dt,
        const std::vector<FE::Real>& phi_previous,
        const std::vector<FE::Real>& phi_transported,
        const std::vector<FE::Real>& w_previous,
        const std::vector<FE::Real>& w_transported,
        std::vector<FE::Real>& reconciled) const
    {
        return level_set::reconcileLevelSetWithKinematicFlux(
            system.meshAccess(), phiDofs(), velocityDofs(),
            /*isovalue=*/0.0, /*tolerance=*/1.0e-12, dt,
            phi_previous, phi_transported, w_previous, w_transported, reconciled);
    }

    [[nodiscard]] FE::Real negativeVolume(const std::vector<FE::Real>& phi_values) const
    {
        const auto result = level_set::computeLevelSetCutCellVolume(
            system.meshAccess(), phiDofs(), level_set::LevelSetVolumeOptions{}, phi_values);
        EXPECT_TRUE(result.success) << result.diagnostic;
        return result.negative_volume;
    }
};

FE::Real ellipse(const Point& x)
{
    // Distorted ellipse centred off the mesh symmetry lines.
    const FE::Real dx = (x[0] - 0.513) / 0.31;
    const FE::Real dy = (x[1] - 0.472) / 0.22;
    return 0.25 * (std::sqrt(dx * dx + dy * dy) - 1.0);
}

} // namespace

TEST(LevelSetKinematicReconciliation, LeavesAConsistentTranslationUnchanged)
{
    SimplexSystem s(buildStructuredTriangleMesh(12), 2);
    const Point w{0.3, -0.2, 0.0};
    const FE::Real dt = 0.01;
    // A planar level set translated by a uniform velocity satisfies the
    // kinematic condition pointwise, so nothing may change.
    const auto plane = [](const Point& x) { return 0.8 * x[0] + 0.6 * x[1] - 0.5371; };
    const auto previous = s.scalar(plane);
    const auto transported = s.scalar([&](const Point& x) {
        return plane(x) - dt * (0.8 * w[0] + 0.6 * w[1]);
    });
    const auto velocity = s.vector([&](const Point&) { return w; });
    std::vector<FE::Real> reconciled;
    const auto result = s.reconcile(dt, previous, transported, velocity, velocity, reconciled);
    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.previous_interface_cells, 0u);
    EXPECT_LT(result.max_abs_correction, 1.0e-14);
    EXPECT_NEAR(result.transported_volume_error, 0.0, 1.0e-13);
    EXPECT_NEAR(result.reconciled_volume_error, 0.0, 1.0e-13);
    ASSERT_EQ(reconciled.size(), transported.size());
    for (std::size_t i = 0; i < reconciled.size(); ++i) {
        EXPECT_NEAR(reconciled[i], transported[i], 1.0e-14);
    }
}

TEST(LevelSetKinematicReconciliation, RemovesALocalVolumeErrorWithoutAGlobalShift)
{
    SimplexSystem s(buildStructuredTriangleMesh(16), 2);
    const FE::Real h = 1.0 / 16.0;
    const FE::Real dt = 1.0e-3;
    const auto previous = s.scalar(ellipse);
    // Zero velocity: any change of the liquid measure is a transport error.
    // Perturb only the right half of the interface band.
    const auto transported = s.scalar([&](const Point& x) {
        const FE::Real bump = x[0] > 0.6 ? 0.03 * h : 0.0;
        return ellipse(x) + bump;
    });
    const auto zero = s.vector([](const Point&) { return Point{0.0, 0.0, 0.0}; });
    std::vector<FE::Real> reconciled;
    const auto result = s.reconcile(dt, previous, transported, zero, zero, reconciled);
    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.sign_preserving_skipped_dofs, 0u);
    EXPECT_EQ(result.sign_preserving_redistributed_dofs, 0u);
    EXPECT_NEAR(result.kinematic_volume_change, 0.0, 1.0e-15);
    EXPECT_GT(std::abs(result.transported_volume_error), 1.0e-5);
    EXPECT_LT(std::abs(result.reconciled_volume_error),
              1.0e-3 * std::abs(result.transported_volume_error));
    EXPECT_NEAR(s.negativeVolume(reconciled), s.negativeVolume(previous),
                1.0e-3 * std::abs(result.transported_volume_error));

    // Locality: nodes away from both interfaces keep their transported value,
    // and the left half, which was transported consistently, barely moves.
    const auto* map = s.phiDofs().getEntityDofMap();
    for (FE::GlobalIndex v = 0; v < map->numVertices(); ++v) {
        const auto dof = static_cast<std::size_t>(map->getVertexDofs(v).front());
        const auto x = s.system.meshAccess().getNodeCoordinates(v);
        if (std::abs(ellipse(x)) > 0.75 * h * 4.0) {
            EXPECT_EQ(reconciled[dof], transported[dof]);
        }
        if (x[0] < 0.4) {
            EXPECT_NEAR(reconciled[dof], transported[dof], 1.0e-12);
        }
    }
    EXPECT_LE(result.max_abs_correction, 0.03 * h * (1.0 + 1.0e-9));
}

TEST(LevelSetKinematicReconciliation, UsesTheKinematicFluxOfARigidRotation)
{
    SimplexSystem s(buildStructuredTriangleMesh(20), 2);
    const FE::Real dt = 2.0e-3;
    const FE::Real omega = 1.7;
    const Point centre{0.49, 0.53, 0.0};
    const auto rotation = [&](const Point& x) {
        return Point{-omega * (x[1] - centre[1]), omega * (x[0] - centre[0]), 0.0};
    };
    const auto rotated = [&](FE::Real angle) {
        return [=](const Point& x) {
            // ellipse evaluated at the point rotated back by angle
            const FE::Real c = std::cos(angle);
            const FE::Real sn = std::sin(angle);
            const FE::Real rx = x[0] - centre[0];
            const FE::Real ry = x[1] - centre[1];
            return ellipse(Point{centre[0] + c * rx + sn * ry,
                                 centre[1] - sn * rx + c * ry, 0.0});
        };
    };
    // The nodal interpolant of the exactly rotated field is not an exact P1
    // transport step, so its sharp measure drifts; a rigid rotation has zero
    // kinematic flux through any closed interface.
    const auto previous = s.scalar(rotated(0.0));
    const auto transported = s.scalar(rotated(omega * dt));
    const auto velocity = s.vector(rotation);
    std::vector<FE::Real> reconciled;
    const auto result = s.reconcile(dt, previous, transported, velocity, velocity, reconciled);
    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.previous_interface_flux, 0.0, 1.0e-13);
    EXPECT_NEAR(result.current_interface_flux, 0.0, 1.0e-13);
    EXPECT_GT(std::abs(result.transported_volume_error), 1.0e-8);
    EXPECT_LT(std::abs(result.reconciled_volume_error),
              1.0e-2 * std::abs(result.transported_volume_error));
}

TEST(LevelSetKinematicReconciliation, PreservesTheSignClassOfEveryNode)
{
    SimplexSystem s(buildStructuredTriangleMesh(8), 2);
    const FE::Real dt = 1.0;
    const auto previous = s.scalar([](const Point& x) { return x[0] - 0.51; });
    // A uniform rise of 0.02 with zero velocity moves the column x = 0.5 from
    // -0.01 to +0.01.  Undoing it would flip those nodes: they move only
    // halfway to the isovalue, and the rest of their share of the volume
    // change goes to the neighbouring columns.
    const auto transported = s.scalar([](const Point& x) { return x[0] - 0.49; });
    const auto zero = s.vector([](const Point&) { return Point{0.0, 0.0, 0.0}; });
    std::vector<FE::Real> reconciled;
    const auto result = s.reconcile(dt, previous, transported, zero, zero, reconciled);
    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.sign_preserving_redistributed_dofs, 9u);
    EXPECT_GT(std::abs(result.transported_volume_error), 1.0e-2);
    // The whole interface is blocked here, so only part of the change can be
    // undone without moving the zero set across the column.
    EXPECT_LT(std::abs(result.reconciled_volume_error),
              0.75 * std::abs(result.transported_volume_error));
    const auto* map = s.phiDofs().getEntityDofMap();
    for (FE::GlobalIndex v = 0; v < map->numVertices(); ++v) {
        const auto dof = static_cast<std::size_t>(map->getVertexDofs(v).front());
        const auto x = s.system.meshAccess().getNodeCoordinates(v);
        EXPECT_EQ(std::signbit(reconciled[dof]), std::signbit(transported[dof]));
        EXPECT_GE(std::abs(reconciled[dof]), 0.5 * std::abs(transported[dof]) - 1.0e-15);
        if (std::abs(x[0] - 0.5) < 1.0e-12) {
            EXPECT_NEAR(reconciled[dof], 0.5 * transported[dof], 1.0e-15);
        }
    }
}

TEST(LevelSetKinematicReconciliation, KeepsTheDegeneracyClassOfEveryCut)
{
    SimplexSystem s(buildStructuredTriangleMesh(8), 2);
    const FE::Real dt = 1.0;
    // The transported interface passes 1.8e-6 from the column x = 0.5.  The
    // correction toward the previous state would cross the column, so its
    // nodes may move only halfway.  Where a column vertex is the lone positive
    // corner of a triangle (every other row, by the alternating diagonals),
    // its cut segment has length 1.8e-6 > sqrt(1e-12); halving would make it
    // nearly tangent, so those vertices keep their value.  The others move
    // halfway.
    const FE::Real gap = 1.8e-6;
    const auto previous = s.scalar([](const Point& x) { return x[0] - 0.51; });
    const auto transported = s.scalar([&](const Point& x) { return x[0] - 0.5 + gap; });
    const auto zero = s.vector([](const Point&) { return Point{0.0, 0.0, 0.0}; });
    std::vector<FE::Real> reconciled;
    const auto result = s.reconcile(dt, previous, transported, zero, zero, reconciled);
    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_GE(result.degeneracy_frozen_dofs, 5u);
    const auto* map = s.phiDofs().getEntityDofMap();
    for (FE::GlobalIndex v = 0; v < map->numVertices(); ++v) {
        const auto dof = static_cast<std::size_t>(map->getVertexDofs(v).front());
        const auto x = s.system.meshAccess().getNodeCoordinates(v);
        if (std::abs(x[0] - 0.5) < 1.0e-12) {
            const auto row = static_cast<int>(std::lround(x[1] * 8.0));
            if (row % 2 == 0) {
                EXPECT_EQ(reconciled[dof], transported[dof]) << "row " << row;
            } else {
                EXPECT_NEAR(reconciled[dof], 0.5 * transported[dof], 1.0e-18) << "row " << row;
            }
        }
    }
}

TEST(LevelSetKinematicReconciliation, ReconcilesTetrahedralCuts)
{
    SimplexSystem s(buildStructuredTetraMesh(6), 3);
    const FE::Real h = 1.0 / 6.0;
    const FE::Real dt = 1.0e-3;
    const auto sphere = [](const Point& x) {
        const FE::Real dx = x[0] - 0.47;
        const FE::Real dy = x[1] - 0.52;
        const FE::Real dz = x[2] - 0.49;
        return std::sqrt(dx * dx + dy * dy + dz * dz) - 0.31;
    };
    // Consistent uniform translation of a plane: no change.
    {
        const Point w{0.2, -0.1, 0.3};
        const auto plane = [](const Point& x) { return 0.48 * x[0] + 0.6 * x[1] + 0.64 * x[2] - 0.8713; };
        const auto previous = s.scalar(plane);
        const auto transported = s.scalar([&](const Point& x) {
            return plane(x) - dt * (0.48 * w[0] + 0.6 * w[1] + 0.64 * w[2]);
        });
        const auto velocity = s.vector([&](const Point&) { return w; });
        std::vector<FE::Real> reconciled;
        const auto result = s.reconcile(dt, previous, transported, velocity, velocity, reconciled);
        ASSERT_TRUE(result.success) << result.diagnostic;
        EXPECT_GT(result.previous_interface_cells, 0u);
        EXPECT_LT(result.max_abs_correction, 1.0e-14);
        // The cut volume of a cube is cubic in the plane offset, so the
        // trapezoidal flux differs from it at third order in dt.
        EXPECT_NEAR(result.transported_volume_error, 0.0, 1.0e-10);
    }
    // A local perturbation at rest is removed from the measure.
    {
        const auto previous = s.scalar(sphere);
        const auto transported = s.scalar([&](const Point& x) {
            return sphere(x) + (x[2] > 0.55 ? 0.02 * h : 0.0);
        });
        const auto zero = s.vector([](const Point&) { return Point{0.0, 0.0, 0.0}; });
        std::vector<FE::Real> reconciled;
        const auto result = s.reconcile(dt, previous, transported, zero, zero, reconciled);
        ASSERT_TRUE(result.success) << result.diagnostic;
        EXPECT_TRUE(result.converged);
        EXPECT_GT(std::abs(result.transported_volume_error), 1.0e-6);
        EXPECT_LT(std::abs(result.reconciled_volume_error),
                  1.0e-3 * std::abs(result.transported_volume_error));
    }
}

TEST(LevelSetKinematicReconciliation, RejectsNonSimplexCellsAndBadInput)
{
    // Quadrilateral background mesh.
    auto base = std::make_shared<svmp::MeshBase>();
    const std::vector<svmp::real_t> x_ref = {0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0};
    const std::vector<svmp::offset_t> offsets = {0, 4};
    const std::vector<svmp::index_t> cell2vertex = {0, 1, 2, 3};
    svmp::CellShape shape{};
    shape.family = svmp::CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(2, x_ref, offsets, cell2vertex, {shape});
    base->finalize();
    auto mesh = svmp::create_mesh(std::move(base));
    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = std::make_shared<FE::spaces::H1Space>(FE::ElementType::Quad4, 1),
        .components = 1,
    });
    auto velocity_space =
        FE::spaces::VectorSpace(FE::spaces::SpaceType::H1, FE::ElementType::Quad4, 1, 2);
    const auto velocity = system.addField(FE::systems::FieldSpec{
        .name = "w", .space = velocity_space, .components = 2});
    system.setup();
    const auto& phi_dofs = system.fieldDofHandler(phi);
    const auto& w_dofs = system.fieldDofHandler(velocity);
    std::vector<FE::Real> p(static_cast<std::size_t>(phi_dofs.getNumDofs()), 0.1);
    p[0] = -0.1;
    std::vector<FE::Real> w(static_cast<std::size_t>(w_dofs.getNumDofs()), 0.0);
    std::vector<FE::Real> out;
    auto result = level_set::reconcileLevelSetWithKinematicFlux(
        system.meshAccess(), phi_dofs, w_dofs, 0.0, 1.0e-12, 0.1, p, p, w, w, out);
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.diagnostic.find("Triangle3"), std::string::npos) << result.diagnostic;

    result = level_set::reconcileLevelSetWithKinematicFlux(
        system.meshAccess(), phi_dofs, w_dofs, 0.0, 1.0e-12, 0.0, p, p, w, w, out);
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.diagnostic.find("time step"), std::string::npos) << result.diagnostic;
}

#endif
