#include "LevelSet/LevelSetKinematicReconciliation.h"

#include "Basis/NodeOrderingConventions.h"
#include "Dofs/DofHandler.h"
#include "Dofs/EntityDofMap.h"
#include "Interfaces/LevelSetInterfaceBuilder.h"
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

// Cut class of every owned cell as the generated-interface builder sees it:
// kind, degeneracy and vertex count of each fragment, then side, full-cell
// flag and pruning (volume fraction below 1e-8) of each volume region.  The
// builder cuts the cell on its reference element, as the generated-interface
// lifecycle does.
std::vector<std::vector<long>> builderCutClasses(const SimplexSystem& s,
                                                 const std::vector<FE::Real>& phi_values)
{
    namespace interfaces = svmp::FE::interfaces;
    interfaces::CutInterfaceDomainRequest request;
    request.source = interfaces::LevelSetInterfaceSource::fromEvaluator(
        "kinematic-reconciliation-test", 1u, 1u);
    request.interface_marker = 1;
    request.isovalue = 0.0;
    request.tolerance = 1.0e-12;
    request.quadrature_order = 1;
    const auto type = s.dimension == 2 ? FE::ElementType::Triangle3 : FE::ElementType::Tetra4;
    const auto corners = static_cast<std::size_t>(s.dimension + 1);
    const auto* map = s.phiDofs().getEntityDofMap();
    const auto& mesh = s.system.meshAccess();
    std::vector<std::vector<long>> classes;
    std::vector<FE::GlobalIndex> nodes;
    mesh.forEachOwnedCell([&](FE::GlobalIndex cell) {
        mesh.getCellNodes(cell, nodes);
        interfaces::LevelSetCellCutInput input;
        input.parent_cell = static_cast<decltype(input.parent_cell)>(cell);
        input.element_type = type;
        for (std::size_t i = 0; i < corners; ++i) {
            const auto xi = FE::basis::ReferenceNodeLayout::get_node_coords(type, i);
            input.node_coordinates.push_back({{xi[0], xi[1], xi[2]}});
            input.level_set_values.push_back(phi_values[static_cast<std::size_t>(
                map->getVertexDofs(nodes[i]).front())]);
        }
        const auto cut = s.dimension == 2 ? interfaces::cutLinearLevelSetCell2D(request, input)
                                          : interfaces::cutLinearLevelSetCell3D(request, input);
        std::vector<long> c;
        for (const auto& fragment : cut.fragments) {
            c.push_back(static_cast<long>(fragment.kind));
            c.push_back(static_cast<long>(fragment.degeneracy));
            c.push_back(static_cast<long>(fragment.vertices.size()));
        }
        c.push_back(-1);
        for (const auto& region : cut.volume_regions) {
            c.push_back(static_cast<long>(region.side));
            c.push_back(region.full_cell_equivalent ? 1 : 0);
            c.push_back(region.volume_fraction > 0.0 && region.volume_fraction < 1.0e-8 ? 1 : 0);
        }
        classes.push_back(std::move(c));
    });
    return classes;
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
    // The transported interface passes 1.5e-7 (1.2e-6 cell widths) from the
    // column x = 0.5.  The correction toward the previous state would cross
    // the column, so its nodes may move only halfway.  Where a column vertex
    // is the lone positive corner of a triangle (every other row, by the
    // alternating diagonals), its cut segment has length 1.2e-6 or 1.7e-6 on
    // the reference triangle, where the generated-interface builder measures
    // it, so it is regular (> sqrt(1e-12)); halving would make it nearly
    // tangent, so those vertices keep their value.  The others move halfway.
    // (The physical length, 1.5e-7, is nearly tangent before and after.)
    const FE::Real gap = 1.5e-7;
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
    const auto before = builderCutClasses(s, transported);
    const auto after = builderCutClasses(s, reconciled);
    ASSERT_EQ(before.size(), after.size());
    for (std::size_t c = 0; c < before.size(); ++c) {
        EXPECT_EQ(before[c], after[c]) << "cell " << c;
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

TEST(LevelSetKinematicReconciliation, KeepsTheBuilderCutClassOfTetrahedra)
{
    SimplexSystem s(buildStructuredTetraMesh(6), 3);
    const FE::Real h = 1.0 / 6.0;
    // The transported plane passes gap = 1.6e-3 h from the vertex plane
    // x = 0.5 and the correction moves it to 0.6 gap.  A tetrahedron whose
    // only positive corner lies on x = 0.5 has a corner fragment of reference
    // area 0.5 t^2 or 0.87 t^2 (t = gap / h): 1.3e-6 or 2.2e-6, regular,
    // before, and 4.6e-7 or 8.0e-7, nearly tangent (<= sqrt(1e-12)), after.
    // Its physical area, about 5e-8, is nearly tangent in both states, so a
    // guard on physical measures would let the builder's class change.
    const FE::Real gap = 1.6e-3 * h;
    const auto previous = s.scalar([&](const Point& x) { return x[0] - 0.5 + 0.6 * gap; });
    const auto transported = s.scalar([&](const Point& x) { return x[0] - 0.5 + gap; });
    const auto zero = s.vector([](const Point&) { return Point{0.0, 0.0, 0.0}; });
    const auto before = builderCutClasses(s, transported);
    // The unguarded correction (the previous state itself) changes classes.
    const auto unguarded = builderCutClasses(s, previous);
    ASSERT_EQ(before.size(), unguarded.size());
    std::size_t changed = 0u;
    for (std::size_t c = 0; c < before.size(); ++c) {
        changed += before[c] != unguarded[c] ? 1u : 0u;
    }
    ASSERT_GT(changed, 0u);

    std::vector<FE::Real> reconciled;
    const auto result = s.reconcile(1.0, previous, transported, zero, zero, reconciled);
    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_GT(result.degeneracy_frozen_dofs, 0u);
    const auto after = builderCutClasses(s, reconciled);
    ASSERT_EQ(before.size(), after.size());
    for (std::size_t c = 0; c < before.size(); ++c) {
        EXPECT_EQ(before[c], after[c]) << "cell " << c;
    }
}

TEST(LevelSetKinematicReconciliation, RedistributesTheShareOfFrozenNodes)
{
    SimplexSystem s(buildStructuredTetraMesh(6), 3);
    const FE::Real h = 1.0 / 6.0;
    // A plane 0.3 h from the vertex plane x = 0.5, except that the vertex
    // (0.5, 0.5, 0.5) lies only 1.6e-3 h from the interface.  The previous
    // state is lower by a quarter of that distance.  Lowering the vertex
    // changes the class of the corner fragments it carries (positive volume
    // fraction below 1e-8), so it and the other nodes of those cells are
    // frozen.  The volume they would have moved goes to the other nodes of
    // their cells, so the measure is restored as if nothing were frozen.
    const FE::Real near = 1.6e-3 * h;
    const auto centre = [](const Point& x) {
        return std::abs(x[0] - 0.5) + std::abs(x[1] - 0.5) + std::abs(x[2] - 0.5) < 1.0e-9;
    };
    const auto transported = s.scalar([&](const Point& x) {
        return centre(x) ? near : x[0] - 0.5 + 0.3 * h;
    });
    std::vector<FE::Real> previous = transported;
    for (auto& value : previous) {
        value -= 0.25 * near;
    }
    const auto zero = s.vector([](const Point&) { return Point{0.0, 0.0, 0.0}; });
    std::vector<FE::Real> reconciled;
    const auto result = s.reconcile(1.0, previous, transported, zero, zero, reconciled);
    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.degeneracy_frozen_dofs, 0u);
    EXPECT_GE(result.sign_preserving_redistributed_dofs, result.degeneracy_frozen_dofs);
    EXPECT_GT(std::abs(result.transported_volume_error), 1.0e-5);
    // Dropping the frozen share would leave about 5% of the error.
    EXPECT_LT(std::abs(result.reconciled_volume_error),
              1.0e-6 * std::abs(result.transported_volume_error));
    for (const auto dof : result.degeneracy_frozen_dof_indices) {
        EXPECT_EQ(reconciled[dof], transported[dof]) << "dof " << dof;
    }
    const auto before = builderCutClasses(s, transported);
    const auto after = builderCutClasses(s, reconciled);
    ASSERT_EQ(before.size(), after.size());
    for (std::size_t c = 0; c < before.size(); ++c) {
        EXPECT_EQ(before[c], after[c]) << "cell " << c;
    }
}

TEST(LevelSetKinematicReconciliation, FreezingDoesNotSpreadThroughRedistribution)
{
    SimplexSystem s(buildStructuredTetraMesh(6), 3);
    const FE::Real h = 1.0 / 6.0;
    // Every vertex of the plane x = 0.5 lies close to the interface: the row
    // y = 0.5 at 1.6e-3 h, the others at 3e-3 h, and the correction lowers
    // each to 3/4 of its distance.  Only cells at the near row change class,
    // but moving their volume into the other rows would push those below the
    // pruning threshold too.  The addition is withheld there, so exactly the
    // nodes that freezing alone would freeze stay frozen and every other node
    // keeps its own correction.
    const auto gap = [&](const Point& x) {
        return (std::abs(x[1] - 0.5) < 1.0e-9 ? 1.6e-3 : 3.0e-3) * h;
    };
    const auto previous = s.scalar([&](const Point& x) { return x[0] - 0.5 + 0.75 * gap(x); });
    const auto transported = s.scalar([&](const Point& x) { return x[0] - 0.5 + gap(x); });
    const auto zero = s.vector([](const Point&) { return Point{0.0, 0.0, 0.0}; });
    std::vector<FE::Real> reconciled;
    const auto result = s.reconcile(1.0, previous, transported, zero, zero, reconciled);
    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.degeneracy_frozen_dofs, 20u);
    EXPECT_LT(std::abs(result.reconciled_volume_error),
              0.12 * std::abs(result.transported_volume_error));
    const auto before = builderCutClasses(s, transported);
    const auto after = builderCutClasses(s, reconciled);
    ASSERT_EQ(before.size(), after.size());
    for (std::size_t c = 0; c < before.size(); ++c) {
        EXPECT_EQ(before[c], after[c]) << "cell " << c;
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
