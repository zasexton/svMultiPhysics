#include "LevelSet/LevelSetKinematicReconciliation.h"

#include "Basis/NodeOrderingConventions.h"
#include "Dofs/DofHandler.h"
#include "Dofs/EntityDofMap.h"
#include "Interfaces/LevelSetInterfaceBuilder.h"
#include "LevelSet/LevelSetInterfacePreservingRedistancing.h"
#include "LevelSet/LevelSetVolume.h"
#include "LevelSet/Maintenance/LevelSetMaintenancePipeline.h"
#include "Tests/Unit/TimeStepping/TimeSteppingTestHelpers.h"
#include "Spaces/H1Space.h"
#include "Spaces/SpaceFactory.h"
#include "Systems/FESystem.h"
#include "TimeStepping/GeneralizedAlpha.h"
#include "TimeStepping/TimeSteppingUtils.h"

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

    SimplexSystem(std::shared_ptr<svmp::Mesh> m, int dim, int velocity_order = 1)
        : mesh(std::move(m)), system(mesh), dimension(dim)
    {
        const auto type = dim == 2 ? FE::ElementType::Triangle3 : FE::ElementType::Tetra4;
        phi = system.addField(FE::systems::FieldSpec{
            .name = "phi",
            .space = std::make_shared<FE::spaces::H1Space>(type, /*order=*/1),
            .components = 1,
        });
        auto velocity_space =
            FE::spaces::VectorSpace(FE::spaces::SpaceType::H1, type, velocity_order, dim);
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
            phi_previous, phi_transported, w_previous, w_transported, reconciled,
            system.fieldRecord(velocity).space.get());
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

// One open simplex. Its negative-phase area for phi=x-a is a-a*a/2.
std::shared_ptr<svmp::Mesh> buildUnitTriangle()
{
    auto base = std::make_shared<svmp::MeshBase>();
    base->build_from_arrays(
        2, {0., 0., 1., 0., 0., 1.}, {0, 3}, {0, 1, 2},
        {svmp::CellShape{svmp::CellFamily::Triangle, 3, 1}});
    base->finalize();
    return svmp::create_mesh(std::move(base));
}

// The single cell uses the canonical reference-node ordering. A component
// block in the velocity field is one scalar node per local Lagrange basis.
std::vector<FE::Real> quadraticShear(const SimplexSystem& s)
{
    std::vector<FE::Real> velocity(s.velocityDofs().getNumDofs(), 0.);
    const auto& dofs = s.velocityDofs().getDofMap().getCellDofs(0);
    const auto type = s.dimension == 2 ? FE::ElementType::Triangle3 : FE::ElementType::Tetra4;
    const auto nodes = FE::basis::ReferenceNodeLayout::get_lagrange_node_coords(type, 2);
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const auto y = nodes[i][1];
        velocity[dofs[i]] = 4. * y * (1. - y);
    }
    return velocity;
}

} // namespace

TEST(LevelSetKinematicReconciliation, ResolvesStationaryFiniteStepAreaBeforePublishing)
{
    SimplexSystem s(buildUnitTriangle(), 2);
    const auto previous = s.scalar([](const Point& x) { return x[0] - 0.25; });
    const auto transported = s.scalar([](const Point& x) { return -0.24 + 0.79 * x[0] - 0.02 * x[1]; });
    const auto zero = s.vector([](const Point&) { return Point{}; });
    std::vector<FE::Real> endpoint;
    const auto result = s.reconcile(0.05, previous, transported, zero, zero, endpoint);
    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_TRUE(result.converged) << result.diagnostic;
    EXPECT_NEAR(s.negativeVolume(endpoint), 0.21875, 1.0e-11);
    EXPECT_NEAR(result.reconciled_volume_error, 0., 1.0e-11);
}

TEST(LevelSetKinematicReconciliation, LeavesOutputAloneWhenTheTransportedTopologyCannotReachTheFlux)
{
    SimplexSystem s(buildUnitTriangle(), 2);
    const auto previous = s.scalar([](const Point&) { return -1.; });
    const auto transported = s.scalar([](const Point&) { return 1.; });
    const auto zero = s.vector([](const Point&) { return Point{}; });
    std::vector<FE::Real> endpoint{3.};
    const auto result = s.reconcile(0.1, previous, transported, zero, zero, endpoint);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.applied);
    EXPECT_FALSE(result.volume_balance_satisfied);
    EXPECT_NEAR(result.reconciled_volume_error, -0.5, 1.0e-14);
    EXPECT_EQ(endpoint, (std::vector<FE::Real>{3.}));
}

TEST(LevelSetKinematicReconciliation, RedistributesTheFluxShareOfAnInterfaceVertex)
{
    SimplexSystem s(buildUnitTriangle(), 2);
    const auto phi = s.scalar([](const Point& x) { return x[0] - x[1]; });
    const auto velocity = s.vector([](const Point&) { return Point{1., 0., 0.}; });
    // The interface passes through the origin. Its value stays pinned and the
    // other two coefficients can rotate the segment to realize the outward flux.
    std::vector<FE::Real> output;
    const auto result = s.reconcile(0.01, phi, phi, velocity, velocity, output);
    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_DOUBLE_EQ(output[s.phiDofs().getEntityDofMap()->getVertexDofs(0).front()], 0.);
    EXPECT_GT(result.sign_preserving_redistributed_dofs, 0u);
    EXPECT_EQ(builderCutClasses(s, output), builderCutClasses(s, phi));
    EXPECT_NEAR(s.negativeVolume(output) - s.negativeVolume(phi),
                result.kinematic_volume_change, result.volume_tolerance);
}

TEST(LevelSetKinematicReconciliation, OppositeUnplacedFluxesOnDisconnectedCutsCannotCancel)
{
    auto mesh = std::make_shared<svmp::MeshBase>();
    const svmp::CellShape triangle{svmp::CellFamily::Triangle, 3, 1};
    mesh->build_from_arrays(2, {0., 0., 1., 0., 0., 1., 2., 0., 3., 0., 2., 1.},
                            {0, 3, 6}, {0, 1, 2, 3, 4, 5}, {triangle, triangle});
    mesh->finalize();
    SimplexSystem s(svmp::create_mesh(std::move(mesh)), 2);
    const auto phi = s.scalar([](const Point& x) { return -x[1]; });
    const auto velocity = s.vector([](const Point& x) { return Point{0., x[0] < 1.5 ? 1. : -1., 0.}; });
    const std::vector<FE::Real> sentinel{42.};
    auto output = sentinel;
    const auto result = s.reconcile(0.01, phi, phi, velocity, velocity, output);
    EXPECT_NEAR(result.kinematic_volume_change, 0., 1.e-15);
    EXPECT_FALSE(result.success);
    EXPECT_GT(result.sign_preserving_skipped_dofs, 0u);
    EXPECT_EQ(output, sentinel);
}

TEST(LevelSetKinematicReconciliation, UsesTheActualQuadraticVelocityOnTheReturnedInterface)
{
    SimplexSystem s(buildUnitTriangle(), 2, 2);
    const auto previous = s.scalar([](const Point& x) { return x[0] - 0.25; });
    // P2 divergence-free shear with zero corner velocities and nonzero edge modes.
    const auto velocity = quadraticShear(s);
    std::vector<FE::Real> endpoint;
    const auto result = s.reconcile(0.01, previous, previous, velocity, velocity, endpoint);
    ASSERT_TRUE(result.success) << result.diagnostic;
    ASSERT_TRUE(result.converged);
    EXPECT_NEAR(result.previous_interface_flux, 9.0 / 16.0, 2.0e-14);
    EXPECT_NEAR(result.transported_volume_error, -0.01 * (9.0 / 16.0), 2.0e-14);
    const auto& phi = s.phiDofs().getDofMap().getCellDofs(0);
    // One endpoint of the segment lies on y=0, the other on x+y=1.
    const auto y = endpoint[phi[1]] / (endpoint[phi[1]] - endpoint[phi[2]]);
    const auto flux = 2.0 * y * y - (4.0 / 3.0) * y * y * y;
    EXPECT_NEAR(result.current_interface_flux, flux, 2.0e-14);
    EXPECT_NEAR(result.kinematic_volume_change, 0.005 * (9.0 / 16.0 + flux), 2.0e-14);
}

TEST(LevelSetKinematicReconciliation, TwoGaussZerosDoNotCertifyAQuadraticTrace)
{
    SimplexSystem s(buildUnitTriangle(), 2, 2);
    const auto previous = s.scalar([](const Point& x) { return x[0] - 0.25; });
    const auto transported = s.scalar([](const Point& x) { return 2.0 * (x[0] - 0.4); });
    const auto velocity = [&](FE::Real height, FE::Real offset) {
        auto w = quadraticShear(s);
        const auto& dofs = s.velocityDofs().getDofMap().getCellDofs(0);
        const auto nodes = FE::basis::ReferenceNodeLayout::get_lagrange_node_coords(
            FE::ElementType::Triangle3, 2);
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            const auto t = nodes[i][1] / height;
            w[dofs[i]] = offset + 1.0 - 6.0 * t + 6.0 * t * t;
        }
        return w;
    };
    std::vector<FE::Real> endpoint;
    const auto result = s.reconcile(0.1, previous, transported, velocity(0.75, 3.0),
                                    velocity(0.6, 0.75), endpoint);
    // At either initial trace, the nonzero quadratic kinematic residual
    // vanishes at both degree-three Gauss points. Its integral alone is zero.
    EXPECT_NEAR(result.previous_interface_flux, 2.25, 1.0e-13);
    EXPECT_NEAR(result.transported_volume_error, -0.03375, 1.0e-13);
    EXPECT_LT(result.volume_tolerance, 1.0e-7);
    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_NEAR(result.reconciled_volume_error, 0.0, 1.0e-9);
}

TEST(LevelSetKinematicReconciliation, IntegratesQuadraticVelocityOnATetrahedralCut)
{
    auto mesh = std::make_shared<svmp::MeshBase>();
    mesh->build_from_arrays(
        3, {0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.},
        {0, 4}, {0, 1, 2, 3}, {svmp::CellShape{svmp::CellFamily::Tetra, 4, 1}});
    mesh->finalize();
    SimplexSystem s(svmp::create_mesh(std::move(mesh)), 3, 2);
    const auto previous = s.scalar([](const Point& x) { return x[0] - 0.25; });
    const auto velocity = quadraticShear(s);
    std::vector<FE::Real> endpoint;
    const auto result = s.reconcile(0.01, previous, previous, velocity, velocity, endpoint);
    ASSERT_TRUE(result.success) << result.diagnostic;
    // Integral of 4 y (1-y) over y,z >= 0, y+z <= 3/4.
    EXPECT_NEAR(result.previous_interface_flux, 45.0 / 256.0, 2.0e-14);
    EXPECT_NEAR(result.reconciled_volume_error, 0.0, 1.0e-12);
}

#if FE_HAS_EIGEN
TEST(LevelSetKinematicReconciliation, RedistancingIsJudgedByItsOwnVolumeAndDisplacement)
{
    // Interface-preserving redistancing moves crossings by the defect of its
    // zero polygon, so it changes the interface flux of the accepted step.
    // Judged by that flux against the transport tolerance (about 1e-10
    // relative) it was refused although it kept the volume to roundoff.  It
    // is judged by its own volume tolerance and by the raw displacement of
    // the redistancing instead.
    namespace ts = FE::timestepping;
    SimplexSystem s(buildStructuredTriangleMesh(14), 2);
    const auto previous = s.scalar([](const Point& x) {
        return ellipse(x) * (0.4 + x[0] + 0.3 * x[1]);
    });
    const auto velocity = s.vector([](const Point& x) { return Point{0.2 * x[0] * x[0], 0., 0.}; });
    const auto factory = ts::test::createTestFactory();
    auto history = ts::TimeHistory::allocate(*factory, s.system.dofHandler().getNumDofs(), 1);
    std::vector<FE::Real> state(s.system.dofHandler().getNumDofs(), 0.);
    std::copy(previous.begin(), previous.end(), state.begin() + s.system.fieldDofOffset(s.phi));
    std::copy(velocity.begin(), velocity.end(), state.begin() + s.system.fieldDofOffset(s.velocity));
    ts::test::setVectorByDof(history.u(), state);
    ts::test::setVectorByDof(history.uPrev(), state);
    history.setStepIndex(1);
    history.setTime(0.001);
    history.setDt(0.001);
    auto config = std::make_shared<level_set::LevelSetMaintenanceConfiguration>();
    config->transport.level_set.field_name = "phi";
    config->transport.velocity.field_name = "w";
    config->transport.kinematic_reconciliation.enabled = true;
    config->transport.interface_preserving_redistancing.enabled = true;
    std::vector<level_set::LevelSetMaintenanceRequest> requests{{.configuration = config}};
    requests.front().runtime.redistancing_accumulated_displacement = 1.;
    const auto baseline = level_set::LevelSetKinematicBaseline{.level_set_field=s.phi, .velocity_field=s.velocity, .step=0, .level_set=previous, .velocity=velocity};
    std::vector<FE::Real> physical_endpoint;
    int published_redistancings = 0;
    const level_set::LevelSetMaintenanceStageObserver observe =
        [&](level_set::MaintenanceSubstage stage, auto, std::span<const FE::Real> after) {
            if (stage == level_set::MaintenanceSubstage::GeometryReconciliation) {
                physical_endpoint.assign(after.begin(), after.end());
            }
            if (stage == level_set::MaintenanceSubstage::Reinitialization) {
                ++published_redistancings;
            }
        };
    ASSERT_TRUE(level_set::applyLevelSetMaintenance({.fe_system = &s.system}, history,
        requests, {}, observe, nullptr, nullptr, {&baseline, 1}));
    ASSERT_FALSE(physical_endpoint.empty());
    EXPECT_EQ(published_redistancings, 1);
    EXPECT_EQ(requests.front().runtime.redistancing_calls, 1u);
    EXPECT_EQ(requests.front().runtime.redistancing_accumulated_displacement, 0.);
    const auto offset = static_cast<std::ptrdiff_t>(s.system.fieldDofOffset(s.phi));
    const auto published = ts::test::getVectorByDof(history.u());
    const std::vector<FE::Real> reconciled(physical_endpoint.begin() + offset,
                                           physical_endpoint.begin() + offset + static_cast<std::ptrdiff_t>(previous.size()));
    const std::vector<FE::Real> redistanced(published.begin() + offset,
                                            published.begin() + offset + static_cast<std::ptrdiff_t>(previous.size()));
    EXPECT_NE(redistanced, reconciled);
    EXPECT_NEAR(s.negativeVolume(redistanced), s.negativeVolume(reconciled), 1.0e-13);
    // Its crossings stay closer to the reconciled interface than those of
    // the raw redistancing ...
    std::vector<FE::Real> raw;
    ASSERT_TRUE(level_set::redistanceLevelSetToZeroPolygon(s.system.meshAccess(), s.phiDofs(), 0.0,
                                                           reconciled, raw).success);
    const auto& mesh = s.system.meshAccess();
    const auto moved = level_set::measureLevelSetZeroPolygonChange(mesh, s.phiDofs(), 0.0, reconciled, redistanced);
    const auto moved_raw = level_set::measureLevelSetZeroPolygonChange(mesh, s.phiDofs(), 0.0, reconciled, raw);
    EXPECT_GT(moved.max_crossing_shift, 0.0);
    EXPECT_LE(moved.max_crossing_shift, moved_raw.max_crossing_shift);
    // ... while the flux they carry differs by more than the transport
    // tolerance allows for the step's volume change.
    const auto flux = [&](const std::vector<FE::Real>& phi) {
        return level_set::measureLevelSetKinematicEndpoint(mesh, s.phiDofs(), s.velocityDofs(),
            *s.system.fieldRecord(s.velocity).space, 0.0, 1.0e-12, phi, velocity).interface_flux;
    };
    std::vector<FE::Real> scratch;
    const auto transport = s.reconcile(0.001, previous, previous, velocity, velocity, scratch);
    ASSERT_TRUE(transport.success) << transport.diagnostic;
    EXPECT_GT(0.5 * 0.001 * std::abs(flux(redistanced) - flux(reconciled)), 1.0e3 * transport.volume_tolerance);
}

TEST(LevelSetKinematicReconciliation, RedistancingKeepsTheAcceptedFieldWhenItsRestorationFails)
{
    // Without a step to integrate over (dt = 0) the volume restoration of the
    // redistanced candidate cannot close.  Redistancing only changes the
    // representation, so the accepted field and its accumulated motion stay.
    namespace ts = FE::timestepping;
    SimplexSystem s(buildStructuredTriangleMesh(14), 2);
    const auto previous = s.scalar([](const Point& x) {
        return ellipse(x) * (0.4 + x[0] + 0.3 * x[1]);
    });
    const auto factory = ts::test::createTestFactory();
    auto history = ts::TimeHistory::allocate(*factory, s.system.dofHandler().getNumDofs(), 1);
    std::vector<FE::Real> state(s.system.dofHandler().getNumDofs(), 0.);
    std::copy(previous.begin(), previous.end(), state.begin() + s.system.fieldDofOffset(s.phi));
    ts::test::setVectorByDof(history.u(), state);
    ts::test::setVectorByDof(history.uPrev(), state);
    history.setStepIndex(1);
    history.setTime(0.001);
    history.setDt(0.0);
    auto config = std::make_shared<level_set::LevelSetMaintenanceConfiguration>();
    config->transport.level_set.field_name = "phi";
    config->transport.velocity.field_name = "w";
    config->transport.interface_preserving_redistancing.enabled = true;
    std::vector<level_set::LevelSetMaintenanceRequest> requests{{.configuration = config}};
    requests.front().runtime.redistancing_accumulated_displacement = 1.;
    int published_redistancings = 0;
    const level_set::LevelSetMaintenanceStageObserver observe =
        [&](level_set::MaintenanceSubstage stage, auto, std::span<const FE::Real>) {
            if (stage == level_set::MaintenanceSubstage::Reinitialization) {
                ++published_redistancings;
            }
        };
    std::vector<FE::LogRecord> logs;
    EXPECT_FALSE(level_set::applyLevelSetMaintenance({.fe_system = &s.system}, history, requests, {},
                                                     observe, &logs));
    EXPECT_EQ(published_redistancings, 0);
    EXPECT_EQ(requests.front().runtime.redistancing_calls, 0u);
    EXPECT_EQ(requests.front().runtime.redistancing_accumulated_displacement, 1.);
    EXPECT_EQ(ts::test::getVectorByDof(history.u()), state);
}
#endif

#if FE_HAS_EIGEN
TEST(LevelSetKinematicReconciliation, RedistancingCarriesTheRateIntoItsRepresentation)
{
    namespace ts = FE::timestepping;
    // phi = 2 (x - x0 - c t) moves at speed c with rate -2c.  Redistancing
    // publishes x - x0 - c t, whose rate is -c.  Kept at -2c, the next
    // generalized-alpha step (rho_inf = 0.5) advances the plane by 1.2 c dt.
    SimplexSystem s(buildStructuredTriangleMesh(12), 2);
    const FE::Real c = 0.3;
    const FE::Real dt = 0.01;
    const FE::Real x0 = 0.4173;
    const auto doubled = s.scalar([&](const Point& x) { return 2.0 * (x[0] - x0); });
    const auto velocity = s.vector([&](const Point&) { return Point{c, 0.0, 0.0}; });
    const auto n = static_cast<std::size_t>(s.system.dofHandler().getNumDofs());
    const auto phi = static_cast<std::size_t>(s.system.fieldDofOffset(s.phi));
    const auto w = static_cast<std::size_t>(s.system.fieldDofOffset(s.velocity));
    std::vector<FE::Real> state(n, 0.0), previous(n, 0.0), rate(n, 0.0);
    for (std::size_t i = 0; i < doubled.size(); ++i) {
        state[phi + i] = doubled[i];
        previous[phi + i] = doubled[i] + 2.0 * c * dt;
        rate[phi + i] = -2.0 * c;
    }
    for (std::size_t i = 0; i < velocity.size(); ++i) state[w + i] = previous[w + i] = velocity[i];
    const auto factory = ts::test::createTestFactory();
    auto history = ts::TimeHistory::allocate(*factory, s.system.dofHandler().getNumDofs(), 1, true);
    ts::test::setVectorByDof(history.u(), state);
    ts::test::setVectorByDof(history.uPrev(), previous);
    ts::test::setVectorByDof(history.uDot(), rate);
    history.setStepIndex(1);
    history.setTime(dt);
    history.setDt(dt);
    auto config = std::make_shared<level_set::LevelSetMaintenanceConfiguration>();
    config->transport.level_set.field_name = "phi";
    config->transport.velocity.field_name = "w";
    config->transport.interface_preserving_redistancing.enabled = true;
    std::vector<level_set::LevelSetMaintenanceRequest> requests{{.configuration = config}};
    requests.front().runtime.redistancing_accumulated_displacement = 1.;
    ASSERT_TRUE(level_set::applyLevelSetMaintenance({.fe_system = &s.system}, history, requests));
    ASSERT_EQ(requests.front().runtime.redistancing_calls, 1u);
    const auto u = ts::test::getVectorByDof(history.u());
    const auto u_dot = ts::test::getVectorByDof(history.uDot());
    const auto u_prev = ts::test::getVectorByDof(history.uPrev());

    // The stage stencil of the injected-rate first-order generalized-alpha
    // scheme, a0 u_stage + a1 u_n + a2 u_dot_n, and the transport residual
    // c d(phi)/dx = c of the redistanced plane.
    const auto p = ts::utils::generalizedAlphaFirstOrderFromRhoInf(0.5);
    const ts::GeneralizedAlphaFirstOrderIntegrator integrator(ts::GeneralizedAlphaFirstOrderIntegratorOptions{
        .alpha_m = p.alpha_m, .alpha_f = p.alpha_f, .gamma = p.gamma, .history_rate_order = 0});
    FE::systems::SystemStateView view;
    view.dt = dt;
    const std::array<std::span<const FE::Real>, 2> slots{};
    view.u_history = slots;
    const auto stencil = integrator.buildContext(1, view).dt1->a;
    for (std::size_t i = phi; i < phi + doubled.size(); ++i) {
        const FE::Real x = doubled[i - phi] / 2.0 + x0;
        EXPECT_NEAR(u[i], x - x0, 1.0e-14);
        EXPECT_NEAR(u_dot[i], -c, 1.0e-12);
        EXPECT_NEAR(u[i] - u_prev[i], -c * dt, 1.0e-14);
        const FE::Real stage = -(stencil[1] * u[i] + stencil[2] * u_dot[i] + c) / stencil[0];
        EXPECT_NEAR((stage - u[i]) / p.alpha_f, -c * dt, 1.0e-12 * c * dt);
    }
    for (std::size_t i = w; i < w + velocity.size(); ++i) EXPECT_EQ(u_dot[i], 0.0);
}
#endif

#if FE_HAS_EIGEN
TEST(LevelSetKinematicReconciliation, RedistancingTriggerSeesHigherOrderVelocityModes)
{
    // A P2 transport velocity that vanishes at every vertex but is (1, 1) at
    // every edge node.  The redistancing trigger sampled vertex velocities
    // only and accumulated no interface motion.
    namespace ts = FE::timestepping;
    SimplexSystem s(buildStructuredTriangleMesh(8), 2, /*velocity_order=*/2);
    const FE::Real dt = 1.0e-3;
    const auto phi = s.scalar([](const Point& x) { return x[0] - 0.4173; });
    std::vector<FE::Real> velocity(static_cast<std::size_t>(s.velocityDofs().getNumDofs()), 1.0);
    const auto* map = s.velocityDofs().getEntityDofMap();
    for (FE::GlobalIndex v = 0; v < map->numVertices(); ++v) {
        for (const auto dof : map->getVertexDofs(v)) velocity[static_cast<std::size_t>(dof)] = 0.0;
    }
    std::vector<FE::Real> state(static_cast<std::size_t>(s.system.dofHandler().getNumDofs()), 0.0);
    std::copy(phi.begin(), phi.end(), state.begin() + s.system.fieldDofOffset(s.phi));
    std::copy(velocity.begin(), velocity.end(), state.begin() + s.system.fieldDofOffset(s.velocity));
    const auto factory = ts::test::createTestFactory();
    auto history = ts::TimeHistory::allocate(*factory, s.system.dofHandler().getNumDofs(), 1);
    ts::test::setVectorByDof(history.u(), state);
    ts::test::setVectorByDof(history.uPrev(), state);
    history.setStepIndex(1);
    history.setTime(dt);
    history.setDt(dt);
    auto config = std::make_shared<level_set::LevelSetMaintenanceConfiguration>();
    config->transport.level_set.field_name = "phi";
    config->transport.velocity.field_name = "w";
    config->transport.interface_preserving_redistancing.enabled = true;
    std::vector<level_set::LevelSetMaintenanceRequest> requests{{.configuration = config}};
    (void)level_set::applyLevelSetMaintenance({.fe_system = &s.system}, history, requests);
    EXPECT_EQ(requests.front().runtime.redistancing_calls, 0u);
    EXPECT_NEAR(requests.front().runtime.redistancing_accumulated_displacement, std::sqrt(2.0) * dt, 1.0e-15);
}
#endif

#if FE_HAS_EIGEN
TEST(LevelSetKinematicReconciliation, PatchBoundsKeepAnIslandCarriedBeyondItsPatch)
{
    // One liquid node translated exactly by three cell widths in one step.
    // Around its new position the previous patch was all gas, so the bound
    // clamped the exact value -1 to +1 and erased the island.  Beyond a
    // one-ring Courant number of one the previous patch range is no bound.
    namespace ts = FE::timestepping;
    SimplexSystem s(buildStructuredTriangleMesh(10), 2);
    const FE::Real dt = 1.0;
    const auto at = [](const Point& x, FE::Real cx) {
        return std::abs(x[0] - cx) < 1.0e-12 && std::abs(x[1] - 0.5) < 1.0e-12;
    };
    const auto previous = s.scalar([&](const Point& x) { return at(x, 0.3) ? -1.0 : 1.0; });
    const auto transported = s.scalar([&](const Point& x) { return at(x, 0.6) ? -1.0 : 1.0; });
    const auto velocity = s.vector([](const Point&) { return Point{0.3, 0.0, 0.0}; });
    std::vector<FE::Real> state(static_cast<std::size_t>(s.system.dofHandler().getNumDofs()), 0.0);
    std::copy(transported.begin(), transported.end(), state.begin() + s.system.fieldDofOffset(s.phi));
    std::copy(velocity.begin(), velocity.end(), state.begin() + s.system.fieldDofOffset(s.velocity));
    const auto factory = ts::test::createTestFactory();
    auto history = ts::TimeHistory::allocate(*factory, s.system.dofHandler().getNumDofs(), 1);
    ts::test::setVectorByDof(history.u(), state);
    ts::test::setVectorByDof(history.uPrev(), state);
    history.setStepIndex(1);
    history.setTime(dt);
    history.setDt(dt);
    auto config = std::make_shared<level_set::LevelSetMaintenanceConfiguration>();
    config->transport.level_set.field_name = "phi";
    config->transport.velocity.field_name = "w";
    config->transport.sign_definite_patch_bounds.enabled = true;
    std::vector<level_set::LevelSetMaintenanceRequest> requests{{.configuration = config}};
    const auto baseline = level_set::LevelSetKinematicBaseline{
        .level_set_field = s.phi, .velocity_field = s.velocity, .step = 0, .level_set = previous, .velocity = velocity};
    (void)level_set::applyLevelSetMaintenance({.fe_system = &s.system}, history, requests, {}, {}, nullptr,
                                              nullptr, {&baseline, 1});
    const auto u = ts::test::getVectorByDof(history.u());
    const auto offset = static_cast<std::size_t>(s.system.fieldDofOffset(s.phi));
    for (std::size_t i = 0; i < transported.size(); ++i) {
        EXPECT_EQ(u[offset + i], transported[i]) << "dof " << i;
    }
}
#endif

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
    EXPECT_TRUE(result.converged) << result.diagnostic;
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
    EXPECT_TRUE(result.converged) << result.diagnostic;
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
    EXPECT_TRUE(result.converged) << result.diagnostic;
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
    auto reconciled = transported;
    const auto result = s.reconcile(dt, previous, transported, zero, zero, reconciled);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.applied);
    EXPECT_TRUE(result.converged) << result.diagnostic;
    EXPECT_GT(std::abs(result.reconciled_volume_error), result.volume_tolerance);
    EXPECT_EQ(reconciled, transported);
    EXPECT_EQ(result.sign_preserving_redistributed_dofs, 9u);
    EXPECT_GT(std::abs(result.transported_volume_error), 1.0e-2);
    // The whole interface is blocked here, so only part of the change can be
    // undone without moving the zero set across the column.
    EXPECT_LT(std::abs(result.reconciled_volume_error),
              0.75 * std::abs(result.transported_volume_error));
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
    auto reconciled = transported;
    const auto result = s.reconcile(dt, previous, transported, zero, zero, reconciled);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.applied);
    EXPECT_TRUE(result.converged) << result.diagnostic;
    EXPECT_GT(std::abs(result.reconciled_volume_error), result.volume_tolerance);
    EXPECT_EQ(reconciled, transported);
    EXPECT_GE(result.degeneracy_frozen_dofs, 5u);
    const auto* map = s.phiDofs().getEntityDofMap();
    for (FE::GlobalIndex v = 0; v < map->numVertices(); ++v) {
        const auto dof = static_cast<std::size_t>(map->getVertexDofs(v).front());
        const auto x = s.system.meshAccess().getNodeCoordinates(v);
        if (std::abs(x[0] - 0.5) < 1.0e-12 &&
            static_cast<int>(std::lround(x[1]*8.0)) % 2 == 0) {
            EXPECT_TRUE(std::binary_search(result.degeneracy_frozen_dof_indices.begin(),
                                           result.degeneracy_frozen_dof_indices.end(), dof));
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
        EXPECT_TRUE(result.converged) << result.diagnostic;
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

    auto reconciled = transported;
    const auto result = s.reconcile(1.0, previous, transported, zero, zero, reconciled);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.applied);
    EXPECT_TRUE(result.converged) << result.diagnostic;
    EXPECT_GT(std::abs(result.reconciled_volume_error), result.volume_tolerance);
    EXPECT_EQ(reconciled, transported);
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
    EXPECT_TRUE(result.converged) << result.diagnostic;
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
    // pruning threshold too. A second, stationary interface near x=0.9
    // has no connection to that cut-cell patch.
    const auto gap = [&](const Point& x) {
        return (std::abs(x[1] - 0.5) < 1.0e-9 ? 1.6e-3 : 3.0e-3) * h;
    };
    const auto previous = s.scalar([&](const Point& x) {
        return x[0] < 0.7 ? x[0] - 0.5 + 0.75 * gap(x) : 0.9 - x[0];
    });
    const auto transported = s.scalar([&](const Point& x) {
        return x[0] < 0.7 ? x[0] - 0.5 + gap(x) : 0.9 - x[0];
    });
    const auto zero = s.vector([](const Point&) { return Point{0.0, 0.0, 0.0}; });
    auto reconciled = transported;
    const auto result = s.reconcile(1.0, previous, transported, zero, zero, reconciled);
    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_TRUE(result.volume_balance_satisfied);
    EXPECT_EQ(result.degeneracy_frozen_dofs, 20u);
    EXPECT_EQ(result.sign_preserving_skipped_dofs, 0u);
    EXPECT_NEAR(s.negativeVolume(reconciled), s.negativeVolume(previous), result.volume_tolerance);
    for (const auto dof : result.degeneracy_frozen_dof_indices) {
        EXPECT_EQ(reconciled[dof], transported[dof]);
    }
    const auto* map = s.phiDofs().getEntityDofMap();
    for (FE::GlobalIndex v = 0; v < map->numVertices(); ++v) {
        if (s.system.meshAccess().getNodeCoordinates(v)[0] > 0.7) {
            const auto i = map->getVertexDofs(v).front();
            EXPECT_EQ(reconciled[i], transported[i]) << "stationary interface vertex " << v;
        }
    }
    const auto before = builderCutClasses(s, transported);
    const auto after = builderCutClasses(s, reconciled);
    ASSERT_EQ(before.size(), after.size());
    for (std::size_t c = 0; c < before.size(); ++c) {
        EXPECT_EQ(before[c], after[c]) << "cell " << c;
    }
}

namespace {

// Root of the P1 trace on the wall where coordinate `axis` is zero, measured
// along the other coordinate `along` of a 2D mesh.
FE::Real wallRoot(const SimplexSystem& s, const std::vector<FE::Real>& values, int axis, int along)
{
    const auto* map = s.phiDofs().getEntityDofMap();
    std::vector<std::pair<FE::Real, FE::Real>> trace;
    for (FE::GlobalIndex v = 0; v < map->numVertices(); ++v) {
        const auto x = s.system.meshAccess().getNodeCoordinates(v);
        if (x[static_cast<std::size_t>(axis)] == 0.0) {
            trace.emplace_back(x[static_cast<std::size_t>(along)],
                               values[static_cast<std::size_t>(map->getVertexDofs(v).front())]);
        }
    }
    std::sort(trace.begin(), trace.end());
    for (std::size_t k = 0; k + 1 < trace.size(); ++k) {
        const auto [a, fa] = trace[k];
        const auto [b, fb] = trace[k + 1];
        if ((fa < 0.0) != (fb < 0.0)) {
            return a + (b - a) * fa / (fa - fb);
        }
    }
    ADD_FAILURE() << "no root on the wall";
    return 0.0;
}

} // namespace

TEST(LevelSetKinematicReconciliation, KeepsAnExactlyTransportedWallRoot)
{
    // At rest the transported field tilts only off the floor: its floor trace
    // is exact.  Restoring the off-floor coefficient recovers both the area and
    // the wall trace; the floor root must not move to 0.2528 or 0.2472.
    SimplexSystem s(buildUnitTriangle(), 2);
    const auto previous = s.scalar([](const Point& x) { return x[0] - 0.25; });
    const auto zero = s.vector([](const Point&) { return Point{}; });
    for (const FE::Real epsilon : {0.01, -0.01}) {
        const auto transported = s.scalar([&](const Point& x) { return x[0] - 0.25 + epsilon * x[1]; });
        std::vector<FE::Real> endpoint;
        const auto result = s.reconcile(1.0, previous, transported, zero, zero, endpoint);
        ASSERT_TRUE(result.success) << result.diagnostic;
        EXPECT_EQ(result.wall_trace_dofs, 3u);
        EXPECT_NEAR(wallRoot(s, endpoint, 1, 0), 0.25, 1.0e-14);
        EXPECT_NEAR(s.negativeVolume(endpoint), 0.21875, 1.0e-14);
    }
}

TEST(LevelSetKinematicReconciliation, FollowsTheWallCharacteristicOfAStrainField)
{
    // u = (x, -y) is tangential on x = 0 and y = 0 and carries x + y - 1/4
    // exactly into exp(-t) x + exp(t) y - 1/4, so the wall roots obey s' = s
    // and s' = -s.  An exact endpoint must keep both.
    SimplexSystem s(buildUnitTriangle(), 2);
    const FE::Real dt = 0.25;
    const auto previous = s.scalar([](const Point& x) { return x[0] + x[1] - 0.25; });
    const auto transported = s.scalar([&](const Point& x) {
        return std::exp(-dt) * x[0] + std::exp(dt) * x[1] - 0.25;
    });
    const auto strain = s.vector([](const Point& x) { return Point{x[0], -x[1], 0.0}; });
    std::vector<FE::Real> endpoint;
    const auto result = s.reconcile(dt, previous, transported, strain, strain, endpoint);
    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_NEAR(wallRoot(s, endpoint, 1, 0), 0.25 * std::exp(dt), 1.0e-12);
    EXPECT_NEAR(wallRoot(s, endpoint, 0, 1), 0.25 * std::exp(-dt), 1.0e-12);
}

TEST(LevelSetKinematicReconciliation, AdvancesAWallRootPastASlopeChangeAtTheVertexBehindIt)
{
    // u = (1, 0) moves the floor root from 0.51 to 0.52.  The floor trace has
    // slope 1/2 behind the vertex at x = 1/2 and slope 1 ahead of it, so the
    // zero of the two characteristic vertex values would lag at 0.515625.
    SimplexSystem s(buildStructuredTriangleMesh(8), 2);
    const FE::Real dt = 0.01;
    const auto trace = [](FE::Real x) { return x >= 0.5 ? x - 0.51 : -0.01 + 0.5 * (x - 0.5); };
    const auto previous = s.scalar([&](const Point& x) { return trace(x[0]) + x[1]; });
    const auto transported = s.scalar([&](const Point& x) { return trace(x[0] - dt) + x[1]; });
    const auto shift = s.vector([](const Point&) { return Point{1.0, 0.0, 0.0}; });
    std::vector<FE::Real> endpoint;
    const auto result = s.reconcile(dt, previous, transported, shift, shift, endpoint);
    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_NEAR(wallRoot(s, endpoint, 1, 0), 0.52, 1.0e-12);
}

TEST(LevelSetKinematicReconciliation, KeepsTheWallCharacteristicUnderAnInteriorTransportError)
{
    // The same strain field on a mesh with interior trace nodes.  An interior
    // transport error that vanishes on both walls is removed by the interior
    // nodes alone; the wall roots stay on their characteristics.
    SimplexSystem s(buildStructuredTriangleMesh(8), 2);
    const FE::Real dt = 0.05;
    const auto exact = [](FE::Real t) {
        return [t](const Point& x) { return std::exp(-t) * x[0] + std::exp(t) * x[1] - 0.5; };
    };
    const auto previous = s.scalar(exact(0.0));
    const auto transported = s.scalar([&](const Point& x) { return exact(dt)(x) + 0.02 * x[0] * x[1]; });
    const auto strain = s.vector([](const Point& x) { return Point{x[0], -x[1], 0.0}; });
    std::vector<FE::Real> endpoint;
    const auto result = s.reconcile(dt, previous, transported, strain, strain, endpoint);
    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_GT(std::abs(result.transported_volume_error), 1.0e-6);
    EXPECT_LE(std::abs(result.reconciled_volume_error), result.volume_tolerance);
    EXPECT_NEAR(wallRoot(s, endpoint, 1, 0), 0.5 * std::exp(dt), 1.0e-12);
    EXPECT_NEAR(wallRoot(s, endpoint, 0, 1), 0.5 * std::exp(-dt), 1.0e-12);
}

TEST(LevelSetKinematicReconciliation, KeepsAnExactWallTraceOnTetrahedra)
{
    // At rest an interior transport error that vanishes on the cube surface
    // leaves every wall trace exact.  The interior nodes alone restore the
    // volume; every root on a boundary edge stays.
    SimplexSystem s(buildStructuredTetraMesh(4), 3);
    const auto plane = [](const Point& x) { return 0.48 * x[0] + 0.6 * x[1] + 0.64 * x[2] - 0.8713; };
    const auto previous = s.scalar(plane);
    const auto transported = s.scalar([&](const Point& x) {
        return plane(x) + 0.8 * x[0] * (1 - x[0]) * x[1] * (1 - x[1]) * x[2] * (1 - x[2]);
    });
    const auto zero = s.vector([](const Point&) { return Point{}; });
    std::vector<FE::Real> endpoint;
    const auto result = s.reconcile(1.0, previous, transported, zero, zero, endpoint);
    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_GT(result.wall_trace_dofs, 0u);
    EXPECT_EQ(result.wall_trace_unresolved_dofs, 0u);
    EXPECT_GT(std::abs(result.transported_volume_error), 1.0e-5);
    EXPECT_NEAR(s.negativeVolume(endpoint), s.negativeVolume(previous), result.volume_tolerance);
    // Edges lying in a face of the cube.
    const auto& mesh = s.system.meshAccess();
    const auto* map = s.phiDofs().getEntityDofMap();
    std::size_t roots = 0u;
    std::vector<FE::GlobalIndex> nodes;
    mesh.forEachOwnedCell([&](FE::GlobalIndex cell) {
        mesh.getCellNodes(cell, nodes);
        for (std::size_t a = 0; a < 4u; ++a) {
            for (std::size_t b = a + 1; b < 4u; ++b) {
                const auto xa = mesh.getNodeCoordinates(nodes[a]);
                const auto xb = mesh.getNodeCoordinates(nodes[b]);
                bool on_face = false;
                for (std::size_t d = 0; d < 3u; ++d) {
                    on_face = on_face || (xa[d] == xb[d] && (xa[d] == 0.0 || xa[d] == 1.0));
                }
                const auto i = static_cast<std::size_t>(map->getVertexDofs(nodes[a]).front());
                const auto j = static_cast<std::size_t>(map->getVertexDofs(nodes[b]).front());
                if (!on_face || (transported[i] < 0.0) == (transported[j] < 0.0)) continue;
                ++roots;
                EXPECT_NEAR(endpoint[i] / (endpoint[i] - endpoint[j]),
                            transported[i] / (transported[i] - transported[j]), 1.0e-14);
            }
        }
    });
    EXPECT_GT(roots, 0u);
}

TEST(LevelSetKinematicReconciliation, KeepsTheVolumeOfEachDisconnectedRegion)
{
    // A diamond of four triangles around its centre holds two liquid tips that
    // share the centre, north and south coefficients; a ring of gas cells
    // keeps both away from the walls.  At rest, perturbing only the east tip
    // must restore it and leave the west tip alone.  Through the shared nodes
    // the west tip used to gain area while the exact-trace remainder of the
    // changed cells widened the accepted volume error.
    auto mesh = std::make_shared<svmp::MeshBase>();
    const svmp::CellShape triangle{svmp::CellFamily::Triangle, 3, 1};
    const std::vector<svmp::real_t> x{0., 0., 1., 0., 0., 1., -1., 0., 0., -1., 2., 0., 2., 2.,
                                      0., 2., -2., 2., -2., 0., -2., -2., 0., -2., 2., -2.};
    const std::vector<svmp::index_t> cells{0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1,
                                           1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7, 8, 2, 8, 3, 3, 8, 9,
                                           3, 9, 10, 3, 10, 4, 4, 10, 11, 4, 11, 12, 4, 12, 1, 1, 12, 5};
    std::vector<svmp::offset_t> offsets;
    for (std::size_t c = 0; c <= cells.size() / 3u; ++c) offsets.push_back(static_cast<svmp::offset_t>(3u * c));
    mesh->build_from_arrays(2, x, offsets, cells, std::vector<svmp::CellShape>(cells.size() / 3u, triangle));
    mesh->finalize();
    SimplexSystem s(svmp::create_mesh(std::move(mesh)), 2);
    const auto tip_node = [](const Point& p) { return std::abs(p[0]) == 1.0 && p[1] == 0.0; };
    const auto previous = s.scalar([&](const Point& p) { return tip_node(p) ? -1.0 : 1.0; });
    const auto transported = s.scalar([&](const Point& p) {
        return tip_node(p) ? (p[0] > 0.0 ? -0.9 : -1.0) : 1.0;
    });
    const auto zero = s.vector([](const Point&) { return Point{}; });
    // The area of one tip: the other tip's node is moved to the positive side.
    const auto tip = [&](const std::vector<FE::Real>& phi, FE::Real side) {
        auto alone = phi;
        const auto* map = s.phiDofs().getEntityDofMap();
        for (FE::GlobalIndex v = 0; v < map->numVertices(); ++v) {
            const auto p = s.system.meshAccess().getNodeCoordinates(v);
            if (tip_node(p) && p[0] == -side) {
                alone[static_cast<std::size_t>(map->getVertexDofs(v).front())] = 1.0;
            }
        }
        return s.negativeVolume(alone);
    };
    const FE::Real area = tip(previous, 1.0);
    ASSERT_NEAR(tip(previous, -1.0), area, 1.0e-15);
    std::vector<FE::Real> endpoint;
    const auto result = s.reconcile(1.0, previous, transported, zero, zero, endpoint);
    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.wall_trace_dofs, 0u);
    EXPECT_NEAR(tip(endpoint, 1.0), area, 1.0e-12);
    EXPECT_NEAR(tip(endpoint, -1.0), area, 1.0e-12);
    EXPECT_LT(result.volume_tolerance, 1.0e-10);
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
