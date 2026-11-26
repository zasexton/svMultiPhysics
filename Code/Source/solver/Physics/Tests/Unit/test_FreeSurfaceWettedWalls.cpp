/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

// Strong wall conditions of an unfitted free surface act on the wetted part
// of their walls (free-surface audit A27).

#include <gtest/gtest.h>

#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"

#include "FE/Assembly/CutIntegrationContext.h"
#include "FE/Backends/Interfaces/BackendFactory.h"
#include "FE/Backends/Interfaces/BackendKind.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Forms/Vocabulary.h"
#include "FE/Interfaces/GeneratedActiveBoundaryDomain.h"
#include "FE/Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"
#include "FE/LevelSet/LevelSetInterfaceLifecycle.h"
#include "FE/Spaces/SpaceFactory.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"
#include "FE/Systems/TimeIntegrator.h"
#include "FE/TimeStepping/TimeLoop.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Core/MeshBase.h"
#  include "Mesh/Fields/MeshFields.h"
#  include "Mesh/Mesh.h"
#  include "Mesh/Topology/CellShape.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace svmp::Physics::test {
namespace {

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

namespace ns = formulations::navier_stokes;

constexpr int kLeft = 41;
constexpr int kRight = 42;
constexpr int kBottom = 43;
constexpr int kTop = 44;
constexpr int kInterface = 45;
constexpr const char* kDomain = "translating_block";
constexpr int kNx = 8;
constexpr int kNy = 4;
constexpr double kH = 1.0 / kNx;

/// A liquid block rear < x < front, y < depth on the floor of the tank
/// [0, 1] x [0, 1/2], as the level set max(x - front, y - depth, rear - x).
struct Block {
    double rear{0.3};
    double front{0.95};
    double depth{0.3};

    [[nodiscard]] double phi(double x, double y) const { return std::max({x - front, y - depth, rear - x}); }
};

/// kNx x kNy square Q1 cells, the walls labelled by side, the vertex level
/// set of @p block.
std::shared_ptr<Mesh> makeTank(const Block& block)
{
    auto base = std::make_shared<MeshBase>();
    std::vector<real_t> coordinates;
    for (int j = 0; j <= kNy; ++j) {
        for (int i = 0; i <= kNx; ++i) {
            coordinates.push_back(static_cast<real_t>(i * kH));
            coordinates.push_back(static_cast<real_t>(j * kH));
        }
    }
    const auto vertex = [](int i, int j) { return static_cast<index_t>(j * (kNx + 1) + i); };
    std::vector<offset_t> offsets{0};
    std::vector<index_t> cells;
    for (int j = 0; j < kNy; ++j) {
        for (int i = 0; i < kNx; ++i) {
            for (const auto v : {vertex(i, j), vertex(i + 1, j), vertex(i + 1, j + 1), vertex(i, j + 1)}) {
                cells.push_back(v);
            }
            offsets.push_back(static_cast<offset_t>(cells.size()));
        }
    }
    const CellShape shape{CellFamily::Quad, 4, 1};
    base->build_from_arrays(2, coordinates, offsets, cells,
                            std::vector<CellShape>(static_cast<std::size_t>(kNx * kNy), shape));
    base->finalize();
    for (index_t face = 0; face < static_cast<index_t>(base->n_faces()); ++face) {
        const auto vertices = base->face_vertices(face);
        if (vertices.size() != 2u) {
            continue;
        }
        const auto on = [&](int c, double value) {
            return std::abs(coordinates[static_cast<std::size_t>(2 * vertices[0] + c)] - value) < 1e-12 &&
                   std::abs(coordinates[static_cast<std::size_t>(2 * vertices[1] + c)] - value) < 1e-12;
        };
        if (on(0, 0.0)) {
            base->set_boundary_label(face, kLeft);
        } else if (on(0, 1.0)) {
            base->set_boundary_label(face, kRight);
        } else if (on(1, 0.0)) {
            base->set_boundary_label(face, kBottom);
        } else if (on(1, kNy * kH)) {
            base->set_boundary_label(face, kTop);
        }
    }
    const auto handle = MeshFields::attach_field(*base, EntityKind::Vertex, "phi", FieldScalarType::Float64, 1);
    auto* phi = MeshFields::field_data_as<real_t>(*base, handle);
    for (index_t v = 0; v < static_cast<index_t>(base->n_vertices()); ++v) {
        phi[static_cast<std::size_t>(v)] = static_cast<real_t>(
            block.phi(coordinates[static_cast<std::size_t>(2 * v)], coordinates[static_cast<std::size_t>(2 * v + 1)]));
    }
    return create_mesh(std::move(base));
}

/// The generated active boundary (the wetted part on the liquid side) of
/// every wall, built as the geometry service builds it.
void addWettedWalls(FE::assembly::CutIntegrationContext& context,
                    const FE::interfaces::LevelSetInterfaceDomain& interface_domain,
                    const FE::assembly::IMeshAccess& mesh,
                    const Block& block)
{
    namespace interfaces = FE::interfaces;
    const auto& source = interface_domain.request();
    for (const int marker : {kLeft, kRight, kBottom, kTop}) {
        interfaces::GeneratedInterfaceBoundaryIntersectionRequest contact;
        contact.source = source.source;
        contact.generated_domain_id = kDomain;
        contact.interface_marker = kInterface;
        contact.boundary_marker = marker;
        contact.intersection_marker = interfaces::stableGeneratedInterfaceBoundaryIntersectionMarker(
            {.source = source.source, .domain_id = kDomain, .interface_marker = kInterface, .boundary_marker = marker});
        contact.tolerance = source.tolerance;
        contact.coefficient_classification_policy = source.coefficient_classification_policy;
        contact.quadrature_order = 2;
        contact.frame = source.frame;
        contact.mesh_geometry_revision = source.mesh_geometry_revision;
        contact.mesh_topology_revision = source.mesh_topology_revision;
        contact.ownership_revision = source.ownership_revision;
        contact.quadrature_policy_key = source.quadrature_policy_key;
        contact.source_value_revision = source.source.value_revision;
        const auto contact_domain =
            interfaces::buildGeneratedInterfaceBoundaryIntersectionDomain(std::move(contact), interface_domain, mesh);

        interfaces::GeneratedActiveBoundaryRequest active;
        active.source = source.source;
        active.generated_domain_id = kDomain;
        active.interface_marker = kInterface;
        active.boundary_marker = marker;
        active.side = FE::geometry::CutIntegrationSide::Negative;
        active.tolerance = source.tolerance;
        active.coefficient_classification_policy = source.coefficient_classification_policy;
        active.quadrature_order = 2;
        active.frame = source.frame;
        active.mesh_geometry_revision = source.mesh_geometry_revision;
        active.mesh_topology_revision = source.mesh_topology_revision;
        active.ownership_revision = source.ownership_revision;
        active.quadrature_policy_key = source.quadrature_policy_key;
        active.source_value_revision = source.source.value_revision;
        interfaces::GeneratedActiveBoundaryScalarField phi;
        phi.value_at_node = [&mesh, block](FE::GlobalIndex node) {
            const auto x = mesh.getNodeCoordinates(node);
            return block.phi(x[0], x[1]);
        };
        context.addGeneratedActiveBoundaryDomain(interfaces::buildGeneratedActiveBoundaryDomain(
            std::move(active), interface_domain, contact_domain, mesh, phi));
    }
}

struct FrontSample {
    double front_speed{0.0};            ///< min of u_x where the front crosses the wet edges
    double max_velocity_error{0.0};     ///< max |u - (U, 0)| over the wet nodes and the front
    double max_pressure{0.0};           ///< max |p| over the wet nodes
};

/// One generalized-alpha step of @p block translating at U = 1 toward the
/// slip wall x = 1 of a tank with slip walls (no gravity, a traction-free
/// surface at p = 0).  The exact solution keeps u = (U, 0) and p = 0 until
/// contact.  @p wetted_walls puts the generated active boundary of the walls
/// in the cut context, which restricts the strong wall rows to the wetted
/// faces; without it the rows cover every node of each wall (the operator
/// before A27).
FrontSample translateBlock(const Block& block, bool wetted_walls)
{
    constexpr double U = 1.0;
    auto mesh = makeTank(block);
    auto scalar = FE::spaces::SpaceFactory::create_h1(FE::ElementType::Quad4, 1);
    auto vector = FE::spaces::SpaceFactory::create_vector_h1(FE::ElementType::Quad4, 1, 2);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{.name = "phi", .space = scalar, .components = 1});
    system.addOperator("equations");
    {
        const auto state = FE::forms::StateField(phi, *scalar, "phi_block");
        const auto test = FE::forms::TestField(phi, *scalar, "eta_block");
        FE::systems::FormInstallOptions install;
        install.compiler_options.jit.enable = false;
        (void)FE::systems::installFormulation(system, "equations", {phi}, (FE::forms::dt(state) * test).dx(), install);
    }

    ns::IncompressibleNavierStokesVMSOptions options;
    options.velocity_field_name = "u";
    options.pressure_field_name = "p";
    options.density = 1.0;
    options.viscosity = 1.0e-3;   // the translation has no strain
    options.enable_convection = false;
    options.enable_vms = true;
    options.jit_policy.enable = false;
    using BC = ns::IncompressibleNavierStokesVMSOptions::VelocityDirichletBC;
    for (const int marker : {kLeft, kRight}) {
        options.velocity_dirichlet.push_back(BC{.boundary_marker = marker, .value = {0.0, 0.0, 0.0},
                                                .active_components = {true, false, false}});
    }
    for (const int marker : {kBottom, kTop}) {
        options.velocity_dirichlet.push_back(BC{.boundary_marker = marker, .value = {0.0, 0.0, 0.0},
                                                .active_components = {false, true, false}});
    }
    ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary surface;
    surface.implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet;
    surface.interface_marker = kInterface;
    surface.level_set_field_name = "phi";
    surface.generated_interface_domain_id = kDomain;
    surface.level_set_isovalue = 0.0;
    surface.active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative;
    surface.external_pressure = 0.0;
    surface.surface_tension = 0.0;
    surface.cut_cell_stabilization.enabled = false;   // no facet sets here; it vanishes on the translation
    surface.small_cut_aggregation = true;
    options.free_surface.push_back(std::move(surface));
    ns::IncompressibleNavierStokesVMSModule(vector, scalar, options).registerOn(system);
    system.setup({});

    const auto u = system.findFieldByName("u");
    const auto p = system.findFieldByName("p");
    const auto dof = [&](FE::FieldId field, int i, int j, std::size_t c) {
        const auto vertex = static_cast<FE::GlobalIndex>(j * (kNx + 1) + i);
        const auto local = system.fieldDofHandler(field).getEntityDofMap()->getVertexDofs(vertex);
        return static_cast<std::size_t>(system.fieldDofOffset(field) + local[c]);
    };
    std::vector<FE::Real> state(static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    for (int j = 0; j <= kNy; ++j) {
        for (int i = 0; i <= kNx; ++i) {
            state[dof(phi, i, j, 0)] = block.phi(i * kH, j * kH);
            state[dof(u, i, j, 0)] = U;
        }
    }

    FE::level_set::LevelSetGeneratedInterfaceOptions cut;
    cut.level_set_field_name = "phi";
    cut.domain_id = kDomain;
    cut.requested_interface_marker = kInterface;
    cut.tolerance = 1.0e-12;
    cut.quadrature_order = 2;
    cut.interface_quadrature_order = 2;
    cut.volume_quadrature_order = 2;
    FE::level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto generated = lifecycle.build(system, cut, state);
    EXPECT_TRUE(generated.success) << generated.diagnostic;
    auto context = std::make_shared<FE::assembly::CutIntegrationContext>();
    context->addGeneratedInterfaceDomain(generated.domain);
    if (wetted_walls) {
        addWettedWalls(*context, generated.domain, system.meshAccess(), block);
    }
    system.setCutIntegrationContext(context);
    system.rebuildConstraintState();

    namespace ts = FE::timestepping;
    const auto factory = FE::backends::BackendFactory::create(FE::backends::BackendKind::Eigen);
    FE::backends::SolverOptions direct;
    direct.method = FE::backends::SolverMethod::Direct;
    auto linear = factory->createLinearSolver(direct);
    FE::systems::TransientSystem transient(system, std::make_shared<FE::systems::BackwardDifferenceIntegrator>());
    constexpr double dt = 0.01;
    ts::TimeLoopOptions loop;
    loop.t0 = 0.0;
    loop.t_end = dt;
    loop.dt = dt;
    loop.scheme = ts::SchemeKind::GeneralizedAlpha;
    loop.generalized_alpha_rho_inf = 0.5;
    loop.newton.residual_op = loop.newton.jacobian_op = "equations";
    loop.newton.abs_tolerance = 1e-12;
    loop.newton.rel_tolerance = 1e-12;
    auto history = ts::TimeHistory::allocate(*factory, system.dofHandler().getNumDofs());
    std::copy(state.begin(), state.end(), history.uPrev().localSpan().begin());
    history.resetCurrentToPrevious();
    history.primeDtHistory(dt);
    history.ensureSecondOrderState(*factory);
    history.uDot().zero();
    const auto report = ts::TimeLoop(loop).run(transient, *factory, *linear, history, {});
    EXPECT_TRUE(report.success) << report.message;

    const auto solved = history.uSpan();
    const auto value = [&](FE::FieldId field, int i, int j, std::size_t c) {
        return static_cast<double>(solved[dof(field, i, j, c)]);
    };
    FrontSample sample;
    sample.front_speed = U;
    const double s = (block.front - (1.0 - kH)) / kH;   // where the front crosses the edges of the last column
    for (int j = 0; j <= kNy; ++j) {
        if (j * kH >= block.depth) {
            continue;
        }
        const double ux = (1.0 - s) * value(u, kNx - 1, j, 0) + s * value(u, kNx, j, 0);
        const double uy = (1.0 - s) * value(u, kNx - 1, j, 1) + s * value(u, kNx, j, 1);
        sample.front_speed = std::min(sample.front_speed, ux);
        sample.max_velocity_error = std::max({sample.max_velocity_error, std::abs(ux - U), std::abs(uy)});
        for (int i = 0; i <= kNx; ++i) {
            if (block.phi(i * kH, j * kH) < 0.0) {
                sample.max_velocity_error = std::max(
                    {sample.max_velocity_error, std::abs(value(u, i, j, 0) - U), std::abs(value(u, i, j, 1))});
                sample.max_pressure = std::max(sample.max_pressure, std::abs(value(p, i, j, 0)));
            }
        }
    }
    return sample;
}

#endif

} // namespace

// A liquid block translating toward a slip wall, without gravity or viscous
// stress, reaches the wall at its own speed: until contact u = (U, 0) and
// p = 0 (free-surface audit A27).  With strong wall rows on the dry wall
// nodes of the cut cells the liquid in the last cell is decelerated (the
// front approaches as d' = -u_front, u_front ~ U d/h) and its pressure rises
// before contact; with the rows on the wetted faces only, the translation is
// the discrete solution.
TEST(FreeSurfaceWettedWalls, TranslatingBlockKeepsItsSpeedInTheLastCellBeforeASlipWall)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    for (const double front : {0.95, 0.99}) {
        SCOPED_TRACE(::testing::Message() << "front=" << front);
        const Block block{.front = front};
        const double d = 1.0 - front;
        const auto whole = translateBlock(block, /*wetted_walls=*/false);
        const auto wetted = translateBlock(block, /*wetted_walls=*/true);
        std::cout << "A27_translating_block front=" << front << " d/h=" << d / kH
                  << " whole_marker_front_speed=" << whole.front_speed
                  << " whole_marker_front_rate=" << whole.front_speed / d
                  << " whole_marker_velocity_error=" << whole.max_velocity_error
                  << " whole_marker_max_pressure=" << whole.max_pressure
                  << " wetted_front_speed=" << wetted.front_speed
                  << " wetted_velocity_error=" << wetted.max_velocity_error
                  << " wetted_max_pressure=" << wetted.max_pressure << '\n';
        EXPECT_LT(wetted.max_velocity_error, 1.0e-8);
        EXPECT_LT(wetted.max_pressure, 1.0e-8);
        EXPECT_LT(whole.front_speed, 0.9) << "the dry wall rows no longer slow the front";
    }
#endif
}

} // namespace svmp::Physics::test
