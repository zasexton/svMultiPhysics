/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

// Focused checks of the fitted-ALE free-surface path (tracker M5, decision
// D5): the boundary measure of fitted ALE terms, the opt-in fitted
// SurfaceStress (Laplace-Beltrami) form on flat and circular surfaces, the
// MeshNitsche kinematic ownership, and component-selected (sliding)
// mesh-motion Dirichlet conditions.

#include <gtest/gtest.h>

#include "Physics/Formulations/MeshMotion/HarmonicMeshMotionModule.h"
#include "Physics/Formulations/MeshMotion/PseudoElasticMeshMotionModule.h"
#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"

#include "FE/Assembly/GlobalSystemView.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Spaces/H1Space.h"
#include "FE/Spaces/ProductSpace.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/TimeIntegrator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Mesh.h"
#  include "Mesh/Topology/CellShape.h"
#endif

namespace svmp {
namespace Physics {
namespace test {
namespace {

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

namespace ns = formulations::navier_stokes;
namespace mm = formulations::mesh_motion;

constexpr int kLeft = 301;
constexpr int kRight = 302;
constexpr int kBottom = 303;
constexpr int kSurface = 304;

// [0, 2] x [0, 1] split into 2 x 1 squares and 4 triangles; the top edge is
// the free surface.
std::shared_ptr<Mesh> makeRectangleTriangleMesh()
{
    auto base = std::make_shared<MeshBase>();
    const std::vector<real_t> x_ref = {
        0.0, 0.0,  1.0, 0.0,  2.0, 0.0,
        0.0, 1.0,  1.0, 1.0,  2.0, 1.0,
    };
    const std::vector<offset_t> offsets = {0, 3, 6, 9, 12};
    const std::vector<index_t> cells = {
        0, 1, 4,  0, 4, 3,
        1, 2, 5,  1, 5, 4,
    };
    CellShape shape{};
    shape.family = CellFamily::Triangle;
    shape.num_corners = 3;
    shape.order = 1;
    base->build_from_arrays(2, x_ref, offsets, cells, std::vector<CellShape>(4, shape));
    base->finalize();
    base->register_label("wall_left", static_cast<label_t>(kLeft));
    base->register_label("wall_right", static_cast<label_t>(kRight));
    base->register_label("wall_bottom", static_cast<label_t>(kBottom));
    base->register_label("free_surface", static_cast<label_t>(kSurface));
    const auto coord = [&](index_t v, int c) {
        return base->X_ref().at(static_cast<std::size_t>(2 * v + c));
    };
    for (index_t face = 0; face < static_cast<index_t>(base->n_faces()); ++face) {
        const auto vertices = base->face_vertices(face);
        if (vertices.size() != 2u) {
            continue;
        }
        const auto all = [&](int c, real_t value) {
            return std::all_of(vertices.begin(), vertices.end(), [&](index_t v) {
                return std::abs(coord(v, c) - value) < real_t(1e-14);
            });
        };
        label_t label = INVALID_LABEL;
        if (all(1, 1.0)) {
            label = kSurface;
        } else if (all(1, 0.0)) {
            label = kBottom;
        } else if (all(0, 0.0)) {
            label = kLeft;
        } else if (all(0, 2.0)) {
            label = kRight;
        }
        if (label != INVALID_LABEL) {
            base->set_boundary_label(face, label);
        }
    }
    return create_mesh(std::move(base));
}

// Fan triangulation of the regular N-gon inscribed in the circle of radius
// R0; every boundary edge belongs to the free surface.
std::shared_ptr<Mesh> makeRegularPolygonDropMesh(int sides, real_t radius)
{
    auto base = std::make_shared<MeshBase>();
    std::vector<real_t> x_ref = {0.0, 0.0};
    for (int i = 0; i < sides; ++i) {
        const real_t theta = real_t(2) * std::numbers::pi_v<real_t> *
                             static_cast<real_t>(i) / static_cast<real_t>(sides);
        x_ref.push_back(radius * std::cos(theta));
        x_ref.push_back(radius * std::sin(theta));
    }
    std::vector<offset_t> offsets = {0};
    std::vector<index_t> cells;
    for (int i = 0; i < sides; ++i) {
        cells.push_back(0);
        cells.push_back(static_cast<index_t>(1 + i));
        cells.push_back(static_cast<index_t>(1 + (i + 1) % sides));
        offsets.push_back(static_cast<offset_t>(cells.size()));
    }
    CellShape shape{};
    shape.family = CellFamily::Triangle;
    shape.num_corners = 3;
    shape.order = 1;
    base->build_from_arrays(2, x_ref, offsets, cells,
                            std::vector<CellShape>(static_cast<std::size_t>(sides), shape));
    base->finalize();
    base->register_label("free_surface", static_cast<label_t>(kSurface));
    for (index_t face = 0; face < static_cast<index_t>(base->n_faces()); ++face) {
        const auto vertices = base->face_vertices(face);
        if (vertices.size() == 2u && vertices[0] != 0 && vertices[1] != 0) {
            base->set_boundary_label(face, static_cast<label_t>(kSurface));
        }
    }
    return create_mesh(std::move(base));
}

struct Spaces {
    std::shared_ptr<FE::spaces::H1Space> scalar;
    std::shared_ptr<FE::spaces::ProductSpace> vector;
};

Spaces triangleSpaces()
{
    auto scalar = std::make_shared<FE::spaces::H1Space>(FE::ElementType::Triangle3, 1);
    return {scalar, std::make_shared<FE::spaces::ProductSpace>(scalar, 2)};
}

ns::IncompressibleNavierStokesVMSOptions coupledALEOptions()
{
    ns::IncompressibleNavierStokesVMSOptions opts;
    opts.velocity_field_name = "u";
    opts.pressure_field_name = "p";
    opts.density = 1.0;
    opts.viscosity = 0.01;
    opts.enable_convection = false;
    opts.enable_vms = false;
    opts.enable_ale = true;
    opts.mesh_velocity_source = ns::ALEMeshVelocitySource::CoupledDisplacement;
    opts.mesh_displacement_field_name = "mesh_displacement";
    opts.mesh_velocity_field_name = "mesh_velocity";
    opts.auto_register_mesh_displacement_field = true;
    return opts;
}

ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary fittedSurface(
    FE::Real external_pressure,
    FE::Real surface_tension,
    bool surface_stress)
{
    ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary bc{};
    bc.implementation = ns::FreeSurfaceImplementation::FittedALE;
    bc.boundary_marker = kSurface;
    bc.external_pressure = external_pressure;
    bc.surface_tension = surface_tension;
    if (surface_stress) {
        bc.surface_tension_form = ns::FreeSurfaceSurfaceTensionForm::SurfaceStress;
        bc.allow_fitted_surface_stress = true;
    }
    bc.normal_kinematic_policy = ns::FreeSurfaceNormalKinematicPolicy::MatchFluidNormalVelocity;
    bc.tangential_mesh_policy = ns::FreeSurfaceTangentialMeshPolicy::Free;
    bc.kinematic_enforcement = ns::FreeSurfaceKinematicEnforcement::MeshNitsche;
    bc.kinematic_nitsche_gamma = 10.0;
    return bc;
}

mm::HarmonicMeshMotionOptions harmonicOptions(FE::Real kappa = 1.0)
{
    mm::HarmonicMeshMotionOptions opts;
    opts.field_name = "mesh_displacement";
    opts.operator_tag = "equations";
    opts.kappa = kappa;
    // MeshNitsche constrains the normal mesh velocity; the harmonic operator
    // acts on the mesh velocity as well.
    opts.quantity = mm::HarmonicQuantity::Velocity;
    return opts;
}

FE::GlobalIndex vertexDof(const FE::systems::FESystem& system,
                          FE::FieldId field,
                          FE::GlobalIndex vertex,
                          int component)
{
    const auto* map = system.fieldDofHandler(field).getEntityDofMap();
    if (map == nullptr) {
        throw std::runtime_error("vertexDof: field has no entity DOF map");
    }
    const auto dofs = map->getVertexDofs(vertex);
    return system.fieldDofOffset(field) + dofs[static_cast<std::size_t>(component)];
}

// A fitted coupled-ALE fluid system, with the harmonic mesh motion registered
// after the fluid, evaluated at u = 0, uniform pressure p0 and a stationary
// mesh displaced by d(x) = A x (Backward Euler, d_prev = d).
struct FittedALEFixture {
    std::shared_ptr<Mesh> mesh;
    Spaces spaces = triangleSpaces();
    std::unique_ptr<FE::systems::FESystem> system;
    FE::FieldId u{FE::INVALID_FIELD_ID};
    FE::FieldId p{FE::INVALID_FIELD_ID};
    FE::FieldId d{FE::INVALID_FIELD_ID};
    std::vector<FE::Real> solution;
    std::vector<FE::Real> previous;
    bool explicit_previous{false};

    FittedALEFixture(std::shared_ptr<Mesh> m,
                     ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary bc,
                     bool ale = true)
        : mesh(std::move(m))
    {
        // Coupled ALE assembles on the trial current configuration; the
        // application builds its FE system the same way for such inputs.
        system = std::make_unique<FE::systems::FESystem>(
            mesh, ale ? svmp::Configuration::Current : svmp::Configuration::Reference);
        auto opts = coupledALEOptions();
        if (!ale) {
            opts.enable_ale = false;
            opts.mesh_velocity_source = ns::ALEMeshVelocitySource::PrescribedData;
            opts.input_configuration_schema_version = 1;
            opts.explicit_legacy_configuration = true;
            bc.kinematic_enforcement = ns::FreeSurfaceKinematicEnforcement::None;
            bc.tangential_mesh_policy = ns::FreeSurfaceTangentialMeshPolicy::SmoothingOnly;
        }
        opts.free_surface.push_back(bc);
        ns::IncompressibleNavierStokesVMSModule fluid(spaces.vector, spaces.scalar, opts);
        fluid.registerOn(*system);
        if (ale) {
            // Sliding walls where the mesh has them (the rectangle); the drop
            // has none.  Without them the harmonic operator has a constant
            // null space and the FE gauge registry adds a mean-zero
            // constraint to the mesh rows.
            auto mesh_options = harmonicOptions();
            for (const auto& [marker, component] :
                 std::array<std::pair<int, int>, 3>{{{kLeft, 0}, {kRight, 0}, {kBottom, 1}}}) {
                if (mesh->local_mesh().list_label_names().count(static_cast<label_t>(marker)) == 0u) {
                    continue;
                }
                mm::HarmonicMeshMotionOptions::DirichletBC wall{};
                wall.boundary_marker = marker;
                wall.active_components = {component == 0, component == 1, false};
                mesh_options.dirichlet.push_back(wall);
            }
            mm::HarmonicMeshMotionModule mesh_motion(spaces.vector, mesh_options);
            mesh_motion.registerOn(*system);
        }
        system->setup();
        u = system->findFieldByName("u");
        p = system->findFieldByName("p");
        d = ale ? system->findFieldByName("mesh_displacement") : FE::INVALID_FIELD_ID;
        solution.assign(static_cast<std::size_t>(system->dofHandler().getNumDofs()), 0.0);
    }

    void setPressure(FE::Real value)
    {
        for (FE::GlobalIndex v = 0; v < static_cast<FE::GlobalIndex>(mesh->n_vertices()); ++v) {
            solution[static_cast<std::size_t>(vertexDof(*system, p, v, 0))] = value;
        }
    }

    // d = A x at every vertex (reference coordinates).
    void setLinearDisplacement(const std::array<std::array<FE::Real, 2>, 2>& a)
    {
        const auto& X = mesh->local_mesh().X_ref();
        for (FE::GlobalIndex v = 0; v < static_cast<FE::GlobalIndex>(mesh->n_vertices()); ++v) {
            const FE::Real x = X[static_cast<std::size_t>(2 * v)];
            const FE::Real y = X[static_cast<std::size_t>(2 * v + 1)];
            for (int c = 0; c < 2; ++c) {
                solution[static_cast<std::size_t>(vertexDof(*system, d, v, c))] =
                    a[static_cast<std::size_t>(c)][0] * x + a[static_cast<std::size_t>(c)][1] * y;
            }
        }
    }

    // Keep the current state as the previous one (a stationary mesh) unless
    // setPreviousMeshDisplacementToZero() was called; then only the mesh
    // displacement starts from zero (so dt(d) = d for dt = 1).
    void setPreviousMeshDisplacementToZero()
    {
        previous = solution;
        for (FE::GlobalIndex v = 0; v < static_cast<FE::GlobalIndex>(mesh->n_vertices()); ++v) {
            for (int c = 0; c < 2; ++c) {
                previous[static_cast<std::size_t>(vertexDof(*system, d, v, c))] = 0.0;
            }
        }
        explicit_previous = true;
    }

    // u = d at every vertex (used with setPreviousToZero and dt = 1, so the
    // fluid velocity equals the mesh velocity dt(d) = d).
    void setVelocityEqualToDisplacement()
    {
        for (FE::GlobalIndex v = 0; v < static_cast<FE::GlobalIndex>(mesh->n_vertices()); ++v) {
            for (int c = 0; c < 2; ++c) {
                solution[static_cast<std::size_t>(vertexDof(*system, u, v, c))] =
                    solution[static_cast<std::size_t>(vertexDof(*system, d, v, c))];
            }
        }
    }

    std::vector<FE::Real> residual()
    {
        if (!explicit_previous) {
            previous = solution;
        }
        FE::systems::SystemStateView state;
        state.dt = 1.0;
        state.u = std::span<const FE::Real>(solution);
        state.u_prev = std::span<const FE::Real>(previous);
        const FE::systems::BackwardDifferenceIntegrator integrator;
        const auto context = integrator.buildContext(1, state);
        state.time_integration = &context;
        if (d != FE::INVALID_FIELD_ID) {
            (void)system->updateCurrentCoordinatesFromMeshDisplacement(state);
        }
        const auto n = system->dofHandler().getNumDofs();
        FE::assembly::DenseVectorView r(n);
        r.zero();
        FE::systems::AssemblyRequest request;
        request.op = "equations";
        request.want_vector = true;
        const auto result = system->assemble(request, state, nullptr, &r);
        EXPECT_TRUE(result.success) << result.error_message;
        std::vector<FE::Real> out(static_cast<std::size_t>(n));
        for (FE::GlobalIndex i = 0; i < n; ++i) {
            out[static_cast<std::size_t>(i)] = r.getVectorEntry(i);
        }
        return out;
    }

    FE::Real velocityRow(const std::vector<FE::Real>& r, FE::GlobalIndex v, int c) const
    {
        return r[static_cast<std::size_t>(vertexDof(*system, u, v, c))];
    }

    FE::Real meshRow(const std::vector<FE::Real>& r, FE::GlobalIndex v, int c) const
    {
        return r[static_cast<std::size_t>(vertexDof(*system, d, v, c))];
    }
};

constexpr std::array<FE::GlobalIndex, 3> kTopVertices{3, 4, 5};

TEST(FittedFreeSurfaceALE, ExternalPressureLoadUsesTheCurrentSurfaceMeasureOnce)
{
    // p_ext n.v on the top edge: the y rows of the fluid momentum sum to
    // p_ext times the current surface length.  The stretch d = (x/2, 0)
    // makes the current length 3 (reference length 2).
    constexpr FE::Real p_ext = 1.25;
    FittedALEFixture ale(makeRectangleTriangleMesh(), fittedSurface(p_ext, 0.0, false));
    ale.setLinearDisplacement({{{0.5, 0.0}, {0.0, 0.0}}});
    const auto r = ale.residual();
    FE::Real load_y = 0.0;
    FE::Real load_x = 0.0;
    for (FE::GlobalIndex v = 0; v < 6; ++v) {
        load_y += ale.velocityRow(r, v, 1);
        load_x += ale.velocityRow(r, v, 0);
    }
    EXPECT_NEAR(load_y, p_ext * 3.0, 1e-12);
    EXPECT_NEAR(load_x, 0.0, 1e-12);
    // Consistent nodal loads: 3/4, 3/2, 3/4 of p_ext on the top vertices.
    EXPECT_NEAR(ale.velocityRow(r, 3, 1), p_ext * 0.75, 1e-12);
    EXPECT_NEAR(ale.velocityRow(r, 4, 1), p_ext * 1.5, 1e-12);
    EXPECT_NEAR(ale.velocityRow(r, 5, 1), p_ext * 0.75, 1e-12);

    // The same boundary on a static mesh carries p_ext times its length 2.
    FittedALEFixture fixed(makeRectangleTriangleMesh(), fittedSurface(p_ext, 0.0, false),
                           /*ale=*/false);
    const auto r0 = fixed.residual();
    FE::Real static_load_y = 0.0;
    for (FE::GlobalIndex v = 0; v < 6; ++v) {
        static_load_y += fixed.velocityRow(r0, v, 1);
    }
    EXPECT_NEAR(static_load_y, p_ext * 2.0, 1e-12);
}

TEST(FittedFreeSurfaceALE, SurfaceStressRequiresTheExplicitOptIn)
{
    auto mesh = makeRectangleTriangleMesh();
    const auto spaces = triangleSpaces();
    auto bc = fittedSurface(0.0, 0.5, true);
    bc.allow_fitted_surface_stress = false;
    {
        FE::systems::FESystem system(mesh);
        auto opts = coupledALEOptions();
        opts.free_surface.push_back(bc);
        ns::IncompressibleNavierStokesVMSModule fluid(spaces.vector, spaces.scalar, opts);
        try {
            fluid.registerOn(system);
            FAIL() << "fitted SurfaceStress without the opt-in must fail closed";
        } catch (const std::invalid_argument& error) {
            EXPECT_NE(std::string(error.what()).find("Allow_fitted_surface_stress"),
                      std::string::npos)
                << error.what();
        }
        EXPECT_TRUE(system.formulationRecords().empty());
    }
    {
        // The opt-in without an explicit SurfaceStress request is rejected.
        auto flagged = fittedSurface(0.0, 0.5, false);
        flagged.allow_fitted_surface_stress = true;
        FE::systems::FESystem system(mesh);
        auto opts = coupledALEOptions();
        opts.free_surface.push_back(flagged);
        ns::IncompressibleNavierStokesVMSModule fluid(spaces.vector, spaces.scalar, opts);
        EXPECT_THROW(fluid.registerOn(system), std::invalid_argument);
    }
    {
        // Prescribed mesh motion may assemble on the reference frame.
        FE::systems::FESystem system(mesh);
        auto opts = coupledALEOptions();
        opts.mesh_velocity_source = ns::ALEMeshVelocitySource::PrescribedData;
        opts.input_configuration_schema_version = 1;
        opts.explicit_legacy_configuration = true;
        auto legacy = fittedSurface(0.0, 0.5, true);
        legacy.kinematic_enforcement = ns::FreeSurfaceKinematicEnforcement::None;
        opts.free_surface.push_back(legacy);
        ns::IncompressibleNavierStokesVMSModule fluid(spaces.vector, spaces.scalar, opts);
        EXPECT_THROW(fluid.registerOn(system), std::invalid_argument);
    }
    {
        FE::systems::FESystem system(mesh);
        auto opts = coupledALEOptions();
        opts.free_surface.push_back(fittedSurface(0.0, 0.5, true));
        ns::IncompressibleNavierStokesVMSModule fluid(spaces.vector, spaces.scalar, opts);
        ASSERT_NO_THROW(fluid.registerOn(system));
        const auto artifact = fluid.effectiveConfigurationArtifact();
        ASSERT_TRUE(artifact.has_value());
        EXPECT_NE(artifact->json.find("\"fitted_surface_stress\":{\"opt_in\":true"),
                  std::string::npos);
        EXPECT_EQ(artifact->json.find("fitted_surface_stress_current_frame_gradient_unqualified"),
                  std::string::npos);
    }
}

TEST(FittedFreeSurfaceALE, SurfaceStressOnAFlatSurfaceHasNoNormalLoad)
{
    // gamma (I - n n) : grad(v) on a flat current surface: no normal load at
    // any node and no tangential load at interior nodes; the end nodes carry
    // the conormal line force -+gamma.
    constexpr FE::Real gamma = 0.5;
    FittedALEFixture ale(makeRectangleTriangleMesh(), fittedSurface(0.0, gamma, true));
    ale.setLinearDisplacement({{{0.5, 0.0}, {0.0, 0.25}}});
    const auto r = ale.residual();
    for (const auto v : kTopVertices) {
        EXPECT_NEAR(ale.velocityRow(r, v, 1), 0.0, 1e-13) << "vertex " << v;
    }
    EXPECT_NEAR(ale.velocityRow(r, 4, 0), 0.0, 1e-13);
    EXPECT_NEAR(ale.velocityRow(r, 3, 0), -gamma, 1e-13);
    EXPECT_NEAR(ale.velocityRow(r, 5, 0), gamma, 1e-13);
}

FE::Real regularPolygonResidual(int sides, FE::Real pressure_factor)
{
    // Reference polygon of radius 1, current polygon of radius 1.3 (d = 0.3 x).
    constexpr FE::Real gamma = 0.7;
    constexpr FE::Real current_radius = 1.3;
    FittedALEFixture drop(makeRegularPolygonDropMesh(sides, 1.0), fittedSurface(0.0, gamma, true));
    drop.setLinearDisplacement({{{0.3, 0.0}, {0.0, 0.3}}});
    const FE::Real half_angle = std::numbers::pi_v<FE::Real> / static_cast<FE::Real>(sides);
    // pressure_factor = 1: the discrete balance gamma / (R cos(pi/N)) of the
    // regular polygon; pressure_factor = cos(pi/N): the Laplace pressure gamma/R.
    drop.setPressure(pressure_factor * gamma / (current_radius * std::cos(half_angle)));
    const auto r = drop.residual();
    const FE::Real nodal_capillary_load = 2.0 * gamma * std::sin(half_angle);
    FE::Real worst = 0.0;
    for (FE::GlobalIndex v = 0; v <= static_cast<FE::GlobalIndex>(sides); ++v) {
        worst = std::max(worst, std::hypot(drop.velocityRow(r, v, 0), drop.velocityRow(r, v, 1)));
    }
    return worst / nodal_capillary_load;
}

TEST(FittedFreeSurfaceALE, SurfaceStressOnACircularDropBalancesAConstantPressure)
{
    // On the regular polygon the Laplace-Beltrami load at a node is
    // 2 gamma sin(pi/N) inward and a constant pressure p loads it with
    // p 2 R sin(pi/N) cos(pi/N) outward: the discrete balance is exact for
    // p = gamma / (R cos(pi/N)), and the Laplace pressure gamma/R leaves the
    // relative residual 1 - cos(pi/N) = O(h^2).  The current radius differs
    // from the reference radius, so the balance checks that the current
    // normal, measure and gradient are used consistently.
    for (const int sides : {16, 32}) {
        EXPECT_LT(regularPolygonResidual(sides, 1.0), 1e-12) << "N = " << sides;
    }
    const FE::Real e16 = regularPolygonResidual(
        16, std::cos(std::numbers::pi_v<FE::Real> / 16.0));
    const FE::Real e32 = regularPolygonResidual(
        32, std::cos(std::numbers::pi_v<FE::Real> / 32.0));
    EXPECT_NEAR(e16, 1.0 - std::cos(std::numbers::pi_v<FE::Real> / 16.0), 1e-12);
    EXPECT_NEAR(e32, 1.0 - std::cos(std::numbers::pi_v<FE::Real> / 32.0), 1e-12);
    EXPECT_NEAR(e16 / e32, 4.0, 0.05);
}

TEST(FittedFreeSurfaceALE, MeshNitscheKeepsTheFluidDynamicConditionAndDeclaresMeshConsistency)
{
    auto mesh = makeRectangleTriangleMesh();
    const auto spaces = triangleSpaces();
    FE::systems::FESystem system(mesh);
    auto opts = coupledALEOptions();
    opts.free_surface.push_back(fittedSurface(0.0, 0.0, false));
    ns::IncompressibleNavierStokesVMSModule fluid(spaces.vector, spaces.scalar, opts);
    fluid.registerOn(system);
    const auto declarations = system.meshNormalBoundaryConstraints();
    ASSERT_EQ(declarations.size(), 1u);
    EXPECT_TRUE(declarations.front().requires_mesh_flux_consistency);
    ASSERT_TRUE(declarations.front().consumer_binding.has_value());
    ASSERT_TRUE(declarations.front().consumer_binding->related_fluid.has_value());
    EXPECT_EQ(declarations.front().consumer_binding->related_fluid->enforcement_kind,
              FE::analysis::EnforcementKind::WeakConsistent);
    EXPECT_EQ(declarations.front().consumer_binding->related_fluid->descriptor_source,
              "Fitted free-surface fluid natural dynamic condition on marker " +
                  std::to_string(kSurface));

    // The harmonic mesh motion registered afterwards installs its consistency
    // term; the pseudo-elastic model has none and fails closed.
    mm::PseudoElasticMeshMotionOptions elastic;
    elastic.field_name = "mesh_displacement";
    mm::PseudoElasticMeshMotionModule elastic_module(spaces.vector, elastic);
    EXPECT_THROW(elastic_module.registerOn(system), std::invalid_argument);
    // A displacement operator and a kappa that breaks coercivity
    // (gamma_N = 10 <= 2 kappa) fail closed as well.
    auto displacement_options = harmonicOptions();
    displacement_options.quantity = mm::HarmonicQuantity::Displacement;
    mm::HarmonicMeshMotionModule displacement_module(spaces.vector, displacement_options);
    EXPECT_THROW(displacement_module.registerOn(system), std::invalid_argument);
    mm::HarmonicMeshMotionModule stiff_module(spaces.vector, harmonicOptions(6.0));
    EXPECT_THROW(stiff_module.registerOn(system), std::invalid_argument);
    mm::HarmonicMeshMotionModule harmonic(spaces.vector, harmonicOptions());
    ASSERT_NO_THROW(harmonic.registerOn(system));
    ASSERT_NO_THROW(system.setup());

    // Registration in the reverse order fails closed.
    FE::systems::FESystem reversed(mesh);
    mm::HarmonicMeshMotionModule first(spaces.vector, harmonicOptions());
    first.registerOn(reversed);
    ns::IncompressibleNavierStokesVMSModule second(spaces.vector, spaces.scalar, opts);
    try {
        second.registerOn(reversed);
        FAIL() << "MeshNitsche after the mesh-motion equation must fail closed";
    } catch (const std::invalid_argument& error) {
        EXPECT_NE(std::string(error.what()).find("before the mesh-motion equation"),
                  std::string::npos)
            << error.what();
    }
}

TEST(FittedFreeSurfaceALE, MeshNitscheRemovesTheHarmonicNormalFlux)
{
    // The harmonic operator acts on the mesh velocity w = dt(d).  With
    // d_prev = 0, dt = 1 and d = A x (the fluid state unchanged from the
    // previous step, u = u_prev = w), w = A x is linear, so the harmonic
    // volume term at a boundary node equals its boundary flux
    // kappa ((grad w) n).psi exactly.  With the consistency term the mesh rows
    // of the interior free-surface node keep only the tangential flux; the
    // normal component vanishes (u = w, so the kinematic penalty is zero).
    FittedALEFixture ale(makeRectangleTriangleMesh(), fittedSurface(0.0, 0.0, false));
    ale.setLinearDisplacement({{{0.1, 0.2}, {0.3, 0.05}}});
    ale.setVelocityEqualToDisplacement();
    ale.setPreviousMeshDisplacementToZero();
    const auto r = ale.residual();
    // Current top edge: the image of y = 1, direction (1.1, 0.3).
    const FE::Real tx = 1.1, ty = 0.3;
    const FE::Real norm = std::hypot(tx, ty);
    const FE::Real nx = -ty / norm, ny = tx / norm;
    const FE::Real normal = ale.meshRow(r, 4, 0) * nx + ale.meshRow(r, 4, 1) * ny;
    const FE::Real tangential = (ale.meshRow(r, 4, 0) * tx + ale.meshRow(r, 4, 1) * ty) / norm;
    EXPECT_NEAR(normal, 0.0, 1e-13);
    // kappa ((grad_x d) n . t) |edge|, grad_x d = G F^{-1}, G the imposed
    // gradient and F = I + G; the two top edges each carry half of it.
    EXPECT_NEAR(tangential, 0.1589918, 1e-6);
    EXPECT_NEAR(ale.meshRow(r, 4, 0), 0.1533895, 1e-6);
    EXPECT_NEAR(ale.meshRow(r, 4, 1), 0.0418335, 1e-6);
}

TEST(FittedFreeSurfaceALE, MeshKinematicRowSamplesTheFluidVelocityOnTheFace)
{
    // The mesh row is a boundary term of the mesh-displacement formulation in
    // which the fluid velocity is a non-primary field.  With dt(d) = 0 and
    // u = (0, phi_4) (one at the free-surface vertex 4 only) its normal
    // component at vertex 4 is -gamma_N/h_n n_y int phi_4^2 ds over the two
    // current top edges.  Regression for face field sampling: the field basis
    // must be evaluated at the face-to-cell mapped points, not at the
    // canonical face points.
    FittedALEFixture ale(makeRectangleTriangleMesh(), fittedSurface(0.0, 0.0, false));
    ale.setLinearDisplacement({{{0.1, 0.2}, {0.3, 0.05}}});
    ale.solution[static_cast<std::size_t>(vertexDof(*ale.system, ale.u, 4, 1))] = 1.0;
    const auto r = ale.residual();
    const FE::Real tx = 1.1, ty = 0.3;
    const FE::Real edge = std::hypot(tx, ty);
    const FE::Real nx = -ty / edge, ny = tx / edge;
    // Both top triangles have current area det(F)/2 = 1.095/2.
    const FE::Real h_n = 2.0 * (1.095 / 2.0) / edge;
    const FE::Real expected = -(10.0 / h_n) * ny * (2.0 * edge / 3.0);
    const FE::Real normal = ale.meshRow(r, 4, 0) * nx + ale.meshRow(r, 4, 1) * ny;
    EXPECT_NEAR(normal, expected, 1e-10 * std::abs(expected));
}

TEST(FittedFreeSurfaceALE, ComponentSelectedMeshDirichletSlidesAlongAWall)
{
    auto mesh = makeRectangleTriangleMesh();
    const auto spaces = triangleSpaces();
    FE::systems::FESystem system(mesh);
    auto opts = harmonicOptions();
    mm::HarmonicMeshMotionOptions::DirichletBC left{};
    left.boundary_marker = kLeft;
    left.active_components = {true, false, false};
    mm::HarmonicMeshMotionOptions::DirichletBC bottom{};
    bottom.boundary_marker = kBottom;
    bottom.active_components = {false, true, false};
    opts.dirichlet = {left, bottom};
    mm::HarmonicMeshMotionModule module(spaces.vector, opts);
    module.registerOn(system);
    system.setup();
    const auto d = system.findFieldByName("mesh_displacement");
    const auto& constraints = system.constraints();
    // Vertex 3 = (0, 1) lies on the left wall only: x fixed, y free.
    EXPECT_TRUE(constraints.isConstrained(vertexDof(system, d, 3, 0)));
    EXPECT_FALSE(constraints.isConstrained(vertexDof(system, d, 3, 1)));
    // Vertex 2 = (2, 0) lies on the bottom only: y fixed, x free.
    EXPECT_FALSE(constraints.isConstrained(vertexDof(system, d, 2, 0)));
    EXPECT_TRUE(constraints.isConstrained(vertexDof(system, d, 2, 1)));
    // Vertex 0 = (0, 0) lies on both: fully fixed.
    EXPECT_TRUE(constraints.isConstrained(vertexDof(system, d, 0, 0)));
    EXPECT_TRUE(constraints.isConstrained(vertexDof(system, d, 0, 1)));
    // Vertex 5 = (2, 1) lies on neither: free.
    EXPECT_FALSE(constraints.isConstrained(vertexDof(system, d, 5, 0)));
    EXPECT_FALSE(constraints.isConstrained(vertexDof(system, d, 5, 1)));

    FE::systems::FESystem none(mesh);
    auto empty = harmonicOptions();
    mm::HarmonicMeshMotionOptions::DirichletBC nothing{};
    nothing.boundary_marker = kLeft;
    nothing.active_components = {false, false, false};
    empty.dirichlet = {nothing};
    mm::HarmonicMeshMotionModule rejected(spaces.vector, empty);
    EXPECT_THROW(rejected.registerOn(none), std::invalid_argument);
}

#endif // SVMP_FE_WITH_MESH

} // namespace
} // namespace test
} // namespace Physics
} // namespace svmp
