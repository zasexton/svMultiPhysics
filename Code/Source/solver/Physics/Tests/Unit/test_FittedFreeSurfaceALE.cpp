/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

// Focused checks of the fitted-ALE free-surface path (tracker M5, decision
// D5): the boundary measure of fitted ALE terms, the fitted SurfaceStress
// (Laplace-Beltrami) form, the default, on flat and circular surfaces, the
// MeshNitsche kinematic ownership, and component-selected (sliding)
// mesh-motion Dirichlet conditions.

#include <gtest/gtest.h>

#include "Physics/Formulations/MeshMotion/HarmonicMeshMotionModule.h"
#include "Physics/Formulations/MeshMotion/PseudoElasticMeshMotionModule.h"
#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"
#include "Physics/Formulations/NavierStokes/NavierStokesBCFactories.h"

#include "FE/Assembly/GlobalSystemView.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Spaces/H1Space.h"
#include "FE/Spaces/ProductSpace.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/SystemManifest.h"
#include "FE/Systems/TimeIntegrator.h"
#include "FE/TimeStepping/GeneralizedAlpha.h"
#include "FE/TimeStepping/TimeSteppingUtils.h"
#include "FE/Tests/Unit/Forms/FormsTestHelpers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <numbers>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Mesh.h"
#  include "Mesh/Topology/CellShape.h"
#endif

namespace svmp {
namespace Physics {
namespace test {
namespace {

TEST(SurfaceStressForms, PreservesPressureAndEnergyTreesOnEachIntegrationFrame)
{
    namespace ns = formulations::navier_stokes;
    using namespace FE::forms;
    auto mesh = std::make_shared<FE::forms::test::SingleTetraMeshAccess>();
    auto scalar = std::make_shared<FE::spaces::H1Space>(FE::ElementType::Tetra4, 1);
    FE::spaces::ProductSpace vector(scalar, 3);
    FE::systems::FESystem system(mesh);
    const auto phi = system.addField({.name = "phi", .space = scalar});
    const auto geometry = FE::level_set::LevelSetGeometry::bind(
        system, {.level_set_field = phi, .domain_id = "surface_stress", .interface_marker = 71});
    const auto v = TestFunction(vector, "v");
    const auto pressure_value = FormExpr::constant(1.25);
    const auto expect_tree = [](const FormExpr& actual, const FormExpr& expected) {
        EXPECT_EQ(FE::systems::formExprTreeHash(actual), FE::systems::formExprTreeHash(expected));
        EXPECT_EQ(actual.toString(), expected.toString());
    };
    const auto check = [&](const FormExpr& normal, const auto& integrate) {
        for (const FE::Real gamma_value : {FE::Real{0.0}, FE::Real{0.072}}) {
            const auto gamma = FormExpr::constant(gamma_value);
            const auto pressure = ns::Factories::exteriorPressureIntegrand(pressure_value, normal, v);
            const auto expected_pressure = pressure_value * inner(normal, v);
            expect_tree(pressure, expected_pressure);
            ns::post::ConservativeBalanceForms diagnostics;
            ns::post::addExteriorPressure(&diagnostics, integrate(pressure));
            expect_tree(diagnostics.external_pressure, integrate(expected_pressure));
            FormExpr residual = integrate(pressure);
            if (gamma_value > 0.0) {
                const auto energy = ns::Factories::surfaceEnergyIntegrand(gamma, normal, v, 3);
                const auto expected_energy = gamma * inner(FormExpr::identity(3) - outer(normal, normal), grad(v));
                expect_tree(energy, expected_energy);
                ns::post::addSurfaceEnergy(&diagnostics, integrate(energy));
                expect_tree(diagnostics.surface_energy, integrate(expected_energy));
                residual = integrate(pressure + energy);
                expect_tree(residual, integrate(expected_pressure + expected_energy));
                EXPECT_NE(FE::systems::formExprTreeHash(residual),
                          FE::systems::formExprTreeHash(integrate(pressure) + integrate(energy)));
            } else {
                expect_tree(residual, integrate(expected_pressure));
                EXPECT_FALSE(diagnostics.surface_energy.isValid());
            }
        }
    };

    const auto fitted = BoundaryDomain::fullPhysical(304);
    check(currentNormal(), [&](const FormExpr& form) { return form.ds(fitted); });
    for (const auto side : {CutVolumeSide::Negative, CutVolumeSide::Positive}) {
        const auto liquid = geometry.side(side);
        const auto expected_normal = side == CutVolumeSide::Negative ? FormExpr::normal() : -FormExpr::normal();
        expect_tree(liquid.outwardNormal(), expected_normal);
        check(liquid.outwardNormal(), [&](const FormExpr& form) { return form.dI(liquid.interface()); });
    }
}

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
    FE::Real surface_tension)
{
    ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary bc{};
    bc.implementation = ns::FreeSurfaceImplementation::FittedALE;
    bc.boundary_marker = kSurface;
    bc.external_pressure = external_pressure;
    bc.surface_tension = surface_tension;
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

    // The time integrator of residual() and jacobian(): Backward Euler unless
    // set; with previous_rate set, the history is (u_n, u_dot_n), the layout
    // of the first-order generalized-alpha stage solve.
    const FE::systems::TimeIntegrator* integrator{nullptr};
    std::vector<FE::Real> previous_rate;

    FittedALEFixture(std::shared_ptr<Mesh> m,
                     std::optional<ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary> bc,
                     ns::IncompressibleNavierStokesVMSOptions opts = coupledALEOptions())
        : mesh(std::move(m))
    {
        // Coupled ALE assembles on the trial current configuration; the
        // application builds its FE system the same way for such inputs.
        system = std::make_unique<FE::systems::FESystem>(mesh, svmp::Configuration::Current);
        if (bc.has_value()) {
            opts.free_surface.push_back(*bc);
        }
        ns::IncompressibleNavierStokesVMSModule fluid(spaces.vector, spaces.scalar, opts);
        fluid.registerOn(*system);
        // Sliding walls where the mesh has them (the rectangle); the drop
        // has none.  Without them the harmonic operator has a constant null
        // space and the FE gauge registry adds a mean-zero constraint to the
        // mesh rows.
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
        system->setup();
        u = system->findFieldByName("u");
        p = system->findFieldByName("p");
        d = system->findFieldByName("mesh_displacement");
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

    void assemble(FE::assembly::GlobalSystemView* J, FE::assembly::GlobalSystemView* R)
    {
        if (!explicit_previous) {
            previous = solution;
        }
        FE::systems::SystemStateView state;
        state.dt = 1.0;
        state.u = std::span<const FE::Real>(solution);
        state.u_prev = std::span<const FE::Real>(previous);
        const std::array<std::span<const FE::Real>, 2> history{
            std::span<const FE::Real>(previous), std::span<const FE::Real>(previous_rate)};
        if (!previous_rate.empty()) {
            state.u_prev2 = history[1];
            state.u_history = history;
        }
        const FE::systems::BackwardDifferenceIntegrator backward_euler;
        const auto context = (integrator != nullptr ? *integrator : backward_euler).buildContext(1, state);
        state.time_integration = &context;
        if (d != FE::INVALID_FIELD_ID) {
            (void)system->updateCurrentCoordinatesFromMeshDisplacement(state);
        }
        FE::systems::AssemblyRequest request;
        request.op = "equations";
        request.want_matrix = J != nullptr;
        request.want_vector = R != nullptr;
        const auto result = system->assemble(request, state, J, R);
        EXPECT_TRUE(result.success) << result.error_message;
    }

    FE::assembly::DenseMatrixView jacobian()
    {
        FE::assembly::DenseMatrixView J(system->dofHandler().getNumDofs());
        J.zero();
        assemble(&J, nullptr);
        return J;
    }

    std::vector<FE::Real> residual()
    {
        const auto n = system->dofHandler().getNumDofs();
        FE::assembly::DenseVectorView r(n);
        r.zero();
        assemble(nullptr, &r);
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
    FittedALEFixture ale(makeRectangleTriangleMesh(), fittedSurface(p_ext, 0.0));
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

    // The same boundary on the undisplaced mesh (d = 0: the current surface
    // is the reference one) carries p_ext times its length 2.
    FittedALEFixture fixed(makeRectangleTriangleMesh(), fittedSurface(p_ext, 0.0));
    const auto r0 = fixed.residual();
    FE::Real static_load_y = 0.0;
    for (FE::GlobalIndex v = 0; v < 6; ++v) {
        static_load_y += fixed.velocityRow(r0, v, 1);
    }
    EXPECT_NEAR(static_load_y, p_ext * 2.0, 1e-12);
}

TEST(FittedFreeSurfaceALE, SurfaceStressIsTheFittedDefaultWithoutAnOptIn)
{
    // Decision D47 Q2(b): SurfaceStress is the only capillary form on fitted
    // boundaries as well, and it needs no opt-in.  Its frame, surface-tension
    // and contact-line checks apply only when the surface energy is
    // installed (gamma != 0).
    constexpr FE::Real gamma = 0.5;
    {
        // On the stretched flat surface the end nodes carry the conormal
        // line force -+gamma.
        FittedALEFixture ale(makeRectangleTriangleMesh(), fittedSurface(0.0, gamma));
        ale.setLinearDisplacement({{{0.5, 0.0}, {0.0, 0.0}}});
        const auto r = ale.residual();
        EXPECT_NEAR(ale.velocityRow(r, 3, 0), -gamma, 1e-13);
        EXPECT_NEAR(ale.velocityRow(r, 5, 0), gamma, 1e-13);
    }

    auto mesh = makeRectangleTriangleMesh();
    const auto spaces = triangleSpaces();
    {
        FE::systems::FESystem system(mesh);
        auto opts = coupledALEOptions();
        opts.free_surface.push_back(fittedSurface(0.0, gamma));
        ns::IncompressibleNavierStokesVMSModule fluid(spaces.vector, spaces.scalar, opts);
        ASSERT_NO_THROW(fluid.registerOn(system));
        const auto artifact = fluid.effectiveConfigurationArtifact();
        ASSERT_TRUE(artifact.has_value());
        EXPECT_NE(artifact->json.find("\"surface_tension_form_effective\":\"SurfaceStress\""),
                  std::string::npos)
            << artifact->json;
        EXPECT_EQ(artifact->json.find("fitted_surface_stress_current_frame_gradient_unqualified"),
                  std::string::npos);
    }
    {
        // Prescribed mesh motion may assemble on the reference frame: the
        // surface energy is rejected there by the frame check, and the
        // pressure load alone by the qualified fitted-ALE contract.
        const auto register_prescribed = [&](FE::Real surface_tension) -> std::string {
            FE::systems::FESystem system(mesh);
            auto opts = coupledALEOptions();
            opts.mesh_velocity_source = ns::ALEMeshVelocitySource::PrescribedData;
            opts.auto_register_mesh_displacement_field = false;
            opts.free_surface.push_back(fittedSurface(0.25, surface_tension));
            ns::IncompressibleNavierStokesVMSModule fluid(spaces.vector, spaces.scalar, opts);
            try {
                fluid.registerOn(system);
            } catch (const std::invalid_argument& error) {
                return error.what();
            }
            return {};
        };
        EXPECT_NE(register_prescribed(gamma).find(
                      "fitted SurfaceStress requires a static mesh or ALE with a "
                      "coupled mesh-displacement unknown"),
                  std::string::npos);
        EXPECT_NE(register_prescribed(0.0).find(
                      "contract requires mesh velocity derived from a coupled "
                      "mesh-displacement unknown"),
                  std::string::npos);
    }
}

TEST(FittedFreeSurfaceALE, SurfaceStressOnAFlatSurfaceHasNoNormalLoad)
{
    // gamma (I - n n) : grad(v) on a flat current surface: no normal load at
    // any node and no tangential load at interior nodes; the end nodes carry
    // the conormal line force -+gamma.
    constexpr FE::Real gamma = 0.5;
    FittedALEFixture ale(makeRectangleTriangleMesh(), fittedSurface(0.0, gamma));
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
    FittedALEFixture drop(makeRegularPolygonDropMesh(sides, 1.0), fittedSurface(0.0, gamma));
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
    opts.free_surface.push_back(fittedSurface(0.0, 0.0));
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
    FittedALEFixture ale(makeRectangleTriangleMesh(), fittedSurface(0.0, 0.0));
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
    FittedALEFixture ale(makeRectangleTriangleMesh(), fittedSurface(0.0, 0.0));
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

// The fields of a fixture DOF: 0 velocity, 1 pressure, 2 mesh displacement.
int fixtureFieldIndex(const FittedALEFixture& ale, FE::GlobalIndex dof)
{
    const std::array<FE::FieldId, 3> fields{ale.u, ale.p, ale.d};
    for (int k = 0; k < 3; ++k) {
        const auto begin = ale.system->fieldDofOffset(fields[static_cast<std::size_t>(k)]);
        const auto end = begin + ale.system->fieldDofHandler(fields[static_cast<std::size_t>(k)]).getNumDofs();
        if (dof >= begin && dof < end) {
            return k;
        }
    }
    throw std::runtime_error("fixtureFieldIndex: DOF outside the fixture fields");
}

// ||J - FD|| / ||FD|| of each (row field, column field) block of the
// assembled Jacobian against central differences of the residual, over the
// unconstrained rows and columns; the mesh moves with every perturbed state.
std::array<std::array<FE::Real, 3>, 3> fixtureJacobianBlockDefects(FittedALEFixture& ale, FE::Real eps)
{
    const auto J = ale.jacobian();
    const auto base = ale.solution;
    const auto& constraints = ale.system->constraints();
    const auto n = ale.system->dofHandler().getNumDofs();
    std::array<std::array<FE::Real, 3>, 3> diff2{}, ref2{};
    for (FE::GlobalIndex col = 0; col < n; ++col) {
        if (constraints.isConstrained(col)) {
            continue;
        }
        ale.solution = base;
        ale.solution[static_cast<std::size_t>(col)] += eps;
        const auto r_plus = ale.residual();
        ale.solution = base;
        ale.solution[static_cast<std::size_t>(col)] -= eps;
        const auto r_minus = ale.residual();
        const auto c = static_cast<std::size_t>(fixtureFieldIndex(ale, col));
        for (FE::GlobalIndex row = 0; row < n; ++row) {
            if (constraints.isConstrained(row)) {
                continue;
            }
            const auto r = static_cast<std::size_t>(fixtureFieldIndex(ale, row));
            const FE::Real fd = (r_plus[static_cast<std::size_t>(row)] - r_minus[static_cast<std::size_t>(row)]) /
                                (FE::Real(2.0) * eps);
            const FE::Real d = J.getMatrixEntry(row, col) - fd;
            diff2[r][c] += d * d;
            ref2[r][c] += fd * fd;
        }
    }
    ale.solution = base;
    std::array<std::array<FE::Real, 3>, 3> defect{};
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            defect[r][c] = ref2[r][c] > 1e-28 ? std::sqrt(diff2[r][c] / ref2[r][c]) : std::sqrt(diff2[r][c]);
        }
    }
    return defect;
}

// A distorted, moving mesh (|d| <= 0.08, w = d - d_prev) under a non-trivial
// velocity and pressure.
void setMovingState(FittedALEFixture& ale)
{
    const auto n = static_cast<FE::GlobalIndex>(ale.solution.size());
    ale.previous.assign(ale.solution.size(), 0.0);
    for (FE::GlobalIndex i = 0; i < n; ++i) {
        const auto k = static_cast<std::size_t>(i);
        const auto x = static_cast<FE::Real>(i);
        if (fixtureFieldIndex(ale, i) == 2) {
            ale.solution[k] = FE::Real(0.08) * std::sin(FE::Real(2.1) * x + FE::Real(0.5));
            ale.previous[k] = FE::Real(0.5) * ale.solution[k];
        } else {
            ale.solution[k] = FE::Real(0.25) * std::sin(FE::Real(1.3) * x + FE::Real(0.2));
            ale.previous[k] = FE::Real(0.9) * ale.solution[k];
        }
    }
    ale.explicit_previous = true;
}

// Characterization of the coupled-ALE Jacobian of the production modules
// (decision 4A): Galerkin Navier-Stokes with ALE convection and the moving-
// control-volume term, then with VMS, then with a fitted free surface
// (surface tension, MeshNitsche kinematic row), each with the harmonic mesh
// motion.  The u and p columns match central differences.  The mesh columns
// of every row block miss part of the shape derivative: the continuity rows
// get no mesh block at all without VMS (q div(u) has no frame-explicit
// terminal), the fluid rows miss the variation of grad and div, of the VMS
// metric and of the measure, and the mesh operator is geometry-constant.  The
// printed defects are what a complete shape-derivative tangent path must
// remove.
TEST(FittedFreeSurfaceALE, CoupledJacobianLacksPartOfTheShapeDerivativeInEveryRowBlock)
{
    struct Configuration {
        const char* name;
        bool vms;
        bool surface;
    };
    for (const auto& configuration : {Configuration{"Galerkin", false, false},
                                      Configuration{"VMS", true, false},
                                      Configuration{"VMS + fitted free surface", true, true}}) {
        SCOPED_TRACE(configuration.name);
        auto opts = coupledALEOptions();
        opts.enable_convection = true;
        opts.enable_vms = configuration.vms;
        std::optional<ns::IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary> surface;
        if (configuration.surface) {
            surface = fittedSurface(0.0, 0.07);
        }
        FittedALEFixture ale(makeRectangleTriangleMesh(), surface, opts);
        setMovingState(ale);
        const auto defect = fixtureJacobianBlockDefects(ale, 1e-6);
        const std::array<const char*, 3> names{"u", "p", "d"};
        for (std::size_t r = 0; r < 3; ++r) {
            std::cout << "[ shape derivative ] " << configuration.name << ", " << names[r] << " rows:";
            for (std::size_t c = 0; c < 3; ++c) {
                std::cout << " " << names[c] << " columns " << defect[r][c];
            }
            std::cout << "\n";
        }
        for (std::size_t r = 0; r < 3; ++r) {
            SCOPED_TRACE(std::string("rows ") + names[r]);
            EXPECT_LT(defect[r][0], 1e-6);
            EXPECT_LT(defect[r][1], 1e-6);
            EXPECT_GT(defect[r][2], 1e-2);
        }
    }
}

// Free-stream preservation on a moving mesh: a uniform velocity U with zero
// pressure on a non-uniformly moving mesh.  The non-conservative and the
// conservative ALE momentum forms leave the fluid rows at round-off; the
// legacy form, the default, adds the moving-control-volume term
// rho div(w) U.v (and its share of the VMS strong residual) to the
// non-conservative transport and does not, under Backward Euler and the
// first-order generalized-alpha stage alike.
TEST(FittedFreeSurfaceALE, FreeStreamOnAMovingMeshIsKeptByTheNonConservativeAndConservativeForms)
{
    const auto alpha = FE::timestepping::utils::generalizedAlphaFirstOrderFromRhoInf(0.5);
    const FE::timestepping::GeneralizedAlphaFirstOrderIntegrator generalized_alpha(
        FE::timestepping::GeneralizedAlphaFirstOrderIntegratorOptions{
            .alpha_m = alpha.alpha_m, .alpha_f = alpha.alpha_f, .gamma = alpha.gamma, .history_rate_order = 0});
    const std::array<std::pair<ns::ALEMomentumForm, const char*>, 3> forms{{
        {ns::ALEMomentumForm::Legacy, "legacy"},
        {ns::ALEMomentumForm::NonConservative, "non-conservative"},
        {ns::ALEMomentumForm::Conservative, "conservative"},
    }};
    for (const bool use_generalized_alpha : {false, true}) {
        for (const auto& [form, form_name] : forms) {
            auto opts = coupledALEOptions();
            opts.enable_convection = true;
            opts.enable_vms = true;
            opts.ale_momentum_form = form;
            FittedALEFixture ale(makeRectangleTriangleMesh(), std::nullopt, opts);
            const auto n = static_cast<FE::GlobalIndex>(ale.solution.size());
            for (FE::GlobalIndex v = 0; v < static_cast<FE::GlobalIndex>(ale.mesh->n_vertices()); ++v) {
                ale.solution[static_cast<std::size_t>(vertexDof(*ale.system, ale.u, v, 0))] = 0.7;
                ale.solution[static_cast<std::size_t>(vertexDof(*ale.system, ale.u, v, 1))] = -0.3;
            }
            ale.previous = ale.solution;
            ale.previous_rate.assign(ale.solution.size(), 0.0);
            for (FE::GlobalIndex i = 0; i < n; ++i) {
                if (fixtureFieldIndex(ale, i) == 2) {
                    const auto x = static_cast<FE::Real>(i);
                    ale.solution[static_cast<std::size_t>(i)] = FE::Real(0.08) * std::sin(FE::Real(2.1) * x + FE::Real(0.5));
                    ale.previous_rate[static_cast<std::size_t>(i)] = FE::Real(0.3) * ale.solution[static_cast<std::size_t>(i)];
                }
            }
            ale.explicit_previous = true;
            if (use_generalized_alpha) {
                ale.integrator = &generalized_alpha;
            } else {
                ale.previous_rate.clear();
            }
            const auto r = ale.residual();
            FE::Real fluid2 = 0.0;
            for (FE::GlobalIndex i = 0; i < n; ++i) {
                if (fixtureFieldIndex(ale, i) != 2) {
                    fluid2 += r[static_cast<std::size_t>(i)] * r[static_cast<std::size_t>(i)];
                }
            }
            const FE::Real fluid = std::sqrt(fluid2);
            const std::string scheme = use_generalized_alpha ? "generalized-alpha" : "Backward Euler";
            std::cout << "[ free stream ] " << scheme << ", " << form_name << " ALE momentum form: |R_fluid| = "
                      << fluid << "\n";
            SCOPED_TRACE(scheme + ", " + form_name);
            if (form == ns::ALEMomentumForm::Legacy) {
                EXPECT_GT(fluid, 1e-3);
            } else {
                EXPECT_LT(fluid, 1e-13);
            }
        }
    }
}

#endif // SVMP_FE_WITH_MESH

} // namespace
} // namespace test
} // namespace Physics
} // namespace svmp
