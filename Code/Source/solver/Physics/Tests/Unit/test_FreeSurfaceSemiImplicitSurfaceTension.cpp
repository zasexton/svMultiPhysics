/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

// Unit tests of the lagged normal-increment capillary term
// (Surface_tension_semi_implicit=NormalIncrement):
//   R_SI = gamma dt_eff int_Gamma grad_Gamma((u-u_ref).n) . grad_Gamma(v.n).
// One affine Tetra4 cell carries a planar generated interface, so the
// velocity block is known in closed form:
//   J_SI[(a,c),(b,d)] = gamma dt_eff |Gamma_K| n_c n_d (P grad N_a).(P grad N_b).

#include <gtest/gtest.h>

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceSemiImplicitSurfaceTension.h"
#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"
#include "Physics/Tests/Unit/PhysicsTestHelpers.h"

#include "FE/Assembly/CutIntegrationContext.h"
#include "FE/Assembly/GlobalSystemView.h"
#include "FE/Assembly/TimeIntegrationContext.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Spaces/SpaceFactory.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/TimeIntegrator.h"
#include "FE/TimeStepping/GeneralizedAlpha.h"
#include "Interfaces/LevelSetInterfaceDomain.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace svmp {
namespace Physics {
namespace test {
namespace {

namespace ns = formulations::navier_stokes;
using FE::Real;

constexpr int kInterfaceMarker = 417;
constexpr Real kSurfaceTension = 0.7;
constexpr Real kTimeStep = 0.1;

// Plane n.x = offset cutting the unit tetra (vertices 0, e_x, e_y, e_z).
struct PlanarCut {
    std::array<Real, 3> normal{};
    Real offset{0.0};
    Real area{0.0};
    std::array<Real, 3> centroid{};
    Real negative_volume{0.0};
};

PlanarCut horizontalCut()
{
    // z = 1/4: the triangle (0,0,h), (1-h,0,h), (0,1-h,h).
    constexpr Real h = 0.25;
    PlanarCut cut;
    cut.normal = {0.0, 0.0, 1.0};
    cut.offset = h;
    cut.area = 0.5 * (1.0 - h) * (1.0 - h);
    cut.centroid = {(1.0 - h) / 3.0, (1.0 - h) / 3.0, h};
    cut.negative_volume = (1.0 - std::pow(1.0 - h, 3)) / 6.0;
    return cut;
}

PlanarCut tiltedCut()
{
    // (x+y+z)/sqrt(3) = s/sqrt(3), s = 0.6: the triangle s e_x, s e_y, s e_z.
    constexpr Real s = 0.6;
    const Real inv_sqrt3 = 1.0 / std::sqrt(3.0);
    PlanarCut cut;
    cut.normal = {inv_sqrt3, inv_sqrt3, inv_sqrt3};
    cut.offset = s * inv_sqrt3;
    cut.area = 0.5 * std::sqrt(3.0) * s * s;
    cut.centroid = {s / 3.0, s / 3.0, s / 3.0};
    cut.negative_volume = s * s * s / 6.0;
    return cut;
}

std::shared_ptr<FE::assembly::CutIntegrationContext> makePlanarCutContext(
    FE::FieldId level_set_field,
    const PlanarCut& cut)
{
    namespace interfaces = FE::interfaces;

    interfaces::CutInterfaceDomainRequest request;
    request.source = interfaces::LevelSetInterfaceSource::fromField(
        level_set_field, /*layout_revision=*/0u, /*value_revision=*/1u);
    request.interface_marker = kInterfaceMarker;
    request.quadrature_order = 0;
    request.interface_quadrature_order = 0;
    request.volume_quadrature_order = 0;
    interfaces::LevelSetInterfaceDomain domain(request);

    // P1 gradients are cellwise constant, so one point at the centroid with
    // the full facet area integrates every term of the block exactly.
    interfaces::CutInterfaceFragment fragment;
    fragment.interface_marker = kInterfaceMarker;
    fragment.parent_cell = 0;
    fragment.local_fragment_index = 0;
    fragment.stable_id = 21;
    fragment.kind = interfaces::CutInterfaceFragmentKind::Polygon;
    fragment.measure = cut.area;
    fragment.normal = cut.normal;
    fragment.quadrature_points.push_back(interfaces::CutInterfaceQuadraturePoint{
        .point = cut.centroid,
        .parent_coordinate = cut.centroid,
        .normal = cut.normal,
        .weight = cut.area,
    });
    domain.addFragment(std::move(fragment));

    constexpr Real parent_volume = 1.0 / 6.0;
    for (const auto side : {FE::geometry::CutIntegrationSide::Negative,
                            FE::geometry::CutIntegrationSide::Positive}) {
        interfaces::CutInterfaceVolumeRegion region;
        region.interface_marker = kInterfaceMarker;
        region.parent_cell = 0;
        region.side = side;
        const bool negative = side == FE::geometry::CutIntegrationSide::Negative;
        region.local_region_index = negative ? 0 : 1;
        region.stable_id = negative ? 22 : 23;
        region.measure = negative ? cut.negative_volume
                                  : parent_volume - cut.negative_volume;
        region.parent_measure = parent_volume;
        region.volume_fraction = region.measure / parent_volume;
        region.centroid = {0.25, 0.25, 0.25};
        region.normal = cut.normal;
        domain.addVolumeRegion(std::move(region));
    }

    auto context = std::make_shared<FE::assembly::CutIntegrationContext>();
    context->addGeneratedInterfaceDomain(
        domain, FE::geometry::CutIntegrationSide::Negative);
    return context;
}

ns::FreeSurfaceBoundary unfittedBoundary(
    ns::FreeSurfaceSurfaceTensionSemiImplicit semi_implicit,
    Real surface_tension = kSurfaceTension)
{
    return ns::FreeSurfaceBoundary{
        .implementation = ns::FreeSurfaceImplementation::UnfittedLevelSet,
        .interface_marker = kInterfaceMarker,
        .level_set_field_name = "phi",
        .geometry_tangent_policy = "RefreshedFrozenQuadrature",
        .active_domain = ns::FreeSurfaceActiveDomain::LevelSetNegative,
        .external_pressure = 0.0,
        .surface_tension = surface_tension,
        .surface_tension_form = ns::FreeSurfaceSurfaceTensionForm::SurfaceStress,
        .surface_tension_semi_implicit = semi_implicit,
        .use_level_set_curvature = false,
        .small_cut_aggregation = false,
    };
}

ns::IncompressibleNavierStokesVMSOptions navierStokesOptions(
    ns::FreeSurfaceBoundary boundary)
{
    ns::IncompressibleNavierStokesVMSOptions options;
    options.velocity_field_name = "u";
    options.pressure_field_name = "p";
    options.density = 1.0;
    options.viscosity = 0.01;
    options.enable_convection = false;
    options.enable_vms = false;
    options.free_surface.push_back(std::move(boundary));
    return options;
}

// One registered single-tetra Navier--Stokes system with a planar cut.
struct Probe {
    std::shared_ptr<SingleTetraMeshAccess> mesh{};
    std::shared_ptr<FE::spaces::FunctionSpace> velocity_space{};
    std::shared_ptr<FE::spaces::FunctionSpace> pressure_space{};
    std::unique_ptr<ns::IncompressibleNavierStokesVMSModule> module{};
    std::unique_ptr<FE::systems::FESystem> system{};
    FE::FieldId phi{FE::INVALID_FIELD_ID};
    FE::FieldId velocity{FE::INVALID_FIELD_ID};
    FE::FieldId reference{FE::INVALID_FIELD_ID};
    std::string log{};
};

std::unique_ptr<Probe> makeProbe(
    ns::FreeSurfaceSurfaceTensionSemiImplicit semi_implicit,
    const PlanarCut& cut)
{
    auto probe = std::make_unique<Probe>();
    probe->mesh = std::make_shared<SingleTetraMeshAccess>();
    probe->velocity_space = FE::spaces::VectorSpace(
        FE::spaces::SpaceType::H1, probe->mesh, /*order=*/1, /*components=*/3);
    probe->pressure_space = FE::spaces::Space(
        FE::spaces::SpaceType::H1, probe->mesh, /*order=*/1, /*components=*/1);
    probe->system = std::make_unique<FE::systems::FESystem>(probe->mesh);
    probe->phi = probe->system->addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = probe->pressure_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });
    probe->module = std::make_unique<ns::IncompressibleNavierStokesVMSModule>(
        probe->velocity_space,
        probe->pressure_space,
        navierStokesOptions(unfittedBoundary(semi_implicit)));
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    probe->module->registerOn(*probe->system);
    probe->log = testing::internal::GetCapturedStdout();
    probe->log += testing::internal::GetCapturedStderr();
    probe->system->setCutIntegrationContext(
        makePlanarCutContext(probe->phi, cut));
    probe->system->setup({}, makeSingleTetraSetupInputs());
    probe->system->setPrescribedFieldCoefficients(
        probe->phi,
        std::vector<Real>{
            -cut.offset,
            cut.normal[0] - cut.offset,
            cut.normal[1] - cut.offset,
            cut.normal[2] - cut.offset,
        });
    probe->velocity = probe->system->findFieldByName("u");
    probe->reference = probe->system->findFieldByName(std::string(
        ns::kFreeSurfaceSemiImplicitReferenceVelocityFieldName));
    return probe;
}

// Global index of velocity component c at vertex a.
FE::GlobalIndex velocityDof(const Probe& probe, int vertex, int component)
{
    const auto* entity_map =
        probe.system->fieldDofHandler(probe.velocity).getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::runtime_error("velocity field has no entity DOF map");
    }
    const auto dofs = entity_map->getVertexDofs(vertex);
    return probe.system->fieldDofOffset(probe.velocity) +
           dofs[static_cast<std::size_t>(component)];
}

// Field-local coefficient vector of a vector field from nodal values.
std::vector<Real> fieldCoefficients(const Probe& probe,
                                    FE::FieldId field,
                                    const std::array<std::array<Real, 3>, 4>& nodal)
{
    const auto& handler = probe.system->fieldDofHandler(field);
    std::vector<Real> coefficients(
        static_cast<std::size_t>(handler.getNumDofs()), 0.0);
    const auto* entity_map = handler.getEntityDofMap();
    for (int vertex = 0; vertex < 4; ++vertex) {
        const auto dofs = entity_map->getVertexDofs(vertex);
        for (int c = 0; c < 3; ++c) {
            coefficients[static_cast<std::size_t>(
                dofs[static_cast<std::size_t>(c)])] =
                nodal[static_cast<std::size_t>(vertex)]
                     [static_cast<std::size_t>(c)];
        }
    }
    return coefficients;
}

enum class Integrator { BackwardEuler, GeneralizedAlphaRhoInfHalf };

// rho_inf = 0.5: alpha_m = 5/6, alpha_f = gamma = 2/3, so
// dt_eff = gamma alpha_f dt / alpha_m = (8/15) dt = 0.5333 dt.
Real expectedEffectiveTimeStep(Integrator integrator)
{
    return integrator == Integrator::BackwardEuler
               ? kTimeStep
               : (2.0 / 3.0) * (2.0 / 3.0) / (5.0 / 6.0) * kTimeStep;
}

struct StateBundle {
    std::vector<Real> u{};
    std::vector<Real> u_prev{};
    std::vector<Real> u_prev2{};
    std::array<std::span<const Real>, 2> history{};
    FE::assembly::TimeIntegrationContext context{};
    FE::systems::SystemStateView view{};
};

std::unique_ptr<StateBundle> makeState(const Probe& probe,
                                       const std::array<std::array<Real, 3>, 4>& velocity,
                                       Integrator integrator)
{
    auto state = std::make_unique<StateBundle>();
    const auto n = static_cast<std::size_t>(
        probe.system->dofHandler().getNumDofs());
    state->u.assign(n, 0.0);
    state->u_prev.assign(n, 0.0);
    state->u_prev2.assign(n, 0.0);
    for (int a = 0; a < 4; ++a) {
        for (int c = 0; c < 3; ++c) {
            state->u[static_cast<std::size_t>(velocityDof(probe, a, c))] =
                velocity[static_cast<std::size_t>(a)][static_cast<std::size_t>(c)];
        }
    }
    state->history = {std::span<const Real>(state->u_prev),
                      std::span<const Real>(state->u_prev2)};
    state->view.dt = kTimeStep;
    state->view.dt_prev = kTimeStep;
    state->view.u = std::span<const Real>(state->u);
    state->view.u_prev = std::span<const Real>(state->u_prev);
    state->view.u_prev2 = std::span<const Real>(state->u_prev2);
    state->view.u_history =
        std::span<const std::span<const Real>>(state->history);
    if (integrator == Integrator::BackwardEuler) {
        const FE::systems::BackwardDifferenceIntegrator be;
        state->context = be.buildContext(1, state->view);
    } else {
        const FE::timestepping::GeneralizedAlphaFirstOrderIntegrator ga(
            FE::timestepping::GeneralizedAlphaFirstOrderIntegratorOptions{
                .alpha_m = 5.0 / 6.0,
                .alpha_f = 2.0 / 3.0,
                .gamma = 2.0 / 3.0,
                .history_rate_order = 0,
            });
        state->context = ga.buildContext(1, state->view);
    }
    state->view.time_integration = &state->context;
    return state;
}

std::vector<Real> residual(Probe& probe, const StateBundle& state)
{
    const auto n = probe.system->dofHandler().getNumDofs();
    FE::assembly::DenseVectorView r(n);
    r.zero();
    FE::systems::AssemblyRequest request;
    request.op = "equations";
    request.want_vector = true;
    const auto result =
        probe.system->assemble(request, state.view, nullptr, &r);
    EXPECT_TRUE(result.success) << result.error_message;
    std::vector<Real> out(static_cast<std::size_t>(n));
    for (FE::GlobalIndex i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i)] = r[i];
    }
    return out;
}

std::vector<Real> jacobian(Probe& probe, const StateBundle& state)
{
    const auto n = probe.system->dofHandler().getNumDofs();
    FE::assembly::DenseMatrixView J(n);
    J.zero();
    FE::systems::AssemblyRequest request;
    request.op = "equations";
    request.want_matrix = true;
    const auto result =
        probe.system->assemble(request, state.view, &J, nullptr);
    EXPECT_TRUE(result.success) << result.error_message;
    std::vector<Real> out(static_cast<std::size_t>(n * n));
    for (FE::GlobalIndex i = 0; i < n; ++i) {
        for (FE::GlobalIndex j = 0; j < n; ++j) {
            out[static_cast<std::size_t>(i * n + j)] = J.getMatrixEntry(i, j);
        }
    }
    return out;
}

constexpr std::array<std::array<Real, 3>, 4> kVelocity{{
    {{0.30, -0.20, 0.50}},
    {{-0.40, 0.10, 0.20}},
    {{0.15, 0.60, -0.35}},
    {{0.05, -0.25, 0.45}},
}};

constexpr std::array<std::array<Real, 3>, 4> kReferenceVelocity{{
    {{-0.10, 0.30, 0.20}},
    {{0.20, -0.15, 0.40}},
    {{0.35, 0.05, 0.10}},
    {{-0.30, 0.25, -0.20}},
}};

// gamma dt_eff |Gamma| n_c n_d (P grad N_a).(P grad N_b), ordered (a,c).
std::array<std::array<Real, 12>, 12> analyticBlock(const PlanarCut& cut,
                                                   Real effective_time_step)
{
    const std::array<std::array<Real, 3>, 4> grad_n{{
        {{-1.0, -1.0, -1.0}},
        {{1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}},
        {{0.0, 0.0, 1.0}},
    }};
    std::array<std::array<Real, 3>, 4> surface_grad{};
    for (std::size_t a = 0; a < 4; ++a) {
        Real normal_part = 0.0;
        for (std::size_t k = 0; k < 3; ++k) {
            normal_part += grad_n[a][k] * cut.normal[k];
        }
        for (std::size_t k = 0; k < 3; ++k) {
            surface_grad[a][k] = grad_n[a][k] - normal_part * cut.normal[k];
        }
    }
    std::array<std::array<Real, 12>, 12> block{};
    const Real scale = kSurfaceTension * effective_time_step * cut.area;
    for (std::size_t a = 0; a < 4; ++a) {
        for (std::size_t b = 0; b < 4; ++b) {
            Real g = 0.0;
            for (std::size_t k = 0; k < 3; ++k) {
                g += surface_grad[a][k] * surface_grad[b][k];
            }
            for (std::size_t c = 0; c < 3; ++c) {
                for (std::size_t d = 0; d < 3; ++d) {
                    block[3 * a + c][3 * b + d] =
                        scale * cut.normal[c] * cut.normal[d] * g;
                }
            }
        }
    }
    return block;
}

// Eigenvalues of a symmetric matrix (cyclic Jacobi).
std::vector<Real> symmetricEigenvalues(std::array<std::array<Real, 12>, 12> m)
{
    constexpr std::size_t n = 12;
    for (int sweep = 0; sweep < 100; ++sweep) {
        Real off = 0.0;
        for (std::size_t p = 0; p < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                off += m[p][q] * m[p][q];
            }
        }
        if (off < 1.0e-30) {
            break;
        }
        for (std::size_t p = 0; p < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                if (std::abs(m[p][q]) < 1.0e-300) {
                    continue;
                }
                const Real theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
                const Real t = (theta >= 0.0 ? 1.0 : -1.0) /
                               (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                const Real c = 1.0 / std::sqrt(t * t + 1.0);
                const Real s = t * c;
                for (std::size_t k = 0; k < n; ++k) {
                    const Real mkp = m[k][p];
                    const Real mkq = m[k][q];
                    m[k][p] = c * mkp - s * mkq;
                    m[k][q] = s * mkp + c * mkq;
                }
                for (std::size_t k = 0; k < n; ++k) {
                    const Real mpk = m[p][k];
                    const Real mqk = m[q][k];
                    m[p][k] = c * mpk - s * mqk;
                    m[q][k] = s * mpk + c * mqk;
                }
            }
        }
    }
    std::vector<Real> eigenvalues(n);
    for (std::size_t i = 0; i < n; ++i) {
        eigenvalues[i] = m[i][i];
    }
    std::sort(eigenvalues.begin(), eigenvalues.end());
    return eigenvalues;
}

struct BlockMeasurement {
    std::array<std::array<Real, 12>, 12> velocity_block{};
    Real off_velocity_max{0.0};
    std::vector<Real> residual_difference{};
    std::vector<Real> residual_on{};
    std::vector<Real> residual_off{};
};

// J_on - J_off on the velocity rows/columns, with u_ref set on the "on"
// system.  Everything outside the velocity-velocity block must vanish.
BlockMeasurement measureBlock(const PlanarCut& cut,
                              Integrator integrator,
                              const std::array<std::array<Real, 3>, 4>& velocity,
                              const std::array<std::array<Real, 3>, 4>& reference)
{
    auto on = makeProbe(ns::FreeSurfaceSurfaceTensionSemiImplicit::NormalIncrement, cut);
    auto off = makeProbe(ns::FreeSurfaceSurfaceTensionSemiImplicit::None, cut);
    EXPECT_NE(on->reference, FE::INVALID_FIELD_ID);
    on->system->setPrescribedFieldCoefficients(
        on->reference, fieldCoefficients(*on, on->reference, reference));
    const auto state_on = makeState(*on, velocity, integrator);
    const auto state_off = makeState(*off, velocity, integrator);
    const auto j_on = jacobian(*on, *state_on);
    const auto j_off = jacobian(*off, *state_off);

    BlockMeasurement out;
    out.residual_on = residual(*on, *state_on);
    out.residual_off = residual(*off, *state_off);
    const auto n = static_cast<std::size_t>(on->system->dofHandler().getNumDofs());
    EXPECT_EQ(n, static_cast<std::size_t>(off->system->dofHandler().getNumDofs()));
    out.residual_difference.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        out.residual_difference[i] = out.residual_on[i] - out.residual_off[i];
    }
    std::vector<int> local_of(n, -1);
    for (int a = 0; a < 4; ++a) {
        for (int c = 0; c < 3; ++c) {
            local_of[static_cast<std::size_t>(velocityDof(*on, a, c))] = 3 * a + c;
            EXPECT_EQ(velocityDof(*on, a, c), velocityDof(*off, a, c));
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const Real difference = j_on[i * n + j] - j_off[i * n + j];
            if (local_of[i] >= 0 && local_of[j] >= 0) {
                out.velocity_block[static_cast<std::size_t>(local_of[i])]
                                  [static_cast<std::size_t>(local_of[j])] = difference;
            } else {
                out.off_velocity_max =
                    std::max(out.off_velocity_max, std::abs(difference));
            }
        }
    }
    return out;
}

TEST(FreeSurfaceSemiImplicitSurfaceTension,
     JacobianBlockIsTheAnalyticPlanarLaplaceBeltramiBlock)
{
    for (const auto& [cut_name, cut] :
         {std::pair{"horizontal", horizontalCut()},
          std::pair{"tilted", tiltedCut()}}) {
        for (const auto integrator : {Integrator::BackwardEuler,
                                      Integrator::GeneralizedAlphaRhoInfHalf}) {
            SCOPED_TRACE(std::string(cut_name) +
                         (integrator == Integrator::BackwardEuler
                              ? " backward Euler"
                              : " generalized-alpha rho_inf=0.5"));
            const Real dt_eff = expectedEffectiveTimeStep(integrator);
            const auto measured =
                measureBlock(cut, integrator, kVelocity, kReferenceVelocity);
            const auto expected = analyticBlock(cut, dt_eff);
            Real scale = 0.0;
            for (const auto& row : expected) {
                for (const auto value : row) {
                    scale = std::max(scale, std::abs(value));
                }
            }
            ASSERT_GT(scale, 0.0);
            EXPECT_LE(measured.off_velocity_max, 1.0e-13 * scale)
                << "the term must touch only the velocity-velocity block";
            for (std::size_t i = 0; i < 12; ++i) {
                for (std::size_t j = 0; j < 12; ++j) {
                    EXPECT_NEAR(measured.velocity_block[i][j], expected[i][j],
                                1.0e-12 * scale)
                        << "entry (" << i << ", " << j << ")";
                    EXPECT_NEAR(measured.velocity_block[i][j],
                                measured.velocity_block[j][i], 1.0e-13 * scale)
                        << "symmetry (" << i << ", " << j << ")";
                }
            }

            // Symmetric positive semidefinite of rank 2: n n^T has rank one
            // and the projected P1 gradients span the 2D tangent plane.
            const auto eigenvalues =
                symmetricEigenvalues(measured.velocity_block);
            EXPECT_GE(eigenvalues.front(), -1.0e-12 * scale);
            const auto positive = std::count_if(
                eigenvalues.begin(), eigenvalues.end(),
                [&](Real value) { return value > 1.0e-10 * scale; });
            EXPECT_EQ(positive, 2);

            // Kernel: every constant field and every tangential field.
            const std::array<Real, 3> constant{0.3, -1.2, 0.7};
            std::array<Real, 3> tangent_a{};
            std::array<Real, 3> tangent_b{};
            {
                const std::array<Real, 3> seed_a{1.0, 0.0, 0.0};
                const std::array<Real, 3> seed_b{0.0, 1.0, 0.0};
                Real na = 0.0;
                Real nb = 0.0;
                for (std::size_t k = 0; k < 3; ++k) {
                    na += seed_a[k] * cut.normal[k];
                    nb += seed_b[k] * cut.normal[k];
                }
                for (std::size_t k = 0; k < 3; ++k) {
                    tangent_a[k] = seed_a[k] - na * cut.normal[k];
                    tangent_b[k] = seed_b[k] - nb * cut.normal[k];
                }
            }
            for (int field = 0; field < 2; ++field) {
                std::array<Real, 12> w{};
                for (std::size_t a = 0; a < 4; ++a) {
                    for (std::size_t c = 0; c < 3; ++c) {
                        w[3 * a + c] =
                            field == 0
                                ? constant[c]
                                : (1.0 + a) * tangent_a[c] +
                                      static_cast<Real>(a * a) * tangent_b[c];
                    }
                }
                for (std::size_t i = 0; i < 12; ++i) {
                    Real action = 0.0;
                    for (std::size_t j = 0; j < 12; ++j) {
                        action += measured.velocity_block[i][j] * w[j];
                    }
                    EXPECT_NEAR(action, 0.0, 1.0e-12 * scale)
                        << (field == 0 ? "constant" : "tangential")
                        << " field, row " << i;
                }
            }

            // The residual is exactly linear: R_on - R_off = J_SI (u - u_ref).
            auto on = makeProbe(
                ns::FreeSurfaceSurfaceTensionSemiImplicit::NormalIncrement, cut);
            for (int a = 0; a < 4; ++a) {
                for (int c = 0; c < 3; ++c) {
                    const auto row = static_cast<std::size_t>(3 * a + c);
                    Real expected_residual = 0.0;
                    for (std::size_t b = 0; b < 4; ++b) {
                        for (std::size_t d = 0; d < 3; ++d) {
                            expected_residual +=
                                expected[row][3 * b + d] *
                                (kVelocity[b][d] - kReferenceVelocity[b][d]);
                        }
                    }
                    EXPECT_NEAR(
                        measured.residual_difference[static_cast<std::size_t>(
                            velocityDof(*on, a, c))],
                        expected_residual, 1.0e-12 * scale)
                        << "residual row (" << a << ", " << c << ")";
                }
            }
        }
    }
}

TEST(FreeSurfaceSemiImplicitSurfaceTension, JacobianMatchesCentralFiniteDifferences)
{
    for (const auto& cut : {horizontalCut(), tiltedCut()}) {
        auto on = makeProbe(
            ns::FreeSurfaceSurfaceTensionSemiImplicit::NormalIncrement, cut);
        on->system->setPrescribedFieldCoefficients(
            on->reference,
            fieldCoefficients(*on, on->reference, kReferenceVelocity));
        const auto state = makeState(
            *on, kVelocity, Integrator::GeneralizedAlphaRhoInfHalf);
        expectOperatorJacobianMatchesCentralFD(
            *on->system, state->view, "equations");
    }
}

TEST(FreeSurfaceSemiImplicitSurfaceTension,
     ReferenceEqualToStateLeavesResidualUnchanged)
{
    for (const auto& cut : {horizontalCut(), tiltedCut()}) {
        for (const auto integrator : {Integrator::BackwardEuler,
                                      Integrator::GeneralizedAlphaRhoInfHalf}) {
            const auto measured =
                measureBlock(cut, integrator, kVelocity, kVelocity);
            Real scale = 0.0;
            Real largest_difference = 0.0;
            std::size_t bitwise_equal = 0u;
            for (std::size_t i = 0; i < measured.residual_on.size(); ++i) {
                scale = std::max(scale, std::abs(measured.residual_off[i]));
                largest_difference = std::max(
                    largest_difference,
                    std::abs(measured.residual_difference[i]));
                bitwise_equal +=
                    measured.residual_on[i] == measured.residual_off[i] ? 1u : 0u;
            }
            ASSERT_GT(scale, 0.0);
            EXPECT_EQ(bitwise_equal, measured.residual_on.size())
                << "largest difference " << largest_difference
                << " relative to " << scale;
            EXPECT_LE(largest_difference, 1.0e-15 * scale);
        }
    }
}

TEST(FreeSurfaceSemiImplicitSurfaceTension, EffectiveTimeStepIsTheIntegratorCoefficient)
{
    const auto cut = horizontalCut();
    const auto be = measureBlock(cut, Integrator::BackwardEuler,
                                 kVelocity, kReferenceVelocity);
    const auto ga = measureBlock(cut, Integrator::GeneralizedAlphaRhoInfHalf,
                                 kVelocity, kReferenceVelocity);
    // Entry (vertex 1, z; vertex 1, z) = gamma dt_eff |Gamma| |P grad N_1|^2.
    const std::size_t zz = 3 * 1 + 2;
    const Real unit = kSurfaceTension * cut.area * 1.0;
    EXPECT_NEAR(be.velocity_block[zz][zz] / unit, kTimeStep, 1.0e-14);
    EXPECT_NEAR(ga.velocity_block[zz][zz] / unit, 0.5333333333333333 * kTimeStep,
                1.0e-14);
    EXPECT_NEAR(ga.velocity_block[zz][zz] / be.velocity_block[zz][zz],
                8.0 / 15.0, 1.0e-13);
}

TEST(FreeSurfaceSemiImplicitSurfaceTension, OptionOffRegistersNothing)
{
    const auto cut = horizontalCut();
    auto off = makeProbe(ns::FreeSurfaceSurfaceTensionSemiImplicit::None, cut);
    EXPECT_EQ(off->reference, FE::INVALID_FIELD_ID);
    EXPECT_EQ(off->log.find("semi-implicit"), std::string::npos);
    const auto artifact = off->module->effectiveConfigurationArtifact();
    ASSERT_TRUE(artifact.has_value());
    EXPECT_EQ(artifact->json.find("surface_tension_semi_implicit"),
              std::string::npos);

    auto on = makeProbe(
        ns::FreeSurfaceSurfaceTensionSemiImplicit::NormalIncrement, cut);
    EXPECT_NE(on->reference, FE::INVALID_FIELD_ID);
    EXPECT_NE(on->log.find(
                  "diagnostic=free_surface_semi_implicit_normal_increment"),
              std::string::npos);
    const auto on_artifact = on->module->effectiveConfigurationArtifact();
    ASSERT_TRUE(on_artifact.has_value());
    EXPECT_NE(on_artifact->json.find(
                  "\"surface_tension_semi_implicit\":\"NormalIncrement\""),
              std::string::npos);
}

void expectRegistrationRejected(ns::FreeSurfaceBoundary boundary,
                                const std::string& expected,
                                int velocity_order = 1)
{
    auto mesh = std::make_shared<SingleTetraMeshAccess>();
    auto velocity_space = FE::spaces::VectorSpace(
        FE::spaces::SpaceType::H1, mesh, velocity_order, 3);
    auto pressure_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, 1, 1);
    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = pressure_space,
        .components = 1,
        .source_kind = FE::systems::FieldSourceKind::PrescribedData,
    });
    ns::IncompressibleNavierStokesVMSModule module(
        velocity_space, pressure_space, navierStokesOptions(std::move(boundary)));
    const auto fields_before = system.registeredFieldCount();
    try {
        testing::internal::CaptureStdout();
        testing::internal::CaptureStderr();
        module.registerOn(system);
        (void)testing::internal::GetCapturedStdout();
        (void)testing::internal::GetCapturedStderr();
        ADD_FAILURE() << "expected rejection: " << expected;
    } catch (const std::exception& error) {
        (void)testing::internal::GetCapturedStdout();
        (void)testing::internal::GetCapturedStderr();
        EXPECT_NE(std::string(error.what()).find(expected), std::string::npos)
            << error.what();
    }
    EXPECT_EQ(system.registeredFieldCount(), fields_before)
        << "a rejected configuration must leave the system unchanged";
}

TEST(FreeSurfaceSemiImplicitSurfaceTension, RejectsConfigurationsOutsideTheValidatedScope)
{
    using SI = ns::FreeSurfaceSurfaceTensionSemiImplicit;
    const std::string requires_prefix =
        "Surface_tension_semi_implicit=NormalIncrement requires";

    {
        SCOPED_TRACE("supplied-curvature traction");
        auto bc = unfittedBoundary(SI::NormalIncrement);
        bc.surface_tension_form =
            ns::FreeSurfaceSurfaceTensionForm::CurvatureTraction;
        bc.curvature = 2.0;
        expectRegistrationRejected(bc, requires_prefix + " Surface_tension_form=SurfaceStress");
    }
    {
        SCOPED_TRACE("zero surface tension");
        expectRegistrationRejected(unfittedBoundary(SI::NormalIncrement, 0.0),
                                   requires_prefix + " a literal positive Surface_tension");
    }
    {
        SCOPED_TRACE("fitted ALE");
        auto bc = unfittedBoundary(SI::NormalIncrement);
        bc.implementation = ns::FreeSurfaceImplementation::FittedALE;
        bc.boundary_marker = 3;
        bc.active_domain = ns::FreeSurfaceActiveDomain::None;
        expectRegistrationRejected(bc, requires_prefix + " an exterior one-phase UnfittedLevelSet");
    }
    {
        SCOPED_TRACE("full-domain diagnostic");
        auto bc = unfittedBoundary(SI::NormalIncrement);
        bc.active_domain = ns::FreeSurfaceActiveDomain::None;
        bc.allow_full_domain_unfitted_free_surface = true;
        expectRegistrationRejected(bc, requires_prefix + " Active_domain=LevelSetNegative");
    }
    {
        SCOPED_TRACE("high-order generated geometry");
        auto bc = unfittedBoundary(SI::NormalIncrement);
        bc.generated_interface_geometry = "HighOrderImplicit";
        expectRegistrationRejected(bc, requires_prefix + " Generated_interface_geometry=LinearCorner");
    }
    {
        SCOPED_TRACE("differentiated geometry");
        auto bc = unfittedBoundary(SI::NormalIncrement);
        bc.geometry_tangent_policy = "DifferentiatedQuadrature";
        expectRegistrationRejected(bc, requires_prefix + " Geometry_tangent_policy=RefreshedFrozenQuadrature");
    }
    {
        SCOPED_TRACE("quadratic velocity");
        expectRegistrationRejected(unfittedBoundary(SI::NormalIncrement),
                                   requires_prefix + " an affine P1",
                                   /*velocity_order=*/2);
    }
}

} // namespace
} // namespace test
} // namespace Physics
} // namespace svmp
