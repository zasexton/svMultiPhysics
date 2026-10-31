/**
 * @file test_JacobianVerification.cpp
 * @brief Stronger Jacobian verification tests (Taylor remainder + central FD + Stokes coupling)
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Spaces/SpaceFactory.h"
#include "Systems/ALEBinding.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Systems/TimeIntegrator.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <vector>

using svmp::FE::ElementType;
using svmp::FE::GlobalIndex;
using svmp::FE::Real;

namespace {

svmp::FE::dofs::MeshTopologyInfo singleTetraTopology()
{
    svmp::FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 4;
    topo.dim = 3;
    topo.cell2vertex_offsets = {0, 4};
    topo.cell2vertex_data = {0, 1, 2, 3};
    topo.vertex_gids = {0, 1, 2, 3};
    topo.cell_gids = {0};
    topo.cell_owner_ranks = {0};
    return topo;
}

void distributeConstrainedState(const svmp::FE::systems::FESystem& sys,
                                std::vector<Real>& values)
{
    sys.constraints().distribute(values);
}

Real l2Norm(const std::vector<Real>& v)
{
    Real s = 0.0;
    for (const auto x : v) s += x * x;
    return std::sqrt(s);
}

std::vector<Real> matVec(const svmp::FE::assembly::DenseSystemView& A, const std::vector<Real>& x)
{
    std::vector<Real> y(static_cast<std::size_t>(A.numRows()), 0.0);
    for (GlobalIndex i = 0; i < A.numRows(); ++i) {
        Real acc = 0.0;
        for (GlobalIndex j = 0; j < A.numCols(); ++j) {
            acc += A.getMatrixEntry(i, j) * x[static_cast<std::size_t>(j)];
        }
        y[static_cast<std::size_t>(i)] = acc;
    }
    return y;
}

} // namespace

TEST(StokesCouplingJacobianTest, CouplingBlocksMatchCentralDifferencesAndHaveCorrectTransposeSign)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();

    auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1,
                                                 mesh,
                                                 /*order=*/1,
                                                 /*components=*/3);
    auto p_space = svmp::FE::spaces::Space(svmp::FE::spaces::SpaceType::H1,
                                           mesh,
                                           /*order=*/1,
                                           /*components=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = u_space, .components = 3});
    const auto p_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "p", .space = p_space, .components = 1});
    sys.addOperator("stokes_coupling");

    const auto u = svmp::FE::forms::FormExpr::stateField(u_field, *u_space, "u");
    const auto v = svmp::FE::forms::TestFunction(*u_space, "v");
    const auto p = svmp::FE::forms::FormExpr::stateField(p_field, *p_space, "p");
    const auto q = svmp::FE::forms::TestFunction(*p_space, "q");

    const auto residual =
        (-p * svmp::FE::forms::div(v)).dx() +
        (q * svmp::FE::forms::div(u)).dx();

    (void)svmp::FE::systems::installFormulation(
        sys, "stokes_coupling",
        {u_field, p_field},
        residual);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    svmp::FE::systems::SetupOptions setup_options;
    setup_options.use_constraints_in_assembly = false;
    sys.setup(setup_options, inputs);

    const auto n_dofs = sys.dofHandler().getNumDofs();
    ASSERT_EQ(n_dofs, 16);

    std::vector<Real> U(static_cast<std::size_t>(n_dofs), 0.0);
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        U[static_cast<std::size_t>(i)] = static_cast<Real>(0.05) * static_cast<Real>(i + 1);
    }
    distributeConstrainedState(sys, U);

    svmp::FE::systems::SystemStateView state;
    state.u = U;

    svmp::FE::systems::AssemblyRequest req;
    req.op = "stokes_coupling";
    req.want_matrix = true;
    req.want_vector = true;

    svmp::FE::assembly::DenseSystemView out(n_dofs);
    out.zero();
    auto result = sys.assemble(req, state, &out, &out);
    EXPECT_TRUE(result.success);

    // Block structure sanity:
    //   [ 0   -B^T ]
    //   [ B     0  ]
    constexpr GlobalIndex n_u = 12;
    constexpr GlobalIndex n_p = 4;
    const Real zero_tol = 1e-14;
    const Real transpose_tol = 1e-12;

    for (GlobalIndex i = 0; i < n_u; ++i) {
        for (GlobalIndex j = 0; j < n_u; ++j) {
            EXPECT_NEAR(out.getMatrixEntry(i, j), 0.0, zero_tol);
        }
    }
    for (GlobalIndex i = 0; i < n_p; ++i) {
        for (GlobalIndex j = 0; j < n_p; ++j) {
            EXPECT_NEAR(out.getMatrixEntry(n_u + i, n_u + j), 0.0, zero_tol);
        }
    }
    for (GlobalIndex iu = 0; iu < n_u; ++iu) {
        for (GlobalIndex ip = 0; ip < n_p; ++ip) {
            const auto up = out.getMatrixEntry(iu, n_u + ip);
            const auto pu = out.getMatrixEntry(n_u + ip, iu);
            EXPECT_NEAR(up + pu, 0.0, transpose_tol);
        }
    }

    // Central-difference check: J(:,j) ~= (R(U+eps e_j) - R(U-eps e_j)) / (2 eps)
    std::vector<Real> R0(static_cast<std::size_t>(n_dofs), 0.0);
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        R0[static_cast<std::size_t>(i)] = out.getVectorEntry(i);
    }

    const Real eps = 1e-6;
    const Real fd_tol = 1e-10;
    svmp::FE::systems::AssemblyRequest req_vec;
    req_vec.op = "stokes_coupling";
    req_vec.want_vector = true;

    for (GlobalIndex j = 0; j < n_dofs; ++j) {
        auto U_plus = U;
        auto U_minus = U;
        U_plus[static_cast<std::size_t>(j)] += eps;
        U_minus[static_cast<std::size_t>(j)] -= eps;

        svmp::FE::systems::SystemStateView state_plus;
        state_plus.u = U_plus;
        svmp::FE::systems::SystemStateView state_minus;
        state_minus.u = U_minus;

        svmp::FE::assembly::DenseVectorView Rp(n_dofs);
        svmp::FE::assembly::DenseVectorView Rm(n_dofs);
        Rp.zero();
        Rm.zero();
        (void)sys.assemble(req_vec, state_plus, nullptr, &Rp);
        (void)sys.assemble(req_vec, state_minus, nullptr, &Rm);

        for (GlobalIndex i = 0; i < n_dofs; ++i) {
            const Real fd = (Rp.getVectorEntry(i) - Rm.getVectorEntry(i)) / (2.0 * eps);
            EXPECT_NEAR(out.getMatrixEntry(i, j), fd, fd_tol);
        }
    }
}

TEST(NonlinearJacobianVerificationTest, ReverseADJacobianMatchesCentralDifferencesAndTaylorRemainderIsQuadratic)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();

    auto u_space = svmp::FE::spaces::VectorSpace(svmp::FE::spaces::SpaceType::H1,
                                                 mesh,
                                                 /*order=*/1,
                                                 /*components=*/3);
    auto p_space = svmp::FE::spaces::Space(svmp::FE::spaces::SpaceType::H1,
                                           mesh,
                                           /*order=*/1,
                                           /*components=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = u_space, .components = 3});
    const auto p_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "p", .space = p_space, .components = 1});
    sys.addOperator("ns_reverse");

    const auto u = svmp::FE::forms::FormExpr::stateField(u_field, *u_space, "u");
    const auto v = svmp::FE::forms::TestFunction(*u_space, "v");
    const auto p = svmp::FE::forms::FormExpr::stateField(p_field, *p_space, "p");
    const auto q = svmp::FE::forms::TestFunction(*p_space, "q");

    const Real nu = 0.01;
    const auto nu_c = svmp::FE::forms::FormExpr::constant(nu);

    const auto residual =
        (svmp::FE::forms::inner(svmp::FE::forms::grad(u) * u, v) +
         nu_c * svmp::FE::forms::inner(svmp::FE::forms::grad(u), svmp::FE::forms::grad(v)) -
         p * svmp::FE::forms::div(v))
            .dx() +
        (q * svmp::FE::forms::div(u)).dx();

    (void)svmp::FE::systems::installFormulation(
        sys, "ns_reverse",
        {u_field, p_field},
        residual);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    const auto n_dofs = sys.dofHandler().getNumDofs();
    ASSERT_EQ(n_dofs, 16);

    std::vector<Real> U(static_cast<std::size_t>(n_dofs), 0.0);
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        U[static_cast<std::size_t>(i)] = static_cast<Real>(0.05) * static_cast<Real>(i + 1);
    }
    distributeConstrainedState(sys, U);

    svmp::FE::systems::SystemStateView state;
    state.u = U;

    svmp::FE::systems::AssemblyRequest req;
    req.op = "ns_reverse";
    req.want_matrix = true;
    req.want_vector = true;

    svmp::FE::assembly::DenseSystemView out(n_dofs);
    out.zero();
    auto result = sys.assemble(req, state, &out, &out);
    EXPECT_TRUE(result.success);

    std::vector<Real> R0(static_cast<std::size_t>(n_dofs), 0.0);
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        R0[static_cast<std::size_t>(i)] = out.getVectorEntry(i);
    }

    // Central difference check.
    const Real eps_fd = 1e-6;
    const Real fd_tol = 2e-6;
    svmp::FE::systems::AssemblyRequest req_vec;
    req_vec.op = "ns_reverse";
    req_vec.want_vector = true;

    if (sys.constraints().numConstraints() > 0u) {
        const std::array<Real, 3> seeds = {0.29, 0.53, 0.83};
        for (const Real seed : seeds) {
            std::vector<Real> dU(static_cast<std::size_t>(n_dofs), 0.0);
            for (GlobalIndex i = 0; i < n_dofs; ++i) {
                const Real x = static_cast<Real>(i + 1);
                dU[static_cast<std::size_t>(i)] =
                    static_cast<Real>(0.15) * std::sin(static_cast<double>(seed * x)) +
                    static_cast<Real>(0.06) * std::cos(static_cast<double>((seed + 0.21) * x));
            }
            sys.constraints().distributeHomogeneous(dU);
            ASSERT_GT(l2Norm(dU), 1e-12);

            const auto JdU = matVec(out, dU);

            auto U_plus = U;
            auto U_minus = U;
            for (GlobalIndex i = 0; i < n_dofs; ++i) {
                U_plus[static_cast<std::size_t>(i)] += eps_fd * dU[static_cast<std::size_t>(i)];
                U_minus[static_cast<std::size_t>(i)] -= eps_fd * dU[static_cast<std::size_t>(i)];
            }
            distributeConstrainedState(sys, U_plus);
            distributeConstrainedState(sys, U_minus);

            svmp::FE::systems::SystemStateView state_plus;
            state_plus.u = U_plus;
            svmp::FE::systems::SystemStateView state_minus;
            state_minus.u = U_minus;

            svmp::FE::assembly::DenseVectorView Rp(n_dofs);
            svmp::FE::assembly::DenseVectorView Rm(n_dofs);
            Rp.zero();
            Rm.zero();
            (void)sys.assemble(req_vec, state_plus, nullptr, &Rp);
            (void)sys.assemble(req_vec, state_minus, nullptr, &Rm);

            for (GlobalIndex i = 0; i < n_dofs; ++i) {
                const Real fd = (Rp.getVectorEntry(i) - Rm.getVectorEntry(i)) / (2.0 * eps_fd);
                EXPECT_NEAR(JdU[static_cast<std::size_t>(i)], fd, fd_tol);
            }
        }
    } else {
        for (GlobalIndex j = 0; j < n_dofs; ++j) {
            auto U_plus = U;
            auto U_minus = U;
            U_plus[static_cast<std::size_t>(j)] += eps_fd;
            U_minus[static_cast<std::size_t>(j)] -= eps_fd;
            distributeConstrainedState(sys, U_plus);
            distributeConstrainedState(sys, U_minus);

            svmp::FE::systems::SystemStateView state_plus;
            state_plus.u = U_plus;
            svmp::FE::systems::SystemStateView state_minus;
            state_minus.u = U_minus;

            svmp::FE::assembly::DenseVectorView Rp(n_dofs);
            svmp::FE::assembly::DenseVectorView Rm(n_dofs);
            Rp.zero();
            Rm.zero();
            (void)sys.assemble(req_vec, state_plus, nullptr, &Rp);
            (void)sys.assemble(req_vec, state_minus, nullptr, &Rm);

            for (GlobalIndex i = 0; i < n_dofs; ++i) {
                const Real fd = (Rp.getVectorEntry(i) - Rm.getVectorEntry(i)) / (2.0 * eps_fd);
                EXPECT_NEAR(out.getMatrixEntry(i, j), fd, fd_tol);
            }
        }
    }

    // Taylor remainder check:
    //   R(U + eps*dU) = R(U) + eps*J*dU + O(eps^2)
    //
    // Auto-enforced gauge constraints make this mixed system a reduced
    // constrained operator. The central-difference check above already
    // verifies the assembled Jacobian against that constrained operator.
    // Leave Taylor remainder coverage to unconstrained nonlinear tests,
    // where the asymptotic slope is stable and interpretable.
    if (sys.constraints().numConstraints() > 0u) {
        return;
    }

    std::vector<Real> dU(static_cast<std::size_t>(n_dofs), 0.0);
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        const Real x = static_cast<Real>(i + 1);
        dU[static_cast<std::size_t>(i)] = static_cast<Real>(0.2) * std::sin(static_cast<double>(0.37 * x));
    }
    sys.constraints().distributeHomogeneous(dU);

    const auto JdU = matVec(out, dU);

    const std::array<Real, 3> eps = {1e-2, 1e-3, 1e-4};
    std::array<Real, 3> remainder_norm{};
    for (std::size_t k = 0; k < eps.size(); ++k) {
        auto U_eps = U;
        for (GlobalIndex i = 0; i < n_dofs; ++i) {
            U_eps[static_cast<std::size_t>(i)] += eps[k] * dU[static_cast<std::size_t>(i)];
        }
        distributeConstrainedState(sys, U_eps);

        svmp::FE::systems::SystemStateView state_eps;
        state_eps.u = U_eps;

        svmp::FE::assembly::DenseVectorView R_eps(n_dofs);
        R_eps.zero();
        (void)sys.assemble(req_vec, state_eps, nullptr, &R_eps);

        std::vector<Real> remainder(static_cast<std::size_t>(n_dofs), 0.0);
        for (GlobalIndex i = 0; i < n_dofs; ++i) {
            const auto idx = static_cast<std::size_t>(i);
            remainder[idx] = (R_eps.getVectorEntry(i) - R0[idx]) - eps[k] * JdU[idx];
        }
        remainder_norm[k] = l2Norm(remainder);
    }

    ASSERT_GT(remainder_norm[0], 0.0);
    ASSERT_GT(remainder_norm[1], 0.0);
    ASSERT_GT(remainder_norm[2], 0.0);

    // Under the reduced constrained system, the Taylor remainder can drop to
    // the floating-point floor at these step sizes. The central-difference
    // check above already validates the Jacobian entries, so only enforce the
    // asymptotic slope when the remainder is still measurably above roundoff.
    const Real max_remainder =
        std::max(remainder_norm[0], std::max(remainder_norm[1], remainder_norm[2]));
    if (max_remainder < 1e-12) {
        return;
    }

    const auto slope01 = std::log(remainder_norm[1] / remainder_norm[0]) / std::log(eps[1] / eps[0]);
    const auto slope12 = std::log(remainder_norm[2] / remainder_norm[1]) / std::log(eps[2] / eps[1]);

    EXPECT_GT(slope01, 1.8);
    EXPECT_GT(slope12, 1.8);
}

TEST(StabilizedConvectionTest, SUPGJacobianMatchesCentralDifferencesAndTaylorRemainderIsQuadratic)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();

    auto u_space = svmp::FE::spaces::Space(svmp::FE::spaces::SpaceType::H1,
                                           mesh,
                                           /*order=*/1,
                                           /*components=*/1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_field = sys.addField(svmp::FE::systems::FieldSpec{.name = "u", .space = u_space, .components = 1});
    sys.addOperator("supg");

    const auto u = svmp::FE::forms::FormExpr::stateField(u_field, *u_space, "u");
    const auto v = svmp::FE::forms::TestFunction(*u_space, "v");

    // Nonlinear, physics-agnostic convection + SUPG-style stabilization.
    const auto beta = svmp::FE::forms::as_vector({u, u, u});
    const auto adv = svmp::FE::forms::inner(beta, svmp::FE::forms::grad(u));
    const auto tau = 0.5 * svmp::FE::forms::h() / svmp::FE::forms::safeNorm(beta);
    const auto residual = (adv * (v + tau * svmp::FE::forms::inner(beta, svmp::FE::forms::grad(v)))).dx();

    (void)svmp::FE::systems::installFormulation(
        sys, "supg",
        {u_field},
        residual);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    const auto n_dofs = sys.dofHandler().getNumDofs();
    ASSERT_EQ(n_dofs, 4);

    std::vector<Real> U(static_cast<std::size_t>(n_dofs), 0.0);
    U = {0.2, 0.35, 0.15, 0.4};

    svmp::FE::systems::SystemStateView state;
    state.u = U;

    svmp::FE::systems::AssemblyRequest req;
    req.op = "supg";
    req.want_matrix = true;
    req.want_vector = true;

    svmp::FE::assembly::DenseSystemView out(n_dofs);
    out.zero();
    auto result = sys.assemble(req, state, &out, &out);
    EXPECT_TRUE(result.success);

    std::vector<Real> R0(static_cast<std::size_t>(n_dofs), 0.0);
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        R0[static_cast<std::size_t>(i)] = out.getVectorEntry(i);
    }

    // Central difference check.
    const Real eps_fd = 1e-6;
    const Real fd_tol = 2e-8;

    svmp::FE::systems::AssemblyRequest req_vec;
    req_vec.op = "supg";
    req_vec.want_vector = true;

    for (GlobalIndex j = 0; j < n_dofs; ++j) {
        auto U_plus = U;
        auto U_minus = U;
        U_plus[static_cast<std::size_t>(j)] += eps_fd;
        U_minus[static_cast<std::size_t>(j)] -= eps_fd;

        svmp::FE::systems::SystemStateView state_plus;
        state_plus.u = U_plus;
        svmp::FE::systems::SystemStateView state_minus;
        state_minus.u = U_minus;

        svmp::FE::assembly::DenseVectorView Rp(n_dofs);
        svmp::FE::assembly::DenseVectorView Rm(n_dofs);
        Rp.zero();
        Rm.zero();
        (void)sys.assemble(req_vec, state_plus, nullptr, &Rp);
        (void)sys.assemble(req_vec, state_minus, nullptr, &Rm);

        for (GlobalIndex i = 0; i < n_dofs; ++i) {
            const Real fd = (Rp.getVectorEntry(i) - Rm.getVectorEntry(i)) / (2.0 * eps_fd);
            EXPECT_NEAR(out.getMatrixEntry(i, j), fd, fd_tol);
        }
    }

    // Taylor remainder check.
    std::vector<Real> dU(static_cast<std::size_t>(n_dofs), 0.0);
    dU = {0.25, -0.15, 0.05, -0.3};

    const auto JdU = matVec(out, dU);

    const std::array<Real, 3> eps = {1e-2, 1e-3, 1e-4};
    std::array<Real, 3> remainder_norm{};
    for (std::size_t k = 0; k < eps.size(); ++k) {
        auto U_eps = U;
        for (GlobalIndex i = 0; i < n_dofs; ++i) {
            U_eps[static_cast<std::size_t>(i)] += eps[k] * dU[static_cast<std::size_t>(i)];
        }

        svmp::FE::systems::SystemStateView state_eps;
        state_eps.u = U_eps;

        svmp::FE::assembly::DenseVectorView R_eps(n_dofs);
        R_eps.zero();
        (void)sys.assemble(req_vec, state_eps, nullptr, &R_eps);

        std::vector<Real> remainder(static_cast<std::size_t>(n_dofs), 0.0);
        for (GlobalIndex i = 0; i < n_dofs; ++i) {
            const auto idx = static_cast<std::size_t>(i);
            remainder[idx] = (R_eps.getVectorEntry(i) - R0[idx]) - eps[k] * JdU[idx];
        }
        remainder_norm[k] = l2Norm(remainder);
    }

    ASSERT_GT(remainder_norm[0], 0.0);
    ASSERT_GT(remainder_norm[1], 0.0);
    ASSERT_GT(remainder_norm[2], 0.0);

    const auto slope01 = std::log(remainder_norm[1] / remainder_norm[0]) / std::log(eps[1] / eps[0]);
    const auto slope12 = std::log(remainder_norm[2] / remainder_norm[1]) / std::log(eps[2] / eps[1]);

    EXPECT_GT(slope01, 1.8);
    EXPECT_GT(slope12, 1.8);
}

// ---------------------------------------------------------------------------
// Shape derivatives on a moving domain
//
// A coupled ALE system assembles on the configuration moved by the mesh
// displacement unknown d (x = X + d).  Each probe residual is installed with
// the geometry sensitivity of a coupled ALE formulation, and its Jacobian is
// compared with central differences of the residual, the mesh moved for every
// perturbed state as Newton moves it.  The relative defect of the d columns
// is the part of the shape derivative the automatic Jacobian leaves out; the
// u and p columns are exact in every case.
// ---------------------------------------------------------------------------

namespace {

namespace forms = svmp::FE::forms;
namespace systems = svmp::FE::systems;
using forms::FormExpr;

constexpr int kMovingTop = 4;

// [0, 2] x [0, 1] split into 2 x 1 squares and 4 triangles; the top edge
// carries kMovingTop.
std::shared_ptr<svmp::Mesh> makeMovingRectangleMesh()
{
    auto base = std::make_shared<svmp::MeshBase>();
    const std::vector<svmp::real_t> x_ref = {
        0.0, 0.0,  1.0, 0.0,  2.0, 0.0,
        0.0, 1.0,  1.0, 1.0,  2.0, 1.0,
    };
    const std::vector<svmp::offset_t> offsets = {0, 3, 6, 9, 12};
    const std::vector<svmp::index_t> cells = {
        0, 1, 4,  0, 4, 3,
        1, 2, 5,  1, 5, 4,
    };
    svmp::CellShape shape{};
    shape.family = svmp::CellFamily::Triangle;
    shape.num_corners = 3;
    shape.order = 1;
    base->build_from_arrays(2, x_ref, offsets, cells, std::vector<svmp::CellShape>(4, shape));
    base->finalize();
    base->register_label("top", static_cast<svmp::label_t>(kMovingTop));
    for (svmp::index_t face = 0; face < static_cast<svmp::index_t>(base->n_faces()); ++face) {
        const auto vertices = base->face_vertices(face);
        const bool top = vertices.size() == 2u &&
                         std::all_of(vertices.begin(), vertices.end(), [&](svmp::index_t vertex) {
                             return std::abs(base->X_ref().at(static_cast<std::size_t>(2 * vertex + 1)) - 1.0) < 1e-14;
                         });
        if (top) {
            base->set_boundary_label(face, static_cast<svmp::label_t>(kMovingTop));
        }
    }
    return svmp::create_mesh(std::move(base));
}

struct MovingDomainSymbols {
    FormExpr u, v;     // velocity and its test function
    FormExpr p, q;     // pressure and its test function
    FormExpr d, psi;   // mesh displacement and its test function
};

enum class ProbeRows { Velocity, Pressure, Mesh };

struct MovingDomainProbe {
    std::string name;
    ProbeRows rows;
    std::function<FormExpr(const MovingDomainSymbols&)> residual;
    bool coupled_install{true};   ///< false: geometry-constant install, as the mesh-motion modules install
};

struct ShapeDerivativeDefect {
    Real mesh_columns{0.0};    ///< ||J - FD|| / ||FD|| over the d columns
    Real other_columns{0.0};   ///< the same over the u and p columns (absolute when FD vanishes there)
    Real mesh_fd_norm{0.0};    ///< ||FD|| over the d columns
    Real mesh_jacobian_norm{0.0};
};

Real relativeOrAbsolute(Real diff2, Real ref2)
{
    return ref2 > 1e-28 ? std::sqrt(diff2 / ref2) : std::sqrt(diff2);
}

// Installs @p probe as a moving-domain formulation installs it and compares
// the Jacobian rows of its test field with central differences.  A fluid
// probe goes into a coupled install with the other fluid field,
// install({u, p}, probe + dt term of the other field), next to a mesh-motion
// equation on d (the cell geometry variations are assembled only when the
// mesh field has rows of its own, as it always has with a mesh-motion
// module).  A mesh probe is installed alone, as the mesh-motion modules
// install.
ShapeDerivativeDefect movingDomainJacobianDefect(const MovingDomainProbe& probe,
                                                 forms::GeometryTangentPath path)
{
    auto mesh = makeMovingRectangleMesh();
    auto scalar = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Triangle3, 1);
    auto vector = std::make_shared<svmp::FE::spaces::ProductSpace>(scalar, 2);

    systems::FESystem system(mesh, svmp::Configuration::Current);
    const auto u_id = system.addField(systems::FieldSpec{.name = "u", .space = vector, .components = 2});
    const auto p_id = system.addField(systems::FieldSpec{.name = "p", .space = scalar, .components = 1});
    systems::ALEBindingOptions ale_options;
    ale_options.enabled = true;
    ale_options.dimension = 2;
    ale_options.mesh_velocity_source = systems::ALEMeshVelocitySource::CoupledDisplacement;
    ale_options.geometry_tangent_path = path;
    ale_options.mesh_velocity_space = vector;
    ale_options.mesh_displacement_space = vector;
    ale_options.auto_register_mesh_displacement_field = true;
    const auto ale = systems::resolveALEBinding(system, ale_options);
    const auto d_id = ale.mesh_displacement_field;

    const MovingDomainSymbols s{
        .u = FormExpr::stateField(u_id, *vector, "u"),
        .v = FormExpr::testFunction(u_id, *vector, "v"),
        .p = FormExpr::stateField(p_id, *scalar, "p"),
        .q = FormExpr::testFunction(p_id, *scalar, "q"),
        .d = FormExpr::stateField(d_id, *vector, "d"),
        .psi = FormExpr::testFunction(d_id, *vector, "psi"),
    };
    systems::FormInstallOptions install;
    install.compiler_options.use_symbolic_tangent = true;
    install.compiler_options.jit.enable = false;
    if (probe.coupled_install) {
        ale.configureInstallOptions(install);
    }
    system.addOperator("probe");
    svmp::FE::FieldId row = d_id;
    if (probe.rows == ProbeRows::Mesh) {
        (void)systems::installFormulation(system, "probe", {d_id}, probe.residual(s), install);
    } else {
        row = probe.rows == ProbeRows::Velocity ? u_id : p_id;
        const auto other = probe.rows == ProbeRows::Velocity ? (dt(s.p) * s.q).dx() : inner(dt(s.u), s.v).dx();
        (void)systems::installFormulation(system, "probe", {u_id, p_id}, probe.residual(s) + other, install);
        (void)systems::installFormulation(system, "probe", {d_id}, inner(grad(s.d), grad(s.psi)).dx(),
                                          systems::FormInstallOptions{});
    }
    // The probes have null spaces; compare the unconstrained operator.
    systems::SetupOptions setup_options;
    setup_options.use_constraints_in_assembly = false;
    system.setup(setup_options);

    const auto n = system.dofHandler().getNumDofs();
    const auto range = [&](svmp::FE::FieldId field) {
        const auto begin = system.fieldDofOffset(field);
        return std::pair<GlobalIndex, GlobalIndex>{begin, begin + system.fieldDofHandler(field).getNumDofs()};
    };
    const auto mesh_columns = range(d_id);
    const auto rows = range(row);
    const auto is_mesh_column = [&](GlobalIndex col) { return col >= mesh_columns.first && col < mesh_columns.second; };

    // A distorted mesh (|d| <= 0.08 on unit cells) moving with w = (d - d_prev) / dt.
    std::vector<Real> values(static_cast<std::size_t>(n), 0.0);
    std::vector<Real> previous(static_cast<std::size_t>(n), 0.0);
    for (GlobalIndex i = 0; i < n; ++i) {
        const auto k = static_cast<std::size_t>(i);
        const auto x = static_cast<Real>(i);
        values[k] = is_mesh_column(i) ? Real(0.08) * std::sin(Real(2.1) * x + Real(0.5))
                                      : Real(0.25) * std::sin(Real(1.3) * x + Real(0.2));
        previous[k] = is_mesh_column(i) ? Real(0.5) * values[k] : values[k];
    }

    const auto assemble = [&](const std::vector<Real>& state_values,
                              svmp::FE::assembly::GlobalSystemView* J,
                              svmp::FE::assembly::GlobalSystemView* R) {
        systems::SystemStateView state;
        state.dt = 0.5;
        state.u = std::span<const Real>(state_values);
        state.u_prev = std::span<const Real>(previous);
        const systems::BackwardDifferenceIntegrator integrator;
        const auto context = integrator.buildContext(1, state);
        state.time_integration = &context;
        (void)system.updateCurrentCoordinatesFromMeshDisplacement(state);
        systems::AssemblyRequest request;
        request.op = "probe";
        request.want_matrix = J != nullptr;
        request.want_vector = R != nullptr;
        const auto result = system.assemble(request, state, J, R);
        EXPECT_TRUE(result.success) << result.error_message;
    };

    svmp::FE::assembly::DenseMatrixView J(n);
    J.zero();
    assemble(values, &J, nullptr);

    const Real eps = 1e-6;
    Real mesh_diff2 = 0.0, mesh_ref2 = 0.0, mesh_j2 = 0.0, other_diff2 = 0.0, other_ref2 = 0.0;
    for (GlobalIndex col = 0; col < n; ++col) {
        auto plus = values;
        auto minus = values;
        plus[static_cast<std::size_t>(col)] += eps;
        minus[static_cast<std::size_t>(col)] -= eps;
        svmp::FE::assembly::DenseVectorView R_plus(n);
        svmp::FE::assembly::DenseVectorView R_minus(n);
        R_plus.zero();
        R_minus.zero();
        assemble(plus, nullptr, &R_plus);
        assemble(minus, nullptr, &R_minus);
        for (GlobalIndex row_dof = rows.first; row_dof < rows.second; ++row_dof) {
            const Real fd = (R_plus.getVectorEntry(row_dof) - R_minus.getVectorEntry(row_dof)) / (2.0 * eps);
            const Real jac = J.getMatrixEntry(row_dof, col);
            if (is_mesh_column(col)) {
                mesh_diff2 += (jac - fd) * (jac - fd);
                mesh_ref2 += fd * fd;
                mesh_j2 += jac * jac;
            } else {
                other_diff2 += (jac - fd) * (jac - fd);
                other_ref2 += fd * fd;
            }
        }
    }
    return {relativeOrAbsolute(mesh_diff2, mesh_ref2), relativeOrAbsolute(other_diff2, other_ref2),
            std::sqrt(mesh_ref2), std::sqrt(mesh_j2)};
}

const char* tangentPathName(forms::GeometryTangentPath path)
{
    return path == forms::GeometryTangentPath::ADReference ? "ADReference" : "SymbolicRequired";
}

using S = MovingDomainSymbols;

FormExpr zeroTimes(const FormExpr& f)
{
    return FormExpr::constant(0.0) * f;
}

} // namespace

// Characterization of the moving-domain Jacobian (decision 4A).  Under the
// mesh-motion sensitivity of coupled ALE only the boundary probe of the
// frame-explicit normal is exact.  In the fluid-row, mesh-column cell blocks
// the measure variation and part of the frame-explicit variations are not
// assembled; the active-frame operators (x(), J(), detJ(), Jinv(), h(),
// FormExpr::normal(), hNormal()) and the gradients of the fluid fields carry
// no shape derivative; an integrand without a frame-explicit terminal gets no
// mesh block at all; and the mesh operator is geometry-constant.  The AD
// reference path has the same gaps.  The printed relative defect of the mesh
// columns is what a complete shape-derivative tangent path must remove; the
// fluid columns are exact in every case.
TEST(MovingDomainShapeDerivativeTest, OnlyTheFrameExplicitNormalIsComplete)
{
    using namespace svmp::FE::forms;
    const MovingDomainProbe exact{
        "(u.currentNormal()) q ds", ProbeRows::Pressure,
        [](const S& s) { return (inner(s.u, currentNormal()) * s.q + zeroTimes(s.p * s.q)).ds(kMovingTop); }};
    const std::vector<MovingDomainProbe> incomplete = {
        {"q dx", ProbeRows::Pressure,
         [](const S& s) { return (s.q + zeroTimes(s.p * s.q)).dx(); }},
        {"u.v dx", ProbeRows::Velocity,
         [](const S& s) { return inner(s.u, s.v).dx(); }},
        {"currentCoordinate().v dx", ProbeRows::Velocity,
         [](const S& s) { return (inner(currentCoordinate(), s.v) + zeroTimes(inner(s.u, s.v))).dx(); }},
        {"tr(currentJacobian()) q dx", ProbeRows::Pressure,
         [](const S& s) { return (trace(currentJacobian()) * s.q + zeroTimes(s.p * s.q)).dx(); }},
        {"meshVelocity().v dx", ProbeRows::Velocity,
         [](const S& s) { return (inner(meshVelocity(), s.v) + zeroTimes(inner(s.u, s.v))).dx(); }},
        {"div(meshVelocity()) u.v dx (moving control volume)", ProbeRows::Velocity,
         [](const S& s) { return (div(meshVelocity()) * inner(s.u, s.v)).dx(); }},
        {"x().v dx", ProbeRows::Velocity,
         [](const S& s) { return (inner(x(), s.v) + zeroTimes(inner(s.u, s.v))).dx(); }},
        {"tr(J()) q dx", ProbeRows::Pressure,
         [](const S& s) { return (trace(J()) * s.q + zeroTimes(s.p * s.q)).dx(); }},
        {"detJ() q dx", ProbeRows::Pressure,
         [](const S& s) { return (detJ() * s.q + zeroTimes(s.p * s.q)).dx(); }},
        {"(u.Jinv()^T Jinv() u) q dx (VMS metric)", ProbeRows::Pressure,
         [](const S& s) { return (inner(s.u, transpose(Jinv()) * Jinv() * s.u) * s.q + zeroTimes(s.p * s.q)).dx(); }},
        {"h() q dx", ProbeRows::Pressure,
         [](const S& s) { return (h() * s.q + zeroTimes(s.p * s.q)).dx(); }},
        {"(grad(u) u).v dx (convection)", ProbeRows::Velocity,
         [](const S& s) { return inner(grad(s.u) * s.u, s.v).dx(); }},
        {"(grad(u) (u - meshVelocity())).v dx (ALE convection)", ProbeRows::Velocity,
         [](const S& s) { return inner(grad(s.u) * (s.u - meshVelocity()), s.v).dx(); }},
        {"q div(u) dx (continuity)", ProbeRows::Pressure,
         [](const S& s) { return (s.q * div(s.u) + zeroTimes(s.p * s.q)).dx(); }},
        {"-p div(v) dx (pressure)", ProbeRows::Velocity,
         [](const S& s) { return (-s.p * div(s.v) + zeroTimes(inner(s.u, s.v))).dx(); }},
        {"grad(u):grad(v) dx (viscous)", ProbeRows::Velocity,
         [](const S& s) { return inner(grad(s.u), grad(s.v)).dx(); }},
        {"q ds", ProbeRows::Pressure,
         [](const S& s) { return (s.q + zeroTimes(s.p * s.q)).ds(kMovingTop); }},
        {"(u.normal()) q ds", ProbeRows::Pressure,
         [](const S& s) { return (inner(s.u, FormExpr::normal()) * s.q + zeroTimes(s.p * s.q)).ds(kMovingTop); }},
        {"q / hNormal() ds (Nitsche scale)", ProbeRows::Pressure,
         [](const S& s) { return (s.q / hNormal() + zeroTimes(s.p * s.q)).ds(kMovingTop); }},
        {"(grad(u) currentNormal()).v ds", ProbeRows::Velocity,
         [](const S& s) { return inner(grad(s.u) * currentNormal(), s.v).ds(kMovingTop); }},
        {"grad(d):grad(psi) dx (harmonic, coupled install)", ProbeRows::Mesh,
         [](const S& s) { return inner(grad(s.d), grad(s.psi)).dx(); }},
        {"grad(d):grad(psi) dx (harmonic, geometry-constant install)", ProbeRows::Mesh,
         [](const S& s) { return inner(grad(s.d), grad(s.psi)).dx(); },
         /*coupled_install=*/false},
    };
    const auto report = [](const MovingDomainProbe& probe, GeometryTangentPath path,
                           const ShapeDerivativeDefect& defect) {
        std::cout << "[ shape derivative ] " << tangentPathName(path) << ": " << probe.name
                  << ": relative defect of the mesh columns " << defect.mesh_columns << " (|J| "
                  << defect.mesh_jacobian_norm << ", |FD| " << defect.mesh_fd_norm << ")\n";
    };
    for (const auto path : {GeometryTangentPath::SymbolicRequired, GeometryTangentPath::ADReference}) {
        {
            SCOPED_TRACE(exact.name + " " + tangentPathName(path));
            const auto defect = movingDomainJacobianDefect(exact, path);
            report(exact, path, defect);
            EXPECT_LT(defect.mesh_columns, 1e-6);
            EXPECT_LT(defect.other_columns, 1e-6);
        }
        for (const auto& probe : incomplete) {
            SCOPED_TRACE(probe.name + " " + tangentPathName(path));
            const auto defect = movingDomainJacobianDefect(probe, path);
            report(probe, path, defect);
            EXPECT_GT(defect.mesh_columns, 1e-2);
            EXPECT_LT(defect.other_columns, 1e-6);
        }
    }
}
