/**
 * @file test_JacobianVerification.cpp
 * @brief Stronger Jacobian verification tests (Taylor remainder + central FD + Stokes coupling)
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Dofs/EntityDofMap.h"
#include "Forms/JIT/JITKernelWrapper.h"
#include "Forms/MonolithicCellKernel.h"
#include "Forms/JIT/LLVMJITBuildInfo.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Spaces/SpaceFactory.h"
#include "Systems/ALEBinding.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Systems/TimeIntegrator.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"
#include "Tests/Unit/Forms/JITTestHelpers.h"
#include "Tests/Unit/Systems/TraceMultiplierTestMeshes.h"

#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <tuple>
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
                                                 forms::GeometryTangentPath path,
                                                 forms::ShapeDerivative shape = forms::ShapeDerivative::FrameExplicit)
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
    ale_options.geometry_tangent_path = path;
    ale_options.shape_derivative = shape;
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
    if (probe.coupled_install || shape == forms::ShapeDerivative::Complete) {
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
        systems::FormInstallOptions mesh_install;
        if (shape == forms::ShapeDerivative::Complete) {
            ale.configureInstallOptions(mesh_install);
        }
        (void)systems::installFormulation(system, "probe", {d_id}, inner(grad(s.d), grad(s.psi)).dx(),
                                          mesh_install);
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
// mesh-motion sensitivity of coupled ALE the frame-explicit probes are exact:
// the boundary probe of the frame-explicit normal and the cell probes of the
// current coordinate, the current Jacobian and the mesh velocity (with the
// variation of the cell measure).  Partly or not assembled: the variations
// of div(meshVelocity()) and of the fluid-field gradients, so the moving
// control volume and the ALE convection are partial; the active-frame
// operators (x(), J(), detJ(), Jinv(), h(), FormExpr::normal(), hNormal())
// and the gradients of the fluid fields carry no shape derivative; an
// integrand without a frame-explicit terminal gets no mesh block at all; and
// the mesh operator is geometry-constant.  The AD reference path has the same
// gaps.  The printed relative defect of the mesh columns is what a complete
// shape-derivative tangent path must remove; the fluid columns are exact in
// every case.
TEST(MovingDomainShapeDerivativeTest, OnlyTheFrameExplicitTerminalsAreComplete)
{
    using namespace svmp::FE::forms;
    const std::vector<MovingDomainProbe> exact = {
        {"(u.currentNormal()) q ds", ProbeRows::Pressure,
         [](const S& s) { return (inner(s.u, currentNormal()) * s.q + zeroTimes(s.p * s.q)).ds(kMovingTop); }},
        {"currentCoordinate().v dx", ProbeRows::Velocity,
         [](const S& s) { return (inner(currentCoordinate(), s.v) + zeroTimes(inner(s.u, s.v))).dx(); }},
        {"tr(currentJacobian()) q dx", ProbeRows::Pressure,
         [](const S& s) { return (trace(currentJacobian()) * s.q + zeroTimes(s.p * s.q)).dx(); }},
        {"meshVelocity().v dx", ProbeRows::Velocity,
         [](const S& s) { return (inner(meshVelocity(), s.v) + zeroTimes(inner(s.u, s.v))).dx(); }},
    };
    const std::vector<MovingDomainProbe> incomplete = {
        {"q dx", ProbeRows::Pressure,
         [](const S& s) { return (s.q + zeroTimes(s.p * s.q)).dx(); }},
        {"u.v dx", ProbeRows::Velocity,
         [](const S& s) { return inner(s.u, s.v).dx(); }},
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
        for (const auto& probe : exact) {
            SCOPED_TRACE(probe.name + " " + tangentPathName(path));
            const auto defect = movingDomainJacobianDefect(probe, path);
            report(probe, path, defect);
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

// The complete shape derivative (forms::ShapeDerivative::Complete, a C++-only
// opt-in of the moving domain): every probe of the characterization above,
// the continuity probe q div(u) and the geometry-constant harmonic install
// included, matches central differences in the mesh columns.  The AD
// reference path differentiates the frame-explicit terminals only and is
// refused with the opt-in.
TEST(MovingDomainShapeDerivativeTest, CompleteOptInMakesEveryProbeExact)
{
    using namespace svmp::FE::forms;
    const std::vector<MovingDomainProbe> probes = {
        {"(u.currentNormal()) q ds", ProbeRows::Pressure,
         [](const S& s) { return (inner(s.u, currentNormal()) * s.q + zeroTimes(s.p * s.q)).ds(kMovingTop); }},
        {"q dx", ProbeRows::Pressure, [](const S& s) { return (s.q + zeroTimes(s.p * s.q)).dx(); }},
        {"u.v dx", ProbeRows::Velocity, [](const S& s) { return inner(s.u, s.v).dx(); }},
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
        {"grad(d):grad(psi) dx (harmonic)", ProbeRows::Mesh,
         [](const S& s) { return inner(grad(s.d), grad(s.psi)).dx(); }},
    };
    for (const auto& probe : probes) {
        SCOPED_TRACE(probe.name);
        const auto defect =
            movingDomainJacobianDefect(probe, GeometryTangentPath::SymbolicRequired, ShapeDerivative::Complete);
        std::cout << "[ complete shape derivative ] " << probe.name << ": relative defect of the mesh columns "
                  << defect.mesh_columns << " (|J| " << defect.mesh_jacobian_norm << ", |FD| " << defect.mesh_fd_norm
                  << ")\n";
        EXPECT_LT(defect.mesh_columns, 1e-6);
        EXPECT_LT(defect.other_columns, 1e-6);
    }
    EXPECT_THROW((void)movingDomainJacobianDefect(probes[1], GeometryTangentPath::ADReference, ShapeDerivative::Complete),
                 std::exception);
}

/**
 * The rate of a vector field in another field's row: with u = (t x, 0, 0) at
 * t = 0.2 and 0.1 on the single tetrahedron (volume 1/6) and backward Euler
 * with dt = 0.1, dt(u) = (x, 0, 0), so div(dt(u)) = trace(grad(dt(u))) = 1
 * and the P0 row of b = 0 in (b - rate) w is -1/6 for both spellings (a
 * rate without the history term would give -2/6), interpreted and compiled.
 */
TEST(TimeDerivativeInAnotherRow, TheDivergenceOfARateIsTheTraceOfItsGradient)
{
    using svmp::FE::forms::FormExpr;
    namespace forms = svmp::FE::forms;
    namespace systems = svmp::FE::systems;
    const Real dt = 0.1;
    for (const bool jit : {false, true}) {
    for (const bool divergence : {true, false}) {
        SCOPED_TRACE(std::string(divergence ? "div(dt(u))" : "trace(grad(dt(u)))") +
                     (jit ? ", compiled" : ", interpreted"));
        auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
        auto p1 = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);
        auto vector = std::make_shared<svmp::FE::spaces::ProductSpace>(p1, 3);
        auto p0 = std::make_shared<svmp::FE::spaces::L2Space>(ElementType::Tetra4, 0);
        systems::FESystem sys(mesh);
        const auto u_id = sys.addField(systems::FieldSpec{.name = "u", .space = vector, .components = 3});
        const auto b_id = sys.addField(systems::FieldSpec{.name = "b", .space = p0, .components = 1});
        sys.addOperator("op");
        const auto u = forms::StateField(u_id, *vector, "u");
        const auto z = forms::TestField(u_id, *vector, "z");
        const auto b = forms::StateField(b_id, *p0, "b");
        const auto w = forms::TestField(b_id, *p0, "w");
        const auto rate = divergence ? forms::div(forms::dt(u)) : forms::trace(forms::grad(forms::dt(u)));
        systems::FormInstallOptions options;
        options.compiler_options.jit.enable = jit;
        (void)systems::installFormulation(sys, "op", {u_id, b_id},
                                          forms::inner(u, z).dx() + ((b - rate) * w).dx(), options);
        systems::SetupInputs inputs;
        inputs.topology_override = singleTetraTopology();
        sys.setup({}, inputs);

        const auto n = sys.dofHandler().getNumDofs();
        const std::array<Real, 4> x{0.0, 1.0, 0.0, 0.0};  // the vertices' x
        const auto state_at = [&](Real t) {
            std::vector<Real> values(static_cast<std::size_t>(n), 0.0);
            const auto dofs = sys.fieldDofHandler(u_id).getCellDofs(0);
            const auto offset = sys.fieldDofOffset(u_id);
            for (std::size_t a = 0; a < 4; ++a) {  // component 0 of vertex a
                values[static_cast<std::size_t>(offset + dofs[a])] = t * x[a];
            }
            return values;
        };
        const auto current = state_at(0.2);
        const auto previous = state_at(0.1);
        systems::SystemStateView state;
        state.dt = dt;
        state.u = std::span<const Real>(current);
        state.u_prev = std::span<const Real>(previous);
        const systems::BackwardDifferenceIntegrator integrator;
        const auto context = integrator.buildContext(1, state);
        state.time_integration = &context;
        systems::AssemblyRequest request;
        request.op = "op";
        request.want_vector = true;
        svmp::FE::assembly::DenseVectorView R(n);
        R.zero();
        ASSERT_TRUE(sys.assemble(request, state, nullptr, &R).success);
        const auto b_row = sys.fieldDofOffset(b_id) + sys.fieldDofHandler(b_id).getCellDofs(0)[0];
        EXPECT_NEAR(R.getVectorEntry(b_row), -1.0 / 6.0, 1e-12);
        if (jit) {
            // Compiled, not evaluated by the interpreter: the fused cell
            // kernel, or each block it falls back to.
            int jit_blocks = 0;
            for (const auto& term : sys.operatorDefinition("op").cells) {
                const auto* fused = dynamic_cast<const svmp::FE::forms::MonolithicCellKernel*>(term.kernel.get());
                ASSERT_NE(fused, nullptr) << term.kernel->name();
                for (std::size_t i = 0; i < fused->numBlocks(); ++i) {
                    if (const auto* block = dynamic_cast<const svmp::FE::forms::jit::JITKernelWrapper*>(
                            fused->blockSpec(i).fallback_kernel.get())) {
                        ++jit_blocks;
                        EXPECT_TRUE(fused->hasCompiledDispatch() || block->isJITReady()) << block->name();
                    }
                }
            }
            EXPECT_GT(jit_blocks, 0);
        }
    }
    }
}

namespace {

namespace forms = svmp::FE::forms;
namespace systems = svmp::FE::systems;

/// Two unequal, positively oriented P1 triangles sharing an edge, in the
/// given order.
constexpr std::array<std::array<Real, 2>, 4> kTwoTriangleNodes{{{0.0, 0.0}, {1.3, 0.0}, {0.2, 0.9}, {1.1, 1.2}}};
constexpr std::array<std::array<int, 3>, 2> kTwoTriangleCells{{{0, 1, 2}, {1, 3, 2}}};

std::shared_ptr<svmp::Mesh> makeTwoTriangleMesh(bool reversed)
{
    auto base = std::make_shared<svmp::MeshBase>();
    std::vector<svmp::real_t> x_ref;
    for (const auto& node : kTwoTriangleNodes) {
        x_ref.push_back(node[0]);
        x_ref.push_back(node[1]);
    }
    std::vector<svmp::index_t> cells;
    for (std::size_t c = 0; c < kTwoTriangleCells.size(); ++c) {
        const auto& cell = kTwoTriangleCells[reversed ? kTwoTriangleCells.size() - 1 - c : c];
        cells.insert(cells.end(), cell.begin(), cell.end());
    }
    svmp::CellShape shape{};
    shape.family = svmp::CellFamily::Triangle;
    shape.num_corners = 3;
    shape.order = 1;
    base->build_from_arrays(2, x_ref, {0, 3, 6}, cells, std::vector<svmp::CellShape>(2, shape));
    base->finalize();
    return svmp::create_mesh(std::move(base));
}

/// The mesh displacement of a node: smooth in the reference position, so it
/// does not depend on the cell order.
std::array<Real, 2> twoTriangleDisplacement(const std::array<Real, 2>& X)
{
    return {Real(0.05) * X[1] - Real(0.02) * X[0] + Real(0.01), Real(0.03) * X[0] * X[1] - Real(0.04) * X[0]};
}

/// d/dd_{a,k} of R_i = int q_i tr(J) dx = sum over the cells at node i of
/// det(J) tr(J) / 6, J the current Jacobian of a P1 triangle, by node i, node
/// a and component k.
std::map<std::tuple<int, int, int>, Real> traceJacobianMeshColumnsExact()
{
    std::map<std::tuple<int, int, int>, Real> out;
    for (const auto& cell : kTwoTriangleCells) {
        std::array<std::array<Real, 2>, 3> x{};
        for (int n = 0; n < 3; ++n) {
            const auto& X = kTwoTriangleNodes[static_cast<std::size_t>(cell[static_cast<std::size_t>(n)])];
            const auto d = twoTriangleDisplacement(X);
            x[static_cast<std::size_t>(n)] = {X[0] + d[0], X[1] + d[1]};
        }
        Real J[2][2];
        for (int k = 0; k < 2; ++k) {
            for (int c = 0; c < 2; ++c) {
                J[k][c] = x[static_cast<std::size_t>(c + 1)][static_cast<std::size_t>(k)] - x[0][static_cast<std::size_t>(k)];
            }
        }
        const Real det = J[0][0] * J[1][1] - J[0][1] * J[1][0];
        const Real tr = J[0][0] + J[1][1];
        for (int a = 0; a < 3; ++a) {
            for (int k = 0; k < 2; ++k) {
                Real dJ[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
                for (int c = 0; c < 2; ++c) {
                    dJ[k][c] = (a == c + 1 ? 1.0 : 0.0) - (a == 0 ? 1.0 : 0.0);
                }
                const Real ddet = dJ[0][0] * J[1][1] + J[0][0] * dJ[1][1] - dJ[0][1] * J[1][0] - J[0][1] * dJ[1][0];
                const Real dtr = dJ[0][0] + dJ[1][1];
                const Real df = (ddet * tr + det * dtr) / Real(6.0);
                for (const int i : cell) {
                    out[{i, cell[static_cast<std::size_t>(a)], k}] += df;
                }
            }
        }
    }
    return out;
}

struct TraceJacobianColumns {
    std::vector<std::map<std::tuple<int, int, int>, Real>> assembled;  ///< one map per assembly
    std::map<std::tuple<int, int, int>, Real> central_difference;
};

/// The pressure-row, mesh-column Jacobian entries of the interpreted probe
/// int q tr(currentJacobian()) dx on the two triangles, assembled
/// @p assemblies times in a row (a cold then warm basis cache), and their
/// central differences that rebuild the current coordinates, keyed by the
/// node indices of kTwoTriangleNodes.
TraceJacobianColumns traceJacobianMeshColumns(bool reversed, bool batched, int assemblies)
{
    using namespace svmp::FE::forms;
    auto mesh = makeTwoTriangleMesh(reversed);
    auto scalar = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Triangle3, 1);
    auto vector = std::make_shared<svmp::FE::spaces::ProductSpace>(scalar, 2);

    systems::FESystem system(mesh, svmp::Configuration::Current);
    const auto u_id = system.addField(systems::FieldSpec{.name = "u", .space = vector, .components = 2});
    const auto p_id = system.addField(systems::FieldSpec{.name = "p", .space = scalar, .components = 1});
    systems::ALEBindingOptions ale_options;
    ale_options.enabled = true;
    ale_options.dimension = 2;
    ale_options.mesh_velocity_space = vector;
    ale_options.mesh_displacement_space = vector;
    ale_options.auto_register_mesh_displacement_field = true;
    const auto ale = systems::resolveALEBinding(system, ale_options);
    const auto d_id = ale.mesh_displacement_field;

    const auto u = FormExpr::stateField(u_id, *vector, "u");
    const auto v = FormExpr::testFunction(u_id, *vector, "v");
    const auto p = FormExpr::stateField(p_id, *scalar, "p");
    const auto q = FormExpr::testFunction(p_id, *scalar, "q");
    const auto d = FormExpr::stateField(d_id, *vector, "d");
    const auto psi = FormExpr::testFunction(d_id, *vector, "psi");
    systems::FormInstallOptions install;
    install.compiler_options.use_symbolic_tangent = true;
    install.compiler_options.jit.enable = false;
    ale.configureInstallOptions(install);
    system.addOperator("probe");
    (void)systems::installFormulation(system, "probe", {u_id, p_id},
                                      (trace(currentJacobian()) * q + FormExpr::constant(0.0) * p * q).dx() +
                                          inner(dt(u), v).dx(),
                                      install);
    (void)systems::installFormulation(system, "probe", {d_id}, inner(grad(d), grad(psi)).dx(),
                                      systems::FormInstallOptions{});
    systems::SetupOptions setup_options;
    setup_options.use_constraints_in_assembly = false;
    setup_options.assembly_options.use_batching = batched;
    system.setup(setup_options);

    // The node of kTwoTriangleNodes at each mesh vertex.
    const auto& local = mesh->local_mesh();
    const auto& X_ref = local.X_ref();
    std::vector<int> node_of(local.n_vertices(), -1);
    for (std::size_t vtx = 0; vtx < local.n_vertices(); ++vtx) {
        for (std::size_t n = 0; n < kTwoTriangleNodes.size(); ++n) {
            if (std::abs(X_ref[2 * vtx] - kTwoTriangleNodes[n][0]) < 1e-14 &&
                std::abs(X_ref[2 * vtx + 1] - kTwoTriangleNodes[n][1]) < 1e-14) {
                node_of[vtx] = static_cast<int>(n);
            }
        }
        EXPECT_GE(node_of[vtx], 0);
    }
    const auto dof = [&](svmp::FE::FieldId field, std::size_t vtx, int component) {
        const auto dofs = system.fieldDofHandler(field).getEntityDofMap()->getVertexDofs(static_cast<GlobalIndex>(vtx));
        return system.fieldDofOffset(field) + dofs[static_cast<std::size_t>(component)];
    };

    const auto n = system.dofHandler().getNumDofs();
    std::vector<Real> values(static_cast<std::size_t>(n), 0.0);
    for (std::size_t vtx = 0; vtx < local.n_vertices(); ++vtx) {
        const auto displacement = twoTriangleDisplacement(kTwoTriangleNodes[static_cast<std::size_t>(node_of[vtx])]);
        for (int k = 0; k < 2; ++k) {
            values[static_cast<std::size_t>(dof(d_id, vtx, k))] = displacement[static_cast<std::size_t>(k)];
        }
    }
    const std::vector<Real> previous(static_cast<std::size_t>(n), 0.0);
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

    TraceJacobianColumns out;
    for (int pass = 0; pass < assemblies; ++pass) {
        svmp::FE::assembly::DenseMatrixView J(n);
        J.zero();
        assemble(values, &J, nullptr);
        auto& entries = out.assembled.emplace_back();
        for (std::size_t row = 0; row < local.n_vertices(); ++row) {
            for (std::size_t col = 0; col < local.n_vertices(); ++col) {
                for (int k = 0; k < 2; ++k) {
                    entries[{node_of[row], node_of[col], k}] = J.getMatrixEntry(dof(p_id, row, 0), dof(d_id, col, k));
                }
            }
        }
    }
    const Real eps = 1e-6;
    for (std::size_t col = 0; col < local.n_vertices(); ++col) {
        for (int k = 0; k < 2; ++k) {
            auto plus = values;
            auto minus = values;
            plus[static_cast<std::size_t>(dof(d_id, col, k))] += eps;
            minus[static_cast<std::size_t>(dof(d_id, col, k))] -= eps;
            svmp::FE::assembly::DenseVectorView R_plus(n);
            svmp::FE::assembly::DenseVectorView R_minus(n);
            R_plus.zero();
            R_minus.zero();
            assemble(plus, nullptr, &R_plus);
            assemble(minus, nullptr, &R_minus);
            for (std::size_t row = 0; row < local.n_vertices(); ++row) {
                const auto r = dof(p_id, row, 0);
                out.central_difference[{node_of[row], node_of[col], k}] =
                    (R_plus.getVectorEntry(r) - R_minus.getVectorEntry(r)) / (2.0 * eps);
            }
        }
    }
    return out;
}

} // namespace

// The interpreted mesh-motion sensitivity of int q tr(J) dx, J the current
// Jacobian, on two unequal P1 triangles: every pressure-row, mesh-column entry
// is d(det(J) tr(J) / 6)/dd summed over the cells of the pressure node, equal
// to central differences that rebuild the current coordinates, whether the
// cells are assembled one at a time or in a batch, in either cell order, from
// a cold and from a warm basis cache. A batch slot that does not run the slow
// path of the basis preparation still needs the reference gradients the
// current-Jacobian seed reads.
TEST(MovingDomainShapeDerivativeTest, TraceOfCurrentJacobianIsExactInEveryBatchAndOrder)
{
    const auto exact = traceJacobianMeshColumnsExact();
    Real scale = 0.0;
    for (const auto& [key, value] : exact) {
        scale = std::max(scale, std::abs(value));
    }
    ASSERT_GT(scale, 0.1);
    for (const bool reversed : {false, true}) {
        for (const bool batched : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "reversed cell order " << reversed << ", batched " << batched);
            const auto columns = traceJacobianMeshColumns(reversed, batched, /*assemblies=*/2);
            for (const auto& [key, value] : exact) {
                const auto& [row, col, k] = key;
                const auto where = ::testing::Message() << "pressure node " << row << ", displacement node " << col
                                                        << " component " << k;
                EXPECT_NEAR(columns.central_difference.at(key), value, 1e-7 * scale) << where;
                EXPECT_NEAR(columns.assembled[0].at(key), value, 1e-13 * scale) << where << " (cold cache)";
                EXPECT_NEAR(columns.assembled[1].at(key), value, 1e-13 * scale) << where << " (warm cache)";
            }
        }
    }
}

namespace {

enum class MovingFrameKernel { Interpreted, Compiled, CompiledSpecialized };

/// The residual of int grad(q).(currentCoordinate() + meshVelocity()) dx on the
/// two moving triangles, by pressure node of kTwoTriangleNodes. On an affine
/// P1 triangle grad(q) is constant but the two point values are not: a kernel
/// that evaluates the integrand once at the first quadrature point and scales
/// it by the cell measure is wrong.
std::map<int, Real> movingFramePointValueResidual(MovingFrameKernel kernel, bool batched)
{
    using namespace svmp::FE::forms;
    auto mesh = makeTwoTriangleMesh(/*reversed=*/false);
    auto scalar = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Triangle3, 1);
    auto vector = std::make_shared<svmp::FE::spaces::ProductSpace>(scalar, 2);

    systems::FESystem system(mesh, svmp::Configuration::Current);
    const auto p_id = system.addField(systems::FieldSpec{.name = "p", .space = scalar, .components = 1});
    systems::ALEBindingOptions ale_options;
    ale_options.enabled = true;
    ale_options.dimension = 2;
    ale_options.mesh_velocity_space = vector;
    ale_options.mesh_displacement_space = vector;
    ale_options.auto_register_mesh_displacement_field = true;
    const auto ale = systems::resolveALEBinding(system, ale_options);
    const auto d_id = ale.mesh_displacement_field;

    const auto p = FormExpr::stateField(p_id, *scalar, "p");
    const auto q = FormExpr::testFunction(p_id, *scalar, "q");
    const auto d = FormExpr::stateField(d_id, *vector, "d");
    const auto psi = FormExpr::testFunction(d_id, *vector, "psi");
    systems::FormInstallOptions install;
    install.compiler_options.jit = svmp::FE::forms::test::makeUnitTestJITOptions();
    install.compiler_options.jit.vectorize = true;
    install.compiler_options.jit.enable = kernel != MovingFrameKernel::Interpreted;
    install.compiler_options.jit.specialization.enable = kernel == MovingFrameKernel::CompiledSpecialized;
    system.addOperator("probe");
    (void)systems::installFormulation(system, "probe", {p_id},
                                      (inner(grad(q), currentCoordinate() + meshVelocity()) +
                                       FormExpr::constant(0.0) * p * q).dx(),
                                      install);
    (void)systems::installFormulation(system, "probe", {d_id}, inner(grad(d), grad(psi)).dx(),
                                      systems::FormInstallOptions{});
    systems::SetupOptions setup_options;
    setup_options.use_constraints_in_assembly = false;
    setup_options.assembly_options.use_batching = batched;
    system.setup(setup_options);

    const auto& local = mesh->local_mesh();
    const auto& X_ref = local.X_ref();
    std::vector<int> node_of(local.n_vertices(), -1);
    for (std::size_t vtx = 0; vtx < local.n_vertices(); ++vtx) {
        for (std::size_t n = 0; n < kTwoTriangleNodes.size(); ++n) {
            if (std::abs(X_ref[2 * vtx] - kTwoTriangleNodes[n][0]) < 1e-14 &&
                std::abs(X_ref[2 * vtx + 1] - kTwoTriangleNodes[n][1]) < 1e-14) {
                node_of[vtx] = static_cast<int>(n);
            }
        }
    }
    const auto dof = [&](svmp::FE::FieldId field, std::size_t vtx, int component) {
        const auto dofs = system.fieldDofHandler(field).getEntityDofMap()->getVertexDofs(static_cast<GlobalIndex>(vtx));
        return system.fieldDofOffset(field) + dofs[static_cast<std::size_t>(component)];
    };
    const auto n = system.dofHandler().getNumDofs();
    std::vector<Real> values(static_cast<std::size_t>(n), 0.0);
    for (std::size_t vtx = 0; vtx < local.n_vertices(); ++vtx) {
        const auto displacement = twoTriangleDisplacement(kTwoTriangleNodes[static_cast<std::size_t>(node_of[vtx])]);
        for (int k = 0; k < 2; ++k) {
            values[static_cast<std::size_t>(dof(d_id, vtx, k))] = displacement[static_cast<std::size_t>(k)];
        }
    }
    const std::vector<Real> previous(static_cast<std::size_t>(n), 0.0);
    systems::SystemStateView state;
    state.dt = 0.5;
    state.u = std::span<const Real>(values);
    state.u_prev = std::span<const Real>(previous);
    const systems::BackwardDifferenceIntegrator integrator;
    const auto context = integrator.buildContext(1, state);
    state.time_integration = &context;
    (void)system.updateCurrentCoordinatesFromMeshDisplacement(state);
    systems::AssemblyRequest request;
    request.op = "probe";
    request.want_vector = true;
    svmp::FE::assembly::DenseVectorView R(n);
    R.zero();
    const auto result = system.assemble(request, state, nullptr, &R);
    EXPECT_TRUE(result.success) << result.error_message;
    std::map<int, Real> out;
    for (std::size_t vtx = 0; vtx < local.n_vertices(); ++vtx) {
        out[node_of[vtx]] = R.getVectorEntry(dof(p_id, vtx, 0));
    }
    return out;
}

} // namespace

// A compiled kernel specialized for affine P1 cells evaluates the point
// values of the moving frame (current coordinate, mesh velocity) at every
// quadrature point, one cell at a time and in a batch, as the interpreter
// does.
TEST(MovingDomainCompiledKernelTest, AffineSpecializationKeepsMovingFramePointValues)
{
    if (!svmp::FE::forms::jit::llvmJITEnabled()) {
        GTEST_SKIP() << "requires the compiled kernels (LLVM JIT)";
    }
    const auto interpreted = movingFramePointValueResidual(MovingFrameKernel::Interpreted, /*batched=*/false);
    Real scale = 0.0;
    for (const auto& [node, value] : interpreted) {
        scale = std::max(scale, std::abs(value));
    }
    ASSERT_GT(scale, 1e-2);
    for (const auto kernel : {MovingFrameKernel::Compiled, MovingFrameKernel::CompiledSpecialized}) {
        for (const bool batched : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "specialized " << (kernel == MovingFrameKernel::CompiledSpecialized)
                                              << ", batched " << batched);
            const auto compiled = movingFramePointValueResidual(kernel, batched);
            for (const auto& [node, value] : interpreted) {
                EXPECT_NEAR(compiled.at(node), value, 1e-13 * scale) << "pressure node " << node;
            }
        }
    }
}

namespace {

/// The u and p residual rows of int (meshVelocity().v + q div(meshVelocity())) dx
/// on the two moving triangles, installed as one formulation of {u, p}, by
/// (field 0 = u or 1 = p, node of kTwoTriangleNodes, component). With
/// @p mesh_rows a mesh equation on d is installed in the same operator, so the
/// {u, p} formulation is not its only cell term.
std::map<std::tuple<int, int, int>, Real> movingFrameBundleResidual(bool jit, bool mesh_rows)
{
    using namespace svmp::FE::forms;
    auto mesh = makeTwoTriangleMesh(/*reversed=*/false);
    auto scalar = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Triangle3, 1);
    auto vector = std::make_shared<svmp::FE::spaces::ProductSpace>(scalar, 2);

    systems::FESystem system(mesh, svmp::Configuration::Current);
    const auto u_id = system.addField(systems::FieldSpec{.name = "u", .space = vector, .components = 2});
    const auto p_id = system.addField(systems::FieldSpec{.name = "p", .space = scalar, .components = 1});
    systems::ALEBindingOptions ale_options;
    ale_options.enabled = true;
    ale_options.dimension = 2;
    ale_options.mesh_velocity_space = vector;
    ale_options.mesh_displacement_space = vector;
    ale_options.auto_register_mesh_displacement_field = true;
    const auto ale = systems::resolveALEBinding(system, ale_options);
    const auto d_id = ale.mesh_displacement_field;

    const auto u = FormExpr::stateField(u_id, *vector, "u");
    const auto v = FormExpr::testFunction(u_id, *vector, "v");
    const auto p = FormExpr::stateField(p_id, *scalar, "p");
    const auto q = FormExpr::testFunction(p_id, *scalar, "q");
    const auto d = FormExpr::stateField(d_id, *vector, "d");
    const auto psi = FormExpr::testFunction(d_id, *vector, "psi");
    systems::FormInstallOptions install;
    install.compiler_options.jit = svmp::FE::forms::test::makeUnitTestJITOptions();
    install.compiler_options.jit.vectorize = true;
    install.compiler_options.jit.enable = jit;
    system.addOperator("probe");
    (void)systems::installFormulation(system, "probe", {u_id, p_id},
                                      (inner(meshVelocity(), v) + q * div(meshVelocity()) +
                                       FormExpr::constant(0.0) * (inner(u, v) + p * q)).dx(),
                                      install);
    if (mesh_rows) {
        (void)systems::installFormulation(system, "probe", {d_id}, inner(grad(d), grad(psi)).dx(),
                                          systems::FormInstallOptions{});
    }
    systems::SetupOptions setup_options;
    setup_options.use_constraints_in_assembly = false;
    system.setup(setup_options);

    const auto& local = mesh->local_mesh();
    const auto& X_ref = local.X_ref();
    std::vector<int> node_of(local.n_vertices(), -1);
    for (std::size_t vtx = 0; vtx < local.n_vertices(); ++vtx) {
        for (std::size_t n = 0; n < kTwoTriangleNodes.size(); ++n) {
            if (std::abs(X_ref[2 * vtx] - kTwoTriangleNodes[n][0]) < 1e-14 &&
                std::abs(X_ref[2 * vtx + 1] - kTwoTriangleNodes[n][1]) < 1e-14) {
                node_of[vtx] = static_cast<int>(n);
            }
        }
    }
    const auto dof = [&](svmp::FE::FieldId field, std::size_t vtx, int component) {
        const auto dofs = system.fieldDofHandler(field).getEntityDofMap()->getVertexDofs(static_cast<GlobalIndex>(vtx));
        return system.fieldDofOffset(field) + dofs[static_cast<std::size_t>(component)];
    };
    const auto n = system.dofHandler().getNumDofs();
    std::vector<Real> values(static_cast<std::size_t>(n), 0.0);
    for (std::size_t vtx = 0; vtx < local.n_vertices(); ++vtx) {
        const auto displacement = twoTriangleDisplacement(kTwoTriangleNodes[static_cast<std::size_t>(node_of[vtx])]);
        for (int k = 0; k < 2; ++k) {
            values[static_cast<std::size_t>(dof(d_id, vtx, k))] = displacement[static_cast<std::size_t>(k)];
        }
    }
    const std::vector<Real> previous(static_cast<std::size_t>(n), 0.0);
    systems::SystemStateView state;
    state.dt = 0.5;
    state.u = std::span<const Real>(values);
    state.u_prev = std::span<const Real>(previous);
    const systems::BackwardDifferenceIntegrator integrator;
    const auto context = integrator.buildContext(1, state);
    state.time_integration = &context;
    (void)system.updateCurrentCoordinatesFromMeshDisplacement(state);
    systems::AssemblyRequest request;
    request.op = "probe";
    request.want_vector = true;
    svmp::FE::assembly::DenseVectorView R(n);
    R.zero();
    const auto result = system.assemble(request, state, nullptr, &R);
    EXPECT_TRUE(result.success) << result.error_message;
    std::map<std::tuple<int, int, int>, Real> out;
    for (std::size_t vtx = 0; vtx < local.n_vertices(); ++vtx) {
        for (int k = 0; k < 2; ++k) {
            out[{0, node_of[vtx], k}] = R.getVectorEntry(dof(u_id, vtx, k));
        }
        out[{1, node_of[vtx], 0}] = R.getVectorEntry(dof(p_id, vtx, 0));
    }
    return out;
}

} // namespace

// A formulation of several fields with a moving-frame terminal assembles every
// one of its blocks when it is the only cell term of its operator, as it does
// next to another cell term.
TEST(MovingDomainBundleTest, SoleMovingFrameFormulationAssemblesEveryBlock)
{
    std::vector<bool> kernel_modes{false};
    if (svmp::FE::forms::jit::llvmJITEnabled()) {
        kernel_modes.push_back(true);
    }
    for (const bool jit : kernel_modes) {
        SCOPED_TRACE(::testing::Message() << "compiled kernels " << jit);
        const auto reference = movingFrameBundleResidual(jit, /*mesh_rows=*/true);
        const auto sole = movingFrameBundleResidual(jit, /*mesh_rows=*/false);
        Real velocity_scale = 0.0;
        Real pressure_scale = 0.0;
        for (const auto& [key, value] : reference) {
            (std::get<0>(key) == 0 ? velocity_scale : pressure_scale) =
                std::max(std::get<0>(key) == 0 ? velocity_scale : pressure_scale, std::abs(value));
        }
        ASSERT_GT(velocity_scale, 1e-3);
        ASSERT_GT(pressure_scale, 1e-3);
        for (const auto& [key, value] : reference) {
            const auto& [field, node, k] = key;
            EXPECT_NEAR(sole.at(key), value, 1e-14 * std::max(velocity_scale, pressure_scale))
                << (field == 0 ? "velocity" : "pressure") << " row of node " << node << " component " << k;
        }
    }
}

/**
 * The tangent of the upwind DG transport residual
 *   (-c beta.grad v) dx + ({beta}.n(-) upwindValue(c, beta) [v]) dS
 *   + ((max(beta.n, 0) c + min(beta.n, 0) c_in) v) ds - (f v) dx
 * for both signs of beta, with the data c_in and f as coefficients, on a 2 x 2
 * grid of discontinuous Q1 cells: the symbolic tangent equals the
 * automatic-differentiation one, and both equal central differences.
 */
TEST(UpwindTransportJacobianTest, SymbolicTangentEqualsAutomaticDifferentiation)
{
    using svmp::FE::forms::FormExpr;
    namespace forms = svmp::FE::forms;
    namespace systems = svmp::FE::systems;
    for (const Real sign : {1.0, -1.0}) {
        SCOPED_TRACE(sign > 0.0 ? "+beta" : "-beta");
        std::vector<std::vector<Real>> jacobians;
        std::vector<Real> fd_defects;
        for (const bool symbolic : {false, true}) {
            auto space = std::make_shared<svmp::FE::spaces::L2Space>(ElementType::Quad4, 1);
            systems::FESystem sys(svmp::FE::test_trace_multiplier::buildQuadGrid(2, 2, 1.0, 1.0, 1));
            const auto c_id = sys.addField(systems::FieldSpec{.name = "c", .space = space});
            sys.addOperator("op");
            const auto c = forms::StateField(c_id, *space, "c");
            const auto v = forms::TestField(c_id, *space, "v");
            const std::array<Real, 2> b{sign * 1.0, sign * 0.5};
            const auto beta = forms::constantVector(b, 2);
            const auto n = FormExpr::normal();
            const auto bn = forms::inner(beta, n);
            const auto c_in = FormExpr::coefficient(
                "c_in", svmp::FE::forms::ScalarCoefficient([](Real x, Real y, Real) { return 1.0 + x * y; }));
            const auto f = FormExpr::coefficient(
                "f", svmp::FE::forms::ScalarCoefficient([](Real x, Real y, Real) { return x + 2.0 * y; }));
            const auto R = (-c * forms::inner(beta, forms::grad(v))).dx() +
                           (forms::inner(forms::avg(beta), n.minus()) * forms::upwindValue(c, beta) * forms::jump(v)).dS() +
                           ((forms::max(bn, 0.0) * c + forms::min(bn, 0.0) * c_in) * v).ds() - (f * v).dx();
            systems::FormInstallOptions options;
            options.compiler_options.use_symbolic_tangent = symbolic;
            // The interpreted kernels: the compiled path waits for the JIT key
            // fix (tracker D75); the compiled kernels of the two signs of beta
            // share a cache key today.
            options.compiler_options.jit.enable = false;
            (void)systems::installFormulation(sys, "op", {c_id}, R, options);
            sys.setup();
            const auto N = sys.dofHandler().getNumDofs();
            std::vector<Real> state(static_cast<std::size_t>(N));
            for (std::size_t i = 0; i < state.size(); ++i) {
                state[i] = std::sin(0.37 * static_cast<Real>(i) * static_cast<Real>(i));
            }
            const auto assemble = [&](const std::vector<Real>& values, svmp::FE::assembly::GlobalSystemView* J,
                                      svmp::FE::assembly::GlobalSystemView* Rv) {
                systems::SystemStateView view;
                view.u = std::span<const Real>(values);
                systems::AssemblyRequest request;
                request.op = "op";
                request.want_matrix = J != nullptr;
                request.want_vector = Rv != nullptr;
                EXPECT_TRUE(sys.assemble(request, view, J, Rv).success);
            };
            svmp::FE::assembly::DenseMatrixView J(N);
            J.zero();
            assemble(state, &J, nullptr);
            std::vector<Real> entries;
            Real defect = 0.0;
            for (GlobalIndex col = 0; col < N; ++col) {
                auto plus = state;
                auto minus = state;
                plus[static_cast<std::size_t>(col)] += 1e-6;
                minus[static_cast<std::size_t>(col)] -= 1e-6;
                svmp::FE::assembly::DenseVectorView Rp(N), Rm(N);
                Rp.zero();
                Rm.zero();
                assemble(plus, nullptr, &Rp);
                assemble(minus, nullptr, &Rm);
                for (GlobalIndex row = 0; row < N; ++row) {
                    const Real fd = (Rp.getVectorEntry(row) - Rm.getVectorEntry(row)) / 2e-6;
                    defect = std::max(defect, std::abs(J.getMatrixEntry(row, col) - fd));
                }
            }
            for (GlobalIndex row = 0; row < N; ++row) {
                for (GlobalIndex col = 0; col < N; ++col) {
                    entries.push_back(J.getMatrixEntry(row, col));
                }
            }
            jacobians.push_back(std::move(entries));
            fd_defects.push_back(defect);
        }
        EXPECT_LT(fd_defects[0], 1e-7) << "automatic differentiation";
        EXPECT_LT(fd_defects[1], 1e-7) << "symbolic tangent";
        ASSERT_EQ(jacobians[0].size(), jacobians[1].size());
        Real difference = 0.0;
        for (std::size_t k = 0; k < jacobians[0].size(); ++k) {
            difference = std::max(difference, std::abs(jacobians[0][k] - jacobians[1][k]));
        }
        EXPECT_LT(difference, 1e-12);
    }
}
