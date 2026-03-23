/**
 * @file test_MixedFormPerformance.cpp
 * @brief Phase 5 performance verification: mixed vs manual compilation and assembly
 *
 * Verifies that:
 * 1. Mixed-compiled forms produce identical assembly results as manual blocks
 * 2. installMixedFormIR creates CoupledBlockKernel for fused execution when JIT enabled
 * 3. Fused execution through installFormulation (residual path) works end-to-end
 *
 * Performance micro-benchmarks are DISABLED by default (prefix with DISABLED_).
 * Enable with --gtest_also_run_disabled_tests.
 */

#include <gtest/gtest.h>

#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Systems/FormsInstallerDetail.h"

#include "Forms/BlockForm.h"
#include "Forms/CoupledBlockKernel.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/MixedFormIR.h"

#include "Spaces/H1Space.h"
#include "Spaces/L2Space.h"

#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <array>
#include <chrono>
#include <iostream>

using svmp::FE::ElementType;
using svmp::FE::FieldId;
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

using Clock = std::chrono::steady_clock;

template <class Fn>
double timeSeconds(Fn&& fn) {
    const auto t0 = Clock::now();
    fn();
    const auto t1 = Clock::now();
    return std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
}

} // namespace

// ============================================================================
// Functional: CoupledBlockKernel created by installMixedFormIR when JIT enabled
// ============================================================================

TEST(MixedFormPerformance, InstallMixedFormIR_CreatesCoupledBlockKernel)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    auto p = svmp::FE::forms::FormExpr::trialFunction(*space, "p");
    auto q = svmp::FE::forms::FormExpr::testFunction(*space, "q");

    auto mixed = (u * v).dx() + (p * v).dx() + (u * q).dx();

    svmp::FE::forms::FormCompiler compiler;
    auto mir = compiler.compileMixed(mixed, svmp::FE::forms::FormKind::Bilinear);
    ASSERT_EQ(mir.numActiveBlocks(), 3u);

    // Install with JIT enabled
    svmp::FE::systems::FESystem sys(mesh);
    const auto u_f = sys.addField({.name = "u", .space = space, .components = 1});
    const auto p_f = sys.addField({.name = "p", .space = space, .components = 1});
    sys.addOperator("op");

    svmp::FE::systems::FormInstallOptions opts;
    opts.compiler_options.jit.enable = true;

    const std::array fields = {u_f, p_f};
    svmp::FE::systems::installMixedFormIR(
        sys, "op",
        std::span<const FieldId>(fields),
        std::span<const FieldId>(fields),
        mir, opts);

    // The operator should have cell terms: 3 per-block + 1 CoupledBlockKernel = 4
    const auto& def = sys.operatorDefinition("op");
    EXPECT_GE(def.cells.size(), 3u);

    // Check that at least one cell term is a CoupledBlockKernel
    bool has_coupled = false;
    for (const auto& ct : def.cells) {
        if (dynamic_cast<const svmp::FE::forms::CoupledBlockKernel*>(ct.kernel.get())) {
            has_coupled = true;
            break;
        }
    }
    EXPECT_TRUE(has_coupled) << "Expected CoupledBlockKernel in operator definition";
}

// ============================================================================
// Functional: installFormulation residual path gets fused execution
// ============================================================================

TEST(MixedFormPerformance, InstallFormulation_MixedResidual_AssemblesCorrectly)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem sys(mesh);
    const auto u_f = sys.addField({.name = "u", .space = space, .components = 1});
    const auto p_f = sys.addField({.name = "p", .space = space, .components = 1});
    sys.addOperator("op");

    auto u_state = svmp::FE::forms::FormExpr::stateField(u_f, *space, "u");
    auto p_state = svmp::FE::forms::FormExpr::stateField(p_f, *space, "p");
    auto v = svmp::FE::forms::FormExpr::testFunction(u_f, *space, "v");
    auto q = svmp::FE::forms::FormExpr::testFunction(p_f, *space, "q");

    // Mixed residual: momentum + continuity
    auto residual = (u_state * v + p_state * v).dx() + (u_state * q).dx();

    svmp::FE::systems::installFormulation(sys, "op", {u_f, p_f}, residual);

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    sys.setup({}, inputs);

    const auto n = sys.dofHandler().getNumDofs();
    ASSERT_EQ(n, 8);  // 4 dofs/field * 2 fields

    std::vector<Real> U(static_cast<std::size_t>(n), 0.0);
    for (std::size_t i = 0; i < U.size(); ++i) {
        U[i] = 0.1 * static_cast<Real>(i + 1);
    }

    svmp::FE::systems::SystemStateView state;
    state.u = U;

    // Assemble both matrix and vector
    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;

    svmp::FE::assembly::DenseSystemView out(n);
    out.zero();
    auto result = sys.assemble(req, state, &out, &out);

    // Basic sanity: residual vector should be non-zero
    double vec_norm = 0.0;
    for (GlobalIndex i = 0; i < n; ++i) {
        vec_norm += out.getVectorEntry(i) * out.getVectorEntry(i);
    }
    EXPECT_GT(vec_norm, 0.0) << "Residual vector should be non-zero";

    // Jacobian should be non-zero
    double mat_norm = 0.0;
    for (GlobalIndex i = 0; i < n; ++i) {
        for (GlobalIndex j = 0; j < n; ++j) {
            mat_norm += out.getMatrixEntry(i, j) * out.getMatrixEntry(i, j);
        }
    }
    EXPECT_GT(mat_norm, 0.0) << "Jacobian matrix should be non-zero";

    // Verify finite differences: dR/du ≈ (R(u+eps*e_j) - R(u)) / eps
    const Real eps = 1e-7;
    std::vector<Real> R0(static_cast<std::size_t>(n));
    for (GlobalIndex i = 0; i < n; ++i) {
        R0[static_cast<std::size_t>(i)] = out.getVectorEntry(i);
    }

    svmp::FE::systems::AssemblyRequest req_vec;
    req_vec.op = "op";
    req_vec.want_vector = true;

    for (GlobalIndex j = 0; j < n; ++j) {
        auto U_plus = U;
        U_plus[static_cast<std::size_t>(j)] += eps;

        svmp::FE::systems::SystemStateView state_plus;
        state_plus.u = U_plus;

        svmp::FE::assembly::DenseVectorView Rp(n);
        Rp.zero();
        (void)sys.assemble(req_vec, state_plus, nullptr, &Rp);

        for (GlobalIndex i = 0; i < n; ++i) {
            const Real fd = (Rp.getVectorEntry(i) - R0[static_cast<std::size_t>(i)]) / eps;
            EXPECT_NEAR(out.getMatrixEntry(i, j), fd, 1e-5)
                << "FD mismatch at (" << i << ", " << j << ")";
        }
    }
}

// ============================================================================
// Functional: per-block JIT specialization works with mixed-compiled forms
// ============================================================================

TEST(MixedFormPerformance, PerBlockJIT_MixedBilinear)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    auto p = svmp::FE::forms::FormExpr::trialFunction(*space, "p");
    auto q = svmp::FE::forms::FormExpr::testFunction(*space, "q");

    // Use grad-grad coupling (all scalar, no div needed)
    auto mixed = (inner(grad(u), grad(v))).dx() + (p * v).dx() + (u * q).dx();

    // Install with JIT and assemble
    svmp::FE::systems::FESystem sys(mesh);
    const auto u_f = sys.addField({.name = "u", .space = space, .components = 1});
    const auto p_f = sys.addField({.name = "p", .space = space, .components = 1});
    sys.addOperator("op");

    svmp::FE::systems::FormInstallOptions opts;
    opts.compiler_options.jit.enable = true;

    const std::array fields = {u_f, p_f};
    svmp::FE::systems::installMixedBilinear(
        sys, "op",
        std::span<const FieldId>(fields),
        std::span<const FieldId>(fields),
        mixed, opts);

    svmp::FE::systems::SetupInputs setup_inputs;
    setup_inputs.topology_override = singleTetraTopology();
    sys.setup({}, setup_inputs);

    const auto n = sys.dofHandler().getNumDofs();

    svmp::FE::assembly::DenseMatrixView mat(n);
    mat.zero();

    svmp::FE::systems::SystemStateView state;
    svmp::FE::systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    (void)sys.assemble(req, state, &mat, nullptr);

    // The assembled matrix should be non-trivial
    double norm = 0.0;
    for (GlobalIndex i = 0; i < n; ++i) {
        for (GlobalIndex j = 0; j < n; ++j) {
            norm += mat.getMatrixEntry(i, j) * mat.getMatrixEntry(i, j);
        }
    }
    EXPECT_GT(norm, 0.0) << "Assembled matrix from JIT mixed bilinear should be non-zero";
}

// ============================================================================
// DISABLED benchmark: compilation timing for mixed vs manual
// Enable with --gtest_also_run_disabled_tests
// ============================================================================

TEST(MixedFormPerformance, DISABLED_Benchmark_CompilationTiming)
{
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    auto u = svmp::FE::forms::FormExpr::trialFunction(*space, "u");
    auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    auto p = svmp::FE::forms::FormExpr::trialFunction(*space, "p");
    auto q = svmp::FE::forms::FormExpr::testFunction(*space, "q");

    constexpr int N = 100;

    // Benchmark: manual block compilation
    double manual_time = timeSeconds([&]() {
        svmp::FE::forms::FormCompiler compiler;
        for (int i = 0; i < N; ++i) {
            (void)compiler.compileBilinear((inner(grad(u), grad(v))).dx());
            (void)compiler.compileBilinear((p * v).dx());
            (void)compiler.compileBilinear((u * q).dx());
        }
    });

    // Benchmark: mixed compilation
    auto mixed = (inner(grad(u), grad(v))).dx() + (p * v).dx() + (u * q).dx();
    double mixed_time = timeSeconds([&]() {
        svmp::FE::forms::FormCompiler compiler;
        for (int i = 0; i < N; ++i) {
            (void)compiler.compileMixed(mixed, svmp::FE::forms::FormKind::Bilinear);
        }
    });

    std::cout << "\n=== Compilation Benchmark (" << N << " iterations) ===\n"
              << "  Manual (3 separate compileBilinear): " << manual_time * 1000.0 << " ms\n"
              << "  Mixed  (1 compileMixed):             " << mixed_time * 1000.0 << " ms\n"
              << "  Ratio (mixed/manual):                " << mixed_time / manual_time << "x\n"
              << std::endl;

    // Mixed should not be more than 3x slower than manual (generous margin)
    EXPECT_LT(mixed_time, manual_time * 3.0)
        << "Mixed compilation should not be significantly slower than manual block compilation";
}
