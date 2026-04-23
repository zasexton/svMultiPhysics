/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_JIT_SIMDBatch.cpp
 * @brief SIMD batch vectorization parity tests.
 *
 * Compiles JIT kernels with simd_batch=true and simd_batch=false, runs both
 * on the same mesh/solution data via full assembly, and verifies that the
 * resulting global matrices and vectors match to machine epsilon.
 *
 * The SIMD batch path processes multiple elements simultaneously using LLVM
 * vector types (e.g., <2 x double> on SSE2).  These tests ensure the
 * vectorized path produces bit-identical results to the scalar batch path.
 *
 * Test cases:
 *   1. ScalarLaplacian        -- inner(grad(u), grad(v)), simplest bilinear form
 *   2. NonlinearReaction      -- inner(grad(u), grad(v)) + u*u*v, nonlinear in u
 *   3. StokesVMS_VV           -- velocity-velocity VMS block (Triangle3 + ProductSpace + field_solutions)
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Assembly/TimeIntegrationContext.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/JIT/JITKernelWrapper.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

#if SVMP_FE_ENABLE_LLVM_JIT

namespace {

// ============================================================================
// Triangle3 mesh for the Stokes VMS test (2D, 4 cells)
// ============================================================================

class FourTriangleMeshAccess final : public assembly::IMeshAccess {
public:
    FourTriangleMeshAccess()
    {
        nodes_ = {
            {0.0, 0.0, 0.0},
            {1.2, 0.0, 0.0},
            {0.5, 0.7, 0.0},
            {0.0, 1.3, 0.0},
            {1.2, 1.3, 0.0},
        };
        cells_ = {{0, 1, 2}, {1, 4, 2}, {4, 3, 2}, {3, 0, 2}};
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 4; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 4; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }
    [[nodiscard]] bool isOwnedCell(GlobalIndex) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex) const override { return ElementType::Triangle3; }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override
    {
        const auto& c = cells_.at(static_cast<std::size_t>(cell_id));
        nodes.assign(c.begin(), c.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex cell_id, std::vector<std::array<Real, 3>>& coords) const override
    {
        const auto& c = cells_.at(static_cast<std::size_t>(cell_id));
        coords.resize(c.size());
        for (std::size_t i = 0; i < c.size(); ++i)
            coords[i] = nodes_.at(static_cast<std::size_t>(c[i]));
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex, GlobalIndex) const override { return 0; }
    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex) const override { return -1; }
    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex) const override { return {0, 0}; }

    void forEachCell(std::function<void(GlobalIndex)> cb) const override { for (GlobalIndex i = 0; i < 4; ++i) cb(i); }
    void forEachOwnedCell(std::function<void(GlobalIndex)> cb) const override { forEachCell(std::move(cb)); }
    void forEachBoundaryFace(int, std::function<void(GlobalIndex, GlobalIndex)>) const override {}
    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)>) const override {}

private:
    std::vector<std::array<Real, 3>> nodes_;
    std::vector<std::array<GlobalIndex, 3>> cells_;
};

inline dofs::DofMap createFourTriangleDofMap()
{
    dofs::DofMap dof_map(4, 5, 3);
    dof_map.setCellDofs(0, std::vector<GlobalIndex>{0, 1, 2});
    dof_map.setCellDofs(1, std::vector<GlobalIndex>{1, 4, 2});
    dof_map.setCellDofs(2, std::vector<GlobalIndex>{4, 3, 2});
    dof_map.setCellDofs(3, std::vector<GlobalIndex>{3, 0, 2});
    dof_map.setNumDofs(5);
    dof_map.setNumLocalDofs(5);
    dof_map.finalize();
    return dof_map;
}

inline dofs::DofMap createFourTriangleVectorDofMap()
{
    dofs::DofMap dof_map(4, 10, 6);
    dof_map.setCellDofs(0, std::vector<GlobalIndex>{0, 5, 1, 6, 2, 7});
    dof_map.setCellDofs(1, std::vector<GlobalIndex>{1, 6, 4, 9, 2, 7});
    dof_map.setCellDofs(2, std::vector<GlobalIndex>{4, 9, 3, 8, 2, 7});
    dof_map.setCellDofs(3, std::vector<GlobalIndex>{3, 8, 0, 5, 2, 7});
    dof_map.setNumDofs(10);
    dof_map.setNumLocalDofs(10);
    dof_map.finalize();
    return dof_map;
}

// ============================================================================
// DofMap for TwoTetraSharedFaceMeshAccess (shared face, 5 nodes)
// ============================================================================

inline dofs::DofMap createTwoTetraDofMap()
{
    dofs::DofMap dof_map(/*n_cells=*/2, /*n_dofs_total=*/5, /*dofs_per_cell=*/4);
    dof_map.setCellDofs(0, std::vector<GlobalIndex>{0, 1, 2, 3});
    dof_map.setCellDofs(1, std::vector<GlobalIndex>{1, 2, 3, 4});
    dof_map.setNumDofs(5);
    dof_map.setNumLocalDofs(5);
    dof_map.finalize();
    return dof_map;
}

// ============================================================================
// Comparison helpers
// ============================================================================

void expectDenseNear(const assembly::DenseMatrixView& A,
                     const assembly::DenseMatrixView& B,
                     Real tol,
                     std::string_view label)
{
    ASSERT_EQ(A.numRows(), B.numRows()) << label;
    ASSERT_EQ(A.numCols(), B.numCols()) << label;
    for (GlobalIndex i = 0; i < A.numRows(); ++i) {
        for (GlobalIndex j = 0; j < A.numCols(); ++j) {
            EXPECT_NEAR(A.getMatrixEntry(i, j), B.getMatrixEntry(i, j), tol)
                << label << " matrix(" << i << "," << j << ")";
        }
    }
}

void expectDenseNear(const assembly::DenseVectorView& a,
                     const assembly::DenseVectorView& b,
                     Real tol,
                     std::string_view label)
{
    ASSERT_EQ(a.numRows(), b.numRows()) << label;
    for (GlobalIndex i = 0; i < a.numRows(); ++i) {
        EXPECT_NEAR(a.getVectorEntry(i), b.getVectorEntry(i), tol)
            << label << " vector(" << i << ")";
    }
}

// ============================================================================
// JITOptions factories
// ============================================================================

[[nodiscard]] forms::JITOptions makeJITOptions(bool simd_batch, bool vectorize = true)
{
    forms::JITOptions opt;
    opt.enable = true;
    opt.optimization_level = 2;
    opt.vectorize = vectorize;
    opt.simd_batch = simd_batch;
    opt.cache_kernels = true;
    return opt;
}

// ============================================================================
// Core comparison routine
// ============================================================================

struct SIMDBatchCompareOptions {
    const assembly::TimeIntegrationContext* time_ctx{nullptr};
    Real time_step{0.0};
    std::optional<std::vector<Real>> prev_solution{};
    std::span<const assembly::FieldSolutionAccess> field_access{};
};

void expectJITBatchOptionsMatch(const assembly::IMeshAccess& mesh,
                                const dofs::DofMap& row_map,
                                const dofs::DofMap& col_map,
                                const spaces::FunctionSpace& test_space,
                                const spaces::FunctionSpace& trial_space,
                                const FormExpr& residual,
                                const std::vector<Real>& U,
                                const forms::JITOptions& lhs_options,
                                const forms::JITOptions& rhs_options,
                                Real vec_tol,
                                Real mat_tol,
                                const SIMDBatchCompareOptions& options = {})
{
    FormCompiler compiler;

    auto lhs_ir = compiler.compileResidual(residual);
    auto rhs_ir = compiler.compileResidual(residual);

    auto lhs_fallback =
        std::make_shared<SymbolicNonlinearFormKernel>(std::move(lhs_ir), NonlinearKernelOutput::Both);
    forms::jit::JITKernelWrapper lhs_kernel(lhs_fallback, lhs_options);
    lhs_kernel.resolveInlinableConstitutives();

    auto rhs_fallback =
        std::make_shared<SymbolicNonlinearFormKernel>(std::move(rhs_ir), NonlinearKernelOutput::Both);
    forms::jit::JITKernelWrapper rhs_kernel(rhs_fallback, rhs_options);
    rhs_kernel.resolveInlinableConstitutives();

    auto runAssembly = [&](assembly::AssemblyKernel& kernel,
                           assembly::DenseMatrixView& J,
                           assembly::DenseVectorView& R) {
        assembly::StandardAssembler assembler;
        if (&row_map == &col_map) {
            assembler.setDofMap(row_map);
        } else {
            assembler.setRowDofMap(row_map);
            assembler.setColDofMap(col_map);
        }
        assembler.setCurrentSolution(U);
        assembler.setTimeIntegrationContext(options.time_ctx);
        assembler.setTimeStep(options.time_step);
        if (options.prev_solution.has_value()) {
            assembler.setPreviousSolution(*options.prev_solution);
        }
        if (!options.field_access.empty()) {
            assembler.setFieldSolutionAccess(options.field_access);
        }
        J.zero();
        R.zero();
        auto result = assembler.assembleBoth(mesh, test_space, trial_space, kernel, J, R);
        EXPECT_GT(result.elements_assembled, 0) << "No cells assembled";
    };

    const GlobalIndex n_rows = row_map.getNumDofs();
    const GlobalIndex n_cols = col_map.getNumDofs();

    assembly::DenseMatrixView J_lhs(n_rows, n_cols);
    assembly::DenseVectorView R_lhs(n_rows);
    runAssembly(lhs_kernel, J_lhs, R_lhs);

    assembly::DenseMatrixView J_rhs(n_rows, n_cols);
    assembly::DenseVectorView R_rhs(n_rows);
    runAssembly(rhs_kernel, J_rhs, R_rhs);

    expectDenseNear(R_lhs, R_rhs, vec_tol, "residual");
    expectDenseNear(J_lhs, J_rhs, mat_tol, "tangent");
}

/**
 * @brief Compare full assembly results between simd_batch=true and simd_batch=false.
 *
 * Both paths use JIT compilation; the only difference is whether the generated
 * kernel uses SIMD vector types for the batch loop.
 */
void expectSIMDBatchMatchesScalar(const assembly::IMeshAccess& mesh,
                                  const dofs::DofMap& row_map,
                                  const dofs::DofMap& col_map,
                                  const spaces::FunctionSpace& test_space,
                                  const spaces::FunctionSpace& trial_space,
                                  const FormExpr& residual,
                                  const std::vector<Real>& U,
                                  Real vec_tol,
                                  Real mat_tol,
                                  const SIMDBatchCompareOptions& options = {})
{
    FormCompiler compiler;

    auto ir_simd = compiler.compileResidual(residual);
    auto ir_scalar = compiler.compileResidual(residual);

    auto fallback_simd =
        std::make_shared<SymbolicNonlinearFormKernel>(std::move(ir_simd), NonlinearKernelOutput::Both);
    forms::jit::JITKernelWrapper simd_kernel(fallback_simd, makeJITOptions(/*simd_batch=*/true));
    simd_kernel.resolveInlinableConstitutives();

    auto fallback_scalar =
        std::make_shared<SymbolicNonlinearFormKernel>(std::move(ir_scalar), NonlinearKernelOutput::Both);
    forms::jit::JITKernelWrapper scalar_kernel(fallback_scalar, makeJITOptions(/*simd_batch=*/false));
    scalar_kernel.resolveInlinableConstitutives();

    auto runAssembly = [&](assembly::AssemblyKernel& kernel,
                           assembly::DenseMatrixView& J,
                           assembly::DenseVectorView& R) {
        assembly::StandardAssembler assembler;
        if (&row_map == &col_map) {
            assembler.setDofMap(row_map);
        } else {
            assembler.setRowDofMap(row_map);
            assembler.setColDofMap(col_map);
        }
        assembler.setCurrentSolution(U);
        assembler.setTimeIntegrationContext(options.time_ctx);
        assembler.setTimeStep(options.time_step);
        if (options.prev_solution.has_value()) {
            assembler.setPreviousSolution(*options.prev_solution);
        }
        if (!options.field_access.empty()) {
            assembler.setFieldSolutionAccess(options.field_access);
        }
        J.zero();
        R.zero();
        auto result = assembler.assembleBoth(mesh, test_space, trial_space, kernel, J, R);
        EXPECT_GT(result.elements_assembled, 0) << "No cells assembled";
    };

    const GlobalIndex n_rows = row_map.getNumDofs();
    const GlobalIndex n_cols = col_map.getNumDofs();

    assembly::DenseMatrixView J_simd(n_rows, n_cols);
    assembly::DenseVectorView R_simd(n_rows);
    runAssembly(simd_kernel, J_simd, R_simd);

    assembly::DenseMatrixView J_scalar(n_rows, n_cols);
    assembly::DenseVectorView R_scalar(n_rows);
    runAssembly(scalar_kernel, J_scalar, R_scalar);

    // Verify parity: SIMD batch must match scalar batch exactly.
    expectDenseNear(R_simd, R_scalar, vec_tol, "residual");
    expectDenseNear(J_simd, J_scalar, mat_tol, "tangent");

    // Sanity check: at least one non-zero matrix entry (avoid vacuous pass).
    bool any_nonzero_simd = false, any_nonzero_scalar = false;
    for (GlobalIndex i = 0; i < n_rows; ++i) {
        for (GlobalIndex j = 0; j < n_cols; ++j) {
            if (std::abs(J_simd.getMatrixEntry(i, j)) > 1e-15) any_nonzero_simd = true;
            if (std::abs(J_scalar.getMatrixEntry(i, j)) > 1e-15) any_nonzero_scalar = true;
        }
    }
    EXPECT_TRUE(any_nonzero_simd) << "SIMD assembled matrix is entirely zero";
    EXPECT_TRUE(any_nonzero_scalar) << "Scalar assembled matrix is entirely zero";
}

void expectSIMDBatchMatchesScalar(const assembly::IMeshAccess& mesh,
                                  const dofs::DofMap& dof_map,
                                  const spaces::FunctionSpace& space,
                                  const FormExpr& residual,
                                  const std::vector<Real>& U,
                                  Real vec_tol,
                                  Real mat_tol,
                                  const SIMDBatchCompareOptions& options = {})
{
    expectSIMDBatchMatchesScalar(mesh, dof_map, dof_map, space, space,
                                residual, U, vec_tol, mat_tol, options);
}

} // anonymous namespace

// ============================================================================
// Test 1: Scalar Laplacian -- inner(grad(u), grad(v))
//
// Simplest bilinear form, no field solutions, no time dependence.
// 2-element Tetra4 mesh with shared face (batch size 2 fills one SSE2 pair).
// ============================================================================

TEST(JITSIMDBatchParity, ScalarLaplacian_Tetra4)
{
    TwoTetraSharedFaceMeshAccess mesh;
    auto dof_map = createTwoTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto residual = inner(grad(u), grad(v)).dx();

    const std::vector<Real> U = {0.1, -0.2, 0.3, -0.1, 0.15};

    expectSIMDBatchMatchesScalar(mesh, dof_map, space, residual, U,
                                /*vec_tol=*/1e-14, /*mat_tol=*/1e-14);
}

// ============================================================================
// Test 2: Nonlinear reaction-diffusion -- inner(grad(u), grad(v)) + u*u*v
//
// Nonlinear residual with u^2 term.  Tangent has solution-dependent
// contribution 2*u*v.  Tests that nonlinear terms are correctly
// vectorized across batch elements with different solution values.
// ============================================================================

TEST(JITSIMDBatchParity, NonlinearReactionDiffusion_Tetra4)
{
    TwoTetraSharedFaceMeshAccess mesh;
    auto dof_map = createTwoTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto residual = (inner(grad(u), grad(v)) + u * u * v).dx();

    const std::vector<Real> U = {0.5, -0.3, 0.7, 0.1, -0.4};

    expectSIMDBatchMatchesScalar(mesh, dof_map, space, residual, U,
                                /*vec_tol=*/1e-14, /*mat_tol=*/1e-14);
}

TEST(JITSIMDBatchParity, SingleCellUnderfilledBatch_Tetra4)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto residual = (inner(grad(u), grad(v)) + FormExpr::constant(0.5) * u * u * v).dx();

    const std::vector<Real> U = {0.25, -0.5, 0.75, -0.1};

    expectSIMDBatchMatchesScalar(mesh, dof_map, space, residual, U,
                                /*vec_tol=*/1e-14, /*mat_tol=*/1e-14);
}

TEST(JITSIMDBatchParity, VectorizationDisabledBatchMatchesScalarPath_Tetra4)
{
    TwoTetraSharedFaceMeshAccess mesh;
    auto dof_map = createTwoTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto residual = (inner(grad(u), grad(v)) + u * u * v).dx();

    const std::vector<Real> U = {0.4, -0.2, 0.6, 0.1, -0.3};

    auto batch_no_vector = makeJITOptions(/*simd_batch=*/true, /*vectorize=*/false);
    auto scalar_no_vector = makeJITOptions(/*simd_batch=*/false, /*vectorize=*/false);

    expectJITBatchOptionsMatch(mesh, dof_map, dof_map, space, space, residual, U,
                               batch_no_vector, scalar_no_vector,
                               /*vec_tol=*/1e-14, /*mat_tol=*/1e-14);
}

TEST(JITSIMDBatchParity, VectorTestScalarTrialCrossBlock_Triangle3)
{
    constexpr int dim = 2;

    FourTriangleMeshAccess mesh;
    auto base = std::make_shared<spaces::H1Space>(ElementType::Triangle3, /*order=*/1);
    spaces::ProductSpace vel_space(base, dim);
    spaces::H1Space pressure_space(ElementType::Triangle3, /*order=*/1);

    auto row_map = createFourTriangleVectorDofMap();
    auto col_map = createFourTriangleDofMap();

    const auto p = TrialFunction(pressure_space, "p");
    const auto v = TestFunction(vel_space, "v");
    const auto residual = (p * div(v)).dx();

    const std::vector<Real> U = {0.2, -0.1, 0.4, 0.05, -0.3};

    expectSIMDBatchMatchesScalar(mesh, row_map, col_map, vel_space, pressure_space,
                                residual, U,
                                /*vec_tol=*/1e-13, /*mat_tol=*/1e-13);
}

// ============================================================================
// Test 3: Stokes VMS velocity-velocity block on Triangle3 (2D)
//
// Tests ProductSpace (vector-valued basis functions) and field_solutions
// (pressure as a DiscreteField).  This is the most complex form tested.
//
// NOTE: While the has_field_solutions guard is active in LLVMGen, both the
// simd_batch=true and simd_batch=false paths will fall through to the same
// scalar batch codegen.  This test still verifies correctness and will become
// a meaningful SIMD parity check once the guard is lifted.
// ============================================================================

TEST(JITSIMDBatchParity, StokesVMS_VV_Triangle3)
{
    constexpr int dim = 2;

    FourTriangleMeshAccess mesh;

    auto base = std::make_shared<spaces::H1Space>(ElementType::Triangle3, /*order=*/1);
    spaces::ProductSpace vel_space(base, dim);
    spaces::H1Space p_space(ElementType::Triangle3, /*order=*/1);

    auto dof_map_u = createFourTriangleVectorDofMap();
    auto dof_map_p = createFourTriangleDofMap();

    const auto rho = FormExpr::constant(Real(1.0));
    const auto mu = FormExpr::constant(Real(0.01));
    const auto eps = FormExpr::constant(Real(1e-12));

    const FieldId p_id = static_cast<FieldId>(1);

    const auto u = TrialFunction(vel_space, "u");
    const auto v = TestFunction(vel_space, "v");
    const auto p = FormExpr::discreteField(p_id, p_space, "p");

    std::vector<FormExpr> zero_vec;
    zero_vec.reserve(dim);
    for (int d = 0; d < dim; ++d) {
        zero_vec.push_back(FormExpr::constant(0.0));
    }
    const auto a = FormExpr::asVector(zero_vec);
    const auto f = FormExpr::asVector(std::move(zero_vec));

    const auto stress = FormExpr::constant(2.0) * mu * sym(grad(u));
    const auto r_m = rho * (dt(u) + grad(u) * a - f) + grad(p) - div(stress);

    const auto inertia = rho * inner(dt(u), v);
    const auto viscous = FormExpr::constant(2.0) * mu * inner(sym(grad(u)), sym(grad(v)));
    const auto forcing = -rho * inner(f, v);

    const auto dt_step = FormExpr::effectiveTimeStep();
    const auto ct_m = FormExpr::constant(Real(1.0));
    const auto ct_c = FormExpr::constant(Real(36.0));

    const auto Jinv_expr = Jinv();
    const auto K = transpose(Jinv_expr) * Jinv_expr;
    const auto nu = mu / rho;

    const auto kT = FormExpr::constant(4.0) * (ct_m * ct_m) / (dt_step * dt_step);
    const auto kU = inner(a, K * a);
    const auto kS = ct_c * doubleContraction(K, K) * (nu * nu);
    const auto tau_m = FormExpr::constant(1.0) / (rho * sqrt(kT + kU + kS + eps));
    const auto tau_c = FormExpr::constant(1.0) / (tau_m * trace(K) + eps);

    const auto u_sub = -tau_m * r_m;
    const auto p_sub = -tau_c * div(u);
    const auto p_adv = p + p_sub;
    const auto pressure_adv = -p_adv * div(v);

    const auto convection_adv = rho * inner(grad(u) * a, v);
    const auto supg = -rho * inner(grad(v) * a, u_sub);

    const auto residual = (inertia + convection_adv + viscous + pressure_adv + forcing + supg).dx();

    const std::vector<Real> U = {
        0.10, -0.20,  0.05,  0.15, -0.08,
        0.03, -0.12,  0.07, -0.04,  0.11,
        0.20, -0.10,  0.05,  0.12, -0.03,
    };

    constexpr Real dt_val = 0.05;
    assembly::TimeIntegrationContext ti;
    ti.integrator_name = "stokes_vms_simd_test";
    ti.dt1 = assembly::TimeDerivativeStencil{
        .order = 1,
        .a = {Real(1.0) / dt_val, -Real(1.0) / dt_val}
    };

    std::vector<Real> prev_U(15, 0.0);

    const std::array<assembly::FieldSolutionAccess, 1> fields = {{
        {.field = p_id, .space = &p_space, .dof_map = &dof_map_p, .dof_offset = 10},
    }};

    SIMDBatchCompareOptions opts;
    opts.time_ctx = &ti;
    opts.time_step = dt_val;
    opts.prev_solution = prev_U;
    opts.field_access = fields;

    expectSIMDBatchMatchesScalar(mesh, dof_map_u, dof_map_u, vel_space, vel_space,
                                residual, U,
                                /*vec_tol=*/1e-12, /*mat_tol=*/1e-12, opts);
}

#endif // SVMP_FE_ENABLE_LLVM_JIT

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp
