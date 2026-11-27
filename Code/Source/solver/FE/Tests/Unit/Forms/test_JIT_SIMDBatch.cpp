/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_JIT_SIMDBatch.cpp
 * @brief Compare compiled cell batches with independently interpreted element assembly.
 *
 * Bilinear forms run through the scalar and SIMD batch backends. Nonlinear
 * residuals and their tangents exercise the compiled backend's effective policy.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Assembly/TimeIntegrationContext.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/JIT/JITKernelWrapper.h"
#include "Forms/JIT/HardwareProfile.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"
#include "Tests/Unit/Forms/JITTestHelpers.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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
// The four triangles above on a moving mesh: every node has its own
// displacement, so the current geometry differs from cell to cell
// ============================================================================

class MovingFourTriangleMeshAccess final : public assembly::IMeshAccess {
public:
    MovingFourTriangleMeshAccess()
    {
        reference_nodes_ = {
            {0.0, 0.0, 0.0},
            {1.2, 0.0, 0.0},
            {0.5, 0.7, 0.0},
            {0.0, 1.3, 0.0},
            {1.2, 1.3, 0.0},
        };
        const std::array<std::array<Real, 2>, 5> displacement{{
            {0.02, -0.01}, {-0.03, 0.04}, {0.05, 0.02}, {0.01, -0.06}, {-0.04, 0.03},
        }};
        current_nodes_ = reference_nodes_;
        for (std::size_t i = 0; i < current_nodes_.size(); ++i) {
            current_nodes_[i][0] += displacement[i][0];
            current_nodes_[i][1] += displacement[i][1];
        }
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
        return current_nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex cell_id, std::vector<std::array<Real, 3>>& coords) const override
    {
        getCellCoordinates(cell_id, assembly::CoordinateFrame::Current, coords);
    }

    [[nodiscard]] bool supportsCoordinateFrame(assembly::CoordinateFrame) const override { return true; }

    void getCellCoordinates(GlobalIndex cell_id,
                            assembly::CoordinateFrame frame,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        const auto& nodes = frame == assembly::CoordinateFrame::Reference ? reference_nodes_ : current_nodes_;
        const auto& c = cells_.at(static_cast<std::size_t>(cell_id));
        coords.resize(c.size());
        for (std::size_t i = 0; i < c.size(); ++i)
            coords[i] = nodes.at(static_cast<std::size_t>(c[i]));
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex, GlobalIndex) const override { return 0; }
    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex) const override { return -1; }
    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex) const override { return {0, 0}; }

    void forEachCell(std::function<void(GlobalIndex)> cb) const override { for (GlobalIndex i = 0; i < 4; ++i) cb(i); }
    void forEachOwnedCell(std::function<void(GlobalIndex)> cb) const override { forEachCell(std::move(cb)); }
    void forEachBoundaryFace(int, std::function<void(GlobalIndex, GlobalIndex)>) const override {}
    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)>) const override {}

private:
    std::vector<std::array<Real, 3>> reference_nodes_;
    std::vector<std::array<Real, 3>> current_nodes_;
    std::vector<std::array<GlobalIndex, 3>> cells_;
};

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

void expectCompiledResidualMatchesInterpreter(const assembly::IMeshAccess& mesh,
                                              const dofs::DofMap& dofs,
                                              const spaces::FunctionSpace& space,
                                              const FormExpr& residual,
                                              const std::vector<Real>& state,
                                              const forms::JITOptions& jit_options,
                                              Real tolerance,
                                              const SIMDBatchCompareOptions& options = {})
{
    FormCompiler compiler;
    SymbolicNonlinearFormKernel interpreted(compiler.compileResidual(residual), NonlinearKernelOutput::Both);
    interpreted.resolveInlinableConstitutives();
    auto fallback = std::make_shared<SymbolicNonlinearFormKernel>(
        compiler.compileResidual(residual), NonlinearKernelOutput::Both);
    forms::jit::JITKernelWrapper compiled(fallback, jit_options);
    compiled.resolveInlinableConstitutives();

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dofs);
    assembler.setCurrentSolution(state);
    assembler.setTimeIntegrationContext(options.time_ctx);
    assembler.setTimeStep(options.time_step);
    if (options.prev_solution) assembler.setPreviousSolution(*options.prev_solution);
    assembler.setFieldSolutionAccess(options.field_access);
    const auto n = dofs.getNumDofs();
    assembly::DenseMatrixView J(n), J_ref(n);
    assembly::DenseVectorView R(n), R_ref(n);
    assembler.assembleBoth(mesh, space, space, interpreted, J_ref, R_ref);
    const auto result = assembler.assembleBoth(mesh, space, space, compiled, J, R);
    ASSERT_EQ(result.elements_assembled, mesh.numCells());
    expectCompiledExecution(compiled, *fallback);
    expectDenseNear(R, R_ref, tolerance, "residual");
    expectDenseNear(J, J_ref, tolerance, "tangent");
}

// Look in the unoptimized IR so ordinary loop vectorization cannot satisfy the check.
void expectTwoLaneArithmetic(const std::filesystem::path& directory, bool expected)
{
    bool found_module = false;
    bool found_vector_multiply = false;
    for (const auto& file : std::filesystem::directory_iterator(directory)) {
        if (!file.path().filename().string().ends_with("_before.ll")) continue;
        found_module = true;
        std::ifstream stream(file.path());
        for (std::string line; std::getline(stream, line); ) {
            if (line.find("fmul") != std::string::npos && line.find("<2 x double>") != std::string::npos) {
                found_vector_multiply = true;
            }
        }
    }
    EXPECT_TRUE(found_module);
    EXPECT_EQ(found_vector_multiply, expected);
}

void expectBilinearBatches(const assembly::IMeshAccess& mesh,
                           const dofs::DofMap& row_map, const dofs::DofMap& col_map,
                           const spaces::FunctionSpace& test_space, const spaces::FunctionSpace& trial_space,
                           const FormExpr& form, std::optional<Real> diagonal = {}, bool inspect_ir = false)
{
    FormCompiler compiler;
    FormKernel interpreted(compiler.compileBilinear(form));
    assembly::AssemblyOptions assembly_options;
    assembly_options.use_batching = true;
    assembly_options.batch_size = 2;
    assembly::StandardAssembler assembler(assembly_options);
    assembler.setRowDofMap(row_map);
    assembler.setColDofMap(col_map);
    assembly::DenseMatrixView reference(row_map.getNumDofs(), col_map.getNumDofs());
    assembler.assembleMatrix(mesh, test_space, trial_space, interpreted, reference);
    if (diagonal) EXPECT_NEAR(reference.getMatrixEntry(0, 0), *diagonal, 1e-14);

    for (const bool simd : {false, true}) {
        auto options = makeJITOptions(simd);
        options.specialization.enable = false;
        options.cache_kernels = false;
        std::filesystem::path directory;
        if (inspect_ir) {
            const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
            directory = std::filesystem::temp_directory_path() / ("svmp_fe_simd_" + std::to_string(stamp));
            ASSERT_TRUE(std::filesystem::create_directory(directory));
            options.dump_llvm_ir = true;
            options.dump_directory = directory.string();
        }
        auto source = std::make_shared<FormKernel>(compiler.compileBilinear(form));
        forms::jit::JITKernelWrapper compiled(source, options);
        assembly::DenseMatrixView actual(row_map.getNumDofs(), col_map.getNumDofs());
        const auto result = assembler.assembleMatrix(mesh, test_space, trial_space, compiled, actual);
        EXPECT_EQ(result.elements_assembled, mesh.numCells());
        expectCompiledExecution(compiled, *source);
        expectDenseNear(actual, reference, 1e-13, simd ? "SIMD batch" : "scalar batch");
        if (inspect_ir) {
            // The cross-cell packing currently supports two lanes. Wider builds
            // and an explicit runtime disable use scalar arithmetic.
            const char* policy = std::getenv("SVMP_SIMD_BATCH");
            const bool enabled = !policy || std::string_view(policy) != "0";
            expectTwoLaneArithmetic(directory, simd && enabled && jit::hardwareProfile().simdDoubles() == 2);
            std::filesystem::remove_all(directory);
        }
    }
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
    const auto form = inner(grad(u), grad(v)).dx();
    // Only the unit tetrahedron touches node zero and |grad N0|² = 3.
    expectBilinearBatches(mesh, dof_map, dof_map, space, space, form, 0.5, true);
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

    expectCompiledResidualMatchesInterpreter(mesh, dof_map, space, residual, U, makeJITOptions(true), 1e-14);
}

TEST(JITSIMDBatchParity, SingleCellUnderfilledBatch_Tetra4)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraDofMap();
    spaces::H1Space space(ElementType::Tetra4, /*order=*/1);

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto form = (inner(grad(u), grad(v)) + 0.5 * u * v).dx();
    // K00 = 1/2 and M00 = 1/60. A single cell leaves the second lane unused.
    expectBilinearBatches(mesh, dof_map, dof_map, space, space, form, 61.0 / 120);
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

    expectCompiledResidualMatchesInterpreter(mesh, dof_map, space, residual, U,
                                             makeJITOptions(true, false), 1e-14);
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
    expectBilinearBatches(mesh, row_map, col_map, vel_space, pressure_space, (p * div(v)).dx());
}

// ============================================================================
// Test 3: Stokes VMS velocity-velocity block on Triangle3 (2D)
//
// Tests ProductSpace (vector-valued basis functions) and field_solutions
// (pressure as a DiscreteField).  This is the most complex form tested.
//
// The nonlinear compiled kernel uses its scalar ABI. The interpreter provides
// an independent reference for the residual and the velocity tangent.
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

    const auto dt_step = FormExpr::timeStep();
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

    expectCompiledResidualMatchesInterpreter(mesh, dof_map_u, vel_space, residual, U,
                                             makeJITOptions(true), 1e-12, opts);
}

// ============================================================================
// Test 4: moving-frame data in a batch
//
// The batched kernel call packs its arguments from the first cell of the batch
// and re-points the per-cell data for the others. The moving-frame data (mesh
// velocity and its divergence, current and reference geometry) differ from cell
// to cell, so the batched assembly must match the cell-by-cell assembly and
// the interpreter.
// ============================================================================

TEST(JITSIMDBatchParity, BatchedCellsSeeTheirOwnMovingFrame_Triangle3)
{
    MovingFourTriangleMeshAccess mesh;
    auto base = std::make_shared<spaces::H1Space>(ElementType::Triangle3, /*order=*/1);
    spaces::ProductSpace space(base, /*components=*/2);
    auto dof_map = createFourTriangleVectorDofMap();
    constexpr FieldId kMeshMotionField = 17;

    const auto u = TrialFunction(space, "u");
    const auto v = TestFunction(space, "v");
    const auto F = currentJacobian() * inv(referenceJacobian());
    const auto residual =
        (inner(meshVelocity(), v) +
         div(meshVelocity()) * v.component(0) +
         currentJacobianDeterminant() * component(F, 0, 1) * v.component(1) +
         inner(u, v)).dx();

    const std::vector<Real> U = {0.10, -0.20, 0.05, 0.15, -0.08, 0.03, -0.12, 0.07, -0.04, 0.11};
    const std::array<assembly::FieldSolutionAccess, 1> fields = {{
        {.field = kMeshMotionField, .space = &space, .dof_map = &dof_map, .dof_offset = 0},
    }};

    auto assemble = [&](assembly::AssemblyKernel& kernel,
                        bool batched,
                        assembly::DenseMatrixView& J,
                        assembly::DenseVectorView& R) {
        assembly::AssemblyOptions options;
        options.use_batching = batched;
        assembly::StandardAssembler assembler(options);
        assembler.setDofMap(dof_map);
        assembler.setFieldSolutionAccess(fields);
        assembler.setMeshMotionFieldAccess(assembly::MeshMotionFieldAccess{
            .mesh_displacement = kMeshMotionField,
            .mesh_velocity = kMeshMotionField,
        });
        assembler.setCurrentSolution(U);
        J.zero();
        R.zero();
        const auto result = assembler.assembleBoth(mesh, space, space, kernel, J, R);
        EXPECT_EQ(result.elements_assembled, 4);
    };

    FormCompiler compiler;
    auto jit_fallback = std::make_shared<SymbolicNonlinearFormKernel>(compiler.compileResidual(residual),
                                                                      NonlinearKernelOutput::Both);
    forms::jit::JITKernelWrapper jit_kernel(jit_fallback, makeJITOptions(/*simd_batch=*/true));
    jit_kernel.resolveInlinableConstitutives();
    SymbolicNonlinearFormKernel interpreted(compiler.compileResidual(residual), NonlinearKernelOutput::Both);
    interpreted.resolveInlinableConstitutives();

    const GlobalIndex n = dof_map.getNumDofs();
    assembly::DenseMatrixView J_batched(n, n), J_cellwise(n, n), J_interpreted(n, n);
    assembly::DenseVectorView R_batched(n), R_cellwise(n), R_interpreted(n);
    assemble(jit_kernel, /*batched=*/true, J_batched, R_batched);
    expectCompiledExecution(jit_kernel, *jit_fallback);
    assemble(jit_kernel, /*batched=*/false, J_cellwise, R_cellwise);
    assemble(interpreted, /*batched=*/false, J_interpreted, R_interpreted);

    expectDenseNear(R_batched, R_cellwise, 1e-14, "batched vs cell-by-cell residual");
    expectDenseNear(J_batched, J_cellwise, 1e-14, "batched vs cell-by-cell tangent");
    expectDenseNear(R_batched, R_interpreted, 1e-13, "compiled vs interpreted residual");
    expectDenseNear(J_batched, J_interpreted, 1e-13, "compiled vs interpreted tangent");

    Real largest = 0.0;
    for (GlobalIndex i = 0; i < n; ++i) {
        largest = std::max(largest, std::abs(R_cellwise.getVectorEntry(i)));
    }
    EXPECT_GT(largest, 1e-3) << "the moving-frame residual is zero";
}

#endif // SVMP_FE_ENABLE_LLVM_JIT

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp
