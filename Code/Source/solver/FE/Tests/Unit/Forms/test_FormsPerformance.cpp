/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_FormsPerformance.cpp
 * @brief Micro-benchmarks for FE/Forms (disabled by default)
 *
 * These are intentionally lightweight and informational; enable with
 * `--gtest_also_run_disabled_tests`.
 */

#include <gtest/gtest.h>

#include "Assembly/StandardAssembler.h"
#include "Forms/Dual.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <chrono>
#include <cmath>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

namespace {

using Clock = std::chrono::steady_clock;

template <class Fn>
double timeSeconds(Fn&& fn)
{
    const auto t0 = Clock::now();
    fn();
    const auto t1 = Clock::now();
    return std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
}

dofs::DofMap makeContiguousSingleCellDofMap(GlobalIndex num_dofs)
{
    dofs::DofMap dof_map(1, num_dofs, static_cast<LocalIndex>(num_dofs));
    std::vector<GlobalIndex> cell_dofs(static_cast<std::size_t>(num_dofs));
    for (GlobalIndex i = 0; i < num_dofs; ++i) cell_dofs[static_cast<std::size_t>(i)] = i;
    dof_map.setCellDofs(0, cell_dofs);
    dof_map.setNumDofs(num_dofs);
    dof_map.setNumLocalDofs(num_dofs);
    dof_map.finalize();
    return dof_map;
}

} // namespace

TEST(FormsPerformance, DISABLED_CellAssemblyThroughput)
{
    SingleTetraMeshAccess mesh;
    FormCompiler compiler;

    auto run = [&](int order, int iters) {
        spaces::H1Space space(ElementType::Tetra4, order);
        auto dof_map = makeContiguousSingleCellDofMap(static_cast<GlobalIndex>(space.dofs_per_element()));
        const auto u = FormExpr::trialFunction(space, "u");
        const auto v = FormExpr::testFunction(space, "v");

        auto ir = compiler.compileBilinear(inner(grad(u), grad(v)).dx());
        FormKernel kernel(std::move(ir));

        assembly::StandardAssembler assembler;
        assembler.setDofMap(dof_map);

        assembly::DenseMatrixView mat(static_cast<GlobalIndex>(dof_map.getNumDofs()));
        mat.zero();

        (void)assembler.assembleMatrix(mesh, space, space, kernel, mat); // warmup
        const double sec = timeSeconds([&]() {
            for (int i = 0; i < iters; ++i) {
                mat.zero();
                (void)assembler.assembleMatrix(mesh, space, space, kernel, mat);
            }
        });
        const double elems_per_sec = (sec > 0.0) ? (static_cast<double>(iters) / sec) : 0.0;
        return elems_per_sec;
    };

    const double p1 = run(/*order=*/1, /*iters=*/2000);
    const double p2 = run(/*order=*/2, /*iters=*/1000);

    SUCCEED() << "P1 elem/s=" << p1 << ", P2 elem/s=" << p2;
}

TEST(FormsPerformance, DISABLED_NonlinearAssemblyThroughput)
{
    SingleTetraMeshAccess mesh;
    FormCompiler compiler;

    spaces::H1Space space(ElementType::Tetra4, 1);
    auto dof_map = createSingleTetraDofMap();

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    auto ir = compiler.compileResidual((u * u * v).dx());
    NonlinearFormKernel kernel(std::move(ir), ADMode::Forward, NonlinearKernelOutput::Both);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setCurrentSolution(std::vector<Real>{0.1, -0.2, 0.3, -0.1});

    assembly::DenseMatrixView J(4);
    assembly::DenseVectorView R(4);
    J.zero();
    R.zero();

    (void)assembler.assembleBoth(mesh, space, space, kernel, J, R); // warmup

    constexpr int kIters = 800;
    const double sec = timeSeconds([&]() {
        for (int i = 0; i < kIters; ++i) {
            J.zero();
            R.zero();
            (void)assembler.assembleBoth(mesh, space, space, kernel, J, R);
        }
    });
    const double elems_per_sec = (sec > 0.0) ? (static_cast<double>(kIters) / sec) : 0.0;
    SUCCEED() << "Residual+Jacobian elem/s=" << elems_per_sec;
}

TEST(FormsPerformance, DISABLED_ADOverheadRatio)
{
    SingleTetraMeshAccess mesh;
    FormCompiler compiler;

    spaces::H1Space space(ElementType::Tetra4, 1);
    auto dof_map = createSingleTetraDofMap();

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto residual = (u * u * v).dx();

    auto ir_a = compiler.compileResidual(residual);
    auto ir_b = compiler.compileResidual(residual);

    NonlinearFormKernel kernel_vec(std::move(ir_a), ADMode::Forward, NonlinearKernelOutput::VectorOnly);
    NonlinearFormKernel kernel_both(std::move(ir_b), ADMode::Forward, NonlinearKernelOutput::Both);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setCurrentSolution(std::vector<Real>{0.1, -0.2, 0.3, -0.1});

    assembly::DenseVectorView R(4);
    assembly::DenseMatrixView J(4);
    R.zero();
    J.zero();

    constexpr int kIters = 1200;
    (void)assembler.assembleVector(mesh, space, kernel_vec, R); // warmup
    const double sec_vec = timeSeconds([&]() {
        for (int i = 0; i < kIters; ++i) {
            R.zero();
            (void)assembler.assembleVector(mesh, space, kernel_vec, R);
        }
    });

    (void)assembler.assembleBoth(mesh, space, space, kernel_both, J, R); // warmup
    const double sec_both = timeSeconds([&]() {
        for (int i = 0; i < kIters; ++i) {
            J.zero();
            R.zero();
            (void)assembler.assembleBoth(mesh, space, space, kernel_both, J, R);
        }
    });

    const double ratio = (sec_vec > 0.0) ? (sec_both / sec_vec) : 0.0;
    SUCCEED() << "Dual/Real-ish overhead ratio (both/vectorOnly) = " << ratio;
}

TEST(FormsPerformance, DISABLED_FormKernelQuadraturePointEvaluation)
{
    SingleTetraMeshAccess mesh;
    FormCompiler compiler;

    spaces::H1Space space(ElementType::Tetra4, 1);
    auto dof_map = createSingleTetraDofMap();

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const auto beta = FormExpr::coefficient("beta", [](Real x, Real y, Real z) {
        return std::array<Real, 3>{x + 1.0, y + 2.0, z + 3.0};
    });

    const auto form =
        (inner(grad(u), grad(v)) +
         inner(beta, grad(v)) +
         (sqrt(u * u + FormExpr::constant(1.0)) + exp(u) + log(u * u + FormExpr::constant(1.0))) * v)
            .dx();

    auto ir = compiler.compileResidual(form);
    NonlinearFormKernel kernel(std::move(ir), ADMode::Forward, NonlinearKernelOutput::VectorOnly);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);
    assembler.setCurrentSolution(std::vector<Real>{0.1, -0.2, 0.3, -0.1});

    assembly::DenseVectorView R(4);
    R.zero();
    (void)assembler.assembleVector(mesh, space, kernel, R); // warmup

    constexpr int kIters = 1500;
    const double sec = timeSeconds([&]() {
        for (int i = 0; i < kIters; ++i) {
            R.zero();
            (void)assembler.assembleVector(mesh, space, kernel, R);
        }
    });
    SUCCEED() << "Vector-only nonlinear eval time/call = " << (sec / static_cast<double>(kIters)) << " s";
}

TEST(FormsPerformance, DISABLED_FormCompilerLatency)
{
    FormCompiler compiler;
    spaces::H1Space space(ElementType::Tetra4, 1);
    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");

    const auto simple = (u * v).dx();
    const auto medium = inner(grad(u), grad(v)).dx();
    const auto heavy = (pow(u, FormExpr::constant(3.0)) * v + inner(grad(u), grad(v))).dx();

    constexpr int kIters = 500;
    const double t_simple = timeSeconds([&]() {
        for (int i = 0; i < kIters; ++i) {
            (void)compiler.compileBilinear(simple);
        }
    });
    const double t_medium = timeSeconds([&]() {
        for (int i = 0; i < kIters; ++i) {
            (void)compiler.compileBilinear(medium);
        }
    });
    const double t_heavy = timeSeconds([&]() {
        for (int i = 0; i < kIters; ++i) {
            (void)compiler.compileBilinear(heavy);
        }
    });

    const double ms_simple = 1e3 * t_simple / static_cast<double>(kIters);
    const double ms_medium = 1e3 * t_medium / static_cast<double>(kIters);
    const double ms_heavy = 1e3 * t_heavy / static_cast<double>(kIters);
    SUCCEED() << "compileBilinear avg ms: simple=" << ms_simple << ", medium=" << ms_medium
              << ", heavy=" << ms_heavy;
}

TEST(FormsPerformance, DISABLED_DualWorkspaceAllocationThroughput)
{
    DualWorkspace ws;
    ws.reset(32);

    constexpr std::size_t kAllocs = 2'000'000;
    (void)ws.alloc(); // warmup
    const double sec = timeSeconds([&]() {
        for (std::size_t i = 0; i < kAllocs; ++i) {
            auto span = ws.alloc();
            (void)span;
        }
    });
    const double allocs_per_sec = (sec > 0.0) ? (static_cast<double>(kAllocs) / sec) : 0.0;
    SUCCEED() << "allocs/s=" << allocs_per_sec;
}

TEST(FormsPerformance, DISABLED_LargeDofScaling)
{
    SingleTetraMeshAccess mesh;
    FormCompiler compiler;

    auto bench_order = [&](int order, int iters) {
        spaces::H1Space space(ElementType::Tetra4, order);
        auto dof_map = makeContiguousSingleCellDofMap(static_cast<GlobalIndex>(space.dofs_per_element()));
        const auto u = FormExpr::trialFunction(space, "u");
        const auto v = FormExpr::testFunction(space, "v");
        auto ir = compiler.compileBilinear(inner(grad(u), grad(v)).dx());
        FormKernel kernel(std::move(ir));

        assembly::StandardAssembler assembler;
        assembler.setDofMap(dof_map);

        assembly::DenseMatrixView mat(static_cast<GlobalIndex>(dof_map.getNumDofs()));
        mat.zero();
        (void)assembler.assembleMatrix(mesh, space, space, kernel, mat); // warmup
        const double sec = timeSeconds([&]() {
            for (int i = 0; i < iters; ++i) {
                mat.zero();
                (void)assembler.assembleMatrix(mesh, space, space, kernel, mat);
            }
        });
        return sec / static_cast<double>(iters);
    };

    const double t_p1 = bench_order(/*order=*/1, /*iters=*/2000);
    const double t_p2 = bench_order(/*order=*/2, /*iters=*/1000);
    SUCCEED() << "time/call: P1=" << t_p1 << " s, P2=" << t_p2 << " s";
}

TEST(FormsPerformance, DISABLED_BoundaryAssemblyThroughput)
{
    SingleTetraOneBoundaryFaceMeshAccess mesh(/*boundary_marker=*/2);
    FormCompiler compiler;
    spaces::H1Space space(ElementType::Tetra4, 1);
    auto dof_map = createSingleTetraDofMap();

    const auto v = FormExpr::testFunction(space, "v");
    auto ir = compiler.compileLinear(v.ds(2));
    FormKernel kernel(std::move(ir));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseVectorView vec(4);
    vec.zero();
    (void)assembler.assembleBoundaryFaces(mesh, 2, space, kernel, nullptr, &vec); // warmup

    constexpr int kIters = 3000;
    const double sec = timeSeconds([&]() {
        for (int i = 0; i < kIters; ++i) {
            vec.zero();
            (void)assembler.assembleBoundaryFaces(mesh, 2, space, kernel, nullptr, &vec);
        }
    });
    const double faces_per_sec = (sec > 0.0) ? (static_cast<double>(kIters) / sec) : 0.0;
    SUCCEED() << "faces/s=" << faces_per_sec;
}

TEST(FormsPerformance, DISABLED_DGInteriorFaceAssemblyThroughput)
{
    TwoTetraSharedFaceMeshAccess mesh;
    FormCompiler compiler;
    spaces::H1Space space(ElementType::Tetra4, 1);
    auto dof_map = createTwoTetraDG_DofMap();

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    auto ir = compiler.compileBilinear((FormExpr::constant(1.0) * inner(jump(u), jump(v))).dS());
    FormKernel kernel(std::move(ir));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView mat(8);
    mat.zero();
    (void)assembler.assembleInteriorFaces(mesh, space, space, kernel, mat, nullptr); // warmup

    constexpr int kIters = 2500;
    const double sec = timeSeconds([&]() {
        for (int i = 0; i < kIters; ++i) {
            mat.zero();
            (void)assembler.assembleInteriorFaces(mesh, space, space, kernel, mat, nullptr);
        }
    });
    const double faces_per_sec = (sec > 0.0) ? (static_cast<double>(kIters) / sec) : 0.0;
    SUCCEED() << "faces/s=" << faces_per_sec;
}

TEST(FormsPerformance, DISABLED_MemoryAllocationProfile)
{
    SingleTetraMeshAccess mesh;
    FormCompiler compiler;
    spaces::H1Space space(ElementType::Tetra4, 1);
    auto dof_map = createSingleTetraDofMap();

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    auto ir = compiler.compileBilinear(inner(grad(u), grad(v)).dx());
    FormKernel kernel(std::move(ir));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView mat(4);
    mat.zero();

    constexpr int kIters = 5000;
    (void)assembler.assembleMatrix(mesh, space, space, kernel, mat); // warmup
    const double sec = timeSeconds([&]() {
        for (int i = 0; i < kIters; ++i) {
            mat.zero();
            (void)assembler.assembleMatrix(mesh, space, space, kernel, mat);
        }
    });
    SUCCEED() << "Ran " << kIters << " assemblies in " << sec << " s; use a dedicated profiler for allocations.";
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp
