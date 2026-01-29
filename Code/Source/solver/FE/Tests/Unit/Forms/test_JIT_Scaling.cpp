/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/StandardAssembler.h"
#include "Dofs/DofMap.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Forms/Index.h"
#include "Forms/JIT/JITKernelWrapper.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Tests/Unit/Forms/JITTestHelpers.h"
#include "Tests/Unit/Forms/PerfTestHelpers.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

namespace {

class MultiTetraMeshAccess final : public assembly::IMeshAccess {
public:
    explicit MultiTetraMeshAccess(GlobalIndex num_cells)
        : num_cells_(num_cells)
    {
    }

    [[nodiscard]] GlobalIndex numCells() const override { return num_cells_; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return num_cells_; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override
    {
        return ElementType::Tetra4;
    }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override
    {
        nodes.clear();
        nodes.push_back(cell_id * 4 + 0);
        nodes.push_back(cell_id * 4 + 1);
        nodes.push_back(cell_id * 4 + 2);
        nodes.push_back(cell_id * 4 + 3);
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        const GlobalIndex cell_id = node_id / 4;
        const int local = static_cast<int>(node_id % 4);
        const Real x0 = static_cast<Real>(cell_id) * 2.0;

        switch (local) {
            case 0: return {x0, 0.0, 0.0};
            case 1: return {x0 + 1.0, 0.0, 0.0};
            case 2: return {x0, 1.0, 0.0};
            case 3: return {x0, 0.0, 1.0};
            default: return {x0, 0.0, 0.0};
        }
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        coords.clear();
        coords.push_back(getNodeCoordinates(cell_id * 4 + 0));
        coords.push_back(getNodeCoordinates(cell_id * 4 + 1));
        coords.push_back(getNodeCoordinates(cell_id * 4 + 2));
        coords.push_back(getNodeCoordinates(cell_id * 4 + 3));
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex /*face_id*/, GlobalIndex /*cell_id*/) const override
    {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override { return -1; }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex> getInteriorFaceCells(GlobalIndex /*face_id*/) const override
    {
        return {-1, -1};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        for (GlobalIndex c = 0; c < num_cells_; ++c) callback(c);
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override
    {
        for (GlobalIndex c = 0; c < num_cells_; ++c) callback(c);
    }

    void forEachBoundaryFace(int /*marker*/, std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

private:
    GlobalIndex num_cells_{0};
};

dofs::DofMap makeRepeatedCellDofMap(GlobalIndex num_cells, GlobalIndex dofs_per_element)
{
    dofs::DofMap dof_map(num_cells, dofs_per_element, static_cast<LocalIndex>(dofs_per_element));
    std::vector<GlobalIndex> dofs(static_cast<std::size_t>(dofs_per_element));
    for (GlobalIndex i = 0; i < dofs_per_element; ++i) dofs[static_cast<std::size_t>(i)] = i;

    for (GlobalIndex c = 0; c < num_cells; ++c) {
        dof_map.setCellDofs(c, dofs);
    }
    dof_map.setNumDofs(dofs_per_element);
    dof_map.setNumLocalDofs(dofs_per_element);
    dof_map.finalize();
    return dof_map;
}

} // namespace

TEST(JITScaling, MultiElementCellAssembly_JITAtLeast1p2xFasterThanInterpreter)
{
    requireLLVMJITOrSkip();
    if (!perfTestsEnabled()) {
        GTEST_SKIP() << "Set SVMP_FE_RUN_PERF_TESTS=1 to enable";
    }

    const GlobalIndex num_cells = static_cast<GlobalIndex>(std::max(1, detail::getenvInt("SVMP_FE_PERF_NUM_CELLS", 50)));
    const int iters = std::max(1, detail::getenvInt("SVMP_FE_PERF_ITERS_MULTI", 2));
    const int repeats = std::max(1, detail::getenvInt("SVMP_FE_PERF_REPEATS", 1));
    const double min_speedup = detail::getenvDouble("SVMP_FE_PERF_MIN_SPEEDUP_MULTI", 1.2);

    MultiTetraMeshAccess mesh(num_cells);
    auto base = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);
    spaces::ProductSpace space(base, 3);
    auto dof_map = makeRepeatedCellDofMap(num_cells, static_cast<GlobalIndex>(space.dofs_per_element()));

    SymbolicOptions sym_opts;
    sym_opts.jit.enable = true;
    FormCompiler compiler(sym_opts);

    const auto u = FormExpr::trialFunction(space, "u");
    const auto v = FormExpr::testFunction(space, "v");
    const Index i("i");
    const Index j("j");
    const auto form = (grad(u)(i, j) * grad(v)(i, j)).dx();

    auto ir_interp = compiler.compileBilinear(form);
    auto ir_jit = compiler.compileBilinear(form);

    auto interp_kernel = std::make_shared<FormKernel>(std::move(ir_interp));
    auto jit_fallback = std::make_shared<FormKernel>(std::move(ir_jit));

    forms::JITOptions jit_opts;
    jit_opts.enable = true;
    jit_opts.optimization_level = 2;
    jit_opts.vectorize = true;
    jit_opts.cache_kernels = true;
    jit_opts.specialization.enable = false;

    forms::jit::JITKernelWrapper jit_kernel(jit_fallback, jit_opts);

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseMatrixView mat(static_cast<GlobalIndex>(dof_map.getNumDofs()));
    mat.zero();

    // Warmup + JIT compile.
    (void)assembler.assembleMatrix(mesh, space, space, *interp_kernel, mat);
    (void)assembler.assembleMatrix(mesh, space, space, jit_kernel, mat);

    const auto run_interp = [&]() {
        for (int k = 0; k < iters; ++k) {
            mat.zero();
            (void)assembler.assembleMatrix(mesh, space, space, *interp_kernel, mat);
        }
    };
    const auto run_jit = [&]() {
        for (int k = 0; k < iters; ++k) {
            mat.zero();
            (void)assembler.assembleMatrix(mesh, space, space, jit_kernel, mat);
        }
    };

    const double sec_interp = detail::bestOfSeconds(repeats, run_interp);
    const double sec_jit = detail::bestOfSeconds(repeats, run_jit);

    const double speedup = (sec_jit > 0.0) ? (sec_interp / sec_jit) : 0.0;
    std::cerr << "JITScaling.MultiElementCellAssembly: cells=" << num_cells
              << " iters=" << iters
              << " repeats=" << repeats
              << " interp_ms/iter=" << (1e3 * sec_interp / static_cast<double>(iters))
              << " jit_ms/iter=" << (1e3 * sec_jit / static_cast<double>(iters))
              << " speedup=" << speedup << "x (min=" << min_speedup << "x)\n";

    EXPECT_GE(speedup, min_speedup)
        << "JIT speedup fell below threshold. interp=" << sec_interp << "s jit=" << sec_jit << "s";
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp
