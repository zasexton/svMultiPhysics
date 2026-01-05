#ifndef SVMP_FE_TESTS_UNIT_TIMESTEPPING_TIMESTEPPING_TEST_HELPERS_H
#define SVMP_FE_TESTS_UNIT_TIMESTEPPING_TIMESTEPPING_TEST_HELPERS_H

#include <gtest/gtest.h>

#include "Backends/Interfaces/BackendFactory.h"
#include "Backends/Interfaces/BackendKind.h"
#include "Backends/Interfaces/GenericVector.h"
#include "Backends/Interfaces/LinearSolver.h"
#include "Backends/Utils/BackendOptions.h"

#include "Core/Types.h"
#include "Dofs/DofHandler.h"

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

namespace svmp::FE::timestepping::test {

[[nodiscard]] inline svmp::FE::dofs::MeshTopologyInfo singleTetraTopology()
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

inline void setVectorByDof(svmp::FE::backends::GenericVector& vec, const std::vector<Real>& values)
{
    ASSERT_EQ(static_cast<std::size_t>(vec.size()), values.size());
    auto view = vec.createAssemblyView();
    ASSERT_NE(view.get(), nullptr);
    view->beginAssemblyPhase();
    for (GlobalIndex i = 0; i < vec.size(); ++i) {
        view->addVectorEntry(i, values[static_cast<std::size_t>(i)], svmp::FE::assembly::AddMode::Insert);
    }
    view->finalizeAssembly();
}

[[nodiscard]] inline std::vector<Real> getVectorByDof(svmp::FE::backends::GenericVector& vec)
{
    std::vector<Real> out(static_cast<std::size_t>(vec.size()), 0.0);
    auto view = vec.createAssemblyView();
    EXPECT_NE(view.get(), nullptr);
    for (GlobalIndex i = 0; i < vec.size(); ++i) {
        out[static_cast<std::size_t>(i)] = view->getVectorEntry(i);
    }
    return out;
}

[[nodiscard]] inline double relativeL2Error(const std::vector<Real>& approx, const std::vector<Real>& exact)
{
    EXPECT_EQ(approx.size(), exact.size());
    double err2 = 0.0;
    double ref2 = 0.0;
    for (std::size_t i = 0; i < approx.size(); ++i) {
        const double diff = static_cast<double>(approx[i] - exact[i]);
        err2 += diff * diff;
        ref2 += static_cast<double>(exact[i]) * static_cast<double>(exact[i]);
    }
    const double denom = (ref2 > 0.0) ? std::sqrt(ref2) : 1.0;
    return std::sqrt(err2) / denom;
}

[[nodiscard]] inline std::unique_ptr<svmp::FE::backends::BackendFactory> createTestFactory()
{
#if defined(FE_HAS_EIGEN) && FE_HAS_EIGEN
    return svmp::FE::backends::BackendFactory::create(svmp::FE::backends::BackendKind::Eigen);
#else
    return nullptr;
#endif
}

[[nodiscard]] inline svmp::FE::backends::SolverOptions directSolve()
{
    svmp::FE::backends::SolverOptions opts;
    opts.method = svmp::FE::backends::SolverMethod::Direct;
    opts.preconditioner = svmp::FE::backends::PreconditionerType::None;
    opts.rel_tol = 1e-14;
    opts.abs_tol = 1e-14;
    opts.max_iter = 1;
    return opts;
}

} // namespace svmp::FE::timestepping::test

#endif // SVMP_FE_TESTS_UNIT_TIMESTEPPING_TIMESTEPPING_TEST_HELPERS_H

