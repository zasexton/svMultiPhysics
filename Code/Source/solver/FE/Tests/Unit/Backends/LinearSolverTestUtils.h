/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_TESTS_UNIT_BACKENDS_LINEAR_SOLVER_TEST_UTILS_H
#define SVMP_FE_TESTS_UNIT_BACKENDS_LINEAR_SOLVER_TEST_UTILS_H

#include "Backends/Interfaces/BackendFactory.h"
#include "Backends/Interfaces/BackendKind.h"
#include "Backends/Interfaces/GenericMatrix.h"
#include "Backends/Interfaces/GenericVector.h"
#include "Core/FEException.h"
#include "Core/Types.h"
#include "Sparsity/SparsityPattern.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace svmp::FE::backends::testutils {

[[nodiscard]] inline bool envFlagEnabled(std::string_view name)
{
    const char* v = std::getenv(std::string(name).c_str());
    if (!v) return false;
    return std::string_view(v) == "1" || std::string_view(v) == "ON" || std::string_view(v) == "TRUE";
}

[[nodiscard]] inline std::string backendName(BackendKind kind)
{
    return std::string(backendKindToString(kind));
}

[[nodiscard]] inline std::unique_ptr<BackendFactory> tryCreateFactory(BackendKind kind, int dof_per_node = 1)
{
    BackendFactory::CreateOptions opts;
    opts.dof_per_node = dof_per_node;
    try {
        return BackendFactory::create(kind, opts);
    } catch (...) {
        return nullptr;
    }
}

[[nodiscard]] inline std::vector<BackendKind> availableSerialBackends()
{
    std::vector<BackendKind> kinds;

    // FSILS is always compiled with FE_ENABLE_ASSEMBLY.
    kinds.push_back(BackendKind::FSILS);

#if defined(FE_HAS_EIGEN) && FE_HAS_EIGEN
    kinds.push_back(BackendKind::Eigen);
#endif

#if defined(FE_HAS_PETSC) && FE_HAS_PETSC
    kinds.push_back(BackendKind::PETSc);
#endif

    // Trilinos tests use a dedicated executable with Tpetra::ScopeGuard.
    return kinds;
}

[[nodiscard]] inline sparsity::SparsityPattern makeDensePattern(GlobalIndex n)
{
    sparsity::SparsityPattern p(n, n);
    for (GlobalIndex r = 0; r < n; ++r) {
        for (GlobalIndex c = 0; c < n; ++c) {
            p.addEntry(r, c);
        }
    }
    p.finalize();
    return p;
}

[[nodiscard]] inline sparsity::SparsityPattern makeDiagonalPattern(GlobalIndex n)
{
    sparsity::SparsityPattern p(n, n);
    for (GlobalIndex i = 0; i < n; ++i) {
        p.addEntry(i, i);
    }
    p.finalize();
    return p;
}

[[nodiscard]] inline sparsity::SparsityPattern makeTridiagPattern(GlobalIndex n)
{
    sparsity::SparsityPattern p(n, n);
    for (GlobalIndex i = 0; i < n; ++i) {
        p.addEntry(i, i);
        if (i > 0) p.addEntry(i, i - 1);
        if (i + 1 < n) p.addEntry(i, i + 1);
    }
    p.finalize();
    return p;
}

[[nodiscard]] inline sparsity::SparsityPattern makeGrid5ptPattern(GlobalIndex nx, GlobalIndex ny)
{
    const GlobalIndex n = nx * ny;
    sparsity::SparsityPattern p(n, n);

    auto idx = [nx](GlobalIndex i, GlobalIndex j) { return j * nx + i; };

    for (GlobalIndex j = 0; j < ny; ++j) {
        for (GlobalIndex i = 0; i < nx; ++i) {
            const GlobalIndex row = idx(i, j);
            p.addEntry(row, row);
            if (i > 0) p.addEntry(row, idx(i - 1, j));
            if (i + 1 < nx) p.addEntry(row, idx(i + 1, j));
            if (j > 0) p.addEntry(row, idx(i, j - 1));
            if (j + 1 < ny) p.addEntry(row, idx(i, j + 1));
        }
    }
    p.finalize();
    return p;
}

[[nodiscard]] inline sparsity::SparsityPattern replicateScalarPatternPerComponent(const sparsity::SparsityPattern& scalar,
                                                                                  int dof)
{
    FE_THROW_IF(!scalar.isFinalized(), InvalidArgumentException, "replicateScalarPatternPerComponent: pattern not finalized");
    FE_THROW_IF(dof <= 0, InvalidArgumentException, "replicateScalarPatternPerComponent: dof must be > 0");

    const GlobalIndex n_scalar = scalar.numRows();
    sparsity::SparsityPattern p(n_scalar * dof, n_scalar * dof);

    const auto row_ptr = scalar.getRowPtr();
    const auto col_idx = scalar.getColIndices();

    for (GlobalIndex r = 0; r < n_scalar; ++r) {
        const GlobalIndex start = row_ptr[static_cast<std::size_t>(r)];
        const GlobalIndex end = row_ptr[static_cast<std::size_t>(r + 1)];
        for (int c = 0; c < dof; ++c) {
            const GlobalIndex rr = r * dof + c;
            for (GlobalIndex k = start; k < end; ++k) {
                const GlobalIndex cc = col_idx[static_cast<std::size_t>(k)] * dof + c;
                p.addEntry(rr, cc);
            }
        }
    }

    p.finalize();
    return p;
}

inline void assembleVector(GenericVector& v, std::span<const Real> values)
{
    auto view = v.createAssemblyView();
    view->beginAssemblyPhase();
    std::vector<GlobalIndex> dofs(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        dofs[i] = static_cast<GlobalIndex>(i);
    }
    view->addVectorEntries(dofs, values, assembly::AddMode::Insert);
    view->finalizeAssembly();
}

[[nodiscard]] inline bool allFinite(std::span<const Real> xs)
{
    for (const auto v : xs) {
        if (!std::isfinite(v)) return false;
    }
    return true;
}

[[nodiscard]] inline Real computeResidualNorm(const BackendFactory& factory,
                                              const GenericMatrix& A,
                                              const GenericVector& x,
                                              const GenericVector& b)
{
    auto Ax = factory.createVector(b.size());
    Ax->zero();
    A.mult(x, *Ax);

    auto r = factory.createVector(b.size());
    r->zero();

    auto rs = r->localSpan();
    const auto bs = b.localSpan();
    const auto axs = Ax->localSpan();
    FE_THROW_IF(rs.size() != bs.size() || rs.size() != axs.size(), FEException,
                "computeResidualNorm: local span size mismatch");
    for (std::size_t i = 0; i < rs.size(); ++i) {
        rs[i] = bs[i] - axs[i];
    }
    return r->norm();
}

[[nodiscard]] inline Real computeRelativeResidual(const BackendFactory& factory,
                                                  const GenericMatrix& A,
                                                  const GenericVector& x,
                                                  const GenericVector& b)
{
    const Real rn = computeResidualNorm(factory, A, x, b);
    const Real bn = b.norm();
    return rn / std::max<Real>(bn, 1e-30);
}

} // namespace svmp::FE::backends::testutils

#endif // SVMP_FE_TESTS_UNIT_BACKENDS_LINEAR_SOLVER_TEST_UTILS_H
