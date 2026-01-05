/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Backends/PETSc/PetscLinearSolver.h"

#if defined(FE_HAS_PETSC)

#include "Backends/Interfaces/BlockMatrix.h"
#include "Backends/Interfaces/BlockVector.h"
#include "Backends/PETSc/PetscMatrix.h"
#include "Backends/PETSc/PetscVector.h"
#include "Core/FEException.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace backends {

PetscLinearSolver::PetscLinearSolver(const SolverOptions& options)
{
    setOptions(options);
}

PetscLinearSolver::~PetscLinearSolver()
{
    if (ksp_) {
        FE_PETSC_CALL(KSPDestroy(&ksp_));
    }
}

PetscLinearSolver::PetscLinearSolver(PetscLinearSolver&& other) noexcept
{
    *this = std::move(other);
}

PetscLinearSolver& PetscLinearSolver::operator=(PetscLinearSolver&& other) noexcept
{
    if (this == &other) {
        return *this;
    }
    if (ksp_) {
        KSPDestroy(&ksp_);
    }
    options_ = other.options_;
    ksp_ = other.ksp_;
    other.ksp_ = nullptr;
    return *this;
}

void PetscLinearSolver::setOptions(const SolverOptions& options)
{
    FE_THROW_IF(options.max_iter <= 0, InvalidArgumentException, "PetscLinearSolver: max_iter must be > 0");
    FE_THROW_IF(options.rel_tol < 0.0, InvalidArgumentException, "PetscLinearSolver: rel_tol must be >= 0");
    FE_THROW_IF(options.abs_tol < 0.0, InvalidArgumentException, "PetscLinearSolver: abs_tol must be >= 0");
    options_ = options;
}

void PetscLinearSolver::ensureKspCreated()
{
    if (ksp_) {
        return;
    }
    FE_PETSC_CALL(KSPCreate(PETSC_COMM_WORLD, &ksp_));
}

namespace {

[[nodiscard]] std::string reasonToString(KSPConvergedReason r)
{
    return std::to_string(static_cast<int>(r));
}

[[nodiscard]] std::string normalizedPetscPrefix(std::string_view prefix)
{
    if (!prefix.empty() && prefix.front() == '-') {
        prefix.remove_prefix(1);
    }
    return std::string(prefix);
}

KSPType toPetscKspType(SolverMethod method)
{
    switch (method) {
        case SolverMethod::Direct: return KSPPREONLY;
        case SolverMethod::CG: return KSPCG;
        case SolverMethod::BiCGSTAB: return KSPBCGS;
        case SolverMethod::GMRES: return KSPGMRES;
        case SolverMethod::FGMRES: return KSPFGMRES;
        case SolverMethod::BlockSchur: return KSPFGMRES;
        default: return KSPCG;
    }
}

PCType toPetscPcType(PreconditionerType pc)
{
    switch (pc) {
        case PreconditionerType::None: return PCNONE;
        case PreconditionerType::Diagonal: return PCJACOBI;
        case PreconditionerType::ILU: return PCILU;
        case PreconditionerType::AMG: return PCGAMG;
        case PreconditionerType::RowColumnScaling: return PCNONE;
        case PreconditionerType::FieldSplit: return PCFIELDSPLIT;
        default: return PCNONE;
    }
}

} // namespace

void PetscLinearSolver::applyBaseOptionsToKsp()
{
    FE_CHECK_NOT_NULL(ksp_, "PetscLinearSolver::ksp");

    FE_PETSC_CALL(KSPSetType(ksp_, toPetscKspType(options_.method)));
    FE_PETSC_CALL(KSPSetTolerances(ksp_,
                                   static_cast<PetscReal>(options_.rel_tol),
                                   static_cast<PetscReal>(options_.abs_tol),
                                   PETSC_DEFAULT,
                                   static_cast<PetscInt>(options_.max_iter)));
    FE_PETSC_CALL(KSPSetInitialGuessNonzero(ksp_, options_.use_initial_guess ? PETSC_TRUE : PETSC_FALSE));

    PC pc = nullptr;
    FE_PETSC_CALL(KSPGetPC(ksp_, &pc));
    FE_CHECK_NOT_NULL(pc, "PetscLinearSolver::pc");

    const bool wants_fieldsplit =
        (options_.preconditioner == PreconditionerType::FieldSplit) || (options_.method == SolverMethod::BlockSchur);

    if (wants_fieldsplit) {
        FE_THROW_IF(options_.method == SolverMethod::Direct, InvalidArgumentException,
                    "PetscLinearSolver: field-split preconditioning requires an iterative method");
        FE_PETSC_CALL(PCSetType(pc, PCFIELDSPLIT));

        PCCompositeType split_type = PC_COMPOSITE_ADDITIVE;
        if (options_.method == SolverMethod::BlockSchur) {
            split_type = PC_COMPOSITE_SCHUR;
        } else {
            switch (options_.fieldsplit.kind) {
                case FieldSplitKind::Additive: split_type = PC_COMPOSITE_ADDITIVE; break;
                case FieldSplitKind::Multiplicative: split_type = PC_COMPOSITE_MULTIPLICATIVE; break;
                case FieldSplitKind::Schur: split_type = PC_COMPOSITE_SCHUR; break;
            }
        }
        FE_PETSC_CALL(PCFieldSplitSetType(pc, split_type));
    } else if (options_.method == SolverMethod::Direct) {
        FE_PETSC_CALL(PCSetType(pc, PCLU));
    } else {
        FE_PETSC_CALL(PCSetType(pc, toPetscPcType(options_.preconditioner)));

        if (options_.preconditioner == PreconditionerType::RowColumnScaling) {
            // Best-effort analogue to FSILS row/column scaling: enable diagonal scaling.
            FE_PETSC_CALL(KSPSetDiagonalScale(ksp_, PETSC_TRUE));
        }
    }

    const auto prefix = normalizedPetscPrefix(options_.petsc_options_prefix);
    if (!prefix.empty()) {
        FE_PETSC_CALL(KSPSetOptionsPrefix(ksp_, prefix.c_str()));
    }

    for (const auto& [key_raw, val] : options_.passthrough) {
        if (key_raw.empty()) continue;
        std::string key = key_raw;
        if (key.front() != '-') {
            key = "-" + prefix + key;
        }
        FE_PETSC_CALL(PetscOptionsSetValue(nullptr, key.c_str(), val.empty() ? nullptr : val.c_str()));
    }
}

void PetscLinearSolver::applyKspFromOptions()
{
    FE_CHECK_NOT_NULL(ksp_, "PetscLinearSolver::ksp");
    FE_PETSC_CALL(KSPSetFromOptions(ksp_));
}

namespace {

[[nodiscard]] std::string defaultFieldName(std::size_t i)
{
    return "field" + std::to_string(i);
}

} // namespace

SolverReport PetscLinearSolver::solve(const GenericMatrix& A_in,
                                      GenericVector& x_in,
                                      const GenericVector& b_in)
{
    const auto* A = dynamic_cast<const PetscMatrix*>(&A_in);
    const auto* A_block = dynamic_cast<const BlockMatrix*>(&A_in);

    if (A) {
        auto* x = dynamic_cast<PetscVector*>(&x_in);
        const auto* b = dynamic_cast<const PetscVector*>(&b_in);

        FE_THROW_IF(!x || !b, InvalidArgumentException, "PetscLinearSolver::solve: backend mismatch");
        FE_THROW_IF(options_.preconditioner == PreconditionerType::FieldSplit || options_.method == SolverMethod::BlockSchur,
                    InvalidArgumentException,
                    "PetscLinearSolver::solve: field-split/BlockSchur requires a BlockMatrix/BlockVector system");
        FE_THROW_IF(A->numRows() != A->numCols(), NotImplementedException,
                    "PetscLinearSolver::solve: rectangular systems not implemented");
        FE_THROW_IF(b->size() != A->numRows() || x->size() != b->size(), InvalidArgumentException,
                    "PetscLinearSolver::solve: size mismatch");

        ensureKspCreated();
        FE_PETSC_CALL(KSPSetOperators(ksp_, A->petsc(), A->petsc()));
        applyBaseOptionsToKsp();
        applyKspFromOptions();

        if (!options_.use_initial_guess) {
            x->zero();
        }

        // Compute initial residual ||b - A x||.
        PetscReal r0 = 0.0;
        {
            Vec r = nullptr;
            FE_PETSC_CALL(VecDuplicate(b->petsc(), &r));
            FE_PETSC_CALL(MatMult(A->petsc(), x->petsc(), r));
            FE_PETSC_CALL(VecAYPX(r, static_cast<PetscScalar>(-1.0), b->petsc())); // r = b - Ax
            FE_PETSC_CALL(VecNorm(r, NORM_2, &r0));
            FE_PETSC_CALL(VecDestroy(&r));
        }

        FE_PETSC_CALL(KSPSolve(ksp_, b->petsc(), x->petsc()));
        x->invalidateLocalCache();

        SolverReport rep;
        rep.initial_residual_norm = static_cast<Real>(r0);

        PetscInt its = 0;
        FE_PETSC_CALL(KSPGetIterationNumber(ksp_, &its));
        rep.iterations = static_cast<int>(its);

        PetscReal rn = 0.0;
        FE_PETSC_CALL(KSPGetResidualNorm(ksp_, &rn));
        rep.final_residual_norm = static_cast<Real>(rn);

        const Real denom = std::max<Real>(rep.initial_residual_norm, 1e-30);
        rep.relative_residual = rep.final_residual_norm / denom;

        KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
        FE_PETSC_CALL(KSPGetConvergedReason(ksp_, &reason));
        rep.converged = (reason > 0);
        rep.message = "petsc: reason=" + reasonToString(reason);

        return rep;
    }

    if (A_block) {
        auto* x = dynamic_cast<BlockVector*>(&x_in);
        const auto* b = dynamic_cast<const BlockVector*>(&b_in);

        FE_THROW_IF(!x || !b, InvalidArgumentException, "PetscLinearSolver::solve: backend mismatch (block)");
        FE_THROW_IF(A_block->backendKind() != BackendKind::PETSc, InvalidArgumentException,
                    "PetscLinearSolver::solve: BlockMatrix backendKind must be PETSc");
        FE_THROW_IF(A_block->numRows() != A_block->numCols(), NotImplementedException,
                    "PetscLinearSolver::solve: rectangular BlockMatrix systems not implemented");
        FE_THROW_IF(b->size() != A_block->numRows() || x->size() != b->size(), InvalidArgumentException,
                    "PetscLinearSolver::solve: size mismatch");

        const auto m = A_block->numRowBlocks();
        const auto n = A_block->numColBlocks();
        FE_THROW_IF(x->numBlocks() != n || b->numBlocks() != m, InvalidArgumentException,
                    "PetscLinearSolver::solve: block vector layout mismatch");
        FE_THROW_IF(m == 0 || n == 0, InvalidArgumentException, "PetscLinearSolver::solve: empty BlockMatrix");

        std::vector<Mat> submats(m * n, nullptr);
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                const auto* blk = A_block->block(i, j);
                if (!blk) continue;
                const auto* pb = dynamic_cast<const PetscMatrix*>(blk);
                FE_THROW_IF(!pb, InvalidArgumentException, "PetscLinearSolver::solve: non-PETSc block in BlockMatrix");
                submats[i * n + j] = pb->petsc();
            }
        }

        std::vector<Vec> x_sub(n, nullptr);
        std::vector<Vec> b_sub(m, nullptr);
        for (std::size_t j = 0; j < n; ++j) {
            const auto& v = x->block(j);
            const auto* pv = dynamic_cast<const PetscVector*>(&v);
            FE_THROW_IF(!pv, InvalidArgumentException, "PetscLinearSolver::solve: non-PETSc block in x");
            x_sub[j] = pv->petsc();
        }
        for (std::size_t i = 0; i < m; ++i) {
            const auto& v = b->block(i);
            const auto* pv = dynamic_cast<const PetscVector*>(&v);
            FE_THROW_IF(!pv, InvalidArgumentException, "PetscLinearSolver::solve: non-PETSc block in b");
            b_sub[i] = pv->petsc();
        }

        Mat A_nest = nullptr;
        Vec x_nest = nullptr;
        Vec b_nest = nullptr;

        FE_PETSC_CALL(MatCreateNest(PETSC_COMM_WORLD,
                                    static_cast<PetscInt>(m),
                                    nullptr,
                                    static_cast<PetscInt>(n),
                                    nullptr,
                                    submats.data(),
                                    &A_nest));
        FE_PETSC_CALL(MatAssemblyBegin(A_nest, MAT_FINAL_ASSEMBLY));
        FE_PETSC_CALL(MatAssemblyEnd(A_nest, MAT_FINAL_ASSEMBLY));

        FE_PETSC_CALL(VecCreateNest(PETSC_COMM_WORLD, static_cast<PetscInt>(n), nullptr, x_sub.data(), &x_nest));
        FE_PETSC_CALL(VecCreateNest(PETSC_COMM_WORLD, static_cast<PetscInt>(m), nullptr, b_sub.data(), &b_nest));

        ensureKspCreated();
        FE_PETSC_CALL(KSPSetOperators(ksp_, A_nest, A_nest));
        applyBaseOptionsToKsp();

        const bool wants_fieldsplit =
            (options_.preconditioner == PreconditionerType::FieldSplit) || (options_.method == SolverMethod::BlockSchur);
        if (wants_fieldsplit) {
            FE_THROW_IF(m != n, InvalidArgumentException,
                        "PetscLinearSolver: field-split requires a square block structure");
            if (options_.method == SolverMethod::BlockSchur) {
                FE_THROW_IF(m != 2, InvalidArgumentException,
                            "PetscLinearSolver: BlockSchur currently requires a 2x2 BlockMatrix");
            }

            PC pc = nullptr;
            FE_PETSC_CALL(KSPGetPC(ksp_, &pc));
            FE_CHECK_NOT_NULL(pc, "PetscLinearSolver::pc");

            const auto& names = options_.fieldsplit.split_names;
            FE_THROW_IF(!names.empty() && names.size() != m, InvalidArgumentException,
                        "PetscLinearSolver: fieldsplit.split_names size mismatch");

            for (std::size_t i = 0; i < m; ++i) {
                const auto name = names.empty() ? defaultFieldName(i) : names[i];
                const PetscInt size_i = static_cast<PetscInt>(A_block->colBlockSize(i));
                const PetscInt off_i = static_cast<PetscInt>(A_block->colBlockOffset(i));
                IS is = nullptr;
                FE_PETSC_CALL(ISCreateStride(PETSC_COMM_WORLD, size_i, off_i, 1, &is));
                FE_PETSC_CALL(PCFieldSplitSetIS(pc, name.c_str(), is));
                FE_PETSC_CALL(ISDestroy(&is));
            }
        }

        applyKspFromOptions();

        if (!options_.use_initial_guess) {
            x->zero();
        }

        PetscReal r0 = 0.0;
        {
            Vec r = nullptr;
            FE_PETSC_CALL(VecDuplicate(b_nest, &r));
            FE_PETSC_CALL(MatMult(A_nest, x_nest, r));
            FE_PETSC_CALL(VecAYPX(r, static_cast<PetscScalar>(-1.0), b_nest)); // r = b - Ax
            FE_PETSC_CALL(VecNorm(r, NORM_2, &r0));
            FE_PETSC_CALL(VecDestroy(&r));
        }

        FE_PETSC_CALL(KSPSolve(ksp_, b_nest, x_nest));

        // Invalidate caches on solution blocks (KSPSolve writes into sub-Vec objects).
        for (std::size_t j = 0; j < x->numBlocks(); ++j) {
            if (auto* pv = dynamic_cast<PetscVector*>(&x->block(j))) {
                pv->invalidateLocalCache();
            }
        }

        SolverReport rep;
        rep.initial_residual_norm = static_cast<Real>(r0);

        PetscInt its = 0;
        FE_PETSC_CALL(KSPGetIterationNumber(ksp_, &its));
        rep.iterations = static_cast<int>(its);

        PetscReal rn = 0.0;
        FE_PETSC_CALL(KSPGetResidualNorm(ksp_, &rn));
        rep.final_residual_norm = static_cast<Real>(rn);

        const Real denom = std::max<Real>(rep.initial_residual_norm, 1e-30);
        rep.relative_residual = rep.final_residual_norm / denom;

        KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
        FE_PETSC_CALL(KSPGetConvergedReason(ksp_, &reason));
        rep.converged = (reason > 0);
        rep.message = "petsc(nest): reason=" + reasonToString(reason);

        FE_PETSC_CALL(VecDestroy(&x_nest));
        FE_PETSC_CALL(VecDestroy(&b_nest));
        FE_PETSC_CALL(MatDestroy(&A_nest));

        return rep;
    }

    FE_THROW(InvalidArgumentException, "PetscLinearSolver::solve: backend mismatch");
}

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // FE_HAS_PETSC
