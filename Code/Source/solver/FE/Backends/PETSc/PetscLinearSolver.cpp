/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Backends/PETSc/PetscLinearSolver.h"
#include "Core/EnvironmentNames.h"

#if defined(FE_HAS_PETSC)

#include "Backends/Interfaces/BlockMatrix.h"
#include "Backends/Interfaces/BlockVector.h"
#include "Backends/PETSc/PetscMatrix.h"
#include "Backends/PETSc/PetscVector.h"
#include "Core/FEException.h"
#include "Core/Logger.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <sstream>
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
    nullspace_basis_ = std::move(other.nullspace_basis_);
    return *this;
}

void PetscLinearSolver::setOptions(const SolverOptions& options)
{
    FE_THROW_IF(options.max_iter <= 0, InvalidArgumentException, "PetscLinearSolver: max_iter must be > 0");
    FE_THROW_IF(options.rel_tol < 0.0, InvalidArgumentException, "PetscLinearSolver: rel_tol must be >= 0");
    FE_THROW_IF(options.abs_tol < 0.0, InvalidArgumentException, "PetscLinearSolver: abs_tol must be >= 0");
    options_ = normalizeSolverOptionsForBackend(options, BackendKind::PETSc);
}

void PetscLinearSolver::ensureKspCreated()
{
    if (ksp_) {
        return;
    }
    FE_PETSC_CALL(KSPCreate(PETSC_COMM_WORLD, &ksp_));
}

namespace {

[[nodiscard]] bool oopTraceEnabled() noexcept
{
    static const bool enabled = [] {
        const char* env = environment::value("SVMP_OOP_SOLVER_TRACE");
        if (env == nullptr) {
            return false;
        }
        std::string v(env);
        std::transform(v.begin(), v.end(), v.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return !(v == "0" || v == "false" || v == "off" || v == "no");
    }();
    return enabled;
}

void traceLog(const std::string& msg)
{
    if (!oopTraceEnabled()) {
        return;
    }
    FE_LOG_INFO(msg);
}

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
        case SolverMethod::PGMRES: return KSPPGMRES;
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

void PetscLinearSolver::applyBaseOptionsToKsp(const SolverOptions& options)
{
    FE_CHECK_NOT_NULL(ksp_, "PetscLinearSolver::ksp");

    FE_PETSC_CALL(KSPSetType(ksp_, toPetscKspType(options.method)));
    FE_PETSC_CALL(KSPSetTolerances(ksp_,
                                   static_cast<PetscReal>(options.rel_tol),
                                   static_cast<PetscReal>(options.abs_tol),
                                   PETSC_DEFAULT,
                                   static_cast<PetscInt>(options.max_iter)));
    FE_PETSC_CALL(KSPSetInitialGuessNonzero(ksp_, options.use_initial_guess ? PETSC_TRUE : PETSC_FALSE));

    PC pc = nullptr;
    FE_PETSC_CALL(KSPGetPC(ksp_, &pc));
    FE_CHECK_NOT_NULL(pc, "PetscLinearSolver::pc");

    const bool wants_fieldsplit =
        (options.preconditioner == PreconditionerType::FieldSplit) ||
        (options.method == SolverMethod::BlockSchur);

    if (wants_fieldsplit) {
        FE_THROW_IF(options.method == SolverMethod::Direct, InvalidArgumentException,
                    "PetscLinearSolver: field-split preconditioning requires an iterative method");
        FE_PETSC_CALL(PCSetType(pc, PCFIELDSPLIT));

        PCCompositeType split_type = PC_COMPOSITE_ADDITIVE;
        if (options.method == SolverMethod::BlockSchur) {
            split_type = PC_COMPOSITE_SCHUR;
        } else {
            switch (options.fieldsplit.kind) {
                case FieldSplitKind::Auto: split_type = PC_COMPOSITE_ADDITIVE; break;
                case FieldSplitKind::Additive: split_type = PC_COMPOSITE_ADDITIVE; break;
                case FieldSplitKind::Multiplicative: split_type = PC_COMPOSITE_MULTIPLICATIVE; break;
                case FieldSplitKind::Schur: split_type = PC_COMPOSITE_SCHUR; break;
            }
        }
        FE_PETSC_CALL(PCFieldSplitSetType(pc, split_type));
    } else if (options.method == SolverMethod::Direct) {
        FE_PETSC_CALL(PCSetType(pc, PCLU));
    } else {
        FE_PETSC_CALL(PCSetType(pc, toPetscPcType(options.preconditioner)));

        if (options.preconditioner == PreconditionerType::RowColumnScaling) {
            // Best-effort analogue to FSILS row/column scaling: enable diagonal scaling.
            FE_PETSC_CALL(KSPSetDiagonalScale(ksp_, PETSC_TRUE));
        }
    }

    const auto prefix = normalizedPetscPrefix(options.petsc_options_prefix);
    if (!prefix.empty()) {
        FE_PETSC_CALL(KSPSetOptionsPrefix(ksp_, prefix.c_str()));
    }

    for (const auto& [key_raw, val] : options.passthrough) {
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

[[nodiscard]] std::vector<std::string> resolveFieldSplitNames(const SolverOptions& options,
                                                              const BlockMatrix& matrix)
{
    if (!options.fieldsplit.split_names.empty()) {
        return options.fieldsplit.split_names;
    }

    std::vector<std::string> names;
    names.reserve(matrix.numColBlocks());
    for (std::size_t i = 0; i < matrix.numColBlocks(); ++i) {
        std::string name = defaultFieldName(i);
        if (options.mixed_block_layout.has_value()) {
            if (const auto* block = options.mixed_block_layout->findBlockByExtent(
                    matrix.colBlockOffset(i), matrix.colBlockSize(i))) {
                name = block->name;
            }
        }
        names.push_back(std::move(name));
    }
    return names;
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

        const auto nullspace = configureNullspace(A->petsc());
        ensureKspCreated();
        FE_PETSC_CALL(KSPSetOperators(ksp_, A->petsc(), A->petsc()));
        applyBaseOptionsToKsp(options_);
        applyKspFromOptions();

        if (!options_.use_initial_guess) {
            x->zero();
        }

        if (oopTraceEnabled()) {
            std::ostringstream oss;
            oss << "PETScLinearSolver::solve: n=" << A->numRows()
                << " method=" << solverMethodToString(options_.method)
                << " pc=" << preconditionerToString(options_.preconditioner)
                << " rel_tol=" << options_.rel_tol
                << " abs_tol=" << options_.abs_tol
                << " max_iter=" << options_.max_iter
                << " use_initial_guess=" << (options_.use_initial_guess ? 1 : 0)
                << " prefix='" << normalizedPetscPrefix(options_.petsc_options_prefix) << "'";
            traceLog(oss.str());
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
        if (nullspace) FE_PETSC_CALL(MatNullSpaceRemove(nullspace, x->petsc()));
        x->invalidateLocalCache();
        x->markModified();

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

        if (oopTraceEnabled()) {
            std::ostringstream oss;
            oss << "PETScLinearSolver::solve: converged=" << (rep.converged ? 1 : 0)
                << " iters=" << rep.iterations
                << " r0=" << rep.initial_residual_norm
                << " rn=" << rep.final_residual_norm
                << " rel=" << rep.relative_residual
                << " msg='" << rep.message << "'";
            traceLog(oss.str());
        }

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

        const auto solve_options =
            normalizeSolverOptionsForBackend(options_, BackendKind::PETSc, /*block_operator_available=*/true);

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

        struct NestedSystem {
            Mat matrix{nullptr};
            Vec x{nullptr}, rhs{nullptr};
            ~NestedSystem()
            {
                VecDestroy(&x);
                VecDestroy(&rhs);
                MatDestroy(&matrix);
            }
        } nest;

        FE_PETSC_CALL(MatCreateNest(PETSC_COMM_WORLD,
                                    static_cast<PetscInt>(m),
                                    nullptr,
                                    static_cast<PetscInt>(n),
                                    nullptr,
                                    submats.data(),
                                    &nest.matrix));
        FE_PETSC_CALL(MatAssemblyBegin(nest.matrix, MAT_FINAL_ASSEMBLY));
        FE_PETSC_CALL(MatAssemblyEnd(nest.matrix, MAT_FINAL_ASSEMBLY));

        FE_PETSC_CALL(VecCreateNest(PETSC_COMM_WORLD, static_cast<PetscInt>(n), nullptr, x_sub.data(), &nest.x));
        FE_PETSC_CALL(VecCreateNest(PETSC_COMM_WORLD, static_cast<PetscInt>(m), nullptr, b_sub.data(), &nest.rhs));

        // Field coefficients are indexed globally within each block, even when
        // PETSc interleaves the locally owned blocks on each rank.
        FE_PETSC_CALL(MatNestSetVecType(nest.matrix, VECNEST));
        const auto nullspace = configureNullspace(nest.matrix);

        ensureKspCreated();
        FE_PETSC_CALL(KSPSetOperators(ksp_, nest.matrix, nest.matrix));
        applyBaseOptionsToKsp(solve_options);

        const bool wants_fieldsplit =
            (solve_options.preconditioner == PreconditionerType::FieldSplit) ||
            (solve_options.method == SolverMethod::BlockSchur);
        if (wants_fieldsplit) {
            FE_THROW_IF(m != n, InvalidArgumentException,
                        "PetscLinearSolver: field-split requires a square block structure");
            if (solve_options.method == SolverMethod::BlockSchur) {
                FE_THROW_IF(m != 2, InvalidArgumentException,
                            "PetscLinearSolver: BlockSchur currently requires a 2x2 BlockMatrix");
            }

            PC pc = nullptr;
            FE_PETSC_CALL(KSPGetPC(ksp_, &pc));
            FE_CHECK_NOT_NULL(pc, "PetscLinearSolver::pc");

            const auto names = resolveFieldSplitNames(solve_options, *A_block);
            FE_THROW_IF(names.size() != m, InvalidArgumentException,
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

        if (oopTraceEnabled()) {
            std::ostringstream oss;
            oss << "PETScLinearSolver::solve(block): blocks=" << m << "x" << n
                << " n_total=" << A_block->numRows()
                << " method=" << solverMethodToString(solve_options.method)
                << " pc=" << preconditionerToString(solve_options.preconditioner)
                << " split_kind=" << fieldSplitKindToString(solve_options.fieldsplit.kind)
                << " rel_tol=" << solve_options.rel_tol
                << " abs_tol=" << solve_options.abs_tol
                << " max_iter=" << solve_options.max_iter
                << " use_initial_guess=" << (solve_options.use_initial_guess ? 1 : 0)
                << " prefix='" << normalizedPetscPrefix(solve_options.petsc_options_prefix) << "'";
            traceLog(oss.str());
        }

        PetscReal r0 = 0.0;
        {
            Vec r = nullptr;
            FE_PETSC_CALL(VecDuplicate(nest.rhs, &r));
            FE_PETSC_CALL(MatMult(nest.matrix, nest.x, r));
            FE_PETSC_CALL(VecAYPX(r, static_cast<PetscScalar>(-1.0), nest.rhs)); // r = b - Ax
            FE_PETSC_CALL(VecNorm(r, NORM_2, &r0));
            FE_PETSC_CALL(VecDestroy(&r));
        }

        FE_PETSC_CALL(KSPSolve(ksp_, nest.rhs, nest.x));
        if (nullspace) FE_PETSC_CALL(MatNullSpaceRemove(nullspace, nest.x));

        // Invalidate caches on solution blocks (KSPSolve writes into sub-Vec objects).
        for (std::size_t j = 0; j < x->numBlocks(); ++j) {
            if (auto* pv = dynamic_cast<PetscVector*>(&x->block(j))) {
                pv->invalidateLocalCache();
                pv->markModified();
            }
        }
        x->markModified();

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

        if (oopTraceEnabled()) {
            std::ostringstream oss;
            oss << "PETScLinearSolver::solve(block): converged=" << (rep.converged ? 1 : 0)
                << " iters=" << rep.iterations
                << " r0=" << rep.initial_residual_norm
                << " rn=" << rep.final_residual_norm
                << " rel=" << rep.relative_residual
                << " msg='" << rep.message << "'";
            traceLog(oss.str());
        }

        return rep;
    }

    FE_THROW(InvalidArgumentException, "PetscLinearSolver::solve: backend mismatch");
}

void PetscLinearSolver::setNullspaceBasis(std::span<const std::vector<double>> basis)
{
    nullspace_basis_.emplace(basis.begin(), basis.end());
}

namespace {

void copyOwnedMode(Vec vec, std::span<const double> coefficients, double scale)
{
    PetscInt first, last;
    FE_PETSC_CALL(VecGetOwnershipRange(vec, &first, &last));
    PetscScalar* values = nullptr;
    FE_PETSC_CALL(VecGetArrayWrite(vec, &values));
    for (PetscInt i = first; i < last; ++i) {
        values[i - first] = static_cast<PetscScalar>(coefficients[i] / scale);
    }
    FE_PETSC_CALL(VecRestoreArrayWrite(vec, &values));
}

void copyMode(Vec vec, std::span<const double> coefficients, double scale)
{
    PetscBool nested;
    FE_PETSC_CALL(PetscObjectTypeCompare(reinterpret_cast<PetscObject>(vec), VECNEST, &nested));
    if (!nested) {
        copyOwnedMode(vec, coefficients, scale);
        return;
    }
    PetscInt count;
    Vec* blocks;
    FE_PETSC_CALL(VecNestGetSubVecs(vec, &count, &blocks));
    for (PetscInt i = 0; i < count; ++i) {
        PetscInt size;
        FE_PETSC_CALL(VecGetSize(blocks[i], &size));
        copyOwnedMode(blocks[i], coefficients.first(size), scale);
        coefficients = coefficients.subspan(size);
    }
}

struct NullspaceWork {
    explicit NullspaceWork(std::size_t count) : modes(count, nullptr) {}
    ~NullspaceWork()
    {
        MatNullSpaceDestroy(&space);
        for (auto& mode : modes) VecDestroy(&mode);
    }
    MatNullSpace space{nullptr};
    std::vector<Vec> modes;
};

} // namespace

MatNullSpace PetscLinearSolver::configureNullspace(Mat matrix) const
{
    const MPI_Comm comm = PetscObjectComm(reinterpret_cast<PetscObject>(matrix));
    // Obtain the maximum and minimum count in one reduction. -1 means the
    // caller owns the matrix's existing attachment, 0 explicitly clears it.
    const long long count = nullspace_basis_ ? static_cast<long long>(nullspace_basis_->size()) : -1;
    long long bounds[2]{count, -count};
    FE_PETSC_CALL(MPI_Allreduce(MPI_IN_PLACE, bounds, 2, MPI_LONG_LONG_INT, MPI_MAX, comm));
    FE_THROW_IF(bounds[0] != -bounds[1], InvalidArgumentException,
                "PetscLinearSolver: nullspace configuration and mode count must agree on every rank");
    if (count < 0) return nullptr;
    if (count == 0) {
        FE_PETSC_CALL(MatSetNullSpace(matrix, nullptr));
        return nullptr;
    }
    const auto& basis = *nullspace_basis_;
    PetscInt columns;
    FE_PETSC_CALL(MatGetSize(matrix, nullptr, &columns));
    int valid = count <= columns && count <= std::numeric_limits<int>::max();
    std::vector<double> scales(basis.size(), 0.);
    for (std::size_t i = 0; i < basis.size(); ++i) {
        if (basis[i].size() != static_cast<std::size_t>(columns)) valid = 0;
        for (double value : basis[i]) {
            if (!std::isfinite(value)) valid = 0;
            scales[i] = std::max(scales[i], std::abs(value));
        }
    }
    FE_PETSC_CALL(MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, comm));
    FE_THROW_IF(!valid, InvalidArgumentException,
                "PetscLinearSolver: nullspace mode counts must agree on all ranks and each finite mode must span the global domain");
    FE_PETSC_CALL(MPI_Allreduce(MPI_IN_PLACE, scales.data(), static_cast<int>(count), MPI_DOUBLE, MPI_MAX, comm));

    NullspaceWork work(basis.size());
    for (std::size_t i = 0; i < basis.size(); ++i) {
        FE_THROW_IF(scales[i] == 0., InvalidArgumentException, "PetscLinearSolver: zero nullspace mode");
        auto& mode = work.modes[i];
        FE_PETSC_CALL(MatCreateVecs(matrix, &mode, nullptr));
        copyMode(mode, basis[i], scales[i]);
        PetscReal initial_norm;
        FE_PETSC_CALL(VecNorm(mode, NORM_2, &initial_norm));
        // Two passes of modified Gram-Schmidt retain orthogonality for an
        // independent, nonorthogonal spanning set. All dots include every owner.
        for (int pass = 0; pass < 2; ++pass) {
            for (std::size_t j = 0; j < i; ++j) {
                PetscScalar dot;
                FE_PETSC_CALL(VecDot(mode, work.modes[j], &dot));
                FE_PETSC_CALL(VecAXPY(mode, -dot, work.modes[j]));
            }
        }
        PetscReal norm;
        FE_PETSC_CALL(VecNorm(mode, NORM_2, &norm));
        FE_THROW_IF(norm <= 64 * std::numeric_limits<PetscReal>::epsilon() * initial_norm,
                    InvalidArgumentException, "PetscLinearSolver: linearly dependent nullspace modes");
        FE_PETSC_CALL(VecScale(mode, static_cast<PetscScalar>(1. / norm)));
    }
    FE_PETSC_CALL(MatNullSpaceCreate(comm, PETSC_FALSE, static_cast<PetscInt>(count), work.modes.data(), &work.space));
    PetscBool is_null;
    FE_PETSC_CALL(MatNullSpaceTest(work.space, matrix, &is_null));
    FE_THROW_IF(!is_null, InvalidArgumentException,
                "PetscLinearSolver: supplied nullspace is not annihilated by the operator");
    FE_PETSC_CALL(MatSetNullSpace(matrix, work.space));
    return work.space; // The matrix now owns a reference.
}

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // FE_HAS_PETSC
