/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Backends/Eigen/EigenLinearSolver.h"

#include "Backends/Eigen/EigenMatrix.h"
#include "Backends/Eigen/EigenVector.h"
#include "Core/FEException.h"

#if defined(FE_HAS_EIGEN)
#include <iostream>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseLU>
#include <unsupported/Eigen/IterativeSolvers>
#endif

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace backends {

#if defined(FE_HAS_EIGEN)

EigenLinearSolver::EigenLinearSolver(const SolverOptions& options)
{
    setOptions(options);
}

void EigenLinearSolver::setOptions(const SolverOptions& options)
{
    FE_THROW_IF(options.max_iter <= 0, InvalidArgumentException, "EigenLinearSolver: max_iter must be > 0");
    FE_THROW_IF(options.rel_tol < 0.0, InvalidArgumentException, "EigenLinearSolver: rel_tol must be >= 0");
    FE_THROW_IF(options.abs_tol < 0.0, InvalidArgumentException, "EigenLinearSolver: abs_tol must be >= 0");
    options_ = normalizeSolverOptionsForBackend(options, BackendKind::Eigen);
}

namespace {

[[nodiscard]] bool check_convergence(const SolverReport& report, const SolverOptions& options) noexcept
{
    const bool has_abs = options.abs_tol > 0.0;
    const bool has_rel = options.rel_tol > 0.0;

    if (!has_abs && !has_rel) {
        return true;
    }

    const bool abs_ok = has_abs && (report.final_residual_norm <= options.abs_tol);
    const bool rel_ok = has_rel && (report.relative_residual <= options.rel_tol);
    return abs_ok || rel_ok;
}

[[nodiscard]] std::string computation_info_name(Eigen::ComputationInfo info)
{
    switch (info) {
        case Eigen::Success: return "success";
        case Eigen::NumericalIssue: return "numerical_issue";
        case Eigen::NoConvergence: return "no_convergence";
        case Eigen::InvalidInput: return "invalid_input";
    }
    return "unknown";
}

struct BlockRange {
    std::string name;
    GlobalIndex begin{0};
    GlobalIndex end{0};
};

[[nodiscard]] std::vector<BlockRange>
build_diagnostic_block_ranges(const SolverOptions& options, GlobalIndex n_rows)
{
    std::vector<BlockRange> ranges;
    if (n_rows <= 0) {
        return ranges;
    }

    if (options.mixed_block_layout.has_value()) {
        bool usable = true;
        for (const auto& block : options.mixed_block_layout->blocks) {
            if (block.size < 0 || block.offset < 0 ||
                block.offset + block.size > n_rows) {
                usable = false;
                break;
            }
        }
        if (usable) {
            for (const auto& block : options.mixed_block_layout->blocks) {
                ranges.push_back(BlockRange{block.name, block.offset, block.offset + block.size});
            }
            return ranges;
        }
    }

    if (options.block_layout.has_value()) {
        const auto total_components = options.block_layout->totalComponents();
        if (total_components > 0 &&
            (n_rows % static_cast<GlobalIndex>(total_components)) == 0) {
            const auto node_count = n_rows / static_cast<GlobalIndex>(total_components);
            GlobalIndex offset = 0;
            for (const auto& block : options.block_layout->blocks) {
                const auto size =
                    node_count * static_cast<GlobalIndex>(std::max(block.n_components, 0));
                ranges.push_back(BlockRange{block.name, offset, offset + size});
                offset += size;
            }
            if (offset == n_rows) {
                return ranges;
            }
        }
    }

    ranges.push_back(BlockRange{"all", 0, n_rows});
    return ranges;
}

template <class Predicate>
[[nodiscard]] std::string first_indices(GlobalIndex n, Predicate predicate, int limit = 8)
{
    std::ostringstream os;
    int emitted = 0;
    for (GlobalIndex i = 0; i < n && emitted < limit; ++i) {
        if (!predicate(i)) {
            continue;
        }
        if (emitted > 0) {
            os << "|";
        }
        os << i;
        ++emitted;
    }
    if (emitted == 0) {
        return "none";
    }
    return os.str();
}

template <class Predicate>
[[nodiscard]] std::string index_runs(GlobalIndex n, Predicate predicate, int limit = 12)
{
    std::ostringstream os;
    int emitted = 0;
    GlobalIndex run_begin = -1;
    GlobalIndex run_end = -1;
    auto flush = [&]() {
        if (run_begin < 0 || emitted >= limit) {
            return;
        }
        if (emitted > 0) {
            os << "|";
        }
        os << run_begin;
        if (run_end != run_begin) {
            os << "-" << run_end;
        }
        ++emitted;
    };

    for (GlobalIndex i = 0; i < n; ++i) {
        if (predicate(i)) {
            if (run_begin < 0) {
                run_begin = i;
                run_end = i;
            } else if (i == run_end + 1) {
                run_end = i;
            } else {
                flush();
                if (emitted >= limit) {
                    break;
                }
                run_begin = i;
                run_end = i;
            }
        } else if (run_begin >= 0) {
            flush();
            if (emitted >= limit) {
                break;
            }
            run_begin = -1;
            run_end = -1;
        }
    }
    if (emitted < limit) {
        flush();
    }
    if (emitted == 0) {
        return "none";
    }
    return os.str();
}

void emit_direct_factorization_diagnostics(const EigenMatrix& A,
                                           const EigenVector& b,
                                           const SolverOptions& options,
                                           Eigen::ComputationInfo info,
                                           const char* phase)
{
    const auto& mat = A.eigen();
    const GlobalIndex n_rows = A.numRows();
    const GlobalIndex n_cols = A.numCols();
    const Real tiny = 1.0e-14;
    const Real identity_tol = 1.0e-12;

    std::vector<Real> row_abs_sum(static_cast<std::size_t>(std::max<GlobalIndex>(n_rows, 0)), 0.0);
    std::vector<Real> col_abs_sum(static_cast<std::size_t>(std::max<GlobalIndex>(n_cols, 0)), 0.0);
    std::vector<int> row_numeric_entries(static_cast<std::size_t>(std::max<GlobalIndex>(n_rows, 0)), 0);
    std::vector<int> col_numeric_entries(static_cast<std::size_t>(std::max<GlobalIndex>(n_cols, 0)), 0);
    std::vector<Real> diag_values(static_cast<std::size_t>(std::max<GlobalIndex>(std::min(n_rows, n_cols), 0)), 0.0);
    std::vector<int> diag_present(diag_values.size(), 0);

    GlobalIndex structural_empty_rows = 0;
    GlobalIndex numeric_nnz = 0;
    GlobalIndex nonfinite_entries = 0;
    Real max_abs = 0.0;
    Real min_abs_numeric = std::numeric_limits<Real>::infinity();

    const auto* outer = mat.outerIndexPtr();
    const auto* inner = mat.innerIndexPtr();
    const auto* values = mat.valuePtr();

    for (GlobalIndex row_g = 0; row_g < n_rows; ++row_g) {
        const auto row = static_cast<EigenMatrix::StorageIndex>(row_g);
        const auto start = outer[row];
        const auto end = outer[row + 1];
        if (end == start) {
            ++structural_empty_rows;
        }
        for (auto k = start; k < end; ++k) {
            const auto col_g = static_cast<GlobalIndex>(inner[k]);
            const Real value = values[k];
            if (!std::isfinite(value)) {
                ++nonfinite_entries;
                continue;
            }
            const Real abs_value = std::abs(value);
            if (abs_value > tiny) {
                ++numeric_nnz;
                ++row_numeric_entries[static_cast<std::size_t>(row_g)];
                if (col_g >= 0 && col_g < n_cols) {
                    ++col_numeric_entries[static_cast<std::size_t>(col_g)];
                }
                min_abs_numeric = std::min(min_abs_numeric, abs_value);
            }
            max_abs = std::max(max_abs, abs_value);
            row_abs_sum[static_cast<std::size_t>(row_g)] += abs_value;
            if (col_g >= 0 && col_g < n_cols) {
                col_abs_sum[static_cast<std::size_t>(col_g)] += abs_value;
            }
            if (row_g == col_g && row_g < static_cast<GlobalIndex>(diag_values.size())) {
                diag_values[static_cast<std::size_t>(row_g)] = value;
                diag_present[static_cast<std::size_t>(row_g)] = 1;
            }
        }
    }

    GlobalIndex zero_rows = 0;
    GlobalIndex zero_cols = 0;
    GlobalIndex zero_diag = 0;
    GlobalIndex missing_diag = 0;
    GlobalIndex identity_rows = 0;

    for (GlobalIndex row_g = 0; row_g < n_rows; ++row_g) {
        if (row_abs_sum[static_cast<std::size_t>(row_g)] <= tiny) {
            ++zero_rows;
        }
        if (row_g < static_cast<GlobalIndex>(diag_values.size())) {
            const auto idx = static_cast<std::size_t>(row_g);
            if (diag_present[idx] == 0) {
                ++missing_diag;
            }
            if (std::abs(diag_values[idx]) <= tiny) {
                ++zero_diag;
            }
            if (row_numeric_entries[idx] == 1 &&
                std::abs(diag_values[idx] - 1.0) <= identity_tol &&
                std::abs(row_abs_sum[idx] - 1.0) <= identity_tol) {
                ++identity_rows;
            }
        }
    }
    for (GlobalIndex col_g = 0; col_g < n_cols; ++col_g) {
        if (col_abs_sum[static_cast<std::size_t>(col_g)] <= tiny) {
            ++zero_cols;
        }
    }

    GlobalIndex rhs_nonfinite = 0;
    GlobalIndex rhs_numeric_entries = 0;
    Real rhs_max_abs = 0.0;
    const auto& rhs = b.eigen();
    for (Eigen::Index i = 0; i < rhs.size(); ++i) {
        const Real value = rhs(i);
        if (!std::isfinite(value)) {
            ++rhs_nonfinite;
            continue;
        }
        const Real abs_value = std::abs(value);
        if (abs_value > tiny) {
            ++rhs_numeric_entries;
        }
        rhs_max_abs = std::max(rhs_max_abs, abs_value);
    }

    std::ostringstream os;
    os << std::setprecision(17);
    os << "[svMultiPhysics::FE] Eigen direct factorization diagnostic"
       << " phase=" << (phase != nullptr ? phase : "unknown")
       << " info=" << computation_info_name(info)
       << " rows=" << n_rows
       << " cols=" << n_cols
       << " structural_nnz=" << mat.nonZeros()
       << " numeric_nnz=" << numeric_nnz
       << " nonfinite_entries=" << nonfinite_entries
       << " structural_empty_rows=" << structural_empty_rows
       << " zero_rows=" << zero_rows
       << " zero_cols=" << zero_cols
       << " missing_diag=" << missing_diag
       << " zero_diag=" << zero_diag
       << " identity_rows=" << identity_rows
       << " max_abs=" << max_abs
       << " min_abs_numeric="
       << (std::isfinite(min_abs_numeric) ? min_abs_numeric : 0.0)
       << " rhs_norm=" << b.eigen().norm()
       << " rhs_numeric_entries=" << rhs_numeric_entries
       << " rhs_nonfinite_entries=" << rhs_nonfinite
       << " rhs_max_abs=" << rhs_max_abs
       << " zero_rows_first="
       << first_indices(n_rows, [&](GlobalIndex row) {
              return row_abs_sum[static_cast<std::size_t>(row)] <= tiny;
          })
       << " zero_cols_first="
       << first_indices(n_cols, [&](GlobalIndex col) {
              return col_abs_sum[static_cast<std::size_t>(col)] <= tiny;
          })
       << " zero_row_runs="
       << index_runs(n_rows, [&](GlobalIndex row) {
              return row_abs_sum[static_cast<std::size_t>(row)] <= tiny;
          })
       << " zero_col_runs="
       << index_runs(n_cols, [&](GlobalIndex col) {
              return col_abs_sum[static_cast<std::size_t>(col)] <= tiny;
          });

    const auto ranges = build_diagnostic_block_ranges(options, n_rows);
    os << " block_summaries=";
    for (std::size_t bi = 0; bi < ranges.size(); ++bi) {
        if (bi > 0) {
            os << ";";
        }
        const auto& range = ranges[bi];
        GlobalIndex block_zero_rows = 0;
        GlobalIndex block_zero_cols = 0;
        GlobalIndex block_zero_diag = 0;
        GlobalIndex block_missing_diag = 0;
        GlobalIndex block_identity_rows = 0;
        Real block_max_row_sum = 0.0;
        Real block_min_positive_row_sum = std::numeric_limits<Real>::infinity();

        for (GlobalIndex row = range.begin; row < range.end && row < n_rows; ++row) {
            const auto idx = static_cast<std::size_t>(row);
            const Real sum = row_abs_sum[idx];
            if (sum <= tiny) {
                ++block_zero_rows;
            } else {
                block_min_positive_row_sum = std::min(block_min_positive_row_sum, sum);
            }
            block_max_row_sum = std::max(block_max_row_sum, sum);
            if (row < static_cast<GlobalIndex>(diag_values.size())) {
                if (diag_present[idx] == 0) {
                    ++block_missing_diag;
                }
                if (std::abs(diag_values[idx]) <= tiny) {
                    ++block_zero_diag;
                }
                if (row_numeric_entries[idx] == 1 &&
                    std::abs(diag_values[idx] - 1.0) <= identity_tol &&
                    std::abs(row_abs_sum[idx] - 1.0) <= identity_tol) {
                    ++block_identity_rows;
                }
            }
        }
        for (GlobalIndex col = range.begin; col < range.end && col < n_cols; ++col) {
            if (col_abs_sum[static_cast<std::size_t>(col)] <= tiny) {
                ++block_zero_cols;
            }
        }

        os << range.name
           << "{begin=" << range.begin
           << ",end=" << range.end
           << ",zero_rows=" << block_zero_rows
           << ",zero_cols=" << block_zero_cols
           << ",zero_rows_first_local="
           << first_indices(std::max<GlobalIndex>(range.end - range.begin, 0),
                            [&](GlobalIndex local_row) {
                                const auto row = range.begin + local_row;
                                return row >= 0 && row < n_rows &&
                                       row_abs_sum[static_cast<std::size_t>(row)] <= tiny;
                            })
           << ",zero_cols_first_local="
           << first_indices(std::max<GlobalIndex>(range.end - range.begin, 0),
                            [&](GlobalIndex local_col) {
                                const auto col = range.begin + local_col;
                                return col >= 0 && col < n_cols &&
                                       col_abs_sum[static_cast<std::size_t>(col)] <= tiny;
                            })
           << ",zero_row_runs_local="
           << index_runs(std::max<GlobalIndex>(range.end - range.begin, 0),
                         [&](GlobalIndex local_row) {
                             const auto row = range.begin + local_row;
                             return row >= 0 && row < n_rows &&
                                    row_abs_sum[static_cast<std::size_t>(row)] <= tiny;
                         })
           << ",zero_col_runs_local="
           << index_runs(std::max<GlobalIndex>(range.end - range.begin, 0),
                         [&](GlobalIndex local_col) {
                             const auto col = range.begin + local_col;
                             return col >= 0 && col < n_cols &&
                                    col_abs_sum[static_cast<std::size_t>(col)] <= tiny;
                         })
           << ",missing_diag=" << block_missing_diag
           << ",zero_diag=" << block_zero_diag
           << ",identity_rows=" << block_identity_rows
           << ",min_positive_row_sum="
           << (std::isfinite(block_min_positive_row_sum) ? block_min_positive_row_sum : 0.0)
           << ",max_row_sum=" << block_max_row_sum
           << "}";
    }

    std::cout << os.str() << std::endl;
}

[[nodiscard]] bool diagnostics_requested()
{
    const char* env = std::getenv("SVMP_FE_EIGEN_FACTOR_DIAGNOSTICS");
    return env != nullptr && env[0] != '\0' && std::string(env) != "0";
}

SolverReport solve_direct(const EigenMatrix& A, EigenVector& x, const EigenVector& b, const SolverOptions& options)
{
    SolverReport report;
    report.initial_residual_norm = (b.eigen() - A.eigen() * x.eigen()).norm();

    Eigen::SparseMatrix<Real, Eigen::ColMajor, EigenMatrix::StorageIndex> Acol = A.eigen();
    Eigen::SparseLU<decltype(Acol)> solver;
    solver.analyzePattern(Acol);
    solver.factorize(Acol);

    if (solver.info() != Eigen::Success) {
        emit_direct_factorization_diagnostics(A, b, options, solver.info(), "factorize");
        FE_THROW(FEException, "EigenLinearSolver (direct): factorization failed");
    } else if (diagnostics_requested()) {
        emit_direct_factorization_diagnostics(A, b, options, solver.info(), "factorize");
    }

    x.eigen() = solver.solve(b.eigen());
    if (solver.info() != Eigen::Success) {
        emit_direct_factorization_diagnostics(A, b, options, solver.info(), "solve");
        FE_THROW(FEException, "EigenLinearSolver (direct): solve failed");
    }

    report.final_residual_norm = (b.eigen() - A.eigen() * x.eigen()).norm();
    const Real denom = std::max<Real>(report.initial_residual_norm, 1e-30);
    report.relative_residual = report.final_residual_norm / denom;
    report.iterations = 1;

    report.converged = check_convergence(report, options);

    report.message = "direct";
    return report;
}

template <typename IterativeSolver>
SolverReport solve_iterative(const EigenMatrix& A,
                             EigenVector& x,
                             const EigenVector& b,
                             IterativeSolver& solver,
                             const SolverOptions& options)
{
    SolverReport report;

    if (!options.use_initial_guess) {
        x.eigen().setZero();
    }

    const auto r0 = (b.eigen() - A.eigen() * x.eigen());
    report.initial_residual_norm = r0.norm();

    solver.setMaxIterations(options.max_iter);
    solver.setTolerance(options.rel_tol);
    solver.compute(A.eigen());

    if (options.use_initial_guess) {
        x.eigen() = solver.solveWithGuess(b.eigen(), x.eigen());
    } else {
        x.eigen() = solver.solve(b.eigen());
    }

    report.iterations = solver.iterations();
    const auto rf = (b.eigen() - A.eigen() * x.eigen());
    report.final_residual_norm = rf.norm();

    const Real denom = std::max<Real>(report.initial_residual_norm, 1e-30);
    report.relative_residual = report.final_residual_norm / denom;

    report.converged = check_convergence(report, options);

    report.message = "iterative";
    return report;
}

} // namespace

SolverReport EigenLinearSolver::solve(const GenericMatrix& A_in,
                                      GenericVector& x_in,
                                      const GenericVector& b_in)
{
    const auto* A = dynamic_cast<const EigenMatrix*>(&A_in);
    auto* x = dynamic_cast<EigenVector*>(&x_in);
    const auto* b = dynamic_cast<const EigenVector*>(&b_in);

    FE_THROW_IF(!A || !x || !b, InvalidArgumentException, "EigenLinearSolver::solve: backend mismatch");
    FE_THROW_IF(options_.preconditioner == PreconditionerType::FieldSplit, NotImplementedException,
                "EigenLinearSolver::solve: field-split preconditioning not supported");
    FE_THROW_IF(A->numRows() != A->numCols(), NotImplementedException,
                "EigenLinearSolver::solve: rectangular systems not implemented");
    FE_THROW_IF(A->numCols() != b->size() || x->size() != b->size(), InvalidArgumentException,
                "EigenLinearSolver::solve: size mismatch");

    switch (options_.method) {
        case SolverMethod::Direct:
            return solve_direct(*A, *x, *b, options_);

        case SolverMethod::CG: {
            FE_THROW_IF(options_.preconditioner == PreconditionerType::AMG, NotImplementedException,
                        "EigenLinearSolver::solve: AMG preconditioning not supported");
            if (options_.preconditioner == PreconditionerType::Diagonal ||
                options_.preconditioner == PreconditionerType::RowColumnScaling) {
                Eigen::ConjugateGradient<EigenMatrix::SparseMat, Eigen::Lower | Eigen::Upper,
                                         Eigen::DiagonalPreconditioner<Real>> solver;
                return solve_iterative(*A, *x, *b, solver, options_);
            }

            Eigen::ConjugateGradient<EigenMatrix::SparseMat, Eigen::Lower | Eigen::Upper,
                                     Eigen::IdentityPreconditioner> solver;
            return solve_iterative(*A, *x, *b, solver, options_);
        }

        case SolverMethod::BiCGSTAB: {
            FE_THROW_IF(options_.preconditioner == PreconditionerType::AMG, NotImplementedException,
                        "EigenLinearSolver::solve: AMG preconditioning not supported");
            if (options_.preconditioner == PreconditionerType::Diagonal ||
                options_.preconditioner == PreconditionerType::RowColumnScaling) {
                Eigen::BiCGSTAB<EigenMatrix::SparseMat, Eigen::DiagonalPreconditioner<Real>> solver;
                return solve_iterative(*A, *x, *b, solver, options_);
            }

            if (options_.preconditioner == PreconditionerType::ILU) {
                Eigen::BiCGSTAB<EigenMatrix::SparseMat, Eigen::IncompleteLUT<Real>> solver;
                return solve_iterative(*A, *x, *b, solver, options_);
            }

            Eigen::BiCGSTAB<EigenMatrix::SparseMat, Eigen::IdentityPreconditioner> solver;
            return solve_iterative(*A, *x, *b, solver, options_);
        }

        case SolverMethod::GMRES:
        case SolverMethod::PGMRES:
        case SolverMethod::FGMRES:
        case SolverMethod::BlockSchur: {
            FE_THROW_IF(options_.preconditioner == PreconditionerType::AMG, NotImplementedException,
                        "EigenLinearSolver::solve: AMG preconditioning not supported");

            const int restart = std::max(1, std::min(options_.max_iter, 30));

            if (options_.preconditioner == PreconditionerType::Diagonal ||
                options_.preconditioner == PreconditionerType::RowColumnScaling) {
                Eigen::GMRES<EigenMatrix::SparseMat, Eigen::DiagonalPreconditioner<Real>> solver;
                solver.set_restart(restart);
                return solve_iterative(*A, *x, *b, solver, options_);
            }

            if (options_.preconditioner == PreconditionerType::ILU) {
                Eigen::GMRES<EigenMatrix::SparseMat, Eigen::IncompleteLUT<Real>> solver;
                solver.set_restart(restart);
                return solve_iterative(*A, *x, *b, solver, options_);
            }

            Eigen::GMRES<EigenMatrix::SparseMat, Eigen::IdentityPreconditioner> solver;
            solver.set_restart(restart);
            return solve_iterative(*A, *x, *b, solver, options_);
        }
        default:
            FE_THROW(NotImplementedException, "EigenLinearSolver::solve: unknown solver method");
    }
}

#endif // FE_HAS_EIGEN

} // namespace backends
} // namespace FE
} // namespace svmp
