/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_EIGEN_LINEAR_SOLVER_H
#define SVMP_FE_BACKENDS_EIGEN_LINEAR_SOLVER_H

#include "Backends/Interfaces/LinearSolver.h"

#include <cstdint>
#include <memory>

namespace svmp {
namespace FE {
namespace backends {

#if defined(FE_HAS_EIGEN)

class EigenMatrix;
class EigenVector;

class EigenLinearSolver final : public LinearSolver {
public:
    /// Counters of the opt-in preconditioner/factorization reuse path.
    struct ReuseStats {
        std::uint64_t solves{0};
        std::uint64_t refreshes{0};
        std::uint64_t reuses{0};
        std::uint64_t stale_retries{0};
        int last_iterations{0};
        bool last_fresh{false};
    };

    explicit EigenLinearSolver(const SolverOptions& options);
    ~EigenLinearSolver() override;

    [[nodiscard]] BackendKind backendKind() const noexcept override { return BackendKind::Eigen; }

    void setOptions(const SolverOptions& options) override;
    [[nodiscard]] const SolverOptions& getOptions() const noexcept override { return options_; }

    [[nodiscard]] SolverReport solve(const GenericMatrix& A,
                                     GenericVector& x,
                                     const GenericVector& b) override;

    /// Statistics of the reuse path (all zero unless reuse_preconditioner is set).
    [[nodiscard]] ReuseStats reuseStats() const noexcept;

private:
    struct ReuseState;

    [[nodiscard]] SolverReport solveWithReuse(const EigenMatrix& A,
                                              EigenVector& x,
                                              const EigenVector& b);

    SolverOptions options_{};
    std::unique_ptr<ReuseState> reuse_;
};

#endif // FE_HAS_EIGEN

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BACKENDS_EIGEN_LINEAR_SOLVER_H

