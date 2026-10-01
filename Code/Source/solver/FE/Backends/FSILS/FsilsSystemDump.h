/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_FSILS_SYSTEM_DUMP_H
#define SVMP_FE_BACKENDS_FSILS_SYSTEM_DUMP_H

#include "Backends/Utils/BackendOptions.h"
#include "Core/Types.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace backends {

class FsilsMatrix;
class FsilsVector;

/**
 * @brief Node-block linear system captured from an FSILS solve.
 *
 * All indices use the backend numbering of a serial FSILS operator: node ids
 * are global FSILS node ids and a scalar unknown is `node * dof + component`.
 * Blocks are stored row-major (`block[r * dof + c]`), rows sorted by node and
 * columns sorted within each row.  Used to replay real solver inputs offline
 * when evaluating linear solvers and preconditioners.
 */
struct FsilsSystemSnapshot {
    int dof{0};
    int n_nodes{0};
    std::vector<std::int64_t> row_ptr{};       ///< n_nodes + 1
    std::vector<std::int32_t> cols{};          ///< nnz_blocks
    std::vector<double> values{};              ///< nnz_blocks * dof * dof
    std::vector<double> rhs{};                 ///< n_nodes * dof
    std::vector<double> solution{};            ///< n_nodes * dof (solver output)
    std::vector<std::int64_t> dirichlet_dofs{};///< backend scalar ids

    // Solver controls and the result of the recorded solve.
    int method{0};
    int preconditioner{0};
    int use_rcs{0};
    int max_iter{0};
    int krylov_dim{0};
    double rel_tol{0.0};
    double abs_tol{0.0};
    int iterations{0};
    int converged{0};
    int native_update_count{0};
    double initial_residual_norm{0.0};
    double final_residual_norm{0.0};
    double solve_seconds{0.0};
    std::vector<BlockDescriptor> blocks{};

    [[nodiscard]] std::int64_t nnzBlocks() const noexcept
    {
        return row_ptr.empty() ? 0 : row_ptr.back();
    }
};

/// True when SVMP_FSILS_DUMP_SYSTEM_PREFIX requests system snapshots.
[[nodiscard]] bool fsilsSystemDumpRequested() noexcept;

/**
 * @brief Write the solved system to `<prefix>.<sequence>.bin` when requested.
 *
 * Diagnostic only and off by default.  Controlled by
 * SVMP_FSILS_DUMP_SYSTEM_PREFIX (path prefix), SVMP_FSILS_DUMP_SYSTEM_SKIP
 * (number of leading solves to skip, default 0) and
 * SVMP_FSILS_DUMP_SYSTEM_MAX (number of snapshots, default 16).  Serial
 * operators only; distributed solves are skipped.
 */
void maybeDumpFsilsSystem(const FsilsMatrix& A,
                          const FsilsVector& b,
                          const FsilsVector& x,
                          std::span<const GlobalIndex> dirichlet_fe_dofs,
                          const SolverOptions& options,
                          const SolverReport& report,
                          double solve_seconds,
                          int native_update_count);

/// Read a snapshot written by maybeDumpFsilsSystem(). Throws on format errors.
[[nodiscard]] FsilsSystemSnapshot readFsilsSystemSnapshot(const std::string& path);

/// Write a snapshot (used by the dump hook and by tests).
void writeFsilsSystemSnapshot(const FsilsSystemSnapshot& snapshot, const std::string& path);

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BACKENDS_FSILS_SYSTEM_DUMP_H
