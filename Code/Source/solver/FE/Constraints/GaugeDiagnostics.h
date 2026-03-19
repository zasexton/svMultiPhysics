/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTRAINTS_GAUGE_DIAGNOSTICS_H
#define SVMP_FE_CONSTRAINTS_GAUGE_DIAGNOSTICS_H

/**
 * @file GaugeDiagnostics.h
 * @brief Numerical validation of inferred nullspace modes
 *
 * After the first assembly, this module can verify that inferred nullspace
 * modes are actually in the operator's nullspace by computing ||A*z|| for
 * each basis vector z and comparing to ||A|| (estimated via random SpMV).
 *
 * Usage:
 *   auto results = validateNullspaceBasis(matrix, factory, basis, opts);
 *   for (const auto& r : results) {
 *       if (!r.passed) { warn(...); }
 *   }
 *
 * Gated behind SVMP_GAUGE_VALIDATE environment variable or explicit opt-in.
 *
 * @see GaugeRegistry for nullspace detection and enforcement
 */

#include "Core/Types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace svmp {
namespace FE {

namespace backends {
class GenericMatrix;
class GenericVector;
class BackendFactory;
} // namespace backends

namespace gauge {

/**
 * @brief Options for nullspace validation
 */
struct ValidationOptions {
    /// Relative tolerance: ||A*z|| / ||A||_est must be below this
    double tolerance{1e-8};

    /// Number of random SpMV iterations for ||A|| estimation
    int norm_estimation_iterations{5};
};

/**
 * @brief Result for a single nullspace mode validation
 */
struct ValidationResult {
    int mode_index{-1};               ///< Index in the basis vector list
    double az_norm{0.0};              ///< ||A * z||
    double z_norm{0.0};              ///< ||z|| (should be ~1.0 for normalized basis)
    double a_norm_estimate{0.0};      ///< Estimated ||A|| (spectral norm estimate)
    double relative_residual{0.0};    ///< ||A*z|| / (||A||_est * ||z||)
    bool passed{false};               ///< relative_residual < tolerance
    std::string description;          ///< Human-readable label
};

/**
 * @brief Validate nullspace basis vectors against an assembled matrix
 *
 * For each basis vector z, computes y = A*z and checks whether
 * ||y|| / (||A||_est * ||z||) < tolerance.
 *
 * The matrix norm is estimated using a few random sparse matrix-vector
 * products (power iteration on A^T A).
 *
 * @param matrix   Assembled operator matrix (must be finalized)
 * @param factory  Backend factory for creating temporary vectors
 * @param basis    Nullspace basis vectors (dense, length n_dofs each)
 * @param opts     Validation options (tolerance, iteration count)
 * @return         One ValidationResult per basis vector
 */
[[nodiscard]] std::vector<ValidationResult>
validateNullspaceBasis(const backends::GenericMatrix& matrix,
                       const backends::BackendFactory& factory,
                       const std::vector<std::vector<double>>& basis,
                       const ValidationOptions& opts = {});

/**
 * @brief Check if nullspace validation is enabled via environment variable
 *
 * Returns true if SVMP_GAUGE_VALIDATE is set to a truthy value (1, true, yes, on).
 */
[[nodiscard]] bool isNullspaceValidationEnabled() noexcept;

/**
 * @brief Format validation results as a human-readable report
 */
[[nodiscard]] std::string formatValidationReport(const std::vector<ValidationResult>& results);

} // namespace gauge
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_GAUGE_DIAGNOSTICS_H
