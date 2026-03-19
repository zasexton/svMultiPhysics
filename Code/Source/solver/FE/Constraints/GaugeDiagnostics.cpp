/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "GaugeDiagnostics.h"

#include "Backends/Interfaces/GenericMatrix.h"
#include "Backends/Interfaces/GenericVector.h"
#include "Backends/Interfaces/BackendFactory.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <sstream>

namespace svmp {
namespace FE {
namespace gauge {

// ============================================================================
// Norm estimation via power iteration on A^T A
// ============================================================================

namespace {

/// Estimate ||A||_2 (spectral norm) via power iteration.
///
/// Uses k iterations of the power method on A^T A:
///   v_{k+1} = A^T (A v_k) / ||A^T (A v_k)||
///   sigma_k = ||A v_k||
///
/// Since we don't have A^T explicitly, we approximate with just ||A v||
/// over a few random starting vectors and take the maximum.
double estimateMatrixNorm(const backends::GenericMatrix& matrix,
                          const backends::BackendFactory& factory,
                          int n_iterations)
{
    const auto n = matrix.numRows();
    if (n <= 0) return 0.0;

    auto v = factory.createVector(n);
    auto w = factory.createVector(n);

    double max_ratio = 0.0;

    // Use a deterministic seed for reproducibility
    std::mt19937_64 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    for (int iter = 0; iter < n_iterations; ++iter) {
        // Fill v with random values
        {
            auto span = v->localSpan();
            for (auto& val : span) {
                val = dist(rng);
            }
        }

        // Normalize
        double v_norm = v->norm();
        if (v_norm < 1e-300) continue;
        v->scale(1.0 / v_norm);

        // w = A * v
        w->zero();
        matrix.mult(*v, *w);

        double w_norm = w->norm();
        if (w_norm > max_ratio) {
            max_ratio = w_norm;
        }
    }

    return max_ratio;
}

} // namespace

// ============================================================================
// Validation
// ============================================================================

std::vector<ValidationResult>
validateNullspaceBasis(const backends::GenericMatrix& matrix,
                       const backends::BackendFactory& factory,
                       const std::vector<std::vector<double>>& basis,
                       const ValidationOptions& opts)
{
    std::vector<ValidationResult> results;

    if (basis.empty()) {
        return results;
    }

    const auto n = matrix.numRows();
    if (n <= 0) {
        return results;
    }

    // Estimate ||A|| using random SpMVs
    const double a_norm = estimateMatrixNorm(matrix, factory,
                                              opts.norm_estimation_iterations);

    // Create workspace vectors
    auto z_vec = factory.createVector(n);
    auto az_vec = factory.createVector(n);

    results.reserve(basis.size());

    for (std::size_t i = 0; i < basis.size(); ++i) {
        ValidationResult result;
        result.mode_index = static_cast<int>(i);
        result.a_norm_estimate = a_norm;

        const auto& z = basis[i];
        if (static_cast<GlobalIndex>(z.size()) != n) {
            result.description = "Mode " + std::to_string(i) +
                                 ": size mismatch (z.size=" + std::to_string(z.size()) +
                                 ", n_dofs=" + std::to_string(n) + ")";
            result.passed = false;
            results.push_back(std::move(result));
            continue;
        }

        // Copy basis vector into GenericVector
        {
            auto span = z_vec->localSpan();
            for (GlobalIndex j = 0; j < n; ++j) {
                span[static_cast<std::size_t>(j)] = z[static_cast<std::size_t>(j)];
            }
        }

        result.z_norm = z_vec->norm();

        // Compute A * z
        az_vec->zero();
        matrix.mult(*z_vec, *az_vec);

        result.az_norm = az_vec->norm();

        // Relative residual
        const double denom = a_norm * result.z_norm;
        if (denom > 1e-300) {
            result.relative_residual = result.az_norm / denom;
        } else {
            // If both A and z are essentially zero, it trivially passes
            result.relative_residual = 0.0;
        }

        result.passed = (result.relative_residual < opts.tolerance);
        result.description = "Mode " + std::to_string(i) +
                             ": ||Az||=" + std::to_string(result.az_norm) +
                             ", ||A||~" + std::to_string(a_norm) +
                             ", ||z||=" + std::to_string(result.z_norm) +
                             ", rel=" + std::to_string(result.relative_residual);

        results.push_back(std::move(result));
    }

    return results;
}

// ============================================================================
// Environment variable check
// ============================================================================

bool isNullspaceValidationEnabled() noexcept
{
    const char* val = std::getenv("SVMP_GAUGE_VALIDATE");
    if (!val) return false;

    // Accept 1, true, yes, on (case-insensitive for first char)
    if (val[0] == '1' || val[0] == 't' || val[0] == 'T' ||
        val[0] == 'y' || val[0] == 'Y') {
        return true;
    }
    // "on"
    if ((val[0] == 'o' || val[0] == 'O') &&
        (val[1] == 'n' || val[1] == 'N')) {
        return true;
    }
    return false;
}

// ============================================================================
// Report formatting
// ============================================================================

std::string formatValidationReport(const std::vector<ValidationResult>& results)
{
    std::ostringstream os;
    os << "=== Nullspace Validation Report ===\n";
    os << "Modes tested: " << results.size() << "\n";

    int passed = 0;
    int failed = 0;
    for (const auto& r : results) {
        if (r.passed) ++passed;
        else ++failed;
    }
    os << "Passed: " << passed << ", Failed: " << failed << "\n\n";

    for (const auto& r : results) {
        os << "  " << (r.passed ? "[PASS]" : "[FAIL]")
           << " " << r.description << "\n";
    }

    return os.str();
}

} // namespace gauge
} // namespace FE
} // namespace svmp
