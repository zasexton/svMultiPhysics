/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "VectorizationHelper.h"

#include <sstream>
#include <stdexcept>

namespace svmp {
namespace FE {
namespace assembly {

// ============================================================================
// VectorizationHelper Implementation
// ============================================================================

VectorizationHelper::VectorizationHelper()
    : options_{}
{
}

VectorizationHelper::VectorizationHelper(const VectorizationOptions& options)
    : options_(options)
{
}

VectorizationHelper::~VectorizationHelper() = default;

VectorizationHelper::VectorizationHelper(VectorizationHelper&&) noexcept = default;

VectorizationHelper& VectorizationHelper::operator=(VectorizationHelper&&) noexcept = default;

// ============================================================================
// Configuration
// ============================================================================

void VectorizationHelper::setOptions(const VectorizationOptions& options)
{
    options_ = options;
}

const VectorizationOptions& VectorizationHelper::getOptions() const noexcept
{
    return options_;
}

bool VectorizationHelper::isSIMDEnabled() const noexcept
{
    if (!options_.enable_simd) {
        return false;
    }

    // Check compile-time SIMD availability
    return math::simd::SIMDCapabilities::has_sse2 ||
           math::simd::SIMDCapabilities::has_avx ||
           math::simd::SIMDCapabilities::has_avx2 ||
           math::simd::SIMDCapabilities::has_avx512;
}

std::size_t VectorizationHelper::effectiveVectorWidth() const noexcept
{
    if (!isSIMDEnabled()) {
        return 1;
    }
    return VectorWidth::default_width;
}

// ============================================================================
// Batch Management
// ============================================================================

std::unique_ptr<ElementBatch<>> VectorizationHelper::createBatch(
    LocalIndex max_dofs, LocalIndex max_qpts, int dim) const
{
    return std::make_unique<ElementBatch<>>(max_dofs, max_qpts, dim);
}

std::size_t VectorizationHelper::optimalBatchSize() const noexcept
{
    if (!isSIMDEnabled()) {
        return 1;
    }

    // Use batch size from options, or default to multiple of SIMD width
    if (options_.batch_size > 0) {
        return options_.batch_size;
    }

    return effectiveVectorWidth() * 4;  // Process 4x SIMD width elements
}

// ============================================================================
// Vectorized Operations
// ============================================================================

Real VectorizationHelper::dot(std::span<const Real> a, std::span<const Real> b) const
{
    if (a.size() != b.size()) {
        throw std::invalid_argument("VectorizationHelper::dot: vectors must have same size");
    }

    if (!isSIMDEnabled() || a.size() < effectiveVectorWidth()) {
        // Scalar fallback
        Real sum = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    return math::simd::dot_simd(a.data(), b.data(), a.size());
}

Real VectorizationHelper::norm(std::span<const Real> a) const
{
    if (!isSIMDEnabled() || a.size() < effectiveVectorWidth()) {
        Real sum = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            sum += a[i] * a[i];
        }
        return std::sqrt(sum);
    }

    return math::simd::norm_simd(a.data(), a.size());
}

void VectorizationHelper::axpy(Real alpha, std::span<const Real> x, std::span<Real> y) const
{
    if (x.size() != y.size()) {
        throw std::invalid_argument("VectorizationHelper::axpy: vectors must have same size");
    }

    if (!isSIMDEnabled() || x.size() < effectiveVectorWidth()) {
        for (std::size_t i = 0; i < x.size(); ++i) {
            y[i] += alpha * x[i];
        }
        return;
    }

    math::simd::axpy_simd(alpha, x.data(), y.data(), x.size());
}

void VectorizationHelper::scale(Real alpha, std::span<const Real> x, std::span<Real> y) const
{
    if (x.size() != y.size()) {
        throw std::invalid_argument("VectorizationHelper::scale: vectors must have same size");
    }

    if (!isSIMDEnabled() || x.size() < effectiveVectorWidth()) {
        for (std::size_t i = 0; i < x.size(); ++i) {
            y[i] = alpha * x[i];
        }
        return;
    }

    math::simd::scale_simd(alpha, x.data(), y.data(), x.size());
}

void VectorizationHelper::gemv(std::span<const Real> A, std::span<const Real> x,
                                std::span<Real> y, std::size_t M, std::size_t N) const
{
    if (A.size() != M * N || x.size() != N || y.size() != M) {
        throw std::invalid_argument("VectorizationHelper::gemv: dimension mismatch");
    }

    if (!isSIMDEnabled() || N < effectiveVectorWidth()) {
        // Scalar fallback
        for (std::size_t i = 0; i < M; ++i) {
            Real sum = 0.0;
            for (std::size_t j = 0; j < N; ++j) {
                sum += A[i * N + j] * x[j];
            }
            y[i] = sum;
        }
        return;
    }

    math::simd::gemv_simd(A.data(), x.data(), y.data(), M, N);
}

// ============================================================================
// Query
// ============================================================================

std::string VectorizationHelper::getSIMDInfo()
{
    std::ostringstream oss;

    oss << "SIMD Capabilities:\n";

    oss << "  SSE2:   " << (math::simd::SIMDCapabilities::has_sse2 ? "yes" : "no") << "\n";
    oss << "  SSE3:   " << (math::simd::SIMDCapabilities::has_sse3 ? "yes" : "no") << "\n";
    oss << "  SSE4.2: " << (math::simd::SIMDCapabilities::has_sse42 ? "yes" : "no") << "\n";
    oss << "  AVX:    " << (math::simd::SIMDCapabilities::has_avx ? "yes" : "no") << "\n";
    oss << "  AVX2:   " << (math::simd::SIMDCapabilities::has_avx2 ? "yes" : "no") << "\n";
    oss << "  AVX512: " << (math::simd::SIMDCapabilities::has_avx512 ? "yes" : "no") << "\n";

    oss << "\n";
    oss << "Vector Widths:\n";
    oss << "  float:  " << VectorWidth::float_width << "\n";
    oss << "  double: " << VectorWidth::double_width << "\n";

    return oss.str();
}

// ============================================================================
// Factory Functions
// ============================================================================

std::unique_ptr<VectorizationHelper> createVectorizationHelper()
{
    return std::make_unique<VectorizationHelper>();
}

std::unique_ptr<VectorizationHelper> createVectorizationHelper(
    const VectorizationOptions& options)
{
    return std::make_unique<VectorizationHelper>(options);
}

} // namespace assembly
} // namespace FE
} // namespace svmp
