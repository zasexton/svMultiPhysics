/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SVMP_FE_ASSEMBLY_VECTORIZATION_HELPER_H
#define SVMP_FE_ASSEMBLY_VECTORIZATION_HELPER_H

/**
 * @file VectorizationHelper.h
 * @brief SIMD vectorization utilities for finite element assembly
 *
 * VectorizationHelper provides tools for accelerating FE assembly through SIMD
 * (Single Instruction, Multiple Data) vectorization. Key capabilities:
 *
 * 1. BATCHED ELEMENT PROCESSING:
 *    Process multiple elements simultaneously using SIMD lanes.
 *    - Pack N elements' data into aligned arrays
 *    - Perform vectorized quadrature operations
 *    - Unpack results into per-element contributions
 *
 * 2. VECTORIZED QUADRATURE LOOPS:
 *    Accelerate inner quadrature point loops.
 *    - Basis function evaluation across multiple quad points
 *    - Jacobian computation with SIMD
 *    - Weighted summation
 *
 * 3. ALIGNED DATA MANAGEMENT:
 *    Tools for SIMD-friendly data layout.
 *    - Aligned allocation (AVX requires 32-byte, AVX512 requires 64-byte)
 *    - Structure-of-Arrays (SoA) layout helpers
 *    - Padding for efficient vectorization
 *
 * 4. PORTABLE ABSTRACTIONS:
 *    Works across different SIMD instruction sets.
 *    - Compile-time detection of AVX, AVX2, AVX512, SSE
 *    - Runtime dispatching for heterogeneous systems
 *    - Scalar fallback when no SIMD available
 *
 * Integration with Math/SIMD.h:
 *    This file builds on the SIMD primitives in Math/SIMD.h and provides
 *    assembly-specific patterns like batched element processing.
 *
 * @see Math/SIMD.h for low-level SIMD operations
 * @see AssemblyContext for data layout consumed by vectorized kernels
 */

#include "Core/Types.h"
#include "Math/SIMD.h"

#include <vector>
#include <array>
#include <span>
#include <memory>
#include <cstddef>
#include <algorithm>

namespace svmp {
namespace FE {
namespace assembly {

// ============================================================================
// Vectorization Configuration
// ============================================================================

/**
 * @brief SIMD vector width for assembly operations
 *
 * Determines how many elements are processed simultaneously.
 */
struct VectorWidth {
    /**
     * @brief Width for double precision
     */
    static constexpr std::size_t double_width =
        math::simd::SIMDCapabilities::double_width();

    /**
     * @brief Width for single precision
     */
    static constexpr std::size_t float_width =
        math::simd::SIMDCapabilities::float_width();

    /**
     * @brief Default width (based on Real type)
     */
    static constexpr std::size_t default_width = double_width;
};

/**
 * @brief Configuration for vectorized assembly
 */
struct VectorizationOptions {
    /**
     * @brief Enable SIMD vectorization
     */
    bool enable_simd{true};

    /**
     * @brief Batch size for element processing
     *
     * Should be multiple of SIMD width for best performance.
     */
    std::size_t batch_size{VectorWidth::default_width * 4};

    /**
     * @brief Use aligned memory allocations
     */
    bool use_aligned_memory{true};

    /**
     * @brief Alignment requirement in bytes
     */
    std::size_t alignment{64};  // AVX512 requirement

    /**
     * @brief Enable runtime SIMD capability checking
     */
    bool runtime_dispatch{false};
};

// ============================================================================
// Aligned Memory Utilities
// ============================================================================

/**
 * @brief Aligned vector type using SIMD allocator
 */
template<typename T>
using AlignedVector = std::vector<T, math::simd::AlignedAllocator<T, 64>>;

/**
 * @brief Check if pointer is aligned
 */
inline bool isAligned(const void* ptr, std::size_t alignment) {
    return (reinterpret_cast<std::uintptr_t>(ptr) % alignment) == 0;
}

/**
 * @brief Round up to alignment boundary
 */
constexpr std::size_t alignUp(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

// ============================================================================
// Element Batch for Vectorized Processing
// ============================================================================

/**
 * @brief Batch of elements for vectorized assembly
 *
 * ElementBatch groups multiple elements together so that their data can be
 * processed using SIMD instructions. Data is laid out in Structure-of-Arrays
 * (SoA) format for efficient vectorization.
 *
 * @tparam BatchSize Number of elements in batch (should match SIMD width)
 */
template<std::size_t BatchSize = VectorWidth::default_width>
class ElementBatch {
public:
    static constexpr std::size_t batch_size = BatchSize;

    // =========================================================================
    // Construction
    // =========================================================================

    ElementBatch() = default;

    explicit ElementBatch(LocalIndex max_dofs, LocalIndex max_qpts, int dim)
        : max_dofs_(max_dofs), max_qpts_(max_qpts), dim_(dim)
    {
        reserve();
    }

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Reserve storage for batch
     */
    void reserve() {
        const std::size_t total_qpts = batch_size * max_qpts_;
        const std::size_t total_dofs = batch_size * max_dofs_;

        // Quadrature data (SoA layout)
        quad_weights_.resize(total_qpts);
        jacobian_dets_.resize(total_qpts);

        // Reference coordinates per quadrature point
        ref_coords_.resize(total_qpts * static_cast<std::size_t>(dim_));

        // Physical coordinates per quadrature point
        phys_coords_.resize(total_qpts * static_cast<std::size_t>(dim_));

        // Basis values: [batch][dof][qpt] -> [batch * dof][qpt] in SoA
        basis_values_.resize(total_dofs * max_qpts_);

        // Basis gradients: [batch][dof][qpt][dim]
        basis_gradients_.resize(total_dofs * max_qpts_ * static_cast<std::size_t>(dim_));

        // Element IDs
        cell_ids_.fill(-1);
        num_active_ = 0;
    }

    /**
     * @brief Clear batch for reuse
     */
    void clear() {
        cell_ids_.fill(-1);
        num_active_ = 0;
    }

    // =========================================================================
    // Element Management
    // =========================================================================

    /**
     * @brief Add element to batch
     *
     * @param cell_id Global cell ID
     * @return Slot index in batch, or -1 if full
     */
    int addElement(GlobalIndex cell_id) {
        if (num_active_ >= batch_size) {
            return -1;  // Batch full
        }

        int slot = static_cast<int>(num_active_);
        cell_ids_[num_active_] = cell_id;
        num_active_++;
        return slot;
    }

    /**
     * @brief Check if batch is full
     */
    [[nodiscard]] bool isFull() const noexcept {
        return num_active_ >= batch_size;
    }

    /**
     * @brief Check if batch is empty
     */
    [[nodiscard]] bool isEmpty() const noexcept {
        return num_active_ == 0;
    }

    /**
     * @brief Get number of active elements
     */
    [[nodiscard]] std::size_t numActive() const noexcept {
        return num_active_;
    }

    /**
     * @brief Get cell ID for slot
     */
    [[nodiscard]] GlobalIndex cellId(std::size_t slot) const {
        return cell_ids_[slot];
    }

    // =========================================================================
    // Data Access (SoA Layout)
    // =========================================================================

    /**
     * @brief Get quadrature weights for all elements
     *
     * Layout: weights[slot * max_qpts + q]
     */
    [[nodiscard]] Real* quadWeights() noexcept {
        return quad_weights_.data();
    }
    [[nodiscard]] const Real* quadWeights() const noexcept {
        return quad_weights_.data();
    }

    /**
     * @brief Get Jacobian determinants for all elements
     *
     * Layout: dets[slot * max_qpts + q]
     */
    [[nodiscard]] Real* jacobianDets() noexcept {
        return jacobian_dets_.data();
    }
    [[nodiscard]] const Real* jacobianDets() const noexcept {
        return jacobian_dets_.data();
    }

    /**
     * @brief Get basis values
     *
     * Layout: values[slot * max_dofs * max_qpts + dof * max_qpts + q]
     */
    [[nodiscard]] Real* basisValues() noexcept {
        return basis_values_.data();
    }
    [[nodiscard]] const Real* basisValues() const noexcept {
        return basis_values_.data();
    }

    /**
     * @brief Get basis gradients
     *
     * Layout: grads[slot * max_dofs * max_qpts * dim + dof * max_qpts * dim + q * dim + d]
     */
    [[nodiscard]] Real* basisGradients() noexcept {
        return basis_gradients_.data();
    }
    [[nodiscard]] const Real* basisGradients() const noexcept {
        return basis_gradients_.data();
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    /**
     * @brief Get offset into per-element array for given slot and quadrature point
     */
    [[nodiscard]] std::size_t quadOffset(std::size_t slot, LocalIndex q) const noexcept {
        return slot * max_qpts_ + q;
    }

    /**
     * @brief Get offset into basis value array
     */
    [[nodiscard]] std::size_t basisOffset(std::size_t slot, LocalIndex dof, LocalIndex q) const noexcept {
        return slot * max_dofs_ * max_qpts_ + dof * max_qpts_ + q;
    }

    /**
     * @brief Get offset into basis gradient array
     */
    [[nodiscard]] std::size_t gradOffset(std::size_t slot, LocalIndex dof,
                                          LocalIndex q, int d) const noexcept {
        return slot * max_dofs_ * max_qpts_ * static_cast<std::size_t>(dim_) +
               dof * max_qpts_ * static_cast<std::size_t>(dim_) +
               q * static_cast<std::size_t>(dim_) + static_cast<std::size_t>(d);
    }

    [[nodiscard]] LocalIndex maxDofs() const noexcept { return max_dofs_; }
    [[nodiscard]] LocalIndex maxQpts() const noexcept { return max_qpts_; }
    [[nodiscard]] int dimension() const noexcept { return dim_; }

private:
    LocalIndex max_dofs_{0};
    LocalIndex max_qpts_{0};
    int dim_{3};
    std::size_t num_active_{0};

    std::array<GlobalIndex, batch_size> cell_ids_{};

    // SoA data storage (aligned)
    AlignedVector<Real> quad_weights_;
    AlignedVector<Real> jacobian_dets_;
    AlignedVector<Real> ref_coords_;
    AlignedVector<Real> phys_coords_;
    AlignedVector<Real> basis_values_;
    AlignedVector<Real> basis_gradients_;
};

// ============================================================================
// Vectorized Quadrature Operations
// ============================================================================

/**
 * @brief Vectorized quadrature loop utilities
 */
class VectorizedQuadrature {
public:
    using simd_ops = math::simd::SIMDOps<Real>;
    using vec_t = typename simd_ops::vec_type;
    static constexpr std::size_t vec_size = simd_ops::vec_size;

    /**
     * @brief Vectorized weighted sum: result = sum_q (w_q * f_q)
     *
     * @param weights Quadrature weights times Jacobian determinant
     * @param values Function values at quadrature points
     * @param num_qpts Number of quadrature points
     * @return Weighted sum
     */
    static Real weightedSum(const Real* weights, const Real* values, std::size_t num_qpts) {
        return math::simd::dot_simd(weights, values, num_qpts);
    }

    /**
     * @brief Vectorized integration weight computation: JxW = |J| * w_ref
     *
     * @param jacobian_dets Array of Jacobian determinants
     * @param ref_weights Reference element quadrature weights
     * @param output Output array for JxW values
     * @param num_qpts Number of quadrature points
     */
    static void computeJxW(const Real* jacobian_dets, const Real* ref_weights,
                           Real* output, std::size_t num_qpts) {
        math::simd::mul_simd(jacobian_dets, ref_weights, output, num_qpts);
    }

    /**
     * @brief Vectorized basis function x weight accumulation
     *
     * Computes: result[i] += sum_q (phi_i(q) * weight(q) * f(q))
     *
     * @param basis Basis function values [num_dofs * num_qpts]
     * @param weights Integration weights [num_qpts]
     * @param f Function values at quadrature points [num_qpts]
     * @param result Output vector [num_dofs]
     * @param num_dofs Number of DOFs
     * @param num_qpts Number of quadrature points
     */
    static void accumulateBasisTimesScalar(
        const Real* basis, const Real* weights, const Real* f,
        Real* result, std::size_t num_dofs, std::size_t num_qpts)
    {
        // Compute weighted function values: wf[q] = weights[q] * f[q]
        AlignedVector<Real> wf(num_qpts);
        math::simd::mul_simd(weights, f, wf.data(), num_qpts);

        // For each DOF, compute inner product with basis
        for (std::size_t i = 0; i < num_dofs; ++i) {
            result[i] += math::simd::dot_simd(&basis[i * num_qpts], wf.data(), num_qpts);
        }
    }

    /**
     * @brief Vectorized gradient dot product
     *
     * Computes: result = sum_q (grad_phi . grad_psi * JxW)
     */
    static Real gradientDotProduct(
        const Real* grad_phi, const Real* grad_psi, const Real* JxW,
        std::size_t num_qpts, int dim)
    {
        Real sum = 0.0;
        for (int d = 0; d < dim; ++d) {
            const Real* grad_phi_d = grad_phi + d * num_qpts;
            const Real* grad_psi_d = grad_psi + d * num_qpts;

            // Compute component-wise product
            AlignedVector<Real> temp(num_qpts);
            math::simd::mul_simd(grad_phi_d, grad_psi_d, temp.data(), num_qpts);

            // Multiply by JxW and sum
            sum += math::simd::dot_simd(temp.data(), JxW, num_qpts);
        }
        return sum;
    }
};

// ============================================================================
// Batched Element Matrix Assembly
// ============================================================================

/**
 * @brief Batched local matrix assembly
 *
 * Computes element matrices for multiple elements simultaneously using SIMD.
 *
 * @tparam BatchSize Number of elements to process together
 */
template<std::size_t BatchSize = VectorWidth::default_width>
class BatchedMatrixAssembly {
public:
    /**
     * @brief Compute local stiffness matrices for batch
     *
     * Assembles: K_e[i,j] = integral grad(phi_i) . grad(phi_j) dx
     *
     * @param batch Element batch with prepared data
     * @param local_matrices Output: batch_size arrays of local matrices
     */
    static void assembleStiffnessMatrices(
        const ElementBatch<BatchSize>& batch,
        std::array<std::vector<Real>, BatchSize>& local_matrices)
    {
        const LocalIndex num_dofs = batch.maxDofs();
        const LocalIndex num_qpts = batch.maxQpts();
        const int dim = batch.dimension();
        const std::size_t num_active = batch.numActive();

        // Initialize output matrices
        for (std::size_t e = 0; e < num_active; ++e) {
            local_matrices[e].resize(static_cast<std::size_t>(num_dofs) * num_dofs, 0.0);
        }

        // For each DOF pair
        for (LocalIndex i = 0; i < num_dofs; ++i) {
            for (LocalIndex j = 0; j <= i; ++j) {  // Exploit symmetry

                // Process all elements in batch
                for (std::size_t e = 0; e < num_active; ++e) {
                    Real sum = 0.0;

                    // Quadrature loop
                    for (LocalIndex q = 0; q < num_qpts; ++q) {
                        Real JxW = batch.jacobianDets()[batch.quadOffset(e, q)] *
                                   batch.quadWeights()[batch.quadOffset(e, q)];

                        // Gradient dot product
                        Real grad_dot = 0.0;
                        for (int d = 0; d < dim; ++d) {
                            Real grad_i = batch.basisGradients()[batch.gradOffset(e, i, q, d)];
                            Real grad_j = batch.basisGradients()[batch.gradOffset(e, j, q, d)];
                            grad_dot += grad_i * grad_j;
                        }

                        sum += grad_dot * JxW;
                    }

                    // Store (exploit symmetry)
                    local_matrices[e][static_cast<std::size_t>(i) * num_dofs + j] = sum;
                    if (i != j) {
                        local_matrices[e][static_cast<std::size_t>(j) * num_dofs + i] = sum;
                    }
                }
            }
        }
    }

    /**
     * @brief Compute local mass matrices for batch
     *
     * Assembles: M_e[i,j] = integral phi_i * phi_j dx
     */
    static void assembleMassMatrices(
        const ElementBatch<BatchSize>& batch,
        std::array<std::vector<Real>, BatchSize>& local_matrices)
    {
        const LocalIndex num_dofs = batch.maxDofs();
        const LocalIndex num_qpts = batch.maxQpts();
        const std::size_t num_active = batch.numActive();

        // Initialize output matrices
        for (std::size_t e = 0; e < num_active; ++e) {
            local_matrices[e].resize(static_cast<std::size_t>(num_dofs) * num_dofs, 0.0);
        }

        // For each DOF pair
        for (LocalIndex i = 0; i < num_dofs; ++i) {
            for (LocalIndex j = 0; j <= i; ++j) {

                for (std::size_t e = 0; e < num_active; ++e) {
                    Real sum = 0.0;

                    for (LocalIndex q = 0; q < num_qpts; ++q) {
                        Real JxW = batch.jacobianDets()[batch.quadOffset(e, q)] *
                                   batch.quadWeights()[batch.quadOffset(e, q)];

                        Real phi_i = batch.basisValues()[batch.basisOffset(e, i, q)];
                        Real phi_j = batch.basisValues()[batch.basisOffset(e, j, q)];

                        sum += phi_i * phi_j * JxW;
                    }

                    local_matrices[e][static_cast<std::size_t>(i) * num_dofs + j] = sum;
                    if (i != j) {
                        local_matrices[e][static_cast<std::size_t>(j) * num_dofs + i] = sum;
                    }
                }
            }
        }
    }
};

// ============================================================================
// VectorizationHelper
// ============================================================================

/**
 * @brief Main helper class for vectorized assembly
 *
 * Provides high-level interface for vectorized finite element assembly.
 */
class VectorizationHelper {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    VectorizationHelper();
    explicit VectorizationHelper(const VectorizationOptions& options);
    ~VectorizationHelper();

    VectorizationHelper(const VectorizationHelper&) = delete;
    VectorizationHelper& operator=(const VectorizationHelper&) = delete;

    VectorizationHelper(VectorizationHelper&&) noexcept;
    VectorizationHelper& operator=(VectorizationHelper&&) noexcept;

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Set vectorization options
     */
    void setOptions(const VectorizationOptions& options);

    /**
     * @brief Get current options
     */
    [[nodiscard]] const VectorizationOptions& getOptions() const noexcept;

    /**
     * @brief Check if SIMD is available and enabled
     */
    [[nodiscard]] bool isSIMDEnabled() const noexcept;

    /**
     * @brief Get effective SIMD width
     */
    [[nodiscard]] std::size_t effectiveVectorWidth() const noexcept;

    // =========================================================================
    // Batch Management
    // =========================================================================

    /**
     * @brief Create element batch for vectorized processing
     *
     * @param max_dofs Maximum DOFs per element
     * @param max_qpts Maximum quadrature points per element
     * @param dim Spatial dimension
     * @return Unique pointer to element batch
     */
    [[nodiscard]] std::unique_ptr<ElementBatch<>> createBatch(
        LocalIndex max_dofs, LocalIndex max_qpts, int dim) const;

    /**
     * @brief Get optimal batch size for current configuration
     */
    [[nodiscard]] std::size_t optimalBatchSize() const noexcept;

    // =========================================================================
    // Vectorized Operations
    // =========================================================================

    /**
     * @brief Vectorized dot product
     */
    [[nodiscard]] Real dot(std::span<const Real> a, std::span<const Real> b) const;

    /**
     * @brief Vectorized norm
     */
    [[nodiscard]] Real norm(std::span<const Real> a) const;

    /**
     * @brief Vectorized axpy: y = alpha*x + y
     */
    void axpy(Real alpha, std::span<const Real> x, std::span<Real> y) const;

    /**
     * @brief Vectorized scale: y = alpha * x
     */
    void scale(Real alpha, std::span<const Real> x, std::span<Real> y) const;

    /**
     * @brief Vectorized matrix-vector product for small matrices
     */
    void gemv(std::span<const Real> A, std::span<const Real> x, std::span<Real> y,
              std::size_t M, std::size_t N) const;

    // =========================================================================
    // Query
    // =========================================================================

    /**
     * @brief Get SIMD capabilities info
     */
    [[nodiscard]] static std::string getSIMDInfo();

private:
    VectorizationOptions options_;
};

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * @brief Create vectorization helper with default options
 */
std::unique_ptr<VectorizationHelper> createVectorizationHelper();

/**
 * @brief Create vectorization helper with specified options
 */
std::unique_ptr<VectorizationHelper> createVectorizationHelper(
    const VectorizationOptions& options);

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_VECTORIZATION_HELPER_H
