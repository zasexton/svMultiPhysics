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

#ifndef SVMP_FE_ASSEMBLY_MATRIX_FREE_ASSEMBLER_H
#define SVMP_FE_ASSEMBLY_MATRIX_FREE_ASSEMBLER_H

/**
 * @file MatrixFreeAssembler.h
 * @brief Matrix-free assembly for iterative solvers
 *
 * MatrixFreeAssembler implements matrix-free operator application where the
 * matrix-vector product A*x is computed without explicitly forming the matrix A.
 * This approach offers:
 *
 * - Memory efficiency: O(1) storage instead of O(nnz)
 * - Cache efficiency: Element data is accessed sequentially
 * - Natural for high-order: Cost scales better with polynomial order
 * - GPU-friendly: Element-local operations map well to accelerators
 *
 * Two modes of operation:
 *
 * 1. **Traditional matrix-free**: Each apply() recomputes element matrices
 *    and applies them to the input vector on-the-fly.
 *
 * 2. **Partial assembly**: Setup phase precomputes and stores element-level
 *    data (geometry, quadrature weights, physics coefficients), then apply()
 *    uses this cached data. More efficient for repeated applies.
 *
 * Reference patterns:
 * - deal.II MatrixFree: Cell loop with FEEvaluation
 * - MFEM partial assembly: Setup + Apply decomposition
 * - libCEED: Q-function + operator decomposition
 *
 * Module boundaries:
 * - This module OWNS: operator application infrastructure, data caching
 * - This module does NOT OWN: physics kernels, solver iteration
 *
 * @see AssemblyLoop for iteration infrastructure
 * @see StandardAssembler for explicit matrix assembly
 */

#include "Core/Types.h"
#include "Core/FEException.h"
#include "Assembler.h"
#include "AssemblyKernel.h"
#include "AssemblyContext.h"
#include "AssemblyLoop.h"

#include <vector>
#include <span>
#include <functional>
#include <memory>
#include <optional>

namespace svmp {
namespace FE {

// Forward declarations
namespace dofs {
    class DofMap;
}

namespace spaces {
    class FunctionSpace;
}

namespace constraints {
    class AffineConstraints;
}

namespace assembly {

// ============================================================================
// Matrix-Free Options
// ============================================================================

/**
 * @brief Assembly level (how much to precompute)
 *
 * Inspired by MFEM assembly levels.
 */
enum class AssemblyLevel : std::uint8_t {
    None,           ///< No precomputation, recompute everything each apply
    Element,        ///< Store element matrices (like traditional FEM)
    Partial,        ///< Store quadrature-level data (geometry, weights)
    Full            ///< Full matrix assembly (for comparison/fallback)
};

/**
 * @brief Options for matrix-free assembly
 */
struct MatrixFreeOptions {
    /**
     * @brief Level of precomputation
     */
    AssemblyLevel assembly_level{AssemblyLevel::Partial};

    /**
     * @brief Number of threads for parallel apply
     */
    int num_threads{0};  // 0 = auto

    /**
     * @brief Use vectorization in kernel evaluation
     */
    bool vectorize{true};

    /**
     * @brief Batch size for vectorized evaluation
     */
    int batch_size{4};

    /**
     * @brief Apply constraints during operator application
     */
    bool apply_constraints{true};

    /**
     * @brief Cache geometry data (Jacobians, physical points)
     */
    bool cache_geometry{true};

    /**
     * @brief Cache basis evaluations
     */
    bool cache_basis{true};

    /**
     * @brief Verbose timing output
     */
    bool verbose{false};
};

/**
 * @brief Statistics from matrix-free operations
 */
struct MatrixFreeStats {
    // Setup phase
    double setup_seconds{0.0};           ///< Setup time
    std::size_t cached_bytes{0};         ///< Memory used for cached data

    // Apply phase
    GlobalIndex num_applies{0};          ///< Number of apply() calls
    double total_apply_seconds{0.0};     ///< Total time in apply()
    double avg_apply_seconds{0.0};       ///< Average time per apply
    double last_apply_seconds{0.0};      ///< Time for last apply

    // Performance metrics
    double gflops{0.0};                  ///< Estimated GFLOP/s
    double bandwidth_gb_s{0.0};          ///< Estimated bandwidth
};

// ============================================================================
// Matrix-Free Kernel Interface
// ============================================================================

/**
 * @brief Extended kernel interface for matrix-free operators
 *
 * Matrix-free kernels must implement the action of the operator on a vector
 * without forming the full element matrix.
 */
class IMatrixFreeKernel {
public:
    virtual ~IMatrixFreeKernel() = default;

    /**
     * @brief Compute action of operator on local input
     *
     * Computes y_local = A_local * x_local for a single element.
     *
     * @param context Assembly context with cached data
     * @param x_local Local input vector (values at element DOFs)
     * @param y_local Output: local result vector (same size as x_local)
     */
    virtual void applyLocal(
        const AssemblyContext& context,
        std::span<const Real> x_local,
        std::span<Real> y_local) = 0;

    /**
     * @brief Setup element data for repeated applies
     *
     * Called once per element during setup phase. Kernel can cache
     * physics-related data for efficient apply().
     *
     * @param context Assembly context
     * @param element_data Output: data to cache for this element
     */
    virtual void setupElement(
        const AssemblyContext& /*context*/,
        std::vector<Real>& /*element_data*/)
    {
        // Default: no additional data to cache
    }

    /**
     * @brief Apply using cached element data
     *
     * @param context Assembly context (geometry/basis may be cached)
     * @param element_data Cached element data from setupElement
     * @param x_local Local input vector
     * @param y_local Output: local result
     */
    virtual void applyLocalCached(
        const AssemblyContext& context,
        std::span<const Real> element_data,
        std::span<const Real> x_local,
        std::span<Real> y_local)
    {
        // Default: ignore cached data, call regular apply
        (void)element_data;
        applyLocal(context, x_local, const_cast<std::span<Real>&>(y_local));
    }

    /**
     * @brief Get size of cached element data
     *
     * @param context Context with element info
     * @return Number of Real values to cache per element
     */
    [[nodiscard]] virtual std::size_t elementDataSize(
        const AssemblyContext& /*context*/) const noexcept
    {
        return 0;  // Default: no cached data
    }

    /**
     * @brief Get required data for kernel evaluation
     */
    [[nodiscard]] virtual RequiredData getRequiredData() const noexcept {
        return RequiredData::BasisValues | RequiredData::BasisGradients |
               RequiredData::Jacobians;
    }

    /**
     * @brief Check if kernel supports batched evaluation
     */
    [[nodiscard]] virtual bool supportsBatched() const noexcept { return false; }

    /**
     * @brief Batched apply for multiple elements
     *
     * @param contexts Array of assembly contexts
     * @param x_locals Array of local input vectors
     * @param y_locals Array of local output vectors
     * @param batch_size Number of elements in batch
     */
    virtual void applyBatched(
        std::span<const AssemblyContext*> /*contexts*/,
        std::span<const std::span<const Real>> /*x_locals*/,
        std::span<std::span<Real>> /*y_locals*/,
        int /*batch_size*/)
    {
        FE_THROW(NotImplementedException, "Batched apply not implemented for this kernel");
    }
};

// ============================================================================
// Matrix-Free Operator
// ============================================================================

/**
 * @brief Matrix-free linear operator
 *
 * Represents a linear operator A that can compute y = A*x without storing A.
 * Used with iterative solvers (CG, GMRES, etc.).
 */
class MatrixFreeOperator {
public:
    /**
     * @brief Apply operator: y = A * x
     *
     * @param x Input vector
     * @param y Output vector (must be pre-allocated)
     */
    virtual void apply(std::span<const Real> x, std::span<Real> y) = 0;

    /**
     * @brief Apply operator: y += A * x
     */
    virtual void applyAdd(std::span<const Real> x, std::span<Real> y) = 0;

    /**
     * @brief Get number of rows
     */
    [[nodiscard]] virtual GlobalIndex numRows() const noexcept = 0;

    /**
     * @brief Get number of columns
     */
    [[nodiscard]] virtual GlobalIndex numCols() const noexcept = 0;

    /**
     * @brief Get diagonal (for preconditioning)
     *
     * @param diag Output: diagonal entries
     */
    virtual void getDiagonal(std::span<Real> diag) = 0;

    virtual ~MatrixFreeOperator() = default;
};

// ============================================================================
// Cached Element Data
// ============================================================================

/**
 * @brief Cached data for a single element
 */
struct MatrixFreeElementData {
    GlobalIndex cell_id{-1};              ///< Element identifier
    ElementType cell_type{ElementType::Unknown};

    // DOF information
    std::vector<GlobalIndex> dofs;        ///< Global DOF indices

    // Geometry cache (if enabled)
    std::vector<Real> jacobians;          ///< Jacobians at quadrature points
    std::vector<Real> det_jacobians;      ///< Jacobian determinants
    std::vector<Real> inv_jacobians;      ///< Inverse Jacobians

    // Basis cache (if enabled)
    std::vector<Real> basis_values;       ///< Basis values at quadrature points
    std::vector<Real> basis_gradients;    ///< Basis gradients (reference)
    std::vector<Real> physical_gradients; ///< Basis gradients (physical)

    // Quadrature
    std::vector<Real> quadrature_weights; ///< JxW at each quadrature point

    // Physics-specific data
    std::vector<Real> kernel_data;        ///< Kernel-specific cached data
};

// ============================================================================
// Matrix-Free Assembler
// ============================================================================

/**
 * @brief Matrix-free assembly and operator application
 *
 * MatrixFreeAssembler provides matrix-free operator evaluation. It can operate
 * in two modes:
 *
 * 1. Setup + Apply: Call setup() once, then apply() multiple times
 * 2. Direct: Call applyDirect() which recomputes everything
 *
 * Usage:
 * @code
 *   MatrixFreeAssembler assembler;
 *   assembler.setMesh(mesh);
 *   assembler.setDofMap(dof_map);
 *   assembler.setSpace(space);
 *   assembler.setKernel(kernel);
 *
 *   // Setup (precompute geometry, basis, etc.)
 *   assembler.setup();
 *
 *   // Apply multiple times (e.g., in iterative solver)
 *   assembler.apply(x, y);  // y = A * x
 *
 *   // Or with iterative solver interface
 *   auto op = assembler.getOperator();
 *   solver.setOperator(op);
 *   solver.solve(b, x);
 * @endcode
 */
class MatrixFreeAssembler {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor
     */
    MatrixFreeAssembler();

    /**
     * @brief Construct with options
     */
    explicit MatrixFreeAssembler(const MatrixFreeOptions& options);

    /**
     * @brief Destructor
     */
    ~MatrixFreeAssembler();

    /**
     * @brief Move constructor
     */
    MatrixFreeAssembler(MatrixFreeAssembler&& other) noexcept;

    /**
     * @brief Move assignment
     */
    MatrixFreeAssembler& operator=(MatrixFreeAssembler&& other) noexcept;

    // Non-copyable
    MatrixFreeAssembler(const MatrixFreeAssembler&) = delete;
    MatrixFreeAssembler& operator=(const MatrixFreeAssembler&) = delete;

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Set mesh access
     */
    void setMesh(const IMeshAccess& mesh);

    /**
     * @brief Set DOF map
     */
    void setDofMap(const dofs::DofMap& dof_map);

    /**
     * @brief Set function space
     */
    void setSpace(const spaces::FunctionSpace& space);

    /**
     * @brief Set test and trial spaces (for non-symmetric operators)
     */
    void setSpaces(const spaces::FunctionSpace& test_space,
                   const spaces::FunctionSpace& trial_space);

    /**
     * @brief Set the matrix-free kernel
     */
    void setKernel(IMatrixFreeKernel& kernel);

    /**
     * @brief Set constraints
     */
    void setConstraints(const constraints::AffineConstraints& constraints);

    /**
     * @brief Set options
     */
    void setOptions(const MatrixFreeOptions& options);

    /**
     * @brief Get current options
     */
    [[nodiscard]] const MatrixFreeOptions& getOptions() const noexcept {
        return options_;
    }

    // =========================================================================
    // Setup
    // =========================================================================

    /**
     * @brief Setup/precompute data for efficient applies
     *
     * Must be called before apply() if using partial assembly.
     * Can be skipped if using applyDirect().
     */
    void setup();

    /**
     * @brief Check if setup has been performed
     */
    [[nodiscard]] bool isSetup() const noexcept { return is_setup_; }

    /**
     * @brief Invalidate setup (call after mesh/space changes)
     */
    void invalidateSetup();

    // =========================================================================
    // Operator Application
    // =========================================================================

    /**
     * @brief Apply operator: y = A * x
     *
     * Uses cached data from setup() for efficient evaluation.
     *
     * @param x Input vector
     * @param y Output vector (will be overwritten)
     */
    void apply(std::span<const Real> x, std::span<Real> y);

    /**
     * @brief Apply operator: y += A * x
     *
     * @param x Input vector
     * @param y Output vector (added to)
     */
    void applyAdd(std::span<const Real> x, std::span<Real> y);

    /**
     * @brief Apply without using cached data
     *
     * Recomputes everything from scratch. Slower but doesn't require setup().
     *
     * @param x Input vector
     * @param y Output vector
     */
    void applyDirect(std::span<const Real> x, std::span<Real> y);

    /**
     * @brief Get operator diagonal (for preconditioning)
     *
     * @param diag Output: diagonal entries
     */
    void getDiagonal(std::span<Real> diag);

    // =========================================================================
    // Operator Interface
    // =========================================================================

    /**
     * @brief Get linear operator interface
     *
     * Returns a MatrixFreeOperator that can be used with iterative solvers.
     */
    std::shared_ptr<MatrixFreeOperator> getOperator();

    /**
     * @brief Get number of rows (test DOFs)
     */
    [[nodiscard]] GlobalIndex numRows() const noexcept;

    /**
     * @brief Get number of columns (trial DOFs)
     */
    [[nodiscard]] GlobalIndex numCols() const noexcept;

    // =========================================================================
    // Residual Assembly
    // =========================================================================

    /**
     * @brief Assemble residual vector
     *
     * For nonlinear problems, assembles F(u) using the current solution
     * set via setCurrentSolution().
     *
     * @param residual Output: residual vector
     */
    void assembleResidual(std::span<Real> residual);

    /**
     * @brief Set current solution for residual evaluation
     *
     * @param solution Current solution vector
     */
    void setCurrentSolution(std::span<const Real> solution);

    // =========================================================================
    // Query and Statistics
    // =========================================================================

    /**
     * @brief Check if assembler is configured
     */
    [[nodiscard]] bool isConfigured() const noexcept;

    /**
     * @brief Get statistics
     */
    [[nodiscard]] const MatrixFreeStats& getStats() const noexcept {
        return stats_;
    }

    /**
     * @brief Reset statistics
     */
    void resetStats();

private:
    // =========================================================================
    // Internal Implementation
    // =========================================================================

    /**
     * @brief Cache geometry data for all elements
     */
    void cacheGeometryData();

    /**
     * @brief Cache basis evaluations for all elements
     */
    void cacheBasisData();

    /**
     * @brief Cache kernel-specific data for all elements
     */
    void cacheKernelData();

    /**
     * @brief Apply for a single element
     */
    void applyElement(
        const MatrixFreeElementData& elem_data,
        std::span<const Real> x_global,
        std::span<Real> y_global);

    /**
     * @brief Apply constraints to input/output vectors
     */
    void applyConstraints(std::span<Real> y) const;

    /**
     * @brief Zero constrained entries
     */
    void zeroConstrainedEntries(std::span<Real> y) const;

    // =========================================================================
    // Data Members
    // =========================================================================

    // Configuration
    MatrixFreeOptions options_;
    const IMeshAccess* mesh_{nullptr};
    const dofs::DofMap* dof_map_{nullptr};
    const spaces::FunctionSpace* test_space_{nullptr};
    const spaces::FunctionSpace* trial_space_{nullptr};
    IMatrixFreeKernel* kernel_{nullptr};
    const constraints::AffineConstraints* constraints_{nullptr};

    // Cached element data
    std::vector<MatrixFreeElementData> element_cache_;

    // Assembly infrastructure
    std::unique_ptr<AssemblyLoop> loop_;

    // Thread-local storage
    std::vector<std::unique_ptr<AssemblyContext>> thread_contexts_;
    std::vector<std::vector<Real>> thread_x_local_;
    std::vector<std::vector<Real>> thread_y_local_;

    // Current solution (for nonlinear problems)
    std::vector<Real> current_solution_;

    // State
    bool is_setup_{false};

    // Dimensions
    GlobalIndex num_rows_{0};
    GlobalIndex num_cols_{0};

    // Statistics
    MatrixFreeStats stats_;
};

// ============================================================================
// Convenience Factory Functions
// ============================================================================

/**
 * @brief Create matrix-free assembler
 */
std::unique_ptr<MatrixFreeAssembler> createMatrixFreeAssembler(
    const MatrixFreeOptions& options = {});

/**
 * @brief Create matrix-free operator from standard kernel
 *
 * Wraps a standard AssemblyKernel as a matrix-free kernel.
 *
 * @param kernel Standard assembly kernel
 * @return Matrix-free kernel wrapper
 */
std::unique_ptr<IMatrixFreeKernel> wrapAsMatrixFreeKernel(
    AssemblyKernel& kernel);

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_MATRIX_FREE_ASSEMBLER_H
