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

#ifndef SVMP_FE_ASSEMBLY_ASSEMBLY_CONTEXT_H
#define SVMP_FE_ASSEMBLY_ASSEMBLY_CONTEXT_H

/**
 * @file AssemblyContext.h
 * @brief Per-thread assembly context and scratch space
 *
 * AssemblyContext provides data to AssemblyKernels during element computation.
 * It serves as the interface between the assembly infrastructure and physics
 * kernels, providing:
 *
 * - Quadrature points and integration weights
 * - Basis function values and gradients
 * - Geometry (Jacobians, physical coordinates, normals)
 * - Solution values at quadrature points (for nonlinear problems)
 * - Scratch space for intermediate computations
 *
 * Design principles:
 * - One context per thread (thread-local storage)
 * - Prepared by assembler based on kernel's RequiredData
 * - Read-only interface for kernels (const methods)
 * - Supports batching/vectorization where possible
 *
 * Module boundary:
 * - Assembly OWNS: context preparation, data caching, scratch management
 * - Physics ACCESSES: read-only data during kernel execution
 *
 * @see AssemblyKernel for the kernel interface
 * @see Assembler for orchestration
 */

#include "Core/Types.h"
#include "AssemblyKernel.h"

#include <vector>
#include <span>
#include <array>
#include <memory>
#include <cstdint>

namespace svmp {
namespace FE {

// Forward declarations
namespace elements {
    class Element;
}

namespace spaces {
    class FunctionSpace;
}

namespace assembly {

// ============================================================================
// Assembly Context Type
// ============================================================================

/**
 * @brief Type of assembly context (cell vs face)
 */
enum class ContextType : std::uint8_t {
    Cell,           ///< Volume/cell integration context
    BoundaryFace,   ///< Boundary face integration context
    InteriorFace    ///< Interior face integration context (DG)
};

// ============================================================================
// Assembly Context
// ============================================================================

/**
 * @brief Context object providing data to assembly kernels
 *
 * AssemblyContext is prepared by the assembler before calling kernel methods.
 * It provides all the data a kernel might need for element computation.
 *
 * Usage in kernel:
 * @code
 * void MyKernel::computeCell(const AssemblyContext& ctx, KernelOutput& output) {
 *     const auto n_dofs = ctx.numTestDofs();
 *     const auto n_qpts = ctx.numQuadraturePoints();
 *
 *     for (LocalIndex q = 0; q < n_qpts; ++q) {
 *         Real w = ctx.integrationWeight(q);
 *         auto x = ctx.physicalPoint(q);
 *
 *         for (LocalIndex i = 0; i < n_dofs; ++i) {
 *             Real phi_i = ctx.basisValue(i, q);
 *             auto grad_i = ctx.physicalGradient(i, q);
 *             // ... compute contributions
 *         }
 *     }
 * }
 * @endcode
 *
 * Thread safety:
 * - Each thread should have its own AssemblyContext instance
 * - The context is prepared (written) by the assembler, then read by kernels
 * - Multiple kernels can read from the same context concurrently
 */
class AssemblyContext {
public:
    // =========================================================================
    // Types
    // =========================================================================

    using Point3D = std::array<Real, 3>;
    using Vector3D = std::array<Real, 3>;
    using Matrix3x3 = std::array<std::array<Real, 3>, 3>;

    // =========================================================================
    // Construction and Setup
    // =========================================================================

    /**
     * @brief Default constructor
     */
    AssemblyContext();

    /**
     * @brief Destructor
     */
    ~AssemblyContext();

    /**
     * @brief Move constructor
     */
    AssemblyContext(AssemblyContext&& other) noexcept;

    /**
     * @brief Move assignment
     */
    AssemblyContext& operator=(AssemblyContext&& other) noexcept;

    // Non-copyable (for efficiency)
    AssemblyContext(const AssemblyContext&) = delete;
    AssemblyContext& operator=(const AssemblyContext&) = delete;

    /**
     * @brief Reserve storage for expected sizes
     *
     * @param max_dofs Maximum DOFs per element
     * @param max_qpts Maximum quadrature points
     * @param dim Spatial dimension (2 or 3)
     */
    void reserve(LocalIndex max_dofs, LocalIndex max_qpts, int dim);

    /**
     * @brief Configure for a specific element
     *
     * Called by assembler before kernel invocation.
     *
     * @param cell_id Current cell index
     * @param test_element Test space element
     * @param trial_element Trial space element (same as test for square)
     * @param required_data Data flags from kernel
     */
    void configure(
        GlobalIndex cell_id,
        const elements::Element& test_element,
        const elements::Element& trial_element,
        RequiredData required_data);

    /**
     * @brief Configure for a face
     *
     * @param face_id Face index
     * @param cell_id Adjacent cell index
     * @param local_face_id Local face index within cell
     * @param element Element definition
     * @param required_data Data flags from kernel
     * @param type Face type (boundary or interior)
     */
    void configureFace(
        GlobalIndex face_id,
        GlobalIndex cell_id,
        LocalIndex local_face_id,
        const elements::Element& element,
        RequiredData required_data,
        ContextType type);

    /**
     * @brief Clear the context for reuse
     */
    void clear();

    // =========================================================================
    // Basic Properties (always available)
    // =========================================================================

    /**
     * @brief Get context type (cell, boundary face, interior face)
     */
    [[nodiscard]] ContextType contextType() const noexcept { return type_; }

    /**
     * @brief Get current cell ID
     */
    [[nodiscard]] GlobalIndex cellId() const noexcept { return cell_id_; }

    /**
     * @brief Get spatial dimension
     */
    [[nodiscard]] int dimension() const noexcept { return dim_; }

    /**
     * @brief Get number of test DOFs (rows)
     */
    [[nodiscard]] LocalIndex numTestDofs() const noexcept { return n_test_dofs_; }

    /**
     * @brief Get number of trial DOFs (columns)
     */
    [[nodiscard]] LocalIndex numTrialDofs() const noexcept { return n_trial_dofs_; }

    /**
     * @brief Get number of quadrature points
     */
    [[nodiscard]] LocalIndex numQuadraturePoints() const noexcept { return n_qpts_; }

    /**
     * @brief Check if test and trial spaces are the same
     */
    [[nodiscard]] bool isSquare() const noexcept {
        return n_test_dofs_ == n_trial_dofs_;
    }

    // =========================================================================
    // Quadrature Data
    // =========================================================================

    /**
     * @brief Get reference quadrature point
     *
     * @param q Quadrature point index
     * @return Reference coordinates
     */
    [[nodiscard]] Point3D quadraturePoint(LocalIndex q) const;

    /**
     * @brief Get all reference quadrature points
     */
    [[nodiscard]] std::span<const Point3D> quadraturePoints() const noexcept {
        return {quad_points_.data(), static_cast<std::size_t>(n_qpts_)};
    }

    /**
     * @brief Get quadrature weight (reference element)
     *
     * @param q Quadrature point index
     * @return Quadrature weight
     */
    [[nodiscard]] Real quadratureWeight(LocalIndex q) const;

    /**
     * @brief Get integration weight (includes Jacobian determinant)
     *
     * Returns w_q * |J_q| for direct use in integration.
     *
     * @param q Quadrature point index
     * @return Integration weight
     */
    [[nodiscard]] Real integrationWeight(LocalIndex q) const;

    /**
     * @brief Get all integration weights
     */
    [[nodiscard]] std::span<const Real> integrationWeights() const noexcept {
        return {integration_weights_.data(), static_cast<std::size_t>(n_qpts_)};
    }

    // =========================================================================
    // Geometry Data
    // =========================================================================

    /**
     * @brief Get physical coordinate at quadrature point
     *
     * @param q Quadrature point index
     * @return Physical (x, y, z) coordinates
     */
    [[nodiscard]] Point3D physicalPoint(LocalIndex q) const;

    /**
     * @brief Get all physical points
     */
    [[nodiscard]] std::span<const Point3D> physicalPoints() const noexcept {
        return {physical_points_.data(), static_cast<std::size_t>(n_qpts_)};
    }

    /**
     * @brief Get Jacobian determinant at quadrature point
     *
     * @param q Quadrature point index
     * @return |J(xi_q)|
     */
    [[nodiscard]] Real jacobianDet(LocalIndex q) const;

    /**
     * @brief Get Jacobian matrix at quadrature point
     *
     * J[i][j] = dx_i / dxi_j
     *
     * @param q Quadrature point index
     * @return 3x3 Jacobian matrix
     */
    [[nodiscard]] Matrix3x3 jacobian(LocalIndex q) const;

    /**
     * @brief Get inverse Jacobian matrix at quadrature point
     *
     * @param q Quadrature point index
     * @return 3x3 inverse Jacobian matrix
     */
    [[nodiscard]] Matrix3x3 inverseJacobian(LocalIndex q) const;

    /**
     * @brief Get surface normal at quadrature point (face contexts only)
     *
     * @param q Quadrature point index
     * @return Outward unit normal vector
     */
    [[nodiscard]] Vector3D normal(LocalIndex q) const;

    // =========================================================================
    // Basis Function Data (Test Space)
    // =========================================================================

    /**
     * @brief Get test basis function value at quadrature point
     *
     * @param i Test basis function index
     * @param q Quadrature point index
     * @return phi_i(xi_q)
     */
    [[nodiscard]] Real basisValue(LocalIndex i, LocalIndex q) const;

    /**
     * @brief Get test basis function values at all quadrature points
     *
     * @param i Test basis function index
     * @return Values at all quadrature points
     */
    [[nodiscard]] std::span<const Real> basisValues(LocalIndex i) const;

    /**
     * @brief Get reference gradient of test basis function
     *
     * @param i Test basis function index
     * @param q Quadrature point index
     * @return grad_xi(phi_i) at xi_q
     */
    [[nodiscard]] Vector3D referenceGradient(LocalIndex i, LocalIndex q) const;

    /**
     * @brief Get physical gradient of test basis function
     *
     * grad_x(phi_i) = J^{-T} * grad_xi(phi_i)
     *
     * @param i Test basis function index
     * @param q Quadrature point index
     * @return grad_x(phi_i) at x_q
     */
    [[nodiscard]] Vector3D physicalGradient(LocalIndex i, LocalIndex q) const;

    // =========================================================================
    // Trial Basis Function Data (for rectangular assembly)
    // =========================================================================

    /**
     * @brief Get trial basis function value
     *
     * @param j Trial basis function index
     * @param q Quadrature point index
     * @return psi_j(xi_q)
     */
    [[nodiscard]] Real trialBasisValue(LocalIndex j, LocalIndex q) const;

    /**
     * @brief Get physical gradient of trial basis function
     *
     * @param j Trial basis function index
     * @param q Quadrature point index
     * @return grad_x(psi_j)
     */
    [[nodiscard]] Vector3D trialPhysicalGradient(LocalIndex j, LocalIndex q) const;

    // =========================================================================
    // Solution Data (for nonlinear problems)
    // =========================================================================

    /**
     * @brief Set solution coefficients for evaluation
     *
     * @param coefficients DOF coefficients for current solution
     */
    void setSolutionCoefficients(std::span<const Real> coefficients);

    /**
     * @brief Get solution value at quadrature point
     *
     * u_h(x_q) = sum_i U_i * phi_i(x_q)
     *
     * @param q Quadrature point index
     * @return Solution value
     */
    [[nodiscard]] Real solutionValue(LocalIndex q) const;

    /**
     * @brief Get solution gradient at quadrature point
     *
     * @param q Quadrature point index
     * @return Physical gradient of solution
     */
    [[nodiscard]] Vector3D solutionGradient(LocalIndex q) const;

    /**
     * @brief Check if solution data is available
     */
    [[nodiscard]] bool hasSolutionData() const noexcept {
        return !solution_values_.empty();
    }

    // =========================================================================
    // Face-Specific Data
    // =========================================================================

    /**
     * @brief Get local face index within cell
     */
    [[nodiscard]] LocalIndex localFaceId() const noexcept { return local_face_id_; }

    /**
     * @brief Get boundary marker (boundary faces only)
     */
    [[nodiscard]] int boundaryMarker() const noexcept { return boundary_marker_; }

    /**
     * @brief Set boundary marker
     */
    void setBoundaryMarker(int marker) { boundary_marker_ = marker; }

    // =========================================================================
    // Data Setting (called by assembler)
    // =========================================================================

    /**
     * @brief Set quadrature data
     */
    void setQuadratureData(
        std::span<const Point3D> points,
        std::span<const Real> weights);

    /**
     * @brief Set physical points
     */
    void setPhysicalPoints(std::span<const Point3D> points);

    /**
     * @brief Set Jacobian data
     */
    void setJacobianData(
        std::span<const Matrix3x3> jacobians,
        std::span<const Matrix3x3> inverse_jacobians,
        std::span<const Real> determinants);

    /**
     * @brief Set integration weights (pre-multiplied by Jacobian det)
     */
    void setIntegrationWeights(std::span<const Real> weights);

    /**
     * @brief Set test basis function data
     */
    void setTestBasisData(
        LocalIndex n_dofs,
        std::span<const Real> values,
        std::span<const Vector3D> gradients);

    /**
     * @brief Set trial basis function data (for rectangular)
     */
    void setTrialBasisData(
        LocalIndex n_dofs,
        std::span<const Real> values,
        std::span<const Vector3D> gradients);

    /**
     * @brief Set physical gradients (after Jacobian transformation)
     */
    void setPhysicalGradients(
        std::span<const Vector3D> test_gradients,
        std::span<const Vector3D> trial_gradients);

    /**
     * @brief Set normal vectors (for face contexts)
     */
    void setNormals(std::span<const Vector3D> normals);

    /**
     * @brief Set solution values at quadrature points
     */
    void setSolutionValues(std::span<const Real> values);

    /**
     * @brief Set solution gradients at quadrature points
     */
    void setSolutionGradients(std::span<const Vector3D> gradients);

private:
    // Context type and identification
    ContextType type_{ContextType::Cell};
    GlobalIndex cell_id_{-1};
    GlobalIndex face_id_{-1};
    LocalIndex local_face_id_{0};
    int boundary_marker_{-1};
    int dim_{3};

    // DOF counts
    LocalIndex n_test_dofs_{0};
    LocalIndex n_trial_dofs_{0};
    LocalIndex n_qpts_{0};

    // Quadrature data
    std::vector<Point3D> quad_points_;
    std::vector<Real> quad_weights_;
    std::vector<Real> integration_weights_;

    // Geometry data
    std::vector<Point3D> physical_points_;
    std::vector<Matrix3x3> jacobians_;
    std::vector<Matrix3x3> inverse_jacobians_;
    std::vector<Real> jacobian_dets_;
    std::vector<Vector3D> normals_;

    // Test basis data (n_test_dofs * n_qpts arrays)
    std::vector<Real> test_basis_values_;           // [i * n_qpts + q]
    std::vector<Vector3D> test_ref_gradients_;      // [i * n_qpts + q]
    std::vector<Vector3D> test_phys_gradients_;     // [i * n_qpts + q]

    // Trial basis data (may be same as test for square)
    std::vector<Real> trial_basis_values_;
    std::vector<Vector3D> trial_ref_gradients_;
    std::vector<Vector3D> trial_phys_gradients_;
    bool trial_is_test_{true};  // Optimization flag

    // Solution data (for nonlinear problems)
    std::vector<Real> solution_coefficients_;
    std::vector<Real> solution_values_;
    std::vector<Vector3D> solution_gradients_;
};

// ============================================================================
// Assembly Context Pool (for thread safety)
// ============================================================================

/**
 * @brief Pool of AssemblyContext objects for multi-threaded assembly
 *
 * Each thread gets its own context to avoid synchronization overhead.
 */
class AssemblyContextPool {
public:
    /**
     * @brief Construct pool with given capacity
     *
     * @param num_threads Number of threads (contexts)
     * @param max_dofs Maximum DOFs per element
     * @param max_qpts Maximum quadrature points
     * @param dim Spatial dimension
     */
    AssemblyContextPool(int num_threads, LocalIndex max_dofs,
                        LocalIndex max_qpts, int dim);

    /**
     * @brief Get context for current thread
     *
     * @param thread_id Thread index (0 to num_threads-1)
     * @return Reference to thread's context
     */
    [[nodiscard]] AssemblyContext& getContext(int thread_id);

    /**
     * @brief Get number of contexts in pool
     */
    [[nodiscard]] int size() const noexcept {
        return static_cast<int>(contexts_.size());
    }

private:
    std::vector<std::unique_ptr<AssemblyContext>> contexts_;
};

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_ASSEMBLY_CONTEXT_H
