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
#include "Core/AlignedAllocator.h"
#include "Core/ParameterValue.h"
#include "AssemblyKernel.h"
#include "Assembly/JIT/FieldSolutions.h"
#include "TimeIntegrationContext.h"

#include <vector>
#include <span>
#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <algorithm>
#include <iterator>
#include <stdexcept>

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
     * @brief Configure for a specific element (space-aware)
     *
     * Preferred overload when assembling vector-valued spaces (e.g. ProductSpace)
     * because the prototype element DOF count may differ from the FunctionSpace
     * DOF count.
     */
    void configure(
        GlobalIndex cell_id,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
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
     * @brief Configure for a face (space-aware)
     */
    void configureFace(
        GlobalIndex face_id,
        GlobalIndex cell_id,
        LocalIndex local_face_id,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
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
     * @brief Get cell domain/subdomain ID (mesh-provided)
     *
     * This value is populated by the assembler from `IMeshAccess::getCellDomainId(cell_id)`.
     */
    [[nodiscard]] int cellDomainId() const noexcept { return cell_domain_id_; }

    /**
     * @brief Get current face ID (face contexts only)
     */
    [[nodiscard]] GlobalIndex faceId() const noexcept { return face_id_; }

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
    // Space Metadata
    // =========================================================================

    [[nodiscard]] FieldType testFieldType() const noexcept { return test_field_type_; }
    [[nodiscard]] FieldType trialFieldType() const noexcept { return trial_field_type_; }

    [[nodiscard]] Continuity testContinuity() const noexcept { return test_continuity_; }
    [[nodiscard]] Continuity trialContinuity() const noexcept { return trial_continuity_; }

    [[nodiscard]] bool testUsesVectorBasis() const noexcept { return test_is_vector_basis_; }
    [[nodiscard]] bool trialUsesVectorBasis() const noexcept { return trial_is_vector_basis_; }

    [[nodiscard]] int testValueDimension() const noexcept { return test_value_dim_; }
    [[nodiscard]] int trialValueDimension() const noexcept { return trial_value_dim_; }

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
        return quad_points_;
    }

    /**
     * @brief Get quadrature weight (reference element)
     *
     * @param q Quadrature point index
     * @return Quadrature weight
     */
    [[nodiscard]] Real quadratureWeight(LocalIndex q) const;

    /**
     * @brief Get all quadrature weights (reference element)
     */
    [[nodiscard]] std::span<const Real> quadratureWeights() const noexcept { return quad_weights_; }

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
        return integration_weights_;
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
        return physical_points_;
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
     * @brief Get all Jacobian matrices (one per quadrature point)
     */
    [[nodiscard]] std::span<const Matrix3x3> jacobians() const noexcept {
        return jacobians_;
    }

    /**
     * @brief Get inverse Jacobian matrix at quadrature point
     *
     * @param q Quadrature point index
     * @return 3x3 inverse Jacobian matrix
     */
    [[nodiscard]] Matrix3x3 inverseJacobian(LocalIndex q) const;

    /**
     * @brief Get all inverse Jacobians (one per quadrature point)
     */
    [[nodiscard]] std::span<const Matrix3x3> inverseJacobians() const noexcept {
        return inverse_jacobians_;
    }

    /**
     * @brief Get all Jacobian determinants (one per quadrature point)
     */
    [[nodiscard]] std::span<const Real> jacobianDets() const noexcept {
        return jacobian_dets_;
    }

    /**
     * @brief Get surface normal at quadrature point (face contexts only)
     *
     * @param q Quadrature point index
     * @return Outward unit normal vector
     */
    [[nodiscard]] Vector3D normal(LocalIndex q) const;

    /**
     * @brief Get all surface normals (face contexts only)
     */
    [[nodiscard]] std::span<const Vector3D> normals() const noexcept {
        return normals_;
    }

    // Interleaved quadrature-point geometry (SoA->AoSoA bridge for JIT kernels):
    //   [q * stride +  0.. 2] physical point xyz
    //   [q * stride +  3..11] Jacobian (row-major 3x3)
    //   [q * stride + 12..20] inverse Jacobian (row-major 3x3)
    //   [q * stride +    21 ] det(J)
    //   [q * stride + 22..24] normal xyz (zero for cell contexts)
    static constexpr std::uint32_t kInterleavedQPointGeometryStride{25u};
    static constexpr std::uint32_t kInterleavedQPointPhysicalOffset{0u};
    static constexpr std::uint32_t kInterleavedQPointJacobianOffset{3u};
    static constexpr std::uint32_t kInterleavedQPointInverseJacobianOffset{12u};
    static constexpr std::uint32_t kInterleavedQPointDetOffset{21u};
    static constexpr std::uint32_t kInterleavedQPointNormalOffset{22u};

    [[nodiscard]] std::span<const Real> interleavedQPointGeometryRaw() const noexcept
    {
        return interleaved_qpoint_geometry_;
    }

    // =========================================================================
    // Entity Measures (optional; prepared if RequiredData::EntityMeasures)
    // =========================================================================

    /**
     * @brief Cell diameter h (max vertex distance)
     *
     * Available for all context types (cell/boundary/interior face) since all
     * are associated with a cell.
     */
    [[nodiscard]] Real cellDiameter() const noexcept { return cell_diameter_; }

    /**
     * @brief Cell volume/measure
     *
     * Defined for cell contexts. Using this for face contexts is an error.
     */
    [[nodiscard]] Real cellVolume() const;

    /**
     * @brief Facet area/measure
     *
     * Defined for face contexts. Using this for cell contexts is an error.
     */
    [[nodiscard]] Real facetArea() const;

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
     * @brief Raw test basis values storage (layout: [i * n_qpts + q])
     *
     * Intended for JIT/kernel ABI packing. Empty when BasisValues were not requested.
     */
    [[nodiscard]] std::span<const Real> testBasisValuesRaw() const noexcept {
        return test_basis_values_;
    }

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

    /**
     * @brief Raw test physical gradients storage (layout: [q * n_test_dofs + i])
     *
     * Intended for JIT/kernel ABI packing. Empty when PhysicalGradients were not requested.
     */
    [[nodiscard]] std::span<const Vector3D> testPhysicalGradientsRaw() const noexcept {
        return test_phys_gradients_;
    }

    // =========================================================================
    // Vector Basis Function Data (Test Space, H(curl)/H(div))
    // =========================================================================

    /**
     * @brief Get test vector basis function value (physical space) at quadrature point
     */
    [[nodiscard]] Vector3D basisVectorValue(LocalIndex i, LocalIndex q) const;

    /**
     * @brief Raw test vector basis values storage (layout: [i * n_qpts + q])
     */
    [[nodiscard]] std::span<const Vector3D> testBasisVectorValuesRaw() const noexcept
    {
        return test_basis_vector_values_;
    }

    /**
     * @brief Get curl of a test basis function (physical space) at quadrature point (H(curl))
     */
    [[nodiscard]] Vector3D basisCurl(LocalIndex i, LocalIndex q) const;

    /**
     * @brief Raw test basis curls storage (layout: [i * n_qpts + q])
     */
    [[nodiscard]] std::span<const Vector3D> testBasisCurlsRaw() const noexcept
    {
        return test_basis_curls_;
    }

    /**
     * @brief Get divergence of a test basis function (physical space) at quadrature point (H(div))
     */
    [[nodiscard]] Real basisDivergence(LocalIndex i, LocalIndex q) const;

    /**
     * @brief Raw test basis divergences storage (layout: [i * n_qpts + q])
     */
    [[nodiscard]] std::span<const Real> testBasisDivergencesRaw() const noexcept
    {
        return test_basis_divergences_;
    }

    // =========================================================================
    // Basis Hessians (optional; prepared if RequiredData::BasisHessians)
    // =========================================================================

    /**
     * @brief Get reference Hessian of test basis function (d^2 phi / dxi^2)
     */
    [[nodiscard]] Matrix3x3 referenceHessian(LocalIndex i, LocalIndex q) const;

    /**
     * @brief Get physical Hessian of test basis function (d^2 phi / dx^2)
     *
     * For affine mappings: H_x(phi) = J^{-T} * H_xi(phi) * J^{-1}.
     */
    [[nodiscard]] Matrix3x3 physicalHessian(LocalIndex i, LocalIndex q) const;

    /**
     * @brief Raw test physical Hessians storage (layout: [i * n_qpts + q])
     *
     * Intended for JIT/kernel ABI packing. Empty when BasisHessians were not requested.
     */
    [[nodiscard]] std::span<const Matrix3x3> testPhysicalHessiansRaw() const noexcept
    {
        return test_phys_hessians_;
    }

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
     * @brief Raw trial basis values storage (layout: [j * n_qpts + q])
     *
     * When the trial space aliases the test space, this returns the test storage.
     */
    [[nodiscard]] std::span<const Real> trialBasisValuesRaw() const noexcept {
        return trial_is_test_ ? std::span<const Real>(test_basis_values_) : std::span<const Real>(trial_basis_values_);
    }

    // =========================================================================
    // Vector Basis Function Data (Trial Space, H(curl)/H(div))
    // =========================================================================

    /**
     * @brief Get trial vector basis function value (physical space) at quadrature point
     */
    [[nodiscard]] Vector3D trialBasisVectorValue(LocalIndex j, LocalIndex q) const;

    /**
     * @brief Raw trial vector basis values storage (layout: [j * n_qpts + q])
     *
     * When the trial space aliases the test space, this returns the test storage.
     */
    [[nodiscard]] std::span<const Vector3D> trialBasisVectorValuesRaw() const noexcept
    {
        return trial_is_test_ ? std::span<const Vector3D>(test_basis_vector_values_)
                              : std::span<const Vector3D>(trial_basis_vector_values_);
    }

    /**
     * @brief Get curl of a trial basis function (physical space) at quadrature point (H(curl))
     */
    [[nodiscard]] Vector3D trialBasisCurl(LocalIndex j, LocalIndex q) const;

    /**
     * @brief Raw trial basis curls storage (layout: [j * n_qpts + q])
     */
    [[nodiscard]] std::span<const Vector3D> trialBasisCurlsRaw() const noexcept
    {
        return trial_is_test_ ? std::span<const Vector3D>(test_basis_curls_)
                              : std::span<const Vector3D>(trial_basis_curls_);
    }

    /**
     * @brief Get divergence of a trial basis function (physical space) at quadrature point (H(div))
     */
    [[nodiscard]] Real trialBasisDivergence(LocalIndex j, LocalIndex q) const;

    /**
     * @brief Raw trial basis divergences storage (layout: [j * n_qpts + q])
     */
    [[nodiscard]] std::span<const Real> trialBasisDivergencesRaw() const noexcept
    {
        return trial_is_test_ ? std::span<const Real>(test_basis_divergences_)
                              : std::span<const Real>(trial_basis_divergences_);
    }

    /**
     * @brief Get physical gradient of trial basis function
     *
     * @param j Trial basis function index
     * @param q Quadrature point index
     * @return grad_x(psi_j)
     */
    [[nodiscard]] Vector3D trialPhysicalGradient(LocalIndex j, LocalIndex q) const;

    /**
     * @brief Raw trial physical gradients storage (layout: [q * n_trial_dofs + j])
     *
     * When the trial space aliases the test space, this returns the test storage.
     */
    [[nodiscard]] std::span<const Vector3D> trialPhysicalGradientsRaw() const noexcept {
        return trial_is_test_ ? std::span<const Vector3D>(test_phys_gradients_) : std::span<const Vector3D>(trial_phys_gradients_);
    }

    /**
     * @brief Get reference Hessian of trial basis function (d^2 psi / dxi^2)
     */
    [[nodiscard]] Matrix3x3 trialReferenceHessian(LocalIndex j, LocalIndex q) const;

    /**
     * @brief Get physical Hessian of trial basis function (d^2 psi / dx^2)
     */
    [[nodiscard]] Matrix3x3 trialPhysicalHessian(LocalIndex j, LocalIndex q) const;

    /**
     * @brief Raw trial physical Hessians storage (layout: [j * n_qpts + q])
     *
     * When the trial space aliases the test space, this returns the test storage.
     */
    [[nodiscard]] std::span<const Matrix3x3> trialPhysicalHessiansRaw() const noexcept
    {
        return trial_is_test_ ? std::span<const Matrix3x3>(test_phys_hessians_)
                              : std::span<const Matrix3x3>(trial_phys_hessians_);
    }

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
     * @brief Access element-local solution coefficients (DOF values)
     *
     * Size is `numTrialDofs()` for the active context.
     */
    [[nodiscard]] std::span<const Real> solutionCoefficients() const noexcept { return solution_coefficients_; }

    [[nodiscard]] bool hasSolutionCoefficients() const noexcept { return !solution_coefficients_.empty(); }

    /**
     * @brief Set previous-step solution coefficients (u^{n-1}) for transient forms
     *
     * This is only required when assembling forms containing symbolic `dt(·,k)`
     * in a transient time-integration context.
     */
    void setPreviousSolutionCoefficients(std::span<const Real> coefficients);

    /**
     * @brief Set previous-previous solution coefficients (u^{n-2}) for higher-order stencils
     */
    void setPreviousSolution2Coefficients(std::span<const Real> coefficients);

    /**
     * @brief Set k-th previous solution coefficients (u^{n-k})
     *
     * @param k History index (k=1 is u^{n-1})
     */
    void setPreviousSolutionCoefficientsK(int k, std::span<const Real> coefficients);

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
     * @brief Get vector-valued solution value at quadrature point
     *
     * Only valid when the trial space is vector-valued (e.g. ProductSpace).
     */
    [[nodiscard]] Vector3D solutionVectorValue(LocalIndex q) const;

    /**
     * @brief Get previous-step solution value at quadrature point
     */
    [[nodiscard]] Real previousSolutionValue(LocalIndex q) const;

    [[nodiscard]] Real previousSolutionValue(LocalIndex q, int k) const;

    [[nodiscard]] Vector3D previousSolutionVectorValue(LocalIndex q) const;

    [[nodiscard]] Vector3D previousSolutionVectorValue(LocalIndex q, int k) const;

    /**
     * @brief Get previous-previous solution value at quadrature point
     */
    [[nodiscard]] Real previousSolution2Value(LocalIndex q) const;

    [[nodiscard]] Vector3D previousSolution2VectorValue(LocalIndex q) const;

    /**
     * @brief Get solution gradient at quadrature point
     *
     * @param q Quadrature point index
     * @return Physical gradient of solution
     */
    [[nodiscard]] Vector3D solutionGradient(LocalIndex q) const;

    /**
     * @brief Get Jacobian of a vector-valued solution at quadrature point
     *
     * Returns a 3x3 matrix with entries J[i][j] = d u_i / d x_j.
     * Only the leading value_dimension() rows and spatial dimension() columns
     * are meaningful.
     */
    [[nodiscard]] Matrix3x3 solutionJacobian(LocalIndex q) const;

    /**
     * @brief Get previous-step solution gradient at quadrature point
     */
    [[nodiscard]] Vector3D previousSolutionGradient(LocalIndex q) const;

    [[nodiscard]] Matrix3x3 previousSolutionJacobian(LocalIndex q) const;

    /**
     * @brief Get previous-previous solution gradient at quadrature point
     */
    [[nodiscard]] Vector3D previousSolution2Gradient(LocalIndex q) const;

    [[nodiscard]] Matrix3x3 previousSolution2Jacobian(LocalIndex q) const;

    /**
     * @brief Get solution Hessian at quadrature point
     *
     * Only available when RequiredData::SolutionHessians was requested.
     */
    [[nodiscard]] Matrix3x3 solutionHessian(LocalIndex q) const;

    /**
     * @brief Get Hessian of a component of a vector-valued solution at quadrature point
     *
     * Only valid when the trial space is vector-valued (e.g. ProductSpace) and
     * RequiredData::SolutionHessians was requested.
     */
    [[nodiscard]] Matrix3x3 solutionComponentHessian(LocalIndex q, int component) const;

    /**
     * @brief Get solution Laplacian at quadrature point
     *
     * Only available when RequiredData::SolutionLaplacians was requested.
     */
    [[nodiscard]] Real solutionLaplacian(LocalIndex q) const;

    /**
     * @brief Get Laplacian of a component of a vector-valued solution at quadrature point
     *
     * Only valid when the trial space is vector-valued (e.g. ProductSpace) and
     * RequiredData::SolutionLaplacians was requested.
     */
    [[nodiscard]] Real solutionComponentLaplacian(LocalIndex q, int component) const;

    // =========================================================================
    // Multi-field Discrete Solution Data (optional)
    // =========================================================================

    /**
     * @brief Clear all stored auxiliary field solution data
     */
    void clearFieldSolutionData() noexcept;

    /**
     * @brief Bind scalar field data (values and optional derivatives) at quadrature points
     *
     * The provided arrays must be sized to numQuadraturePoints().
     */
    void setFieldSolutionScalar(FieldId field,
                                std::span<const Real> values,
                                std::span<const Vector3D> gradients = {},
                                std::span<const Matrix3x3> hessians = {},
                                std::span<const Real> laplacians = {});

    /**
     * @brief Bind vector field data (values and optional derivatives) at quadrature points
     *
     * The provided arrays must be sized to numQuadraturePoints() for values/jacobians and
     * numQuadraturePoints()*value_dimension for component_hessians/component_laplacians.
     */
    void setFieldSolutionVector(FieldId field,
                                int value_dimension,
                                std::span<const Vector3D> values,
                                std::span<const Matrix3x3> jacobians = {},
                                std::span<const Matrix3x3> component_hessians = {},
                                std::span<const Real> component_laplacians = {});

    /**
     * @brief Bind previous (history) scalar field values at quadrature points
     *
     * History indexing convention: k=1 is u^{n-1}, k=2 is u^{n-2}, ...
     */
    void setFieldPreviousSolutionScalarK(FieldId field, int k, std::span<const Real> values);

    /**
     * @brief Bind previous (history) vector field values at quadrature points
     *
     * History indexing convention: k=1 is u^{n-1}, k=2 is u^{n-2}, ...
     */
    void setFieldPreviousSolutionVectorK(FieldId field, int k, int value_dimension, std::span<const Vector3D> values);

    [[nodiscard]] bool hasFieldSolutionData(FieldId field) const noexcept;
    [[nodiscard]] FieldType fieldSolutionFieldType(FieldId field) const;
    [[nodiscard]] int fieldSolutionValueDimension(FieldId field) const;

    [[nodiscard]] Real fieldValue(FieldId field, LocalIndex q) const;
    [[nodiscard]] Vector3D fieldVectorValue(FieldId field, LocalIndex q) const;
    [[nodiscard]] Vector3D fieldGradient(FieldId field, LocalIndex q) const;
    [[nodiscard]] Matrix3x3 fieldJacobian(FieldId field, LocalIndex q) const;
    [[nodiscard]] Matrix3x3 fieldHessian(FieldId field, LocalIndex q) const;
    [[nodiscard]] Real fieldLaplacian(FieldId field, LocalIndex q) const;
    [[nodiscard]] Matrix3x3 fieldComponentHessian(FieldId field, LocalIndex q, int component) const;
    [[nodiscard]] Real fieldComponentLaplacian(FieldId field, LocalIndex q, int component) const;

    [[nodiscard]] Real fieldPreviousValue(FieldId field, LocalIndex q, int k) const;
    [[nodiscard]] Vector3D fieldPreviousVectorValue(FieldId field, LocalIndex q, int k) const;

    /**
     * @brief JIT-friendly, flat views of auxiliary field solution data
     *
     * This provides a stable table of POD entries intended for JIT ABI packing
     * (DiscreteField/StateField lowering). The table is rebuilt whenever field
     * solution data is modified.
     */
    [[nodiscard]] std::span<const jit::FieldSolutionEntryV1> jitFieldSolutionTable() const noexcept
    {
        return jit_field_solution_table_;
    }

    /**
     * @brief Check if solution data is available
     */
    [[nodiscard]] bool hasSolutionData() const noexcept {
        return !solution_values_.empty() || !solution_vector_values_.empty();
    }

    [[nodiscard]] bool hasPreviousSolutionData() const noexcept;
    [[nodiscard]] bool hasPreviousSolution2Data() const noexcept;

    /**
     * @brief Number of stored previous solution states (history depth)
     *
     * Indexing convention: k=1 is u^{n-1}, so history count is the max valid k.
     */
    [[nodiscard]] std::size_t previousSolutionHistoryCount() const noexcept
    {
        return history_solution_data_.size();
    }

    /**
     * @brief Element-local coefficients for u^{n-k} (k>=1)
     *
     * Intended for JIT/kernel ABI packing. Returns an empty span if k is out
     * of range or the coefficients were not set.
     */
    [[nodiscard]] std::span<const Real> previousSolutionCoefficientsRaw(int k) const noexcept
    {
        if (k <= 0) return {};
        const auto idx = static_cast<std::size_t>(k - 1);
        if (idx >= history_solution_data_.size()) return {};
        return history_solution_data_[idx].coefficients;
    }

    void clearPreviousSolutionData() noexcept
    {
        clearPreviousSolutionDataK(1);
    }

    void clearPreviousSolution2Data() noexcept
    {
        clearPreviousSolutionDataK(2);
    }

    void clearPreviousSolutionDataK(int k) noexcept;

    void clearAllPreviousSolutionData() noexcept;

    // =========================================================================
    // Material State (optional)
    // =========================================================================

    /**
     * @brief Bind per-integration-point state storage for the current cell
     *
     * The assembler should call this when the active kernel requests
     * RequiredData::MaterialState.
     */
    void setMaterialState(std::byte* cell_state_base,
                          std::size_t bytes_per_qpt,
                          std::size_t stride_bytes,
                          std::size_t alignment_bytes = alignof(std::max_align_t)) noexcept;

    void setMaterialState(std::byte* cell_state_old_base,
                          std::byte* cell_state_work_base,
                          std::size_t bytes_per_qpt,
                          std::size_t stride_bytes,
                          std::size_t alignment_bytes = alignof(std::max_align_t)) noexcept;

    [[nodiscard]] bool hasMaterialState() const noexcept { return material_state_work_base_ != nullptr; }
    [[nodiscard]] std::size_t materialStateBytesPerQpt() const noexcept { return material_state_bytes_per_qpt_; }
    [[nodiscard]] std::size_t materialStateStrideBytes() const noexcept { return material_state_stride_bytes_; }
    [[nodiscard]] std::size_t materialStateAlignmentBytes() const noexcept { return material_state_alignment_bytes_; }

    [[nodiscard]] const std::byte* materialStateOldBase() const noexcept { return material_state_old_base_; }
    [[nodiscard]] std::byte* materialStateWorkBase() const noexcept { return material_state_work_base_; }

    /**
     * @brief Access the state block for a single integration point
     *
     * Returns a mutable byte span so kernels can update state in-place.
     */
    [[nodiscard]] std::span<std::byte> materialState(LocalIndex q) const;

    /**
     * @brief Access the "old" state block for a single integration point
     */
    [[nodiscard]] std::span<const std::byte> materialStateOld(LocalIndex q) const;

    /**
     * @brief Access the "work/current" state block for a single integration point
     */
    [[nodiscard]] std::span<std::byte> materialStateWork(LocalIndex q) const;

    // =========================================================================
    // Transient / time integration context
    // =========================================================================

    /**
     * @brief Bind history/convolution weights for JIT-friendly history operators
     *
     * Indexing convention: weights[k-1] corresponds to u^{n-k}, k >= 1.
     *
     * The provided memory must remain valid for the duration of a kernel call.
     */
    void setHistoryWeights(std::span<const Real> weights) noexcept { history_weights_ = weights; }

    [[nodiscard]] std::span<const Real> historyWeights() const noexcept { return history_weights_; }

    /**
     * @brief Attach a transient time-integration context for symbolic `dt(·,k)` lowering
     *
     * When null, kernels containing `dt(...)` must fail with a clear diagnostic.
     */
    void setTimeIntegrationContext(const TimeIntegrationContext* ctx) noexcept { time_integration_ = ctx; }

    [[nodiscard]] const TimeIntegrationContext* timeIntegrationContext() const noexcept { return time_integration_; }

    // =========================================================================
    // System state context (optional)
    // =========================================================================

    void setTime(Real time) noexcept { time_ = time; }
    void setTimeStep(Real dt) noexcept { dt_ = dt; }

    [[nodiscard]] Real time() const noexcept { return time_; }
    [[nodiscard]] Real timeStep() const noexcept { return dt_; }

    void setRealParameterGetter(
        const std::function<std::optional<Real>(std::string_view)>* get_real_param) noexcept
    {
        get_real_param_ = get_real_param;
    }

    [[nodiscard]] const std::function<std::optional<Real>(std::string_view)>* realParameterGetter() const noexcept
    {
        return get_real_param_;
    }

    void setParameterGetter(
        const std::function<std::optional<params::Value>(std::string_view)>* get_param) noexcept
    {
        get_param_ = get_param;
    }

    [[nodiscard]] const std::function<std::optional<params::Value>(std::string_view)>* parameterGetter() const noexcept
    {
        return get_param_;
    }

    void setUserData(const void* user_data) noexcept { user_data_ = user_data; }
    [[nodiscard]] const void* userData() const noexcept { return user_data_; }

    // =========================================================================
    // JIT / Coupled scalar slot views (optional)
    // =========================================================================

    /**
     * @brief Bind a flat array of Real-valued parameter slots for JIT-friendly access.
     *
     * When set, Forms terminals like ParameterRef(slot) can load the value
     * without name-based lookup.
     */
    void setJITConstants(std::span<const Real> constants) noexcept { jit_constants_ = constants; }
    [[nodiscard]] std::span<const Real> jitConstants() const noexcept { return jit_constants_; }

    /**
     * @brief Bind coupled boundary-condition scalar arrays (integrals + aux state).
     *
     * When set, Forms terminals like BoundaryIntegralRef(slot) and
     * AuxiliaryStateRef(slot) can load values directly.
     */
    void setCoupledValues(std::span<const Real> integrals,
                          std::span<const Real> aux_state) noexcept
    {
        coupled_integrals_ = integrals;
        coupled_aux_state_ = aux_state;
    }

    [[nodiscard]] std::span<const Real> coupledIntegrals() const noexcept { return coupled_integrals_; }
    [[nodiscard]] std::span<const Real> coupledAuxState() const noexcept { return coupled_aux_state_; }

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

    /**
     * @brief Set cell domain/subdomain ID (for multi-domain coefficients/materials)
     */
    void setCellDomainId(int domain_id) noexcept { cell_domain_id_ = domain_id; }

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
     * @brief Set test vector basis values (physical space; layout: [i * n_qpts + q])
     */
    void setTestVectorBasisValues(LocalIndex n_dofs, std::span<const Vector3D> values);

    /**
     * @brief Set trial vector basis values (physical space; layout: [j * n_qpts + q])
     */
    void setTrialVectorBasisValues(LocalIndex n_dofs, std::span<const Vector3D> values);

    /**
     * @brief Set curl of test basis functions (physical space; layout: [i * n_qpts + q])
     */
    void setTestBasisCurls(LocalIndex n_dofs, std::span<const Vector3D> curls);

    /**
     * @brief Set curl of trial basis functions (physical space; layout: [j * n_qpts + q])
     */
    void setTrialBasisCurls(LocalIndex n_dofs, std::span<const Vector3D> curls);

    /**
     * @brief Set divergence of test basis functions (physical space; layout: [i * n_qpts + q])
     */
    void setTestBasisDivergences(LocalIndex n_dofs, std::span<const Real> divergences);

    /**
     * @brief Set divergence of trial basis functions (physical space; layout: [j * n_qpts + q])
     */
    void setTrialBasisDivergences(LocalIndex n_dofs, std::span<const Real> divergences);

    /**
     * @brief Set reference Hessians for test basis functions
     */
    void setTestBasisHessians(
        LocalIndex n_dofs,
        std::span<const Matrix3x3> hessians);

    /**
     * @brief Set reference Hessians for trial basis functions
     */
    void setTrialBasisHessians(
        LocalIndex n_dofs,
        std::span<const Matrix3x3> hessians);

    /**
     * @brief Set physical gradients (after Jacobian transformation, layout: [q * n_dofs + dof])
     */
    void setPhysicalGradients(
        std::span<const Vector3D> test_gradients,
        std::span<const Vector3D> trial_gradients);

    /**
     * @brief Set physical Hessians (after Jacobian transformation)
     */
    void setPhysicalHessians(
        std::span<const Matrix3x3> test_hessians,
        std::span<const Matrix3x3> trial_hessians);

    /**
     * @brief Set normal vectors (for face contexts)
     */
    void setNormals(std::span<const Vector3D> normals);

    /**
     * @brief Set entity measures (cell diameter, cell volume, facet area)
     *
     * For cell contexts, set facet_area=0. For face contexts, set cell_volume=0.
     */
    void setEntityMeasures(Real cell_diameter, Real cell_volume, Real facet_area);

    /**
     * @brief Set solution values at quadrature points
     */
    void setSolutionValues(std::span<const Real> values);

    /**
     * @brief Set solution gradients at quadrature points
     */
    void setSolutionGradients(std::span<const Vector3D> gradients);

private:
    template <class T>
    class ArenaArray {
    public:
        using value_type = T;
        using pointer = T*;
        using const_pointer = const T*;
        using iterator = T*;
        using const_iterator = const T*;

        void bind(pointer ptr, std::size_t capacity) noexcept
        {
            data_ = ptr;
            size_ = 0;
            capacity_ = capacity;
        }

        void clear() noexcept { size_ = 0; }

        [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
        [[nodiscard]] std::size_t size() const noexcept { return size_; }
        [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
        [[nodiscard]] pointer data() noexcept { return data_; }
        [[nodiscard]] const_pointer data() const noexcept { return data_; }

        [[nodiscard]] iterator begin() noexcept { return data_; }
        [[nodiscard]] iterator end() noexcept { return data_ + size_; }
        [[nodiscard]] const_iterator begin() const noexcept { return data_; }
        [[nodiscard]] const_iterator end() const noexcept { return data_ + size_; }

        [[nodiscard]] T& operator[](std::size_t index) noexcept { return data_[index]; }
        [[nodiscard]] const T& operator[](std::size_t index) const noexcept { return data_[index]; }

        template <class It>
        void assign(It first, It last)
        {
            const auto n = static_cast<std::size_t>(std::distance(first, last));
            checkCapacity(n);
            std::copy(first, last, data_);
            size_ = n;
        }

        void assign(std::size_t n, const T& value)
        {
            checkCapacity(n);
            std::fill_n(data_, n, value);
            size_ = n;
        }

        void resize(std::size_t n)
        {
            checkCapacity(n);
            if (n > size_) {
                std::fill(data_ + size_, data_ + n, T{});
            }
            size_ = n;
        }

        void resize(std::size_t n, const T& value)
        {
            checkCapacity(n);
            if (n > size_) {
                std::fill(data_ + size_, data_ + n, value);
            }
            size_ = n;
        }

        [[nodiscard]] std::span<T> span() noexcept { return {data_, size_}; }
        [[nodiscard]] std::span<const T> span() const noexcept { return {data_, size_}; }
        [[nodiscard]] operator std::span<const T>() const noexcept { return span(); }

    private:
        void checkCapacity(std::size_t n) const
        {
            if (n > capacity_) {
                throw std::length_error(
                    "AssemblyContext::ArenaArray: requested size exceeds reserved arena capacity");
            }
        }

        pointer data_{nullptr};
        std::size_t size_{0};
        std::size_t capacity_{0};
    };

    template <class T>
    using JITAlignedVector = std::vector<T, AlignedAllocator<T, jit::kJITPointerAlignmentBytes>>;

    void rebuildJITFieldSolutionTable();
    void rebuildInterleavedQPointGeometry();
    void ensureArenaCapacity(LocalIndex required_dofs, LocalIndex required_qpts);

    // Context type and identification
    ContextType type_{ContextType::Cell};
    GlobalIndex cell_id_{-1};
    int cell_domain_id_{0};
    GlobalIndex face_id_{-1};
    LocalIndex local_face_id_{0};
    int boundary_marker_{-1};
    int dim_{3};

    // DOF counts
    LocalIndex n_test_dofs_{0};
    LocalIndex n_trial_dofs_{0};
    LocalIndex n_qpts_{0};

    // Space metadata (needed for vector-valued TrialFunction evaluation)
    FieldType test_field_type_{FieldType::Scalar};
    FieldType trial_field_type_{FieldType::Scalar};
    Continuity test_continuity_{Continuity::C0};
    Continuity trial_continuity_{Continuity::C0};
    bool test_is_vector_basis_{false};
    bool trial_is_vector_basis_{false};
    int test_value_dim_{1};
    int trial_value_dim_{1};
    LocalIndex arena_max_dofs_{0};
    LocalIndex arena_max_qpts_{0};

    // Single contiguous storage arena for high-frequency context arrays.
    JITAlignedVector<std::byte> arena_storage_;

    // Quadrature data
    ArenaArray<Point3D> quad_points_;
    ArenaArray<Real> quad_weights_;
    ArenaArray<Real> integration_weights_;

    // Geometry data
    ArenaArray<Point3D> physical_points_;
    ArenaArray<Matrix3x3> jacobians_;
    ArenaArray<Matrix3x3> inverse_jacobians_;
    ArenaArray<Real> jacobian_dets_;
    ArenaArray<Vector3D> normals_;
    ArenaArray<Real> interleaved_qpoint_geometry_;

    // Entity measures (optional)
    Real cell_diameter_{0.0};
    Real cell_volume_{0.0};
    Real facet_area_{0.0};

    // Test basis data (n_test_dofs * n_qpts arrays)
    ArenaArray<Real> test_basis_values_;           // [i * n_qpts + q]
    ArenaArray<Vector3D> test_ref_gradients_;      // [i * n_qpts + q]
    ArenaArray<Vector3D> test_phys_gradients_;     // [q * n_test_dofs + i]

    // Test vector-basis data (H(curl)/H(div)) (n_test_dofs * n_qpts arrays)
    ArenaArray<Vector3D> test_basis_vector_values_; // [i * n_qpts + q]
    ArenaArray<Vector3D> test_basis_curls_;         // [i * n_qpts + q]
    ArenaArray<Real> test_basis_divergences_;       // [i * n_qpts + q]

    // Trial basis data (may be same as test for square)
    ArenaArray<Real> trial_basis_values_;
    ArenaArray<Vector3D> trial_ref_gradients_;
    ArenaArray<Vector3D> trial_phys_gradients_;    // [q * n_trial_dofs + j]
    ArenaArray<Vector3D> trial_basis_vector_values_;
    ArenaArray<Vector3D> trial_basis_curls_;
    ArenaArray<Real> trial_basis_divergences_;
    bool trial_is_test_{true};  // Optimization flag

    // Optional basis Hessians (n_dofs * n_qpts arrays)
    ArenaArray<Matrix3x3> test_ref_hessians_;
    ArenaArray<Matrix3x3> test_phys_hessians_;
    ArenaArray<Matrix3x3> trial_ref_hessians_;
    ArenaArray<Matrix3x3> trial_phys_hessians_;

    // Solution data (for nonlinear problems)
    ArenaArray<Real> solution_coefficients_;
    ArenaArray<Real> solution_values_;
    ArenaArray<Vector3D> solution_vector_values_;
    ArenaArray<Vector3D> solution_gradients_;
    ArenaArray<Matrix3x3> solution_jacobians_;
    ArenaArray<Matrix3x3> solution_hessians_;
    ArenaArray<Real> solution_laplacians_;
    ArenaArray<Matrix3x3> solution_component_hessians_;  // [q * trial_value_dim + c]
    ArenaArray<Real> solution_component_laplacians_;     // [q * trial_value_dim + c]

    struct FieldSolutionData {
        FieldId id{INVALID_FIELD_ID};
        FieldType field_type{FieldType::Scalar};
        int value_dim{1};

        JITAlignedVector<Real> values{};
        JITAlignedVector<Vector3D> vector_values{};
        // Previous (history) values packed by time index:
        //   history_values[(k-1)*n_qpts + q] == field value at quadrature point q from u^{n-k}.
        JITAlignedVector<Real> history_values{};
        JITAlignedVector<Vector3D> history_vector_values{};
        JITAlignedVector<Vector3D> gradients{};
        JITAlignedVector<Matrix3x3> jacobians{};
        JITAlignedVector<Matrix3x3> hessians{};
        JITAlignedVector<Real> laplacians{};
        JITAlignedVector<Matrix3x3> component_hessians{};  // [q * value_dim + c]
        JITAlignedVector<Real> component_laplacians{};     // [q * value_dim + c]
    };

    std::vector<FieldSolutionData> field_solution_data_{};
    JITAlignedVector<jit::FieldSolutionEntryV1> jit_field_solution_table_{};

    // Transient history solution data (optional)
    struct HistorySolutionData {
        JITAlignedVector<Real> coefficients{};
        JITAlignedVector<Real> values{};
        JITAlignedVector<Vector3D> vector_values{};
        JITAlignedVector<Vector3D> gradients{};
        JITAlignedVector<Matrix3x3> jacobians{};
        JITAlignedVector<Matrix3x3> component_hessians{};  // [q * value_dim + c]
        JITAlignedVector<Real> component_laplacians{};     // [q * value_dim + c]
    };

    // Indexing: history_solution_data_[k-1] corresponds to u^{n-k}, k >= 1.
    std::vector<HistorySolutionData> history_solution_data_{};

    // Optional per-cell material state storage (owned externally)
    std::byte* material_state_old_base_{nullptr};
    std::byte* material_state_work_base_{nullptr};
    std::size_t material_state_bytes_per_qpt_{0};
    std::size_t material_state_stride_bytes_{0};
    std::size_t material_state_alignment_bytes_{alignof(std::max_align_t)};

    // Optional time/parameter context (owned by Systems)
    Real time_{0.0};
    Real dt_{0.0};
    const std::function<std::optional<Real>(std::string_view)>* get_real_param_{nullptr};
    const std::function<std::optional<params::Value>(std::string_view)>* get_param_{nullptr};
    const void* user_data_{nullptr};

    // Optional pre-resolved scalar slot arrays (owned externally; stable during a kernel call)
    std::span<const Real> jit_constants_{};
    std::span<const Real> coupled_integrals_{};
    std::span<const Real> coupled_aux_state_{};
    std::span<const Real> history_weights_{};

    // Optional transient time integration context (owned by Systems/TimeStepping)
    const TimeIntegrationContext* time_integration_{nullptr};

    // Remember last requested data flags so setSolutionCoefficients can compute only what is needed.
    RequiredData required_data_{RequiredData::None};
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
