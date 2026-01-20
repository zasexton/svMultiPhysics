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

#ifndef SVMP_FE_ASSEMBLY_ASSEMBLY_KERNEL_H
#define SVMP_FE_ASSEMBLY_ASSEMBLY_KERNEL_H

/**
 * @file AssemblyKernel.h
 * @brief Interface for element kernels (Physics-facing)
 *
 * AssemblyKernel defines the interface between Assembly and Physics modules.
 * Physics implementations provide kernels that compute local element matrices
 * and vectors; Assembly orchestrates calling these kernels and inserting
 * the results into global structures.
 *
 * Kernel types:
 * - Cell kernels: Compute contributions for volume integrals
 * - Boundary face kernels: Compute Neumann/Robin/Nitsche boundary terms
 * - Interior face kernels: Compute DG interface terms
 *
 * Design principles:
 * - Physics-agnostic interface (Assembly doesn't know what PDE is being solved)
 * - Kernels request data via RequiredData flags
 * - AssemblyContext provides requested data (geometry, basis, quadrature, solution)
 * - Kernels are stateless for thread safety (state in AssemblyContext)
 *
 * Module boundary:
 * - Assembly OWNS: kernel invocation, context preparation
 * - Physics OWNS: kernel implementation, weak form definition
 *
 * @see AssemblyContext for data provided to kernels
 * @see Assembler for the orchestration interface
 */

#include "Core/Types.h"
#include "Core/ParameterValue.h"

#include <span>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <string>
#include <memory>
#include <functional>

namespace svmp {
namespace FE {

// Forward declarations
namespace assembly {
class AssemblyContext;
}

namespace assembly {

// ============================================================================
// Required Data Flags
// ============================================================================

/**
 * @brief Flags indicating what data a kernel needs
 *
 * Kernels specify their data requirements via getRequiredData().
 * The assembler uses this to prepare the AssemblyContext before
 * calling the kernel.
 *
 * Using a bitfield allows efficient combination of requirements.
 */
enum class RequiredData : std::uint32_t {
    None                = 0,

    // Geometry data
    PhysicalPoints      = 1 << 0,   ///< Physical coordinates at quadrature points
    Jacobians           = 1 << 1,   ///< Jacobian matrices at quadrature points
    JacobianDets        = 1 << 2,   ///< Jacobian determinants (for integration)
    InverseJacobians    = 1 << 3,   ///< Inverse Jacobians (for gradient transform)
    Normals             = 1 << 4,   ///< Surface normals (for faces)
    Tangents            = 1 << 5,   ///< Surface tangents (for edges/faces)
    EntityMeasures      = 1 << 6,   ///< Cell/face measures (volume/area/diameter)

    // Basis function data
    BasisValues         = 1 << 8,   ///< Basis function values at quadrature points
    BasisGradients      = 1 << 9,   ///< Reference gradients of basis functions
    PhysicalGradients   = 1 << 10,  ///< Physical gradients (transformed)
    BasisHessians       = 1 << 11,  ///< Second derivatives (for C1 elements)
    BasisCurls          = 1 << 12,  ///< Curl of basis functions (H(curl))
    BasisDivergences    = 1 << 13,  ///< Divergence of basis functions (H(div))

    // Quadrature data
    QuadraturePoints    = 1 << 16,  ///< Reference quadrature point coordinates
    QuadratureWeights   = 1 << 17,  ///< Quadrature weights
    IntegrationWeights  = 1 << 18,  ///< Weights * |J| (ready for integration)

    // Solution data (for nonlinear problems)
    SolutionCoefficients= 1 << 19,  ///< Element-local DOF coefficients for the current solution
    SolutionValues      = 1 << 20,  ///< Solution values at quadrature points
    SolutionGradients   = 1 << 21,  ///< Solution gradients at quadrature points
    SolutionHessians    = 1 << 22,  ///< Solution Hessians
    SolutionLaplacians  = 1 << 23,  ///< Solution Laplacians

    // Material/coefficient data
    MaterialState       = 1 << 24,  ///< Material state from previous iteration

    // Face-specific data
    FaceOrientations    = 1 << 28,  ///< Face orientation info (for DG)
    NeighborData        = 1 << 29,  ///< Data from neighboring cell (for DG)

    // Common combinations
    BasicGeometry       = PhysicalPoints | JacobianDets | InverseJacobians,
    BasicBasis          = BasisValues | BasisGradients,
    BasicQuadrature     = QuadratureWeights | IntegrationWeights,
    Standard            = BasicGeometry | BasicBasis | BasicQuadrature,
    WithSolution        = Standard | SolutionValues | SolutionGradients,
    DGFace              = Standard | Normals | FaceOrientations | NeighborData
};

// Bitwise operators for RequiredData
inline constexpr RequiredData operator|(RequiredData a, RequiredData b) {
    return static_cast<RequiredData>(
        static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b));
}

inline constexpr RequiredData operator&(RequiredData a, RequiredData b) {
    return static_cast<RequiredData>(
        static_cast<std::uint32_t>(a) & static_cast<std::uint32_t>(b));
}

inline constexpr RequiredData& operator|=(RequiredData& a, RequiredData b) {
    a = a | b;
    return a;
}

inline constexpr bool hasFlag(RequiredData flags, RequiredData flag) {
    return (static_cast<std::uint32_t>(flags) & static_cast<std::uint32_t>(flag)) != 0;
}

// ============================================================================
// Material State Specification (optional)
// ============================================================================

/**
 * @brief Description of per-integration-point state storage required by a kernel
 *
 * Kernels that request RequiredData::MaterialState should override
 * AssemblyKernel::materialStateSpec() to indicate how much storage is needed
 * at each integration point.
 */
struct MaterialStateSpec {
    std::size_t bytes_per_qpt{0};
    std::size_t alignment{alignof(std::max_align_t)};
};

// ============================================================================
// Multi-field Solution Requirements (optional)
// ============================================================================

/**
 * @brief Requirement for accessing an additional FE field as a discrete coefficient
 *
 * Kernels can declare that they need values and/or derivatives of other fields
 * (identified by FieldId) at the current quadrature points. The assembler is
 * responsible for gathering the requested field data and binding it into the
 * AssemblyContext before kernel evaluation.
 *
 * Only solution-related bits in RequiredData are meaningful here:
 * - SolutionValues
 * - SolutionGradients (scalar gradients / vector Jacobians)
 * - SolutionHessians (scalar Hessians / vector component Hessians)
 * - SolutionLaplacians (scalar Laplacians / vector component Laplacians)
 */
struct FieldRequirement {
    FieldId field{INVALID_FIELD_ID};
    RequiredData required{RequiredData::None};
};

// ============================================================================
// Kernel Result Structure
// ============================================================================

/**
 * @brief Output from a kernel computation
 *
 * Contains the local element matrix and/or vector computed by the kernel.
 * The assembler inserts these into the global system.
 */
struct KernelOutput {
    /// Local element matrix (row-major, n_test_dofs x n_trial_dofs)
    std::vector<Real> local_matrix;

    /// Local element vector (n_test_dofs)
    std::vector<Real> local_vector;

    /// Number of test DOFs (rows)
    LocalIndex n_test_dofs{0};

    /// Number of trial DOFs (columns)
    LocalIndex n_trial_dofs{0};

    /// Whether matrix was computed
    bool has_matrix{false};

    /// Whether vector was computed
    bool has_vector{false};

    /**
     * @brief Reserve storage for given DOF counts
     */
    void reserve(LocalIndex n_test, LocalIndex n_trial, bool need_matrix, bool need_vector) {
        n_test_dofs = n_test;
        n_trial_dofs = n_trial;

        has_matrix = need_matrix;
        has_vector = need_vector;

        if (need_matrix) {
            const auto size =
                static_cast<std::size_t>(n_test) * static_cast<std::size_t>(n_trial);
            local_matrix.resize(size);
            std::fill(local_matrix.begin(), local_matrix.end(), 0.0);
        } else {
            local_matrix.clear();
        }

        if (need_vector) {
            const auto size = static_cast<std::size_t>(n_test);
            local_vector.resize(size);
            std::fill(local_vector.begin(), local_vector.end(), 0.0);
        } else {
            local_vector.clear();
        }
    }

    /**
     * @brief Clear all data
     */
    void clear() {
        std::fill(local_matrix.begin(), local_matrix.end(), 0.0);
        std::fill(local_vector.begin(), local_vector.end(), 0.0);
    }

    /**
     * @brief Access matrix entry (row-major)
     */
    Real& matrixEntry(LocalIndex row, LocalIndex col) {
        return local_matrix[static_cast<std::size_t>(row * n_trial_dofs + col)];
    }

    const Real& matrixEntry(LocalIndex row, LocalIndex col) const {
        return local_matrix[static_cast<std::size_t>(row * n_trial_dofs + col)];
    }

    /**
     * @brief Access vector entry
     */
    Real& vectorEntry(LocalIndex row) {
        return local_vector[static_cast<std::size_t>(row)];
    }

    const Real& vectorEntry(LocalIndex row) const {
        return local_vector[static_cast<std::size_t>(row)];
    }
};

// ============================================================================
// Assembly Kernel Interface
// ============================================================================

/**
 * @brief Abstract interface for element kernels
 *
 * Physics modules implement this interface to define how local element
 * matrices and vectors are computed. The assembly infrastructure calls
 * these methods during the element loop.
 *
 * Implementation notes:
 * - Kernels should be stateless (all state in AssemblyContext)
 * - getRequiredData() is called once before assembly begins
 * - Cell/face methods may be called concurrently from multiple threads
 *
 * Example implementation (Poisson):
 * @code
 * class PoissonKernel : public AssemblyKernel {
 * public:
 *     RequiredData getRequiredData() const override {
 *         return RequiredData::Standard;
 *     }
 *
 *     void computeCell(const AssemblyContext& ctx, KernelOutput& output) override {
 *         const auto n_dofs = ctx.numTestDofs();
 *         output.reserve(n_dofs, n_dofs, true, true);
 *
 *         for (LocalIndex q = 0; q < ctx.numQuadraturePoints(); ++q) {
 *             const Real w = ctx.integrationWeight(q);
 *
 *             for (LocalIndex i = 0; i < n_dofs; ++i) {
 *                 auto grad_i = ctx.physicalGradient(i, q);
 *
 *                 for (LocalIndex j = 0; j < n_dofs; ++j) {
 *                     auto grad_j = ctx.physicalGradient(j, q);
 *                     output.matrixEntry(i, j) += w * dot(grad_i, grad_j);
 *                 }
 *
 *                 output.vectorEntry(i) += w * ctx.basisValue(i, q) * f(ctx.physicalPoint(q));
 *             }
 *         }
 *     }
 * };
 * @endcode
 */
class AssemblyKernel {
public:
    virtual ~AssemblyKernel() = default;

    // =========================================================================
    // Data Requirements
    // =========================================================================

    /**
     * @brief Specify what data this kernel needs
     *
     * Called once before assembly begins. The assembler uses this to
     * configure the AssemblyContext appropriately.
     *
     * @return Bitfield of RequiredData flags
     */
    [[nodiscard]] virtual RequiredData getRequiredData() const = 0;

    /**
     * @brief Optional multi-field (discrete-coefficient) requirements
     *
     * Kernels may override this to request access to other FE fields beyond the
     * current test/trial spaces. The default implementation requests no
     * additional fields.
     */
    [[nodiscard]] virtual std::vector<FieldRequirement> fieldRequirements() const { return {}; }

    /**
     * @brief Optional per-integration-point state requirement
     *
     * If getRequiredData() includes RequiredData::MaterialState, kernels should
     * override this to return a non-zero bytes_per_qpt (and desired alignment).
     */
	    [[nodiscard]] virtual MaterialStateSpec materialStateSpec() const noexcept { return {}; }

	    /**
	     * @brief Optional parameter requirements for this kernel
	     *
	     * Kernels may declare required and optional parameters (with defaults) for
	     * validation by higher-level systems. The assembly layer does not
	     * interpret these values directly.
	     */
	    [[nodiscard]] virtual std::vector<params::Spec> parameterSpecs() const { return {}; }

	    /**
	     * @brief Optional setup-time resolution of parameter symbols to slot refs
	     *
	     * FE/Forms expressions can reference scalar parameters by name
	     * (FormExprType::ParameterSymbol). For JIT-friendly evaluation, Systems can
	     * resolve those names to stable integer slots and bind a flat constants
	     * array into the AssemblyContext (see AssemblyContext::jitConstants()).
	     *
	     * Kernels that contain ParameterSymbol nodes may override this hook to
	     * rewrite their internal representation to use ParameterRef(slot) nodes.
	     * The default implementation is a no-op.
	     *
	     * @param slot_of_real_param Mapping from parameter key -> slot index.
	     */
	    virtual void resolveParameterSlots(
	        const std::function<std::optional<std::uint32_t>(std::string_view)>& slot_of_real_param)
	    {
	        (void)slot_of_real_param;
	    }

	    /**
	     * @brief Optional setup-time lowering of inlinable constitutive calls
	     *
	     * FE/Forms supports `FormExprType::Constitutive` as a type-erased bridge to
	     * material-point operators. In "JIT-fast" mode, models may opt in to a
	     * symbolic expansion that rewrites constitutive calls into plain FormExpr
	     * (no virtual dispatch in the hot path) plus explicit material-state
	     * reads/writes.
	     *
	     * Kernels that embed FE/Forms expressions may override this hook to apply
	     * that lowering during system setup. The default implementation is a no-op.
	     */
	    virtual void resolveInlinableConstitutives() {}

	    // =========================================================================
	    // Cell (Volume) Integration
	    // =========================================================================

    /**
     * @brief Compute element matrix/vector for a cell
     *
     * This is the main integration routine for volume terms.
     * Called once per cell during assembly.
     *
     * @param ctx AssemblyContext with prepared data for this cell
     * @param output Output structure for local matrix/vector
     *
     * The context provides:
     * - Quadrature points and weights
     * - Basis function values and gradients
     * - Geometry (Jacobians, physical coordinates)
     * - Solution values (for nonlinear problems)
     *
     * @note This method may be called concurrently from multiple threads.
     *       Implementation must be thread-safe (no shared mutable state).
     */
    virtual void computeCell(
        const AssemblyContext& ctx,
        KernelOutput& output) = 0;

    /**
     * @brief Check if kernel supports cell integration
     */
    [[nodiscard]] virtual bool hasCell() const noexcept { return true; }

    // =========================================================================
    // Boundary Face Integration
    // =========================================================================

    /**
     * @brief Compute element matrix/vector for a boundary face
     *
     * Used for Neumann, Robin, or Nitsche boundary conditions.
     *
     * @param ctx AssemblyContext with prepared data for this face
     * @param boundary_marker Boundary label/marker ID
     * @param output Output structure for local matrix/vector
     *
     * Additional context data available for faces:
     * - Surface normal vectors
     * - Face quadrature (restricted to boundary)
     */
    virtual void computeBoundaryFace(
        const AssemblyContext& ctx,
        int boundary_marker,
        KernelOutput& output);

    /**
     * @brief Check if kernel supports boundary face integration
     */
    [[nodiscard]] virtual bool hasBoundaryFace() const noexcept { return false; }

    // =========================================================================
    // Interior Face Integration (DG)
    // =========================================================================

    /**
     * @brief Compute matrix/vector for an interior face (DG)
     *
     * Used for DG interface terms (numerical fluxes).
     *
     * @param ctx_minus Context for the "minus" side cell
     * @param ctx_plus Context for the "plus" side cell
     * @param output_minus Output contributions to minus cell equations
     * @param output_plus Output contributions to plus cell equations
     * @param coupling_minus_plus Coupling from plus DOFs to minus equations
     * @param coupling_plus_minus Coupling from minus DOFs to plus equations
     *
     * For DG methods, interior faces contribute to both adjacent cells.
     * The four output structures represent:
     * - output_minus: terms in minus cell equations involving minus cell DOFs
     * - output_plus: terms in plus cell equations involving plus cell DOFs
     * - coupling_minus_plus: terms in minus equations involving plus DOFs
     * - coupling_plus_minus: terms in plus equations involving minus DOFs
     */
    virtual void computeInteriorFace(
        const AssemblyContext& ctx_minus,
        const AssemblyContext& ctx_plus,
        KernelOutput& output_minus,
        KernelOutput& output_plus,
        KernelOutput& coupling_minus_plus,
        KernelOutput& coupling_plus_minus);

    /**
     * @brief Check if kernel supports interior face integration
     */
    [[nodiscard]] virtual bool hasInteriorFace() const noexcept { return false; }

    // =========================================================================
    // Interface Face Integration (subset via InterfaceMesh)
    // =========================================================================

    /**
     * @brief Compute matrix/vector for an interface face (InterfaceMesh subset)
     *
     * This is analogous to computeInteriorFace, but is dispatched only for faces
     * belonging to a registered InterfaceMesh (e.g., a material interface).
     *
     * @param ctx_minus Context for the "minus" side cell
     * @param ctx_plus Context for the "plus" side cell
     * @param interface_marker Interface surface identifier (as specified by `.dI(marker)`)
     * @param output_minus Output contributions to minus cell equations
     * @param output_plus Output contributions to plus cell equations
     * @param coupling_minus_plus Coupling from plus DOFs to minus equations
     * @param coupling_plus_minus Coupling from minus DOFs to plus equations
     */
    virtual void computeInterfaceFace(
        const AssemblyContext& ctx_minus,
        const AssemblyContext& ctx_plus,
        int interface_marker,
        KernelOutput& output_minus,
        KernelOutput& output_plus,
        KernelOutput& coupling_minus_plus,
        KernelOutput& coupling_plus_minus);

    /**
     * @brief Check if kernel supports interface face integration
     */
    [[nodiscard]] virtual bool hasInterfaceFace() const noexcept { return false; }

    // =========================================================================
    // Query
    // =========================================================================

    /**
     * @brief Get kernel name for debugging/logging
     */
    [[nodiscard]] virtual std::string name() const { return "GenericKernel"; }

    /**
     * @brief Maximum time-derivative order referenced by this kernel
     *
     * Kernels compiled from continuous-time formulations (e.g., containing
     * `dt(u,k)` terms) can override this to signal that the problem is
     * transient and requires time discretization (handled by FE/TimeStepping).
     *
     * Default is 0 (purely spatial / steady kernel).
     */
    [[nodiscard]] virtual int maxTemporalDerivativeOrder() const noexcept { return 0; }

    /**
     * @brief Check if kernel produces symmetric matrices
     *
     * Used by assembler to potentially optimize insertion.
     */
    [[nodiscard]] virtual bool isSymmetric() const noexcept { return false; }

    /**
     * @brief Check if kernel produces only matrix (no RHS)
     */
    [[nodiscard]] virtual bool isMatrixOnly() const noexcept { return false; }

    /**
     * @brief Check if kernel produces only vector (no matrix)
     */
    [[nodiscard]] virtual bool isVectorOnly() const noexcept { return false; }
};

// ============================================================================
// Common Kernel Implementations
// ============================================================================

/**
 * @brief Base class for bilinear form kernels (matrix only)
 */
class BilinearFormKernel : public AssemblyKernel {
public:
    [[nodiscard]] bool isMatrixOnly() const noexcept override { return true; }
};

/**
 * @brief Base class for linear form kernels (vector only)
 */
class LinearFormKernel : public AssemblyKernel {
public:
    [[nodiscard]] bool isVectorOnly() const noexcept override { return true; }
};

/**
 * @brief Mass matrix kernel (u, v)_L2
 *
 * Computes the mass matrix: M_ij = integral(phi_i * phi_j)
 */
class MassKernel : public BilinearFormKernel {
public:
    /**
     * @brief Construct with optional scaling coefficient
     */
    explicit MassKernel(Real coefficient = 1.0);

    [[nodiscard]] RequiredData getRequiredData() const override;
    void computeCell(const AssemblyContext& ctx, KernelOutput& output) override;
    [[nodiscard]] std::string name() const override { return "MassKernel"; }
    [[nodiscard]] bool isSymmetric() const noexcept override { return true; }

private:
    Real coefficient_;
};

/**
 * @brief Stiffness matrix kernel (grad u, grad v)_L2
 *
 * Computes the stiffness matrix: K_ij = integral(grad phi_i . grad phi_j)
 */
class StiffnessKernel : public BilinearFormKernel {
public:
    /**
     * @brief Construct with optional scaling coefficient
     */
    explicit StiffnessKernel(Real coefficient = 1.0);

    [[nodiscard]] RequiredData getRequiredData() const override;
    void computeCell(const AssemblyContext& ctx, KernelOutput& output) override;
    [[nodiscard]] std::string name() const override { return "StiffnessKernel"; }
    [[nodiscard]] bool isSymmetric() const noexcept override { return true; }

private:
    Real coefficient_;
};

/**
 * @brief Source term kernel (f, v)_L2
 *
 * Computes RHS contribution: b_i = integral(f * phi_i)
 */
class SourceKernel : public LinearFormKernel {
public:
    using SourceFunction = std::function<Real(Real, Real, Real)>;

    /**
     * @brief Construct with source function f(x, y, z)
     */
    explicit SourceKernel(SourceFunction source);

    /**
     * @brief Construct with constant source
     */
    explicit SourceKernel(Real constant_source);

    [[nodiscard]] RequiredData getRequiredData() const override;
    void computeCell(const AssemblyContext& ctx, KernelOutput& output) override;
    [[nodiscard]] std::string name() const override { return "SourceKernel"; }

private:
    SourceFunction source_;
};

/**
 * @brief Combined Poisson kernel: -Delta u = f
 *
 * Computes both stiffness matrix and source RHS.
 */
class PoissonKernel : public AssemblyKernel {
public:
    using SourceFunction = std::function<Real(Real, Real, Real)>;

    /**
     * @brief Construct with source function
     */
    explicit PoissonKernel(SourceFunction source);

    /**
     * @brief Construct with constant source
     */
    explicit PoissonKernel(Real constant_source = 0.0);

    [[nodiscard]] RequiredData getRequiredData() const override;
    void computeCell(const AssemblyContext& ctx, KernelOutput& output) override;
    [[nodiscard]] std::string name() const override { return "PoissonKernel"; }
    [[nodiscard]] bool isSymmetric() const noexcept override { return true; }

private:
    SourceFunction source_;
};

// ============================================================================
// Kernel Utilities
// ============================================================================

/**
 * @brief Combine multiple kernels into one
 *
 * Useful for multi-term weak forms like convection-diffusion.
 */
class CompositeKernel : public AssemblyKernel {
public:
    /**
     * @brief Add a kernel with optional scaling
     */
    void addKernel(std::shared_ptr<AssemblyKernel> kernel, Real scale = 1.0);

    [[nodiscard]] RequiredData getRequiredData() const override;
    void computeCell(const AssemblyContext& ctx, KernelOutput& output) override;
    void computeBoundaryFace(const AssemblyContext& ctx, int boundary_marker,
                             KernelOutput& output) override;
    [[nodiscard]] bool hasBoundaryFace() const noexcept override;
    [[nodiscard]] std::string name() const override { return "CompositeKernel"; }

private:
    struct KernelEntry {
        std::shared_ptr<AssemblyKernel> kernel;
        Real scale;
    };
    std::vector<KernelEntry> kernels_;
    KernelOutput temp_output_;
};

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_ASSEMBLY_KERNEL_H
