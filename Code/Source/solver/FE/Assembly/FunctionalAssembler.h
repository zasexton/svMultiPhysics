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

#ifndef SVMP_FE_ASSEMBLY_FUNCTIONAL_ASSEMBLER_H
#define SVMP_FE_ASSEMBLY_FUNCTIONAL_ASSEMBLER_H

/**
 * @file FunctionalAssembler.h
 * @brief Assembly of scalar functionals (Quantities of Interest)
 *
 * FunctionalAssembler computes scalar quantities by integrating over the mesh.
 * This is essential for:
 *
 * - Output quantities: energy, drag/lift coefficients, flux integrals
 * - Norms: L2 norm, H1 seminorm, error norms
 * - Goal-oriented error estimation: dual-weighted residual (DWR) indicators
 * - Stabilization diagnostics: SUPG/GLS consistency measures
 *
 * Unlike matrix/vector assembly, functional assembly produces a single scalar
 * value (or a small set of scalars). The assembly pattern is:
 *
 *   J(u) = sum_K integral_K j(u, grad u, x) dx
 *
 * where j is the integrand provided by a FunctionalKernel.
 *
 * Features:
 * - Cell integration for volume functionals
 * - Boundary integration for surface functionals (drag, lift)
 * - Thread-parallel reduction with deterministic summation order
 * - MPI reduction for distributed meshes
 * - Multiple functional computation in single mesh pass
 *
 * Module boundaries:
 * - This module OWNS: loop orchestration, reduction, parallel summation
 * - This module does NOT OWN: physics integrand (FunctionalKernel)
 *
 * @see AssemblyContext for data provided to kernels
 * @see FunctionalKernel for the integrand interface
 */

#include "Core/Types.h"
#include "Core/FEException.h"
#include "Assembler.h"
#include "AssemblyKernel.h"
#include "AssemblyContext.h"

#include <vector>
#include <span>
#include <functional>
#include <memory>
#include <string>
#include <cmath>

namespace svmp {
namespace FE {

// Forward declarations
namespace dofs {
    class DofMap;
}

namespace spaces {
    class FunctionSpace;
}

namespace assembly {

struct TimeIntegrationContext;

// ============================================================================
// Functional Kernel Interface
// ============================================================================

/**
 * @brief Abstract interface for computing scalar integrands
 *
 * FunctionalKernel defines how to compute the local contribution to a
 * scalar functional at each quadrature point or element.
 *
 * Example implementation (L2 norm of solution):
 * @code
 * class L2NormKernel : public FunctionalKernel {
 * public:
 *     Real evaluateCell(const AssemblyContext& ctx, LocalIndex q) override {
 *         Real u = ctx.solutionValue(q);
 *         return u * u;  // Will be multiplied by integration weight
 *     }
 *
 *     bool requiresSquareRoot() const override { return true; }
 * };
 * @endcode
 */
class FunctionalKernel {
public:
    virtual ~FunctionalKernel() = default;

    // =========================================================================
    // Data Requirements
    // =========================================================================

    /**
     * @brief Specify what data this kernel needs
     *
     * @return Bitfield of RequiredData flags
     */
    [[nodiscard]] virtual RequiredData getRequiredData() const noexcept {
        return RequiredData::Standard | RequiredData::SolutionValues;
    }

    /**
     * @brief Optional multi-field requirements
     *
     * For functionals defined using FE/Forms, this can be used to request that
     * additional field values/derivatives be bound into the AssemblyContext
     * (see AssemblyContext::setFieldSolutionScalar/Vector).
     *
     * Default: no extra fields.
     */
    [[nodiscard]] virtual std::vector<FieldRequirement> fieldRequirements() const { return {}; }

    // =========================================================================
    // Cell (Volume) Integration
    // =========================================================================

    /**
     * @brief Evaluate integrand at a single quadrature point
     *
     * Returns the unweighted integrand value. The assembler will multiply
     * by the integration weight (JxW).
     *
     * @param ctx Assembly context with prepared data
     * @param q Quadrature point index
     * @return Integrand value j(x_q)
     */
    [[nodiscard]] virtual Real evaluateCell(
        const AssemblyContext& ctx,
        LocalIndex q) = 0;

    /**
     * @brief Evaluate over entire cell (alternative interface)
     *
     * For kernels that need to see all quadrature points at once
     * (e.g., for jump terms). Default implementation sums point-wise values.
     *
     * @param ctx Assembly context
     * @return Total contribution from this cell (already weighted)
     */
    [[nodiscard]] virtual Real evaluateCellTotal(const AssemblyContext& ctx) {
        Real sum = 0.0;
        for (LocalIndex q = 0; q < ctx.numQuadraturePoints(); ++q) {
            sum += evaluateCell(ctx, q) * ctx.integrationWeight(q);
        }
        return sum;
    }

    /**
     * @brief Check if kernel supports cell integration
     */
    [[nodiscard]] virtual bool hasCell() const noexcept { return true; }

    // =========================================================================
    // Boundary Face Integration
    // =========================================================================

    /**
     * @brief Evaluate integrand at a boundary face quadrature point
     *
     * @param ctx Assembly context for face
     * @param q Quadrature point index
     * @param boundary_marker Boundary label
     * @return Integrand value
     */
    [[nodiscard]] virtual Real evaluateBoundaryFace(
        const AssemblyContext& ctx,
        LocalIndex q,
        int boundary_marker) {
        (void)ctx; (void)q; (void)boundary_marker;
        return 0.0;
    }

    /**
     * @brief Evaluate over entire boundary face (alternative interface)
     *
     * Default implementation sums point-wise values multiplied by integration weights (JxW).
     *
     * @param ctx Assembly context for face
     * @param boundary_marker Boundary label
     * @return Total contribution from this face (already weighted)
     */
    [[nodiscard]] virtual Real evaluateBoundaryFaceTotal(const AssemblyContext& ctx,
                                                         int boundary_marker) {
        Real sum = 0.0;
        for (LocalIndex q = 0; q < ctx.numQuadraturePoints(); ++q) {
            sum += evaluateBoundaryFace(ctx, q, boundary_marker) * ctx.integrationWeight(q);
        }
        return sum;
    }

    /**
     * @brief Check if kernel supports boundary face integration
     */
    [[nodiscard]] virtual bool hasBoundaryFace() const noexcept { return false; }

    // =========================================================================
    // Post-Processing
    // =========================================================================

    /**
     * @brief Post-process the assembled value
     *
     * For norms, this typically involves taking a square root.
     * Default returns the value unchanged.
     *
     * @param raw_value Raw assembled value
     * @return Processed result
     */
    [[nodiscard]] virtual Real postProcess(Real raw_value) const noexcept {
        return raw_value;
    }

    /**
     * @brief Check if result should be square-rooted (for norms)
     */
    [[nodiscard]] virtual bool requiresSquareRoot() const noexcept { return false; }

    // =========================================================================
    // Query
    // =========================================================================

    /**
     * @brief Get kernel name for debugging/logging
     */
    [[nodiscard]] virtual std::string name() const { return "GenericFunctional"; }

    /**
     * @brief Check if functional is linear in the solution
     *
     * Linear functionals can use superposition for efficiency.
     */
    [[nodiscard]] virtual bool isLinear() const noexcept { return false; }
};

// ============================================================================
// Common Functional Kernels
// ============================================================================

/**
 * @brief L2 norm of solution: ||u||_L2 = sqrt(integral u^2 dx)
 */
class L2NormKernel : public FunctionalKernel {
public:
    [[nodiscard]] RequiredData getRequiredData() const noexcept override {
        return RequiredData::IntegrationWeights | RequiredData::SolutionValues;
    }

    [[nodiscard]] Real evaluateCell(const AssemblyContext& ctx, LocalIndex q) override {
        Real u = ctx.solutionValue(q);
        return u * u;
    }

    [[nodiscard]] Real postProcess(Real raw_value) const noexcept override {
        return std::sqrt(raw_value);
    }

    [[nodiscard]] bool requiresSquareRoot() const noexcept override { return true; }
    [[nodiscard]] std::string name() const override { return "L2Norm"; }
};

/**
 * @brief H1 seminorm: |u|_H1 = sqrt(integral |grad u|^2 dx)
 */
class H1SeminormKernel : public FunctionalKernel {
public:
    [[nodiscard]] RequiredData getRequiredData() const noexcept override {
        return RequiredData::IntegrationWeights | RequiredData::SolutionGradients;
    }

    [[nodiscard]] Real evaluateCell(const AssemblyContext& ctx, LocalIndex q) override {
        auto grad_u = ctx.solutionGradient(q);
        Real norm_sq = 0.0;
        for (int d = 0; d < ctx.dimension(); ++d) {
            norm_sq += grad_u[static_cast<std::size_t>(d)] * grad_u[static_cast<std::size_t>(d)];
        }
        return norm_sq;
    }

    [[nodiscard]] Real postProcess(Real raw_value) const noexcept override {
        return std::sqrt(raw_value);
    }

    [[nodiscard]] bool requiresSquareRoot() const noexcept override { return true; }
    [[nodiscard]] std::string name() const override { return "H1Seminorm"; }
};

/**
 * @brief L2 error norm: ||u - u_h||_L2
 *
 * Requires setting the exact solution function.
 */
class L2ErrorKernel : public FunctionalKernel {
public:
    using ExactSolutionFunc = std::function<Real(Real, Real, Real)>;

    explicit L2ErrorKernel(ExactSolutionFunc exact)
        : exact_solution_(std::move(exact)) {}

    [[nodiscard]] RequiredData getRequiredData() const noexcept override {
        return RequiredData::IntegrationWeights | RequiredData::SolutionValues |
               RequiredData::PhysicalPoints;
    }

    [[nodiscard]] Real evaluateCell(const AssemblyContext& ctx, LocalIndex q) override {
        Real u_h = ctx.solutionValue(q);
        auto x = ctx.physicalPoint(q);
        Real u_exact = exact_solution_(x[0], x[1], x[2]);
        Real error = u_h - u_exact;
        return error * error;
    }

    [[nodiscard]] Real postProcess(Real raw_value) const noexcept override {
        return std::sqrt(raw_value);
    }

    [[nodiscard]] bool requiresSquareRoot() const noexcept override { return true; }
    [[nodiscard]] std::string name() const override { return "L2Error"; }

private:
    ExactSolutionFunc exact_solution_;
};

/**
 * @brief H1 error seminorm: |u - u_h|_H1
 */
class H1ErrorKernel : public FunctionalKernel {
public:
    using ExactGradientFunc = std::function<std::array<Real, 3>(Real, Real, Real)>;

    explicit H1ErrorKernel(ExactGradientFunc exact_grad)
        : exact_gradient_(std::move(exact_grad)) {}

    [[nodiscard]] RequiredData getRequiredData() const noexcept override {
        return RequiredData::IntegrationWeights | RequiredData::SolutionGradients |
               RequiredData::PhysicalPoints;
    }

    [[nodiscard]] Real evaluateCell(const AssemblyContext& ctx, LocalIndex q) override {
        auto grad_uh = ctx.solutionGradient(q);
        auto x = ctx.physicalPoint(q);
        auto grad_exact = exact_gradient_(x[0], x[1], x[2]);

        Real norm_sq = 0.0;
        for (int d = 0; d < ctx.dimension(); ++d) {
            Real diff = grad_uh[static_cast<std::size_t>(d)] -
                        grad_exact[static_cast<std::size_t>(d)];
            norm_sq += diff * diff;
        }
        return norm_sq;
    }

    [[nodiscard]] Real postProcess(Real raw_value) const noexcept override {
        return std::sqrt(raw_value);
    }

    [[nodiscard]] bool requiresSquareRoot() const noexcept override { return true; }
    [[nodiscard]] std::string name() const override { return "H1Error"; }

private:
    ExactGradientFunc exact_gradient_;
};

/**
 * @brief Total energy functional: E(u) = 0.5 * integral |grad u|^2 dx
 */
class EnergyKernel : public FunctionalKernel {
public:
    [[nodiscard]] RequiredData getRequiredData() const noexcept override {
        return RequiredData::IntegrationWeights | RequiredData::SolutionGradients;
    }

    [[nodiscard]] Real evaluateCell(const AssemblyContext& ctx, LocalIndex q) override {
        auto grad_u = ctx.solutionGradient(q);
        Real norm_sq = 0.0;
        for (int d = 0; d < ctx.dimension(); ++d) {
            norm_sq += grad_u[static_cast<std::size_t>(d)] * grad_u[static_cast<std::size_t>(d)];
        }
        return 0.5 * norm_sq;
    }

    [[nodiscard]] std::string name() const override { return "Energy"; }
};

/**
 * @brief Volume integral: integral dx
 */
class VolumeKernel : public FunctionalKernel {
public:
    [[nodiscard]] RequiredData getRequiredData() const noexcept override {
        return RequiredData::IntegrationWeights;
    }

    [[nodiscard]] Real evaluateCell(const AssemblyContext& /*ctx*/, LocalIndex /*q*/) override {
        return 1.0;  // Just integrate 1.0
    }

    [[nodiscard]] std::string name() const override { return "Volume"; }
    [[nodiscard]] bool isLinear() const noexcept override { return true; }
};

/**
 * @brief Boundary flux integral: integral_Gamma f . n ds
 */
class BoundaryFluxKernel : public FunctionalKernel {
public:
    /**
     * @brief Construct for integrating grad(u) . n
     */
    BoundaryFluxKernel() = default;

    [[nodiscard]] RequiredData getRequiredData() const noexcept override {
        return RequiredData::IntegrationWeights | RequiredData::SolutionGradients |
               RequiredData::Normals;
    }

    [[nodiscard]] bool hasCell() const noexcept override { return false; }
    [[nodiscard]] bool hasBoundaryFace() const noexcept override { return true; }

    [[nodiscard]] Real evaluateCell(const AssemblyContext& /*ctx*/,
                                     LocalIndex /*q*/) override {
        return 0.0;  // No cell contribution
    }

    [[nodiscard]] Real evaluateBoundaryFace(
        const AssemblyContext& ctx,
        LocalIndex q,
        int /*boundary_marker*/) override
    {
        auto grad_u = ctx.solutionGradient(q);
        auto n = ctx.normal(q);
        Real flux = 0.0;
        for (int d = 0; d < ctx.dimension(); ++d) {
            flux += grad_u[static_cast<std::size_t>(d)] * n[static_cast<std::size_t>(d)];
        }
        return flux;
    }

    [[nodiscard]] std::string name() const override { return "BoundaryFlux"; }
};

// ============================================================================
// Functional Assembly Options
// ============================================================================

/**
 * @brief Options for functional assembly
 */
struct FunctionalAssemblyOptions {
    /**
     * @brief Number of threads for parallel assembly
     */
    int num_threads{1};

    /**
     * @brief Use deterministic reduction (stable floating-point sum)
     */
    bool deterministic{true};

    /**
     * @brief Use Kahan summation for improved accuracy
     */
    bool use_kahan_summation{false};

    /**
     * @brief Verbose output
     */
    bool verbose{false};
};

/**
 * @brief Result from functional assembly
 */
struct FunctionalResult {
    Real value{0.0};                 ///< Computed functional value
    bool success{true};              ///< Whether computation succeeded
    std::string error_message;       ///< Error message if failed

    GlobalIndex elements_processed{0};
    GlobalIndex faces_processed{0};
    double elapsed_seconds{0.0};

    operator Real() const noexcept { return value; }
    operator bool() const noexcept { return success; }
};

// ============================================================================
// Functional Assembler
// ============================================================================

/**
 * @brief Assembler for scalar functionals (Quantities of Interest)
 *
 * FunctionalAssembler computes scalar quantities by integrating over the mesh.
 *
 * Usage:
 * @code
 *   FunctionalAssembler assembler;
 *   assembler.setMesh(mesh);
 *   assembler.setDofMap(dof_map);
 *   assembler.setSpace(space);
 *   assembler.setSolution(solution);
 *
 *   // Compute L2 norm
 *   L2NormKernel kernel;
 *   Real l2_norm = assembler.assembleScalar(kernel);
 *
 *   // Compute multiple functionals in one pass
 *   std::vector<FunctionalKernel*> kernels = {&kernel1, &kernel2};
 *   auto values = assembler.assembleMultiple(kernels);
 * @endcode
 */
class FunctionalAssembler {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    FunctionalAssembler();
    explicit FunctionalAssembler(const FunctionalAssemblyOptions& options);
    ~FunctionalAssembler();

    FunctionalAssembler(FunctionalAssembler&& other) noexcept;
    FunctionalAssembler& operator=(FunctionalAssembler&& other) noexcept;

    // Non-copyable
    FunctionalAssembler(const FunctionalAssembler&) = delete;
    FunctionalAssembler& operator=(const FunctionalAssembler&) = delete;

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
     * @brief Specify which FieldId the configured space/solution represent
     *
     * This enables FE/Forms-driven functional kernels (which use DiscreteField /
     * StateField nodes) to access the current solution through the
     * AssemblyContext field-solution API.
     *
     * Current limitation: only the primary field is supported (multi-field
     * functional evaluation is not yet implemented here).
     */
    void setPrimaryField(FieldId field) noexcept;

    /**
     * @brief Set current solution for evaluation
     *
     * @param solution Solution vector (coefficients)
     */
    void setSolution(std::span<const Real> solution);

    /**
     * @brief Provide a global-indexed solution view (MPI-aware)
     *
     * When set, the assembler will read coefficients from
     * `solution_view->getVectorEntry(dof)` instead of indexing the solution span.
     */
    void setSolutionView(const GlobalSystemView* solution_view) noexcept;

    /**
     * @brief Set previous-step solution coefficients (u^{n-1})
     */
    void setPreviousSolution(std::span<const Real> solution);

    /**
     * @brief Set previous-previous solution coefficients (u^{n-2})
     */
    void setPreviousSolution2(std::span<const Real> solution);

    /**
     * @brief Set k-th previous solution coefficients (u^{n-k}, k>=1)
     */
    void setPreviousSolutionK(int k, std::span<const Real> solution);

    void setPreviousSolutionView(const GlobalSystemView* solution_view) noexcept;
    void setPreviousSolution2View(const GlobalSystemView* solution_view) noexcept;
    void setPreviousSolutionViewK(int k, const GlobalSystemView* solution_view);

    /**
     * @brief Attach a transient time-integration context for `dt(·,k)` lowering
     */
    void setTimeIntegrationContext(const TimeIntegrationContext* ctx) noexcept;

    void setTime(Real time) noexcept;
    void setTimeStep(Real dt) noexcept;

    void setRealParameterGetter(
        const std::function<std::optional<Real>(std::string_view)>* get_real_param) noexcept;

    void setParameterGetter(
        const std::function<std::optional<params::Value>(std::string_view)>* get_param) noexcept;

    void setUserData(const void* user_data) noexcept;

    /**
     * @brief Bind a flat array of Real-valued parameter slots for JIT-friendly kernels.
     */
    void setJITConstants(std::span<const Real> constants) noexcept;

    /**
     * @brief Bind coupled boundary-condition scalar arrays (integrals + auxiliary state).
     */
    void setCoupledValues(std::span<const Real> integrals,
                          std::span<const Real> aux_state) noexcept;

    /**
     * @brief Bind history/convolution weights for history operators
     *
     * Indexing convention: weights[k-1] corresponds to u^{n-k}, k >= 1.
     *
     * The provided memory must remain valid for the duration of an assembly call.
     */
    void setHistoryWeights(std::span<const Real> weights) noexcept;

    /**
     * @brief Set options
     */
    void setOptions(const FunctionalAssemblyOptions& options);

    /**
     * @brief Get current options
     */
    [[nodiscard]] const FunctionalAssemblyOptions& getOptions() const noexcept;

    // =========================================================================
    // Scalar Assembly
    // =========================================================================

    /**
     * @brief Assemble a scalar functional over cells
     *
     * @param kernel Functional kernel
     * @return Assembled value (or FunctionalResult for detailed info)
     */
    [[nodiscard]] Real assembleScalar(FunctionalKernel& kernel);

    /**
     * @brief Assemble scalar with detailed result
     */
    [[nodiscard]] FunctionalResult assembleScalarDetailed(FunctionalKernel& kernel);

    /**
     * @brief Assemble boundary functional
     *
     * @param kernel Functional kernel
     * @param boundary_marker Boundary label to integrate over
     * @return Assembled value
     */
    [[nodiscard]] Real assembleBoundaryScalar(
        FunctionalKernel& kernel,
        int boundary_marker);

    // =========================================================================
    // Multiple Functionals
    // =========================================================================

    /**
     * @brief Assemble multiple functionals in a single mesh pass
     *
     * More efficient than separate calls when computing multiple QoIs.
     *
     * @param kernels Vector of functional kernels
     * @return Vector of assembled values (same order as input)
     */
    [[nodiscard]] std::vector<Real> assembleMultiple(
        std::span<FunctionalKernel* const> kernels);

    // =========================================================================
    // Goal-Oriented Error Estimation Support
    // =========================================================================

    /**
     * @brief Compute element-wise error indicators for goal-oriented adaptivity
     *
     * Uses the dual-weighted residual (DWR) approach:
     *   eta_K = integral_K R(u_h) * z_h dx
     *
     * where R is the residual operator and z_h is the dual solution.
     *
     * @param primal_solution Primal (forward) solution
     * @param dual_solution Dual (adjoint) solution
     * @param kernel Residual kernel
     * @return Element-wise error indicators
     */
    [[nodiscard]] std::vector<Real> computeGoalOrientedIndicators(
        std::span<const Real> primal_solution,
        std::span<const Real> dual_solution,
        FunctionalKernel& kernel);

    // =========================================================================
    // Norm Computations (Convenience)
    // =========================================================================

    /**
     * @brief Compute L2 norm of current solution
     */
    [[nodiscard]] Real computeL2Norm();

    /**
     * @brief Compute H1 seminorm of current solution
     */
    [[nodiscard]] Real computeH1Seminorm();

    /**
     * @brief Compute L2 error against exact solution
     */
    [[nodiscard]] Real computeL2Error(
        std::function<Real(Real, Real, Real)> exact_solution);

    /**
     * @brief Compute H1 error against exact gradient
     */
    [[nodiscard]] Real computeH1Error(
        std::function<std::array<Real, 3>(Real, Real, Real)> exact_gradient);

    /**
     * @brief Compute total energy
     */
    [[nodiscard]] Real computeEnergy();

    /**
     * @brief Compute mesh volume
     */
    [[nodiscard]] Real computeVolume();

    // =========================================================================
    // Query
    // =========================================================================

    /**
     * @brief Check if assembler is configured
     */
    [[nodiscard]] bool isConfigured() const noexcept;

    /**
     * @brief Get last assembly result
     */
    [[nodiscard]] const FunctionalResult& getLastResult() const noexcept;

private:
    // =========================================================================
    // Internal Implementation
    // =========================================================================

    void bindFieldSolutionData(AssemblyContext& context, std::span<const FieldRequirement> reqs);

    /**
     * @brief Core assembly loop for cells
     */
    Real assembleCellsCore(FunctionalKernel& kernel, FunctionalResult& result);

    /**
     * @brief Core assembly loop for boundary faces
     */
    Real assembleBoundaryCore(
        FunctionalKernel& kernel,
        int boundary_marker,
        FunctionalResult& result);

    /**
     * @brief Prepare context for element
     */
    void prepareContext(GlobalIndex cell_id, RequiredData required_data);

    /**
     * @brief Kahan summation accumulator
     */
    struct KahanAccumulator {
        Real sum{0.0};
        Real compensation{0.0};

        void add(Real value) {
            Real y = value - compensation;
            Real t = sum + y;
            compensation = (t - sum) - y;
            sum = t;
        }

        Real get() const { return sum; }
    };

    /**
     * @brief Parallel reduction with deterministic order
     */
    Real parallelReduce(
        const std::vector<Real>& local_values,
        bool use_kahan);

    // =========================================================================
    // Data Members
    // =========================================================================

    FunctionalAssemblyOptions options_;
    const IMeshAccess* mesh_{nullptr};
    const dofs::DofMap* dof_map_{nullptr};
    const spaces::FunctionSpace* space_{nullptr};
    FieldId primary_field_{INVALID_FIELD_ID};
    std::vector<Real> solution_;
    const GlobalSystemView* solution_view_{nullptr};
    std::vector<std::vector<Real>> previous_solutions_{};
    std::vector<const GlobalSystemView*> previous_solution_views_{};

    const TimeIntegrationContext* time_integration_{nullptr};
    Real time_{0.0};
    Real dt_{0.0};
    const std::function<std::optional<Real>(std::string_view)>* get_real_param_{nullptr};
    const std::function<std::optional<params::Value>(std::string_view)>* get_param_{nullptr};
    const void* user_data_{nullptr};
    std::span<const Real> jit_constants_{};
    std::span<const Real> coupled_integrals_{};
    std::span<const Real> coupled_aux_state_{};
    std::span<const Real> history_weights_{};

    // Working storage
    AssemblyContext context_;
    std::vector<GlobalIndex> cell_dofs_;

    // Thread-local contexts (for parallel assembly)
    std::vector<std::unique_ptr<AssemblyContext>> thread_contexts_;

    // Results
    FunctionalResult last_result_;
};

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * @brief Create a functional assembler with default options
 */
std::unique_ptr<FunctionalAssembler> createFunctionalAssembler();

/**
 * @brief Create a functional assembler with specified options
 */
std::unique_ptr<FunctionalAssembler> createFunctionalAssembler(
    const FunctionalAssemblyOptions& options);

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_FUNCTIONAL_ASSEMBLER_H
