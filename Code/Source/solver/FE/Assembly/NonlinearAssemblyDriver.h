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

#ifndef SVMP_FE_ASSEMBLY_NONLINEAR_ASSEMBLY_DRIVER_H
#define SVMP_FE_ASSEMBLY_NONLINEAR_ASSEMBLY_DRIVER_H

/**
 * @file NonlinearAssemblyDriver.h
 * @brief Orchestration of residual and Jacobian assembly for nonlinear problems
 *
 * NonlinearAssemblyDriver provides the assembly infrastructure for nonlinear
 * finite element problems. It coordinates:
 *
 * - Residual vector assembly: F(u)
 * - Jacobian matrix assembly: J = dF/du
 * - Combined residual/Jacobian assembly (efficient for Newton methods)
 * - Finite difference Jacobian verification
 * - AD-assisted assembly (when Sacado is available)
 *
 * Key design principles:
 * - Efficient combined assembly (compute local F and J together)
 * - Support for different Jacobian computation strategies
 * - Verification tools for debugging new physics implementations
 * - Integration with nonlinear solvers (Newton, SNES, etc.)
 *
 * Assembly modes:
 * - AssembleResidualOnly: Only F(u), no Jacobian
 * - AssembleJacobianOnly: Only J, residual assumed already computed
 * - AssembleBoth: Combined assembly (most efficient for Newton)
 *
 * Module boundaries:
 * - This module OWNS: assembly orchestration for nonlinear problems
 * - This module does NOT OWN: physics kernels, solver iteration, material models
 *
 * @see AssemblyKernel for nonlinear kernel interface
 * @see AssemblyLoop for the underlying loop infrastructure
 */

#include "Core/Types.h"
#include "Core/FEException.h"
#include "Assembler.h"
#include "AssemblyKernel.h"
#include "AssemblyLoop.h"
#include "GlobalSystemView.h"

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

// Forward declaration
class AssemblyConstraintDistributor;

// ============================================================================
// Nonlinear Assembly Options
// ============================================================================

/**
 * @brief What to assemble
 */
enum class NonlinearAssemblyMode : std::uint8_t {
    ResidualOnly,    ///< Only assemble F(u)
    JacobianOnly,    ///< Only assemble J = dF/du
    Both             ///< Assemble both F(u) and J (efficient combined)
};

/**
 * @brief Jacobian computation strategy
 */
enum class JacobianStrategy : std::uint8_t {
    Analytic,        ///< Use analytically-derived Jacobian from kernel
    AD_Forward,      ///< Use forward-mode automatic differentiation (Sacado)
    AD_Reverse,      ///< Use reverse-mode automatic differentiation
    FiniteDifference ///< Finite difference approximation (for debugging)
};

/**
 * @brief Options for nonlinear assembly
 */
struct NonlinearAssemblyOptions {
    /**
     * @brief What to assemble
     */
    NonlinearAssemblyMode mode{NonlinearAssemblyMode::Both};

    /**
     * @brief How to compute Jacobian
     */
    JacobianStrategy jacobian_strategy{JacobianStrategy::Analytic};

    /**
     * @brief Perturbation for finite difference Jacobian
     */
    Real fd_perturbation{1e-7};

    /**
     * @brief Use relative perturbation for finite difference
     *
     * If true: h = fd_perturbation * (1 + |u_i|)
     * If false: h = fd_perturbation
     */
    bool fd_relative_perturbation{true};

    /**
     * @brief Loop options for cell/face iteration
     */
    LoopOptions loop_options{};

    /**
     * @brief Apply constraints during assembly
     */
    bool apply_constraints{true};

    /**
     * @brief Zero residual/Jacobian before assembly
     */
    bool zero_before_assembly{true};

    /**
     * @brief Include boundary terms
     */
    bool include_boundary{true};

    /**
     * @brief Boundary markers to include
     */
    std::vector<int> boundary_markers;

    /**
     * @brief Include interior faces (for DG)
     */
    bool include_interior_faces{false};

    /**
     * @brief Verbose output
     */
    bool verbose{false};
};

/**
 * @brief Result of Jacobian verification
 */
struct JacobianVerificationResult {
    bool passed{false};             ///< Whether verification passed
    Real max_abs_error{0.0};        ///< Maximum absolute error
    Real max_rel_error{0.0};        ///< Maximum relative error
    Real avg_abs_error{0.0};        ///< Average absolute error
    GlobalIndex worst_row{-1};      ///< Row with worst error
    GlobalIndex worst_col{-1};      ///< Column with worst error
    std::size_t num_entries_checked{0}; ///< Number of entries verified
    std::string message;            ///< Diagnostic message
};

/**
 * @brief Statistics from nonlinear assembly
 */
struct NonlinearAssemblyStats {
    GlobalIndex num_cells{0};               ///< Cells processed
    GlobalIndex num_boundary_faces{0};      ///< Boundary faces processed
    GlobalIndex num_interior_faces{0};      ///< Interior faces processed
    double residual_assembly_seconds{0.0};  ///< Time for residual assembly
    double jacobian_assembly_seconds{0.0};  ///< Time for Jacobian assembly
    double total_seconds{0.0};              ///< Total assembly time
    Real residual_norm{0.0};                ///< Norm of assembled residual
};

// ============================================================================
// Nonlinear Kernel Interface
// ============================================================================

/**
 * @brief Extended kernel interface for nonlinear problems
 *
 * Kernels for nonlinear problems must implement at least one of:
 * - computeResidual: Evaluate F(u) for given solution u
 * - computeJacobian: Evaluate J = dF/du
 * - computeBoth: Combined computation (most efficient)
 */
class INonlinearKernel {
public:
    virtual ~INonlinearKernel() = default;

    /**
     * @brief Compute element residual vector
     *
     * @param context Assembly context with solution values
     * @param output Output for local residual
     */
    virtual void computeResidual(
        AssemblyContext& context,
        KernelOutput& output) = 0;

    /**
     * @brief Compute element Jacobian matrix
     *
     * @param context Assembly context with solution values
     * @param output Output for local Jacobian
     */
    virtual void computeJacobian(
        AssemblyContext& context,
        KernelOutput& output) = 0;

    /**
     * @brief Compute both residual and Jacobian (for efficiency)
     *
     * Default implementation calls computeResidual and computeJacobian separately.
     *
     * @param context Assembly context
     * @param output Output for both local residual and Jacobian
     */
    virtual void computeBoth(
        AssemblyContext& context,
        KernelOutput& output)
    {
        computeResidual(context, output);
        computeJacobian(context, output);
    }

    /**
     * @brief Check if kernel can compute residual
     */
    [[nodiscard]] virtual bool canComputeResidual() const noexcept { return true; }

    /**
     * @brief Check if kernel can compute Jacobian
     */
    [[nodiscard]] virtual bool canComputeJacobian() const noexcept { return true; }

    /**
     * @brief Check if kernel has optimized combined computation
     */
    [[nodiscard]] virtual bool hasOptimizedBoth() const noexcept { return false; }

    /**
     * @brief Get required data for residual evaluation
     */
    [[nodiscard]] virtual RequiredData getResidualRequiredData() const noexcept {
        return RequiredData::SolutionValues | RequiredData::BasisGradients;
    }

    /**
     * @brief Get required data for Jacobian evaluation
     */
    [[nodiscard]] virtual RequiredData getJacobianRequiredData() const noexcept {
        return getResidualRequiredData();
    }
};

// ============================================================================
// Nonlinear Assembly Driver
// ============================================================================

/**
 * @brief Driver for nonlinear FE assembly
 *
 * NonlinearAssemblyDriver coordinates the assembly of residuals and Jacobians
 * for nonlinear finite element problems. It integrates with:
 *
 * - AssemblyLoop for element/face iteration
 * - INonlinearKernel for physics evaluation
 * - GlobalSystemView for matrix/vector storage
 * - Constraints module for BC handling
 *
 * Usage:
 * @code
 *   NonlinearAssemblyDriver driver;
 *   driver.setMesh(mesh);
 *   driver.setDofMap(dof_map);
 *   driver.setSpace(space);
 *   driver.setKernel(kernel);
 *   driver.setConstraints(constraints);
 *
 *   // During Newton iteration:
 *   while (!converged) {
 *       driver.setCurrentSolution(u);
 *
 *       // Assemble F(u) and J
 *       driver.assembleBoth(jacobian_view, residual_view);
 *
 *       // Solve J * delta_u = -F
 *       solve(jacobian_view, residual_view, delta_u);
 *
 *       u += delta_u;
 *   }
 * @endcode
 */
class NonlinearAssemblyDriver {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor
     */
    NonlinearAssemblyDriver();

    /**
     * @brief Construct with options
     */
    explicit NonlinearAssemblyDriver(const NonlinearAssemblyOptions& options);

    /**
     * @brief Destructor
     */
    ~NonlinearAssemblyDriver();

    /**
     * @brief Move constructor
     */
    NonlinearAssemblyDriver(NonlinearAssemblyDriver&& other) noexcept;

    /**
     * @brief Move assignment
     */
    NonlinearAssemblyDriver& operator=(NonlinearAssemblyDriver&& other) noexcept;

    // Non-copyable
    NonlinearAssemblyDriver(const NonlinearAssemblyDriver&) = delete;
    NonlinearAssemblyDriver& operator=(const NonlinearAssemblyDriver&) = delete;

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
     * @brief Set test and trial spaces (for non-symmetric problems)
     */
    void setSpaces(const spaces::FunctionSpace& test_space,
                   const spaces::FunctionSpace& trial_space);

    /**
     * @brief Set the nonlinear kernel
     */
    void setKernel(INonlinearKernel& kernel);

    /**
     * @brief Set constraints
     */
    void setConstraints(const constraints::AffineConstraints& constraints);

    /**
     * @brief Set assembly options
     */
    void setOptions(const NonlinearAssemblyOptions& options);

    /**
     * @brief Get current options
     */
    [[nodiscard]] const NonlinearAssemblyOptions& getOptions() const noexcept {
        return options_;
    }

    // =========================================================================
    // Solution Management
    // =========================================================================

    /**
     * @brief Set current solution vector for assembly
     *
     * The solution values are used to evaluate F(u) and J(u).
     *
     * @param solution Current solution vector
     */
    void setCurrentSolution(std::span<const Real> solution);

    /**
     * @brief Set current solution from GlobalSystemView
     */
    void setCurrentSolution(const GlobalSystemView& solution_view);

    /**
     * @brief Get current solution
     */
    [[nodiscard]] std::span<const Real> getCurrentSolution() const noexcept {
        return current_solution_;
    }

    // =========================================================================
    // Assembly Operations
    // =========================================================================

    /**
     * @brief Assemble only the residual vector F(u)
     *
     * @param residual_view Output: global residual vector
     * @return Assembly statistics
     */
    NonlinearAssemblyStats assembleResidual(GlobalSystemView& residual_view);

    /**
     * @brief Assemble only the Jacobian matrix J = dF/du
     *
     * @param jacobian_view Output: global Jacobian matrix
     * @return Assembly statistics
     */
    NonlinearAssemblyStats assembleJacobian(GlobalSystemView& jacobian_view);

    /**
     * @brief Assemble both residual and Jacobian (most efficient)
     *
     * For Newton methods, this is the preferred approach as many computations
     * can be shared between F and J evaluation.
     *
     * @param jacobian_view Output: global Jacobian matrix
     * @param residual_view Output: global residual vector
     * @return Assembly statistics
     */
    NonlinearAssemblyStats assembleBoth(
        GlobalSystemView& jacobian_view,
        GlobalSystemView& residual_view);

    // =========================================================================
    // Jacobian Verification
    // =========================================================================

    /**
     * @brief Verify Jacobian using finite differences
     *
     * Computes J_fd using finite differences and compares to the analytic J.
     * Useful for debugging new physics implementations.
     *
     * @param jacobian_view Analytic Jacobian (must be assembled first)
     * @param tol Tolerance for comparison
     * @return Verification result
     */
    JacobianVerificationResult verifyJacobianFD(
        const GlobalSystemView& jacobian_view,
        Real tol = 1e-5);

    /**
     * @brief Verify Jacobian for specific rows
     *
     * @param jacobian_view Analytic Jacobian
     * @param rows Rows to verify
     * @param tol Tolerance
     * @return Verification result
     */
    JacobianVerificationResult verifyJacobianFD(
        const GlobalSystemView& jacobian_view,
        std::span<const GlobalIndex> rows,
        Real tol = 1e-5);

    /**
     * @brief Compute finite difference Jacobian
     *
     * Assembles the full Jacobian using finite differences.
     * Expensive but useful for debugging.
     *
     * @param fd_jacobian_view Output: FD Jacobian matrix
     * @return Assembly statistics
     */
    NonlinearAssemblyStats assembleJacobianFD(GlobalSystemView& fd_jacobian_view);

    // =========================================================================
    // Query
    // =========================================================================

    /**
     * @brief Get last assembly statistics
     */
    [[nodiscard]] const NonlinearAssemblyStats& getLastStats() const noexcept {
        return last_stats_;
    }

    /**
     * @brief Check if driver is properly configured
     */
    [[nodiscard]] bool isConfigured() const noexcept;

private:
    // =========================================================================
    // Internal Implementation
    // =========================================================================

    /**
     * @brief Prepare assembly context with solution values
     */
    void prepareSolutionValues(AssemblyContext& context, GlobalIndex cell_id, RequiredData required_data);

    /**
     * @brief Cell loop for residual assembly
     */
    void residualCellLoop(GlobalSystemView& residual_view);

    /**
     * @brief Cell loop for Jacobian assembly
     */
    void jacobianCellLoop(GlobalSystemView& jacobian_view);

    /**
     * @brief Cell loop for combined assembly
     */
    void combinedCellLoop(GlobalSystemView& jacobian_view, GlobalSystemView& residual_view);

    /**
     * @brief Boundary loop for residual
     */
    void residualBoundaryLoop(GlobalSystemView& residual_view);

    /**
     * @brief Boundary loop for Jacobian
     */
    void jacobianBoundaryLoop(GlobalSystemView& jacobian_view);

    /**
     * @brief Apply constraints to assembled system
     */
    void applyConstraints(GlobalSystemView* jacobian_view, GlobalSystemView* residual_view);

    /**
     * @brief Compute residual norm
     */
    Real computeResidualNorm(const GlobalSystemView& residual_view);

    /**
     * @brief Compute FD Jacobian column
     */
    void computeFDColumn(GlobalIndex col,
                         std::vector<Real>& column,
                         Real perturbation);

    // =========================================================================
    // Data Members
    // =========================================================================

    // Configuration
    NonlinearAssemblyOptions options_;
    const IMeshAccess* mesh_{nullptr};
    const dofs::DofMap* dof_map_{nullptr};
    const spaces::FunctionSpace* test_space_{nullptr};
    const spaces::FunctionSpace* trial_space_{nullptr};
    INonlinearKernel* kernel_{nullptr};
    const constraints::AffineConstraints* constraints_{nullptr};

    // Constraint distributor
    std::unique_ptr<AssemblyConstraintDistributor> constraint_distributor_;

    // Assembly loop
    std::unique_ptr<AssemblyLoop> loop_;

    // Current solution
    std::vector<Real> current_solution_;

    // Scratch space
    std::vector<Real> perturbed_solution_;
    std::vector<Real> residual_base_;
    std::vector<Real> residual_perturbed_;

    // Statistics
    NonlinearAssemblyStats last_stats_;
};

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * @brief Create a nonlinear assembly driver
 */
std::unique_ptr<NonlinearAssemblyDriver> createNonlinearAssemblyDriver(
    const NonlinearAssemblyOptions& options = {});

/**
 * @brief One-shot residual assembly
 */
NonlinearAssemblyStats assembleResidual(
    const IMeshAccess& mesh,
    const dofs::DofMap& dof_map,
    const spaces::FunctionSpace& space,
    INonlinearKernel& kernel,
    std::span<const Real> solution,
    GlobalSystemView& residual_view,
    const constraints::AffineConstraints* constraints = nullptr);

/**
 * @brief One-shot residual and Jacobian assembly
 */
NonlinearAssemblyStats assembleBoth(
    const IMeshAccess& mesh,
    const dofs::DofMap& dof_map,
    const spaces::FunctionSpace& space,
    INonlinearKernel& kernel,
    std::span<const Real> solution,
    GlobalSystemView& jacobian_view,
    GlobalSystemView& residual_view,
    const constraints::AffineConstraints* constraints = nullptr);

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_NONLINEAR_ASSEMBLY_DRIVER_H
