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

#ifndef SVMP_FE_ASSEMBLY_PARALLEL_ASSEMBLER_H
#define SVMP_FE_ASSEMBLY_PARALLEL_ASSEMBLER_H

/**
 * @file ParallelAssembler.h
 * @brief MPI-parallel assembly coordination
 *
 * ParallelAssembler extends StandardAssembler for distributed-memory
 * parallelism using MPI. It handles:
 *
 * - Iteration over locally owned cells
 * - Ghost contribution management via GhostContributionManager
 * - Communication of off-process contributions
 * - Synchronization points for collective operations
 *
 * Ghost policies:
 * - OwnedRowsOnly: Only assemble to locally owned rows (simpler, faster)
 * - ReverseScatter: Assemble everywhere, then reverse-scatter to owners (more accurate)
 *
 * Communication strategies:
 * - Blocking: Exchange all ghost contributions at end of assembly
 * - Non-blocking: Overlap communication with computation (if supported)
 * - Batched: Accumulate contributions, exchange periodically
 *
 * Determinism requirements:
 * - Results must be reproducible given the same mesh partition
 * - Ghost accumulation uses stable ordering for floating-point consistency
 *
 * @see GhostContributionManager for ghost contribution handling
 * @see StandardAssembler for serial assembly base
 */

#include "Assembler.h"
#include "StandardAssembler.h"
#include "GhostContributionManager.h"
#include "Core/FEConfig.h"

#include <memory>
#include <vector>

#if FE_HAS_MPI
#  include <mpi.h>
#endif

namespace svmp {
namespace FE {

// Forward declarations
namespace dofs {
    class DofMap;
    class DofHandler;
    class GhostDofManager;
}

namespace assembly {

/**
 * @brief MPI-parallel assembler
 *
 * ParallelAssembler orchestrates distributed assembly across MPI ranks.
 * Each rank assembles contributions for its local cells, and ghost
 * contributions are communicated via reverse scatter.
 *
 * Usage:
 * @code
 *   ParallelAssembler assembler(MPI_COMM_WORLD);
 *   assembler.setDofHandler(dof_handler);
 *   assembler.setGhostPolicy(GhostPolicy::ReverseScatter);
 *   assembler.initialize();
 *
 *   // Assembly produces distributed matrix/vector
 *   auto result = assembler.assembleBoth(mesh, space, kernel,
 *                                        matrix_view, vector_view);
 *
 *   // Finalize triggers ghost exchange
 *   assembler.finalize(&matrix_view, &vector_view);
 * @endcode
 */
class ParallelAssembler : public Assembler {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor (uses MPI_COMM_WORLD)
     */
    ParallelAssembler();

#if FE_HAS_MPI
    /**
     * @brief Construct with MPI communicator
     */
    explicit ParallelAssembler(MPI_Comm comm);
#endif

    /**
     * @brief Construct with options
     */
    explicit ParallelAssembler(const AssemblyOptions& options);

    /**
     * @brief Destructor
     */
    ~ParallelAssembler() override;

    /**
     * @brief Move constructor
     */
    ParallelAssembler(ParallelAssembler&& other) noexcept;

    /**
     * @brief Move assignment
     */
    ParallelAssembler& operator=(ParallelAssembler&& other) noexcept;

    // Non-copyable
    ParallelAssembler(const ParallelAssembler&) = delete;
    ParallelAssembler& operator=(const ParallelAssembler&) = delete;

    // =========================================================================
    // Configuration (Assembler interface)
    // =========================================================================

    void setDofMap(const dofs::DofMap& dof_map) override;
    void setRowDofMap(const dofs::DofMap& dof_map, GlobalIndex row_offset = 0) override;
    void setColDofMap(const dofs::DofMap& dof_map, GlobalIndex col_offset = 0) override;
    void setDofHandler(const dofs::DofHandler& dof_handler) override;
    void setConstraints(const constraints::AffineConstraints* constraints) override;
    void setSparsityPattern(const sparsity::SparsityPattern* sparsity) override;
    void setOptions(const AssemblyOptions& options) override;
    void setCurrentSolution(std::span<const Real> solution) override;
    void setCurrentSolutionView(const GlobalSystemView* solution_view) override;
    void setFieldSolutionAccess(std::span<const FieldSolutionAccess> fields) override;
    void setPreviousSolution(std::span<const Real> solution) override;
    void setPreviousSolution2(std::span<const Real> solution) override;
    void setPreviousSolutionView(const GlobalSystemView* solution_view) override;
    void setPreviousSolution2View(const GlobalSystemView* solution_view) override;
    void setPreviousSolutionK(int k, std::span<const Real> solution) override;
    void setPreviousSolutionViewK(int k, const GlobalSystemView* solution_view) override;
    void setTimeIntegrationContext(const TimeIntegrationContext* ctx) override;
    void setTime(Real time) override;
    void setTimeStep(Real dt) override;
    void setRealParameterGetter(
        const std::function<std::optional<Real>(std::string_view)>* get_real_param) noexcept override;
    void setParameterGetter(
        const std::function<std::optional<params::Value>(std::string_view)>* get_param) noexcept override;
    void setUserData(const void* user_data) noexcept override;
    void setJITConstants(std::span<const Real> constants) noexcept override;
    void setCoupledValues(std::span<const Real> integrals,
                          std::span<const Real> aux_state) noexcept override;
    void setMaterialStateProvider(IMaterialStateProvider* provider) noexcept override;
    [[nodiscard]] const AssemblyOptions& getOptions() const noexcept override;

    // =========================================================================
    // Parallel-specific configuration
    // =========================================================================

#if FE_HAS_MPI
    /**
     * @brief Set MPI communicator
     */
    void setComm(MPI_Comm comm);

    /**
     * @brief Get MPI communicator
     */
    [[nodiscard]] MPI_Comm getComm() const noexcept { return comm_; }
#endif

    /**
     * @brief Set ghost DOF manager for ownership information
     */
    void setGhostDofManager(const dofs::GhostDofManager& ghost_manager);

    /**
     * @brief Set ghost contribution policy
     */
    void setGhostPolicy(GhostPolicy policy);

    /**
     * @brief Get current ghost policy
     */
    [[nodiscard]] GhostPolicy getGhostPolicy() const noexcept {
        return ghost_policy_;
    }

    /**
     * @brief Enable/disable communication overlap
     */
    void setOverlapCommunication(bool overlap) { overlap_comm_ = overlap; }

    // =========================================================================
    // Matrix Assembly
    // =========================================================================

    [[nodiscard]] AssemblyResult assembleMatrix(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view) override;

    // =========================================================================
    // Vector Assembly
    // =========================================================================

    [[nodiscard]] AssemblyResult assembleVector(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& space,
        AssemblyKernel& kernel,
        GlobalSystemView& vector_view) override;

    // =========================================================================
    // Combined Assembly
    // =========================================================================

    [[nodiscard]] AssemblyResult assembleBoth(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view,
        GlobalSystemView& vector_view) override;

    // =========================================================================
    // Face Assembly
    // =========================================================================

    [[nodiscard]] AssemblyResult assembleBoundaryFaces(
        const IMeshAccess& mesh,
        int boundary_marker,
        const spaces::FunctionSpace& space,
        AssemblyKernel& kernel,
        GlobalSystemView* matrix_view,
        GlobalSystemView* vector_view) override;

    [[nodiscard]] AssemblyResult assembleBoundaryFaces(
        const IMeshAccess& mesh,
        int boundary_marker,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView* matrix_view,
        GlobalSystemView* vector_view) override;

    [[nodiscard]] AssemblyResult assembleInteriorFaces(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view,
        GlobalSystemView* vector_view) override;

    // =========================================================================
    // Lifecycle
    // =========================================================================

    void initialize() override;
    void finalize(GlobalSystemView* matrix_view, GlobalSystemView* vector_view) override;
    void reset() override;

    // =========================================================================
    // Query
    // =========================================================================

    [[nodiscard]] std::string name() const override { return "ParallelAssembler"; }
    [[nodiscard]] bool isConfigured() const noexcept override;
    [[nodiscard]] bool supportsRectangular() const noexcept override { return true; }
    [[nodiscard]] bool supportsDG() const noexcept override { return true; }
    [[nodiscard]] bool supportsFullContext() const noexcept override { return true; }
    [[nodiscard]] bool supportsSolution() const noexcept override { return true; }
    [[nodiscard]] bool supportsSolutionHistory() const noexcept override { return true; }
    [[nodiscard]] bool supportsTimeIntegrationContext() const noexcept override { return true; }
    [[nodiscard]] bool supportsDofOffsets() const noexcept override { return true; }
    [[nodiscard]] bool supportsFieldRequirements() const noexcept override { return true; }
    [[nodiscard]] bool supportsMaterialState() const noexcept override { return true; }
    [[nodiscard]] bool isThreadSafe() const noexcept override { return false; }

    /**
     * @brief Get MPI rank
     */
    [[nodiscard]] int rank() const noexcept { return my_rank_; }

    /**
     * @brief Get number of MPI ranks
     */
    [[nodiscard]] int numRanks() const noexcept { return world_size_; }

    /**
     * @brief Get ghost contribution manager for advanced use
     */
    [[nodiscard]] GhostContributionManager& getGhostManager() noexcept {
        return ghost_manager_;
    }

private:
    // =========================================================================
    // Internal Implementation
    // =========================================================================

    void beginGhostAssemblyIfNeeded();

    /**
     * @brief Core assembly loop for cells (parallel version)
     */
    AssemblyResult assembleCellsParallel(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView* matrix_view,
        GlobalSystemView* vector_view,
        bool assemble_matrix,
        bool assemble_vector);

    /**
     * @brief Exchange ghost contributions
     */
    void exchangeGhostContributions();

    /**
     * @brief Apply received ghost contributions to global system
     */
    void applyReceivedContributions(
        GlobalSystemView* matrix_view,
        GlobalSystemView* vector_view);

    /**
     * @brief Insert local contribution, routing ghosts to manager
     */
    void insertLocalWithGhostHandling(
        const KernelOutput& output,
        std::span<const GlobalIndex> row_dofs,
        std::span<const GlobalIndex> col_dofs,
        GlobalSystemView* matrix_view,
        GlobalSystemView* vector_view);

    // Configuration
    AssemblyOptions options_;
    const dofs::DofMap* dof_map_{nullptr};
    const dofs::DofHandler* dof_handler_{nullptr};
    const constraints::AffineConstraints* constraints_{nullptr};
    const sparsity::SparsityPattern* sparsity_{nullptr};
    const dofs::GhostDofManager* ghost_dof_manager_{nullptr};

    // MPI state
#if FE_HAS_MPI
    MPI_Comm comm_{MPI_COMM_WORLD};
#endif
    int my_rank_{0};
    int world_size_{1};

    // Ghost handling
    GhostPolicy ghost_policy_{GhostPolicy::ReverseScatter};
    GhostContributionManager ghost_manager_;
    bool overlap_comm_{false};

    // Local assembler for element operations
    StandardAssembler local_assembler_;

    // Working storage
    AssemblyContext context_;
    KernelOutput kernel_output_;
    std::vector<GlobalIndex> row_dofs_;
    std::vector<GlobalIndex> col_dofs_;
    std::vector<GhostContribution> owned_contributions_;

    // State
    bool initialized_{false};
    bool assembly_in_progress_{false};
};

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * @brief Create a parallel assembler
 */
std::unique_ptr<Assembler> createParallelAssembler();

#if FE_HAS_MPI
/**
 * @brief Create a parallel assembler with specified communicator
 */
std::unique_ptr<Assembler> createParallelAssembler(MPI_Comm comm);
#endif

/**
 * @brief Create a parallel assembler with options
 */
std::unique_ptr<Assembler> createParallelAssembler(const AssemblyOptions& options);

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_PARALLEL_ASSEMBLER_H
