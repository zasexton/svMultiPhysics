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

#ifndef SVMP_FE_ASSEMBLY_ASSEMBLER_H
#define SVMP_FE_ASSEMBLY_ASSEMBLER_H

/**
 * @file Assembler.h
 * @brief Abstract base interface for assembly strategies
 *
 * The Assembler class defines the abstract interface for all assembly strategies
 * in the FE library. Assembly is the process of accumulating element-local
 * contributions into global matrices and vectors.
 *
 * Design principles:
 * - Assembly = Orchestrate + Insert, NOT Physics
 * - Physics kernels provide local element matrices/vectors
 * - Assembly orchestrates the element loop and insertion
 * - Backend storage is abstracted via GlobalSystemView
 *
 * Key features:
 * - Support for rectangular operators (test_space != trial_space)
 * - Constraint-aware assembly (via ConstraintDistributor integration)
 * - Thread-safety requirements clearly documented
 * - Deterministic results for reproducibility
 *
 * Module boundaries:
 * - This module OWNS: loop orchestration, insertion mechanics, strategy selection
 * - This module does NOT OWN: physics kernels, matrix storage, DOF numbering,
 *   constraint definitions, sparsity patterns
 *
 * @see StandardAssembler for traditional element-by-element assembly
 * @see ParallelAssembler for MPI-parallel assembly
 * @see GlobalSystemView for backend-neutral insertion
 */

#include "Core/Types.h"
#include "Core/FEException.h"
#include "Core/ParameterValue.h"
#include "Assembly/TimeIntegrationContext.h"
#include "Assembly/Coloring.h"

#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>
#include <string>
#include <string_view>
#include <functional>
#include <optional>

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
namespace svmp {
class InterfaceMesh;
}
#endif

namespace svmp {

namespace FE {

// Forward declarations
namespace dofs {
    class DofMap;
    class DofHandler;
}

namespace constraints {
    class AffineConstraints;
}

namespace sparsity {
    class SparsityPattern;
}

namespace spaces {
    class FunctionSpace;
}

namespace assembly {

// Forward declarations
class GlobalSystemView;
class AssemblyKernel;
class AssemblyContext;

/**
 * @brief View of per-integration-point state storage for a single cell
 *
 * The assembler is responsible for binding a view for the current cell into
 * AssemblyContext when a kernel requests RequiredData::MaterialState.
 */
struct MaterialStateView {
    std::byte* data_old{nullptr};      ///< Base pointer for "old" state (may be null/alias)
    std::byte* data_work{nullptr};     ///< Base pointer for "work/current" state
    std::size_t bytes_per_qpt{0};      ///< Payload bytes per integration point
    std::size_t stride_bytes{0};       ///< Stride between integration points (>= bytes_per_qpt)
    std::size_t alignment{alignof(std::max_align_t)}; ///< Alignment guarantee for base pointers

    [[nodiscard]] explicit operator bool() const noexcept { return data_work != nullptr; }
};

/**
 * @brief Mapping information for accessing a field's solution coefficients in a global state vector
 *
 * This describes how a field identified by FieldId is embedded into the global
 * system state (typically a block-structured vector). Assemblers that support
 * multi-field workflows can use this to gather per-cell coefficients for
 * additional fields requested by kernels.
 */
struct FieldSolutionAccess {
    FieldId field{INVALID_FIELD_ID};
    const spaces::FunctionSpace* space{nullptr};
    const dofs::DofMap* dof_map{nullptr};
    GlobalIndex dof_offset{0};
};

/**
 * @brief Provider interface for per-cell material state
 *
 * Implemented by higher-level orchestration layers (typically FE/Systems)
 * to allocate and provide stable storage for kernels that request state.
 */
class IMaterialStateProvider {
public:
    virtual ~IMaterialStateProvider() = default;

    [[nodiscard]] virtual MaterialStateView getCellState(const AssemblyKernel& kernel,
                                                         GlobalIndex cell_id,
                                                         LocalIndex num_qpts) = 0;

    /**
     * @brief Provide per-integration-point state storage for a boundary face
     *
     * The default implementation returns an empty view (unsupported).
     */
    [[nodiscard]] virtual MaterialStateView getBoundaryFaceState(const AssemblyKernel& /*kernel*/,
                                                                 GlobalIndex /*face_id*/,
                                                                 LocalIndex /*num_qpts*/)
    {
        return {};
    }

    /**
     * @brief Provide per-integration-point state storage for an interior face (DG/contact)
     *
     * The default implementation returns an empty view (unsupported).
     */
    [[nodiscard]] virtual MaterialStateView getInteriorFaceState(const AssemblyKernel& /*kernel*/,
                                                                 GlobalIndex /*face_id*/,
                                                                 LocalIndex /*num_qpts*/)
    {
        return {};
    }

    /**
     * @brief Prepare "work" state from "old" at the start of a time step
     *
     * Default is a no-op.
     */
    virtual void beginTimeStep() {}

    /**
     * @brief Commit the current time step (e.g., swap old/work buffers)
     *
     * Default is a no-op.
     */
    virtual void commitTimeStep() {}
};

// ============================================================================
// Assembly Options and Configuration
// ============================================================================

/**
 * @brief Assembly mode for matrix/vector operations
 */
enum class AssemblyMode : std::uint8_t {
    Add,       ///< Add values to existing entries (default)
    Insert,    ///< Insert values, replacing existing entries
    Flush      ///< Force communication/finalization
};

/**
 * @brief Threading strategy for assembly
 */
enum class ThreadingStrategy : std::uint8_t {
    Sequential,     ///< Single-threaded, serial assembly
    Colored,        ///< Graph coloring for race-free parallel assembly
    WorkStream,     ///< Task pipeline with scratch/copy pattern
    Atomic          ///< Atomic operations for thread safety
};

/**
 * @brief Policy level for `assembler_name="Auto"` selection
 *
 * Conservative is the default and only policy that may be enabled implicitly.
 * More aggressive policies should remain explicitly opt-in.
 */
enum class AutoSelectionPolicy : std::uint8_t {
    Conservative,
    Performance,
    Experimental
};

/**
 * @brief Ghost contribution policy for parallel assembly
 */
enum class GhostPolicy : std::uint8_t {
    OwnedRowsOnly,   ///< Only assemble to locally owned rows
    ReverseScatter   ///< Assemble everywhere, then reverse-scatter ghosts to owners
};

/**
 * @brief Assembly options and parameters
 */
struct AssemblyOptions {
    // Threading
    ThreadingStrategy threading{ThreadingStrategy::Sequential};
    int num_threads{1};                  ///< Number of threads (0 = auto-detect)

    // Parallel (MPI)
    GhostPolicy ghost_policy{GhostPolicy::ReverseScatter};
    bool overlap_communication{true};    ///< Overlap computation and communication

    // Assembly mode
    AssemblyMode default_mode{AssemblyMode::Add};
    bool use_constraints{true};          ///< Apply constraints during assembly

    // Determinism
    bool deterministic{true};            ///< Require deterministic results
    bool stable_insertion_order{true};   ///< Stable ordering for floating-point

    // Performance hints
    bool use_batching{false};            ///< Batch element operations
    int batch_size{32};                  ///< Elements per batch
    bool cache_element_data{false};      ///< Cache geometry/basis evaluations

    // Scheduling / locality (decorator opt-in)
    bool schedule_elements{false};       ///< Reorder element traversal (cache/locality)
    int schedule_strategy{0};            ///< 0=Natural,1=Hilbert,2=Morton,3=RCM,4=Complexity,5=CacheBlocked

    // Coloring (decorator opt-in)
    bool use_coloring{false};            ///< Enable graph-colored parallel assembly
    ColoringOptions coloring{};          ///< Coloring algorithm/options (used when use_coloring=true)

    // Auto-selection policy (used only when assembler_name="Auto")
    AutoSelectionPolicy auto_policy{AutoSelectionPolicy::Conservative};

    // Debugging
    bool check_sparsity{false};          ///< Verify insertions match sparsity pattern
    bool check_finite_values{false};     ///< Throw if local output contains NaN/Inf
    bool verbose{false};                 ///< Print assembly progress
};

/**
 * @brief Result of an assembly operation
 */
struct AssemblyResult {
    bool success{true};
    std::string error_message;

    // Statistics
    GlobalIndex elements_assembled{0};
    GlobalIndex boundary_faces_assembled{0};
    GlobalIndex interior_faces_assembled{0};
    GlobalIndex interface_faces_assembled{0};
    double elapsed_time_seconds{0.0};

    // Performance metrics
    GlobalIndex matrix_entries_inserted{0};
    GlobalIndex vector_entries_inserted{0};

    operator bool() const noexcept { return success; }
};

// ============================================================================
// Mesh Topology Abstraction (for mesh-independent assembly)
// ============================================================================

/**
 * @brief Abstract interface for mesh iteration during assembly
 *
 * This interface allows the assembler to iterate over mesh cells and faces
 * without depending on a specific mesh library. Implementations can be
 * provided for MeshBase, Mesh, or external mesh libraries.
 */
class IMeshAccess {
public:
    virtual ~IMeshAccess() = default;

    /// Number of locally stored cells (owned + ghost)
    [[nodiscard]] virtual GlobalIndex numCells() const = 0;

    /// Number of locally owned cells
    [[nodiscard]] virtual GlobalIndex numOwnedCells() const = 0;

    /// Number of boundary faces
    [[nodiscard]] virtual GlobalIndex numBoundaryFaces() const = 0;

    /// Number of interior faces (for DG)
    [[nodiscard]] virtual GlobalIndex numInteriorFaces() const = 0;

    /// Spatial dimension (2 or 3)
    [[nodiscard]] virtual int dimension() const = 0;

    /// Check if cell is locally owned
    [[nodiscard]] virtual bool isOwnedCell(GlobalIndex cell_id) const = 0;

    /// Get element type for a cell
    [[nodiscard]] virtual ElementType getCellType(GlobalIndex cell_id) const = 0;

    /// Get domain/subdomain label for a cell (used for multi-domain/multiphysics setups)
    ///
    /// Default implementation returns 0 for all cells, allowing mesh adapters
    /// that do not support domain labels to remain valid.
    [[nodiscard]] virtual int getCellDomainId(GlobalIndex /*cell_id*/) const { return 0; }

    /// Get node indices for a cell
    /// @param cell_id Cell identifier
    /// @param nodes Output vector to fill with node indices
    virtual void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const = 0;

    /// Get coordinates of a single node
    /// @param node_id Node identifier
    /// @return 3D coordinates of the node
    [[nodiscard]] virtual std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const = 0;

    /// Get coordinates of all nodes of a cell
    /// @param cell_id Cell identifier
    /// @param coords Output vector to fill with node coordinates (size = num_nodes_per_cell)
    virtual void getCellCoordinates(GlobalIndex cell_id,
                                    std::vector<std::array<Real, 3>>& coords) const = 0;

    /// Get the local face index within a cell for a boundary face
    /// @param face_id Global face identifier
    /// @param cell_id Cell that contains this face
    /// @return Local face index within the cell (0-based)
    [[nodiscard]] virtual LocalIndex getLocalFaceIndex(GlobalIndex face_id,
                                                        GlobalIndex cell_id) const = 0;

    /// Get boundary face label/marker
    [[nodiscard]] virtual int getBoundaryFaceMarker(GlobalIndex face_id) const = 0;

    /// Get cells adjacent to an interior face (for DG)
    [[nodiscard]] virtual std::pair<GlobalIndex, GlobalIndex>
    getInteriorFaceCells(GlobalIndex face_id) const = 0;

    /// Iterate over cells, calling callback for each
    virtual void forEachCell(std::function<void(GlobalIndex)> callback) const = 0;

    /// Iterate over owned cells only
    virtual void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const = 0;

    /// Iterate over boundary faces with given marker
    virtual void forEachBoundaryFace(int marker,
                                     std::function<void(GlobalIndex, GlobalIndex)> callback) const = 0;

    /// Iterate over all interior faces
    virtual void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> callback) const = 0;
};

// ============================================================================
// Assembler Base Interface
// ============================================================================

/**
 * @brief Abstract base class for all assembly strategies
 *
 * The Assembler interface defines the contract for assembling global matrices
 * and vectors from element contributions. Concrete implementations provide
 * different assembly strategies (serial, colored, work-stream, matrix-free).
 *
 * Usage pattern:
 * @code
 *   auto assembler = std::make_unique<StandardAssembler>();
 *
 *   // Configure
 *   assembler->setDofMap(dof_map);
 *   assembler->setConstraints(constraints);
 *   assembler->setSparsityPattern(sparsity);
 *   assembler->setOptions(options);
 *
 *   // Assemble matrix
 *   assembler->assembleMatrix(mesh_access, test_space, trial_space, kernel, matrix_view);
 *
 *   // Or assemble both matrix and vector
 *   assembler->assembleBoth(mesh_access, test_space, trial_space, kernel,
 *                           matrix_view, vector_view);
 * @endcode
 *
 * Thread safety:
 * - Configuration methods (set*) are NOT thread-safe
 * - Assembly methods may be thread-safe depending on the strategy
 * - After setup, read-only accessors are thread-safe
 */
class Assembler {
public:
    virtual ~Assembler() = default;

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Set the DOF map for local-to-global mapping
     *
     * @param dof_map The DOF map (must be finalized)
     *
     * The DOF map provides the mapping from element-local DOF indices to
     * global system indices. It must remain valid during assembly.
     */
    virtual void setDofMap(const dofs::DofMap& dof_map) = 0;

    /**
     * @brief Set the row (test) DOF map for rectangular assembly
     *
     * Default implementation forwards to @ref setDofMap and requires zero offset.
     */
    virtual void setRowDofMap(const dofs::DofMap& dof_map, GlobalIndex row_offset = 0) {
        FE_THROW_IF(row_offset != 0, FEException,
                    "Assembler::setRowDofMap: offsets not supported by this assembler");
        setDofMap(dof_map);
    }

    /**
     * @brief Set the column (trial) DOF map for rectangular assembly
     *
     * Default implementation forwards to @ref setDofMap and requires zero offset.
     */
    virtual void setColDofMap(const dofs::DofMap& dof_map, GlobalIndex col_offset = 0) {
        FE_THROW_IF(col_offset != 0, FEException,
                    "Assembler::setColDofMap: offsets not supported by this assembler");
        setDofMap(dof_map);
    }

    /**
     * @brief Set the DOF handler (alternative to setDofMap)
     *
     * @param dof_handler The DOF handler (must be finalized)
     */
    virtual void setDofHandler(const dofs::DofHandler& dof_handler) = 0;

    /**
     * @brief Set constraints for constraint-aware assembly
     *
     * @param constraints The closed AffineConstraints object
     *
     * If set, constraints are applied during assembly via the
     * ConstraintDistributor pattern. Pass nullptr to disable.
     */
    virtual void setConstraints(const constraints::AffineConstraints* constraints) = 0;

    /**
     * @brief Set sparsity pattern for matrix assembly
     *
     * @param sparsity The finalized sparsity pattern
     *
     * Optional but recommended for backends that benefit from preallocation.
     */
    virtual void setSparsityPattern(const sparsity::SparsityPattern* sparsity) = 0;

    /**
     * @brief Set assembly options
     *
     * @param options The assembly options
     */
    virtual void setOptions(const AssemblyOptions& options) = 0;

    /**
     * @brief Set the current solution vector for kernels that request solution data
     *
     * Assemblers that support solution-aware kernels (RequiredData::SolutionValues, etc.)
     * should gather per-cell coefficients and populate AssemblyContext via
     * AssemblyContext::setSolutionCoefficients(...) before invoking the kernel.
     *
     * Default implementation is a no-op; kernels requesting solution data will
     * still fail unless the concrete assembler provides it.
     */
    virtual void setCurrentSolution(std::span<const Real> /*solution*/) {}

    /**
     * @brief Set the current solution vector via a global-indexed view
     *
     * This supports backends whose local storage is not directly indexable by
     * global DOF (e.g., distributed/ghosted vectors). The provided view must
     * implement getVectorEntry(dof) for any global DOF queried by the assembler.
     *
     * Default implementation is a no-op.
     *
     * @param solution_view Global-indexed vector view (nullable to clear)
     */
    virtual void setCurrentSolutionView(const GlobalSystemView* /*solution_view*/) {}

    /**
     * @brief Provide accessors for multi-field discrete solution data
     *
     * Default implementation is a no-op. Assemblers that support kernels using
     * additional discrete FE fields should override this and use the provided
     * registry to populate AssemblyContext field data.
     */
    virtual void setFieldSolutionAccess(std::span<const FieldSolutionAccess> /*fields*/) {}

    /**
     * @brief Set the previous-step solution vector (u^{n-1}) for transient assembly
     *
     * Default implementation is a no-op.
     */
    virtual void setPreviousSolution(std::span<const Real> /*solution*/) {}

    /**
     * @brief Set the previous-previous solution vector (u^{n-2}) for transient assembly
     *
     * Default implementation is a no-op.
     */
    virtual void setPreviousSolution2(std::span<const Real> /*solution*/) {}

    /**
     * @brief Set the previous-step solution vector via a global-indexed view
     *
     * This supports backends whose local storage is not directly indexable by
     * global DOF (e.g., distributed/ghosted vectors or permuted layouts).
     *
     * Default implementation is a no-op.
     *
     * @param solution_view Global-indexed vector view (nullable to clear)
     */
    virtual void setPreviousSolutionView(const GlobalSystemView* /*solution_view*/) {}

    /**
     * @brief Set the previous-previous solution vector via a global-indexed view
     *
     * Default implementation is a no-op.
     *
     * @param solution_view Global-indexed vector view (nullable to clear)
     */
    virtual void setPreviousSolution2View(const GlobalSystemView* /*solution_view*/) {}

    /**
     * @brief Set the k-th previous solution vector (u^{n-k}) for transient assembly
     *
     * Default implementation forwards k=1,2 to setPreviousSolution/setPreviousSolution2.
     * Higher k values are ignored unless overridden by a concrete assembler.
     */
    virtual void setPreviousSolutionK(int k, std::span<const Real> solution)
    {
        if (k == 1) {
            setPreviousSolution(solution);
        } else if (k == 2) {
            setPreviousSolution2(solution);
        }
    }

    /**
     * @brief Set the k-th previous solution vector via a global-indexed view
     *
     * Default implementation forwards k=1,2 to setPreviousSolutionView/setPreviousSolution2View.
     * Higher k values are ignored unless overridden by a concrete assembler.
     */
    virtual void setPreviousSolutionViewK(int k, const GlobalSystemView* solution_view)
    {
        if (k == 1) {
            setPreviousSolutionView(solution_view);
        } else if (k == 2) {
            setPreviousSolution2View(solution_view);
        }
    }

    /**
     * @brief Attach a time-integration context for symbolic `dt(·,k)` lowering
     *
     * When null, kernels containing `dt(...)` must fail with a clear diagnostic.
     *
     * Default implementation is a no-op.
     */
    virtual void setTimeIntegrationContext(const TimeIntegrationContext* /*ctx*/) {}

    /**
     * @brief Set the current physical time (optional)
     *
     * Default implementation is a no-op.
     */
    virtual void setTime(Real /*time*/) {}

    /**
     * @brief Set the current time step size dt (optional)
     *
     * Default implementation is a no-op.
     */
    virtual void setTimeStep(Real /*dt*/) {}

    /**
     * @brief Provide a parameter getter (optional)
     *
     * This is typically backed by systems::SystemStateView::getRealParam and is
     * forwarded into AssemblyContext for downstream evaluation (e.g., Forms/Constitutive).
     *
     * Default implementation is a no-op.
     */
    virtual void setRealParameterGetter(
        const std::function<std::optional<Real>(std::string_view)>* /*get_real_param*/) noexcept
    {
    }

    /**
     * @brief Provide a typed parameter getter (optional)
     *
     * This is typically backed by systems::SystemStateView::getParam and is
     * forwarded into AssemblyContext for downstream evaluation (e.g., Forms/Constitutive).
     *
     * Default implementation is a no-op.
     */
    virtual void setParameterGetter(
        const std::function<std::optional<params::Value>(std::string_view)>* /*get_param*/) noexcept
    {
    }

    /**
     * @brief Provide an opaque user context pointer (optional)
     *
     * This is forwarded into AssemblyContext and can be used to supply
     * application-owned services to constitutive evaluations.
     *
     * Default implementation is a no-op.
     */
    virtual void setUserData(const void* /*user_data*/) noexcept {}

    /**
     * @brief Bind a flat array of Real-valued parameter slots for JIT-friendly kernels.
     *
     * Default implementation is a no-op.
     */
    virtual void setJITConstants(std::span<const Real> /*constants*/) noexcept {}

    /**
     * @brief Bind coupled boundary-condition scalar arrays (integrals + auxiliary state).
     *
     * Default implementation is a no-op.
     */
    virtual void setCoupledValues(std::span<const Real> /*integrals*/,
                                  std::span<const Real> /*aux_state*/) noexcept
    {
    }

    /**
     * @brief Provide per-cell state storage for kernels requesting RequiredData::MaterialState
     *
     * Default implementation is a no-op. Assemblers that support state (e.g.
     * StandardAssembler) should override and bind the provided storage into
     * AssemblyContext during element preparation.
     */
    virtual void setMaterialStateProvider(IMaterialStateProvider* /*provider*/) noexcept {}

    /**
     * @brief Get current options
     */
    [[nodiscard]] virtual const AssemblyOptions& getOptions() const noexcept = 0;

    // =========================================================================
    // Matrix Assembly
    // =========================================================================

    /**
     * @brief Assemble a matrix (bilinear form)
     *
     * Iterates over mesh cells, computes element matrices using the kernel,
     * and inserts contributions into the global matrix.
     *
     * @param mesh Mesh access interface for iteration
     * @param test_space Test function space (rows)
     * @param trial_space Trial function space (columns)
     * @param kernel Element kernel for computing local matrices
     * @param matrix_view Global matrix insertion interface
     * @return Assembly result with statistics and status
     *
     * Supports rectangular assembly: test_space != trial_space.
     */
    [[nodiscard]] virtual AssemblyResult assembleMatrix(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view) = 0;

    /**
     * @brief Assemble a matrix (square bilinear form)
     *
     * Convenience overload for square matrices where test_space == trial_space.
     */
    [[nodiscard]] virtual AssemblyResult assembleMatrix(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view) {
        return assembleMatrix(mesh, space, space, kernel, matrix_view);
    }

    // =========================================================================
    // Vector Assembly
    // =========================================================================

    /**
     * @brief Assemble a vector (linear form)
     *
     * Iterates over mesh cells, computes element vectors using the kernel,
     * and inserts contributions into the global vector.
     *
     * @param mesh Mesh access interface for iteration
     * @param space Function space
     * @param kernel Element kernel for computing local vectors
     * @param vector_view Global vector insertion interface
     * @return Assembly result with statistics and status
     */
    [[nodiscard]] virtual AssemblyResult assembleVector(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& space,
        AssemblyKernel& kernel,
        GlobalSystemView& vector_view) = 0;

    // =========================================================================
    // Combined Assembly
    // =========================================================================

    /**
     * @brief Assemble both matrix and vector (coupled)
     *
     * More efficient than separate calls when both are needed, as element
     * data can be reused between matrix and vector computation.
     *
     * @param mesh Mesh access interface
     * @param test_space Test function space
     * @param trial_space Trial function space
     * @param kernel Element kernel
     * @param matrix_view Global matrix insertion interface
     * @param vector_view Global vector insertion interface
     * @return Assembly result
     */
    [[nodiscard]] virtual AssemblyResult assembleBoth(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view,
        GlobalSystemView& vector_view) = 0;

    /**
     * @brief Assemble both matrix and vector (square)
     *
     * Convenience overload for square systems.
     */
    [[nodiscard]] virtual AssemblyResult assembleBoth(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view,
        GlobalSystemView& vector_view) {
        return assembleBoth(mesh, space, space, kernel, matrix_view, vector_view);
    }

    // =========================================================================
    // Face Assembly (for DG and boundary conditions)
    // =========================================================================

    /**
     * @brief Assemble boundary face contributions
     *
     * @param mesh Mesh access interface
     * @param boundary_marker Boundary label/marker to iterate
     * @param space Function space
     * @param kernel Element kernel with boundary face method
     * @param matrix_view Matrix insertion (can be null for vector-only)
     * @param vector_view Vector insertion (can be null for matrix-only)
     * @return Assembly result
     */
    [[nodiscard]] virtual AssemblyResult assembleBoundaryFaces(
        const IMeshAccess& mesh,
        int boundary_marker,
        const spaces::FunctionSpace& space,
        AssemblyKernel& kernel,
        GlobalSystemView* matrix_view,
        GlobalSystemView* vector_view) = 0;

    /**
     * @brief Assemble boundary face contributions (rectangular)
     *
     * Supports bilinear forms where test_space != trial_space on the boundary.
     * Default implementation supports only the square case.
     */
    [[nodiscard]] virtual AssemblyResult assembleBoundaryFaces(
        const IMeshAccess& mesh,
        int boundary_marker,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView* matrix_view,
        GlobalSystemView* vector_view) {
        FE_THROW_IF(&test_space != &trial_space, FEException,
                    "Assembler::assembleBoundaryFaces: rectangular boundary assembly not implemented");
        return assembleBoundaryFaces(mesh, boundary_marker, test_space, kernel, matrix_view, vector_view);
    }

    /**
     * @brief Assemble interior face contributions (for DG)
     *
     * @param mesh Mesh access interface
     * @param test_space Test function space
     * @param trial_space Trial function space
     * @param kernel Element kernel with interior face method
     * @param matrix_view Matrix insertion
     * @param vector_view Vector insertion (can be null)
     * @return Assembly result
     */
    [[nodiscard]] virtual AssemblyResult assembleInteriorFaces(
        const IMeshAccess& mesh,
        const spaces::FunctionSpace& test_space,
        const spaces::FunctionSpace& trial_space,
        AssemblyKernel& kernel,
        GlobalSystemView& matrix_view,
        GlobalSystemView* vector_view) = 0;

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    /**
     * @brief Assemble interface-face contributions (InterfaceMesh subset)
     *
     * Intended for interface-coupled formulations assembled on a user-provided
     * InterfaceMesh (e.g., multi-material coupling surfaces, mortar/Nitsche).
     *
     * Default implementation throws (unsupported).
     */
    [[nodiscard]] virtual AssemblyResult assembleInterfaceFaces(
        const IMeshAccess& /*mesh*/,
        const svmp::InterfaceMesh& /*interface_mesh*/,
        int /*interface_marker*/,
        const spaces::FunctionSpace& /*test_space*/,
        const spaces::FunctionSpace& /*trial_space*/,
        AssemblyKernel& /*kernel*/,
        GlobalSystemView& /*matrix_view*/,
        GlobalSystemView* /*vector_view*/)
    {
        FE_THROW(FEException, "Assembler::assembleInterfaceFaces: not implemented");
    }
#endif

    // =========================================================================
    // Lifecycle
    // =========================================================================

    /**
     * @brief Initialize the assembler before assembly loops
     *
     * Called once before beginning assembly. Can be used to set up
     * thread-local storage, communication patterns, etc.
     */
    virtual void initialize() = 0;

    /**
     * @brief Finalize after assembly is complete
     *
     * Called after all assembly operations. Triggers any necessary
     * communication (e.g., ghost accumulation) and cleanup.
     *
     * @param matrix_view Matrix to finalize (can be null)
     * @param vector_view Vector to finalize (can be null)
     */
    virtual void finalize(GlobalSystemView* matrix_view,
                          GlobalSystemView* vector_view) = 0;

    /**
     * @brief Reset the assembler for reuse
     *
     * Clears internal state while keeping configuration.
     */
    virtual void reset() = 0;

    // =========================================================================
    // Query
    // =========================================================================

    /**
     * @brief Get assembler strategy name (for debugging/logging)
     */
    [[nodiscard]] virtual std::string name() const = 0;

    /**
     * @brief Check if assembler is properly configured
     */
    [[nodiscard]] virtual bool isConfigured() const noexcept = 0;

    /**
     * @brief Check if assembler supports rectangular assembly
     */
    [[nodiscard]] virtual bool supportsRectangular() const noexcept { return true; }

    /**
     * @brief Check if assembler supports DG (interior face) assembly
     */
    [[nodiscard]] virtual bool supportsDG() const noexcept { return false; }

    /**
     * @brief Check if assembler populates full FE context data (quad/geometry/basis)
     *
     * Assemblers that only call `AssemblyContext::configure(...)` without filling
     * quadrature/basis/geometry buffers should return false.
     */
    [[nodiscard]] virtual bool supportsFullContext() const noexcept { return false; }

    /**
     * @brief Check if assembler supports solution-dependent kernels
     *
     * This implies honoring `setCurrentSolution(...)` and binding solution
     * coefficients/values/derivatives into `AssemblyContext` as requested by
     * `AssemblyKernel::getRequiredData()`.
     */
    [[nodiscard]] virtual bool supportsSolution() const noexcept { return false; }

    /**
     * @brief Check if assembler supports transient (history) data
     *
     * This implies honoring `setPreviousSolutionK(...)`.
     */
    [[nodiscard]] virtual bool supportsSolutionHistory() const noexcept { return false; }

    /**
     * @brief Check if assembler supports attaching `TimeIntegrationContext`
     */
    [[nodiscard]] virtual bool supportsTimeIntegrationContext() const noexcept { return false; }

    /**
     * @brief Check if assembler supports multi-field offsets via setRowDofMap/setColDofMap
     */
    [[nodiscard]] virtual bool supportsDofOffsets() const noexcept { return false; }

    /**
     * @brief Check if assembler supports kernels requiring additional discrete fields
     */
    [[nodiscard]] virtual bool supportsFieldRequirements() const noexcept { return false; }

    /**
     * @brief Check if assembler supports kernels requiring per-integration-point state
     */
    [[nodiscard]] virtual bool supportsMaterialState() const noexcept { return false; }

    /**
     * @brief Check if assembler is thread-safe
     */
    [[nodiscard]] virtual bool isThreadSafe() const noexcept { return false; }
};

// ============================================================================
// Assembler Factory
// ============================================================================

// Forward declarations for selection-aware factory overload.
struct FormCharacteristics;
struct SystemCharacteristics;

/**
 * @brief Create an assembler with the specified strategy
 *
 * @param strategy Threading strategy to use
 * @return Unique pointer to the created assembler
 */
std::unique_ptr<Assembler> createAssembler(ThreadingStrategy strategy);

/**
 * @brief Create an assembler with full options
 *
 * @param options Assembly options
 * @return Unique pointer to the created assembler
 */
std::unique_ptr<Assembler> createAssembler(const AssemblyOptions& options);

/**
 * @brief Create an assembler using a name + form/system characteristics (opt-in)
 *
 * This overload enables rule-based selection (e.g., `assembler_name="Auto"`)
 * and decorator composition (caching/scheduling/vectorization).
 *
 * DG incompatibility is a hard error (no fallback).
 */
std::unique_ptr<Assembler> createAssembler(const AssemblyOptions& options,
                                           std::string_view assembler_name,
                                           const FormCharacteristics& form,
                                           const SystemCharacteristics& system);

/**
 * @brief Create an assembler and optionally emit a human-readable selection report
 */
std::unique_ptr<Assembler> createAssembler(const AssemblyOptions& options,
                                           std::string_view assembler_name,
                                           const FormCharacteristics& form,
                                           const SystemCharacteristics& system,
                                           std::string* selection_report);

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_ASSEMBLER_H
