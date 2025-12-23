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

#ifndef SVMP_FE_ASSEMBLY_ASSEMBLY_SCHEDULER_H
#define SVMP_FE_ASSEMBLY_ASSEMBLY_SCHEDULER_H

/**
 * @file AssemblyScheduler.h
 * @brief Optimize assembly order and scheduling for cache efficiency
 *
 * AssemblyScheduler provides strategies for ordering element assembly to
 * maximize cache utilization and improve parallel load balance. Key features:
 *
 * 1. ELEMENT ORDERING STRATEGIES:
 *    - Natural ordering (mesh traversal order)
 *    - Space-filling curve ordering (Hilbert/Morton) for spatial locality
 *    - Graph-based reordering (RCM/Cuthill-McKee) for bandwidth reduction
 *    - Complexity-based ordering (sort by element cost)
 *
 * 2. CACHE-AWARE SCHEDULING:
 *    - Block elements that share DOFs together
 *    - Minimize working set size during assembly
 *    - Prefetch hints for next element batch
 *
 * 3. NUMA-AWARE DISTRIBUTION:
 *    - Partition elements across NUMA nodes
 *    - Pin threads to NUMA domains
 *    - Minimize cross-NUMA memory traffic
 *
 * 4. DYNAMIC LOAD BALANCING:
 *    - Estimate element complexity (quadrature points, DOFs)
 *    - Balance work across threads
 *    - Runtime adjustment based on execution history
 *
 * 5. INTEGRATION WITH COLOREDASSEMBLER:
 *    - Optimize color ordering for better cache behavior
 *    - Balance color sizes for parallel efficiency
 *
 * Module boundaries:
 * - This module OWNS: element ordering, scheduling strategies, load balancing
 * - This module does NOT OWN: actual assembly loops, kernel computation
 *
 * @see ColoredAssembler for integration with colored parallel assembly
 * @see AssemblyLoop for the main loop infrastructure
 */

#include "Core/Types.h"
#include "Assembler.h"

#include <vector>
#include <span>
#include <memory>
#include <functional>
#include <optional>

namespace svmp {
namespace FE {

// Forward declarations
namespace dofs {
    class DofMap;
}

namespace assembly {

// ============================================================================
// Scheduling Strategies
// ============================================================================

/**
 * @brief Element ordering strategy
 */
enum class OrderingStrategy : std::uint8_t {
    Natural,           ///< Original mesh ordering
    Hilbert,           ///< Hilbert space-filling curve
    Morton,            ///< Morton (Z-order) curve
    RCM,               ///< Reverse Cuthill-McKee
    SloanProfile,      ///< Sloan profile/wavefront reduction
    ComplexityBased,   ///< Order by element complexity
    CacheBlocked,      ///< Group elements sharing DOFs
    Custom             ///< User-provided ordering
};

/**
 * @brief NUMA distribution strategy
 */
enum class NUMAStrategy : std::uint8_t {
    None,              ///< No NUMA awareness
    Interleaved,       ///< Interleave across NUMA nodes
    FirstTouch,        ///< First-touch policy (thread affinity)
    Partitioned,       ///< Explicit element partitioning to nodes
    Automatic          ///< Auto-detect and optimize
};

/**
 * @brief Load balancing mode
 */
enum class LoadBalanceMode : std::uint8_t {
    Static,            ///< Fixed distribution (no rebalancing)
    Dynamic,           ///< Runtime work stealing
    Guided,            ///< Decreasing chunk sizes
    Adaptive           ///< Learn from execution history
};

// ============================================================================
// Scheduler Options
// ============================================================================

/**
 * @brief Configuration options for assembly scheduling
 */
struct SchedulerOptions {
    /**
     * @brief Element ordering strategy
     */
    OrderingStrategy ordering{OrderingStrategy::Natural};

    /**
     * @brief NUMA distribution strategy
     */
    NUMAStrategy numa{NUMAStrategy::None};

    /**
     * @brief Load balancing mode
     */
    LoadBalanceMode load_balance{LoadBalanceMode::Static};

    /**
     * @brief Number of threads (0 = auto-detect)
     */
    int num_threads{0};

    /**
     * @brief Block size for cache-blocked ordering
     */
    std::size_t cache_block_size{64};

    /**
     * @brief Enable prefetching hints
     */
    bool enable_prefetch{true};

    /**
     * @brief Prefetch distance (elements ahead)
     */
    int prefetch_distance{2};

    /**
     * @brief Recompute ordering when mesh changes
     */
    bool auto_reorder{false};

    /**
     * @brief Verbose output for debugging
     */
    bool verbose{false};
};

// ============================================================================
// Element Complexity Estimation
// ============================================================================

/**
 * @brief Estimated complexity of an element
 */
struct ElementComplexity {
    GlobalIndex cell_id{-1};           ///< Cell identifier
    LocalIndex num_dofs{0};            ///< Number of DOFs
    LocalIndex num_qpts{0};            ///< Number of quadrature points
    int polynomial_order{0};           ///< Polynomial order
    double estimated_flops{0.0};       ///< Estimated floating-point operations
    double estimated_memory{0.0};      ///< Estimated memory access (bytes)

    /**
     * @brief Compute arithmetic intensity (FLOPS/byte)
     */
    [[nodiscard]] double arithmeticIntensity() const noexcept {
        return estimated_memory > 0.0 ? estimated_flops / estimated_memory : 0.0;
    }

    /**
     * @brief Combined cost estimate for load balancing
     */
    [[nodiscard]] double cost() const noexcept {
        // Simple model: cost proportional to DOFs^2 * quadrature points
        return static_cast<double>(num_dofs) * num_dofs * num_qpts;
    }
};

/**
 * @brief Element complexity estimator
 *
 * Estimates computational cost of assembling each element for load balancing.
 */
class ComplexityEstimator {
public:
    ComplexityEstimator() = default;
    ~ComplexityEstimator() = default;

    ComplexityEstimator(const ComplexityEstimator&) = default;
    ComplexityEstimator& operator=(const ComplexityEstimator&) = default;
    ComplexityEstimator(ComplexityEstimator&&) = default;
    ComplexityEstimator& operator=(ComplexityEstimator&&) = default;

    /**
     * @brief Estimate complexity for a single element
     *
     * @param cell_type Element type
     * @param num_dofs Number of DOFs on element
     * @param num_qpts Number of quadrature points
     * @param polynomial_order Polynomial order
     * @return Complexity estimate
     */
    [[nodiscard]] ElementComplexity estimate(
        ElementType cell_type,
        LocalIndex num_dofs,
        LocalIndex num_qpts,
        int polynomial_order) const;

    /**
     * @brief Estimate complexity for all elements in mesh
     *
     * @param mesh Mesh access interface
     * @param dof_map DOF map for DOF counts
     * @return Vector of complexity estimates
     */
    [[nodiscard]] std::vector<ElementComplexity> estimateAll(
        const IMeshAccess& mesh,
        const dofs::DofMap& dof_map) const;

    /**
     * @brief Update estimates based on actual timing
     *
     * @param cell_id Cell identifier
     * @param actual_time Measured assembly time
     */
    void updateFromTiming(GlobalIndex cell_id, double actual_time);

    /**
     * @brief Get total estimated cost
     */
    [[nodiscard]] double totalCost(const std::vector<ElementComplexity>& complexities) const;

    /**
     * @brief Get load imbalance factor
     *
     * Ratio of max thread cost to average thread cost.
     * Value of 1.0 indicates perfect balance.
     */
    [[nodiscard]] double loadImbalance(
        const std::vector<ElementComplexity>& complexities,
        int num_threads) const;
};

// ============================================================================
// Scheduling Result
// ============================================================================

/**
 * @brief Result of scheduling computation
 */
struct SchedulingResult {
    /**
     * @brief Reordered element indices
     *
     * If ordering is non-null, process elements in this order:
     *   for (i = 0; i < n; i++) process(ordering[i])
     */
    std::vector<GlobalIndex> ordering;

    /**
     * @brief Thread assignments for each element
     *
     * thread_assignment[ordering[i]] = thread that should process element
     */
    std::vector<int> thread_assignment;

    /**
     * @brief Per-thread element ranges (for static scheduling)
     *
     * Thread t processes elements [thread_ranges[t], thread_ranges[t+1])
     */
    std::vector<std::size_t> thread_ranges;

    /**
     * @brief Per-thread work (for load balance info)
     */
    std::vector<double> thread_work;

    /**
     * @brief Scheduling computation time
     */
    double scheduling_seconds{0.0};

    /**
     * @brief Estimated load imbalance factor
     */
    double estimated_imbalance{1.0};
};

// ============================================================================
// Space-Filling Curve Utilities
// ============================================================================

/**
 * @brief Space-filling curve coordinate encoding
 */
class SpaceFillingCurve {
public:
    /**
     * @brief Compute Hilbert index for 2D point
     *
     * @param x X coordinate (0 to n-1)
     * @param y Y coordinate (0 to n-1)
     * @param n Grid dimension (must be power of 2)
     * @return Hilbert curve index
     */
    [[nodiscard]] static std::uint64_t hilbert2D(
        std::uint32_t x, std::uint32_t y, std::uint32_t n);

    /**
     * @brief Compute Hilbert index for 3D point
     *
     * @param x X coordinate
     * @param y Y coordinate
     * @param z Z coordinate
     * @param n Grid dimension (must be power of 2)
     * @return Hilbert curve index
     */
    [[nodiscard]] static std::uint64_t hilbert3D(
        std::uint32_t x, std::uint32_t y, std::uint32_t z, std::uint32_t n);

    /**
     * @brief Compute Morton (Z-order) index for 2D point
     *
     * @param x X coordinate
     * @param y Y coordinate
     * @return Morton curve index
     */
    [[nodiscard]] static std::uint64_t morton2D(std::uint32_t x, std::uint32_t y);

    /**
     * @brief Compute Morton (Z-order) index for 3D point
     *
     * @param x X coordinate
     * @param y Y coordinate
     * @param z Z coordinate
     * @return Morton curve index
     */
    [[nodiscard]] static std::uint64_t morton3D(
        std::uint32_t x, std::uint32_t y, std::uint32_t z);

    /**
     * @brief Discretize coordinate to grid cell
     *
     * @param coord Physical coordinate
     * @param min_coord Bounding box minimum
     * @param max_coord Bounding box maximum
     * @param num_cells Number of grid cells
     * @return Grid cell index
     */
    [[nodiscard]] static std::uint32_t discretize(
        Real coord, Real min_coord, Real max_coord, std::uint32_t num_cells);
};

// ============================================================================
// NUMA Utilities
// ============================================================================

/**
 * @brief NUMA topology and utilities
 */
class NUMATopology {
public:
    /**
     * @brief Default constructor
     */
    NUMATopology();

    /**
     * @brief Destructor
     */
    ~NUMATopology();

    /**
     * @brief Move constructor
     */
    NUMATopology(NUMATopology&& other) noexcept;

    /**
     * @brief Move assignment
     */
    NUMATopology& operator=(NUMATopology&& other) noexcept;

    // Non-copyable
    NUMATopology(const NUMATopology&) = delete;
    NUMATopology& operator=(const NUMATopology&) = delete;

    /**
     * @brief Query NUMA availability
     */
    [[nodiscard]] bool isNUMAAvailable() const noexcept;

    /**
     * @brief Get number of NUMA nodes
     */
    [[nodiscard]] int numNodes() const noexcept;

    /**
     * @brief Get number of CPUs per NUMA node
     */
    [[nodiscard]] int cpusPerNode() const noexcept;

    /**
     * @brief Get NUMA node for current thread
     */
    [[nodiscard]] int currentNode() const;

    /**
     * @brief Get NUMA node for a given CPU
     */
    [[nodiscard]] int nodeForCPU(int cpu) const;

    /**
     * @brief Pin current thread to NUMA node
     *
     * @param node NUMA node index
     * @return true if successful
     */
    bool pinToNode(int node);

    /**
     * @brief Pin current thread to specific CPU
     *
     * @param cpu CPU index
     * @return true if successful
     */
    bool pinToCPU(int cpu);

    /**
     * @brief Get CPUs in a NUMA node
     */
    [[nodiscard]] std::vector<int> getCPUsInNode(int node) const;

    /**
     * @brief Compute optimal thread-to-node mapping
     *
     * @param num_threads Number of threads
     * @return Vector of NUMA node assignments for each thread
     */
    [[nodiscard]] std::vector<int> computeThreadMapping(int num_threads) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================================================
// Assembly Scheduler
// ============================================================================

/**
 * @brief Main scheduler class for optimizing assembly order
 *
 * AssemblyScheduler analyzes the mesh and DOF structure to produce an
 * optimized element ordering for assembly. It can be used with any
 * assembler that supports custom iteration order.
 *
 * Usage:
 * @code
 *   AssemblyScheduler scheduler;
 *   scheduler.setMesh(mesh);
 *   scheduler.setDofMap(dof_map);
 *   scheduler.setOptions(options);
 *
 *   auto result = scheduler.computeSchedule();
 *
 *   // Use with custom loop
 *   for (GlobalIndex i : result.ordering) {
 *       assembleElement(i);
 *   }
 *
 *   // Or integrate with ColoredAssembler
 *   colored_assembler.setElementOrder(result.ordering);
 * @endcode
 */
class AssemblyScheduler {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor
     */
    AssemblyScheduler();

    /**
     * @brief Construct with options
     */
    explicit AssemblyScheduler(const SchedulerOptions& options);

    /**
     * @brief Destructor
     */
    ~AssemblyScheduler();

    /**
     * @brief Move constructor
     */
    AssemblyScheduler(AssemblyScheduler&& other) noexcept;

    /**
     * @brief Move assignment
     */
    AssemblyScheduler& operator=(AssemblyScheduler&& other) noexcept;

    // Non-copyable
    AssemblyScheduler(const AssemblyScheduler&) = delete;
    AssemblyScheduler& operator=(const AssemblyScheduler&) = delete;

    // =========================================================================
    // Configuration
    // =========================================================================

    /**
     * @brief Set mesh access interface
     */
    void setMesh(const IMeshAccess& mesh);

    /**
     * @brief Set DOF map
     */
    void setDofMap(const dofs::DofMap& dof_map);

    /**
     * @brief Set scheduler options
     */
    void setOptions(const SchedulerOptions& options);

    /**
     * @brief Get current options
     */
    [[nodiscard]] const SchedulerOptions& getOptions() const noexcept;

    /**
     * @brief Set element centroids for spatial ordering
     *
     * @param centroids Vector of (x, y, z) centroids for each cell
     */
    void setCentroids(std::span<const std::array<Real, 3>> centroids);

    /**
     * @brief Set element complexities for load balancing
     */
    void setComplexities(std::span<const ElementComplexity> complexities);

    /**
     * @brief Set custom element ordering callback
     *
     * @param comparator Comparison function for sorting elements
     */
    void setCustomComparator(
        std::function<bool(GlobalIndex, GlobalIndex)> comparator);

    // =========================================================================
    // Schedule Computation
    // =========================================================================

    /**
     * @brief Compute optimized assembly schedule
     *
     * @return Scheduling result with element ordering and thread assignments
     */
    [[nodiscard]] SchedulingResult computeSchedule();

    /**
     * @brief Compute schedule for specific element subset
     *
     * @param elements Elements to schedule
     * @return Scheduling result
     */
    [[nodiscard]] SchedulingResult computeSchedule(
        std::span<const GlobalIndex> elements);

    /**
     * @brief Compute natural (identity) ordering
     */
    [[nodiscard]] SchedulingResult computeNaturalSchedule();

    /**
     * @brief Compute Hilbert curve ordering
     */
    [[nodiscard]] SchedulingResult computeHilbertSchedule();

    /**
     * @brief Compute Morton curve ordering
     */
    [[nodiscard]] SchedulingResult computeMortonSchedule();

    /**
     * @brief Compute RCM ordering
     */
    [[nodiscard]] SchedulingResult computeRCMSchedule();

    /**
     * @brief Compute complexity-based ordering
     */
    [[nodiscard]] SchedulingResult computeComplexitySchedule();

    /**
     * @brief Compute cache-blocked ordering
     */
    [[nodiscard]] SchedulingResult computeCacheBlockedSchedule();

    // =========================================================================
    // Color Optimization (for ColoredAssembler)
    // =========================================================================

    /**
     * @brief Optimize element order within each color
     *
     * Reorders elements within each color bucket for better cache behavior
     * while preserving the coloring.
     *
     * @param colors Element colors
     * @param num_colors Number of colors
     * @return Optimized ordering (respecting color boundaries)
     */
    [[nodiscard]] std::vector<GlobalIndex> optimizeColorOrder(
        std::span<const int> colors, int num_colors);

    /**
     * @brief Optimize color processing order
     *
     * Determines optimal order to process colors for cache efficiency.
     *
     * @param colors Element colors
     * @param num_colors Number of colors
     * @return Optimal color processing order
     */
    [[nodiscard]] std::vector<int> optimizeColorSequence(
        std::span<const int> colors, int num_colors);

    // =========================================================================
    // Load Balancing
    // =========================================================================

    /**
     * @brief Compute thread assignments for static load balancing
     *
     * @param ordering Element ordering
     * @param num_threads Number of threads
     * @return Thread assignment for each element
     */
    [[nodiscard]] std::vector<int> computeStaticAssignment(
        std::span<const GlobalIndex> ordering, int num_threads);

    /**
     * @brief Compute thread ranges for static scheduling
     *
     * @param ordering Element ordering
     * @param num_threads Number of threads
     * @return Per-thread element ranges
     */
    [[nodiscard]] std::vector<std::size_t> computeThreadRanges(
        std::span<const GlobalIndex> ordering, int num_threads);

    /**
     * @brief Get work stealing hints for dynamic load balancing
     *
     * @param ordering Element ordering
     * @param num_threads Number of threads
     * @param chunk_size Base chunk size
     * @return Chunk boundaries for work stealing
     */
    [[nodiscard]] std::vector<std::size_t> getWorkStealingChunks(
        std::span<const GlobalIndex> ordering,
        int num_threads,
        std::size_t chunk_size);

    // =========================================================================
    // Query
    // =========================================================================

    /**
     * @brief Check if scheduler is configured
     */
    [[nodiscard]] bool isConfigured() const noexcept;

    /**
     * @brief Get NUMA topology
     */
    [[nodiscard]] const NUMATopology& getNUMATopology() const;

    /**
     * @brief Get complexity estimator
     */
    [[nodiscard]] ComplexityEstimator& getComplexityEstimator();

    /**
     * @brief Get last scheduling result
     */
    [[nodiscard]] const SchedulingResult& getLastResult() const noexcept;

private:
    // =========================================================================
    // Internal Implementation
    // =========================================================================

    /**
     * @brief Compute element centroids from mesh
     */
    void computeCentroids();

    /**
     * @brief Compute element complexities from mesh and dof map
     */
    void computeComplexities();

    /**
     * @brief Build element connectivity graph for cache-blocked ordering
     */
    void buildConnectivityGraph();

    /**
     * @brief Apply Hilbert curve ordering
     */
    void applyHilbertOrdering(std::vector<GlobalIndex>& ordering);

    /**
     * @brief Apply Morton curve ordering
     */
    void applyMortonOrdering(std::vector<GlobalIndex>& ordering);

    /**
     * @brief Apply RCM ordering
     */
    void applyRCMOrdering(std::vector<GlobalIndex>& ordering);

    /**
     * @brief Apply cache-blocked ordering
     */
    void applyCacheBlockedOrdering(std::vector<GlobalIndex>& ordering);

    /**
     * @brief Distribute elements to threads
     */
    void distributeToThreads(SchedulingResult& result);

    // =========================================================================
    // Data Members
    // =========================================================================

    SchedulerOptions options_;

    const IMeshAccess* mesh_{nullptr};
    const dofs::DofMap* dof_map_{nullptr};

    // Element data
    std::vector<std::array<Real, 3>> centroids_;
    std::vector<ElementComplexity> complexities_;
    bool has_centroids_{false};
    bool has_complexities_{false};

    // Custom comparator
    std::function<bool(GlobalIndex, GlobalIndex)> custom_comparator_;

    // Connectivity graph (for cache-blocked ordering)
    std::vector<std::vector<GlobalIndex>> element_neighbors_;

    // NUMA topology
    NUMATopology numa_topology_;

    // Complexity estimator
    ComplexityEstimator complexity_estimator_;

    // Last result
    SchedulingResult last_result_;
};

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * @brief Create scheduler with default options
 */
std::unique_ptr<AssemblyScheduler> createAssemblyScheduler();

/**
 * @brief Create scheduler with specified options
 */
std::unique_ptr<AssemblyScheduler> createAssemblyScheduler(
    const SchedulerOptions& options);

/**
 * @brief Create scheduler with specified ordering strategy
 */
std::unique_ptr<AssemblyScheduler> createAssemblyScheduler(
    OrderingStrategy ordering);

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_ASSEMBLY_SCHEDULER_H
