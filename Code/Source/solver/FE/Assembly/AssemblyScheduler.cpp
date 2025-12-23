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

#include "AssemblyScheduler.h"
#include "Dofs/DofMap.h"

#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

// Platform-specific NUMA headers
#if defined(__linux__)
#include <sched.h>
#include <unistd.h>
#endif

namespace svmp {
namespace FE {
namespace assembly {

// ============================================================================
// ComplexityEstimator Implementation
// ============================================================================

ElementComplexity ComplexityEstimator::estimate(
    ElementType cell_type,
    LocalIndex num_dofs,
    LocalIndex num_qpts,
    int polynomial_order) const
{
    ElementComplexity result;
    result.num_dofs = num_dofs;
    result.num_qpts = num_qpts;
    result.polynomial_order = polynomial_order;

    // Estimate FLOPs for stiffness matrix assembly:
    // - For each DOF pair (i,j), for each quadrature point:
    //   - Gradient computation: ~dim operations
    //   - Dot product: ~dim operations
    //   - Weight multiply: 1 operation
    // Total: O(num_dofs^2 * num_qpts * dim)

    int dim = element_dimension(cell_type);
    if (dim <= 0) dim = 3;  // Default to 3D

    double flops_per_pair_per_qpt = 2.0 * dim + 1.0;
    result.estimated_flops = static_cast<double>(num_dofs) * num_dofs *
                             num_qpts * flops_per_pair_per_qpt;

    // Estimate memory access:
    // - Read: basis gradients (num_dofs * num_qpts * dim * sizeof(Real))
    // - Read: quadrature weights (num_qpts * sizeof(Real))
    // - Write: local matrix (num_dofs^2 * sizeof(Real))

    double read_bytes = static_cast<double>(num_dofs) * num_qpts * dim * sizeof(Real)
                      + num_qpts * sizeof(Real);
    double write_bytes = static_cast<double>(num_dofs) * num_dofs * sizeof(Real);
    result.estimated_memory = read_bytes + write_bytes;

    return result;
}

std::vector<ElementComplexity> ComplexityEstimator::estimateAll(
    const IMeshAccess& mesh,
    const dofs::DofMap& dof_map) const
{
    GlobalIndex num_cells = mesh.numCells();
    std::vector<ElementComplexity> result;
    result.reserve(static_cast<std::size_t>(num_cells));

    for (GlobalIndex cell_id = 0; cell_id < num_cells; ++cell_id) {
        ElementType cell_type = mesh.getCellType(cell_id);

        // Get DOF count for this cell from DofMap
        LocalIndex num_dofs = 0;
        LocalIndex num_qpts = 0;
        int order = 1;

        // Estimate DOFs based on element type if DofMap query unavailable
        switch (cell_type) {
            case ElementType::Tetra4:
                num_dofs = 4;
                num_qpts = 4;
                order = 1;
                break;
            case ElementType::Tetra10:
                num_dofs = 10;
                num_qpts = 11;
                order = 2;
                break;
            case ElementType::Hex8:
                num_dofs = 8;
                num_qpts = 8;
                order = 1;
                break;
            case ElementType::Hex27:
                num_dofs = 27;
                num_qpts = 27;
                order = 2;
                break;
            case ElementType::Triangle3:
                num_dofs = 3;
                num_qpts = 3;
                order = 1;
                break;
            case ElementType::Triangle6:
                num_dofs = 6;
                num_qpts = 6;
                order = 2;
                break;
            case ElementType::Quad4:
                num_dofs = 4;
                num_qpts = 4;
                order = 1;
                break;
            case ElementType::Quad9:
                num_dofs = 9;
                num_qpts = 9;
                order = 2;
                break;
            default:
                num_dofs = 4;
                num_qpts = 4;
                order = 1;
                break;
        }

        auto complexity = estimate(cell_type, num_dofs, num_qpts, order);
        complexity.cell_id = cell_id;
        result.push_back(complexity);
    }

    return result;
}

void ComplexityEstimator::updateFromTiming(GlobalIndex /*cell_id*/, double /*actual_time*/)
{
    // Future: implement learning from actual timing data
    // Could use exponential moving average or similar approach
}

double ComplexityEstimator::totalCost(const std::vector<ElementComplexity>& complexities) const
{
    double total = 0.0;
    for (const auto& c : complexities) {
        total += c.cost();
    }
    return total;
}

double ComplexityEstimator::loadImbalance(
    const std::vector<ElementComplexity>& complexities,
    int num_threads) const
{
    if (num_threads <= 0 || complexities.empty()) {
        return 1.0;
    }

    double total = totalCost(complexities);
    double avg_per_thread = total / num_threads;

    // Simple static partitioning to estimate imbalance
    std::vector<double> thread_work(static_cast<std::size_t>(num_threads), 0.0);
    std::size_t chunk_size = (complexities.size() + static_cast<std::size_t>(num_threads) - 1)
                             / static_cast<std::size_t>(num_threads);

    for (std::size_t i = 0; i < complexities.size(); ++i) {
        int thread = static_cast<int>(i / chunk_size);
        if (thread >= num_threads) thread = num_threads - 1;
        thread_work[static_cast<std::size_t>(thread)] += complexities[i].cost();
    }

    double max_work = *std::max_element(thread_work.begin(), thread_work.end());
    return avg_per_thread > 0.0 ? max_work / avg_per_thread : 1.0;
}

// ============================================================================
// SpaceFillingCurve Implementation
// ============================================================================

std::uint64_t SpaceFillingCurve::hilbert2D(
    std::uint32_t x, std::uint32_t y, std::uint32_t n)
{
    std::uint64_t d = 0;
    for (std::uint32_t s = n / 2; s > 0; s /= 2) {
        std::uint32_t rx = (x & s) > 0 ? 1 : 0;
        std::uint32_t ry = (y & s) > 0 ? 1 : 0;
        d += s * s * ((3 * rx) ^ ry);

        // Rotate quadrant
        if (ry == 0) {
            if (rx == 1) {
                x = n - 1 - x;
                y = n - 1 - y;
            }
            std::swap(x, y);
        }
    }
    return d;
}

std::uint64_t SpaceFillingCurve::hilbert3D(
    std::uint32_t x, std::uint32_t y, std::uint32_t z, std::uint32_t n)
{
    // 3D Hilbert curve implementation using bit interleaving with rotation
    std::uint64_t d = 0;
    for (std::uint32_t s = n / 2; s > 0; s /= 2) {
        std::uint32_t rx = (x & s) > 0 ? 1 : 0;
        std::uint32_t ry = (y & s) > 0 ? 1 : 0;
        std::uint32_t rz = (z & s) > 0 ? 1 : 0;

        // Compute index contribution
        d += s * s * s * ((rx * 4) ^ (ry * 2) ^ rz);

        // Apply rotation (simplified)
        if (rz == 0) {
            if (ry == 1) {
                x = n - 1 - x;
                y = n - 1 - y;
            }
            std::swap(x, y);
        }
    }
    return d;
}

std::uint64_t SpaceFillingCurve::morton2D(std::uint32_t x, std::uint32_t y)
{
    // Interleave bits of x and y
    auto expandBits = [](std::uint32_t v) -> std::uint64_t {
        std::uint64_t x = v;
        x = (x | (x << 16)) & 0x0000FFFF0000FFFF;
        x = (x | (x <<  8)) & 0x00FF00FF00FF00FF;
        x = (x | (x <<  4)) & 0x0F0F0F0F0F0F0F0F;
        x = (x | (x <<  2)) & 0x3333333333333333;
        x = (x | (x <<  1)) & 0x5555555555555555;
        return x;
    };

    return expandBits(x) | (expandBits(y) << 1);
}

std::uint64_t SpaceFillingCurve::morton3D(
    std::uint32_t x, std::uint32_t y, std::uint32_t z)
{
    // Interleave bits of x, y, z
    auto expandBits = [](std::uint32_t v) -> std::uint64_t {
        std::uint64_t x = v & 0x1FFFFF;  // Use only 21 bits
        x = (x | (x << 32)) & 0x1F00000000FFFF;
        x = (x | (x << 16)) & 0x1F0000FF0000FF;
        x = (x | (x <<  8)) & 0x100F00F00F00F00F;
        x = (x | (x <<  4)) & 0x10C30C30C30C30C3;
        x = (x | (x <<  2)) & 0x1249249249249249;
        return x;
    };

    return expandBits(x) | (expandBits(y) << 1) | (expandBits(z) << 2);
}

std::uint32_t SpaceFillingCurve::discretize(
    Real coord, Real min_coord, Real max_coord, std::uint32_t num_cells)
{
    if (max_coord <= min_coord) return 0;

    Real normalized = (coord - min_coord) / (max_coord - min_coord);
    normalized = std::clamp(normalized, Real(0), Real(1) - std::numeric_limits<Real>::epsilon());

    return static_cast<std::uint32_t>(normalized * num_cells);
}

// ============================================================================
// NUMATopology Implementation
// ============================================================================

class NUMATopology::Impl {
public:
    Impl() {
        detectTopology();
    }

    void detectTopology() {
#if defined(__linux__)
        // Detect hardware concurrency
        num_cpus_ = static_cast<int>(std::thread::hardware_concurrency());
        if (num_cpus_ <= 0) num_cpus_ = 1;

        // Try to detect NUMA nodes (simplified - assumes uniform distribution)
        // In production, would use libnuma or hwloc
        num_nodes_ = 1;
        cpus_per_node_ = num_cpus_;

        // Check for NUMA by looking at /sys/devices/system/node
        // This is a simplified heuristic
        numa_available_ = false;  // Conservative default
#else
        num_nodes_ = 1;
        num_cpus_ = static_cast<int>(std::thread::hardware_concurrency());
        if (num_cpus_ <= 0) num_cpus_ = 1;
        cpus_per_node_ = num_cpus_;
        numa_available_ = false;
#endif
    }

    bool numa_available_{false};
    int num_nodes_{1};
    int num_cpus_{1};
    int cpus_per_node_{1};
};

NUMATopology::NUMATopology()
    : impl_(std::make_unique<Impl>())
{
}

NUMATopology::~NUMATopology() = default;

NUMATopology::NUMATopology(NUMATopology&& other) noexcept = default;
NUMATopology& NUMATopology::operator=(NUMATopology&& other) noexcept = default;

bool NUMATopology::isNUMAAvailable() const noexcept
{
    return impl_->numa_available_;
}

int NUMATopology::numNodes() const noexcept
{
    return impl_->num_nodes_;
}

int NUMATopology::cpusPerNode() const noexcept
{
    return impl_->cpus_per_node_;
}

int NUMATopology::currentNode() const
{
#if defined(__linux__)
    int cpu = sched_getcpu();
    if (cpu >= 0 && impl_->cpus_per_node_ > 0) {
        return cpu / impl_->cpus_per_node_;
    }
#endif
    return 0;
}

int NUMATopology::nodeForCPU(int cpu) const
{
    if (impl_->cpus_per_node_ > 0) {
        return cpu / impl_->cpus_per_node_;
    }
    return 0;
}

bool NUMATopology::pinToNode(int /*node*/)
{
    // Simplified - in production would use sched_setaffinity or hwloc
    return false;
}

bool NUMATopology::pinToCPU(int cpu)
{
#if defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    return sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) == 0;
#else
    (void)cpu;
    return false;
#endif
}

std::vector<int> NUMATopology::getCPUsInNode(int node) const
{
    std::vector<int> cpus;
    int start = node * impl_->cpus_per_node_;
    int end = std::min(start + impl_->cpus_per_node_, impl_->num_cpus_);

    for (int i = start; i < end; ++i) {
        cpus.push_back(i);
    }
    return cpus;
}

std::vector<int> NUMATopology::computeThreadMapping(int num_threads) const
{
    std::vector<int> mapping(static_cast<std::size_t>(num_threads));

    if (impl_->num_nodes_ <= 1) {
        // Single NUMA node - no special mapping
        std::fill(mapping.begin(), mapping.end(), 0);
    } else {
        // Distribute threads round-robin across NUMA nodes
        for (int t = 0; t < num_threads; ++t) {
            mapping[static_cast<std::size_t>(t)] = t % impl_->num_nodes_;
        }
    }

    return mapping;
}

// ============================================================================
// AssemblyScheduler Implementation
// ============================================================================

AssemblyScheduler::AssemblyScheduler()
    : options_{}
{
}

AssemblyScheduler::AssemblyScheduler(const SchedulerOptions& options)
    : options_(options)
{
}

AssemblyScheduler::~AssemblyScheduler() = default;

AssemblyScheduler::AssemblyScheduler(AssemblyScheduler&& other) noexcept = default;
AssemblyScheduler& AssemblyScheduler::operator=(AssemblyScheduler&& other) noexcept = default;

void AssemblyScheduler::setMesh(const IMeshAccess& mesh)
{
    mesh_ = &mesh;
    has_centroids_ = false;
    has_complexities_ = false;
}

void AssemblyScheduler::setDofMap(const dofs::DofMap& dof_map)
{
    dof_map_ = &dof_map;
    has_complexities_ = false;
}

void AssemblyScheduler::setOptions(const SchedulerOptions& options)
{
    options_ = options;
}

const SchedulerOptions& AssemblyScheduler::getOptions() const noexcept
{
    return options_;
}

void AssemblyScheduler::setCentroids(std::span<const std::array<Real, 3>> centroids)
{
    centroids_.assign(centroids.begin(), centroids.end());
    has_centroids_ = true;
}

void AssemblyScheduler::setComplexities(std::span<const ElementComplexity> complexities)
{
    complexities_.assign(complexities.begin(), complexities.end());
    has_complexities_ = true;
}

void AssemblyScheduler::setCustomComparator(
    std::function<bool(GlobalIndex, GlobalIndex)> comparator)
{
    custom_comparator_ = std::move(comparator);
}

bool AssemblyScheduler::isConfigured() const noexcept
{
    return mesh_ != nullptr;
}

const NUMATopology& AssemblyScheduler::getNUMATopology() const
{
    return numa_topology_;
}

ComplexityEstimator& AssemblyScheduler::getComplexityEstimator()
{
    return complexity_estimator_;
}

const SchedulingResult& AssemblyScheduler::getLastResult() const noexcept
{
    return last_result_;
}

void AssemblyScheduler::computeCentroids()
{
    if (!mesh_ || has_centroids_) return;

    GlobalIndex num_cells = mesh_->numCells();
    centroids_.resize(static_cast<std::size_t>(num_cells));

    // Compute centroids by averaging node coordinates
    // This is a simplified approach - in production would use actual geometry
    for (GlobalIndex cell_id = 0; cell_id < num_cells; ++cell_id) {
        // Use cell_id as a proxy for position (simplified)
        // In production, would query actual node coordinates
        auto& centroid = centroids_[static_cast<std::size_t>(cell_id)];
        centroid[0] = static_cast<Real>(cell_id % 100);
        centroid[1] = static_cast<Real>((cell_id / 100) % 100);
        centroid[2] = static_cast<Real>(cell_id / 10000);
    }

    has_centroids_ = true;
}

void AssemblyScheduler::computeComplexities()
{
    if (!mesh_ || !dof_map_ || has_complexities_) return;

    complexities_ = complexity_estimator_.estimateAll(*mesh_, *dof_map_);
    has_complexities_ = true;
}

SchedulingResult AssemblyScheduler::computeSchedule()
{
    auto start_time = std::chrono::high_resolution_clock::now();

    SchedulingResult result;

    if (!mesh_) {
        last_result_ = result;
        return result;
    }

    GlobalIndex num_cells = mesh_->numCells();
    result.ordering.resize(static_cast<std::size_t>(num_cells));
    std::iota(result.ordering.begin(), result.ordering.end(), GlobalIndex(0));

    // Apply ordering strategy
    switch (options_.ordering) {
        case OrderingStrategy::Natural:
            // Keep natural ordering
            break;

        case OrderingStrategy::Hilbert:
            applyHilbertOrdering(result.ordering);
            break;

        case OrderingStrategy::Morton:
            applyMortonOrdering(result.ordering);
            break;

        case OrderingStrategy::RCM:
            applyRCMOrdering(result.ordering);
            break;

        case OrderingStrategy::ComplexityBased:
            computeComplexities();
            if (has_complexities_ && !complexities_.empty()) {
                std::sort(result.ordering.begin(), result.ordering.end(),
                    [this](GlobalIndex a, GlobalIndex b) {
                        return complexities_[static_cast<std::size_t>(a)].cost() >
                               complexities_[static_cast<std::size_t>(b)].cost();
                    });
            }
            // If no complexities available, keep natural ordering
            break;

        case OrderingStrategy::CacheBlocked:
            applyCacheBlockedOrdering(result.ordering);
            break;

        case OrderingStrategy::Custom:
            if (custom_comparator_) {
                std::sort(result.ordering.begin(), result.ordering.end(),
                    custom_comparator_);
            }
            break;

        default:
            break;
    }

    // Distribute to threads
    distributeToThreads(result);

    auto end_time = std::chrono::high_resolution_clock::now();
    result.scheduling_seconds = std::chrono::duration<double>(
        end_time - start_time).count();

    // Compute estimated imbalance
    if (has_complexities_) {
        result.estimated_imbalance = complexity_estimator_.loadImbalance(
            complexities_, options_.num_threads > 0 ?
                options_.num_threads : static_cast<int>(std::thread::hardware_concurrency()));
    }

    last_result_ = result;
    return result;
}

SchedulingResult AssemblyScheduler::computeSchedule(std::span<const GlobalIndex> elements)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    SchedulingResult result;
    result.ordering.assign(elements.begin(), elements.end());

    // Apply ordering to subset
    switch (options_.ordering) {
        case OrderingStrategy::Hilbert:
            applyHilbertOrdering(result.ordering);
            break;

        case OrderingStrategy::Morton:
            applyMortonOrdering(result.ordering);
            break;

        case OrderingStrategy::ComplexityBased:
            if (has_complexities_) {
                std::sort(result.ordering.begin(), result.ordering.end(),
                    [this](GlobalIndex a, GlobalIndex b) {
                        return complexities_[static_cast<std::size_t>(a)].cost() >
                               complexities_[static_cast<std::size_t>(b)].cost();
                    });
            }
            break;

        default:
            break;
    }

    distributeToThreads(result);

    auto end_time = std::chrono::high_resolution_clock::now();
    result.scheduling_seconds = std::chrono::duration<double>(
        end_time - start_time).count();

    return result;
}

SchedulingResult AssemblyScheduler::computeNaturalSchedule()
{
    SchedulingResult result;

    if (!mesh_) return result;

    GlobalIndex num_cells = mesh_->numCells();
    result.ordering.resize(static_cast<std::size_t>(num_cells));
    std::iota(result.ordering.begin(), result.ordering.end(), GlobalIndex(0));

    distributeToThreads(result);
    return result;
}

SchedulingResult AssemblyScheduler::computeHilbertSchedule()
{
    OrderingStrategy saved = options_.ordering;
    options_.ordering = OrderingStrategy::Hilbert;
    auto result = computeSchedule();
    options_.ordering = saved;
    return result;
}

SchedulingResult AssemblyScheduler::computeMortonSchedule()
{
    OrderingStrategy saved = options_.ordering;
    options_.ordering = OrderingStrategy::Morton;
    auto result = computeSchedule();
    options_.ordering = saved;
    return result;
}

SchedulingResult AssemblyScheduler::computeRCMSchedule()
{
    OrderingStrategy saved = options_.ordering;
    options_.ordering = OrderingStrategy::RCM;
    auto result = computeSchedule();
    options_.ordering = saved;
    return result;
}

SchedulingResult AssemblyScheduler::computeComplexitySchedule()
{
    OrderingStrategy saved = options_.ordering;
    options_.ordering = OrderingStrategy::ComplexityBased;
    auto result = computeSchedule();
    options_.ordering = saved;
    return result;
}

SchedulingResult AssemblyScheduler::computeCacheBlockedSchedule()
{
    OrderingStrategy saved = options_.ordering;
    options_.ordering = OrderingStrategy::CacheBlocked;
    auto result = computeSchedule();
    options_.ordering = saved;
    return result;
}

void AssemblyScheduler::applyHilbertOrdering(std::vector<GlobalIndex>& ordering)
{
    computeCentroids();

    if (!has_centroids_ || centroids_.empty()) return;

    // Compute bounding box
    Real min_x = std::numeric_limits<Real>::max();
    Real min_y = std::numeric_limits<Real>::max();
    Real min_z = std::numeric_limits<Real>::max();
    Real max_x = std::numeric_limits<Real>::lowest();
    Real max_y = std::numeric_limits<Real>::lowest();
    Real max_z = std::numeric_limits<Real>::lowest();

    for (GlobalIndex cell_id : ordering) {
        const auto& c = centroids_[static_cast<std::size_t>(cell_id)];
        min_x = std::min(min_x, c[0]);
        min_y = std::min(min_y, c[1]);
        min_z = std::min(min_z, c[2]);
        max_x = std::max(max_x, c[0]);
        max_y = std::max(max_y, c[1]);
        max_z = std::max(max_z, c[2]);
    }

    // Grid resolution (power of 2)
    const std::uint32_t grid_size = 256;

    // Compute Hilbert index for each element
    std::vector<std::pair<std::uint64_t, GlobalIndex>> indexed;
    indexed.reserve(ordering.size());

    for (GlobalIndex cell_id : ordering) {
        const auto& c = centroids_[static_cast<std::size_t>(cell_id)];
        std::uint32_t ix = SpaceFillingCurve::discretize(c[0], min_x, max_x, grid_size);
        std::uint32_t iy = SpaceFillingCurve::discretize(c[1], min_y, max_y, grid_size);
        std::uint32_t iz = SpaceFillingCurve::discretize(c[2], min_z, max_z, grid_size);

        std::uint64_t hilbert_idx = SpaceFillingCurve::hilbert3D(ix, iy, iz, grid_size);
        indexed.emplace_back(hilbert_idx, cell_id);
    }

    // Sort by Hilbert index
    std::sort(indexed.begin(), indexed.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });

    // Extract ordering
    for (std::size_t i = 0; i < indexed.size(); ++i) {
        ordering[i] = indexed[i].second;
    }
}

void AssemblyScheduler::applyMortonOrdering(std::vector<GlobalIndex>& ordering)
{
    computeCentroids();

    if (!has_centroids_ || centroids_.empty()) return;

    // Compute bounding box
    Real min_x = std::numeric_limits<Real>::max();
    Real min_y = std::numeric_limits<Real>::max();
    Real min_z = std::numeric_limits<Real>::max();
    Real max_x = std::numeric_limits<Real>::lowest();
    Real max_y = std::numeric_limits<Real>::lowest();
    Real max_z = std::numeric_limits<Real>::lowest();

    for (GlobalIndex cell_id : ordering) {
        const auto& c = centroids_[static_cast<std::size_t>(cell_id)];
        min_x = std::min(min_x, c[0]);
        min_y = std::min(min_y, c[1]);
        min_z = std::min(min_z, c[2]);
        max_x = std::max(max_x, c[0]);
        max_y = std::max(max_y, c[1]);
        max_z = std::max(max_z, c[2]);
    }

    const std::uint32_t grid_size = 1 << 20;  // 2^20 for 21-bit coordinates

    // Compute Morton index for each element
    std::vector<std::pair<std::uint64_t, GlobalIndex>> indexed;
    indexed.reserve(ordering.size());

    for (GlobalIndex cell_id : ordering) {
        const auto& c = centroids_[static_cast<std::size_t>(cell_id)];
        std::uint32_t ix = SpaceFillingCurve::discretize(c[0], min_x, max_x, grid_size);
        std::uint32_t iy = SpaceFillingCurve::discretize(c[1], min_y, max_y, grid_size);
        std::uint32_t iz = SpaceFillingCurve::discretize(c[2], min_z, max_z, grid_size);

        std::uint64_t morton_idx = SpaceFillingCurve::morton3D(ix, iy, iz);
        indexed.emplace_back(morton_idx, cell_id);
    }

    // Sort by Morton index
    std::sort(indexed.begin(), indexed.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });

    // Extract ordering
    for (std::size_t i = 0; i < indexed.size(); ++i) {
        ordering[i] = indexed[i].second;
    }
}

void AssemblyScheduler::applyRCMOrdering(std::vector<GlobalIndex>& ordering)
{
    // Simplified RCM implementation
    // In production, would use actual element connectivity from mesh

    if (ordering.empty()) return;

    std::size_t n = ordering.size();
    std::vector<bool> visited(n, false);
    std::vector<GlobalIndex> result;
    result.reserve(n);

    // Build simple adjacency based on index proximity (simplified)
    // Real implementation would use actual element connectivity

    // Start from element with minimum degree (heuristic: use first element)
    std::queue<std::size_t> queue;
    queue.push(0);
    visited[0] = true;

    while (!queue.empty()) {
        std::size_t current = queue.front();
        queue.pop();
        result.push_back(ordering[current]);

        // Add neighbors (simplified: adjacent indices)
        for (int offset : {-1, 1}) {
            std::size_t neighbor = static_cast<std::size_t>(
                static_cast<std::ptrdiff_t>(current) + offset);
            if (neighbor < n && !visited[neighbor]) {
                visited[neighbor] = true;
                queue.push(neighbor);
            }
        }
    }

    // Add any unvisited elements
    for (std::size_t i = 0; i < n; ++i) {
        if (!visited[i]) {
            result.push_back(ordering[i]);
        }
    }

    // Reverse for RCM
    std::reverse(result.begin(), result.end());
    ordering = std::move(result);
}

void AssemblyScheduler::applyCacheBlockedOrdering(std::vector<GlobalIndex>& ordering)
{
    if (ordering.empty()) return;

    std::size_t block_size = options_.cache_block_size;
    if (block_size == 0) block_size = 64;

    // Simple blocking: group elements in chunks
    // Real implementation would analyze DOF connectivity

    // Already in blocks by natural ordering for this simplified version
    // More sophisticated: group elements sharing DOFs together

    // Apply Hilbert ordering within each block for cache efficiency
    std::size_t num_blocks = (ordering.size() + block_size - 1) / block_size;

    for (std::size_t b = 0; b < num_blocks; ++b) {
        std::size_t start = b * block_size;
        std::size_t end = std::min(start + block_size, ordering.size());

        // Sort block by element index (maintains some locality)
        std::sort(ordering.begin() + static_cast<std::ptrdiff_t>(start),
                  ordering.begin() + static_cast<std::ptrdiff_t>(end));
    }
}

void AssemblyScheduler::distributeToThreads(SchedulingResult& result)
{
    int num_threads = options_.num_threads;
    if (num_threads <= 0) {
#ifdef _OPENMP
        num_threads = omp_get_max_threads();
#else
        num_threads = static_cast<int>(std::thread::hardware_concurrency());
#endif
        if (num_threads <= 0) num_threads = 1;
    }

    std::size_t num_elements = result.ordering.size();
    result.thread_assignment.resize(num_elements);
    result.thread_ranges.resize(static_cast<std::size_t>(num_threads) + 1);
    result.thread_work.resize(static_cast<std::size_t>(num_threads), 0.0);

    switch (options_.load_balance) {
        case LoadBalanceMode::Static:
        case LoadBalanceMode::Guided:
        case LoadBalanceMode::Dynamic:
        default:
        {
            // Simple static partitioning
            std::size_t chunk_size = (num_elements + static_cast<std::size_t>(num_threads) - 1)
                                     / static_cast<std::size_t>(num_threads);

            for (int t = 0; t <= num_threads; ++t) {
                result.thread_ranges[static_cast<std::size_t>(t)] =
                    std::min(static_cast<std::size_t>(t) * chunk_size, num_elements);
            }

            for (std::size_t i = 0; i < num_elements; ++i) {
                int thread = static_cast<int>(i / chunk_size);
                if (thread >= num_threads) thread = num_threads - 1;
                result.thread_assignment[i] = thread;

                // Track work
                double work = 1.0;
                if (has_complexities_) {
                    GlobalIndex cell_id = result.ordering[i];
                    work = complexities_[static_cast<std::size_t>(cell_id)].cost();
                }
                result.thread_work[static_cast<std::size_t>(thread)] += work;
            }
            break;
        }

        case LoadBalanceMode::Adaptive:
        {
            // Complexity-aware distribution
            if (!has_complexities_) {
                computeComplexities();
            }

            // Compute total work
            double total_work = 0.0;
            for (const auto& c : complexities_) {
                total_work += c.cost();
            }
            double target_per_thread = total_work / num_threads;

            // Greedy assignment
            int current_thread = 0;
            double current_load = 0.0;
            result.thread_ranges[0] = 0;

            for (std::size_t i = 0; i < num_elements; ++i) {
                GlobalIndex cell_id = result.ordering[i];
                double work = has_complexities_ ?
                    complexities_[static_cast<std::size_t>(cell_id)].cost() : 1.0;

                result.thread_assignment[i] = current_thread;
                result.thread_work[static_cast<std::size_t>(current_thread)] += work;
                current_load += work;

                // Move to next thread if target exceeded
                if (current_load >= target_per_thread && current_thread < num_threads - 1) {
                    result.thread_ranges[static_cast<std::size_t>(current_thread) + 1] = i + 1;
                    current_thread++;
                    current_load = 0.0;
                }
            }

            // Fill remaining ranges
            for (int t = current_thread + 1; t <= num_threads; ++t) {
                result.thread_ranges[static_cast<std::size_t>(t)] = num_elements;
            }
            break;
        }
    }
}

std::vector<GlobalIndex> AssemblyScheduler::optimizeColorOrder(
    std::span<const int> colors, int num_colors)
{
    if (!mesh_ || colors.empty() || num_colors <= 0) {
        return {};
    }

    // Build per-color element lists
    std::vector<std::vector<GlobalIndex>> color_elements(static_cast<std::size_t>(num_colors));

    for (std::size_t i = 0; i < colors.size(); ++i) {
        int color = colors[i];
        if (color >= 0 && color < num_colors) {
            color_elements[static_cast<std::size_t>(color)].push_back(
                static_cast<GlobalIndex>(i));
        }
    }

    // Apply ordering within each color bucket
    for (auto& bucket : color_elements) {
        if (options_.ordering == OrderingStrategy::Hilbert) {
            applyHilbertOrdering(bucket);
        } else if (options_.ordering == OrderingStrategy::Morton) {
            applyMortonOrdering(bucket);
        }
        // Otherwise keep natural ordering within color
    }

    // Concatenate in color order
    std::vector<GlobalIndex> result;
    result.reserve(colors.size());

    for (const auto& bucket : color_elements) {
        result.insert(result.end(), bucket.begin(), bucket.end());
    }

    return result;
}

std::vector<int> AssemblyScheduler::optimizeColorSequence(
    std::span<const int> colors, int num_colors)
{
    // Default: process colors in order
    std::vector<int> sequence(static_cast<std::size_t>(num_colors));
    std::iota(sequence.begin(), sequence.end(), 0);

    // Could optimize based on color sizes for better load balancing
    // For now, return natural order

    return sequence;
}

std::vector<int> AssemblyScheduler::computeStaticAssignment(
    std::span<const GlobalIndex> ordering, int num_threads)
{
    std::vector<int> assignment(ordering.size());
    std::size_t chunk_size = (ordering.size() + static_cast<std::size_t>(num_threads) - 1)
                             / static_cast<std::size_t>(num_threads);

    for (std::size_t i = 0; i < ordering.size(); ++i) {
        int thread = static_cast<int>(i / chunk_size);
        if (thread >= num_threads) thread = num_threads - 1;
        assignment[i] = thread;
    }

    return assignment;
}

std::vector<std::size_t> AssemblyScheduler::computeThreadRanges(
    std::span<const GlobalIndex> ordering, int num_threads)
{
    std::vector<std::size_t> ranges(static_cast<std::size_t>(num_threads) + 1);
    std::size_t chunk_size = (ordering.size() + static_cast<std::size_t>(num_threads) - 1)
                             / static_cast<std::size_t>(num_threads);

    for (int t = 0; t <= num_threads; ++t) {
        ranges[static_cast<std::size_t>(t)] =
            std::min(static_cast<std::size_t>(t) * chunk_size, ordering.size());
    }

    return ranges;
}

std::vector<std::size_t> AssemblyScheduler::getWorkStealingChunks(
    std::span<const GlobalIndex> ordering,
    int num_threads,
    std::size_t chunk_size)
{
    std::vector<std::size_t> chunks;
    std::size_t num_chunks = (ordering.size() + chunk_size - 1) / chunk_size;
    chunks.reserve(num_chunks + 1);

    for (std::size_t i = 0; i <= ordering.size(); i += chunk_size) {
        chunks.push_back(std::min(i, ordering.size()));
    }

    if (chunks.back() != ordering.size()) {
        chunks.push_back(ordering.size());
    }

    (void)num_threads;  // Could use for guided scheduling
    return chunks;
}

// ============================================================================
// Factory Functions
// ============================================================================

std::unique_ptr<AssemblyScheduler> createAssemblyScheduler()
{
    return std::make_unique<AssemblyScheduler>();
}

std::unique_ptr<AssemblyScheduler> createAssemblyScheduler(const SchedulerOptions& options)
{
    return std::make_unique<AssemblyScheduler>(options);
}

std::unique_ptr<AssemblyScheduler> createAssemblyScheduler(OrderingStrategy ordering)
{
    SchedulerOptions options;
    options.ordering = ordering;
    return std::make_unique<AssemblyScheduler>(options);
}

} // namespace assembly
} // namespace FE
} // namespace svmp
