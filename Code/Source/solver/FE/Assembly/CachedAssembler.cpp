/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "CachedAssembler.h"
#include "StandardAssembler.h"
#include "Dofs/DofMap.h"
#include "Dofs/DofHandler.h"
#include "Constraints/AffineConstraints.h"
#include "Constraints/ConstraintDistributor.h"
#include "Sparsity/SparsityPattern.h"
#include "Spaces/FunctionSpace.h"
#include "Elements/Element.h"

#include <chrono>
#include <algorithm>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace svmp {
namespace FE {
namespace assembly {

// ============================================================================
// ElementMatrixCache Implementation
// ============================================================================

ElementMatrixCache::ElementMatrixCache() = default;

ElementMatrixCache::ElementMatrixCache(const CacheOptions& options)
    : options_(options)
{
}

ElementMatrixCache::~ElementMatrixCache() = default;

void ElementMatrixCache::setOptions(const CacheOptions& options)
{
    std::lock_guard<std::mutex> lock(mutex_);
    options_ = options;
}

const CacheOptions& ElementMatrixCache::getOptions() const noexcept
{
    return options_;
}

bool ElementMatrixCache::contains(GlobalIndex cell_id) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_.find(cell_id) != cache_.end();
}

const CachedElementData* ElementMatrixCache::get(GlobalIndex cell_id)
{
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = cache_.find(cell_id);
    if (it == cache_.end()) {
        stats_.cache_misses++;
        return nullptr;
    }

    stats_.cache_hits++;
    it->second.last_access = access_counter_.fetch_add(1);
    return &it->second;
}

bool ElementMatrixCache::insert(GlobalIndex cell_id, CachedElementData data)
{
    std::lock_guard<std::mutex> lock(mutex_);

    // Check memory limit
    if (options_.max_memory_bytes > 0) {
        std::size_t new_bytes = data.memoryBytes();
        while (stats_.memory_bytes + new_bytes > options_.max_memory_bytes &&
               !cache_.empty()) {
            evictOne();
        }

        if (stats_.memory_bytes + new_bytes > options_.max_memory_bytes) {
            return false;  // Cannot fit even after eviction
        }
    }

    // Check element count limit
    if (options_.max_elements > 0 && cache_.size() >= options_.max_elements) {
        evictOne();
    }

    data.last_access = access_counter_.fetch_add(1);
    std::size_t bytes = data.memoryBytes();

    auto result = cache_.insert_or_assign(cell_id, std::move(data));
    if (result.second) {  // New insertion
        stats_.memory_bytes += bytes;
        stats_.total_elements = cache_.size();
    }

    return true;
}

void ElementMatrixCache::invalidate(GlobalIndex cell_id)
{
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = cache_.find(cell_id);
    if (it != cache_.end()) {
        stats_.memory_bytes -= it->second.memoryBytes();
        cache_.erase(it);
        stats_.invalidations++;
        stats_.total_elements = cache_.size();
    }
}

void ElementMatrixCache::invalidateAll()
{
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.invalidations += cache_.size();
    cache_.clear();
    stats_.memory_bytes = 0;
    stats_.total_elements = 0;
}

void ElementMatrixCache::clear()
{
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
    stats_ = CacheStats{};
}

CacheStats ElementMatrixCache::getStats() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    CacheStats result = stats_;
    result.updateHitRate();
    return result;
}

void ElementMatrixCache::resetStats()
{
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.cache_hits = 0;
    stats_.cache_misses = 0;
    stats_.evictions = 0;
    stats_.invalidations = 0;
}

std::size_t ElementMatrixCache::memoryUsage() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_.memory_bytes;
}

void ElementMatrixCache::evictToLimit()
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (options_.max_memory_bytes > 0) {
        while (stats_.memory_bytes > options_.max_memory_bytes && !cache_.empty()) {
            evictOne();
        }
    }

    if (options_.max_elements > 0) {
        while (cache_.size() > options_.max_elements) {
            evictOne();
        }
    }
}

void ElementMatrixCache::reserve(std::size_t num_elements)
{
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.reserve(num_elements);
}

void ElementMatrixCache::evictOne()
{
    // Must be called with mutex held

    if (cache_.empty()) {
        return;
    }

    GlobalIndex victim = -1;

    switch (options_.eviction) {
        case EvictionPolicy::LRU: {
            // Find least recently used
            std::size_t min_access = std::numeric_limits<std::size_t>::max();
            for (const auto& [id, data] : cache_) {
                if (data.last_access < min_access) {
                    min_access = data.last_access;
                    victim = id;
                }
            }
            break;
        }

        case EvictionPolicy::FIFO: {
            // First element (arbitrary for unordered_map, but consistent)
            victim = cache_.begin()->first;
            break;
        }

        case EvictionPolicy::None:
            return;  // No eviction
    }

    if (victim >= 0) {
        auto it = cache_.find(victim);
        if (it != cache_.end()) {
            stats_.memory_bytes -= it->second.memoryBytes();
            cache_.erase(it);
            stats_.evictions++;
            stats_.total_elements = cache_.size();
        }
    }
}

// ============================================================================
// CachedAssembler Construction
// ============================================================================

CachedAssembler::CachedAssembler()
    : cache_(std::make_unique<ElementMatrixCache>()),
      standard_assembler_(std::make_unique<StandardAssembler>())
{
}

CachedAssembler::CachedAssembler(const CacheOptions& options)
    : cache_options_(options),
      cache_(std::make_unique<ElementMatrixCache>(options)),
      standard_assembler_(std::make_unique<StandardAssembler>())
{
}

CachedAssembler::~CachedAssembler() = default;

CachedAssembler::CachedAssembler(CachedAssembler&& other) noexcept = default;

CachedAssembler& CachedAssembler::operator=(CachedAssembler&& other) noexcept = default;

// ============================================================================
// Configuration
// ============================================================================

void CachedAssembler::setDofMap(const dofs::DofMap& dof_map)
{
    dof_map_ = &dof_map;
    standard_assembler_->setDofMap(dof_map);
}

void CachedAssembler::setDofHandler(const dofs::DofHandler& dof_handler)
{
    dof_handler_ = &dof_handler;
    dof_map_ = &dof_handler.getDofMap();
    standard_assembler_->setDofHandler(dof_handler);
}

void CachedAssembler::setConstraints(const constraints::AffineConstraints* constraints)
{
    constraints_ = constraints;
    standard_assembler_->setConstraints(constraints);

    if (constraints_ && constraints_->isClosed()) {
        constraint_distributor_ = std::make_unique<constraints::ConstraintDistributor>(*constraints_);
    } else {
        constraint_distributor_.reset();
    }
}

void CachedAssembler::setSparsityPattern(const sparsity::SparsityPattern* sparsity)
{
    sparsity_ = sparsity;
    standard_assembler_->setSparsityPattern(sparsity);
}

void CachedAssembler::setOptions(const AssemblyOptions& options)
{
    options_ = options;
    standard_assembler_->setOptions(options);
}

const AssemblyOptions& CachedAssembler::getOptions() const noexcept
{
    return options_;
}

bool CachedAssembler::isConfigured() const noexcept
{
    return dof_map_ != nullptr;
}

void CachedAssembler::setCacheOptions(const CacheOptions& options)
{
    cache_options_ = options;
    cache_->setOptions(options);
}

const CacheOptions& CachedAssembler::getCacheOptions() const noexcept
{
    return cache_options_;
}

CacheStats CachedAssembler::getCacheStats() const
{
    return cache_->getStats();
}

void CachedAssembler::resetCacheStats()
{
    cache_->resetStats();
}

// ============================================================================
// Lifecycle
// ============================================================================

void CachedAssembler::initialize()
{
    if (!isConfigured()) {
        throw std::runtime_error("CachedAssembler::initialize: not configured");
    }

    standard_assembler_->initialize();
    initialized_ = true;
}

void CachedAssembler::finalize(GlobalSystemView* matrix_view, GlobalSystemView* vector_view)
{
    standard_assembler_->finalize(matrix_view, vector_view);
}

void CachedAssembler::reset()
{
    standard_assembler_->reset();
    cache_->clear();
    cache_populated_ = false;
    initialized_ = false;
}

// ============================================================================
// Assembly Operations
// ============================================================================

AssemblyResult CachedAssembler::assembleMatrix(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView& matrix_view)
{
    if (cache_options_.strategy == CacheStrategy::None) {
        return standard_assembler_->assembleMatrix(mesh, test_space, trial_space,
                                                   kernel, matrix_view);
    }

    return assembleCellsCore(mesh, test_space, trial_space, kernel,
                             &matrix_view, nullptr, true, false, true);
}

AssemblyResult CachedAssembler::assembleVector(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& space,
    AssemblyKernel& kernel,
    GlobalSystemView& vector_view)
{
    // Vector assembly typically depends on time-varying data, so don't cache
    return standard_assembler_->assembleVector(mesh, space, kernel, vector_view);
}

AssemblyResult CachedAssembler::assembleBoth(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView& matrix_view,
    GlobalSystemView& vector_view)
{
    if (cache_options_.strategy == CacheStrategy::None) {
        return standard_assembler_->assembleBoth(mesh, test_space, trial_space,
                                                 kernel, matrix_view, vector_view);
    }

    return assembleCellsCore(mesh, test_space, trial_space, kernel,
                             &matrix_view, &vector_view, true, true, true);
}

AssemblyResult CachedAssembler::assembleBoundaryFaces(
    const IMeshAccess& mesh,
    int boundary_marker,
    const spaces::FunctionSpace& space,
    AssemblyKernel& kernel,
    GlobalSystemView* matrix_view,
    GlobalSystemView* vector_view)
{
    // Boundary face assembly - delegate to standard assembler
    return standard_assembler_->assembleBoundaryFaces(mesh, boundary_marker, space,
                                                      kernel, matrix_view, vector_view);
}

AssemblyResult CachedAssembler::assembleInteriorFaces(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView& matrix_view,
    GlobalSystemView* vector_view)
{
    // Interior face assembly - delegate to standard assembler
    return standard_assembler_->assembleInteriorFaces(mesh, test_space, trial_space,
                                                      kernel, matrix_view, vector_view);
}

// ============================================================================
// Cache-Specific Assembly
// ============================================================================

AssemblyResult CachedAssembler::populateCache(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel)
{
    // Assemble without inserting into any global system
    return assembleCellsCore(mesh, test_space, trial_space, kernel,
                             nullptr, nullptr, true, false, true);
}

AssemblyResult CachedAssembler::assembleMatrixFromCache(GlobalSystemView& matrix_view)
{
    AssemblyResult result;
    auto start_time = std::chrono::steady_clock::now();

    if (!cache_populated_) {
        throw std::runtime_error("CachedAssembler::assembleMatrixFromCache: cache not populated");
    }

    matrix_view.beginAssemblyPhase();

    auto stats = cache_->getStats();
    result.elements_assembled = static_cast<GlobalIndex>(stats.total_elements);

    // Iterate over all cached elements
    // Note: This requires iterating the cache, which is not exposed directly
    // In practice, we'd store the cell IDs or iterate over mesh

    // For now, rely on the fact that cache is populated for all mesh cells
    // and use assembleMatrixHybrid which handles both cases

    auto end_time = std::chrono::steady_clock::now();
    result.elapsed_time_seconds = std::chrono::duration<double>(end_time - start_time).count();

    return result;
}

AssemblyResult CachedAssembler::assembleMatrixHybrid(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView& matrix_view)
{
    return assembleCellsCore(mesh, test_space, trial_space, kernel,
                             &matrix_view, nullptr, true, false, true);
}

// ============================================================================
// Cache Management
// ============================================================================

void CachedAssembler::invalidateCache()
{
    cache_->invalidateAll();
    cache_populated_ = false;
}

void CachedAssembler::invalidateElements(std::span<const GlobalIndex> cell_ids)
{
    for (GlobalIndex id : cell_ids) {
        cache_->invalidate(id);
        if (invalidation_callback_) {
            invalidation_callback_(id);
        }
    }
}

bool CachedAssembler::isCachePopulated() const noexcept
{
    return cache_populated_;
}

std::size_t CachedAssembler::numCachedElements() const
{
    return cache_->getStats().total_elements;
}

std::size_t CachedAssembler::cacheMemoryUsage() const
{
    return cache_->memoryUsage();
}

void CachedAssembler::setInvalidationCallback(std::function<void(GlobalIndex)> callback)
{
    invalidation_callback_ = std::move(callback);
}

// ============================================================================
// Internal Methods
// ============================================================================

AssemblyResult CachedAssembler::assembleCellsCore(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView* matrix_view,
    GlobalSystemView* vector_view,
    bool assemble_matrix,
    bool assemble_vector,
    bool populate_cache)
{
    AssemblyResult result;
    auto start_time = std::chrono::steady_clock::now();

    if (!initialized_) {
        initialize();
    }

    if (matrix_view && assemble_matrix) {
        matrix_view->beginAssemblyPhase();
    }
    if (vector_view && assemble_vector && vector_view != matrix_view) {
        vector_view->beginAssemblyPhase();
    }

    const RequiredData required_data = kernel.getRequiredData();
    const bool is_rectangular = (&test_space != &trial_space);

    // Reserve cache capacity
    if (populate_cache) {
        cache_->reserve(static_cast<std::size_t>(mesh.numCells()));
    }

    // Context for computing elements (when cache miss)
    AssemblyContext context;
    context.reserve(dof_map_->getMaxDofsPerCell(), 27, mesh.dimension());
    KernelOutput kernel_output;

    mesh.forEachCell([&](GlobalIndex cell_id) {
        // Try cache first
        const CachedElementData* cached = cache_->get(cell_id);

        if (cached != nullptr) {
            // Cache hit - use cached data
            insertFromCache(*cached, matrix_view, vector_view);
            result.elements_assembled++;
            return;
        }

        // Cache miss - compute element contributions
        auto test_dofs = dof_map_->getCellDofs(cell_id);
        std::vector<GlobalIndex> row_dofs(test_dofs.begin(), test_dofs.end());
        std::vector<GlobalIndex> col_dofs = row_dofs;

        if (is_rectangular) {
            // Handle rectangular assembly (different test/trial spaces)
            // For now, use same DOFs
        }

        // Prepare context
        ElementType cell_type = mesh.getCellType(cell_id);
        const auto& test_element = test_space.getElement(cell_type, cell_id);
        const auto& trial_element = trial_space.getElement(cell_type, cell_id);
        context.configure(cell_id, test_element, trial_element, required_data);

        // Compute
        kernel_output.clear();
        kernel.computeCell(context, kernel_output);

        // Insert into global system
        if (matrix_view && kernel_output.has_matrix) {
            matrix_view->addMatrixEntries(row_dofs, col_dofs, kernel_output.local_matrix);
        }
        if (vector_view && kernel_output.has_vector) {
            vector_view->addVectorEntries(row_dofs, kernel_output.local_vector);
        }

        // Populate cache
        if (populate_cache && assemble_matrix && kernel_output.has_matrix) {
            CachedElementData data;
            data.local_matrix = kernel_output.local_matrix;
            data.row_dofs = std::move(row_dofs);
            data.col_dofs = std::move(col_dofs);
            data.element_type = cell_type;
            data.has_matrix = true;
            data.has_vector = false;

            if (assemble_vector && kernel_output.has_vector) {
                data.local_vector = kernel_output.local_vector;
                data.has_vector = true;
            }

            cache_->insert(cell_id, std::move(data));
        }

        result.elements_assembled++;
    });

    if (populate_cache) {
        cache_populated_ = true;
    }

    auto end_time = std::chrono::steady_clock::now();
    result.elapsed_time_seconds = std::chrono::duration<double>(end_time - start_time).count();

    return result;
}

void CachedAssembler::insertFromCache(
    const CachedElementData& cached,
    GlobalSystemView* matrix_view,
    GlobalSystemView* vector_view)
{
    if (matrix_view && cached.has_matrix) {
        matrix_view->addMatrixEntries(cached.row_dofs, cached.col_dofs, cached.local_matrix);
    }

    if (vector_view && cached.has_vector) {
        vector_view->addVectorEntries(cached.row_dofs, cached.local_vector);
    }
}

// ============================================================================
// Factory Functions
// ============================================================================

std::unique_ptr<Assembler> createCachedAssembler()
{
    return std::make_unique<CachedAssembler>();
}

std::unique_ptr<Assembler> createCachedAssembler(const CacheOptions& options)
{
    return std::make_unique<CachedAssembler>(options);
}

} // namespace assembly
} // namespace FE
} // namespace svmp
