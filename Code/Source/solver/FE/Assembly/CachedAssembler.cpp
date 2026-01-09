/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "CachedAssembler.h"
#include "StandardAssembler.h"
#include "Spaces/FunctionSpace.h"

#include <algorithm>
#include <limits>

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

    const auto existing_it = cache_.find(cell_id);
    const bool has_existing = existing_it != cache_.end();
    const std::size_t old_bytes = has_existing ? existing_it->second.memoryBytes() : 0u;

    data.last_access = access_counter_.fetch_add(1);
    data.insertion_order = insertion_counter_.fetch_add(1);
    const std::size_t new_bytes = data.memoryBytes();

    // Check memory limit
    if (options_.max_memory_bytes > 0) {
        while (stats_.memory_bytes - old_bytes + new_bytes > options_.max_memory_bytes &&
               !cache_.empty()) {
            evictOne();
        }

        if (stats_.memory_bytes - old_bytes + new_bytes > options_.max_memory_bytes) {
            return false;  // Cannot fit even after eviction
        }
    }

    // Check element count limit
    if (options_.max_elements > 0 && !has_existing && cache_.size() >= options_.max_elements) {
        evictOne();
    }

    auto result = cache_.insert_or_assign(cell_id, std::move(data));
    if (result.second) {  // New insertion
        stats_.memory_bytes += new_bytes;
        stats_.total_elements = cache_.size();
    } else if (has_existing) {
        stats_.memory_bytes = stats_.memory_bytes - old_bytes + new_bytes;
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

    if (options_.eviction == EvictionPolicy::None) {
        return;
    }

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
            // Evict first inserted entry.
            std::size_t min_order = std::numeric_limits<std::size_t>::max();
            for (const auto& [id, data] : cache_) {
                if (data.insertion_order < min_order) {
                    min_order = data.insertion_order;
                    victim = id;
                }
            }
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

namespace {

class NullSystemView final : public GlobalSystemView {
public:
    explicit NullSystemView(bool has_matrix = true, bool has_vector = true) noexcept
        : has_matrix_(has_matrix), has_vector_(has_vector)
    {
    }

    void addMatrixEntries(std::span<const GlobalIndex> /*dofs*/,
                          std::span<const Real> /*local_matrix*/,
                          AddMode /*mode*/ = AddMode::Add) override
    {
    }

    void addMatrixEntries(std::span<const GlobalIndex> /*row_dofs*/,
                          std::span<const GlobalIndex> /*col_dofs*/,
                          std::span<const Real> /*local_matrix*/,
                          AddMode /*mode*/ = AddMode::Add) override
    {
    }

    void addMatrixEntry(GlobalIndex /*row*/, GlobalIndex /*col*/, Real /*value*/,
                        AddMode /*mode*/ = AddMode::Add) override
    {
    }

    void setDiagonal(std::span<const GlobalIndex> /*dofs*/,
                     std::span<const Real> /*values*/) override
    {
    }

    void setDiagonal(GlobalIndex /*dof*/, Real /*value*/) override {}

    void zeroRows(std::span<const GlobalIndex> /*rows*/, bool /*set_diagonal*/ = true) override {}

    void addVectorEntries(std::span<const GlobalIndex> /*dofs*/,
                          std::span<const Real> /*local_vector*/,
                          AddMode /*mode*/ = AddMode::Add) override
    {
    }

    void addVectorEntry(GlobalIndex /*dof*/, Real /*value*/, AddMode /*mode*/ = AddMode::Add) override {}

    void setVectorEntries(std::span<const GlobalIndex> /*dofs*/,
                          std::span<const Real> /*values*/) override
    {
    }

    void zeroVectorEntries(std::span<const GlobalIndex> /*dofs*/) override {}

    void beginAssemblyPhase() override { phase_ = AssemblyPhase::Building; }
    void endAssemblyPhase() override { phase_ = AssemblyPhase::Flushing; }
    void finalizeAssembly() override { phase_ = AssemblyPhase::Finalized; }

    [[nodiscard]] AssemblyPhase getPhase() const noexcept override { return phase_; }

    [[nodiscard]] bool hasMatrix() const noexcept override { return has_matrix_; }
    [[nodiscard]] bool hasVector() const noexcept override { return has_vector_; }
    [[nodiscard]] GlobalIndex numRows() const noexcept override { return 0; }
    [[nodiscard]] GlobalIndex numCols() const noexcept override { return 0; }
    [[nodiscard]] std::string backendName() const override { return "NullSystemView"; }

    void zero() override {}

private:
    bool has_matrix_{true};
    bool has_vector_{true};
    AssemblyPhase phase_{AssemblyPhase::NotStarted};
};

class CachedKernel final : public AssemblyKernel {
public:
    CachedKernel(AssemblyKernel& inner,
                 ElementMatrixCache& cache,
                 bool want_matrix,
                 bool want_vector,
                 bool populate_cache)
        : inner_(inner),
          cache_(cache),
          populate_cache_(populate_cache),
          want_matrix_(want_matrix),
          want_vector_(want_vector)
    {
    }

    [[nodiscard]] RequiredData getRequiredData() const override { return inner_.getRequiredData(); }
    [[nodiscard]] std::vector<FieldRequirement> fieldRequirements() const override { return inner_.fieldRequirements(); }
    [[nodiscard]] MaterialStateSpec materialStateSpec() const noexcept override { return inner_.materialStateSpec(); }
    [[nodiscard]] std::vector<params::Spec> parameterSpecs() const override { return inner_.parameterSpecs(); }

    [[nodiscard]] bool hasCell() const noexcept override { return inner_.hasCell(); }
    [[nodiscard]] bool hasBoundaryFace() const noexcept override { return inner_.hasBoundaryFace(); }
    [[nodiscard]] bool hasInteriorFace() const noexcept override { return inner_.hasInteriorFace(); }

    [[nodiscard]] std::string name() const override { return inner_.name(); }
    [[nodiscard]] int maxTemporalDerivativeOrder() const noexcept override { return inner_.maxTemporalDerivativeOrder(); }
    [[nodiscard]] bool isSymmetric() const noexcept override { return inner_.isSymmetric(); }
    [[nodiscard]] bool isMatrixOnly() const noexcept override { return inner_.isMatrixOnly(); }
    [[nodiscard]] bool isVectorOnly() const noexcept override { return inner_.isVectorOnly(); }

    void computeCell(const AssemblyContext& ctx, KernelOutput& out) override
    {
        const GlobalIndex cell_id = ctx.cellId();
        const CachedElementData* cached = cache_.get(cell_id);

        if (cached != nullptr &&
            (!want_matrix_ || cached->has_matrix) &&
            (!want_vector_ || cached->has_vector)) {
            out.reserve(ctx.numTestDofs(), ctx.numTrialDofs(),
                        /*need_matrix=*/want_matrix_,
                        /*need_vector=*/want_vector_);

            if (want_matrix_) {
                out.local_matrix = cached->local_matrix;
            }
            if (want_vector_) {
                out.local_vector = cached->local_vector;
            }
            return;
        }

        inner_.computeCell(ctx, out);

        if (!populate_cache_) {
            return;
        }

        if (!out.has_matrix && !out.has_vector) {
            return;
        }

        CachedElementData data;
        if (out.has_matrix) {
            data.local_matrix = out.local_matrix;
            data.has_matrix = true;
        }
        if (out.has_vector) {
            data.local_vector = out.local_vector;
            data.has_vector = true;
        }

        (void)cache_.insert(cell_id, std::move(data));
    }

    void computeBoundaryFace(const AssemblyContext& ctx, int boundary_marker, KernelOutput& out) override
    {
        inner_.computeBoundaryFace(ctx, boundary_marker, out);
    }

    void computeInteriorFace(const AssemblyContext& ctx_minus,
                             const AssemblyContext& ctx_plus,
                             KernelOutput& out_minus,
                             KernelOutput& out_plus,
                             KernelOutput& coupling_mp,
                             KernelOutput& coupling_pm) override
    {
        inner_.computeInteriorFace(ctx_minus, ctx_plus, out_minus, out_plus, coupling_mp, coupling_pm);
    }

private:
    AssemblyKernel& inner_;
    ElementMatrixCache& cache_;
    bool populate_cache_{true};
    bool want_matrix_{true};
    bool want_vector_{false};
};

} // namespace

CachedAssembler::CachedAssembler()
    : DecoratorAssembler(createStandardAssembler()),
      cache_(std::make_unique<ElementMatrixCache>())
{
}

CachedAssembler::CachedAssembler(const CacheOptions& options)
    : DecoratorAssembler(createStandardAssembler()),
      cache_options_(options),
      cache_(std::make_unique<ElementMatrixCache>(options))
{
}

CachedAssembler::CachedAssembler(std::unique_ptr<Assembler> base, const CacheOptions& options)
    : DecoratorAssembler(std::move(base)),
      cache_options_(options),
      cache_(std::make_unique<ElementMatrixCache>(options))
{
}

CachedAssembler::~CachedAssembler() = default;
CachedAssembler::CachedAssembler(CachedAssembler&& other) noexcept = default;
CachedAssembler& CachedAssembler::operator=(CachedAssembler&& other) noexcept = default;

void CachedAssembler::setCacheOptions(const CacheOptions& options)
{
    cache_options_ = options;
    cache_->setOptions(options);
    if (options.eviction != EvictionPolicy::None) {
        cache_->evictToLimit();
    }
    cache_populated_ = cache_->getStats().total_elements > 0;
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

void CachedAssembler::reset()
{
    base().reset();
    cache_->clear();
    cache_populated_ = false;
}

AssemblyResult CachedAssembler::assembleMatrix(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView& matrix_view)
{
    if (cache_options_.strategy == CacheStrategy::None) {
        return base().assembleMatrix(mesh, test_space, trial_space, kernel, matrix_view);
    }

    CachedKernel cached_kernel(kernel, *cache_, /*want_matrix=*/true, /*want_vector=*/false, /*populate_cache=*/true);
    auto result = base().assembleMatrix(mesh, test_space, trial_space, cached_kernel, matrix_view);
    cache_populated_ = cache_->getStats().total_elements > 0;
    return result;
}

AssemblyResult CachedAssembler::assembleVector(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& space,
    AssemblyKernel& kernel,
    GlobalSystemView& vector_view)
{
    return base().assembleVector(mesh, space, kernel, vector_view);
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
        return base().assembleBoth(mesh, test_space, trial_space, kernel, matrix_view, vector_view);
    }

    CachedKernel cached_kernel(kernel, *cache_, /*want_matrix=*/true, /*want_vector=*/true, /*populate_cache=*/true);
    auto result = base().assembleBoth(mesh, test_space, trial_space, cached_kernel, matrix_view, vector_view);
    cache_populated_ = cache_->getStats().total_elements > 0;
    return result;
}

AssemblyResult CachedAssembler::assembleBoundaryFaces(
    const IMeshAccess& mesh,
    int boundary_marker,
    const spaces::FunctionSpace& space,
    AssemblyKernel& kernel,
    GlobalSystemView* matrix_view,
    GlobalSystemView* vector_view)
{
    return base().assembleBoundaryFaces(mesh, boundary_marker, space, kernel, matrix_view, vector_view);
}

AssemblyResult CachedAssembler::assembleInteriorFaces(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView& matrix_view,
    GlobalSystemView* vector_view)
{
    return base().assembleInteriorFaces(mesh, test_space, trial_space, kernel, matrix_view, vector_view);
}

AssemblyResult CachedAssembler::populateCache(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel)
{
    if (cache_options_.strategy == CacheStrategy::None) {
        return {};
    }

    NullSystemView null_matrix(/*has_matrix=*/true, /*has_vector=*/false);
    CachedKernel cached_kernel(kernel, *cache_, /*want_matrix=*/true, /*want_vector=*/false, /*populate_cache=*/true);
    auto result = base().assembleMatrix(mesh, test_space, trial_space, cached_kernel, null_matrix);
    cache_populated_ = cache_->getStats().total_elements > 0;
    return result;
}

AssemblyResult CachedAssembler::assembleMatrixFromCache(GlobalSystemView& /*matrix_view*/)
{
    FE_THROW_IF(!cache_populated_, FEException, "CachedAssembler::assembleMatrixFromCache: cache not populated");
    FE_THROW(FEException,
             "CachedAssembler::assembleMatrixFromCache: not implemented (requires mesh/space/kernel context)");
}

AssemblyResult CachedAssembler::assembleMatrixHybrid(
    const IMeshAccess& mesh,
    const spaces::FunctionSpace& test_space,
    const spaces::FunctionSpace& trial_space,
    AssemblyKernel& kernel,
    GlobalSystemView& matrix_view)
{
    return assembleMatrix(mesh, test_space, trial_space, kernel, matrix_view);
}

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
    cache_populated_ = cache_->getStats().total_elements > 0;
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
